"""Training loop for the policy+value network.

Loss: policy_weight * cross-entropy(policy logits, MCTS visit distribution)
     + value_weight * MSE(value, game outcome z)
     + weight_decay * L2 (via AdamW).

Optimiser: AdamW with cosine or step LR decay (with a short warmup).
A held-out validation fraction (default 5%) is used to track the best
checkpoint by val loss. Checkpoints save the model state dict plus
optimiser state and training metadata so training can resume.

Data: ``.npz`` shards produced by selfplay.py (planes/policy/value).
All shards in a directory are loaded into memory — fine for the smoke
run and for modest datasets; a streaming loader is future work for
AlphaZero-scale data volumes.

Augmentation: when enabled, each batch is transformed with random
dihedral board symmetries (rotations + reflections, see augment.py)
before the forward pass. Policy targets are transformed identically;
value targets are unchanged.

Progress is reported through the ``on_event`` callback (dicts with an
``"event"`` key: ``train_step``, ``epoch``, ``checkpoint``); the CLI
scripts forward these to the run logger (JSON lines on stdout +
metrics.jsonl). Human-readable lines are printed to stderr.
"""

from __future__ import annotations

import glob
import os
import sys
import time
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F

from . import augment
from .model import GomokuNet


def load_shards(data_dirs: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load all ``.npz`` shards under the given directories (recursive)
    and concatenate them into (planes, policy, value) arrays."""
    planes_list, policy_list, value_list = [], [], []
    for d in data_dirs:
        for path in sorted(glob.glob(os.path.join(d, "**", "*.npz"), recursive=True)):
            with np.load(path) as z:
                planes_list.append(z["planes"])
                policy_list.append(z["policy"])
                value_list.append(z["value"])
    if not planes_list:
        raise FileNotFoundError(f"no .npz shards found under {data_dirs}")
    return (
        np.concatenate(planes_list),
        np.concatenate(policy_list),
        np.concatenate(value_list),
    )


def save_checkpoint(
    path: str,
    model: GomokuNet,
    optimizer: torch.optim.Optimizer,
    step: int,
    epoch: int,
    extra: dict | None = None,
) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "step": step,
        "epoch": epoch,
        "channels": getattr(model, "channels", None),
        "blocks": getattr(model, "blocks", None),
        "architecture": getattr(model, "architecture_name", None),
        "arch_params": getattr(model, "arch_params", None),
        "num_parameters": model.num_parameters(),
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def load_checkpoint(
    path: str, model: GomokuNet, optimizer: torch.optim.Optimizer | None = None,
    device: str = "cpu",
) -> dict:
    payload = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(payload["model_state"])
    if optimizer is not None and "optimizer_state" in payload:
        optimizer.load_state_dict(payload["optimizer_state"])
    return payload


def _evaluate_loss(
    model: GomokuNet,
    planes: np.ndarray,
    policy_targets: np.ndarray,
    values: np.ndarray,
    batch_size: int,
    policy_weight: float,
    value_weight: float,
    device: torch.device,
) -> tuple[float, float, float]:
    """Mean (loss, policy loss, value loss) over a held-out set."""
    model.eval()
    total = total_p = total_v = 0.0
    n = len(planes)
    with torch.no_grad():
        for i in range(0, n, batch_size):
            x = torch.from_numpy(planes[i : i + batch_size]).to(device)
            pt = torch.from_numpy(policy_targets[i : i + batch_size]).to(device)
            z = torch.from_numpy(values[i : i + batch_size]).to(device).unsqueeze(1)
            logits, value = model(x)
            log_probs = F.log_softmax(logits, dim=1)
            p_loss = -(pt * log_probs).sum(dim=1).mean()
            v_loss = F.mse_loss(value, z)
            loss = policy_weight * p_loss + value_weight * v_loss
            w = len(x)
            total += float(loss.item()) * w
            total_p += float(p_loss.item()) * w
            total_v += float(v_loss.item()) * w
    model.train()
    return total / n, total_p / n, total_v / n


def train(
    model: GomokuNet,
    planes: np.ndarray,
    policy_targets: np.ndarray,
    values: np.ndarray,
    epochs: int = 10,
    batch_size: int = 64,
    lr: float = 1e-3,
    lr_schedule: str = "cosine",
    lr_step_size: int = 10,
    lr_gamma: float = 0.5,
    weight_decay: float = 1e-4,
    policy_weight: float = 1.0,
    value_weight: float = 1.0,
    warmup_steps: int = 10,
    augmentation: bool = True,
    val_fraction: float = 0.05,
    device: str = "cpu",
    checkpoint_dir: str | None = None,
    checkpoint_every_epochs: int = 0,
    optimizer: torch.optim.Optimizer | None = None,
    start_step: int = 0,
    start_epoch: int = 0,
    seed: int = 0,
    log_every: int = 10,
    on_event: Callable[[dict], None] | None = None,
    stop_flag=None,
    schedule_epochs: int | None = None,
) -> dict:
    """Run the training loop. Returns a summary dict with loss history,
    best val loss, checkpoint paths, and ``interrupted`` (True when
    ``stop_flag`` stopped the run early — see below).

    ``stop_flag`` is an :class:`interrupt.InterruptFlag` (or any object
    with a ``requested`` attribute). When set, the current batch
    finishes, an ``interrupted.pt`` checkpoint (weights + optimiser +
    step) is written to ``checkpoint_dir`` (when given), and the loop
    returns normally with ``interrupted: True`` and
    ``interrupted_checkpoint`` set — the CLI then reports the exact
    resume command.

    ``schedule_epochs``: the epoch number the run ends at
    (``start_epoch + epochs``). The cosine LR schedule then spans the
    whole run — total steps = steps-per-epoch x schedule_epochs — so
    a resumed run continues the original decay exactly instead of
    restarting a fresh decay over the remaining epochs. (Passing the
    run's total epoch count here is only correct when the run ends at
    that epoch; the CLI passes ``start_epoch + epochs``.)
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    device_t = torch.device(device)
    model.to(device_t)
    model.train()

    n = len(planes)
    perm_all = rng.permutation(n)
    n_val = int(round(n * val_fraction)) if val_fraction > 0 else 0
    val_idx, train_idx = perm_all[:n_val], perm_all[n_val:]
    x_tr, p_tr, z_tr = planes[train_idx], policy_targets[train_idx], values[train_idx]
    n_tr = len(x_tr)

    steps_per_epoch = max(1, (n_tr + batch_size - 1) // batch_size)
    if schedule_epochs is not None:
        total_steps = steps_per_epoch * schedule_epochs
    else:
        total_steps = start_step + steps_per_epoch * epochs
    if optimizer is None:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )

    def lr_at(step: int, epoch: int) -> float:
        if step < warmup_steps:
            return lr * (step + 1) / max(1, warmup_steps)
        if lr_schedule == "step":
            return lr * (lr_gamma ** (epoch // lr_step_size))
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * lr * (1.0 + np.cos(np.pi * min(1.0, progress)))

    def emit(event: str, **fields) -> None:
        if on_event is not None:
            on_event({"event": event, **fields})

    history: list[dict] = []
    best_val_loss: float | None = None
    best_path: str | None = None
    interrupted = False
    interrupted_path: str | None = None
    step = start_step
    for epoch in range(start_epoch, start_epoch + epochs):
        perm = rng.permutation(n_tr)
        epoch_loss = epoch_ploss = epoch_vloss = 0.0
        batches = 0
        t0 = time.time()
        for i in range(0, n_tr, batch_size):
            if stop_flag is not None and stop_flag.requested:
                interrupted = True
                break
            idx = perm[i : i + batch_size]
            batch_planes = x_tr[idx]
            batch_policy = p_tr[idx]
            if augmentation:
                batch_planes, batch_policy = augment.random_augment_batch(
                    batch_planes, batch_policy, rng
                )
            x = torch.from_numpy(np.ascontiguousarray(batch_planes)).to(device_t)
            pt = torch.from_numpy(np.ascontiguousarray(batch_policy)).to(device_t)
            z = torch.from_numpy(z_tr[idx]).to(device_t).unsqueeze(1)
            current_lr = lr_at(step, epoch)
            for group in optimizer.param_groups:
                group["lr"] = current_lr
            policy_logits, value = model(x)
            log_probs = F.log_softmax(policy_logits, dim=1)
            policy_loss = -(pt * log_probs).sum(dim=1).mean()
            value_loss = F.mse_loss(value, z)
            loss = policy_weight * policy_loss + value_weight * value_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())
            epoch_ploss += float(policy_loss.item())
            epoch_vloss += float(value_loss.item())
            batches += 1
            step += 1
            if step % log_every == 0:
                emit(
                    "train_step",
                    step=step,
                    epoch=epoch + 1,
                    loss=float(loss.item()),
                    policy_loss=float(policy_loss.item()),
                    value_loss=float(value_loss.item()),
                    lr=current_lr,
                )
        if interrupted:
            # Partial epoch: not recorded as a completed epoch (the
            # step count in the interrupt checkpoint is the truth).
            break
        entry = {
            "epoch": epoch + 1,
            "loss": epoch_loss / batches,
            "policy_loss": epoch_ploss / batches,
            "value_loss": epoch_vloss / batches,
            "val_loss": None,
            "seconds": time.time() - t0,
            "num_samples": n_tr,
        }
        if n_val > 0:
            val_loss, _, _ = _evaluate_loss(
                model, planes[val_idx], policy_targets[val_idx], values[val_idx],
                batch_size, policy_weight, value_weight, device_t,
            )
            entry["val_loss"] = val_loss
            if best_val_loss is None or val_loss < best_val_loss:
                best_val_loss = val_loss
                if checkpoint_dir:
                    best_path = os.path.join(checkpoint_dir, "best.pt")
                    save_checkpoint(best_path, model, optimizer, step, epoch + 1)
        history.append(entry)
        emit("epoch", **entry)
        print(
            f"epoch {epoch + 1}/{start_epoch + epochs}: loss={entry['loss']:.4f} "
            f"policy={entry['policy_loss']:.4f} value={entry['value_loss']:.4f}"
            + (f" val={entry['val_loss']:.4f}" if entry["val_loss"] is not None else "")
            + f" ({entry['seconds']:.1f}s)",
            file=sys.stderr, flush=True,
        )
        if checkpoint_dir and checkpoint_every_epochs and (epoch + 1) % checkpoint_every_epochs == 0:
            ckpt_path = os.path.join(checkpoint_dir, f"checkpoint-{step}.pt")
            save_checkpoint(ckpt_path, model, optimizer, step, epoch + 1)
            emit("checkpoint", path=ckpt_path, step=step, epoch=epoch + 1)

    if interrupted and checkpoint_dir:
        # `epoch` is the 0-based index of the unfinished epoch, which
        # is exactly the count of completed epochs — resume continues
        # from there (re-running the partial epoch from its start).
        interrupted_path = os.path.join(checkpoint_dir, "interrupted.pt")
        save_checkpoint(interrupted_path, model, optimizer, step, epoch)
        emit("checkpoint", path=interrupted_path, step=step, epoch=epoch,
             interrupted=True)
        print(
            f"interrupted at step {step} (epoch {epoch} complete); "
            f"checkpoint -> {interrupted_path}",
            file=sys.stderr, flush=True,
        )

    return {
        "history": history,
        "total_steps": step,
        "num_samples": n,
        "num_train_samples": n_tr,
        "num_val_samples": n_val,
        "best_val_loss": best_val_loss,
        "best_checkpoint": best_path,
        "optimizer": optimizer,
        "interrupted": interrupted,
        "interrupted_checkpoint": interrupted_path,
    }
