"""Training loop for the policy+value network.

Loss: policy_weight * cross-entropy(policy logits, MCTS visit distribution)
     + value_weight * MSE(value, game outcome z)
     + threat_weight * cross-entropy(threat logits, threat class)
       (multi-task only, v38 — see ``gomoku_ml.threat``; 0 by default)
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


_MMAP_FILES = ("planes.npy", "policy.npy", "values.npy")


def _load_mmap_bundle(data_dir: str) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Load a memmap bundle built by ``ml/scripts/build_mmap_bundle.py``.

    Returns ``(planes, policy, values)`` as read-only memmaps when
    ``data_dir`` contains all three bundle files, else ``None``.
    Raises ``ValueError`` on a partial bundle (never silently mix a
    bundle with ``.npz`` shards).
    """
    present = [f for f in _MMAP_FILES if os.path.exists(os.path.join(data_dir, f))]
    if not present:
        return None
    if len(present) != len(_MMAP_FILES):
        raise ValueError(
            f"partial memmap bundle in {data_dir}: found {present}, "
            f"expected {_MMAP_FILES}"
        )
    return tuple(
        np.load(os.path.join(data_dir, f), mmap_mode="r") for f in _MMAP_FILES
    )


def load_shards(data_dirs: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load all ``.npz`` shards under the given directories (recursive)
    and concatenate them into (planes, policy, value) arrays.

    When a single data directory contains a memmap bundle
    (``planes.npy``/``policy.npy``/``values.npy``, built by
    ``ml/scripts/build_mmap_bundle.py``), the bundle is returned as
    read-only memmaps instead — same content, same order, ~200 MB of
    anonymous memory instead of the full dataset size. The training
    loop indexes the returned arrays lazily (see ``train``), so a
    memmap never materialises a full in-RAM copy.

    Peak memory is ~1x the data size, not ~2x: the outputs are
    pre-allocated from a shapes-only first pass (``np.load`` is lazy
    — reading ``.shape`` does not read array data) and filled shard
    by shard, instead of building per-shard lists and calling
    ``np.concatenate`` (which holds the lists AND the copy alive at
    once). Same return contract as before (same arrays, same order,
    same dtypes — the dtypes come from the first shard). This matters
    at 10k-game scale: the v25 training load is ~3.1 GB, and the old
    ~6.2 GB peak exceeded this VM's cell limit (OOM-killed 2026-11-16
    — see ``research/iterations/iter-25.md`` §2.4).
    """
    if len(data_dirs) == 1:
        bundle = _load_mmap_bundle(data_dirs[0])
        if bundle is not None:
            return bundle
    paths: list[str] = []
    for d in data_dirs:
        paths.extend(
            sorted(glob.glob(os.path.join(d, "**", "*.npz"), recursive=True))
        )
    if not paths:
        raise FileNotFoundError(f"no .npz shards found under {data_dirs}")
    total = 0
    plane_shape = policy_shape = value_shape = None
    plane_dtype = policy_dtype = value_dtype = None
    for path in paths:
        with np.load(path) as z:
            total += z["planes"].shape[0]
            if plane_shape is None:
                plane_shape = z["planes"].shape[1:]
                policy_shape = z["policy"].shape[1:]
                value_shape = z["value"].shape[1:]
                plane_dtype = z["planes"].dtype
                policy_dtype = z["policy"].dtype
                value_dtype = z["value"].dtype
    planes = np.empty((total,) + plane_shape, dtype=plane_dtype)
    policy = np.empty((total,) + policy_shape, dtype=policy_dtype)
    values = np.empty((total,) + value_shape, dtype=value_dtype)
    offset = 0
    for path in paths:
        with np.load(path) as z:
            n = z["planes"].shape[0]
            planes[offset : offset + n] = z["planes"]
            policy[offset : offset + n] = z["policy"]
            values[offset : offset + n] = z["value"]
            offset += n
    return planes, policy, values


def load_threat_labels(data_dirs: list[str]) -> np.ndarray:
    """Load threat-class labels aligned with :func:`load_shards`.

    Returns the int64 ``threat`` array in exactly the order
    ``load_shards`` returns the planes (same glob order, same bundle
    order), so ``threats[i]`` is the label for ``planes[i]``. From a
    memmap bundle (single dir containing ``threats.npy``) the labels
    come back as a read-only memmap; from ``.npz`` shards they are
    read from each shard's ``threat`` key. Raises ``FileNotFoundError``
    when no threat labels exist under the given directories, and
    ``ValueError`` when some shards carry them and others do not.
    """
    if len(data_dirs) == 1:
        threat_path = os.path.join(data_dirs[0], "threats.npy")
        if os.path.exists(threat_path):
            return np.load(threat_path, mmap_mode="r")
        if os.path.exists(os.path.join(data_dirs[0], "planes.npy")):
            raise ValueError(
                f"memmap bundle in {data_dirs[0]} has no threats.npy "
                f"(rebuild it from threat-labelled shards)"
            )
    paths: list[str] = []
    for d in data_dirs:
        paths.extend(
            sorted(glob.glob(os.path.join(d, "**", "*.npz"), recursive=True))
        )
    if not paths:
        raise FileNotFoundError(f"no .npz shards found under {data_dirs}")
    total = 0
    dtype = None
    for path in paths:
        with np.load(path) as z:
            if "threat" not in z:
                raise ValueError(
                    f"shard {path} has no 'threat' array (multi-task "
                    f"training needs threat labels on every shard)"
                )
            total += z["threat"].shape[0]
            if dtype is None:
                dtype = z["threat"].dtype
    threats = np.empty((total,), dtype=dtype)
    offset = 0
    for path in paths:
        with np.load(path) as z:
            n = z["threat"].shape[0]
            threats[offset : offset + n] = z["threat"]
            offset += n
    return threats


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


def _evaluate_threat(
    model: GomokuNet,
    planes: np.ndarray,
    threats: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> tuple[float, float]:
    """Mean (threat cross-entropy, threat accuracy) over a held-out
    set. Multi-task only (v38): the model must have a threat head."""
    model.eval()
    total = 0.0
    correct = 0
    n = len(planes)
    with torch.no_grad():
        for i in range(0, n, batch_size):
            x = torch.from_numpy(planes[i : i + batch_size]).to(device)
            t = torch.from_numpy(threats[i : i + batch_size]).to(device)
            _logits, _value, threat_logits = model(x, return_threat=True)
            total += float(F.cross_entropy(threat_logits, t).item()) * len(x)
            correct += int((threat_logits.argmax(dim=1) == t).sum().item())
    model.train()
    return total / n, correct / n


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
    threats: np.ndarray | None = None,
    threat_weight: float = 0.0,
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

    Multi-task (v38): when ``threats`` is given (int64 class labels
    aligned with ``planes``) and ``threat_weight`` > 0, the loss
    gains ``threat_weight * cross_entropy(threat_logits, threats)``
    and the model must have a threat head (``has_threat_head``).
    Threat labels are symmetry-invariant, so dihedral augmentation
    needs no change. Best-checkpoint selection then uses the VALUE
    val loss (the inference-relevant metric — the threat term is
    larger in magnitude and would dominate a total-loss selection);
    the summary's ``best_val_loss`` is the selection metric and
    ``best_selection`` names it ("value" or "total"). Without
    ``threats`` the loop is byte-identical in behaviour to before.
    """
    multi_task = threats is not None and threat_weight > 0.0
    if threats is not None and not multi_task:
        raise ValueError(
            "threat labels given but threat_weight is 0 — pass a "
            "positive threat_weight or no labels"
        )
    if multi_task and not getattr(model, "has_threat_head", False):
        raise ValueError(
            "multi-task training needs a model with a threat head "
            "(architecture 'resnet-multitask')"
        )
    if threats is not None and len(threats) != len(planes):
        raise ValueError(
            f"threat labels ({len(threats)}) do not match planes "
            f"({len(planes)})"
        )
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    device_t = torch.device(device)
    model.to(device_t)
    model.train()

    n = len(planes)
    perm_all = rng.permutation(n)
    n_val = int(round(n * val_fraction)) if val_fraction > 0 else 0
    val_idx, train_idx = perm_all[:n_val], perm_all[n_val:]
    # Lazy indexing: batches index the full arrays via train_idx
    # (``planes[train_idx[idx]]``) instead of materialising 95%
    # copies (``planes[train_idx]``). Same batches, same order, same
    # RNG stream — but no ~2.9 GB copy at 10k-game scale, and a
    # memmap input is never pulled fully into RAM. The per-batch
    # fancy index returns a fresh writable copy (augmentation and
    # torch.from_numpy both need one).
    n_tr = len(train_idx)

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
        epoch_loss = epoch_ploss = epoch_vloss = epoch_tloss = 0.0
        epoch_tcorrect = epoch_tcount = 0
        batches = 0
        t0 = time.time()
        for i in range(0, n_tr, batch_size):
            if stop_flag is not None and stop_flag.requested:
                interrupted = True
                break
            idx = perm[i : i + batch_size]
            gidx = train_idx[idx]
            batch_planes = planes[gidx]
            batch_policy = policy_targets[gidx]
            if augmentation:
                batch_planes, batch_policy = augment.random_augment_batch(
                    batch_planes, batch_policy, rng
                )
            x = torch.from_numpy(np.ascontiguousarray(batch_planes)).to(device_t)
            pt = torch.from_numpy(np.ascontiguousarray(batch_policy)).to(device_t)
            z = torch.from_numpy(values[gidx]).to(device_t).unsqueeze(1)
            current_lr = lr_at(step, epoch)
            for group in optimizer.param_groups:
                group["lr"] = current_lr
            if multi_task:
                t = torch.from_numpy(threats[gidx]).to(device_t)
                policy_logits, value, threat_logits = model(x, return_threat=True)
                threat_loss = F.cross_entropy(threat_logits, t)
                threat_acc = float(
                    (threat_logits.argmax(dim=1) == t).float().mean().item()
                )
            else:
                policy_logits, value = model(x)
                threat_loss = None
                threat_acc = None
            log_probs = F.log_softmax(policy_logits, dim=1)
            policy_loss = -(pt * log_probs).sum(dim=1).mean()
            value_loss = F.mse_loss(value, z)
            loss = policy_weight * policy_loss + value_weight * value_loss
            if multi_task:
                loss = loss + threat_weight * threat_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())
            epoch_ploss += float(policy_loss.item())
            epoch_vloss += float(value_loss.item())
            if multi_task:
                epoch_tloss += float(threat_loss.item())
                epoch_tcorrect += int(round(threat_acc * len(x)))
                epoch_tcount += len(x)
            batches += 1
            step += 1
            if step % log_every == 0:
                step_fields = {
                    "step": step,
                    "epoch": epoch + 1,
                    "loss": float(loss.item()),
                    "policy_loss": float(policy_loss.item()),
                    "value_loss": float(value_loss.item()),
                    "lr": current_lr,
                }
                if multi_task:
                    step_fields["threat_loss"] = float(threat_loss.item())
                    step_fields["threat_acc"] = threat_acc
                emit("train_step", **step_fields)
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
        if multi_task:
            entry["threat_loss"] = epoch_tloss / batches
            entry["threat_acc"] = epoch_tcorrect / max(1, epoch_tcount)
        if n_val > 0:
            val_loss, _, val_vloss = _evaluate_loss(
                model, planes[val_idx], policy_targets[val_idx], values[val_idx],
                batch_size, policy_weight, value_weight, device_t,
            )
            entry["val_loss"] = val_loss
            entry["val_value_loss"] = val_vloss
            if multi_task:
                val_tloss, val_tacc = _evaluate_threat(
                    model, planes[val_idx], threats[val_idx],
                    batch_size, device_t,
                )
                entry["val_threat_loss"] = val_tloss
                entry["val_threat_acc"] = val_tacc
            # Best-checkpoint selection: VALUE val loss for multi-task
            # runs (the threat term would dominate a total-loss
            # selection), total val loss otherwise (unchanged).
            best_key = val_vloss if multi_task else val_loss
            if best_val_loss is None or best_key < best_val_loss:
                best_val_loss = best_key
                if checkpoint_dir:
                    best_path = os.path.join(checkpoint_dir, "best.pt")
                    save_checkpoint(best_path, model, optimizer, step, epoch + 1)
        history.append(entry)
        emit("epoch", **entry)
        print(
            f"epoch {epoch + 1}/{start_epoch + epochs}: loss={entry['loss']:.4f} "
            f"policy={entry['policy_loss']:.4f} value={entry['value_loss']:.4f}"
            + (f" threat={entry['threat_loss']:.4f}"
               f" threat_acc={entry['threat_acc']:.4f}" if multi_task else "")
            + (f" val={entry['val_loss']:.4f}" if entry["val_loss"] is not None else "")
            + (f" val_value={entry['val_value_loss']:.4f}"
               if multi_task and entry["val_loss"] is not None else "")
            + (f" val_threat_acc={entry['val_threat_acc']:.4f}"
               if multi_task and entry["val_loss"] is not None else "")
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
        "best_selection": "value" if multi_task else "total",
        "best_checkpoint": best_path,
        "optimizer": optimizer,
        "interrupted": interrupted,
        "interrupted_checkpoint": interrupted_path,
    }
