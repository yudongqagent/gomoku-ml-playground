"""Tests for long-run support: graceful interrupt, resume correctness,
data manifest, resume-safe shard numbering, and the live self-play feed."""

import json
import os
import signal
import subprocess
import sys

import numpy as np
import pytest
import torch

from gomoku_ml import selfplay
from gomoku_ml.architectures import build_architecture
from gomoku_ml.config import ConfigError, resolve_config
from gomoku_ml.interrupt import InterruptFlag
from gomoku_ml.mcts import MCTS
from gomoku_ml.train import load_checkpoint, save_checkpoint, train

REPO_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
SCRIPTS_DIR = os.path.join(REPO_ROOT, "ml", "scripts")


def _tiny_mcts():
    model = build_architecture("resnet", {"channels": 8, "blocks": 1})
    return MCTS(model)


def _tiny_data(n=64, seed=0):
    rng = np.random.default_rng(seed)
    planes = rng.integers(0, 2, size=(n, 8, 15, 15)).astype(np.float32)
    policy = rng.random((n, 225)).astype(np.float32)
    policy /= policy.sum(axis=1, keepdims=True)
    values = rng.choice([-1.0, 1.0], size=n).astype(np.float32)
    return planes, policy, values


# --- interrupt flag -------------------------------------------------------

def test_interrupt_flag_exit_codes():
    flag = InterruptFlag()
    assert flag.exit_code() == 0
    flag.requested = True
    flag.signal_name = "SIGINT"
    assert flag.exit_code() == 128 + signal.SIGINT
    flag.signal_name = "SIGTERM"
    assert flag.exit_code() == 128 + signal.SIGTERM


def test_interrupt_flag_catches_signal():
    flag = InterruptFlag().install()
    try:
        os.kill(os.getpid(), signal.SIGTERM)
        assert flag.requested
        assert flag.signal_name == "SIGTERM"
    finally:
        flag.uninstall()


# --- resume correctness ---------------------------------------------------

def test_resume_continues_rather_than_restarts(tmp_path):
    """Train 1 epoch, save a checkpoint, resume for 1 more epoch:
    the step count continues (not restarts), the epoch count
    continues, the optimiser state is restored, and the weights keep
    changing (training actually continued)."""
    planes, policy, values = _tiny_data(n=64)
    ckpt_dir = str(tmp_path / "checkpoints")
    model = build_architecture("resnet", {"channels": 8, "blocks": 1})
    summary1 = train(
        model, planes, policy, values,
        epochs=1, batch_size=16, val_fraction=0.0,
        checkpoint_dir=ckpt_dir, seed=0,
    )
    steps_epoch_1 = summary1["total_steps"]
    assert steps_epoch_1 == 4  # 64 samples / batch 16
    ckpt1 = os.path.join(ckpt_dir, "epoch1.pt")
    save_checkpoint(ckpt1, model, summary1["optimizer"], steps_epoch_1, 1)
    weights_1 = {k: v.clone() for k, v in model.state_dict().items()}

    # Resume into a fresh model + fresh optimiser.
    model2 = build_architecture("resnet", {"channels": 8, "blocks": 1})
    opt2 = torch.optim.AdamW(model2.parameters(), lr=1e-3)
    assert len(opt2.state) == 0
    payload = load_checkpoint(ckpt1, model2, opt2)
    assert int(payload["step"]) == 4
    assert int(payload["epoch"]) == 1
    assert len(opt2.state) > 0, "optimiser state was not restored"
    summary2 = train(
        model2, planes, policy, values,
        epochs=1, batch_size=16, val_fraction=0.0,
        checkpoint_dir=ckpt_dir, seed=0,
        optimizer=opt2, start_step=4, start_epoch=1,
    )
    assert summary2["total_steps"] == 8, "step count restarted instead of continuing"
    weights_2 = model2.state_dict()
    changed = any(
        not torch.equal(weights_1[k], weights_2[k]) for k in weights_1
    )
    assert changed, "weights did not change after the resumed epoch"


# --- training interrupt ---------------------------------------------------

def test_train_interrupt_writes_resumable_checkpoint(tmp_path):
    """A stop flag set mid-run stops training promptly, writes
    interrupted.pt with weights + optimiser + step, and the run
    reports interrupted=True."""
    planes, policy, values = _tiny_data(n=256)
    flag = InterruptFlag()

    def on_event(rec):
        if rec["event"] == "train_step":
            flag.requested = True  # interrupt after the first logged step

    ckpt_dir = str(tmp_path / "checkpoints")
    model = build_architecture("resnet", {"channels": 8, "blocks": 1})
    summary = train(
        model, planes, policy, values,
        epochs=5, batch_size=16, val_fraction=0.0,
        checkpoint_dir=ckpt_dir, seed=0, log_every=1,
        on_event=on_event, stop_flag=flag,
    )
    assert summary["interrupted"] is True
    interrupted_path = summary["interrupted_checkpoint"]
    assert interrupted_path is not None
    assert os.path.exists(interrupted_path)
    assert 0 < summary["total_steps"] < 5 * 16  # stopped early, after some work
    # The interrupt checkpoint loads and carries optimiser state.
    model2 = build_architecture("resnet", {"channels": 8, "blocks": 1})
    opt2 = torch.optim.AdamW(model2.parameters(), lr=1e-3)
    payload = load_checkpoint(interrupted_path, model2, opt2)
    assert int(payload["step"]) == summary["total_steps"]
    assert len(opt2.state) > 0


# --- self-play manifest / numbering / live feed ---------------------------

def test_selfplay_manifest_and_resume_numbering(tmp_path):
    """Two generation calls into the same dir: the second continues
    numbering (no overwrite) and the manifest lists all shards with
    sample counts."""
    out_dir = str(tmp_path / "data")
    mcts = _tiny_mcts()
    paths1 = selfplay.generate_games(
        mcts, num_games=2, num_simulations=4, out_dir=out_dir,
        run_name="test", max_moves=40, temperature_moves=2,
    )
    assert len(paths1) == 2
    assert [os.path.basename(p) for p in paths1] == [
        "test_game0000.npz", "test_game0001.npz"]
    manifest = selfplay.load_manifest(out_dir)
    assert manifest["games"] == 2
    assert manifest["total_samples"] == sum(s["samples"] for s in manifest["shards"])
    assert all(s["samples"] > 0 for s in manifest["shards"])

    paths2 = selfplay.generate_games(
        mcts, num_games=2, num_simulations=4, out_dir=out_dir,
        run_name="test", max_moves=40, temperature_moves=2,
    )
    assert [os.path.basename(p) for p in paths2] == [
        "test_game0002.npz", "test_game0003.npz"]
    manifest = selfplay.load_manifest(out_dir)
    assert manifest["games"] == 4
    # The first shards were not overwritten (same file list, 4 files).
    assert sorted(os.path.basename(p) for p in
                  [f for f in os.listdir(out_dir) if f.endswith(".npz")]) == [
        "test_game0000.npz", "test_game0001.npz",
        "test_game0002.npz", "test_game0003.npz"]


def test_selfplay_live_feed_written(tmp_path):
    """current_game.json is written during a run and ends in the
    finished state with the full move list and the winner."""
    out_dir = str(tmp_path / "data")
    live_path = str(tmp_path / "current_game.json")
    mcts = _tiny_mcts()
    selfplay.generate_games(
        mcts, num_games=1, num_simulations=4, out_dir=out_dir,
        run_name="test", live_path=live_path,
        max_moves=40, temperature_moves=2,
    )
    assert os.path.exists(live_path)
    with open(live_path, encoding="utf-8") as f:
        feed = json.load(f)
    assert feed["game"] == 1
    assert feed["status"] == "finished"
    assert feed["winner"] in (1, -1, 0)
    assert isinstance(feed["moves"], list) and len(feed["moves"]) > 0
    assert all(set(m) == {"x", "y"} for m in feed["moves"])
    assert feed["to_move"] in (1, -1)
    # The move list matches the saved shard.
    with np.load(os.path.join(out_dir, "test_game0000.npz")) as z:
        moves = [int(m) for m in z["moves"]]
        assert int(z["winner"]) == feed["winner"]
    assert [(m["x"], m["y"]) for m in feed["moves"]] == [
        (m % 15, m // 15) for m in moves
    ]


def test_play_game_abort_mid_game():
    """A stop flag set mid-game aborts that game: it is marked
    aborted, has no outcome, and stops after a few moves."""
    mcts = _tiny_mcts()
    flag = InterruptFlag()
    moves_seen = {"n": 0}

    def on_move(rec):
        moves_seen["n"] += 1
        if moves_seen["n"] >= 3:
            flag.requested = True

    result = selfplay.play_game(
        mcts, num_simulations=4, max_moves=40, temperature_moves=2,
        on_move=on_move, stop_flag=flag,
    )
    assert bool(result["aborted"]) is True
    assert len(result["moves"]) == 3


def test_generate_games_interrupt_stops_cleanly(tmp_path):
    """A stop flag set after the first game stops generation: the
    completed game is saved and manifested, no partial game is
    started, and the manifest lists exactly the completed games."""
    out_dir = str(tmp_path / "data")
    mcts = _tiny_mcts()
    flag = InterruptFlag()

    def on_event(rec):
        if rec["event"] == "selfplay_game":
            flag.requested = True

    paths = selfplay.generate_games(
        mcts, num_games=5, num_simulations=4, out_dir=out_dir,
        run_name="test", stop_flag=flag, on_event=on_event,
        max_moves=40, temperature_moves=2,
    )
    assert len(paths) == 1
    assert flag.requested
    manifest = selfplay.load_manifest(out_dir)
    assert manifest["games"] == 1
    assert manifest["shards"][0]["file"] == "test_game0000.npz"


def test_selfplay_top_up_semantics(tmp_path):
    """Top-up (server coordination item): re-running selfplay.py with
    the same config tops up to the target total instead of appending
    a full batch — 2 games, resume, still 2 games; raise the target
    to 3, resume, 3 games. Game numbers stay unique and dense, so the
    server's dedupe-by-game-number keeps working. Runs the real CLI
    as a subprocess (each run is a fresh process, as in production)."""
    import yaml

    cfg_path = tmp_path / "tiny.yaml"
    with open(cfg_path, "w", encoding="utf-8") as f:
        yaml.safe_dump({
            "run_name": "topup",
            "seed": 0,
            "architecture": {"name": "resnet", "params": {"channels": 8, "blocks": 1}},
            "selfplay": {"games": 2, "sims_per_move": 4, "max_moves": 40,
                         "temperature_moves": 2},
        }, f)
    run_dir = str(tmp_path / "run")

    def run_cli(extra_args):
        proc = subprocess.run(
            [sys.executable, os.path.join(SCRIPTS_DIR, "selfplay.py"),
             "--config", str(cfg_path), "--run-dir", run_dir, *extra_args],
            capture_output=True, text=True, timeout=300,
        )
        assert proc.returncode == 0, proc.stderr[-2000:]

    def shard_names():
        data_dir = os.path.join(run_dir, "data")
        return sorted(f for f in os.listdir(data_dir) if f.endswith(".npz"))

    run_cli([])
    assert shard_names() == ["topup_game0000.npz", "topup_game0001.npz"]
    # Resume with the same config: top-up, no new games.
    run_cli([])
    assert shard_names() == ["topup_game0000.npz", "topup_game0001.npz"]
    manifest = selfplay.load_manifest(os.path.join(run_dir, "data"))
    assert manifest["games"] == 2
    # Raise the target: exactly one more game, continuing numbering.
    run_cli(["--games", "3"])
    assert shard_names() == [
        "topup_game0000.npz", "topup_game0001.npz", "topup_game0002.npz",
    ]
    manifest = selfplay.load_manifest(os.path.join(run_dir, "data"))
    assert manifest["games"] == 3


# --- config ---------------------------------------------------------------

def test_config_new_eval_and_selfplay_keys():
    cfg = resolve_config({
        "run_name": "x",
        "selfplay": {"checkpoint_every_games": 5},
        "eval": {"opponents": ["random", "heuristic", "mcts"],
                 "mcts_sims": 50, "tactical": False,
                 "tactical_sims": 10, "auto": True},
    }, check_files=False)
    assert cfg["selfplay"]["checkpoint_every_games"] == 5
    assert cfg["eval"]["opponents"] == ["random", "heuristic", "mcts"]
    assert cfg["eval"]["auto"] is True
    assert cfg["eval"]["tactical"] is False
    # Defaults for configs that omit the new keys (e.g. server-written).
    cfg2 = resolve_config({"run_name": "x"}, check_files=False)
    assert cfg2["eval"]["auto"] is False
    assert cfg2["eval"]["tactical"] is True
    assert cfg2["eval"]["mcts_sims"] == 100
    assert cfg2["selfplay"]["checkpoint_every_games"] == 10


def test_config_preset_block_allowed():
    """The optional top-level preset: block (Trainer UI display) is
    accepted and passed through; a malformed one fails fast."""
    cfg = resolve_config({
        "run_name": "x",
        "preset": {"name": "Long run", "description": "d",
                   "throughput_note": "t"},
    }, check_files=False)
    assert cfg["preset"]["name"] == "Long run"
    with pytest.raises(ConfigError):
        resolve_config({"preset": "not-a-mapping"}, check_files=False)
    with pytest.raises(ConfigError):
        resolve_config({"preset": {"name": 42}}, check_files=False)


def test_config_rejects_bad_new_keys():
    with pytest.raises(ConfigError):
        resolve_config({"eval": {"opponents": ["nope"]}}, check_files=False)
    with pytest.raises(ConfigError):
        resolve_config({"eval": {"auto": "yes"}}, check_files=False)
    with pytest.raises(ConfigError):
        resolve_config({"selfplay": {"checkpoint_every_games": -1}},
                       check_files=False)


# --- CLI smoke ------------------------------------------------------------

def test_evaluate_cli_help_and_bad_checkpoint():
    py = sys.executable
    help_proc = subprocess.run(
        [py, os.path.join(SCRIPTS_DIR, "evaluate.py"), "--help"],
        capture_output=True, text=True, timeout=120,
    )
    assert help_proc.returncode == 0
    assert "--opponent-checkpoint" in help_proc.stdout
    assert "--previous" in help_proc.stdout
    assert "--tactical" in help_proc.stdout
    bad_proc = subprocess.run(
        [py, os.path.join(SCRIPTS_DIR, "evaluate.py"),
         "--checkpoint", "/nonexistent/ckpt.pt"],
        capture_output=True, text=True, timeout=120,
    )
    assert bad_proc.returncode == 2
    assert "checkpoint not found" in bad_proc.stderr
