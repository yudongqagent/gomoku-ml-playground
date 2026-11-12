"""Tests for run tracking (runs.py)."""

import json
import os
import re

from gomoku_ml import runs
from gomoku_ml.config import resolve_config


def test_new_run_id_format():
    run_id = runs.new_run_id("smoke")
    assert re.fullmatch(r"\d{8}-\d{6}-smoke", run_id), run_id


def test_resolve_run_dir_explicit_and_default(tmp_path):
    explicit = runs.resolve_run_dir(str(tmp_path), "/tmp/my-run", "smoke")
    assert explicit == "/tmp/my-run"
    default = runs.resolve_run_dir(str(tmp_path), None, "smoke")
    assert default.startswith(os.path.join(str(tmp_path), "runs", ""))
    assert default.endswith("-smoke")


def test_run_logging_and_artifacts(tmp_path):
    run_dir = str(tmp_path / "20261112-000000-test")
    cfg = resolve_config({"run_name": "test"})
    with runs.Run(run_dir) as run:
        assert os.path.isdir(run.data_dir)
        assert os.path.isdir(run.checkpoint_dir)
        run.save_config(cfg)
        run.log_event("epoch", epoch=1, loss=1.23, policy_loss=1.0, value_loss=0.23)
        run.log_event("selfplay_game", game=1, games_total=2, moves=42, winner="Black")
        run.info("hello human")
        run.write_eval({"random": {"wins": 4, "losses": 0, "draws": 0, "elo_diff_vs_opponent": None}})
    # metrics.jsonl: one parseable JSON object per line.
    with open(os.path.join(run_dir, "metrics.jsonl")) as f:
        records = [json.loads(line) for line in f if line.strip()]
    assert [r["event"] for r in records] == ["epoch", "selfplay_game"]
    assert records[0]["loss"] == 1.23
    assert all("timestamp" in r for r in records)
    # Config snapshot round-trips.
    with runs.Run(run_dir) as run:
        loaded = run.load_config()
    assert loaded["run_name"] == "test"
    assert loaded["architecture"]["name"] == "resnet"
    # Eval round-trips.
    with runs.Run(run_dir) as run:
        assert run.load_eval()["random"]["wins"] == 4
    # Human log exists.
    assert "hello human" in open(os.path.join(run_dir, "run.log")).read()


def test_model_card_generation(tmp_path):
    run_dir = str(tmp_path / "20261112-000000-test")
    cfg = resolve_config({
        "run_name": "test",
        "architecture": {"name": "mobilenet", "params": {"channels": 32, "blocks": 4}},
    })
    with runs.Run(run_dir) as run:
        run.save_config(cfg)
        run.log_event("epoch", epoch=1, loss=1.5, policy_loss=1.2, value_loss=0.3,
                      val_loss=1.4, num_samples=100)
        run.write_eval({"heuristic": {"wins": 3, "losses": 1, "draws": 0,
                                      "elo_diff_vs_opponent": 150.0}})
    card_path = runs.generate_model_card(run_dir)
    card = open(card_path).read()
    assert "mobilenet" in card
    assert "32" in card  # channels
    assert "vs heuristic: 3W-1L-0D" in card
    # Honesty note present.
    assert "says almost nothing about absolute strength" in card
    # Regenerating is safe (idempotent).
    runs.generate_model_card(run_dir)
    assert open(card_path).read() == card
