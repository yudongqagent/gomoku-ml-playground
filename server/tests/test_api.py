"""API tests for the training server (pytest + httpx TestClient).

The ML subprocesses are faked (see _fake_run_command): the fake emits
the same stdout lines the real scripts print and creates the same
artifact files, so the full job lifecycle — queue, progress parsing,
metrics, eval, export, model registration — is exercised without
torch. No test ever touches real GitHub: release tests only dry-run,
and the real-upload path is tested only for its auth-missing error.
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.jobs import JobError, JobManager
from app.main import create_app


# -- fake ML runner ----------------------------------------------------

async def _fake_run_command(self, job, cmd, phase):
    """Stand-in for JobManager._run_command.

    Emits the stdout lines the real ML scripts print (parsed by the
    real _handle_line) and creates the artifact files the real scripts
    would write. Core ML export fails, as it does on this machine
    (coremltools not installed). Honors the cancel flag like the real
    runner does.
    """
    from app.jobs import JobCancelled

    self._check_cancelled(job)
    await asyncio.sleep(0.15)  # keep jobs alive long enough to cancel
    if job["id"] in self._cancel_requested:
        raise JobCancelled("cancelled by user")
    script = Path(cmd[1]).name
    self._handle_line(job, phase, f"$ {' '.join(cmd)}")

    def _arg(flag):
        return cmd[cmd.index(flag) + 1] if flag in cmd else None

    if script == "selfplay.py":
        out_dir = Path(_arg("--out-dir"))
        out_dir.mkdir(parents=True, exist_ok=True)
        games = int(job["config"]["selfplay"]["games"])
        # Mirror the real script's top-up resume: --games is the TOTAL
        # wanted; numbering continues after existing shards and only
        # the missing games are generated.
        existing = len(list(out_dir.glob("*.npz")))
        total = games
        for i in range(1, max(0, games - existing) + 1):
            game_no = existing + i
            (out_dir / f"{job['run_id']}_game{game_no - 1:04d}.npz").write_bytes(b"fake-shard")
            self._handle_line(
                job, phase,
                f"  game {game_no}/{total}: 42 moves, winner=Black, 0.1s",
            )
    elif script == "train.py":
        out = Path(_arg("--out"))
        out.write_bytes(b"fake-checkpoint")
        epochs = int(job["config"]["train"]["epochs"])
        for e in range(1, epochs + 1):
            self._handle_line(
                job, phase,
                f"epoch {e}/{epochs}: loss={1.0 / e:.4f} "
                f"policy={0.8 / e:.4f} value={0.2 / e:.4f} (0.1s)",
            )
    elif script == "evaluate.py":
        self._handle_line(job, phase, "  vs random: 4W-0L-0D (Elo diff None)")
        self._handle_line(job, phase, "  vs heuristic: 2W-2L-0D (Elo diff 0.0)")
    elif script == "export_onnx.py":
        out = Path(_arg("--out"))
        out.write_bytes(b"fake-onnx" * 100)
        fp16 = out.with_name(out.stem + ".fp16.onnx")
        fp16.write_bytes(b"fake-onnx-fp16" * 50)
        self._handle_line(job, phase, f"exported ONNX (opset 13) -> {out}")
    elif script == "export_coreml.py":
        raise JobError(
            "step export failed (exit 1): coremltools is not installed"
        )
    else:
        raise AssertionError(f"unexpected script in fake runner: {script}")
    self._set_phase_progress(job, phase, 1.0)
    return 0


@pytest.fixture()
def settings(tmp_path):
    return Settings(data_dir=tmp_path / "data", model_repo="owner/repo")


@pytest.fixture()
def client(settings, monkeypatch):
    monkeypatch.setattr(JobManager, "_run_command", _fake_run_command)
    app = create_app(settings=settings)
    with TestClient(app) as c:
        yield c


def _wait_for_terminal(client, job_id, timeout=30.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        resp = client.get(f"/api/jobs/{job_id}")
        assert resp.status_code == 200
        job = resp.json()
        if job["status"] in ("completed", "failed", "cancelled"):
            return job
        time.sleep(0.2)
    raise AssertionError(f"job {job_id} did not reach a terminal state in {timeout}s")


# -- health / architectures --------------------------------------------

def test_health(client):
    assert client.get("/api/health").json() == {"status": "ok"}


def test_architectures_returns_registry(client):
    resp = client.get("/api/architectures")
    assert resp.status_code == 200
    archs = resp.json()["architectures"]
    ids = [a["id"] for a in archs]
    # The real ML registry (via ml/scripts/list_architectures.py under
    # the ML interpreter): resnet, mobilenet, resnet-se.
    assert ids == ["resnet", "mobilenet", "resnet-se"]
    for a in archs:
        assert a["name"] == a["id"]
        assert a["description"]
        assert "channels" in a["params"]
        assert "blocks" in a["params"]
    resnet_se = next(a for a in archs if a["id"] == "resnet-se")
    assert "reduction" in resnet_se["params"]


# -- job validation ------------------------------------------------------

def test_create_job_validation_errors(client):
    # Unknown job type.
    resp = client.post("/api/jobs", json={"type": "nope"})
    assert resp.status_code == 422
    # Negative games.
    resp = client.post("/api/jobs", json={
        "type": "selfplay", "hyperparameters": {"games": -3},
    })
    assert resp.status_code == 422
    # Unknown architecture.
    resp = client.post("/api/jobs", json={"type": "selfplay", "architecture": "nope"})
    assert resp.status_code == 400
    # Bad arch params (validated against the registry schema: channels
    # minimum is 4; unknown keys rejected).
    resp = client.post("/api/jobs", json={
        "type": "selfplay", "arch_params": {"channels": 2},
    })
    assert resp.status_code == 400
    resp = client.post("/api/jobs", json={
        "type": "selfplay", "arch_params": {"unknown_key": 1},
    })
    assert resp.status_code == 400
    # Non-numeric arch param value is a schema error.
    resp = client.post("/api/jobs", json={
        "type": "selfplay", "arch_params": {"channels": "big"},
    })
    assert resp.status_code == 422
    # resnet-se accepts its extra 'reduction' param.
    resp = client.post("/api/jobs", json={
        "type": "selfplay", "architecture": "resnet-se",
        "arch_params": {"channels": 16, "blocks": 2, "reduction": 4},
        "hyperparameters": {"games": 1},
    })
    assert resp.status_code == 201
    job = _wait_for_terminal(client, resp.json()["id"])
    assert job["status"] == "completed", job.get("error")
    assert job["config"]["architecture"] == {
        "name": "resnet-se",
        "params": {"channels": 16, "blocks": 2, "reduction": 4},
    }


def test_get_unknown_job_404(client):
    assert client.get("/api/jobs/job-does-not-exist").status_code == 404
    assert client.post("/api/jobs/job-does-not-exist/cancel").status_code == 404


# -- full job lifecycle (fake ML) ----------------------------------------

def test_full_job_lifecycle_and_model_registration(client):
    resp = client.post("/api/jobs", json={
        "type": "full",
        "name": "test smoke",
        "architecture": "resnet",
        "arch_params": {"channels": 16, "blocks": 2},
        "hyperparameters": {
            "games": 2, "epochs": 2, "mcts_sims": 8,
            "eval_games": 4, "eval_sims": 8,
        },
    })
    assert resp.status_code == 201
    job_id = resp.json()["id"]
    assert resp.json()["status"] == "queued"

    job = _wait_for_terminal(client, job_id)
    assert job["status"] == "completed", job.get("error")
    assert job["progress"] == 1.0
    assert job["games_generated"] == 2
    assert len(job["metrics"]) == 2
    assert job["metrics"][-1]["epoch"] == 2
    assert job["eval"]["random"]["wins"] == 4
    assert job["eval"]["heuristic"]["wins"] == 2
    assert job["model_id"] is not None
    assert job["model_id"].startswith("model-")

    # Job appears in the list, newest first.
    jobs = client.get("/api/jobs").json()["jobs"]
    assert jobs[0]["id"] == job_id

    # Model is registered with artifacts + honest card.
    model_id = job["model_id"]
    detail = client.get(f"/api/models/{model_id}")
    assert detail.status_code == 200
    detail = detail.json()
    assert detail["trained_games"] == 2
    assert "onnx" in detail["formats"]
    assert "onnx-fp16" in detail["formats"]
    assert "coreml" not in detail["formats"]  # best-effort, skipped
    assert detail["onnx_url"] == f"/api/models/{model_id}/download?format=onnx"
    assert "has not been evaluated against human players" in detail["model_card"]
    assert "beats the best human" not in detail["model_card"]

    # Download works for existing formats, 404s for missing ones.
    dl = client.get(f"/api/models/{model_id}/download?format=onnx")
    assert dl.status_code == 200
    assert dl.content.startswith(b"fake-onnx")
    assert dl.headers["access-control-allow-origin"] == "*"
    assert client.get(
        f"/api/models/{model_id}/download?format=coreml"
    ).status_code == 404
    assert client.get(
        "/api/models/model-nope/download?format=onnx"
    ).status_code == 404

    # Throughput stats in the detail view (fake games: 42 moves, 0.1s).
    detail_job = client.get(f"/api/jobs/{job_id}").json()
    assert detail_job["games_per_hour"] is not None  # 2 games finished
    assert detail_job["buffer_size"] == 84  # 2 games x 42 moves
    assert detail_job["buffer_size_estimate"] is False
    assert detail_job["mcts"]["sims_per_move"] == 8
    assert detail_job["mcts"]["avg_game_length"] == 42.0
    assert detail_job["current_game"] is None  # no current_game.json yet
    assert len(detail_job["eval_history"]) == 1

    # A train-only job can now reuse the completed job's data.
    resp = client.post("/api/jobs", json={
        "type": "train", "hyperparameters": {"epochs": 1},
    })
    assert resp.status_code == 201
    job2 = _wait_for_terminal(client, resp.json()["id"])
    assert job2["status"] == "completed", job2.get("error")
    assert job2["model_id"] is not None


def test_train_job_without_data_fails_cleanly(client):
    resp = client.post("/api/jobs", json={"type": "train"})
    job = _wait_for_terminal(client, resp.json()["id"])
    assert job["status"] == "failed"
    assert "self-play data" in job["error"]


def test_cancel_running_job_marks_cancelled(client):
    resp = client.post("/api/jobs", json={
        "type": "full", "hyperparameters": {"games": 2, "epochs": 2},
    })
    job_id = resp.json()["id"]
    # Wait until the job is actually running, then cancel: the terminal
    # state must be 'cancelled', never 'failed'.
    deadline = time.time() + 5.0
    while time.time() < deadline:
        if client.get(f"/api/jobs/{job_id}").json()["status"] == "running":
            break
        time.sleep(0.05)
    cancelled = client.post(f"/api/jobs/{job_id}/cancel")
    assert cancelled.status_code == 200
    job = _wait_for_terminal(client, job_id)
    assert job["status"] == "cancelled"
    assert "cancelled" in (job["error"] or "")


def test_json_event_parsing(tmp_path):
    """Unit test: the new ML JSON events drive progress/metrics/eval."""
    from app.models_registry import ModelStore

    settings = Settings(data_dir=tmp_path / "data")
    manager = JobManager(settings, ModelStore(settings))
    job = manager.create_job({
        "type": "full", "name": "unit", "architecture": "resnet",
        "arch_params": {}, "hyperparameters": {},
        "base_checkpoint": None,
    })
    job["config"] = {"train": {"epochs": 2}, "selfplay": {"games": 2}}
    manager._handle_line(
        job, "selfplay",
        '{"event": "selfplay_game", "game": 1, "games_total": 2, '
        '"moves": 73, "winner": "Black", "seconds": 578.0}',
    )
    assert job["games_generated"] == 1
    assert abs(job["progress"] - 0.225) < 1e-9  # 0.45 weight * 1/2
    manager._handle_line(
        job, "train",
        '{"event": "epoch", "epoch": 1, "loss": 1.2345, '
        '"policy_loss": 1.0, "value_loss": 0.2345, "val_loss": 1.1}',
    )
    assert len(job["metrics"]) == 1
    assert job["metrics"][0]["loss"] == 1.2345
    assert job["metrics"][0]["val_loss"] == 1.1
    manager._handle_line(
        job, "eval",
        '{"event": "eval_result", "opponent": "random", "wins": 4, '
        '"losses": 0, "draws": 0, "elo_diff_vs_opponent": null}',
    )
    assert job["eval"]["random"]["wins"] == 4
    # Plain-text lines still parse as a fallback.
    manager._handle_line(job, "selfplay", "  game 2/2: 50 moves, winner=White, 1.0s")
    assert job["games_generated"] == 2
    # The plain-text epoch line for an epoch already recorded via the
    # JSON event is deduplicated (the scripts print both).
    manager._handle_line(
        job, "train",
        "epoch 1/2: loss=1.2345 policy=1.0000 value=0.2345 (0.1s)",
    )
    assert len(job["metrics"]) == 1
    manager._handle_line(
        job, "train",
        "epoch 2/2: loss=0.9876 policy=0.8000 value=0.1876 (0.1s)",
    )
    assert len(job["metrics"]) == 2


def test_pause_resume_conflicts(client):
    # Pause/resume on unknown jobs -> 404.
    assert client.post("/api/jobs/nope/pause").status_code == 404
    assert client.post("/api/jobs/nope/resume").status_code == 404
    # Pause/resume on a terminal job -> 409.
    resp = client.post("/api/jobs", json={
        "type": "selfplay", "hyperparameters": {"games": 1},
    })
    job_id = resp.json()["id"]
    _wait_for_terminal(client, job_id)
    assert client.post(f"/api/jobs/{job_id}/pause").status_code == 409
    assert client.post(f"/api/jobs/{job_id}/resume").status_code == 409
    # Pause on a queued job -> 409 (only running jobs pause).
    resp_a = client.post("/api/jobs", json={
        "type": "full", "hyperparameters": {"games": 2, "epochs": 2},
    })
    resp_b = client.post("/api/jobs", json={"type": "selfplay"})
    job_b = resp_b.json()["id"]
    assert client.post(f"/api/jobs/{job_b}/pause").status_code == 409
    client.post(f"/api/jobs/{job_b}/cancel")
    _wait_for_terminal(client, resp_a.json()["id"])


def test_pause_resume_real_subprocess(tmp_path):
    """Pause/resume against a real subprocess (sleep): SIGSTOP freezes
    it (state T in /proc), SIGCONT thaws it, cancel kills it."""
    import asyncio
    import signal

    from app.jobs import JobCancelled
    from app.models_registry import ModelStore

    settings = Settings(data_dir=tmp_path / "data")
    manager = JobManager(settings, ModelStore(settings))
    job = manager.create_job({
        "type": "selfplay", "name": "sig", "architecture": "resnet",
        "arch_params": {}, "hyperparameters": {},
        "base_checkpoint": None,
    })
    job["status"] = "running"

    async def scenario():
        task = asyncio.create_task(
            manager._run_command(job, ["sleep", "30"], "selfplay")
        )
        for _ in range(100):
            if manager._current_proc is not None:
                break
            await asyncio.sleep(0.05)
        proc = manager._current_proc
        assert proc is not None

        def proc_state():
            status = Path(f"/proc/{proc.pid}/status").read_text()
            return next(
                line.split()[1] for line in status.splitlines()
                if line.startswith("State:")
            )

        await manager.pause_job(job["id"])
        assert job["status"] == "paused"
        assert proc_state() == "T"  # stopped
        await manager.resume_job(job["id"])
        assert job["status"] == "running"
        assert proc_state() in ("S", "R")  # sleeping / running
        await manager.cancel_job(job["id"])
        assert job["cancelled_at"] is not None
        with pytest.raises(JobCancelled):
            await task

    asyncio.run(scenario())


def test_resume_interrupted_job_skips_completed_phases(client, settings):
    """An interrupted full job with self-play done resumes at train:
    self-play is skipped (shards complete), train/eval/export run."""
    job_id = "job-test-resume-1"
    _write_job_json(
        settings.data_dir, job_id,
        run_id="test-resume-1", status="interrupted", phase="train",
        progress=0.45, hyperparameters={"games": 1, "epochs": 1},
        completed_phases=["selfplay"],
        note="server restarted — resume from last checkpoint",
    )
    data_dir = settings.data_dir / "jobs" / job_id / "data"
    data_dir.mkdir(parents=True)
    (data_dir / "test-resume-1_game0000.npz").write_bytes(b"fake-shard")
    resp = client.post(f"/api/jobs/{job_id}/resume")
    assert resp.status_code == 200
    assert resp.json()["status"] == "queued"
    job = _wait_for_terminal(client, job_id)
    assert job["status"] == "completed", job.get("error")
    assert job["completed_phases"] == ["selfplay", "train", "eval", "export"]
    assert job["model_id"] is not None
    assert job["note"] is None
    # Eval history has the eval phase entry.
    assert len(job["eval_history"]) == 1
    assert job["eval_history"][0]["summary"]["random"] == "4W-0L-0D"


def test_resume_selfplay_tops_up_games(client, settings):
    """Resume with self-play incomplete: the re-run TOPS UP to the
    configured total (numbering continues after existing shards), it
    does not generate the full count again — games_generated ends at
    the configured total and the pre-interruption game stays in the
    stats (no reset)."""
    job_id = "job-test-resume-2"
    _write_job_json(
        settings.data_dir, job_id,
        run_id="test-resume-2", status="interrupted", phase="selfplay",
        progress=0.1, hyperparameters={"games": 2, "epochs": 1},
        games_generated=1, completed_phases=[],
        stats={"games_finished": 1, "total_moves": 42,
               "total_game_seconds": 0.1, "last_game_recorded": 1},
        note="server restarted — resume from last checkpoint",
    )
    data_dir = settings.data_dir / "jobs" / job_id / "data"
    data_dir.mkdir(parents=True)
    (data_dir / "test-resume-2_game0000.npz").write_bytes(b"fake-shard")
    resp = client.post(f"/api/jobs/{job_id}/resume")
    assert resp.status_code == 200
    job = _wait_for_terminal(client, job_id)
    assert job["status"] == "completed", job.get("error")
    # 1 pre-existing game + 1 generated on resume = 2 total (the
    # configured target); the pre-interruption game stays in the
    # stats (no reset).
    assert job["games_generated"] == 2
    assert job["stats"]["games_finished"] == 2
    assert job["stats"]["total_moves"] == 84
    assert len(list(data_dir.glob("*.npz"))) == 2


def test_cancel_records_cancelled_at(client):
    resp = client.post("/api/jobs", json={"type": "selfplay"})
    job_id = resp.json()["id"]
    cancelled = client.post(f"/api/jobs/{job_id}/cancel")
    assert cancelled.status_code == 200
    assert cancelled.json()["cancelled_at"] is not None


def test_cancel_queued_job(client):
    resp_a = client.post("/api/jobs", json={
        "type": "full", "hyperparameters": {"games": 2, "epochs": 2},
    })
    job_a = resp_a.json()["id"]
    resp_b = client.post("/api/jobs", json={"type": "selfplay"})
    job_b = resp_b.json()["id"]
    cancelled = client.post(f"/api/jobs/{job_b}/cancel")
    assert cancelled.status_code == 200
    assert cancelled.json()["status"] == "cancelled"
    # Cancelling a terminal job is a 400.
    assert client.post(f"/api/jobs/{job_b}/cancel").status_code == 400
    # Job A still completes.
    assert _wait_for_terminal(client, job_a)["status"] == "completed"


def test_sse_stream_for_completed_job(client):
    resp = client.post("/api/jobs", json={
        "type": "selfplay", "hyperparameters": {"games": 1},
    })
    job_id = resp.json()["id"]
    _wait_for_terminal(client, job_id)
    lines = []
    with client.stream("GET", f"/api/jobs/{job_id}/events") as stream:
        assert stream.status_code == 200
        assert "text/event-stream" in stream.headers["content-type"]
        for line in stream.iter_lines():
            lines.append(line)
    events = [
        json.loads(line[len("data: "):])
        for line in lines if line.startswith("data: ")
    ]
    assert any(e.get("type") == "done" and e.get("status") == "completed"
               for e in events)
    assert any(e.get("type") == "log" for e in events)


# -- presets ----------------------------------------------------------------

def test_presets_endpoint(client):
    resp = client.get("/api/presets")
    assert resp.status_code == 200
    presets = resp.json()["presets"]
    ids = [p["id"] for p in presets]
    assert "smoke" in ids
    assert "full" in ids
    # longrun-cpu.yaml landed with the ML eval/long-run follow-up.
    assert "longrun-cpu" in ids
    for p in presets:
        assert p["name"]
        assert p["config_file"] == f"ml/configs/{p['id']}.yaml"
        assert "throughput_note" in p


def test_create_job_with_preset(client):
    resp = client.post("/api/jobs", json={
        "type": "selfplay", "preset": "smoke",
        "hyperparameters": {"games": 1},
    })
    assert resp.status_code == 201
    job = _wait_for_terminal(client, resp.json()["id"])
    assert job["status"] == "completed"
    assert job["preset"] == "smoke"
    resp = client.post("/api/jobs", json={
        "type": "selfplay", "preset": "nope",
    })
    assert resp.status_code == 422


# -- eval endpoints ----------------------------------------------------------

def _completed_full_job(client):
    resp = client.post("/api/jobs", json={
        "type": "full", "hyperparameters": {
            "games": 1, "epochs": 1, "eval_games": 2, "eval_sims": 8,
        },
    })
    assert resp.status_code == 201
    job = _wait_for_terminal(client, resp.json()["id"])
    assert job["status"] == "completed", job.get("error")
    return job


def test_evaluate_job_endpoint(client):
    parent = _completed_full_job(client)
    parent_id = parent["id"]
    # The built-in eval phase already wrote an eval report.
    report = client.get(f"/api/jobs/{parent_id}/eval")
    assert report.status_code == 200
    assert report.json()["ladder"]["random"]["wins"] == 4
    assert report.json()["elo_estimate"] is None  # not produced yet
    assert report.json()["generated_at"]

    resp = client.post(f"/api/jobs/{parent_id}/evaluate",
                       json={"games": 2, "opponents": ["random"]})
    assert resp.status_code == 201
    eval_job = _wait_for_terminal(client, resp.json()["id"])
    assert eval_job["status"] == "completed", eval_job.get("error")
    assert eval_job["type"] == "eval"
    assert eval_job["parent_job_id"] == parent_id
    assert eval_job["eval"]["random"]["wins"] == 4
    # Parent eval_history gained the standalone eval entry.
    parent_after = client.get(f"/api/jobs/{parent_id}").json()
    assert len(parent_after["eval_history"]) == 2


def test_evaluate_validation(client):
    assert client.post("/api/jobs/nope/evaluate", json={}).status_code == 404
    # A selfplay job has no checkpoint to evaluate.
    resp = client.post("/api/jobs", json={
        "type": "selfplay", "hyperparameters": {"games": 1},
    })
    job_id = resp.json()["id"]
    _wait_for_terminal(client, job_id)
    assert client.post(
        f"/api/jobs/{job_id}/evaluate", json={}).status_code == 400
    assert client.get(f"/api/jobs/{job_id}/eval").status_code == 404
    # "checkpoint" opponent is not supported by this endpoint.
    parent = _completed_full_job(client)
    resp = client.post(f"/api/jobs/{parent['id']}/evaluate",
                       json={"opponents": ["checkpoint"]})
    assert resp.status_code == 400


def test_evaluate_model_endpoint(client):
    parent = _completed_full_job(client)
    model_id = parent["model_id"]
    resp = client.post(f"/api/models/{model_id}/evaluate",
                       json={"games": 2})
    assert resp.status_code == 201
    eval_job = _wait_for_terminal(client, resp.json()["id"])
    assert eval_job["status"] == "completed", eval_job.get("error")
    assert eval_job["model_id"] == model_id
    assert client.post("/api/models/nope/evaluate", json={}).status_code == 404
    # Bundled models have no checkpoint here.
    resp = client.post("/api/models/gomoku-smoke-v1/evaluate", json={})
    assert resp.status_code in (400, 404)  # 404 if manifest absent


# -- live game feed ------------------------------------------------------------

def test_current_game_polling(tmp_path):
    from app.models_registry import ModelStore

    settings = Settings(data_dir=tmp_path / "data")
    manager = JobManager(settings, ModelStore(settings))
    job = manager.create_job({
        "type": "selfplay", "name": "feed", "architecture": "resnet",
        "arch_params": {}, "hyperparameters": {},
        "base_checkpoint": None,
    })
    job_dir = manager.job_dir(job["id"])
    # Absent file: no crash, current_game stays None.
    manager._poll_current_game(job)
    assert job["current_game"] is None
    # First write: game_move published.
    (job_dir / "current_game.json").write_text(json.dumps({
        "game": 1, "moves": [{"x": 7, "y": 7}], "to_move": -1,
        "status": "playing", "winner": None,
    }))
    manager._poll_current_game(job)
    assert job["current_game"]["moves"] == [{"x": 7, "y": 7}]
    history = list(manager._event_history[job["id"]])
    assert any(e["type"] == "game_move" for e in history)
    # Same content, new mtime, no new moves: no duplicate event.
    import os
    os.utime(job_dir / "current_game.json", (2000000000, 2000000000))
    manager._poll_current_game(job)
    history2 = list(manager._event_history[job["id"]])
    assert len([e for e in history2 if e["type"] == "game_move"]) == 1
    # Finished game: game_finished published.
    (job_dir / "current_game.json").write_text(json.dumps({
        "game": 1,
        "moves": [{"x": 7, "y": 7}, {"x": 8, "y": 8}],
        "to_move": 1, "status": "finished", "winner": 1,
    }))
    manager._poll_current_game(job)
    history3 = list(manager._event_history[job["id"]])
    assert any(e["type"] == "game_finished" and e["winner"] == 1
               for e in history3)
    # Torn JSON is skipped silently.
    (job_dir / "current_game.json").write_text('{"game": 1, "moves": [')
    manager._poll_current_game(job)
    assert job["current_game"]["status"] == "finished"


# -- models ---------------------------------------------------------------

def test_models_list_shape(client):
    resp = client.get("/api/models")
    assert resp.status_code == 200
    assert isinstance(resp.json()["models"], list)


def test_model_detail_404(client):
    assert client.get("/api/models/nope").status_code == 404


# -- releases (dry-run only; never touches GitHub) -------------------------

def _make_fixture_model(settings, model_id="model-20261111-000000-abc123"):
    model_dir = settings.models_dir / model_id
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "model.pt").write_bytes(b"fake-checkpoint")
    (model_dir / "model.onnx").write_bytes(b"fake-onnx")
    (model_dir / "model_card.md").write_text("# Fixture model\n\nEval: 4W-0L-0D vs random.\n")
    meta = {
        "id": model_id,
        "name": "Fixture model",
        "run_id": "20261111-000000-abc123",
        "job_id": "job-20261111-000000-abc123",
        "architecture": "resnet",
        "created_at": "2026-11-11T00:00:00Z",
        "trained_games": 8,
        "metrics": {"final_loss": 0.5},
        "eval": {"random": {"wins": 4, "losses": 0, "draws": 0,
                            "elo_diff_vs_opponent": None}},
        "formats_detail": {
            "checkpoint": {"file": "model.pt", "size": 15},
            "onnx": {"file": "model.onnx", "size": 9},
        },
        "source": "server",
    }
    (model_dir / "model.json").write_text(json.dumps(meta, indent=2))
    return model_id


def test_release_dry_run_endpoint(client, settings):
    model_id = _make_fixture_model(settings)
    resp = client.post(f"/api/models/{model_id}/release", json={"dry_run": True})
    assert resp.status_code == 200
    result = resp.json()
    assert result["dry_run"] is True
    assert result["repo"] == "owner/repo"
    assert result["tag"] == "model-20261111-000000-abc123"
    assert result["title"] == "Fixture model"
    names = [a["name"] for a in result["assets"]]
    assert names == ["model.pt", "model.onnx", "model_card.md"]
    assert result["total_size"] == 15 + 9 + len(
        "# Fixture model\n\nEval: 4W-0L-0D vs random.\n".encode()
    )


def test_release_dry_run_default(client, settings):
    model_id = _make_fixture_model(settings)
    resp = client.post(f"/api/models/{model_id}/release", json={})
    assert resp.status_code == 200
    assert resp.json()["dry_run"] is True


def test_release_real_without_auth_fails_without_touching_github(
    client, settings, monkeypatch
):
    # No gh auth and no token in this env: the real path must refuse
    # before any GitHub call. (gh auth status returns non-zero here.)
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    model_id = _make_fixture_model(settings)
    resp = client.post(f"/api/models/{model_id}/release", json={"dry_run": False})
    assert resp.status_code == 400
    assert "no GitHub auth available" in resp.json()["detail"]


def test_release_unknown_model_404(client):
    resp = client.post("/api/models/nope/release", json={"dry_run": True})
    assert resp.status_code == 404


def test_release_without_repo_configured_is_400(tmp_path, monkeypatch):
    monkeypatch.setattr(JobManager, "_run_command", _fake_run_command)
    settings = Settings(data_dir=tmp_path / "data", model_repo="")
    _make_fixture_model(settings)
    app = create_app(settings=settings)
    with TestClient(app) as c:
        resp = c.post(
            "/api/models/model-20261111-000000-abc123/release",
            json={"dry_run": True},
        )
    assert resp.status_code == 400
    assert "no model repo configured" in resp.json()["detail"]


# -- default release repo + repo-existence warning ----------------------

def test_default_model_repo():
    assert Settings().model_repo == "yudongqagent/gomoku-ml-playground"


class _FakeResp:
    def __init__(self, status_code):
        self.status_code = status_code


def test_check_repo_exists(monkeypatch):
    import httpx
    from app import releases

    monkeypatch.setattr(httpx, "get", lambda *a, **k: _FakeResp(200))
    assert releases.check_repo_exists("owner/repo") is True
    monkeypatch.setattr(httpx, "get", lambda *a, **k: _FakeResp(404))
    assert releases.check_repo_exists("owner/repo") is False
    monkeypatch.setattr(httpx, "get", lambda *a, **k: _FakeResp(403))
    assert releases.check_repo_exists("owner/repo") is None

    def _boom(*a, **k):
        raise httpx.ConnectError("no network", request=None)

    monkeypatch.setattr(httpx, "get", _boom)
    assert releases.check_repo_exists("owner/repo") is None


def test_dry_run_warns_but_succeeds_when_repo_missing(settings, monkeypatch):
    """A missing repo is a warning in the dry-run plan, never a failure."""
    from app import releases

    model_id = _make_fixture_model(settings)
    meta = json.loads(
        (settings.models_dir / model_id / "model.json").read_text()
    )
    monkeypatch.setattr(releases, "check_repo_exists", lambda repo: False)
    plan = releases.plan_release(settings, model_id, meta, check_repo=True)
    assert plan["repo"] == "owner/repo"  # settings override, not the default
    assert plan["repo_exists"] is False
    assert "does not exist yet" in plan["repo_warning"]
    assert plan["tag"] == "model-20261111-000000-abc123"
    assert len(plan["assets"]) == 3


def test_dry_run_uses_default_repo_when_unset(tmp_path, monkeypatch):
    from app import releases

    settings = Settings(data_dir=tmp_path / "data")
    model_id = _make_fixture_model(settings)
    meta = json.loads(
        (settings.models_dir / model_id / "model.json").read_text()
    )
    monkeypatch.setattr(releases, "check_repo_exists", lambda repo: None)
    plan = releases.plan_release(settings, model_id, meta, check_repo=True)
    assert plan["repo"] == "yudongqagent/gomoku-ml-playground"


# -- restart recovery --------------------------------------------------------

def _write_job_json(data_dir, job_id, **overrides):
    jobs_dir = data_dir / "jobs" / job_id
    jobs_dir.mkdir(parents=True, exist_ok=True)
    record = {
        "id": job_id, "run_id": "old", "name": "old", "type": "full",
        "architecture": "resnet", "arch_params": {}, "hyperparameters": {},
        "base_checkpoint": None, "status": "running", "phase": "train",
        "progress": 0.5, "created_at": "2026-11-11T00:00:00Z",
        "started_at": "2026-11-11T00:01:00Z", "finished_at": None,
        "config": None, "metrics": [], "games_generated": 3, "eval": {},
        "logs": [], "error": None, "model_id": None, "artifacts": {},
    }
    record.update(overrides)
    (jobs_dir / "job.json").write_text(json.dumps(record))
    return record


def test_recover_marks_running_and_paused_jobs_interrupted(tmp_path):
    data_dir = tmp_path / "data"
    _write_job_json(data_dir, "job-running")
    _write_job_json(data_dir, "job-paused", status="paused")
    _write_job_json(data_dir, "job-done", status="completed")
    settings = Settings(data_dir=data_dir)
    from app.models_registry import ModelStore

    manager = JobManager(settings, ModelStore(settings))
    manager.recover()
    running = manager.get_job("job-running")
    assert running["status"] == "interrupted"
    assert "resume from last checkpoint" in running["note"]
    assert running["finished_at"] is None  # resumable, not finished
    assert manager.get_job("job-paused")["status"] == "interrupted"
    assert manager.get_job("job-done")["status"] == "completed"


# -- enriched eval shape (regression) -----------------------------------

_ENRICHED_EVAL = {
    "checkpoint": "final.pt",
    "generated_at": "2026-11-12T00:00:00",
    "ladder": {
        "random": {"wins": 30, "losses": 10, "draws": 0,
                   "elo_diff_vs_opponent": 190.8},
        "heuristic": {"wins": 0, "losses": 40, "draws": 0,
                      "elo_diff_vs_opponent": None},
    },
    "elo_estimate": {"estimate": -609.2, "interval": [-731.0, -487.3],
                     "interval_inconsistent": False},
    "tactical": {"total": 20, "passed": 0, "pass_rate": 0.0},
    "tactical_baseline_heuristic": {"total": 20, "passed": 20},
    "verdict": {"tier": "below heuristic", "label": "Below heuristic",
                "meets_good_level": False,
                "explanation": "Win rate vs heuristic is 0.00."},
    "opponent_checkpoint": None,
    "ts_match": None,
}


def test_eval_history_handles_enriched_eval_shape(tmp_path):
    """Regression: _record_eval_history must not crash on the enriched
    eval.json (ladder nested under 'ladder', plus verdict/elo/tactical
    keys that are not per-opponent dicts)."""
    from app.models_registry import ModelStore

    settings = Settings(data_dir=tmp_path / "data")
    manager = JobManager(settings, ModelStore(settings))
    job = manager.create_job({
        "type": "full", "name": "enriched", "architecture": "resnet",
        "arch_params": {}, "hyperparameters": {},
        "base_checkpoint": None,
    })
    job["eval"] = dict(_ENRICHED_EVAL)
    manager._record_eval_history(job, "final.pt")
    entry = job["eval_history"][-1]
    assert entry["summary"] == {"random": "30W-10L-0D",
                                "heuristic": "0W-40L-0D"}
    assert entry["verdict"]["tier"] == "below heuristic"
    assert entry["elo_estimate"]["estimate"] == -609.2
    # Legacy flat shape still works.
    job["eval"] = {"random": {"wins": 4, "losses": 0, "draws": 0}}
    manager._record_eval_history(job, "best.pt")
    assert job["eval_history"][-1]["summary"] == {"random": "4W-0L-0D"}


def test_render_model_card_handles_enriched_eval_shape():
    """Regression: the fallback model card must render the enriched
    eval.json (nested ladder, verdict dict, elo dict) without crashing
    or printing raw dicts."""
    from app.models_registry import render_model_card

    meta = {
        "id": "model-x", "name": "x", "run_id": "r1", "job_id": "job-x",
        "created_at": "2026-11-12", "architecture": "resnet",
        "trained_games": 8,
        "metrics": {"final_loss": 5.7, "final_policy_loss": 5.4,
                    "final_value_loss": 0.3, "epochs": 3},
        "eval": dict(_ENRICHED_EVAL),
        "formats_detail": {"onnx": {"file": "model.onnx", "size": 123}},
    }
    job = {"config": {
        "model": {"channels": 16, "blocks": 2},
        "selfplay": {"games": 8, "sims_per_move": 32, "temperature": 1.0},
        "train": {"epochs": 3, "batch_size": 32, "lr": 0.001},
    }}
    card = render_model_card(meta, job)
    assert "vs random: 30W-10L-0D" in card
    assert "vs heuristic: 0W-40L-0D" in card
    assert "Below heuristic" in card
    assert "-609" in card
    assert "Tactical test suite: 0/20 passed" in card
    assert "{'tier'" not in card and '"tier"' not in card
