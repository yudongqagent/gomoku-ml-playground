"""Job queue + runner for training jobs.

One job runs at a time (FIFO, concurrency=1 — a deliberate v1 choice;
training is CPU/GPU-bound and parallel runs would thrash each other).
Each job runs the ML pipeline scripts (ml/scripts/*.py) as subprocesses
under the ML venv's interpreter, streams their stdout as log/progress
events, and on completion exports + registers a model.

Job lifecycle:
    queued -> running -> completed | failed | cancelled
    running <-> paused            (SIGSTOP / SIGCONT on the subprocess)
    running/paused --server restart--> interrupted --resume--> queued

Pause freezes the subprocess in place (SIGSTOP: alive, no CPU); resume
thaws it (SIGCONT). A failed or interrupted job can be resumed too: it
re-enters the queue and re-runs from the phase it was interrupted in,
reusing completed phases' artifacts (see _execute resume mode and
server/README.md "Resume semantics").

On server restart, jobs left running/paused are marked ``interrupted``
with a note pointing at POST /api/jobs/{id}/resume. They are NOT
auto-resumed: the operator decides (a restart may mean the machine was
repurposed, and a blind resume could burn hours of CPU).

Every event is persisted: job.json is rewritten on every event (atomic
temp-file + rename), job.log gets every stdout line, and metrics.jsonl
gets every server-normalized event (the ML run also appends its own
native events to the same file — server lines carry "source": "server"
and a "type" key instead of "event", so the ML model-card generator,
which filters on "event", is unaffected). Persisted job.json is the
source of truth: recover() reloads it on startup, and get_job() falls
back to disk when a job isn't in memory.

Progress parsing: the ML scripts emit JSON event lines on stdout (one
JSON object per line, each with an ``"event"`` key — see
``gomoku_ml/runs.py``) AND human-readable plain-text lines. The parser
accepts both: JSON events (``selfplay_game``, ``epoch``, ``train_step``,
``eval_result``, ``run_finished``, ...) drive progress/metrics/eval,
and plain lines are stored as log events and matched against regexes
for games / epochs / eval results as a fallback.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import secrets
import signal
import subprocess
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import yaml

from .config import ML_DIR, SERVER_DIR, Settings
from .models_registry import ModelStore

log = logging.getLogger(__name__)

TERMINAL_STATUSES = {"completed", "failed", "cancelled"}

# Phase weights (fraction of overall progress) per job type.
PHASE_WEIGHTS: dict[str, dict[str, float]] = {
    "selfplay": {"selfplay": 1.0},
    "train": {"train": 0.85, "eval": 0.08, "export": 0.07},
    "full": {"selfplay": 0.45, "train": 0.40, "eval": 0.08, "export": 0.07},
    "eval": {"eval": 1.0},
}

# Phases in run order per job type (for resume: re-run from the
# interrupted phase, reusing completed phases' artifacts).
PHASE_ORDER: dict[str, list[str]] = {
    "selfplay": ["selfplay"],
    "train": ["train", "eval", "export"],
    "full": ["selfplay", "train", "eval", "export"],
    "eval": ["eval"],
}

# Plain-text line formats printed by the ML scripts (gomoku_ml/*).
# Game lines look like: "  game 1/4: 73 moves, winner=Black, 578.0s"
_RE_GAME = re.compile(
    r"game (\d+)/(\d+)(?:: (\d+) moves, winner=[A-Za-z]+, ([0-9.]+)s)?"
)
_RE_STEP = re.compile(r"step (\d+)/(\d+) loss=([0-9.]+)")
_RE_EPOCH = re.compile(
    r"epoch (\d+)/(\d+): loss=([0-9.]+) policy=([0-9.]+) value=([0-9.]+)"
)
_RE_EVAL = re.compile(
    r"vs (\w+): (\d+)W-(\d+)L-(\d+)D \(Elo diff (None|-?[0-9.]+)\)"
)

_LOG_TAIL = 100
_EVENT_HISTORY = 500


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _new_run_id() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"{stamp}-{secrets.token_hex(3)}"


def _ensure_job_defaults(job: dict) -> None:
    """Fill fields missing from job.json files written by older server
    versions, so restarts stay backward compatible."""
    job.setdefault("preset", None)
    job.setdefault("cancelled_at", None)
    job.setdefault("resumed_at", None)
    job.setdefault("note", None)
    job.setdefault("eval_history", [])
    job.setdefault("parent_job_id", None)
    job.setdefault("eval_checkpoint", None)
    job.setdefault("completed_phases", [])
    job.setdefault("resume", False)
    job.setdefault("current_game", None)
    job.setdefault("stats", {
        "games_finished": 0, "total_moves": 0, "total_game_seconds": 0.0,
    })
    job.setdefault("logs", [])
    job.setdefault("metrics", [])
    job.setdefault("eval", {})
    job.setdefault("artifacts", {})


class JobError(Exception):
    """A job step failed; carries the message recorded on the job."""


class JobCancelled(JobError):
    """The job was cancelled by the user (maps to status 'cancelled',
    not 'failed')."""


# Torch intra-op threads per phase, passed as TORCH_THREADS to the ML
# subprocesses (ml/scripts/common.py honours the env var). Self-play and
# eval run one tiny forward pass at a time, so intra-op parallelism is
# pure synchronisation overhead on this small VM — 1 thread is fastest
# there. Training with real batch sizes benefits from 2.
TORCH_THREADS: dict[str, int] = {
    "selfplay": 1,
    "train": 2,
    "eval": 1,
    "export": 1,
}


class JobManager:
    def __init__(self, settings: Settings, model_store: ModelStore) -> None:
        self.settings = settings
        self.model_store = model_store
        self.jobs: dict[str, dict[str, Any]] = {}
        self._queue: asyncio.Queue[str] = asyncio.Queue()
        self._worker_task: asyncio.Task | None = None
        self._current_proc: asyncio.subprocess.Process | None = None
        self._current_job_id: str | None = None
        self._cancel_requested: set[str] = set()
        self._event_history: dict[str, deque] = {}
        self._subscribers: dict[str, set[asyncio.Queue]] = {}
        self._lock = asyncio.Lock()
        # Live game feed state (per job): last seen current_game.json
        # mtime + the last published (game, moves, status) triple.
        self._game_feed_mtime: dict[str, float] = {}
        self._game_feed_last: dict[str, tuple] = {}
        self._watcher_task: asyncio.Task | None = None

    # -- paths ---------------------------------------------------------

    def job_dir(self, job_id: str) -> Path:
        return self.settings.jobs_dir / job_id

    # -- persistence ---------------------------------------------------

    def _save(self, job: dict, force: bool = False) -> None:
        """Persist job.json (atomic temp-file + rename) on every call.

        Deliberately unthrottled: persisted job.json is the source of
        truth for restart recovery, so every event rewrites it. Event
        rates are low (one line per game/step at most), and job.json is
        small (logs are capped at 100 lines; the full log is job.log).
        """
        path = self.job_dir(job["id"]) / "job.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(job, indent=2), encoding="utf-8")
        tmp.replace(path)

    def _append_log(self, job: dict, line: str) -> None:
        with open(self.job_dir(job["id"]) / "job.log", "a", encoding="utf-8") as f:
            f.write(line + "\n")
        job["logs"].append(line)
        if len(job["logs"]) > _LOG_TAIL:
            job["logs"] = job["logs"][-_LOG_TAIL:]

    # -- events (SSE) --------------------------------------------------

    def publish(self, job_id: str, event: dict) -> None:
        event = {"job_id": job_id, **event}
        history = self._event_history.setdefault(job_id, deque(maxlen=_EVENT_HISTORY))
        history.append(event)
        for queue in list(self._subscribers.get(job_id, ())):
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:
                pass
        # Persist every event to metrics.jsonl (server-normalized shape,
        # "source": "server"). The ML run appends its own native events
        # to the same file; both writers append whole lines. A logging
        # failure must never break the job, so errors are swallowed.
        try:
            record = {
                "type": event.get("type"),
                "timestamp": _now_iso(),
                "source": "server",
                **{k: v for k, v in event.items()
                   if k not in ("type", "job_id")},
            }
            with open(self.job_dir(job_id) / "metrics.jsonl",
                      "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
        except OSError as exc:
            log.debug("could not append to metrics.jsonl for %s: %s", job_id, exc)

    def subscribe(self, job_id: str) -> tuple[list[dict], asyncio.Queue]:
        """Return (replayed history, live queue) for an SSE subscriber."""
        history = list(self._event_history.get(job_id, ()))
        queue: asyncio.Queue = asyncio.Queue(maxsize=1000)
        self._subscribers.setdefault(job_id, set()).add(queue)
        return history, queue

    def unsubscribe(self, job_id: str, queue: asyncio.Queue) -> None:
        self._subscribers.get(job_id, set()).discard(queue)

    # -- lifecycle -----------------------------------------------------

    async def start(self) -> None:
        self.recover()
        self._worker_task = asyncio.create_task(self._worker_loop())

    async def stop(self) -> None:
        if self._worker_task:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
        if self._current_proc and self._current_proc.returncode is None:
            self._current_proc.terminate()
            try:
                await asyncio.wait_for(self._current_proc.wait(), timeout=5)
            except asyncio.TimeoutError:
                self._current_proc.kill()

    def recover(self) -> None:
        """Load persisted jobs after a (re)start.

        Jobs left running/paused when the server went down are marked
        ``interrupted`` — their subprocesses are gone, but their
        checkpoints and partial progress survive on disk, so they can
        be resumed via POST /api/jobs/{id}/resume. They are NOT
        auto-resumed: the operator decides (a restart may mean the
        machine was repurposed). Queued jobs are re-queued.
        """
        jobs_dir = self.settings.jobs_dir
        if not jobs_dir.exists():
            return
        for path in sorted(jobs_dir.glob("*/job.json")):
            try:
                job = json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError) as exc:
                log.warning("skipping unreadable job file %s: %s", path, exc)
                continue
            _ensure_job_defaults(job)
            if job["status"] in ("running", "paused"):
                job["status"] = "interrupted"
                job["note"] = (
                    "server restarted — resume from last checkpoint via "
                    f"POST /api/jobs/{job['id']}/resume"
                )
                self._save(job, force=True)
            self.jobs[job["id"]] = job
            if job["status"] == "queued":
                self._queue.put_nowait(job["id"])

    def create_job(self, payload: dict) -> dict:
        """Create a job record + job dir, enqueue it, return the record."""
        run_id = _new_run_id()
        job_id = f"job-{run_id}"
        job: dict[str, Any] = {
            "id": job_id,
            "run_id": run_id,
            "name": payload.get("name") or f"{payload['type']} {run_id}",
            "type": payload["type"],
            "architecture": payload.get("architecture", "resnet"),
            "arch_params": payload.get("arch_params", {}),
            "hyperparameters": payload.get("hyperparameters", {}),
            "preset": payload.get("preset"),
            "base_checkpoint": payload.get("base_checkpoint"),
            "status": "queued",
            "phase": None,
            "progress": 0.0,
            "created_at": _now_iso(),
            "started_at": None,
            "finished_at": None,
            "cancelled_at": None,
            "resumed_at": None,
            "note": None,
            "config": None,
            "metrics": [],
            "games_generated": 0,
            "eval": {},
            "eval_history": [],
            "logs": [],
            "error": None,
            "model_id": payload.get("model_id"),
            "parent_job_id": payload.get("parent_job_id"),
            "eval_checkpoint": payload.get("eval_checkpoint"),
            "artifacts": {},
            "completed_phases": [],
            "resume": False,
            "current_game": None,
            "stats": {
                "games_finished": 0,
                "total_moves": 0,
                "total_game_seconds": 0.0,
            },
        }
        self.job_dir(job_id).mkdir(parents=True, exist_ok=True)
        self._save(job, force=True)
        self.jobs[job_id] = job
        self._queue.put_nowait(job_id)
        log.info("created job %s (%s)", job_id, job["type"])
        return job

    def get_job(self, job_id: str) -> dict | None:
        """Job by id, from memory, falling back to the disk record."""
        job = self.jobs.get(job_id)
        if job is not None:
            return job
        path = self.job_dir(job_id) / "job.json"
        if not path.exists():
            return None
        try:
            job = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None
        _ensure_job_defaults(job)
        self.jobs[job_id] = job
        return job

    def list_jobs(self) -> list[dict]:
        return sorted(
            self.jobs.values(), key=lambda j: j["created_at"], reverse=True
        )

    def detail(self, job: dict) -> dict:
        """Full job detail for GET /api/jobs/{id}: the record plus
        computed throughput stats (games/hour, buffer size, MCTS)."""
        return {**job, **self.throughput_stats(job)}

    def throughput_stats(self, job: dict) -> dict:
        """Live throughput stats, computed from the record + wall clock.

        games_per_hour is null until >= 2 games (one game gives no
        rate). buffer_size is the exact sample count once games have
        finished (each move is one training sample); before that it is
        an estimate (games x avg game length) or null. Elapsed time
        runs from first start and includes paused time.
        """
        stats = job.get("stats") or {}
        games_finished = int(stats.get("games_finished", 0))
        total_moves = int(stats.get("total_moves", 0))
        total_seconds = float(stats.get("total_game_seconds", 0.0))
        avg_game_length = (
            total_moves / games_finished if games_finished else None
        )
        avg_move_time_ms = (
            total_seconds * 1000.0 / total_moves if total_moves else None
        )
        # The ML selfplay_game events carry their own running rate
        # (games/hour since generation started) — prefer it when present,
        # since the wall-clock rate includes startup + pause time.
        games_per_hour: float | None = stats.get("games_per_hour_ml")
        started_at = job.get("started_at")
        games = int(job.get("games_generated", 0))
        if games_per_hour is None and games >= 2 and started_at:
            try:
                start = datetime.fromisoformat(started_at.replace("Z", "+00:00"))
                end_raw = job.get("finished_at")
                end = (
                    datetime.fromisoformat(end_raw.replace("Z", "+00:00"))
                    if end_raw
                    else datetime.now(timezone.utc)
                )
                hours = (end - start).total_seconds() / 3600.0
                if hours > 0:
                    games_per_hour = round(games / hours, 1)
            except ValueError:
                pass
        # Exact buffer size from the ML data manifest when present
        # (data/manifest.json totals the samples in every shard on disk
        # — the source of truth for the replay buffer).
        manifest_samples = self._manifest_total_samples(job)
        if manifest_samples is not None:
            buffer_size: int | None = manifest_samples
            buffer_size_estimate = False
        elif total_moves:
            buffer_size = total_moves
            buffer_size_estimate = False
        elif games and avg_game_length:
            buffer_size = int(round(games * avg_game_length))
            buffer_size_estimate = True
        else:
            buffer_size = None
            buffer_size_estimate = True
        cfg = job.get("config") or {}
        sims = (cfg.get("selfplay") or {}).get("sims_per_move")
        return {
            "games_per_hour": games_per_hour,
            "buffer_size": buffer_size,
            "buffer_size_estimate": buffer_size_estimate,
            "mcts": {
                "sims_per_move": sims,
                "avg_game_length": (
                    round(avg_game_length, 1)
                    if avg_game_length is not None else None
                ),
                "avg_move_time_ms": (
                    round(avg_move_time_ms, 1)
                    if avg_move_time_ms is not None else None
                ),
            },
        }

    def _manifest_total_samples(self, job: dict) -> int | None:
        """Total training samples from data/manifest.json, if present."""
        path = self.job_dir(job["id"]) / "data" / "manifest.json"
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, ValueError):
            return None
        total = manifest.get("total_samples") if isinstance(manifest, dict) else None
        return int(total) if isinstance(total, int) else None

    def summary(self, job: dict) -> dict:
        return {
            "id": job["id"],
            "name": job["name"],
            "type": job["type"],
            "architecture": job["architecture"],
            "status": job["status"],
            "phase": job["phase"],
            "progress": job["progress"],
            "created_at": job["created_at"],
            "started_at": job["started_at"],
            "finished_at": job["finished_at"],
            "games_generated": job["games_generated"],
            "model_id": job["model_id"],
            "parent_job_id": job.get("parent_job_id"),
            "error": job["error"],
            "note": job.get("note"),
        }

    async def cancel_job(self, job_id: str) -> dict | None:
        """Cancel a queued, running or paused job. Returns None if unknown.

        Records cancelled_at. A paused job is SIGCONTed before the kill
        so the SIGTERM is actually delivered (a stopped process can't
        act on signals until it continues).
        """
        job = self.get_job(job_id)
        if job is None:
            return None
        if job["status"] in TERMINAL_STATUSES:
            raise JobError(f"job is already {job['status']}")
        if job["status"] in ("interrupted",):
            job["status"] = "cancelled"
            job["finished_at"] = _now_iso()
            job["cancelled_at"] = _now_iso()
            job["error"] = "cancelled while interrupted"
            job["note"] = None
            self._save(job, force=True)
            self.publish(job_id, {"type": "done", "status": "cancelled"})
            return job
        if job["status"] == "queued":
            # The worker skips cancelled jobs when it dequeues them.
            job["status"] = "cancelled"
            job["finished_at"] = _now_iso()
            job["cancelled_at"] = _now_iso()
            job["error"] = "cancelled before starting"
            self._save(job, force=True)
            self.publish(job_id, {"type": "done", "status": "cancelled"})
            return job
        # Running or paused: the read loop notices the flag and kills
        # the process (SIGCONT first if paused — see _kill_proc).
        self._cancel_requested.add(job_id)
        proc = self._current_proc
        if proc is not None and proc.returncode is None:
            await self._kill_proc(proc)
        job["cancelled_at"] = _now_iso()
        self._save(job, force=True)
        return job

    async def pause_job(self, job_id: str) -> dict | None:
        """Pause a running job: SIGSTOP the subprocess (alive, no CPU).

        Returns None if unknown. Raises JobError for anything that
        isn't currently running (the endpoint maps it to 409).
        """
        job = self.get_job(job_id)
        if job is None:
            return None
        if job["status"] != "running":
            raise JobError(
                f"only a running job can be paused (job is {job['status']})"
            )
        job["status"] = "paused"
        self._save(job, force=True)
        proc = self._current_proc
        if proc is not None and proc.returncode is None:
            try:
                os.kill(proc.pid, signal.SIGSTOP)
            except ProcessLookupError:
                pass
        self.publish(job_id, {"type": "paused", "status": "paused"})
        log.info("paused job %s", job_id)
        return job

    async def resume_job(self, job_id: str) -> dict | None:
        """Resume a job.

        * paused -> running via SIGCONT (same process continues).
        * interrupted/failed -> re-queued; the worker re-runs it from
          the phase it was interrupted in, reusing completed phases'
          artifacts (see _execute resume mode).
        Returns None if unknown. Raises JobError for running, queued,
        completed or cancelled jobs (the endpoint maps it to 409).
        """
        job = self.get_job(job_id)
        if job is None:
            return None
        if job["status"] == "paused":
            proc = self._current_proc
            if proc is not None and proc.returncode is None:
                try:
                    os.kill(proc.pid, signal.SIGCONT)
                except ProcessLookupError:
                    pass
            job["status"] = "running"
            job["resumed_at"] = _now_iso()
            self._save(job, force=True)
            self.publish(job_id, {"type": "resumed", "status": "running"})
            log.info("resumed paused job %s", job_id)
            return job
        if job["status"] in ("interrupted", "failed"):
            job["status"] = "queued"
            job["resume"] = True
            job["error"] = None
            job["note"] = None
            job["finished_at"] = None
            job["resumed_at"] = _now_iso()
            self._save(job, force=True)
            self._queue.put_nowait(job_id)
            self.publish(job_id, {"type": "resumed", "status": "queued"})
            log.info("re-queued %s job %s for resume",
                     job["type"], job_id)
            return job
        raise JobError(
            f"only a paused, interrupted or failed job can be resumed "
            f"(job is {job['status']})"
        )

    # -- worker --------------------------------------------------------

    async def _worker_loop(self) -> None:
        while True:
            job_id = await self._queue.get()
            job = self.jobs.get(job_id)
            if job is None or job["status"] != "queued":
                continue  # cancelled while waiting, or removed
            try:
                await self._run_job(job)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # never let the worker die
                log.exception("job %s crashed the runner: %s", job_id, exc)
                if job["status"] == "running":
                    self._finish(job, "failed", error=str(exc))

    async def _run_job(self, job: dict) -> None:
        resume = bool(job.get("resume"))
        job["resume"] = False
        job["status"] = "running"
        if not job.get("started_at"):
            job["started_at"] = _now_iso()
        job["finished_at"] = None
        self._current_job_id = job["id"]
        self._save(job, force=True)
        self.publish(job["id"], {
            "type": "progress", "progress": job["progress"],
            "resumed": resume,
        })
        self._watcher_task = asyncio.create_task(self._watch_current_game(job))
        try:
            await self._execute(job, resume=resume)
            self._finish(job, "completed")
        except asyncio.CancelledError:
            raise
        except JobCancelled as exc:
            self._finish(job, "cancelled", error=str(exc))
        except JobError as exc:
            self._finish(job, "failed", error=str(exc))
        finally:
            if self._watcher_task is not None:
                self._watcher_task.cancel()
                try:
                    await self._watcher_task
                except asyncio.CancelledError:
                    pass
                self._watcher_task = None
            self._current_job_id = None
            self._current_proc = None
            self._cancel_requested.discard(job["id"])

    def _finish(self, job: dict, status: str, error: str | None = None) -> None:
        job["status"] = status
        job["finished_at"] = _now_iso()
        if status == "completed":
            job["progress"] = 1.0
        if error:
            job["error"] = error
        self._save(job, force=True)
        if status in ("completed", "cancelled"):
            self.publish(job["id"], {
                "type": "done", "status": status,
                "model_id": job.get("model_id"),
            })
        else:
            self.publish(job["id"], {
                "type": "error", "status": status,
                "message": error or status,
            })

    def _check_cancelled(self, job: dict) -> None:
        if job["id"] in self._cancel_requested:
            raise JobCancelled("cancelled by user")

    # -- config --------------------------------------------------------

    def _resolve_config(self, job: dict) -> dict:
        """Build the resolved training YAML for this job.

        Starts from the preset config (ml/configs/<preset>.yaml, or
        smoke.yaml when no preset is given) and applies the request's
        architecture / hyperparameters on top. The resolved config is
        written to the job dir as config.yaml.

        Architecture goes into the ``architecture: {name, params}``
        section the current ML scripts read (gomoku_ml/architectures);
        the legacy ``model: {channels, blocks}`` key is kept too for
        older scripts (the config validator explicitly allows it).
        """
        preset = job.get("preset") or "smoke"
        base_path = ML_DIR / "configs" / f"{preset}.yaml"
        if not base_path.exists():
            raise JobError(f"preset config not found: {base_path}")
        with open(base_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        hp = job.get("hyperparameters") or {}
        arch = job.get("arch_params") or {}
        cfg["run_name"] = job["run_id"]
        arch_name = job.get("architecture", "resnet")
        base_params = (cfg.get("architecture") or {}).get("params") or {}
        params = {**base_params, **{k: int(v) for k, v in arch.items()}}
        cfg["architecture"] = {"name": arch_name, "params": params}
        cfg.setdefault("model", {}).update({
            "channels": int(params.get("channels", 16)),
            "blocks": int(params.get("blocks", 2)),
        })
        sp = cfg.setdefault("selfplay", {})
        if hp.get("games") is not None:
            sp["games"] = int(hp["games"])
        if hp.get("mcts_sims") is not None:
            sp["sims_per_move"] = int(hp["mcts_sims"])
        if hp.get("temperature") is not None:
            sp["temperature"] = float(hp["temperature"])
        tr = cfg.setdefault("train", {})
        if hp.get("epochs") is not None:
            tr["epochs"] = int(hp["epochs"])
        if hp.get("lr") is not None:
            tr["lr"] = float(hp["lr"])
        if hp.get("batch_size") is not None:
            tr["batch_size"] = int(hp["batch_size"])
        return cfg

    def _resolve_base_checkpoint(self, job: dict) -> str | None:
        """Resolve base_checkpoint: a model id, or a filesystem path."""
        ref = job.get("base_checkpoint")
        if not ref:
            return None
        model = self.model_store.get_model(ref)
        if model is not None:
            ckpt = model.get("formats", {}).get("checkpoint", {})
            path = ckpt.get("path")
            if path and Path(path).exists():
                return path
            raise JobError(
                f"base_checkpoint model {ref!r} has no checkpoint artifact"
            )
        path = Path(ref)
        if path.exists():
            return str(path)
        raise JobError(f"base_checkpoint not found: {ref!r} (not a model id or path)")

    def _find_latest_data_dir(self) -> Path | None:
        """Data dir of the most recent completed selfplay/full job."""
        for job in self.list_jobs():
            if job["type"] in ("selfplay", "full") and job["status"] == "completed":
                data_dir = self.job_dir(job["id"]) / "data"
                if data_dir.exists() and list(data_dir.glob("*.npz")):
                    return data_dir
        return None

    # -- execution -----------------------------------------------------

    def _latest_checkpoint(self, job_dir: Path) -> Path | None:
        """Newest checkpoint in a job dir by mtime (for train resume).

        Considers checkpoints/*.pt (best.pt, final.pt,
        checkpoint-<step>.pt) and the --out copy final.pt. The newest
        file has the most training progress.
        """
        candidates = list((job_dir / "checkpoints").glob("*.pt"))
        out_copy = job_dir / "final.pt"
        if out_copy.exists():
            candidates.append(out_copy)
        if not candidates:
            return None
        return max(candidates, key=lambda p: p.stat().st_mtime)

    def _best_checkpoint(self, job_dir: Path) -> Path | None:
        """Checkpoint to evaluate/export: best.pt (lowest val loss)
        when the run wrote one, else final.pt."""
        for candidate in (
            job_dir / "checkpoints" / "best.pt",
            job_dir / "checkpoints" / "final.pt",
            job_dir / "final.pt",
        ):
            if candidate.exists():
                return candidate
        return None

    def _mark_phase_done(self, job: dict, phase: str) -> None:
        if phase not in job["completed_phases"]:
            job["completed_phases"].append(phase)
        self._save(job, force=True)

    def _record_eval_history(
        self, job: dict, checkpoint: Path | str
    ) -> None:
        """Append one entry to the job's eval_history.

        Handles both eval shapes: the enriched eval.json
        (``{ladder: {...}, elo_estimate, tactical, verdict, ...}``)
        and the legacy flat ladder dict
        (``{opponent: {wins, losses, draws}}``).
        """
        eval_data = job.get("eval") or {}
        ladder = eval_data.get("ladder") if isinstance(eval_data, dict) else None
        if not isinstance(ladder, dict):
            ladder = eval_data  # legacy flat shape
        summary = {
            name: f"{r.get('wins', 0)}W-{r.get('losses', 0)}L-"
                  f"{r.get('draws', 0)}D"
            for name, r in ladder.items()
            if isinstance(r, dict)
        }
        entry: dict = {
            "checkpoint": Path(checkpoint).name,
            "at": _now_iso(),
            "summary": summary,
        }
        if isinstance(eval_data, dict):
            if eval_data.get("verdict") is not None:
                entry["verdict"] = eval_data["verdict"]
            if eval_data.get("elo_estimate") is not None:
                entry["elo_estimate"] = eval_data["elo_estimate"]
        job.setdefault("eval_history", []).append(entry)
        self._save(job, force=True)

    async def _execute(self, job: dict, resume: bool = False) -> None:
        """Run a job's phases, skipping completed ones on resume.

        Resume semantics (see server/README.md "Resume semantics"):
        - selfplay: the ML script is resume-safe by design — numbering
          continues after existing shards and it TOPS UP to the
          configured total (a job configured for 4 games, interrupted
          after 3, ends with 4 shards, not 7; data/manifest.json is
          the source of truth). The server does NOT reset counters on
          resume: game numbers continue, and the per-game stats dedupe
          by game number handles both the top-up case and the
          overwrite case (older scripts). Skipped entirely if all
          shards exist.
        - train: re-run with --resume <newest checkpoint> when one
          exists, passing --epochs = remaining epochs (train.py runs
          `epochs` MORE epochs from the resumed epoch, so the full
          count would overshoot). Skipped if already done.
        - eval/export: re-run whenever train ran in this invocation or
          they never completed; export is skipped if the model is
          already registered.
        """
        if job["type"] == "eval":
            await self._execute_eval_job(job)
            return

        cfg = self._resolve_config(job)
        job["config"] = cfg
        job_dir = self.job_dir(job["id"])
        config_path = job_dir / "config.yaml"
        config_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
        self._save(job, force=True)

        base_ckpt = self._resolve_base_checkpoint(job)
        data_dir = job_dir / "data"
        scripts = ML_DIR / "scripts"
        py = self.settings.ml_python
        # The job dir IS the ML run dir: the scripts write metrics.jsonl,
        # eval.json, model_card.md and checkpoints/ into it.
        run_dir = str(job_dir)
        done = set(job.get("completed_phases") or [])

        if job["type"] in ("selfplay", "full"):
            shards = list(data_dir.glob("*.npz")) if data_dir.exists() else []
            want_games = int(cfg.get("selfplay", {}).get("games", 0))
            if "selfplay" in done and len(shards) >= want_games:
                log.info("job %s: selfplay already complete, skipping",
                         job["id"])
                self._set_phase_progress(job, "selfplay", 1.0)
            else:
                # No counter reset here: the ML script continues shard
                # numbering after existing shards and TOPS UP to the
                # configured total (resume-safe), so game numbers keep
                # counting up and the per-game stats dedupe by game
                # number. Resetting would lose the pre-interruption
                # games from the stats.
                self._set_phase(job, "selfplay")
                cmd = [
                    py, str(scripts / "selfplay.py"),
                    "--config", str(config_path),
                    "--run-dir", run_dir,
                    "--out-dir", str(data_dir),
                    "--run-name", job["run_id"],
                ]
                if base_ckpt:
                    cmd += ["--checkpoint", base_ckpt]
                await self._run_command(job, cmd, "selfplay")
                self._mark_phase_done(job, "selfplay")
                done.add("selfplay")

        if job["type"] in ("train", "full"):
            train_ran = False
            checkpoint_path = self._best_checkpoint(job_dir)
            if "train" in done and checkpoint_path is not None:
                log.info("job %s: train already complete, skipping",
                         job["id"])
                self._set_phase_progress(job, "train", 1.0)
            else:
                if job["type"] == "train":
                    # Train-only jobs reuse the latest completed
                    # self-play data.
                    latest = self._find_latest_data_dir()
                    if latest is None:
                        raise JobError(
                            "train job needs self-play data: run a selfplay "
                            "or full job first (no completed job with data "
                            "found)"
                        )
                    data_dir = latest
                self._set_phase(job, "train")
                out_path = job_dir / "final.pt"
                cmd = [
                    py, str(scripts / "train.py"),
                    "--config", str(config_path),
                    "--run-dir", run_dir,
                    "--data", str(data_dir),
                    "--out", str(out_path),
                ]
                resume_ckpt = (
                    self._latest_checkpoint(job_dir) if resume else None
                )
                if resume_ckpt is not None:
                    # Continue from the newest checkpoint: write a
                    # resume-config.yaml (config + resume: <ckpt>) and
                    # run only the REMAINING epochs — train.py runs
                    # `epochs` more epochs from the resumed epoch, so
                    # passing the full count would overshoot the total.
                    total_epochs = int(cfg.get("train", {}).get("epochs", 0))
                    done_epochs = max(
                        (int(m.get("epoch", 0))
                         for m in job.get("metrics", [])
                         if isinstance(m, dict) and "epoch" in m),
                        default=0,
                    )
                    remaining = max(1, total_epochs - done_epochs)
                    resume_cfg = {**cfg, "resume": str(resume_ckpt)}
                    resume_path = job_dir / "resume-config.yaml"
                    resume_path.write_text(
                        yaml.safe_dump(resume_cfg, sort_keys=False),
                        encoding="utf-8",
                    )
                    cmd[cmd.index(str(config_path))] = str(resume_path)
                    cmd += ["--epochs", str(remaining)]
                    self.publish(job["id"], {
                        "type": "log",
                        "line": f"resuming training from {resume_ckpt} "
                                f"(epoch {done_epochs}/{total_epochs}, "
                                f"{remaining} remaining)",
                    })
                elif base_ckpt:
                    cmd += ["--resume", base_ckpt]
                await self._run_command(job, cmd, "train")
                self._mark_phase_done(job, "train")
                done.add("train")
                train_ran = True
                checkpoint_path = self._best_checkpoint(job_dir)
                if checkpoint_path is None:
                    raise JobError(
                        "training finished but no checkpoint was written"
                    )

            # Eval ladder (small, quick — honest numbers for the model card).
            if "eval" in done and not train_ran:
                log.info("job %s: eval already complete, skipping", job["id"])
                self._set_phase_progress(job, "eval", 1.0)
            else:
                assert checkpoint_path is not None
                await self._run_eval_phase(job, checkpoint_path, run_dir)
                self._record_eval_history(job, checkpoint_path)
                self._mark_phase_done(job, "eval")
                done.add("eval")

            # Export + register the model.
            if job.get("model_id") and "export" in done:
                log.info("job %s: model %s already registered, skipping "
                         "export", job["id"], job["model_id"])
                self._set_phase_progress(job, "export", 1.0)
            else:
                assert checkpoint_path is not None
                self._set_phase(job, "export")
                artifacts = await self._export(job, checkpoint_path)
                job["artifacts"] = artifacts
                model = self.model_store.register_from_job(job, artifacts)
                job["model_id"] = model["id"]
                self._mark_phase_done(job, "export")
                self.publish(job["id"], {
                    "type": "log",
                    "line": f"registered model {model['id']}",
                })

    async def _run_eval_phase(
        self, job: dict, checkpoint_path: Path, run_dir: str
    ) -> None:
        """Run the eval ladder for a training job and record results."""
        self._set_phase(job, "eval")
        hp = job.get("hyperparameters") or {}
        eval_games = int(hp.get("eval_games") or 6)
        eval_sims = int(hp.get("eval_sims") or 50)
        opponents = hp.get("opponents") or ["random", "heuristic"]
        cmd = [
            self.settings.ml_python, str(ML_DIR / "scripts" / "evaluate.py"),
            "--checkpoint", str(checkpoint_path),
            "--run-dir", run_dir,
            "--games", str(eval_games),
            "--sims", str(eval_sims),
            "--opponents", ",".join(opponents),
        ]
        await self._run_command(job, cmd, "eval")
        # eval.json in the run dir is authoritative (the eval script
        # writes it); stdout parsing already filled job["eval"] as a
        # fallback for scripts that don't write the file.
        job_dir = self.job_dir(job["id"])
        eval_path = job_dir / "eval.json"
        if eval_path.exists():
            try:
                job["eval"] = json.loads(eval_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                log.warning("could not parse %s: %s", eval_path, exc)
        else:
            eval_path.write_text(
                json.dumps(job["eval"], indent=2), encoding="utf-8"
            )
        self._save(job, force=True)

    async def _execute_eval_job(self, job: dict) -> None:
        """Run a standalone eval job (created by the evaluate endpoints).

        Evaluates a fixed checkpoint and writes the report into the eval
        job dir and, when linked to a parent job, into the parent job
        dir as eval.json (updating the parent's eval + eval_history).
        """
        job_dir = self.job_dir(job["id"])
        ckpt = job.get("eval_checkpoint")
        if not ckpt or not Path(ckpt).exists():
            raise JobError(f"eval checkpoint not found: {ckpt}")
        hp = job.get("hyperparameters") or {}
        self._set_phase(job, "eval")
        cmd = [
            self.settings.ml_python, str(ML_DIR / "scripts" / "evaluate.py"),
            "--checkpoint", str(ckpt),
            "--run-dir", str(job_dir),
            "--games", str(int(hp.get("eval_games") or 10)),
            "--sims", str(int(hp.get("eval_sims") or 50)),
            "--opponents", ",".join(hp.get("opponents") or ["random", "heuristic"]),
        ]
        await self._run_command(job, cmd, "eval")
        eval_path = job_dir / "eval.json"
        if eval_path.exists():
            try:
                job["eval"] = json.loads(eval_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                log.warning("could not parse %s: %s", eval_path, exc)
        else:
            eval_path.write_text(
                json.dumps(job["eval"], indent=2), encoding="utf-8"
            )
        self._record_eval_history(job, ckpt)
        self._mark_phase_done(job, "eval")
        # Propagate to the parent job (if any).
        parent_id = job.get("parent_job_id")
        if parent_id:
            parent = self.get_job(parent_id)
            if parent is not None:
                parent_dir = self.job_dir(parent_id)
                (parent_dir / "eval.json").write_text(
                    json.dumps(job["eval"], indent=2), encoding="utf-8"
                )
                parent["eval"] = job["eval"]
                self._record_eval_history(parent, ckpt)
                self._save(parent, force=True)
                self.publish(parent_id, {
                    "type": "metric", "phase": "eval",
                    "eval": job["eval"],
                })

    def _set_phase(self, job: dict, phase: str) -> None:
        job["phase"] = phase
        self._save(job)
        self.publish(job["id"], {"type": "progress", "progress": job["progress"],
                                 "phase": phase})

    async def _export(self, job: dict, checkpoint_path: Path) -> dict:
        """Export checkpoint -> ONNX (+fp16) + Core ML (best-effort).

        Returns an artifacts dict of {format: {path, size}} / notes.
        Core ML failure never fails the job: it is recorded with the
        reason (e.g. coremltools not installed on this machine).
        """
        job_dir = self.job_dir(job["id"])
        scripts = ML_DIR / "scripts"
        py = self.settings.ml_python
        artifacts: dict[str, Any] = {}
        onnx_path = job_dir / "model.onnx"
        cmd = [
            py, str(scripts / "export_onnx.py"),
            "--checkpoint", str(checkpoint_path),
            "--out", str(onnx_path),
            "--quantize", "fp16",
        ]
        await self._run_command(job, cmd, "export")
        if not onnx_path.exists():
            raise JobError("ONNX export finished but no .onnx file was written")
        artifacts["onnx"] = {"path": str(onnx_path), "size": onnx_path.stat().st_size}
        fp16_path = job_dir / "model.fp16.onnx"
        if fp16_path.exists():
            artifacts["onnx-fp16"] = {
                "path": str(fp16_path), "size": fp16_path.stat().st_size,
            }
        else:
            artifacts["onnx-fp16"] = {
                "path": None, "size": None,
                "note": "fp16 variant not produced "
                        "(onnxconverter-common may be missing; see job log)",
            }
        # Core ML: best-effort only.
        coreml_path = job_dir / "model.mlpackage"
        cmd = [
            py, str(scripts / "export_coreml.py"),
            "--checkpoint", str(checkpoint_path),
            "--out", str(coreml_path),
        ]
        try:
            await self._run_command(job, cmd, "export")
        except JobError as exc:
            artifacts["coreml"] = {"path": None, "size": None, "note": str(exc)}
            self.publish(job["id"], {
                "type": "log",
                "line": f"Core ML export skipped: {exc}",
            })
        else:
            if coreml_path.exists():
                artifacts["coreml"] = {
                    "path": str(coreml_path), "size": None,
                    "note": "mlpackage directory; zipped at registration",
                }
            else:
                artifacts["coreml"] = {
                    "path": None, "size": None,
                    "note": "export_coreml.py exited 0 but wrote no package",
                }
        artifacts["checkpoint"] = {
            "path": str(checkpoint_path),
            "size": checkpoint_path.stat().st_size,
        }
        return artifacts

    # -- subprocess + line parsing -------------------------------------

    async def _run_command(
        self, job: dict, cmd: list[str], phase: str
    ) -> int:
        """Run one ML script, streaming + parsing its stdout.

        Raises JobError on non-zero exit or on cancellation.
        """
        await self._wait_if_paused(job)
        log.info("job %s [%s]: $ %s", job["id"], phase, " ".join(cmd))
        self._append_log(job, f"$ {' '.join(cmd)}")
        self.publish(job["id"], {
            "type": "log", "line": f"$ {' '.join(cmd)}", "phase": phase,
        })
        env = {
            **os.environ,
            "PYTHONUNBUFFERED": "1",
            # Per-phase torch thread count (see TORCH_THREADS above).
            # Only set when the operator hasn't already set it.
            "TORCH_THREADS": os.environ.get(
                "TORCH_THREADS", str(TORCH_THREADS.get(phase, 1))
            ),
        }
        proc = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT, env=env,
        )
        self._current_proc = proc
        try:
            assert proc.stdout is not None
            while True:
                if job["id"] in self._cancel_requested:
                    await self._kill_proc(proc)
                    raise JobCancelled("cancelled by user")
                if job["status"] == "paused":
                    # The child is SIGSTOPped; just wait for resume or
                    # cancel instead of burning readline timeouts.
                    await asyncio.sleep(0.5)
                    continue
                try:
                    # Timeout so a cancel request is noticed even while
                    # the child is quiet (self-play prints once per game).
                    line_bytes = await asyncio.wait_for(
                        proc.stdout.readline(), timeout=5
                    )
                except asyncio.TimeoutError:
                    continue
                if not line_bytes:
                    break
                line = line_bytes.decode("utf-8", errors="replace").rstrip("\n")
                self._handle_line(job, phase, line)
            rc = await proc.wait()
        finally:
            self._current_proc = None
        if job["id"] in self._cancel_requested:
            raise JobCancelled("cancelled by user")
        if rc != 0:
            raise JobError(
                f"step {phase} failed (exit {rc}): "
                f"{' '.join(cmd[:3])}... — see job log tail"
            )
        # Phase finished: advance progress to the end of this phase.
        self._set_phase_progress(job, phase, 1.0)
        return rc

    @staticmethod
    async def _kill_proc(proc: asyncio.subprocess.Process) -> None:
        """SIGTERM, then SIGKILL after 5s if the process survives.

        SIGCONT first: a paused (SIGSTOPped) process can't act on
        SIGTERM until it continues, so without this the kill would hang
        until the job is resumed. SIGCONT on a running process is a
        harmless no-op.
        """
        if proc.returncode is not None:
            return
        try:
            os.kill(proc.pid, signal.SIGCONT)
        except ProcessLookupError:
            return
        proc.terminate()
        try:
            await asyncio.wait_for(proc.wait(), timeout=5)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()

    async def _wait_if_paused(self, job: dict) -> None:
        """Block while the job is paused (the subprocess is SIGSTOPped,
        so nothing progresses; this just keeps the worker from starting
        the next step). Raises JobCancelled if a cancel arrives."""
        while job["status"] == "paused":
            self._check_cancelled(job)
            await asyncio.sleep(0.5)
        self._check_cancelled(job)

    def _handle_line(self, job: dict, phase: str, line: str) -> None:
        self._append_log(job, line)
        self.publish(job["id"], {"type": "log", "line": line, "phase": phase})
        self._parse_line(job, phase, line)
        self._save(job)

    def _parse_line(self, job: dict, phase: str, line: str) -> None:
        """Parse one stdout line into progress / metric / eval events.

        Accepts JSON lines (``{"event": "progress"|"metric"|"log", ...}``)
        for forward compatibility, and the plain-text formats the ML
        scripts actually print today.
        """
        stripped = line.strip()
        if stripped.startswith("{"):
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError:
                payload = None
            if isinstance(payload, dict) and "event" in payload:
                self._handle_json_event(job, phase, payload)
                return
        m = _RE_GAME.search(line)
        if m and phase == "selfplay":
            done, total = int(m.group(1)), int(m.group(2))
            job["games_generated"] = done
            moves = int(m.group(3)) if m.group(3) else None
            seconds = float(m.group(4)) if m.group(4) else None
            self._record_game_stats(job, done, moves, seconds)
            self._set_phase_progress(job, phase, done / max(total, 1))
            self.publish(job["id"], {
                "type": "progress", "progress": job["progress"],
                "phase": phase, "games_done": done, "games_total": total,
            })
            return
        m = _RE_EPOCH.search(line)
        if m and phase == "train":
            epoch, epochs = int(m.group(1)), int(m.group(2))
            # Dedupe: the scripts emit BOTH a JSON "epoch" event and a
            # plain-text epoch line for the same epoch. The JSON event
            # arrives first and carries val_loss, so skip the plain line
            # when that epoch is already recorded.
            if any(
                isinstance(mt, dict) and mt.get("epoch") == epoch
                for mt in job["metrics"]
            ):
                return
            entry = {
                "epoch": epoch,
                "loss": float(m.group(3)),
                "policy_loss": float(m.group(4)),
                "value_loss": float(m.group(5)),
            }
            job["metrics"].append(entry)
            self._set_phase_progress(job, phase, epoch / max(epochs, 1))
            self.publish(job["id"], {
                "type": "metric", "phase": phase, **entry,
                "progress": job["progress"],
            })
            return
        m = _RE_STEP.search(line)
        if m and phase == "train":
            step, total = int(m.group(1)), int(m.group(2))
            self._set_phase_progress(job, phase, step / max(total, 1))
            return
        m = _RE_EVAL.search(line)
        if m and phase == "eval":
            name = m.group(1)
            elo = m.group(5)
            job["eval"][name] = {
                "wins": int(m.group(2)),
                "losses": int(m.group(3)),
                "draws": int(m.group(4)),
                "elo_diff_vs_opponent": None if elo == "None" else float(elo),
            }
            self.publish(job["id"], {
                "type": "metric", "phase": phase, "opponent": name,
                **job["eval"][name],
            })

    def _handle_json_event(self, job: dict, phase: str, payload: dict) -> None:
        """Handle one JSON event line from the ML scripts.

        Event shapes (gomoku_ml/runs.py + the pipeline modules):
        - ``run_started`` / ``run_finished`` / ``checkpoint`` /
          ``export_verified`` / ``export_finished``: informational; the
          full line is already in the log.
        - ``selfplay_game``: {game, games_total, moves, winner, seconds}
        - ``train_step``: {step, epoch, loss, policy_loss, value_loss}
        - ``epoch``: {epoch, loss, policy_loss, value_loss, val_loss}
        - ``eval_result``: {opponent, wins, losses, draws,
          elo_diff_vs_opponent}
        - ``progress`` / ``metric``: generic forward-compat shapes.
        """
        event = payload.get("event")
        if event == "selfplay_game":
            done = int(payload.get("game", 0))
            total = int(payload.get("games_total", 0))
            job["games_generated"] = done
            if payload.get("games_per_hour") is not None:
                job.setdefault("stats", {})["games_per_hour_ml"] = round(
                    float(payload["games_per_hour"]), 1
                )
            self._record_game_stats(
                job, done, payload.get("moves"), payload.get("seconds")
            )
            if total:
                self._set_phase_progress(job, phase, done / total)
            self.publish(job["id"], {
                "type": "progress", "progress": job["progress"],
                "phase": phase, "games_done": done, "games_total": total,
                "moves": payload.get("moves"), "winner": payload.get("winner"),
            })
        elif event == "epoch":
            entry = {
                "epoch": int(payload.get("epoch", 0)),
                "loss": payload.get("loss"),
                "policy_loss": payload.get("policy_loss"),
                "value_loss": payload.get("value_loss"),
                "val_loss": payload.get("val_loss"),
            }
            job["metrics"].append(entry)
            epochs = int((job.get("config") or {}).get("train", {}).get("epochs", 0))
            if epochs:
                self._set_phase_progress(job, phase, entry["epoch"] / epochs)
            self.publish(job["id"], {
                "type": "metric", "phase": phase, **entry,
                "progress": job["progress"],
            })
        elif event == "train_step":
            # Fine-grained loss signal; progress comes from epoch events.
            self.publish(job["id"], {
                "type": "metric", "phase": phase,
                "step": payload.get("step"), "epoch": payload.get("epoch"),
                "loss": payload.get("loss"),
                "policy_loss": payload.get("policy_loss"),
                "value_loss": payload.get("value_loss"),
            })
        elif event == "eval_result":
            name = payload.get("opponent")
            if name:
                job["eval"][name] = {
                    "wins": int(payload.get("wins", 0)),
                    "losses": int(payload.get("losses", 0)),
                    "draws": int(payload.get("draws", 0)),
                    "elo_diff_vs_opponent": payload.get("elo_diff_vs_opponent"),
                }
                self.publish(job["id"], {
                    "type": "metric", "phase": phase, "opponent": name,
                    **job["eval"][name],
                })
        elif event == "progress":
            frac = payload.get("fraction")
            if frac is not None:
                self._set_phase_progress(job, phase, float(frac))
            if "games_done" in payload:
                job["games_generated"] = int(payload["games_done"])
            self.publish(job["id"], {"type": "progress",
                                     "progress": job["progress"], "phase": phase})
        elif event == "metric":
            entry = {k: v for k, v in payload.items()
                     if k not in ("event", "timestamp")}
            job["metrics"].append(entry)
            self.publish(job["id"], {"type": "metric", "phase": phase, **entry})
        # run_started / run_finished / checkpoint / export_* carry no
        # per-step progress; the line itself is already in the log.

    def _set_phase_progress(self, job: dict, phase: str, frac: float) -> None:
        weights = PHASE_WEIGHTS[job["type"]]
        base = 0.0
        for name, weight in weights.items():
            if name == phase:
                job["progress"] = min(1.0, base + weight * max(0.0, min(1.0, frac)))
                return
            base += weight

    def _record_game_stats(
        self, job: dict, game_no: int,
        moves: int | float | None, seconds: int | float | None,
    ) -> None:
        """Fold one finished self-play game into the running stats.

        Deduped by game number: the scripts emit both a plain-text
        game line and a JSON selfplay_game event for the same game.
        """
        stats = job.setdefault("stats", {})
        if game_no <= int(stats.get("last_game_recorded", 0)):
            return
        stats["last_game_recorded"] = game_no
        stats["games_finished"] = int(stats.get("games_finished", 0)) + 1
        if moves is not None:
            stats["total_moves"] = int(stats.get("total_moves", 0)) + int(moves)
        if seconds is not None:
            stats["total_game_seconds"] = (
                float(stats.get("total_game_seconds", 0.0)) + float(seconds)
            )

    # -- live self-play feed ------------------------------------------

    def _poll_current_game(self, job: dict) -> None:
        """One poll of <job-dir>/current_game.json (the live game viewer).

        The ML self-play script writes this file every move (contract
        with the ML workstream; not implemented there yet — until it
        lands, the file is simply absent and current_game stays null).
        On change, updates job["current_game"] and publishes game_move /
        game_finished SSE events. Never raises: a missing or torn file
        (mid-write read) is skipped silently.
        """
        path = self.job_dir(job["id"]) / "current_game.json"
        try:
            mtime = path.stat().st_mtime
        except OSError:
            return
        if mtime == self._game_feed_mtime.get(job["id"]):
            return
        try:
            current = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError, ValueError):
            return  # torn read mid-write; the next poll retries
        self._game_feed_mtime[job["id"]] = mtime
        if not isinstance(current, dict) or "moves" not in current:
            return
        job["current_game"] = current
        self._save(job)
        game = current.get("game")
        moves = current.get("moves") or []
        status = current.get("status")
        key = (game, len(moves), status)
        if key == self._game_feed_last.get(job["id"]):
            return
        self._game_feed_last[job["id"]] = key
        if status == "finished":
            self.publish(job["id"], {
                "type": "game_finished", "game": game,
                "winner": current.get("winner"), "moves": moves,
            })
        else:
            self.publish(job["id"], {
                "type": "game_move", "game": game, "moves": moves,
                "to_move": current.get("to_move"),
            })

    async def _watch_current_game(self, job: dict) -> None:
        """Poll current_game.json ~1/s while the job runs."""
        while job["status"] in ("running", "paused"):
            self._poll_current_game(job)
            await asyncio.sleep(1.0)
