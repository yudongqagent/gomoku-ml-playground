"""FastAPI app for the Gomoku ML Playground training server.

Run from server/:
    uvicorn app.main:app --reload --port 8000

API contract (the web UI codes against this exactly):
    GET  /api/health
    GET  /api/architectures
    POST /api/jobs
    GET  /api/jobs
    GET  /api/jobs/{id}
    POST /api/jobs/{id}/cancel
    POST /api/jobs/{id}/pause           (running -> paused, SIGSTOP)
    POST /api/jobs/{id}/resume          (paused -> running, SIGCONT;
                                        interrupted/failed -> re-queue)
    POST /api/jobs/{id}/evaluate        (create an eval job for the job)
    GET  /api/jobs/{id}/eval            (latest eval report)
    GET  /api/jobs/{id}/events          (SSE)
    GET  /api/presets
    GET  /api/models
    GET  /api/models/{id}
    GET  /api/models/{id}/download?format=onnx|onnx-fp16|coreml|checkpoint
    POST /api/models/{id}/release       (dry-run by default)
    POST /api/models/{id}/evaluate      (create an eval job for the model)
"""

from __future__ import annotations

import asyncio
import json
import logging
import subprocess
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse

import yaml

from .config import ML_DIR, Settings, load_settings
from .jobs import TERMINAL_STATUSES, JobError, JobManager
from .models_registry import ModelStore
from .releases import ReleaseError, release_model
from .schemas import EvaluateRequest, JobCreate, ReleaseRequest

log = logging.getLogger(__name__)

VITE_ORIGINS = ["http://localhost:5173", "http://localhost:4173"]

# Fallback architecture list, used only when the ML package's registry
# is unavailable (gomoku_ml.architectures / list_architectures.py do not
# exist yet — the ML workstream ships ResNet only so far). The server
# never 500s on /api/architectures.
FALLBACK_ARCHITECTURES = [
    {
        "id": "resnet",
        "name": "ResNet (AlphaZero-style)",
        "description": (
            "Residual policy+value network: 3x3 conv stem, residual "
            "blocks, policy head (225 logits) + value head (tanh). "
            "The only architecture shipped by the ML pipeline so far."
        ),
        "params": {
            "channels": {
                "type": "integer", "default": 128,
                "minimum": 8, "maximum": 512,
                "description": "Convolution channels throughout the net.",
            },
            "blocks": {
                "type": "integer", "default": 10,
                "minimum": 1, "maximum": 20,
                "description": "Number of residual blocks.",
            },
        },
    },
]


def _normalize_architectures(raw: object) -> list[dict]:
    """Normalise the ML registry into the API contract shape.

    The registry returns ``{"architectures": [...]}`` (or a bare list)
    where each entry has ``name`` (the id), ``description`` and a
    ``params`` schema. The web UI codes against ``id`` + ``name``, so
    ``id`` is taken from ``id`` when present, else from ``name``.
    """
    if isinstance(raw, dict):
        raw = raw.get("architectures", [])
    if not isinstance(raw, list):
        return []
    entries = []
    for e in raw:
        if not isinstance(e, dict) or "name" not in e and "id" not in e:
            continue
        arch_id = e.get("id") or e.get("name")
        entries.append({
            "id": arch_id,
            "name": e.get("name", arch_id),
            "description": e.get("description", ""),
            "params": e.get("params", {}),
        })
    return entries


# In-process cache for the architecture registry: the subprocess path
# imports torch (~seconds per call), so results are cached for 5 min.
_ARCH_CACHE: dict = {"at": 0.0, "entries": None}
_ARCH_CACHE_TTL = 300.0


def list_architectures(ml_python: str | None = None) -> list[dict]:
    """Architecture registry, best-effort from the ML package.

    1. Import gomoku_ml.architectures.list_architectures() (works when
       the server shares an env with torch installed).
    2. Run ml/scripts/list_architectures.py as a subprocess under the
       ML interpreter (the server venv has no torch, so the ML venv's
       python is required here).
    3. Fall back to the static ResNet-only list with a warning log.
    Never raises: the endpoint must not 500.
    Results are cached for 5 minutes (the subprocess imports torch).
    """
    import time

    now = time.monotonic()
    if (
        _ARCH_CACHE["entries"] is not None
        and now - _ARCH_CACHE["at"] < _ARCH_CACHE_TTL
    ):
        return _ARCH_CACHE["entries"]
    # 1. Import the registry.
    try:
        if str(ML_DIR) not in sys.path:
            sys.path.insert(0, str(ML_DIR))
        from gomoku_ml import architectures  # type: ignore

        entries = _normalize_architectures(architectures.list_architectures())
        if entries:
            _ARCH_CACHE.update({"at": now, "entries": entries})
            return entries
    except Exception as exc:  # noqa: BLE001 — registry is best-effort
        log.debug("gomoku_ml.architectures unavailable: %s", exc)
    # 2. Subprocess the list script with the ML interpreter.
    script = ML_DIR / "scripts" / "list_architectures.py"
    if script.exists():
        try:
            result = subprocess.run(
                [ml_python or sys.executable, str(script)],
                capture_output=True, text=True, timeout=60,
            )
            if result.returncode == 0:
                entries = _normalize_architectures(json.loads(result.stdout))
                if entries:
                    _ARCH_CACHE.update({"at": now, "entries": entries})
                    return entries
            else:
                log.debug("list_architectures.py failed: %s",
                          result.stderr.strip()[:300])
        except Exception as exc:  # noqa: BLE001 — best-effort
            log.debug("list_architectures.py failed: %s", exc)
    # 3. Static fallback.
    log.warning(
        "ML architecture registry unavailable; serving static "
        "ResNet-only list"
    )
    _ARCH_CACHE.update({"at": now, "entries": FALLBACK_ARCHITECTURES})
    return FALLBACK_ARCHITECTURES


def validate_arch_params(
    arch_id: str, params: dict, architectures: list[dict]
) -> str | None:
    """Validate arch_params against the registry schema for one arch.

    Mirrors gomoku_ml.architectures.ArchitectureSpec.validate_params:
    unknown keys, non-integers, and out-of-range values are rejected.
    Returns an error message, or None when valid.
    """
    spec = next((a for a in architectures if a["id"] == arch_id), None)
    if spec is None:
        return f"unknown architecture {arch_id!r}"
    schema = spec.get("params", {})
    unknown = sorted(set(params) - set(schema))
    if unknown:
        return (
            f"architecture '{arch_id}': unknown parameter(s) {unknown}; "
            f"known parameters: {sorted(schema)}"
        )
    for key, value in params.items():
        param_spec = schema[key]
        if param_spec.get("type") == "integer":
            if isinstance(value, bool) or not isinstance(value, int):
                return (
                    f"architecture '{arch_id}': parameter '{key}' must be "
                    f"an integer, got {value!r}"
                )
        if "minimum" in param_spec and value < param_spec["minimum"]:
            return (
                f"architecture '{arch_id}': parameter '{key}' must be >= "
                f"{param_spec['minimum']}, got {value}"
            )
        if "maximum" in param_spec and value > param_spec["maximum"]:
            return (
                f"architecture '{arch_id}': parameter '{key}' must be <= "
                f"{param_spec['maximum']}, got {value}"
            )
    return None


def list_presets() -> list[dict]:
    """Training presets: one per ml/configs/*.yaml.

    Reads a top-level ``preset:`` block ({name, description,
    throughput_note}) when present; otherwise falls back to a
    filename-based entry. Never raises: an unreadable file is skipped
    with a warning log.
    """
    configs_dir = ML_DIR / "configs"
    presets: list[dict] = []
    if not configs_dir.exists():
        return presets
    for path in sorted(configs_dir.glob("*.yaml")):
        preset_id = path.stem
        try:
            with open(path, encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
        except (OSError, yaml.YAMLError) as exc:
            log.warning("skipping unreadable preset config %s: %s", path, exc)
            continue
        block = cfg.get("preset") or {}
        presets.append({
            "id": preset_id,
            "name": block.get("name") or preset_id,
            "description": block.get("description") or "",
            "config_file": f"ml/configs/{path.name}",
            "throughput_note": block.get("throughput_note") or "",
        })
    return presets


EVAL_OPPONENTS = {"random", "heuristic"}


def _create_eval_job(
    manager: JobManager,
    payload: EvaluateRequest | None,
    *,
    name: str,
    parent_job_id: str | None = None,
    model_id: str | None = None,
    eval_checkpoint: str,
) -> dict:
    """Validate an evaluate request and create the eval job."""
    payload = payload or EvaluateRequest()
    opponents = payload.opponents or ["random", "heuristic"]
    bad = sorted(set(opponents) - EVAL_OPPONENTS)
    if bad:
        raise HTTPException(
            status_code=400,
            detail=(
                f"unsupported opponent(s) {bad}; supported here: "
                f"{sorted(EVAL_OPPONENTS)} ('checkpoint' needs a "
                "--vs-checkpoint, which this endpoint doesn't take)"
            ),
        )
    return manager.create_job({
        "type": "eval",
        "name": name,
        "architecture": "resnet",
        "arch_params": {},
        "hyperparameters": {
            "eval_games": payload.games,
            "eval_sims": payload.sims,
            "opponents": opponents,
        },
        "base_checkpoint": None,
        "parent_job_id": parent_job_id,
        "model_id": model_id,
        "eval_checkpoint": eval_checkpoint,
    })


def create_app(
    settings: Settings | None = None,
    manager: JobManager | None = None,
) -> FastAPI:
    settings = settings or load_settings()
    model_store = ModelStore(settings)
    manager = manager or JobManager(settings, model_store)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        await manager.start()
        yield
        await manager.stop()

    app = FastAPI(title="Gomoku ML Playground — training server", lifespan=lifespan)
    app.state.settings = settings
    app.state.manager = manager
    app.state.model_store = model_store

    app.add_middleware(
        CORSMiddleware,
        allow_origins=VITE_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # -- health ------------------------------------------------------

    @app.get("/api/health")
    def health() -> dict:
        return {"status": "ok"}

    # -- architectures -----------------------------------------------

    @app.get("/api/architectures")
    def architectures() -> dict:
        return {"architectures": list_architectures(settings.ml_python)}

    # -- jobs --------------------------------------------------------

    @app.post("/api/jobs", status_code=201)
    async def create_job(payload: JobCreate) -> dict:
        archs = list_architectures(settings.ml_python)
        known = {a["id"] for a in archs}
        if payload.architecture not in known:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"unknown architecture {payload.architecture!r}; "
                    f"known: {sorted(known)}"
                ),
            )
        error = validate_arch_params(
            payload.architecture, payload.arch_params, archs
        )
        if error:
            raise HTTPException(status_code=400, detail=error)
        if payload.preset is not None:
            known_presets = {p["id"] for p in list_presets()}
            if payload.preset not in known_presets:
                raise HTTPException(
                    status_code=422,
                    detail=(
                        f"unknown preset {payload.preset!r}; "
                        f"known: {sorted(known_presets)}"
                    ),
                )
        job = manager.create_job(payload.model_dump())
        return {"id": job["id"], "status": job["status"]}

    @app.get("/api/jobs")
    def list_jobs() -> dict:
        return {"jobs": [manager.summary(j) for j in manager.list_jobs()]}

    @app.get("/api/jobs/{job_id}")
    def get_job(job_id: str) -> dict:
        job = manager.get_job(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="job not found")
        return manager.detail(job)

    @app.post("/api/jobs/{job_id}/cancel")
    async def cancel_job(job_id: str) -> dict:
        job = manager.get_job(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="job not found")
        try:
            updated = await manager.cancel_job(job_id)
        except JobError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        assert updated is not None
        return manager.detail(updated)

    @app.post("/api/jobs/{job_id}/pause")
    async def pause_job(job_id: str) -> dict:
        job = manager.get_job(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="job not found")
        try:
            updated = await manager.pause_job(job_id)
        except JobError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        assert updated is not None
        return manager.detail(updated)

    @app.post("/api/jobs/{job_id}/resume")
    async def resume_job(job_id: str) -> dict:
        job = manager.get_job(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="job not found")
        try:
            updated = await manager.resume_job(job_id)
        except JobError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        assert updated is not None
        return manager.detail(updated)

    @app.post("/api/jobs/{job_id}/evaluate", status_code=201)
    async def evaluate_job(
        job_id: str, payload: EvaluateRequest | None = None
    ) -> dict:
        """Create an eval job for a training job's latest checkpoint."""
        parent = manager.get_job(job_id)
        if parent is None:
            raise HTTPException(status_code=404, detail="job not found")
        checkpoint = manager._best_checkpoint(manager.job_dir(job_id))
        if checkpoint is None:
            raise HTTPException(
                status_code=400,
                detail=f"job {job_id!r} has no checkpoint to evaluate yet",
            )
        eval_job = _create_eval_job(
            manager, payload,
            name=f"eval of {job_id}",
            parent_job_id=job_id,
            eval_checkpoint=str(checkpoint),
        )
        return {"id": eval_job["id"], "status": eval_job["status"]}

    @app.get("/api/jobs/{job_id}/eval")
    def get_job_eval(job_id: str) -> dict:
        """Latest eval report for a job, from its eval.json (if any).

        Passes through whatever the ML eval script produced: the
        enriched eval.json
        ({checkpoint, generated_at, ladder, elo_estimate, tactical,
        verdict, ...}); older runs stored a plain ladder dict
        ({opponent: {wins, losses, draws, elo_diff_vs_opponent}}).
        """
        job = manager.get_job(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="job not found")
        eval_path = manager.job_dir(job_id) / "eval.json"
        if not eval_path.exists():
            raise HTTPException(
                status_code=404,
                detail=f"no eval report for job {job_id!r} yet — "
                       "run POST /api/jobs/{id}/evaluate",
            )
        try:
            content = json.loads(eval_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            raise HTTPException(
                status_code=500, detail=f"unreadable eval.json: {exc}"
            ) from exc
        if not isinstance(content, dict):
            raise HTTPException(
                status_code=500, detail="eval.json did not parse to a mapping"
            )
        generated_at = content.get("generated_at")
        if generated_at is None:
            import datetime as _dt

            generated_at = _dt.datetime.fromtimestamp(
                eval_path.stat().st_mtime, tz=_dt.timezone.utc
            ).isoformat().replace("+00:00", "Z")
        return {
            "ladder": content.get("ladder", content),
            "elo_estimate": content.get("elo_estimate"),
            "tactical": content.get("tactical"),
            "verdict": content.get("verdict"),
            "generated_at": generated_at,
        }

    @app.get("/api/presets")
    def presets() -> dict:
        return {"presets": list_presets()}

    @app.get("/api/jobs/{job_id}/events")
    async def job_events(job_id: str, request: Request) -> StreamingResponse:
        job = manager.get_job(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="job not found")

        async def event_stream() -> AsyncIterator[str]:
            history, queue = manager.subscribe(job_id)
            try:
                # Replay recent history so a (re)connecting client sees
                # what it missed, then stream live events.
                for event in history:
                    yield f"data: {json.dumps(event)}\n\n"
                if job["status"] in TERMINAL_STATUSES:
                    # Synthesize the terminal event only if the replayed
                    # history doesn't already end with one (the worker
                    # publishes done/error at finish time; after a
                    # server restart the in-memory history is empty and
                    # the synthesis below is what closes the stream).
                    if not history or history[-1].get("type") not in (
                        "done", "error",
                    ):
                        terminal = (
                            {"type": "done", "status": job["status"]}
                            if job["status"] == "completed"
                            else {"type": "error", "status": job["status"],
                                  "message": job.get("error") or job["status"]}
                        )
                        yield f"data: {json.dumps(terminal)}\n\n"
                    return
                while True:
                    if await request.is_disconnected():
                        break
                    try:
                        event = await asyncio.wait_for(queue.get(), timeout=15)
                    except asyncio.TimeoutError:
                        yield ": heartbeat\n\n"
                        continue
                    yield f"data: {json.dumps(event)}\n\n"
                    if event.get("type") in ("done", "error"):
                        break
            finally:
                manager.unsubscribe(job_id, queue)

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    # -- models ------------------------------------------------------

    @app.get("/api/models")
    def list_models() -> dict:
        return {"models": model_store.list_models()}

    @app.get("/api/models/{model_id}")
    def get_model(model_id: str) -> dict:
        detail = model_store.get_model_detail(model_id)
        if detail is None:
            raise HTTPException(status_code=404, detail="model not found")
        return detail

    @app.get("/api/models/{model_id}/download")
    def download_model(
        model_id: str,
        format: str = Query("onnx", pattern="^(onnx|onnx-fp16|onnx_fp16|coreml|checkpoint)$"),
    ) -> FileResponse:
        format = format.replace("onnx_fp16", "onnx-fp16")  # accept both spellings
        path = model_store.artifact_path(model_id, format)
        if path is None:
            raise HTTPException(
                status_code=404,
                detail=f"model {model_id!r} has no {format!r} artifact",
            )
        media_types = {
            "onnx": "application/octet-stream",
            "onnx-fp16": "application/octet-stream",
            "coreml": "application/zip",
            "checkpoint": "application/octet-stream",
        }
        # Dev-only: allow any origin to fetch model files so
        # onnxruntime-web in the browser can load them cross-origin.
        # In production, serve models from the same origin instead.
        return FileResponse(
            path,
            filename=path.name,
            media_type=media_types[format],
            headers={"Access-Control-Allow-Origin": "*"},
        )

    @app.post("/api/models/{model_id}/evaluate", status_code=201)
    async def evaluate_model(
        model_id: str, payload: EvaluateRequest | None = None
    ) -> dict:
        """Create an eval job for a registered model's checkpoint."""
        meta = model_store.get_model(model_id)
        if meta is None:
            raise HTTPException(status_code=404, detail="model not found")
        if meta.get("source") != "server":
            raise HTTPException(
                status_code=400,
                detail="bundled models have no checkpoint to evaluate here",
            )
        checkpoint = model_store.artifact_path(model_id, "checkpoint")
        if checkpoint is None:
            raise HTTPException(
                status_code=400,
                detail=f"model {model_id!r} has no checkpoint artifact",
            )
        eval_job = _create_eval_job(
            manager, payload,
            name=f"eval of {model_id}",
            model_id=model_id,
            eval_checkpoint=str(checkpoint),
        )
        return {"id": eval_job["id"], "status": eval_job["status"]}

    @app.post("/api/models/{model_id}/release")
    def release(model_id: str, payload: ReleaseRequest) -> dict:
        meta = model_store.get_model(model_id)
        if meta is None:
            raise HTTPException(status_code=404, detail="model not found")
        if meta.get("source") != "server":
            raise HTTPException(
                status_code=400,
                detail="bundled models cannot be released from this server",
            )
        try:
            result = release_model(
                settings, model_id, meta, dry_run=payload.dry_run
            )
        except ReleaseError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return result

    return app


app = create_app()
