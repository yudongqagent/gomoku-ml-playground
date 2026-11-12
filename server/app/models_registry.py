"""Model store: registered models from training jobs + bundled models.

Server models live under ``server/data/models/<model-id>/`` with
``model.json`` metadata + artifacts. Bundled models come from
``public/models/models.json`` (written by the ML workstream's
run_smoke.py); they are served by the web app itself, not this server,
so their ``onnx_url`` points at the Vite static path ``/models/<file>``.
"""

from __future__ import annotations

import json
import logging
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import PUBLIC_MODELS_DIR, Settings

log = logging.getLogger(__name__)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


class ModelStore:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    # -- server models -------------------------------------------------

    def _server_models(self) -> list[dict]:
        models: list[dict] = []
        models_dir = self.settings.models_dir
        if not models_dir.exists():
            return models
        for path in sorted(models_dir.glob("*/model.json")):
            try:
                meta = json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError) as exc:
                log.warning("skipping unreadable model metadata %s: %s", path, exc)
                continue
            meta["source"] = "server"
            models.append(meta)
        return models

    # -- bundled models (public/models/models.json) --------------------

    def _bundled_models(self) -> list[dict]:
        manifest_path = PUBLIC_MODELS_DIR / "models.json"
        if not manifest_path.exists():
            return []
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            log.warning("could not read bundled models manifest: %s", exc)
            return []
        models: list[dict] = []
        for entry in manifest.get("models", []):
            url = entry.get("url", "")
            models.append({
                "id": entry.get("id"),
                "name": entry.get("name"),
                "run_id": None,
                "job_id": None,
                "architecture": entry.get("architecture", "resnet"),
                "created_at": entry.get("created_at"),
                "onnx_url": url,  # served by the web app, not this server
                "formats": ["onnx"] if url.endswith(".onnx") else [],
                "metrics": {},
                "eval": entry.get("eval", {}),
                "trained_games": entry.get("trainedGames"),
                "description": entry.get("description"),
                "notes": entry.get("notes"),
                "source": "bundled",
            })
        return models

    # -- public API ----------------------------------------------------

    def list_models(self) -> list[dict]:
        models = self._server_models() + self._bundled_models()
        models.sort(key=lambda m: (m.get("created_at") or ""), reverse=True)
        return [self._public_view(m) for m in models]

    def get_model(self, model_id: str) -> dict | None:
        for model in self._server_models() + self._bundled_models():
            if model.get("id") == model_id:
                return model
        return None

    def get_model_detail(self, model_id: str) -> dict | None:
        model = self.get_model(model_id)
        if model is None:
            return None
        detail = self._public_view(model)
        if model.get("source") == "server":
            card_path = (
                self.settings.models_dir / model_id / "model_card.md"
            )
            detail["model_card"] = (
                card_path.read_text(encoding="utf-8")
                if card_path.exists()
                else None
            )
        else:
            detail["model_card"] = model.get("notes")
        detail["formats_detail"] = model.get("formats_detail", {})
        return detail

    def _public_view(self, model: dict) -> dict:
        """The shape the web UI codes against."""
        if model.get("source") == "server":
            formats_detail = model.get("formats_detail", {})
            formats = sorted(
                name for name, info in formats_detail.items()
                if info.get("file")
            )
            onnx_url = (
                f"/api/models/{model['id']}/download?format=onnx"
                if "onnx" in formats
                else None
            )
        else:
            formats = model.get("formats", [])
            onnx_url = model.get("onnx_url")
        return {
            "id": model.get("id"),
            "name": model.get("name"),
            "run_id": model.get("run_id"),
            "job_id": model.get("job_id"),
            "architecture": model.get("architecture"),
            "created_at": model.get("created_at"),
            "onnx_url": onnx_url,
            "formats": formats,
            "metrics": model.get("metrics", {}),
            "eval": model.get("eval", {}),
            "trained_games": model.get("trained_games"),
            "description": model.get("description"),
            "notes": model.get("notes"),
            "source": model.get("source"),
        }

    def artifact_path(self, model_id: str, fmt: str) -> Path | None:
        """Resolve a download format to a file on disk (server models only)."""
        model = self.get_model(model_id)
        if model is None or model.get("source") != "server":
            return None
        formats_detail = model.get("formats_detail", {})
        info = formats_detail.get(fmt)
        if not info or not info.get("file"):
            return None
        path = self.settings.models_dir / model_id / info["file"]
        return path if path.exists() else None

    # -- registration --------------------------------------------------

    def register_from_job(self, job: dict, artifacts: dict) -> dict:
        """Copy a completed job's artifacts into the model store.

        Model id: ``model-<run-id>``. Writes model.json + model_card.md +
        a copy of the resolved config + eval.json. Never claims strength
        beyond the recorded eval numbers (see render_model_card).
        """
        model_id = f"model-{job['run_id']}"
        model_dir = self.settings.models_dir / model_id
        model_dir.mkdir(parents=True, exist_ok=True)
        job_dir = self.settings.jobs_dir / job["id"]

        formats_detail: dict[str, Any] = {}

        def _copy(fmt: str, src: str | None, dest_name: str) -> None:
            if not src:
                return
            src_path = Path(src)
            if not src_path.exists():
                return
            dest = model_dir / dest_name
            shutil.copy2(src_path, dest)
            formats_detail[fmt] = {
                "file": dest_name,
                "size": dest.stat().st_size,
            }

        ckpt = artifacts.get("checkpoint", {})
        _copy("checkpoint", ckpt.get("path"), "model.pt")
        onnx = artifacts.get("onnx", {})
        _copy("onnx", onnx.get("path"), "model.onnx")
        fp16 = artifacts.get("onnx-fp16", {})
        if fp16.get("path"):
            _copy("onnx-fp16", fp16.get("path"), "model.fp16.onnx")
        else:
            formats_detail["onnx-fp16"] = {
                "file": None, "size": None,
                "note": fp16.get("note", "fp16 variant not produced"),
            }
        coreml = artifacts.get("coreml", {})
        if coreml.get("path") and Path(coreml["path"]).exists():
            zip_path = model_dir / "model.mlpackage.zip"
            pkg = Path(coreml["path"])
            with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
                for file in pkg.rglob("*"):
                    if file.is_file():
                        zf.write(file, Path("model.mlpackage") / file.relative_to(pkg))
            formats_detail["coreml"] = {
                "file": "model.mlpackage.zip",
                "size": zip_path.stat().st_size,
            }
        else:
            formats_detail["coreml"] = {
                "file": None, "size": None,
                "note": coreml.get("note", "Core ML export not produced"),
            }

        metrics = job.get("metrics") or []
        final_metric = metrics[-1] if metrics else {}
        meta: dict[str, Any] = {
            "id": model_id,
            "name": job.get("name") or model_id,
            "run_id": job["run_id"],
            "job_id": job["id"],
            "architecture": job.get("architecture", "resnet"),
            "arch_params": job.get("arch_params", {}),
            "created_at": _now_iso(),
            "trained_games": job.get("games_generated", 0),
            "metrics": {
                "final_loss": final_metric.get("loss"),
                "final_policy_loss": final_metric.get("policy_loss"),
                "final_value_loss": final_metric.get("value_loss"),
                "epochs": len(metrics),
            },
            "eval": job.get("eval", {}),
            "formats_detail": formats_detail,
            "source": "server",
            "notes": (
                "Trained on this server from self-play games. Strength is "
                "whatever the eval ladder below says — nothing more. This "
                "model has not been evaluated against human players."
            ),
        }
        (model_dir / "model.json").write_text(
            json.dumps(meta, indent=2), encoding="utf-8"
        )
        # Model card: prefer the ML run's own auto-generated card (the
        # train/eval scripts regenerate it from config + metrics.jsonl +
        # eval.json, and it carries an explicit honest-strength note).
        # Fall back to the server-rendered card if the run wrote none.
        ml_card = job_dir / "model_card.md"
        if ml_card.exists():
            shutil.copy2(ml_card, model_dir / "model_card.md")
        else:
            (model_dir / "model_card.md").write_text(
                render_model_card(meta, job), encoding="utf-8"
            )
        # Keep the resolved config + eval + metrics next to the model
        # for provenance.
        for name in ("config.yaml", "eval.json", "metrics.jsonl"):
            src = job_dir / name
            if src.exists():
                shutil.copy2(src, model_dir / name)
        log.info("registered model %s from job %s", model_id, job["id"])
        return meta


def render_model_card(meta: dict, job: dict) -> str:
    """Honest model card: config + metrics + eval numbers only.

    Never claims the model beats any human or any strength level beyond
    the recorded eval-ladder results.
    """
    cfg = job.get("config") or {}
    model_cfg = cfg.get("model", {})
    sp = cfg.get("selfplay", {})
    tr = cfg.get("train", {})
    lines = [
        f"# Model card: {meta.get('name')}",
        "",
        f"- **Model id:** `{meta.get('id')}`",
        f"- **Run id:** `{meta.get('run_id')}` (job `{meta.get('job_id')}`)",
        f"- **Created:** {meta.get('created_at')}",
        f"- **Architecture:** {meta.get('architecture')} "
        f"(channels={model_cfg.get('channels')}, "
        f"blocks={model_cfg.get('blocks')})",
        f"- **Self-play games trained on:** {meta.get('trained_games')}",
        f"- **Self-play settings:** {sp.get('games')} games, "
        f"{sp.get('sims_per_move')} MCTS sims/move, "
        f"temperature {sp.get('temperature')}",
        f"- **Training:** {tr.get('epochs')} epochs, "
        f"batch size {tr.get('batch_size')}, lr {tr.get('lr')}",
        "",
        "## Training metrics",
        "",
    ]
    metrics = meta.get("metrics", {})
    lines += [
        f"- Final loss: {metrics.get('final_loss')}",
        f"- Final policy loss: {metrics.get('final_policy_loss')}",
        f"- Final value loss: {metrics.get('final_value_loss')}",
        f"- Epochs: {metrics.get('epochs')}",
        "",
        "## Eval ladder",
        "",
        "Win/loss/draw counts against scripted opponents, from the "
        "`ml/scripts/evaluate.py` ladder run after training. These are "
        "the only strength numbers measured for this model.",
        "",
    ]
    eval_data = meta.get("eval") or {}
    # Enriched eval.json nests the ladder under "ladder"; older runs
    # stored a flat {opponent: {...}} dict. Handle both, and skip any
    # non-opponent value (verdict strings, elo dicts, ...) so a new
    # eval key can never crash card rendering.
    ladder = eval_data.get("ladder") if isinstance(eval_data, dict) else None
    if not isinstance(ladder, dict):
        ladder = eval_data
    ladder = {k: v for k, v in ladder.items() if isinstance(v, dict)}
    if not ladder:
        lines.append("No eval run recorded for this model.")
    for opponent, result in ladder.items():
        lines.append(
            f"- vs {opponent}: {result.get('wins')}W-"
            f"{result.get('losses')}L-{result.get('draws')}D "
            f"(Elo diff vs opponent: {result.get('elo_diff_vs_opponent')})"
        )
    if isinstance(eval_data, dict):
        verdict = eval_data.get("verdict")
        if isinstance(verdict, dict):
            label = verdict.get("label") or verdict.get("tier")
            explanation = verdict.get("explanation")
            lines += ["", f"**Verdict:** {label}"]
            if explanation:
                lines.append(str(explanation))
        elif verdict:
            lines += ["", f"**Verdict:** {verdict}"]
        elo = eval_data.get("elo_estimate")
        if isinstance(elo, dict) and elo.get("estimate") is not None:
            interval = elo.get("interval")
            interval_txt = (
                f" (95% interval {interval[0]:.0f}–{interval[1]:.0f})"
                if isinstance(interval, list) and len(interval) == 2
                else ""
            )
            lines.append(
                f"- Ladder Elo (relative to this ladder only): "
                f"{elo.get('estimate'):.0f}{interval_txt}"
            )
        tactical = eval_data.get("tactical")
        if isinstance(tactical, dict):
            passed = tactical.get("passed")
            total = tactical.get("total")
            if passed is not None and total is not None:
                lines.append(
                    f"- Tactical test suite: {passed}/{total} passed"
                )
    lines += [
        "",
        "## Strength note",
        "",
        "This model has not been evaluated against human players, and "
        "no claim is made about how it would fare against any human. "
        "Strength depends overwhelmingly on training scale (games, sims, "
        "network size); see `ml/configs/full.yaml` for the honest compute "
        "estimate for reaching genuinely strong play.",
        "",
        "## Formats",
        "",
    ]
    for fmt, info in (meta.get("formats_detail") or {}).items():
        if info.get("file"):
            lines.append(f"- {fmt}: `{info['file']}` ({info.get('size')} bytes)")
        else:
            lines.append(f"- {fmt}: not available — {info.get('note')}")
    lines.append("")
    return "\n".join(lines)
