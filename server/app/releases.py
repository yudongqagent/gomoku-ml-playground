"""GitHub release publishing for trained models.

Two modes:

* Dry-run (default, and the only mode the web UI uses first): validates
  config + asset existence and returns exactly what WOULD be uploaded
  (repo, tag, title, body preview, asset list with sizes). No GitHub
  call is made.
* Real upload: prefers the ``gh`` CLI (``gh release create``) when it is
  installed and authenticated; falls back to the GitHub REST API via
  httpx (needs ``GITHUB_TOKEN``).

Tag per run: ``model-<run-id>``. Release title = model name. Body = the
model card markdown (honest eval numbers only — see models_registry).
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import httpx

from .config import Settings

log = logging.getLogger(__name__)

GITHUB_API = "https://api.github.com"
GITHUB_UPLOADS = "https://uploads.github.com"


class ReleaseError(Exception):
    """Release planning/upload failed; message is safe to show to the UI."""


def gh_available() -> bool:
    return shutil.which("gh") is not None


def gh_authenticated() -> bool:
    if not gh_available():
        return False
    result = subprocess.run(
        ["gh", "auth", "status"], capture_output=True, timeout=15
    )
    return result.returncode == 0


def _github_request_kwargs() -> dict:
    """httpx kwargs for talking to api.github.com from this sandbox.

    The sandbox routes egress through a proxy with a custom CA bundle,
    and its ``no_proxy`` list contains bracketed IPv6 entries (``[::1]``)
    that crash httpx's env parsing (InvalidURL, which is NOT an
    httpx.HTTPError in httpx 0.28). So: explicit proxy + trust_env=False
    (skip env parsing) + the sandbox CA bundle when present.
    """
    proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy")
    verify = os.environ.get("SSL_CERT_FILE") or True
    return {"trust_env": False, "proxy": proxy, "verify": verify}


def check_repo_exists(repo: str) -> bool | None:
    """Check whether a (public) GitHub repo exists, no auth needed.

    Returns True if it exists, False if GitHub says 404, None if the
    check itself failed (network error, rate limit, unexpected status)
    — callers must treat None as "unknown", never as a failure.
    """
    try:
        resp = httpx.get(
            f"{GITHUB_API}/repos/{repo}", timeout=10, **_github_request_kwargs()
        )
    except Exception as exc:  # noqa: BLE001 — best-effort check, never crash
        log.debug("repo existence check failed for %s: %s", repo, exc)
        return None
    if resp.status_code == 200:
        return True
    if resp.status_code == 404:
        return False
    log.debug("repo existence check for %s: unexpected status %s",
              repo, resp.status_code)
    return None


def _asset_list(model_dir: Path, meta: dict) -> list[dict]:
    """Assets to upload, in upload order, from the model dir."""
    assets: list[dict] = []
    formats = meta.get("formats_detail", {})
    # Order: checkpoint, ONNX, fp16 ONNX, Core ML zip, model card.
    for fmt, fallback_name in (
        ("checkpoint", "model.pt"),
        ("onnx", "model.onnx"),
        ("onnx-fp16", "model.fp16.onnx"),
        ("coreml", "model.mlpackage.zip"),
    ):
        info = formats.get(fmt) or {}
        name = info.get("file")
        if not name:
            continue
        path = model_dir / name
        if not path.exists():
            continue
        assets.append({
            "name": name,
            "path": str(path),
            "size": path.stat().st_size,
            "format": fmt,
        })
    card = model_dir / "model_card.md"
    if card.exists():
        assets.append({
            "name": "model_card.md",
            "path": str(card),
            "size": card.stat().st_size,
            "format": "model_card",
        })
    return assets


def plan_release(
    settings: Settings, model_id: str, meta: dict, repo: str | None = None,
    check_repo: bool = False,
) -> dict:
    """Build the release plan for a model.

    Raises ReleaseError if the repo is unset, the model has no run id,
    or no uploadable assets exist. With ``check_repo=True`` it also
    checks (unauthenticated, public API) whether the repo exists — a
    missing repo is reported as a WARNING in the plan, never a hard
    failure: the dry-run must validate everything it can without the
    repo existing yet.
    """
    repo = repo or settings.model_repo
    if not repo or "/" not in repo:
        raise ReleaseError(
            "no model repo configured: set GOMOKU_MODEL_REPO=owner/repo "
            "(or model_repo in server/config.yaml)"
        )
    run_id = meta.get("run_id")
    if not run_id:
        raise ReleaseError(
            f"model {model_id!r} has no run_id (bundled models cannot be "
            "released from here)"
        )
    model_dir = settings.models_dir / model_id
    if not model_dir.exists():
        raise ReleaseError(f"model directory not found: {model_dir}")
    card_path = model_dir / "model_card.md"
    body = (
        card_path.read_text(encoding="utf-8")
        if card_path.exists()
        else f"# {meta.get('name', model_id)}\n\nNo model card recorded.\n"
    )
    assets = _asset_list(model_dir, meta)
    if not assets:
        raise ReleaseError(f"model {model_id!r} has no uploadable assets")
    tag = f"model-{run_id}"
    repo_exists: bool | None = None
    repo_warning: str | None = None
    if check_repo:
        repo_exists = check_repo_exists(repo)
        if repo_exists is False:
            repo_warning = (
                f"repo {repo} does not exist yet — create it first "
                "(Yudong creates the empty public repo himself); "
                "a real release upload will fail until it exists"
            )
        elif repo_exists is None:
            repo_warning = (
                f"could not verify whether repo {repo} exists "
                "(network/API check failed); proceeding anyway"
            )
    return {
        "model_id": model_id,
        "repo": repo,
        "tag": tag,
        "title": meta.get("name") or model_id,
        "body": body,
        "body_preview": body[:500] + ("…" if len(body) > 500 else ""),
        "assets": assets,
        "total_size": sum(a["size"] for a in assets),
        "method": (
            "gh" if gh_available() and gh_authenticated()
            else "api" if settings.github_token
            else "unavailable"
        ),
        "gh_available": gh_available(),
        "gh_authenticated": gh_authenticated(),
        "github_token_set": bool(settings.github_token),
        "repo_exists": repo_exists,
        "repo_warning": repo_warning,
    }


def release_model(
    settings: Settings, model_id: str, meta: dict,
    dry_run: bool = True, repo: str | None = None, check_repo: bool = False,
) -> dict:
    """Dry-run or perform a GitHub release upload for a model."""
    plan = plan_release(settings, model_id, meta, repo=repo, check_repo=check_repo)
    if dry_run:
        return {"dry_run": True, **plan}
    if plan["method"] == "gh":
        result = _upload_with_gh(plan)
    elif plan["method"] == "api":
        result = _upload_with_api(settings, plan)
    else:
        raise ReleaseError(
            "no GitHub auth available: run `gh auth login`, or set "
            "GITHUB_TOKEN for the REST API fallback"
        )
    return {"dry_run": False, **plan, **result}


def _upload_with_gh(plan: dict) -> dict:
    cmd = [
        "gh", "release", "create", plan["tag"],
        "--repo", plan["repo"],
        "--title", plan["title"],
        "--notes", plan["body"],
    ] + [a["path"] for a in plan["assets"]]
    log.info("creating release via gh: %s %s", plan["repo"], plan["tag"])
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        raise ReleaseError(
            f"`gh release create` failed (exit {result.returncode}): "
            f"{result.stderr.strip() or result.stdout.strip()}"
        )
    return {"release_url": result.stdout.strip().splitlines()[-1]
            if result.stdout.strip() else None}


def _upload_with_api(settings: Settings, plan: dict) -> dict:
    token = settings.github_token
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    with httpx.Client(timeout=120, **_github_request_kwargs()) as client:
        resp = client.post(
            f"{GITHUB_API}/repos/{plan['repo']}/releases",
            headers=headers,
            json={
                "tag_name": plan["tag"],
                "name": plan["title"],
                "body": plan["body"],
            },
        )
        if resp.status_code not in (200, 201):
            raise ReleaseError(
                f"GitHub release create failed ({resp.status_code}): "
                f"{resp.text[:500]}"
            )
        release = resp.json()
        upload_url = release["upload_url"].split("{")[0]
        for asset in plan["assets"]:
            with open(asset["path"], "rb") as f:
                data = f.read()
            up = client.post(
                f"{upload_url}?name={asset['name']}",
                headers={**headers, "Content-Type": "application/octet-stream"},
                content=data,
                timeout=600,
            )
            if up.status_code not in (200, 201):
                raise ReleaseError(
                    f"asset upload failed for {asset['name']} "
                    f"({up.status_code}): {up.text[:300]}"
                )
        return {"release_url": release.get("html_url")}
