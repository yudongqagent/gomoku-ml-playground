#!/usr/bin/env python3
"""Publish a trained model as a GitHub release (standalone CLI).

Dry-run is the default and the ONLY mode that should be used until a
model repo exists and `gh auth login` (or GITHUB_TOKEN) is set up:

    python server/scripts/publish_release.py --model-id model-20261112-093000-ab12cd --dry-run

To actually create the release (requires GOMOKU_MODEL_REPO + auth):

    python server/scripts/publish_release.py --model-id <id> --execute

Tag: model-<run-id>. Assets: checkpoint .pt, .onnx, .fp16.onnx (if
present), .mlpackage.zip (if present), model_card.md. Release body is
the model card (honest eval numbers only).
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import load_settings  # noqa: E402
from app.models_registry import ModelStore  # noqa: E402
from app.releases import ReleaseError, release_model  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", required=True, help="model id, e.g. model-20261112-093000-ab12cd")
    parser.add_argument("--repo", default=None, help="override GOMOKU_MODEL_REPO (owner/repo)")
    parser.add_argument("--data-dir", default=None, help="override data dir (default: server/data)")
    parser.add_argument("--dry-run", dest="dry_run", action="store_true", default=True,
                        help="validate + print what would be uploaded; no GitHub call (default)")
    parser.add_argument("--execute", dest="dry_run", action="store_false",
                        help="actually create the GitHub release (requires auth)")
    parser.add_argument("--json", action="store_true", help="print the full plan/result as JSON")
    args = parser.parse_args()

    settings = load_settings(data_dir=args.data_dir)
    store = ModelStore(settings)
    meta = store.get_model(args.model_id)
    if meta is None:
        print(f"error: model not found: {args.model_id}", file=sys.stderr)
        return 1
    if meta.get("source") != "server":
        print(f"error: bundled model {args.model_id!r} cannot be released from here",
              file=sys.stderr)
        return 1

    try:
        # The CLI checks repo existence (unauthenticated public API call);
        # a missing repo is a warning, never a failure — the dry-run must
        # validate everything it can without the repo existing yet.
        result = release_model(
            settings, args.model_id, meta,
            dry_run=args.dry_run, repo=args.repo, check_repo=True,
        )
    except ReleaseError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(result, indent=2))
        return 0

    if result["dry_run"]:
        print("DRY RUN — nothing was uploaded to GitHub.")
    else:
        print("RELEASE CREATED.")
    if result.get("repo_warning"):
        print(f"  WARNING:     {result['repo_warning']}")
    print(f"  repo:        {result['repo']} "
          f"(exists: {result.get('repo_exists')})")
    print(f"  tag:         {result['tag']}")
    print(f"  title:       {result['title']}")
    print(f"  method:      {result['method']} "
          f"(gh installed: {result['gh_available']}, "
          f"gh authenticated: {result['gh_authenticated']}, "
          f"GITHUB_TOKEN set: {result['github_token_set']})")
    print(f"  assets ({len(result['assets'])}, "
          f"total {result['total_size']} bytes):")
    for asset in result["assets"]:
        print(f"    - {asset['name']} ({asset['size']} bytes)")
    print("  body preview:")
    for line in result["body_preview"].splitlines()[:12]:
        print(f"    {line}")
    if result.get("release_url"):
        print(f"  release_url: {result['release_url']}")
    if result["dry_run"] and result["method"] == "unavailable":
        print()
        print("NOTE: no GitHub auth is configured yet — a real upload would "
              "fail. Run `gh auth login` or set GITHUB_TOKEN. (The default "
              "repo is yudongqagent/gomoku-ml-playground; override with "
              "GOMOKU_MODEL_REPO if needed.)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
