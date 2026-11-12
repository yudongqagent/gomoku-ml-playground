#!/usr/bin/env python3
"""jobctl — operate training jobs from the command line.

API-based: the training server must be running
(uvicorn app.main:app --port 8000, from server/).

Usage:
    python scripts/jobctl.py list
    python scripts/jobctl.py status <job-id>
    python scripts/jobctl.py pause <job-id>
    python scripts/jobctl.py resume <job-id>
    python scripts/jobctl.py cancel <job-id>
    python scripts/jobctl.py logs <job-id> [--tail N]

    python scripts/jobctl.py --base-url http://localhost:8000 list
"""

from __future__ import annotations

import argparse
import sys

import httpx


def _client(base_url: str) -> httpx.Client:
    # trust_env=False: the sandbox's no_proxy list contains bracketed
    # IPv6 entries ([::1]) that crash httpx's env parsing, and the
    # local API needs no proxy anyway.
    return httpx.Client(base_url=base_url, timeout=30, trust_env=False)


def _get(client: httpx.Client, path: str) -> dict:
    resp = client.get(path)
    _check(resp)
    return resp.json()


def _post(client: httpx.Client, path: str) -> dict:
    resp = client.post(path)
    _check(resp)
    return resp.json()


def _check(resp: httpx.Response) -> None:
    if resp.status_code >= 400:
        try:
            detail = resp.json().get("detail", resp.text)
        except ValueError:
            detail = resp.text
        print(f"error: HTTP {resp.status_code}: {detail}", file=sys.stderr)
        raise SystemExit(1)


def _fmt_job(job: dict) -> str:
    status = job["status"]
    extra = ""
    if job.get("phase"):
        extra += f" [{job['phase']}]"
    if job.get("progress") is not None:
        extra += f" {job['progress'] * 100:.0f}%"
    if job.get("games_generated"):
        extra += f" games={job['games_generated']}"
    return (
        f"{job['id']}  {job['type']:<9} {status:<12}{extra}  {job['name']}"
    )


def cmd_list(client: httpx.Client, args: argparse.Namespace) -> int:
    jobs = _get(client, "/api/jobs")["jobs"]
    if not jobs:
        print("no jobs")
        return 0
    for job in jobs:
        print(_fmt_job(job))
    return 0


def cmd_status(client: httpx.Client, args: argparse.Namespace) -> int:
    job = _get(client, f"/api/jobs/{args.job_id}")
    print(_fmt_job(job))
    print(f"  created:   {job['created_at']}")
    print(f"  started:   {job.get('started_at')}")
    print(f"  finished:  {job.get('finished_at')}")
    if job.get("resumed_at"):
        print(f"  resumed:   {job['resumed_at']}")
    if job.get("cancelled_at"):
        print(f"  cancelled: {job['cancelled_at']}")
    if job.get("note"):
        print(f"  note:      {job['note']}")
    if job.get("error"):
        print(f"  error:     {job['error']}")
    print(f"  games/hour: {job.get('games_per_hour')}")
    print(f"  buffer:     {job.get('buffer_size')} samples"
          + (" (estimate)" if job.get("buffer_size_estimate") else ""))
    print(f"  mcts:       {job.get('mcts')}")
    if job.get("eval"):
        print(f"  eval:       {job['eval']}")
    if job.get("model_id"):
        print(f"  model:      {job['model_id']}")
    return 0


def _transition(client: httpx.Client, job_id: str, action: str) -> int:
    job = _post(client, f"/api/jobs/{job_id}/{action}")
    print(_fmt_job(job))
    return 0


def cmd_pause(client: httpx.Client, args: argparse.Namespace) -> int:
    return _transition(client, args.job_id, "pause")


def cmd_resume(client: httpx.Client, args: argparse.Namespace) -> int:
    return _transition(client, args.job_id, "resume")


def cmd_cancel(client: httpx.Client, args: argparse.Namespace) -> int:
    return _transition(client, args.job_id, "cancel")


def cmd_logs(client: httpx.Client, args: argparse.Namespace) -> int:
    job = _get(client, f"/api/jobs/{args.job_id}")
    lines = job.get("logs") or []
    for line in lines[-args.tail:]:
        print(line)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://localhost:8000",
                        help="training server URL (default: %(default)s)")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list", help="list jobs, newest first")
    p = sub.add_parser("status", help="show a job's detail + throughput")
    p.add_argument("job_id")
    p = sub.add_parser("pause", help="pause a running job (SIGSTOP)")
    p.add_argument("job_id")
    p = sub.add_parser("resume", help="resume a paused/interrupted/failed job")
    p.add_argument("job_id")
    p = sub.add_parser("cancel", help="cancel a queued/running/paused job")
    p.add_argument("job_id")
    p = sub.add_parser("logs", help="show a job's log tail")
    p.add_argument("job_id")
    p.add_argument("--tail", type=int, default=50)
    args = parser.parse_args()
    try:
        with _client(args.base_url) as client:
            return {
                "list": cmd_list,
                "status": cmd_status,
                "pause": cmd_pause,
                "resume": cmd_resume,
                "cancel": cmd_cancel,
                "logs": cmd_logs,
            }[args.command](client, args)
    except httpx.ConnectError:
        print(f"error: cannot reach the server at {args.base_url} — "
              "is it running?", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
