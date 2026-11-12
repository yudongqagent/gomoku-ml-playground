"""Gomoku ML Playground — training server (FastAPI backend).

Runs training jobs on the server (never in the browser), owns a FIFO job
queue, streams progress to the web UI over SSE, stores checkpoints and
exported models, and can publish models to GitHub releases (dry-run by
default; see releases.py).
"""
