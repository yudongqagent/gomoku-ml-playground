# LIMITS — honest strength, gaps, and what it takes

This file states plainly what this project can and cannot do today.
Nothing here claims superhuman play, and no number is rounded up.

## What v1 actually plays like

Measured, not promised:

- **L3 Advanced / L4 Expert (search-based)** are the strongest
  out-of-the-box players: strong club/amateur level. They find
  immediate wins, block open fours, and spot many forced sequences —
  but they have no quiescence search, VCF follows fours only, and
  search branching is capped (~12–14 moves). A strong human player
  will beat them.
- **L5 Master with the bundled demo net** (`gomoku-smoke-v1`, 8
  self-play games, 16 channels / 2 blocks) plays at roughly L4 level
  via its MCTS fallback and the weak net. The demo net on its own was
  fully evaluated (40 games/opponent, 32 sims, 2026-11-12):
  vs random **30W–10L** (win rate 0.75, 95% CI [0.60, 0.86]),
  vs the 1-ply heuristic **0W–40L**, vs heuristic-MCTS **0W–40L**,
  tactical suite **0/20** (the heuristic baseline scores 20/20),
  ladder Elo **−609.2** (interval [−731.0, −487.3]), verdict
  **"Below heuristic"**. It is included to prove the pipeline
  end-to-end, not to be a strong opponent. (The earlier 4-game smoke
  eval showed 3W–1L vs random and could not resolve the 25% loss
  rate — this is why the harness reports confidence intervals.)
- **Win rates shown in the UI** are estimates: search levels use an
  uncalibrated logistic squash of the eval score; MCTS levels use
  visit counts / mean values. They are useful for comparing moves,
  not calibrated probabilities.

## What "beats the best human easily" actually takes

Superhuman Gomoku is an AlphaZero-scale endeavour:

- **~1M+ self-play games** (~10¹⁰–10¹¹ network evaluations).
- **~1,000–10,000 GPU-hours** on A100/H100-class hardware — weeks to
  months on a single A100, plausibly 1–4 weeks on 8× A100.
- A full training loop (self-play → train → eval → promote the
  better net, repeated) with a large replay buffer and careful
  hyperparameter tuning — the pipeline in this repo, run at that
  scale, not a different pipeline.

What smaller budgets reach (honest expectations):

| Budget | Realistic outcome |
|---|---|
| Smoke run (~2 min CPU) | A demo net that loses to the heuristic. Proves the pipeline. |
| A few hours on CPU (`longrun-cpu` preset) | At best a weak club-ish player that sometimes beats the heuristic. Not an expert. |
| One `full.yaml` round (~5,000 games, 400 sims; ~1–2 days on a single A100) | A competent club-level player. Not superhuman. |
| AlphaZero-scale (above) | The only known route to superhuman. |

Measured throughput on the build VM (~1.5 shared CPUs, torch CPU,
2026-11-12 — re-measure on your hardware): self-play **~150
games/hour at 100 sims**; one full `longrun-cpu` round (200 games +
20 epochs + auto-eval) **≈ 2–2.5 hours**; a 6-hour run is ~2 rounds
(~400 games, ~28k samples). Auto-eval of 40 games × 3 opponents +
tactics takes ~12 minutes at this scale (longer as models improve
and games lengthen).

**On-device training at that scale is not realistic.** The intended
flow is: train on GPU servers, deploy the frozen model on device
(ONNX in the browser, Core ML natively). The A20 Pro's Neural Engine
is a deployment target, not a training cluster.

## What "a good level" means here (operational definition)

Defined in `ml/README.md` and used by the eval harness verdicts:

> A model plays at a **good level** when it wins ≥50% against the
> heuristic-MCTS ladder opponent as both colors over ≥40 games, with
> a ≥80% tactical test pass rate.

This is a v1 working definition tied to this repo's ladder — it will
be recalibrated once stronger opponents exist. Ladder Elo in eval
reports is relative to this ladder's anchors only and is **not**
comparable to human ratings.

## Known gaps and limits

- **No quiescence search** (horizon effect), **VCF follows fours
  only** (not open-three threats, no defender counter-threats),
  **MCTS is single-threaded, batch=1** (one net call per expansion).
- **Core ML export was not executed here**: the script is
  code-complete, but `coremltools` is not installed on the Linux
  build VM. Run it on a Mac:
  `pip install coremltools && python ml/scripts/export_coreml.py
  --checkpoint ml/runs/<run-id>/checkpoints/final.pt --out
  GomokuNet.mlpackage`.
- **fp16 ONNX quantisation** is implemented (`--quantize`) but needs
  `onnxconverter-common`, which is not in `ml/.venv`; dynamic int8
  quantisation is documented with the exact command, not automated.
- **In the browser the Neural Engine is not directly accessible.**
  WebGPU via ONNX Runtime Web is the fastest web path; Core ML is the
  path to the Neural Engine via a native wrapper (see
  `docs/ios-coreml.md`). No full Xcode project is included.
- **ONNX Runtime Web's WASM bundle is ~29.5 MB** (JSEP/WebGPU build)
  — a first-load-size consideration for deployment.
- **The training server runs one job at a time** (FIFO). No GPU
  scheduling, no distributed self-play, no multi-user support.
- **Train-only jobs report `buffer_size: null`** (their data lives in
  another job's directory).
- **Self-play resume tops up** to the configured game count (a
  resumed job finishes the originally requested N games total, it
  does not generate N more). Training resume runs *additional*
  epochs — the interrupt message prints the exact `--epochs
  <remaining>` to finish the original plan.
- **No live-browser end-to-end test** of the UI was possible in this
  environment; the UI is verified at the code/contract level, plus
  served-build smoke checks (see the verification notes in the
  handoff).
- **Rules**: standard Gomoku only — no Renju forbidden moves, no
  Swap2, no captures. Overline (6+) counts as a win.
- **GitHub release upload**: the target repo
  `yudongqagent/gomoku-ml-playground` does not exist yet. Dry-runs
  work and warn; a real upload requires the repo plus `gh` auth or
  `GITHUB_TOKEN`. No release has been published.

## What to train next

1. Run the `longrun-cpu` preset for a few hours; read the eval
   report (ladder + tactics + verdict) — it tells you where the
   model actually stands.
2. Scale the loop: more self-play games per round, larger net
   (`resnet` 64–128 channels), more MCTS sims, and promote only
   checkpoints that beat the previous best (checkpoint-vs-checkpoint
   eval exists for this).
3. Compare architectures with `ml/scripts/compare_runs.py` — that is
   the point of the registry (`resnet` vs `mobilenet` vs
   `resnet-se`).
4. When a model beats the full ladder convincingly, the next step is
   GPU training at `full.yaml` scale, then iterating the loop — not
   more CPU hours on a tiny net.
