# ML batch 6 plan — targeted White-collapse fix, 3-ply retrain of the patch data, parity-corrected labels

## Starting point
- The program now runs two tracks (see `research/TRACKS.md`). This is
  the ML track's batch after batch 13 (heuristic).
- **ML champion is v44** (3-ply-label distilled value net, ResNet
  16ch/2 blocks, α=0.125) — the first ML version ever to pass all
  4 clauses (ML batch 4, 2026-11-20): vs v8 0.67, CI clear of 0.50;
  ladder Elo 98.1 [−12.1, 208.2] (40-game legs; the interval
  includes 0 — the pass is on the point estimate, per the
  pre-registered rule); tactical 20/20 + 29/29; vs v0 0.58. First
  ML version ever to beat the ladder MCTS outright (25–15).
- **ML track's own bar (re-anchored on v44 when it passed):**
  (1) h2h vs v44 ≥100 games, win rate > 0.50 with Wilson 95% CI
  lower bound > 0.50; (2) ladder Elo > 98.1; (3) tactical ≥
  20/20 + 29/29 on the corrected suite; (4) all test suites green.
- **v37 remains the strongest ML version by head-to-head** (beats
  v44 0.69–0.31, v43 0.63, v42 0.68) even though v44 holds the
  champion title by the bar. Non-transitivity is the norm in the
  ML track — new versions are measured against v37 as well as the
  champion.
- **Versions v54–v56** (batch 13 owns v51–v53; confirm the next
  numbers from `research/VERSIONS.md` at start).
- The tactical suite is the CORRECTED suite (batch 9 added (9,7) to
  winning-quiet-setup-2's expected list). Re-verify v44's tactical
  on it first.
- **White collapse vs heuristic opponents is the top ML-track
  problem** (ML batches 4–5): v42 White 0.28 vs v0 / 0.15 vs v7;
  v43 White 0.21 vs v0; v44 White 0.16 vs v37; v49 White 0.22 vs
  v0, ladder heuristic White 3-17-0; v50-as-White 0.175 (3W-16L-1D)
  against the frozen baseline heuristic — while White is 0.44 in
  the mirror and 0.55 vs the ladder MCTS, so the collapse is
  opponent-dependent (heuristic opponents only). v48's diagnosis
  (ML batch 5): the White collapse is a GENERAL White-side
  value-head defect (v44's net undervalues White's wins and fails
  to see White is losing, on all four opponents, worst vs ML
  opponents), not heuristic-specific — but the data patch
  redistributed rather than improved (White fixed, Black fell by
  at least as much). This batch tests a TARGETED fix, not another
  global lever.
- **The label-depth lever is fully mapped at both parities and
  CLOSED** (ML batch 5): 1-ply 0.515/Elo 52.5, 2-ply 0.75/−34.9,
  3-ply 0.67/98.1, 4-ply 0.475/−137.0. Depth helps once (1→2) and
  saturates; the 2-ply recipe remains the strongest labelling.
  This batch works on label *quality*, not more depth.
- **The α lever is CLOSED for the 2-ply net** (ML batch 5, v50):
  v44's Elo gain came from its 3-ply labels, not the lower α (the
  mirror match vs v37 is 0.51 — the same net at α=0.125 vs 0.25
  is statistically indistinguishable over 100 games). If α is
  revisited, it should be on a 3-ply+ net, where the interaction
  with label depth is untested.
- **The ladder is 40 games per opponent per leg** (batch 11's
  re-basing — it worked: no saturation, intervals narrowed).
  Point estimates remain comparable to historical versions (same
  anchors, same formula). The ladder driver has auto-resume
  (`eval_version.py --out`, on master) — use it; every ladder
  JSON must be a committed artifact.
- The match drivers now seed per game (`seed + game_index`) with a
  uniqueness audit on every leg (ML batch 3 tooling, on master).

## Tooling prerequisite (do FIRST, before any by-side holdout numbers)
- **Fix `eval_value_holdout.py`'s side-to-move inference on distill
  shards** (ML batch 5 finding): its index-inferred side-to-move
  is wrong on distill shards (attenuates the true White gap by up
  to ~60%); iter-37/iter-44 by-side numbers are unreliable. Do
  not trust any by-side holdout numbers until this is fixed and
  the fix is verified (the fixed numbers should reproduce the
  known White gap on v44's holdout).

## Battery design (carried from ML batches 4–5)
- **Opening variety for ML-vs-ML legs.** It worked: the vs-v44 legs
  in ML batch 5 ran 96–100 unique games in 100. Use the same
  pre-registered `research/openings-ml4.json` (10 openings) for
  the vs-v44 legs in this batch. Report the uniqueness audit
  prominently for every leg.
- **α sweeps:** 20-game sweeps rank noisily (ML batch 5: winner's
  regression 0.55 → 0.475 at 100 games) — use more games per arm
  (at least 40 per arm, or a two-stage sweep: 20-game screen then
  a 40-game confirmation of the top two arms).

## Candidates, in priority order
1. **H1 (v54) — White-side training positions weighted toward
   heuristic-opponent games (the targeted White-collapse fix).**
   Use v48's diagnosis + v49's legs as evidence (v49 is the
   sharpest case: White 0.22 vs v0, ladder heuristic White
   3-17-0; v49's Black-vs-v37 0.68 vs its White 0.31 may locate
   the mechanism). Weight White-side training positions toward
   games against heuristic opponents (v0, the ladder MCTS, v7) —
   or apply a White-specific value correction — NOT another
   global lever. Future patch data should include ladder-MCTS
   legs (ladder White vs mcts was 0.15). Pre-register the
   weighting scheme and the mechanism check (does the played
   White gap vs heuristic opponents close WITHOUT Black-side
   strength falling?) before any eval. The v48 lesson: a fix
   that redistributes (White up, Black down by at least as much)
   is not an improvement — the battery must show a net gain.
2. **H2 (v55) — retrain the v48 patch data with 3-ply labels
   (v44's recipe).** Test whether the cost was the label depth
   or the data mix: v48's patch data (v37's 2-ply base + 17,354
   new quiet positions from v44's four battery legs, 49,452
   train positions) retrained with 3-ply labels instead of
   2-ply. If it plays at v44's level, the cost was the label
   depth; if it plays at v48's level, the cost was the data mix.
   Pre-register the labelling procedure and the α sweep before
   any eval. Do NOT retry another 2-ply retrain on that mix.
3. **H3 (v56) — parity-corrected 3-ply labels.** The 3-ply labels
   are systematically optimistic (mean +0.1094 vs 2-ply's
   ≈ −0.06; fraction positive 0.734 vs 0.234) — the odd-horizon
   parity shift that cost v44 played strength vs v37. Correct
   it by centering the labels (subtracting the mean shift) or by
   a symmetric label (the average of the 3-ply label and its
   negation under colour swap — pick ONE correction and
   pre-register it). If the correction recovers v37-level played
   strength while keeping v44's Elo, the parity shift was the
   cost; if it plays at v44's level, the shift was not the
   binding constraint. Pre-register the correction and the α
   sweep before any eval.

Max 3 versions, each pre-registered in its iteration log BEFORE any
eval.

## Standing lessons — ML track
- **Dedup:** the "10k dataset" is 3,244 UNIQUE games, not 7,099 —
  and v8's "4,350" was 1,449 (the B1 parallel generator seeds
  worker w with seed+w, so consecutive resume seeds replay each
  other's streams). Future legs must space resume seeds ≥ workers
  apart, and dedup-aware holdouts (by unique game) are essential.
- **Training at this scale needs the memmap path** (in-RAM load
  OOM-killed at 3.2 GB; load_shards + memmap bundle + lazy-index
  train() is bit-identical to the RAM path).
- **CLOSED levers** — do not reopen without a genuinely new
  mechanism: "more data" for the v8 recipe (v25 — 2.24x unique
  games played WORSE); the learned policy prior cannot replace the
  v1 prior (v26 — 0.20 vs v8); dedup + colour-balance as a strength
  lever (v30 — the mechanism test passed but the engine didn't
  improve; the White fix is opponent-dependent); scale does NOT
  break the distillation ceiling (v31 — 4x capacity with better
  holdout plays worse; holdout transfer does not predict played
  strength at this scale); the VCF check is near-neutral on an
  already-tactically-perfect engine (v32 — 282 fires, zero games'
  outcomes changed); the threshold tie-break for the value engine
  (v36 — net q-gaps are unimodal noise, max 0.0569 vs the rollout
  leaf's signal gaps 0.130–0.417); multi-task threat auxiliary in
  this form (v38 — the threat head scores 0% on forcing positions;
  forcing examples must come from engine positions and must not
  share the value trunk); history dropout in this form (v42/v43 —
  the 2-ply labels NEED the history planes; both gated versions
  passed their no-history gates decisively and played worse than
  the nets they were distilled from); label depth at both parities
  (v49 — 1-ply 0.515/52.5, 2-ply 0.75/−34.9, 3-ply 0.67/98.1,
  4-ply 0.475/−137.0; depth helps once, 1→2, then saturates;
  2-ply remains the strongest labelling); α for the 2-ply net
  (v50 — the mirror match vs v37 is 0.51; v44's Elo gain came
  from its 3-ply labels, not the lower α).
- **B1 data collapses White** (v25, v31 — both collapse as White;
  v24's engine-only data did not). Second data point for the
  Black-skewed-training-data hypothesis (Black wins 53.1% of B1
  games). The distilled nets (v24, v37) do NOT collapse as White
  vs ML opponents — but DO collapse vs heuristic opponents (v42,
  v43, v44, v49, v50) — the White-collapse problem this batch
  addresses with a targeted fix.
- **Holdout robustness does not predict played strength — now
  six instances** (v31, v38, v42, v43, v48, v49). The 100-game
  battery is the verdict, not the holdout.
- **History-dependence tracks parity** (ML batch 5): even-parity
  v37/v49 collapse without history (drops 0.195/0.179);
  odd-parity v44 doesn't (0.007).
- Small-sample smokes mislead; dev seeds flatter (v31: 0.70 vs v8
  on dev, 0.40 on validation).
- **Fast loops run against v0 as well as v44.**

## Method (carried over)
- The fast 10-game loop stays the finding method: play 10 games, read
  every move, find each game's losing move (solver/deeper-search
  verified, never guessed), patch, regression-test, repeat.
- The 4-clause ML bar stays the validation gate.
- All match drivers auto-resume from partial `--out` files (ML batch
  1 H1's patch, on master) and now seed per game (`seed +
  game_index`, ML batch 3 tooling, on master) with a uniqueness
  audit on every leg — use that durability and hygiene; batteries
  must survive VM replacements, and every leg reports its
  distinct-game count and de-dup rate.

## Merge discipline (batch-3 lesson)
- Run the full suite PLUS targeted per-version tests after every merge.
- Watch for the recurring traps: the silent auto-merge breakage on
  `ValueMCTS.__init__` (ML batch 1, ML batch 2 — duplicate
  parameters with NO conflict flag; check for duplicate parameters
  before running anything), the dead-code root-selection trap
  (batch 8, batch 9), and the shared-tail trap (batch 8, batch 9,
  ML batch 5).

## Constraints
- Branch `research/ml-batch-6` from master (current HEAD — verify
  with git rev-parse before branching). Commit this plan first.
- Use your own git worktree (e.g. `~/workspace/projects/gomoku-ml6/`).
- `ml/gomoku_ml/baseline.py` stays FROZEN (ladder anchor). Kernel
  re-sync (NumPy/WebGPU/Metal) only when a version passes the bar.
- No cloud GPU spend, no money, no wallets, no releases, no Colab
  driving, no Space edits. Yudong runs nothing.
- Push the branch when done. Push to master ONLY if a version passes
  all 4 clauses of the ML bar, with the same plain-merge approach as
  batch 3. A bar pass changes the ML champion and the daily
  cross-track match — say so explicitly in the report.
- The daily cross-track cron runs each morning ~08:37 PST; do not
  touch it or its output files. The daily pairing is v52 vs v44
  (swap rules), run in the main worktree (see TRACKS.md).
