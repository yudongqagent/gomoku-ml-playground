# ML batch 4 plan — history robustness (v24 recipe + v37 net), 3-ply labels

## Starting point
- The program now runs two tracks (see `research/TRACKS.md`). This is
  the ML track's batch after batch 11 (heuristic).
- **ML champion is v8** (Elo 0.0, vs v0 0.545, tactical 20/20 +
  29/29 — best net-using version). **The ML bar is the same
  4-clause shape with v8 as reference:** (1) h2h vs v8 ≥100 games,
  win rate > 0.50 with Wilson 95% CI lower bound > 0.50; (2)
  ladder Elo > 0.0; (3) tactical ≥ 20/20 + 29/29 on the corrected
  suite; (4) all test suites green.
- **Versions v42–v44** (batch 11 owns v39–v41; confirm the next
  numbers from `research/VERSIONS.md` at start).
- The tactical suite is the CORRECTED suite (batch 9 added (9,7) to
  winning-quiet-setup-2's expected list). Re-verify v8's tactical
  on it first (adding an expected move can only raise pass rates,
  but verify).
- **v37 is the strongest ML version ever by head-to-head** (0.75 vs
  v8, CI clear of 0.50 — the first ML version ever to beat v8) but
  NOT the champion (it failed clause 2 on a cold 20-game ladder
  leg, Elo −34.9). New versions are measured against v8 for the
  bar and against v37 for head-to-head progress.
- **History-dependence is the top ML-track risk** (ML batch 3): both
  new nets (v37, v38) collapse on engine positions without history
  planes (v37: v7 holdout 0.7395 → 0.5163; v38: no-history sign
  0.5035), while v24's net is history-robust. This batch attacks
  it directly.
- **The ladder is now 40 games per opponent per leg** (batch 11's
  re-basing — narrows the ±~180 Elo noise that decided v37's
  clause 2). Point estimates remain comparable to historical
  versions (same anchors, same formula); v8's Elo 0.0 was measured
  at 20 games/opponent but the comparison holds on the same scale.
  The ladder driver has auto-resume (`eval_version.py --out`, on
  master) — use it; every ladder JSON must be a committed artifact.
- The match drivers now seed per game (`seed + game_index`) with a
  uniqueness audit on every leg (ML batch 3 tooling, on master).

## Battery design (pre-registered change)
- **Opening variety for ML-vs-ML legs.** The mirror-match
  determinism collapse persists despite per-game seeding (v36-vs-v8
  produced only 21 unique games in 100 — engine similarity, not
  RNG sharing). For the vs-v8 legs in this batch, use a fixed set
  of diverse opening positions (pre-register the opening set —
  e.g. a spread of first-move placements — before any eval) so
  the leg's effective N is real. Head-to-head fairness is
  preserved (both engines play the same openings, alternating
  colours). Report the uniqueness audit prominently for every leg.
  This re-bases the vs-v8 comparison across batches, but the bar
  is about beating v8 in the same battery, not about cross-batch
  comparison.

## Candidates, in priority order
1. **H1 (v42) — history augmentation / history-dropout on v24's
   recipe.** The top priority from ML batch 3. Pre-register the
   augmentation/dropout scheme (e.g. random history-plane dropout
   during training, or training on both full-planes and no-history
   versions of each position) before any eval. **Gate: the
   no-history holdout must match full-planes BEFORE any games are
   played** — if the gate fails, the version does not go to battery
   (report the holdout numbers and stop; do not burn battery time
   on a net that fails its own gate).
2. **H2 (v43) — history dropout on the v37 net.** 2-ply labels +
   history-robustness: combine the two strongest ML findings
   (v37's 2-ply distillation, which moved played strength +0.235
   over v24, with the history-robustness fix). Pre-register the
   dropout scheme and the α sweep before any eval. Same gate as
   H1: no-history holdout must match full-planes before any games.
3. **H3 (v44) — 3-ply labels.** Extend v37's 2-ply success one ply
   deeper (if 2-ply helped +0.235, 3-ply may help more — or hit
   diminishing returns). Pre-register the labelling procedure and
   the α sweep before any eval. Measure the labelling cost first
   (2-ply was 11.15 ms/position, 1.61x 1-ply; 3-ply will be
   ~3–4x — still feasible for 37k positions). Measure the net's
   accuracy by side to move on the holdout (the White-collapse
   check from v30's method).

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
  share the value trunk).
- **B1 data collapses White** (v25, v31 — both collapse as White;
  v24's engine-only data did not). Second data point for the
  Black-skewed-training-data hypothesis (Black wins 53.1% of B1
  games). The distilled nets (v24, v37) do NOT collapse as White.
- **Holdout transfer does not predict played strength** (v31, v38
  — better holdout, worse play; v37 — 0.7519 holdout, 0.75 vs v8).
  The 100-game battery is the verdict, not the holdout.
- Small-sample smokes mislead; dev seeds flatter (v31: 0.70 vs v8
  on dev, 0.40 on validation).
- **Fast loops run against v0 as well as v8.**

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
  (batch 8, batch 9), and the shared-tail trap (batch 8, batch 9).

## Constraints
- Branch `research/ml-batch-4` from master (current HEAD — verify
  with git rev-parse before branching). Commit this plan first.
- Use your own git worktree (e.g. `~/workspace/projects/gomoku-ml4/`).
- `ml/gomoku_ml/baseline.py` stays FROZEN (ladder anchor). Kernel
  re-sync (NumPy/WebGPU/Metal) only when a version passes the bar.
- No cloud GPU spend, no money, no wallets, no releases, no Colab
  driving, no Space edits. Yudong runs nothing.
- Push the branch when done. Push to master ONLY if a version passes
  all 4 clauses of the ML bar, with the same plain-merge approach as
  batch 3. A bar pass changes the ML champion and the daily
  cross-track match — say so explicitly in the report.
- The daily cross-track cron runs each morning ~08:37 PST; do not
  touch it or its output files. The daily pairing is unchanged
  (v18 vs v8, swap rules).
