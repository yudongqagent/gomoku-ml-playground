# ML batch 3 plan — port v27's threshold, deeper-teacher distillation, multi-task value net

## Starting point
- The program now runs two tracks (see `research/TRACKS.md`). This is
  the ML track's batch after batch 10 (heuristic).
- **ML champion is v8** (Elo 0.0, vs v0 0.545, tactical 20/20 +
  29/29 — best net-using version). **The ML bar is the same
  4-clause shape with v8 as reference:** (1) h2h vs v8 ≥100 games,
  win rate > 0.50 with Wilson 95% CI lower bound > 0.50; (2)
  ladder Elo > 0.0; (3) tactical ≥ 20/20 + 29/29 on the corrected
  suite; (4) all test suites green.
- **Versions v36–v38** (batch 10 owns v33–v35; confirm the next
  numbers from `research/VERSIONS.md` at start).
- The tactical suite is the CORRECTED suite (batch 9 added (9,7) to
  winning-quiet-setup-2's expected list). Re-verify v8's tactical
  on it first (adding an expected move can only raise pass rates,
  but verify).
- Master contains the v18/v23/v27/v28/v29 code (batch-9 merge), so
  the rollout-leaf teacher is on master. **v24 stays the strongest
  ML version on ladder Elo (52.5) and the only one to tie v8
  (0.515)** — the distilled rollout leaf is the benchmark to beat.
- **The value head is the binding constraint**, confirmed at 4x
  capacity (v31): the prior ranks the right move first and the
  net's value head overrides it (v26, v31 — the same failure mode).
  The heuristic leaf's blind spots (batch 10, H2/H3) are the same
  failure mode in miniature. The search horizon (100 sims) is the
  shared limit both tracks keep hitting.

## Tooling (do FIRST, alongside the version work)
- **Per-game seeding (`seed + game_index`) + uniqueness audit as
  standard battery hygiene.** Batch 10 found a resume seed-replay
  bug: `selfplay_batch.py` creates engines once with fixed seeds,
  so on auto-resume their RNG streams restart and the resumed
  segment replays the prefix when segment-start parity aligns
  (v34: games 0–33 = 66–99, 63 unique; v33: 71 unique). Worse,
  engine determinism alone collapses vs-v0 legs to 41–45 unique
  games even with NO resume (RNG is consumed only on exact ties;
  one White line repeats 16x). Nominal "100 games" overstates the
  independent sample by up to 2.3x. No batch-10 verdict turns on
  it (every clause-1 call stands on both the nominal and
  de-duplicated cuts, and v27's bar pass survives de-dup at
  0.7766, CI lower 0.68), but the fix is overdue. This is a
  pre-registered scheme change: it re-bases secondary-metric
  comparisons across batches, but head-to-head fairness within a
  battery is preserved (both engines play the same lines).
  Implement per-game seeding in the match drivers, add a
  uniqueness audit to the battery report (distinct-game count and
  de-dup rate for every leg), and verify with a kill-and-resume
  test (the resumed battery must NOT replay the prefix).
- The ladder driver now has auto-resume (batch 10's tooling,
  `eval_version.py --out`, on master) — use it; every ladder JSON
  must be a committed artifact, no recovery from raw logs.

## Candidates, in priority order
1. **H1 (v36) — v27's q-gap threshold tie-break ported to the value
   engine (v8 base).** The heuristic track's best mechanism,
   applied where the value engine has the same visit-tie problem.
   Pre-register τ (0.05, same as v27) and the exact tie-break rule
   before any eval.
2. **H2 (v37) — deeper-teacher distillation.** Label positions
   with 2-ply rollout-leaf values (the teacher plays 2 plies
   before evaluating). Targets the value head's quiet-setup
   blindness (v31's failure: the distilled net overvalues
   immediate threat patterns and undervalues quiet 2-ply setups).
   Pre-register the labelling procedure, the architecture (v24's
   ResNet 16ch/2 blocks), and the α sweep before any eval.
3. **H3 (v38) — multi-task value net.** Value + next-move threat
   classification as an auxiliary task. Targets the same failure
   mode from the training side (teach the net to distinguish
   threat patterns from quiet setups, so the value head stops
   overriding a correct prior on quiet positions). Pre-register
   the auxiliary task and loss weights before any eval.

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
  outcomes changed).
- **B1 data collapses White** (v25, v31 — both collapse as White;
  v24's engine-only data did not). Second data point for the
  Black-skewed-training-data hypothesis (Black wins 53.1% of B1
  games).
- Small-sample smokes mislead (the 100-game battery is the verdict);
  dev seeds flatter (v31: 0.70 vs v8 on dev, 0.40 on validation).
- **Fast loops run against v0 as well as v8.**

## Method (carried over)
- The fast 10-game loop stays the finding method: play 10 games, read
  every move, find each game's losing move (solver/deeper-search
  verified, never guessed), patch, regression-test, repeat.
- The 4-clause ML bar stays the validation gate.
- All match drivers auto-resume from partial `--out` files (ML batch
  1 H1's patch, on master) — use that durability; batteries must
  survive VM replacements.

## Merge discipline (batch-3 lesson)
- Run the full suite PLUS targeted per-version tests after every merge.
- Watch for the recurring traps: the silent auto-merge breakage on
  `ValueMCTS.__init__` (ML batch 1, ML batch 2 — duplicate
  parameters with NO conflict flag; check for duplicate parameters
  before running anything), the dead-code root-selection trap
  (batch 8, batch 9), and the shared-tail trap (batch 8, batch 9).

## Constraints
- Branch `research/ml-batch-3` from master (current HEAD — verify
  with git rev-parse before branching). Commit this plan first.
- Use your own git worktree (e.g. `~/workspace/projects/gomoku-ml3/`).
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
