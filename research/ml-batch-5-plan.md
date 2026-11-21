# ML batch 5 plan — White-collapse diagnosis, 4-ply labels, α ablation

## Starting point
- The program now runs two tracks (see `research/TRACKS.md`). This is
  the ML track's batch after batch 12 (heuristic).
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
- **Versions v48–v50** (batch 12 owns v45–v47; confirm the next
  numbers from `research/VERSIONS.md` at start).
- The tactical suite is the CORRECTED suite (batch 9 added (9,7) to
  winning-quiet-setup-2's expected list). Re-verify v44's tactical
  on it first (adding an expected move can only raise pass rates,
  but verify).
- **White collapse vs heuristic opponents is the top ML-track
  problem** (ML batch 4): v42 White 0.28 vs v0 / 0.15 vs v7;
  v43 White 0.21 vs v0; v44 White 0.16 vs v37 — while the same
  nets' White vs v8 is 0.59–0.67. The distilled nets don't
  collapse vs ML opponents but do vs heuristic ones. This batch
  diagnoses it first.
- **The label-depth lever is mapped** (ML batch 4): 1-ply
  0.515/Elo 52.5, 2-ply 0.75/−34.9, 3-ply 0.67/98.1 — depth helps
  once (1→2), then the odd-horizon parity shift dominates (3-ply
  labels are systematically optimistic, mean +0.1094 vs 2-ply's
  ≈ −0.06; fraction positive 0.734 vs 0.234; sign agreement with
  2-ply only 0.492). This batch tests whether depth helps again
  at even parity (4-ply).
- **The ladder is 40 games per opponent per leg** (batch 11's
  re-basing — it worked: no saturation, intervals narrowed).
  Point estimates remain comparable to historical versions (same
  anchors, same formula). The ladder driver has auto-resume
  (`eval_version.py --out`, on master) — use it; every ladder
  JSON must be a committed artifact.
- The match drivers now seed per game (`seed + game_index`) with a
  uniqueness audit on every leg (ML batch 3 tooling, on master).

## Battery design (carried from ML batch 4)
- **Opening variety for ML-vs-ML legs.** It worked: the vs-v8 legs
  in ML batch 4 ran 91–97 unique games in 100 (v42: 91, v43: 97,
  v44: 93), vs 21 for v36-vs-v8 without openings. Use the same
  pre-registered `research/openings-ml4.json` (10 openings) for
  the vs-v44 legs in this batch. Report the uniqueness audit
  prominently for every leg.

## Candidates, in priority order
1. **H1 (v48) — diagnose the White collapse vs heuristic
   opponents, then patch the mechanism found.** The top priority
   from ML batch 4. Measure the value head's accuracy by side to
   move on heuristic-opponent positions vs ML-opponent positions
   (the v30 method: holdout accuracy split by side to move and by
   opponent type). Find whether it's the value head misreading
   greedy-heuristic positions or a search-horizon effect. Then
   patch the mechanism found — e.g. if the value head misreads
   greedy-heuristic positions, train on positions from games
   against the heuristic anchor (v0) and the ladder MCTS; if it's
   a search-horizon effect, the patch is search-side (and must
   not touch a CLOSED lever). Pre-register the diagnosis
   (measurements + the decision rule for which mechanism it is)
   and the patch before any eval. If the diagnosis finds no
   patchable mechanism, say so and fall back to the α-ablation
   direction (H3's area, with a distinct mechanism).
2. **H2 (v49) — 4-ply labels.** The even horizon should reverse
   the parity shift that cost v44, testing whether depth helps
   again at even parity. Pre-register the labelling procedure and
   the α sweep before any eval. Measure the labelling cost first
   (2-ply was 11.15 ms/position, 1.61x 1-ply; 3-ply was measured
   in ML batch 4 — check iter-44 for the number; 4-ply will be
   ~4–6x 1-ply, still feasible for 37k positions). Measure the
   net's accuracy by side to move on the holdout (the
   White-collapse check from v30's method).
3. **H3 (v50) — 2-ply labels at α=0.125 (the α ablation).** v44
   differs from v37 in TWO ways (3-ply labels AND α=0.125 vs
   v37's α=0.25). This version isolates the α effect: the v37 net
   (2-ply labels, identical positions) trained/evaluated at
   α=0.125. If it plays at v44's level, v44's Elo gain came from
   the lower α, not the 3-ply labels; if it plays at v37's level,
   the labels drove the gain. Pre-register the α and the
   comparison (vs v37 and vs v44) before any eval. This is a
   clean ablation, not a new mechanism — report it as such.

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
  the nets they were distilled from).
- **B1 data collapses White** (v25, v31 — both collapse as White;
  v24's engine-only data did not). Second data point for the
  Black-skewed-training-data hypothesis (Black wins 53.1% of B1
  games). The distilled nets (v24, v37) do NOT collapse as White
  vs ML opponents — but DO collapse vs heuristic opponents (v42,
  v43, v44) — the White-collapse problem this batch diagnoses.
- **Holdout robustness does not predict played strength — now
  four instances** (v31, v38, v42, v43). The 100-game battery is
  the verdict, not the holdout.
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
  (batch 8, batch 9), and the shared-tail trap (batch 8, batch 9).

## Constraints
- Branch `research/ml-batch-5` from master (current HEAD — verify
  with git rev-parse before branching). Commit this plan first.
- Use your own git worktree (e.g. `~/workspace/projects/gomoku-ml5/`).
- `ml/gomoku_ml/baseline.py` stays FROZEN (ladder anchor). Kernel
  re-sync (NumPy/WebGPU/Metal) only when a version passes the bar.
- No cloud GPU spend, no money, no wallets, no releases, no Colab
  driving, no Space edits. Yudong runs nothing.
- Push the branch when done. Push to master ONLY if a version passes
  all 4 clauses of the ML bar, with the same plain-merge approach as
  batch 3. A bar pass changes the ML champion and the daily
  cross-track match — say so explicitly in the report.
- The daily cross-track cron runs each morning ~08:37 PST; do not
  touch it or its output files. The daily pairing is v18 vs v44
  (swap rules), run in the main worktree (see TRACKS.md).
