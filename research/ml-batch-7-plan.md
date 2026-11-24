# ML batch 7 plan — White fix that survives 3-ply labelling, v54 α ablation, centering correction

## Starting point
- The program now runs two tracks (see `research/TRACKS.md`). This is
  the ML track's batch after batch 14 (heuristic).
- **ML bar reference is v44** (3-ply labels, α=0.125; ladder Elo
  98.1 [−12.1, 208.2]; vs v8 0.67, CI [0.5731, 0.7544]; tactical
  20/20 + 29/29; first ML version ever to beat the ladder MCTS
  outright, 25–15) — the first ML version ever to pass all 4
  clauses (ML batch 4, 2026-11-20). **ML champion is also v44.**
- **v37 is the strongest ML version by head-to-head** (beats v44
  0.69–0.31) but NOT champion (failed clause 2 in ML batch 3 on
  a cold 20-game ladder leg, Elo −34.9). New versions are
  measured against v37 as well as the champion.
- **The 4-clause bar** (`research/README.md`), re-anchored on v44:
  (1) h2h vs v44 ≥100 games, win rate > 0.50 with Wilson 95% CI
  lower bound > 0.50; (2) ladder Elo > 98.1 (see the MCTS-leg
  rule below); (3) tactical ≥ 20/20 + 29/29; (4) all suites
  green.
- **Clause 2 MCTS-leg rule (NEW, pre-registered for this batch):**
  the ladder Elo is computed from the heuristic anchor leg only
  and is blind to the MCTS leg — ML batch 6 showed both v54
  (Elo 168.4) and v56 (Elo 137.0) passing Elo while losing the
  MCTS leg outright (16-23-1, 14-25-1). From this batch, clause 2
  requires BOTH the Elo point estimate > 98.1 AND no MCTS-leg
  loss (mcts leg win rate ≥ 0.50 on the point estimate). This
  rule is prospective (applies to ML batch 7 onwards) and does
  not change any past verdict (v54/v56 already failed clause 1;
  every past version that passed clause 1 — v37, v44, v50 — has
  an MCTS leg ≥ 0.50). Report the MCTS leg alongside the Elo in
  every VERSIONS.md row.
- **Versions v60–v62** (batch 14 owns v57–v59; confirm the next
  numbers from `research/VERSIONS.md` at start).
- The tactical suite is the CORRECTED suite (batch 9 added (9,7)
  to winning-quiet-setup-2's expected list). Re-verify v44's
  tactical on it first.
- **The ladder is 40 games per opponent per leg** (batch 11's
  re-basing). The ladder driver has auto-resume
  (`eval_version.py --out`, on master) — use it; every ladder
  JSON must be a committed artifact.
- The match drivers now seed per game (`seed + game_index`) with
  a uniqueness audit on every leg (ML batch 3 tooling, on
  master). Opening variety for the vs-v44 legs
  (`research/openings-ml4.json`, 10 openings) — it fixed the
  mirror-match collapse (v36-vs-v8 was 21 unique in 100 without
  openings; ML batch 4 legs ran 91–97 unique).
- **α sweeps:** 40-game sweeps have now failed twice (ML batch 5,
  v54) — winners regress at 100 games. Confirm sweep winners at
  100 games, or sweep at 100 games directly. Do not trust a
  40-game sweep ranking.
- **Data premise (ML batch 6 correction):** the distill game
  records and shards survive intact in the per-worker worktrees
  (v37's shards at
  `~/workspace/projects/gomoku-ml3-h2/ml/runs/ml3-h2/data/`,
  copies in `gomoku-ml1-h1`, `gomoku-ml4-h3`,
  `gomoku-ml5-h2`). Search the per-worker worktrees before
  assuming any data is gone.
- **Tooling (ML batch 6, on the batch-6 branch):** the fixed
  `eval_value_holdout.py` (reads side-to-move from plane 3 of
  the stored encoding — 37.9% of positions were mislabelled on
  distill shards). The fix is on the `research/ml-batch-6`
  branch, NOT on master (ML batch 6 was 0/3, so nothing merged).
  Cherry-pick or re-apply the fix (commit `038232c` on
  `research/ml-batch-6`) before trusting any by-side holdout
  numbers. **Do NOT trust by-side holdout numbers from the
  unfixed script** — iter-37/iter-44 by-side numbers are
  unreliable.

## Candidates, in priority order
1. **H1 (v60) — a White fix that survives odd-parity 3-ply
   labelling.** The v55 consequence is the sharpest open ML
   problem: the patch's White fix held at 2-ply but did not
   survive 3-ply labels (White 0.31 vs v0 vs v48's 0.52). The
   positions that matter are White under pressure vs *stronger*
   opponents — exactly what neither patch contained (v48's patch
   data came from v44's battery legs vs v0/v7/v37; v54's White
   fix only transferred to the weakest opponents). Pre-register
   the data mix (include White-under-pressure positions vs
   stronger opponents — v37/v44 legs, ladder MCTS legs) and the
   mechanism check (the played White gap must close WITHOUT
   Black-side strength falling; the v48/v54 lessons: a fix that
   redistributes or only transfers to weak opponents is not an
   improvement). Use 3-ply labels (v44's recipe) so the fix is
   tested in the labelling it must survive.
2. **H2 (v61) — v54's net at α=0.125.** The α ablation to
   separate net from α (like v50 for v37). No retraining; α
   verified as inference-time before any eval. Pre-registered
   decision rule: if v61 plays at v54's level (vs v44 ~0.385),
   the α is not the cause of v54's weakness; if it plays at
   v44's level or better, the α was the cause. (v50's precedent:
   v37's net at α=0.125 played at v37's level, not v44's —
   the mirror match vs v37 was 0.51 — proving v44's Elo gain
   came from its 3-ply labels, not the lower α.)
3. **H3 (v62) — centering correction for 3-ply labels.** Subtract
   the mean shift from the labels (the CENTERING form, not the
   symmetric form — H3 in ML batch 6 tried the symmetric form).
   Pre-register the mean prediction (labels centred to mean ≈ 0)
   and the terminal-preservation check (the symmetric form's
   terminal-asymmetry mechanism — wins halved, losses surviving
   — suggests centering may hit the same wall; check terminal
   preservation explicitly before training). If the terminal
   check fails (centering distorts terminals the way
   symmetrisation did), say so before training — do not train a
   version whose correction is already refuted.

Max 3 versions, each pre-registered in its iteration log BEFORE any
eval.

## Standing lessons — ML track
- **CLOSED levers** — do not reopen without a genuinely new
  mechanism: the learned policy prior cannot replace the v1
  prior (v26); "more data" for the v8 recipe (v25); dedup +
  colour-balanced retrain (v30); bigger value net / scale (v31
  — holdout transfer does not predict played strength; the
  value head is the binding constraint); VCF-at-root offensive
  on the value engine (v32 — near-neutral); threshold tie-break
  on the value engine (v36 — net q-gaps are noise); multi-task
  threat auxiliary in this form (v38); history dropout in this
  form (v42/v43 — the 2-ply labels NEED the history planes);
  label depth at both parities (v49 — 1-ply 0.515/Elo 52.5,
  2-ply 0.75/−34.9, 3-ply 0.67/98.1, 4-ply 0.475/−137.0; depth
  helps once, 1→2, then saturates; 2-ply remains the strongest
  labelling); α for the 2-ply net (v50 — mirror match vs v37
  is 0.51; v44's Elo gain came from its 3-ply labels, not the
  lower α); symmetric-label correction in this form (v56 —
  removed 70% of the gradient gap but did not centre the
  labels, sym mean −0.1292); White-side data patching in this
  form (v48 redistribution, v54 non-transfer — the fix works
  vs the weakest opponents only).
- **Holdout does not predict played strength — now EIGHT
  instances** (v31, v38, v42, v43, v48, v49, v56, v54); v54's
  is the sharpest (every diagnostic positive, played strength
  fell). Do not use holdout as a strength gate.
- **History-dependence tracks parity** (even-parity v37/v49
  collapse without history, drops 0.195/0.179; odd-parity v44
  doesn't, 0.007); the symmetric-label correction INCREASED
  history dependence (v56 no-history collapse v7 0.739→0.375,
  worse than v37's).
- **White collapse vs heuristic opponents is the top ML-track
  problem** (v44 White 0.16 vs v37; v42 White 0.28 vs v0 / 0.15
  vs v7; v43 White 0.21 vs v0). The collapse is opponent-
  dependent in played games (heuristic opponents only — v50
  White 0.175 vs heuristic, 0.44 in the mirror, 0.55 vs the
  ladder MCTS). The White collapse is a GENERAL White-side
  value-head defect (v48's diagnosis — the net undervalues
  White's wins and fails to see White is losing, on all four
  opponents, worst vs ML opponents), not heuristic-specific.
- **The ladder Elo and played strength now openly diverge**
  (ML batch 6): v54 has the best ML Elo (168.4) and the worst
  h2h vs v44 (0.385); both v54 and v56 pass Elo while losing
  the MCTS leg outright — the Elo is computed from the
  heuristic anchor leg only and is blind to the MCTS leg.
  Hence the clause-2 MCTS-leg rule above.
- **The ML label-depth naming counts plies BEFORE the teacher
  leaf**, so an "N-ply label" is an (N+2)-ply effective
  horizon — state depth in total rollout plies and check
  against `_ROLLOUT_PLIES` before the plan is written (batch
  13's H3 premise failure: the plan's "2-ply is untested" was
  FALSE — v18/v27's leaf is already 2-ply).
- **The "10k dataset" is 3,244 UNIQUE games, not 7,099** — and
  v8's "4,350" was 1,449. The B1 parallel generator seeds
  worker *w* with `seed + w`, so consecutive resume seeds
  replay each other's streams. Future legs must space resume
  seeds ≥ workers apart, and dedup-aware holdouts (by unique
  game) are now essential.
- **Training at this scale needs the memmap path** — the in-RAM
  load OOM-killed (3.2 GB vs host memory swings); the
  pre-allocated `load_shards` + memmap bundle + lazy-index
  `train()` (all tested, bit-identical to the RAM path) make
  it robust.
- **Thread-pin flag (ML batch 6, from H3):**
  `OMP_NUM_THREADS=1 TORCH_THREADS=1` env vars are overridden
  by `ValueMCTS.__init__`'s hard-coded
  `torch.set_num_threads(2)` — use the wrapper approach
  (`run_1thread.py`, iter-56 §5.0) for single-threaded runs.

## Method (carried over)
- The fast 10-game loop stays the finding method: play 10 games, read
  every move, find each game's losing move (solver/deeper-search
  verified, never guessed), patch, regression-test, repeat.
- The 4-clause bar stays the validation gate.
- All match drivers auto-resume from partial `--out` files (ML batch
  1 H1's patch, on master) and now seed per game (`seed +
  game_index`, ML batch 3 tooling, on master) with a uniqueness
  audit on every leg — use that durability and hygiene; batteries
  must survive VM replacements, and every leg reports its
  distinct-game count and de-dup rate.
- Fast loops run vs v0 AND v44.

## Merge discipline (batch-3 lesson)
- Run the full suite PLUS targeted per-version tests after every merge.
- Watch for the recurring traps: the silent auto-merge breakage on
  `ValueMCTS.__init__` (ML batch 1, ML batch 2 — duplicate
  parameters with NO conflict flag; check for duplicate parameters
  before running anything), the dead-code root-selection trap
  (batch 8, batch 9), and the shared-tail trap (batch 8, batch 9,
  ML batch 5).

## Constraints
- Branch `research/ml-batch-7` from master (current HEAD — verify
  with git rev-parse before branching). Commit this plan first.
- Use your own git worktree (e.g. `~/workspace/projects/gomoku-ml7/`).
- `ml/gomoku_ml/baseline.py` stays FROZEN (ladder anchor). Kernel
  re-sync (NumPy/WebGPU/Metal) only when a version passes the bar.
- No cloud GPU spend, no money, no wallets, no releases, no Colab
  driving, no Space edits. Yudong runs nothing.
- Push the branch when done. Push to master ONLY if a version passes
  all 4 clauses of the bar, with the same plain-merge approach as
  batch 3. A bar pass changes the ML champion and the daily
  cross-track match — say so explicitly in the report.
- The daily cross-track cron runs each morning ~08:37 PST; do not
  touch it or its output files. The daily pairing is v52 vs v44
  (swap rules), run in the main worktree (see TRACKS.md).
