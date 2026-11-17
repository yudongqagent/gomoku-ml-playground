# ML batch 2 plan — dedup + balance, bigger value net, VCF for the value engine

## Starting point
- The program now runs two tracks (see `research/TRACKS.md`). This is
  the ML-model track's batch after heuristic batch 9.
- **ML champion stays v8** (Elo 0.0, vs v0 0.545, tactical 20/20 +
  29/29). **ML bar unchanged** (same 4-clause shape, reference v8):
  (1) h2h vs v8 ≥100 games, win rate > 0.50 with Wilson 95% CI lower
  bound > 0.50; (2) ladder Elo > 0.0; (3) tactical ≥ v8's (20/20 +
  29/29); (4) all test suites green. A new ML champion is declared
  when a version passes all 4 clauses of the ML bar.
- **Versions v30–v32** (batch 9 owns v27–v29; confirm the next
  numbers from `research/VERSIONS.md` at start).
- **The tactical suite is now the CORRECTED suite** (batch 9 added
  (9,7) to winning-quiet-setup-2's expected list). Re-verify v8's
  tactical on the corrected suite before any eval — adding an
  expected move can only raise pass rates, so v8 should still be
  29/29, but verify, don't assume.
- Master now contains the v18/v23/v27/v28/v29 code (batch-9 merge),
  so the rollout-leaf distillation teacher is on master — no
  cherry-picking needed. The v8 checkpoint is gitignored data at
  `ml/runs/b3-h2/value-net.pt` (present in the main worktree; copy
  it into any worktree that must run v8).

## Candidates, in priority order
1. **H1 (v30) — dedup + colour-balanced retrain.** ML batch 1's
   v25 refuted "more data" for the v8 recipe, but the failure was
   specific: White collapsed (vs v8 Black 0.73 / White 0.15), and
   the training data is Black-skewed (Black wins 53.1%) and
   duplicated (the "10k dataset" is 3,244 unique games, not
   7,099). Retrain v8's recipe on the DEDUPED dataset (one copy
   per unique game) with COLOUR-BALANCED outcomes. Pre-register
   the balancing method (e.g. downsample Black-win games or
   upweight White-win games) and the holdout design (dedup-aware,
   by unique game) before any eval. Measure net accuracy by side
   to move before and after — v25's White-collapse hypothesis
   predicts the White-to-move accuracy gap closes.
2. **H2 (v31) — bigger value net on distilled labels.** The value
   head, not the prior, is the binding constraint (v26: the prior
   ranks the setup move 1st on all 3 tactical failures; the value
   head overrides it). v24 proved distillation works as ML
   (holdout sign accuracy 0.7766 vs v8's 0.504 on the same rollout
   labels) but plays only at v8's level — at v8's size (ResNet
   16ch/2 blocks). Scale it up: a larger value net (e.g. ResNet
   32ch/4 blocks, the round-1/2 architecture) trained on rollout-
   leaf labels from the DEDUPED dataset. Pre-register the
   architecture and the α sweep (distilled values are small,
   mean|v| ~0.06 — α=1.0 won v24's sweep); the 100-game battery
   is the verdict.
3. **H3 (v32) — VCF at root for the value-MCTS engine.** v24's
   finding was that the 100-sim search horizon is a shared limit
   (v7 makes the same miss as v24); batch 9's v29 proved a
   VCF-at-root check fixes tactical failures deterministically
   for the rollout-leaf engine (all three of v18's failures).
   Apply the search-side lever to the ML engine (v8 base):
   an offensive VCF check at the root, played immediately when
   found. Pre-register whether the defensive check is included
   — v29's defensive check is a VERIFIED FLAW (picks preventions
   by prior noise; the check's move lost 3/3 while the search's
   won 3/3), so the default is OFFENSIVE-ONLY unless the fast
   loop validates a VCF-filtered search (choose among preventions
   by search, not prior — the batch-10 direction from H3).

Max 3 versions, each pre-registered in its iteration log BEFORE any
eval.

## Standing lessons — ML track
- **Dataset dedup (ML batch 1):** the B1 parallel generator seeds
  worker *w* with `seed + w`, so consecutive resume seeds replay
  each other's streams. Future legs must space resume seeds ≥
  workers apart, and holdouts must be dedup-aware (by unique game).
  Count UNIQUE games, never recorded games.
- **Training at this scale needs the memmap path** — the in-RAM load
  OOM-killed (3.2 GB vs host memory swings); the pre-allocated
  `load_shards` + memmap bundle + lazy-index `train()` is
  bit-identical to the RAM path. Use it.
- **CLOSED levers** — do not reopen without a genuinely new
  mechanism: "more data" for the v8 recipe (v25: 2.24× unique
  games, weaker engine); the learned policy prior as a replacement
  for the v1 prior (v26: 0.20 vs v8 — the value head, not the
  prior, is the binding constraint).
- The value head is the binding constraint (v26). Distillation
  works as ML but a faithful copy of the rollout leaf plays only
  at v8's level at v8's size (v24) — the open question is whether
  scale (H2) or search (H3) breaks that ceiling.
- Small-sample smokes mislead (the 100-game battery is the verdict);
  v24's 0.65 smoke at α=1.0 did not reproduce at 100 games.
- Fast loops run against v0 as well as v8 (iter-17 §5 lesson,
  carried over).

## Method (carried over)
- The fast 10-game loop stays the finding method: play 10 games, read
  every move, find each game's losing move (solver/deeper-search
  verified, never guessed), patch, regression-test, repeat.
- The 4-clause ML bar stays the validation gate.
- All three match drivers now auto-resume from partial `--out`
  files (ML batch 1 H1's patch, on master via the batch-9 merge)
  — use that durability; batteries must survive VM replacements.

## Merge discipline (batch-3 lesson)
- Run the full suite PLUS targeted per-version tests after every merge.

## Constraints
- Branch `research/ml-batch-2` from master (`17d08db` plus the
  TRACKS.md doc commit — read HEAD at start). Commit this plan first.
- Use your own git worktree (e.g. `~/workspace/projects/gomoku-ml2/`).
- `ml/gomoku_ml/baseline.py` stays FROZEN (ladder anchor). Kernel
  re-sync (NumPy/WebGPU/Metal) only when a version passes the bar.
- No cloud GPU spend, no money, no wallets, no releases, no Colab
  driving, no Space edits. Yudong runs nothing.
- Push the branch when done. Push to master ONLY if a version passes
  all 4 clauses of the ML bar, with the same plain-merge approach as
  batch 3.
- The daily cross-track cron runs each morning ~08:37 PST; do not
  touch it or its output files. The daily pairing is unchanged
  (v18 vs v8, swap rules).
