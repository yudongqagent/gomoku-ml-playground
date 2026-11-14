# Batch 4 plan — build on v7

## Starting point
- v7 (H1, MCTS + capped attack-gradient leaf) is the first version to pass all
  4 clauses and is now on master (`bf866f7`, merge `49a0475`).
- **v7 is the new reference version.** The bar is unchanged; the reference
  moves from v1 to v7:
  1. Head-to-head vs v7, ≥100 games, alternating colours, win rate (draws =
     half) > 0.50 with Wilson 95% CI lower bound > 0.50.
  2. Ladder Elo strictly higher than v7's 190.8 at identical game
     counts/opponents (20 games/opponent: random, heuristic, mcts).
  3. Tactical pass rate ≥ v7's (20/20 old suite + 29/29 hard suite).
  4. All test suites green (ml pytest, npm test, tsc).
- Every version also plays 100 games vs v0 and a 100-game swap match vs v7
  (tempo judge); record non-transitivity prominently.
- `ml/gomoku_ml/baseline.py` stays FROZEN (ladder anchor). Kernel re-sync
  (NumPy/WebGPU/Metal) only when a version passes the bar.

## Candidates, in priority order
1. **H1 — q-vs-visits root decision.** Worked game (v7 vs v0 game 16, ply 14):
   the saving move had higher q but tied on visits and lost the RNG
   coin-flip. Decide the root move by q, or break visit-ties by q.
   Pre-register the exact variant(s) before any eval.
2. **H2 — White weakness / colour-conditional value training.** v7 as White
   vs v1 is only 0.52 (18 of 21 losses as White); v8 as White was 0.29.
   Colour-conditional value function, or colour-aware leaf/priors.
3. **H3 — value net trained on heuristic-opponent games, not weak self-play.**
   v8's lesson: more weak-self-play data is not the fix (the 4,350-game net
   generalised, holdout 0.7165, but played worse at full weight —
   distribution mismatch + optimizer's curse; VALUE_WEIGHT 0.25 recovered
   playability). Train on games against the heuristic / stronger opponents.
4. **Complete the 10k dataset** (4,350/10,000 done, 5,650 remain, ~5.3–7.8h
   at the measured B1 rate) if a stage needs it. Data at
   `~/workspace/projects/gomoku-b3-b1/ml/runs/b3-b1-10k/data/` (gitignored).

Max 3 versions, each pre-registered in its iteration log BEFORE any eval.

## Method (carried over from batch 3)
- The fast 10-game loop stays the finding method: play 10 games, read every
  move, find each game's losing move (solver/deeper-search verified, never
  guessed), patch the algorithm to reproduce the perfect move, record the
  critical position + correct move as a regression test, repeat. Think like
  a human master about weak points and weak strategies.
- The 4-clause bar stays the validation gate — a patch that fixes 10 games
  but loses Elo is not progress.
- Open questions from batch 3 to keep in mind: v7's colour-split inversion
  vs v0 (0.6650 vs v0 < 0.7050 vs v1; 13 Black losses to v0 vs 3 to v1);
  v7's advantage lives largely in the empty-board opening (swap edge not
  established, 0.57, CI includes 0.50).

## Merge discipline (batch-3 lesson)
- H1's leaf refactor silently broke H2's ValueMCTS (instance attribute
  shadowed the method) — no git conflict flagged it; targeted tests caught
  it (12 failures). When merging parallel workstreams, run the full suite
  PLUS targeted per-version tests after every merge.

## Constraints
- Branch `research/batch-4` from master (`bf866f7`). Commit this plan first.
- No cloud GPU spend, no money, no wallets, no releases, no Colab driving,
  no Space edits. Yudong runs nothing.
- The v8 checkpoint (`ml/runs/b3-h2/value-net.pt`) is gitignored data, not
  in git — a fresh clone cannot run v8. If H2/H3 needs it, check the main
  worktree copy still exists or retrain.
- Push the branch when done. Push to master ONLY if a version passes all
  4 clauses, with the same plain-merge approach as batch 3.
