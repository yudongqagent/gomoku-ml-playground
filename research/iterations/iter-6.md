# Iteration 6 — v6: MCTS with heuristic prior (batch 2, H2)

**Status: pre-registered — implementation in progress, no evals run
yet.** Comparison target and eval seeds recorded BEFORE the first
eval (see §1). This is the third and last version of batch 2
(max 3 versions: v4, v5, v6).

## 1. Design and pre-registration

**Comparison target: v1** (primary, same as v4/v5 for
comparability). The bar's clause 1 is head-to-head vs v1 (normal
rules, >= 40 games, alternating colours), clause 2 is ladder Elo
vs v1's 70.4 (identical 20 games/opponent, mcts_sims=100),
clause 3 is tactical pass rate vs v1's on the hard suite (12/29)
and the old suite (20/20). **Secondary head-to-heads, also
pre-registered:** v6 vs v5 (100 games) and v6 vs v0 (100 games);
plus a direct v6-vs-HeuristicMCTSPlayer match if the harness
allows it cheaply (same sims, better priors/leaf — the clean A/B
of this iteration's change). v3's numbers (Elo 107.5, hard suite
26/29) are reported alongside for transparency.

**The improvement: v6 = PUCT MCTS with v1 scores as priors and
the terminal-aware leaf value.** A new module
(`ml/gomoku_ml/mcts_heuristic.py`, standalone — it does not touch
`mcts.py`, which the B2 batched-leaf workstream owns) implements
AlphaZero-style PUCT with:

- Priors from v1's 1-ply scores (5-window, broken-four-aware):
  softmax over log1p(scores) with temperature 2, over ALL
  candidate moves (no top-K cutoff — iter-5 §2.2 showed the
  saving moves rank 11th-12th by 1-ply score).
- Leaf values from the terminal-aware evaluation
  (`scoring._leaf_value`), squashed to [-1, 1] as
  x / (|x| + 200_000) (an open four maps to ±0.98, a single four
  to ±0.09, a quiet position to ~0). Terminal win/loss/draw are
  exact (±1/0), as in the baseline MCTS.
- Win/block scans before searching (as in every version).
- Move choice: most-visited root child, ties by the seeded RNG.
- Sims: 100, c_puct: 1.5 — the SAME settings as the ladder's
  `HeuristicMCTSPlayer`, so v6's edge (if any) comes purely from
  the better priors (v1 vs v0 scoring) and the better leaf value
  (terminal-aware vs crude local attack). Registered as version 6
  in `scoring.ENGINE_VERSIONS` with an `"mcts"` flag;
  `ScoringPlayer.choose_move` dispatches to the MCTS when set, so
  the existing harness (`selfplay_batch.py`, `eval_version.py`)
  resolves `v6` unchanged.

**Why this change — traced to the analysed gaps (iter-5 §2.2-2.3).**
The 2-ply minimax lineage failed on two counts that MCTS
addresses directly: (1) candidate coverage — the saving move in
v5-vs-v0 game 1 ranked 11th-12th by 1-ply score, outside the
top-8; PUCT explores all candidates, guided (not restricted) by
the priors. (2) Threat breadth — the static 2-ply leaf values
the BEST attack, not how many threats exist, so it ranked the
losing (4,9) above the winning (7,11); MCTS evaluates by searched
win rates across many lines, so a line where Black wins across
many continuations scores badly however good the single best
attack looks.

**Eval seeds (pre-registered):** h2h vs v1: v6=100 / v1=200;
h2h vs v5: v6=100 / v5=200; h2h vs v0: v6=100 / v0=200; ladder
seed 100, 20 games/opponent, mcts_sims=100; swap 100 games vs v1,
balanced placement, placement_seed 11, tempo judge (the harness
default since H3; recorded in the JSON).

## 2. Loss analysis for iteration 6

(TODO: filled after the head-to-heads — same method as iter-4/5:
open-four race, missed blocks, missed preventions, colour split,
win/loss plies, plus at least one worked game with a solver
oracle if v6 still loses games the same way.)

## 3. v6 evaluation

(TODO: head-to-heads vs v1 / v5 / v0, ladder, swap match, tactical
on both suites.)

## 4. Bar check (v6 vs v1)

(TODO)

## 5. What this means

(TODO)
