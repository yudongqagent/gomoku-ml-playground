# Iteration 5 — v5: v4 + terminal-aware minimax leaf evaluation (batch 2, H1 fix)

**Status: pre-registered — implementation in progress, no evals run
yet.** Comparison target and eval seeds recorded BEFORE the first
eval (see §1).

## 1. Design and pre-registration

**Comparison target: v1** (primary). v5 builds on v4, which builds
on v1; v1 is the version v4 was compared against, so keeping the
same target preserves comparability. The bar's clause 1 is
head-to-head vs v1 (normal rules, >= 40 games, alternating
colours), clause 2 is ladder Elo vs v1's 70.4 (identical 20
games/opponent, mcts_sims=100), clause 3 is tactical pass rate vs
v1's (old suite 20/20; hard suite when it lands). **Secondary
head-to-heads, also pre-registered:** v5 vs v4 (100 games — the
leaf fix should beat v4 directly) and v5 vs v0 (100 games — the
matchup v4 failed, 0.37). If v5 beats both v1 and v4, it passes
clause 1 under either reading of "previous version". v3's numbers
(Elo 107.5, h2h vs v1 0.47) are reported alongside for
transparency.

**The improvement: v5 = v4 + terminal-aware leaf evaluation**
(`ENGINE_VERSIONS[5]`, `minimax_topk=8`, `terminal_aware=True`).
The search is unchanged (2-ply minimax over the top-8 1-ply
candidates, opponent modelled as v1). Only the leaf value
changes. After the modelled reply, count each side's winning
moves on the leaf board:

- Opponent with >= 2 winning moves: terminal loss, -WIN_SCORE,
  with NO compensation from our attack score (an open four cannot
  be out-attacked).
- Exactly 1 winning move for a side: capped at four-value
  (20,000), not WIN_SCORE — the player to move blocks it, so it
  is a forcing threat, not a win.
- Symmetric for our winning moves (>= 2 is a terminal win).
- Otherwise: best-attack difference over non-winning moves, plus
  the four-value cap for a single winning move each side.

**Why this change — traced to the analysed gap (iter-4 §2.3-2.4).**
In v4-vs-v0 game 1, v4 played (8,4) instead of the block (7,4) /
(3,8) because the leaf value scored the block at ≈ -10M: after
the block, Black's modelled reply creates a four with exactly one
winning square, and `theirs = 10,000,006` reads as a terminal
loss — but White to move blocks that square and the game
continues. The VCF solver proved (8,4) loses by force while the
blocks show no forced win within 15 plies. The same inversion
accounts for all 77 analysed missed preventions (14 vs v1, 63 vs
v0): in every case the preventing move was ranked #1 by 1-ply
score and v4's minimax overrode it. The terminal-aware leaf
repairs the inversion on the game-1 board: the block values at
≈ -20,000 (Black's single four, capped) instead of ≈ -10M, beating
the attack (8,4) at -180,155 (Black's creator still live). v5
must choose (3,8) or (7,4) on the game-1 regression position —
recorded here as a required check before the evals.

**Eval seeds (pre-registered):** h2h vs v1: v5=100 / v1=200;
h2h vs v4: v5=100 / v4=200; h2h vs v0: v5=100 / v0=200; ladder
seed 100, 20 games/opponent, mcts_sims=100; swap 100 games vs v1,
balanced placement, placement_seed 11, judge = the harness
default at run time (recorded explicitly in §3; v4's swap match
used the old attack judge, so a judge change is a comparability
caveat, not a bar issue — the swap match is secondary).

## 2. Loss analysis for iteration 5

(TODO: filled after the head-to-heads — same method as iter-4:
open-four race, missed blocks, missed preventions, colour split,
win/loss plies, plus at least one worked game with a solver
oracle if v5 still loses games the same way.)

## 3. v5 evaluation

(TODO: head-to-heads vs v1 / v4 / v0, ladder, swap match, tactical
on both suites.)

## 4. Bar check (v5 vs v1)

(TODO)

## 5. What this means

(TODO)
