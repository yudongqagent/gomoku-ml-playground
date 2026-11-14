# Iteration 4 — v4: 2-ply minimax over v1's top candidates (batch 2, H1)

**Status: complete — v4 REJECTED** (clause 2 fails: ladder Elo
-70.4 vs v1's 70.4; confirmed by the 100-game v0 match, 0.37).
Comparison target recorded BEFORE the first eval (see §1).

## 1. Design and pre-registration

**Comparison target: v1.** v4 builds on v1 (2-ply minimax over v1's
top candidates), so the bar's clause 1 is head-to-head vs v1
(normal rules, >= 40 games, alternating colours), clause 2 is
ladder Elo vs v1's 70.4 (identical 20 games/opponent, mcts_sims=100),
and clause 3 is tactical pass rate vs v1's score on the NEW harder
suite (batch 2, Step 0). v3's ladder Elo (107.5) is higher than
v1's, but v3 lost head-to-head to v1 (0.47) and is a rejected
branch, not the version v4 builds on — v1 is the comparison target,
and v3's numbers are reported alongside for transparency.

**The improvement: v4 = v1 + 2-ply minimax** (`ENGINE_VERSIONS[4]`,
`minimax_topk=8`). After the win/block scans, for each of the top-8
1-ply candidates: play the opponent's v1 best reply (opponent
modelled as v1 — win/block scans + best 1-ply move, no recursion),
then evaluate the resulting position by best-attack difference
(attack-only scores). The total score is colour-symmetric at
defence_weight=1.0, so a total-score difference would always be 0 —
attack-only is what makes the static evaluation meaningful (a side
with a winning move next scores 10M; a side facing an open four
scores 200k). Ties on minimax value break toward the better 1-ply
score. Unlike v3's safety filter (which only avoids losing
creators), minimax also FINDS attacking sequences: a move creating
a double threat while the opponent's reply is forced scores highly.

**Why top-8:** on the iter-3 game-13 regression position the
preventing move (10, 7) ranks 8th by 1-ply score (tied at 20,097);
a smaller top-K would miss it. Cost measured on a random 24-stone
board: v4 `choose_move` ~81 ms mean vs v1 ~3.7 ms (~22x slower).

**Harness fix (batch 2):** `ml/scripts/eval_version.py` measured
tactical with `_choose_move(..., defence_weight)` only — it dropped
the search flags, so v3's 20/20 was measured AS v1. Fixed: the
tactical probe now uses the version's full move choice (threat
check / minimax), with rng=None preserved for determinism. For
v1/v2 this changes nothing; v3's tactical score is re-measured
with the fixed driver for reference (see §4).

## 2. Loss analysis for iteration 4

### 2.1 Aggregate: v4's losses are still all open-four losses

| Match (100 games, normal rules) | v4 result | Losses via open four | Missed immediate blocks | Threatened turns | Clean missed preventions |
|---|---|---|---|---|---|
| v4 vs v1 (seeds 100/200) | 79W-21L-0D | 21/21 | 0 | 54 | 14, in 12 games |
| v4 vs v0 (seeds 100/200) | 37W-63L-0D | 63/63 | 0 | 156 | 63, in 63 games |

Definitions (same as batch 1): a win is "via open four" if, on the
board just after the winner's previous move, the winner held >= 2
winning moves. A "threatened turn" is a v4 turn with the opponent
holding an open-four creator and no immediate win for either side.
A "clean missed prevention" is a threatened turn where v4 played a
non-forcing move leaving >= 1 creator live while a preventing
alternative existed.

Correction recorded: the first version of the classifier checked
the board just BEFORE the winning move (after the defender's
block), which gave 0/21 — an off-by-one. The corrected classifier
checks the board after the winner's previous move (before the
defender's reply) and gives 21/21 and 63/63. The numbers above are
from the corrected classifier (`/tmp/analyze_v4_losses.py`).

Colour splits: vs v1, v4 as Black 49W-1L, as White 30W-20L (20 of
21 losses as White). Vs v0, v4 as Black 36W-14L, as White 1W-49L
(49 of 63 losses as White; v4 won ONE game in 50 as White against
the frozen heuristic). Plies: vs v1, v4 wins in 28.4 plies mean,
loses in 44.5; vs v0, v4 wins in 42.9, loses in 20.6 — v4 loses
fast as White against v0. The fastest losses are 15 plies
(games 1, 3, 5 vs v0, all as White; games 1 and 5 are move-identical
— deterministic engines, same seeds).

### 2.2 The top-K hypothesis is refuted — the value function is the cause

For every clean missed prevention, was the preventing move in
v4's top-8 candidates by 1-ply score, or excluded by the top-K
cutoff? (`/tmp/diag_prevention_rank.py`)

- vs v1: 14/14 missed preventions had the preventing move IN the
  top-8 — ranked **#1 by 1-ply score in all 14 cases**.
- vs v0: 63/63 missed preventions had the preventing move IN the
  top-8 — ranked **#1 by 1-ply score in all 63 cases**.
- 0 missed preventions in either match were top-K exclusions.

The preventing move is always ranked #1 by 1-ply score (blocking
a creator scores ~200,000 defence). v4's minimax then OVERRIDES
the #1 1-ply move and plays something else. The flaw is in the
minimax leaf value, not the candidate set and not the search depth.
§2.3 works one game in full to show exactly how.

### 2.3 Worked case: v4-vs-v0 game 1 — the losing move, verified

Game 1 (v4 as White, v0 as Black, Black wins in 15 plies). Moves:

1. B (7,7) 2. W (5,5) 3. B (7,6) 4. W (7,5) 5. B (6,5)
6. W (5,4) 7. B (5,6) 8. W (8,6) 9. B (6,4) 10. W (6,6)
11. B (4,7) 12. W (8,4) 13. B (3,8) 14. W (7,4) 15. B (2,9) wins

**The position before White's move 12.** Black stones: (4,7),
(5,6), (6,4), (6,5), (7,6), (7,7). White stones: (5,4), (5,5),
(6,6), (7,5), (8,6). Black holds an open three — (4,7), (5,6),
(6,5) on the anti-diagonal — with TWO creators: (3,8) and (7,4).
Either creator makes an open four (two winning squares).

**The losing move: White (8,4)** (v4's choice, reproduced by
`_choose_move` with minimax on this exact board). It builds
White's own open three — (8,4), (7,5), (6,6) — and ignores Black's.
Black replies (3,8): open four (3,8), (4,7), (5,6), (6,5) with
winning squares (2,9) and (7,4). White blocks (7,4); Black plays
(2,9) and wins on move 15.

**The best moves instead: (3,8) or (7,4)** — block one end of
Black's open three. They are ranked #1 and #2 by 1-ply score
(200,012 and 200,027). After White (3,8), Black's best reply
creates a single four and White blocks its one winning square
(verified below); the game continues.

**Why v4 chose (8,4): the minimax values on this board**
(recomputed with the same code path as `_minimax_choose`):

| Candidate | 1-ply score | Minimax value | Modelled v1 reply | Black winning moves after reply |
|---|---|---|---|---|
| (7,4) — block | 200,027 (#1) | -9,995,914 | (3,8) | 1 |
| (3,8) — block | 200,012 (#2) | -9,995,929 | (8,3) | 1 |
| (8,3) | 20,090 | -9,942,009 | (3,8) | 2 |
| (2,9) | 20,012 | -9,995,922 | (7,4) | 2 |
| (5,7) | 8,016 | -180,397 | (8,4) | 0 |
| (8,7) | 4,487 | -9,942,003 | (7,4) | 2 |
| (6,7) | 4,173 | -9,995,923 | (7,4) | 2 |
| (8,4) — played | 4,098 | **-180,155** | (5,7) | 0 |

The blocks — the #1 and #2 1-ply moves — are valued at ≈ -10M,
the worst in the top-8; the losing attack (8,4) is valued best
at -180,155. Two compounding flaws in the leaf value
(`ours - theirs` on the board after the modelled reply):

1. **A single blockable winning move counts as -10M.** After White
   blocks (7,4), Black's modelled reply (3,8) creates a four with
   exactly ONE winning square (2,9). White to move blocks it and
   the game continues — but the leaf eval scores
   `theirs = 10,000,006` (a winning move is worth WIN_SCORE) and
   the position reads as a terminal loss. The value function does
   not distinguish "opponent has one winning move" (we block it)
   from "opponent has two winning moves" (open four — we lose).
2. **Our attack score compensates for the opponent's win.**
   `ours - theirs` lets White's open three (ours = 20,006) offset
   Black's threats. After (8,4), the modelled v1 reply is (5,7)
   — v1 blocks White's open three instead of converting Black's
   creator (a 1-ply tie-break) — so the leaf shows 0 Black winning
   moves and the line looks safe at -180,155. But the REAL
   opponent v0 converts the creator ((3,8)) instead of blocking,
   because v0's attack (10,000) beats its defence (90% = 9,000).
   The opponent model (v1) and the real opponent (v0) diverge
   exactly here.

**Verification with the VCF solver** (`src/ai/search.ts`,
`findForcedWin`, run via vite-node on this position):

- After White (8,4): Black has a forced VCF win — first move
  (7,4) (an immediate open-four creation; (3,8) wins the same
  way). The loss is PROVEN, not just suspected.
- After White (3,8): no forced VCF win for Black within 9 or 15
  plies. After White (7,4): likewise none within 9 or 15 plies.

**Verification by playout** (fixed engines from the position
before move 12, v0 as Black):

| White move 12 | v4 as White | v1 as White |
|---|---|---|
| (8,4) — played | Black wins after 3 plies | Black wins after 3 plies |
| (3,8) — block | Black wins after 17 plies | Black wins after 29 plies |
| (7,4) — block | Black wins after 29 plies | Black wins after 37 plies |

Honest caveat: the block is the best move by every oracle
available — it avoids the proven forced loss and extends the game
6–12x — but the playout engines still eventually lose, so the
position after the block is NOT proven drawn or won for White.
What is proven is that (8,4) loses by force and the blocks do not
lose by force within the solver's horizon. The same flaw pattern
(a missed rank-#1 prevention, decided by an open four) accounts
for all 77 missed preventions across both matches (§2.2).

### 2.4 What changes in the algorithm because of this (v5)

The leaf evaluation must be **terminal-aware**: after the modelled
reply, count each side's winning moves on the leaf board.

- Opponent with >= 2 winning moves: terminal loss, -WIN_SCORE,
  with NO compensation from our attack score (an open four cannot
  be out-attacked).
- Exactly 1 winning move for a side: capped at four-value (20,000),
  not WIN_SCORE — the player to move blocks it, so it is a forcing
  threat, not a win.
- Symmetric for our winning moves (>= 2 is a terminal win).

On the game-1 board this repairs the inversion: the block (7,4)
values at ≈ -20,000 (Black's single four, capped) instead of
≈ -10M, beating the attack (8,4) at -180,155 (Black's creator
still live). v5 = v4 + this terminal-aware leaf evaluation; the
design traces directly to this analysed gap, not a blind tweak.
See iter-5.md.

## 3. v4 evaluation

### 3.1 Head-to-head vs v1 (clause 1) — PASS

100 games, normal rules, seeds v4=100 / v1=200, alternating
colours (`research/iterations/batch-v4-vs-v1.json`, 89.9 s):

**v4: 79W-21L-0D, win rate 0.7900, 95% CI [0.7002, 0.8583].**
As Black 49W-1L; as White 30W-20L. Plies mean 31.8 (min 15,
max 98); v4 wins in 28.4 plies mean, loses in 44.5. A 20-game
smoke match first gave 17W-3L-0D (0.85, CI [0.6396, 0.9476];
as Black 10W-0L, as White 7W-3L).

### 3.2 Ladder (clause 2) — FAIL

`eval_version.py --version 4 --ladder-games 20 --mcts-sims 100
--seed 100` (identical counts/opponents to v1's measurement):

| Opponent | v4 | v1 (same counts) |
|---|---|---|
| random | 20W-0L-0D (1.0000, CI [0.8389, 1.0000]) | 20W-0L-0D |
| heuristic (v0) | 8W-12L-0D (0.4000, CI [0.2188, 0.6134]); as Black 8W-2L, as White 0W-10L | 11W-7L-2D (0.600) |
| mcts (100 sims) | 4W-16L-0D (0.2000, CI [0.0807, 0.4160]); as Black 4W-6L, as White 0W-10L | 9W-10L-1D (0.475) |
| **Ladder Elo** | **-70.4, interval [-221.1, 80.2]** | **70.4, interval [-80.2, 221.1]** |

v4's ladder Elo (-70.4) is BELOW v1's (70.4). The intervals
overlap, so the 20-game Elo difference alone is not statistically
significant — but the 100-game match below confirms it is real.

### 3.3 Head-to-head vs v0 (secondary; confirms the ladder)

100 games, normal rules, seeds v4=100 / v0=200
(`research/iterations/batch-v4-vs-v0.json`, 118.8 s):

**v4: 37W-63L-0D, win rate 0.3700, 95% CI [0.2818, 0.4678].**
As Black 36W-14L; as White 1W-49L. Plies mean 28.9; v4 wins in
42.9 plies mean, loses in 20.6. For comparison, v1 vs v0 pooled
200 games was 0.55 and v3 vs v0 was 0.6150 — v4 is decisively
worse than both against the frozen heuristic, despite beating v1
0.79 head-to-head. Non-transitivity: v4 > v1 > v0 > v4.

### 3.4 Swap-rule match vs v1 (secondary)

100 games, standard swap, balanced placement, placement_seed 11
(`research/iterations/batch-v4-vs-v1-swap.json`, 198.7 s):

**v4: 74W-23L-3D, win rate 0.7550, 95% CI [0.6623, 0.8288].**
As Black 40W-10L; as White 34W-13L-3D. Plies mean 39.3 (min 16,
max 225). Second player took Black in 100/100 games (the OLD
colour judge — H3's tempo-aware judge was not yet in place);
winners by colour: Black 53, White 44, draws 3.

### 3.5 Tactical

Old 20-position suite, fixed driver (full move choice):
**v4 20/20**; v1 20/20; v3 re-measured with the fixed driver
20/20 (unchanged — the threat check does not affect the old
suite). The old suite still does not discriminate. NEW harder
suite (29 positions, batch 2 Step 0 — see
`research/iterations/tactical-suite-v2.md`): **v4 25/29**
(prevention 13/16, setup 6/7, winning 2/2, vcf 4/4; fails
`prev-v1bl-g2`, `prev-v2v1-g36`, `prev-v2v1-g60`,
`setup-v1v0s-g92`), vs v1's 12/29 (2/16 prevention) and v3's
26/29 (16/16 prevention). v4 passes all 4 VCF positions (the
minimax finds forced wins) but misses 3 preventions.

### 3.6 Cost

v4 `choose_move` ~81 ms mean on a random 24-stone board vs v1
~3.7 ms (~22x slower). Match pace: 100 games vs v1 in 89.9 s
(~0.9 s/game); vs v0 in 118.8 s. Affordable for this research;
the flaw is in the value function, not the speed.

## 4. Bar check (v4 vs v1)

| Clause | Requirement | v4 | Verdict |
|---|---|---|---|
| 1. Head-to-head vs v1 | >= 40 games, win rate > 0.50, Wilson 95% CI lower bound > 0.50 | 100 games, 0.7900, CI [0.7002, 0.8583] | **PASS** |
| 2. Ladder Elo | strictly higher than v1's 70.4, identical counts/opponents | -70.4 (interval [-221.1, 80.2]) | **FAIL** |
| 3. Tactical | pass rate >= v1's | old suite 20/20 = v1's 20/20; hard suite 25/29 vs v1's 12/29 | **PASS** |
| 4. Tests green | ml pytest, npm test | targeted suites green at implementation (69 tests: test_minimax, test_scoring, test_research, test_swap_rule, test_game); full suites run at batch end | not finally assessed |

**Verdict: v4 is REJECTED.** Clause 2 fails: ladder Elo -70.4 is
not strictly higher than v1's 70.4, and the independent 100-game
match vs v0 (0.37, CI entirely below 0.50) confirms v4 is
genuinely worse than v1 against the ladder anchor. v4 is NOT
pushed to master. The head-to-head pass (0.79) makes this the
sharpest possible demonstration of why the bar has four clauses:
a version can dominate its parent head-to-head and still be worse
overall.

## 5. What this means

- **H1's 2-ply minimax works as a search idea but its leaf value
  is broken.** It wins the open-four race against v1 (79-21)
  because the opponent model (v1) is accurate there, but the
  `ours - theirs` leaf value cannot tell a blockable four from an
  open four, and lets our attack compensate for the opponent's
  win. Against v0 — whose replies the v1 model mispredicts — the
  search actively misleads: v4 overrides its own #1-ranked
  blocking move in all 77 analysed missed preventions.
- **The fix is precise and traced to the analysed gap** (§2.4):
  v5 keeps v4's search and replaces the leaf value with a
  terminal-aware evaluation (>= 2 winning moves = terminal;
  exactly 1 = capped at four-value). If the diagnosis is right,
  v5 should recover the v0 matchup while keeping the v1 edge.
  iter-5.md records the pre-registration before v5's eval.
- **Method note:** the loss analysis caught this only because it
  went per-game with a solver oracle (Yudong's direction, batch 2
  update). The aggregate stats (0.79 h2h) looked like a clear win;
  the 100-game v0 match + the game-1 workup showed the opposite.
  Every future version gets the same treatment: a concrete losing
  position, the best move with solver verification, and a design
  change that traces to it.
- **Standing:** master untouched (still 78c733a). v4 lives only
  on branch `research/batch-2`.
