# Iteration 5 — v5: v4 + terminal-aware minimax leaf evaluation (batch 2, H1 fix)

**Status: complete — v5 REJECTED** (clause 2 fails: ladder Elo
-107.5 vs v1's 70.4; v0 match 0.28, worse than v4's 0.37).
Comparison target and eval seeds recorded BEFORE the first eval
(see §1).

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

### 2.1 Aggregate: the leaf fix works mechanically — but v5 got worse vs v0

| Match (100 games, normal rules) | v5 result | Losses via open four | Missed blocks | Threatened turns | Clean missed preventions (rate) |
|---|---|---|---|---|---|
| v5 vs v1 | 54W-24L-22D | 24/24 | 0 | 95 | 8 in 8 games (8.4%) |
| v5 vs v4 | 49W-51L-0D | 51/51 | 0 | 164 | 28 in 28 games (17.1%) |
| v5 vs v0 | 24W-68L-8D | 68/68 | 0 | 508 | 16 in 16 games (3.1%) |

For comparison, v4's missed-prevention rates were 25.9% vs v1
(14/54) and 40.4% vs v0 (63/156). The terminal-aware leaf did
exactly what it was designed to do: v5 now blocks the #1-ranked
prevention instead of overriding it. Yet v5 vs v0 fell from 0.37
(v4) to 0.28, and v5 vs v4 is a dead heat (0.49). Defence alone
does not win Gomoku: v5 blocks correctly but never builds winning
threats of its own, and v0's steady attacking play grinds it down
in long games (v5-vs-v0 plies mean 61.3 vs v4's 28.9; v5-vs-v1
mean 88.3 with 22 full-board draws at 225 plies).

Colour splits: vs v1, v5 as Black 39W-5L-6D (0.84), as White
15W-19L-16D (0.46) — the White weakness persists. Vs v0, v5 as
Black 15W-28L-7D (0.37), as White 9W-40L-1D (0.19): v5 now loses
as BOTH colours to the frozen heuristic (v4 won 36-14 as Black).
Vs v4, the result is entirely colour-driven: v5 as Black 41W-9L,
as White 8W-42L — Black wins ~82% of v5-vs-v4 games regardless of
engine, so 49-51 is what two equal engines produce under
alternating colours. v5 ≈ v4 in strength.

### 2.2 Worked case: v5-vs-v0 game 1 — the losing move is a quiet one

Game 1 (v5 as White, v0 as Black, Black wins in 57 plies). The
decisive point is NOT a missed block or a missed prevention — v5
blocks correctly throughout. It is a slow strategic loss:

- **First unstoppable turn: White's move 42.** Black holds 2
  creators [(7,4), (11,8)]; White has no forcing reply (no win,
  no four). White blocks (7,4); Black's other creator decides
  the game 15 plies later.
- **Last preventable turn: White's move 30.** Before it, Black
  has exactly ONE creator, (4,9), and no winning move. v5 played
  **(4,9)** — ranked #1 by 1-ply score (200,417) — blocking
  Black's only creator. But after Black's reply (3,9), Black has
  1 winning move (2,10) PLUS 6 moves that each create an
  additional four: White must block (2,10), then faces a stream
  of single-four threats it can only answer one per move. Black
  wins 26 plies later in the playout (27 in the actual game).
- **The best moves instead: (7,11) or (6,12).** After either,
  Black's reply leaves Black with 1 creator and 0 winning moves
  — and the playouts (v0 as Black vs v5 as White, terminal
  checked at start) end with **White winning after 71 plies**
  ((7,11)) and **59 plies** ((6,12)).

**Why v5 missed them — two compounding reasons, both verified:**

1. **Top-K exclusion (this time for real).** (7,11) ranks 11th
   and (6,12) 12th by 1-ply score (20,026 / 20,019) — OUTSIDE
   the top-8. v5's minimax never considers them. (Contrast
   iter-4: v4's missed preventions were always rank #1. The
   saving moves here are quiet positional moves, not blocks.)
2. **The leaf value would rank them LAST even if considered.**
   (7,11)'s terminal-aware leaf value is -199,443 and (6,12)'s
   -199,442, vs (4,9)'s -19,989. The leaf sees Black's single
   remaining creator after (7,11) as a 200k open-four threat,
   while (4,9)'s leaf shows only Black's single winning move
   capped at 20k. The static leaf evaluates the BEST attack, not
   the BREADTH of threats: it cannot see that six single-four
   threats (a forced blocking sequence White eventually loses)
   are worse than one open-four threat (which White answers next
   move). Note the scorer's part in this: the six creators after
   (4,9) score only ~20,000 each as attacks (broken/double-threat
   shapes the 5-window scan values as ordinary fours), so the
   leaf is blind to their multiplicity.

Methods note: the first walk-back for this game had a bug — it
did not check whether Black's modelled reply wins immediately,
and wrongly credited (11,12) at move 36 (Black's reply (10,11)
completes five on the diagonal (6,7)-(10,11)). The corrected
walk-back (with the terminal check) finds move 30 instead, and
the playouts above use a terminal check at start. The (11,12)
claim is withdrawn.

### 2.3 What changes in the algorithm because of this (v6)

The 2-ply minimax lineage (v4/v5) has hit its ceiling: a static
leaf cannot value threat breadth, and a top-8 candidate set
ranked by 1-ply score excludes quiet saving moves. v6 is H2 —
**MCTS with the heuristic as prior**: PUCT search over ALL
candidate moves (no top-K cutoff), with v1 scores as priors and
heuristic static values at the leaves, so the engine evaluates
positions by searched win rates across many lines instead of a
single 2-ply leaf. That directly addresses both gaps found here:
candidate coverage (PUCT explores moves the 1-ply ranking buries)
and threat breadth (a line where Black wins across many
continuations scores badly however good the single best attack
looks). See iter-6.md.

## 3. v5 evaluation

### 3.1 Head-to-head vs v1 (clause 1) — PASS (vs the pre-registered target)

100 games, normal rules, seeds v5=100 / v1=200
(`research/iterations/batch-v5-vs-v1.json`, 481.4 s):

**v5: 54W-24L-22D, win rate 0.6500, 95% CI [0.5525, 0.7364].**
As Black 39W-5L-6D; as White 15W-19L-16D. Plies mean 88.3
(min 19, max 225); all 22 draws are full-board 225-ply games.
A 20-game smoke match first gave 13W-4L-3D (0.7250, CI [0.5059,
0.8716]).

### 3.2 Head-to-head vs v4 (secondary) — TIE

100 games, normal rules, seeds v5=100 / v4=200
(`research/iterations/batch-v5-vs-v4.json`, 324.8 s):

**v5: 49W-51L-0D, win rate 0.4900, 95% CI [0.3942, 0.5865].**
As Black 41W-9L; as White 8W-42L. The result is entirely
colour-driven (Black wins ~82% of v5-vs-v4 games regardless of
engine), so 49-51 is what two equal engines produce. v5 does
NOT beat its immediate predecessor — flagged here because the
pre-registration said beating both v1 and v4 would settle clause
1 under either reading of "previous version". Clause 1 passes
against the pre-registered target (v1) but not against v4.

### 3.3 Head-to-head vs v0 (secondary) — WORSE than v4

100 games, normal rules, seeds v5=100 / v0=200
(`research/iterations/batch-v5-vs-v0.json`, 368.9 s):

**v5: 24W-68L-8D, win rate 0.2800, 95% CI [0.2014, 0.3749].**
As Black 15W-28L-7D; as White 9W-40L-1D. Plies mean 61.3. v4
managed 0.37 against v0; v5's defensive leaf made it worse
(0.28), losing as both colours.

### 3.4 Ladder (clause 2) — FAIL

`eval_version.py --version 5 --ladder-games 20 --mcts-sims 100
--seed 100` (identical counts/opponents to v1's measurement):

| Opponent | v5 | v4 | v1 (same counts) |
|---|---|---|---|
| random | 20W-0L-0D (1.0000) | 20W-0L-0D | 20W-0L-0D |
| heuristic (v0) | 7W-13L-0D (0.3500, CI [0.1812, 0.5671]); as Black 4W-6L, as White 3W-7L | 8W-12L-0D (0.4000) | 11W-7L-2D (0.600) |
| mcts (100 sims) | 11W-9L-0D (0.5500, CI [0.3421, 0.7418]); as Black 7W-3L, as White 4W-6L | 4W-16L-0D (0.2000) | 9W-10L-1D (0.475) |
| **Ladder Elo** | **-107.5, interval [-262.0, 46.9]** | **-70.4** | **70.4** |

v5's ladder Elo (-107.5) is BELOW v1's (70.4) and below v4's
(-70.4). One bright spot: v5 BEATS the MCTS ladder opponent
(11-9, 0.55) — better than v1 (0.475) and far better than v4
(0.20). v5's defensive solidity works against the search-based
MCTS player but fails against the heuristic's direct attacking
play. Non-transitivity, again: v5 > mcts > v4 ≈ v5, while
v0 > v5 and v0 > v4.

### 3.5 Swap-rule match vs v1 (secondary, tempo judge)

100 games, standard swap, balanced placement, placement_seed 11,
tempo judge (the harness default since H3)
(`research/iterations/batch-v5-vs-v1-swap.json`, 1067.1 s):

**v5: 62W-16L-22D, win rate 0.7300, 95% CI [0.6357, 0.8073].**
As Black 32W-4L-14D; as White 30W-12L-8D. second_colour: Black
76, White 24 — matching H3's tempo split on the same placement
seed. Winners by colour: Black 44, White 34, draws 22. Plies
mean 86.2. The attack-judge swap variant (for direct
comparability with v4's 0.755 under the attack judge) was NOT
run: the swap match took 1067 s and the attack variant was
killed to save ~18 min for the ladder and v6. The judge is
recorded in the JSON (`"judge": "tempo"`); v4's swap used the
attack judge, so the 0.73 vs 0.755 comparison mixes a judge
change and is a secondary-metric caveat only.

### 3.6 Tactical and cost

Old 20-position suite, fixed driver: **v5 20/20** (= v1's 20/20;
v3/v4 also 20/20). NEW harder suite (29 positions): **v5 25/29**
(prevention 15/16, setup 6/7, winning 2/2, vcf 2/4; fails
`prev-v1bl-g2`, `setup-v1v0s-g92`, `vcf-race-block-or-win`,
`vcf-race-block-or-win-white`), vs v1's 12/29 and v3's 26/29.
v5's prevention (15/16) beats v4's (13/16) — the terminal-aware
leaf helps prevention — but v5 fails both VCF race positions
(the defensive leaf blocks instead of winning the race), where
v4 passes all 4 VCF positions. Cost: v5 is slower than v4 per move (extra
`_winning_moves` calls in the leaf) AND plays much longer games
(88.3 plies mean vs v1, vs v4's 31.8): 100 games vs v1 took
481.4 s (~4.8 s/game) vs v4's 89.9 s (~0.9 s/game). Affordable
for research, but the direction is wrong — slower AND weaker vs
v0.

## 4. Bar check (v5 vs v1)

| Clause | Requirement | v5 | Verdict |
|---|---|---|---|
| 1. Head-to-head vs v1 | >= 40 games, win rate > 0.50, Wilson 95% CI lower bound > 0.50 | 100 games, 0.6500, CI [0.5525, 0.7364] | **PASS** (vs v1; vs v4 it ties 0.49 — see §3.2) |
| 2. Ladder Elo | strictly higher than v1's 70.4, identical counts/opponents | -107.5 (interval [-262.0, 46.9]) | **FAIL** |
| 3. Tactical | pass rate >= v1's | old suite 20/20 = v1's 20/20; hard suite 25/29 vs v1's 12/29 | **PASS** |
| 4. Tests green | ml pytest, npm test | targeted suites green at implementation (test_minimax + test_scoring: 47 passed); full suites run at batch end | not finally assessed |

**Verdict: v5 is REJECTED.** Clause 2 fails decisively: ladder
Elo -107.5 is not strictly higher than v1's 70.4, and the
independent 100-game match vs v0 (0.28, CI entirely below 0.50)
confirms v5 is genuinely worse than v1 against the ladder
anchor — worse even than v4 (0.37). v5 is NOT pushed to master.

## 5. What this means

- **The leaf fix worked mechanically and failed strategically.**
  Missed preventions collapsed (v0: 40.4% -> 3.1%), the game-1
  regression passes (v5 plays the block), and v5 beats the MCTS
  ladder opponent for the first time (0.55). But correct defence
  without attack loses: v5 blocks everything, builds nothing, and
  v0 grinds it down (0.28). The 2-ply minimax lineage has hit its
  ceiling — the worked game (§2.2) shows the saving moves are
  quiet positional moves ranked 11th-12th (outside the top-8)
  that a static leaf ranks LAST because it values the best attack
  rather than the breadth of threats.
- **v6 is H2 (MCTS with heuristic prior)** — the last version of
  this batch. It addresses both gaps: PUCT explores all
  candidates (no top-K cutoff) and evaluates by searched win
  rates across many lines (threat breadth, not best attack). If
  v6 also fails, the batch's conclusion is that this search
  family cannot beat the ladder anchor on CPU, and the next batch
  should change the evaluation (learned value function) rather
  than the search. See iter-6.md.
- **Standing:** master untouched (still 78c733a). v4 and v5 live
  only on branch `research/batch-2`.
