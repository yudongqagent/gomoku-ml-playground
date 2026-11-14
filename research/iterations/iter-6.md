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
- Leaf values are terminal-aware and cheap: the player to move
  with any winning move is +1 (wins next move); the opponent
  with >= 2 winning moves is -1 (cannot block all); the
  opponent with exactly 1 is -0.25 (must block — the search
  resolves which block); otherwise 0. Quiet positions are valued
  by the search itself, not by a static heuristic — a
  threat-space search. NOTE (design refinement, decided BEFORE
  any eval, from a cost measurement): the pre-registration first
  described the leaf as `scoring._leaf_value` squashed to [-1, 1],
  but `_leaf_value` costs ~9.4 ms/call (two full attack scans),
  which would make a 100-sim move cost ~1.3 s and a 100-game
  match ~45 min. The winning-move counts cost ~0.4 ms (via the
  cheap `_winning_moves`) and carry the terminal information
  that matters — a single winning move is resolved by the search
  in 1 ply, and a static best-attack gradient is exactly what v5
  showed to be misleading (it values the BEST attack, not the
  BREADTH of threats). The implemented leaf is the sharper,
  cheaper version of the same terminal-aware idea; if v6's evals
  show the missing attack gradient is the problem, the
  `_leaf_value` leaf (possibly kernel-accelerated) is the
  next-batch follow-up.
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

### 2.1 Aggregates (vs v1, 100 games)

37 losses, **37/37 via open four** (the winner held >= 2 winning
moves before the final move). **0 missed immediate blocks. 0
clean missed preventions in 0 games** (221 threatened turns) —
v6 NEVER misses a clean prevention. Its defence is perfect by
every measure used in this program (suite 29/29, 0 missed
blocks, 0 missed preventions), yet it loses 37 games: every loss
comes from a position where no prevention exists, i.e. the race
was lost earlier — v1 built the attack faster. The MCTS with a
terminal-only leaf defends perfectly but does not attack: in
quiet positions all leaves are 0, so the search has no gradient
toward building an attack, and v6's move choice falls back to
priors (= v1's scoring) plus MCTS noise. v6 is v1 with perfect
defence and no better attack — and that is not enough to beat v1.

Across all three 100-game matches: **166 losses, 166/166 via
open four, 0 missed blocks, 1 clean missed prevention** (0.1% of
1,061 threatened turns; the single one is in the v0 match).
vs v5: 72/72 via open four, 0 missed preventions (510 threatened
turns). vs v0: 57/57 via open four, 1 missed prevention (330
threatened turns). The defence is essentially perfect — and v6
still loses 166 games. The program's central finding, sharpened:
**no version has ever lost by failing to defend; every version
loses by failing to attack first.** v1/v4/v5/v6 all defend
airtight (0 missed blocks everywhere); what differs is who
builds the first unstoppable threat, and v6 builds it last.

### 2.2 Worked games — the negative result (no single losing move)

Three v6 losses were walked back with the corrected method
(`/tmp/workup_v6_game.py`): v1 game 1 (63 plies, v6 as White),
v1 game 3 (29 plies, v6 as White), v0 game 0 (16 plies, v6 as
Black). **In all three, the playout oracle draws from every
tested v6 turn (17 turns total) — no single losing move exists
by the oracle.** v6's losses are distributed drift, not blunders:
the position stays drawable for many moves, v6 never finds a
blunder to make (it defends perfectly), but it also never builds
enough counterplay, and the opponent's attack eventually breaks
through in the real game while the playout (fresh RNG) finds the
drawing defence.

Two findings from the workup, both honestly reported:

**(a) The game-3 finding is WITHDRAWN.** The walk-back's first
version reported ply 20 as the last preventable turn (v6 played
(6,7), losing; (5,6) avoids the loss, playout: draw). Verification (`/tmp/verify_v6_game3.py`) refuted it: playouts after BOTH
moves draw at seeds 0, 1, 2, and (6,7) is the SAFER move by
creator count (0 Black creators after (6,7) vs 1 after (5,6);
(6,7) is 1-ply rank #1, (5,6) rank #2). The error was
methodological: the first version compared the alternative's
playout against the HISTORICAL game outcome, but the playout
engines have fresh RNG state, so even the actual move's playout
diverges from the real game. The corrected method requires the
actual move's playout to lose FIRST (same seeds) before an
alternative counts — and under that bar, no turn in any of the
three games qualifies. (The iter-5 walk-back is unaffected: there
the actual move's playout reproduced the loss.)

**(b) v0 game 0, ply 13 — the clearest "blunder" is not one.**
v6 (Black) played (10,5) while White held an open three on row 6
((7,6),(8,6),(9,6)); White played (10,6) (four), Black blocked
(6,6), White played (11,6) and won with the five
(7,6)-(11,6). But blocking the three does not save the game:
after (10,6) White still has 2 creators ((10,5), (6,9)), and
after (6,6) the same 2. No single Black move at ply 13 leaves
White without a creator — the position was already multi-threat,
lost through accumulation before ply 13. (The classifier's single
clean missed prevention in the v0 match is in a different game.)

**(c) A mechanism for the passivity (game 3, ply 20).** v6's root
visit counts show the most-visited rule picking the high-prior
move over the high-value move: (6,7) had 26 visits (q 0.038,
prior 0.149) while (5,6) had 3 visits (q 0.333, prior 0.081).
With a sparse terminal-only leaf (most leaves 0), visit counts
track priors more than values — the search does not exploit its
own value estimates. A leaf with an attack gradient would give
the q values something to separate on; without it, v6 plays its
priors (v1's scoring) plus noise.

**What changed in the algorithm because of this iteration:**
nothing further can be built into v6 — v6 is the last version of
the batch, and the finding is that the MCTS-with-terminal-leaf
design is the wrong leaf, not a tunable parameter. The
follow-up (next batch) is an MCTS leaf WITH an attack gradient
(`_leaf_value`, kernel-accelerated if the cost is prohibitive),
or a learned value function — see §5.

## 3. v6 evaluation

### 3.1 Smoke (20 games vs v1, seeds 100/200)

**6W-10L-4D, win rate 0.4000, CI [0.2188, 0.6134]**, 185.8 s
(~9.3 s/game; plies mean 89.6, min 25, max 225); colour split
perfectly symmetric (as Black 3W-5L-2D, as White 3W-5L-2D). Below 0.50 but
n=20 with a wide CI — not conclusive; the pre-registered
100-game match decides. Loss analysis on the smoke (§2) already
showed the pattern: 0 clean missed preventions — v6 defends
correctly but loses the race.

### 3.2 Tactical suites

Old 20-position suite, fixed driver: **v6 20/20** (= v1's 20/20).
NEW harder suite (29 positions): **v6 29/29 — a perfect score**
(prevention 16/16, setup 7/7, winning 2/2, vcf 4/4), the first
version to pass all 29, including the 3 positions v3 failed
(`winning-quiet-setup-2` and both `vcf-race-block-or-win`
positions). v6's search is tactically perfect on single-move
decisions — and that makes the match results (§3.3) the more
important finding: the suite measures prevention, setups and
forced wins, NOT attack-building over many moves, and v6's
terminal-only leaf gives no attack gradient in quiet positions.

### 3.3 Head-to-head vs v1 (100 games, seeds 100/200)

**44W-37L-19D, win rate 0.5350, CI [0.4378, 0.6296]**, 885.4 s
(~8.9 s/game). As Black 24W-16L-10D (0.58); as White 20W-21L-9D
(0.49) — roughly symmetric, unlike v4/v5's extreme colour
splits. Plies mean 85.2 (min 19, max 225); win plies 48.5, loss
plies 57.1 (v6 loses SLOWER than it wins — it grinds). All 19
draws are full-board 225-ply games.

**Clause 1 FAILS**: the win rate (0.5350) exceeds 0.50 but the
Wilson 95% CI lower bound (0.4378) does not. v6 is statistically
indistinguishable from v1 over 100 games — the MCTS search adds
nothing measurable against v1 in game play, despite the perfect
tactical-suite score (§3.2).

### 3.4 Head-to-head vs v5 (100 games, seeds 100/200)

**26W-72L-2D, win rate 0.2700, CI [0.1927, 0.3643]**, 581.5 s.
As Black 18W-32L-0D (0.36); as White 8W-40L-2D (0.18). Plies
mean 48.6 (min 18, max 225); win plies 46.5, loss plies 44.4.
v6 loses badly to v5 in both colours — v5's minimax FINDS
attacking sequences (2-ply + best-attack leaf) while v6's MCTS
only defends. v5's attack decides games quickly (48.6 plies vs
85.2 vs v1).

### 3.5 Head-to-head vs v0 (100 games, seeds 100/200)

**39W-57L-4D, win rate 0.4100, CI [0.3187, 0.5080]**, 479.9 s.
As Black 27W-21L-2D (0.56 — v6 BEATS v0 as Black); as White
12W-36L-2D (0.26). Plies mean 51.5 (min 16, max 225); win plies
58.0, loss plies 34.8 (v6 loses FAST to v0 — v0's attack wins
quickly when v6 is White). The extreme colour split is the
passivity story in miniature: as Black the first-move tempo
gives v6 the attack for free and its perfect defence closes
games; as White it must CREATE the attack to overcome Black's
tempo, and it cannot.

### 3.6 Ladder (20 games/opponent, mcts_sims=100, seed 100)

vs random 20W-0L (1.0000); vs heuristic 7W-13L (0.3500, CI
[0.1812, 0.5671]; as Black 4W-6L, as White 3W-7L); vs mcts 9W-11L
(0.4500, CI [0.2582, 0.6579]; as Black 6W-4L, as White 3W-7L).
**Elo -107.5, interval [-262.0, 46.9]** — identical to v5's Elo
(the vs-heuristic result is the same 7W-13L), and far below v1's
70.4. **Clause 2 FAILS.**

The vs-mcts number is the pre-registered A/B, and it goes the
WRONG way: v6 loses 9W-11L to the ladder's HeuristicMCTSPlayer
at the same 100 sims / c_puct 1.5, despite better priors (v1 vs
v0 scoring) and a better leaf (terminal-aware vs crude local
attack). The difference is the leaf GRADIENT: the baseline's
`_local_attack_value` gives the search an attack signal in quiet
positions; v6's terminal-only leaf gives none. Better priors do
not compensate for a missing attack gradient. (n=20, CI [0.26,
0.66] — the direct 50-game match in §3.7 sharpens this.)

### 3.7 Swap match vs v1 (100 games, balanced, placement_seed 11, tempo judge)

**46W-34L-20D, win rate 0.5600, CI [0.4623, 0.6533]**, 897.7 s
(secondary metric, not the bar). As Black 24W-15L-11D (0.59);
as White 22W-19L-9D (0.53). second_colour Black 76 / White 24
(matches H3's tempo split); winners Black 43, White 37, draws 20
(Black-favouring by 6 — the swap balance depends on the engines
too, not just the judge). Plies mean 84.0 (min 17, max 225).

### 3.8 Direct A/B vs HeuristicMCTSPlayer (50 games, seed 100)

**21W-29L-0D, win rate 0.4200, CI [0.2937, 0.5577]**, 358.5 s.
As Black 15W-10L (0.60); as White 6W-19L (0.24) — the same
extreme colour pattern as vs v0. The direct match confirms the
ladder (§3.6): at identical 100 sims / c_puct 1.5, v6's better
priors (v1 vs v0 scoring) and better leaf (terminal-aware vs
crude local attack) do NOT beat the baseline MCTS in game play.
Combined with the ladder (9W-11L at 20 games), v6 is at best
equal to the baseline MCTS and likely slightly worse. The
baseline's crude local-attack leaf provides an attack gradient
that v6's terminal-only leaf lacks — and in Gomoku, the attack
gradient is worth more than the prior quality. (Note: the first
run of this match was killed by a VM replacement after ~7 s;
this is the re-run on the fresh VM.)

## 4. Bar check (v6 vs v1)

| Clause | Requirement | v6 result | Verdict |
|---|---|---|---|
| 1. Head-to-head vs v1 | >= 40 games, win rate > 0.50, Wilson 95% CI lower bound > 0.50 | 100 games: 44W-37L-19D, 0.5350, CI [0.4378, 0.6296] | **FAIL** (lower bound 0.4378 < 0.50) |
| 2. Ladder Elo | strictly higher than v1's 70.4, identical games/opponents | -107.5 (interval [-262.0, 46.9]), vs v1's 70.4 | **FAIL** |
| 3. Tactical | pass rate >= v1's | old 20/20 = v1's 20/20; hard 29/29 vs v1's 12/29 | **PASS** |
| 4. Test suites | ml pytest + npm test green | ml pytest 192 passed; npm test 55 passed | **PASS** |

**Verdict: v6 REJECTED.** Clauses 1 and 2 both fail. v6 is the
tactically strongest version ever built in this program (hard
suite 29/29, 1 missed prevention in 1,061 threatened turns) and
it still loses to v0 (0.41), ties v1 (0.535, CI includes 0.50),
and loses badly to v5 (0.27). The MCTS search adds nothing
measurable in game play over v1's 1-ply scoring — the search
finds every defence and builds no attack.

## 5. What this means

**The search direction has hit its ceiling on this ladder.**
Batch 2 tried three search designs (v4 2-ply minimax, v5
terminal-aware minimax, v6 MCTS with heuristic prior) and all
three failed clauses 1–2, all for the same reason: they defend
perfectly and attack worse than the 1-ply engines. The ladder
anchor (frozen v0, Elo 0) is beaten only by v1 (0.55 h2h) and v3
(0.615 h2h) — both 1-ply scorers. Every search version loses to
v0 (v4 0.37, v5 0.28, v6 0.41). The pattern across all six
versions: **defence is solved (0 missed blocks everywhere, 1
missed prevention in v6's 1,061 threatened turns); the open-four
race is won by whoever attacks first; and every search tried so
far attacks slower than 1-ply scoring.**

The next batch should change the EVALUATION, not the search:

1. **An MCTS leaf with an attack gradient** (v7 candidate):
   `_leaf_value` (best-attack difference, terminal-aware) as the
   v6 leaf, kernel-accelerated via the portable kernel (the
   NumPy kernel scores all 225 moves in ~0.75 ms vs ~9.4 ms for
   the Python `_leaf_value`). The v6-vs-baseline-MCTS A/B (§3.6)
   shows the leaf gradient matters more than the priors.
2. **A learned value function**: train a small net to predict
   game outcome from self-play positions (the B2 batched
   self-play path now makes data generation 1.89x faster on CPU),
   and use it as the MCTS leaf. This is the AlphaZero direction
   the project was built for — the heuristic leaf has been
   explored to its limit.
3. **Attack-first search**: a search that explicitly prioritises
   creating threats (e.g. PUCT with a prior bonus for moves that
   create open threes / double threats, or a two-phase search:
   find our best attack line first, then check the opponent's
   replies). The loss analysis (§2.1) shows the race is lost in
   quiet positions where no defence is needed — the engine must
   use those moves to attack.

What v6 contributes regardless: the hard-suite 29/29 shows MCTS
search DOES find tactical moves 1-ply scoring misses (the
iter-5 game-1 saving move, the 3 positions v3 failed) — the
search machinery works; it needs a value function that rewards
attacking, not just surviving.
