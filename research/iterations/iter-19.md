# Iteration 19 — v19: blended rollout terminal (batch 7, H2)

**Status: IN PROGRESS — pre-registration (§1) written before any eval;
implementation not yet run.** Comparison target is v7 (the batch-7
reference). This is the H2 workstream of batch 7 (H1 is v18, the
rollout leaf for both colours; H3 is v20, the v1 leaf vs greedy
play); work happens on branch `research/batch-7-h2` in the
`gomoku-b7-h2` worktree. v19 does NOT change the root decision rule
(CLOSED/EXHAUSTED), does NOT change sims/priors/book (CLOSED for
White), does NOT change the rollout policy or horizon, and does NOT
touch the Black leaf (v19-as-Black IS v7, exactly as v16 was).

**Method (carried over from batch 3/4/5/6, with the iter-17 §5
correction):** the fast 10-game loop — play 10 games, read EVERY
move of every game, find THE losing move per game (verified with
the solver or a deeper search — never guessed), patch the
algorithm so that, replayed to that critical position, it plays
the perfect move, and record each critical position + correct move
as a regression test. Then play 10 more games and repeat. Max 4
fast-loop iterations (40 games) before validation, or stop when an
iteration finds no new patchable weakness class. Patches must fix
the STRUCTURAL cause, not the single position. The full 4-clause
validation battery (100-game matches + ladder) runs on the FINAL
patched v19 only. **Fast-loop games run against v0 as well as v7
in this batch (iter-17 §5): v17's weakness only showed at battery
scale against v0, where the fast loop never looked.** Fast-loop
games use development seeds (1001/2001, 1002/2002, ...) distinct
from the bar battery's pre-registered seeds (100/200), so no
development game overlaps an eval game.

## 1. Design and pre-registration

**Comparison target: v7** (primary, the batch-7 reference: ladder
Elo 190.8, on master at `bf866f7`). The bar's clause 1 is
head-to-head vs v7 (normal rules, 100 games, alternating colours,
win rate > 0.50 with Wilson 95% CI lower bound > 0.50), clause 2
is ladder Elo strictly higher than v7's 190.8 (identical 20
games/opponent: random, heuristic, mcts, mcts_sims=100), clause 3
is tactical pass rate >= v7's (old suite 20/20, hard suite 29/29),
clause 4 is all test suites green (ml pytest, npm test, tsc).
**Secondary head-to-heads, also pre-registered:** v19 vs v0 (100
games) and a 100-game swap match vs v7 under the tempo judge (the
harness default). Non-transitivity is recorded prominently.

**The improvement: v19 = v16 with BLENDED rollout terminals.**
Everything else is v16 unchanged: the White-only 2-ply rollout
leaf (2 plies under the v1 rollout policy, symmetric v1 gradient
at the end), v7's leaf when the engine plays Black, 100 sims,
c_puct 1.5, v1 priors over all candidates, win/block scans before
search, most-visited root child. v16's two fast-loop patches are
kept verbatim: (1) the v1 rollout policy + gradient-evaluated
rollout end (a rollout-found must-block is a prediction, not a
fact — iter-16 §2.1); (2) open-four priority in the rollout policy
(own v1 attack >= 200,000 beats the argmax — iter-16 §2.2). The
ONLY change is what a rollout terminal is worth.

**The diagnosed weakness (iter-16 §3.1, §5).** v16's rollout leaf
compresses all q values toward -1.0 in creator-rich positions:
the 2-ply horizon reaches a forcing loss from nearly every quiet
leaf, so almost every leaf returns exactly -1.000 and the search
loses the discrimination that v7's smooth gradient preserves. On
the hard suite's `prev-v2v1s-g12` (a forced loss for White), the
good moves (the counter-fours (12,2)/(13,2)) sit at q -0.75 vs
-1.00 for the blockers — a 0.25 gap the blockers' 3x prior
advantage (0.130 vs 0.041) ties into a 5-5 visit tie, broken by
RNG toward (7,3). The good moves are still least-bad by q, but
visits tie. v16's evaluation is more accurate than the suite's
(the position is a forced loss; the suite's expected moves are a
weak-v1-oracle artifact), but the FLATTENING is a genuine weakness
beyond this one position: in any creator-rich position the search
cannot rank defences it knows are all losing, because the leaf
says they are all equally lost.

**The blend, exactly.** When the rollout reaches a terminal
(±1 — a win/loss found inside the rollout, or at the rollout's
end), v16 returns the raw terminal value. v19 returns the terminal
BLENDED with the symmetric v1 gradient evaluated at the terminal
position:

```
blended(term, board_T, player_T) = clamp(term + g, -1.0, 1.0)
    where g = _attack_gradient_v1(board_T, player_T)
```

`board_T` is the position where the rollout reached the terminal
(the position BEFORE the winning move is played — the terminal is
a prediction that the player to move there wins, not a finished
game), `player_T` is the player to move there, and the blend is
formed in `player_T`'s perspective and sign-adjusted to the leaf
player's perspective exactly as v16 sign-adjusts the raw terminal
(negated on odd plies). The gradient is v14's symmetric v1 form
(cap ±0.2, scale 7.5, v1 thresholds) — the SAME gradient v16
already evaluates at quiet rollout ends, so no new constant is
introduced: the blend has zero tunable parameters beyond v16's.
A draw terminal (term 0.0, board full) is returned unchanged —
a full board has no empty cells, so its gradient is 0 anyway.

**What does NOT change.** Leaf-start terminals keep their exact
values (+1/-1/-0.25): those are FACTS about the real position,
not predictions. The rollout-end must-block stays
gradient-evaluated (v16 patch 2 — unchanged). Quiet rollout ends
stay the plain v1 gradient. The blend applies ONLY to
rollout-found ±1 terminals — the same principle as patch 2
(a rollout-found value is a prediction under the 1-ply policy,
not a fact), extended from the must-block to wins and losses.

**Why this form, and why the clamp.** The terminal is the
dominant signal (a concrete 2-ply sequence); the gradient is a
tie-breaker among lines with the same terminal outcome. The
gradient's ±0.2 cap — the same cap iter-7 chose so a static
heuristic never outweighs an exact terminal signal — bounds the
shift: a blended rollout loss lands in [-1.0, -0.8] and a blended
rollout win in [0.8, 1.0], so the terminal's dominance over quiet
leaves (±0.2) and the must-block (-0.25) is preserved exactly,
and the ordering vs non-terminal leaves is unchanged. The blend
reorders ONLY among lines that share a terminal outcome — which
is precisely the flattened set. The clamp is the
predictions-never-exceed-facts rule: a rollout win with a
favourable gradient must not outrank an exact win at the leaf
start (+1.0), and a rollout loss with an unfavourable gradient
must not sink below an exact loss (-1.0). With the clamp, the
discrimination appears where it is needed: among LOSING lines,
the ones with counterplay breadth at the terminal position rise
above -1.0 while hopeless lines stay pinned at -1.0 (least-bad
separates from hopeless); among WINNING lines, weak wins with
negative gradient drop below 1.0 (robust wins separate from
fragile ones).

**Expected effect.** In creator-rich positions the q gap between
good defences (counterplay breadth survives to the terminal
position) and bad ones (no counterplay) widens from v16's
compressed ~0.25 toward the gradient's full spread, so PUCT
visits follow q instead of tying on priors — the g12 mechanism,
addressed at the leaf rather than the root (a root tie-break rule
would be the CLOSED root-decision family). In real games White's
defence selection should improve the same way: among lines the
rollout calls losing, prefer the one that keeps counterplay. No
change is expected in quiet positions (no terminals to blend) or
in the Black games (v19-as-Black IS v7, as v16 was). The cost is
one extra `attack_v1_both` call per rollout terminal (~24% of
leaves per iter-16 §1.1, ~4 ms each) — roughly +1 ms per leaf
call, affordable.

**What would refute it.** If the battery shows the blend does not
move the h2h vs v7 (v16: 0.5750, CI lower 0.4771) or the tactical
suite (v16: 28/29) — e.g. because the gradient at terminal
positions is almost always aligned with the terminal (positive at
wins, negative at losses) and the clamp pins nearly everything —
then the blend is inert and the q-compression diagnosis needs a
different mechanism (e.g. a deeper rollout, already refuted at 4
plies, or a static double-creator term). The 100-game battery is
the verdict; single-position diagnostics cannot validate a global
search change.

**Eval seeds (pre-registered):** h2h vs v7: v19=100 / v7=200; h2h
vs v0: v19=100 / v0=200; ladder seed 100, 20 games/opponent,
mcts_sims=100; swap 100 games vs v7, balanced placement,
placement_seed 11, tempo judge (the harness default; recorded in
the JSON). Fast-loop dev seeds 1001/2001, 1002/2002, ...
(iteration k uses 1000+k / 2000+k, per iter-8 §4's policy).

### 1.1 Battery (the verdict — pre-registered seeds and commands)

On the FINAL v19 only (after the fast loop):

- 100 games vs v7, seeds 100/200 (clause 1).
- 100 games vs v0, seeds 100/200.
- 100-game swap vs v7, balanced placement, placement_seed 11,
  tempo judge (the harness default), seeds 100/200.
- Ladder 20 games/opponent (random, heuristic, mcts;
  mcts_sims=100), seed 100 (clause 2).
- Tactical old 20/20 + hard 29/29, full-dispatch driver
  (`ml/scripts/eval_tactical_full.py`, `player.choose_move` —
  NOT `eval_version.py`'s probe, which bypasses the MCTS
  dispatch) (clause 3).
- ml pytest + npm test + tsc green (clause 4).

**Integration.** `ml/gomoku_ml/mcts_heuristic.py` gains the
blended terminal (`_blend_rollout_terminal` + the
`rollout2_v1_blend` leaf mode, sharing `_rollout_move` and
`_attack_gradient_v1` with v16); `ml/gomoku_ml/scoring.py`
gains ONLY the `ENGINE_VERSIONS[19]` registry entry.
`kernel_v1.py` (+ its tests and fixtures) is v16's exact port
(cherry-picked from `research/batch-6`, NOT re-ported).
`baseline.py` is frozen and untouched.

## 2. Loss analysis for iteration 19

The fast-loop worked games (§2.1-2.4) ARE the loss analysis.
Aggregate counts (§2.5) are secondary.

### 2.1 Fast-loop iteration 1 — 10 games vs v7 + 10 games vs v0

**Batch vs v7:** `research/iterations/fastloop1-v19-vs-v7.json`
(seeds v19=1001 / v7=2001 — development seeds, distinct from the
bar battery's 100/200). **3W-7L-0D, win rate 0.3000, CI [0.1078,
0.6032]**, 445.0 s. Split: as Black 1W-4L (0.20), as White 2W-3L
(0.40) — below v16's first loop (5W-4L-1D; White 4W-1L) on the
same seeds, but n=5 per colour is noise, and the Black games are
v7-vs-v7 mirrors decided by seed (the RNG stream diverges because
v19's White games tie-break differently). Classifier
(`ml/scripts/classify_batch_losses.py`): 7/7 losses via open four,
0 missed blocks, 0 clean missed preventions in 28 threatened turns
(in losses). Defence is still perfect; the losses are lost races.

**Batch vs v0:** `research/iterations/fastloop1-v19-vs-v0.json`
(seeds v19=1001 / v0=2001). **9W-1L-0D, win rate 0.9000, CI
[0.5958, 0.9821]**, 196.7 s. Split: as Black 4W-1L (0.80), as
White 5W-0L (1.00) — White perfect against the frozen baseline,
as v16 was. Classifier: the single loss (game 2, Black — v7's
move) via open four, 0 missed blocks, 0 missed preventions.
No White weakness shows against v0 in this iteration.

**The losing move in each White loss vs v7** (walk-back,
`ml/scripts/walkback_loss.py`, corrected oracle method — the
actual move's v1-vs-v1 playout must lose at all 3 seeds before an
alternative counts; seeds 0,1,2; JSONs:
`research/iterations/walkback-fl1-g{5,7,9}.json`):

| Game | Last preventable turn | v19 played (loses 3/3) | Perfect move (oracle) |
|---|---|---|---|
| 5 (White, 89 plies) | none | — (ply 81 draws 3/3; ply 83+ lose 3/3, no save) | — |
| 7 (White, 65 plies) | none | — (ply 55 draws 3/3; ply 53 loses 3/3, no save) | — |
| 9 (White, 31 plies) | ply 19 | (11,6) | (9,6) / (10,5) / (10,7) / (4,4) / (3,4) |

**Games 5 and 7 — no losing move (beyond the oracle's horizon).**
In both, White was fine (oracle draws 3/3) as late as 2 plies
before the end, then Black's single move made it unwinnable with
no saving reply: game 5, Black's (5,4) at ply 82 starts a forcing
sequence (four at (5,5), then (4,5), (3,4), (2,3) wins — a
diagonal four built over 3 Black moves, each forcing White's
block); game 7, Black's quiet (7,2) at ply 54 sets up the
(10,9)-(10,12) kill 5 plies later. The v1 oracle is too weak to
see Black's killer move — the same pattern as v16's iteration-2
games 5/9, and the same conclusion: verifying a save here needs a
stronger oracle than v1 (v7-vs-v7 playouts), and the weakness
class (forcing sequences / quiet setups resolving in 3–6 plies,
beyond the 2-ply rollout horizon) is the OPEN class from iter-16
§2.2, not a new one. Not patched.

**Game 9, ply 19 (White to move) — the iteration's finding, and
the blend is EXONERATED as its cause.** v19 played (11,6), which
loses 3/3; five alternatives save ((9,6) and (10,7) win 3/3,
(10,5) draws 3/3, (4,4) wins 3/3). Critical analysis (root dumps,
fresh player, seed 0): **v16 plays (11,6) too, with byte-identical
root stats** (20 visits, our-q +0.109; the saves at 4–6 visits,
q ≈ -0.05 to 0.00) — the blend changed NOTHING on this position,
so this loss is v16's weakness inherited, not a blend regression.
v7 plays the save (10,5) (12 visits, q +0.066 vs -0.049 for
(11,6)). Mechanism, verified by direct computation: Black holds
FOUR creators ((9,6), (11,6), (10,5), (10,7)); (11,6) blocks the
wrong one. The real kill is Black's QUIET setup (11,5) at ply 20
— it builds the diagonal (8,2),(9,3),(10,4),(11,5),(12,6) that
wins at ply 30 — a move the 1-ply rollout policy never plays
(it plays the argmax creator (10,5) instead) and a kill that
resolves 5 plies out, beyond the 2-ply horizon. Worse, the
rollout leaf actively misranks the position: after (11,6), the
rollout exchange (Black (10,5), White blocks (10,7)) ends in a
position where White's gradient is +0.198 (nearly maxed — White's
four-makers (4,4)/(3,4) are still live), while after the save
(9,6) the rollout exchange ends at +0.143. The rollout leaf
rewards the move that leaves White's threats on the board over
the move that spends a block on the real killer — its value is
dominated by the noise of the 1-ply policy's arbitrary replies.

**Candidate patch measured and REFUTED — leaf-start gradient
blend.** The natural fix (blend the rollout-end gradient with the
gradient at the LEAF-START position, where Black's creator count
is visible before the exchange erases it) does not fix this
position: the leaf-start gradients are (11,6) +0.1822 for White
vs (9,6) +0.1907, (10,5) +0.1941 — a 0.008–0.012 edge for the
saves, far smaller than the rollout's 0.054 edge for the loser,
so even a 50/50 blend still ranks (11,6) first (0.190 vs 0.167
for (9,6)). v7's advantage here comes from its SEARCH compounding
the static signal over depth, not from the leaf level — and
weighting the static leaf more heavily would just be v7 with
extra steps, destroying the rollout's White gains. The other
candidate fixes are out of scope for a fast-loop patch: a deeper
rollout is REFUTED (4-ply false terminals, iter-16 §1.1), a
3-ply rollout is a new mechanism needing its own pre-registration,
and a VCF/VCT solver at the root is a different architecture.
**No patch in iteration 1.** The position is recorded as a
regression xfail (`fl1-g9-ply19`), not dropped.

**Black losses (games 0, 2, 4, 6) — v7's moves, recorded only.**
Walk-backs (`research/iterations/walkback-fl1-g{0,2,4,6}.json`):
game 0 ply 12 played (11,9), 3 saves; game 2 ply 40 played
(5,11), 4 saves; game 4 ply 20 played (9,7), 1 save; game 6 ply
20 played (10,8), 1 save. v19-as-Black IS v7, so these are out of
scope for patching; recorded as regression xfails.

**Regression tests:** `ml/gomoku_ml/regression_positions_v19.json`
gains 5 positions (all xfail: fl1-g9-ply19, the open White
weakness, plus the 4 Black positions). Targeted suites:
test_regression_v19 + test_regression_v16 5 passed, 7 xfailed.

**Iteration 1 verdict: no new patchable weakness class.** But the
blend is a new mechanism and one iteration is thin validation —
the specific risk is a loss where v19 plays a loser v16 does NOT
play (a blend-caused regression), which iteration 1 did not
produce. Iteration 2 runs to check for exactly that.

### 2.2 Fast-loop iteration 2 — 10 games vs v7 + 10 games vs v0 (LOOP STOPS)

**Batch vs v7:** `research/iterations/fastloop2-v19-vs-v7.json`
(seeds v19=1002 / v7=2002). **8W-2L-0D, win rate 0.8000, CI
[0.4902, 0.9433]**, 516.3 s. Split: as Black 3W-2L (0.60), as
White **5W-0L (1.00)** — the iteration-1 White 2W-3L was noise;
across both iterations White is 7W-3L (0.70). Classifier: 2/2
losses via open four, 0 missed blocks, 0 missed preventions in 4
threatened turns (losses). Defence still solved.

**Batch vs v0:** `research/iterations/fastloop2-v19-vs-v0.json`
(seeds v19=1002 / v0=2002). **7W-2L-1D, win rate 0.7500, CI
[0.4422, 0.9191]**, 220.3 s. Split: as Black 2W-2L-1D (0.50), as
White **5W-0L (1.00)** — White perfect against v0 in both
iterations (10W-0L). Classifier: 2/2 losses via open four, 0
missed blocks, 0 missed preventions in 6 threatened turns.

**Walk-backs — all four losses in this iteration are Black
games, i.e. v7's moves (v19-as-Black IS v7), recorded only:**
vs v7 game 6 ply 30 played (12,6), 1 save; game 8 ply 40 played
(9,11), 2 saves (`research/iterations/walkback-fl2-g{6,8}.json`);
vs v0 game 2 ply 14 played (5,8), 1 save; game 4 ply 16 played
(8,7), 2 saves (`research/iterations/walkback-fl2v0-g{2,4}.json`).

**No White loss in this iteration against either opponent, hence
no new patchable White weakness class — the fast loop STOPS here
(2 iterations) per its stopping rule.** In particular, no
blend-caused regression appeared: across 20 White games (10 vs
v7, 10 vs v0) v19 lost 3, all in iteration 1, all beyond-horizon
losses shared with v16 — never a move v16 plays correctly and
v19 gets wrong. Fast-loop totals over 40 games: v19 27W-12L-1D
(0.6875); as White 17W-3L (0.85), as Black 10W-9L-1D (0.525).
The 100-game battery is the verdict — the fast loop's 40 games
are development signal, not the bar.

### 2.3 Fast-loop iteration 3 — NOT RUN (loop stopped after iteration 2)

### 2.4 Fast-loop iteration 4 — NOT RUN (loop stopped after iteration 2)

### 2.5 Aggregate counts (secondary)

Fast-loop defence (2 iterations, 40 games, 12 losses): 12/12
losses via open four, 0 missed blocks, 0 clean missed preventions
(41 threatened turns in losses). Defence remains solved; every
loss is a lost race, as in all batches since v6. White lost 3 of
20 games (all vs v7, all iteration 1); Black lost 9 of 20 (v7's
mirror games).

## 3. v19 evaluation

Battery on the FINAL version (after the fast loop stopped).
Pre-registered seeds 100/200 throughout (§1.1). The fast loop made
no patches, so the battery version is the pre-registered design
exactly (commit `a4713ea`).

### 3.1 Tactical suites — 20/20 old, 27/29 hard (CLAUSE 3 FAILS)

`ml/scripts/eval_tactical_full.py --version 19 --suite both
--seed 100` (identical protocol to v7's and v16's measurements):
old 20/20, hard **27/29** — one WORSE than v16's 28/29. v7 under
the identical protocol: 20/20 + 29/29. The two failures are both
prevention positions, White to move:

**`prev-v2v1s-g12` (chose (7,3), expected (12,2)/(13,2)) — the
blend is INERT here, as the pre-battery root dump predicted.**
v16 fails this position identically. The root dumps are
byte-identical between v16 and v19 (good moves q -0.75 at 5
visits, blockers q -1.00 at 5 visits, RNG tie-break picks (7,3)):
no leaf in the 100-sim search changes value under the blend,
because the gradient at the terminal positions in this subtree is
aligned with the terminal (negative at losses) and the clamp pins
everything at -1.0. The counterplay breadth the blend was meant
to reward — White's four (12,2) — is already spent/blocked by
the time the rollout reaches a terminal, so White's breadth at
the terminal position is low. The pre-registration's refutation
scenario (§1, "the gradient at terminal positions is almost
always aligned with the terminal") is exactly what happens on
g12. (The position is still a forced loss for White and v19's
evaluation is still more accurate than the suite's — iter-16
§3.1's analysis stands — but the blend does not change the
outcome.)

**`prev-v1v1s-g45` (chose (10,4), expected (11,7)/(12,7)) — a
genuine blend REGRESSION.** v16 passes this position (28/29,
choosing (11,7)) under the identical protocol; v19 fails it.
Mechanism (root dumps, fresh player, seed 0): v16 gives the
losing move (10,4) 1 visit at q -1.000 and the good moves 2
visits each at q -0.375, so the good moves win 2-1. v19's blend
SOFTENS (10,4)'s -1.000 to -0.932 — (10,4) is an attacking move
that creates White threats while failing to prevent Black's open
four, so it holds counterplay breadth at the terminal position —
and the softened q plus its 3x prior advantage (0.133 vs 0.042)
earns it a second visit, tying the good moves 2-2-2; under the
battery's RNG state the tie-break picks (10,4). (With a fresh
player v19 plays (12,7) — a pass — so the failure is the
tie-plus-RNG, but the blend created the tie.) The blend's core
assumption — "losing with counterplay breadth is better than
losing without it" — is wrong here: the right moves PREVENT the
loss and need no counterplay, while the losing move's counterplay
is exactly what makes it attractive to a softened leaf. The blend
cannot distinguish "counterplay that saves the game" from
"counterplay that loses more slowly". This is the refutation of
the blend as a general mechanism, at the tactical level: it
reorders among losing lines, and the reordering is not always
right.

### 3.2 Head-to-head vs v7 — 52W-37L-11D, 0.5750 (CLAUSE 1 FAILS)

`research/iterations/batch-v19-vs-v7.json` (100 games, normal
rules, seeds 100/200, 6540.8 s): **v19 52W-37L-11D, win rate
0.5750, 95% CI [0.4771, 0.6673]** — aggregates recomputed from
the raw records (match the script's printed summary exactly).
Split: as Black 17W-26L-7D (0.410), as White 35W-11L-4D (0.740).
Clause 1 requires the CI lower bound > 0.50; 0.4771 < 0.50, so
**clause 1 FAILS** — the same verdict as v16, with IDENTICAL
headline aggregates (v16: 52W-37L-11D, 0.5750, CI [0.4771,
0.6673]). The identity is a coincidence of the totals, not of
the games: the colour splits moved sharply in opposite
directions. White improved substantially (0.740 vs v16's 0.640,
+5 wins — the strongest White split vs v7 in the program, above
v16's 0.64 and v13's best), which is the blend working as
designed in real games: among losing lines, preferring the one
that keeps counterplay wins more White games. But the Black
mirror games ran 5 games worse (0.410 vs v16's 0.510) — and the
Black games are v7-vs-v7 mirrors decided by seed (v19-as-Black
IS v7; the games differ from v16's only because the RNG stream
diverges after v19's White games consume different tie-breaks),
so the -5 is seed noise, not a blend effect. The net is the
same 0.5750: the White gain was exactly cancelled by Black
mirror noise. This is the starkest demonstration yet that the
Black mirror games cap the combined h2h — the White split would
need to reach ~0.80 for the combined number to clear the bar
while Black runs at 0.41. (Two runtime restarts interrupted this
match, at 10/100 and at 82/100 games; each re-run with the same
seeds reproduced the saved partial EXACTLY — all records
identical — and the final games are new. Determinism verified
twice; the restarts cost only wall-clock time.) Classifier:
37/37 losses via open four, 0 missed blocks, 0 clean missed
preventions in 115 threatened turns (losses). Defence still
solved.

### 3.3 Head-to-head vs v0 — 84W-16L-0D, 0.8400

`research/iterations/batch-v19-vs-v0.json` (100 games, normal
rules, seeds 100/200, 1937.4 s): **v19 84W-16L-0D, win rate
0.8400, 95% CI [0.7558, 0.8990]** — aggregates recomputed from
the raw records. Split: as Black 34W-16L-0D (0.68), as White
**50W-0L-0D (1.000)**. The headline AND the splits are identical
to v16's (84W-16L-0D; Black 34W-16L, White 50W-0L): against the
frozen baseline the blend changes nothing at the aggregate
level — White is perfect under both versions, and all 16 losses
are as Black (v19-as-Black IS v7; v7 lost 13 as Black to v0 in
its own battery, 16 here is seed/RNG-stream noise). Far above
v7's 0.6650 vs v0 (64W-31L-5D), and the entire gain is
White-side, as with v16. (One runtime restart interrupted this
match at 19/100 games; the re-run with the same seeds reproduced
the saved partial EXACTLY. Determinism verified a third time.)
Classifier: 16/16 losses via open four, 0 missed blocks, 0 clean
missed preventions in 70 threatened turns (losses). **Ordering
is transitive: v19 > v7 (0.5750) > v0 (v7 0.6650, v19 0.8400) —
no non-transitivity**, unlike v4 (v4>v1>v0>v4) and v14
(v14>v7>heuristic>v14). The v19 gain over v7 is a genuine
strength gain that extends down the ladder, not a matchup
artefact — the same conclusion as v16's.

### 3.4 Swap match vs v7 — 72W-26L-2D, 0.7300 (edge ESTABLISHED)

`research/iterations/batch-v19-vs-v7-swap.json` (100 games,
swap rule, balanced placement, placement_seed 11, tempo judge,
seeds 100/200, 3142.5 s): **v19 72W-26L-2D, win rate 0.7300,
95% CI [0.6357, 0.8073]** — aggregates recomputed from the raw
records. Split: as Black 37W-12L-1D (0.75), as White
35W-14L-1D (0.71). The CI lower bound 0.6357 > 0.50, so the
swap edge is ESTABLISHED — the second-strongest swap result in
the program (v17 0.74), well above v16's 0.615 (CI lower
0.5171) and v7's own unestablished 0.57. Notably, the swap edge
(0.73) is much larger than the normal-rules h2h (0.5750):
v19's advantage is NOT an opening artefact — it survives the
swap, which neutralizes the empty-board first-move advantage
where much of v7's edge lived (v7 was 0.89 as Black under
normal rules vs 0.52 under swap). Under swap the blend is
active in roughly half the games (White) and v19 wins as both
colours (0.75 Black / 0.71 White), so the midgame strength is
real in both seats. Classifier: 26/26 losses via open four, 0
missed blocks, 0 clean missed preventions in 103 threatened
turns (losses). Defence still solved.

### 3.5 Ladder — Elo 381.7 [146.3, 617.1] (CLAUSE 2 PASSES)

`research/iterations/ladder-v19.json` (`eval_version.py
--version 19 --ladder-games 20 --mcts-sims 100 --seed 100`,
60 games, 773.7 s): vs random 20W-0L-0D (1.0000), vs
heuristic 18W-2L-0D (0.9000; Black 8W-2L, White 10W-0L), vs
mcts 9W-11L-0D (0.4500; Black 4W-6L, White 5W-5L). **Ladder
Elo 381.7, interval [146.3, 617.1]** — strictly above v7's
190.8 [21.8, 359.9] at identical counts/opponents, so **clause
2 PASSES**. The estimate and interval are IDENTICAL to v16's
(381.7 [146.3, 617.1]) — the per-opponent counts are the same
too (random 20-0, heuristic 18-2, mcts 9-11). The blend buys
nothing on the ladder: v19 still loses to the baseline MCTS
anchor (9-11, vs v7's 8-12), and the ladder Elo is driven
entirely by the heuristic matchup, where v19 matches v16
exactly. The ladder cannot distinguish v19 from v16; the
differences live only in the v7 h2h colour splits (§3.2) and
the swap match (§3.4).

### 3.6 Test suites (CLAUSE 4 PASSES)

Final tree, all green: ml pytest **269 passed, 7 skipped, 11
xfailed, 0 failures** (279 s). Delta vs the v16 port baseline
(256 passed, 7 skipped, 4 xfailed) is exactly +13 passed
(12 new tests in `test_mcts_heuristic.py` + the regression
smoke test) and +7 xfailed (the 7 v19 regression positions,
all expected-fail). `npm test`: **55/55 passed** (after `npm
ci`; fresh worktrees have no `node_modules`). `tsc --noEmit`:
clean, exit 0. `baseline.py` diff vs master is empty — the
frozen file was never touched. **Clause 4 PASSES.**

## 4. Bar check (v19 vs v7)

| Clause | Requirement | v19 | Verdict |
|---|---|---|---|
| 1 | h2h vs v7, win rate > 0.50, Wilson 95% CI lower > 0.50 | 52W-37L-11D, 0.5750, CI [0.4771, 0.6673] | **FAIL** |
| 2 | Ladder Elo > v7's 190.8, identical counts/opponents | 381.7 [146.3, 617.1] | **PASS** |
| 3 | Tactical >= v7's 20/20 + 29/29 | 20/20 + 27/29 | **FAIL** |
| 4 | All test suites green | pytest 269/0 fail, npm 55/55, tsc clean | **PASS** |

**v19 is REJECTED (2/4).** It passes the same two clauses as
v16 (ladder, tests) and fails the same two (h2h, tactical) —
but it fails tactical WORSE than v16 (27/29 vs 28/29) because
the blend itself regresses g45, while improving the White h2h
split (0.74 vs 0.64) and the swap match (0.73 vs 0.615). The
headline h2h, v0, and ladder numbers are identical to v16's;
the blend's effects are real but confined to the White seat
and the swap format, and they come with a tactical cost.

## 5. What this means

**The blend is refuted as a general mechanism.** It does
exactly what §1 predicted it would do in real games — it
restores discrimination among losing lines, and White wins
more games for it (0.74 vs v7, the best White split in the
program; swap 0.73, second-best). But the same discrimination
is what breaks g45: the blend cannot distinguish "counterplay
that saves the game" from "counterplay that loses more
slowly", and on g45 it rewards the losing move's counterplay
until the tie-break picks it. A mechanism that trades a
tactical regression for a White-split gain is not a step
forward — and the combined h2h does not move at all, because
the Black mirror games (v7-vs-v7, seed-decided) swallow the
White gain whole. The blend's failure mode is now precise:
it is a good tie-breaker among lines that all lose, and a bad
judge of which lines lose.

**What changed because of a specific loss: nothing was
patched.** The fast loop (2 iterations, 40 games) found no
patchable weakness class: the one concrete White losing move
(fl1-g9 ply 19, (11,6)) is shared byte-identically by v16,
so the blend is exonerated; the candidate patch (leaf-start
gradient blend) was measured and refuted on the position
itself (50/50 blend still ranks the loser first); the other
White losses have no losing move (the oracle draws, then a
single Black move makes the position unwinnable with no
saving reply — the same OPEN class as iter-16 §2.2); all
remaining losses are Black games, i.e. v7's moves. The
battery version is the pre-registered design exactly. The
regression file records all 7 positions as xfails for the
next iteration.

**The White-side story is now the program's clearest
pattern.** v16's rollout leaf fixed White against v0 (50W-0L)
and improved White vs v7 (0.64); v19's blend pushes White
vs v7 to 0.74 and the swap to 0.73. White is no longer the
weakness in the v7 matchup — Black is, and Black is v7
itself. Any further gain in the combined h2h must come from
the Black seat, which is exactly batch 7's H1 (v18: rollout
leaf for both colours). H2's job was to fix the leaf's
q-compression for White; it did, and the price was one
tactical position. The next lever is not another leaf tweak
— it is H1's Black-side rollout, or a value function that
sees quiet setups (the fl1-g9 killer was a quiet setup 5
plies out, invisible to a 1-ply policy and to any gradient).

**Defence remains solved.** Across the battery (v7, v0,
swap: 79 losses), 79/79 losses are via open four, with 0
missed blocks and 0 clean missed preventions in 288
threatened turns. The blend did not touch defence — it
reorders losing lines, and every loss is still an open-four
race lost on attack, not on defence.

**Methodological note.** Three runtime restarts interrupted
this battery (v7 match at 10/100 and 82/100, v0 match at
19/100). Each re-run with identical seeds reproduced the
saved partial EXACTLY, record for record — determinism
verified three times. The per-game JSON flush is what makes
this recoverable; without it, ~5 hours of compute would
have been lost. The restarts cost only wall-clock time.
