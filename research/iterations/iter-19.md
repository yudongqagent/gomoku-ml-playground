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

## 4. Bar check (v19 vs v7)

## 5. What this means
