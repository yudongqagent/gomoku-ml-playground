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

## 3. v19 evaluation

Battery on the FINAL version (after the fast loop stopped).
Pre-registered seeds 100/200 throughout (§1.1).

## 4. Bar check (v19 vs v7)

## 5. What this means
