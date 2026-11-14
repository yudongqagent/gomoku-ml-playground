# Iteration 10 — v10: White weakness / colour-asymmetric search (batch 4, H2)

**Status: PRE-REGISTERED — design, eval seeds and White-measurement
plan recorded BEFORE any eval (see §1).** Comparison target is v7
(the batch-4 reference). This is the H2 workstream of batch 4
(H1 is the q-vs-visits root decision, a separate workstream on
`research/batch-4-h1` — v10 does NOT change the root decision
rule); work happens on branch `research/batch-4-h2` in the
`gomoku-b4-h2` worktree.

## 1. Design and pre-registration

**Comparison target: v7** (primary, the batch-4 reference:
ladder Elo 190.8, vs v1 0.7050). The bar's clause 1 is
head-to-head vs v7 (normal rules, 100 games, alternating
colours, win rate > 0.50 with Wilson 95% CI lower bound >
0.50), clause 2 is ladder Elo strictly higher than v7's 190.8
(identical 20 games/opponent, mcts_sims=100), clause 3 is
tactical pass rate >= v7's (old suite 20/20, hard suite 29/29),
clause 4 is all test suites green. **Secondary head-to-heads,
also pre-registered:** v10 vs v0 (100 games) and a 100-game
swap match vs v7 under the tempo judge (the harness default).

**The improvement: v10 = v7 + the 2-ply threat check at the
root + colour-asymmetric search depth.** In full:

- Base is v7 EXACTLY: `HeuristicMCTS` with
  `leaf_mode="gradient"` (capped ±0.2 attack-gradient leaf),
  c_puct 1.5, v1 priors over all candidates, win/block scans
  before search, most-visited root child with RNG tie-break.
- Addition 1 (colour-SYMMETRIC, directed by the brief): v3's
  2-ply threat check at the MCTS root, ported exactly as in
  v8 (`ValueMCTS.choose_move`, iter-8 §4.1): if the opponent
  has creators, the root choice is restricted to safe moves
  (forcing, or no creator remains live); if no root child is
  safe, the most-visited move is played (best effort). With
  no creators the choice is unchanged. v7 does NOT have the
  threat check; iter-8 §7 finding 3 calls it a durable
  portable gain (fixed v8's hard tactical 24/29 → 29/29;
  defence airtight).
- Addition 2 (colour-CONDITIONAL — the H2 hypothesis): the
  root search runs **200 sims when the engine plays White
  and 100 sims when it plays Black**. NOTHING else varies by
  colour: same priors, same leaf, same c_puct, same threat
  check, same decision rule. The colour is known at
  `choose_move` time (the `player` argument is the engine's
  own colour for the game).

**Why more search for White, and why NOT a colour-conditional
value net.** The brief offers three H2 designs (colour-
conditional value training, more search for White, colour-
aware leaf/priors); this workstream chooses more search for
White, for evidence-based reasons:

1. v8's lesson (iter-8 §7 finding 1) is that a value net
   trained on weak self-play is WORSE than no value function
   unless shrunk hard — the 4,350-game net generalises
   (holdout 0.72) yet at full weight loses 0-10 to v0, and
   at α=0.25 the net leaf still lands at ladder Elo 0.0
   against v7's gradient leaf at 190.8. The net leaf is a
   step BACK from the gradient leaf; grafting a
   (colour-conditional) net onto v7's leaf would contaminate
   the best leaf in the program with the weakest. v10 uses
   NO net value, so no α sweep is needed (the sweep rule in
   the brief applies only to designs that use a net value).
2. The White gap is better explained as an EXPLORATION /
   tempo problem than a value-estimation problem. Iter-7
   §2.1's fast loop found exactly one White weakness class
   the cap patch did NOT fix: game 3, ply 23 — the saving
   move (9,12) is a forcing move ranked 8th by v1 (prior
   0.04), starved of visits in 100 sims (recorded as the
   `fl1-g3-ply23` regression xfail in test_regression_v7.py).
   Doubling White sims directly targets visit starvation of
   low-prior forcing moves, with no new tuning knob and no
   distribution-mismatch risk.
3. The threat check (addition 1) complements the sims: when
   Black holds creators, the root choice is restricted to
   safe moves — which INCLUDES forcing counterplay — so
   White is biased toward forcing moves exactly in the
   positions where it is behind in the race.

**What varies by colour (exact, the pre-registered
contract):** the sims count ONLY — 200 for White, 100 for
Black. Verified by a `last_sims_used` diagnostic on the MCTS
(set in `choose_move`) and asserted in tests
(`test_mcts_white.py`): a White `choose_move` on a
search-reaching position runs exactly 200 simulations, a
Black one exactly 100.

**Built-in mechanism check (pre-registered, NOT an eval):**
v10, as White with 200 sims, should play the saving move
(9,12) on the `fl1-g3-ply23` regression position (v7's open
xfail: forcing move starved by a low prior in 100 sims). If
it does, the visit-starvation mechanism is confirmed on a
real position; if it does not, the sims increase does not
fix the known exploration weakness and the design's
mechanism is in doubt before any match is played. This check
runs as a unit test, not a match.

**Pre-eval observation (recorded BEFORE any eval, during
implementation testing — the design stands as pre-registered;
the evals decide).** The built-in mechanism check FAILS: v10
as White with 200 sims still plays the losing block (4,9) on
`fl1-g3-ply23`, not the saving forcing move (9,12). Root
statistics (same position, gradient leaf, seed 0):

| sims | (4,9) visits / q | (9,12) visits / q |
|---|---|---|
| 100 (v7) | 15 / 0.052 | 7 / 0.009 |
| 200 (v10 White) | 26 / 0.045 | 22 / 0.016 |
| 400 (probe) | 113 / 0.012 | 29 / 0.021 |

200 sims close the visit gap (7→22 vs 15→26) but do not flip
the move, and at 400 sims the search converges ON the block
(113 vs 29). The q-values do not distinguish the moves
(0.01–0.05 throughout), so this position's real problem is a
VALUE problem — the search does not see that (9,12) wins —
not pure visit starvation, and more sims alone does not fix
it. (The threat check engages on this position but both
moves are safe, so it cannot decide between them.) The
position is recorded as an xfail test
(`test_v10_white_plays_forcing_save_on_fl1_g3_ply23`, the v7
pattern) so the behaviour stays visible. Following iter-7's
precedent (§1 pre-eval observation), the pre-registered
design is NOT amended on the basis of one diagnostic
position — amending after seeing v10 behave badly on a
known-hard position would be moving the goalposts, and the
evals are the pre-registered decision procedure. The fast
loop (§2) is where a patchable White weakness class would
show up; if the matches show no White gain, the sims
hypothesis is refuted and the next batch should look at the
leaf's value signal for White, not at more search.

**How the White improvement is measured (per-colour splits
are MANDATORY in every match reported here).** White is the
target; the overall match score alone cannot show whether
the colour-conditional change did its job:

- PRIMARY: v10-as-White win rate vs v7-as-Black in the
  100-game h2h (n=50). Reference points: v7-as-White vs v1
  was 0.52 (n=50); v8-as-White vs v1 was 0.29 (n=50).
  A v10-as-White rate > 0.50 against v7-as-Black — the
  reference engine holding Black's first-move edge — is a
  clear White gain. (Caveat: no v7-vs-v7 self-play baseline
  exists, so the White split is judged against the 0.50
  line and the v7/v8 reference splits, not against a
  same-engine baseline.)
- SECONDARY: per-colour splits in every other match — vs v0
  (v7-as-White 0.62, 30W-18L-2D), swap vs v7 (v7's swap
  splits vs v1: Black 0.52, White 0.62), ladder (v7 White
  splits: vs heuristic 7W-3L = 0.70, vs mcts 3W-7L = 0.30).
- v10-as-Black vs v7-as-White is also reported: it should
  track v7's Black game (modulo the threat check), so a
  Black-side regression would show the threat check costing
  attack, not the White change helping.

**Eval seeds (pre-registered, same as iter-7/iter-8):**
h2h vs v7: v10=100 / v7=200; h2h vs v0: v10=100 / v0=200;
ladder seed 100, 20 games/opponent, mcts_sims=100; swap 100
games vs v7, balanced placement, placement_seed 11, tempo
judge (the harness default; recorded in the JSON).

**Fast loop (the finding method, per the batch-4 plan).**
Opponent is v7 (the reference — losses against it are the
ones that matter for the bar); development seeds 500/600,
501/601, ... distinct from the bar battery's 100/200. If an
iteration produces no losses against v7, switch the probe to
v0 (fast; v0's crude play exposed v7's as-Black inversion
and produces White losses). Max 4 iterations (40 games),
stop when an iteration finds no new patchable weakness
class. Each iteration: play 10 games, read every move, find
each game's losing move (solver/deeper-search verified,
never guessed), patch the structural cause, record the
critical position + correct move as a regression test
(`ml/gomoku_ml/regression_positions_v10.json` +
`ml/tests/test_regression_v10.py`).

**Cost estimate (pre-registered, from v7's measured
timings).** v7's 100-game matches took ~3600 s (vs v0/v1)
and ~6444 s (swap); the ladder took 1469 s. v10's White
moves run 2x sims, so matches cost ~1.5x: ~5400 s per
100-game normal match, ~9700 s for the swap, ~2200 s for
the ladder. Total battery ~6 h, run sequentially in the
background, one match at a time. Matches flush after every
game (`selfplay_batch.py`), so a VM replacement loses at
most the in-progress game; a re-run with the same seeds is
deterministic.

**Integration.** New module `ml/gomoku_ml/mcts_white.py`
subclassing `HeuristicMCTS` (the v8 pattern — `mcts_value.py`
is the model); `mcts_heuristic.py` is NOT edited (the
parallel H1 workstream owns the root-decision change there,
and the batch-3 merge lesson says to keep workstreams in
separate modules). `ml/gomoku_ml/scoring.py` gains ONLY the
`ENGINE_VERSIONS[10]` registry entry and a minimal dispatch
hook in `ScoringPlayer.__init__` (a version dict with a
`"white_sims"` key instantiates `WhiteMCTS`).
`baseline.py` is frozen and untouched.

## 2. Loss analysis (fast loop)

(TBD — fast-loop iterations go here as they run; the worked
games ARE the loss analysis.)

## 3. v10 evaluation (FULL VALIDATION BATTERY — final patched v10 only)

(TBD — tactical, h2h vs v7, h2h vs v0, swap vs v7, ladder.)

## 4. Bar check (v10 vs v7)

(TBD.)

## 5. What this means

(TBD.)
