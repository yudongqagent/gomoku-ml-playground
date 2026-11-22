# Iteration 52 — v52: threat-space search (TSS) at the root (batch 13, H2)

**Status: PRE-REGISTERED — design fixed below (§1) BEFORE any eval of
v52. Cost bench (§1.4), mechanism check (§3), fast loop (§4), and
battery (§5) follow.**

**METHOD (carried over).** The fast 10-game loop stays the finding
method: play 10 games, read every move, find each game's losing move
(solver/deeper-search verified, never guessed), patch, regression-test,
repeat — max 4 iterations, or stop when an iteration finds no new
patchable weakness class. The 4-clause bar stays the validation
gate. Fast-loop games use development seeds distinct from the bar
battery's pre-registered seeds (100/200). **Fast loops run against
v0 as well as v27** (iter-17 §5). The full battery runs on the FINAL
version only. The 100-game battery is the verdict — small-sample
smokes mislead; single-position diagnostics cannot validate a
global search change; the v1 walk-back oracle shares the
argmax-of-sum flaw (iter-18) — every nominated save needs playouts
or exact computation.

## 0. The problem

v27 (the bar reference: v18's 2-ply rollout leaf both colours + the
q-gap threshold tie-break τ=0.05) is a 100-sim MCTS whose effective
horizon is ~4-6 plies: the tree itself reaches deeper in forcing
lines, but visit allocation follows the v1 prior, and the 2-ply
rollout leaf only sees two plies past the leaf. Forcing wins that
need 3+ attacker moves — especially wins whose first move is an
open-three creator rather than a four-creator — fall outside that
horizon. v29's VCF-at-root check covers only fours-only sequences
(`evaluate.starts_forced_win`, depth 3): a win whose first move
creates an open three (not a four) is invisible to it, and the
check's defensive half is a verified design flaw (iter-29 §3.4:
preventions picked by prior noise, 3e-6 apart, losing 3/3 while the
search's move wins 3/3) — and VCF-prevention for the v27 family is
now CLOSED in all three forms (v29 prior-pick, v33 search-pick,
v35 search-first; re-confirmed v45, batch 12). The gap this
iteration targets is the OFFENSIVE one: deep forcing wins through
open threes and double threats that neither the MCTS nor the VCF
check sees.

Threat-space search (TSS, Allis) is the classic Gomoku solver
technique for exactly this gap: search ONLY forcing moves, so the
search reaches much deeper than a full-width search, and play the
winning forcing sequence if one exists. It is a genuinely new
mechanism — distinct from v29's VCF-at-root (fours only) and from
the MCTS (all moves).

## 1. Design and pre-registration

**Comparison target: v27** (the bar reference, batch-13 plan:
ladder Elo 636.4). Clause 1 is head-to-head vs v27 (normal rules,
100 games, alternating colours, win rate > 0.50 with Wilson 95%
CI lower bound > 0.50); clause 2 is ladder Elo strictly higher
than v27's 636.4 (identical 40 games/opponent: random, heuristic,
mcts, mcts_sims=100 — batch 11's re-basing); clause 3 is tactical
pass rate >= v27's (20/20 old suite + 29/29 hard suite, on the
CORRECTED suite — batch 9's pre-registered program correction
adds (9,7) to winning-quiet-setup-2); clause 4 is all test suites
green (ml pytest, npm test, tsc). **Secondary head-to-heads, also
pre-registered:** v52 vs v0 (100 games) and a 100-game swap match
vs v27 under the tempo judge (the harness default). A new strength
champion ALSO needs to pass the bar AND win a 100-game direct
challenge vs v18 (champion rule v2, TRACKS.md). Non-transitivity
is recorded prominently.

**Base: v27 exactly** — 100 sims, c_puct 1.5, sharp v1 priors,
rollout leaf both colours, τ=0.05 threshold tie-break, most-
visited root. The TSS is a root hook ONLY: when it finds a forcing
win it plays it, otherwise the engine is exactly v27 (verified by
`test_v52_falls_through_to_v27_when_tss_silent`).

### 1.1 The TSS move set (pre-registered)

The attacker (the side to move at the root) searches ONLY forcing
moves:

* **Four-creating moves** — moves after which the attacker has >= 1
  winning move next turn. Enumerated exactly (a kernel-prefiltered
  enumeration, `mcts_heuristic._four_creating_moves_fast`, verified
  0-mismatch against `evaluate.four_creating_moves` in
  `test_tss.py`): the prefilter is the **v1 kernel** (>= 20000) —
  the v0 kernel's contiguous-run scoring is BLIND to broken-four
  creators (XX_XX; a broken-four creator's v0 score is ~100-200),
  which the oracle test caught (22 mismatches on the first
  v0-only attempt). Each candidate is exact-tested locally (the
  winning moves after the move must involve it). Contract note:
  returns every near move when the player ALREADY holds a winning
  move (its test is "a winning move exists after the move"); the
  fast version returns only genuine creators. TSS never enumerates
  in the pre-existing case — every caller checks for a winning move
  first — so the two agree everywhere TSS looks.
* **True-open-three-creating moves** — moves creating a _XXX_
  (contiguous, both ends empty). Enumerated with the **v0 kernel
  only**: v0 attack >= 1000 (the (3,2) weight is 1000, the (4,1) /
  (4,2) weights are 1000/10000, and no combination of sub-1000
  per-direction values reaches 1000 — the largest is 100), minus
  four-creators. Verified 0-mismatch against a brute-force run-scan
  oracle in `test_tss.py`. NOTE the v0 kernel, not v1: v1's
  window-based `open_ends` counts the cells just OUTSIDE the
  5-window, so v1 scores a genuine open three as a mere THREE
  (400) when the cells two out from the run are occupied — the
  oracle test caught exactly this (an open-three creator with v1
  attack 405, v0 1021). v1 is blind to such threes (a pre-existing
  v1 property — its priors undervalue those moves); v0's run-based
  scoring sees the run directly.
* **Gap-three creators are EXCLUDED** (pre-registered): a gap three
  (XX_X) is killed by occupying the gap, so it rarely forces;
  including them would multiply the branching for near-zero win
  rate.
* **Double threats are NOT a separate move class**: a move creating
  >= 2 threats is in the union above (each threat is a four or an
  open three); the search detects double threats by counting threats
  after the move.

### 1.2 The search (pre-registered)

`mcts_heuristic.tss_winning_move(board, player)` — deterministic
(no RNG):

* An immediate winning move is returned first (standing fact: it IS
  a VCF start but is NOT necessarily a four-creator — the fast test
  checks winning moves first).
* Forcing moves are tried in (v1 prior desc, flat index asc) order
  at the root; in the recursion, four-creators first, then v1
  attack score desc (order affects only speed in the recursion).
  The FIRST move starting a forced win is returned.
* **Depth: 4 attacker forcing moves** (`TSS_DEPTH = 4`). VCF uses
  depth 3 (after the first move); TSS at 4 attacker moves with
  open-three threats reaches ~7-8 plies, genuinely beyond the MCTS
  + 2-ply-rollout horizon (~4-6 plies).
* **Defender replies (exact — every reply must lose for the
  attacker to win):**
  * after a four (exactly 1 winning square): the block is the ONLY
    reply (anything else loses at once);
  * after open three(s): replies are the squares blocking ALL of the
    attacker's open threes (recomputed from the resulting board —
    the batch-11 non-monotonicity lesson: full re-check after every
    move; the attacker may hold a standing three from an earlier
    move in the sequence, e.g. after a counter-four exchange) UNION
    the defender's four-creating moves (counter-threats: a
    counter-four forces the attacker to block instead of
    converting). If the reply set is empty, the attacker wins
    (double threat, no defence). Defender open-three creators are
    NOT replies: they do not stop the attacker converting the
    standing threat next move (open three -> open four -> win), so
    they cannot refute a line;
  * after a defender counter-four: the attacker's reply is the
    forced block (>= 2 defender winning moves refutes the line).
  * if the defender holds a winning move when the attacker is to
    move, the attacker MUST block it (>= 2 refutes the line); the
    block must itself be forcing for the sequence to continue.
* **Limitation (pre-registered):** every attacker move in a sequence
  must be forcing, except forced blocks after defender counter-fours.
  Sequences needing a quiet defensive interlude are out of scope —
  that is what the MCTS is for.
* **TSS is OFFENSIVE only.** There is no defensive prevention check:
  VCF-prevention for the v27 family is CLOSED (v29/v33/v35/v45). If
  the opponent holds a forcing win, the MCTS handles it as v27 does.
  The hook runs AFTER the win/block scans in `choose_move`, so an
  immediate win or a forced block always takes precedence over the
  TSS.

### 1.3 Mechanism check (pre-registered)

**Check A — constructed positions.** 7 positions, all fixed BEFORE
this check runs (the 5 constructed ones live in
`ml/gomoku_ml/tss_mechanism_positions.json`; the 2 VCF controls
come from the tactical suites):

* T1 `tss-double-three-black`: Black (7,6),(8,6),(6,7),(6,8);
  White (3,3),(11,11),(12,3). Black to move. Expected (6,6):
  creates two open threes (horizontal ends (5,6),(9,6); vertical
  ends (6,5),(6,9)), no common block, White has no counter-four.
* T2 `tss-double-three-white`: colours swapped. White to move.
  Expected (6,6).
* T3 `tss-three-sequence-black`: Black (5,7),(6,7),(7,8),(5,9),
  (6,9); White (2,2),(12,2),(2,12),(12,12),(11,3). Black to move.
  Expected {(7,7),(7,9),(8,9),(8,7)} — every winning first move,
  verified by hand:
  * (7,7): open three (horizontal); White blocks an end; (7,9)
    makes two open threes (horizontal (5,9),(6,9),(7,9) +
    vertical (7,7),(7,8),(7,9)) → win.
  * (7,9): open three (horizontal); White blocks an end; (7,7)
    makes two open threes → win.
  * (8,9): diagonal open three (6,7),(7,8),(8,9); White blocks an
    end; (7,9) makes the open four (5,9),(6,9),(7,9),(8,9) → win.
  * (8,7): diagonal open three (6,9),(7,8),(8,7); White blocks an
    end; (7,7) makes the open four (5,7),(6,7),(7,7),(8,7) → win.
  * Refuted (TSS must NOT return them): (4,9) — White blocks
    (7,9), killing the vertical pair; (4,7) — White blocks (7,7),
    killing the double threat; (5,6) — White blocks (8,9),
    breaking the follow-up; (5,10) — White blocks (8,7).
  * NOTE: (7,6)-style first moves are NOT in the set — the
    position was designed so the winning first moves are all
    open-three creators (no four-creator wins at the root), which
    is what makes T3 a TSS-specific test: `starts_forced_win`
    returns False on every expected move (none creates a four).
* T4 `tss-three-sequence-white`: colours swapped. White to move.
  Expected {(7,7),(7,9),(8,9),(8,7)} by symmetry.
* T5 `vcf-1` (old suite) and T6 `vcf-two-four-sequence` (hard
  suite): VCF controls — TSS must return a VCF start (it subsumes
  the offensive VCF check).
* T7 `tss-counter-four-refutation`: T1's Black stones plus White
  (11,11),(12,11),(13,11) — White holds an open three, so the
  counter-four (10,11) (creating an open four) refutes Black's
  double threat. TSS must return None.

**Check A pass criterion: TSS correct on >= 6/7 positions.**
MCTS-miss sub-criterion: on >= 1 of T1–T4, v27 (seed 100) plays a
move OUTSIDE the expected set while TSS returns a move inside it.

**Check B — mined positions.** 50 v27-vs-v27 games (normal rules,
alternating colours, seeds 777/1777 — dev seeds, distinct from the
battery's 100/200). At every turn, run TSS (production budget)
without perturbing the game (TSS consumes no RNG) and record fires,
confirms (TSS move == v27's move), and differs (TSS move != v27's
move). Each differ (first 5, to bound runtime) is verified with 3
playouts from the position: play the TSS move, then v52 on the
attacker's side vs v27 defending (seeds 0/1/2). A differ is
VERIFIED if the attacker wins 3/3 against v27's defence.
**Check B pass criterion: >= 1 verified differ.** If 0 differs
occur in 50 games, the check is INCONCLUSIVE (not failed) — the
mechanism may simply be rare in v27-vs-v27 play — and the fast
loop is the arbiter of the in-game fire rate.

Script: `ml/scripts/check_tss_mechanism.py` (Check A) and
`ml/scripts/mine_tss_positions.py` (Check B).

### 1.4 Cost budget (pre-registered, set from the bench)

The cost is bounded by a DETERMINISTIC node budget (a node count,
not a wall clock — so resumed runs reproduce exactly), never a
time budget. The budget is set from the bench below
(`ml/scripts/bench_tss.py`, run BEFORE this pre-registration was
committed, with a high provisional budget of 100000 nodes so the
bench sees the unconstrained distribution; 100 quiet v7 self-play
positions + 100 creator-rich v27 midgame positions, seed 777):

| bench (n=100) | fired | budget exceeded | time median / p90 / p99 / max / mean | nodes median / p90 / p99 / max / mean |
|---|---|---|---|---|
| quiet (v7 self-play) | 11 (11%) | 0 | 3.5ms / 3230ms / 9917ms / 10578ms / 869ms | 0 / 740 / 2273 / 2273 / 196 |
| creator-rich (v27 midgame) | 5 (5%) | 0 | 111ms / 5896ms / 24991ms / 53189ms / 2221ms | 20 / 1394 / 2763 / 12916 / 464 |

Times measured under 2–3-way CPU contention with the other batch-13
workers (load ~3.6 on ~1.5 CPUs), so they are inflated roughly 2x
over uncontended; node counts are contention-independent and are
what the budget rests on. The distribution is extremely
heavy-tailed: the median call is cheap (3.5ms / 111ms) but the
tail is dominated by "prove no win exists" searches (the mean is
~10x the median). No call hit the 100000-node provisional budget
(max 12916), so the search terminates well within budget — the
budget bounds TIME, not termination.

**Pre-registered budget: TSS_NODE_BUDGET = 3000 nodes** (the
creator-rich p99 node count, 2763, rounded up). It binds only on
~1% of positions (the extreme tail) and caps the worst-case move
time at ~6s uncontended (~10s contended) instead of the observed
53s max. On budget exhaustion the search aborts and returns None
(fall through to the MCTS) — never a partial answer.

Two honest caveats. First, the plan's p99-time-under-1s criterion
is NOT met: the p99 time is ~25s contended (~5s uncontended),
because the per-node cost is ~2ms uncontended, dominated by
kernel calls (a cProfile of 20 creator-rich positions: 49% of
time in `scoring._winning_moves`, 39% in `kernel_v1.attack_v1_both`).
A budget low enough for p99 < 1s (~500 nodes) would abort a
quarter of all searches (p90 = 1394 nodes) and gut the mechanism,
so the budget bounds the tail instead of the median. Second, the
bench above is the PRE-refactor code: profiling drove a refactor
(local winning-move checks on the lines through the just-played
stone, replacing full-board scans wherever the winning moves must
involve that stone; the enumeration unified into `_forcing_sets`
with one v1 call for four-creators — v0 is blind to broken fours —
and one v0 call for open-three creators — v1 is blind to some
open threes; the redundant third kernel call for move ordering
eliminated), which cut the profiled time 11.1s → 9.5s (~15%) with
IDENTICAL node counts (1423/1328 nodes on the same 20 positions —
the refactor changed cost, not the search). A first refactor
attempt (v0-only enumeration) was caught by the oracle test: 22
mismatches, all broken-four creators missed by v0's
contiguous-run scoring. The refactor is verified by 21 tests
(including a new local-vs-full winning-move equivalence test).
The bench re-run on the final code (same seed, same positions;
`research/iterations/tss-bench-final.txt`):

| bench (n=100) | fired | time median / p90 / p99 / max / mean | nodes median / p90 / p99 / max / mean |
|---|---|---|---|
| quiet (v7 self-play) | 11 (11%) | 2.7ms / 2088ms / 6245ms / 6699ms / 550ms | 0 / 740 / 2273 / 2273 / 196 |
| creator-rich (v27 midgame) | 5 (5%) | 86ms / 4229ms / 14769ms / 45214ms / 1673ms | 20 / 1394 / 2763 / 12916 / 464 |

Node counts are IDENTICAL to the pre-refactor run (the refactor
changed cost, not the search — the search explores the same nodes
in the same order); times are ~25–37% lower (mean 869→550ms
quiet, 2221→1673ms creator-rich; the tail positions benefited
most). The 3000-node budget still binds only on ~1% of positions
(p99 = 2763 nodes) and caps the worst case at ~4s uncontended
(the 45214ms max is the 12916-node tail position, contended;
at ~2ms/node uncontended, 3000 nodes ≈ 6s).

For comparability: v29's VCF check measured median 90ms / p90
2.9s / max 36s / mean 1.65s on the same quiet-position bench, and
~7s/move in creator-rich games. v52's profile is comparable
(mean ~0.9–2.2s contended, worst case bounded to ~6s uncontended
by the budget), and the battery is affordable at ~50 min/leg.

### 1.5 Eval seeds (pre-registered)

- Battery h2h vs v27: v52=100 / v27=200; vs v0: v52=100 / v0=200.
- Ladder: seed 100, 40 games/opponent, mcts_sims=100
  (`ml/scripts/eval_version.py --version 52 --ladder-games 40
  --out research/iterations/ladder-v52.json` — the auto-resume
  driver).
- Swap: 100 games vs v27, balanced placement, placement_seed
  11, tempo judge, seeds 100/200.
- Tactical: `eval_tactical_full.py --version 52 --suite both
  --seed 100`. v27's tactical is re-verified first on the
  corrected suite.
- Challenge match (ONLY if v52 passes all 4 clauses): 100
  games vs v18, seeds 100/200, normal rules.
- Fast-loop dev seeds: 1052/2052 (vs v0), 1053/2053 (vs v27)
  — distinct from the battery's 100/200 and from batch-12's
  loop seeds (1045/2045, 1046/2046).
- Harness commands follow iter-27 §1.4 / iter-45 §1.5 exactly
  (`selfplay_batch.py`, `eval_version.py` for the ladder,
  `eval_tactical_full.py` for tactical). All match drivers
  AUTO-RESUME from partial `--out` files — if a run is killed,
  relaunch the SAME command. Per-game seeding
  (`seed + game_index`) with a uniqueness audit on every leg.

### 1.6 What does NOT change

100 sims, c_puct 1.5, sharp v1 priors, the rollout leaf, the
τ=0.05 tie-break, most-visited root. No q-based root decision
(CLOSED). No sims change (more sims is NOT monotonically better).
No prior change. No defensive VCF check (CLOSED). No leaf change.
`baseline.py` stays FROZEN. No kernel re-sync unless v52 passes
the bar.

## 2. Implementation

`ml/gomoku_ml/mcts_heuristic.py` — the TSS block sits between
`_rollout3_leaf_value` and `class HeuristicMCTS`:

* `_winning_moves_through(board, player, move)` — winning moves on
  the 4 lines through the just-played stone (local check; the
  winning moves must involve that stone whenever none pre-existed).
* `_forcing_sets(board, player)` — the enumeration: `(fours,
  threes, a1)`. Four-creators via the v1 kernel (>= 20000,
  catches broken fours) + a local exact test; open-three creators
  via the v0 kernel (>= 1000, minus fours). Two kernel calls, one
  per class — each kernel is blind to the other's patterns.
* `_four_creating_moves_fast`, `open_three_creating_moves`,
  `_all_open_threes`, `_forcing_moves_and_fours`,
  `tss_forcing_moves` — the public enumeration surface.
* `_tss_after_attacker_move(board, attacker, depth_left, budget,
  last_move)` / `_tss_win(board, attacker, depth_left, budget,
  last_defender_move)` — the mutually recursive search. The
  threats check is local to the last attacker move; the defender's
  entry check is local to the last defender move; the attacker's
  entry check stays the FULL scan (a safety net — provably
  redundant, but it converts any proof error into a correct True
  rather than a missed win).
* `tss_winning_move(board, player, depth=4, node_budget=3000,
  stats=None)` — the root entry point. Immediate win first; then
  forcing moves in prior order; first forced win returned.
* `HeuristicMCTS.__init__` gains `tss_root: bool = False`; the
  hook in `choose_move` runs AFTER the `vcf_root` block (i.e.
  after the win/block scans), so an immediate win or forced block
  always takes precedence. When TSS returns None the engine is
  exactly v27 (verified by
  `test_v52_falls_through_to_v27_when_tss_silent`).

`ml/gomoku_ml/scoring.py` — `ENGINE_VERSIONS[52]` = v27 +
`"tss_root": True`; `ScoringPlayer.__init__` passes `tss_root`
through to `HeuristicMCTS`.

Tests: `ml/tests/test_tss.py`, 21 tests, all passing — oracle
equivalences (four-creators vs `evaluate.four_creating_moves`,
open-three creators vs a brute-force run-scan oracle, local vs
full winning-move check), all mechanism positions, suite VCF
positions, quiet-position None, immediate-win-first, depth
accounting, node-budget abort, v52 registration, v52 plays the
TSS win on T1, v52 == v27 fall-through on 5 TSS-silent positions.

## 3. Mechanism check

### Check A — constructed positions: PASSES, 7/7

`ml/scripts/check_tss_mechanism.py` →
`research/iterations/tss-mechanism-check-a.json` (criterion:
TSS correct on >= 6/7):

| position | TSS | correct | v27 (seed 100) | v27 correct |
|---|---|---|---|---|
| T1 tss-double-three-black | (6,6) | yes | (6,6) | yes |
| T2 tss-double-three-white | (6,6) | yes | (6,6) | yes |
| T3 tss-three-sequence-black | (8,7) | yes | (8,7) | yes |
| T4 tss-three-sequence-white | (8,7) | yes | (8,7) | yes |
| T7 tss-counter-four-refutation | None | yes | (10,11) | n/a |
| T5 vcf-1 (control) | (8,3) | yes (VCF start) | (8,3) | yes |
| T6 vcf-two-four-sequence (control) | (9,6) | yes (VCF start) | (9,6) | yes |

The mechanism is CORRECT: TSS finds the double-three win, the
two-move open-three sequence (in both colours), both VCF controls
(it subsumes the offensive VCF check), and correctly returns None
on the counter-four refutation (White's counter-four (10,11)
refutes Black's double threat).

**But the MCTS-miss sub-criterion FAILS: 0/4 on T1–T4.** v27 plays
the correct forcing move on ALL FOUR constructed positions. The
constructed positions are too shallow/simple for the MCTS to miss
— v27's MCTS + 2-ply rollout already sees a double three and a
two-move open-three sequence. So Check A proves TSS is correct
but does NOT demonstrate a case where TSS finds something the
MCTS misses. (On T7, v27 plays (10,11) — the correct defensive
block of White's counter-four; the script marks it
v27_correct=False only because T7's expected set is empty by
definition. That is the right defensive play, not a miss.)

The value of TSS, if any, must live in DEEPER sequences (beyond
the MCTS + 2-ply-rollout horizon) or in positions where the
MCTS's visit allocation dilutes the forcing line — exactly what
Check B (mined real-game positions) arbitrates.

### Check B — mined positions

[To be filled when the mining run finishes. Note: the first
mining attempt was interrupted by a runtime restart at game 9/50
(387 turns, 48 fires, 21 differs) and restarted from scratch —
the mining script has no resume logic. The partial numbers are
consistent with the final run's early games.]

## 4. Fast loop

Iteration 1, leg 1 (vs v0, 10 games, seeds 1052/2052):
**v52 10W-0L-0D** (win rate 1.0000, Wilson 95% CI [0.7225,
1.0000]) — `research/iterations/fastloop-v52-vs-v0-i1.json`.
Clean sweep, no losses to analyze. (v27 beats v0 0.9650, so
this is the expected smoke-test outcome, not evidence of
strength.) Uniqueness: only 6/10 unique games (de-dup 6W-0L-0D,
CI [0.6097, 1.0]) — the known vs-v0 determinism collapse (v0 is
fully deterministic and v52's opening is prior-dominated, so
short games coincide); the verdict stands on the de-dup cut.

Iteration 1, leg 2 (vs v27, 10 games, seeds 1053/2053):
**v52 6W-3L-1D** (win rate 0.6500, Wilson 95% CI [0.3537,
0.8631]) — `research/iterations/fastloop-v52-vs-v27-i1.json`
(recomputed from raw records; 10/10 unique games, de-dup win
rate 0.65). The CI includes 0.50 — 10 games prove nothing; the
battery is the verdict. The run was interrupted by a runtime
restart at 6/10 games and resumed from the partial file
(`resumed_from_game: 6`).

Loss workup (`classify_batch_losses.py` +
`ml/scripts/workup_v52_losses.py`): all 3 losses via open four
(3/3), **0 missed blocks** (in losses: 0), 0 clean missed
preventions in 10 threatened turns. Defence is solved. TSS
fired on **0 v52 turns in all 3 losses** and 0 turns in the
draw (225 plies) — no false positives, and no v52-specific
losing move to patch: the losses are pure v27-equivalent play
(v52 == v27 on every turn of the losses), v27 outplaying the
base engine positionally.

In all 6 wins, TSS fired 2–5x per game, always in the endgame
(last ~7 plies) and always followed — converting already-won
positions. The pattern is "win-more" so far: TSS fires exactly
when v52 is already winning and never when it is losing. Whether
TSS turns any v27-loss into a v52-win — the only thing that
moves the battery — is unanswered by 10 games.

**Fast-loop verdict: NO PATCH.** The mechanism behaves as
designed (fires only on genuine forcing wins, never wrongly,
never in losses). Proceed to the battery.

## 5. Battery

TODO.

## 6. Verdict

TODO.
