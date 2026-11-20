# Iteration 45 — v45: fix v35's `_leaves_opp_vcf` hole, then measure (batch 12, H1)

**Status: PRE-REGISTERED — design fixed below (§1) BEFORE any eval of
v45. Port (§2), mechanism checks (§3), fast loop (§4), and battery
(§5) follow.**

**METHOD (carried over).** The fast 10-game loop stays the finding
method: play 10 games, read every move, find each game's losing move
(solver/deeper-search verified, never guessed — for rollout-leaf
engines, playouts or exact computation, NOT the v1 walk-back oracle
alone — iter-18 §2.1), patch, regression-test, repeat — max 4
iterations, or stop when an iteration finds no new patchable
weakness class. The 4-clause bar stays the validation gate.
Fast-loop games use development seeds distinct from the bar
battery's pre-registered seeds (100/200). **Fast loops run against
v0 as well as v27** (the bar reference is v27 now, not v7). The
full battery runs on the FINAL version only. The 100-game battery
is the verdict — small-sample smokes mislead; single-position
diagnostics cannot validate a global search change; the v1
walk-back oracle shares the argmax-of-sum flaw (iter-18) — every
nominated save needs playouts or exact computation.

## 0. The problem

v35 (batch 10, H3: v27 + the VCF-at-root check with a SEARCH-FIRST
defensive step) was REJECTED 2/4 (vs v27 0.53, CI [0.4329, 0.6249];
ladder Elo 636.4, an exact tie with v27; tactical 20/20 + 29/29;
vs v0 0.975; swap vs v27 0.62, CI clear of 0.50). Its search-first
gate — `_leaves_opp_vcf`, which decides whether the search's chosen
move is a prevention — re-tests only the opponent's ORIGINAL VCF
starts on the resulting board, by a monotonicity argument: our
move adds our stone, which "can only REMOVE the opponent's forcing
options". Batch 11 (iter-39 §1.1) showed the argument is FALSE:
**VCF starts are NOT monotone under our move.** Occupying a setup
square can CREATE opponent VCF starts by removing one of our own
defensive replies. v35's gate therefore has a hole: it can return
False (claiming a prevention) when the opponent in fact holds a
VCF start after the candidate move — and then the check does not
override, and v35 plays a move that loses by force.

This workstream fixes the hole and measures whether the fixed check
moves strength vs v27. The fix is small and exact (§1.2); the
measurement is the standard battery plus the check's fire rate and
the hole's incidence in real games (§1.4). The VCF-prevention lever
is CLOSED as a strength lever for the v27 family (iter-35 §7: the
check eliminates missed preventions and the remaining losses are
horizon losses) — this version does not reopen it. It repairs a
known defect in a merged-but-unshipped version and measures the
repair. If the fixed check does not move strength, that is the
result, and the hole is closed as a defect regardless.

**Base engine.** v35 exactly as on `research/batch-10` @ `6c6ceb8`
(ported by hand to this branch, §2 — v35 is NOT on master): v27's
search (100 sims, c_puct 1.5, sharp v1 priors, rollout leaf both
colours, τ=0.05 threshold tie-break, most-visited root) + v29's
full VCF-at-root check with the search-first defensive step.

## 1. Design and pre-registration

**Comparison target: v27** (the bar reference, batch-12 plan:
ladder Elo 636.4). Clause 1 is head-to-head vs v27 (normal
rules, 100 games, alternating colours, win rate > 0.50 with
Wilson 95% CI lower bound > 0.50); clause 2 is ladder Elo
strictly higher than v27's 636.4 (**40 games/opponent** — batch
11's re-basing; leg-result reading for saturation: a 40-0 leg
strictly dominates a 39-0-1 leg); clause 3 is tactical pass rate
>= v27's (20/20 old suite + 29/29 hard suite, on the CORRECTED
suite); clause 4 is all test suites green (ml pytest, npm test,
tsc). **Secondary head-to-heads, also pre-registered:** v45 vs v0
(100 games) and a 100-game swap match vs v27 under the tempo
judge (balanced placement, placement_seed 11). **Strength
champion:** v18 (ladder Elo 636.4). Champion rule v2 (TRACKS.md,
direct challenge): IF v45 passes all 4 clauses, it plays a
100-game challenge match vs v18 (seeds 100/200, normal rules) and
takes the title only if it wins that match (win rate > 0.50 with
Wilson 95% CI lower bound > 0.50) AND its ladder Elo beats
636.4 (automatic for a bar-passer). The coordinator makes the
champion call; this log reports the challenge match if it runs.

### 1.1 The hole, precisely

Definitions (from `ml/gomoku_ml/evaluate.py`): a **VCF start** is
a move starting a forced win by consecutive fours
(`starts_forced_win`, depth 3 — the move must create a four, or
be an immediate win; after the defender's forced block the
attacker still forces a win within 3 more of their own moves).
A **four-creator** is a move after which the player has ≥ 1
winning move next turn. A **prevention** is a move after which
the opponent holds NO VCF start.

v35's `_leaves_opp_vcf(board, player, move)`:
1. computes the opponent's four-creators on the CURRENT board
   and, among them, the VCF starts (`opp_vcf`);
2. returns False at once if `opp_vcf` is empty ("our move cannot
   create one — monotonicity");
3. otherwise re-tests ONLY `opp_vcf` on the resulting board
   (`b2`) and returns True iff one survives.

Step 2 and step 3 both assume VCF starts are monotone under our
move. They are not. The mechanism (verified from the
`starts_forced_win` / `forces_win` code, not just asserted): in
the VCF recursion the defender's only replies are forced blocks,
BUT a line is refuted when the defender holds an immediate
winning move after the attacker's four (`winning_moves(b2,
-defender)` is non-empty — the defender wins at once instead of
blocking). Our candidate move can occupy a square that BECOMES
the defender's winning reply later in the forcing sequence — the
five is completed by forced blocks played during the sequence,
so on the current board the square is not yet a winning move
and our candidate is not yet a winning move either — and by
occupying it early we remove the refutation, so the opponent's
forcing sequence now works. A four-creator that was not a VCF
start before becomes one after our move. (Four-creators
themselves ARE monotone — our stone can only remove the
opponent's — and winning moves are monotone too; only the
VCF-start property, which depends on the defender's replies
inside the recursion, is non-monotone.)

The hole therefore has two faces, and the old test misses both:
- **(B1)** the opponent holds NO VCF start on the current board
  (`opp_vcf` empty) but our move creates one — step 2 returns
  False without testing anything;
- **(B2)** the opponent holds VCF starts, our move removes all
  of them, but creates a NEW one from a non-start creator —
  step 3 re-tests only the originals and returns False.

A third, adjacent hole (batch 11's second finding, iter-39 §2):
an immediate winning move IS a VCF start
(`starts_forced_win` returns True for it at once) but is NOT
necessarily a four-creator — so a test that enumerates only
four-creators misses it. The verification audit caught 2 false
candidates from exactly this bug. Any full re-check must test
winning moves first.

### 1.2 The fix (pre-registered)

**v45 = v35 + the fixed search-first defensive VCF check.**
Exactly two changes, both in `ml/gomoku_ml/mcts_heuristic.py`,
gated behind a new flag `vcf_full_recheck` (default False — v35
is v35 exactly when it is off):

1. **The gate fully re-checks VCF after the candidate move**
   (`_leaves_opp_vcf_full`). On the resulting board `b2`: test
   winning moves FIRST (an immediate opponent win on `b2` is a
   VCF start — return True); then test EVERY four-creator on
   `b2` with `starts_forced_win` (depth 3, the same VCF
   definition as the rest of the check) and return True iff any
   is a VCF start. No computation on the current board at all —
   the current board's VCF starts are irrelevant to whether the
   opponent holds one after our move. This closes both (B1) and
   (B2) and the winning-move hole. It is typically CHEAPER than
   the subset test (one board's VCF enumeration instead of two:
   the subset test enumerates VCF starts on the current board
   AND re-tests on `b2`).
2. **The defensive selector runs UNGATED when the fixed gate
   fires** (`_vcf_defensive_move_ungated`). v35's
   `_vcf_defensive_move` early-exits (`return None`) when the
   opponent holds no VCF start on the current board — sound
   under the false monotonicity assumption, fatal without it:
   in case (B1) the fixed gate fires but the gated selector
   returns None, no override happens, and the fix is dead
   exactly where it matters. The ungated selector is the same
   candidate scan (highest-prior candidate after which the
   opponent holds no VCF start, testing all ORIGINAL
   four-creators on each resulting board — that subset IS sound,
   because four-creators are monotone) without the early exit.
   When the gate is open the two selectors return the same move
   (same test set, same first-success rule); they differ only in
   case (B1), where only the ungated one can answer. The ungated
   scan runs ONLY when the fixed gate has fired, so it is never
   wasted on a quiet position.

**What stays the same.** Everything else is v35 exactly: the
search (100 sims, c_puct 1.5, sharp v1 priors, rollout leaf both
colours, τ=0.05 threshold tie-break, most-visited root); the
offensive VCF step (short-circuits before the search, unchanged);
the search-first structure (the defensive check overrides ONLY
a non-prevention — the fast loop's §4.2 design decision is
inherited, not revisited); the prevention definition (0 VCF
starts after, depth-3 VCF); the prior-based prevention selection
(the g32 prior-noise limitation is inherited, not fixed — fixing
it would be a different version). `_vcf_root_move` (v29's path)
is untouched. `baseline.py` is untouched (frozen ladder anchor).

**What the fix is NOT.** Not a new check, not a new trigger, not
a threshold: the check fires in exactly the positions where the
search's move is a non-prevention — v45 only defines
"non-prevention" correctly. Where the old and new gates agree
(the common case), v45 plays EXACTLY v35's move.

### 1.3 The g0-ply-29 instance (pre-registered test position)

Batch 11's verified instance (batch-v33-vs-v27 g0 ply 29, White
to move): after the actual move (0,6) Black holds 0 VCF starts /
0 creators / 1 setup [(5,9)]; occupying (5,9) removes the setup
but leaves Black with FIVE VCF starts. Recomputed here during
implementation (§3.0): the exact count reconciles as 4
four-creator VCF starts [(3,8), (4,8), (2,10), (1,11)] PLUS the
immediate winning move (0,6) — the winning-move hole again.
The position is a forced block in a real game (Black holds the
immediate win (0,6), so `choose_move` blocks it before any
search), so it tests the GATE functions directly, not the full
engine: the fixed gate must return True for (5,9) (and for the
other tested alternatives), with the exact count 5.

**Honesty note (recorded before any eval).** At g0-ply-29 the old
and new gates AGREE on every tested move (both True — 4 original
VCF starts survive (5,9), so the subset test catches it; the
full scan over all candidate moves is in §3.0). The instance
demonstrates the PHENOMENON (a setup-removal leaving VCF starts
live — the mining filter's lesson) but does NOT discriminate the
old test from the new one. The discriminating unit test (old
False, new True) therefore needs a position where the search's
move removes ALL original VCF starts while creating a new one —
found by the scan in §3.0, recorded there with its verification.
If no discriminating position is found by the scan, that itself
is a finding (the hole may be real but vanishingly rare in
reachable positions) and is reported as such — the battery and
the hole-incidence measurement (§1.4) are the verdict either way.

### 1.4 Measurement (pre-registered)

The battery (§1, seeds below) measures strength. Three
additional measurements answer whether the FIX (not just the
version) did anything:

1. **Check fire rate.** The fraction of v45 turns where the
   defensive check overrides the search's move (search move ≠
   played move), per leg, from the instrumented replay (§3.3).
   Also reported: the gate-fire rate (fixed gate True) and the
   no-prevention rate (gate True but no prevention exists).
2. **Hole incidence.** The fraction of v45 turns where the OLD
   gate says "prevention" (False) and the NEW gate says
   "non-prevention" (True) — i.e. turns where v35's check would
   have stayed closed and v45's opens — split by case (B1: no
   VCF start on the current board; B2: originals removed, new
   one created). Measured by the instrumented replay over the
   battery legs: the replay re-runs each recorded game with the
   audit flag on (both gates evaluated at every v45 turn, the
   old answer logged but never affecting play) and verifies the
   replayed moves match the recorded moves exactly. If the hole
   incidence is 0 across all battery legs, the fix changed
   nothing in games and any strength difference from v35 is
   noise — said explicitly in the verdict.
3. **Clean missed preventions + defence audit.** The
   `classify_losses.py` rate for v45 (v35's family runs
   ~0.4–1.3%; v35 itself: 0 in 448 threatened turns) and the
   defence classification (all losses via open four, 0 missed
   blocks). The fix should keep the missed-prevention rate at
   v35's 0 — a non-zero rate would mean the fixed check still
   misses preventions (e.g. via the gated/ungated selector or
   the prior-noise choice), and is reported as a defect.

**Cost.** Mean seconds/move for v45 vs v35's (~1.65s/move mean
on quiet positions, ~7s/move creator-rich) on the battery legs;
the fixed gate is expected to be no more expensive than the
subset test (one board's enumeration instead of two), and the
ungated scan runs only on gate fires.

### 1.5 Eval seeds (pre-registered)

- Battery h2h vs v27: v45=100 / v27=200; vs v0: v45=100 /
  v0=200.
- Ladder: seed 100, 40 games/opponent, mcts_sims=100
  (`ml/scripts/eval_version.py --version 45 --ladder-games 40
  --out research/iterations/ladder-v45.json` — the auto-resume
  driver).
- Swap: 100 games vs v27, balanced placement, placement_seed
  11, tempo judge, seeds 100/200.
- Tactical: `eval_tactical_full.py --version 45 --suite both
  --seed 100`. v27's tactical is re-verified first on the
  corrected suite.
- Challenge match (ONLY if v45 passes all 4 clauses): 100
  games vs v18, seeds 100/200, normal rules.
- Fast-loop dev seeds: 1045/2045 (vs v0), 1046/2046 (vs v27)
  — distinct from the battery's 100/200 and from batch-11's
  loop seeds.
- Harness commands follow iter-27 §1.4 / iter-39 §1.4 exactly
  (`selfplay_batch.py`, `eval_version.py` for the ladder,
  `eval_tactical_full.py` for tactical). All match drivers
  AUTO-RESUME from partial `--out` files — if a run is
  killed, relaunch the SAME command. Per-game seeding
  (`seed + game_index`) with a uniqueness audit on every leg.

## 2. Port (v35 from research/batch-10)

v35 is NOT on master — its code lives only on `research/batch-10`
@ `6c6ceb8`. Ported BY HAND (a cherry-pick of the batch-10 merge
would drag in v33/v34 too):

- `ml/gomoku_ml/mcts_heuristic.py`: `_vcf_root_move` split into
  `_vcf_offensive_move` / `_vcf_defensive_move` (the split is a
  pure refactor — v29's selection unchanged, verified by the v29
  tests below); `_vcf_prevention_pool` (v33's method, ported
  verbatim — unused by any version on this branch, kept for
  completeness); the `vcf_root_search_first` flag + search-first
  flow in `choose_move`; `_leaves_opp_vcf` as a dispatcher over
  `_leaves_opp_vcf_subset` (v35's original, verbatim) and
  `_leaves_opp_vcf_full` (v45's fix). NOT ported: v33's
  `vcf_filtered_search` / `_search_restricted` / `_root_pool`,
  v34's `tie_break_q_tactical_only`, v33's `evaluate.py`
  four-creator optimisation (A/B-identical to master's
  enumeration — iter-33 §2.3 — so the port uses master's).
- `ml/gomoku_ml/scoring.py`: registry entries for v35 (verbatim
  from batch-10, incl. the comment) and v45 (v35's entry +
  `"vcf_full_recheck": True`); the factory passes both new flags
  to `HeuristicMCTS` (defaults keep every other version
  untouched).
- `ml/gomoku_ml/regression_positions_v35.json` and
  `ml/tests/test_regression_v35.py`: copied verbatim from
  batch-10.

**Port verification (COMPLETE — the port matches batch-10
behaviour exactly):**
- **A/B** (`research/iterations/ab_v35_port.py`, same script run
  in both trees with the same Python): v35.choose_move (seed 0)
  plus `_vcf_root_move`, `_vcf_offensive_move`,
  `_vcf_defensive_move`, `_leaves_opp_vcf` on 58 positions (8 v35
  regression + 20 old tactical + 29 hard tactical + g0-ply-29):
  **0 differences / 58** (artifacts
  `research/iterations/ab-v35-port-b12h1.json` and
  `ab-v35-port-b10.json`). The tactical-suite JSONs are
  byte-identical between the trees, so the comparison is on
  identical inputs.
- **Pool** (`research/iterations/pool_ab.py`, 2 positions — the
  full-pool scan has no early exit and takes ~60s/position, so
  it is verified separately, not on all 58): g4-ply44 pool
  [[13,7],[14,7],[13,11],[13,6],[13,9]] and g10-ply44 pool
  [[3,6],[1,9],[0,9],[7,4],[7,3]] — IDENTICAL in both trees.
- **Regression tests** (ported `test_regression_v35.py`): 8
  passed + 2 xfailed (g32, g44 — the documented limitations),
  exactly as on batch-10 (iter-35 §3.1).
- **v29 unaffected by the split**: `test_mcts_heuristic.py -k
  "vcf or v29"`: 6 passed. (Full suite at §6.)

## 3. Mechanism checks

### 3.0 The g0-ply-29 instance, recomputed (pre-eval)

Batch 11's instance, recomputed here from the batch file
(batch-v33-vs-v27 g0, moves[:29], White to move):
- Black holds a WINNING move on the current board: (0,6) — the
  actual move v27 played. The position is a FORCED BLOCK in a
  real game (`choose_move` blocks it before any search), so it
  tests the gate functions directly, not the engine.
- Black holds 15 four-creator VCF starts on the current board
  (plus the immediate win = 16 VCF starts total).
- After (5,9): Black holds exactly **5** VCF starts — 4
  four-creator starts [(3,8), (4,8), (2,10), (1,11)] PLUS the
  immediate win (0,6) — reconciling the mining record's 5 and
  confirming batch 11's second finding (the win is a VCF start
  but NOT a four-creator: (0,6) is not in the four-creator list
  after (5,9)). After the actual (0,6): 0 creators, 0 VCF.
- **Old vs new at g0-ply-29: AGREE on all 98 candidate moves**
  (full scan, `scan_g0_all.py` — the scan output is summarised
  here; the script was a /tmp one-off). Every non-blocking move
  leaves ≥1 surviving ORIGINAL VCF start, so the subset test
  returns True (correctly) everywhere the full test does. The
  instance demonstrates the PHENOMENON (a setup-removal leaving
  VCF starts live) but does NOT discriminate the gates.
- **Targeted scan** (`research/iterations/scan_hole_instances.py`,
  artifact `research/iterations/hole-scan.json`): the mining
  turns with small EXACT VCF counts (v33 g9 ply 60, g38 ply 31,
  g51 ply 32 — all test moves + the actual move) and all 8 v35
  regression positions (played + expected moves): **0
  disagreements**. The late-game mining positions hold 10–15
  original VCF starts, so "remove them all" is nearly
  impossible there; the v35 regression positions hold 1–5, but
  the tested moves are clean preventions or clear
  non-preventions.
- **Broad scan** (`research/iterations/scan_hole_broad.py`,
  artifact `research/iterations/hole-scan-broad.json`): v27's
  ACTUAL (search) move at its last-4 live turns in the three
  normal-rules batch-10 legs. IN PROGRESS (resumable; 2 games
  scanned at last check, 0 disagreements in 7 live turns) — it
  is a 3–4h job at 1 CPU and is paused (SIGSTOP) during the
  port verification and battery, which own the CPU. If it finds
  a live disagreement, the position becomes the discriminating
  regression test (added to `regression_positions_v45.json` +
  `test_regression_v45.py`); if it finds none, that is itself
  the finding, and the hole-incidence measurement (§5) over the
  battery legs is the verdict on a much bigger sample.

### 3.1 Regression tests (v45)

`ml/tests/test_regression_v45.py` (positions in
`ml/gomoku_ml/regression_positions_v45.json`):
- g0-ply-29: the actual (0,6) leaves 0 VCF (both gates False);
  (5,9) leaves exactly 5 VCF starts (fixed gate True; the
  immediate win (0,6) among them and NOT a four-creator);
  the old gate also returns True there (the documented
  agreement, pinned, not hidden).
- Winning-moves-first: a synthetic half-open-four board where
  the opponent's ONLY VCF start after our move is the immediate
  win — the fixed gate returns True; blocking it returns False.
  (First version of this test built an OPEN four — two winning
  squares — and failed on the exact assertion; fixed to a
  half-open four.)
- Ungated ≡ gated when the gate is open (all gate-open v35
  regression positions).
- Flag wiring: v45 all three flags on; v35 full_recheck off;
  v27 none.
- v45 == v35's move on all 8 v35 regression positions (the
  gates agree there, so the fix must not change play).
- Result: **all v45 tests pass** (see §6 for the full-suite
  count, which includes them).

### 3.2 v45 vs v35 (pre-eval behaviour)

v45 and v35 differ ONLY where the gates disagree. The scans
above found no disagreement on any tested position, so v45 is
expected to play v35's move everywhere in the fast loop and
battery unless a live hole instance turns up — in which case
the instrumented replay (§3.3) records it. The fast loop and
battery are therefore ALSO a measurement of the hole's
incidence: any v45-vs-v35 move difference in a game is a hole
instance by construction.

### 3.3 Instrumentation (for §5)

`HeuristicMCTS.vcf_fire_log` (None by default — zero cost in
games): when set to a list by a replay script, each search-first
turn appends `(search_move, gate_fired, old_gate,
defensive_move)`. `vcf_audit_old_gate` additionally evaluates
v35's subset test at each logged turn (purely for the
hole-incidence measurement — never affects play). The replay
script (`research/iterations/replay_hole_incidence.py`,
§5) re-runs each recorded battery game with the audit on and
verifies the replayed moves match the recorded moves exactly
(the v39 `replay_with_fires.py` pattern), then aggregates the
fire rate, the hole incidence (split B1/B2), and the
no-prevention rate.

## 4. Fast loop

Pre-registered seeds (dev seeds, distinct from the battery's
100/200). v45's design is frozen — the fast loop is defect
shake-out, not redesign; no refinement is made unless a genuine
defect is found.

### 4.1 vs v0 (seeds 1045/2045)

**10W-0L-0D** (1.0000, CI [0.7225, 1.0000]) in 533.4s. No losses
to read.

### 4.2 vs v27 (seeds 1046/2046)

**7W-2L-1D** (0.7500, CI [0.4422, 0.9191]) in 1239.9s. Games: g0
W (29 plies), g1 W (28), g2 W (35), g3 draw (225 plies — the
maximal-length draw, ~8 min of the 1240s), g4 W (31), g5 L
(White, 41), g6 W (53), g7 W (48), g8 L (Black, 30), g9 W (54).
Loss classification (`classify_losses.py`): **2 losses, 2 via
open four, 0 missed blocks, 0 clean missed preventions in 11
threatened turns** — defence stays solved; no patchable defect
class. The two losses are "lost to a better attack", not a
check failure: the replay below shows the gate fired twice in
g5 and found NO prevention either time (no-prevention, not a
missed prevention).

### 4.3 Replay (fire rate + hole incidence on the fast loop)

`replay_hole_incidence.py` on the 10-game fast loop vs v27
(artifact `research/iterations/replay-fast-v45-vs-v27.json`):
**233 v45 search turns, 5 gate fires (2.15%), 1 override
(0.43%), 4 no-prevention, 0 hole instances (0 B1, 0 B2), 0
replay mismatches.** The 0 mismatches prove the single-engine
replay reproduces v45's moves exactly; the 0 holes (the old
gate computed on every turn in this run — the pre-optimisation
instrumentation) prove v45 played v35's move on every turn of
all 10 games. The fix changed nothing in the fast loop.

### 4.4 Cost note (ungated selector)

The ungated selector tests four-creators in natural order (no
VCF-starts-first reordering), so `any()` short-circuits later
than v35's gated selector in creator-rich positions — a
suspected cost defect (moves are unaffected: `any()` is
order-independent). Measured: v45 runs ~84s/game vs v27
(excluding the 225-ply draw) vs v35's ~74s/game battery
average — ~14% slower, not the feared 2–3x. Gate fires are
rare (2.15%), so the selector rarely runs. NO ordering fix:
the saving (~5–10% of battery time) is not worth the
refactor + re-verification risk. The battery is expected at
roughly v35's pace (~7–10h for 420 games under contention).

### 4.5 Instrumentation optimisation (post fast-loop)

The audit test (`vcf_audit_old_gate`) originally ran on EVERY
logged turn. A hole is old=False AND new=True by definition,
so the old gate is irrelevant when the fixed gate doesn't
fire — the instrumentation now computes `old` only when
`leaves` is True (mcts_heuristic.py, `choose_move`). The hole
measurement is unaffected; the battery replays run ~2x faster.
(The fast-loop replay above ran with the old instrumentation;
its 0-holes result is a superset — old computed on all 233
turns.)

### 4.6 Verdict on the fast loop

No defect found, no refinement made. Proceed to the battery.

## 5. Battery

TODO.

## 6. Bar check (v45 vs v27)

TODO.

## 7. What this means

TODO.
