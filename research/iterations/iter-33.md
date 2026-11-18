# Iteration 33 — v33: VCF-filtered search on v29 (batch 10, H1)

**Status: PRE-REGISTERED — design fixed below (§1) BEFORE any eval of
v33. Implementation (§2), cost guard (§3), fast loop (§4), and
battery (§5) follow.**

**METHOD (carried over).** The fast 10-game loop stays the finding
method: play 10 games, read every move, find each game's losing move
(solver/deeper-search verified, never guessed — for rollout-leaf
engines, playouts or exact computation, not the v1 walk-back oracle
alone), patch, regression-test, repeat — max 4 iterations, or stop
when an iteration finds no new patchable weakness class. The 4-clause
bar stays the validation gate. Fast-loop games use development seeds
distinct from the bar battery's pre-registered seeds (100/200).
**Fast loops run against v0 as well as v27** (the bar reference is
v27 now, not v7). The full battery runs on the FINAL version only.
The 100-game battery is the verdict — small-sample smokes mislead;
single-position diagnostics cannot validate a global search change.

## 0. The problem

v29 (batch 9, H3: VCF-at-root on v18) passes all 4 clauses of the
heuristic bar, but its defensive VCF check has a VERIFIED FLAW
(iter-29 §3.4): it picks preventions by v1 prior, and preventions
are near-tied on prior BY CONSTRUCTION (they occupy the opponent's
threat squares, so their 1-ply scores are similar). In both White
losses of the v29-vs-v0 fast loop, the defensive check overrode the
search at ply 9 and the override was the losing move:

| Game | Ply | Check played | Search played | Check playout | Search playout |
|---|---|---|---|---|---|
| 1 | 9 | (5,5) | (9,5) | loses 3/3 | WINS 3/3 |
| 3 | 9 | (5,5) | (5,9) | loses 3/3 | WINS 3/3 |

The priors differed by **3e-6** (0.20661768 vs 0.20661458 — noise),
and both candidate preventions left Black with IDENTICAL exact
threat profiles (0 VCF starts, 0 creators, 2 setups, same sets) —
the exact analysis cannot distinguish them either. The search (v18's
rollout-leaf search) CAN distinguish a winning prevention from a
losing one via rollouts; the check bypasses it. The flaw is in the
DESIGN (the selection rule), not the implementation — and the fix is
a different design, not a patch within v29's: **choose among
preventions by SEARCH, not prior** — VCF-filtered search.

The cost of the flaw at battery scale: v29 vs v0 0.6650 (far below
v18's 0.97), White 0.60 — the prior-noise prevention picks bleed
tempo vs phantom threats, especially as White (iter-29 §4.4).

**The v32 caveat (ML batch 2, H3 — iter-32 §4.3/§6).** If the
prevention pool contains NO good prevention, no selection rule fixes
it: in v32's vs-v0 game 4, White held six VCF starts, the saving
move (8,5) leaves four live, and BOTH exact preventions lose 3/3 —
a search-chosen prevention would lose there too. The fast loop MUST
verify the pool contains a winner before claiming the fix (§1.4).

## 1. Design and pre-registration

**Comparison target: v27** (the bar reference, batch-10 plan:
ladder Elo 636.4). Clause 1 is head-to-head vs v27 (normal rules,
100 games, alternating colours, win rate > 0.50 with Wilson 95% CI
lower bound > 0.50); clause 2 is ladder Elo strictly higher than
v27's 636.4 (identical 20 games/opponent: random, heuristic, mcts,
mcts_sims=100); clause 3 is tactical pass rate >= v27's (20/20 old
suite + 29/29 hard suite, on the CORRECTED suite — batch 9 added
(9,7) to winning-quiet-setup-2); clause 4 is all test suites green
(ml pytest, npm test, tsc). **Secondary head-to-heads, also
pre-registered:** v33 vs v0 (100 games) and a 100-game swap match
vs v27 under the tempo judge (the harness default). If — and ONLY
if — v33 passes all 4 clauses, it also plays a 100-game challenge
match vs v18 (seeds 100/200, normal rules) under champion rule v2
(TRACKS.md); the coordinator makes the champion call.

**Base: v29 exactly** — v18's 2-ply rollout leaf in both colours
(`leaf_mode="rollout2_v1_both"`), 100 sims, c_puct 1.5, sharp v1
priors (temperature 2.0), most-visited root child
(`root_decision="visits"`, no tie-break), win/block scans before
the search. The ONLY change is the defensive step of the VCF-at-root
check.

### 1.1 The version — VCF-filtered search (selection rule, pre-registered)

Hook: `HeuristicMCTS.choose_move`, AFTER the existing immediate-win
and single-block scans, BEFORE the MCTS search — the same position
as v29's hook. Gated by a new registry flag `vcf_filtered_search`
(True only for v33; mutually exclusive with `vcf_root` — the two
defensive rules conflict, so setting both is a configuration error).

1. **Offensive VCF (UNCHANGED from v29, step 1).** Among our
   four-creating moves, any that starts a forced win
   (`evaluate.starts_forced_win`, depth 3 — the tactical suite's
   own VCF definition) is played immediately, highest v1 prior
   first (ties: lowest flat move index — deterministic, no RNG),
   skipping the search entirely.
2. **Defensive VCF-filtered search (REPLACES v29's step 2).** If
   any opponent four-creating move starts a forced win for the
   opponent (the SAME gate as v29's step 2 — the per-candidate
   loop runs only under a live VCF threat), compute the FULL
   prevention pool: every candidate move after which the
   opponent has NO VCF-starting move (the SAME exact definition
   as v29's step 2, the same subset + ordering optimisations —
   test only the opponent's ORIGINAL four-creators, VCF starts
   first — but WITHOUT v29's first-success early exit, since the
   search needs the whole pool). Then:
   - **Pool empty** → fall through to the unrestricted search
     (v29's behaviour when no prevention exists).
   - **Pool of exactly 1** → play it directly (provably
     identical to what a 1-child search would pick; saves the
     search).
   - **Pool of ≥ 2** → run the normal 100-sim MCTS with the
     root RESTRICTED to the pool: root children = pool moves
     only, priors = the v1 priors RENORMALISED over the pool
     (a proper prior distribution over the restricted action
     set — PUCT's exploration/exploitation balance stays
     calibrated as in the full search), pick most-visited
     (`root_decision="visits"`, v29/v18's rule, no tie-break).
     All deeper nodes expand normally over all candidates.
3. If neither the offensive step fires nor the gate opens, the
   engine is exactly v18 (same as v29).

**Why this should fix the flaw.** In iter-29 §3.4 the full-pool
search already picked the winning prevention in both games — the
rollout leaf distinguishes preventions the prior and the exact
analysis cannot. Concentrating all 100 sims on 2–3 pool moves
(each gets ~30–50 visits instead of ~1–5 in a full search) gives
the search its best chance to distinguish, exactly where the
decision matters. The filter keeps v29's tactical guarantee (a
prevention is always played when one exists — the pool members
all remove the opponent's VCF starts), while the search chooses
WHICH prevention on rollout evidence instead of prior noise.

**Pre-registered choices (fixed now, not tunable later).**
Renormalised priors over the pool; 100 sims on the restricted
root (the same budget as the full search — concentrating it is
the point); pool-of-1 plays directly; no threshold, no q-based
decision (CLOSED), no sims change, no prior change, no leaf
change, no threat check beyond the VCF filter. `baseline.py`
stays FROZEN. No kernel re-sync unless v33 passes the bar.

### 1.2 What does NOT change

100 sims, c_puct 1.5, sharp v1 priors (temperature 2.0),
most-visited root child, terminal leaf values, the v18 rollout
leaf, the win/block scans, the offensive VCF step. No q-based
root decision (CLOSED). No sims change (more sims is NOT
monotonically better). No prior change. `baseline.py` stays
FROZEN.

### 1.3 Cost guard

The full-pool scan costs MORE than v29's first-success scan, and
the search still runs after it (v29 skips the search when the
check fires) — so v33's per-move cost is >= v29's wherever the
gate opens. The cost profile is measured EARLY (before the
battery) with a v33 bench (`bench_vcf_root.py`, v33 mode: 100
quiet v7 self-play positions, seed 777, same set as v29's bench)
and reported in §3: median, p90, max, mean, fire rate. No design
change is made after any eval on the basis of cost alone — cost
is reported, strength is what the battery measures. The battery
drivers auto-resume, so a slow battery is survivable.

### 1.4 Fast-loop protocol and the pool-contains-a-winner check

Fast-loop dev seeds: **1037/2037 (vs v0), 1038/2038 (vs v27)** —
distinct from the battery's 100/200, from batch 9's 1029–1032,
and from ML batch 2's 1035/1036. 10 games per leg, max 2
iterations; every loss read and worked up with `workup_loss.py`
playouts (seeds 0/1/2) or exact computation.

**Pre-registered reproduction check (runs BEFORE the battery).**
The two iter-29 §3.4 positions are the regression test for the
fix: reconstruct them from the committed fast-loop record
(`~/workspace/b9h3-logs/fast-v29-vs-v0.json`, games 1 and 3 —
v29-vs-v0, seeds 1031/2031) and verify v33 plays the search's
winning prevention at ply 9 — (9,5) in game 1, (5,9) in game 3
— where v29 played the losing (5,5). If v33 does NOT reproduce
the winning prevention at both positions, the design has failed
its own motivating test and the fast loop stops for diagnosis
before any battery leg runs.

**The pool-contains-a-winner check (the v32 caveat).** At every
fast-loop position where the filter fires in a game v33 LOSES,
record the pool and whether it contains a playout-verified
winner (via `targeted_playout.py` on the pool members). If a
loss's pool is all-losers (the v0-g4 case), that loss is NOT a
selection-rule failure — it is recorded as a wrong-pool loss,
and no selection-rule claim is made from it. The fix is claimed
only for losses whose pool contains a winner that v33 failed to
pick (or picked, and won).

### 1.5 Eval seeds (pre-registered)

- Battery h2h vs v27: v33=100 / v27=200; vs v0: v33=100 / v0=200.
- Ladder: seed 100, 20 games/opponent, mcts_sims=100.
- Swap: 100 games vs v27, balanced placement, placement_seed
  11, tempo judge (the harness default).
- Tactical: `eval_tactical_full.py --suite both --seed 100`.
- Challenge (only if all 4 clauses pass): 100 games vs v18,
  seeds 100/200, normal rules.
- Fast-loop dev seeds: 1037/2037 (vs v0), 1038/2038 (vs v27).
- Harness commands follow iter-29 §1.3 exactly
  (`fast_loop_match.py`, `selfplay_batch.py --swap-rule`,
  `eval_version.py` for the ladder — now with auto-resume,
  batch-10 tooling — `eval_tactical_full.py` for tactical).
  The match drivers AUTO-RESUME from partial `--out` files —
  if a run is killed, relaunch the SAME command; the saved
  games stand and the RNG restart is recorded as
  `resumed_from_game`.

## 2. Implementation

Branch `research/batch-10-h1`, worktree
`~/workspace/projects/gomoku-b10-h1/` (from coordinator commit
`a4ed543`). Three files:

1. **`ml/gomoku_ml/mcts_heuristic.py`** —
   `HeuristicMCTS.__init__` gains `vcf_filtered_search: bool =
   False` (mutually exclusive with `vcf_root` — setting both
   raises `ValueError`) and initialises `self._root_pool = None`
   (set by `_search_restricted` for one search, cleared in a
   finally block; initialised in `__init__`, not lazily, so
   subclasses inheriting `_run_one` — `ValueMCTS` — always have
   the attribute). `_vcf_root_move`'s offensive step is factored
   out unchanged into `_vcf_offensive_move` (v29's step 1,
   reused by v33); v29's defensive step is untouched. New:
   `_vcf_prevention_pool` (the FULL pool — None when the gate is
   closed, else the list of candidates after which the opponent
   has no VCF start, prior-desc order; same subset + ordering
   optimisations as v29's step 2, no first-success early exit)
   and `_search_restricted` (the normal 100-sim MCTS with the
   root restricted to the pool). `choose_move` checks the
   filtered path first (offensive → pool → 1-play-direct /
   restricted-search / fall-through), then v29's `vcf_root`
   branch. `_run_one`'s root expansion (`last_move is None`)
   filters priors to `self._root_pool` and renormalises over the
   pool when set; deeper nodes always expand over all candidates.
2. **`ml/gomoku_ml/scoring.py`** — v33 registered in
   `ENGINE_VERSIONS` (`leaf_mode="rollout2_v1_both"`,
   `vcf_filtered_search: True` — and NOT `vcf_root`, by design);
   `ScoringPlayer.__init__` passes the flag through to
   `HeuristicMCTS`.
3. **`ml/tests/test_mcts_heuristic.py`** — 8 new tests (§2.1).

`baseline.py` untouched (frozen).

### 2.1 Tests (new, all passing)

- `test_v33_registered_and_filtered_search_enabled` — registry
  + flags; v29/v18 unchanged.
- `test_vcf_filtered_search_mutually_exclusive_with_vcf_root`.
- `test_vcf_prevention_pool_contains_both_iter29_preventions` —
  the pool at both §3.4 positions is exactly the two preventions
  ([(5,5), (9,5)] / [(5,5), (5,9)]), and every pool member is
  re-verified as a prevention (no opponent VCF start on the
  resulting board). The pool CONTAINS a winner at both positions
  — the v32 caveat check passes here.
- `test_vcf_filtered_search_reproduces_iter29_winning_preventions`
  — the pre-registered reproduction check (§1.4): v33 plays (9,5)
  / (5,9) at the two §3.4 positions; also asserts `_root_pool`
  is cleared after the filtered search (no state leak).
- `test_vcf_filtered_search_fixes_v18_tactical_failures` — v33
  plays a prevention in the expected set at all three of v18's
  hard-suite failures (g26 → (9,4), g79 → (7,9), v1bl-g2 → (7,5)).
- `test_vcf_filtered_search_offensive_still_fires` (vcf-1/vcf-2).
- `test_vcf_filtered_search_quiet_position_falls_through_to_v18`
  — where neither step fires, v33 IS v18 exactly (low sims).
- `test_vcf_filtered_search_does_not_mutate_board`,
  `test_v33_choose_move_deterministic_for_same_seed`.

Targeted run (v33 + vcf + v29 + v18 selection): 20 passed.

### 2.2 Pre-battery verification (smoke, not eval)

- **Reproduction check (§1.4): PASS.** At both iter-29 §3.4
  positions (reconstructed from
  `~/workspace/b9h3-logs/fast-v29-vs-v0.json`), the pool is the
  two preventions and v33 plays the search's winning one —
  (9,5) in game 1, (5,9) in game 3 — where v29 played the losing
  (5,5). Controls at the same positions: v29's check plays (5,5);
  v18's unrestricted search plays (9,5)/(5,9) — the restricted
  search agrees with the full search here, as expected (the
  filter's value-add is the GUARANTEE of a prevention plus the
  concentrated sims, not a different pick at these positions).
  Pool scans 0.22s/0.21s; total v33 move time ~2.1s, the same as
  v18's search — the pool scan is negligible next to the search
  at these positions.
- **Tactical prevention positions: PASS.** g26 pool [(9,4),
  (8,4)] → (9,4); g79 pool [(7,9), (8,7), (6,7), (7,8)] → (7,9);
  v1bl-g2 pool [(6,5), (7,5)] → (7,5). Every pool member is in
  the expected set at all three — the search cannot pick wrong
  there, so the tactical guarantee survives the design change.
  Offensive VCF positions vcf-1/vcf-2: (8,3)/(8,5) as expected.
- **Cost flag (see §3):** the full-pool scan is slow on
  creator-rich tactical positions — g26 31.0s, g79 26.4s,
  v1bl-g2 8.3s (vs v29's early-exit 2.0s/5.0s/2.2s). The §3.4
  positions scan in 0.22s, so the cost is strongly
  position-dependent; the bench (§3) and the fast loop (§4)
  quantify the real-game cost.

## 3. Cost guard

`bench_vcf_root.py --engine v33` (new mode, §2): 100 quiet v7
self-play positions (seed 777 — the SAME set as v29's bench).
**Offensive fires 6/100; gate open 29/100; filter engaged
(non-empty pool) 23/100. Median 89.4ms, p90 9.72s, max 48.6s,
mean 2.73s.**

Comparison with v29's bench on the same set (fired 29/100,
median 90.5ms, p90 2.94s, max 36.4s, mean 1.65s): the gate opens
on exactly the same 29 positions (the gate is unchanged); of
those, 23 have a non-empty pool and 6 have an empty pool (the
full scan is paid, then the unrestricted search runs). The mean
check overhead is 1.65x v29's (2.73s vs 1.65s), the p90 3.3x
(9.7s vs 2.9s) — the full-pool scan has no first-success early
exit, and the tail is heavier. The median is unchanged (89ms):
gate-closed positions dominate it. Two further costs vs v29 are
NOT in this bench: v29 SKIPS the search when the check fires
(29% of positions), while v33 runs the search after the filter
engages (23%) and after an empty pool (6%) — v33 pays check AND
search on most gate-open moves. The fast loop (§4) measures the
real in-game cost; the battery legs will run slower than v29's
(~1.5–2x projected) and the drivers auto-resume, so a slow
battery is survivable.

**No cost optimisation was made.** An exact optimisation of the
shared `four_creating_moves` (a four-creator must sit in a
5-window with ≥3 attacker stones) was considered and REJECTED:
it would change shared `evaluate.py` code (used by v29/v32 and
the tactical suite) for a cost problem the pre-registration says
to report, not act on (§1.3), and v29's battery was feasible at
~5.5h — v33's at ~1.5–2x that is feasible too, just long. If the
fast loop shows a far worse in-game cost than projected, that is
a finding to report to the coordinator, not a silent redesign.

## 4. Fast loop

TODO.

## 5. Battery

TODO.

## 6. Verdict

TODO.
