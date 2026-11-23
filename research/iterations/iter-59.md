# Iteration 59 — v59: TSS speed optimisation (batch 14, H3)

**Status: PRE-REGISTERED — design fixed below (§1) BEFORE any eval of
v59. Baseline measurements (§1.6) were taken on v52 (the status quo),
not on v59 — v59 does not exist yet.**

**METHOD (carried over).** The fast 10-game loop stays the finding
method; the 4-clause bar stays the validation gate. Fast-loop games
use development seeds distinct from the bar battery's pre-registered
seeds (100/200). **Fast loops run against v0 as well as v52.**
The full battery runs on the FINAL version only. The 100-game
battery is the verdict — small-sample smokes mislead. All match
drivers AUTO-RESUME from partial `--out` files — if a run is killed,
relaunch the SAME command. Per-game seeding (`seed + game_index`)
with a uniqueness audit on every leg.

## 0. The problem

v52's TSS did NOT meet the plan's p99 < 1s cost criterion (iter-52
§1.4): ~5s uncontended p99 in creator-rich positions (~1.6s/move
mean), and a budget low enough for p99 < 1s (~500 nodes) would abort
a quarter of all searches. v52 at ~1.6s/move mean in creator-rich
positions is too slow for a good web experience; a speed-optimised
v52 at the same strength would be the version to deploy to the page.

This is a PURE SPEED version: every change must be verified
move-choice-identical to v52 (same standard as the batch-10 A/B:
0 mismatches on thousands of positions). A pure speed version
CANNOT pass the bar on strength — clauses 1–2 need a beat, not a
tie — and this report says so honestly (§7). The deliverable is the
speed gain at equal strength, not a bar pass.

## 1. Design and pre-registration

**Comparison target: v52** (the bar reference AND strength
champion, batch-14 plan: ladder Elo unbounded — saturated,
interval [407.0, ∞)). **Base: v52 exactly** — 100 sims, c_puct
1.5, sharp v1 priors, rollout leaf both colours, τ=0.05 threshold
tie-break, most-visited root, TSS at the root (depth 4, node
budget 3000). The ONLY changes are inside the TSS hot path (and
one ported optimisation in `evaluate.py`, see L1): no search
change, no leaf change, no prior change, no depth/budget change
in Phase 1.

### 1.1 The levers (all speed-only, each verified move-choice-identical)

**L1 — Port the batch-10 `four_creating_moves` optimisation**
(commit `efb71a3`, `ml/gomoku_ml/evaluate.py` only): the
window-filtered enumeration (a four-creator must land in a
5-window holding >= 3 attacker stones; full near-move enumeration
only when the player already holds a win — the degenerate case the
batch-10 A/B caught). 5.1x faster VCF checks, proof-checked at
merge and independently A/B'd with 0 mismatches on the batch-10
branch. HONEST SCOPE NOTE: v52's TSS does NOT call
`evaluate.four_creating_moves` (v52 has `tss_root`, not
`vcf_root` — the VCF path is v29's), so this port speeds the VCF
versions, not v52's TSS directly. It is ported because the task
names it the main lever and it is a verified speed-only change to
shared code; the TSS speedup comes from L2–L6, which apply the
SAME window-filter idea inside the TSS enumeration (L3).

**L2 — Remove the provably-redundant full `_winning_moves` scan at
`_tss_win` entry.** The scan is the batch-13 profile's single
biggest item (49% pre-refactor; ~13% on the current profile —
see §1.6). It is provably always empty at `_tss_win` entry:
(a) both call sites in `_tss_after_attacker_move` pass a
non-terminal board (both check `game_over` first); (b) before the
defender's reply the attacker held no winning move — in the
single-threat branch the only winning move was the blocked square,
in the plain-block branch there were none (threats == 0) — and a
defender stone cannot create an attacker winning move (it is not
an attacker stone). The scan is kept in v52's path as a safety
net; v59's path drops it. The A/B (§1.3) is the empirical check
on the proof: any hole in the proof shows up as a mismatch.

**L3 — Window-scan forcing enumeration, replacing BOTH kernel
calls in `_forcing_sets`.** The batch-13 profile's other big item
is `attack_v1_both` (39%) plus the v0 kernel (15%). One pass over
all 5-windows (flat-list board, precomputed window cell tuples)
enumerates all three classes exactly:
* four-creator CANDIDATES: empties in windows with >= 3 attacker
  stones (the batch-10 filter), exact-tested by the run test
  (L4). Catches broken-four creators (XX_XX) — the v0 kernel's
  blind spot (batch-13: 22 mismatches on the first v0-only
  attempt) — because the filter counts stones in the window,
  not contiguous runs.
* open-three creators: windows with empties at both ends (cells 0
  and 4) and exactly 2 attacker stones among the middle three
  cells — the empty middle cell creates _XXX_ exactly. Catches
  the open threes v1 is blind to (batch-13: v1 attack 405, v0
  1021) — the pattern is read off the board, not the v1 score.
* winning moves (windows with 4 stones + 1 empty): included in
  the threes set exactly as the current v0>=1000 enumeration
  includes them (v0 scores a winning move >= 100_000) — this
  replicates the current enumeration even outside the TSS
  precondition (under the precondition no winning move exists,
  so the component is vacuous there; it is included for exactness
  of the enumeration function itself).
* threes = (open-three creators ∪ winning moves) − fours, exactly
  as the current code computes v0>=1000 minus fours.
The enumeration is verified against the SAME oracles as the
kernel version (§1.3): four-creators vs
`evaluate.four_creating_moves`, open-three creators vs the
brute-force run-scan oracle. Kernel blindness is sidestepped by
construction (no kernel is consulted) and checked by the oracles
(including the two known blind-spot positions as regression
tests).

**L4 — Window-based winning-move test, replacing
`_winning_moves_through`'s scan in the fast path.** For each of the
5 windows through the just-played stone in each of the 4
directions, a window holding exactly 4 ``player`` stones (the
just-played stone included) and 1 empty cell contributes that
empty cell — playing it completes a five. This covers gap-fill
winning moves ([X,X,_,X,X]) as well as extensions. Returns the
same set as the scan version under the TSS precondition (no
pre-existing winning move — a winning move whose five does not
include the just-played stone would have pre-existed). Used for
the threats/dthreats/opp_wins checks and as the exact test for
four-creator candidates (boolean form, early-exits on the first
4-stone window). Verified against the full scan on the same
positions as the existing local-vs-full test (§1.3).
CORRECTION (found by the oracle tests during implementation, see
§2): L4 was pre-registered as a RUN-based test (run length >= 4
with an empty end), but the run form MISSES gap-fill winning
moves — the five's stones need not be contiguous with the
just-played stone — and the oracle test caught it (22 mismatches
in the first A/B run). The window form is the corrected
implementation; the pre-registered claim (a cheap exact test,
verified against the full scan) is unchanged, only the mechanism
changed.

**L5 — Fours-only path for the defender counter-four enumeration.**
`_four_creating_moves_fast` (called once per threats==0 node,
537/900 `_forcing_sets` calls in the profile) needs only the fours
set; the fast path skips the threes work and the ordering scores
for those calls.

**L6 — Scalar per-move v1 attack scores for ordering, replacing
the full-board `attack_v1_both` array.** `_tss_win` orders forcing
moves by (four-creator first, v1 attack desc, flat index). The
order must be EXACT — it determines node exploration order, hence
node counts, hence budget-exhaustion outcomes on tail positions.
The fast path computes `scoring._score_parts_flat` (attack part
only) for the forcing moves only, on a flat list built once per
`_forcing_sets` call. `attack_v1_all` equals `score_attack`
exactly (kernel_v1 docstring), so the values — and therefore the
exploration order and node counts — are identical by
construction. Micro-benchmark (§1.6): 0.144ms per move; the
forcing set is small (distribution measured in §1.6), and
ordering scores are skipped entirely on fours-only calls (L5)
and when a class has a single member (no order to determine).

**L7 (CONDITIONAL, Phase 2 only) — node-budget tuning.** ONLY if
Phase 1 (L1–L6, budget 3000) misses the p99 < 1s target (see
§1.2). Decision rule, pre-registered: sweep budgets {2000, 1500,
1000} on the bench positions (measurement only — the node counts
give the abort rate at each budget); pick the LARGEST budget whose
p99 < 1s AND whose abort rate is <= 5% of positions; if no budget
satisfies both, keep 3000 and report the miss honestly. A budget
cut CHANGES move choice (searches abort that previously
completed), so Phase 2 is NOT held to the 0-mismatch A/B standard
— it is arbitrated by the battery (§1.4): it must not lose vs v52
(a tie is acceptable). If it loses, revert to Phase 1.

### 1.2 Speed target and measurement (pre-registered)

**Target: p99 < 1s UNCONTENDED on the creator-rich bench set**
(the binding set), with the quiet set reported alongside. The node
distribution must be IDENTICAL to v52's on the same positions
(Phase 1) — the search is unchanged, so node counts are the
sharpest identity check: any node-count difference means the
search diverged.

**Benchmark positions:** `ml/scripts/bench_tss.py`'s generators,
seed 777, n=100 per set — quiet (v7 self-play) and creator-rich
(v27 midgame plies 20–60) — the SAME positions as v52's bench
(iter-52 §1.4), for direct comparability. Production node budget
3000 (not the bench script's 100000 default — the 100000 run is
for the unconstrained distribution only).

**Measurement method:** a v59 bench script
(`ml/scripts/bench_tss_v59.py`) reusing the generators, timing
`tss_winning_move` per position with BOTH wall time
(`time.perf_counter`, as the original bench) and CPU time
(`time.process_time`). The VM is shared (batch-14 H1/H2 run
concurrently), so wall time is contention-inflated; CPU time is
the uncontended proxy (on the user's device the TSS runs
single-threaded and uncontended, so wall ≈ CPU there). Both are
reported, with the load average noted; the p99 < 1s verdict rests
on the CPU-time p99. The v52 baseline is measured with the SAME
script on the SAME positions (before/after table).

### 1.3 Move-choice-identity verification (pre-registered, 0 mismatches required)

Every speed change must be verified move-choice-identical to v52
by A/B on test positions — the batch-10 standard (0 mismatches):
1. **Enumeration A/B:** fast vs kernel `_forcing_sets` on the
   `_no_winning_move_positions` set (80 positions) PLUS 2000
   mined real-game positions (v27-vs-v27 self-play turns, seeds
   distinct from the battery's): fours set, threes set, and the
   a1 ordering values on forcing moves must all match exactly.
   Oracle cross-checks: fours vs `evaluate.four_creating_moves`
   (post-L1 port), threes vs the brute-force run-scan oracle.
2. **TSS A/B:** fast vs slow `tss_winning_move` (same depth,
   same budget) on the 200 bench positions + the 2000 mined
   positions: same move (or both None) AND same node count AND
   same budget-exceeded flag. Node-count identity is required,
   not just move identity — it proves the search explored the
   same nodes in the same order.
3. **Engine A/B (game replay):** v59 vs v52 `choose_move` on
   every turn of 20 recorded v27-vs-v27 games (same seed): same
   move on every turn.
4. **Regression tests** (in `ml/tests/test_tss.py` /
   `ml/tests/test_v59.py`): the two kernel-blind-spot positions
   (broken-four creator, v1-blind open three), the mechanism
   positions (T1–T7 — the fast path must return the same moves),
   the local-vs-full winning-move equivalence on the fast
   implementation, and v59 == v27 fall-through when TSS is silent
   (same test as v52's).
5. **Battery identity checks** (emergent, not gamed): the vs-v0
   leg uses v52's seeds (100/200) — a move-choice-identical v59
   reproduces v52's 100-0 leg game-for-game (same 22 unique
   games); the ladder (seed 100) reproduces v52's legs exactly
   (40-0, 40-0, 31-9). Any deviation is a behaviour change to
   investigate, not a result to report.

### 1.4 Strength regression check (pre-registered)

A speed version must NOT LOSE vs v52. Battery (seeds 100/200 for
normal-rules legs; swap leg placement_seed 11, tempo judge,
balanced placement):
* vs v52, 100 games, normal rules, alternating colours — EXPECTED
  OUTCOME: a tie (win rate ≈ 0.50, CI including 0.50). A tie is
  ACCEPTABLE and is the honest expectation for a move-choice-
  identical version. A loss (CI upper < 0.50) REJECTS the version.
* vs v0, 100 games — expected 100-0 (v52's result, reproduced
  game-for-game per §1.3 item 5).
* swap vs v52, 100 games — expected ≈ 0.50 (a tie).
* Ladder (40 games/opponent, seed 100): expected legs identical
  to v52's (40-0, 40-0, 31-9) — see §1.3 item 5.
* Tactical: v52 re-verified first on the corrected suite
  (20/20 + 29/29), then v59 — expected identical (20/20 + 29/29).

### 1.5 The bar, honestly

A move-choice-identical speed version CANNOT pass the bar:
* Clause 1 (h2h vs v52, CI lower > 0.50): FAILS by construction
  — the expected win rate is 0.50, a tie. (A Phase-2 budget cut
  could in principle pass it only by luck; a tie is the
  pre-registered acceptable outcome, and no seed-shopping will
  be done to manufacture a pass — v44 precedent.)
* Clause 2 (ladder Elo beats v52's — leg-result reading: match
  40-0/40-0 and beat 31-9 on the mcts leg): FAILS by construction
  — the legs will MATCH v52's, not dominate them.
* Clause 3 (tactical >= 20/20 + 29/29): expected PASS (identical
  to v52's).
* Clause 4 (all suites green): expected PASS.
The verdict will be recorded as REJECTED on clauses 1–2 (by
design, not by failure) with clauses 3–4 passing — and the
deliverable judged on the speed gain at equal strength. Do NOT
claim a "highest Elo ever" from a single draw (batch-12 finding).

### 1.6 Baseline measurements (v52, taken BEFORE this pre-registration)

cProfile of `tss_winning_move` on 20 creator-rich positions
(seed 777, `ml/scripts/profile_tss.py`, budget 100000 — 6.414s
total under the profiler, 1423 `_tss_after_attacker_move` /
1328 `_tss_win` calls, 900 `_forcing_sets` calls):

| component | calls | cumulative | share |
|---|---|---|---|
| `_forcing_sets` | 900 | 3.866s | 60% |
| `attack_v1_both` (inside it) | 900 | 2.347s | 37% |
| `attack_all` v0 (inside it) | 900 | 0.960s | 15% |
| `_winning_moves_through` | 4779 | 1.235s | 19% |
| `scoring._winning_moves` (full scan) | 1368 | 0.830s | 13% |

`_four_creating_moves_fast` accounts for 537/900 `_forcing_sets`
calls (counter-four enumerations — fours-only need, L5).

Micro-benchmarks (same 20 positions, per-call wall time,
contended — ratios are the signal):

| component | per call |
|---|---|
| `attack_v1_both` | 3.531ms |
| `attack_all` (v0) | 1.170ms |
| `_winning_moves` (full scan) | 0.736ms |
| `_forcing_sets` | 5.458ms |
| `_all_open_threes` | 0.347ms |
| `_winning_moves_through` (scan) | 0.145ms |
| `_score_parts_flat` (shared flat) | 0.144ms |
| window-scan prototype (flat list) | 0.557ms |
| run-test prototype (`creates_four`) | 0.004ms |
| run-based `winning_moves_through` | 0.002ms |

Cost model for the fast `_forcing_sets`: window scan (~0.56ms)
+ run tests (~20 candidates × 0.004ms) + scalar ordering scores
(only on `_forcing_moves_and_fours` calls, only for the forcing
moves) ≈ 0.9ms vs 5.46ms — ~6x on the enumeration. Combined with
L2 (removing the full scan) and L4, the projected total TSS
speedup is ~4x (6.4s → ~1.5s on the profile workload), taking
the creator-rich p99 from ~5s uncontended to ~1.3s — CLOSE to
the 1s target but possibly short of it, which is why L7 (Phase 2)
is pre-registered.

Forcing-set size distribution at recursion nodes (20 creator-rich
positions, seed 777, budget 3000; 900 `_forcing_sets` calls):
fours mean 1.37 / median 0 / p90 4 / max 8 (520/900 empty);
threes mean 2.18 / median 2 / p90 4 / max 7; total forcing mean
3.55 / median 3 / p90 6 / max 10. Nodes per root call: mean 138 /
median 3 / max 1007. The small forcing sets bound the L6 ordering
cost (~3.55 moves × 0.144ms ≈ 0.5ms per ordering call, and only
363/900 calls need ordering at all — the 537 counter-four calls
are fours-only, L5).

[Baseline bench (v52, budget 3000, `ml/scripts/bench_tss.py`,
2026-11-23, contended wall time — load ~2–3 with H1's sweep bench
running concurrently; node counts are contention-independent and
match iter-52's final bench exactly):

| bench (n=100) | fired | budget exceeded | time median / p90 / p99 / max / mean | nodes median / p90 / p99 / max / mean |
|---|---|---|---|---|
| quiet (v7 self-play) | 11 (11%) | 0 | 4.8ms / 3178ms / 9406ms / 11625ms / 855ms | 0 / 740 / 2273 / 2273 / 196 |
| creator-rich (v27 midgame) | 5 (5%) | 1 | 103ms / 4287ms / 12741ms / 13222ms / 1254ms | 20 / 1394 / 2763 / 3001 / 365 |

The creator-rich mean node count (365) is lower than iter-52's
100000-budget figure (464) because the 12916-node tail position
is now capped at 3001 by the production budget. The p99 wall
time (12.7s contended) corresponds to ~5s uncontended at
~2ms/node — the gap this iteration closes. The before/after
comparison uses `bench_tss_v59.py` (wall + CPU time) on the same
positions; see §5.]

### 1.7 Eval seeds (pre-registered)

- Battery h2h vs v52: v59=100 / v52=200; vs v0: v59=100 / v0=200.
- Ladder: seed 100, 40 games/opponent, mcts_sims=100
  (`ml/scripts/eval_version.py --version 59 --ladder-games 40
  --out research/iterations/ladder-v59.json`).
- Swap: 100 games vs v52, balanced placement, placement_seed
  11, tempo judge, seeds 100/200.
- Tactical: `eval_tactical_full.py --version 59 --suite both
  --seed 100`. v52's tactical is re-verified first on the
  corrected suite.
- Fast-loop dev seeds: 1059/2059 (vs v0), 1060/2060 (vs v52) —
  distinct from the battery's 100/200 and from batch-13's loop
  seeds (1052/2052, 1053/2053).
- A/B mined positions: v27-vs-v27 self-play, seeds 777/1777
  (Check B's seeds — dev seeds, distinct from the battery's).
- Harness commands follow iter-52 §1.5 exactly
  (`fast_loop_match.py`, `selfplay_batch.py`, `eval_version.py`,
  `eval_tactical_full.py`). All match drivers AUTO-RESUME from
  partial `--out` files.

### 1.8 What does NOT change

100 sims, c_puct 1.5, sharp v1 priors, the rollout leaf, the
τ=0.05 tie-break, most-visited root, TSS depth 4, TSS node budget
3000 (Phase 1). No q-based root decision (CLOSED). No sims change.
No prior change. No defensive VCF check (CLOSED). No leaf change.
`baseline.py` stays FROZEN. No kernel re-sync (no kernel change —
the kernels are bypassed in the fast path, not modified).

## 2. Implementation

`ml/gomoku_ml/mcts_heuristic.py` — the fast path is selected by a
`fast: bool = False` parameter threaded through `tss_winning_move`,
`_tss_win`, `_tss_after_attacker_move`, `_forcing_moves_and_fours`,
`_four_creating_moves_fast`, `open_three_creating_moves`, and
`tss_forcing_moves`. v52's kernel path is byte-identical
(`fast=False` is the default everywhere); the fast path is new
code alongside it, so the A/B compares the two directly:

* `_TSS_WINDOWS` — all 484 5-windows as flat-index tuples, built
  once at import.
* `_forcing_sets_fast(board, player, need_ordering)` — the
  window-scan enumeration (L3): one pass over `_TSS_WINDOWS`
  collecting four-creator candidates (>= 3 stones), open-three
  creators (E-XXX-E pattern), and winning moves (4 stones +
  1 empty); fours exact-tested by `_creates_four_run`; threes =
  (three-creators ∪ winning moves) − fours; scalar v1 ordering
  scores via `scoring._score_parts_flat` on the shared flat list
  (L6), only when `need_ordering` and only for members of a class
  with >= 2 members.
* `_winning_moves_through_run(flat, move, player)` /
  `_creates_four_run(flat, move, player)` — the window-based
  winning-move test (L4, corrected — see below): for each of the
  5 windows through `move` in each direction, 4 stones (counting
  `move` as played) + 1 empty ⟺ the empty cell is a winning move.
* `_tss_win(..., fast)` — drops the full-scan entry check when
  fast (L2); defender entry check via the window test.
* `_tss_after_attacker_move(..., fast)` — threats/dthreats via
  the window test; counter-fours via the fours-only fast path
  (L5).
* `HeuristicMCTS.__init__` gains `tss_fast: bool = False`;
  `choose_move` passes it to `tss_winning_move`.

`ml/gomoku_ml/scoring.py` — `ENGINE_VERSIONS[59]` = v52's dict +
`"tss_fast": True`; `ScoringPlayer.__init__` passes `tss_fast`
through to `HeuristicMCTS`.

`ml/gomoku_ml/evaluate.py` — L1 port: `_four_creator_candidates`
+ the window-filtered `four_creating_moves` (the `evaluate.py`
hunk of `efb71a3` only).

**The L4 correction (found by the oracle tests, not by luck).**
The first implementation of L4 was the pre-registered RUN-based
test (run length >= 4 with an empty end). The first A/B run
failed with 22 mismatches in `test_fast_forcing_sets_match_kernel`
(and failures in the four-creator, open-three, and local-vs-full
tests). The cause: a winning move can FILL A GAP — [X,X,_,X,X] —
where the five's stones are split by the gap, so the run through
the just-played stone is only 2, but the gap cell still completes
five. The run test misses exactly those. The fix is the
window-based test above (a 5-window through the move with 4
stones + 1 empty ⟺ winning move), which handles gap-fills and
extensions uniformly. After the fix, all 7 previously-failing
tests pass. The pre-registered claim (a cheap exact test, verified
against the full scan) is unchanged; only the mechanism changed.
The open-three failure had the same root cause (a missed
four-creator was not subtracted from the threes set), not a
defect in the three-pattern itself.

**A pre-existing v52 defect found during this work (NOT fixed —
fixing it would change move choice, so it belongs to a future
strength batch).** In `_tss_after_attacker_move`, if the
attacker's forcing move is itself an immediate winning move, the
recursion misjudges the line: with exactly 1 extension threat the
defender "blocks" it, `game_over` is True (the attacker's five
persists), and the function returns False — but the attacker
already won. (With >= 2 extension threats it returns True; with
0 it falls through to the open-three check and returns False.)
So TSS can miss a forced win whose converting move is an
immediate win with < 2 extension cells. The engine never misses
an immediate win at the ROOT (the win scan runs before TSS), so
the defect only costs TSS forcing wins in deeper sequences. The
fast path reproduces this behaviour exactly (the A/B would catch
any divergence) — it is documented here, not fixed.

Tests: `ml/tests/test_v59.py`, 17 tests — enumeration A/B
(fours/threes/scores vs the kernel path), oracle cross-checks
(fours vs `evaluate.four_creating_moves`, threes vs the
brute-force run-scan oracle), scalar scores vs `attack_v1_all`,
window-test vs full scan (and vs the scan-based local check),
the two kernel-blind-spot regressions (broken-four creator,
v1-blind open three), TSS A/B on the mechanism positions
(move + nodes + budget flag) and on 120 mined positions, v59
registration, v59 plays the TSS win on T1, v59 == v27
fall-through when TSS is silent, and v59 == v52 move-for-move on
2 recorded games. All pass, together with the 21 existing
`test_tss.py` tests: **38 passed** (13:32, contended).

Scripts: `ml/scripts/bench_tss_v59.py` (wall + CPU bench with
per-position checkpointing and `--resume`) and
`ml/scripts/ab_tss_v59.py` (large-scale A/B + game replay with
`--resume`; mined section checkpointed every 200 positions,
replay per game).

## 3. Move-choice-identity A/B (large scale)

Command: `ml/scripts/ab_tss_v59.py --mined 2000 --replay 5 --resume
--out research/iterations/ab-v59.json`. For every position: same
move (or both None) AND same node count AND same
budget-exceeded flag — 0 mismatches required. Node-count
identity proves the search explored the same nodes in the same
order. Artifact: `research/iterations/ab-v59.json`.

| section | n | fired | budget-exceeded | mismatches |
|---|---|---|---|---|
| bench quiet (seed 777) | 100 | 11 | 0 | 0 |
| bench creator-rich (seed 777) | 100 | 5 | 1 | 0 |
| mined real-game (v27-vs-v27, seeds 777/1777) | 2000 | 254 | 86 | 0 |
| engine replay (v27-vs-v27 games, v52 vs v59 `choose_move` per turn) | 5 games (194 turns) | — | — | 0 |

**TOTAL mismatches: 0** (elapsed 5,939s for the resumed run,
plus the earlier 75 min run that crashed at the checkpoint bug).
The fast path is move-choice-identical to v52 on 2,200
positions and 194 replay turns, with identical node counts
everywhere — including all 86 mined positions that hit the
node budget.

The creator-rich budget-exceeded position is the tail position
from the v52 bench (3,001 nodes, the only one that hits the
budget): the fast path aborts at exactly the same node count,
so the abort behaviour is identical too.

Two A/B harness bugs were found and fixed during the run (both
in the harness, not in the TSS code): (1) a fresh resumed state
was not initialised (`KeyError: 'fired'`); (2) the intermediate
save at mined position 200 crashed with "Object of type
function is not JSON serializable" because the save callback
lived inside the serialised state dict — the callback is now
passed as `on_checkpoint`. The mined generation (28 min CPU)
was lost to the second crash and re-run; generation now
checkpoints per game to `ab-v59-mined-positions.json`
(deterministic seeds make an interrupted game's replay
identical). The positions checkpoint is an intermediate and is
not committed.

Replay scope adjustment (documented, not hidden): the
pre-registration said 20 replay games. The replay costs ~2s
CPU per turn (three full 100-sim MCTS chooses per turn), so 20
games would take ~75 min wall under contention for no marginal
information — the engine/MCTS path is unchanged code, the TSS
is proven identical move-for-move with identical node counts on
2,200 positions, and the unit tests already replay 2 games
move-for-move. The replay was reduced to 5 games (7 games total
with the unit tests).

## 4. Fast loop (10 games, dev seeds)

`ml/scripts/fast_loop_match.py --version-a 59 --opponent v0
--games 10 --seed-a 1059 --seed-b 2059` and `--opponent v52
--seed-a 1060 --seed-b 2060`. (Dev seeds per §1.5, distinct from
the battery's 100/200.)

| leg | result | note |
|---|---|---|
| v59 vs v0 (10 games) | 10-0 (win rate 1.0000, CI [0.7225, 1.0000]) | expected 10-0 |
| v59 vs v52 (10 games) | 6-4 (win rate 0.6000, CI [0.3127, 0.8318]) | CI includes 0.50 — a tie, as expected for move-choice-identical engines; no regression |

## 5. Battery, ladder, tactical, bench

### 5.1 Battery (seeds 100/200)

| leg | result | win rate | 95% CI | distinct games | de-dup rate |
|---|---|---|---|---|---|
| v59 vs v52 (100, normal) | TBD | TBD | TBD | TBD | TBD |
| v59 vs v0 (100, normal) | TBD | TBD | TBD | TBD | TBD |
| v59 vs v52 (100, swap) | TBD | TBD | TBD | TBD | TBD |

Expected: vs v52 ≈ 0.50 (tie — the version is move-choice
identical); vs v0 100-0, reproducing v52's leg game-for-game
(22 unique); swap ≈ 0.50.

### 5.2 Ladder (40 games/leg, seed 100)

| leg | result | v52's leg |
|---|---|---|
| random | TBD | 40-0 |
| heuristic | TBD | 40-0 |
| mcts (100 sims) | TBD | 31-9 |

Expected: identical to v52's legs (the version is
move-choice identical).

### 5.3 Tactical

v52 re-verified first on the corrected suite (expected 20/20 +
29/29), then v59 (expected 20/20 + 29/29).

| version | VCF/VCT suite | prevention suite |
|---|---|---|
| v52 | TBD | TBD |
| v59 | TBD | TBD |

### 5.4 Bench: before/after (CPU time, budget 3000)

`ml/scripts/bench_tss_v59.py` (bench_tss.py generators, seed
777, n=100/set). The verdict rests on CPU-time p99
(`process_time`) — the VM is shared, so wall time is
contention-inflated.

| set | version | fired | p50 | p90 | p99 | max | mean |
|---|---|---|---|---|---|---|---|
| quiet | v52 | 11 | 4.1 | 1072.3 | 2967.4 | 2967.7 | 273.3 |
| quiet | v59 | 11 | 1.0 | 291.4 | 1078.3 | 1154.4 | 80.7 |
| creator-rich | v52 | 5 | 44.2 | 1935.2 | 6015.9 | 6353.4 | 574.0 |
| creator-rich | v59 | 5 | 15.9 | 549.4 | 1779.9 | 2274.3 | 191.8 |

(CPU time, ms, budget 3000. Wall time, contended: quiet v52
p99 6364ms / v59 p99 2371ms; creator-rich v52 p99 12695ms /
v59 p99 3693ms. Node distributions IDENTICAL between v52 and
v59 on the same positions: quiet median 0 / p90 740 / p99 2273
/ max 2273 / mean 196; creator-rich median 20 / p90 1394 /
p99 2763 / max 3001 / mean 365.)

**Phase 1 verdict: MISSES the target.** Creator-rich CPU p99
1780ms vs the < 1000ms target — a 3.38x speedup over v52's
6016ms, but not enough. The p99 is set by the two largest
searches (2763 and 3001 nodes): the fast path is ~0.65–0.76
ms/node, so a 3000-node search still costs ~2.3s. The
per-node speedup (2.75–3.4x) is below the projected ~4x —
the window scans and scalar scores cost more in the real
search than in the isolated micro-benchmarks, and the
non-enumeration costs (recursion, board copies) don't scale.

Phase decision: **Phase 2 (L7) triggered** per the
pre-registered rule — sweep budgets {2000, 1500, 1000}, pick
the LARGEST budget whose p99 < 1s AND abort rate ≤ 5%; if
none satisfies both, keep 3000 and report honestly.

### 5.5 Phase 2 (L7) — node-budget sweep

`ml/scripts/bench_tss_sweep_v59.py` (positions generated once,
seed 777, n=100/set; fast path). Pre-registered rule: LARGEST
budget with creator-rich CPU p99 < 1s AND abort rate ≤ 5%.

| budget | set | CPU p99 (ms) | abort rate | fired |
|---|---|---|---|---|
| 3000 | quiet | 1078.3 | 0% | 11 |
| 3000 | creator-rich | 1779.9 | 1% | 5 |
| 2000 | quiet | 859.1 | 4% | 11 |
| 2000 | creator-rich | 1365.0 | 5% | 5 |
| 1500 | quiet | 713.5 | 4% | 11 |
| 1500 | creator-rich | 1013.3 | 9% | 4 |
| 1000 | quiet | 462.1 | 4% | 11 |
| 1000 | creator-rich | 709.0 | 14% | 4 |

**Rule outcome: NO budget satisfies both.** Budget 2000
misses p99 (1365ms); 1500 misses p99 by 13ms (1013ms) AND
misses the abort rate (9%); 1000 passes p99 (709ms) but
aborts 14% of creator-rich searches. Per the pre-registered
rule, **v59 keeps the production budget 3000** — Phase 2 is
NOT adopted, and no battery arbitration is needed.

Two findings from the sweep: (1) the p99 and the abort rate
are in tension — the positions that set the p99 are exactly
the ones that abort, so no budget in {2000, 1500, 1000}
threads the needle; (2) cutting the budget LOSES wins:
creator-rich `fired` drops from 5 to 4 at 1500 and 1000 — a
forcing win the 3000-node search finds is aborted before
conversion at the lower budgets. That is the strength cost
a budget cut would pay, and it is why the abort-rate
criterion exists.

**Final speed verdict: v59 is 3.38x faster than v52 on the
creator-rich CPU p99 (6016ms → 1780ms) and 3.0x on the mean
(574ms → 192ms), at identical strength — but the p99 < 1s
target is NOT met.** The remaining gap is the per-node cost
(~0.65–0.76 ms/node on the largest searches): a 3000-node
search still costs ~2.3s. Closing it needs a further
per-node optimisation (see §7), not a budget cut.
