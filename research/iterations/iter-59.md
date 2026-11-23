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
