# Iteration 57 — v57: TSS depth/budget sweep (batch 14, H1)

**Status: PRE-REGISTERED — design fixed below (§1) BEFORE any eval of
v57. Cost bench (§2), mechanism check (§3), fast loop (§4), and
battery (§5) follow.**

**METHOD (carried over).** The fast 10-game loop stays the finding
method: play 10 games, read every move, find each game's losing move
(solver/deeper-search verified, never guessed), patch, regression-test,
repeat — max 4 iterations, or stop when an iteration finds no new
patchable weakness class. The 4-clause bar stays the validation
gate. Fast-loop games use development seeds distinct from the bar
battery's pre-registered seeds (100/200). **Fast loops run against
v0 as well as v52** (iter-17 §5). The full battery runs on the FINAL
version only. The 100-game battery is the verdict — small-sample
smokes mislead; single-position diagnostics cannot validate a
global search change; the v1 walk-back oracle shares the
argmax-of-sum flaw (iter-18) — every nominated save needs playouts
or exact computation.

**Runtime note (2026-11-23):** a runtime security notice fired once
while reading the project's own source (a `grep` over
`ml/gomoku_ml/scoring.py` + `research/VERSIONS.md`) — the known
false-positive pattern on this codebase. The flagged output was
skipped and the work continued via the project's own files and
APIs, per the tasking. No other notice recurred.

## 0. The problem

v52's TSS runs at depth 4 attacker forcing moves (~7–8 total plies)
with a deterministic node budget of 3000. Two open questions, and
the sweep exists to answer both before any strength claim:

1. **Does deeper TSS find forcing wins depth-4 misses?** The MCTS
   + 2-ply-rollout horizon is ~4–6 plies; depth-4 TSS already
   reaches ~7–8 plies and finds real wins the MCTS misses (~2.4
   per game, iter-52 §3 Check B, 5/5 verified differs). Forcing
   wins needing 5–6 attacker moves (~9–12 plies) would be
   invisible to both. Do such wins occur in real games?
2. **Does the budget bind first?** At depth 4 the 3000-node budget
   binds on only ~1% of positions (creator-rich p99 = 2763 nodes,
   iter-52 §1.4). A deeper search grows the tree — at fixed
   budget, a depth-5 search can LOSE wins depth-4 finds: the
   budget is shared across root moves, and the deeper tree can
   exhaust it before reaching the winning branch (on exhaustion
   the search aborts and returns None — never a partial answer).
   So deeper depth at the same budget is not a superset: it trades
   gains (deeper wins found) against losses (shallow wins aborted
   away). The sweep must measure both sides of that trade.

A larger budget (6000 nodes) is the other axis: it rescues the
~1% tail aborts at depth 4 and may let depth 5–6 searches complete
where 3000 aborts them. Cost is the constraint throughout: the
per-node cost is ~2ms uncontended, so 6000 nodes ≈ 12s worst-case
per move, and the depth-5/6 trees are much larger than depth-4's.

## 1. Design and pre-registration

**Comparison target: v52** (the bar reference AND strength
champion, batch-13 plan: ladder Elo unbounded — saturated,
interval [407.0, ∞)). Clause 1 is head-to-head vs v52 (normal
rules, 100 games, alternating colours, win rate > 0.50 with Wilson
95% CI lower bound > 0.50); clause 2 is the LEG-RESULT reading
(pre-registered in TRACKS.md): strictly dominate v52's legs
(random 40-0, heuristic 40-0, mcts 31-9) — i.e. match the two
perfect legs and beat 31-9 on the mcts leg; clause 3 is tactical
pass rate >= v52's (20/20 old suite + 29/29 hard suite, on the
CORRECTED suite — batch 9's pre-registered program correction
adds (9,7) to winning-quiet-setup-2); clause 4 is all test suites
green (ml pytest, npm test, tsc). **Secondary head-to-heads, also
pre-registered:** v57 vs v0 (100 games) and a 100-game swap match
vs v52 under the tempo judge (the harness default). A new strength
champion ALSO needs to pass the bar AND win a 100-game direct
challenge vs v52 (champion rule v2, TRACKS.md) — note the
challenge opponent IS the bar reference here, so a bar-passer has
already beaten v52 once; the challenge is a second, independent
100-game match at fresh seeds (300/400 — pre-registered here, not
the battery's 100/200, so it is not a re-run of clause 1).
Non-transitivity is recorded prominently.

**Base: v52 exactly** — 100 sims, c_puct 1.5, sharp v1 priors,
rollout leaf both colours, τ=0.05 threshold tie-break, most-
visited root, TSS at the root. The ONLY change is the TSS depth
and node budget, via two new `HeuristicMCTS` parameters
(`tss_depth`, `tss_node_budget`, defaults = v52's 4/3000, so v52
is bit-identical). The TSS is a root hook ONLY: when it finds a
forcing win it plays it, otherwise the engine is exactly v52
(verified by a fall-through test mirroring
`test_v52_falls_through_to_v27_when_tss_silent`).

### 1.1 Depth counting — stated in total plies (batch-13 lesson)

The ML track's label-depth naming counted plies BEFORE the teacher
leaf, so an "N-ply label" was an (N+2)-ply effective horizon — a
real premise failure (iter-53 §6, ML batch 6). TSS depth is stated
here in TOTAL plies, checked against `_ROLLOUT_PLIES`:

* TSS depth D = D attacker forcing moves. A forcing sequence
  alternates attacker move / defender reply, so D attacker moves
  + (D−1) defender replies = 2D−1 plies to the last attacker move,
  +1 conversion ply = a **2D-ply total horizon**.
* Depth 4 → **7–8 total plies** (v52, as documented in
  `mcts_heuristic.py`: "TSS at 4 attacker moves with open-three
  threats reaches ~7-8 plies").
* Depth 5 → **9–10 total plies**. Depth 6 → **11–12 total plies**.
* Checked against `_ROLLOUT_PLIES = 2`
  (`ml/gomoku_ml/mcts_heuristic.py:757`): the MCTS leaf rolls out
  exactly 2 plies under the teacher's own policy, so the MCTS +
  2-ply-rollout horizon is ~4–6 plies. Depth-4 TSS (7–8 plies)
  already exceeds it; depth 5–6 extend the forcing-win horizon to
  9–12 plies. The sweep asks whether forcing wins live in the
  9–12 ply band that neither the MCTS nor depth-4 TSS sees.

### 1.2 The sweep grid (pre-registered)

Depth × budget, 6 cells; (4, 3000) is the v52 control:

| cell | depth (attacker moves) | total plies | node budget |
|---|---|---|---|
| (4, 3000) | 4 | 7–8 | 3000 — v52 control |
| (4, 6000) | 4 | 7–8 | 6000 — budget only |
| (5, 3000) | 5 | 9–10 | 3000 |
| (5, 6000) | 5 | 9–10 | 6000 |
| (6, 3000) | 6 | 11–12 | 3000 |
| (6, 6000) | 6 | 11–12 | 6000 |

No depth-7 cell: the tree grows roughly exponentially in depth,
and depth 6 already tests whether the budget binds before depth
pays. No budget-12000 cell: 6000 nodes ≈ 12s worst-case per move
uncontended is already past any deployable bound; if 6000 binds,
the answer is "the budget binds first", not a bigger budget.

### 1.3 Kernel blindness (batch 13) — no enumeration change

The sweep varies ONLY depth and budget. The TSS enumeration is
unchanged: four-creators via the **v1 kernel** (>= 20000 — the v0
kernel's contiguous-run scoring is BLIND to broken-four creators,
XX_XX) + a local exact test, and true-open-three creators via the
**v0 kernel** (>= 1000, minus four-creators — v1's window-based
`open_ends` is BLIND to some open threes, scoring a genuine open
three as a mere THREE (400) when the cells two out from the run
are occupied). Each kernel enumerates one class; neither alone
suffices. Verified by the EXISTING oracle tests in
`ml/tests/test_tss.py` (four-creators vs
`evaluate.four_creating_moves`, open-three creators vs a
brute-force run-scan oracle, local vs full winning-move check) —
re-run as part of the v57 suite — plus new differential tests
(§1.6) proving the sweep cells use the same enumeration as v52
(the cells differ only in depth/budget, never in which moves are
enumerated).

### 1.4 Cost bench (pre-registered — measured FIRST, before any strength eval)

`ml/scripts/bench_tss_sweep.py` (new): generates the SAME 100
quiet + 100 creator-rich positions as iter-52's bench (same
generators, same seed 777 — so the (4, 3000) cell reproduces
iter-52 §1.4's numbers and the other cells are directly
comparable), then benches all 6 cells on those positions. Each
cell is benched at its OWN production budget (3000/6000) — NOT
the 100000-node provisional budget iter-52 used: at depth 5–6
the unconstrained tree may explode, the budget is the
termination guarantee, and the **budget_exceeded rate is the
decision-relevant measurement** (it answers "does the budget
bind first?" directly). Reports per cell: fire rate,
budget_exceeded rate, time distribution (median/p90/p99/max/mean),
node distribution (median/p90/p99/max/mean). Times are measured
under whatever CPU contention the batch workers impose — node
counts are contention-independent and are what the budget rests
on (iter-52 §1.4).

### 1.5 Mechanism check (pre-registered) — differential mining

**Check M — mined positions, differential vs the (4, 3000)
control.** 50 v27-vs-v27 games (normal rules, alternating colours,
seeds 777/1777 — the SAME setup as iter-52's Check B, so the
positions are directly comparable; v27-vs-v27 games are the probe
set because they are TSS-naive — richer in missed forcing wins
than v52-vs-v52 games, where v52 already takes the shallow wins).
At every turn, run TSS at the control (4, 3000) AND at each of the
other 5 cells (TSS consumes no RNG, so the probes do not perturb
the game) and record, per cell:

* **gain**: the cell fires (returns a move) where the control
  returns None — a forcing win only the deeper/bigger search sees;
* **loss**: the control fires where the cell returns None — a
  win the cell LOSES (only possible via budget abort: in the
  unbounded case depth-5 ⊇ depth-4, since every depth-4 sequence
  is explored with at least as much depth_left at depth 5 — so
  every loss must coincide with `budget_exceeded=True`, and the
  script asserts this);
* **abort**: the cell's `budget_exceeded` flag (recorded for
  every call, whether or not it is a loss).

Each gain is a PROVEN forcing win by the TSS itself (the search
is exact over defender replies), but — following iter-52's Check
B — the first 5 gains per cell are verified end-to-end with 3
playouts each: play the cell's move, then the cell-engine
attacking vs v52 defending (seeds 0/1/2). A gain is VERIFIED if
the attacker wins 3/3 (the playout checks the follow-through: the
cell-engine must re-find the forcing line on later turns). The
cell-engine for playouts is a v52 engine with `tss_depth` /
`tss_node_budget` overridden to the cell (the plumbing in §1
makes this a plain attribute set on the constructed MCTS).

**Check M pass criterion: >= 1 verified gain for at least one
cell.** If 0 gains occur in 50 games for every cell, the check is
INCONCLUSIVE on mechanism (not failed) — deeper wins may simply
be rarer than 1 in ~2500 turns — and the selection rule's
fallback (§1.6) applies. The mining checkpoints after every game
(same pattern as `mine_tss_positions.py`) so VM replacements lose
nothing; the mining is deterministic and a resume reproduces the
uninterrupted run exactly.

Script: `ml/scripts/mine_tss_sweep.py` (new).

### 1.6 Selection rule (pre-registered) — how the sweep winner is chosen

Per cell, from Check M: `net = (# verified gains) − (# losses)`.
Losses need no playout verification — the control's move is a
TSS-proven forcing win, so a loss is a real missed win by
construction.

1. A cell is ELIGIBLE if it has >= 1 verified gain AND net > 0.
2. Among eligible cells, the winner is the one with the highest
   net; ties broken by the lowest creator-rich mean time on the
   §1.4 bench.
3. **Cost gate:** the winner's creator-rich mean time must be <=
   3x v52's (4, 3000) creator-rich mean on the same bench. If the
   top cell fails the gate, fall to the next eligible cell (by
   net, then time) that passes it.
4. **Fallback — if NO cell is eligible:** the sweep finds no
   deeper-win mechanism. v57 = **(4, 6000)**, the budget-only
   cell — the only remaining mechanism (rescuing the ~1% tail
   aborts at depth 4) — and the report says so honestly. The
   depth lever is then CLOSED (no verified deeper win in 50
   mined games) and only the budget question remains open.
5. The fast loop and the full battery run on the sweep winner
   ONLY (a 100-game battery per cell would be ~30+ hours of CPU).

**AMENDMENT (2026-11-23, made BEFORE the verification phase ran —
the mining was at 46/50 games when the defect below was recognised
from the interim counts; the verification playouts had not started
and no verified-gain count existed yet).** The net formula as
written compares a CAPPED count against an UNCAPPED count:
`# verified gains` counts only the first 5 gains per cell
(`MAX_VERIFY = 5` in the mining script — a runtime bound on the
playout verification), while `# losses` counts every loss in all
50 games. That asymmetry is a defect, identifiable from the rule
text alone: it systematically punishes cells for finding many
wins (only 5 can ever count) while counting all their losses, so
a cell with 22 gains and 5 losses (raw net +17) would score net
<= 0 and be declared ineligible, while a cell with 4 gains and 0
losses (raw net +4) would win — the rule would pick the cell
with LESS mechanism evidence. That defeats the rule's own
purpose, and the pre-registration itself states the correcting
principle (§1.5): "Each gain is a PROVEN forcing win by the TSS
itself (the search is exact over defender replies)" — the
playouts are a spot-check that the follow-through works, not the
sole evidence for a gain. The amended rule, applied uniformly to
all cells:

* `net = (total gains) − (total losses)` — both complete counts
  from the mining (gains are TSS-proven wins, losses are
  TSS-proven missed wins).
* Eligibility: >= 1 playout-verified gain among the first 5
  checked (the soundness gate — the mechanism's follow-through
  works in real play) AND net > 0.
* Winner: highest net; tie-break lowest creator-rich mean time;
  cost gate ≤3x; fallback (4, 6000) — all unchanged.
* The LITERAL-rule result (net with capped verified gains) is
  ALSO reported in §3 as a sensitivity analysis, so the
  coordinator can see both readings and adjudicate. If the two
  readings disagree on the winner, the report says so
  explicitly and the amended-rule winner is the version run
  through the battery, with the disagreement flagged.

New tests (`ml/tests/test_tss_sweep.py`): the sweep cells use the
same enumeration as v52 (differential: `tss_forcing_moves`
identical — it takes no depth/budget); depth-5 ⊇ depth-4 on the
T1–T4 mechanism positions at a high budget (no budget to bind);
a budget-abort-loses-a-win case (a position where depth-4/3000
fires and depth-5 at a TINY budget aborts — proving the loss
mechanism exists); v57 registration; v57 plays the TSS win on
T1; v57 == v52 fall-through on TSS-silent positions.

### 1.7 Eval seeds (pre-registered)

- Fast loop: vs v0 and vs v52, 10 games each, seeds 1001/2001
  (the tasking's seeds — distinct from the battery's 100/200 and
  from batch-13's loop seeds).
- Battery h2h vs v52: v57=100 / v52=200; vs v0: v57=100 / v0=200.
- Ladder: seed 100, 40 games/opponent, mcts_sims=100
  (`ml/scripts/eval_version.py --version 57 --ladder-games 40
  --out research/iterations/ladder-v57.json` — the auto-resume
  driver).
- Swap: 100 games vs v52, balanced placement, placement_seed
  11, tempo judge, seeds 100/200.
- Tactical: `eval_tactical_full.py --version 57 --suite both
  --seed 100`. v52's tactical is re-verified first on the
  corrected suite.
- Challenge match (ONLY if v57 passes all 4 clauses): 100
  games vs v52, seeds 300/400 (fresh — NOT the battery's
  100/200), normal rules.
- Harness commands follow iter-52 §1.5 exactly
  (`fast_loop_match.py`, `selfplay_batch.py`, `eval_version.py`
  for the ladder, `eval_tactical_full.py` for tactical). All
  match drivers AUTO-RESUME from partial `--out` files — if a
  run is killed, relaunch the SAME command. Per-game seeding
  (`seed + game_index`) with a uniqueness audit on every leg.

### 1.8 What does NOT change

100 sims, c_puct 1.5, sharp v1 priors, the rollout leaf, the
τ=0.05 tie-break, most-visited root. No q-based root decision
(CLOSED). No sims change (more sims is NOT monotonically better).
No prior change. No defensive VCF check (CLOSED). No leaf change.
No enumeration change (§1.3). `baseline.py` stays FROZEN. No
kernel re-sync unless v57 passes the bar.

## 2. Cost bench

`ml/scripts/bench_tss_sweep.py` →
`research/iterations/tss-sweep-bench.json` (seed 777, 100 quiet +
100 creator-rich positions — the same positions as iter-52's bench;
the (4, 3000) cell reproduces iter-52 §1.4's node counts EXACTLY:
quiet median 0 / p90 740 / p99 2273 / mean 196; creator-rich median
20 / p90 1394 / p99 2763 / mean 365 — so the bench positions are the
same and the cells are directly comparable). Times measured under
CPU contention with the other batch-14 workers (and, for the first
segment, my own pytest run), so they are inflated ~2x over
uncontended; node counts are contention-independent and are what
the budget rests on. The first bench attempt was SIGTERM'd by a VM
replacement after ~30 min (2 cells done); the resume reproduced the
run exactly (per-cell resume from the JSON).

| cell | quiet fired / exc | rich fired / exc | rich time mean (×v52) | rich nodes mean / p90 |
|---|---|---|---|---|
| (4, 3000) = v52 | 11 / 0 (0%) | 5 / 1 (1%) | 1533ms (1.0x) | 365 / 1394 |
| (4, 6000) | 11 / 0 (0%) | 5 / 1 (1%) | 1387ms (0.9x) | 395 / 1394 |
| (5, 3000) | 10 / 10 (10%) | 6 / 19 (19%) | 2484ms (1.6x) | 813 / 3001 |
| (5, 6000) | 11 / 4 (4%) | 7 / 8 (8%) | 3792ms (2.5x) | 1139 / 4246 |
| (6, 3000) | 9 / 24 (24%) | 6 / 28 (28%) | 3625ms (2.4x) | 1047 / 3001 |
| (6, 6000) | 10 / 14 (14%) | 9 / 22 (22%) | 6248ms (4.1x) | 1823 / 6001 |

Full per-cell distributions (median/p90/p99/max time and nodes)
are in the JSON. Headline findings:

1. **The budget binds first — decisively.** At depth 5 the 3000-
   node budget binds on 10% (quiet) / 19% (creator-rich) of
   positions, vs 0%/1% at depth 4; at depth 6 it binds on 24%/
   28%. At depth 5/3000 and depth 6/3000 the creator-rich node
   p90 IS the budget (3001) — the budget, not the search, is the
   binding constraint. Even at 6000 nodes, depth 6 aborts 14%/
   22% of searches.
2. **The budget-only cell (4, 6000) is a near-no-op on the bench:**
   identical fires (11/5) and identical aborts (0/1) to v52. The
   single creator-rich position that exceeds at 3000 needs 12916
   nodes unconstrained (iter-52 §1.4), so it exceeds at 6000 too
   (max 6001). The only difference is nodes mean 365→395 (the
   aborted search runs longer before dying). If the sweep falls
   back to (4, 6000), expect v57 ≈ v52 in games.
3. **Depth 5 trades losses against gains already on the bench:**
   quiet fires DROP 11→10 at (5, 3000) — a loss (a win the
   control finds that the deeper search aborts away) — while
   creator-rich fires rise 5→6/7 — possible gains. The mining
   (§3) arbitrates the net.
4. **Depth 6 looks dominated:** fires drop on quiet at both
   budgets (11→9/10) while aborts explode; the only bright spot
   is creator-rich fires 9 at (6, 6000) — but at 4.1x the mean
   cost.
5. **Cost gate (§1.6, ≤3x v52's creator-rich mean):** (5, 6000)
   at 2.5x PASSES; (6, 3000) at 2.4x passes; **(6, 6000) at 4.1x
   FAILS** the gate — it cannot be the sweep winner even if it
   leads on net verified gains (the gate falls through to the
   next eligible cell).

## 3. Mechanism check

**Cross-worker note (received mid-run from H2/v58, 2026-11-23):**
H2's diagnosis independently found (1) depth-6 TSS aborts far
more than depth-4: 17.5% abort rate in mining positions, 55% in
losing positions at 10k nodes, and 5 positions still aborting
even at 50k nodes — i.e. the node budget must scale with depth,
and a depth-6 config at budget 3000 aborts too often to be
meaningful; (2) H2's M3 tested White turns only and found ZERO
additional White forcing wins at depth 6 (0/55 completed
searches). Both findings are consistent with this sweep's own
bench (§2: depth 6 aborts 24–28% at budget 3000) and are
interpretation context only — they do not change the
pre-registered grid or selection rule (the grid already pairs
depth 6 with budget 6000, and the selection rule's net + cost
gate already penalise abort-heavy cells). When §3 is written up,
the gains will be broken down by colour (each mined gain records
`to_move`): if depth-5/6 gains concentrate on Black turns, that
aligns with H2's M3 and the depth benefit, if any, comes through
a different population than White-turn detection.

*(results to be filled after Check M runs)*

## 4. Fast loop

*(to be filled)*

## 5. Battery

*(to be filled)*

## 6. Verdict

*(to be filled)*
