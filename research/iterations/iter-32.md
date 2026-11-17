# Iteration 32 — v32: VCF at root for the value-MCTS engine (ML batch 2, H3)

**Status: PRE-REGISTERED — design fixed below (§1) BEFORE any eval of
v32. Port (§2), cost guard (§3), fast loop (§4), and battery (§5)
follow.**

**METHOD (carried over).** The fast 10-game loop stays the finding
method: play 10 games, read every move, find each game's losing move
(solver/deeper-search verified, never guessed), patch, regression-test,
repeat — max 4 iterations, or stop when an iteration finds no new
patchable weakness class. The 4-clause ML bar stays the validation
gate. Fast-loop games use development seeds distinct from the bar
battery's pre-registered seeds (100/200). **Fast loops run against
v0 as well as v8** (iter-17 §5, carried over into the ML track).
The full battery runs on the FINAL version only. The 100-game
battery is the verdict — small-sample smokes mislead.

## 0. The problem

v24's finding (iter-24 §5, ML batch 1): the 100-sim search horizon
is a SHARED limit — v7 makes the same miss as v24 (White's i10
instead of d9, vs-v0 game 1 ply 19; d9 saves 2/3 in same-engine
playouts). The miss is a preventive move 3–4 plies deep that the
100-sim search cannot propagate to the root. Batch 9's v29 proved
a VCF-at-root check fixes tactical failures deterministically for
the rollout-leaf engine (all three of v18's q-compression failures,
iter-29 §3.1) and PASSED all 4 clauses of the heuristic bar — but
v29's DEFENSIVE check is a VERIFIED FLAW: it picks preventions by
prior noise (the check's move lost 3/3 while the search's won 3/3,
iter-29 §3.4), costing tempo vs phantom threats especially as
White (vs v0 0.6650, far below v18's 0.97).

This workstream applies the search-side lever to the ML engine
(v8 base). Note the asymmetry with v29: v8 is ALREADY 29/29 on
the tactical suite (re-verified by the coordinator on the
CORRECTED suite at seed 100), so the tactical gain here is nil —
the value is in GAME STRENGTH: catching horizon misses (forced
wins the 100-sim search misses) that cost real games.

## 1. Design and pre-registration

**Comparison target: v8** (the ML champion and bar reference:
ladder Elo 0.0, vs v0 0.545, tactical 20/20 + 29/29, checkpoint
`ml/runs/b3-h2/value-net.pt`, ResNet 16ch/2 blocks, α=0.25).
The ML bar's clause 1 is head-to-head vs v8 (normal rules, 100
games, alternating colours, win rate > 0.50 with Wilson 95% CI
lower bound > 0.50), clause 2 is ladder Elo strictly higher than
v8's 0.0 (identical 20 games/opponent: random, heuristic, mcts,
mcts_sims=100), clause 3 is tactical ≥ v8's (old suite 20/20,
hard suite 29/29, run at `--seed 100` on the CORRECTED suite),
clause 4 is all test suites green (ml pytest, npm test, tsc).
**Secondary head-to-heads, also pre-registered:** v32 vs v7 (100
games), v32 vs v0 (100 games), and a 100-game swap match vs v7
under the tempo judge (the harness default). Non-transitivity is
recorded prominently.

### 1.1 The version

**v32 = v8's ValueMCTS + value leaf, with an OFFENSIVE-ONLY VCF
check at the root, played immediately when found.** Ported from
v29's `_vcf_root_move` (on master via the batch-9 merge), NOT
re-derived: the offensive step is v29's step 1 exactly — among
our four-creating moves, any that starts a forced win
(`evaluate.starts_forced_win`, depth 3 — the tactical suite's
own VCF definition) is played immediately, highest v1 prior
first (ties: lowest flat move index — deterministic, no RNG),
skipping the search entirely. The hook sits in
`ValueMCTS.choose_move` AFTER the existing immediate-win and
single-block scans, BEFORE the MCTS search — the same position
as v29's hook in `HeuristicMCTS.choose_move` (the hook must be
added to `ValueMCTS.choose_move` explicitly: it overrides the
parent's `choose_move` entirely, so v29's hook does not reach
it). If the check does not fire, the engine is exactly v8.

**The defensive check is OFF — pre-registered default.** v29's
defensive check is a verified flaw (iter-29 §3.4): preventions
are near-tied on prior by construction (they occupy the
opponent's threat squares), so the highest-prior pick is a
coin flip that loses tempo vs phantom threats. The ONLY way a
defensive check enters v32 is the batch-10 direction from
iter-29: a **VCF-filtered search** — when the opponent holds a
VCF start, restrict the root pool to prevention moves (moves
after which the opponent has NO VCF start) and let the SEARCH
choose among them by visits, exactly as v8's existing 2-ply
threat check restricts the pool to safe moves. Inclusion rule
(pre-registered, high bar): the fast loop must produce **≥2
verified losses where a VCF prevention saves** (workup_loss
playouts, seeds 0/1/2) **AND** a VCF-filtered-search prototype
must reproduce the saving move at those positions. If either
condition fails, the defensive check stays OFF and v32 ships
offensive-only. No prior-based defensive override is ever
added — that design is CLOSED by v29's evidence.

### 1.2 The d9 diagnostic (pre-registered, diagnostic only)

v24's verified miss (White i10 instead of d9, vs-v0 game 1 ply
19) is a DEFENSIVE prevention, not an offensive VCF start — so
the offensive-only check is NOT expected to catch it. The
position is tested explicitly as a diagnostic (does v32 play
d9? does the defensive check, run as a diagnostic with
`defensive=True`, play d9?), and the result is recorded in §4
— but it is NOT a design trigger by itself: one position
cannot override the pre-registered inclusion rule (§1.1), and
v29's evidence says a prior-based defensive override would
cost more than it saves. The diagnostic answers "does the
offensive check catch horizon misses of this class?" — the
honest prior is NO for defensive preventions, YES for missed
offensive forced wins (the class the check is built for).

### 1.3 What does NOT change

100 sims, c_puct 1.5, v1 priors (temperature 2.0), most-visited
root child, VALUE_WEIGHT 0.25, the v8 checkpoint
(`ml/runs/b3-h2/value-net.pt`), v8's 2-ply threat check at the
root, terminal leaf cases exact. No q-based root decision
(CLOSED). No sims change (more sims is NOT monotonically
better). No prior change. No threat check beyond the VCF root
check above. `baseline.py` stays FROZEN. No kernel re-sync
unless v32 passes the bar.

### 1.4 Cost guard

v29's check averaged ~1.65s/move on quiet positions (median
90ms, p90 2.9s, max 36s) and ~7s/move in creator-rich games —
but v29's cost was dominated by the DEFENSIVE full-candidate
scan, which v32 does not run. The offensive-only check tests
only OUR four-creators (a small set, often empty) and returns
on the first VCF start. The cost profile is measured EARLY
(before the battery) with a value-engine bench
(`bench_vcf_root.py --engine v32`: v8 self-play positions,
ValueMCTS with the check enabled) and reported in §3. No
design change is made after any eval on the basis of cost
alone — cost is reported, strength is what the battery
measures. If the bench shows a heavy tail, the battery legs
are still run in full (the drivers auto-resume).

### 1.5 Port verification (before any eval)

1. **Offensive check fires:** v32's check plays the forced-win
   start on v29's 4 VCF smoke positions (vcf-1 (8,3), vcf-2
   (8,5), vcf-race-block-or-win (8,3), vcf-two-four-sequence
   (9,6)) — the same moves v29's check plays.
2. **Fall-through equivalence:** where the check does not fire,
   v32 IS v8 exactly (same leaf, same priors, same search,
   same threat check) — unit test at low sims, same seeds.
3. **Offensive-only:** on v18's three tactical-failure
   prevention positions (prev-v1v0s-g26, prev-v1v1s-g79,
   prev-v1bl-g2), v32's check does NOT fire (those are
   defensive preventions, which the offensive-only check
   cannot play) — the defensive step is provably absent, not
   merely unused.
4. **v8 tactical re-verified** on the CORRECTED suite in this
   worktree (`eval_tactical_full.py --version 8 --suite both
   --seed 100`): 20/20 + 29/29 expected.

### 1.6 Eval seeds (pre-registered)

- Battery h2h vs v8: v32=100 / v8=200; vs v0: v32=100 / v0=200;
  vs v7: v32=100 / v7=200.
- Ladder: seed 100, 20 games/opponent, mcts_sims=100.
- Swap: 100 games vs v7, balanced placement, placement_seed
  11, tempo judge (the harness default).
- Tactical: `eval_tactical_full.py --suite both --seed 100`.
- Fast-loop dev seeds: 1035/2035 (vs v0), 1036/2036 (vs v8) —
  distinct from the battery's 100/200, from ML batch 1's
  1001/2001, and from batch 9's 1029–1032/2029–2032.
- Harness commands follow iter-24 §2 exactly
  (`fast_loop_match.py`, `selfplay_batch.py --swap-rule`,
  `eval_version.py` for the ladder, `eval_tactical_full.py`
  for tactical). The match drivers AUTO-RESUME from partial
  `--out` files — if a run is killed, relaunch the SAME
  command; the saved games stand and the RNG restart is
  recorded as `resumed_from_game`.

## 2. Port

### 2.0 Implementation

Three files, all on branch `research/ml-batch-2-h3`:

1. **`ml/gomoku_ml/mcts_heuristic.py`** — `_vcf_root_move` gains
   a `defensive: bool = True` parameter; step 2 (the defensive
   prevention scan) is skipped when `defensive=False`. The
   default preserves v29 exactly (its `choose_move` calls the
   method without the argument). No other change to the
   heuristic engine.
2. **`ml/gomoku_ml/mcts_value.py`** — `ValueMCTS.__init__`
   gains `vcf_root: bool = False` and `vcf_defensive: bool =
   False` (both default off = v8 exactly); `choose_move`
   hooks the check after the win/block scans and the
   single-candidate early return, before the search — the same
   position as v29's hook in the parent's `choose_move`.
   The hook had to be added explicitly: `ValueMCTS.choose_move`
   overrides the parent's entirely, so v29's hook does not
   reach the value engine. The check call passes
   `defensive=self.vcf_defensive`.
3. **`ml/gomoku_ml/scoring.py`** — v32 registered in
   `ENGINE_VERSIONS` (same v8 checkpoint
   `ml/runs/b3-h2/value-net.pt`, `vcf_root: True`,
   `vcf_defensive: False`); the `ValueMCTS` branch of
   `ScoringPlayer.__init__` passes both flags through.

`baseline.py` untouched (frozen).

### 2.1 Port verification (§1.5) — all four checks pass

1. **Offensive check fires:** v32's check plays (8,3) on vcf-1
   and (8,5) on vcf-2 (unit test), and the full tactical run
   (§5) confirms the race/two-four VCF positions too — the same
   moves v29's check plays.
2. **Fall-through equivalence:** where the check does not fire,
   v32 IS v8 exactly — unit test
   (`test_v32_falls_through_to_v8_where_check_does_not_fire`,
   low sims, shared random checkpoint and seed, 8 sparse
   boards).
3. **Offensive-only:** on all three of v18's tactical-failure
   prevention positions (prev-v1v0s-g26, prev-v1v1s-g79,
   prev-v1bl-g2), v32's check does NOT fire
   (`defensive=False` returns None), while the defensive step
   DOES fire on the same positions (control) — the defensive
   step is provably absent, not merely unused.
4. **v8 tactical re-verified** on the CORRECTED suite in this
   worktree: **20/20 old + 29/29 hard** (`eval_tactical_full.py
   --version 8 --suite both --seed 100`).

New tests (5, in `ml/tests/test_mcts_value.py`): registration
+ offensive-only flags, offensive fires on vcf-1/vcf-2,
defensive step absent on the three prevention positions,
fall-through equivalence, board not mutated. All pass, and
the v29 tests still pass (the `defensive` default preserves
v29's behaviour). Targeted run: 13 passed (v32 + v29 + v8
selection).

## 3. Cost guard

`bench_vcf_root.py --engine v32` (new mode: v8 self-play
positions, ValueMCTS with the offensive-only check, seed 777,
n=100): **fired 6/100 (6%), median 14.5ms, p90 104.6ms, max
279.2ms, mean 43.9ms.**

The offensive-only check is ~38x cheaper than v29's on mean
(43.9ms vs 1.65s) and the tail is gone (max 279ms vs 36s):
v29's cost was dominated by the defensive full-candidate
scan, which v32 does not run. The fire rate is also much
lower (6% vs v29's 29%) — most of v29's firings were the
defensive step. At ~44ms mean overhead next to a ~1s search,
the battery runs at essentially v8's pace; no design change
on cost grounds (§1.4), and none is needed.

## 4. Fast loop

### 4.0 The d9 diagnostic (§1.2)

`ml/scripts/diag_d9.py` reconstructs v24's vs-v0 game 1 ply-19
position (White to move; actual i10 loses 3/3, d9 saves 2/3):

- White holds 2 four-creators, **0 VCF starts** — the offensive
  check cannot fire, as predicted (§1.2).
- Black holds **12 VCF starts** — the defensive check WOULD fire
  (and, run as a diagnostic, plays d9, the save).
- **v32 plays d9 anyway** — via the search, not the check — and
  **v8 plays d9 too** (iter-24 §5 already noted v8 plays d9 "via
  visit dynamics"). The i10 miss was v24/v7's, not v8's: there
  is nothing for the check to catch here, and the diagnostic
  does NOT trigger the defensive-inclusion rule (§1.1 — one
  position, and the rule needs ≥2 verified fast-loop losses
  plus a validated VCF-filtered search).
- Side observation: `starts_forced_win` reports 12 Black VCF
  starts at a position that is NOT lost (d9 saves 2/3) — the
  iter-29 §2.3 caveat again (the VCF definition does not model
  defender counter-threats beyond the next move).

### 4.1 Legs

**vs v0 (seeds 1035/2035): 4W-5L-1D (0.4500, CI [0.2014,
0.7263]) in 498s.** As Black 3-2, as White 1-3-1. Defence
solved (5/5 losses via open four, 0 missed blocks, 0 clean
missed preventions in 26 threatened turns).

**vs v8 (seeds 1036/2036): 5W-5L-0D (0.5000, CI [0.2366,
0.7634]).** As Black 1-4, as White 4-1 (White won 8/10 games
overall). Defence solved (5/5 via open four, 0 missed blocks,
0 clean missed preventions in 19 threatened turns).

**Sample duplication (both legs).** The fast-loop samples are
heavily duplicated: the vs-v8 leg has only **4 distinct games
in 10** (games 0/3/4 identical; games 1/2/5/7/8 identical —
the same lines with colours swapped), and the vs-v0 leg has
**7 distinct in 10** (games 1/5/9 identical; games 6/8
identical). The effective sample is far smaller than 10 games
per leg, and the win rates are very noisy — the battery is
the verdict, as always. The duplicates also show the two
engines play IDENTICALLY on the shared lines: where the
check fired, v8's search found the same VCF move anyway
(proven by the colour-swapped duplicates), so on these lines
the check changed no move — it only skipped the search. The
check's value-add, if any, must come from positions where the
search MISSES the VCF start, which the fast loop did not
produce as a distinct loss.

### 4.2 VCF fires in real games

`ml/scripts/replay_vcf_fires.py` re-runs the check at every
v32 turn of the recorded games (0 mismatches both legs — the
recorded move equals the check's move whenever it fired,
validating the port end-to-end in real games):

| Leg | Checked turns | Fires | Fire rate | Check time median / p90 / max / mean |
|---|---|---|---|---|
| vs v0 | 243 | 7 | 2.9% | 98ms / 514ms / 6.2s / 261ms |
| vs v8 | 109 | 5 | 4.6% | 13ms / 104ms / 198ms / 40ms |

Pooled: **12 fires, all in games v32 won; none in the 10
losses or the draw.** Every one of v32's 9 fast-loop wins
featured at least one VCF fire. The in-game check cost is
heavier than the quiet-position bench (§3) — real games reach
creator-rich positions the bench does not — but still far
below v29's profile (max 6.2s vs 36s). The fires cluster in
the endgame (e.g. game 3 vs v0: 4 consecutive fires at plies
75–81 closing out a won game), i.e. the check converts wins
the position already holds.

### 4.3 Loss workups

All 10 losses worked up — 6 DISTINCT losses (duplicates
collapsed: vs-v8 g0=g4, g2=g8; vs-v0 g1=g5=g9). Each workup:
last 4 v32 turns, top-5 1-ply + VCF + deep-sims alternatives,
seeds 0/1/2. Raw JSONs in `~/workspace/ml2h3-logs/
workup-v32-*.json`. (The first workup batch was killed by a
runtime restart that wiped /tmp; the workups were re-run from
the committed fast-loop JSONs — v8-g0 and v8-g2 reproduced
exactly, confirming the pre-restart findings.)

| Loss | v32 | Plies | Verified save (playout) | VCF prevention? |
|---|---|---|---|---|
| vs-v8 g0 (=g4) | B | 34 | ply 26 (11,6) 3/3 | **YES** — White held exactly 2 VCF starts, (11,6) and (7,10); BOTH remove all White VCF starts. The defensive check picks the save by prior (0.1219 vs 0.1162) — a 0.0057 gap, not noise-level, but the two preventions are indistinguishable to the exact analysis |
| vs-v8 g2 (=g8) | B | 24 | ply 16 (6,8) 3/3 | NO — White held 0 VCF starts at ply 16; a horizon save with no exact signature |
| vs-v8 g9 | W | 25 | none in window | horizon loss |
| vs-v0 g1 (=g5=g9) | W | 25 | none in window | horizon loss |
| vs-v0 g2 | B | 28 | none in window | horizon loss |
| vs-v0 g4 | B | 34 | ply 26 (8,5) 3/3 | NO — and the opposite of a prevention: White held SIX VCF starts; the save (8,5) leaves FOUR live, while BOTH exact preventions lose 3/3 (the actual (11,10) and the check's pick (10,9), prior 0.0512 vs 0.0510) |

Three findings:

1. **Only ONE distinct VCF-prevention save exists** (v8-g0's
   (11,6)). The defensive-inclusion rule (§1.1) needs ≥2.
2. **v0-g4 refutes the prevention concept itself, not just the
   prior pick.** White's six "VCF starts" do not convert
   against real defence: the saving move ignores four of them,
   and both exact preventions lose. `starts_forced_win`
   over-reports forced wins (iter-29 §2.3's caveat, in extreme
   form) — so a defensive check that restricts to preventions
   would pick a losing move here, whether it chooses by prior
   OR by search (the prevention pool's tested members both lose
   3/3). The batch-10 direction (VCF-filtered search) fails at
   this position too, for a reason no selection rule fixes:
   the pool itself is wrong.
3. **No patchable weakness class.** The two non-VCF saves
   (v8-g2's (6,8), v0-g4's (8,5)) have no exact signature a
   rule could reproduce; the three horizon losses have no save
   at all; the single VCF-prevention save cannot carry a
   design change (and the check that would play it is the
   verified-flawed one). Fast loop STOPS after iteration 1.

### 4.4 Defensive-inclusion decision (§1.1)

**The defensive check stays OFF. v32 ships offensive-only.**

Condition 1 fails (1 VCF-prevention save, need ≥2), and
v0-g4 shows the second condition's premise is false as well:
even a search-chosen prevention would lose there. The
offensive-only design stands as pre-registered. No code
changes from the fast loop — the battery runs on the
implementation of §2.

## 5. Battery

TODO — battery, filled in when run.

## 6. Verdict

TODO.
