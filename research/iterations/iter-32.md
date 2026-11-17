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

TODO — implementation, port verification (§1.5), tests.

## 3. Cost guard

TODO — bench numbers, fire rate, time/move distribution.

## 4. Fast loop

TODO — 10 games vs v0 (seeds 1035/2035), 10 vs v8 (seeds
1036/2036), loss workups, the d9 diagnostic (§1.2), the
defensive-inclusion decision (§1.1).

## 5. Battery

TODO — battery, filled in when run.

## 6. Verdict

TODO.
