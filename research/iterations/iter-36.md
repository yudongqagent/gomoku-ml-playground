# Iteration 36 — v36: v27's q-gap threshold tie-break ported to the value engine (ML batch 3, H1)

**Status: IN PROGRESS — pre-registration (§1).** Comparison target is
v8 (the ML champion: ladder Elo 0.0, vs v0 0.545, tactical 20/20 +
29/29 — re-verified on the CORRECTED suite at `--seed 100` before
any v36 work: old 20/20, hard 29/29). This is the H1 workstream of
ML batch 3 (H2 is v37, deeper-teacher distillation; H3 is v38,
multi-task value net); work happens on branch
`research/ml-batch-3-h1` in the `gomoku-ml3-h1` worktree, branched
from `research/ml-batch-3` @ `d4dc549` (the batch-3 plan +
per-game seeding tooling). v36 does NOT change the net (v8's
checkpoint `ml/runs/b3-h2/value-net.pt`, byte-identical — no new
checkpoint file), does NOT change sims/priors/VALUE_WEIGHT
(CLOSED), does NOT change the threat check, and does NOT include
the VCF check (v32 showed it near-neutral on this engine). The
ONLY change from v8 is the threshold tie-break on the root
selection in §1 — a port of v27's mechanism (iter-27 §1.1) to the
value engine. It is NOT a reopening of the closed q-based
root-decisions lever (see §1.1).

**Method (carried over):** the fast 10-game loop — play 10 games,
read EVERY move of every game, find THE losing move per game
(verified with the solver or a deeper search — never guessed),
patch the algorithm so that, replayed to that critical position,
it plays the perfect move, and record each critical position +
correct move as a regression test. Then play 10 more games and
repeat. Max 4 fast-loop iterations before validation, or stop when
two consecutive iterations find no new patchable weakness class.
Patches must fix the STRUCTURAL cause, not the single position.
**The fast loop runs against v0 as well as v8** — each iteration
plays 10 games vs v8 AND 10 games vs v0, and losses from both
opponents are walked back. The full 4-clause validation battery
(100-game matches + ladder) runs on the FINAL patched v36 only.
Fast-loop games use development seeds (1001/2001, 1002/2002, ...)
distinct from the bar battery's pre-registered seeds (100/200),
so no development game overlaps an eval game.

## 0. The q-gap measurement (BEFORE pre-registration — diagnostic, not an eval)

v8's q values come from the NET, not the rollout leaf, so the
q-gap distribution was measured FIRST (`/tmp/qgap_v8.py`, v8's
100-sim search, fresh seed 0, on the effective root pool — after
v8's 2-ply threat filter, exactly as `ValueMCTS.choose_move`
applies it). Two corpora: all 49 tactical positions (old 20 +
hard 29) and 220 midgame positions sampled every 4th ply from
ply 8 of v8's recorded battery games (games 0–9 vs v0 and vs v1,
`batch-v8-vs-v0.json` / `batch-v8-vs-v1.json`). 269 positions
total; 31 scan-decided (win/block/single candidate), 238
searched.

**Result — the gaps do NOT separate.** This is the finding the
brief anticipated, and it is reported here, not used to re-tune
τ after the fact.

| Quantity | Value |
|---|---|
| Searched positions | 238 |
| Exact top-visit ties | 59 (24.8% of searched) |
| Gap = 0 exactly | 3 |
| Gap in (0, 0.01] | 36 |
| Gap in (0.01, 0.05] | 17 |
| Gap in (0.05, 0.10] | 3 |
| Gap > 0.10 | **0** |
| Gap median / mean / max | 0.0048 / 0.0095 / **0.0569** |

(One script bug, corrected here from the raw records: the
histogram labelled the (0, 0.01] bin "0"; the counts above are
recomputed from `/tmp/qgap_v8.json` and sum to 59.)

Compare v27's rollout leaf: signal gaps 0.130–0.417 (exact
terminal values — loser −1.000 vs saver +0.250 at the leaf),
documented noise gap 0.016 — an 8x separation. For the value
engine the LARGEST gap in 59 ties (0.0569) is 2.3x SMALLER than
the rollout leaf's smallest signal gap, and the median gap
(0.0048) is 3x smaller than the rollout leaf's NOISE gap. The
distribution is unimodal noise: no bimodal split, no terminal
signal. The three gaps above 0.05 are all between small
net-scale values (0.085 vs 0.028; −0.025 vs −0.079; −0.155 vs
−0.208) — no ±1.0 terminal value is involved in any of them.

**Why (structural, not positional).** The net value is shrunk
by VALUE_WEIGHT = 0.25, so quiet-leaf values live in
[−0.25, 0.25]; a root child's q is a mean over sims that are
mostly net leaves, so q values cluster in a narrow band
(median our-q_max 0.108, median our-q_mean −0.001 across
searched positions; median 0 children per position at an exact
terminal ±1.0 our-q). The rollout leaf's exact terminal values
are what made its gaps large; the net leaf has no equivalent.

**Robustness.** 12 of the 59 ties are 49-way ties at 2 visits
from two very long v8-vs-v1 games (g1, g9; plies 52–184) where
the search spreads thinly over 73–132 children — a sampling
artefact of the corpus, all with gaps ≤ 0.0122. Excluding those
two games: 187 positions, 158 searched, 30 tied (19.0% tie
rate), 2 gaps > 0.05, 0 > 0.10, max 0.0569. The conclusion is
unchanged: ties are common (~19–25% of searched positions —
the visit-tie problem EXISTS for the value engine), but the
gaps carry no separable signal.

**Tactical positions.** 8/29 hard-suite positions have root
ties (all threat-filtered pools, gaps 0.0002–0.0527); only
prev-v1v1s-g45 (gap 0.0527, tied our-q −0.155 vs −0.208)
exceeds 0.05. The old suite has 0 ties among its 4 searched
positions (16 scan-decided). v8 already passes 29/29, so the
tie-break has NO tactical recovery available here — it can only
change (and possibly break) correct choices.

**Consequence for the sweep, stated before running it.** At
τ=0.10 the tie-break fires on 0/59 measured ties — v36 at
τ=0.10 is behaviourally v8 (up to RNG-stream identity: when
the rule does not fire, the code path is v8's exactly). At
τ=0.05 it fires on 3/59 ties (~5% of ties, ~1.3% of searched
positions). The sweep values stay {0.05, 0.10} as
pre-registered in the batch plan — the measurement weakens the
hypothesis but does not change the test. If the battery shows
v36 ≈ v8, that is the port not transferring, and the mechanism
is understood (this section), not a mystery.

## 1. Design and pre-registration

**Comparison target: v8** (primary, the ML champion: ladder Elo
0.0, on master). The ML bar's clause 1 is head-to-head vs v8
(normal rules, 100 games, alternating colours, win rate > 0.50
with Wilson 95% CI lower bound > 0.50), clause 2 is ladder Elo
strictly higher than v8's 0.0 (identical 20 games/opponent:
random, heuristic, mcts, mcts_sims=100), clause 3 is tactical
pass rate >= v8's on the CORRECTED suite (old suite 20/20, hard
suite 29/29), clause 4 is all test suites green (ml pytest, npm
test, tsc). **Secondary head-to-heads, also pre-registered:**
v36 vs v0 (100 games), v36 vs v7 (100 games), and a 100-game
swap match vs v7 under the tempo judge (the harness default).
Non-transitivity is recorded prominently.

**The improvement: v36 = v8 + v27's threshold tie-break at the
root.** Everything else is v8 unchanged: the v8 checkpoint,
100 sims, c_puct 1.5, v1 priors over all candidates, win/block
scans before search, most-visited root child, the 2-ply threat
check restricting the root pool to safe moves, VALUE_WEIGHT
0.25 on the net leaf. The threshold tie-break is the ONLY
change from v8.

### 1.1 The threshold rule (pre-registered, exact)

v27's rule (iter-27 §1.1), ported to `ValueMCTS.choose_move`'s
root selection. The port is necessary because `ValueMCTS`
re-implements root selection inline — it never calls the
parent's `_pick_root_move` / `_choose_root_child` (the threat
filter needs the pool first). The rule operates on the
EFFECTIVE pool (after the threat filter), mirroring
`_pick_root_move` exactly:

1. **When it applies:** only at the root, only when two or more
   children in the effective pool share the maximum visit count
   (an exact integer tie at the top) after all sims. If a unique
   child has the most visits, v36 behaves EXACTLY like v8 — no q
   is consulted, no RNG is consumed. Near-ties (a 1-visit gap)
   are NOT ties and are not touched.
2. **The q gap:** among the visit-tied children, let our-q =
   `−child.q` (our perspective; `child.q` is the OPPONENT's —
   see iter-23 §1.1 rule 2, verified against the backup code;
   the same backup code serves ValueMCTS). Let q_max be the
   highest our-q and q_second the second-highest DISTINCT our-q
   value among the tied children. If all tied children share the
   same our-q exactly, the gap is 0. The gap is q_max − q_second.
3. **Fire condition:** the tie-break fires ONLY if gap > τ (the
   threshold, strictly greater). When it fires, v36 plays the
   child with the highest our-q; if two or more share q_max
   exactly (float equality), fall back to RNG choice among that
   subset when an RNG is set, else the first in dict order.
4. **Below threshold:** when gap ≤ τ, the tie-break does NOT
   fire at all — v36 behaves EXACTLY like v8: RNG choice among
   ALL visit-tied children when an RNG is set, else the first in
   dict order. (Not restricted to the q-tied subset: below the
   threshold q carries no reliable signal, so no q information
   is used.)

**Interpolation check (by construction).** τ = 0 reproduces
v23's rule on the value engine (any positive gap fires, and a
zero gap means all tied children share q_max, so the q-tied
subset IS all tied children). τ → ∞ reproduces v8 exactly (the
rule never fires). v36 sits strictly between them.

**Why this test is justified (and why it is not a reopening of
the closed lever).** The visit-tie problem is real for the
value engine (§0: ~19–25% of searched positions tie at the
top), and v27's threshold is the heuristic track's best
mechanism. But §0 already shows the premise is weakened: net q
gaps do not separate signal from noise the way rollout-leaf
gaps did. This iteration is therefore a TRANSFER TEST, not a
strength claim: does the mechanism survive the change of leaf?
The expected answer, honestly stated before the sweep, is
"mostly no" — the rule fires rarely at τ=0.05 and never at
τ=0.10 in the measured corpus. Pure q is still catastrophic
(0.20), and v36 still never consults q except on an exact
visit tie with a large gap.

**What would refute the hypothesis.** If the selected
threshold's battery vs-v8 rate is ~0.50 with the CI including
0.50 and ladder Elo ≈ 0.0, the port does not transfer: the
threshold mechanism needs terminal-scale q gaps, and the net
leaf does not produce them. If BOTH sweep thresholds fail the
tactical gate (§1.2), the tie-break actively breaks v8's
tactical play (it can only change correct choices — v8 is
already 29/29).

### 1.2 Threshold sweep (pre-registered values and selection rule)

**Sweep values: τ ∈ {0.05, 0.10}.** Same values as v27 (the
batch plan's pre-registration). For the value engine both sit
ABOVE essentially the whole measured gap distribution (max
0.0569): 0.05 admits 3/59 measured ties, 0.10 admits 0/59. The
sweep therefore tests whether the rare τ=0.05 fires carry any
signal in real games, and whether τ=0.10 is a true no-op. A
smaller τ (e.g. 0.01) would fire on ~1/3 of ties, but that
would be post-hoc tuning to §0's measurement — NOT swept.

**Sweep protocol (per τ):**
- Tactical old + hard at `--seed 100` (the established protocol).
- 40 games vs v0 (alternating colours, 100 sims), seeds
  3001/4001 — fresh dev seeds, distinct from the battery's
  100/200 and the fast loop's 1001+/2001+. vs v0 is the direct
  measure of whether the rare fires help or hurt against a weak
  opponent (v8's vs-v0 rate is 0.545, the weakest of its
  numbers — the most room to move).

**Selection rule (stated BEFORE seeing any numbers; v27's rule
verbatim, with v8 as the reference):**
1. **Tactical gate.** A τ is eligible only if it scores 20/20
   old + 29/29 hard on the corrected suite at `--seed 100`. If
   exactly one τ passes the gate, select it. If both fail, select
   the one with the higher hard-suite pass count (ties → the
   smaller τ) and record the hypothesis as weakened — the battery
   still runs on the selected τ, since the battery is the verdict.
2. **Among eligible τ:** select the one with the higher vs-v0 win
   rate over the 40-game sweep match.
3. **Tie rule.** If the two vs-v0 win rates differ by ≤ 1 game
   (0.025), select τ = 0.05 — the smaller threshold, which stays
   closer to v23's demonstrated mechanism.

The sweep candidates are registered as TEMPORARY versions 92
(τ=0.05) and 93 (τ=0.10) so the standard drivers run them
unchanged; both entries are removed when v36 is registered with
the selected threshold. (Numbered 92/93, not v27's 90/91: v27's
regression test asserts 90/91 are absent, so reusing those
numbers would break the suite during the sweep. The numbers are
a tooling detail, not part of the hypothesis — the sweep values,
protocol, and selection rule are unchanged.) The 100-game battery is the verdict, not
the sweep: the sweep's job is to catch a catastrophic threshold
and to pick between two near-identical candidates, not to measure
strength (40 games cannot distinguish τ=0.05 from τ=0.10 finely
— the two differ only on ties with q gaps in (0.05, 0.10], which
are ~5% of ties).

**Tactical expectation (pre-registered).** At τ=0.10, no
measured tie fires — expected tactical 20/20 old + 29/29 hard,
identical choices to v8. At τ=0.05, exactly one measured
tactical position fires (prev-v1v1s-g45, gap 0.0527): whether
it keeps the expected move depends on whether the q-max child
is an expected move — the mechanism check (§1.3) will show it
before the sweep. **Expected tactical: 20/20 old + 29/29 hard
at both τ**, with the prev-v1v1s-g45 fire as the one watched
position.

**Game-strength expectation (pre-registered, honest).** v36 is
expected to play at ≈ v8's level: the rule fires on ~1% of
moves at τ=0.05 and ~0% at τ=0.10, so the vs-v8 battery rate
should sit near 0.50 with the CI including 0.50 (clause 1
expected to FAIL), and ladder Elo near 0.0 (clause 2 expected
to FAIL or pass only on noise). A pass on either clause would
mean the rare fires carry real signal despite §0 — the battery
decides. The value of this iteration is the transfer test
itself: it prices the threshold mechanism for the value
engine, exactly as batch 10 priced VCF-prevention for the v27
family.

**Cost (from v8's measurements — the leaf function is
unchanged, so the cost is identical by construction).** Sweep:
~10 min per τ (40 games vs v0; v8's 100-game vs-v0 battery took
907 s) + tactical. Fast loop: ~1 h for 2 iterations. Battery:
100 games vs v8 ~30–60 min, vs v0 ~15 min, vs v7 ~30–60 min,
swap ~30–60 min, ladder ~30 min, tactical ~1 min. Affordable.

**Eval seeds (pre-registered):** h2h vs v8: v36=100 / v8=200;
h2h vs v0: v36=100 / v0=200; h2h vs v7: v36=100 / v7=200;
ladder seed 100, 20 games/opponent, mcts_sims=100; swap 100
games vs v7, balanced placement, placement_seed 11, tempo judge
(the harness default), seeds 100/200; tactical `--seed 100`.
Fast-loop dev seeds 1001/2001, 1002/2002, ... (iteration k uses
1000+k / 2000+k, per iter-8 §4's policy). Sweep seeds 3001/4001.

### 1.3 Mechanism check (post-implementation, pre-sweep)

(TODO — after implementation, before any sweep game: root dumps
on the 8 tied tactical-hard positions at both τ, verifying the
dispatch does what §1.1 says, especially prev-v1v1s-g45.)

### 1.4 Battery (the verdict — pre-registered seeds and commands)

On the FINAL v36 only (after the fast loop):

- 100 games vs v8, seeds 100/200 (clause 1).
- 100 games vs v0, seeds 100/200.
- 100 games vs v7, seeds 100/200.
- 100-game swap vs v7, balanced placement, placement_seed 11,
  tempo judge (the harness default), seeds 100/200.
- Ladder 20 games/opponent (random, heuristic, mcts;
  mcts_sims=100), seed 100 (clause 2).
- Tactical old 20/20 + hard 29/29, full-dispatch driver
  (`ml/scripts/eval_tactical_full.py`, `player.choose_move` —
  NOT `eval_version.py`'s probe, which bypasses the MCTS
  dispatch) (clause 3).
- ml pytest + npm test + tsc green (clause 4).

**Integration.** `ml/gomoku_ml/mcts_value.py`: `ValueMCTS`
gains `tie_break_q` / `tie_break_q_threshold` parameters
(defaults False / 0.0 — v8's behaviour exactly, so v8 and all
earlier versions are unaffected) passed through to the parent
`__init__`, and `choose_move`'s root selection gains the
threshold tie-break on the effective pool (mirroring
`HeuristicMCTS._pick_root_move` — same rule, same candidate
set restricted to the pool). `ml/gomoku_ml/scoring.py` gains
ONLY the `ENGINE_VERSIONS[36]` registry entry (v8's entry +
the selected `"tie_break_q"` / `"tie_break_q_threshold"`)
plus passing the flags through to `ValueMCTS` in the
`value_checkpoint` construction branch. `baseline.py` is
frozen and untouched. **Merge-trap note (flagged to the
coordinator): this iteration DOES touch `ValueMCTS.__init__`'s
signature** — it adds two keyword parameters with defaults
(`tie_break_q: bool = False`,
`tie_break_q_threshold: float = 0.0`), a compatible extension:
every existing caller passes the same arguments as before and
gets identical behaviour. The silent auto-merge breakage on
this signature (ML batch 1, ML batch 2 — duplicate parameters
with NO conflict flag) is the reason this is called out
explicitly; the coordinator's merge should check for duplicate
parameters before running anything.

### 1.5 Threshold selection — sweep results (the pre-registered rule applied)

(TODO — after the sweep.)

## 2. Loss analysis for iteration 36

### 2.1 Fast-loop iteration 1 — 10 games vs v8 + 10 games vs v0

(TODO)

### 2.2 Fast-loop iteration 2

(TODO)

### 2.3 Fast-loop iteration 3 — NOT RUN unless iterations 1–2 find a patchable class

### 2.4 Fast-loop iteration 4 — NOT RUN unless iterations 1–3 find a patchable class

### 2.5 Aggregate counts (secondary)

(TODO)

## 3. v36 evaluation

(TODO — battery on the FINAL version only.)

### 3.1 Tactical (clause 3)

(TODO)

### 3.2 Head-to-head vs v8 (clause 1)

(TODO)

### 3.3 Head-to-head vs v0

(TODO)

### 3.4 Head-to-head vs v7

(TODO)

### 3.5 Swap match vs v7

(TODO)

### 3.6 Ladder (clause 2)

(TODO)

### 3.7 Defence classification

(TODO)

### 3.8 Test suites (clause 4)

(TODO)

## 4. Bar check (v36 vs v8)

(TODO)

## 5. What this means

(TODO)
