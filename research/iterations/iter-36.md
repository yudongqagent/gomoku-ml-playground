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

After implementation and before any sweep game, v92 (τ=0.05)
and v93 (τ=0.10) were run on the 8 tied tactical-hard positions
(fresh-seed-0 root dumps, `/tmp/mech_v36.py`). This is an
implementation check, not pre-registration evidence and not an
eval — it verifies the dispatch does what §1.1 says it does.

**Root dumps reproduce §0 exactly** (same ties, same q values,
same priors — the search is unchanged; only the final selection
differs). All 8 positions are threat-filtered pools, as in §0:

| Position | Root state (fresh seed 0) | Gap | τ=0.05 | τ=0.10 |
|---|---|---|---|---|
| prev-v1v0s-g13 | 4-way tie at 4 visits | 0.0097 | no fire → (11,7) EXPECTED | no fire → (11,7) EXPECTED |
| prev-v2v0-g2 | 2-way tie at 4 visits | 0.0036 | no fire → (6,5) EXPECTED | no fire → (6,5) EXPECTED |
| prev-v2v1s-g12 | 2-way tie at 4 visits | 0.0126 | no fire → (13,2) EXPECTED | no fire → (13,2) EXPECTED |
| prev-v2v1s-g28 | 2-way tie at 4 visits | 0.0002 | no fire → (4,9) EXPECTED | no fire → (4,9) EXPECTED |
| prev-v1v0s-g48 | 2-way tie at 5 visits | 0.0087 | no fire → (4,9) EXPECTED | no fire → (4,9) EXPECTED |
| prev-v1v1s-g79 | 4-way tie at 6 visits | 0.0125 | no fire → (8,7) EXPECTED | no fire → (8,7) EXPECTED |
| prev-v2v1s-g30 | 2-way tie at 4 visits | 0.0356 | no fire → (5,4) EXPECTED | no fire → (5,4) EXPECTED |
| prev-v1v1s-g45 | 2-way tie at 5 visits | 0.0527 | **FIRES → (11,7) EXPECTED** | no fire → (12,7) EXPECTED |

Exactly one position fires at τ=0.05 (prev-v1v1s-g45, gap
0.0527 — the only measured tactical gap above 0.05), and the
fired pick (11,7) is an expected move — v8's RNG picked the
other expected move (12,7). At τ=0.10 nothing fires and every
choice equals v8's, including the RNG picks (the no-fire code
path consumes RNG identically to v8 — verified: all 8 choices
match v8's move-for-move at both τ except the single fire).
Note prev-v1v1s-g79: the q-max move is (7,9) but the gap
(0.0125) is below both thresholds, so all three versions play
(8,7) by RNG — both are expected moves. **Expected tactical:
20/20 old + 29/29 hard at both τ** — the sweep's tactical gate
(§1.2 rule 1) should pass for both candidates.

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

**Tactical gate (§1.2 rule 1):** both τ pass — 20/20 old + 29/29
hard at `--seed 100`, as the mechanism check (§1.3) predicted.
Both eligible.

**Sweep artefact names.** The first sweep run wrote to
`sweep-tau005-vs-v0.json` / `sweep-tau010-vs-v0.json` — the same
names as v27's committed sweep artefacts — and the driver
correctly skipped both matches as already complete (v27's
39W-0L-1D). No v36 games were played and no v27 artefact was
modified. The matches were re-run to distinct files
(`sweep-v36-tau005-vs-v0.json` /
`sweep-v36-tau010-vs-v0.json`); only those files are v36 sweep
evidence.

**40-game vs-v0 sweep (seeds 3001/4001, alternating colours,
100 sims; aggregates recomputed from the raw records, matching
the drivers' headers exactly):**

| τ | Result | Win rate | 95% CI | As Black | As White | Unique games | De-dup rate |
|---|---|---|---|---|---|---|---|
| 0.05 | 20W-20L-0D | 0.5000 | [0.3520, 0.6480] | 11W-9L (0.55) | 9W-11L (0.45) | 31/40 | 0.5161 |
| 0.10 | 20W-20L-0D | 0.5000 | [0.3520, 0.6480] | 11W-9L (0.55) | 9W-11L (0.45) | 31/40 | 0.5161 |

The two thresholds are INDISTINGUISHABLE on the sweep:
identical W/L/D, identical colour splits, identical uniqueness
audits. 37/40 games are move-for-move identical between the two
matches; the 3 diverging games (12, 26, 36) all keep the same
winner — the τ=0.05 fires (the only difference between the
engines) change lines but no outcomes. **Per the pre-registered
tie rule (§1.2 rule 3: rates within 1 game → the smaller
threshold), v36 = τ 0.05.** The temporary sweep entries (92/93)
were removed on registration. Sweep artefacts:
`research/iterations/sweep-v36-tau005-vs-v0.json`,
`research/iterations/sweep-v36-tau010-vs-v0.json`.

**What the sweep does and does not show.** Both thresholds sit
at 0.5000 vs v0 — within noise of v8's battery 0.545 (54W-45L-1D,
CI [0.4475, 0.6391]), and the sweep CIs overlap it heavily. The
tie-break changes essentially nothing against v0 at either τ,
exactly as §0 predicted (the rule fires on ~1% of moves at
τ=0.05 and ~0% at τ=0.10). The 100-game battery is the verdict —
but the sweep already prices the mechanism: a near-no-op. The
only open question for the battery is whether the rare τ=0.05
fires carry any signal vs v8 itself (clause 1), where the two
engines differ ONLY in those fires.

**Runtime restart during the sweep (recorded, no games lost).**
A runtime restart killed the sweep chain mid-match (τ=0.10 at
20/40 games) and wiped /tmp — the q-gap measurement's raw
records (`/tmp/qgap_v8.json`) and the mechanism-check records
(`/tmp/mech_v36.json`) are lost, but every number reported from
them was recomputed and recorded in §0/§1.3 before the restart,
and the committed log is the record. The chain was relaunched
from the persistent scripts directory
(`~/workspace/gomoku-ml3-h1-scripts/`, created after the
restart so a second wipe cannot strand the run) and the driver
auto-resumed from the partial `--out` file (`resuming from
game 20`). Resume integrity verified two ways: (a) all 20 saved
prefix games match the pre-restart snapshot (winners, plies,
colours) taken from the recovery log; (b) the resumed match's
duplicate structure is IDENTICAL to the uninterrupted τ=0.05
control's (duplicate groups [4,20], [6,18,24],
[9,25,27,29,31,39], [30,38] in both) — the groups straddling
the resume boundary are engine determinism (the same line
recurring from different per-game seeds), not resume replay,
because the never-resumed control has them too. Per-game
seeding reproduced the uninterrupted run exactly. v36's own
tactical (post-registration): 20/20 old + 29/29 hard at
`--seed 100`, as expected.

## 2. Loss analysis for iteration 36

### 2.1 Fast-loop iteration 1 — 10 games vs v8 + 10 games vs v0

**Batches:** `research/iterations/fastloop1-v36-vs-v8.json`
(seeds v36=1001 / v8=2001) and
`research/iterations/fastloop1-v36-vs-v0.json` (seeds v36=1001 /
v0=2001) — development seeds, distinct from the bar battery's
100/200. **vs v8: 3W-7L-0D, win rate 0.3000, CI [0.1078,
0.6032]**, 473.4 s. Split: as Black 2W-3L (0.40), as White 1W-4L
(0.20). **vs v0: 3W-6L-1D, win rate 0.3500, CI [0.1369,
0.6463]**, 631.0 s (game 7 was a 225-ply full-board draw).
Split: as Black 3W-2L (0.60), as White 0W-4L-1D (0.10).
Classifier (both batches): 13/13 losses via open four, 0 missed
blocks, 0 clean missed preventions in 66 threatened turns.
Defence is still perfect. (n=10 per opponent — development
signal, not the bar; the fast loop's job is finding weaknesses,
not measuring strength.)

**The fire diagnostic exonerates the tie-break in both legs**
(`~/workspace/gomoku-ml3-h1-scripts/fire_diag.py` — replays
every v36 turn with the search re-run and the RNG state
snapshotted, so each fire is exact and the v8 counterfactual
pick is known; every replayed pick matched the recorded move).
vs v8: 132 v36 turns, 109 searched, 10 tied (9.2%), **0 fires**
— v36 played EXACTLY as v8 would have in all 10 games, so the
3W-7L is v8-vs-v8 seed noise, not a tie-break effect. vs v0:
267 turns, 220 searched, 58 tied (26.4%), **1 fire** — game 7
ply 25 (gap 0.0623, 2-way tie, v36 played 205 where v8's RNG
would have played 155), in the game that drew anyway. The
tie-break changed one move in 20 games and decided zero games.

**The 3-7 vs v8 is a colour lottery over 3 lines, not 10
independent games.** The vs-v8 leg collapses to **3 distinct
lines in 10 games** (uniqueness audit): line [0,7] (White wins
2/2), line [1,2,3,5,8,9] (Black wins 6/6), line [4,6] (White
wins 2/2). Every line was won by the same colour in every
occurrence — both engines are so deterministic that the line
determines the winner, and v36's score is the colour-assignment
lottery (it held the losing colour in 7 of 10). The driver's
de-dup rate (0.0) is an artefact of its first-occurrence rule:
the same lines contain wins under the opposite colour
assignment. Per-game seeding does not prevent this collapse —
it prevents resume replay, but the lines recur because the
engines barely consume RNG (RNG is spent only on exact ties,
and the tie-break fires ~0% of the time). The vs-v0 leg
collapses less (9 distinct lines in 10) because v0's play
varies more. The battery will face the same collapse on the
vs-v8 leg — its uniqueness audit is reported prominently in §3
and clause 1 is read on both cuts.

**Game 1 vs v8 (White, 25 plies) — the walkback nominates ply
11, and the stronger oracle REFUTES it.**
(`research/iterations/walkback-fl1-g1.json`, v1 oracle, seeds
0,1,2): plies 23/21/19/17/15/13 lose 3/3 with 0 saving
alternatives; ply 11 nominates (actual (9,8) loses 3/3; saves
(5,8) [draw,draw,draw] and (5,7) [win,win,win]); ply 9 does not
lose at all seeds. Under v8-vs-v8
(`~/workspace/gomoku-ml3-h1-scripts/verify_nomination.py`,
seeds 0,1,2): the ACTUAL move draws 3/3 and the (5,8) save
draws 3/3 — the nomination is refuted outright (the actual must
lose first before an alternative counts, and it does not lose
at all). The (5,7) leg was killed after ~14 min on marathon
playout games; it cannot resurrect the nomination, which
already fails on the actual's draws. **The root dump
exonerates the tie-break independently**: at ply 11 the
threat-filtered pool has 2 children and NO tie — the actual
move is the unique visit leader (16 visits, our-q −0.1288)
over the (5,8) save (12 visits, our-q −0.1858), and (5,7) is
not in the safe pool at all. No tie-break variant could have
picked a save: the actual wins on visits AND on q. The loss is
distributed drift in v8's search/leaf, not a root-decision
failure — and not patchable within v36's scope (its only change
is the tie-break).

**Iteration 1 verdict: no patchable weakness class.** The
tie-break fired once in 329 searched turns and decided zero
games; the one nominated losing move is refuted by the stronger
oracle and unreachable by any tie-break. The loop does NOT stop
here (same pre-commitment as iter-27 §2.1: two consecutive
barren iterations, 40 games). Iteration 2 runs on fresh seeds
(1002/2002); if it also finds no patchable class, the loop
stops. No regression positions added: no loss yields a verified
losing move with a verified save the tie-break could be patched
to play.

### 2.2 Fast-loop iteration 2 — 10 games vs v8 + 10 games vs v0 (LOOP STOPS)

**Batches:** `research/iterations/fastloop2-v36-vs-v8.json`
(seeds v36=1002 / v8=2002) and
`research/iterations/fastloop2-v36-vs-v0.json` (seeds v36=1002 /
v0=2002). **vs v8: 8W-2L-0D, win rate 0.8000, CI [0.4902,
0.9433]**, 219.5 s — the mirror image of iteration 1's 3W-7L.
Split: as Black 4W-1L (0.80), as White 4W-1L (0.80). **vs v0:
6W-4L-0D, win rate 0.6000, CI [0.3127, 0.8318]**, 295.2 s.
Split: as Black 4W-1L (0.80), as White 2W-3L (0.40).
Classifier (both batches): 6/6 losses via open four, 0 missed
blocks, 0 clean missed preventions in 34 threatened turns.
Defence still solved.

**The reversal confirms iteration 1 was seed noise.** The two
iterations bracket 0.50 almost symmetrically (3-7 then 8-2 vs
v8; pooled 11W-9L, 0.55). The vs-v8 leg again collapses to a
handful of lines (6 distinct in 10; the [1,3,5,9] White line
occurred 4 times, all with v36 as White → 4W). The fire
diagnostic again shows an inert tie-break — vs v8: 129 turns,
101 searched, 10 tied (9.9%), **0 fires**; vs v0: 232 turns,
184 searched, 30 tied (16.3%), **4 fires (2.17%)**, 3 changing
the move (game 1 ply 25 fired but the RNG agreed; game 6 ply 24
changed the move in a game v36 lost; game 9 ply 25 and ply 107
changed moves in a game v36 won). Across the whole fast loop:
614 searched turns, 5 fires (0.81%), 4 changed moves.

**The game 6 fire (the only fire in a fast-loop loss) is
exonerated.** At ply 24 (v36 as Black, 2-way tie, gap 0.0766)
the tie-break fired and played (9,5) where v8's RNG would have
played (3,8); v36 lost the game. Under v8-vs-v8
(`verify_nomination.py`, seeds 0,1,2): the fired move (9,5)
draws 3/3 AND v8's move (3,8) draws 3/3 — the two moves are
equivalent under strong play, so the fire did not cost the
game; the loss came later (distributed drift). The tie-break is
exonerated in the one fire-in-a-loss, exactly as it was in the
sweep's diverging games (§1.5) and the ply-11 nomination (§2.1).

**No patchable weakness class in this iteration either — the
fast loop STOPS here (2 iterations)** per the pre-commitment in
§2.1 (two consecutive barren iterations, 40 games). v36's only
losses in the loop are v8's distributed-drift losses plus one
fire in a lost game (game 6, §2.2 above); the battery is the
verdict on whether the rare fires help or hurt at scale.

### 2.3 Fast-loop iteration 3 — NOT RUN (loop stopped after iteration 2)

### 2.4 Fast-loop iteration 4 — NOT RUN (loop stopped after iteration 2)

### 2.5 Aggregate counts (secondary)

Fast-loop totals (2 iterations, 40 games): vs v8 11W-9L-0D
(0.5500); vs v0 9W-10L-1D (0.4750). Colour splits vs v8: as
Black 6W-4L (0.60), as White 5W-5L (0.50). Fast-loop defence:
19/19 losses via open four, 0 missed blocks, 0 clean missed
preventions (100 threatened turns in losses). Defence remains
solved. Fire totals: 614 searched v36 turns, 5 tie-break fires
(0.81%), 4 changed moves, 0 games decided by a fire (the game 6
fire is exonerated by the v8-vs-v8 verification — both moves
draw 3/3). No patches were made in this iteration —
v36-as-batteried is v36-as-pre-registered (τ=0.05, selected by
the §1.2 rule).

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
