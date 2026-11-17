# Iteration 28 — v28: ours-only Black demotion (batch 9, H2)

**Status: PRE-REGISTERED — design fixed below (§1) BEFORE any eval of
v28. Step 0 (§1.1), implementation (§1.2), direction checks (§1.3),
fast loop (§2), and battery (§3) follow.**

**METHOD (carried over from batch 3/4/5/6/7/8).** The fast 10-game
loop stays the finding method: play 10 games, read every move, find
each game's losing move (solver/deeper-search verified, never
guessed — for rollout-leaf engines, playouts or exact computation,
not the v1 oracle alone, which shares the argmax-of-sum flaw),
patch the algorithm to reproduce the perfect move in those
positions (regression test), repeat — max 4 iterations, or stop when
an iteration finds no new patchable weakness class. The 4-clause bar
stays the validation gate. Fast-loop games use development seeds
distinct from the bar battery's pre-registered seeds (100/200).
**Fast loops run against v0 as well as v7** — v20's whole mechanism
came from the v0 loop (iter-20 §2.0). The full battery runs on the
FINAL version only. The 100-game battery is the verdict —
small-sample smokes mislead (v8, v11, v15's two n=40 false positives
in one iteration); single-position diagnostics cannot validate a
global search change.

**BASE ENGINE.** v28 is built on v22 (batch 8, H2: v14's PUCT MCTS
with the v1-kernel gradient leaf and Black-only gap-three
demotion), ported exactly from `research/batch-8` (cherry-picked
v22 commits, including the batch-7-h3 ml-tree port: `kernel_v1.py`,
v14/v20 leaf code, tests). The port is verified to reproduce v22's
behaviour BEFORE the condition changes (§1.2): v22's tactical must
be 29/29 on the corrected suite, and a 20-game smoke vs v0 must
show v22's colour-split pattern (strong Black, weak White). v17's
root rule (tie-prior-forcing) and threat check are NOT in v28: the
root-decision lever is EXHAUSTED, and the batch-9 plan tasks H2
with fixing the leaf's condition, not re-combining (v17: the
combination does not add). The fast loop therefore runs on v28
itself, the clean v1-leaf engine with a plain most-visited root
and the ours-only demotion.

## 0. The problem (from batch 8, H2 — v22)

v22 = v14 + gap-three demotion in BLACK's breadth only, in BOTH
terms of the gradient difference (iter-22 §1). Results (iter-22
§3–§4, REJECTED 2/4):

- vs v7: 54W-38L-8D, 0.5800, CI [0.4821, 0.6720] — clause 1 FAIL.
  Perfectly symmetric: Black 0.58, White 0.58 (first of any
  v1-leaf version).
- vs v0: 61W-30L-9D, 0.6550, CI [0.5577, 0.7409]. Black **0.93**
  (ZERO losses in 50 games), White **0.38** (30 losses) — the
  most extreme colour split in the program.
- Ladder Elo 147.2 [-13.2, 307.6] — clause 2 FAIL (v7: 190.8).
  Tactical 20/20 + 29/29 PASS. Swap vs v7 0.535, no edge.

The aggregate vs v0 (0.6550) matched the pre-registered
prediction (≈0.66) but through the OPPOSITE colour mechanism:
Black far exceeded the prediction (0.93 vs ≈0.72) while White
fell below it (0.38 vs ≈0.60). The decomposition (iter-22 §3.4,
clean: each comparison changes exactly one thing) is the
iteration's sharpest result:

- v14 (no demotion): Black 0.60, White 0.60.
- v20 (demote both colours, both terms): Black 0.72, White 0.45.
- v22 (demote Black only, both terms): Black 0.93, White 0.38.

So demoting BLACK's breadth (v14 → v22) is worth **+0.33 for
Black and −0.22 for White**; demoting WHITE's breadth on top
(v22 → v20) is worth **−0.21 for Black and +0.07 for White**.
The design's premise — "White's gap-three breadth is signal" —
is WRONG in aggregate against greedy play: White's gap-three
breadth is signal only as the OPPONENT's breadth (Black fearing
White's counterplay fully is worth +0.21 for Black). As White's
OWN breadth it is phantom (fast-loop games 7/9: 18 and 13
gap-three makers, the top 1.5% of the distribution — v22 falls
for the gap-saturated position where v20 does not).

The mechanism for White's −0.22 is the THEIRS-term demotion:
at White-to-move leaves, Black's gap-three breadth is demoted,
so White UNDER-FEARS Black's threats — the same mechanism as
fast-loop games 7/9, measured at battery scale against greedy
play, where Black's gap threes convert most directly. And
White's own full breadth does not save it: counting it fully
makes White overvalue its own gap-rich positions while the
theirs-term demotion makes White under-fear Black's. Both errors
point the same way — White over-optimistic — and the battery
prices them at 30 White losses.

The open follow-up (iter-22 §5, batch-9 plan H2): demote Black's
gap threes ONLY in the ours term (when Black is to move), never
in the theirs term — White would fear Black's gap threes fully,
which should repair White's −0.22 without touching Black's
+0.33. That is v28.

Note the second-order fact, recorded here because it constrains
the prediction: v22 also collapses against the terminal-leaf
ladder MCTS (0.25, worst of any v1-leaf version), which caps its
ladder Elo at 147.2. The leading hypothesis (iter-22 §3.6,
UNVERIFIED) is the full theirs-term: v22 counts the opponent's
gap-three breadth at full weight, so against the terminal-leaf
MCTS's forcing play v22 over-fears and plays passively. v28
makes the theirs term full in BOTH parities (Black's
theirs-breadth undemoted at White-to-move leaves), so UNDER THAT
HYPOTHESIS v28 could be WORSE vs the ladder MCTS, not better.
The same change that should repair White vs v0 may deepen the
vs-mcts collapse. This tension is stated in advance: the battery
decides, and the ladder is the clause at risk. The vs-mcts
forensics itself is H3's task (v29), not this iteration's.

## 1. Design and pre-registration

**Comparison target: v7** (the reference, batch-9 plan: ladder Elo
190.8, on master at `bf866f7`). Clause 1 is head-to-head vs v7
(normal rules, >= 100 games, alternating colours, win rate > 0.50
with Wilson 95% CI lower bound > 0.50); clause 2 is ladder Elo
strictly higher than v7's 190.8 (identical 20 games/opponent:
random, heuristic, mcts, mcts_sims=100); clause 3 is tactical pass
rate >= v7's (20/20 old suite + 29/29 hard suite, on the
CORRECTED suite — batch-9 program correction: (9,7) added to
winning-quiet-setup-2's expected list, commit 0ee51d8); clause 4
is all test suites green (ml pytest, npm test, tsc). **Secondary
head-to-heads, also pre-registered:** v28 vs v0 (100 games) and a
100-game swap match vs v7 under the tempo judge (the harness
default). Non-transitivity is recorded prominently. A new
strength champion ALSO needs to beat v18 on BOTH h2h-vs-v7 rate
(0.775) AND ladder Elo (636.4).

**The improvement: v28 = v14 with Black's gap-three makers
demoted ONLY in the OURS term of the gradient difference — i.e.
only at Black-to-move leaves, where Black's breadth IS the ours
term. White's breadth is never demoted (as in v22); Black's
breadth in the THEIRS term (at White-to-move leaves) is never
demoted either.** Everything else is v14/v22 unchanged: 100
sims, c_puct 1.5, v1 priors over all candidates (temperature
2.0 — the sharp prior is load-bearing at 100 sims; flattening
is catastrophic, iter-13 §2.4), win/block scans before search,
most-visited root child with RNG tie-breaks, terminal leaf
values identical. The ONLY change is the quiet-leaf gradient's
breadth rule, stated exactly:

- Leaf with Black to move: diff = breadth_gap(black) −
  breadth_v1(white). (IDENTICAL to v22's rule at this parity:
  Black's own breadth demoted, White's theirs-breadth full.)
- Leaf with White to move: diff = breadth_v1(white) −
  breadth_v1(black). (IDENTICAL to v14's rule at this parity:
  no demotion anywhere; Black's theirs-breadth counted FULLY.)

where `breadth_gap` is v20's `_breadth_v1_gap` (four-makers +
true open-three makers strong, gap-three makers at 0.25, v0
attack >= 1000 as the true/gap filter) and `breadth_v1` is
v14's undemoted `_breadth_v1` (all three-tier moves strong).
Equivalently: **v28 ≡ v22 at Black-to-move leaves and v28 ≡
v14 at White-to-move leaves** (modulo the re-derived tanh scale,
§1.1, which applies to both parities). The change vs v22 is
confined to White-to-move leaves, in both colours' games: when
the engine plays White, its own-turn leaves now fear Black's
gap threes fully; when it plays Black, its opponent-turn leaves
now rate Black's threats fully instead of discounting them.

**NOT the same as iter-22's rejected option 3.** Iter-22 §1
considered and rejected "demote only in the OURS term (the side
to move's own breadth)" — but that option was COLOUR-BLIND: it
would demote White's own breadth at White-to-move leaves (the
v20 White problem, unchanged). v28 is colour-conditional AND
term-conditional: the demotion applies only to BLACK's breadth,
and only when Black's breadth is the ours term. At
White-to-move leaves NOTHING is demoted. The four tested
combinations are now: v14 (no demotion), v20 (both colours,
both terms), v22 (Black only, both terms), v28 (Black only,
ours term only). The untested complement — Black only, theirs
term only — is not this iteration's task.

**Why this condition (the mechanism, from v22's battery).** The
decomposition prices demoting Black's breadth at +0.33 Black /
−0.22 White, and the two terms are separable: the +0.33 is an
OURS-term effect (the ply-10 fix — the v1 gradient's +4.0
preference for the losing cluster move came entirely from
Black's OWN gap-three makers at a Black-to-move leaf), while
the −0.22 is a THEIRS-term effect (White under-fearing Black's
threats at White-to-move leaves, where Black's breadth is the
theirs term). v28 keeps the ours-term demotion (Black's +0.33
should survive) and removes the theirs-term demotion (White's
−0.22 should be repaired: White fears Black's gap threes fully
and stops overvaluing its own gap-rich positions relative to
Black's threats). White's own breadth stays full, as in v22 —
the fast-loop saturation finding (games 7/9) says White's own
gap-three breadth is phantom at saturated positions, but that
is a separate, unpatched cost (a saturation threshold or
kill-square dedup would be its own iteration, not this one);
the battery measures whether the theirs-term repair alone
recovers White.

**Player-swap antisymmetry is DELIBERATELY BROKEN by this
condition, and that is pre-registered, not a bug.** v22's leaf
is antisymmetric under player swap (value(board, WHITE) =
−value(board, BLACK)) because each colour's breadth uses the
same rule in both terms. v28's does not: value(board, BLACK)
= 0.2·tanh((gap(B) − v1(W))/s) while value(board, WHITE) =
0.2·tanh((v1(W) − v1(B))/s), and −value(board, BLACK) =
0.2·tanh((v1(W) − gap(B))/s) ≥ value(board, WHITE) since
gap(B) ≤ v1(B). The break has a definite direction:
**White-to-move leaves are valued LOWER than the mirror of
Black-to-move leaves** — White's positions are worth less than
the antisymmetric value, because Black's full undemoted breadth
counts against White in the theirs term. That IS the intended
effect (White fears Black's gap threes fully). MCTS backup
remains valid: each board has a fixed player to move by
stone-count parity, so in a real game tree a node is only ever
evaluated with its correct player, and the backup (negate along
the path) is correct as long as each leaf returns the value for
the player to move at that leaf — which v28 does. Precedent:
v16's rollout leaf is colour-gated (when the engine plays
Black, v16 IS v7 exactly), so a non-antisymmetric leaf is not
new in this program. The tests assert the inequality direction
and boundedness, and assert that v28 matches v22 exactly at
Black-to-move leaves and v14 exactly at White-to-move leaves
(up to scale) — NOT player-swap antisymmetry.

**What does NOT change (pre-registered exclusions).** Broken-four
makers (20,006) stay STRONG for both colours: a broken four has
exactly one winning square, the same as a straight four — both
force a block identically; the defect is specific to gap THREE
makers (non-forcing, killable), not broken fours (forcing). The
developing tier [80, 4000) is unchanged. The cap stays ±0.2 — a
static heuristic must never outweigh an exact terminal signal. No
root-rule change (EXHAUSTED — v28 keeps v14's plain most-visited
root). No exploration change (CLOSED). No sims change (100 both
colours — more search is NOT monotonically better). No prior
change (sharp prior load-bearing). No threat check, no opening
book, no q-based root decision (CLOSED). No change to
`baseline.py` (FROZEN). No kernel re-sync (NumPy/WebGPU/Metal)
unless v28 passes the bar. The v1 kernel is used directly, not
re-ported; the v0 kernel is used as the filter, unchanged.

**The scale question (pre-registered rule, Step-0 value).** The
ours-only demotion changes the breadth-difference distribution,
so the tanh scale is re-derived from the V28 distribution by
v7's rule (scale ~= 1.8 x std) — the same rule v7 (4.0 from std
2.26), v14 (7.5 from std 4.09), v20 (5.5 from std 3.02), and v22
(7.0 from std 3.75) used — measured in Step 0 (§1.1) BEFORE any
v28 eval. Because the rule is PARITY-dependent, the Step-0
sample must cover BOTH parities, unlike iter-22's all-White-to-
move sample: same replay source (batch-v7-vs-v1.json, first 20
games), EVERY ply from ply 8 onward (0-indexed move index ≥ 8 —
alternating parity, so both Black-to-move and White-to-move
leaves are represented; the iter-22 sample's every-4th-ply rule
would give all-White-to-move again), quiet positions only,
pooled std across both parities. [CORRECTION, made before Step 0
ran: the first draft of this paragraph said "every 2nd ply from
ply 8 (plies 8, 10, 12, … — alternating parity)" — that is
arithmetically wrong, since even move indices are ALL White to
move; the corrected rule above (every ply ≥ 8) is the one Step 0
uses. No v28 eval had run when this was corrected.] The scale is
fixed here from the measured distribution, NOT tuned on any v28
eval result. Value: **8.0** (1.8 x std 4.54 = 8.17, rounded to the
nearest 0.5).

**What this change is meant to fix, and the expected effect
(stated in advance).** Against greedy v0: Black should HOLD near
v22's 0.93 (the ours-term demotion is untouched at Black-to-move
leaves, and Black's +0.33 was an ours-term effect), while White
should RECOVER toward v14's 0.60 from v22's 0.38 (the theirs-term
repair) — expected vs v0 ≈ 0.74–0.77 if the two effects hold.
Against v7 the expectation is genuinely unclear, and is stated
as such: v22 was 0.58/0.58; the theirs-term repair changes
White-to-move leaves in BOTH colours' searches, and whether the
White recovery outweighs any Black cost against a searching
opponent is what the battery measures. The ladder should improve
over v22's 147.2 on the vs-heuristic leg if White recovers, but
the vs-mcts leg may stay collapsed or worsen under the over-fear
hypothesis (§0) — clause 2 is the risk. Tactical should stay
20/20 + 29/29 (v22 was perfect; the change only alters
White-to-move leaf values, and tactical positions are mostly
forced).

**What would count as confirmation vs refutation (pre-registered,
the 100-game battery is the verdict).**
- CONFIRMATION: vs v0 White recovers to ≥ 0.55 (from v22's 0.38)
  while Black holds ≥ 0.85 (v22: 0.93) — the terms separate as
  predicted, and the theirs-term demotion was the cause of
  White's −0.22.
- REFUTATION (White): vs v0 White stays ≤ 0.45 — the theirs-term
  demotion was NOT the (main) cause of White's −0.22; White's
  own full breadth (the saturation phantom, games 7/9) or
  something else dominates, and the ours-only condition does not
  repair White.
- REFUTATION (Black): vs v0 Black falls to ≤ 0.80 — the
  ours-term demotion alone does not carry Black's edge, i.e.
  v22's Black strength came partly from the theirs-term demotion
  at White-to-move leaves (White under-fearing Black helped Black
  too), and the terms do NOT separate.
- Either refutation kills the condition as a mechanism claim even
  if the aggregate vs v0 happens to rise; the bar clauses are
  judged on their own terms in §4.

**Eval seeds (pre-registered):** h2h vs v7: v28=100 / v7=200; h2h
vs v0: v28=100 / v0=200; ladder seed 100, 20 games/opponent,
mcts_sims=100; swap 100 games vs v7, balanced placement,
placement_seed 11, tempo judge (the harness default; recorded in
the JSON). Harness commands follow iter-7 §3 exactly
(`fast_loop_match.py`, `eval_scoring.py` for the ladder,
`eval_tactical_full.py` for tactical, `selfplay_batch.py
--swap-rule` for swap). Fast-loop dev seeds 1015/2015 (iter 1
vs v0) and 1016/2016 (iter 2 vs v7) are distinct from the
battery's 100/200 AND from iter-20's loop seeds (1011/2011,
1012/2012) and iter-22's (1013/2013, 1014/2014), so these are
fresh games.

### 1.1 Step 0 — cost and distribution measurement

`ml/scripts/bench_v28_leaf.py` on 1,343 boards replayed from
batch-v7-vs-v1.json (first 20 games, EVERY ply from ply 8
onward — the iter-28 §1 sampling rule, both parities; 1,151
quiet: 577 Black to move, 574 White to move). Median of 3 rounds
x 50 calls.

| Call | Median ms/call |
|---|---:|
| v28 `_attack_gradient_v1_gap_ours` (full) | 6.89 (min 5.98, max 7.09) |
| v28 `_gradient_v1_gap_ours_leaf_value` (full) | 8.23 (min 7.87, max 9.44) |
| v22 `_attack_gradient_v1_gap_black` (full) | 8.22 (min 6.21, max 11.31) |
| v20 `_attack_gradient_v1_gap` (full) | 6.11 (min 4.42, max 7.35) |
| v14 `_attack_gradient_v1` (full) | 4.37 (min 3.88, max 5.83) |
| v7 `_attack_gradient` (full) | 2.12 (min 1.92, max 2.13) |
| v6 `_terminal_leaf_value` | 1.79 (min 0.59, max 1.83) |

The v28 gradient costs ~6.9 ms per quiet leaf on this run vs
~8.2 ms for v22's (absolute numbers are contention-noisy; the
ordering is what matters: v28 ≈ v22 ≈ v20, one extra kernel call
over v14). At up to ~100 leaf calls per move that is at most
~+0.3 s/move over v14, ~+10 min per 100-game match. Affordable
— the same cost class as v22.

Ours-only breadth-difference distribution on the 1,151 quiet
positions (pooled): mean -2.89, std 4.54, p10 -9.00, p50 -2.00,
p90 +1.50; |diff| >= 2 in 59.9%, >= 4 in 38.7%, >= 6 in 25.1%.
Per parity: Black to move (n=577) mean -2.94, std 4.32; White to
move (n=574) mean -2.85, std 4.75 — the two parities have close
spreads, so a single pooled scale is reasonable for both. (On
the same sample: v14 mean -2.11, std 4.63; v20 mean -1.51, std
3.04; v22 mean -1.87, std 4.20.) Scale: 1.8 x 4.54 = 8.17 ->
**8.0** (pre-registered above). The pooled mean (-2.89) is more
negative than v22's (-1.87) — the intended direction: at
White-to-move leaves v28 counts Black's full undemoted breadth
in the theirs term, so White's positions look worse for White
than under v22. Mean gap-three makers per position: Black 2.41,
White 2.15. The ours-only demotion changes the breadth
difference vs v22 in 27.9% of quiet positions (mean |change|
1.03), vs v14 in 22.7% (mean 0.78), and vs v20 in 56.4% (mean
1.81) — a substantial revision of the signal vs v22, not a
tweak at the margin, and concentrated exactly where the design
says it should be (White-to-move leaves).

### 1.2 Implementation (pre-registered)

- `ml/gomoku_ml/mcts_heuristic.py`: `_attack_gradient_v1_gap_ours`
  (one `attack_v1_both` + one `attack_both` call; at Black-to-move
  leaves Black's breadth via `_breadth_v1_gap` and White's via
  `_breadth_v1`; at White-to-move leaves both via `_breadth_v1`;
  tanh scale from §1.1, cap ±0.2),
  `_gradient_v1_gap_ours_leaf_value` (terminal values identical
  by construction), registered as `leaf_mode="gradient_v1_gap_ours"`
  in `_LEAF_FUNCTIONS`. Both breadth functions are reused
  unchanged; the v1 kernel (`kernel_v1.py`) and the v0 kernel
  (`kernel_numpy.py`) are used directly, unchanged.
- `ml/gomoku_ml/scoring.py`: version 28 in `ENGINE_VERSIONS`
  (`{"name": "scoring-v28", "defence_weight": 1.0, "threat_check":
  False, "mcts": True, "mcts_sims": 100, "leaf_mode":
  "gradient_v1_gap_ours"}`), dispatching through the existing
  generic `HeuristicMCTS` path (no new dispatch code).
- Tests: `ml/tests/test_mcts_heuristic.py` gains v28 leaf tests
  (ours-only unit cases on the ply-10 board: v28 == v22 at
  Black-to-move leaves and v28 == v14 at White-to-move leaves
  up to scale; terminal identity with v14/v20/v22; the
  antisymmetry-break inequality direction + boundedness; scale
  test); `ml/tests/test_regression_v28.py` +
  `ml/gomoku_ml/regression_positions_v28.json` record the §2
  critical positions (plus the v20/v22 positions re-checked in
  §1.3).

**Port verification (before the condition changed).** v22 was
ported from `research/batch-8` by cherry-pick (commits 8516a6e,
the batch-7-h3 ml-tree port, and 39792ce, the v22
implementation, code only — `iter-22.md` stays on batch-8) and
verified to reproduce v22's behaviour BEFORE any v28 change:
v22's tactical is 20/20 old + 29/29 hard on the CORRECTED suite
(`eval_tactical_full.py --version 22 --suite both --seed 100`),
and a 20-game smoke vs v0 (seeds 100/200) reproduces the
batch-8 battery's first 20 games EXACTLY — 14W-5L-1D, 0
mismatches in moves, winners, and ply counts, colour split
Black 9W-0L-1D / White 5W-5L-0D (the v22 pattern: strong Black,
weak White). The full ml suite on the ported tree was 338
passed / 7 skipped / 38 xfailed with 2 failures, BOTH caused by
the coordinator's batch-9 suite correction (commit 0ee51d8),
not the port: `test_hard_suite_expected_moves_are_exact` (the
(9,7) addition is a longer-route win the setup/VCF computation
does not find) and `test_hard_suite_v1_baseline_reproduces`
(v1 plays (9,7), so it now passes 13/29, not the recorded 12).
Both tests were fixed on this branch with documented
justifications (see the `PLAYOUT_VERIFIED_MOVES` exception and
the V1_PASSED 12→13 comment in `test_tactical_hard.py`); after
the fix, `test_tactical_hard.py` + `test_mcts_heuristic.py`
(including the 6 new v28 leaf tests) run 78 passed / 1 xfailed.

### 1.3 Direction checks (NOT evals — identity/direction only)

After implementation and before the battery, v28's search (100
sims, seed 0) is run on the 9 v20 regression positions
(`regression_positions_v20.json`) to check the patch has the
intended DIRECTIONAL effect — whether it plays the oracle-verified
save where v14 played the loser, and whether it keeps v22's saves.
These are single-position diagnostics: they CANNOT validate the
version (standing lesson), they only check the patch does what it
was designed to do at the positions that motivated it. The battery
(§3) is the verdict.

**Results (measured, seed 0).** v28 plays the save at **0 of 9**
positions; v22 at **2 of 9**; v20 at **3 of 9**; v14 at **0 of 9**:

| Position | v28 plays | v22 plays | v20 plays | v14 plays | Expected (save) | Verdict |
|---|---|---|---|---|---|---|
| bl-ply8 | (7,10) | (7,10) | (7,10) | (7,10) | (8,5) | miss (all) |
| bl-ply10 | (8,5) | (8,5) | (8,5) | (7,9) | (5,12) | miss (v28 avoids the loser, not the save) |
| bl-ply12 | (6,11) | (6,11) | (5,12) | (6,11) | (5,12) | miss (v28 keeps v22's regression) |
| bl-ply14 | (4,9) | (6,9) | (6,9) | (4,9) | (6,9) | **miss (v28 REGRESSES vs v22)** |
| bl-ply16 | (9,4) | (9,4) | (9,4) | (9,4) | (6,9) | miss (all — prior gap) |
| fl1-g6-ply28 | (7,5) | (4,9) | (0,9) | (7,5) | (4,9)/(0,9) | **miss (v28 REGRESSES vs v22)** |
| fl1-g6-ply34 | (6,11) | (6,11) | (6,11) | (6,11) | (4,2) | miss (all — starvation) |
| fl1-g5-ply27 | (11,9) | (11,9) | (6,10) | (8,9) | (7,6)/... | miss (all — beyond-horizon; v28 same as v22) |
| fl2-g5-ply65 | (11,10) | (12,11) | (11,9) | (11,9) | (13,6)/... | miss (all — beyond-horizon; v28 plays a new move) |

(Coordinates are (x, y) throughout, as in the walk-back JSONs
and the iter-22 §1.3 table — the v22/v20/v14 columns reproduce
iter-22 §1.3 exactly, which cross-checks the driver AND the port:
the ported v22/v20/v14 play identically to batch 8's.)

**v28 regresses BOTH of v22's saves.** At bl-ply14 v28 plays
(4,9) like v14 instead of the forcing open three (6,9); at
fl1-g6-ply28 v28 plays (7,5) like v14 instead of either save
(v22's winning (4,9) or v20's drawing (0,9)). Both positions are
Black to move, where v28's leaf IS v22's leaf — so the
regressions are NOT 1-ply gradient effects: they are deeper-tree
effects of the theirs-term repair, the mirror image of v22's
bl-ply12 regression. Iter-22 §1.3 attributed bl-ply12 to
White's full gap-three breadth at every White-to-move leaf
making White's replies look better for White and the save's
lines worse for Black on backup. Under v28 the sign flips:
Black's gap-three breadth counts fully at every White-to-move
leaf, so White's replies look WORSE for White (better for Black
on backup), and the search prefers the flashier move over the
forcing save. The condition is a trade, not a pure win — the
same lesson as v22's §1.3, now in the other direction. Whether
that cost matters in aggregate is what the battery measures; no
single-position diagnostic can say. bl-ply12 stays a miss (v28
does not recover v20's save there), and the two White positions
are missed by all four engines, with v28 playing a new move at
fl2-g5-ply65 ((11,10), vs v22's (12,11) and v20/v14's (11,9)) —
whether it is better is unknown (it is not an oracle-verified
save) and is not claimed. All 9 positions are xfails in
`test_regression_v28.py` — v28 plays 0 saves, like v14; the
file pins the direction-check result, not a pass.

## 2. Loss analysis for iteration 28

The fast loop runs on v28 itself (the pre-registered engine),
against v0 AND v7. Its purpose is finding patchable weakness
classes the ours-only demotion introduces or leaves open — not
validating v28 (the battery does that).

### 2.1 Fast-loop iteration 1 — 10 games vs v0

**Batch:** `research/iterations/fastloop1-v28-vs-v0.json` (seeds
v28=1015 / v0=2015 — development seeds, distinct from the bar
battery's 100/200 AND from iter-20's loop seeds 1011/2011 and
1012/2012 and iter-22's 1013/2013 and 1014/2014, so these are
fresh games). **4W-5L-1D, win rate 0.4500, CI [0.2014, 0.7263]**,
358.0 s. Per colour: as Black 4W-0L-1D (51, 225, 49, 143, 111
plies), as White **0W-5L-0D** (25, 29, 29, 21, 21 plies). Mean
game length 70.4 plies. The White collapse is the headline — all
five White games lost, all fast (21–29 plies) — but 10 games is
a smoke, not a verdict, and the control below shows these seeds
are hard for White for v22 too. Classifier: 5/5 losses via open
four, 0 missed blocks, 0 clean missed preventions in 31
threatened turns (losses). Defence is still perfect; the losses
are lost races, as in every batch.

**The losing move in each game** (walk-back, corrected oracle
method, seeds 0,1,2 — then ENGINE-SWAP VERIFIED with fresh v28
vs fresh v0, the iter-20 §2.0 instrument; the oracle alone is
not trusted for MCTS games, and this iteration shows why):

| Game | Last preventable turn | v28 played (oracle) | Perfect move (oracle) | Engine-swap verdict |
|---|---|---|---|---|
| 1 (25 plies, White) | ply 17 | (4,5) | (7,6) draws 1/3 | **WITHDRAWN** — save loses +9, control loses +7 |
| 3 (29 plies, White) | ply 15 | (7,9) | (10,9) draws 3/3 | **WITHDRAWN** — save loses +13, control loses +13 |
| 5 (29 plies, White) | ply 15 | (9,7) | (9,10) draws 3/3 | **WITHDRAWN** — save loses +13, control loses +13 |
| 7 (21 plies, White) | ply 15 | (6,8) | (6,4) wins 1 / draws 2 | **CONFIRMED** — save wins +28, control loses +5 |
| 9 (21 plies, White) | ply 15 | (8,8) | (4,8) draws 1 / wins 2 | **WITHDRAWN** — save wins 1/3 (L+23, W+34, L+69), control loses +5 |

**Four of the five oracle verdicts are WITHDRAWN.** Games 3 and
5 are the sharpest case: the oracle's save loses in EXACTLY as
many plies as the control (+13 both) — the positions were
already lost by the walk-back window, and the v1-vs-v1 oracle's
"draw" reflects 1-ply dynamics, not MCTS-vs-greedy play. Game 1:
the save loses +9 vs the control's +7 — the save only delays
the loss by 2 plies. Game 9: the save wins 1/3 but loses 2/3
(and its losses run +23/+69 vs the control's +5 — it delays the
loss substantially without saving the game at 2 of 3 seeds).
This is the iter-22 §2.2 caveat again: the v1 walk-back oracle
invents saves for MCTS games. Only game 7 survives as a
verified losing move.

**Game 7 ply 15 — the clean case, but SHARED by all engines.**
Actual (6,8): v28-fresh, v22-fresh, v20-fresh, v14-fresh, and
v7-fresh ALL play it; the save (6,4) wins 3/3 under engine-swap
(+28 plies, a slow grind) while the actual loses 3/3 in +5.
v28-root: actual 19 visits, our-q −0.016; save 12 visits, our-q
−0.077 — priors nearly equal (0.1117 vs 0.1171, the save's is
HIGHER), so this is a VALUE error, not starvation: the tree
prefers the loser on q. The 1-ply gradients after each move do
NOT separate them (for White: v28 +0.197 vs +0.195, v22 +0.198
vs +0.198, v14 +0.195 vs +0.192) — the save's value materialises
deeper than the 100-sim horizon, where the search cannot see
that (6,4) wins slowly while (6,8) loses fast. After the actual,
White holds 19 gap-three makers — the same saturation class as
v22's fast-loop games 7/9 — but v20 (which demotes White's gap
threes) and v7 (v0 kernel) miss it too, so the saturation alone
does not explain it. This is the shared 100-sim search-horizon
limit (the same class ML batch 1's v24 finding documented: v7
makes the same miss as v24), not a v28-specific defect — and
not a patch target for this iteration: no leaf change
distinguishes these two moves at 1 ply, and the root-decision
lever is EXHAUSTED.

**Control: v22 at the same seeds.** v22 vs v0, seeds 1015/2015
(`~/workspace/b9h2-logs/control-v22-vs-v0-1015.json`, not
committed): **6W-4L-0D, Black 5W-0L-0D, White 1W-4L-0D**. v22-as-
White also loses 4 of 5 at these seeds — the White collapse is
largely a SEEDS effect, not a v28 effect. The v28-vs-v22
difference at these seeds is exactly 2 games: game 2 (v28 draws
as Black in 225 plies, v22 wins in 63) and game 9 (v28 loses as
White in 21, v22 wins in 52). Ten games cannot attribute a
2-game difference to the condition; the battery decides.

**No patch from iteration 1; the loop continues to iteration 2
vs v7** (per the method: loops run against v0 AND v7) before the
stopping rule is applied. The one confirmed losing move is
shared by all engines; the four withdrawn verdicts show the
oracle inventing saves; the control shows the White collapse is
not v28-specific. The confirmed position (fl1-g7-ply15) is
recorded in `regression_positions_v28.json` as an xfail — v28
misses the save, as do v22/v20/v14/v7.

### 2.2 Fast-loop iteration 2 — 10 games vs v7

**Batch:** `research/iterations/fastloop2-v28-vs-v7.json` (seeds
v28=1016 / v7=2016 — development seeds, distinct from the bar
battery's 100/200). **6W-2L-2D, win rate 0.7000, CI [0.3968,
0.8922]**, 647.0 s. Per colour: as Black 4W-0L-1D (games 0, 2, 4,
6, 8), as White 2W-2L-1D (games 1, 3 won; 5, 9 lost; 7 drawn).
Classifier: 2/2 losses via open four, 0 missed blocks, 0 clean
missed preventions in 4 threatened turns (losses). Defence is
still perfect; the losses are lost races, as in every batch.

**The losing move in each game** (walk-back, corrected oracle
method, seeds 0,1,2 — then ENGINE-SWAP VERIFIED with fresh v28
vs fresh v7, the iter-20 §2.0 instrument):

| Game | Last preventable turn | v28 played (loses 3/3) | Perfect move (oracle) | Engine-swap verdict |
|---|---|---|---|---|
| 5 (19 plies, White) | ply 13 | (8,7) | (10,7) draws 2/3 | **WITHDRAWN** — save D+211/L+45/L+49, control loses +5 |
| 9 (21 plies, White) | ply 15 | (7,11) | (7,9) draws 2/3 | **WITHDRAWN** — save loses +5, control loses +5 |

**Both oracle verdicts are WITHDRAWN.** Game 9 is the cleanest
withdrawal: the save loses in exactly +5 plies, identical to the
control — the position was already lost by the walk-back window,
and v7-fresh playing the save (the only engine to do so) is luck
as much as skill. Game 5 is subtler: the save draws once (a
211-ply grind) and loses twice, but its losses run +45/+49
plies against the control's +5 — the save delays the loss
hugely without saving the game at 2 of 3 seeds, so the verdict
is withdrawn under the ≥2/3 rule. Fresh-engine check at ply 13:
v28-fresh plays (10,9) (NOT the match's actual (8,7) — the
fresh RNG stream diverges), v22/v14/v7 play the actual (8,7),
v20 plays (10,8), and NOBODY plays the oracle-save (10,7). The
position is a coin-flip among several moves under 100-sim
search; there is no v28-specific signal in either loss.

**No patch from iteration 2; loop stops here (stopping rule).**
Iteration 2 found no new patchable weakness class: both losses
are withdrawn oracle verdicts on positions that were already
lost (game 9) or horizon-limited coin-flips (game 5), and
neither shows a v28-specific mechanism. Combined with
iteration 1 — whose one confirmed losing move is shared by all
engines — the fast loop's finding is that v28-as-pre-registered
has no patchable v28-specific weakness class at these seeds.
The battery (§3) measures the aggregate; v28-as-batteried is
v28-as-pre-registered.

### 2.3 Fast-loop iteration 3 — NOT RUN (loop stopped after iteration 2)

### 2.4 Fast-loop iteration 4 — NOT RUN (loop stopped after iteration 2)

### 2.5 Aggregate counts (secondary)

Fast loop pooled (20 games, 7 v28 losses):

- Losses via open four: **7/7** (5 vs v0 + 2 vs v7)
- Missed blocks: **0** in all games
- Clean missed preventions: **0** in 35 threatened turns (losses)
- Losses with a verified losing move (oracle + engine-swap):
  1 of 7 (fl1-g7); 6 of 7 oracle verdicts WITHDRAWN by
  engine-swap (fl1-g1, fl1-g3, fl1-g5, fl1-g9, fl2-g5, fl2-g9)
- Losses where the confirmed move is v28-specific: **0** (the
  one confirmed move, fl1-g7-ply15, is played by all engines —
  a shared 100-sim search-horizon value error)

Defence is solved, as in every batch: every loss is an open-four
race lost, never a defensive error.

## 3. v28 evaluation

**Engine note:** the fast loop produced NO patch (§2.2), so the
battery runs on the pre-registered v28 exactly as implemented in
§1.2 (v14's MCTS + the ours-only Black-demoted v1 gradient leaf,
scale 8.0, cap ±0.2). Tactical was measured on this engine before
the battery (§3.2) and is re-run in §3.5 for the record.

### 3.1 Smoke — NOT RUN (no smoke; the fast loop is the smoke)

Per the batch-5 method the 10-game fast loop (§2.1-2.2) is the
only small-sample evidence, and small-sample smokes mislead (v8,
v11, v15's two n=40 false positives) — the 100-game battery is
the verdict.

### 3.2 Tactical suites — pre-battery (re-run on the final engine in §3.5)

Old 20-position suite, full-dispatch driver
(`ml/scripts/eval_tactical_full.py --version 28 --suite both
--seed 100`): **v28 20/20**. Hard suite (29 positions): **v28
29/29** — a perfect score, matching v7's 20/20 + 29/29 on the
CORRECTED suite. The ours-only demotion costs nothing
tactically: the direction-check regressions (§1.3) are
search-level move-choice differences at unforced positions, not
tactical failures.

### 3.3 Head-to-head vs v7 (100 games, seeds 100/200) — FINAL BATTERY

**v28 beats v7: 53W-39L-8D, win rate 0.5700, 95% CI [0.4722,
0.6627]**, 5189.8 s. File:
`research/iterations/batch-v28-vs-v7.json` (seeds 100/200). Per
colour: as Black 26W-19L-5D (0.5700), as White 27W-20L-3D
(0.5700) — a PERFECTLY SYMMETRIC split again, like v22's
0.58/0.58 (v14: 0.72/0.48; v20: 0.61/0.53). The ours-only
demotion keeps v22's colour balance exactly but buys nothing
net against the searching opponent: the aggregate (0.5700) sits
1 game per 100 BELOW v22's 0.5800 and 3 below v14's 0.6000, and
matches v20's 0.5700. Mean game length 58.44 plies (min 18, max
225); wins mean 41.0 plies, losses mean 48.0 — v28 wins slightly
FASTER than it loses, as in v22. 86 distinct 12-ply lines in 100
games (no transposition collapse). Loss classifier: 39/39 losses
via open four, **0 missed blocks**, 172 threatened turns in
losses, **0 clean missed preventions** — defence is solved.
**Clause 1 FAILS** (0.5700 > 0.50, but the CI lower bound 0.4722
< 0.50 — the edge over v7 is not established).

### 3.4 Head-to-head vs v0 (100 games, seeds 100/200) — FINAL BATTERY

**v28 beats v0: 52W-43L-5D, win rate 0.5450, 95% CI [0.4475,
0.6391]**, 2750.0 s. File:
`research/iterations/batch-v28-vs-v0.json` (seeds 100/200). Per
colour: as Black **39W-6L-5D (0.8300)**; as White **13W-37L-0D
(0.2600)** — the worst White vs v0 of any v1-leaf version (v14:
0.60, v20: 0.45, v22: 0.38). The aggregate (0.5450) is below
v22's 0.6550, v14's 0.6000, and v20's 0.5850: the ours-only
demotion is the WORST of the four tested combinations vs v0.
Mean game length 53.41 plies (min 21, max 225); wins mean 58.0
plies, losses mean 27.9 — v28 wins very slow grinds and loses
fast races, the familiar pattern in its extreme form. White's 37
losses average 25.7 plies, 26 of them in ≤30 plies. 66 distinct
12-ply lines (v0's greedy play funnels the lines). Loss
classifier: 43/43 losses via open four, **0 missed blocks**, 237
threatened turns in losses, **0 clean missed preventions** —
defence is still solved.

**Both pre-registered refutations trigger — and the mechanism is
the OPPOSITE of the prediction.** White 0.26 ≤ 0.45 (REFUTATION
(White)): the theirs-term demotion was NOT the cause of White's
−0.22 — removing it made White WORSE (0.38 → 0.26), not better.
Black 0.83 > 0.80, so REFUTATION (Black) does NOT trigger
(narrowly): the ours-term demotion carries most of Black's edge
(0.93 → 0.83), but not all of it. The decomposition (clean: each
comparison changes exactly one thing) now prices the two terms
separately:

- v14 (no demotion): Black 0.60, White 0.60.
- v20 (demote both, both terms): Black 0.72, White 0.45.
- v22 (demote Black, both terms): Black 0.93, White 0.38.
- v28 (demote Black, ours term only): Black 0.83, White 0.26.

So the OURS-term demotion alone (v14 → v28) is worth **+0.23 for
Black and −0.34 for White**, while the THEIRS-term demotion on
top (v28 → v22) is worth **+0.10 for Black and +0.12 for
White**. The design's premise is EXACTLY BACKWARDS for White:
the theirs-term demotion HELPS White (+0.12), and the ours-term
demotion is what HURTS White (−0.34). The mechanism, stated as
the best explanation (not proven — the untested complement,
Black-only theirs-term demotion, would be the controlled test):
when White plays, Black's demoted breadth at Black-to-move
leaves — the OPPONENT's turns in White's search — makes Black's
replies look weak, so White under-fears Black's counterplay in
its own search and plays into Black's threats; the theirs-term
demotion (v22) partially compensates by making White's
own-turn leaves look better (Black's threats discounted),
restoring White's willingness to play counterplay. Removing it
(v28) leaves White with the worst of both: under-fearing Black
at Black's turns while over-fearing Black at its own turns.
The deeper lesson (§5): v28's leaf is the first INCOHERENT leaf
in the program — it breaks player-swap antisymmetry WITHIN a
game (v16's colour gate is per-game, so v16's leaf is coherent
within a game; v28's is per-leaf). The battery prices that
incoherence at White 0.26.

### 3.5 Tactical suites — FINAL BATTERY (re-run on the final engine)

**v28: 20/20 old + 29/29 hard** (full-dispatch driver,
`ml/scripts/eval_tactical_full.py`, `--seed 100`, run on the
final engine after the battery). Identical to the pre-battery run
(§3.2) and to v7's scores on the CORRECTED suite. The ours-only
demotion costs no tactical accuracy: every forced win, block,
prevention, setup, and VCF sequence is still found. **Clause 3
PASSES.**

### 3.6 Ladder (20 games/opponent, mcts_sims=100, seed 100) — FINAL BATTERY

**Ladder Elo: -34.9, interval [-183.3, 113.6]** (anchors:
random=-800, heuristic=0; relative to this ladder only).
Per-opponent (20 games each, seed 100):
- vs random: 20W-0L-0D (1.0000; as Black 10W-0L-0D, as White
  10W-0L-0D).
- vs heuristic (= v0): 9W-11L-0D (0.4500, CI [0.2582, 0.6579];
  as Black 8W-2L-0D, as White 1W-9L-0D) — the White collapse
  (§3.4) shows here too, at the ladder's 20-game scale.
- vs mcts: 13W-7L-0D (0.6500, CI [0.4329, 0.8188]; as Black
  8W-2L-0D, as White 5W-5L-0D).
Log: `~/workspace/b9h2-logs/ladder-v28.log` (raw); the EVAL_JSON
is committed as `research/iterations/ladder-v28.json`.
**Clause 2 FAILS** (-34.9 < v7's 190.8; the interval's upper
bound 113.6 does not reach 190.8). The Elo is identical to
v14's (-34.9) — the vs-heuristic leg (0.45) is the binding
constraint, and it is the WORST vs-heuristic of any v1-leaf
version at the ladder scale (v14: 9-11 too — the same 0.45, in
fact; v20: 0.65, v22: 0.70).

**The vs-mcts collapse did NOT reproduce — and that refutes
v22's leading hypothesis.** v28 beats the ladder MCTS 13-7
(0.65), matching v20's 0.65 and well above v7's own 0.40 and
v22's 0.25 (the worst of any v1-leaf version). Iter-22 §3.6
hypothesized the collapse came from the FULL theirs-term (v22
counting the opponent's gap-three breadth at full weight, so it
over-fears and plays passively). v28 counts MORE opponent
breadth at full weight than v22 (Black's theirs-breadth is
undemoted at White-to-move leaves) yet beats the MCTS — so the
over-fear hypothesis is REFUTED as the cause of v22's collapse.
The difference between v22 and v28 is ONLY the White-to-move
leaves (v28 ≡ v14 there), so the collapse tracks the
THEIRS-TERM DEMOTION itself: demoting Black's breadth at
White-to-move leaves is what loses to the terminal-leaf MCTS,
not counting it fully. The mechanism is H3's forensics question
(v29); this battery constrains it to the theirs-term demotion.

### 3.7 Swap match vs v7 (100 games, balanced, placement_seed 11, tempo judge) — FINAL BATTERY

**v28 beats v7 under the swap rule: 50W-44L-6D, win rate
0.5300, 95% CI [0.4329, 0.6249]**, 4589.2 s. File:
`research/iterations/batch-v28-vs-v7-swap.json` (seeds 100/200,
no resume). Per colour: as Black 30W-16L-4D (0.6400); as White
20W-28L-2D (0.4200) — the White weakness (§3.4) persists under
swap, unlike v23's swap White (0.78). The swap match (0.53) is
essentially identical to v22's (0.535) and to the normal-rules
h2h (0.57): the swap does not rescue v28, and the swap choice
does not neutralise v28's colour asymmetry the way it did for
v23 (Black won 58/100 here, vs 41/100 in v23's swap). Both
engines choose colours identically (tempo judge is shared):
second player took Black 38 / White 12 each. v28 as opener:
0.48; v28 as second player: 0.58 — the colour choice is worth
about +0.10 to whoever holds it, as the swap rule intends.
Mean game length 53.32 plies (min 16, max 225); v28's 44 losses
average 40.6 plies. 97 distinct 12-ply lines from 23 distinct
opening placements. Loss classifier: 44/44 via open four,
**0 missed blocks**, 179 threatened turns, **0 clean missed
preventions** — defence is solved across the entire battery
(pooled: 126/126 losses via open four, 0 missed blocks, 0 clean
missed preventions in 588 threatened turns).

## 4. Bar check (v28 vs v7)

(TBD.)

## 5. What this means

(TBD.)
