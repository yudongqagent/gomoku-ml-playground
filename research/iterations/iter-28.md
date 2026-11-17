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
eval result. Value: **TBD in §1.1** (1.8 x measured std, rounded
to the nearest 0.5).

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

(TBD — run after the port is verified, before any v28 eval.)

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

### 1.3 Direction checks (NOT evals — identity/direction only)

(TBD — after implementation, before the battery: v28's search
(100 sims, seed 0) on the 9 v20 regression positions
(`regression_positions_v20.json`) to check the patch has the
intended DIRECTIONAL effect. Single-position diagnostics CANNOT
validate the version; the battery (§3) is the verdict.)

## 2. Loss analysis for iteration 28

(TBD.)

## 3. v28 evaluation

(TBD.)

## 4. Bar check (v28 vs v7)

(TBD.)

## 5. What this means

(TBD.)
