# Iteration 27 — v27: q-gap threshold tie-break for rollout-leaf engines (batch 9, H1)

**Status: IN PROGRESS — pre-registration (§1).** Comparison target is
v7 (the batch-9 reference: ladder Elo 190.8, on master at
`bf866f7`). This is the H1 workstream of batch 9 (H2 is v28,
ours-only Black demotion; H3 is v29, v22 vs-mcts forensics);
work happens on branch `research/batch-9-h1` in the
`gomoku-b9-h1` worktree, branched from `research/batch-9` @
`0ee51d8` (NOT from master — the coordinator branch already
contains the batch-9 plan, the auto-resume match drivers ported
from ML batch 1, and the corrected tactical suite).
v27 does NOT change the leaf (v18's rollout leaf, byte-identical),
does NOT change sims/priors/book (CLOSED), does NOT change
exploration constants (CLOSED without a new mechanism), and does
NOT include the threat check. The ONLY change from v23 is the
threshold on the root tie-break rule in §1 — a narrow refinement
of v23's mechanism, targeting v23's documented vs-v0/Elo
regression. It is NOT a reopening of the closed q-based
root-decisions lever (see §1.1).

**Method (carried over from batch 3/4/5/6/7/8):** the fast 10-game
loop — play 10 games, read EVERY move of every game, find THE
losing move per game (verified with the solver or a deeper search —
never guessed; for rollout-leaf engines, playouts or exact
computation, NOT the v1 walk-back oracle alone — iter-18 §2.1),
patch the algorithm so that, replayed to that critical position,
it plays the perfect move, and record each critical position +
correct move as a regression test. Then play 10 more games and
repeat. Max 4 fast-loop iterations before validation, or stop when
two consecutive iterations find no new patchable weakness class.
Patches must fix the STRUCTURAL cause, not the single position.
**The fast loop runs against v0 as well as v7** (iter-17 §5) —
each iteration plays 10 games vs v7 AND 10 games vs v0, and losses
from both opponents are walked back. The full 4-clause validation
battery (100-game matches + ladder) runs on the FINAL patched v27
only. Fast-loop games use development seeds (1001/2001, 1002/2002,
...) distinct from the bar battery's pre-registered seeds
(100/200), so no development game overlaps an eval game.

**Port note.** v23's code was ported by cherry-pick, not
re-derived: `8d43e8b` (v18 pre-registration), `f836e89` (v16 port),
`6bc865e` (v18), `02b2d47` (v23 pre-registration), `c4ba4d8` (v23
implementation) from `research/batch-8` / `research/batch-8-h3`.
Port verification (BEFORE any v27 work): v23's tactical on the
CORRECTED suite is 20/20 old + 29/29 hard at `--seed 100` —
reproducing the coordinator's verification, with the three
documented tie fixes choosing (8,4), (6,7), (7,5) and
winning-quiet-setup-2 choosing (9,7), now in the expected list.
The port is faithful; v27 builds on it.

## 1. Design and pre-registration

**Comparison target: v7** (primary, the batch-9 reference: ladder
Elo 190.8, on master at `bf866f7`). The bar's clause 1 is
head-to-head vs v7 (normal rules, 100 games, alternating colours,
win rate > 0.50 with Wilson 95% CI lower bound > 0.50), clause 2 is
ladder Elo strictly higher than v7's 190.8 (identical 20
games/opponent: random, heuristic, mcts, mcts_sims=100), clause 3 is
tactical pass rate >= v7's on the CORRECTED suite (old suite 20/20,
hard suite 29/29 — the suite fix added (9,7) to
winning-quiet-setup-2's expected list; batch-9 plan), clause 4 is
all test suites green (ml pytest, npm test, tsc).
**Secondary head-to-heads, also pre-registered:** v27 vs v0 (100
games) and a 100-game swap match vs v7 under the tempo judge (the
harness default). Non-transitivity is recorded prominently.
**Strength champion:** v18 (h2h vs v7 0.775, ladder Elo 636.4). A
new strength champion needs to beat v18 on BOTH h2h-vs-v7 rate
AND ladder Elo on the same battery.

**The improvement: v27 = v23 + a threshold on the q-gap.**
Everything else is v23 unchanged: 100 sims, c_puct 1.5,
v1 priors over all candidates, win/block scans before search,
most-visited root child, rollout leaf both colours (leaf mode
`rollout2_v1_both`, v16's function byte-identical including both
fast-loop patches). The threshold is the ONLY change from v23.

### 1.1 The threshold rule (pre-registered, exact)

v23's rule (iter-23 §1.1): at the root, on an exact top-visit
tie, break the tie by correctly-signed q (our-q = `−child.q`);
if q also ties exactly, RNG/first among the q-tied subset.
v27 gates that rule on the SIZE of the q gap:

1. **When it applies:** only at the root, only when two or more
   root children share the maximum visit count (an exact integer
   tie at the top) after all sims. If a unique child has the most
   visits, v27 behaves EXACTLY like v18/v23 — no q is consulted,
   no RNG is consumed. Near-ties (a 1-visit gap) are NOT ties and
   are not touched.
2. **The q gap:** among the visit-tied children, let our-q =
   `−child.q` (our perspective; `child.q` is the OPPONENT's —
   see iter-23 §1.1 rule 2, verified against the backup code).
   Let q_max be the highest our-q and q_second the
   second-highest DISTINCT our-q value among the tied children.
   If all tied children share the same our-q exactly, the gap is
   0. The gap is q_max − q_second.
3. **Fire condition:** the tie-break fires ONLY if gap > τ (the
   threshold, strictly greater). When it fires, v27 behaves
   EXACTLY like v23: play the child with the highest our-q; if
   two or more share q_max exactly (float equality), fall back
   to RNG choice among that subset when an RNG is set, else the
   first in dict order.
4. **Below threshold:** when gap ≤ τ, the tie-break does NOT
   fire at all — v27 behaves EXACTLY like v18: RNG choice among
   ALL visit-tied children when an RNG is set, else the first in
   dict order. (Not restricted to the q-tied subset: below the
   threshold q carries no reliable signal, so no q information
   is used.)

**Interpolation check (by construction).** τ = 0 reproduces v23
exactly: any positive gap fires, and a zero gap means all tied
children share q_max, so v23's rule-4 subset IS all tied
children. τ → ∞ reproduces v18 exactly (the rule never fires).
v27 sits strictly between them, and the threshold's only job is
to separate terminal signal from gradient noise.

**Why this retest is justified (and why it is not a reopening
of the closed lever).** v23's battery (iter-23 §3/§5) showed the
tie-break's two effects are separable and both real: it recovers
all 3 q-compression tactical ties (25/29 → 28/29 on the old
suite) and sets the best h2h vs v7 in the program (0.795) and the
strongest swap result (0.72) — but it costs ~5 games/100 vs v0
(0.91 vs v18's 0.97, all as Black; v18's rate lies outside v23's
CI) and 125 ladder Elo (511.5 vs 636.4, traceable to one
heuristic game: a Black draw became a Black loss). The fast loop
found the mechanism twice (iter-23 §2.2): the tie-break picks a
verified losing move over a verified saving move on a q gap of
0.016 — gradient noise, both q values small positives from the
rollout leaf's capped gradient, the threat beyond the 2-ply
horizon. The tactical fixes' gaps are 0.13–0.42 (v1bl-g2 0.130,
g26/g79 0.417) — exact terminal values (loser −1.000 vs saver
+0.250 at the leaf). The gap SIZE separates the two regimes:
signal gaps are ≥0.130, the documented noise gap is 0.016 — an
8x separation. The threshold is the minimal mechanism that keeps
the tie-break where q is exact and disables it where q is noise.
This is a refinement of v23's narrow mechanism, not a reopening
of q-based root decisions as a general mechanism: pure q is still
catastrophic (0.20), and v27 still never consults q except on an
exact visit tie with a large gap.

**What would refute the hypothesis.** If the selected
threshold's battery vs-v0 rate stays at or below v23's 0.91, or
its ladder Elo stays at or below v23's 511.5, the threshold
hypothesis is refuted: v23's vs-v0/Elo cost is not (only)
sub-threshold noise. If BOTH sweep thresholds fail the tactical
gate (§1.2), the threshold is not the separator the documented
gaps suggest.

### 1.2 Threshold sweep (pre-registered values and selection rule)

**Sweep values: τ ∈ {0.05, 0.10}.** Both sit strictly between
the documented noise gap (0.016) and the smallest documented
signal gap (0.130, v1bl-g2 — the binding minimum): 0.05 filters
the known noise by 3x, 0.10 by 6x, and both preserve all three
tactical fixes (0.130 > 0.10, margin 0.030 at τ=0.10). A value
≥0.13 would break the v1bl-g2 fix and is NOT swept. The sweep
tests whether the extra filtering of 0.10 over 0.05 matters in
real games (ties with q gaps in (0.05, 0.10] are the only
difference between the two candidates).

**Sweep protocol (per τ):**
- Tactical old + hard at `--seed 100` (the established protocol;
  ~30 s each).
- 40 games vs v0 (alternating colours, 100 sims), seeds
  3001/4001 — fresh dev seeds, distinct from the battery's
  100/200 and the fast loop's 1001+/2001+. vs v0 is the direct
  measure of the regression v27 targets (v23's cost was vs v0
  and the ladder heuristic, both weak opponents).

**Selection rule (stated BEFORE seeing any numbers):**
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
   closer to v23's demonstrated mechanism (both filter the
   documented 0.016 noise by ≥3x, and the smaller threshold keeps
   the tie-break active on more ties, matching v23's
   program-best h2h vs v7).

The sweep candidates are registered as TEMPORARY versions 90
(τ=0.05) and 91 (τ=0.10) so the standard drivers run them
unchanged; both entries are removed when v27 is registered with
the selected threshold. The 100-game battery is the verdict, not
the sweep: the sweep's job is to catch a catastrophic threshold
and to pick between two near-identical candidates, not to measure
strength (40 games cannot distinguish τ=0.05 from τ=0.10 finely
— the two differ only on ties with q gaps in (0.05, 0.10], which
are rare).

**Tactical expectation (pre-registered).** At both τ, all three
tie fixes fire (gaps 0.417, 0.417, 0.130 — all > 0.10) and no
other position changes (the rule only fires on exact visit ties,
and the suite's other positions are decided by unique visit
leaders or the win/block scans). **Expected tactical: 20/20 old
+ 29/29 hard at both τ** — clause 3 should PASS on the corrected
suite (winning-quiet-setup-2 now passes via the suite fix, which
is a program-wide correction, not a v27 effect).

**Game-strength expectation (pre-registered, honest).** vs v0
should recover toward v18's 0.97 (v23's 0.91 cost came from
sub-0.05-gap misfires, all as Black — the threshold converts
those from deterministic-wrong to 50/50 RNG coin flips, so the
recovery is partial, not total: a coin flip saves half the
misfires on average). Ladder Elo should recover toward v18's
636.4 IF the lost heuristic game was a noise-gap misfire (one
game decides the point estimate — the ladder is a noisy
instrument at 20 games/opponent). vs v7 should stay near v23's
0.795 (the tie-break's effect there was ~neutral-to-favourable).
No claim of beating v18 on both champion criteria is made: the
goal is v23's tactical recovery PLUS v18's game strength. The
battery decides.

**Cost (from v23's measurements — the leaf function is
unchanged, so the cost is identical by construction).** Sweep:
~15 min per τ (40 games vs v0) + tactical. Fast loop: ~1 h for
2 iterations. Battery: 100 games vs v7 ~2.1 h, vs v0 ~35 min,
swap ~55 min, ladder ~30 min, tactical ~1 min. Affordable.

**Eval seeds (pre-registered):** h2h vs v7: v27=100 / v7=200;
h2h vs v0: v27=100 / v0=200; ladder seed 100, 20
games/opponent, mcts_sims=100; swap 100 games vs v7, balanced
placement, placement_seed 11, tempo judge (the harness
default), seeds 100/200; tactical `--seed 100`. Fast-loop dev
seeds 1001/2001, 1002/2002, ... (iteration k uses 1000+k /
2000+k, per iter-8 §4's policy). Sweep seeds 3001/4001.

### 1.3 Mechanism check (post-implementation, pre-sweep)

After implementation and before any sweep game, v27 was run on
the 4 tactical positions (fresh-seed-0 root dumps,
`/tmp/dump_v27_mechanism.py`) and on the full tactical suites at
`--seed 100` at both τ, plus the two fl2-g6/g8 misfire positions
across seeds 0–9. This is an implementation check, not
pre-registration evidence and not an eval — it verifies the
dispatch does what §1 says it does.

**Root dumps reproduce iter-18 §3.1 / iter-23 §1.3 exactly**
(same ties, same q values, same priors — the search is unchanged;
only the final selection differs). A first version of the dump
script mixed up the two JSON formats (tactical positions use int
`to_move`, regression positions use `"black"`/`"white"` strings)
and dumped the wrong player to move; the fixed script reproduces
the documented states:

| Position | Root state (fresh seed 0) | Gap | τ=0.05 | τ=0.10 |
|---|---|---|---|---|
| prev-v1v0s-g26 | 4-way tie at 3 visits; expected our-q −0.5833, losers −1.000 | 0.4167 | FIRES → (9,4) EXPECTED | FIRES → (9,4) EXPECTED |
| prev-v1v1s-g79 | 7-way tie at 3 visits; expected our-q −0.5833, losers −1.000 | 0.4167 | FIRES → (7,9) EXPECTED | FIRES → (7,9) EXPECTED |
| prev-v1bl-g2 | 2-way tie at 8 visits; expected (7,5) our-q −0.5938 vs (7,9) −0.7244 | 0.1306 | FIRES → (7,5) EXPECTED | FIRES → (7,5) EXPECTED |
| winning-quiet-setup-2 | NOT a tie: 79 visits on (9,7) | — | (9,7), as v23 | (9,7), as v23 |

**Misfire positions (fl2-g6/g8 ply23):** the 2-way tie at 8
visits has gap 0.0155 at both positions (the documented 0.016).
v23 (τ=0.0) fires and picks the LOSING move at all 10 seeds
(10/10 (9,5) / (5,5) — deterministic-wrong, as documented).
v27 at BOTH τ does not fire and picks the saving move 7/10 seeds
((8,10)/(6,10) 7 times, the loser 3 times, identical picks at
τ=0.05 and τ=0.10) — the coin flip the hypothesis predicts
(7/10 vs a 50% expectation is within noise: P(X≥7) ≈ 0.17 for
Bin(10, 0.5)). A seed-0 pass on these positions is a coin flip
that landed well, NOT evidence the position is fixed — and the
v27 regression file records them accordingly (see §2).

**Full suites at `--seed 100`: both τ score 20/20 old + 29/29
hard**, with the three tie fixes choosing (8,4), (6,7), (7,5)
— identical choices to v23's port verification. Both thresholds
pass the tactical gate (§1.2 rule 1); the selection rule now
goes to the 40-game vs-v0 sweep matches.

### 1.4 Battery (the verdict — pre-registered seeds and commands)

On the FINAL v27 only (after the fast loop):

- 100 games vs v7, seeds 100/200 (clause 1).
- 100 games vs v0, seeds 100/200.
- 100-game swap vs v7, balanced placement, placement_seed 11,
  tempo judge (the harness default), seeds 100/200.
- Ladder 20 games/opponent (random, heuristic, mcts;
  mcts_sims=100), seed 100 (clause 2).
- Tactical old 20/20 + hard 29/29, full-dispatch driver
  (`ml/scripts/eval_tactical_full.py`, `player.choose_move` —
  NOT `eval_version.py`'s probe, which bypasses the MCTS
  dispatch) (clause 3).
- ml pytest + npm test + tsc green (clause 4).

**Integration.** `ml/gomoku_ml/mcts_heuristic.py` gains a
`tie_break_q_threshold` parameter on `HeuristicMCTS` (default
0.0 — v23's behaviour exactly, so v23 and all earlier versions
are unaffected) and `ml/gomoku_ml/scoring.py` gains ONLY the
`ENGINE_VERSIONS[27]` registry entry (v23's entry + the selected
`"tie_break_q_threshold"`) plus passing the flag through to
`HeuristicMCTS`. The v1 kernel (`kernel_v1.py`), the rollout
function, and v23's dispatch are checked out unchanged from
`research/batch-8-h3`. `baseline.py` is frozen and untouched.

### 1.5 Threshold selection — sweep results (the pre-registered rule applied)

**Tactical gate (§1.2 rule 1):** both τ pass — 20/20 old + 29/29
hard at `--seed 100`, identical choices to v23's port
verification (g26 → (8,4), g79 → (6,7), v1bl-g2 → (7,5),
winning-quiet-setup-2 → (9,7)). Both eligible.

**40-game vs-v0 sweep (seeds 3001/4001, alternating colours,
100 sims; aggregates recomputed from the raw records, matching
the drivers' headers exactly):**

| τ | Result | Win rate | 95% CI | As Black | As White | Time |
|---|---|---|---|---|---|---|
| 0.05 | 39W-0L-1D | 0.9875 | [0.8909, 0.9987] | 19W-0L-1D (0.975) | 20W-0L (1.000) | 1505.2 s |
| 0.10 | 39W-0L-1D | 0.9875 | [0.8909, 0.9987] | 19W-0L-1D (0.975) | 20W-0L (1.000) | 1307.1 s |

The two thresholds are INDISTINGUISHABLE on the sweep: identical
W/L/D and identical colour splits (the single draw in each is a
225-ply full-board game). Only 8/40 games are move-for-move
identical between the two matches — the thresholds DO diverge in
play (games 0–5 identical, then the RNG streams desync after a
differently-resolved tie) — but the divergence changes no
outcome at this sample size. **Per the pre-registered tie rule
(§1.2 rule 3: rates within 1 game → the smaller threshold),
v27 = τ 0.05.** The temporary sweep entries (90/91) were removed
on registration. Sweep artefacts:
`research/iterations/sweep-tau005-vs-v0.json`,
`research/iterations/sweep-tau010-vs-v0.json`.

**What the sweep does and does not show.** Zero losses in 80
combined sweep games vs v0 (v23's battery: 9 losses/100, all as
Black) is consistent with the hypothesis — the threshold
removes the Black-side misfire class — but the sweep seeds differ
from the battery seeds and n=40 cannot measure a 0.91→0.97
shift precisely. The 100-game battery is the verdict. v27's own
tactical (post-registration): 20/20 old + 29/29 hard at
`--seed 100`, as expected.

## 2. Loss analysis for iteration 27

(TODO — the fast-loop worked games ARE the loss analysis.)

## 3. v27 evaluation

(TODO — battery on the FINAL version only.)

## 4. Bar check (v27 vs v7)

(TODO.)

## 5. What this means

(TODO.)
