# Iteration 27 — v27: q-gap threshold tie-break for rollout-leaf engines (batch 9, H1)

**Status: COMPLETE — v27 PASSES all 4 clauses (see §4).** Comparison target is
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

### 2.1 Fast-loop iteration 1 — 10 games vs v7 + 10 games vs v0 (pre-patch engine)

**Batches:** `research/iterations/fastloop1-v27-vs-v7.json`
(seeds v27=1001 / v7=2001) and
`research/iterations/fastloop1-v27-vs-v0.json` (seeds v27=1001 /
v0=2001) — development seeds, distinct from the bar battery's
100/200. **vs v7: 7W-2L-1D, win rate 0.7500, CI [0.4422,
0.9191]**, 611.0 s. Split: as Black 4W-0L-1D (0.90), as White
3W-2L (0.60). **vs v0: 10W-0L-0D, win rate 1.0000, CI [0.7225,
1.0000]**, 309.3 s. Classifier (both batches): 2/2 losses via
open four, 0 missed blocks, 0 clean missed preventions in 8
threatened turns (losses). Defence is still perfect. (n=10 per
opponent — development signal, not the bar; the fast loop's job
is finding weaknesses, not measuring strength.)

**Both losses — games 1 and 9 vs v7 (White, 21 and 23 plies) —
have NO patchable losing move.** Game 9's walk-back
(`research/iterations/walkback-fl1-g9.json`, v1 oracle, seeds
0,1,2) nominates nothing: at every White turn the actual either
does not lose at all seeds (ply 15/13/11/9/7) or loses 3/3 with
0 saving alternatives (ply 21/19/17). The position went from
drawable at ply 15 to lost by ply 17 with no verified save —
distributed drift.

Game 1's walk-back (`research/iterations/walkback-fl1-g1.json`)
nominates ply 13 (actual (7,8) loses 3/3 under the v1 oracle;
saves (9,8) [win,win,win] and (7,4) [draw,win,draw]) and ply 17
(actual (6,4) loses 3/3; save (9,4) [draw,draw,draw]). **The
stronger oracle REFUTES the ply-13 nomination outright**
(`/tmp/verify_fl1_g1.py`, corrected — see the off-by-one note
below): under v7-vs-v7, actual (7,8) is [−1,+1,−1] (wins 1/3)
while BOTH "saves" lose 3/3 — (9,8) [−1,−1,−1], (7,4)
[−1,−1,−1]. The v1 oracle's saves are worse than the actual move
under strong play; the position at ply 13 is bad for White
(loses 2/3 from the actual) but no tested alternative does
better. The loss is distributed drift, not a single losing move.

**The threshold is exonerated in both nominations** (corrected
root dumps, fresh seed 0). At ply 13 the actual (7,8) is a
UNIQUE visit leader (18 visits, our-q −0.1197); the save (7,4)
is second at 8 visits with WORSE q (−0.3226), and (9,8) is not
in the top 6. The tie-break never fires (no tie), and no root
rule can promote a move that loses on visits AND on q —
promoting it would be the CLOSED q-based root-decisions lever.
At ply 17 there IS a 2-way visit tie at 6 visits — but both
children share our-q −1.000 EXACTLY (gap 0.0), so the threshold
correctly does not fire and the choice is RNG over the tied
pair; the save (9,4) holds 5 visits with better q (−0.8027)
but cannot be promoted without reopening the CLOSED lever.
This is q-compression in its purest form (every candidate loses
equally under the leaf), the same class as iter-23 §2.1's
move-27 finding — and no tie-break variant could have saved it,
since the q-tied set contains no saving move.

**Off-by-one note (method).** The first verification run used
the wrong ply convention: `walkback_loss.py`'s "ply" is the
0-based move index, so the board before ply N is the first N
moves — the first script passed N−1 and verified the wrong
positions (and the wrong player to move) for 42 minutes before
the root dumps exposed it (the "ply 17" dump reproduced Black's
(5,4), the move at index 16). The corrected run above uses
ply_before = N. The wasted run's numbers are discarded, not
reported as evidence.

**Iteration 1 verdict: no new patchable weakness class — but
the loop does NOT stop here** (same pre-commitment as iter-23
§2.1: two consecutive barren iterations, 40 games). Iteration
2 runs on fresh seeds (1002/2002); if it also finds no
patchable class, the loop stops. No regression positions added:
neither loss yields a verified losing move with a verified save
the engine could be patched to play.

### 2.2 Fast-loop iteration 2 — 10 games vs v7 + 10 games vs v0 (LOOP STOPS)

**Batches:** `research/iterations/fastloop2-v27-vs-v7.json`
(seeds v27=1002 / v7=2002) and
`research/iterations/fastloop2-v27-vs-v0.json` (seeds v27=1002 /
v0=2002). **vs v7: 7W-2L-1D, win rate 0.7500, CI [0.4422,
0.9191]**, 686.7 s — identical to iteration 1. Split: as Black
4W-1L (0.80), as White 3W-1L-1D (0.70). **vs v0: 10W-0L-0D, win
rate 1.0000, CI [0.7225, 1.0000]**, 250.8 s. Classifier (both
batches): 2/2 losses via open four, 0 missed blocks, 0 clean
missed preventions in 7 threatened turns (losses). Defence
still solved.

**Game 9 (White, 29 plies) has NO patchable losing move.**
Its walk-back (`research/iterations/walkback-fl2-g9.json`)
nominates nothing: the actual never loses 3/3 with a saving
alternative (ply 27/25 lose 3/3 with 0 saves; every earlier ply
fails to lose at all seeds). Distributed drift.

**Game 2 (Black, 34 plies) nominates ply 22 — and the
verification refutes it, with one finding worth recording.**
The v1 oracle says actual (5,0) loses 3/3 with three saves
((6,2), (6,7), (5,1), all drawing 3/3). Under v7-vs-v7
(`/tmp/verify_fl1_g1.py`'s `run_oracle`, corrected ply
convention): actual (5,0) [1,1,1] (Black loses 3/3 — confirmed
losing), but (6,2) [1,1,1] and (6,7) [1,1,1] ALSO lose 3/3 —
the v1 oracle's draws were wrong — and (5,1) is [1,1,−1],
saving at only 1/3 seeds: a split, not a verified save.

**The threshold FIRED at ply 22 — and it did not matter.** The
root (fresh seed 0) has a 2-way tie at 8 visits between (5,0)
(our-q −0.1605) and (6,2) (our-q −0.3086): gap 0.1481 > τ=0.05,
so the tie-break fires and picks (5,0). Both tied moves lose
3/3 under v7-vs-v7, so the choice between them is outcome-free.
Two lessons: (a) a "signal-sized" gap (0.1481, above the 0.130
tactical minimum) does NOT guarantee the higher-q move saves —
in real games it can separate two losers; the gap-size
separation holds for the tactical positions, where the gaps
reflect exact terminal values, but a large gap in a murky
midgame is not the same evidence. (b) The only move with any
saving signal, (5,1) (wins 1/3), holds 7 visits — one short of
the tie — with the HIGHEST our-q (−0.1429). No root rule can
promote a 7-visit move over 8-visit leaders; that is the CLOSED
q-based root-decisions lever, and 1/3 is too weak to record
even as a regression xfail. No patch.

**No patchable weakness class in this iteration either — the
fast loop STOPS here (2 iterations)** per the pre-commitment
in §2.1 (two consecutive barren iterations, 40 games). v27's
only losses in the loop are four distributed-drift games (two
per iteration) plus one threshold-fire between two losers; the
battery is the verdict on whether the threshold costs or saves
games at scale.

### 2.3 Fast-loop iteration 3 — NOT RUN (loop stopped after iteration 2)

### 2.4 Fast-loop iteration 4 — NOT RUN (loop stopped after iteration 2)

### 2.5 Aggregate counts (secondary)

Fast-loop totals (2 iterations, 40 games): vs v7 14W-4L-2D
(0.7500); vs v0 20W-0L-0D (1.0000). Colour splits vs v7: as
Black 8W-1L-1D (0.85), as White 6W-3L-1D (0.65). Fast-loop
defence: 4/4 losses via open four, 0 missed blocks, 0 clean
missed preventions (15 threatened turns in losses). Defence
remains solved. No patches were made in this iteration —
v27-as-batteried is v27-as-pre-registered (τ=0.05, selected by
the §1.2 rule).

## 3. v27 evaluation

Battery run on the FINAL v27 (τ=0.05, as pre-registered — no
patches were made in the fast loop, §2.5). Pre-registered seeds
(§1.2): h2h 100/200, ladder seed 100, swap seeds 100/200 with
balanced placement (placement_seed 11, tempo judge), tactical
`--seed 100`. All aggregates below were recomputed from the raw
records in the committed artefacts and match the drivers'
headers exactly (`/tmp/verify_v27.py`).

**Runtime interruptions (recorded, no games lost).** The battery
chain was interrupted TWICE by runtime restarts without warning:
once ~50 min in, at 53/100 vs-v7 games saved; once during the
swap match, at 16/100 games saved. Both times the SAME chain was
relaunched and the drivers auto-resumed from the partial
`--out` files (`resumed_from_game: 53` and `resumed_from_game:
16` in the artefacts; the completed matches were skipped on
relaunch). The elapsed times below are the resume legs' as
printed by the drivers, not total wall time.

### 3.1 Tactical (clause 3)

`ml/scripts/eval_tactical_full.py --version 27 --suite both
--seed 100` (full-dispatch driver, `player.choose_move`):
**old 20/20, hard 29/29** on the CORRECTED suite. Artefact:
`research/iterations/tactical-v27.json`. The three tie fixes
choose (8,4), (6,7), (7,5) as in the mechanism check (§1.3);
winning-quiet-setup-2 chooses (9,7). `eval_version.py`'s probe
independently confirms old 20/20 (§3.5).

### 3.2 Head-to-head vs v7 (clause 1)

`ml/scripts/fast_loop_match.py --version-a 27 --opponent v7
--games 100 --seed-a 100 --seed-b 200`: **77W-19L-4D, win rate
0.7900, 95% CI [0.7002, 0.8583]**, 2655.4 s (resume leg, 47
games). Colour splits: as Black 41W-8L-1D (0.8300), as White
36W-11L-3D (0.7500). Artefact:
`research/iterations/batch-v27-vs-v7.json`.

Context: v27's 0.7900 sits between v18's 0.775 (73W-18L-9D) and
v23's 0.7950 (75W-16L-9D) — above the strength champion, 2 games
below v23, well within noise of both (all three CIs overlap
heavily). The threshold costs nothing measurable vs v7: the
tie-break's vs-v7 effect was ~neutral-to-favourable, and
filtering its sub-0.05 firings does not move the match.

### 3.3 Head-to-head vs v0

`ml/scripts/fast_loop_match.py --version-a 27 --opponent v0
--games 100 --seed-a 100 --seed-b 200`: **96W-3L-1D, win rate
0.9650, 95% CI [0.9085, 0.9871]**, 3549.2 s. Colour splits: as
Black 46W-3L-1D (0.9300), as White 50W-0L-0D (1.0000). Artefact:
`research/iterations/batch-v27-vs-v0.json`.

Context: this is the regression v27 was built to fix, and it is
fixed. v23 scored 0.91 (91W-9L, all 9 losses as Black); v27
recovers 5 of those games and lands at 0.9650 — statistically
indistinguishable from v18's 0.9700 (96W-2L-2D; v27 has one more
loss and one fewer draw, same win count). The 3 remaining losses
are all as Black, the same colour the misfires hit — the
threshold converts deterministic-wrong picks into coin flips
(§1.3), so a residual Black-side cost is expected, not a
surprise. The sweep's 0 losses in 80 games (§1.5) did not
reproduce at battery seeds, as §1.5 warned it might not.

### 3.4 Swap match vs v7

`ml/scripts/selfplay_batch.py --engine-a v27 --engine-b v7
--games 100 --swap-rule --placement balanced --placement-seed
11 --judge tempo --seed-a 100 --seed-b 200`: **76W-21L-3D, win
rate 0.7750, 95% CI [0.6839, 0.8458]**, 3345.3 s (resume leg,
84 games). Colour splits: as Black 40W-9L-2D (0.8039, 51
games), as White 36W-12L-1D (0.7449, 49 games). Artefact:
`research/iterations/batch-v27-vs-v7-swap.json`.

Context: **0.7750 is the best swap result in the program**
(v23: 0.72, v18: not measured under swap vs v7 in its own
battery — v18's swap figure in iter-18 is vs v1; v7's swap vs
v1 was 0.57 with the CI including 0.50). Unlike v7, whose edge
mostly lived in the empty-board opening, v27's edge survives
the swap choice almost intact (0.7750 swap vs 0.7900 normal).
The second player (tempo judge) chose Black in 77/100 games and
White in 23/100 — the balanced openings mostly favour Black
slightly, and v27 wins strongly from either side (0.80 as
Black, 0.74 as White).

### 3.5 Ladder (clause 2)

`ml/scripts/eval_version.py --version 27 --ladder-games 20
--mcts-sims 100 --seed 100` (note: the brief said
`eval_scoring.py`, but that script has no `--version` flag;
`eval_version.py` is the ladder script, as iter-23 §3.5 used):
**ladder Elo 636.4, interval [240.3, 1032.5]**. Artefact:
`research/iterations/ladder-v27.json`.

| Opponent | Result | Win rate | 95% CI | As Black | As White |
|---|---|---|---|---|---|
| random | 20W-0L-0D | 1.0000 | [0.8389, 1.0000] | 10W-0L-0D | 10W-0L-0D |
| heuristic | 19W-0L-1D | 0.9750 | [0.7995, 0.9974] | 9W-0L-1D | 10W-0L-0D |
| mcts | 13W-7L-0D | 0.6500 | [0.4329, 0.8188] | 8W-2L-0D | 5W-5L-0D |

Context: **636.4 EXACTLY TIES v18's 636.4** — the highest ladder
Elo in the program. The tie is exact, not approximate: the Elo
estimate is computed from the vs-heuristic score only
(19W-0L-1D → 636.4258...), and v27 scored exactly v18's
19W-0L-1D against the heuristic. v23 scored 0.95 vs the
heuristic (511.5) — the single Black loss the threshold was
built to prevent. The full 125-Elo regression is recovered.
The vs-mcts leg (13W-7L) is one game weaker than v18's 14W-6L
but does not enter the Elo estimate (the mcts opponent has no
anchor); v27 is the third version ever to beat the ladder MCTS
outright (after v18 and v21).

### 3.6 Defence classification

`ml/scripts/classify_losses.py` on all three battery batches:

| Batch | Losses | Via open four | Missed blocks | Clean missed preventions | Threatened turns |
|---|---|---|---|---|---|
| vs v7 | 19 | 19 | 0 | 0 | 69 |
| vs v0 | 3 | 3 | 0 | 0 | 14 |
| swap vs v7 | 21 | 21 | 0 | 0 | 100 |
| **Pooled** | **43** | **43** | **0** | **0** | **183** |

Defence remains solved: every battery loss is an open-four
race lost, never a missed block or a missed prevention.

### 3.7 Test suites (clause 4)

All suites run on the FINAL tree (after the H2 test fix,
commit a949606, and all battery artefacts):

- **ml pytest: 297 passed, 7 skipped, 12 xfailed, 0 failures**
  (282 s). The 12 xfails are the known regression xfails
  (v7's, v23's carried-over fl1-g0-ply12 and fl2-g7-ply9, and
  v27's own 2 — see `ml/tests/test_regression_v27.py`).
- **npm test: 55/55** (4 test files).
- **`tsc --noEmit`: clean**, exit 0.
- **`ml/gomoku_ml/baseline.py`: 0 lines changed** vs the base
  branch (`git diff 0ee51d8` — the frozen ladder anchor is
  untouched).

## 4. Bar check (v27 vs v7)

| Clause | Requirement | v27 | Verdict |
|---|---|---|---|
| 1. h2h vs v7 | win rate > 0.50, Wilson 95% CI lower bound > 0.50 | 77W-19L-4D, 0.7900, CI [0.7002, 0.8583] | **PASS** |
| 2. Ladder Elo | > v7's 190.8 | 636.4, interval [240.3, 1032.5] | **PASS** |
| 3. Tactical | ≥ v7's 20/20 + 29/29 (corrected suite) | 20/20 old + 29/29 hard | **PASS** |
| 4. Suites green | ml pytest, npm test, tsc | 297 passed / 0 failed; npm 55/55; tsc clean | **PASS** |

**v27 PASSES all 4 clauses.** It is the second version in the
program ever to do so (after v7), and the first rollout-leaf
version. Per the brief, it goes to master and becomes the
heuristic track's new bar reference, and the daily cross-track
match's heuristic champion question is for the coordinator:
v27 beats v18 on h2h vs v7 (0.7900 > 0.775) but TIES v18 on
ladder Elo (636.4 = 636.4, exact), so it does NOT meet the
strict "beat on BOTH" champion rule — see §5.

## 5. What this means

**v27 PASSES all 4 clauses** (pending §3.7's suite numbers —
clauses 1–3 are decided; see §4). It is the second version in
the program ever to pass the bar (after v7), and the first
rollout-leaf version to do so. The threshold hypothesis is
CONFIRMED on every axis it targeted:

- **vs v0: 0.91 → 0.9650** — v23's regression recovered (5 of
  the 9 lost games back), statistically indistinguishable from
  v18's 0.97.
- **Ladder Elo: 511.5 → 636.4** — the full 125-Elo regression
  recovered, exactly tying the program best.
- **Tactical: 28/29 (old suite) → 29/29 (corrected)** — all
  three q-compression tie fixes kept; the threshold fires on
  every documented signal gap (0.130–0.417) and blocks the
  documented noise gap (0.016).
- **Swap: 0.72 → 0.7750** — the best swap result in the
  program, improving on v23 rather than merely keeping it.
- **vs v7: 0.7950 → 0.7900** — 2 games below v23, within noise;
  the threshold costs nothing measurable against the reference.

**What changed because of a specific loss: nothing was patched.**
The fast loop ran 2 iterations (40 games) and found no
patchable weakness class (§2): all 4 losses were distributed
drift, the one threshold fire in the loop (§2.2, ply 22)
separated two verified losers, and the only move with any
saving signal sat one visit below the tie — promotable only by
the CLOSED q-based root-decisions lever. v27-as-batteried is
v27-as-pre-registered (τ=0.05, selected by the §1.2 tie rule
after both thresholds tied the sweep at 39W-0L-1D). The
threshold itself was designed from v23's documented losses
(iter-23 §2.2's 0.016-gap misfires), not from v27's own games.

**Champion status — the honest read.** v27 beats v18 on h2h
vs v7 (0.7900 > 0.775) but TIES v18 on ladder Elo (636.4 =
636.4, exact — same 19W-0L-1D vs the heuristic). The strict
champion rule needs a beat on BOTH, so v27 is NOT the new
strength champion by the strict reading; v18 keeps the title
on the tie. But the tie is a tie at the program's ceiling, and
v27 is the version that passes the bar — v18 never did
(tactical 25/29). The two-track structure's heuristic bar
reference should now be v27, not v7: future versions are
measured against the strongest bar-passing version, and v27 is
it. (The coordinator decides the champion recording; the
numbers here are exact.)

**What the threshold teaches.** The gap-size separation is real
but not absolute. In the tactical positions, gaps ≥0.130 carry
exact terminal values and the threshold keeps every fix. In
real games, a "signal-sized" gap (0.1481 at fast-loop ply 22)
separated two verified losers — a large gap in a murky midgame
is not the same evidence as a large gap in a tactical position,
because the rollout leaf's q values are estimates, not exact
values, outside forced lines. The threshold's job was never to
make q reliable; it was to stop q from being consulted where it
is KNOWN noise (sub-0.05 gaps, mostly q-compression ties), and
on that job it delivered exactly what §1.2 predicted: the
misfires became coin flips, half the lost games came back, and
the Elo followed. Further threshold tuning (τ in (0.05, 0.10])
is a dead end — the sweep showed the two thresholds
indistinguishable, and the remaining vs-v0 losses are coin
flips, not threshold artefacts.

**What did NOT change.** The leaf is v18's, byte-identical. The
search is v23's. The only delta from v23 is one gated
comparison at the root. v27's improvement over v23 is entirely
a root-decision refinement — and it is the last such refinement
available: the root-decision lever is now exhausted in both
directions (pure q catastrophic, tie-break thresholded to its
useful limit, visit leaders untouchable without reopening a
closed lever). The next heuristic gains must come from the
search or the leaf, not the root rule.

**Test-repair note (flagged to the coordinator).** The batch-9
suite correction (0ee51d8) broke 2 tests in
`ml/tests/test_tactical_hard.py` on the base branch (verified
by reproducing both failures on a bare 0ee51d8 worktree). I
fixed them first (commit 3012cea), then — per the
coordinator's note — replaced my fix verbatim with H2's
verified fix (commit a3c4f73 on `research/batch-9-h2`, taken
via `git checkout origin/research/batch-9-h2 -- 
ml/tests/test_tactical_hard.py`; my commit a949606). The two
fixes are functionally identical (V1_PASSED 12→13, a
playout-verified (9,7) exception); H2's is canonical so all
batch-9 branches converge. The coordinator's merge should take
either — they agree.
