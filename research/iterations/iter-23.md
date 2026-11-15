# Iteration 23 — v23: q-aware visit-tie break for rollout-leaf engines (batch 8, H3)

**Status: IN PROGRESS — pre-registration (§1).** Comparison target is
v7 (the batch-8 reference: ladder Elo 190.8, on master at
`bf866f7`). This is the H3 workstream of batch 8 (H1 is v21,
v18+blend; H2 is v22, conditional gap-three demotion); work happens
on branch `research/batch-8-h3` in the `gomoku-b8-h3` worktree.
v23 does NOT change the leaf (v18's rollout leaf, byte-identical),
does NOT change sims/priors/book (CLOSED), does NOT change
exploration constants (CLOSED without a new mechanism), and does
NOT include the threat check. The ONLY change from v18 is the
root tie-break rule in §1 — a narrow, mechanism-specific test
targeting v18's documented q-compression tie failure mode, framed
as clause-3 recovery for rollout-leaf engines. It is NOT a
reopening of the CLOSED q-based root-decisions lever (see §1.1).

**Method (carried over from batch 3/4/5/6/7):** the fast 10-game
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
battery (100-game matches + ladder) runs on the FINAL patched v23
only. Fast-loop games use development seeds (1001/2001, 1002/2002,
...) distinct from the bar battery's pre-registered seeds
(100/200), so no development game overlaps an eval game.

## 1. Design and pre-registration

**Comparison target: v7** (primary, the batch-8 reference: ladder
Elo 190.8, on master at `bf866f7`). The bar's clause 1 is
head-to-head vs v7 (normal rules, 100 games, alternating colours,
win rate > 0.50 with Wilson 95% CI lower bound > 0.50), clause 2 is
ladder Elo strictly higher than v7's 190.8 (identical 20
games/opponent: random, heuristic, mcts, mcts_sims=100), clause 3 is
tactical pass rate >= v7's (old suite 20/20, hard suite 29/29),
clause 4 is all test suites green (ml pytest, npm test, tsc).
**Secondary head-to-heads, also pre-registered:** v23 vs v0 (100
games) and a 100-game swap match vs v7 under the tempo judge (the
harness default). Non-transitivity is recorded prominently.

**The improvement: v23 = v18 + a q-aware visit-tie break at the
root.** Everything else is v18 unchanged: 100 sims, c_puct 1.5,
v1 priors over all candidates, win/block scans before search,
most-visited root child, rollout leaf both colours (leaf mode
`rollout2_v1_both`, v16's function byte-identical including both
fast-loop patches). The tie-break is the ONLY change from v18.

### 1.1 The tie-break rule (pre-registered, exact)

Current v18 root selection (`HeuristicMCTS.choose_move`): after
all sims, take the children with the maximum visit count; if
exactly one, play it; if several tie, pick by RNG
(`rng.choice(best)`) or, with no RNG, the first in dict order.
v23 replaces ONLY the tie case:

1. **When it applies:** only at the root, only when two or more
   root children share the maximum visit count (an exact integer
   tie at the top) after all sims. If a unique child has the most
   visits, v23 behaves EXACTLY like v18 — no q is consulted, no
   RNG is consumed. Near-ties (a 1-visit gap) are NOT ties and
   are not touched.
2. **The q used:** our-perspective q, defined as `−child.q`.
   The sign is verified against the backup code, not taken on
   trust: `_run_one` accumulates `value_sum` from the
   perspective of the player to move AT THAT NODE (it returns
   the value from `player`'s perspective, and the parent
   negates on backup: `value = -self._run_one(...)`), so for a
   root child — whose player to move is the opponent — `child.q`
   is the OPPONENT's mean value. Our-perspective q is
   `−child.q`, exactly as `_select_child` exploits it
   (`-child.q` + exploration). Batch 4's sign error (iter-7
   §2.6's inverted claim, corrected in batch 4) must not recur.
3. **Selection:** among the visit-tied children, play the one
   with the HIGHEST our-q (`−child.q`).
4. **If q also ties:** if two or more visit-tied children share
   the highest our-q EXACTLY (float equality — plausible when
   the underlying value sums and visit counts are identical,
   e.g. two children at −1.000), fall back to v18's behaviour
   restricted to that subset: `rng.choice` among the q-tied
   children when an RNG is set, else the first in dict order.
   No epsilon, no tolerance: exact equality only. The documented
   failures have q gaps of 0.13–0.42, far above float noise.

**Scope:** the root final selection only. `_select_child`
(PUCT) is unchanged — the tie-break never influences search,
only the move played. The leaf, priors, sims, c_puct, and the
win/block scans are unchanged.

**Why this retest is justified (and why it is not a reopening
of the closed lever).** q-based root decisions are CLOSED as a
general mechanism: pure q (replacing visits entirely) was
catastrophic (0.20 vs v7, tactical 22/29), and v9's tie-by-q
variant smoked 0.45 on the v0/v1 leaf. But both of those were
on the GRADIENT/terminal leaf, where q is a smooth, informative
signal and visits are the better aggregator. On the ROLLOUT
leaf the situation is inverted and DOCUMENTED (iter-18 §3.1):
q-compression flattens the leaf's signal (many rollout-ends at
exactly ±0.2), the q gap between a saving move and a losing
move shrinks, and with ~84 candidates splitting 100 sims the
losers' ~3x prior advantage equalises the visits — so the
search ties moves the leaf already discriminates. In all three
tie failures the leaf KNOWS the answer (chosen −1.000 vs
expected +0.250 at the leaf; our-q −1.000 vs −0.583 at the
root) but cannot convert it into visits. The tie-break is the
minimal mechanism that converts the leaf's existing knowledge
into the played move, and it fires ONLY where visits carry no
information at all (an exact tie). This is clause-3 recovery
for rollout-leaf engines, not a strength lever: the
root-decision lever is EXHAUSTED for strength (±10 games/100,
v9/v12), and no strength gain is expected or claimed.

### 1.2 The four failures it targets (iter-18 §3.1, reproduced exactly at --seed 100)

| Position | To move | v18 chose | Expected | Root state (fresh seed 0) |
|---|---|---|---|---|
| prev-v1v0s-g26 | Black | (12,3) | (8,4)/(9,4) | 4-way tie at 3 visits; expected our-q −0.583, losers −1.000 |
| prev-v1v1s-g79 | Black | (5,1) | (6,7)/(8,7)/(7,8)/(7,9) | 7-way tie at 3 visits; expected our-q −0.583, losers −1.000 |
| prev-v1bl-g2 | Black | (7,9) | (6,5)/(7,5) | 2-way tie at 8 visits; expected (7,5) our-q −0.594, loser (7,9) −0.724 |
| winning-quiet-setup-2 | Black | (9,7) | (7,7)/(8,7)/(8,9)/(10,10) | NOT a tie: 79 visits on (9,7) vs 16 on (8,7); (9,7) wins 3/3 under v7-vs-v7 — a suite-list gap, not a defect |

**Tactical expectation (pre-registered, honest).** The tie-break
should flip exactly the three tie failures: in each, the
expected move holds strictly better our-q than every loser it
ties with, so rule 3 picks an expected move. The fourth failure
is NOT a tie and is NOT expected to flip — v23 will play (9,7)
there exactly as v18 did (the position's own search has no root
tie, so the rule never fires; RNG-stream divergence across the
suite cannot change a no-tie position's outcome). **Expected
tactical: 20/20 old + 28/29 hard**, with winning-quiet-setup-2
as the sole remaining failure — which means clause 3 is
expected to FAIL on that one suite-list-gap position alone,
unless the suite's expectation list is widened (a suite change,
not an engine change, and not part of this iteration). The
test's value is the mechanism isolation: if v23 recovers
exactly the three documented tie failures and no other
position regresses, the q-compression tie failure mode is
confirmed as fixed by the minimal mechanism, and the remaining
gap is a measurement artefact, not an engine defect. Any
OTHER failure (a position v18 passed that v23 fails) is a
regression to report prominently — the tie-break could
misfire where v18's RNG tie-break happened to pick right.

**Game-strength expectation (pre-registered).** The tie-break
fires rarely in real games (only on exact top-visit ties) and
when it fires it picks the higher-q move, so the expected
effect on h2h is ~neutral — the exhausted root-decision lever
brackets it at ±10 games/100. v23's game strength should
reproduce v18's (h2h vs v7 ~0.775, ladder Elo ~636) within
noise. A large deviation in either direction would be
evidence the tie-break is NOT the neutral tactical-recovery
mechanism it is framed as, and would need explaining. The
battery decides.

**Cost (from v18's measurements — the leaf function is
unchanged, so the cost is identical by construction).** A
100-game match vs v7 is ~1.75 h (v18's took 6301 s), vs v0
~1.25 h, swap ~1.5 h, ladder ~30 min, tactical ~10 min.
Affordable.

**Eval seeds (pre-registered):** h2h vs v7: v23=100 / v7=200;
h2h vs v0: v23=100 / v0=200; ladder seed 100, 20
games/opponent, mcts_sims=100; swap 100 games vs v7, balanced
placement, placement_seed 11, tempo judge (the harness
default), seeds 100/200; tactical `--seed 100` (the
established protocol — rollout-leaf tactical results are
seed-dependent). Fast-loop dev seeds 1001/2001, 1002/2002, ...
(iteration k uses 1000+k / 2000+k, per iter-8 §4's policy).

### 1.3 Mechanism check (post-implementation, pre-fast-loop)

After implementation and before any fast-loop game, v23 was run
on the 4 failing tactical positions (fresh-seed-0 root dumps,
`/tmp/dump_v23_tactical_fails.py`) and on the full tactical
suites at `--seed 100` (the established protocol). This is an
implementation check, not pre-registration evidence and not an
eval — it verifies the dispatch does what §1 says it does.

**Root dumps reproduce iter-18 §3.1 exactly** (same ties, same
q values, same priors — the search is unchanged; only the final
selection differs):

| Position | v18 root state (seed 0) | v23 picks |
|---|---|---|
| prev-v1v0s-g26 | 4-way tie at 3 visits; expected our-q −0.583, losers −1.000 | (9,4) EXPECTED |
| prev-v1v1s-g79 | 7-way tie at 3 visits; expected our-q −0.583, losers −1.000 | (7,9) EXPECTED |
| prev-v1bl-g2 | 2-way tie at 8 visits; expected (7,5) our-q −0.594 vs (7,9) −0.724 | (7,5) EXPECTED |
| winning-quiet-setup-2 | NO tie: 79 visits on (9,7) vs 16 on (8,7) | (9,7), as v18 |

Two details worth recording. First, on g26 and g79 the q-tie
fallback (rule 4) fires: the expected moves share the top our-q
exactly (−0.583 on both), so v23 falls back to RNG choice among
them — harmless, because every q-tied move is expected. Second,
in the ISOLATED seed-0 dumps v18's RNG happens to pick an
expected move on g26 ((9,4)) and g79 ((7,8)) — the suite-run
failures came from the RNG stream state at those positions when
one player instance runs the whole suite, which is exactly why
the tie-break (deterministic where q separates, RNG only among
equal-q moves) is the fix rather than luck.

**Full suites at `--seed 100`: old 20/20, hard 28/29** — exactly
the pre-registered expectation (§1.2). All three tie failures
flip (g26 chose (8,4), g79 chose (6,7), v1bl-g2 chose (7,5)),
NO other position regresses, and the sole remaining failure is
winning-quiet-setup-2 (chose (9,7)) — the suite-list gap, not a
tie, not a defect. The q-compression tie failure mode is
confirmed fixed by the minimal mechanism; the remaining gap is
a measurement artefact (the suite's expectation list omits a
move that wins 3/3 under v7-vs-v7).

**v18 regression positions under v23** (seed 0, the v23
regression file): the 3 White mechanism positions pass, the 2
xfails stay xfails (fl1-g0-ply12: v23 plays (7,10) as v18 did;
fl2-g7-ply9: v23 plays (6,4) — the rollout leaf prefers (6,4)
too, so the q-break cannot flip a twin choice the leaf itself
ranks wrong), fl1-g6-ply78 passes. The tie-break changes
nothing where no tie exists, as designed.

**v18 behaviour preserved by the refactor.** `choose_move`'s
root selection was extracted into `_pick_root_move` (same code,
same order) so the rule is unit-testable; the full
`test_mcts_heuristic` + v7/v16/v18 regression suites pass
unchanged (56 passed, 6 xfailed), and v23's own unit tests (5)
plus its regression file (8 passed, 3 xfailed) are green.

### 1.4 Battery (the verdict — pre-registered seeds and commands)

On the FINAL v23 only (after the fast loop):

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
`tie_break_q` flag on `HeuristicMCTS` (default False — v18 and
all earlier versions unaffected) and `ml/gomoku_ml/scoring.py`
gains ONLY the `ENGINE_VERSIONS[23]` registry entry (v18's
entry + `"tie_break_q": True`) plus passing the flag through
to `HeuristicMCTS`. The v1 kernel (`kernel_v1.py`), the
rollout function, and v18's dispatch are checked out unchanged
from `research/batch-7-h1`. `baseline.py` is frozen and
untouched.

## 2. Loss analysis for iteration 23

(TODO — the fast-loop worked games ARE the loss analysis.)

### 2.1 Fast-loop iteration 1 — 10 games vs v7 + 10 games vs v0 (pre-patch engine)

**Batches:** `research/iterations/fastloop1-v23-vs-v7.json`
(seeds v23=1001 / v7=2001) and
`research/iterations/fastloop1-v23-vs-v0.json` (seeds v23=1001 /
v0=2001) — development seeds, distinct from the bar battery's
100/200. **vs v7: 8W-0L-2D, win rate 0.9000, CI [0.5958,
0.9821]**, 1149.9 s. Split: as Black 3W-0L-2D (0.80), as White
5W-0L (1.00). **vs v0: 9W-1L-0D, win rate 0.9000, CI [0.5958,
0.9821]**, 296.2 s. Split: as Black 4W-1L (0.80), as White
5W-0L (1.00). Classifier (both batches): 1/1 loss via open
four, 0 missed blocks, 0 clean missed preventions in 6
threatened turns (losses). Defence is still perfect. (n=10 per
opponent — development signal, not the bar; the fast loop's job
is finding weaknesses, not measuring strength.)

**The only loss — game 4 vs v0 (Black, 36 plies) — has NO
patchable losing move.** White won with the vertical five at
x=5 ((5,4) completing (5,4)–(5,8)) after Black faced an open
four at move 35. Walk-back
(`research/iterations/walkback-fl1-g4.json`, v1 oracle, seeds
0,1,2) nominates nothing: at every Black turn from move 27 on,
the actual loses 3/3 with 0 saving alternatives, and at moves
21–25 the actual "wins" under the oracle. **The oracle's
move-25 verdict is an artifact** — re-verification with the
stronger oracles (`/tmp/verify_fl1_g4.py`,
`/tmp/verify_fl1_g4b.py`, per the iter-18 §2.1 rule) reverses
it:

- Move 21, actual (8,4): wins 3/3 under v7-vs-v7.
- Move 23, actual (3,5): wins 3/3 under v7-vs-v7 — and every
  top 1-ply alternative LOSES 3/3. (3,5) was the best move.
- Move 25, actual (3,6): loses 3/3 under v7-vs-v7 AND 3/3
  under the actual matchup (v23-vs-v0). The v1 oracle's "wins
  2/3" is wrong because v1-as-White never plays the killer
  (5,8). Alternatives (5,8), (5,5), (4,8) all lose 3/3 under
  both oracles. (7,6) wins 2/3 under v7-vs-v7 but loses 3/3
  under v23-vs-v0 — a split, oracle-dependent edge, not a
  verified save: the position after (7,6) still loses when
  v23 itself plays it out.
- Move 27, actual (11,11): loses 3/3 under both oracles.
  (5,5) wins 3/3 under v23-vs-v0 but loses 3/3 under
  v7-vs-v7 — again a split: a save against the weak opponent
  only (v0-as-White misses the follow-up; v7-as-White finds
  it). Patching toward it would be fitting the weak opponent.

So the game went from winning at move 23 to lost by move 25,
with White's (5,6) creating a double structure (the row-6
three plus the x=5 column) that no single Black reply handles
under both oracles. Distributed drift (the iter-6 pattern), no
single verified losing move. **No patch.** The (7,6)
nomination is too weak to record even as a regression xfail
(it loses under the actual matchup); it is documented here,
not in the regression file.

**The tie-break is exonerated in this loss** (root dumps,
`/tmp/verify_fl1_g4.py`): at move 25 the root has a UNIQUE
visit leader — (3,6) at 15 visits — so the tie-break never
fired, and v18's root is identical. At move 27 there IS a
4-way visit tie at 2 visits — but all four children share the
same our-q EXACTLY (−0.375), so rule 4 falls back to RNG and
v23 picks (11,11); v18's RNG picks (11,11) too at seed 0.
This is q-compression in its purest form (every candidate
loses equally under the leaf), and the tie-break cannot
discriminate where the leaf itself cannot. The loss is a
value/horizon failure, not a tie-break failure — and no
tie-break variant could have saved it, since the q-tied set
contains no saving move under either oracle.

**Iteration 1 verdict: no new patchable weakness class — but
the loop does NOT stop here** (same pre-commitment as iter-18
§2.1: two consecutive barren iterations, 40 games). Iteration
2 runs on fresh seeds (1002/2002); if it also finds no
patchable class, the loop stops. No regression positions added.

### 2.2 Fast-loop iteration 2 — 10 games vs v7 + 10 games vs v0 (LOOP STOPS)

**Batches:** `research/iterations/fastloop2-v23-vs-v7.json`
(seeds v23=1002 / v7=2002) and
`research/iterations/fastloop2-v23-vs-v0.json` (seeds v23=1002 /
v0=2002). **vs v7: 7W-2L-1D, win rate 0.7500, CI [0.4422,
0.9191]**, 759.1 s. Split: as Black 3W-2L (0.60), as White
4W-0L-1D (0.90). **vs v0: 10W-0L-0D, win rate 1.0000**,
623.4 s. Classifier (both batches): 2/2 losses via open four,
0 missed blocks, 0 clean missed preventions in 10 threatened
turns (losses). Defence still solved.

**Both losses — games 6 and 8 vs v7 (Black, 30 plies each) —
have VERIFIED losing moves, and both are tie-break misfires.**
Walk-back (`research/iterations/walkback-fl2-g6.json`,
`walkback-fl2-g8.json`, v1 oracle, seeds 0,1,2) nominates move
23 in both games (actual loses 3/3, 4 saving alternatives
each) — and this time the stronger oracles CONFIRM the
nominations outright (`/tmp/verify_fl2_g6g8.py`,
`/tmp/verify_fl2_g6g8b.py`):

| Game | Move 23 | v7-vs-v7 | v23-vs-v7 |
|---|---|---|---|
| g6 actual (9,5) | loses | [−1,−1,−1] | [−1,−1,−1] |
| g6 save (4,7) | wins | [1,1,1] | [1,1,1] |
| g6 save (8,10) — the other tied move | wins | [1,1,1] | [0,1,1] |
| g6 saves (9,7)/(9,10)/(11,7) | save | [1,1,0]/[0,1,1]/[1,1,1] | — |
| g8 actual (5,5) | loses | [−1,−1,−1] | — |
| g8 save (10,7) | wins | [1,1,1] | [1,1,1] |
| g8 save (6,10) — the other tied move | saves | [−1,1,1] | [1,1,1] |
| g8 saves (5,7)/(3,7)/(5,10) | save | [1,1,1]/[1,1,1]/[−1,1,1] | — |

In both games White won with a five on row 10 (g6: (10,10)
completing (6–10,10); g8: (4,10) completing (4–8,10)), and
Black's move 23 played elsewhere while the row-10 build was
already underway. The two positions are near-mirror images of
each other — same root shape, IDENTICAL q values — so this is
really ONE structural observation, seen twice.

**The tie-break misfire.** At move 23 in both games the root
has a 2-way visit tie at 8 visits, and the tie-break picks the
LOSING tied move over the SAVING tied move on a q gap of
0.016: g6 (9,5) our-q +0.076 over (8,10) +0.060; g8 (5,5)
+0.076 over (6,10) +0.060. v18's RNG picks the saving move at
seed 0 in both — but that is 50/50 luck, not a fix. The q gap
here is gradient noise, not terminal signal: both q values are
small positives from the rollout leaf's capped gradient (the
threat materialises beyond the 2-ply horizon, so the leaf
cannot see it), whereas the tactical fixes' gaps (0.13–0.42)
reflect exact terminal values (loser −1.000 vs saver +0.250 at
the leaf). The tie-break trusts q where q is exact and
misleads where q is noise — the mechanism's boundary, found in
real games.

**No patch — and why.** The fast-loop method demands a patch
that reproduces the perfect move, and no available patch does:

- The best saves, (4,7) and (10,7), hold ≤6 visits and LOWER
  q than the played moves at the root. No root rule can
  promote them — promoting a 5-visit move over an 8-visit
  leader is the CLOSED q-based root-decisions lever (pure q
  was catastrophic), and the leaf/prior/exploration levers
  that could are CLOSED. Their value materialises beyond the
  2-ply horizon (White's row-10 five takes 4+ plies to build)
  — the same beyond-horizon class as iter-13's White saves.
- The tied moves (8,10)/(6,10) cannot be distinguished from
  the losers by any root signal: visits tied, priors equal
  (0.031), q wrong by 0.016. A q-gap THRESHOLD (apply the
  tie-break only when the gap exceeds, say, 0.05–0.1 — which
  would preserve all three tactical fixes, gaps ≥0.130) does
  NOT fix these positions: it converts a deterministic-wrong
  choice into a 50/50 RNG coin flip. It only helps if q is
  SYSTEMATICALLY misleading in quiet ties, and the evidence
  for that is n=1 (one structural position, mirrored). The
  threshold value would be fit to that n=1 (0.016 vs 0.130)
  and risks the v1bl-g2 fix (gap 0.130, margin 0.03 at a 0.1
  threshold). It is recorded here as the next iteration's
  candidate hypothesis — to be tested with battery-scale
  evidence if the battery shows a game-strength cost — not
  patched in on two mirrored games.

The two positions ARE recorded as regression xfails
(`fl2-g6-ply23`, `fl2-g8-ply23` — v23 plays the losing moves),
so the tie-break's boundary case is documented for future work
rather than dropped.

**No patchable weakness class in this iteration either — the
fast loop STOPS here (2 iterations)** per the pre-commitment
in §2.1 (two consecutive barren iterations, 40 games). The
Black leaf's only losses in the loop are these two tie-break
misfires plus iteration 1's distributed-drift loss; the
battery is the verdict on whether the tie-break costs games
at scale.

### 2.3 Fast-loop iteration 3 — NOT RUN (loop stopped after iteration 2)

### 2.4 Fast-loop iteration 4 — NOT RUN (loop stopped after iteration 2)

### 2.5 Aggregate counts (secondary)

Fast-loop totals (2 iterations, 40 games): vs v7 15W-2L-3D
(0.8250); vs v0 19W-1L-0D (0.9500). Colour splits vs v7: as
Black 6W-2L-2D (0.70), as White 9W-0L-1D (0.95). Fast-loop
defence: 3/3 losses via open four, 0 missed blocks, 0 clean
missed preventions (16 threatened turns in losses). Defence
remains solved. Two of the three losses are the tie-break
misfire class (one structural position, mirrored); the third
is distributed drift with no patchable move. No patches were
made in this iteration — v23-as-batteried is v23-as-
pre-registered.

## 3. v23 evaluation

(TODO — battery on the FINAL version only.)

### 3.1 Tactical suites

(TODO)

### 3.2 Head-to-head vs v7

(TODO)

### 3.3 Head-to-head vs v0

(TODO)

### 3.4 Swap-rule head-to-head vs v7

(TODO)

### 3.5 Ladder

(TODO)

### 3.6 Test suites (clause 4)

(TODO)

## 4. Bar check (v23 vs v7)

(TODO)

## 5. What this means

(TODO)
