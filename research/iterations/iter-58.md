# Iteration 58 — v58: diagnose v52's swap/colour interaction, then patch the mechanism found (batch 14, H2)

**Status: PRE-REGISTERED — diagnosis design fixed below (§1) BEFORE
any new eval of v58. The patch design (§3) will be written here
AFTER the diagnosis is in and BEFORE any implementation, per the
tasking. If the diagnosis finds no patchable mechanism, §3 says so
plainly and the work STOPS (batch 13 H3 precedent).**

**METHOD (carried over).** The fast 10-game loop stays the finding
method IF a patch is built: play 10 games, read every move, find
each game's losing move (solver/deeper-search verified, never
guessed), patch, regression-test, repeat. The 4-clause bar stays
the validation gate. Fast-loop games use development seeds
distinct from the bar battery's pre-registered seeds (100/200).
**Fast loops run against v0 as well as v52** (iter-17 §5). The
full battery runs on the FINAL version only. The 100-game battery
is the verdict — small-sample smokes mislead; single-position
diagnostics cannot validate a global search change.

## 0. The problem

v52 wins as White under swap rules but loses as White under normal
rules:

| leg (100 games) | v52 as Black | v52 as White |
|---|---|---|
| vs v27, normal rules | 39W-11L-0D (0.78) | 21W-27L-2D (0.44) |
| vs v27, swap rules | 38W-11L-3D (0.76) | 26W-22L-0D (0.54) |
| vs v18, normal rules (challenge) | 38W-11L-1D (0.77) | 23W-25L-2D (0.48) |

The swap White improvement (+0.10 vs v27) is concentrated in ONE
cell: v52-as-Black is flat (0.78 → 0.76) and v27-as-White is flat
(0.22 → 0.24, the control — see §1.1). The White gap as a
single-flaw framing is CLOSED (v41, v46 — the gap is
matchup-dependent, not a systematic defect), so this iteration
needs a genuinely new mechanism, not another White patch — the
swap/colour interaction is the new angle.

## 1. Diagnosis design (pre-registered)

### 1.1 Exploratory motivation (post-hoc re-analysis of committed battery data)

The following was computed from the already-committed v52 battery
JSONs (`batch-v52-vs-v27.json`, `batch-v52-vs-v27-swap.json`) BEFORE
this pre-registration was written. It is EXPLORATORY — it motivates
the hypotheses and the confirmatory measurements below, and is
labelled as such. No new evals were run before this section was
written. The confirmatory measurements (§1.3) are pre-registered
here with decision rules (§1.4) before they run.

* **Selection (mechanism (a)) looks refuted by arithmetic.** In the
  swap leg the second player chose Black 88/100 times (the tempo
  judge almost always prefers Black). v52's 48 White games split
  into CHOSEN White (v52 as second player picked White): n=5,
  3W-2L (0.600), and FORCED White (v52 as opener, v27 chose
  Black): n=43, 23W-20L (0.535). The chosen subset is 5/48 games
  — even at a 1.00 win rate it could only lift the White rate to
  (5 + 43×0.44)/48 = 0.499, below the observed 0.542; and the
  forced subset ALONE (0.535, n=43) already shows the improvement
  over normal-rules White (0.44). The swap White games are mostly
  HARD openings (the opponent took the better colour), yet v52
  wins them at 0.535. Selection cannot be the main explanation.
  (Also: the judge's White picks LOSE — the chosen colour won only
  4/12 when the second player took White, vs 52/88 + 3 draws when
  it took Black — so the judge's White choices do not favour the
  chooser at all.)
* **The v27 control: no general White-under-swap effect.**
  v27-as-White (vs v52-as-Black): 0.22 normal → 0.24 swap (flat).
  If the swap opening favoured White generally, v27-White should
  improve too. It does not. The improvement is specific to the
  (v52, White, swap) combination. Since v52 = v27 + TSS, the
  mechanism must involve the TSS — or the opponent's Black being
  weaker under swap (mechanism (c)).
* **The opening dominates the outcome.** 24 distinct openings in
  the swap leg; v52's per-opening win rate ranges from 0.17
  (opening (112,142,81), n=6) to 1.00 (several openings, n=2–3).
  In the 12 openings with ≥2 games in each role, v52-as-second
  (mostly Black) beats or ties v52-as-opener (mostly White) in
  10/12 — but the OPENING effect dwarfs the role effect (e.g.
  (112,142,81): 0.00 as opener, 0.25 as second; (112,126,81):
  1.00 as opener, 0.88 as second). Outcomes track the opening, not
  the choice.
* **Normal-rules White games all start identically**: v27-as-Black
  plays move 112 (centre) first in 50/50 games. Under swap,
  White faces 24 different openings. Swap White games are SHORTER
  (44.0 plies mean) than normal White games (47.3) — directionally
  consistent with White converting forcing wins faster under swap,
  but weak evidence on its own.
* **Game-length note:** swap games overall are LONGER than normal
  games (46.8 vs 42.6 plies mean) — the White-games shortening is
  specific to v52-as-White, not a general swap property.

### 1.2 Candidate mechanisms

* **(a) Selection** — the tempo judge picks the stronger colour for
  the second player, so the swap White games are a selected subset
  (easier openings) rather than a v52-White improvement.
  *Status after §1.1: refuted by arithmetic (see above). The
  confirmatory measurements below do not re-test (a); the
  refutation stands on the committed battery data.*
* **(b) TSS-mediated** — v52's TSS behaves differently as White
  under swap because the opening positions differ: the balanced
  3-stone opening gives White a developed position and a fairer
  race, so White reaches forcing positions more often, and only
  v52 (with TSS) converts them — v27 (no TSS) cannot, which is why
  the v27 control is flat. Two sub-questions:
  * (b1) OPPORTUNITY: does White get MORE forcing-win opportunities
    under swap (higher TSS fire/differ rate as White)?
  * (b2) DETECTION: does production TSS (depth 4, budget 3000)
    MISS White forcing wins that a deeper search would find — i.e.
    is White's normal-rules deficit partly a detection gap (patchable)
    rather than purely an opportunity gap (structural, not patchable)?
* **(c) Black-opening handicap (residual)** — the balanced placement
  handicaps Black's opening (Black's second stone is the
  gap-minimiser, not the engine's choice), and the handicap bites
  v27 (whose edge is opening/MCTS play) but not v52 (whose edge is
  TSS, which does not depend on the opening choice). Under (c),
  v52-White's swap improvement is really a v27-Black degradation —
  a harness property, NOT patchable in v52.
* **(d) Something else** — e.g. a TSS budget asymmetry (the node
  budget binding more often for White than Black), or an
  opening-specific TSS effect (White's differs clustering in
  specific openings). The mining records budget-exceeded rates by
  colour and differs by opening, so (d) is covered by the same
  measurements.

### 1.3 Measurements (pre-registered)

All measurements use the main repo's venv Python
(`~/workspace/projects/gomoku-ml-playground/ml/.venv/bin/python`)
with `PYTHONPATH=ml` in this worktree. All drivers checkpoint per
game and auto-resume (batteries must survive VM replacements).

**M1 — TSS mining, v27-vs-v27, normal vs swap (the decisive test
of (b1)).** New script `ml/scripts/mine_tss_swap.py` (adapted from
`mine_tss_positions.py`, Check B):
* 50 games per leg, seeds 777/1777 (SAME as Check B — the normal
  leg replicates Check B: it must reproduce 2481 turns / 288 fires
  / 121 differs, and adds by-colour breakdowns).
* Swap leg: balanced placement, placement_seed 11 (the battery's
  placement seed), tempo judge; per-game placement RNG seeded
  `11 + game_index`, exactly as `selfplay_batch.py` does. The
  mining's swap openings are therefore the battery's first 50
  openings.
* At every turn, run production TSS (`tss_winning_move`, depth 4,
  budget 3000) WITHOUT perturbing the game (TSS consumes no RNG)
  and record, BY COLOUR: turns, fires, confirms (TSS move ==
  v27's move), differs (TSS move != v27's move), budget-exceeded
  count, node sum. Record game outcomes (winner colour, plies) and,
  for swap games, the opening and per-opening differ counts.
* Record qualifying White turns for M3 (see below): the first 30
  per leg where production TSS returns None AND the forcing move
  set is non-empty AND Black holds no winning move (no forced
  block) AND the search completed (budget NOT exceeded). Board
  stored as black/white stone lists.
* NO playout verification (Check B already verified differs convert
  5/5; TSS fires are exact by construction — the search proves a
  forced win. This mining measures RATES, not verification).

**M2 — Battery replay (TSS rates in the ACTUAL v52-vs-v27 games).**
New script `ml/scripts/replay_tss_battery.py`: replay the recorded
move lists of `batch-v52-vs-v27.json` (normal) and
`batch-v52-vs-v27-swap.json` (swap); at every turn run production
TSS for the side to move and record, by (engine, colour): turns,
fires, confirms, differs. For v52's turns, a fire MUST coincide
with v52's recorded move (v52 plays the TSS move when it fires) —
this is a consistency check on the replay (expect ~100%; any
mismatch is a replay bug, investigated before the numbers are
used). For v27's turns, differs are opportunities v27 missed in
the real battery. Deterministic; checkpoint per game.

**M3 — Depth-6 spot-check (the decisive test of (b2), detection
vs opportunity).** New script `ml/scripts/spotcheck_tss_depth.py`:
on the qualifying White turns recorded by M1 (first 20 per leg,
normal-rules leg FIRST — that is where a patch would apply), run
`tss_winning_move(board, WHITE, depth=6, node_budget=10000)` and
record: found a win / None-completed / None-budget-exceeded.
A depth-6 find on a turn where depth-4 found nothing is a
DETECTION gap (patchable in principle); no finds means TSS
detection is complete for White and the deficit is opportunity
(structural, not patchable). If the per-call cost is prohibitive
(depth-6 explores far more nodes), the sample is reported as run
with the abort rate stated — no silent sample-size change.

### 1.4 Decision rules (pre-registered)

* **(a) Selection:** REFUTED (arithmetic, §1.1). Not re-tested.
* **(b1) Opportunity:** (b) CONFIRMED if White's differ rate in M1
  is ≥1.5× under swap vs normal rules AND the difference exceeds
  2 binomial SEs; (b) REFUTED if the ratio is in [0.8, 1.25] on
  BOTH M1 (White differ rate) and M2 (White fire rate in v52's own
  games / White differ rate on v27's turns in the battery). A ratio
  in (1.25, 1.5) is INCONCLUSIVE — M2 breaks the tie under the same
  thresholds; if still inconclusive, the diagnosis reports
  inconclusive and does NOT build a patch on it.
* **(b2) Detection (only tested if (b1) is confirmed):** a detection
  gap is CONFIRMED if depth-6 finds a win in ≥3/20 sampled
  normal-rules White turns (15%); NO GAP if 0/20; 1–2/20 is
  AMBIGUOUS and counts as NO PATCH (a 5–10% find rate on a selected
  subsample is too small to move the battery, and H1 owns the depth
  lever — a White-specific depth patch on ambiguous evidence would
  duplicate H1's sweep).
* **(c) Black-opening handicap:** the RESIDUAL explanation. If (b)
  is refuted, the improvement is attributed to the opponent's Black
  being handicapped by the balanced placement (a harness property),
  corroborated by M1's game outcomes (Black win rate in v27
  self-play under swap vs normal — reported as a secondary, not a
  decision input, since the swap choice confounds it).
* **Patch decision:** a patch is built ONLY if (b) is confirmed AND
  M3 (or a (d) sub-measurement, e.g. a White-specific budget
  asymmetry) identifies a SPECIFIC patchable sub-mechanism. If (b)
  is confirmed but no sub-mechanism is found (pure opportunity gap),
  or if (b) is refuted, or if M3 is ambiguous — NO PATCH. The
  diagnosis is the deliverable; iter-58.md says so plainly and the
  work STOPS (batch 13 H3 precedent). The patch design, if any, is
  written in §3 BEFORE implementation.
* **H1 overlap note:** H1 (v57) owns the global TSS depth/budget
  sweep. A v58 patch on depth/budget is justified ONLY by a
  White-specific asymmetry found in M1/M3 (e.g. White's wins are
  deeper, or the budget binds for White specifically) — a global
  depth/budget change belongs to v57, and v58 will not duplicate
  it. The coordinator adjudicates any overlap.

### 1.5 What does NOT change (if a patch is built)

100 sims, c_puct 1.5, sharp v1 priors, the rollout leaf, the τ=0.05
tie-break, most-visited root, v52's TSS as the base. No q-based root
decision (CLOSED). No sims change. No prior change. No defensive
VCF/TSS check (CLOSED in all forms). No leaf change. No White
opening book / White search / White prior change (CLOSED — v41,
v46). `baseline.py` stays FROZEN. No kernel re-sync unless v58
passes the bar.

### 1.6 Eval seeds (pre-registered, for use ONLY if a patch is built)

* Fast-loop dev seeds: 1058/2058 (vs v0), 1059/2059 (vs v52) —
  distinct from the battery's 100/200 and from batch-13's loop
  seeds (1052/2052, 1053/2053).
* Battery h2h vs v52: v58=100 / v52=200; vs v0: v58=100 / v0=200.
* Ladder: seed 100, 40 games/opponent, mcts_sims=100
  (`ml/scripts/eval_version.py --version 58 --ladder-games 40
  --out research/iterations/ladder-v58.json`).
* Swap: 100 games vs v52, balanced placement, placement_seed 11,
  tempo judge, seeds 100/200.
* Tactical: `eval_tactical_full.py --version 58 --suite both
  --seed 100`. v52's tactical is re-verified first on the
  corrected suite (expect 20/20 + 29/29).
* Challenge match (ONLY if v58 passes all 4 clauses): 100 games
  vs v52 (the champion), seeds 100/200, normal rules. Note: clause
  1 IS vs v52, so a bar-passer has already beaten the champion
  h2h; the challenge is then a confirmatory re-run at the same
  seeds — the coordinator decides whether a second match is needed
  (a same-seed re-run would replay identical games; fresh seeds
  would be seed-shopping in reverse. Pre-registered position: if
  v58 passes the bar, the title question is decided by the clause-1
  match itself plus the ladder leg-result reading, and no separate
  challenge match is run).
* Harness commands follow iter-52 §1.5 exactly
  (`fast_loop_match.py`, `selfplay_batch.py`, `eval_version.py`,
  `eval_tactical_full.py`). All match drivers AUTO-RESUME from
  partial `--out` files — if a run is killed, relaunch the SAME
  command. Per-game seeding with a uniqueness audit on every leg.

### 1.7 The bar (4 clauses, ALL required; re-anchored on v52)

(1) h2h vs v52 ≥100 games, win rate > 0.50 with Wilson 95% CI
lower bound > 0.50; (2) ladder Elo beats v52's — judged by the
LEG-RESULT reading: strictly dominate v52's legs (random 40-0,
heuristic 40-0, mcts 31-9), i.e. match the two perfect legs and
beat 31-9 on the mcts leg; (3) tactical ≥ 20/20 + 29/29 on the
corrected suite; (4) all test suites green (ml pytest, npm test
55/55, tsc clean). Fail any clause = REJECTED. Do NOT claim a
"highest Elo ever" from a single draw. Do NOT re-run at another
seed to shop for a pass — no seed-shopping, in either direction
(v44 precedent).

## 2. Diagnosis results

### 2.1 M1 — TSS mining, v27-vs-v27, normal vs swap (COMPLETE)

`ml/scripts/mine_tss_swap.py`, 50 games/leg, seeds 777/1777; swap
leg balanced placement, placement_seed 11, tempo judge.
Artifacts: `research/iterations/tss-mine-v58-normal.json`,
`tss-mine-v58-swap.json` (analysis:
`ml/scripts/analyse_tss_mine_v58.py`). The run survived three VM
replacements (SIGTERM at normal 10/50, swap 4/50, swap 45/50);
per-game checkpointing resumed each leg exactly.

**Check B replication: EXACT.** The normal leg reproduces
iter-52's Check B numbers precisely: 2481 turns / 288 fires /
121 differs. The script and the by-colour breakdowns are validated.

| leg / colour | turns | fires | fire rate | differs | differ rate | confirms | budget-exc. | mean nodes |
|---|---|---|---|---|---|---|---|---|
| normal Black | 1256 | 188 | 0.1497 | 86 | 0.0685 | 102 | 52 (4.14%) | 327.5 |
| normal White | 1225 | 100 | 0.0816 | 35 | 0.0286 | 65 | 68 (5.55%) | 366.1 |
| swap Black | 1387 | 133 | 0.0959 | 51 | 0.0368 | 82 | 95 (6.85%) | 429.7 |
| swap White | 1405 | 128 | 0.0911 | 59 | 0.0420 | 69 | 73 (5.20%) | 378.5 |

Game outcomes (v27 self-play): normal Black 29W / White 19W / 2D
(Black win rate 0.60), mean 49.6 plies; swap Black 28W / White
18W / 4D (Black win rate 0.60), mean 55.8 plies. Swap games are
longer (more turns per game), which the per-turn rates already
account for.

**Decision quantities (mechanism (b), pre-registered §1.4):**

* White differ rate: normal 0.0286 (35/1225) → swap 0.0420
  (59/1405), **ratio 1.47, z = +1.87**. Below BOTH confirm
  thresholds (ratio ≥ 1.5 AND z > 2). **INCONCLUSIVE.**
* White fire rate: normal 0.0816 → swap 0.0911, ratio 1.12,
  z = +0.86. No gap.
* Black differ rate: normal 0.0685 → swap 0.0368, ratio 0.54,
  z = −3.63. Black's missed-forcing-win rate FALLS significantly
  under swap.
* Black fire rate: normal 0.1497 → swap 0.0959, ratio 0.64,
  z = −4.20. Black has significantly FEWER forcing wins available
  under swap.

**Reading.** Under swap, the colours' forcing-opportunity rates
CONVERGE: Black loses forcing wins (significantly) and White gains
some (not significantly). The total differ rate actually falls
(121/2481 = 4.88% normal → 110/2792 = 3.94% swap) because Black's
loss exceeds White's gain. The White gain (ratio 1.47, z = 1.87)
is suggestive but does NOT clear the pre-registered bar — M1 alone
cannot confirm (b). Per §1.4, M2 (battery replay) is the
tiebreaker.

**Mechanism (d) sub-measurements (from the same mining):**

* Budget asymmetry: normal White 5.55% vs Black 4.14% (White
  binds more); swap Black 6.85% vs White 5.20% (Black binds
  more). The asymmetry FLIPS between rules — there is NO stable
  White budget disadvantage. The budget-asymmetry sub-mechanism is
  refuted. (Side note: the budget binds on ~5% of turns in real
  games, not the ~1% of iter-52's bench — the bench positions were
  quieter than real-game positions.)
* Per-opening White differs (swap leg) are spread across openings
  (max 7 in (112,142,81) and (112,144,176); 19 of 24 openings have
  ≥1 White differ). No single opening drives the White differ
  increase. The opening-specific sub-mechanism is weak.
* Black's win rate in v27 self-play is 0.60 under BOTH rules — the
  swap choice compensates for the opening handicap in self-play,
  so the self-play Black win rate does not discriminate (c) (as
  flagged in §1.4, it was a secondary, not a decision input).

**Qualifying White turns for M3:** 30 recorded per leg (normal and
swap), as pre-registered.

### 2.2 M2 — Battery replay (scope amendment, then run)

**SCOPE AMENDMENT (made BEFORE any M2 eval ran).** §1.3
pre-registered M2 as "TSS at every turn". A timing probe (3 swap
games, `/tmp` output, not part of the measurement) measured
~100 s/game full-scope, i.e. ~5.5 h for both 100-game legs — too
slow for a tiebreaker, and the daily cross-track cron adds further
contention from 08:37. M2 is therefore scoped to **White turns
only** (both engines, all 100 games per leg), via the new
`--white-only` flag on `replay_tss_battery.py`. Justification:
both pre-registered M2 decision quantities (§1.4) are White-turn
quantities — v52's White fire rate in its own games, and v27's
White differ rate in the battery. Black turns are a non-decision
control, and M1 already measured Black's rates under both rules
(they fall significantly under swap). The decision rule (§1.4) is
UNCHANGED. The probe also validated the replay: 0 consistency
violations in 3 games (v52's recorded moves match TSS fires).

**M2 results (COMPLETE).** Artifacts:
`tss-replay-v58-normal.json`, `tss-replay-v58-swap.json` (White
turns only, all 100 games per leg). The run survived one VM
replacement (SIGTERM at swap 80/100); per-game checkpointing
resumed exactly. Consistency check: **0 violations in both legs**
— v52's recorded moves match TSS fires everywhere, so the replay
is valid.

| leg / engine-White | turns | fires | fire rate | differs | differ rate | confirms | budget-exc. |
|---|---|---|---|---|---|---|---|
| normal A-white (v52) | 1167 | 118 | 0.1011 | 0 | 0.0000 | 118 | 45 (3.86%) |
| swap A-white (v52) | 996 | 158 | 0.1586 | 0 | 0.0000 | 158 | 59 (5.92%) |
| normal B-white (v27) | 928 | 75 | 0.0808 | 29 | 0.0312 | 46 | 51 (5.50%) |
| swap B-white (v27) | 1211 | 72 | 0.0595 | 30 | 0.0248 | 42 | 63 (5.20%) |

(A-white differs are 0 by construction — v52 plays the TSS move
when it fires.)

**Decision quantities (§1.4): the two M2 quantities SPLIT.**

* v52's White fire rate: normal 0.1011 → swap 0.1586, **ratio
  1.57, z = +3.95** — clears BOTH confirm thresholds (≥1.5, z>2).
  On the letter of the pre-registered rule, M2 CONFIRMS (b).
* v27's White differ rate: normal 0.0312 → swap 0.0248, **ratio
  0.79, z = −0.89** — no rise. (v27-White's fire rate also FELL:
  ratio 0.74, z = −1.90.)

**Why the split does not settle (b).** The two quantities measure
opportunities against DIFFERENT opponents: v52-White faces weak
v27-Black, while v27-White faces strong v52-Black (whose TSS does
not depend on the opening, so it is not weakened by swap). More
importantly, the fire rate is confounded by win-more: it rises
when v52-White wins more and plays shorter games, even with NO new
opportunities. Decomposition of the 1.57× fire-rate rise: win rate
0.44 → 0.54 (1.23×) · shorter v52-White games (23.3 → 20.8 White
turns/game, 1.12×) · fires per win 5.62 → 6.08 (1.08×) ≈ 1.49×
(≈ the observed 1.57×; residual is fires in lost/drawn games).
Fires per win — the closest thing to "opportunities per won game"
in the aggregate data — rises only 1.08×. The fire-rate "confirm"
is therefore largely an EFFECT of winning more, not evidence of
new forcing opportunities. The differ rate (opportunities the
MCTS misses) is the cleaner causal quantity, and it does not rise
in the battery (M2 B-white 0.79×) and rises only suggestively in
self-play (M1 1.47×, z = 1.87, below threshold).

**M2b — attribution follow-up (POST-HOC, motivated by M2's split
result; not pre-registered).** The aggregate data cannot say
whether v52-White's EXTRA swap wins (+5: 26 vs 21) were TSS-driven.
M2b replays the v52-White games only (50 normal + 48 swap),
White turns, recording PER-GAME fire counts alongside the outcome
(`--per-game --only-a-colour white`). Decision logic: if the extra
swap wins disproportionately involve White TSS fires, (b) is
causally supported and M3 runs; if the extra wins are
disproportionately fire-free (won by outplaying a weakened
v27-Black without TSS), (b) is causally refuted and the fire-rate
"confirm" is win-more. M2b does not change the pre-registered
decision rule — it tests whether the rule's fire-rate quantity was
confounded, and the patch decision follows the CAUSAL evidence.

### 2.3 M2b — attribution (per-game fires for v52-White games)

(to be filled in when M2b completes)

## 3. Patch design

(to be written AFTER the diagnosis is in and BEFORE any
implementation — or to record NO PATCHABLE MECHANISM and stop)

## 4. Fast loop

## 5. Battery

## 6. Verdict
