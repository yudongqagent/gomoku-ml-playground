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

**COMPLETE.** Artifacts: `tss-replay-v58-normal-pergame.json`,
`tss-replay-v58-swap-pergame.json` (v52-White games only, White
turns, per-game fire counts + outcome). No interruption; both legs
exit 0. Aggregate fires match M2 exactly (118/1167 normal,
158/996 swap). 0 violations.

| leg / outcome | games | White TSS fires | fires/game | games with ≥1 fire | fire rate |
|---|---|---|---|---|---|
| normal WIN | 21 | 118 | 5.62 | 21 (100%) | 0.2593 (118/455) |
| normal LOSS | 27 | 0 | 0.00 | 0 (0%) | 0.0000 (0/488) |
| normal DRAW | 2 | 0 | 0.00 | 0 (0%) | 0.0000 (0/224) |
| swap WIN | 26 | 148 | 5.69 | 26 (100%) | 0.2874 (148/515) |
| swap LOSS | 22 | 10 | 0.45 | 2 (9%) | 0.0208 (10/481) |

**The finding is stark: TSS firing is (near-)NECESSARY for
v52-White to win.** All 47 v52-White wins across both legs
involved at least one White TSS fire (100%); v52-White fired in
only 2 of 49 losses (both under swap, 3 and 7 fires — positions
where White held a forcing win but still lost, i.e. the forcing
win did not convert, or the fire came too late). Fires per win are
nearly identical under both rules (5.62 normal, 5.69 swap) — the
extra swap wins are NOT "more fires per win", they are MORE GAMES
in which White reaches a firing position (26/48 = 54% of swap
games vs 21/50 = 42% of normal games).

**Attribution verdict: (b) is causally supported.** The extra swap
wins are 100% TSS-involved, and the swap White improvement runs
through White reaching forcing-win positions more often (54% vs
42% of games). The M2 fire-rate "confirm" (1.57×, z = 3.95) is
therefore not merely win-more: the fire rate rises because White
reaches forcing positions in more games, and reaching one is what
wins White's games. The remaining causal caveat is the
confirm/differ split WITHIN v52's won games (would the MCTS have
found those wins anyway?): M1's self-play split says under swap a
higher fraction of White's fires are differs (46% vs 35% normal),
i.e. under swap White's forcing wins are more often ones the MCTS
would miss — consistent with TSS adding real value for White under
swap. A direct per-fire confirm/differ check in v52's won games
would need the firing positions recorded (not saved by this
replay); it is NOT run, because the patch decision turns on M3,
not on this split.

**What M2b does NOT show:** a patchable mechanism. M2b shows White
wins by reaching forcing positions, and reaches them more often
under swap — an OPPORTUNITY difference driven by the opening
(balanced 3-stone start + choice vs Black's first-move advantage
from the empty board). Under normal rules White's opportunity
deficit is structural (Black moves first; Black builds threats
earlier — M1: Black's fire rate 15.0% vs White's 8.2%). No engine
change creates forcing opportunities that do not exist. The ONLY
patchable form of (b) is a DETECTION gap — White HAS forcing wins
under normal rules that production TSS (depth 4) misses. That is
exactly what M3 tests.

### 2.4 M3 — depth-6 spot-check (detection vs opportunity)

**COMPLETE.** `ml/scripts/spotcheck_tss_depth.py`: depth-6 TSS,
node budget 10000, on the first 20 qualifying White turns per leg
from M1 (production TSS returned None, forcing set non-empty, no
forced block, search completed). Artifacts:
`tss-spotcheck-v58-normal.json`, `tss-spotcheck-v58-swap.json`.
No interruption; both legs exit 0.

| leg | n | found | none (completed) | none (budget-exceeded) | find rate | nodes median / max |
|---|---|---|---|---|---|---|
| normal | 20 | 0 | 17 | 3 | 0.000 | 817 / 10001 |
| swap | 20 | 0 | 16 | 4 | 0.000 | 2548 / 10001 |

**Per the pre-registered decision rule (§1.4): 0/20 = NO GAP.**
Depth-6 TSS finds ZERO additional White forcing wins beyond what
depth-4 finds, in 33 completed searches across both legs. TSS
detection is complete for White at depth 4: White's normal-rules
deficit is a pure OPPORTUNITY gap (White genuinely has fewer
forcing wins because Black moves first), not a detection gap.

Caveats, stated plainly:
* 7/40 searches aborted on the 10000-node budget (3 normal, 4
  swap) and are inconclusive for those positions — a win COULD
  hide behind an abort (the winning move late in prior order,
  budget consumed by earlier subtrees). A budget follow-up
  (below) re-runs all 7 aborted positions at budget 50000.
* The sample is the FIRST 20 qualifying turns (games 0–2, plies
  7–31 normal / 4–36 swap) — early-game concentrated, no late-game
  positions. Late-game White turns with forcing moves but no
  depth-4 win are rare (by the late game someone usually holds a
  forcing win or the game is decided), but the sample does not
  cover them.
* The qualifying population is v27-vs-v27 mining games, not v52's
  own lost games. M3b (below) tests the decision-relevant
  population directly.

**M3 budget follow-up (POST-HOC robustness check, COMPLETE).** The
7 aborted positions re-run at depth 6, budget 50000
(`tss-spotcheck-v58-{normal,swap}-budget50k.json`): **0/7 finds**.
5/7 resolved to completed no-wins (nodes 10k–47k); 2 normal-leg
positions (game 2, plies 21 and 27) still abort even at 50000
nodes — extremely bushy "prove no win" searches, still
inconclusive. M3's final tally: 0 finds in 38 completed depth-6
searches (17+1 normal, 16+4 swap), 2 inconclusive.

**M3b — depth-6 spot-check on v52-White's lost games (POST-HOC
robustness check, the decision-relevant population, COMPLETE).**
`ml/scripts/collect_loss_turns.py` replayed v52-White's 29
non-wins (27 losses + 2 draws) from `batch-v52-vs-v27.json`: all
712 White turns had production TSS = None (confirming M2b's 0
fires in losses), 30 qualifying turns collected (plies 5–67,
games 1/5/9/11). Depth-6 (budget 10000) on the first 20
(`tss-spotcheck-v58-losses.json`): **0/20 finds** (9 completed,
11 aborted — a 55% abort rate, far higher than M3-mining's 17.5%:
losing positions have bushier search trees, Black holding strong
counter-replies). The 11 aborted positions are re-run at budget
50000 (`tss-spotcheck-v58-losses-budget50k.json`) — result below.

**M3b budget follow-up (POST-HOC robustness check, COMPLETE).**
The 11 aborted M3b positions re-run at depth 6, budget 50000
(`tss-spotcheck-v58-losses-budget50k.json`): **0/11 finds** (8
resolved to completed no-wins, nodes 10k–47k; 3 still abort even
at 50000 nodes).

**FINAL depth-6 tally: 55 completed searches, 0 finds; 5 still
inconclusive** (2 M3-mining + 3 M3b positions abort even at 50000
nodes — extremely bushy "prove no win" searches). 95% CI upper
bound on the find rate: 5.5% (rule of three). Depth-6 TSS finds
nothing White is missing at depth 4, in two independent
populations (v27 self-play mining turns and v52's own lost
games).

### 2.5 Diagnosis synthesis

**The question:** why does v52 win as White under swap (0.54 vs
v27) but lose as White under normal rules (0.44 vs v27, 0.48 vs
v18)?

**The answer, in one paragraph:** it is an OPPORTUNITY gap plus a
Black handicap, not a v52 defect. Under swap, the balanced 3-stone
opening gives White a fairer race, so White reaches forcing-win
positions more often (54% of games vs 42% under normal rules) —
and reaching one is what wins White's games (TSS firing is
necessary for v52-White to win: 47/47 wins involve a fire).
Simultaneously the swap rule takes forcing wins AWAY from Black
(Black's fire rate falls 15.0% → 9.6%, significantly), because the
balanced placement denies Black its chosen opening — and that
handicap bites v27 (whose edge is opening/MCTS play) but not v52
(whose edge is TSS). So v52-White improves under swap because it
faces a weaker v27-Black AND gets more forcing opportunities,
while v27-White does not improve because it faces v52-Black,
which the swap rule does not weaken. There is no patchable
mechanism in v52: TSS detection is complete for White at depth 4
(0 finds in 55 completed depth-6 searches), and no engine change
creates forcing opportunities that do not exist under normal
rules.

**Mechanism-by-mechanism verdict:**

| mechanism | verdict | evidence |
|---|---|---|
| (a) Selection (swap choice) | REFUTED | Chosen-White subset is 5/48 games; forced-White alone is 0.535 (n=43) vs normal 0.44. Arithmetic, not inference. The judge's White picks lose (4/12). |
| (b) TSS-mediated (White gets more forcing opportunities under swap) | SUPPORTED as the explanation, but a pure OPPORTUNITY gap — NOT patchable | M1 White differ rate 1.47× (z=1.87, just below threshold); M2 v52-White fire rate 1.57× (z=3.95); M2b: TSS firing necessary for v52-White wins (47/47), firing games 54% swap vs 42% normal. M3+M3b: NO detection gap (0/55 depth-6 finds). |
| (c) Black-opening handicap | SUPPORTED as a contributor | M1: Black's fire rate 0.64× (z=−4.20) and differ rate 0.54× (z=−3.63) under swap, both significant. Explains why v27-White (vs v52-Black) is flat while v52-White (vs v27-Black) improves. Harness property, not patchable in v52. |
| (d) Budget asymmetry | REFUTED | Flips between rules (normal W 5.55% > B 4.14%; swap B 6.85% > W 5.20%). No stable White disadvantage. |
| (d) Opening-specific TSS | WEAK | White differs spread across 19/24 openings (max 7 in one). No single opening drives the effect. |

**Side findings (for the program record):**
* The TSS node budget binds on ~5% of turns in real games
  (M1: 4.1–6.9% by leg/colour), not the ~1% of iter-52's bench —
  the bench positions were quieter than real-game positions.
  H3's speed work (v59) should use the real-game rate.
* Depth-6 TSS aborts far more often than depth-4 (17.5% of
  mining turns, 55% of losing-position turns at a 10k budget; 5
  positions still abort at 50k). Any depth increase (H1's v57
  sweep) must scale the budget or aborts will eat the gains —
  and a win late in prior order can hide behind an abort.
* v52-White's losses are TSS-silent: 0 fires in 27 normal-rules
  losses (0/488 turns) and 0 in 2 draws. When White loses, it is
  not because TSS missed a win — it is because no forcing win
  existed at any turn (at depth 4 or 6).

## 3. Patch design

**NO PATCHABLE MECHANISM — WORK STOPS HERE (batch 13 H3
precedent).**

The diagnosis found the mechanism (an opportunity gap + a Black
handicap), and neither half is patchable in v52:
* The opportunity gap is structural: under normal rules Black
  moves first and builds threats earlier, so White genuinely has
  fewer forcing wins. No engine change creates forcing wins that
  do not exist. The White search/prior/book levers that might
  steer White toward forcing positions are CLOSED (v15, v41, v46).
* The Black handicap is a harness property (the balanced placement
  denies Black its chosen opening) and an opponent property (it
  bites v27, not v52). Nothing in v52's code causes it or can
  exploit it further — v52 already converts every forcing win it
  reaches.
* The only patchable form of (b) — a detection gap — is refuted:
  depth-6 TSS finds 0 additional White wins in 55 completed
  searches across two populations. TSS detection is complete for
  White at depth 4.

No v58 engine is built. No registry entry, no VERSIONS.md row, no
fast loop, no battery, no ladder, no challenge match. The
diagnosis in §2 is the deliverable. The measurement scripts
(`mine_tss_swap.py`, `replay_tss_battery.py`,
`spotcheck_tss_depth.py`, `collect_loss_turns.py`,
`analyse_tss_mine_v58.py`) and all JSON artifacts are committed on
this branch for the coordinator to merge as research artifacts;
no engine code changed (`ml/gomoku_ml/` untouched).

## 4. Fast loop

N/A — no version built (see §3).

## 5. Battery

N/A — no version built (see §3).

## 6. Verdict

**v58: NO PATCHABLE MECHANISM FOUND — diagnosis delivered, no
version built, work stopped per the tasking.**

The swap/colour interaction is understood (see §2.5) and the
White gap is now triply confirmed non-patchable in its current
framings (v41 forensics, v46 population statistics, v58
opportunity/detection decomposition). The heuristic bar reference
and strength champion remain v52; the daily cross-track pairing
(v52 vs v44) is unchanged.

**For the next batch:**
1. Do NOT retry White-side patches on this framing. The
   opportunity gap is structural and the detection gap is closed
   (0/55). Any future White work needs a genuinely new mechanism,
   not a re-framing of the gap.
2. H1's depth/budget sweep (v57) should note M3's result: depth-6
   finds nothing White misses at depth 4 — but M3 tested WHITE
   turns only. If depth helps globally, it is through a different
   population (Black's wins, or positions without forcing moves),
   and the budget must scale with depth (55% abort rate in losing
   positions at 10k nodes; 5 positions abort even at 50k).
3. The one speculative direction M2b suggests: v52-White wins
   ONLY by reaching a forcing position (47/47 wins), and reaches
   one in only 42% of normal-rules games. A lever that steers
   White's early/midgame toward forcing positions — e.g. a
   threat-potential term in the leaf evaluation rewarding
   progress toward forcing structures, distinct from the CLOSED
   prior/book levers — is the only untested angle on White. It is
   an idea, not a recommendation: it would need its own
   pre-registered mechanism check (does the term actually raise
   White's firing-game rate?) before any battery.
4. The TSS budget's real-game bind rate (~5%, not ~1%) belongs in
   H3's (v59) speed analysis, and the 5 ultra-bushy positions
   (still aborting at 50k nodes) are a TSS robustness note for
   any future depth work.


### 2.5 Diagnosis synthesis

(to be filled in when the follow-ups complete)

## 3. Patch design

(to be written AFTER the diagnosis is in and BEFORE any
implementation — or to record NO PATCHABLE MECHANISM and stop)

## 4. Fast loop

## 5. Battery

## 6. Verdict

---

# RE-SCOPE — v58: threat-potential term in the leaf evaluation (new worker, 2026-11-23)

The diagnosis above (§§0–6) found no patchable mechanism for
v52's swap/colour interaction and stopped per its tasking. The
coordinator re-scoped H2 to the ONE speculative direction the
diagnosis leaves open (§10 / §6 item 3): a threat-potential term
in the leaf evaluation. Everything below is the re-scope's work,
clearly separated from the diagnosis. The diagnosis (§§0–6),
its measurement scripts, and its JSON artifacts are unchanged.

**Status: PRE-REGISTERED — design fixed below (§11) BEFORE any
eval of the re-scoped v58. The cost bench (§11.4) ran BEFORE
this pre-registration was committed (the iter-52 §1.4 precedent:
the bench informs the budget; it is a cost measurement, not a
strength eval). The mechanism check (§13) runs only after this
section is committed.**

## 11. Re-scope design and pre-registration

### 11.1 The term (ONE definition)

**Definition.** At the rollout-end position — the position where
v52's leaf evaluates `_attack_gradient_v1`, after exactly 2
rollout plies under `_rollout_move` — let `F(p)` be the EXACT
number of forcing moves (the TSS move set: four-creators ∪
true-open-three-creators) held by player `p`, enumerated by the
TSS machinery: four-creators via the v1 kernel prefilter (>=
20000) + a local exact test (`_winning_moves_through`), open-
three creators via the v0 kernel (>= 1000) minus four-creators.
The term is:

```
T = 0.05 * tanh((F(ours) - F(theirs)) / 1.0)
```

where "ours"/"theirs" are from the side-to-move's perspective at
the rollout end. The leaf value is:

```
value = clamp(_attack_gradient_v1 + T, -0.2, +0.2)
```

**Why this definition (justification).**
* *The rollout end, not the leaf start:* forcing potential that
  survives the opponent's best reply (the rollout's ply 2) is
  what matters — a forcing move the opponent kills on the reply
  is not potential. The gradient is evaluated at the rollout end
  for the same reason (iter-16: the leaf horizon extends past the
  forcing exchange); the term follows the same logic.
* *Forcing moves (both classes, counted equally):* both classes
  are TSS forcing moves, and TSS fires on either — reaching a
  firing position is the diagnosis's mechanism (M2b: TSS firing
  is necessary for v52-White to win, 47/47), so the count of
  forcing moves is the direct measure of proximity to a firing
  position. Four-creators and open-three creators are NOT
  weighted differently: a win through open threes is as much a
  win as one through fours (the TSS itself treats them as one
  move class, ordering four-creators first only for speed).
* *Weight 0.05, scale 1.0:* a one-forcing-move advantage shifts
  the leaf by 0.05·tanh(1) ≈ 0.038 — about 40% of the gradient's
  typical midgame magnitude (~0.10) — a meaningful nudge that
  cannot dominate the gradient. The tanh saturates at a
  difference of 2 (0.0482, 96% of the cap): holding 2+ more
  forcing moves than the opponent is already "clearly ahead in
  threat potential". The weight is half the gradient's cap and a
  quarter of the must-block (0.25).
* *Cap (the iter-7 invariant):* the SUM is clamped to ±0.2, so
  quiet leaves never outweigh the smallest exact terminal value
  (the must-block 0.25). The term's own bound is ±0.05.
* *Sign:* positive when we hold more forcing moves than the
  opponent, negative when fewer; antisymmetric under colour swap,
  like the gradient.
* *Where it does NOT apply:* terminals are exact (leaf-start
  terminals, and terminals reached inside the rollout, return
  exact values — the term never touches them). On a rollout-end
  must-block (the opponent holds exactly 1 winning move) the term
  is SKIPPED and the gradient alone is returned: the opponent's
  forcing count is degenerate when the opponent already holds a
  winning move (the local four-creator test would count cells
  near the pre-existing winning square as creators — the
  `_four_creating_moves_fast` contract note), and the must-block
  dominates the position anyway.
* *Both colours:* the term applies at every leaf, for whichever
  side is to move. It is NOT a White patch (see §11.2). The
  mechanism check measures White's firing rate only because that
  is the diagnosis's quantity.

**What is NEW vs the gradient (the honest account).** The
gradient's strong tier (v1 attack >= 4000) already counts
four-creators and v1-visible open-three creators, so the term
partly re-weights moves the gradient already sees — forcing
moves count ~2.4x generic strong moves (once in the breadth,
once in the term). The term's GENUINELY NEW signal is the
v1-blind open-three creators: moves v1 scores below 4000 (in the
constructed case, 6 — below even the developing threshold) that
create a true open three. Those moves are invisible to the
gradient AND to the priors (both use v1 scores); the term is the
only signal valuing them. The bench (§11.4) measures how often
the term is nonzero; the mechanism check (§11.3) measures
whether it changes behaviour.

### 11.2 Distinction from CLOSED levers (explicit)

* **The leaf-blind direction (v40) is closed, but v40 was about
  the leaf MIS-RANKING on specific prevention instances** — and
  the finding was that the leaf ranks the unsafe move worst
  already; the mis-ranking lives in backed-up root q / visit
  allocation (a search problem). This is a different mechanism:
  a NEW POSITIVE term for threat creation. The leaf's ranking of
  preventions is untouched (terminals and the must-block are
  exact, as in v52); nothing here re-opens v40's mis-ranking
  question.
* **The prior/book levers (v13, v16) are about the PRIOR, not
  the leaf value.** v58's priors are v52's exactly (sharp v1
  priors, no floor, no temperature). No opening book, no
  White-search change.
* **The White gap as a single-flaw framing (v41, v46, v58
  diagnosis) is closed.** This term is not a White patch: it
  applies to both colours, and the White firing rate is a
  measurement choice (the diagnosis's quantity), not a
  White-specific mechanism.
* **Rollout-depth changes (v53) are closed.** The rollout stays
  exactly 2 plies under the same policy; only the rollout-end
  evaluation gains the term.
* **The blend (v19, v21) is closed.** No blending of terminal
  values with the gradient; terminals stay exact.

### 11.3 Mechanism check (pre-registered)

The tasking's stop rule: if the mechanism check shows the term
does not change forcing-position reach rates, say so and stop —
do not force a version. The DECISIVE quantity is the reach rate
(Check B); Check A is diagnostic (does the term change move
choice at all?).

**Check A — move-choice on mined miss positions.**
Mine: replay v52-White's 27 normal-rules losses from the
committed `batch-v52-vs-v27.json`. At each White turn, enumerate
White's forcing moves (`tss_forcing_moves`); record turns where
White has >= 1 forcing move, v52's recorded move is NOT a
forcing move, White holds no winning move, and Black holds no
winning move (else the win/block scans force the move). Collect
the FIRST 20 such turns (positions stored as stone lists).
Sanity: TSS must be silent on all 20 (v52 fired 0 times in
these losses — M2b — and v58's TSS is identical, so any forcing
move v58 plays comes from the MCTS + term, not the hook).
Check: run v58 (seed 100) on each position; count how often v58
plays a forcing move.
Decision: PASS if >= 6/20 (30%); FAIL if <= 2/20 (10%); WEAK
3–5/20. (v52 plays a forcing move on 0/20 by construction.)
Script: `ml/scripts/collect_forcing_miss_positions.py` (mining)
+ `ml/scripts/check_threat_term_mechanism.py` (Check A).

**Check B — firing rate in test games (DECISIVE).**
40 games v58 vs v27, normal rules, alternating colours, dev
seeds 1060/2060 (fresh — distinct from the battery's 100/200,
the fast loop's 1001/2001, and Check B's own mining seeds).
Mine White's TSS rates with the diagnosis's
`replay_tss_battery.py --white-only` (per-game fires too, via
`--per-game --only-a-colour white`): White fire rate per turn,
firing-game rate (fraction of v58-White games with >= 1 fire),
fires on White turns in v58-White losses.
Baseline (M2, v52-White vs v27, normal rules): fire rate
0.1011 (118/1167 turns), firing-game rate 42% (21/50), fires
in losses 0 (0/488 turns).
Decision: PASS if the White fire rate is >= 0.15 (a 1.5x rise)
with z > 2 vs the M2 baseline (pooled two-proportion z-test),
OR >= 5 fires on White turns in v58-White losses (fires in
losses are unconfounded by win-more: v52-White fired 0 times
in its losses); FAIL if the fire rate is <= 0.12 AND 0 fires in
losses; WEAK otherwise.
**Proceed to the battery ONLY if Check B PASSES.** If Check B
FAILS or is WEAK, STOP — report the numbers for the next batch
and do not build a battery on an undemonstrated mechanism. (A
WEAK Check B is not rescued by Check A: move-choice change
without a reach-rate change means the forcing moves the term
plays do not lead to firing positions.)
Check B's leg also yields a 40-game v58-vs-v27 win rate as a
SECONDARY (not a strength verdict — the battery is vs v52).

### 11.4 Cost budget (pre-registered from the bench)

The bench (`ml/scripts/bench_threat_leaf.py`, run BEFORE this
pre-registration was committed — the iter-52 §1.4 precedent;
results in `research/iterations/threat-leaf-bench-v58.json`,
403 positions from the first 10 v52-vs-v27 battery games, plus
10 full 100-sim searches):

| quantity | v52 leaf | v58 leaf |
|---|---|---|
| mean per-call time | 11.97 ms | 14.49 ms |
| median per-call time | 6.40 ms | 7.42 ms |
| p90 per-call time | 56.12 ms | 57.86 ms |
| max per-call time | 69.64 ms | 94.70 ms |
| **mean cost ratio** | — | **1.211x** |

**Pre-registered ceiling: mean leaf-time ratio <= 1.5x**
(measured 1.211x — PASSES with headroom; v53's ceiling was
2.0x). The ratio is contention-independent in the sense that
both leaves are timed in the same process on the same
positions; the absolute times are inflated by CPU contention
with the parallel batch-14 workers. The term's added cost is
two v0 `attack_all` calls (one per colour) plus local exact
tests, reusing the gradient's single `attack_v1_both` call.
Battery affordability: v52's battery leg took ~50 min; the
leaf is ~21% slower but TSS dominates creator-rich move time,
so v58's legs are expected at ~55–60 min. If a leg exceeds 90
min the overrun is reported (no hard stop — the cost ceiling
is on the leaf, not the leg).

Bench characterisation (pre-registration measurements, NOT the
mechanism check — they motivate the test gates in §12, and
Check A/B are the independent arbiter): the v58 leaf differs
from the v52 leaf on 58.1% of sampled positions (234/403, mean
|diff| 0.0400 where differing — matching the design: a
one-forcing-move difference shifts the leaf by ~0.038); the
term is nonzero on 82.1% of quiet rollout ends (234/285); the
forcing-difference histogram at quiet rollout ends is
negative-skewed (the rollout's last mover — the opponent's
reply — holds more forcing moves; the term measures this
honestly, as the gradient does for breadth). Move choice on 8
TSS-silent midgame positions (100 sims, seed 100): agree 7,
differ 1 — the term changes move choice sometimes (12.5%),
where the choice is close.

### 11.5 What does NOT change

v52 exactly, except the leaf: 100 sims, c_puct 1.5, sharp v1
priors, the 2-ply rollout (same policy), the τ=0.05 tie-break,
most-visited root, TSS at the root (depth 4, budget 3000).
No q-based root decision (CLOSED). No sims change. No prior
change. No defensive VCF/TSS check (CLOSED). No White opening
book / White search / White prior change (CLOSED).
`baseline.py` stays FROZEN. No kernel re-sync unless v58 passes
the bar.

### 11.6 Eval seeds (pre-registered)

* Mechanism Check B: v58 vs v27, 40 games, seeds 1060/2060.
* Fast loop: vs v0 and vs v52, 10 games each, seeds 1001/2001
  (per the tasking).
* Battery h2h vs v52: v58=100 / v52=200; vs v0: v58=100 / v0=200.
* Ladder: seed 100, 40 games/opponent, mcts_sims=100
  (`ml/scripts/eval_version.py --version 58 --ladder-games 40
  --out research/iterations/ladder-v58.json`).
* Swap: 100 games vs v52, balanced placement, placement_seed 11,
  tempo judge, seeds 100/200.
* Tactical: `eval_tactical_full.py --version 58 --suite both
  --seed 100`. v52's tactical is re-verified first on the
  corrected suite (expect 20/20 + 29/29).
* Challenge match (ONLY if v58 passes all 4 clauses): per §1.6's
  pre-registered position, clause 1 IS vs v52 (the champion), so
  a bar-passer has already beaten the champion h2h; the title
  question is decided by the clause-1 match plus the ladder
  leg-result reading, and no separate challenge match is run.
* Harness commands follow iter-52 §1.5 exactly. All match
  drivers AUTO-RESUME from partial `--out` files — if a run is
  killed, relaunch the SAME command. Per-game seeding with a
  uniqueness audit on every leg.

### 11.7 The bar (4 clauses, ALL required; re-anchored on v52)

As §1.7: (1) h2h vs v52 ≥100 games, win rate > 0.50 with Wilson
95% CI lower bound > 0.50; (2) ladder Elo beats v52's — judged
by the LEG-RESULT reading: strictly dominate v52's legs (random
40-0, heuristic 40-0, mcts 31-9), i.e. match the two perfect
legs and beat 31-9 on the mcts leg; (3) tactical ≥ 20/20 +
29/29; (4) all test suites green. Fail any clause = REJECTED.
Do NOT claim a "highest Elo ever" from a single draw. Do NOT
re-run at another seed to shop for a pass — no seed-shopping,
in either direction (v44 precedent).

## 12. Implementation (re-scope)

`ml/gomoku_ml/mcts_heuristic.py` — the v58 block sits between
`tss_winning_move` and `class HeuristicMCTS`:
* `_THREAT_TERM_WEIGHT = 0.05`, `_THREAT_TERM_SCALE = 1.0`
  (pre-registered, §11.1).
* `_forcing_counts_both(board, black1, white1)` — exact
  forcing-move counts `(black, white)` from precomputed v1
  arrays (shared with the gradient) + one v0 `attack_all` per
  colour + local exact tests. Precondition: neither side holds
  a winning move (the caller establishes it).
* `_threat_potential_term(f_ours, f_theirs)` — the pure term
  function, `W * tanh((F(ours) - F(theirs)) / S)`.
* `_gradient_plus_threat(board, player)` — gradient + term from
  one `attack_v1_both` call, sum clamped to ±0.2.
* `_rollout_threat_leaf_value(board, player)` — the full leaf,
  written FRESH from the §11 spec (the v53 precedent: the
  differential test tests the spec against v52's leaf, not an
  alias against itself). Terminal handling identical to
  `_rollout_leaf_value`; the must-block at the rollout end
  returns the gradient alone (term skipped).
* `HeuristicMCTS` gains `leaf_mode="rollout2_threat_v58"`
  (validation, dispatch, `_rollout_threat_v58_leaf_value`,
  docstring entry).

`ml/gomoku_ml/scoring.py` — `ENGINE_VERSIONS[58]` = v52 +
`"leaf_mode": "rollout2_threat_v58"` (TSS root hook, tie-break,
sims unchanged).

Tests: `ml/tests/test_threat_term_v58.py`, 15 tests, all
passing — oracle equivalences (forcing counts vs
`_forcing_sets` and vs the brute-force oracles, 0 mismatches),
the v1-blind open-three case (v1 attack 6, v0 1030 — counted
by the term, invisible to the gradient's strong tier), the
broken-four creator case (v0-blind, counted via the v1
prefilter + exact test), term values/antisymmetry/bound, leaf
terminal identity (leaf-start + rollout terminals), the
quiet-path decomposition (v58 == clamp(v52 + term)), the
must-block skip, the ±0.2 clamp, colour-swap symmetry, the
difference gate (> 20%; bench 58.1%), the cost ceiling (1.5x;
bench 1.211x), registration, and TSS-hook precedence (v58
plays the TSS move on T1).
One test correction during development: the first antisymmetry
test asserted `leaf(B, BLACK) == -leaf(-B, WHITE)`, which is
the WRONG invariant for a rollout leaf (the rollout path
depends on who moves first) — the code was correct and the
test was fixed to the colour-swap symmetry
`leaf(B, BLACK) == leaf(-B, WHITE)`, which holds exactly.

Bench: `ml/scripts/bench_threat_leaf.py` →
`research/iterations/threat-leaf-bench-v58.json` (see §11.4).

## 13. Mechanism check (re-scope)

### 13.1 Check A — move-choice on mined miss positions: FAILS, 0/20

Mining (`ml/scripts/collect_forcing_miss_positions.py` →
`research/iterations/threat-miss-positions-v58.json`): in
v52-White's 27 normal-rules losses, White held forcing moves on
214/488 White turns (44%) and played a non-forcing move on 122
of them — the raw material is plentiful. The first 20 miss turns
were collected (games 1/5/9/11/21, plies 13–67); TSS was silent
on all 20 (0 firing candidates skipped, as logically expected —
had TSS fired, v52 would have played the TSS move).

Check (`ml/scripts/check_threat_term_mechanism.py`, seed 100 →
`research/iterations/threat-mechanism-check-a-v58.json`):
**v58 plays a forcing move on 0/20 (0.00); v52 control 0/20.
Verdict: FAIL** (pre-registered: PASS >= 6/20, FAIL <= 2/20).
v58 agrees with v52's move on 18/20; on the 2 it differs, both
moves are quiet (the term changes move choice sometimes, but
never toward a forcing move on these positions).

**Why — the mechanism fails at the SEARCH level, not the leaf
level (diagnostic, post-hoc).** Position [0] (game 1, ply 13)
worked up in detail: the forcing moves (9,5)/(10,5) are
four-creators — the 1-ply leaf after playing them is +0.25 for
White (Black must block), vs −0.007 for the recorded quiet move
(6,3), under BOTH leaves, and the term raises the forcing
moves' leaf further (+0.14 → +0.19 for (6,4)). But the root
search (100 sims, instrumented) shows: the forcing moves hold
the HIGHEST priors (0.089) and get 6–7 visits each — no prior
starvation — yet their backed-up our-q is NEGATIVE (−0.029 /
−0.039 for v52; −0.024 / −0.052 for v58), while the chosen
quiet move (6,3) has our-q +0.12 on 37 visits. The mean backup
dilutes the +0.25 expansion leaf with deeper leaves where
White is losing (the position is a White loss; after Black's
forced block White's attack is spent), and the term's leaf-level
boost (+0.04) is diluted to ~+0.005 in backed-up q — far below
the ~0.15 q gap to the chosen move. The leaf is right and the
search does not follow it: the v40 pattern (the mis-ranking
lives in backed-up root q / visit allocation), now confirmed
for a POSITIVE leaf term as well as for preventions. A leaf
term of weight 0.05 cannot flip root decisions where the q gap
is large; it can only tip close calls (the bench's 1/8
move-choice difference). Whether the search's q is RIGHT that
the forcing moves are worse is a separate question — Check B
is the in-game arbiter.

### 13.2 Check B — firing rate in test games (DECISIVE): PASSES

Leg: 40 games v58 vs v27, normal rules, alternating colours,
seeds 1060/2060 (`research/iterations/checkb-v58-vs-v27.json`,
70 min under 3-way contention with the H1/H3 workers; no
interruption). **v58 33W-6L-1D, win rate 0.8375, Wilson 95% CI
[0.6948, 0.9211]** (de-dup: 30/40 unique, 0.7833, CI [0.6087,
0.8937]) — a secondary, not a strength verdict (the battery is
vs v52). Per-colour: v58 as Black 16W-4L-0D (0.800, vs
v52-as-Black 0.78 vs v27 — flat); **v58 as White 17W-2L-1D
(0.875, vs v52-as-White 0.44 — nearly doubled)**. The
improvement is concentrated exactly where the diagnosis said
the mechanism lives: White.

Mining (`replay_tss_battery.py --white-only` →
`tss-replay-checkb-v58.json`; the run survived one VM
replacement at 34/40 games, resumed exactly from the
checkpoint; 0 consistency violations):

| engine-White | turns | fires | fire rate | differs | differ rate |
|---|---|---|---|---|---|
| v58 (A-white) | 625 | 103 | **0.1648** | 0 | 0.0000 |
| v27 (B-white) | 407 | 25 | 0.0614 | 8 | 0.0197 |
| v52 baseline (M2, A-white) | 1167 | 118 | 0.1011 | 0 | 0.0000 |
| v27 baseline (M2, B-white) | 928 | 75 | 0.0808 | 29 | 0.0312 |

**Decision quantities (§11.3): v58-White's fire rate is 0.1648
vs the 0.1011 baseline — ratio 1.63x, pooled z = +3.91.**
PASS (>= 0.15 with z > 2). The fires-in-losses alternative is
0 (see below) — the pass rests on the fire rate alone.

Per-game attribution (`--per-game --only-a-colour white` →
`tss-replay-checkb-v58-pergame.json`, 20 v58-White games):

| outcome | games | White TSS fires | fires/game | games with >= 1 fire | fire rate |
|---|---|---|---|---|---|
| WIN | 17 | 103 | 6.06 | 17 (100%) | 0.2187 (103/471) |
| LOSS | 2 | 0 | 0.00 | 0 (0%) | 0.0000 (0/42) |
| DRAW | 1 | 0 | 0.00 | 0 (0%) | 0.0000 (0/112) |

**Firing-game rate: 17/20 = 0.85 (v52-White baseline 21/50 =
0.42).** Fires per win 6.06 vs 5.62 — nearly unchanged. Fires
in losses 0 (v52: 0 in 27 losses — v58 has only 2 losses, so
this is weak evidence either way).

**Reading, with the win-more caveat stated plainly.** The
fire-rate rise is driven by MORE FIRING GAMES (0.85 vs 0.42),
not more fires per win (6.06 vs 5.62) — exactly the signature
of "reaches forcing positions in more games" (the diagnosis's
§10 hypothesis). The M2b win-more caveat (the fire rate rises
when White wins more, even with no new opportunities) applies
to comparing fire rates across DIFFERENT opponents or rules —
but here the comparison is v58 vs v52 against the SAME
opponent (v27), and the term is the ONLY difference between
the engines: the term causes both the firing-rate rise and
the win-rate rise (White 0.44 → 0.875). The term raises the
REACH rate, not conversion-in-losses (fires in losses stay 0:
when White does not reach a forcing position, it still loses,
as v52 does). The B-white control corroborates: v27-White's
fire rate FELL to 0.0614 (from 0.0808) against v58-Black —
v58 reaches its own forcing positions faster and ends games
before White can fire. (One caveat on the turn counts: the
single v58-White draw ran 112 White turns — a near-full-board
game — inflating White turns/game to 31.2 vs v52's 23.3; the
fire rate is per-turn, so the long draw dilutes it, making
0.1648 conservative.)

**Mechanism verdict: Check B PASSES. Proceed to the fast loop
+ battery.** Check A (0/20) and Check B (PASS) together say:
the term does not flip move choice on the mined loss
positions (where the search's q gap is large — §13.1), but
over full games it steers White toward forcing positions
often enough to double White's firing-game rate and win rate
vs v27. The fast loop (vs v52 — the bar reference, which also
has TSS) is the next arbiter: both sides convert forcing
wins there, so the edge should be smaller.

## 14. Fast loop (re-scope)

Iteration 1, leg 1 (vs v0, 10 games, seeds 1001/2001):
**v58 10W-0L-0D** (win rate 1.0000, Wilson 95% CI [0.7225,
1.0000]) — `research/iterations/fastloop-v58-vs-v0-i1.json`
(231s). Clean sweep, no losses to analyze. (v52 beats v0
1.0000 over 100 games, so this is the expected smoke-test
outcome, not evidence of strength.)

Iteration 1, leg 2 (vs v52, 10 games, seeds 1001/2001):
**v58 7W-2L-1D** (win rate 0.7500, Wilson 95% CI [0.4422,
0.9191]) — `research/iterations/fastloop-v58-vs-v52-i1.json`
(1983s; v58-vs-v52 games are the slowest in the program —
both engines run TSS at every root — and game 3 ran to a
225-ply full-board draw). 10/10 unique games, de-dup win rate
0.75. The CI includes 0.50 — 10 games prove nothing; the
battery is the verdict. Per-colour: v58 as Black 3W-2L-0D,
as White 4W-0L-1D.

Loss workup (`classify_batch_losses.py` + TSS replay of the
loss games): all 2 losses via open four (2/2), **0 missed
blocks**, 0 clean missed preventions in 13 threatened turns.
Defence is solved. TSS fired on **0 v58 turns in both losses**
(game 2: 0/29 turns; game 6: 0/25), while v52 fired 4x (first
ply 51) and 8x (first ply 31) respectively and converted — the
same pattern as v52's own fast loop (TSS fired 0x in all 3
losses): the losses are positional (v52-as-White reaching
forcing positions v58-as-Black did not), with no v58-specific
losing move to patch. Consistent with Check B's flat Black
result (0.80 vs 0.78): the term's Black-side reach effect is
small.

**Fast-loop verdict: NO PATCH.** The mechanism behaves as
designed (no false positives, no v58-specific blunder class).
Proceed to the battery.

## 15. Battery (re-scope)

### Tactical (run before the battery legs — clause-3 verdict first)

v52 re-verified first on the corrected suite:
**old 20/20, hard 29/29**
(`research/iterations/tactical-v52-reverify.txt`).
v58 at `--seed 100`: **old 20/20, hard 29/29**
(`research/iterations/tactical-v58.txt`). **Clause 3 PASSES.**
The term does not break any tactical position (terminals are
exact, as in v52; the tie-break interaction v40 warned about
does not materialise here).

### vs v52, 100 games, normal rules

**v58 82W-15L-3D** (win rate 0.8350, Wilson 95% CI [0.7501,
0.8951]) — `research/iterations/batch-v58-vs-v52.json`
(recompute-verified; leg survived 3 runtime restarts,
auto-resume intact). **Clause 1 PASSES decisively** — the CI
lower bound (0.7501) clears 0.50 by 0.25. Uniqueness 67/100
games; de-dup 49W-15L-3D, rate 0.7537, CI [0.6387, 0.8412] —
still a clear pass on unique games only. Per-colour: v58 as
Black 40W-9L-1D (0.81), as White 42W-6L-2D (0.86) — the term
helps both colours, White slightly more (consistent with the
Check B mechanism: White's forcing-position reach rate rises).
Defence audit: all 15 losses via open four, 0 missed blocks,
0 clean missed preventions in 82 threatened turns — defence
is solved.

### vs v0, 100 games, normal rules

### swap vs v52, 100 games

### Ladder (40 games/opponent)

## 16. Verdict (re-scope)
