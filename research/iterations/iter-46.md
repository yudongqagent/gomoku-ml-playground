# Iteration 46 — v46: White-side statistics, then patch what the statistics point to (batch 12, H2)

**Status: PRE-REGISTERED — §1 was written and committed BEFORE any
statistics collection or eval of v46. Statistics (§2), patch or
fallback decision (§3), fast loop (§4), and battery (§5) follow.**

**Method (carried over).** The fast 10-game loop stays the finding
method: play 10 games, read every move, find each game's losing move
(solver/deeper-search verified, never guessed — for rollout-leaf
engines, playouts or exact computation, NOT the v1 walk-back oracle
alone — iter-18 §2.1), patch, regression-test, repeat. The 4-clause
bar stays the validation gate. Fast-loop games use development seeds
distinct from the bar battery's pre-registered seeds (100/200) and
from the statistics corpus seeds (§1.1). **Fast loops run against
v0 as well as v27** (the bar reference is v27 now, not v7). The full
battery runs on the FINAL version only. The 100-game battery is the
verdict — small-sample smokes mislead; single-position diagnostics
cannot validate a global search change; the v1 walk-back oracle
shares the argmax-of-sum flaw (iter-18) — every nominated save needs
playouts or exact computation.

## 0. The problem

Batch 11 H3 (iter-41) read v27's White losses loss-by-loss and found
**no patchable mechanism**: the losses are distributed drift +
no-signature horizon losses; the three class-7 instances share no
positive mechanism (leaf profiles heterogeneous, setup-occupation
anti-correlated at all three); and in fresh self-play White is NOT
weak (7W-2L-1D) — the "White gap" is matchup-dependent, not a
systematic v27-White defect. The White gap as a single-flaw framing
is CLOSED. This workstream reframes the question statistically:
instead of reading individual losses, collect White-side play
statistics across many games and ask whether ANY colour asymmetry
exists at the population level — and if it does, whether it has a
shared signature a patch can target. If the statistics show no
asymmetry worth patching, the workstream falls back to the
blunder-dilution direction with a mechanism distinct from H3's
(§1.2).

**What is already known (starting point, not re-derived):**
- v27's White losses vs v7 (iter-35 §2, iter-41 §2): 6 horizon, 2
  distinct VCF-prevention positions (CLOSED lever), 2 search-horizon
  no-signature. No class has ≥3 verified instances sharing one
  mechanism.
- v27 itself is White-weak against stronger opposition: vs v7 its
  19 losses split 8 Black / 11 White; its fast loop lost 3 of 4
  games as White (iter-27 §2).
- In v27-vs-v27 self-play White is NOT weak (iter-41 corpus B:
  7W-2L-1D for White across 10 games; both engines win more as
  White). Any asymmetry the statistics find must therefore be
  checked against BOTH matchups (self-play and vs v0) — a
  self-play-only asymmetry and a vs-v0-only asymmetry point at
  different mechanisms, and a "White gap" that appears in neither
  is not a systematic defect.
- The leaf is not the binding constraint in general (v11, v14),
  EXCEPT the v1-kernel leaf's behaviour against greedy play IS
  the binding constraint for v1-leaf versions (v17, v20), and the
  leaf's blind spots were batch 11's target (now closed: the
  leaf-blind direction, v40 — the remaining mis-rankings are
  averaging/visit-allocation, i.e. search problems).

**Base engine.** v27 exactly as on master (100 sims, c_puct 1.5,
sharp v1 priors, rollout leaf both colours, τ=0.05 threshold
tie-break, most-visited root). A statistics patch builds on v27,
not on v18 — v27 is the bar reference, and the patch must beat IT.

## 1. Design and pre-registration

**Comparison target: v27** (the bar reference, batch-12 plan:
ladder Elo 636.4). Clause 1 is head-to-head vs v27 (normal rules,
100 games, alternating colours, win rate > 0.50 with Wilson 95%
CI lower bound > 0.50); clause 2 is ladder Elo strictly higher
than v27's 636.4 (**40 games/opponent** — the re-based ladder,
batch-11 plan; leg-result reading for saturation: a 40-0 leg
strictly dominates a 39-0-1 leg); clause 3 is tactical pass rate
>= v27's (20/20 old suite + 29/29 hard suite, on the CORRECTED
suite — re-verify v27's tactical first); clause 4 is all test
suites green (ml pytest, npm test, tsc). **Secondary
head-to-heads, also pre-registered:** v46 vs v0 (100 games) and a
100-game swap match vs v27 under the tempo judge (the harness
default). **Strength champion:** v18 (ladder Elo 636.4).
Champion rule v2 (TRACKS.md, direct challenge): IF v46 passes all
4 clauses, it plays a 100-game challenge match vs v18 (seeds
100/200, normal rules) and takes the title only if it wins that
match (win rate > 0.50 with Wilson 95% CI lower bound > 0.50)
AND its ladder Elo beats 636.4 (automatic for a bar-passer). The
coordinator makes the champion call; this log reports the
challenge match if it runs.

### 1.1 Phase 1 — White-side statistics (pre-registered protocol)

**Corpus (two legs, both pre-registered before collection):**
- **(S1) v27 vs v27 self-play, normal rules, 60 games**, seeds
  1046/2046. Alternating colours → each engine plays 30 as Black
  and 30 as White; pooled, Black and White each play 60 games.
  Self-play is the cleanest asymmetry test: the engines differ
  only by RNG stream, so any colour gap is a property of the
  colour, not of the opponent.
- **(S2) v27 vs v0, normal rules, 20 games**, seeds 1047/2047.
  v27 as Black 10, as White 10. Tests whether any asymmetry
  reproduces against a weak, greedy opponent (iter-41's corpus B
  found none in 10 games: 10W-0L-0D, no White losses).
- Total: 80 games, ~3,200 turns, ~1,600 per colour. Games are
  recorded with the standard driver (`selfplay_batch.py`, per-game
  seeding, uniqueness audit) as
  `research/iterations/stats-v27-selfplay-60.json` and
  `research/iterations/stats-v27-vs-v0-20.json`.

**Statistics (all split by colour × ply band; bands: opening
plies 1–12, midgame 13–32, endgame 33+; ply = move number,
1-indexed).** Computed by the committed script
`ml/scripts/white_stats.py` from the game records — no
hand-counting. Every statistic is an exact computation from the
game records (VCF solver / creator enumeration / kernel scores),
never a guessed losing move:

1. **Win rate by colour** (per leg and pooled): Black wins / White
   wins / draws. The headline asymmetry check.
2. **Threatened-turn rate:** fraction of turns where the opponent
   holds ≥1 open-four creator and no immediate winning move
   (the classifier's definition — prevention was relevant).
3. **Clean missed prevention rate (creator-based):** among
   threatened turns, the fraction where the played move was not
   preventing (not forcing and left a creator live) while a
   preventing alternative existed (`classify_batch_losses.py`'s
   exact definition). The primary ERROR statistic.
4. **Forced-block share:** fraction of turns where the opponent
   holds exactly 1 winning move (the reply is forced). A
   reactive-play measure: a colour that spends more turns
   forced-blocking is playing more reactively.
5. **Threat-creation rate:** among QUIET turns (opponent holds no
   winning move and no creator before our move), the fraction of
   moves after which the mover holds ≥1 open-four creator, and
   separately ≥1 VCF start. The attack-generation measure.
6. **First-threat ply:** mean ply at which each colour first holds
   a VCF start / first holds a creator, per game (games where a
   colour never holds one are excluded from that colour's mean
   and counted separately).
7. **VCF-start exposure:** mean opponent VCF starts remaining
   after our move, on turns where the opponent held ≥1 VCF start
   before our move (did we clear the live threat?).
8. **Clean missed VCF prevention rate (VCF oracle):** among
   VCF-threatened turns (opponent holds ≥1 VCF start before our
   move, no immediate winning move), the fraction where the
   played move left ≥1 of the ORIGINAL VCF starts live while a
   preventing alternative existed (a candidate move after which
   the opponent holds 0 of the original VCF starts). The
   secondary ERROR statistic. Caveat, recorded here in advance:
   the prevention test covers the original VCF starts only (the
   v35-style check — v35's hole: our move can CREATE a new VCF
   start by removing one of our own defensive replies, and this
   test does not catch that; the error is defined against the
   live threat, so the hole affects the alternative-enumeration
   only, and only in the direction of counting a prevention that
   creates a new threat — the rate is therefore an upper bound
   on true VCF-prevention failures, and the same bound applies
   to both colours, so the colour COMPARISON is unbiased).
9. **Decisive-phase distribution:** for each loss, the ply of the
   first losing move by the VCF-solver method
   (`find_losing_moves.py`: first loser move after which the
   opponent forces a win at depth 4 AND some near reply avoids
   it; "already lost" if no reply avoids it), by colour of the
   loser and ply band. Says WHERE games are lost, not just how
   often.
10. **v1-score regret (secondary, diagnostic only):** mean gap
    between the best candidate's v1 total score and the played
    move's v1 total score, by colour × band. NOT an error
    measure (the search is allowed to disagree with v1) — a
    diagnostic for whether one colour's search deviates further
    from its prior ranking.

**Patch rule.** A patchable asymmetry requires ALL THREE of:
- **(a) A colour gap on an error statistic:** on statistic 3 or
  8, in the midgame band (13–32) or the endgame band (33+),
  White's rate exceeds Black's by **> 5.0 percentage points**
  AND the two-proportion z-test 95% CI for the difference
  excludes 0. (The opening band is excluded from (a): too few
  threatened turns for a stable rate. Statistic 1 alone is NOT
  enough — a win-rate gap without an error-rate gap is
  first-move advantage / drift, not a patchable mechanism.
  Statistics 4–7, 9, 10 are supporting evidence only: they can
  nominate WHERE to look, but the patch decision rests on 3/8.)
- **(b) A shared signature:** the excess White errors in the
  qualifying band share ONE positive signature Y, established by
  inspecting all White error instances in that band: Y must hold
  for ≥ 60% of the White error instances and < 25% of matched
  Black controls (same band, threatened turns where Black
  prevented successfully). Y must be a checkable property of
  the error positions (e.g. "the preventing move is a quiet move
  ranked below 8th by v1 prior", "the threat is a setup the
  rollout policy never plays"). The signature is stated in §2
  BEFORE the patch is designed in §3.
- **(c) A genuinely new patch:** a patch for Y exists that is not
  on the CLOSED list (q-based root decisions general;
  root-decision for strength; the tie-break lever; White
  search/prior/book; exploration tuning; gap-three demotion in
  any combination; the blend; combination-stacking; more sims;
  VCF-prevention for the v27 family in ANY form — prior-pick
  v29, search-pick v33, search-first v35: all priced; the
  leaf-blind direction, v40; setup-prevention in this form,
  v39; the White gap as a single-flaw framing, v41), is not a
  fitted constant, and reproduces the preventing move at the
  error positions (regression test).

**Decision rule.** If (a) fails on every statistic × band →
"no asymmetry worth patching" → the fallback (§1.2) triggers.
If (a) holds but (b) or (c) fails → "asymmetry without a
patchable mechanism" → the fallback (§1.2) triggers. If
(a)+(b)+(c) all hold → v46 = v27 + the statistics patch, and
the fallback does NOT trigger. The decision is recorded in §3
with the numbers that drove it. A negative statistics result is
a program finding, not a failed iteration — the numbers are
committed in full either way.

### 1.2 Phase 2 — fallback: leaf-level top-K min aggregation (pre-registered)

Triggers ONLY if Phase 1 finds no patchable asymmetry under the
rules of §1.1. It does NOT trigger merely because the statistics
are slow or the patch is hard — the §1.1 decision rule is the
trigger.

**Base: v27** exactly as on master. The fallback changes ONLY the
rollout leaf's opponent-reply step, not the search, the priors,
the root rule, the tie-break, or the sims.

**The mechanism (blunder dilution, iter-40 §1.1a).** In g61/g43
the unsafe move creates our own open-four race threats, so
opponent blunder replies return +1 leaves that dilute the
creator-win average — every leaf value is correct, but the
average lies. The mis-ranking is MCTS averaging over the
opponent's moves instead of taking the opponent's best move.
H3 (v47) is taking the BACKUP-rule change (how values aggregate
up the tree). This fallback is the LEAF-level variant: make the
leaf itself pessimistic about the opponent's reply, so a move
whose only defence is "the opponent might blunder" is valued at
its true worst case before any averaging happens.

**The patch, exactly (pre-registered).** New leaf mode
`rollout2_v1_both_mink4`, identical to v27's `rollout2_v1_both`
except in `_rollout_leaf_value`'s quiet case:
- The leaf-start terminal check is unchanged (exact ±1 / ±0.25).
- Rollout ply 0 (our move) is unchanged: `_rollout_move`'s
  win/block/open-four-priority/argmax policy.
- Rollout ply 1 (the opponent's reply): instead of the single
  argmax reply, evaluate the opponent's **top-K = 4** candidate
  replies by v1 total score (the SAME score `_rollout_move`
  argmaxes over: own v1 attack + opponent's v1 attack at the
  square, from the opponent's perspective; ties by lowest flat
  index, as in the argmax) and return the **MIN** (worst case
  for us) of the resulting values.
- Each reply's value uses the rollout-end rule, unchanged from
  v27: our winning move after the reply → +1; opponent holds
  ≥2 winning moves after the reply → −1; otherwise the symmetric
  v1 gradient at the resulting position (a must-block at the
  rollout end is gradient-evaluated, NOT −0.25 — iter-16 §2.1).
- Forced replies are unchanged: if the opponent has a winning
  move at ply 1 → −1; if we hold ≥2 winning moves → +1; if we
  hold exactly 1 winning move → the forced block is the single
  reply (the min over one reply is that reply's value). The min
  applies ONLY in the quiet case where the opponent has a free
  choice.
- **Pure min, NOT a min-mean blend.** The mechanism is "the
  average lies; the worst case is the truth" — a blend would
  reintroduce the dilution the patch exists to remove. If pure
  min is too pessimistic, the battery (vs v0 AND vs v27) will
  show it — that is a measurable outcome, not a reason to
  pre-blend.
- K = 4: the creator in the diagnosed instances ranks at the top
  of the v1 ranking (g61: creator prior ~0.099 vs blunders
  ~0.100), so K=4 covers the creator plus the blunder replies
  that dilute it, at ~2x leaf cost (4 gradient evaluations
  instead of 1). No K sweep: K is fixed by the mechanism (the
  creator + its diluting blunders), not fitted.

**Cost guard.** Mean move time measured on the 100-position
bench set (`ml/scripts/bench_vcf_root.py`'s bench positions are
VCF-specific; use a fresh 20-position quiet-position bench from
v27 self-play games) before the battery and reported in §3; the
battery decides whether the cost is worth it. Expected ~2x leaf
cost → ~1.5–2x move time.

**What the fallback is and is not.** It is a LEAF change
addressing a diagnosed averaging artefact at the point where the
opponent's reply is modelled. It is NOT H3's backup-rule change
(different mechanism, different code path — the backup rule is
untouched here), NOT a VCF check (no `starts_forced_win` anywhere
in the patch), NOT a root-decision change (the tie-break and
most-visited rule are untouched), and NOT a prior/sims change.
Caution (batch-12 plan): v0 blunders constantly, so a
min-aggregation may hurt vs v0 — the battery (vs v0 AND vs v27)
will tell. If the fallback triggers, v46 = v27 + this leaf
change, registered as version 46 with
`leaf_mode="rollout2_v1_both_mink4"`.

### 1.3 Eval seeds (pre-registered)

- Battery h2h vs v27: v46=100 / v27=200; vs v0: v46=100 /
  v0=200.
- Ladder: seed 100, **40 games/opponent** (re-based, batch-11
  plan), mcts_sims=100
  (`ml/scripts/eval_version.py --version 46 --ladder-games 40
  --out research/iterations/ladder-v46.json` — the auto-resume
  driver).
- Swap: 100 games vs v27, balanced placement, placement_seed
  11, tempo judge (the harness default), seeds 100/200.
- Tactical: `eval_tactical_full.py --version 46 --suite both
  --seed 100`. v27's tactical is re-verified FIRST on the same
  command with `--version 27`.
- Challenge match (ONLY if v46 passes all 4 clauses): 100 games
  vs v18, seeds 100/200, normal rules.
- Fast-loop dev seeds (patched engine): 1146/2146 (vs v27) and
  1147/2147 (vs v0) — fresh seeds distinct from the battery's
  100/200, the statistics legs' 1046/2046 and 1047/2047, and
  batch-9/10/11 loop seeds.
- Harness commands follow iter-27 §1.4 / iter-41 §1.3 exactly
  (`selfplay_batch.py`, `eval_version.py` for the ladder,
  `eval_tactical_full.py` for tactical). All match drivers
  AUTO-RESUME from partial `--out` files — if a run is killed,
  relaunch the SAME command. The drivers seed per game (`seed +
  game_index`) and audit uniqueness — every leg reports its
  distinct-game count and de-dup rate.

## 2. Phase 1 results — the statistics (80 games, 3,783 turns)

**Corpus as pre-registered.** (S1) 60-game v27 self-play, seeds
1046/2046, 58 unique games
(`research/iterations/stats-v27-selfplay-60.json`, 35W-23L-2D
from engine A's perspective); (S2) 20-game v27 vs v0, seeds
1047/2047, 16 unique games
(`research/iterations/stats-v27-vs-v0-20.json`, 20W-0L-0D).
Analysis by the committed `ml/scripts/white_stats.py`; merged
output `research/iterations/white-stats-v46.json` (legs:
sp-0-9, sp-10-39, sp-40-59, vs-v0-20; per-leg outputs
`white-stats-v46-sp1039.json`, `white-stats-v46-sp4059.json`,
`white-stats-v46-v0.json`). Total analysis elapsed 12.0 h
across legs (VCF enumeration + losing-move analysis per game;
the batch-12 H1 workstream shared the VM throughout).
**Pickup note:** the original H2 worker was killed by a restart
drain mid-analysis; the pickup resumed from the per-game
checkpoints (sp-10-39: 16→30, sp-40-59: 13→20, vs-v0: 8→20 —
no completed game was re-analysed). The sp-0-9 leg's checkpoint
covers games 0-9 of the 60-game file, so it cannot go through
`--merge`'s completeness check (which requires next_game ==
the input file's game count); it was merged directly from its
checkpoint state, verified to cover exactly games 0-9
(first-threat game ids 0–9, all error/decisive game ids ⊆
0–9), and the merge asserted self-play coverage 0–59 exactly
once. The error records from that leg still carry the input
file's basename (`stats-v27-selfplay-60.json`) — cosmetic
only; the game numbers identify them.

### 2.1 Statistic 1 — win rate by colour

| Leg | Black wins | White wins | Draws | Black rate [95% CI] | White rate [95% CI] |
|---|---|---|---|---|---|
| Self-play (60) | 35 | 23 | 2 | 0.5833 [0.4573, 0.6994] | 0.3833 [0.2709, 0.5098] |
| vs v0 (20) | 10 | 10 | 0 | 0.5000 [0.2993, 0.7007] | 0.5000 [0.2993, 0.7007] |
| Pooled (80) | 45 | 33 | 2 | 0.5625 | 0.4125 |

In self-play the engines are identical (they differ only by RNG
stream), so Black's 35–23 edge is first-move advantage plus
noise (engine A won 20 as Black and 15 as White). Against v0,
v27 went 10–0 as Black AND 10–0 as White — White is not weak
against the greedy opponent. Per §1.1, statistic 1 alone is
NOT enough: a win-rate gap without an error-rate gap is
first-move advantage / drift, not a patchable mechanism.

### 2.2 The patch-rule statistics (3 and 8) — the read-off

Pooled gaps (White minus Black), the pre-registered read-off
from `white_stats.summarize`:

| Band | Statistic | Black | White | Diff | 95% CI | (a)? |
|---|---|---|---|---|---|---|
| midgame | 3 creator | 0.0164 (3/183) | 0.0164 (3/183) | 0.0000 | [-0.0260, 0.0260] | no |
| midgame | 8 VCF | 0.0410 (8/195) | 0.0455 (9/198) | +0.0044 | [-0.0358, 0.0446] | no |
| endgame | 3 creator | 0.0000 (0/127) | 0.0229 (3/131) | +0.0229 | [-0.0027, 0.0485] | no |
| endgame | 8 VCF | 0.0355 (5/141) | 0.0872 (13/149) | +0.0518 | [-0.0028, 0.1064] | no |
| opening | 3 creator | 0.0000 (0/27) | 0.0000 (0/84) | 0.0000 | [0.0, 0.0] | excluded |
| opening | 8 VCF | 0.0000 (0/27) | 0.0000 (0/84) | 0.0000 | [0.0, 0.0] | excluded |

**(a) FAILS on every statistic × band.** The closest approach
is endgame VCF: the point estimate (+5.18 pp) clears the 5 pp
magnitude bar, but the 95% CI [-0.28 pp, +10.64 pp] includes
0 — it misses by 0.28 pp on the lower bound. (a) requires
BOTH conditions, so it fails. Midgame is a dead tie on
creators (3 errors each) and +0.44 pp on VCF.

**The signal does not reproduce across slices/matchups**
(per-leg gaps, White minus Black):

| Leg | Midgame VCF | Endgame VCF |
|---|---|---|
| sp-0-9 | +0.0400 (1/25 vs 0/31), CI incl. 0 | +0.0500 (1/20 vs 0/10), CI incl. 0 |
| sp-10-39 | −0.0211 (2/82 vs 3/66), CI incl. 0 | +0.0868 (6/47 vs 2/49), CI incl. 0 |
| sp-40-59 | +0.0678 (4/59 vs 0/41), CI excl. 0 | −0.0592 (1/48 vs 2/25), CI incl. 0 |
| vs-v0-20 | −0.0252 (2/32 vs 5/57), CI incl. 0 | +0.1295 (5/34 vs 1/57), CI excl. 0 |
| self-play pooled | +0.0204, CI [-0.0186, 0.0595] | +0.0219, CI [-0.0431, 0.0870] |

Two single-slice gaps exclude 0 (sp-40-59 midgame VCF,
vs-v0 endgame VCF) — but they point in opposite directions
across slices (sp-10-39 midgame VCF is −0.0211, sp-40-59
endgame VCF is −0.0592), and the self-play-only pooled gaps
are +0.0204 / +0.0219 with CIs including 0. That is what
slice noise looks like, not a systematic defect. The
pre-registered rule reads the pooled gap, and the pooled gap
fails (a).

### 2.3 Supporting statistics (2, 4–7, 9, 10)

Pooled rates, per colour × band (3,783 turns: Black 1,915,
White 1,868):

| Statistic | Band | Black | White |
|---|---|---|---|
| 2 threatened-turn rate | opening | 0.0563 | 0.1750 |
| 2 threatened-turn rate | midgame | 0.2460 | 0.2521 |
| 2 threatened-turn rate | endgame | 0.1838 | 0.1979 |
| 4 forced-block share | opening | 0.0000 | 0.0250 |
| 4 forced-block share | midgame | 0.1075 | 0.1391 |
| 4 forced-block share | endgame | 0.1302 | 0.1949 |
| 5 quiet creator creation | opening | 0.1854 | 0.1016 |
| 5 quiet creator creation | midgame | 0.4536 | 0.4123 |
| 5 quiet creator creation | endgame | 0.4576 | 0.4111 |
| 5 quiet VCF creation | opening | 0.2119 | 0.1068 |
| 5 quiet VCF creation | midgame | 0.5633 | 0.5190 |
| 5 quiet VCF creation | endgame | 0.5379 | 0.4881 |
| 7 VCF-start exposure | midgame | 0.2718 | 0.5707 |
| 7 VCF-start exposure | endgame | 1.1986 | 0.7248 |
| 10 v1 regret (mean) | opening | 613.8 | 178.8 |
| 10 v1 regret (mean) | midgame | 9520.5 | 8654.2 |
| 10 v1 regret (mean) | endgame | 11872.7 | 11716.5 |

- **6 First threat:** Black first holds a VCF start at mean
  ply 10.7 (n=79, never 1) and a creator at 10.9; White at
  15.6 (n=80, never 0) and 15.8. Black threatens ~5 plies
  earlier — first-move advantage, as expected.
- **9 Decisive phase:** White losses 45 (24 midgame, 21
  endgame), Black losses 33 (11 midgame, 22 endgame);
  already-lost at first-loser detection in 28/45 White
  losses and 25/33 Black losses — most losses are
  already-lost drift, consistent with iter-41's loss-by-loss
  finding (distributed drift + no-signature horizon losses).
- **Error instances:** 44 total (creator: Black 3 / White 3
  midgame, White 3 endgame; VCF: Black 8 / White 9 midgame,
  Black 5 / White 13 endgame), against 680 controls
  (threatened turns where the mover prevented). The 13 White
  endgame VCF errors' best-preventing v1 ranks are 1–6
  (mostly 1–3) and their played v1 ranks mostly 1–2: the
  preventing move is usually top-ranked by v1 and the engine
  plays its top move and still misses — there is no
  "preventing move ranked below 8th" signature to patch.
  (Moot: (a) already fails, so (b) is never reached — but
  the instance inspection independently shows no shared
  low-rank signature hiding in the closest band.)

**Reading.** Every positive finding is first-move-advantage
shaped: Black threatens earlier (~5 plies) and creates more
often; White faces more threats in the opening and blocks
more often (forced-block share higher in every band). But
there is NO error-rate gap behind any of it — White prevents
as well as Black does (midgame creator errors tied 3–3;
midgame VCF +0.44 pp). White's worse record is the position
it defends, not the defence it plays.

### 2.4 Verdict — "no asymmetry worth patching"

(a) fails on every statistic × band (the closest, endgame
VCF, clears the magnitude bar but its CI includes 0 by
0.28 pp, and the signal does not reproduce across
slices/matchups). Per the §1.1 decision rule this is **"no
asymmetry worth patching" → the fallback (§1.2) triggers**.
This is the population-level confirmation of iter-41's
loss-by-loss finding: the White gap is not a systematic
v27-White defect, and there is no statistics-nominated patch
to build. A negative statistics result is a program finding,
not a failed iteration — the numbers above are committed in
full (`research/iterations/white-stats-v46.json`).

## 3. Phase 2 — the fallback triggers: v46 = v27 + leaf-level top-K min aggregation

**Decision.** The §1.1 decision rule fires the fallback:
(a) failed on every statistic × band (§2.2), so v46 is v27
with the pre-registered leaf change of §1.2
(`leaf_mode="rollout2_v1_both_mink4"`), registered as version
46. No statistics patch was designed — there was nothing to
patch.

**Implementation** (`ml/gomoku_ml/mcts_heuristic.py`,
`_rollout_leaf_value_mink4` + `_rollout_end_value`, +145
lines; registry entry in `ml/gomoku_ml/scoring.py`, version
46 = v27's entry with only `leaf_mode` changed). The change
is confined to the rollout leaf's quiet-case opponent-reply
step: the MIN over the opponent's top-K = 4 replies by v1
total score (pure min, no blend; K = 4 fixed by the
mechanism). Terminal positions, the ply-0 rollout move, and
all forced replies are v27-identical (verified by test, not
just by reading); the backup rule, priors, root rule, and
the τ=0.05 tie-break are untouched. The documented corner
case stands: if the opponent holds an open-four creation AND
four or more higher-total replies, the creator falls outside
the top-K and the min misses the −1 — that needs four 200k+
replies outranking a 200k+ creator, and the battery decides
whether it matters in practice. This is the LEAF-level
blunder-dilution variant, distinct from v47's backup-rule
change.

**Tests.** 7 new v46 tests in `ml/tests/test_mcts_heuristic.py`
(registration + v27-untouched control, unknown-leaf-mode
rejection, terminal equality vs v27, independent min
recomputation on two corpus positions for both colours,
mink4 ≤ v27 leaf on quiet midgame positions, no board
mutation, determinism). Full ml pytest green: **458 passed,
7 skipped, 58 xfailed**; npm test **55/55**; `tsc --noEmit`
clean. (Pickup note: the implementation + tests were written
by the original H2 worker and left uncommitted; the pickup
verified the diff against §1.2 line by line, ran the tests,
and committed them — no failures were left behind.)

**Cost guard** (§1.2). Mean move time on a fresh 20-position
quiet-position bench from v27 self-play games (games 0–17,
19, 20 at ply 15–16 — the earliest quiet ply in 15–40;
game 18 skipped, no quiet position in plies 15–40; engines
alternated per position; one warm-up position discarded;
artefact `research/iterations/bench-v46-cost.json`):
**v27 mean 1.80 s** (median 1.83, range 1.18–2.32),
**v46 mean 2.83 s** (median 2.99, range 1.53–3.77),
**ratio 1.57×** — inside the expected ~1.5–2× (4 gradient
evaluations instead of 1 at the leaf). v46 is slower at
every one of the 20 positions. Absolute times are inflated
by the H1 workstream sharing the VM during the bench; the
alternating-engines ratio is the robust number. The battery
decides whether the cost is worth it.

## 4. Fast loop (10 games vs v27, 10 games vs v0)

`ml/scripts/fast_loop_match.py --version-a 46 --opponent v27
--games 10 --seed-a 1146 --seed-b 2146`
(`research/iterations/fl-v46-vs-v27.json`) and the same vs v0
with seeds 1147/2147 (`fl-v46-vs-v0.json`):

- **vs v27: 6W-4L-0D, win rate 0.6000, 95% CI
  [0.3127, 0.8318]**, 828.2 s, 10/10 unique. As Black 4W-1L,
  as White 2W-3L. (Game 8 ran 161 plies, a v46 win as Black.)
- **vs v0: 9W-1L-0D, win rate 0.9000, 95% CI
  [0.5958, 0.9821]**, 588.3 s, 10/10 unique. The §1.2 caution
  (min-aggregation hurting vs v0's constant blunders) does not
  show up in 10 games; the 100-game battery leg decides.

**Loss analysis** (`ml/scripts/find_losing_moves.py`, depth-4
VCF solver, solver-verified losing moves, never guessed):

| Loss | Losing ply (1-indexed) | Played | Saving moves | Already lost? |
|---|---|---|---|---|
| vs v27 g0 (Black, 44 plies) | 41 | (3,4) | — | yes |
| vs v27 g3 (White, 41 plies) | 38 | (8,4) | — | yes |
| vs v27 g5 (White, 35 plies) | 30 | (7,5) | (3,9) | no |
| vs v27 g7 (White, 39 plies) | 34 | (9,4) | (6,4) | no |
| vs v0 g4 (Black, 28 plies) | 23 | (10,4) | (11,3), (6,5), (7,5) | no |

Two losses are already-lost drift (no reply avoids the loss).
For the three with saving moves, the control was run: at each
position, with the game's seed, **v27 chooses the identical
losing move** — v46 reproduces its game move exactly at all
three positions (determinism check) and v27 makes the same
mistake at all three. These are shared search-horizon misses,
not v46 defects: the min-aggregation is not implicated in any
fast-loop loss. **No defect found; no patch; proceed to the
battery.**
