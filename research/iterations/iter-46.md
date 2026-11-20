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
