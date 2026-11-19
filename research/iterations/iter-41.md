# Iteration 41 — v41: White-side forensics → targeted patch (batch 11, H3)

**Status: FORENSICS IN PROGRESS — the mechanism (§3) is pre-registered
BEFORE any eval of v41. Forensics (§2), patch or fallback (§3), fast
loop (§4), and battery (§5) follow.**

**METHOD (carried over).** The fast 10-game loop stays the finding
method: play 10 games, read every move, find each game's losing move
(solver/deeper-search verified, never guessed — for rollout-leaf
engines, playouts or exact computation, NOT the v1 walk-back oracle
alone — iter-18 §2.1), patch the algorithm to reproduce the perfect
move in those positions (regression test), repeat — max 4
iterations, or stop when an iteration finds no new patchable
weakness class. The 4-clause bar stays the validation gate.
Fast-loop games use development seeds distinct from the bar
battery's pre-registered seeds (100/200). **Fast loops run against
v0 as well as v27** (the bar reference is v27 now, not v7). The
full battery runs on the FINAL version only. The 100-game battery
is the verdict — small-sample smokes mislead; single-position
diagnostics cannot validate a global search change; the v1
walk-back oracle shares the argmax-of-sum flaw (iter-18) — every
nominated save needs playouts or exact computation.

## 0. The problem

White remains the persistent gap across all batch-10 versions:
v33 White 0.42, v35 White 13W-35L-2D vs v27 (0.28) while Black
went 38W-10L-2D (0.78). v27 itself is White-weak against
stronger opposition: vs v7 its 19 losses split 8 Black / 11
White, and its fast loop (iter-27 §2) lost 3 of 4 games as
White. This workstream reads v27's WHITE losses specifically —
from v35's forensics corpus (iter-35 §2: 11 White losses vs
v7, classified) and from fresh White games (v27-as-White vs
v27-as-Black and vs v0) — finds any shared mechanism, and
patches it.

**What is already known about v27's White losses (iter-35 §2,
re-stated here as the starting point, not re-derived):**

| White loss | Plies | iter-35 class |
|---|---|---|
| g7 | 31 | horizon (no save in last 4 turns) |
| g15 | 23 | horizon |
| g21 | 41 | missed VCF prevention (check fixes) |
| g35 | 55 | horizon |
| g45 | 23 | horizon (same line as g15 to ply 16) |
| g65 | 41 | horizon |
| g67 | 65 | horizon |
| g73 | 31 | missed VCF prevention (check fixes; same position as g91) |
| g77 | 21 | search-horizon, no signature (ply 13!) |
| g91 | 29 | missed VCF prevention (same position as g73) |
| g93 | 57 | search-horizon, no signature (400-sim proposal saves) |

So of 11 White losses: 6 horizon (5 distinct lines), 2
distinct VCF-prevention positions, 2 search-horizon
no-signature. The VCF-prevention mechanism is CLOSED as a
strength lever for the v27 family (v35 priced it: 0 clean
missed preventions in 448 threatened turns, yet clause 1
failed 0.53) — re-doing v35 is NOT this workstream's patch.
The live questions are: (a) do the 6 horizon losses share a
mechanism EARLIER than the last-4-turn workup window (the
window may simply be too short to see the losing move)?
(b) do the 2 search-horizon losses (g77, g93) share a
mechanism with each other or with the horizon losses?
(c) do fresh White games against v27-as-Black (a stronger
Black than v7) show the same classes, or new ones?

**The mechanism MUST be a value/leaf or horizon effect.**
White search/prior/book (batch 5: v10's 200 White sims went
0W-50L as White; v13's prior-temperature flattening was
catastrophic) and more sims (v10) are CLOSED levers and are
NOT the answer. Note what v10's failure implies: deeper
search AMPLIFIES White's problem rather than fixing it —
consistent with a leaf-value error that the search
converges on, not a visit-starvation gap. The patch rule
(§1.1) enforces this: a patch that reopens a CLOSED lever
is not allowed.

**Base engine (forensics patch).** v27 exactly as on master
(100 sims, c_puct 1.5, sharp v1 priors, rollout leaf both
colours, τ=0.05 threshold tie-break, most-visited root). A
forensics patch builds on v27, not on v18 — v27 is the bar
reference, and the patch must beat IT. v33–v35's code lives
on `research/batch-10` — read there if needed, do NOT
cherry-pick it.

## 1. Design and pre-registration

**Comparison target: v27** (the bar reference, batch-11 plan:
ladder Elo 636.4). Clause 1 is head-to-head vs v27 (normal
rules, 100 games, alternating colours, win rate > 0.50 with
Wilson 95% CI lower bound > 0.50); clause 2 is ladder Elo
strictly higher than v27's 636.4 (**40 games/opponent this
batch — the re-based ladder**, batch-11 plan; leg-result
reading for saturation: a 40-0 leg strictly dominates a
39-0-1 leg); clause 3 is tactical pass rate >= v27's
(20/20 old suite + 29/29 hard suite, on the CORRECTED
suite); clause 4 is all test suites green (ml pytest, npm
test, tsc). **Secondary head-to-heads, also pre-registered:**
v41 vs v0 (100 games) and a 100-game swap match vs v27
under the tempo judge (the harness default). **Strength
champion:** v18 (ladder Elo 636.4). Champion rule v2
(TRACKS.md, direct challenge): IF v41 passes all 4
clauses, it plays a 100-game challenge match vs v18
(seeds 100/200, normal rules) and takes the title only if
it wins that match (win rate > 0.50 with Wilson 95% CI
lower bound > 0.50) AND its ladder Elo beats 636.4
(automatic for a bar-passer). The coordinator makes the
champion call; this log reports the challenge match if it
runs.

### 1.1 Phase 1 — forensics (pre-registered protocol)

**Corpus (two parts):**
- **(A) v27's 11 White battery losses vs v7** (iter-35 §2's
  corpus, restricted to White; workup JSONs in
  `~/workspace/b10h3-logs/`). The VCF-prevention instances
  (g21, g73/g91) are NOT re-worked — their mechanism is
  CLOSED. The 6 horizon losses (g7, g15, g35, g45, g65,
  g67) and the 2 search-horizon losses (g77, g93) are the
  corpus. g45 is the same line as g15 to ply 16 — it
  counts as ONE distinct line with g15, not two instances.
- **(B) Fresh White games:** v27-as-White vs v27-as-Black
  (10-game self-play leg, dev seeds 1141/2141) and
  v27-as-White vs v0 (10-game leg, dev seeds 1142/2142).
  White losses from (B) are worked up with the same
  instruments. (B) tests whether the corpus-(A) classes
  reproduce against a stronger Black (v27, not v7) and a
  weaker one (v0).

**Workup instruments:**
- **Threat timelines** (`~/workspace/b11h3-logs/scripts/
  threat_timeline.py` — exact computation, no playouts):
  at every White turn, Black's VCF starts / open-four
  creators / setup moves before White's move, after
  White's move, and after Black's reply (the
  targeted_playout.py definitions; VCF-start enumeration
  pre-filtered to four-makers, exact by
  `starts_forced_win`'s four-creation requirement). The
  timeline finds WHERE a horizon loss becomes lost — the
  turn where Black's threat counts jump and never come
  back. White-win controls (2 games) are timeline-scanned
  the same way: a mechanism must separate losses from
  wins, not merely describe losses.
- **Extended-window workups** (`workup_loss.py`,
  `--last-turns 8`): the iter-35 workups used the last 4
  turns; a horizon loss's losing move may sit 5–8 turns
  back. Same-engine playout oracle (v27 vs v7 / v27 vs
  v27 / v27 vs v0), seeds 0/1/2, top-5 + 400-sim + VCF
  alternatives. A turn QUALIFIES only when the actual
  move's playout loses at ≥2/3 seeds; an alternative
  SAVES only when it draws or wins at ≥2/3 of the SAME
  seeds (the corrected method, iter-6 §2.2(a)). The v1
  walk-back oracle (`walkback_loss.py`) is NEVER used
  alone to verify a save. **Exact audit (iter-35 §2.3's
  rule):** a save is verified only if it passes BOTH the
  playout qualification AND the exact audit (0 opponent
  VCF starts after the save) — playouts can produce
  false saves against a weak opponent.
- **Root dumps** (`diag_root.py`, 100 sims, seed 0) and
  **rollout-leaf diagnostics** (b10h3's `diag_rollout.py`
  pattern: 1-ply rollout-leaf values of loser vs save)
  at every verified losing position, to diagnose which
  class (4–7) applies. **Exact threat analysis**
  (`targeted_playout.py --no-playout`) on every verified
  save to test whether an exact signature distinguishes
  it from the loser.

**Classification scheme** (iter-35 §1.1's, carried over —
a loss with a verified losing move is classified by the
ROOT CAUSE of that move, not by the loss's surface):

1. Missed block; 2. Missed forced win; 3. Horizon loss
   (no saving move in the window — now the EXTENDED
   window); 4. Prior starvation; 5. Leaf value error;
   6. Root-decision (visits-over-q); 7. Search-horizon
   value error (no exact signature, or a signature to be
   named). Class 1–2 instances are recorded but the
   VCF-prevention mechanism is CLOSED — a patch may not
   be a VCF check in any form (prior-pick v29,
   search-pick v33, search-first v35: all priced).

**Patch rule.** If a class has ≥3 verified instances
sharing ONE mechanism AND a patch reproduces the perfect
move at those positions (regression test), the patch IS
v41, on the v27 base. The mechanism MUST be a value/leaf
or horizon effect (see §0). A patch that needs a free
parameter fitted to the forensics positions is NOT
allowed — the mechanism must be principled (a feature, a
rule, or a search change), not a fitted constant. A patch
that reopens a CLOSED lever (q-based root decisions as a
general mechanism, root-decision for strength, the
tie-break lever, White search/prior/book, exploration
tuning, gap-three demotion in any combination, the
blend, combination-stacking, more sims, VCF-prevention
for the v27 family in any form) is NOT allowed — the
mechanism must be genuinely new. Max 2 patch iterations:
if the first patch fails to reproduce the saves at the
verified positions, ONE refinement is allowed; if that
fails too, the fallback (§1.2) triggers. The patched
engine is fast-looped (vs v0 AND v27, 10 games each,
fresh dev seeds) and then runs the full battery (§1.3).

**Extension rule.** If after corpus (A) + (B) no single
class has ≥3 verified instances, the forensics concludes
"no patchable mechanism found" and the fallback (§1.2)
triggers. The classification table is still committed in
full in §2 — a negative forensics result is a program
finding, not a failed iteration.

**Division of labour with H1/H2 (batch 11).** H1 (v39)
mines the setup signature (v0-g4/g32: a setup position
where v27's leaf is blind) — a GENERAL leaf-blindness
signature, both colours. H2 (v40) patches the leaf-blind
flavour of missed preventions (H2's confident misses
where the leaf ranks the unsafe move best). This
workstream's mechanism must be WHITE-SPECIFIC: a
mechanism that explains White's losses specifically,
not the general setup/leaf-blind mechanisms H1/H2 own.
If the forensics converges on H1's or H2's mechanism,
that is recorded as cross-validation for their
workstream, not patched here — and the fallback (§1.2)
triggers.

### 1.2 Phase 2 — fallback: cheaper VCF check for v29

Triggers ONLY if Phase 1 finds nothing patchable under
the rules of §1.1. It does NOT trigger merely because the
forensics is slow or the patch is hard — the patch rule's
two-iteration limit / the extension rule is the trigger.
(Carried over verbatim from iter-35 §1.2, which did not
need it — the fallback is batch 10 H3's unused design.)

**Base: v29** (v18's 2-ply rollout leaf in both colours +
the VCF-at-root check, 100 sims, c_puct 1.5, sharp v1
priors, most-visited root) — exactly as on master. The
fallback changes ONLY the VCF check's cost, not its
design: the offensive VCF step and the defensive
prevention step keep their exact semantics (depth-3 VCF
definition, highest-prior selection), and the engine is
otherwise exactly v29.

**The cheaper check (exact, pre-registered).** Two changes
to `HeuristicMCTS._vcf_root_move`, both in the defensive
step (the expensive part — the offensive step tests only
our few four-creators; the defensive step scans all
candidates, and the no-prevention full scan is the 36s
worst case, iter-29 §3.2):

1. **Defensive candidate limit (K=8).** Test only the top-8
   candidates by (prior desc, flat index asc) as
   preventions. If none of the top 8 prevents, fall
   through to the search instead of scanning the
   remaining candidates. Rationale: the selected
   prevention is the highest-prior one, so when a
   prevention exists in the top 8 the move is IDENTICAL
   to v29's; a prevention ranked below 8th by prior is a
   move the search would rarely pick anyway, and falling
   through to the search there is the safe default. This
   bounds the defensive scan to 8 candidates (from ~30+),
   cutting the worst case roughly 4x.
2. **VCF-starts-only re-check.** On each resulting board,
   test only the opponent's ORIGINAL VCF starts
   (`opp_vcf`), not all four-creators (`ordered = opp_vcf
   + others`). Rationale (monotonicity): our move adds
   our stone, which can only REMOVE the opponent's
   forcing options — a four-creator that was not a VCF
   start on the original board cannot become one after
   our move (the opponent's subsequent four-creators are
   a subset, and the defender's block options for a
   genuine four are unchanged — a block square our stone
   occupies means the four is already blocked). So
   testing the non-VCF four-creators is redundant. This
   is a pure speedup IF the monotonicity argument holds
   — verified empirically before any eval (mechanism
   check, §3.0): the cheaper check must produce
   IDENTICAL moves to v29's check on all 7 smoke-test
   positions (iter-29 §3.1) and all 49 tactical-suite
   positions, plus a 100-position sample of quiet v7
   self-play positions (the §3.2 bench set), wherever the
   candidate limit does not bind. Any disagreement is
   recorded and the monotonicity claim is withdrawn (the
   candidate limit alone is the fallback then).

**Why NOT depth-limiting.** The batch-10 plan allows
"depth/candidate-limited". Depth-limiting (depth 2
instead of 3) changes the exact VCF definition: a
prevention that removes all depth-2 VCF starts but
leaves a depth-3 VCF start live would be wrongly played
as a prevention, and the three tactical fixes (g26,
g79, v1bl-g2) may need depth 3. The candidate limit +
monotonicity re-check keep the depth-3 definition exact,
so the tactical fixes are preserved by construction
(verified by the mechanism check and the battery
clause 3).

**What the fallback is and is not.** It is a COST
reduction for v29 (target: mean root-check time well
below v29's 1.65s on the bench set, worst case bounded),
keeping v29's tactical 29/29. It is NOT expected to
beat v27 on strength — v29 scores 0.715 vs v7 and Elo
240.8, well below v27's 0.7900 / 636.4, and a cheaper
check does not add strength. The battery measures it
honestly against the v27 bar; the cost numbers are the
finding. If the fallback triggers, v41 = v29 + cheaper
check, registered as version 41 with a `vcf_root`
variant flag (e.g. `vcf_root_cheap`), and the fast loop
runs vs v0 AND v27 as usual.

**Cost guard.** Median/p90/max root-check time measured
on the 100-position bench set
(`ml/scripts/bench_vcf_root.py`) before the battery and
reported in §3; no design change is made after any eval
on the basis of cost alone.

### 1.3 Eval seeds (pre-registered)

- Battery h2h vs v27: v41=100 / v27=200; vs v0: v41=100 /
  v0=200.
- Ladder: seed 100, **40 games/opponent** (re-based,
  batch-11 plan), mcts_sims=100
  (`ml/scripts/eval_version.py --version 41 --ladder-games 40
  --out research/iterations/ladder-v41.json` — the auto-resume
  driver).
- Swap: 100 games vs v27, balanced placement, placement_seed
  11, tempo judge (the harness default), seeds 100/200.
- Tactical: `eval_tactical_full.py --version 41 --suite both
  --seed 100`.
- Challenge match (ONLY if v41 passes all 4 clauses): 100
  games vs v18, seeds 100/200, normal rules.
- Fast-loop dev seeds (patched engine): fresh seeds distinct
  from the battery's 100/200, the forensics legs'
  1141/2141 and 1142/2142, and batch-9/10 loop seeds.
- Harness commands follow iter-27 §1.4 / iter-35 §1.3 exactly
  (`selfplay_batch.py`, `eval_version.py` for the ladder,
  `eval_tactical_full.py` for tactical). All match drivers
  AUTO-RESUME from partial `--out` files — if a run is
  killed, relaunch the SAME command. The drivers now seed
  per game (`seed + game_index`) and audit uniqueness —
  every leg reports its distinct-game count and de-dup rate.

## 2. Forensics (Phase 1)

### 2.0 Corpus (A) — v27's White losses: the starting classification

From iter-35 §2 (not re-derived): 11 White losses vs v7 —
6 horizon (g7, g15, g35, g45=g15's line, g65, g67), 2
distinct VCF-prevention positions (g21, g73/g91 — CLOSED
lever, not re-worked), 2 search-horizon no-signature (g77
ply 13, g93 ply 49). The horizon losses are the corpus:
the iter-35 workups tested only the last 4 White turns,
and in all 6 the position was already lost at the
earliest tested turn (or, g15/g45 ply 15, the actual did
not qualify: [-1,1,1]). The extended-window hypothesis
(§1.1): the losing move sits 5–8 turns back.

**Threat timelines** (`~/workspace/b11h3-logs/scripts/
threat_timeline.py`, exact computation) trace Black's
VCF starts / creators / setups at every White turn.
g15's timeline (first): Black's setup count explodes
2 → 74 after Black's ply-8 reply, and White is purely
reactive thereafter (own vcf/cr/st ≈ 0 until ply 19),
removing Black's VCF starts every move while Black
regenerates 2–10 per reply; at ply 19 White fails to
remove all 6 and the game ends. g7's timeline matches:
setup explosion 1 → 53 after Black's ply-8 reply, White
reactive, White's own VCF starts (ply 17: 2) answered by
Black's reply, White fails to remove all VCF starts at
ply 21 (after: 1) and Black's count explodes to 92–102.
g35's timeline (55 plies): the same ply 7–8 setup
explosion (2 → 65), a quiet middle, then a mutual-VCF
race from ply 23 (White builds 2 → 4 → 16 VCF starts,
Black answers each; Black's ply-30 reply creates 23).
Timelines for g65/g67 + White-win controls (g13, g17)
in progress at the time of writing.

### 2.1 Game 7 — the extended window FINDS the losing move (ply 19, then ply 17)

**Ply 21 (5th-from-last White turn, just outside iter-35's
window): NO save.** Actual (12,4) loses 3/3; all 4
alternatives lose ((4,4) [1,-1,-1], 0 VCF after — a clean
prevention that still loses). The position was already
lost by ply 21.

**Ply 19 — VERIFIED losing move.** Actual (11,6) loses
3/3; THREE saves verify (playout + exact audit, 0
opponent VCF starts after): (9,6) [-1,1,1], (10,5)
[1,1,1], (10,7) [1,1,1]. (4,4) loses 3/3 despite 0 VCF
after. At ply 19 the position is QUIET (Black holds 0
VCF starts, 0 creators, 0 setups before White's move),
and all five moves are IDENTICAL on the 1-ply exact
counts (0/0/0 after each) — no 1-ply signature
distinguishes saves from losers.

**2-ply exact analysis REFUTES the "deny VCF-start
creation" signature** (`two_ply_exact.py`: Black's
VCF-start-creating replies per White move): actual
(11,6) 15 creating replies, max 12 VCF; save (9,6)
IDENTICAL (15, max 12); saves (10,5)/(10,7) 6 replies,
max 5; loser (4,4) the BEST numbers (4 replies, max 1)
yet loses 3/3. Minimising the opponent's VCF-start
creation does NOT save — it is anti-correlated here.
(4,4) plays away from the action; the saves contest
Black's threat region (9–10, 5–7). Black's worst reply
after (11,6)/(9,6) is (10,5) itself (12 VCF starts, 3
creators) — the save (10,5) OCCUPIES Black's setup
square.

**Root + leaf diagnosis (ply 19, seed 0, 100 sims): LEAF
VALUE ERROR (class 5).** Rollout-leaf ours-perspective:
(4,4) +0.2500 (highest — the must-block value: (4,4)
creates a White four), actual (11,6) +0.1977, save
(10,7) +0.1918, save (10,5) +0.1723, save (9,6) +0.1433
(lowest). The leaf ranks BOTH losers above ALL THREE
saves. The root agrees: (11,6) is the visit leader (rank
1, cut from the dump's tail), (4,4) 6 visits our-q
−0.010, saves at 4 visits with WORSE q (−0.034/−0.046/
−0.034); priors all ≈0.0525–0.0527 (noise-tied). The
v1-gradient leaves (v14/v20/v22) are saturated near the
+0.2 cap for all five (+0.198/+0.199) — the breadth
differences (Black's gap3 9 → 5 after (10,5)/(10,7))
cannot separate them. The rollout policy never plays
Black's quiet setup (10,5) (low v1 total), so the leaf
never demonstrates the danger; the gradient saturates.

**Ply 17 — also a verified losing move (earlier).**
Actual (5,4) loses 3/3; FIVE saves verify: (9,6)
[-1,1,1], (10,5) [-1,1,1], (10,7) [1,1,1], (11,6)
[1,-1,1] (the ply-19 actual — tempo matters: it saves
a move earlier), (10,9) [1,1,1]. Note (5,4) is White's
FLASHY move — it creates own vcf=2, cr=2, st=83 — and
it loses, while the quiet saves win: Black answers the
attack (ply 19 before: White vcf=0) and White loses
the tempo to play the real save. The saves at ply 17
and ply 19 are the SAME squares ((9,6), (10,5),
(10,7)) — a stable defensive structure, one game, one
mechanism instance (not two independent instances).

**Mechanism reading (provisional, 1 instance):** the
leaf is blind to Black's quiet setup move (10,5) — the
setup signature H1 (v39) is mining. The save occupies
Black's setup square. This is H1's mechanism, not a
White-specific one (the known signature instance,
v0-g4/g32, is a BLACK loss). Cross-validation for H1,
pending the other games (§2.2–2.4): if all White
losses are setup instances, the fallback (§1.2)
triggers per §1.1's division of labour.

### 2.2 Game 35 — race loss, no verified losing move at the transition plies

**Ply 29 (last quiet White turn before Black's 23-VCF
explosion at ply 30): NO save.** Actual (9,6) [1,-1,-1]
qualifies; all 5 alternatives lose 3/3. **Ply 27
(White's big attack — (9,4) creates own vcf=16): NO
save.** Actual [1,-1,-1] qualifies; all 5 alternatives
lose ((11,5) saves 1/3 only; the 400-sim proposal (9,10)
loses 3/3). The race loss is distributed drift or was
decided earlier at a turn where the actual does not
qualify. Class 3 (horizon loss) stands. Note the
structural finding: White builds enormous attacks (own
vcf 2 → 4 → 16 → 23 → 104 across plies 23–39) and
Black answers every one, while Black's threats
eventually become unanswerable (100+ VCF starts from
ply 45). White's attacks do not convert against a
defender whose defence is solved; Black's do, because
White runs out of preventions first — a tempo asymmetry,
not a patchable mechanism (no single move changes it:
the alternatives at both tested plies lose).

### 2.3 Game 15 (and g45, same line to ply 16) — no verified losing move

**Ply 13: actual (6,7) [-1,1,1] does NOT qualify**
(wins 2/3 in playouts vs v7). Ply 15 (iter-35 workup):
[-1,1,1], does not qualify. Ply 17+: lost with no save
in the window. The position is drawable against
v7-level play at plies 13–15 but lost against the
actual game's Black (whose ply-16 (10,5) creates 2 VCF
starts) — the playout oracle is too weak to see the
loss, and the protocol (same-engine oracle) cannot
name a losing move. Class 3 (horizon loss) stands.
This is the converse of iter-35 §2.3's false-save
finding: a weak playout opponent produces false
NON-qualifications as well as false saves.

### 2.4 Games 65, 67, 77, 93; corpus (B) self-play losses

**g65/g67 are ONE distinct line (shared prefix: 29
plies).** g65's timeline: setup explosion at ply 5→6
(st 2→43), a LONG quiet phase (plies 13–33, almost all
counts zero — White builds NOTHING, own st=0
throughout), then Black's explosion at ply 35→36 (st
2→99 after Black's reply); White fails to remove all
VCF starts at ply 37 (after: 3) and Black's count
explodes to 106. Transition ply 35 workup: RUNNING at
the time of writing (result recorded in §2.5). g67's
timeline: all-quiet through ply 41 (0/0/0 everywhere)
— the slowest positional loss in the corpus; Black
holds 5 VCF starts by ply 57 (iter-35) with no
prevention existing. Class 3 (horizon loss) for both
lines.

**g77 ply 13 and g93 ply 49 (iter-35's class-7
instances, re-audited here for the setup mechanism).**
Setup-overlap check (`setup_overlap.py`): g77 — Black
holds 4 setup squares [(7,6), (6,7), (8,7), (7,9)];
the ACTUAL (7,6) occupies one and LOSES, save (8,7)
occupies one and saves, save (7,5) occupies a
VCF-creating (non-setup) square and saves, save (7,4)
occupies NEITHER and saves. g93 — Black holds 6 setup
squares [(8,3), (8,5), (11,5), (11,6), (11,7),
(11,10)]; the ACTUAL (8,5) occupies one and LOSES,
the save (5,4) occupies NONE and saves. **Setup
occupation is ANTI-correlated with saving in both
games.** The setup mechanism is REFUTED for g77/g93 —
and for g7 too (ply 19: Black holds ZERO setup moves
on the current board; the (10,5) setup threat exists
only AFTER White's move). All three class-7 instances
have NO exact signature (1-ply, 2-ply, and
setup-overlap all refuted). Their leaf profiles are
heterogeneous (g7/g93: leaf ranks losers above saves;
g77: leaf ranks 2/3 saves above the actual), so no
shared leaf error exists to patch either.

**Corpus (B) — fresh White games.** v27-vs-v27
self-play (10 games, seeds 1141/2141): **5W-4L-1D,
with WHITE winning 7 of 9 decisive games** — the
White gap does NOT reproduce in self-play; if anything
it inverts (both engines win more as White: A-as-White
4W-1L, B-as-White 3W-1L-1D). v27 vs v0 (10 games,
seeds 1142/2142): 10W-0L-0D, no White losses. Corpus
(B) yields only 2 White losses: self-play g9 (A, 53
plies) and g4 (B, 27 plies). g9's timeline: quiet
until ply 19 (White attacks, own vcf=10), Black
answers with a setup explosion (st=93) at ply 20;
White reactive thereafter, fails to remove all VCF
starts at ply 29 (after: 4). Transition-ply workups
(g9 ply 25, g65 ply 35) RUNNING at the time of
writing (results in §2.5).

### 2.5 Final classification (corpus A + B)

| Class | Games / lines | Distinct positions | Patchable? |
|---|---|---|---|
| 1. Missed VCF prevention | g21, g73/g91 (2 positions) | 2 | NO — CLOSED (v35) |
| 3. Horizon loss | g15/g45 (1 line), g35, g65/g67 (1 line) | 3 lines | NO — no verified losing move at any tested turn (g15 plies 13/15 don't qualify; g35 plies 27/29 no save) |
| 7. Search-horizon, no signature | g7 (ply 17/19), g77, g93 | 3 | NO — no shared positive mechanism (see §2.4: leaf profiles heterogeneous; all exact signatures refuted) |
| 2. Missed forced win | 0 | 0 | — |
| 4. Prior starvation | 0 | 0 | — |
| 5. Leaf value error (with signature) | 0 | 0 | — (g7/g93 are leaf errors WITHOUT a signature → class 7) |
| 6. Root-decision (visits-over-q) | 0 as primary | — | — (g77's best-q save sits one visit below the tie — CLOSED lever) |
| Corpus (B) self-play | g9, g4 (2 losses) | 2 | NO — g9 ply 25 actual [1,-1,1] does not qualify; g65 ply 35 (corpus A) qualifies but no alternative saves (all lose 3/3) |

**Forensics conclusion: NO PATCHABLE MECHANISM FOUND.**
The patch rule (§1.1) requires ≥3 verified instances
sharing ONE mechanism. Class 7 has 3 instances (g7,
g77, g93) but is defined by the ABSENCE of a shared
mechanism — the three instances are heterogeneous on
every tested axis (leaf profile, root profile, setup
signature, 2-ply threat counts, gradient saturation:
g7 saturated near the +0.2 cap, g77/g93 not). No
principled patch distinguishes the saves from the
losers at all three positions; any patch would fit
2–3 positions, which §1.1 forbids. The horizon losses
(3 distinct lines) have no verified losing move at any
tested turn (g15 plies 13/15 don't qualify; g35 plies
27/29 and g65 ply 35 qualify but no alternative saves;
self-play g9 ply 25 doesn't qualify). The
VCF-prevention instances are CLOSED.
**Per §1.1's extension rule, the fallback (§1.2)
triggers: v41 = v29 + the cheaper VCF check.** The
negative result is itself the finding: White's deficit
is distributed drift + no-signature horizon losses,
not a single patchable flaw — and in self-play White
is not weak at all (7-2-1), so the "White gap" is
opponent- and matchup-dependent, not a systematic
v27-White defect.

**Forensics completeness note.** The g67 timeline and
the White-win control timelines (g13, g17) and the
self-play g4 timeline were still running when the
conclusion was reached; they were stopped (CPU needed
for the fallback battery) because no pending timeline
could change the conclusion — the conclusion rests on
the verified-save analysis (§2.1–2.5), not on the
timelines, and the timelines already completed (g7,
g15, g35, g65, g9-partial) all show the same
structural pattern. The classification table above is
complete for corpus (A) + (B) as pre-registered.

## 3. Patch (v41) — the fallback, pre-registered

**Forensics verdict (§2.5): no patchable mechanism.**
The fallback (§1.2) triggers: **v41 = v29 + the cheaper
VCF check** (batch 10 H3's unused fallback, carried
verbatim here). v29 is the base (on master) — NOT v27,
because the cheap check is a modification of v29's VCF
check, and v29's tactical fixes are the thing being
kept.

### 3.1 The change (two parts, defensive step only)

In `HeuristicMCTS._vcf_root_move` (step 2, defensive
prevention), gated by the new `vcf_root_cheap` flag
(registry key, threaded through `get_player`):

1. **Candidate limit K=8.** Test only the top 8
   prevention candidates by (prior desc, flat index
   asc). If none of the top 8 prevents, the check
   returns None and the search decides. A prevention
   that ranks below 8th is MISSED — this change CAN
   change the selected move (by design; the prevention
   is usually the obvious block, which is high-prior).
2. **VCF-starts-only re-check.** On each resulting
   board, test only the opponent's ORIGINAL VCF starts
   (`opp_vcf`), not all original four-creators
   (`ordered = opp_vcf + others`). Monotonicity claim:
   our stone can only REMOVE opponent forcing options
   (it blocks attacker lines, occupies attacker
   squares, and can only ADD defender counter-threats),
   so a four-creator that was not a VCF start on the
   original board cannot become one after our move.

The offensive step is unchanged (it tests only our few
four-creators and is cheap). NO depth-limiting (would
change the exact VCF definition; tactical fixes may
need depth 3).

### 3.2 Mechanism check (BEFORE any battery)

Change (2) is a claimed move-identity and must be
verified empirically: **v41's `_vcf_root_move` must
produce IDENTICAL moves to v29's on all 49 tactical
positions + the 100-position bench set
(`ml/scripts/bench_vcf_root.py`, seed 777), wherever
the candidate limit does not bind** (i.e. where the
v29 prevention ranks in the top 8, or where neither
finds a prevention). Where the limit binds, the move
may legitimately differ (v41 falls through to the
search); those positions are counted and reported
separately. **Any disagreement where the limit does
not bind → the monotonicity claim is WITHDRAWN and
the candidate limit alone is the fallback** (change
(2) reverted). Cost guard: median/p90/max root-check
time on the bench set, v29 vs v41. v29 baseline
(iter-29 §3.2): fired 29/100, median 90.5ms, p90
2.94s, max 36.4s, mean 1.65s.

**RESULT — mechanism check PASSES (script
`ml/scripts/mechanism_v41.py`, log
`~/workspace/b11h3-logs/mechanism-v41.json`, exit 0):
149/149 positions AGREE, zero disagreements of any
kind.** 0 no-bind disagreements (the monotonicity
claim holds on all 149 positions — change (2)
stands), 0 limit-binds disagreements (the candidate
limit never binds on this set — change (1) stands as
a pure worst-case cap here). v29's 71 fires rank:
0 (52), 1 (2), 2 (2), 3 (7), 4 (6), 5 (1), 7 (1) —
max rank 7, exactly at the K=8 boundary (ranks 0–7):
the limit's safety margin is thin on this set, so
the battery is the real test of whether any game
position has a prevention below rank 8. Bench-only
(n=100, comparable to iter-29's baseline): v29 fired
29, median 95.0ms, p90 2.72s, max 38.3s, mean 1.71s
(reproduces iter-29's baseline closely — same bench
set); v41 fired 29, median 98.4ms, p90 3.23s, max
13.8s, mean 0.91s — **mean 1.9x faster, max 2.8x
faster**; median/p90 roughly unchanged (the gate +
offensive step, which are unchanged, dominate the
typical position). The savings concentrate in the
no-prevention tail (bench:86 38.3s→10.2s, bench:50
33.1s→3.3s, bench:23 29.0s→3.2s — v29 tests ALL
candidates × all four-creators when no prevention
exists; v41 tests 8 × VCF starts only). A few
individual positions measure slower for v41
(bench:15 8.6s→13.8s, bench:84 8.4s→11.0s — both
rank-0 preventions, where v41 tests strictly fewer
moves); those are measurement noise from the
contended VM (H1/H2 running concurrently), not a
real effect — the mean improvement is driven by the
10x tail savings, which are too large for noise.
Tactical (n=49): v29 mean 2.17s → v41 2.11s (the
gate dominates threat-rich positions). **Both changes
stand. v41 proceeds to the fast loop and battery.**

### 3.3 What v41 is (and is not)

v41 is a COST reduction, NOT a strength patch. It
keeps v29's tactical fixes (the VCF-at-root check that
solved v27's missed preventions) while reducing the
~1.65s/move cost. It is NOT expected to beat v27 on
strength: v29 scored 0.715 vs v7 (Elo 240.8) — well
below v27's 0.78/636.4 — and v41 differs from v29 only
by occasionally missing a low-prior prevention. The
battery will measure whether the cost reduction costs
any strength (vs v29's numbers) and whether the
tactical suite stays green. **The honest verdict is
likely REJECT on clause 1/2 (strength), with the
deliverable being the cost reduction + the forensics
finding that White's gap is not a single patchable
flaw.** The bar is still run in full — the battery is
the verdict, not the expectation.

## 4. Fast loop

v41 is a cost reduction, not a strength patch, so the
fast loop is a smoke test (does it play sensibly? does
it crash?), not a patching loop — there is no strength
mechanism to iterate on. Dev seeds 3141/4141 (vs v0)
and 3142/4142 (vs v27), fresh (distinct from battery
100/200, forensics 1141/2141, 1142/2142, and batch-9/10
seeds 1001/2001, 1031/2031, 1035–1038/2035–2038,
5001/6001).

**vs v0 (10 games): 5W-5L-0D (0.50, CI [0.2366,
0.7634]), 8/10 unique games (de-dup 0.50), 742s.**
Within noise of v29's 0.665 vs v0 (the CI includes
it); n=10 has no verdict power. The losses are
v29-style: game 0, v41-as-Black loses to an open-four
creation (White's diagonal (5,7),(6,8),(7,9),(8,10)
with both ends open — unblockable once created, not a
missed block). No crashes, no anomalous play.

**vs v27 (10 games): 6W-3L-1D (0.65, CI [0.3537,
0.8631]), 10/10 unique games, 1156s.** Small n — the
CI is wide and includes 0.50; no verdict power.
v41-as-Black 3W-1L-1D, v41-as-White 3W-2L. Black won
6 of 9 decisive games (colour effect, not a v41
signal). Game 6 was a 225-ply draw (full board).
Resumed after a VM restart killed the chain at 5
games (4W-1L) — per-game seeding reproduced the run
exactly; the 5 recorded games were kept, not
restarted (resumed_from_game 5).

**Tactical (clause 3): 20/20 old + 29/29 hard at
`--seed 100`** — v41 keeps all of v29's tactical
fixes, including the three prevention positions
(g26, g79, v1bl-g2) and the VCF positions. **Clause 3
PASSES.** Artefact: `~/workspace/b11h3-logs/tactical-v41.log`.

**Tests (clause 4):** full ml pytest **410 passed, 7
skipped, 58 xfailed, 0 failures** (883s; a VM restart
killed the first run at ~75% with no failures
recorded, re-run clean); npm test 55/55; `npx tsc
--noEmit` clean. **Clause 4 PASSES** (6 new v41 unit
tests included in the 410).

## 5. Battery

COMPLETE (all legs auto-resumed across 4 VM restarts;
per-game seeding reproduces the uninterrupted run —
no leg was restarted, no seed was changed).

**vs v27, 100 games, normal rules, seeds 100/200:
47W-43L-10D, 0.52, CI [0.4232, 0.6154].** 99/100
unique games (de-dup 0.5253, CI [0.4278, 0.6208]).
Per-colour: as Black 29W-18L-3D (0.61), as White
18W-25L-7D (0.43) — the White gap persists (0.43 vs
0.61). Elapsed 6691s. Artefact:
`research/iterations/batch-v41-vs-v27.json`.

**vs v0, 100 games, seeds 100/200: 59W-40L-1D,
0.595, CI [0.497, 0.686].** 41/100 unique games
(engine determinism vs v0; de-dup 0.5976, CI
[0.4454, 0.733] — the duplicates do not change the
verdict). Per-colour: as Black 34W-15L-1D (0.69), as
White 25W-25L-0D (0.50). Below v29's 0.665 vs v0
(CI [0.568, 0.750]) on the point estimate, but the
intervals overlap — not a significant difference.
Elapsed 5917s. Artefact:
`research/iterations/batch-v41-vs-v0.json`.

**Swap vs v27, 100 games, tempo judge, balanced
placement, placement_seed 11: 56W-39L-5D, 0.585, CI
[0.487, 0.6767].** 98/100 unique (de-dup 0.5867, CI
[0.4878, 0.6792]). Per-colour: as Black 31W-18L-3D
(0.625), as White 25W-21L-2D (0.542). Swap > normal
(0.585 vs 0.52), the same direction as v29 (0.735
swap vs 0.715 normal vs v7). Elapsed 2077s.
Artefact: `research/iterations/batch-v41-vs-v27-swap.json`.

**Ladder, 40 games/opponent (re-based this batch):
Elo 88.7, interval [-20.6, 198.1].** Legs: random
40-0-0 (1.0, CI [0.912, 1.0]); heuristic (v0)
25W-15L-0D (0.625, CI [0.470, 0.758]; Black 12-8,
White 13-7); mcts 28W-12L-0D (0.70, CI [0.546,
0.819]; Black 18-2, White 10-10 — the White gap
again, this time vs the ladder MCTS). The Elo is
driven by the weak heuristic leg (0.625 at 40 games
→ 88.7), below v29's 240.8. Artefact:
`research/iterations/ladder-v41.json`.

**Defence audit (pooled, `classify_losses.py` on all
3 legs): 122 losses, 122/122 via open four, 0
missed blocks, 1 clean missed prevention in 654
threatened turns (0.15%).** Defence stays solved.
The single missed prevention (swap game 15, ply 18)
is the K=8 candidate limit binding in real play —
the first observed instance (the mechanism check's
149 positions had max prevention rank 7): White held
11 VCF starts; v29's full check finds the prevention
(10,10) (rank 11 of 85, prior 0.03699 vs 0.03712 for
the rank-7 candidate — a 0.00014 prior gap), which
removes all 11 VCF starts (11 creators remain, 0 VCF
starts); v41's top-8 check misses it, plays the
top-prior (10,8) via the search, and loses 4 plies
later. Cost of the limit in the battery: 1 game in
300 (0.33%). This is the pre-registered trade-off
(§3.1), not a bug — but it confirms the limit's
safety margin is thin (prior noise decides whether a
quiet prevention makes the top 8).

## 6. Bar check (v41 vs v27)

| Clause | Requirement | v41 | Verdict |
|---|---|---|---|
| 1. h2h vs v27 | ≥100 games, rate > 0.50, Wilson CI lower > 0.50 | 47W-43L-10D, 0.52, CI [0.4232, 0.6154] | **FAIL** |
| 2. Ladder Elo | > 636.4 | 88.7, interval [-20.6, 198.1] | **FAIL** |
| 3. Tactical | ≥ 20/20 + 29/29 (corrected suite) | 20/20 + 29/29 | **PASS** |
| 4. Tests | all suites green | pytest 410 passed / 0 failed; npm 55/55; tsc clean | **PASS** |

**Verdict: REJECTED (2/4)** — as pre-registered in
§3.3. v41 is a cost reduction, not a strength patch,
and it does not beat v27. No challenge match vs v18
(requires all 4 clauses). v18 remains the heuristic
champion; v27 remains the bar reference.

## 7. What this means

1. **The White-side forensics found no patchable
   mechanism (§2.5).** White's losses are
   distributed drift + no-signature horizon losses;
   the three class-7 instances (g7, g77, g93) share
   no positive mechanism (leaf profiles
   heterogeneous; setup, 2-ply, and saturation
   signatures all refuted). And in self-play White
   is not weak at all (7-2-1) — the "White gap" is
   matchup-dependent, not a systematic v27-White
   defect. The fallback triggered, as pre-registered.
2. **v41 delivers the cost reduction it promised
   (§3.2).** Bench mean 1.71s → 0.91s (1.9x), max
   38.3s → 13.8s (2.8x), with identical moves to v29
   on 149/149 mechanism-check positions. The
   monotonicity claim (VCF-starts-only re-check)
   held everywhere tested; the K=8 candidate limit
   bound exactly once in 300 battery games (swap
   g15), costing 1 game. The tactical suite stays
   20/20 + 29/29.
3. **v41's strength is ~v29's, with a surprising
   vs-v27 number.** 0.52 vs v27 (CI [0.42, 0.62])
   is above v29's implied level vs v27, and 0.585
   under swap is above 0.50 — but the CIs include
   0.50 and the ladder Elo (88.7) is below v29's
   240.8, driven by the weak vs-v0 leg (0.595).
   The vs-v27 number is most plausibly noise (the
   CI is wide); there is no mechanism by which
   missing a prevention helps. Do not over-read it.
4. **The White gap is the durable finding across
   v41's battery** (Black 0.61-0.69 vs White
   0.43-0.50 in every leg; ladder MCTS Black 18-2
   vs White 10-10), and the forensics says it is
   not one patchable flaw. Future White work needs
   a different framing — the deficit is distributed
   across the game, not concentrated in a
   classifiable losing move.
