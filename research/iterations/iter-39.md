# Iteration 39 — v39: setup-signature mining + patch (batch 11, H1)

**Status: MINING IN PROGRESS.** This is the H1 workstream of batch
11 (H2 is v40, leaf-blind patch; H3 is v41, White-side forensics);
work happens on branch `research/batch-11-h1` in the
`gomoku-b11-h1` worktree, branched from `research/batch-11` @
`2188b53`. Comparison target is **v27** (the batch-11 bar
reference: ladder Elo 636.4). v39 is the next free number
(confirmed at start: batch-10 owns v33–v35 on
`research/batch-10`, ML batch 3 owns v36–v38 on
`research/ml-batch-3`, the local registry goes to v29).

**Method (carried over):** the fast 10-game loop — play 10 games,
read EVERY move of every game, find THE losing move per game
(verified with the solver or a deeper search — never guessed; for
rollout-leaf engines, playouts or exact computation, NOT the v1
walk-back oracle alone — iter-18 §2.1), patch, regression-test,
repeat. Fast loops run against v0 AND v27. The full 4-clause
validation battery runs on the FINAL patched v39 only. Fast-loop
games use development seeds (1001/2001, 1002/2002, ...) distinct
from the bar battery's pre-registered seeds (100/200).

## 1. Design and pre-registration

**Comparison target: v27** (ladder Elo 636.4). The bar's clause 1
is head-to-head vs v27 (normal rules, ≥100 games, alternating
colours, win rate > 0.50 with Wilson 95% CI lower bound > 0.50),
clause 2 is ladder Elo strictly higher than v27's 636.4
(**re-based this batch, pre-registered: 40 games/opponent** —
batch-11 plan; leg-result reading for saturation: a 40-0 leg
strictly dominates a 39-0-1 leg), clause 3 is tactical ≥ 20/20
old + 29/29 hard on the CORRECTED suite (batch 9 added (9,7) to
winning-quiet-setup-2), clause 4 is all test suites green (ml
pytest, npm test, tsc). **Secondary head-to-heads, also
pre-registered:** v39 vs v0 (100 games) and a 100-game swap
match vs v27 under the tempo judge (balanced placement,
placement_seed 11). **Strength champion:** v18 (Elo 636.4) — a
bar-passer takes the title by DIRECT CHALLENGE (100-game match
vs v18, CI lower > 0.50 and Elo above 636.4); the challenge
match runs ONLY for a bar-passer.

### 1.0 The setup signature (pre-registered definition)

Batch 10 H3's forensics (iter-35 §2.1) found ONE distinct setup
position (v0-g4/g32 ply 52, a transpose) where v27's leaf is
blind. Definitions (from `ml/scripts/targeted_playout.py`):

- **VCF start**: a move starting a forced win by consecutive
  fours (`evaluate.starts_forced_win`, depth 3).
- **Creator**: an open-four creator — a move leaving the player
  ≥ 2 winning moves.
- **Setup**: a QUIET move after which the player holds ≥ 2
  open-four creators (and which is not itself a VCF start or an
  immediate win).

**The setup signature** (a v27 turn in a lost game): after v27's
actual move, the opponent holds **≥ 1 setups with 0 VCF starts
and 0 creators**, and some alternative move removes **ALL**
opponent setups (0 VCF starts / 0 creators / 0 setups after it)
while the actual move leaves them live. The known instance: 2
setups after the actual (11,3); both verified saves ((9,4),
(5,8)) leave 0/0/0. The 0-creators condition is part of the
signature: if the opponent holds a creator after the actual
move, the threat is one ply closer (a creator-prevention miss,
a different class); the setup signature is specifically the
threat TWO plies out, which v27's 2-ply rollout leaf cannot see
(the leaf evaluates the position after the opponent's reply —
a setup is a move the opponent has NOT yet played).

**What counts as an instance.** A mined candidate counts as a
verified setup-signature instance iff ALL of: (1) the actual
move's playout loses at ≥ 2 of 3 seeds (match engines, seeds
0/1/2 — the workup protocol's corrected method); (2) the
removing move's playout draws/wins at ≥ 2 of the same 3 seeds;
(3) the exact audit (iter-35 §2.3): 0 opponent VCF starts, 0
creators, 0 setups after the removing move (recomputed at
verification, not trusted from the filter). Instances are
counted as DISTINCT positions: identical boards and transposes
(x↔y and the other board symmetries) collapse to one, as in
iter-35 (g73/g91 identical, v0-g4/g32 transpose).

**The patch rule (pre-registered).** If the mining finds **≥ 5
distinct verified instances**, the setup signature is a real
strength lever and v39 = v27 + a setup-prevention patch (exact
form pre-registered in §1.2 after the mining, BEFORE any eval —
the patch design depends on what the instances look like: which
removal mechanism dominates, whether the removing moves are
setup squares, creator blocks, or quiet moves). If the mining
finds **fewer than ~5 distinct instances**, the signature is too
rare to be a strength lever — say so in this log and fall back
to the leaf-blind direction (§1.3), with a mechanism DISTINCT
from H2's confident-miss flavour (H2 owns: unique visit leader
+ leaf ranks the unsafe move best — iter-34 §3.6's g61/g73).

### 1.1 Mining protocol (pre-registered)

**Corpus.** v27's losses in the batch-10 legs — all fresh, none
worked up in iter-35 (iter-35 worked up v27's own normal
battery: 19 losses vs v7 + 3 vs v0, yielding the 1 known
distinct setup position): batch-v33-vs-v27 (49 losses),
batch-v34-vs-v27 (45), batch-v35-vs-v27 (51), the three swap
legs (57 + 56 + 59), and batch-v27-vs-v7-swap (21) — 338 losses
nominal, deduplicated by move sequence (the batch-10 legs
contain seed-replay duplicates). v27's own normal battery is
EXCLUDED from the mining count (already worked up; the known
instance is the baseline, not a new find) but is used to
VALIDATE the filter (it must re-find v0-g4 ply 52).

**Filter** (`research/iterations/mine_setup_signature.py`, exact
analysis only, no playouts): for each unique v27 loss, v27's
last 4 turns (the workup window — the known instance sits at
the 4th-from-last turn). Per turn, checks in cost order:
creators after the actual move (0 required), VCF starts after
the actual move (0 required — else the VCF-prevention class,
CLOSED), then setups after the actual move (≥ 1 required).
Pre-conditioned turns (0/0/≥1) go to a second pass testing
removing moves in source-priority order — setup squares, the
creators the opponent would hold after each setup (occupying
one blocks the setup), top-5 1-ply moves, and (lazily, only if
no earlier source removes) the 400-sim MCTS proposal — stopping
at the first full removal (0/0/0). Turns with > 6 setups are
recorded as capped, not tested (a single move is implausible
as a full removal there; revisit if the yield is close to the
bar). The filter is validated on the known instance BEFORE the
full run: it must flag v0-g4 ply 52 with (9,4) and (5,8) among
the removing moves.

**Verification** (`research/iterations/verify_setup_candidate.py`):
the workup protocol (§1.0, clauses 1–3) with the MATCH engines
(v27 vs the leg's winner version). v33/v34/v35 exist only in
the batch-10 worktree, so those candidates verify there; v7
candidates verify here. Cross-tree faithfulness checked at
start: game.py, workup_loss.py, targeted_playout.py are
byte-identical between the trees; evaluate.py differs only by
the v33 four-creator optimisation (verified A/B-identical, 0
mismatches — iter-33 §2.3); scoring.py/mcts_heuristic.py differ
only by the v33–v35 additions (default-off flags for v27).

**Cost note (profiled).** The naive exact analysis is ~7s per
position, dominated by `_vcf_starts` on rich boards (profile:
60% of time) and `_setups` (~5s, O(C²) winning-moves calls).
The filter's cost ordering (creators → VCF → setups, with the
400-sim proposal lazy) cuts the validation game from 112s to
30s. The full run is resumable (per-game JSON writes).

**Fast exact filter (validated by A/B).** The setups scan is
window-filtered: a setup move must lie in a 5-window holding ≥
2 of the player's stones (proof in the code — a setup's creator
completes a five through the setup move), and a new creator's
four must include the setup move (so only windows THROUGH the
setup move are tested). Both filters need the precondition that
the player holds 0 creators on the input board (both call sites
check creators first). Validated by
`research/iterations/ab_setup_filter.py` (inner check +
direct check vs the slow exact versions, 0 mismatches required;
artefact `research/iterations/setup-mining-ab.json`). The
verification script keeps the SLOW exact functions as an
independent cross-check on the filter.

**VCF starts are NOT monotone under our move (finding).**
Winning moves, creators and setups are: our stone can only
remove the opponent's (it can never complete an opponent five).
But a VCF start can APPEAR after our move — verified on
batch-v33-vs-v27 g0 ply 29: after the actual move the opponent
holds 0 VCF starts / 0 creators / 1 setup [(5,9)]; occupying
(5,9) removes the setup but leaves the opponent with FIVE VCF
starts (our stone removed one of our own defensive replies, so
the opponent's forcing sequence now works). Consequence: a
setup-removal test MUST re-check VCF starts after the move
(the filter does — the 0/0/0 requirement), and any future
check must NOT assume VCF monotonicity (only creators/setups
are monotone). This is also why the single-setup turn at g0 ply
29 has NO removing move: the only setup square is a poisoned
removal.

### 1.2 The patch (pre-registered AFTER mining, BEFORE any eval)

[To be written when the mining completes: the exact patch
definition, selection rule, cost analysis, and what would
refute it. The patch is pre-registered here before the fast
loop or any eval game.]

### 1.3 Fallback: leaf-blind direction (if mining yields < ~5)

[To be written only if the fallback triggers: the diagnosis
(leaf mis-ranking with a mechanism DISTINCT from H2's
confident-miss flavour), the patch, and the pre-registration.]

### 1.4 Eval seeds (pre-registered)

h2h vs v27: v39=100 / v27=200; h2h vs v0: v39=100 / v0=200;
ladder seed 100, **40 games/opponent**, mcts_sims=100 (re-based,
batch-11 plan); swap 100 games vs v27, balanced placement,
placement_seed 11, tempo judge, seeds 100/200; tactical
`--seed 100`. Fast-loop dev seeds 1001/2001, 1002/2002, ...
(per iter-8 §4's policy). All match drivers auto-resume from
partial `--out` files and seed per game (`seed + game_index`)
with a uniqueness audit on every leg (ML batch 3 tooling, on
master) — batteries must survive VM replacements, and every
leg reports its distinct-game count and de-dup rate.

## 2. Mining results

[To be written when the mining completes.]

## 3. Fast loop

[To be written.]

## 4. v39 evaluation

[To be written.]

## 5. Bar check (v39 vs v27)

[To be written.]

## 6. What this means

[To be written.]
