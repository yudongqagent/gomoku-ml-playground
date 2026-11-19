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

**NOT TAKEN — the setup signature fails the count.** The mining
found 9 filter candidates in 293 unique v27 losses; the
verification + turn workups verified **2 distinct
setup-signature instances** (v34-swap g46 ply 24, removing
(8,9) saves 3/3 with a clean 0/0/0 audit; v35 g87 ply 36,
removing (4,2) draws 3/3 with a clean audit). Together with
the 1 known instance from iter-35 (v0-g4 ply 52), that is
**3 distinct instances — below the pre-registered ≥5 bar**.
The setup signature is too rare to be a strength lever on its
own, and the workups show why: at 2 of the 4 saving turns the
save is NOT a setup removal (g58's (5,6) leaves 2 setups,
g26's (5,8) leaves the same 1 setup), so a pure
setup-prevention patch cannot reproduce them. The fallback
(§1.3) triggers.

### 1.3 Fallback: leaf-blind direction (TRIGGERED — the
setup-signature count is 3 < 5)

**The diagnosis (verified on 4 saving turns + 2
confidently-wrong turns).** In quiet positions where the
opponent holds setups (0 VCF starts / 0 creators after the
actual move), the rollout leaf ranks the losing quiet move
above a saving quiet move, and the 100-sim search follows
the leaf (visit leader) or cannot discriminate (flat root,
max 4 visits):

| Turn | Save (3/3) | Leaf: actual q vs save q | Root |
|---|---|---|---|
| v35 g58 ply 19 | (5,6) [1,1,1] | −0.012 vs −0.112 | flat (max 4) |
| v34-swap g46 ply 24 | (10,7), (8,9) [1,1,1] | +0.054 vs +0.001/+0.018 | actual leads (12) |
| v35 g26 ply 25 | (5,8) [1,1,1] | −0.066 vs −0.137 | flat (max 4) |
| v35 g87 ply 36 | (4,2) [0,0,0] | −0.077 vs −0.132 | actual leads (8) |

Two more turns show the same blindness with no save at the
turn (the loss was earlier): v34 g55 ply 34 (actual 63
visits, q +0.365) and v33-swap g62 ply 40 (actual 61 visits,
q +0.915 — the leaf is near-certain a 3/3-losing move wins).
The rollout policy never plays setup moves, so the leaf
cannot see setup maturation; the loss arrives 10+ moves
later through quiet play, not a forced win.

**DISTINCT from H2's confident-miss flavour** (iter-34 §3.6,
g61/g73): there the actual move leaves opponent VCF starts —
an immediate forced win the leaf cannot see. Here the
actual leaves NO VCF starts and NO creators in any of the 6
turns — the threat is a setup (2 plies out), not a VCF start
(0 plies out). The mechanism is setup-tempo blindness, not
VCF blindness. H2 owns the VCF flavour; v39 owns the setup
flavour.

**Refuted alternatives (from the same forensics).**
Deeper search does NOT fix it: the 400-sim MCTS proposal at
g26 is (11,8), which loses 3/3 — more search on a wrong leaf
confirms the wrong move (g62's 61-visit, q +0.92 root is the
extreme case). A leaf-value setup term is infeasible at
acceptable cost (a setups computation per leaf call is
~0.3s × 100 sims). A rollout-policy setup rule is
infeasible (a setups computation per rollout move). What
remains is a root-level exact check — the v35 shape.

**The patch (pre-registered): v39 = v27 + setup-at-root
check (search-first, full-removal-only).**
After the search picks its move `m`, compute the opponent's
threats after `m` (exact: VCF via the fixed `_has_vcf_start`,
creators, setups via the window-filtered `_setups_limited`).
The check fires ONLY in the quiet case: 0 VCF starts / 0
creators / ≥1 setups after `m` (and ≤6 setups — above that
a single removal is implausible, the mining's cap). When it
fires, test candidates for a FULL removal (0 VCF / 0 creators
/ 0 setups after the candidate — the VCF re-check is a FULL
re-check, NOT the original-starts subset, because VCF starts
are NOT monotone under our move: occupying a setup square
can CREATE opponent VCF starts by removing one of our own
defensive replies, mining batch-v33-vs-v27 g0 ply 29).
Candidates, in priority order: the setup squares, the top-5
1-ply moves, the creators-after-setup. Collect ALL full
removals (no first-success exit); play the one the search
ranks highest (most visits, then our-q, then prior —
unvisited removals rank below all visited ones, by prior
among themselves). This selection rule is the fix for v35's
documented flaw (prior-based choice among preventions is
noise at 1e-4 margins). If `m` itself is a full removal the
gate is closed (0 setups after `m`) and the search's move
stands. If no full removal exists, the search's move stands.
Forced blocks and immediate wins short-circuit before the
search, so the check never touches them (g49/g65 are
unaffected by construction).

**Coverage on the forensics corpus.** The check reproduces
the verified save at 2 of the 4 saving turns (g46, g87 —
both full removals). At g58 it plays (3,8) (loses 3/3, the
actual's outcome — neutral). At g26 it plays (6,7) ([-1,1,-1]
— 1 draw vs the actual's 0 — a marginal gain). At the 3
no-save searched turns (g55, g62, g97) it plays the removal,
which loses 3/3 like the actual (neutral). The check is
weakly positive on every forensics turn and fixes 2/4 saving
turns. The battery measures whether that is enough strength
against v27 itself.

**Why this is not a CLOSED lever.** Not v35's VCF-prevention
(setups, not VCF starts — one ply earlier; the check fires
where v35's gate is closed). Not v3's creator check (creators
are 1-ply threats; setups are 2-ply — the "earlier threat
detection" iter-35 §7 asked for). Not a q-based root
decision or a tie-break change (an exact check, like v29's
and v35's; the root rule is v27's unchanged). Not
combination-stacking (one check for one verified mechanism,
the v35 precedent). Cost (~0.5s/move gate, ~3s/move worst
case when the gate opens) is reported, not acted on.

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

**Mining in progress** (224/293 unique v27 losses scanned at last
check): 9 filter candidates so far (listed in
`research/iterations/setup-mining.json`).

**Verification (in progress): 0 verified of 5 tested.** Every
tested candidate's actual move qualifies (loses 3/3 in playouts),
but NO removing move saves:

| Candidate | Actual playout | Removing move playout | Audit | Verdict |
|---|---|---|---|---|
| v34 g55 ply 34 | [-1,-1,-1] | (8,12) [-1,-1,-1] | CLEAN 0/0/0 | removal does not save |
| v35 g26 ply 25 | [-1,-1,-1] | (6,7) [-1,1,-1] | CLEAN 0/0/0 | 1/3 non-losses, not a save |
| v35 g49 ply 72 | [-1,-1,-1] | (1,9) [-1,-1,-1] | DIRTY (VCF 1) | false candidate (filter bug, below) |
| v35 g58 ply 19 | [-1,-1,-1] | (3,8) [-1,-1,-1] | CLEAN 0/0/0 | removal does not save |
| v35 g65 ply 88 | [-1,-1,-1] | (3,2) [-1,-1,-1] | DIRTY (VCF 1) | false candidate (filter bug, below) |

The pattern so far: removing all setups is NOT sufficient to save
these games — the positions were already lost (or the loss comes
from elsewhere). The known instance (v0-g4 ply 52, where the
saves DID save 3/3) looks like the exception, not the rule.

**Filter bug found by the verification audit (fixed).** The fast
`_has_vcf_start` tested only four-creating moves — but an
immediate winning move is itself a VCF start
(`starts_forced_win` returns True for it at once) and is NOT
necessarily a four-creator. After the test move (1,9) in v35 g49
ply 72, White holds the immediate win (0,12) — the square v27's
actual move blocked — and the filter wrongly recorded a clean
removal. Fixed by checking winning moves first; validated 12/12
against the slow exact enumeration (including both known-failing
boards). The bug only ADDS false candidates (a missed VCF start
opens the gate wider), never removes true ones, and the
verification's slow exact audit is the safeguard — so the
verified-instance count is unaffected. Games mined before the fix
may contain false pre-conditions; the final statistics audit
them with the fixed function.

**Fallback forensics — turn workups** (`workup_turn.py`,
**COMPLETE — 9/9 turns**): for each candidate turn, the full
workup alternative set (top-5 1-ply, 400-sim MCTS, VCF moves,
mining setup moves) is playout-tested, with a 100-sim v27 root
dump (visits / our-q / prior) at the turn. **4 turns have a
saving move (5 saves); 5 have none.**

| Turn | Saves (3/3) | Save vs setups | Root dump (actual vs save) |
|---|---|---|---|
| v35 g58 ply 19 | (5,6) [1,1,1] | partial removal (3→2 setups) | flat (max 4 visits); actual q −0.012, save q −0.112 — leaf ranks the loser above the save |
| v34-swap g46 ply 24 | (10,7), (8,9) [1,1,1] | FULL removals (1→0) — (8,9) IS a verified setup-signature instance | actual is visit leader (12) with best q (+0.054 vs +0.001/+0.018) |
| v35 g26 ply 25 | (5,8) [1,1,1] | NON-removal (1→1, same setup (6,11) remains) | flat (max 4 visits); actual q −0.066, save q −0.137 |
| v34 g55 ply 34 | none (0/6) | — (position already lost; the losing move was earlier) | CONCENTRATED: actual 63 visits, q +0.365 — the leaf is confidently wrong |
| v35 g87 ply 36 | (4,2) [0,0,0] | FULL removal (2→0) — a verified setup-signature instance | actual leads (8 visits); actual q −0.077, save q −0.132 |
| v33-swap g62 ply 40 | none | — (already lost) | CONCENTRATED: actual 61 visits, q +0.915 — the leaf is near-certain a 3/3-loser wins |
| v35 g49 ply 72 | none (0/4) | — | FORCED BLOCK (opponent's only win) — no search ran; not a leaf issue |
| v33-swap g97 ply 81 | none (0/4) | — (already lost) | actual 7 visits q −0.157; removal (1,3) 1 visit q −0.121 (better q, still loses 3/3) |
| v35 g65 ply 88 | none (0/4) | — | FORCED BLOCK — no search ran; not a leaf issue |

Shared mechanism across the 4 saving turns: **in quiet
positions where the opponent holds setups (0 VCF / 0
creators), the rollout leaf ranks the losing quiet move
above a saving quiet move** — and the 100-sim search follows
the leaf (visit leader) or cannot discriminate (flat root,
max 4 visits). This is DISTINCT from H2's confident-miss
flavour (iter-34 §3.6): there the actual leaves opponent VCF
starts (an immediate forced win the leaf cannot see); here
the actual leaves NO VCF starts — the loss comes 10+ moves
later through setup maturation the rollout policy never
plays. The saves do NOT share an exact signature: 3 of 5 are
full removals, 1 is a partial removal, 1 removes nothing —
so a pure setup-removal check fixes only the g46/g87 flavour
(and would play the 3/3-losing removal (3,8) at g58). No
save creates our own creators (all 0); the v1 score does not
separate saves from actuals (g46's losing actual has the
dominant v1, 20165 vs 8017/4410). The leaf value — not the
prior, not the root rule — is the binding constraint.

**Verified setup-signature instances: 2 from this mining**
(g46 (8,9), g87 (4,2)) **+ 1 known (v0-g4) = 3 distinct —
below the pre-registered ≥5 bar.** The setup patch as the
primary hypothesis is NOT taken; the fallback (§1.3)
triggers. Full per-turn data:
`research/iterations/turn-workups.json`.

## 3. Fast loop

### 3.0 Mechanism checks (post-implementation, pre-eval)

- **Regression tests** (`ml/tests/test_regression_v39.py`):
  **7 passed** — v39 plays a verified save at g46 ((10,7))
  and g87 ((4,2)), the full removal at g58 ((3,8)) and g26
  ((6,7)), and the forced block at g49 ((0,12)); the VCF
  non-monotonicity unit test pins that (1,9) is NOT a removal
  at g49 (leaves White the immediate win (0,12)); with
  `setup_root` off the engine is v27 exactly.
- **Full ml suite**: **411 passed, 7 skipped, 58 xfailed, 0
  failures** (the 7 skips are pre-existing checkpoint-dependent
  ones). The `choose_move` restructure breaks nothing.
- **Cost**: v39 takes 1.2–3.5s/move on the forensics positions
  vs v27's 0.9–2.2s — the check adds ~0.3–1.4s/move.

### 3.1 vs v0 (10 games, seeds 1039/2039): v39 7W-0L-3D (0.85)

No losses — but 3 draws, ALL 225-ply full-board draws, ALL
as Black (games 2, 4, 6 — 3 distinct lines, 9/10 unique
games). As White v39 wins 5/5; as Black only 2W-0L-3D. The
check's passive cost shows against a weak opponent: as
Black (the attacking side) the setup check overrides
attacking moves with removals and v39 fails to convert.
This is v3's lesson recurring — a threat-avoidance check
costs wins. The vs-v27 fast loop (the clause-1 signal) is
the deciding evidence; if v39 also struggles there, the
check's trigger needs narrowing.

### 3.2 vs v27 (10 games, seeds 1040/2040): v39 3W-4L-3D (0.45)

v39 loses the fast loop. Three more 225-ply full-board draws
(games 1, 6, 9). Colour split: as Black 1W-2L-1D; as White
2W-2L-2D. The check is the ONLY difference from v27, so the
draws and the deficit are the check's cost — but the
counterfactual is not simple (see §3.3).

### 3.3 Diagnosis + fast-loop refinement (RESOLVED)

**Instrumented replay** (`replay_with_fires.py`, all 10 vs-v0
games MATCH the recorded moves exactly): the check fires on
**32/513 v39 turns (6.2%)** — about 1 in 16 moves. The fire
rate is similar in wins (8.2–8.3%) and draws (5.3–6.2%), so
frequency alone does not explain the draws. **7 of the 32
fires (22%) override to a removal with ZERO search visits**
— a move the search never tried, chosen by the check alone.
The counterfactual at game 2 ply 16 (the earliest fire in a
draw) shows the override HELPING: the removal (10,8) wins
2/3 in playouts vs the search's move (4,4) at 1W-1L-1D — so
single overrides are not simply bad; the cost is the
accumulation of overrides diverting v39 from v27's winning
lines into quiet positions where v0 survives to a full board.

**Refinement (pre-registered here, before the re-run fast
loop): the check only overrides to a removal the search has
visited at least twice.** A 0-visit removal has no search
evidence and a 1-visit removal has a single rollout's noise;
2+ visits means PUCT returned to the move. The threshold is
set by the forensics, not tuned: the minimum visits among
the verified saves/removals is 2 (g87's (4,2), g26's (6,7)),
so ≥2 keeps every verified save while blocking the
no-evidence overrides. This changes §1.3's selection rule
(which allowed unvisited removals, ranked by prior) — the
change is fast-loop-driven, in v35's §4.2 pattern, and is
recorded here before any eval of the refined version.

### 3.4 Re-run fast loops (refined v39)

## 4. v39 evaluation

[To be written.]

## 5. Bar check (v39 vs v27)

[To be written.]

## 6. What this means

[To be written.]
