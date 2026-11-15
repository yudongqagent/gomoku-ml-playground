# Iteration 16 — v16: White 2-ply rollout leaf (batch 6, H2)

**Status: IN PROGRESS — pre-registration (§1) written before any eval;
implementation complete; fast loop not yet run.** Comparison target is
v7 (the batch-6 reference). This is the H2 workstream of batch 6 (H1
is v15, exploration of low-prior saving moves; H3 is v17, the
combination); work happens on branch `research/batch-6-h2` in the
`gomoku-b6-h2` worktree. v16 does NOT change the root decision rule
(CLOSED/EXHAUSTED), does NOT change sims/priors/book (CLOSED for
White), and does NOT include the threat check (see §1).

**Method (carried over from batch 3/4/5):** the fast 10-game loop —
play 10 games against an opponent that produces losses to learn from
(v7 is the default — losses against the reference are the ones that
matter for the bar; v0 is the fallback if an iteration produces no
losses against v7), read EVERY move of every game, find THE losing
move per game (verified with the solver or a deeper search — never
guessed), patch the algorithm so that, replayed to that critical
position, it plays the perfect move, and record each critical
position + correct move as a regression test. Then play 10 more games
and repeat. Max 4 fast-loop iterations (40 games) before validation,
or stop when an iteration finds no new patchable weakness class.
Patches must fix the STRUCTURAL cause, not the single position. The
full 4-clause validation battery (100-game matches + ladder) runs on
the FINAL patched v16 only. Fast-loop games use development seeds
(1001/2001, 1002/2002, ...) distinct from the bar battery's
pre-registered seeds (100/200), so no development game overlaps an
eval game.

## 1. Design and pre-registration

**Comparison target: v7** (primary, the batch-6 reference: ladder
Elo 190.8, on master at `bf866f7`). The bar's clause 1 is
head-to-head vs v7 (normal rules, 100 games, alternating colours,
win rate > 0.50 with Wilson 95% CI lower bound > 0.50), clause 2 is
ladder Elo strictly higher than v7's 190.8 (identical 20
games/opponent: random, heuristic, mcts, mcts_sims=100), clause 3 is
tactical pass rate >= v7's (old suite 20/20, hard suite 29/29),
clause 4 is all test suites green (ml pytest, npm test, tsc).
**Secondary head-to-heads, also pre-registered:** v16 vs v0 (100
games) and a 100-game swap match vs v7 under the tempo judge (the
harness default). Non-transitivity is recorded prominently.

**The improvement: v16 = v7's PUCT MCTS with a White-only 2-ply
rollout leaf.** Everything else is v7 unchanged: 100 sims, c_puct
1.5, v1 priors over all candidates, win/block scans before search,
most-visited root child, terminal leaf values identical. The ONLY
change is the quiet-leaf value WHEN THE ENGINE PLAYS WHITE: instead
of v7's static v0-kernel gradient, the leaf rolls out exactly 2
plies (opponent reply + our reply) under a win/block/attack-argmax
policy and evaluates the SYMMETRIC v1-kernel gradient at the
resulting position. When the engine plays Black, v16 IS v7 exactly
(same leaf function, same priors, same search) — the Black games are
a built-in control.

**The rollout, exactly (see `mcts_heuristic._rollout_leaf_value`
and `_rollout_move`):**

- The leaf first applies v7's terminal check: +1 (player to move
  has a winning move), -1 (opponent has >= 2), -0.25 (opponent has
  exactly 1). Terminal positions return the exact terminal value;
  the rollout runs ONLY from quiet positions.
- Rollout policy per ply (player P to move): a winning move wins
  (terminal +1 for P); >= 2 opponent winning moves loses (terminal
  -1 for P); exactly 1 opponent winning move is blocked (forced);
  else the argmax of P's v0-kernel attack scores over empty cells
  (ties by lowest flat index — deterministic). A terminal reached
  inside the rollout returns the exact terminal value, sign-adjusted
  to the leaf player's perspective.
- After exactly 2 plies, the value is the symmetric v1 gradient at
  the resulting position: `0.2 * tanh((breadth_v1(ours) -
  breadth_v1(theirs)) / 7.5)` (v14's form: v1 thresholds strong >=
  4000, developing [80, 4000), scale 7.5, cap +/-0.2), with a
  terminal check first (a terminal at the rollout's end returns the
  terminal value, not the gradient).
- 2 plies, not 1 and not 4. A 1-ply (opponent-reply-only) leaf
  measures robustness to the opponent's best attack, which favours
  the block — the save's value comes from OUR next move, so the
  rollout must include it. 4 plies was measured and rejected (§1.1):
  the crude attack-argmax policy blunders at that depth, returning
  false terminal losses (-1.000) for 3 of the 10 saving moves.
- The rollout policy uses the v0 kernel (cheap; the policy only
  needs to be sane); the v1 kernel does the evaluating (broken-four
  aware). The v1 kernel is v14's exact port (`kernel_v1.py`, 0
  mismatches vs frozen + live references) — used directly, NOT
  re-ported.

**Why a rollout, and what it is meant to fix.** The diagnosed White
failure (iter-13 §2.2): in 4 of 5 critical positions the search's
correctly-signed q prefers the LOSING move — a top-v1-score block or
four-maker — over the oracle-verified save, which is quiet
counterplay whose value materialises several plies out (the v1-vs-v1
oracle sees it only via deep playouts). The save's value is
beyond-horizon for a static leaf. The rollout extends the leaf's
horizon past the forcing exchange: after the save, Black's best
reply attacks and White's best reply counterattacks, and the v1
gradient at the resulting position sees White's live counterattack;
after the block, White has spent the move defending and the
post-exchange gradient sees no White attack. The rollout is a cheap
approximation of the oracle's signal — the value function White's
play has been missing — built from the v1 kernel, not a net (the
net-value road is closed by v8/v11's lessons).

**Why White-only.** The White gap is the diagnosed problem (v7 as
White vs v1: 0.52; 18 of 21 losses as White; v14 as White vs v7:
0.48). Black's game is already strong (v7 as Black vs v1: 0.89).
Confining the change to White keeps v16 minimal and isolable (one
change, one colour) and gives the battery a built-in control: the
Black split should reproduce v7-vs-v7, and any overall edge must
come from the White split.

**Threat check: NOT included.** v16 must stay a minimal, isolable
change — the ONLY difference from v7 is the White leaf value.
v13's 2-ply threat check at the root was exonerated as a cause of
v10's collapse (iter-10 §3.3) and is a durable portable gain for
defence (iter-8 §7), but iter-13 measured it as worth ~nothing
against v7 (0.5250, CI includes 0.50) and called it infrastructure,
not an improvement. Including it would add a confound without an
expected gain; it belongs in v17's combination.

**Eval seeds (pre-registered):** h2h vs v7: v16=100 / v7=200; h2h
vs v0: v16=100 / v0=200; ladder seed 100, 20 games/opponent,
mcts_sims=100; swap 100 games vs v7, balanced placement,
placement_seed 11, tempo judge (the harness default; recorded in
the JSON). Fast-loop dev seeds 1001/2001, 1002/2002, ...
(iteration k uses 1000+k / 2000+k, per iter-8 §4's policy).

### 1.1 Step 0 — cost, distribution, and the refuted candidates

**Cost and distribution** (`ml/scripts/bench_v16_leaf.py`, 346
boards replayed from batch-v7-vs-v1.json, first 20 games, every 4th
ply from ply 8; median of 3 rounds x 50 calls):

| Call | Median ms/call |
|---|---|
| v16 `_rollout_leaf_value` (2-ply + v1 gradient) | 10.13 (min 8.61, max 11.54) |
| v7 `_gradient_leaf_value` | 3.64 (min 3.54, max 4.00) |
| symmetric v1 gradient `_attack_gradient_v1` | 3.99 (min 2.78, max 5.58) |
| v6 `_terminal_leaf_value` | 2.25 (min 2.23, max 3.38) |

The rollout leaf costs ~2.8x v7's leaf. At ~99 leaf evals per move
that is ~+0.64 s per White move over v7's ~0.95 s/move (Black moves
unchanged), so a 100-game match is ~80 min under contention.
Affordable. Rollout-leaf values on the 290 quiet positions: mean
-0.150, std 0.305, p10 -0.250, median -0.070, p90 +0.052, range
[-1.000, +1.000]; 24.14% terminal-valued (|v| >= 0.25 — the 2-ply
horizon reaches forcing sequences), 75.86% in the gradient band
(|v| <= 0.2). The rollout leaf is far more discriminating than the
static gradient (std 0.305 vs ~0.1). The negative mean is expected:
the sampled positions come from real games where the side to move is
often responding; the MCTS compares sibling moves, so the spread
matters more than the offset. The 2-ply -1.0s are structurally
verified (the opponent's reply creates >= 2 winning moves — a
genuine double threat the policy found, not a policy blunder),
unlike the 4-ply false -1.0s below.

**Candidate refutations (all measured BEFORE pre-registration, on
the 10 White-to-move critical positions in the v7/v13/v14 regression
files — oracle-verified losing move + saving move each; v7's 5 +
v13's 2 + v14's 3. Scripts in /tmp: `check_v16_critical.py` (leaf
level), `check_v16_rollout.py` (rollout leaf level),
`check_v16_search.py` / `check_v16_rollout_search.py` (search
level: the actual 100-sim MCTS on each position). The v13/v14
regression files were read from `research/batch-5`; the v7 file is
on this branch. Baselines: v7 plays the save on 4/5 of its own
positions (g3 xfail) and 0/5 of the v13/v14 positions (all xfails
by construction). A mechanism must beat 0/5 on the hard positions
to show signal; regressing below 4/5 on v7's positions is a veto.**

| Candidate | Leaf level (saves preferred /10) | Search level (v7 pos / v13-v14 pos) | Verdict |
|---|---|---|---|
| v7 (control) | 3/10 | 4/5 + 0/5 | — |
| Symmetric v1 gradient (v14-style) | 4/10 | (not run — v14's battery is the evidence) | no signal beyond v14 |
| **White asymmetric gradient** (v1, theirs discounted 0.5) | 3/10 | 4/5 + 0/5 | **REFUTED** |
| White asymmetric, alpha 0.25 / 0.0 | 3/10 / 3/10 | — | same |
| **Symmetric must-block gradient term** (+/-0.1 on the -0.25 anchor) | — | 1/5 + 0/5 | **REFUTED (regresses)** |
| One-sided must-block term ((-0.1, 0]) | — | 4/5 + 0/5 | no regression, no signal |
| **2-ply rollout leaf (v16)** | 4/10 | **3/5 + 2/5** | **CHOSEN** |
| 4-ply rollout leaf | 4/10 | — | rejected (false -1.000s) |

- **Asymmetric gradient: refuted.** At the leaf level it prefers
  the save in 3/10 (same as v7), and it saturates: post-move leaf
  values cluster at +0.19-0.20 (the asymmetric diff's positive mean
  4.19, std 4.69 — measured on the same 346-board sample, scale
  1.8xstd = 8.44 — pushes tanh into its flat region after White's
  move). At the search level it plays 0/5 v13/v14 saves (same as
  v7) and plays the known loser on v13-g7-ply25 where v7 plays
  something else. The block raises White's breadth too, so no
  breadth re-weighting separates the block from the counterplay —
  iter-7 §2.1's one-sided-gradient refutation, reproduced on the v1
  kernel.
- **Symmetric must-block term: refuted (regresses).** Adding a
  +/-0.1 v1-gradient term to the -0.25 anchor (iter-7 §5(d)'s
  candidate) softens the -0.25 penalties on the blocking move's
  lines (the block raises White's breadth, so the penalty becomes
  -0.15) — the same myopia the cap fixed — and the search regressed
  from 4/5 to 1/5 saves on v7's own positions. The one-sided
  variant ((-0.1, 0] — breadth can only worsen the anchor, never
  soften it) restores 4/5 + 0/5: no regression, but no signal
  either.
- **4-ply rollout: rejected.** At the leaf level it prefers the
  save in 4/10, but returns false terminal losses (-1.000) for 3
  of the 10 saving moves (v7-g1, v14-g1, v14-g3) — the crude
  attack-argmax policy blunders at that depth, losing games the
  oracle says the save draws/wins. The policy is too weak to be
  trusted at 4 plies.
- **2-ply rollout: chosen.** At the leaf level it prefers the save
  in 4/10, including finding a forced win (+1.000) from
  v13-g9-ply15's save within 2 plies. At the search level it plays
  2/5 v13/v14 saves (v13-g9-ply15, v14-g3-ply13) — the ONLY
  candidate to flip any hard position — at the cost of 1 v7
  position (g5: plays [9,8], a non-save non-loser, instead of a
  save). Net 5/10 saves vs v7's 4/10. The diagnostic is thin (n=10,
  and iter-13's lesson stands: diagnostics do not predict match
  strength) — but it is the only candidate with ANY positive
  search-level signal, and its rationale (the save's value lives
  past the forcing exchange) is the one the diagnosis points to.
  The battery decides.

**Pre-registration note.** The three refuted candidates were
measured during design, BEFORE this section was written and before
any eval (no match has been run). Reconsidering the mechanism on
design measurements is not moving the goalposts: the bar, the
comparison target, the seeds, and the battery were fixed before the
first candidate was measured, and no eval result informed the
choice. The fast loop (§2) may still patch v16 on new weaknesses it
finds; any patch is recorded there, and the battery runs on the
final version only.

### 1.2 Battery (the verdict — pre-registered seeds and commands)

On the FINAL v16 only (after the fast loop):

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

**Integration.** `ml/gomoku_ml/mcts_heuristic.py` gains the
rollout leaf (`_rollout_leaf_value`, `_rollout_move`,
`_attack_gradient_v1`, `_breadth_v1` ported from v14) and the
`rollout2_v1` leaf mode; `ml/gomoku_ml/scoring.py` gains ONLY
the `ENGINE_VERSIONS[16]` registry entry. `kernel_v1.py` (+ its
tests and fixtures) is checked out unchanged from
`research/batch-5`. `baseline.py` is frozen and untouched.

## 2. Loss analysis for iteration 16

The fast-loop worked games (§2.1-2.4) ARE the loss analysis.
Aggregate counts (§2.5) are secondary.

### 2.1 Fast-loop iteration 1 — 10 games vs v7 (pre-patch engine)

**Batch:** `research/iterations/fastloop1-v16-vs-v7.json`
(seeds v16=1001 / v7=2001 — development seeds, distinct from the bar
battery's 100/200). **5W-4L-1D, win rate 0.5500, CI [0.2737,
0.7986]**, 580.5 s. Split: as Black 1W-3L-1D (0.30), as White
4W-1L-0D (0.80) — v16-as-White beats v7-as-Black 4/5 in the first
live test of the rollout leaf, while v16-as-Black (which IS v7)
loses the mirror match 1W-3L-1D on seed differences. (The first run
of this batch was killed by a runtime restart at 4/10 games with a
flushed partial JSON; the re-run with the same seeds reproduced the
4-game prefix EXACTLY — determinism verified.) Classifier
(`ml/scripts/classify_batch_losses.py`): 4/4 losses via open four,
0 missed blocks, 0 clean missed preventions in 14 threatened turns
(in losses). Defence is still perfect; the losses are lost races.

**The losing move in each game** (walk-back,
`ml/scripts/walkback_loss.py`, corrected oracle method — the actual
move's v1-vs-v1 playout must lose at all 3 seeds before an
alternative counts; seeds 0,1,2; JSONs:
`research/iterations/walkback-fl1-g{0,2,6,7}.json`):

| Game | Last preventable turn | v16 played (loses 3/3) | Perfect move (oracle) |
|---|---|---|---|
| 0 (Black, 18 plies) | ply 12 | (11,9) | (9,7) / (5,5) / (7,8) |
| 2 (Black, 78 plies) | none | — (no alternative saves at any turn) | — |
| 6 (Black, 90 plies) | ply 78 | (9,14) | (5,7) / (9,13) / (1,9) / (6,6) |
| 7 (White, 105 plies) | ply 89 | (8,9) | (12,0) / (12,1) / (11,12) / (1,9) |

Games 0 and 6 are Black losses — v16-as-Black IS v7, so these are
v7's losing moves, not v16's; fixing them would mean changing the
Black leaf, which is out of scope (the Black games are the built-in
control). They are recorded as regression xfails (§2.1, below), not
patched. Game 2 has no losing move (distributed drift, the iter-6
result). **Game 7 is the iteration's patchable finding — the only
White loss, and the rollout leaf's first live failure.**

**Game 7, ply 89 (White to move, quiet, 0 creators) — the rollout
leaf reverses v7's correct choice.** The saving move (12,0) is v1
rank 1 (score 20004, FORCING — creates a four), prior 0.073; the
played (8,9) is v1 rank 5 (score 4091), prior 0.033, non-forcing.
Critical analysis (`analyse_critical.py` + the rollout leaf mode):
v7's gradient MCTS plays the save (7 visits vs 2 for the loser);
v6's terminal MCTS also plays it (13 vs 5); v16's rollout MCTS
plays the loser (20 visits vs 5), and v16's correctly-signed q
prefers the loser too (our-q +0.111 vs -0.095). Mechanism, verified
by direct computation: the rollout leaf values ALL THREE of (8,9),
(12,0) and (1,9) at +0.25 for White — after the quiet (8,9), the
rollout plays Black (13,4) then White (12,1) (the forcing move) and
returns +0.25, as if White had played it immediately. **The quiet
move borrows the forcing move's value**, because the rollout's
opponent (v0 attack-argmax) never prevents White's forcing sequence.
v7's static leaf, by contrast, scores (8,9) at -0.181 vs (12,0) at
+0.25 — the gap the rollout erases.

**Patch part 1 — v1 rollout policy (structural).** Black's v1-best
reply after (8,9) is the prevention (12,0) (v1 score 20004) or
(12,1) (20000); the rollout's (13,4) scores 81 in v1 terms. v0's
contiguous-run scoring is blind to the broken four, so BOTH v0
policies (attack-argmax, and score_all — measured, same sequence)
miss the prevention. The rollout policy is now v1's 1-ply best move
for both sides: argmax over empty cells of (own v1 attack +
opponent's v1 attack) from one `attack_v1_both` call (v1's total
score up to the double-threat bonus, which the defence term omits).
Verified: after (8,9), Black's rollout reply is now the prevention
(12,0) and the leaf drops from +0.25 to -0.16. But the patch alone
did NOT fix the position: patched v16 played (11,9) — oracle-tested
at losing 3/3, like (8,9).

**Patch part 2 — gradient-evaluated rollout end (structural).**
After (11,9), the rollout plays Black (11,11) then White (12,0)
(the forcing move) and returns +0.25 — again borrowing the forcing
move's value, this time via a rollout-found four 2 plies out. The
flaw: a rollout-found must-block (-0.25) was treated as an exact
terminal value, but it is a PREDICTION under the 1-ply rollout
policy, not a fact about the position — while (12,0)'s +0.25 is a
fact (White HAS a four on the board). The fix: at the rollout's
end, only wins/losses (+/-1, game over) are terminal; a must-block
is a position to evaluate with the gradient. The must-block at the
START (the real position) keeps v7's exact -0.25. Verified: patched
v16 plays **(12,0)** on g7-ply89 — the save — and the mechanism
checks hold (search-level: v7 positions 3/5, v13/v14 positions 2/5,
unchanged from the unpatched rollout; the two hard flips,
v13-g9-ply15 and v14-g3-ply13, still flip).

**Cost of the patch** (bench re-run): rollout leaf 14.01 ms/call
(was 10.13 — the v1 policy adds ~4 ms: three v1-kernel calls per
leaf). White moves ~1.9 s; a 100-game match ~90-100 min. Affordable.
Rollout-leaf values on quiet positions: mean -0.067, std 0.191
(was -0.150 / 0.305 — the gradient-evaluated end compresses the
rollout-terminal mass, as designed).

**Regression tests:** `ml/gomoku_ml/regression_positions_v16.json`
gains 3 positions (5 total): fl1-g7-ply89 (PASSING — the patch
plays (12,0)) and fl1-g0-ply12 / fl1-g6-ply78 (documented xfails —
Black positions, v16-as-Black is v7). Targeted suites after the
patch: test_regression_v16 + test_mcts_heuristic 35 passed, 3
xfailed (2 new Black xfails + the known v7 iter-5 xfail).

### 2.2 Fast-loop iteration 2 — 10 games vs v7 (patched engine)

**Batch:** `research/iterations/fastloop2-v16-vs-v7.json`
(seeds v16=1002 / v7=2002). **6W-4L-0D, win rate 0.6000, CI
[0.3127, 0.8318]**, 450.0 s. Split: as Black 3W-2L (0.60), as White
3W-2L (0.60) — balanced, and the White split holds at 0.60 after
the iteration-1 patch (4W-1L in iteration 1 was partly seed luck).
Classifier: 4/4 losses via open four, 0 missed blocks, 0 missed
preventions in 18 threatened turns (losses). Defence still solved.

**Walk-backs** (`research/iterations/walkback-fl2-g{5,6,8,9}.json`):

| Game | Last preventable turn | v16 played (loses 3/3) | Perfect move (oracle) |
|---|---|---|---|
| 5 (White, 45 plies) | none | — (ply 41/43 lose 3/3, no save; ply ≤39 wins 3/3) | — |
| 6 (Black, 38 plies) | ply 30 | (12,6) | 1 save |
| 8 (Black, 72 plies) | ply 64 | (12,12) | 2 saves (ply 56: 4 saves) |
| 9 (White, 49 plies) | none | — (ply 47 loses 3/3, no save; ply ≤45 wins 3/3) | — |

Games 6 and 8 are Black losses (v7's moves, out of scope). The two
White losses share a pattern: the oracle says White was WINNING
3/3 as late as 2 plies before the end, then Black's single move
made it unwinnable with no saving reply. The v1 oracle is too weak
to see Black's killer move — so the walk-back finds no losing move
for White in either game. Both were investigated directly instead.

**Game 9, ply 45 — the rollout policy picks a losing move over a
winning move (PATCHED).** After White's (5,5), Black has SEVEN
four-creating moves. The rollout leaf valued the position at
-1.000 for Black — a hallucinated White win — because the
rollout-policy Black reply was (6,8), which lets White's (10,11)
create a double four. (6,8) won the v1-argmax (total 200422: own
401 + opp 200021, a block of one White four) over Black's (6,3)
(total 200090) by 332 points — but (6,3) creates an OPEN four
(own v1 attack 200080), leaving Black TWO winning moves ((8,1)
and (3,6)), a forced win; the real Black (v7) played (6,3) and won.
Root cause: the argmax of (own + opp) v1 attack conflates attack
and defence in one sum, so a defensive block can beat a forced win.
**Patch:** in `_rollout_move`, after the forced block (an opponent
four on the board must be answered first — otherwise the opponent
wins immediately) and before the argmax: if the player's best own
v1 attack is >= OPEN_FOUR (200,000), play the argmax of own attack
alone. The threshold is exact — no non-forcing combination reaches
200,000 (the next-largest bonus is DOUBLE_OPEN_THREE_BONUS =
50,000), and an open four (or double closed four) leaves >= 2
winning moves. Verified: the patched policy plays (6,3), the leaf
after (5,5) is +1.000 for Black (correct), and the mechanism
checks hold (3/5 v7 + 2/5 hard). Leaf-level regression test:
`test_rollout_move_prefers_open_four_over_defensive_block`
(game-9 board embedded). Note the patch does NOT change the move
at ply 45 itself: patched v16 still plays (5,5) — correctly, since
all 8 top White alternatives are -1.000 (the position was already
lost; (5,5) blocks Black's immediate five, v1 total 10,004,022)
and v7 plays (5,5) there too. The patch's value is upstream: the
corrected leaf steers White's search away from such positions
earlier. (Side observation: the v1 oracle itself shares this flaw —
v1-as-Black would also play (6,8) over (6,3) — which is why the
walk-back oracle called (5,5) winning.)

**Game 5, ply 39 — double-creator beyond the rollout horizon
(OPEN, not patched).** After White's (7,3), Black has TWO creators
((8,3) and (8,4)). The rollout plays Black (8,3), White blocks
(8,4) (which also occupies the second creator square), and the
gradient says White is fine (+0.136) — but the real Black plays
(8,4), White blocks (8,3), and Black's follow-up (7,4) wins 2
plies later. Two defects compound: (i) the rollout policy picks
(8,3) over (8,4) among Black's creators (both score ~20,000; the
difference is 2 plies out, invisible to any 1-ply policy), and
(ii) the 2-ply rollout cannot see Black's (7,4) follow-up. The
iteration-2 patch does not touch this position (leaf still -0.136,
reply still (8,3) — verified). A fix would need either a deeper
rollout (4-ply already refuted: false terminals) or a static
double-creator term, and verifying a save needs a stronger oracle
than v1 (v7-vs-v7 playouts) since the v1 oracle calls (7,3)
winning. Recorded as the open weakness class to watch in
iteration 3; if it recurs, it is the next patch candidate.

### 2.3 Fast-loop iteration 3 — NOT RUN YET

### 2.4 Fast-loop iteration 4 — NOT RUN YET

### 2.5 Aggregate counts (secondary)

## 3. v16 evaluation

## 4. Bar check (v16 vs v7)

## 5. What this means
