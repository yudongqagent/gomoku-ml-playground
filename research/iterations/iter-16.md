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

### 2.1 Fast-loop iteration 1 — NOT RUN YET

### 2.2 Fast-loop iteration 2 — NOT RUN YET

### 2.3 Fast-loop iteration 3 — NOT RUN YET

### 2.4 Fast-loop iteration 4 — NOT RUN YET

### 2.5 Aggregate counts (secondary)

## 3. v16 evaluation

## 4. Bar check (v16 vs v7)

## 5. What this means
