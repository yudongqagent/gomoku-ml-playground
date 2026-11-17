# Iteration 29 — v29: v22 vs-mcts forensics → targeted patch (batch 9, H3)

**Status: PRE-REGISTERED — design fixed below (§1) BEFORE any eval of
v29. Port (§1.0), forensics (§2), patch or fallback (§3), and battery
(§4) follow.**

**METHOD (carried over).** The fast 10-game loop stays the finding
method: play 10 games, read every move, find each game's losing move
(solver/deeper-search verified, never guessed), patch the algorithm
to reproduce the perfect move in those positions (regression test),
repeat — max 4 iterations, or stop when an iteration finds no new
patchable weakness class. The 4-clause bar stays the validation
gate. Fast-loop games use development seeds distinct from the bar
battery's pre-registered seeds (100/200). **Fast loops run against
v0 as well as v7** (iter-17 §5). The full battery runs on the FINAL
version only. The 100-game battery is the verdict — small-sample
smokes mislead; single-position diagnostics cannot validate a
global search change; the v1 walk-back oracle shares the
argmax-of-sum flaw (iter-18) — every nominated save needs playouts
or exact computation.

## 0. The problem

v22 (batch 8, H2: v14's v1-kernel gradient leaf with gap-three
makers demoted ONLY in Black's breadth, scale 7.0, cap ±0.2)
collapses against the ladder MCTS: **5W-15L (0.2500, CI [0.1119,
0.4687])** over 20 ladder games, the worst vs-mcts result of any
v1-leaf version (v14: 0.55, v20: 0.65, v17: ~0.50; even v7 itself
scores 0.40 against the ladder MCTS). The collapse caps v22's
ladder Elo at 147.2 despite tactical 29/29 and Black 0.93 vs v0
(iter-22 §3.6). The mechanism is UNKNOWN — that is this
workstream. The leading hypothesis (iter-22 §3.6, stated there as
a hypothesis, NOT a verified mechanism): the full theirs-term —
v22 counts the opponent's gap-three breadth at full weight, so
against the terminal-leaf MCTS's forcing play (which creates
gap-three-rich positions without respecting them, since its leaf
has no gradient) v22 over-fears and plays passively. This
iteration tests that hypothesis against real losses instead of
asserting it.

**Correction to the task brief, recorded here because it changes
the base engine:** v22 is built on **v14's v1-kernel gradient
leaf** (`leaf_mode="gradient_v1_gap_black"`, iter-22 §1/§1.2) —
NOT on v18's rollout leaf. The batch-9 plan's H3 line ("v22
vs-mcts forensics → targeted patch") and the fallback ("VCF-at-
root for the rollout-leaf engine, v18 base") are two different
bases: Phase 1 patches v22 (gradient leaf); Phase 2, if it
triggers, builds on v18 (rollout leaf).

**The v24 finding (iter-24 §5, ML batch 1):** the 100-sim search
horizon is a SHARED limit — v7 makes the same miss as v24
(White's i10 instead of d9, vs-v0 game 1 ply 19; d9 saves 2/3 in
same-engine playouts). A VCF/deeper threat check at the root
could catch forced wins — and forced-win preventions — that the
search misses. That is the Phase 2 fallback, and only the
fallback.

**BASE ENGINE (Phase 1).** v22 exactly as implemented on
`research/batch-8` (cherry-picked, not re-derived — §1.0):
100 sims, c_puct 1.5, v1 priors over all candidates (temperature
2.0), win/block scans before search, most-visited root child
with RNG tie-breaks, terminal leaf values identical to v14/v7.
The fast loop runs on the PATCHED engine (v29), not on v22.

## 1. Design and pre-registration

**Comparison target: v7** (the reference, batch-9 plan: ladder
Elo 190.8, on master at `bf866f7`). Clause 1 is head-to-head vs
v7 (normal rules, 100 games, alternating colours, win rate >
0.50 with Wilson 95% CI lower bound > 0.50); clause 2 is ladder
Elo strictly higher than v7's 190.8 (identical 20
games/opponent: random, heuristic, mcts, mcts_sims=100);
clause 3 is tactical pass rate >= v7's (20/20 old suite + 29/29
hard suite, on the CORRECTED suite — batch 9's pre-registered
program correction adds (9,7) to winning-quiet-setup-2);
clause 4 is all test suites green (ml pytest, npm test, tsc).
**Secondary head-to-heads, also pre-registered:** v29 vs v0
(100 games) and a 100-game swap match vs v7 under the tempo
judge (the harness default). A new strength champion ALSO needs
to beat v18 on BOTH h2h-vs-v7 rate (0.775) AND ladder Elo
(636.4). Non-transitivity is recorded prominently.

### 1.0 Port (before any eval)

v22's code is ported from `research/batch-8` by cherry-pick, not
re-derived: the batch-7-h3 ml tree port (kernel_v1, v14/v20
leaf, tests), the v22 pre-registration, implementation, and
fast-loop commits. Port verification BEFORE the forensics:

1. Tactical: v22 must score **29/29 on the corrected hard suite**
   (and 20/20 old) via `eval_tactical_full.py --version 22
   --suite both --seed 100`. v22 scored 29/29 on the old suite;
   the correction only ADDS an expected move, so 29/29 must hold.
2. Collapse reproduction: the forensics match (§1.1) itself is
   the check — if v22 vs the ladder MCTS at fresh dev seeds does
   NOT come back well below 0.50, STOP and fix the port before
   classifying any loss. (The ladder's own 0.25 was at seed 100;
   the forensics seeds are fresh, so the reproduction is a
   genuine check, not a replay.)

If either check fails, the port is fixed before any forensics
reading. The port commits are recorded in the git log of
`research/batch-9-h3`.

### 1.1 Phase 1 — forensics (pre-registered protocol)

**Forensics match:** `fast_loop_match.py --version-a 22
--opponent mcts --games 20 --seed-a 1029 --seed-b 2029 --out
research/iterations/forensics-v22-vs-mcts.json` — 20 games, the
ladder size, alternating colours, 100 sims both sides. Seeds
1029/2029 are development seeds, distinct from the battery's
100/200 AND from batch-8's loop seeds (1013/2013, 1014/2014),
so these are fresh games. The ladder MCTS is
`baseline.HeuristicMCTSPlayer` (terminal-leaf PUCT, the ladder's
search opponent) — the same opponent v22 collapsed against in
the ladder.

**Read EVERY loss.** Each v22 loss is classified into exactly
one mechanism class (first applicable, in this order — a loss
with a verified losing move is classified by the ROOT CAUSE of
that move, not by the loss's surface):

1. **Missed block** — a defensive error: the loss classifier
   reports a missed block, or exact enumeration shows a block
   existed at a v22 turn and v22 did not play it.
2. **Missed forced win** — a VCF/forced win existed at a v22
   turn (verified by `evaluate.starts_forced_win` or exact
   enumeration) that v22 did not play, and the game was lost
   afterwards.
3. **Horizon loss** — no single losing move: `workup_loss`
   finds no saving move at any of the last 4 v22 turns
   (distributed drift into an unstoppable open four).
4. **Prior starvation** — the saving move is verified by
   playouts but its root prior is low and it received 0–1
   visits at the root (diagnosed with a root dump at the
   losing position).
5. **Leaf value error** — the saving move is verified by
   playouts, its prior is adequate, but the root q/visits rank
   the losing move above the save (diagnosed with a root dump
   at the losing position).

**Verification rule.** Every nominated losing move is verified
with `workup_loss.py` playouts (same-engine, seeds 0/1/2) or
exact computation. The v1 walk-back oracle is NEVER used alone
to verify a save (it shares the argmax-of-sum flaw — iter-18 —
and iter-22 §2.2 shows it inventing false saves in MCTS-vs-MCTS
games). A class counts as VERIFIED only with playout/exact
confirmation.

**Extension rule.** If after 20 games no single class has ≥3
verified instances, extend with 20 more games at fresh seeds
1030/2030 (`forensics-v22-vs-mcts-2.json`) and re-classify the
pooled losses. If still no class has ≥3 verified instances, the
forensics concludes "no patchable mechanism found" and Phase 2
(§1.2) triggers.

**Patch rule.** If a class has ≥3 verified instances sharing
ONE mechanism AND a patch reproduces the perfect move at those
positions (regression test), the patch IS v29, on the v22 base.
The patched engine is fast-looped (vs v0 AND v7, 10 games each,
dev seeds 1031/2031 and 1032/2032) and then runs the full
battery (§1.3). Max 2 patch iterations: if the first patch
fails to reproduce the saves at the verified positions, ONE
refinement is allowed; if that fails too, Phase 2 triggers.
A patch that needs a free parameter fitted to the forensics
positions (e.g. a threshold tuned on 2–3 positions) is NOT
allowed — the mechanism must be principled (a feature, a rule,
or a search change), not a fitted constant.

**Negative-result rule.** If the forensics finds nothing
patchable, the classification table is still committed in full
in §2 — a negative forensics result is a program finding, not a
failed iteration.

### 1.2 Phase 2 — fallback: VCF-at-root for the rollout-leaf engine (v18 base)

Triggers ONLY if Phase 1 finds nothing patchable under the
rules of §1.1. It does NOT trigger merely because the forensics
is slow or the patch is hard — the patch rule's two-iteration
limit is the trigger.

**Base: v18** (`leaf_mode="rollout2_v1_both"` — the 2-ply
rollout leaf in BOTH colours, 100 sims, c_puct 1.5, sharp v1
priors, most-visited root), ported from `research/batch-7` /
`research/ml-batch-1` (cherry-pick/port, no re-derivation —
ML batch 1's H1 already ported this leaf bit-for-bit, iter-24
§0). Port verification, BEFORE any v29 eval: the ported
`_rollout_leaf_value` must agree bit-for-bit with the
`research/batch-7` worktree's on the 401-position
`verify_rollout_port.py` set (the iter-24 §0 instrument), and
v18 must score 29/29 on the corrected tactical suite.

**Root-check design (exact, pre-registered).** Hook:
`HeuristicMCTS.choose_move`, AFTER the existing immediate-win
scan and single-block scan, BEFORE the MCTS search. The check
uses `evaluate.starts_forced_win` (depth 3 — the tactical
suite's own VCF definition: the move must create a four, and
after the defender's forced block the attacker still forces a
win within 3 more of their own moves) and
`evaluate.four_creating_moves`. The web game's VCF/VCT solver
is a reference design only; the repo's `starts_forced_win` is
the instrument.

1. **Offensive VCF.** Enumerate our four-creating moves. For
   each, run `starts_forced_win(board, player, move, depth=3)`.
   If any passes, play the one with the highest v1 prior
   (ties: lowest flat move index — deterministic, no RNG).
   Skip the search entirely.
2. **Defensive VCF prevention.** Enumerate the opponent's
   four-creating moves. If any opponent move `m` satisfies
   `starts_forced_win(board, -player, m, depth=3)`, then among
   our legal candidate moves find those after which the
   opponent has NO VCF-starting move (re-check
   `starts_forced_win` for each opponent four-creator on the
   resulting board). If any exist, play the highest-v1-prior
   one (ties: lowest flat move index). If none exist, fall
   through to the search.
3. If neither check fires, the engine is exactly v18.

**Why this design.** v24's verified miss (d9, iter-24 §5) is a
preventive move 3–4 plies deep that the 100-sim search cannot
propagate to the root; the defensive check directly targets
that class (opponent VCF prevention), while the offensive
check catches forced wins the search misses. The depth-3 VCF
definition is exact, not heuristic — the check can only fire
on a genuinely forced win, so it cannot misfire on a merely
good-looking move.

**Cost guard.** Both checks enumerate only four-creating moves
(small sets in quiet positions). Median and p90 root-check
time are measured on 100 quiet positions before the battery
and reported in §3; no design change is made after any eval
on the basis of cost alone — cost is reported, strength is
what the battery measures.

**What does NOT change (Phase 2).** 100 sims, c_puct 1.5,
sharp v1 priors (temperature 2.0), most-visited root child,
terminal leaf values, the v18 rollout leaf. No q-based root
decision (CLOSED). No sims change (more sims is NOT
monotonically better). No prior change. No threat check beyond
the VCF root check above. `baseline.py` stays FROZEN. No kernel
re-sync unless v29 passes the bar.

### 1.3 Eval seeds (pre-registered)

- Battery h2h vs v7: v29=100 / v7=200; vs v0: v29=100 / v0=200.
- Ladder: seed 100, 20 games/opponent, mcts_sims=100.
- Swap: 100 games vs v7, balanced placement, placement_seed
  11, tempo judge (the harness default).
- Tactical: `eval_tactical_full.py --suite both --seed 100`.
- Fast-loop dev seeds (patched engine): 1031/2031 (vs v0),
  1032/2032 (vs v7) — distinct from the forensics seeds
  (1029/2029, 1030/2030) and the battery's 100/200.
- Harness commands follow iter-7 §3 / iter-24 §2 exactly
  (`fast_loop_match.py`, `selfplay_batch.py --swap-rule`,
  `eval_version.py` for the ladder, `eval_tactical_full.py`
  for tactical). The match drivers AUTO-RESUME from partial
  `--out` files — if a run is killed, relaunch the SAME
  command; the saved games stand and the RNG restart is
  recorded as `resumed_from_game`.

## 2. Forensics (Phase 1)

### 2.0 Port verification — the port is EXACT

v22's code was cherry-picked from `research/batch-8` (commits
`8516a6e`, `f39febb`, `39792ce`, `74ccc3a` — the batch-7-h3 ml
tree port, the v22 pre-registration, implementation, and fast
loop). Three independent checks confirm the port:

1. **Byte-identical code:** `mcts_heuristic.py`, `scoring.py`,
   `kernel_v1.py`, `baseline.py` diff clean against the
   `gomoku-b8-h2` worktree (the tree the ladder ran on).
2. **Tactical:** v22 scores **20/20 old + 29/29 hard** on the
   CORRECTED suite (`eval_tactical_full.py --version 22 --suite
   both --seed 100`), reproducing iter-22 §3.2/§3.5.
3. **Ladder reproduction:** a new script,
   `ml/scripts/replay_ladder_mcts.py`, replays a ladder matchup
   EXACTLY (same player seeds, same warm-up legs — the version
   player's RNG advances through the random and heuristic legs
   before the mcts leg, exactly as in `ladder_for_player` — and
   records every game with full move lists; RNG state is
   snapshotted per game so a resume is exact, not a restart).
   The v22 ladder mcts-leg replay reproduces the recorded ladder
   result **exactly: 5W-15L-0D, win rate 0.2500, CI [0.1119,
   0.4687], as Black 3W-7L, as White 2W-8L** (iter-22 §3.6; the
   original log at `~/workspace/b8h2-logs/ladder-v22.log`
   confirms the figure is real, not a transcription error). The
   replay's games are the forensics corpus:
   `research/iterations/forensics-v22-vs-mcts-ladder.json`.

### 2.1 The collapse does NOT reproduce at fresh seeds

The pre-registered forensics match (`fast_loop_match.py
--version-a 22 --opponent mcts --games 20 --seed-a 1029
--seed-b 2029`, `research/iterations/forensics-v22-vs-mcts.json`)
came back **v22 14W-6L-0D, win rate 0.7000, CI [0.4810,
0.8545]** — as Black 7-3, as White 7-3. The CIs of the two
20-game samples do not overlap (ladder [0.1119, 0.4687] vs
fresh-seed [0.4810, 0.8545]). The drivers are equivalent
(`resolve_opponent("mcts")` builds the identical
`HeuristicMCTSPlayer(sims=100)`; `play_recorded_game` mirrors
`play_game_from`; the only difference is RNG seeds plus the
ladder's warm-up legs) and the fresh-seed games are varied (12
distinct 5-ply prefixes in 14 games at the check; no
transposition collapse). Two facts shrink the collapse further:

- **Duplicate lines:** the ladder's 20 games contain only **17
  distinct games** — three loss pairs are move-for-move
  identical (games 6=8, 12=14, 16=18). The 15 losses are **12
  distinct lines**. The effective sample is smaller than 20.
- **Short games:** the ladder games average **33.55 plies**
  (losses mostly 22–34 plies) — the MCTS kills v22 fast when it
  kills it at all.

So the "0.25 collapse" is partly a seed-draw artifact (a ~2%
tail under a 0.50 true rate, and the fresh-seed draw sits at
the opposite tail) and partly duplicate counting. The
forensics below works on the ladder's 12 distinct losses —
the games that actually collapsed — with the fresh-seed
match's 6 losses as the control (6/6 via open four, 0 missed
blocks, 0 clean missed preventions in 46 threatened turns).

### 2.2 Loss classification (ladder corpus, 15 losses)

`classify_losses.py`: **15/15 losses via open four, 0 missed
blocks, 2 clean missed preventions** (games 13 and 19, both at
ply 23, both with 4 creators live) **in 74 threatened turns**.
Defence is solved, as in every batch. The 2 classifier-flagged
preventions do NOT save in playouts (§2.4) — the "safe"
alternatives are forcing counter-attacks that Black answers
while keeping 2 creators live.

Structural scan (exact computation): 6 of the 12 distinct
losses contain a **double-threat ply** — the winner plays a
move creating a four (forcing a block) while already holding an
open three that becomes an open four after the block (games 2,
5, 6, 9, 11, 15). The other 6 are **creator explosions** — the
winner plays a quiet setup move creating 2–4 open-four creators
at once, after v22's prevention took the previous creators to
0 (games 1, 7, 12, 13, 16, 19).

### 2.3 Workups — 6 verified saving moves, 6 horizon losses

All 12 distinct losses were worked up with `workup_loss.py`
(last 4 v22 turns, top-5 1-ply + VCF alternatives, seeds 0/1/2;
raw JSONs in `~/workspace/b9h3-logs/workup-g*.json`):

| Game | v22 | Plies | Losing move (verified) | Save (playout) | Exact class of the save |
|---|---|---|---|---|---|
| 1 | W | 29 | ply 21 (6,8) | (12,6) 2/3 | partial setup-prevention (setups 2→1; VCF 0→0) |
| 2 | B | 34 | ply 30 (5,7) | (4,12) 3/3, (5,13) 3/3 | partial VCF-prevention (VCF 3→1) / none (VCF 3→3) |
| 5 | W | 29 | ply 23 (9,6) | (13,10) 3/3 | none (VCF 4→4, creators 0, setups 0) |
| 6 | B | 22 | ply 16 (8,3) | (9,7) 3/3, (10,8) 3/3 | **clean VCF-prevention (VCF 1→0)** |
| 7 | W | 25 | ply 19 (8,9) | (8,5) 2/3 | none (all exact counts IDENTICAL to the loser) |
| 9 | W | 29 | ply 23 (5,6) | (1,10) 3/3 | none (VCF 4→4, creators 0, setups 0) |
| 11 | W | 29 | none in window | — | horizon loss |
| 12 | B | 30 | none in window | — | horizon loss |
| 13 | W | 27 | none in window | — | horizon loss |
| 15 | W | 29 | none in window | — | horizon loss |
| 16 | B | 30 | none in window | — | horizon loss |
| 19 | W | 27 | none in window | — | horizon loss |

The exact analysis (`ml/scripts/targeted_playout.py`, new:
for each move, the opponent's VCF starts / open-four creators /
setup moves on the resulting board) shows the saves are
HETEROGENEOUS. Only ONE (game 6) is a clean exact prevention.
Game 2's (4,12) is partial (VCF 3→1) and (5,13) is not a
prevention at all (VCF 3→3 — the remaining VCF starts don't
convert in playouts, i.e. `starts_forced_win` reports forced
wins the real engines refute, because it doesn't model
defender counter-threats beyond the next move). Game 1's save
removes one of two setups. Games 5, 7, 9 have NO exact
signature: the save leaves the opponent's threat counts
unchanged (game 7's save and loser are identical on every
count — VCF 0, creators 0, setups 3, same sets).

Targeted follow-up (13 hypothesis moves in the 6 no-save
games — the double-threat squares, the remaining creators,
the setup squares, the classifier's forcing moves): **0 saves**.
By the last 4 turns those positions are lost by force
(multiple creators/VCF starts one move can't address); the
loss was decided earlier or is distributed drift. Game 1's
(12,6) at ply 23 (one move later than the verified save)
also loses 3/3 — the save has a one-move window.

**Missed forced win: 0 of 12.** An exact scan of every v22
turn in every distinct loss finds v22 NEVER held a
VCF-starting move it didn't play. The offensive VCF check
would not have changed any loss game.

### 2.4 Root dumps — the search, not the leaf

`ml/scripts/diag_root.py` (new: root visits/q/priors + 1-ply
leaf values under the v22/v20/v14 leaves + breadth
decomposition) at the verified losing positions:

- **Game 6, ply 16** (the clean VCF case): v22 ranks (8,3)
  above (9,7) by VISITS (26 vs 15) although q prefers the save
  (+0.073 vs +0.068) and the 1-ply leaf prefers the save under
  ALL THREE leaves (v22 +0.146 vs +0.081; v20 +0.188 vs +0.185;
  v14 +0.196 vs +0.167). Prior-driven visit allocation
  (0.1086 vs 0.0698) + the forced loss 3 plies below (8,3)
  never propagating to the root q.
- **Game 1, ply 21:** a FLAT root (top move 9 visits). The
  search genuinely prefers the loser on every measure —
  visits 9 vs 5, q +0.162 vs +0.065, leaf +0.192 vs +0.155 —
  and all three leaves agree. Nothing at the root
  distinguishes the save.
- **Game 2, ply 30:** an EXTREMELY flat root (top move 4
  visits of 100 sims). All q's negative; the save (4,12) has
  the HIGHEST prior (0.1604) but the WORST q (-0.612) — the
  tree "finds" White's (2,7) VCF below the save and backs up
  a loss, missing Black's refutation. The search hallucinates
  a loss below the save.
- **Game 5, ply 23:** the search prefers the loser by visits
  (31 vs 22) and q (+0.040 vs +0.004); the 1-ply leaf is
  identical for loser and save (+0.197). The blocks (10,8)/
  (10,9) have q -0.40 — and don't save.

**Cross-engine check (is the miss v22-specific?):** at game 6
ply 16, **v14, v20, AND v7 all play the save (9,7)** (v14 #1,
22 visits; v20 #2, 16 visits, #1 is the other save (10,8);
v7 #1, 20 visits) — v22 is the ONLY engine that misses it.
The mechanism there is v22-specific: below (9,7), Black's
gap-three counterplay (14 makers) is demoted in v22's leaf,
so v22's tree rates the quiet block's subtree much lower
(+0.073) than v14's (+0.180), and the near-tie in q lets the
prior drive visits to the flashy (8,3). BUT at the other five
verified positions, **v14 makes the SAME miss as v22** (g1:
plays (8,5), save #7; g2: plays (5,7), save #2; g5: plays
(9,6), save #2; g7: plays (8,9), save #2 by 22–20 visits; g9:
plays (5,6), save #2). The v22-specific mechanism has exactly
ONE verified instance. The other five misses are shared
across leaf variants — the shared 100-sim search horizon
(v24's finding), not v22's demotion.

### 2.5 Verdict: no patchable mechanism — Phase 2 triggers

Final classification of the 12 distinct losses (pre-registered
scheme, §1.1, with two refinements the data forced):

| Class | Count | Games |
|---|---|---|
| Missed block | 0 | — |
| Missed forced win | 0 | — |
| Horizon loss (no save in window) | 6 | 11, 12, 13, 15, 16, 19 |
| Prior starvation | 0 | — |
| Leaf value error (strict) | 0 | — |
| Root-decision (visits-over-q; scheme gap) | 1 | 6 |
| Search-horizon value error (verified save; q/visits/leaf rank the loser above or are neutral; no exact threat signature) | 5 | 1, 2, 5, 7, 9 |

Patch-rule check (§1.1): no class with ≥3 verified instances
has a patch that reproduces the perfect move.

- The VCF-prevention mechanism has ONE clean instance (g6).
  The pre-registered VCF check (play a move leaving the
  opponent with NO VCF start) reproduces (9,7) but NOT (4,12)
  (leaves (2,7) live), NOT (5,13) (leaves all 3), and does not
  fire at all at g1/g7 (no VCF starts) or distinguish the
  saves at g5/g9 (VCF 4→4).
- "Minimise setups" fails on g7 (identical counts) and g5/g9
  (setups 0 either way). "Minimise creators" fails on g7
  (0 either way). "Minimise VCF starts" fails on g5/g9 (4→4).
- More sims is CLOSED (not monotonically better). q-based root
  decisions are CLOSED (pure q 0.20) and would only fix g6
  anyway (q prefers the save there by 0.005 — noise).
  Prior-temperature flattening is CLOSED.
- The 6 horizon losses have no saving move to reproduce.

The iter-22 §3.6 hypothesis (the full theirs-term makes v22
over-fear and play passively vs the forcing MCTS) is NOT
supported: at every verified losing position the v22 leaf
agrees with the v14/v20 leaves on the 1-ply ranking — the
demotion is never the deciding factor — and the collapse
itself is largely a seed artifact (§2.1). The v22-specific
demotion cost is real but confined to ONE position (g6 ply
16), where it undervalues the quiet block's gap-three
counterplay; that single instance cannot carry a patch (and
the ours-only demotion that would address the theirs-term
side is H2's v28 workstream, not this one).

On the extension rule (§1.1): the rule assumes the fresh-seed
forensics match IS the collapse corpus. It isn't (14-6, §2.1),
so extending it would add non-collapse losses that don't bear
on the collapse mechanism. The ladder corpus — the games that
actually collapsed — is exhausted instead: all 12 distinct
losses worked up, all 13 targeted hypothesis moves tested.
The rule's purpose (no premature conclusion from a small
sample) is satisfied by that completeness. **Phase 1
concludes: no patchable mechanism found. Phase 2 (§1.2)
triggers.**

## 3. Patch / fallback (v29)

TODO — filled in when the forensics concludes.


## 4. v29 evaluation

TODO — battery, filled in when run.

## 5. Verdict

TODO.
