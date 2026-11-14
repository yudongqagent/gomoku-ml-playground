# Iteration 11 — v11: value net trained on heuristic-opponent games (batch 4, H3)

**Status: pre-registered — design, data mix, training setup, α sweep
grid and eval seeds recorded BEFORE any data generation, training or
eval (see §1).** This is the H3 workstream of batch 4 (v7 is the
reference version; H1 q-vs-visits and H2 White weakness are parallel
workstreams). Work happens on branch `research/batch-4-h3` in the
`gomoku-b4-h3` worktree.

## 1. Design and pre-registration

**Comparison target: v7** (the batch-4 reference, ladder Elo 190.8).
The bar (batch-4 plan, unchanged from batch 3 except the reference):
1. Head-to-head vs v7, 100 games, alternating colours, win rate
   (draws = half) > 0.50 with Wilson 95% CI lower bound > 0.50.
2. Ladder Elo strictly higher than v7's 190.8 at identical game
   counts/opponents (20 games/opponent: random, heuristic, mcts;
   mcts_sims=100).
3. Tactical pass rate ≥ v7's (20/20 old suite + 29/29 hard suite).
4. All test suites green (ml pytest, npm test, tsc).
**Secondary head-to-heads, also pre-registered:** v11 vs v0 (100
games) and a 100-game swap match vs v7 under the tempo judge (the
harness default). v8's numbers (Elo 0.0, vs v1 0.48) are reported
alongside for transparency — v11's hypothesis is tested against
v8's failure mode, not just against v7.

**The improvement: v11 = v8's ValueMCTS (v6's PUCT MCTS + learned
value leaf, 100 sims, c_puct 1.5, v1 priors, win/block scans before
search, v3's 2-ply threat check at the MCTS root) with the value
net TRAINED ON GAMES AGAINST THE HEURISTIC OPPONENT instead of
weak neural self-play.** The search, the leaf structure (terminal
cases exact, net value shrunk by α in quiet positions only) and
the threat check are unchanged from v8 — the ONLY intended
difference is the training distribution. Any difference from v8
is attributable to the data (plus the re-swept α, see §1.3).

**Why — traced to iter-8 §7 finding 1.** v8's 4,350-game net
generalised (holdout 0.7165 full / 0.6278 no-history) yet at full
weight (α=1.0) lost 0-10 to v0 where v6's terminal-only leaf
scores 0.50. The cause: distribution mismatch (the net's values
are unreliable in the distribution heuristic opponents actually
play) + optimizer's curse (MCTS exploits the net's
overvaluations). Shrinking to α=0.25 recovered 0.60 vs v0 and
0.48 vs v1. Scaling weak self-play is NOT the fix — 8.7x more of
the same data fixed generalisation (0.44 → 0.72) and made game
strength WORSE (0.30 vs v1 at α=1.0). The hypothesis under test:
a net trained on games against the heuristic opponent will have
values that are reliable in the distribution it actually faces,
so it will play well at HIGHER α (less shrink needed) and beat
v8's game strength at its best α.

### 1.1 Data mix (FIXED before generation — this is the pre-registered budget and it does not change with results)

**4,000 games, round-2 checkpoint net vs the frozen heuristic
(`ml/gomoku_ml/baseline.py` `choose_move`, the v0 ladder anchor),
50 sims/move, via the B1 parallel path** (workers=4, concurrency
8 per worker, max-batch-size 32). The generator net is the
round-2 checkpoint (resnet 32ch/4blocks, 192,714 params — copied
from `gomoku-b3-b1/ml/runs/b3-b1/final.pt`, the same checkpoint
B1 used), and the self-play settings are B1's own (c_puct 1.5,
Dirichlet α 0.3 / ε 0.25, temperature 1.0 for 30 moves,
max_moves 225, no resign threshold). **Keeping the generator
identical to B1 isolates the opponent variable: same net, same
sims, same scale (~4k games), only the opponent changes from
self to heuristic.** That is the controlled A/B against v8.

- **Colour alternation:** even-numbered games (per worker) the net
  plays Black, odd-numbered the net plays White — 2,000 games per
  colour, so both colours' positions are covered.
- **Positions recorded from BOTH sides** (player-to-move
  perspective, as in the shard schema). Policy target: the MCTS
  visit distribution for neural moves; one-hot on the played move
  for heuristic moves (the heuristic has no visit distribution;
  the policy head is not used by ValueMCTS — only the value head
  is — so the one-hot policy is a harmless training target).
- **Opponent choice: v0 (baseline), NOT v1.** The task specifies
  `baseline.choose_move` (the frozen ladder anchor); it is also
  the opponent the ladder measures against (heuristic anchor =
  0 Elo), so it is the distribution the value net most needs to
  match. No opponent mix in the primary run — a mix would
  confound the A/B (two variables at once).
- **No mixing with the B1 self-play dataset in the primary run.**
  The 4,350-game self-play set stays a separate, untouched
  baseline; mixing it in would double the data scale AND change
  the distribution, so a win could not be attributed to the
  distribution. A mixed-data variant is a follow-up ONLY if the
  pure heuristic-opponent net fails AND the fast loop points at
  data diversity as the cause.
- **Why the round-2 net is the learner, not v7 or v8 (considered
  and rejected):** v7/v8 are unbatched HeuristicMCTS
  (~0.35–0.95 s/move) — 4,000 games × ~45 plies × ~0.5 s ≈ 25 h
  for one side alone, and the batched path cannot run them. The
  round-2 net via the batched path reaches 4,000 games in a few
  hours (B1 measured 1,073.5 games/hr at 50 sims for pure
  self-play; heuristic moves cost ~nothing, so this run should
  be at least as fast — the rate is measured, not assumed).
- **Known limitation, stated before generation:** the round-2 net
  is below the heuristic (0W-40L in the longrun eval), so the
  heuristic is expected to win the large majority of the 4,000
  games. Value targets are from the player-to-move perspective,
  so counts stay roughly balanced (heuristic-side positions get
  +1, neural-side positions get -1), but the net will mostly see
  positions where the heuristic is winning. That is the
  distribution the leaf actually faces in games against heuristic
  opponents — the hypothesis is that this is a feature, not a
  bug — but the outcome split is recorded with the data (§2) and
  the α sweep + battery are the arbiter, not the argument here.

**Hold-out (decided BEFORE generation).** 10% of games — 400
games, every 10th shard by sorted name (indices 9, 19, …, 3999)
— are held out as validation and NEVER used in training:
**3,600 train games (~153k samples), 400 val games (~17k
samples)**, the same game-level protocol as v8 (§2.1). The
value head's hold-out accuracy is the sign agreement between
the net's predicted value and the actual game outcome on
non-draw positions of held-out games, reported BEFORE any
match, computed two ways: (a) full planes as stored in the
shards, and (b) history planes (2, 4–7) zeroed, approximating
leaf conditions. Both numbers are recorded; (b) is the one
that predicts leaf quality. Comparison with v8's 0.7165 /
0.6278 is reported with the caveat that the distributions
differ (self-play holdout vs heuristic-opponent holdout) — it
is a sanity check, not an apples-to-apples comparison.

**Cross-distribution checks (pre-registered, diagnostic, cheap
— run with `ml/scripts/eval_value_holdout.py`, no training):**
(1) v8's net (`ml/runs/b3-h2/value-net.pt`, copied from the main
worktree) evaluated on MY heuristic-opponent holdout — if it
scores poorly there, that directly confirms the distribution
mismatch hypothesis; (2) MY net evaluated on v8's self-play
holdout (`gomoku-b3-h2/ml/runs/b3-h2-4350/val-data/`, 435 games)
— the reverse check. These two numbers measure the mismatch
itself, independently of game strength.

### 1.2 Training (pre-registered)

**Same architecture and recipe as v8's 4,350-game retrain
(`ml/configs/b3-h2-train-4350.yaml`): smoke ResNet 16ch/2
blocks, trained from scratch with `ml/scripts/train.py`
(policy+value loss; only the value head is used as the leaf):
5 epochs, batch 64, lr 0.002 cosine, warmup 20 steps, weight
decay 1e-4, augmentation on, internal val_fraction 0.05 (for
best.pt selection only — the game-level hold-out above is the
reported accuracy), torch threads 1
(`OMP_NUM_THREADS=1 TORCH_THREADS=1`, per B1's finding).**
Config: `ml/configs/b4-h3-train.yaml`. Run dir:
`ml/runs/b4-h3-value`. No architecture change is pre-registered;
if the fast loop demands one, it is recorded as a deviation.

**Checkpoint selection (pre-registered rule, same as v8).**
After training, compute the hold-out value sign-accuracy for
BOTH `best.pt` (best internal val loss) and `final.pt`; use the
higher-accuracy one (no-history measure, the leaf predictor) as
the leaf net, copied to `ml/runs/b4-h3/value-net.pt` (the path
the v11 registry entry points at); record BOTH accuracies here.
No other selection criterion is used.

### 1.3 Integration and the α sweep (pre-registered)

**Integration.** `ValueMCTS` gains a `value_weight` parameter
(default 0.25 = v8's value, so v8 is byte-identical in behaviour)
used as the instance-level shrink in `_value_leaf_value`; the
module-level `VALUE_WEIGHT` stays as the default. `ScoringPlayer`
passes `ENGINE_VERSIONS[version].get("value_weight")` through,
and the checkpoint env override generalises to a per-version
`"value_checkpoint_env"` key (v11: `GOMOKU_V11_CHECKPOINT`;
default `GOMOKU_V8_CHECKPOINT`, so v8 is unchanged).
`ml/gomoku_ml/scoring.py` gains ONLY the `ENGINE_VERSIONS[11]`
registry entry (`{"name": "scoring-v11", ..., "mcts": True,
"mcts_sims": 100, "value_checkpoint":
"ml/runs/b4-h3/value-net.pt", "value_checkpoint_env":
"GOMOKU_V11_CHECKPOINT", "value_weight": <α from the sweep>}`).
`baseline.py` is frozen and untouched; `mcts_heuristic.py`,
`mcts.py` and the sequential `selfplay.py` path are untouched
(the batched/parallel path gains the opponent option — see §1.4).
The 2-ply threat check at the MCTS root is kept (it lives in
`ValueMCTS.choose_move`, inherited unchanged — per iter-8 §7
finding 3 it is a durable portable gain).

**α (VALUE_WEIGHT) sweep grid (pre-registered — 0.25 is NOT
assumed).** α ∈ {1.0, 0.5, 0.25, 0.125}, 20-game smokes vs v0,
seeds 1001/2001 (the same seeds as v8's sweep matches, for
comparability — v8 used 10 games; 20 here for slightly tighter
CIs). v8's sweep for reference: α=1.0 → 0.00, 0.5 → 0.20,
0.25 → 0.60 vs v0. **Selection rule: highest win rate; ties
broken toward LARGER α** (prefer using the net more when equal
— the hypothesis under test is that heuristic-opponent training
makes higher α viable). If all α score ≤ 0.50 vs v0, the best
is taken anyway and the fast loop investigates. The sweep is a
selection procedure on 20 games — noisy; the battery is the
arbiter. The sweep runs on the trained net BEFORE the fast loop
and the battery, and the chosen α is recorded in the registry
entry only after the sweep completes.

### 1.4 Data generator (pre-registered design)

`generate_games_batched` gains an `opponent` option
(`None` = self-play, unchanged behaviour; `"heuristic"` = one
colour per game, alternating by game number, is played by
`baseline.choose_move` instead of the neural MCTS). The
heuristic side never enters `search_many` (its moves cost ~ms,
no net eval); its policy target is one-hot on the played move.
Threaded through `generate_games_parallel` (via `game_kwargs`,
so the spawn spec is unchanged in shape) and the
`ml/scripts/selfplay.py` CLI (`--opponent heuristic`). B1's
invariants are preserved: per-worker shard dirs + validated
merge, 1 torch thread per worker, parent drains the result
queue continuously (the deadlock fix), seeds differ per worker.
Tests: a short heuristic-opponent run (few games, 2 workers)
verifying shard validity, both colours covered, one-hot
heuristic policies, and merge correctness.

### 1.5 Fast loop (pre-registered)

Per the direction change (iter-7 §1 / iter-8 §1.1): the fast
10-game loop stays the finding method. Max 4 iterations of 10
games, or stop when an iteration finds no new weakness class.
Iteration 1 vs v0 (dev seeds 500/600 — distinct from the bar
battery's 100/200 and the α sweep's 1001/2001); later
iterations vs v0 (seeds 501/601, …) or vs v7 if v0 produces no
losses. For each lost game, THE losing move is found (verified
with the solver / deeper search / corrected playout oracle —
never guessed), the structural cause is patched, and the
critical position + correct move is recorded as a regression
test (`ml/gomoku_ml/regression_positions_v11.json` +
`ml/tests/test_regression_v11.py`). Each iteration is reported
concretely in §4. The fast-loop worked games ARE the loss
analysis; aggregate classification is secondary.

### 1.6 Eval order and seeds (pre-registered)

1. Data generation (§2) → hold-out split → training (§2) →
   hold-out accuracy + cross-distribution checks (§2), all
   BEFORE any match.
2. Integration (§1.3) + leaf cost measurement (same bench as
   iter-8 §3, `ml/scripts/bench_value_leaf.py` with the v11
   checkpoint).
3. Tactical old suite (20) + hard suite (29) using
   `player.choose_move` with FULL dispatch
   (`ml/scripts/eval_tactical_full.py --version 11 --suite both
   --seed 0`).
4. α sweep (§1.3) → registry entry records the chosen α.
5. FAST HEURISTIC LOOP (§1.5) on the swept v11.
6. FULL VALIDATION BATTERY on the final patched v11 (same
   harness commands as iter-8 §5 / iter-7 §3, reference moved
   to v7):
   - tactical old+hard (re-run);
   - h2h vs v7, 100 games, seeds 100/200 →
     `research/iterations/batch-v11-vs-v7.json` (clause 1);
   - h2h vs v0, 100 games, seeds 100/200 →
     `batch-v11-vs-v0.json`;
   - 100-game swap match vs v7, balanced, placement_seed 11,
     tempo judge → `batch-v11-vs-v7-swap.json`;
   - ladder 20 games/opponent via `ml/scripts/eval_version.py
     --version 11 --ladder-games 20 --mcts-sims 100 --seed 100`
     (clause 2 vs v7's 190.8).
   Any battery numbers measured before the fast loop completes
   are INTERMEDIATE (pre-patch) results, labelled as such; only
   the final battery counts for the bar.
7. Full ml pytest before reporting (clause 4; npm test + tsc
   verified by the coordinator).

**Eval seeds (pre-registered):** h2h vs v7: v11=100 / v7=200;
h2h vs v0: v11=100 / v0=200; ladder seed 100, 20
games/opponent, mcts_sims=100; swap 100 games vs v7, balanced
placement, placement_seed 11, tempo judge (the harness default;
recorded in the JSON). Fast-loop dev seeds 500/600, 501/601, …;
α-sweep seeds 1001/2001.
