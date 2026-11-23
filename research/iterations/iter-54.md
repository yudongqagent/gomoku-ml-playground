# Iteration 54 — v54: White-side training positions weighted toward heuristic-opponent games (ML batch 6, H1)

**Status: IN PROGRESS — pre-registered; no eval run yet.**
Comparison target is v44 (the ML track's champion and bar
reference: ladder Elo 98.1, vs v8 0.67, tactical 20/20 +
29/29 — see iter-44.md). This is the H1 workstream of ML
batch 6 (H2 is v55, 3-ply retrain of the v48 patch data;
H3 is v56, parity-corrected 3-ply labels); work happens on
branch `research/ml-batch-6-h1` in the `gomoku-ml6-h1`
worktree, from the coordinator branch HEAD `038232c` (ML
batch 6 plan + fixed `eval_value_holdout.py`).

**Method (carried over):** the fast 10-game loop — play 10
games, read EVERY move, find THE losing move per game
(solver/deeper-search verified, never guessed), patch,
regression-test, repeat. Fast loops run against v0 as well
as v44. Max 4 fast-loop iterations before validation, or
stop when an iteration finds no new patchable weakness
class. The full 4-clause validation battery runs on the
FINAL v54 only. Fast-loop games use development seeds
(1001/2001, ...), never the validation battery's seeds
(100/200).

## 0. Setup (before pre-registration)

**Checkpoints** (gitignored — copied into this worktree at
the registry paths, md5-verified):
- v44: `ml/runs/ml4-h3/value-net.pt`, md5
  `d22af606f7d2ee0254f372da895ae7ad` ✓ (matches iter-44).
- v37: `ml/runs/ml3-h2/value-net.pt`, md5
  `2fde492aae6d44cbe945e15bd5f1f959` ✓ (matches iter-37).
- v8: `ml/runs/b3-h2/value-net.pt`, md5
  `5deb26f3df67883f24f04800d5356da6` ✓ (matches the main
  worktree copy).

**Data — the task premise was wrong, and the correction
matters.** The tasking said ALL distill game records and
shards are gone and v54 must regenerate the base with
fresh self-play. That is FALSE for the copies in the ML
batch 5 H1 worktree (`gomoku-ml5-h1/ml/runs/ml5-h1/`):
v37's base shards AND v48's patch games survive there,
and they verify EXACTLY against the published records:
- Base shards (`data-base/`): **37,227 positions total
  (33,849 train / 3,378 holdout)**, per-source 21,595 v7
  / 9,932 v18 / 5,700 random — exact match with v37's
  record (iter-37 §1). Label means (−0.054 v7, −0.087
  v18; positive fraction 0.218/0.159) are consistent
  with 2-ply labels, not 3-ply (mean +0.109, posfrac
  0.734). Shard counts match v24's protocol exactly
  (440+48 v7, 207+22 v18, 90+10 random).
- Patch games (`games-vs-v*.json`): move lists and
  winners match the committed battery JSONs in this
  tree (`research/iterations/batch-v44-vs-v0.json`,
  `batch-v44-vs-v7.json`) with **0 mismatches** in 200
  games.
- v48's patch shards (`data-new/`): per-leg totals
  5,116 / 3,998 / 3,753 / 4,487 — exact match with
  iter-48 §7.1.
So v54 reuses the verified base shards (copied verbatim,
exactly as v48 did with v37's shards) instead of
regenerating them. This is scientifically BETTER than
regeneration: the identical base makes v54 a clean A/B
against v48 on the patch alone, where fresh base games
would introduce a second variable. Only the
ladder-MCTS patch games (§2) need fresh generation —
they never existed. The regeneration premise and this
correction are reported to the coordinator.

**Tools copied in** (from the ml5-h1 worktree; not on
master because ML batch 5 was never merged):
`ml/scripts/diag_white_collapse.py` (the §2 diagnosis
script, reused for the §3 diagnostic) and
`ml/scripts/score_shards_by_side.py` (TRUE-side shard
scoring). The fixed `eval_value_holdout.py` on this
branch reads side-to-move from plane 3 (exact) — all
by-side holdout numbers in this log use it, never the
old index inference (iter-48 §0).

**No code ports needed** beyond §2's two small additions.
This tree already carries everything ML batch 4 ported:
`_rollout2_leaf_value` / `_rollout3_leaf_value`,
`generate_distill_data.py label --label-mode
{rollout,rollout2,rollout3}`, the memmap `train.py`,
`build_mmap_bundle.py`, `sweep_alpha.py`,
`find_losing_moves.py`, `targeted_playout.py`,
`fast_loop_match.py` (with the `mcts` ladder opponent),
and the v37/v42/v43/v44 registry entries.
`ml/gomoku_ml/baseline.py` is FROZEN and untouched.

## 1. The problem (ML batches 4–5's numbers)

The distilled nets collapse as White against heuristic
opponents — but the collapse is version-dependent, and
v48's fix redistributed rather than improved:

| Version | White vs v0 | White vs v7 | Ladder heur White | Ladder mcts White |
|---|---|---|---|---|
| v37 (2-ply) | 0.44 | 0.31 | — | — |
| v42 (1-ply+dropout) | 0.28 | 0.15 | 0.20 | 0.25 |
| v43 (2-ply+dropout) | 0.21 | — | — | 3-16-1 |
| v44 (3-ply) | 0.59 | 0.40 | 0.60 (11-7-2) | 0.55 (11-9-0) |
| v48 (2-ply+patch) | 0.52 | 0.27 | 0.35 (7-13-0) | 0.15 (3-17-0) |
| v49 (4-ply) | 0.22 | 0.47 | 0.15 (3-17-0) | 0.45 (9-11-0) |

v48's diagnosis (iter-48 §6): a GENERAL White-side
value-head defect — v44's net reproduces its teacher
worse as White than as Black on EVERY opponent (gaps
−0.065 v0, −0.066 v7, −0.103 v8, −0.076 v37), worst on
White-WIN positions (undervalues White's winning
chances) and White-LOSE positions (fails to see White
is losing). v48's data patch (ALL quiet positions from
v44's four battery legs, both sides, 2-ply labels, on
top of v37's base) closed the holdout White gap
(−0.008 pooled) and the played White gap vs v0 (0.52)
and v7 (White nominally better than Black) — but Black
fell by at least as much on every leg (Black vs v7
0.26; vs v44 0.44 overall; Elo −34.9). A redistribution,
not an improvement.

**Hypothesis for the redistribution (v54's design
target):** the patch's Black-side positions injected
optimistic Black labels. In v44's battery games vs weak
opponents (v0), v44-as-Black wins easily, so the
Black-to-move patch positions skew toward
Black-winning — training on them makes the net
overvalue Black's chances and under-defend as Black.
The White-side patch positions were the fix; the
Black-side patch positions were the cost. v54 tests
this by adding ONLY White-side patch positions (the
fix) and ZERO Black-side patch positions (the cost).
If v54's White gap closes with Black intact, the
hypothesis is supported; if Black falls anyway, the
cost is in the White positions themselves (or in the
2-ply retrain) and the hypothesis is refuted.

## 2. Design (pre-registered)

**v54 = v8's search + a new net.** Search, priors, sims,
threat check, terminal cases: v8/v44 unchanged, so the
battery is a clean A/B of the net — and, against v48's
numbers, of the White-only patch against the both-sides
patch on the identical base.

**Recipe (v37's exactly):** ResNet 16ch/2 blocks from
scratch, 15 epochs, batch 64, lr 0.002 cosine, warmup
20 steps, weight decay 1e-4, value MSE only
(policy_weight 0.0), augmentation on, val_fraction
0.05, seed 0, memmap path (mandatory), checkpoint
selection best-by-internal-val-loss. Config
`ml/configs/ml6-h1.yaml` (v37's `ml3-h2.yaml` with run
name/dir changed). Labels: **2-ply**
(`_rollout2_leaf_value`) — v37's recipe, the strongest
labelling (the label-depth lever is CLOSED at both
parities; this batch works on label quality/data mix,
not depth).

**Training data — base:** v37's 2-ply-labelled shards,
copied VERBATIM from the surviving ml5-h1 copy and
verified (§0): 33,849 train / 3,378 holdout = 37,227
positions (v7/v18/random). Same base v48 used.

**Training data — patch (the weighting scheme):**
White-to-move QUIET positions ONLY, 2-ply labelled,
from games against heuristic opponents ONLY. Five
100-game legs:
- (a) v44 vs v0, validation seeds — the committed
  battery leg `batch-v44-vs-v0.json` (via
  `games-vs-v0.json`, verified 0 mismatches). Expected
  yield: **2,536 White positions** (iter-48 §6.1's
  diagnosis count on the same games — the label step
  must reproduce it exactly).
- (b) v44 vs v7, validation seeds — the committed
  battery leg `batch-v44-vs-v7.json`. Expected yield:
  **1,903 White positions** (iter-48 §6.1).
- (c) v44 vs v0, FRESH, dev seeds 1001/2001 —
  `fast_loop_match.py --version-a 44 --opponent v0
  --games 100`. Expected yield: ~2,500 White.
- (d) v44 vs v7, FRESH, dev seeds 1001/2001 —
  `fast_loop_match.py --version-a 44 --opponent v7
  --games 100`. Expected yield: ~1,900 White.
- (e) v44 vs ladder MCTS, FRESH, dev seeds 1001/2001 —
  `fast_loop_match.py --version-a 44 --opponent mcts
  --games 100` (the `mcts` opponent is
  `baseline.HeuristicMCTSPlayer` at 100 sims — the
  ladder's exact opponent, already supported by
  `fast_loop_match.py`). Expected yield: ~1,800 White.
  The ladder MCTS is the opponent v48's patch missed
  (ladder White vs mcts 0.15); this leg is the new
  distribution in v54's patch.

Expected total patch: **~10,600 unique White positions**
(~9,500 train after the game-level holdout: every 10th
game held out, same protocol as v48). Total train set
≈ 43,350 positions, ~57% White (vs 50% in the base).
**No duplication:** the weighting is the selection
itself (White patch positions weight 1, Black patch
positions weight 0, ML-opponent positions weight 0),
plus the doubled v0/v7 sampling (validation + dev
legs). Duplication was considered and rejected: it
would leak twin positions across train.py's
sample-level val split and bias best-checkpoint
selection; unique fresh games buy the same dose
cleanly.

**Explicitly excluded:** every Black-to-move position
from the patch games (weight 0 — the §1 hypothesis);
every position from ML-opponent games (the v8/v37 legs
v48 included are OUT — the plan's "weighted toward
heuristic-opponent games").

**Caveat (recorded, not hidden):** legs (a)/(b) come
from v44's battery games at validation seeds 100/200,
and v54's battery replays the same seeds. v54 ≠ v44,
so the games diverge from the first differing move;
and the v24/v37 precedent is the same shape (trained
on v7 self-play games, then played vs v7). Legs
(c)/(d)/(e) use dev seeds, so only ~42% of the patch
carries this caveat. The game-level holdout on the new
positions is kept for the holdout diagnostic, not for
the battery.

**Implementation (two small additions, both tested):**
- `generate_distill_data.py label` gains `--side
  {both,black,white}` (default `both`): the side to
  move is known exactly at label time (`p` in the
  labelling loop), so the filter is exact — no plane
  decoding, no index inference. The side filter is
  recorded in the LABEL_JSON output.
- New `ml/scripts/battery_to_games.py`: converts a
  battery/fast-loop JSON (records with moves/winner)
  to the games-file format the label step takes
  (winner mapped black→1 / white→−1 / draw→0) —
  reproducible provenance for legs (a)/(b) and (c)–(e).
  (v48's conversion was uncommitted; this one is
  committed so H2/H3 and future batches reuse it.)
- Tests: `--side white` labels exactly the
  White-to-move quiet positions (values match direct
  `_rollout2_leaf_value` calls; shard count and
  per-position sides verified via plane 3); the
  converter reproduces `games-vs-v0.json` from
  `batch-v44-vs-v0.json` exactly.

**α sweep:** {0.125, 0.25, 0.5}, **40 games each** vs
v44, dev seeds 1001/2001, via `sweep_alpha.py
--version 54` (40 games/arm per the batch-6 plan —
20-game sweeps rank noisily; ML batch 5's winner
regressed 0.55 → 0.475 at 100 games). Selection: best
win rate; tie-break the smaller α (conservative);
no-signal rule (all three arms within 2 wins) →
α=0.25 (v37's winner). The registry entry's
`value_weight` starts at 0.25 (placeholder, v37's
winner) and is updated to the sweep winner once,
after the sweep.

## 3. Mechanism check (pre-registered)

The plan's question: **does the played White gap vs
heuristic opponents close WITHOUT Black-side strength
falling?** Measured on the validation battery's
100-game legs (50-game colour splits, SE ≈ 0.07 per
split — thresholds are set with that noise in mind):

1. **White closure:** v54 White vs v0 ≥ 0.50 AND White
   vs v7 ≥ 0.40. (v44's levels: 0.59 / 0.40; v37's:
   0.44 / 0.31; v48's: 0.52 / 0.27.)
2. **No Black fall:** v54 Black vs v0 ≥ 0.52 AND Black
   vs v7 ≥ 0.45. (v44's Black: 0.57 / 0.50, minus a
   0.05 tolerance for 50-game noise; v48's Black vs v7
   was 0.26 — the redistribution failure to avoid.)
3. **Net gain:** clauses 1+2 (the bar) decide whether
   the fix is an improvement rather than a
   redistribution. The mechanism check passes only if
   (1) and (2) both hold AND the battery shows a net
   gain vs v44.
4. **Ladder (secondary, 20-game splits — noisy):**
   ladder heuristic White ≥ 0.40 and ladder mcts White
   ≥ 0.40. (v44: 0.60 / 0.55; v48: 0.35 / 0.15; v49:
   0.15 / 0.45.)
5. **Diagnostic (does NOT decide):**
   `diag_white_collapse.py` on the v54 net vs its
   2-ply target, on v44's battery-leg positions — the
   White-side sign-agreement gap on heuristic-opponent
   positions should close vs v44's net (−0.065 v0,
   −0.066 v7); the fixed `eval_value_holdout.py`
   by-side numbers on the patch holdouts should agree.
   Standing caveat: holdout does not predict played
   strength (six instances) — the battery is the
   verdict.

## 4. Eval seeds and battery (pre-registered)

**The ML bar (reference v44, Elo 98.1):** (1) h2h vs
v44 ≥ 100 games, win rate > 0.50 with Wilson 95% CI
lower bound > 0.50; (2) ladder Elo > 98.1; (3)
tactical ≥ 20/20 + 29/29 on the corrected suite;
(4) all test suites green. Fail any clause =
rejected. Holdout numbers do NOT predict played
strength — the 100-game battery is the verdict.

**Seeds:** h2h vs v44: v54=100 / v44=200; vs v0:
v54=100 / v0=200; vs v7: v54=100 / v7=200; vs v37:
v54=100 / v37=200; ladder seed 100, 40 games /
opponent; swap vs v7, balanced placement,
placement_seed 11, tempo judge; tactical --seed
100. The vs-v44 and vs-v37 legs use the
pre-registered `research/openings-ml4.json` (10
openings) via `--openings`. Per-game seeding +
uniqueness audit on every leg; report distinct-game
counts and de-dup rates.

**Battery runbook** (validation seeds 100/200):
```
# Clause 1: vs v44, 100 games, pre-registered openings
selfplay_batch.py --engine-a v54 --engine-b v44 --games 100 --openings research/openings-ml4.json --seed-a 100 --seed-b 200 --out research/iterations/batch-v54-vs-v44.json
# Clause 2: ladder (40 games/opponent, mcts_sims 100)
eval_version.py --version 54 --ladder-games 40 --out research/iterations/ladder-v54.json
# Clause 3: tactical, both suites, seed 100
eval_tactical_full.py --version 54 --suite both --seed 100
# Clause 4: ml pytest + npm test + tsc
# Secondary: vs v0, vs v7, vs v37 (openings), swap vs v7 (balanced/tempo, placement_seed 11)
selfplay_batch.py --engine-a v54 --engine-b v0 --games 100 --seed-a 100 --seed-b 200 --out research/iterations/batch-v54-vs-v0.json
selfplay_batch.py --engine-a v54 --engine-b v7 --games 100 --seed-a 100 --seed-b 200 --out research/iterations/batch-v54-vs-v7.json
selfplay_batch.py --engine-a v54 --engine-b v37 --games 100 --openings research/openings-ml4.json --seed-a 100 --seed-b 200 --out research/iterations/batch-v54-vs-v37.json
selfplay_batch.py --engine-a v54 --engine-b v7 --games 100 --swap-rule --placement balanced --placement-seed 11 --judge tempo --seed-a 100 --seed-b 200 --out research/iterations/batch-v54-vs-v7-swap.json
```

## 5. Eval order (pre-registered)

(1) Patch-game generation (legs (c)–(e), dev seeds) +
labelling (all five legs, `--side white`, 2-ply) +
bundle build — with the §2 expected yields verified
per leg before training. (2) Train + holdout
diagnostics on the new net (by TRUE side — §0; the
fixed `eval_value_holdout.py` + `diag_white_collapse.py`
+ `score_shards_by_side.py`). (3) Leaf-cost
measurement. (4) α sweep (§2; 40 games/arm). (5) FAST
LOOP: up to 4 iterations of 10 games vs v44 AND 10
games vs v0 (dev seeds). (6) FULL VALIDATION BATTERY
on the final v54 (§4). Any battery numbers measured
before the fast loop completes are INTERMEDIATE,
labelled as such; only the final battery counts for
the bar.

## 6. Data build (in progress)

### 6.1 Patch legs (a)/(b) — labelled, yields EXACT

`battery_to_games.py` on the committed battery JSONs
reproduces ml5-h1's `games-vs-v0.json` /
`games-vs-v7.json` EXACTLY (semantic equality; the
byte diff is JSON spacing only). Labelling
(`--label-mode rollout2 --side white`, holdout every
10th game; log lines in the LABEL_JSON outputs):

| Leg | Games | Plies | White quiet | ±1 labels | Held-out games |
|---|---|---|---|---|---|
| (a) vs v0 | 100 | 6,342 | **2,536** | 132 (5.2%) | 10 |
| (b) vs v7 | 100 | 4,896 | **1,903** | 138 (7.3%) | 10 |
| **Total** | **200** | **11,238** | **4,439** | **270 (6.1%)** | **20** |

Both yields match the §2 pre-registered expectations
EXACTLY (2,536 / 1,903 — iter-48 §6.1's diagnosis
counts on the same games). Cross-verification: the
White-only shards are an EXACT subset of v48's
both-sides shards for the same legs (same games,
same 2-ply labels): 4,439 positions, max |value
diff| 0.0, planes identical — the `--side` filter is
verified end-to-end on real data, not just on the
unit-test toy game.

### 6.2 Patch legs (c)–(e) — generating

Chain `ml/runs/ml6-h1/chain-patch-games.sh` (launched
2026-11-22 ~07:11 PST): fresh 100-game legs v44 vs
v0 / v7 / ladder-MCTS at dev seeds 1001/2001, via
`fast_loop_match.py`, sequential, each auto-resuming.
Rate under contention is ~150 s/game wall-clock
(load ~3.5 — the champion-change match and sibling
batch-6 workers share the VM). Three VM replacements
so far (17:42, 19:02, 21:02 UTC); each relaunch
resumed cleanly from the partial JSON (game 64, game
85, then leg (d) game 55) — zero games lost.

| Leg | Result | White quiet (labelled) |
|---|---|---|
| (c) vs v0, dev seeds | 52W-43L-5D (complete) | **2,634** (6,609 plies; expected ~2,500) |
| (d) vs v7, dev seeds | 48W-49L-3D (complete) | **1,877** (4,681 plies; expected ~1,900) |
| (e) vs ladder MCTS, dev seeds | 47W-52L-1D (complete) | **1,671** (4,418 plies; expected ~1,800) |

All three legs complete 2026-11-22 ~22:32 UTC
(`PATCH_GAMES_DONE`, no further restarts after 21:02).
Note: v44 LOSES to the ladder MCTS on dev seeds
(47-52-1) while beating it 25-15 on the ladder's seed
100 — different seeds, different games; these are
training-data games, not evals.

Patch totals (all five legs, game-level holdout every
10th game):

| Leg | White train | White holdout | White total |
|---|---|---|---|
| (a) vs v0 | 2,240 | 296 | 2,536 |
| (b) vs v7 | 1,756 | 147 | 1,903 |
| (c) vs v0-dev | 2,463 | 171 | 2,634 |
| (d) vs v7-dev | 1,705 | 172 | 1,877 |
| (e) vs mcts-dev | 1,531 | 140 | 1,671 |
| **Total patch** | **9,695** | **926** | **10,621** |

Total patch 10,621 White positions vs the §2
expectation of ~10,600 — on target. Memmap bundle
`ml/runs/ml6-h1/data-mmap/`: 1,187 shards, **43,544
samples** (33,849 base + 9,695 patch), planes
(43544, 8, 15, 15). Measured White share of the
bundle ≈ 0.60 (sampled 1/7) — the §2 estimate of
~57% was arithmetic-sloppy; the actual is ~60%
White vs 50% in the base.

## 7. Training (actuals)

**Training COMPLETE 2026-11-22 ~22:52 UTC** (started
~22:34 UTC, ~18 min, no interruptions): config
`ml/configs/ml6-h1.yaml`, 15 epochs, batch 64,
memmap bundle (43,544 samples; 41,367 train / 2,177
val, seed-0 split). Run dir `ml/runs/ml6-h1-train/`;
log `ml/runs/ml6-h1/train.log`. 9,705 steps,
~46–117 s/epoch.

| Epoch | Train loss | Val loss |
|---|---|---|
| 1 | 0.1017 | 0.0999 |
| 2 | 0.0963 | 0.0984 |
| 3 | 0.0946 | 0.0976 |
| 4 | 0.0927 | 0.0963 |
| 5 | 0.0905 | 0.0940 |
| 6 | 0.0889 | 0.0911 |
| 7 | 0.0865 | 0.0907 |
| 8 | 0.0838 | 0.0882 |
| 9 | 0.0831 | 0.0888 |
| 10 | 0.0809 | 0.0879 |
| 11 | 0.0795 | 0.0864 |
| 12 | 0.0778 | 0.0856 |
| 13 | 0.0779 | **0.0849** |
| 14 | 0.0766 | 0.0851 |
| 15 | 0.0769 | 0.0849 |

Val loss decreased overall (small bump at epoch 9)
— no overfitting by epoch 15. **Best checkpoint:
epoch 13 (`best.pt`, val 0.0849)** — best_val_loss
0.0848611 (epoch 15's 0.0849072 is slightly higher).

**Checkpoint-selection correction (recorded, not
hidden):** `train.py --out` saves the FINAL-epoch
model, not best.pt — verified: `--out` vs `final.pt`
max |ΔW| = 0.0, vs `best.pt` max |ΔW| = 1294.0 (that
figure is the batch-norm `num_batches_tracked`
counter, 9705 − 8411 steps — not a weight). The
pre-registration (§2) requires best-by-val-loss, so
the registry checkpoint is `best.pt` copied to
`ml/runs/ml6-h1/value-net.pt` (md5-verified identical
to best.pt, max |ΔW| 0.0). Registry md5:
`8b7cff2b0a56c4343a544579468aa9ce`. Architecture
verified: ResNet 16ch/2 blocks, 126,759 params, 48
tensors — identical counts to v24's, v37's, v44's,
and v8's checkpoints.

### 7.3 Holdout diagnostics (actuals)

Game-level holdouts (never trained on), targets =
2-ply labels, via the FIXED `eval_value_holdout.py`
(plane-3 sides — §0). v54 net =
`ml/runs/ml6-h1/value-net.pt`; v37 net =
`ml/runs/ml3-h2/value-net.pt` (same targets, same
positions — direct comparison). Per-holdout logs
`ml/runs/ml6-h1/holdout-*.log`.

| Holdout | n pos | v54 sign (full) | v54 sign (no-hist) | v37 sign (full) | v37 sign (no-hist) |
|---|---|---|---|---|---|
| v7 (base) | 2,050 | 0.7574 | 0.7605 | 0.7395 | 0.5163 |
| v18 (base) | 790 | 0.8423 | 0.8423 | 0.8436 | 0.5846 |
| random (base) | 538 | 0.6435 | 0.5422 | 0.6623 | 0.6604 |
| vs-v0 (new) | 296 | 0.5805 | 0.5730 | 0.6479 | 0.6891 |
| vs-v7 (new) | 147 | 0.6759 | 0.7379 | 0.7034 | 0.6690 |
| vs-v0-dev (new) | 171 | 0.8246 | 0.8304 | 0.6667 | 0.6140 |
| vs-v7-dev (new) | 172 | 0.8246 | 0.8421 | 0.7895 | 0.5906 |
| vs-mcts-dev (new) | 140 | 0.6786 | 0.6571 | 0.6857 | 0.7357 |
| **base pooled** | **3,378** | **0.7591** | — | **0.7519** | — |
| **new pooled** | **926** | **0.7047** | — | **0.6935** | — |

v54 beats v37's net on base pooled (+0.007) and new
pooled (+0.011) — small edges, in the same direction
as v48's (v48 beat v37 on 6/7 holdouts). Per-leg the
new holdouts are MIXED and noisy (10 games each,
game-clustered): v54 wins big on the dev legs
(vs-v0-dev 0.825 vs 0.667, vs-v7-dev 0.825 vs 0.790)
but LOSES on the validation-seed legs (vs-v0 0.581
vs 0.648, vs-v7 0.676 vs 0.703) and ties on
mcts-dev (0.679 vs 0.686). No clean per-leg read —
the pooled numbers are the read, and they are
near-level. (The validation-leg losses are
surprising — v54 trained on those legs' other 90
games — but at n=145–267 from 10 games each, the
cells are dominated by which games were held out;
v48 §7.3 recorded the same opposite swings.)

**The v54 net is HISTORY-ROBUST on engine
positions** (v7 no-hist 0.7605 vs full 0.7574, v18
no-hist = full 0.8423; patch holdouts no-hist ≈
full) — the v37 collapse (0.7395 → 0.5163) is gone,
like v44's and v48's nets. The exception is random
positions (no-hist 0.5422 vs full 0.6435) — the net
leans on history for random-playout positions, the
least game-like source; engine positions are what
the leaf sees in real games.

**By TRUE side on the base holdouts** (fixed
`eval_value_holdout.py`, 2-ply labels):

| Split | v54 Black | v54 White | White − Black | v37 Black | v37 White |
|---|---|---|---|---|---|
| v7 | 0.7137 | 0.8021 | +0.088 | 0.7127 | 0.7670 |
| v18 | 0.8785 | 0.8052 | −0.073 | 0.8785 | 0.8078 |
| random | 0.5243 | 0.7632 | +0.239 | 0.5543 | 0.7707 |
| **base pooled** | **0.7226** | **0.7964** | **+0.074** | **0.7269** | **0.7775** |

On the base holdouts v54's White is BETTER than
Black (+0.074) — the White gap is closed and
inverted, as it is for v37's net (+0.051). (The
defect v48 diagnosed lives on the battery-leg
positions, not the base holdouts — the diag script
below is the sharper instrument.) The patch
holdouts are White-only by construction, so they
carry no by-side split. **Caveat (standing):
holdout does not predict played strength — the
battery is the verdict.**

### 7.4 Diag — the White-side gap is CLOSED and inverted (§3.5)

`diag_white_collapse.py` on v44's four battery legs
(17,354 quiet positions, TRUE sides), subjects v54
(2-ply target), v44 (3-ply), v37 (2-ply). Log
`ml/runs/ml6-h1/diag.log`, JSON
`ml/runs/ml6-h1/diag-white-collapse.json`. Labelling
19.8 min (killed once by a VM replacement at 23:03
UTC; re-run clean).

**Reproduction check:** the v44 subject's cells
reproduce iter-48 §6.1 EXACTLY (v0 Black 0.7820 /
White 0.7170, v7 0.7651 / 0.6993, v8 0.7805 / 0.6771,
v37 0.7562 / 0.6805, overall 0.7344) — the
generalized `--subject` script and the position set
are verified against the published diagnosis.

**v54 net vs its 2-ply target (TRUE sides, full planes):**

| Cell | n | sign agreement | no-hist | Pearson r |
|---|---|---|---|---|
| v0: Black | 2,580 | 0.7256 | 0.7110 | 0.0751 |
| v0: White | 2,536 | 0.7627 | 0.7652 | 0.2358 |
| v7: Black | 2,095 | 0.7450 | 0.7440 | 0.0937 |
| v7: White | 1,903 | 0.7980 | 0.7986 | 0.2145 |
| v8: Black | 1,938 | 0.7585 | 0.7515 | 0.0784 |
| v8: White | 1,815 | 0.7971 | 0.8046 | 0.0666 |
| v37: Black | 2,357 | 0.7691 | 0.7627 | 0.0739 |
| v37: White | 2,130 | 0.7777 | 0.7762 | 0.1798 |
| heuristic-pooled: Black | 4,675 | 0.7344 | 0.7260 | 0.0853 |
| heuristic-pooled: White | 4,439 | 0.7781 | 0.7798 | 0.2265 |
| ml-pooled: Black | 4,295 | 0.7642 | 0.7576 | 0.0778 |
| ml-pooled: White | 3,945 | 0.7868 | 0.7895 | 0.1152 |
| overall | 17,354 | 0.7649 | 0.7620 | 0.1408 |

The v54 net's White-side accuracy EXCEEDS its
Black-side accuracy on EVERY opponent (v0 +0.037,
v7 +0.053, v8 +0.039, v37 +0.009; heuristic-pooled
+0.044, ml-pooled +0.023) — on the exact position
set where v44's net showed −0.065 / −0.066 /
−0.103 / −0.076. **The §3.5 diagnostic PASSES
decisively.** Two readings: (a) the v0/v7 cells are
partly trained-on (90% of those legs' games are in
the patch — v54's numbers there closely match
v48's trained-on row, Black 0.7399 / White 0.7837);
(b) the v8/v37 cells are FULLY unseen for v54 (no
ML-opponent positions in the patch) and White still
beats Black there — the fix generalizes beyond the
patch distribution. The by-outcome cut reverses
both halves of v48's defect on the heuristic legs:
White-WIN 0.726/0.763 vs Black-WIN 0.650/0.688 (v0/v7),
White-LOSE 0.867/0.849 vs Black-LOSE 0.820/0.812.
The net is history-robust on these positions too
(overall no-hist 0.7620 vs full 0.7649).

**Leaf cost** (`ml/runs/ml6-h1/leaf-bench.txt`):
`bench_value_leaf.py --checkpoint
ml/runs/ml6-h1/value-net.pt --positions 20
--leaf-samples 200 --sims 100`: per-move at 100 sims
mean 2,646 ms / median 2,545 ms / p95 3,904 ms; net
leaf evals per move mean 81.7 (24–99); per-leaf net
eval median 1.80 ms. Faster than v48's bench (mean
4,120 ms under load 7.45) — same architecture; the
difference is contention.

## 8. α sweep and fast loops

### 8.1 α sweep (2026-11-22 ~23:30 – 2026-11-23 ~02:37 UTC)

`sweep_alpha.py --version 54 --alphas 0.125 0.25 0.5 --games 40
--opponent v44 --seed-a 1001 --seed-b 2001` (dev seeds, no
openings). One VM replacement killed the run during game 21
of the α=0.25 leg; the driver resumed from game 21
(7W-13L-1D) with per-game seeding — nothing lost. Log:
`ml/runs/ml6-h1/sweep-alpha.log`; per-alpha JSONs in
`research/iterations/alpha-v54-<α>-vs-v44.json`.

| α | W-L-D | win rate | 95% CI | as Black | as White | unique |
|---|-------|----------|--------|----------|----------|--------|
| 0.125 | 14-24-2 | 0.3750 | [0.2422, 0.5297] | 10-10-0 (0.500) | 4-14-2 (0.250) | 40/40 |
| 0.25 | 13-24-3 | 0.3625 | [0.2317, 0.5174] | 8-12-0 (0.400) | 5-12-3 (0.325) | 40/40 |
| **0.5** | **17-22-1** | **0.4375** | [0.2960, 0.5899] | 11-9-0 (0.550) | 6-13-1 (0.325) | 40/40 |

Win rate is U-shaped in α (0.375 / 0.3625 / 0.4375) —
the same shape v48's sweep showed (0.375 / 0.250 /
0.450), NOT the monotone-decreasing shape v44's sweep
showed. **Selection (§2 rule, best win rate): α=0.5.**
Registry updated (`scoring.py`,
`value_weight: 0.5`). The no-signal rule does not apply
(arms differ by up to 4 wins). The 40-game CIs overlap
heavily, so the α ranking is a point-estimate call, not
a proven ordering — and ALL THREE arms sit below 0.50
against v44, so the sweep's honest message is that v54
loses to v44 on dev seeds at every α, like v48. The
100-game clause-1 leg (validation seeds, openings) is
the real verdict.

Colour note: White is still the weak side vs v44 at
every α (0.250 / 0.325 / 0.325 vs Black 0.500 / 0.400 /
0.550) — the diag's closed White accuracy gap (§7.4)
has NOT (yet) shown up as played White strength against
v44. These are 20-game colour splits (SE ≈ 0.11), so
the battery's 50-game splits are the real read; but the
direction is a warning, not a confirmation.

### 6.3 Baseline test suites (before training)

Full suite run on this branch (niced, under load ~3.5,
1:44:25): pytest 502 passed / 1 skipped / 58 xfailed
with 2 failures, both `test_longrun.py` subprocess
`TimeoutExpired`s; npm 54/55 with 1 timeout failure;
tsc clean. All three failures are contention flakes:
the two pytest tests pass on re-run (2 passed in
238 s) and npm passes 55/55 on re-run. Baseline is
green; clause 4 re-runs the full suite at the end.
