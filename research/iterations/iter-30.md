# Iteration 30 — v30: dedup + colour-balanced retrain of v8's recipe (ML batch 2, H1)

**Status: pre-registered — design, balancing method, holdout
design, training recipe, α sweep and eval seeds recorded BEFORE
any eval (see §1).** This is the H1 workstream of ML batch 2
(H2 is v31, the bigger value net on distilled labels; H3 is
v32, VCF-at-root for the value engine); work happens on branch
`research/ml-batch-2-h1` in the `gomoku-ml2-h1` worktree. v30
does NOT change the search, the leaf structure, the threat
check, sims/priors, or the training recipe — the ONLY intended
difference from v8 is the value net's training data (deduped +
colour-balanced, see §1.1) plus a re-swept α (see §1.4). The
v11/v25 pattern: same code, different training data + α.

**Method (carried over):** the fast 10-game loop — play 10
games, read EVERY move of every game, find THE losing move per
game (verified with the solver or a deeper search — never
guessed), patch the structural cause, record each critical
position + correct move as a regression test, repeat. Max 4
fast-loop iterations before validation, or stop when an
iteration finds no new patchable weakness class. For an ML
version the "patch" may be a data, architecture, or training
change, pre-registered like any other — and any such change
re-runs the battery from scratch on the new form. The fast loop
runs against v0 as well as v8 (the reference); each iteration
plays 10 games vs v8 AND 10 games vs v0. Fast-loop games use
development seeds (1001/2001, 1002/2002, ...) distinct from
the bar battery's pre-registered seeds (100/200), so no
development game overlaps an eval game. The full 4-clause
validation battery runs on the FINAL v30 only.

## 1. Design and pre-registration

**Comparison target: v8** (primary, the ML track's champion:
ladder Elo 0.0, vs v0 0.545, tactical 20/20 + 29/29 on the
CORRECTED suite — re-verified by the coordinator; see
`research/TRACKS.md`). The ML bar's clause 1 is head-to-head
vs v8 (normal rules, 100 games, alternating colours, win rate
> 0.50 with Wilson 95% CI lower bound > 0.50), clause 2 is
ladder Elo strictly higher than v8's 0.0 (identical 20
games/opponent: random, heuristic, mcts, mcts_sims=100),
clause 3 is tactical pass rate >= v8's (old suite 20/20, hard
suite 29/29, run at `--seed 100`, the battery protocol),
clause 4 is all test suites green (ml pytest, npm test, tsc).
**Secondary head-to-heads, also pre-registered:** v30 vs v0
(100 games), v30 vs v7 (100 games), and a 100-game swap match
vs v7 under the tempo judge (the harness default).
Non-transitivity is recorded prominently.

**The improvement: v30 = v8's ValueMCTS with the value net
retrained on the DEDUPED + COLOUR-BALANCED 10k dataset.**
Search, leaf structure (terminal cases exact, net value shrunk
by α in quiet positions only), the 2-ply threat check at the
root, 100 sims, c_puct 1.5, v1 priors — all v8's, unchanged
(see `ml/gomoku_ml/mcts_value.py` and
`research/iterations/iter-8.md` §1). Any difference from v8
is attributable to the training data (plus the re-swept α, see
§1.4).

**The claim under test (from iter-25 §5.5/§7):** v25 refuted
"more data" for this recipe, but the failure was specific:
White collapsed (vs v8 Black 0.73 / White 0.15), and the
training data is Black-skewed (Black wins 53.1%) and
duplicated (games weighted up to 7x). The hypothesis is that
the net sees mostly Black-win positions and evaluates White
positions poorly — too pessimistically and/or with too little
discrimination between good and bad White moves — so the
search cannot find saving defences as White. v30 removes
both suspected causes at once (dedup + balance); the
by-side-to-move holdout accuracy (§1.2) is the direct test of
the mechanism, and the battery is the verdict on strength.
If v30 passes, a follow-up can separate dedup from balance;
if it fails, both levers are closed together for this recipe.

### 1.1 Data plan (FIXED before training)

Source: the completed B1 10k dataset — 10,000 shards / 3,244
unique games at
`~/workspace/projects/gomoku-b3-b1/ml/runs/b3-b1-10k/data/`
(verified by v25, iter-25 §2.2). Built by
`ml/scripts/build_dedup_balanced_train.py`:

- **Dedup:** one shard per unique game (its first occurrence),
  grouped by MD5 of the `moves` array
  (`split_holdout_unique.split_games` — the same dedup v25
  used, so the unique-game set is identical by construction).
- **Holdout exclusion first:** the holdout games (§1.2) are
  removed BEFORE balancing, so the balance counts are over
  train unique games only.
- **Colour balance — DOWNSAMPLE Black-win games** (the
  majority winner class) to the minority (White-win) count,
  with a fixed RNG seed (0; `np.random.default_rng(0)`).
  Chosen over upweighting White-win games for three reasons:
  (1) it keeps v8's training recipe byte-identical — loss
  upweighting would need a trainer code change (per-sample
  weights), a second variable; (2) upweighting by duplicating
  White-win games would reintroduce duplication, defeating
  the dedup; (3) the data loss is small — the skew is mild
  (53.1% Black wins among shards), so downsampling removes
  only ~6% of unique train games (~170 of ~2,920). Draws
  (none in this dataset) would be kept unconditionally.
  The dropped games are listed in the build manifest, so the
  selection is auditable and reproducible.
- **Expected size:** ~2,738 unique train games (~117k
  samples) vs v25's 9,004 shards (~386k samples) and v8's
  3,915 shards (~167k samples). The position-level balance
  (side-to-move × target) is reported in the manifest as a
  check that game-level balancing produced position-level
  balance (each game alternates sides, so it should, up to
  game-length differences by winner).

**Exposure note (honest confound, recorded now):** dedup
reduces each unique game's exposures from ~15 (v8: 3,915
shards / ~1,304 unique train games ≈ 3x per epoch × 5
epochs; v25: 9,004 / 2,920 ≈ 3.08x × 5) to 5 (1x × 5
epochs). Epochs stay at 5 — the recipe is unchanged except
the data (see §1.3) — so the exposure reduction is part of
the intervention, not compensated by tuning. If v30 fails,
a 15-epoch dedup variant is the follow-up that separates
"dedup" from "fewer exposures"; it is NOT run in this
iteration (one variable at a time).

### 1.2 Holdout protocol (pre-registered — v25's split, reused exactly)

v25's split (iter-25 §1.2) is already dedup-aware (by unique
game; all copies of a holdout game excluded from train), so
v30 REUSES it exactly: the same 324 holdout unique games
(every 10th unique game in first-occurrence order), the same
180-game clean subset (holdout games first occurring at shard
index ≥ 4350 — games NEITHER v8's nor v25's net trained on),
the same 435-game leaky v8 holdout (labelled LEAKY, for
like-for-like comparison with v8's published 0.7165 only).
The build script recomputes the dedup and FAILS if the
holdout set does not match v25's `split-manifest.json`
exactly — the identical holdout makes the v8/v25/v30
before/after comparison exact (same positions, no
split noise). The val dirs are copied from the ML batch 1 H2
worktree (`gomoku-ml1-h2/ml/runs/ml1-h2-10k/val-*`) into this
worktree at `ml/runs/ml2-h1-10k/val-*` (gitignored data).

**By-side-to-move accuracy (the mechanism test, new in this
iteration):** `ml/scripts/eval_value_holdout.py` now also
reports sign agreement split by side to move (Black-to-move
vs White-to-move positions; the side is exact — Black plays
ply 0 and the sides alternate). Measured BEFORE training for
the v8 and v25 nets ("before") and after training for the v30
net ("after"), on the honest holdout AND the clean subset,
full planes AND no-history planes. v25's White-collapse
hypothesis predicts: v25's net shows a White-to-move accuracy
gap vs its Black-to-move accuracy, and v30's net closes it.
Holdout is diagnostic — the battery, not the holdout, is the
verdict.

**Checkpoint selection (pre-registered rule, same as v8/v25):**
after training, compute the holdout value sign-accuracy for
BOTH `best.pt` (best internal val loss) and `final.pt` on the
HONEST holdout; use the higher-accuracy one (no-history
measure, the one that predicts leaf quality) as the leaf net,
copied to `ml/runs/ml2-h1/value-net.pt` (the path the
registry entry points at); record BOTH accuracies here. Tie
→ best.pt. No other selection criterion is used.

### 1.3 Training config (pre-registered — v8's recipe, unchanged)

Config `ml/configs/ml2-h1-train.yaml` (copy of v8's
`b3-h2-train-4350.yaml` with only the run name, data dir, and
comments changed): smoke ResNet 16ch/2 blocks, trained from
scratch with `ml/scripts/train.py` (policy+value loss; only
the value head is used as the leaf): **5 epochs**, batch 64,
lr 0.002 cosine, warmup 20 steps, weight decay 1e-4,
augmentation on, internal val_fraction 0.05 (for best.pt
selection only — the game-level holdout above is the reported
accuracy), torch threads 1 (`OMP_NUM_THREADS=1
TORCH_THREADS=1`, per B1's finding). Run dir
`ml/runs/ml2-h1-value`. **Memmap path** (iter-25 §2.4):
train from the bundle built by
`ml/scripts/build_mmap_bundle.py` (`ml/runs/ml2-h1-10k/
data-mmap/`); the in-RAM load OOM-kills at this scale. No
architecture change, no hyperparameter tuning: the variable
under test is the data (dedup + balance).

### 1.4 α sweep (pre-registered — v25's protocol, both opponents)

Grid α ∈ {1.0, 0.5, 0.25, 0.125}, via
`ml/scripts/sweep_value_weight.py` (overrides
`player._mcts.value_weight` directly, so the registry records
the winner only AFTER the sweep). Two sweep runs, same seeds
(1001/2001), paired per α: 20 games vs v0 AND 20 games vs
v8. Selection rule: highest COMBINED win rate over the 40
games; ties broken toward LARGER α (v11's tie-break: prefer
using the net more when equal). The sweep winner is recorded
in the registry entry (`ENGINE_VERSIONS[30]["value_weight"]`)
before the battery. Outputs:
`research/iterations/sweep-v30-vs-v0.json`,
`research/iterations/sweep-v30-vs-v8.json`.

### 1.5 Integration (pre-registered)

- `ml/gomoku_ml/mcts_value.py` + `scoring.py`: port v25's
  per-version α (`ValueMCTS.__init__` takes `value_weight`,
  default `VALUE_WEIGHT` = 0.25, so v8 is byte-identical in
  behaviour) and the v25-style dispatch (per-version env var
  + `value_weight` passthrough) — ported from ML batch 1 H2
  (commits 68040f7/21e072a/1e3dc2d on
  `research/ml-batch-1-h2`), NOT from a merge (that branch
  predates batch 9; only the ML-track files were ported).
- `ml/gomoku_ml/scoring.py`: ONLY the `ENGINE_VERSIONS[30]`
  registry entry (`{"name": "scoring-v30", ...,
  "value_checkpoint": "ml/runs/ml2-h1/value-net.pt",
  "value_checkpoint_env": "GOMOKU_V30_CHECKPOINT",
  "value_weight": <sweep winner>}` — path relative to the repo
  root, resolved in `ScoringPlayer.__init__`) plus the
  dispatch update above. `baseline.py` is frozen and
  untouched; `mcts_heuristic.py`, `mcts.py`, `selfplay.py`
  untouched.
- `ml/gomoku_ml/train.py`: port v25's memmap path
  (pre-allocated `load_shards` + `_load_mmap_bundle` +
  lazy-index `train()`) from the same commits, plus
  `ml/scripts/build_mmap_bundle.py`,
  `ml/scripts/split_holdout_unique.py`,
  `ml/scripts/sweep_value_weight.py`,
  `ml/scripts/find_losing_moves.py` and their tests.
- `ml/scripts/eval_value_holdout.py`: by-side-to-move
  accuracy (§1.2).
- `ml/scripts/build_dedup_balanced_train.py` (new) +
  `ml/tests/test_dedup_balance.py` (new).
- Per-leaf net eval cost and per-move cost at 100 sims are
  measured and recorded before the matches
  (`ml/scripts/bench_value_leaf.py`).

### 1.6 Eval seeds (pre-registered)

h2h vs v8: v30=100 / v8=200; h2h vs v0: v30=100 / v0=200; h2h
vs v7: v30=100 / v7=200; ladder seed 100, 20 games/opponent,
mcts_sims=100; swap 100 games vs v7, balanced placement,
placement_seed 11, tempo judge; tactical `--seed 100`.

### 1.7 Eval order (pre-registered)

(1) Data build + verification (§2) — dedup count, holdout
match vs the manifest, balance counts, position-level check.
(2) BEFORE baseline: holdout accuracy by side to move for the
v8 and v25 nets on the honest holdout + clean subset (§2).
(3) Training (§1.3) + holdout accuracy (§1.2) + checkpoint
selection. (4) Leaf-cost measurement. (5) Tactical old+hard
(full dispatch, `--seed 100`). (6) α sweep (§1.4) — registry
updated to the winner. (7) FAST LOOP (§4): up to 4 iterations
of 10 games vs v8 AND 10 games vs v0. (8) FULL VALIDATION
BATTERY on the final v30: tactical re-run, h2h vs v8 100
games (`ml/scripts/fast_loop_match.py`, seeds 100/200) ->
`research/iterations/batch-v30-vs-v8.json`, 100 games vs v0
-> `batch-v30-vs-v0.json`, 100 games vs v7 ->
`batch-v30-vs-v7.json`, 100-game swap match vs v7
(`ml/scripts/selfplay_batch.py --swap-rule`, balanced,
placement_seed 11, tempo judge) ->
`batch-v30-vs-v7-swap.json`, ladder 20 games/opponent via
`ml/scripts/eval_version.py`. Any battery numbers measured
before the fast loop completes are INTERMEDIATE, labelled as
such; only the final battery counts for the bar.

## 2. Data build and before-baseline

### 2.1 Dedup + balance build (COMPLETE 2026-11-17)

`ml/scripts/build_dedup_balanced_train.py --data
~/workspace/projects/gomoku-b3-b1/ml/runs/b3-b1-10k/data
--split-manifest ml/runs/ml2-h1-10k/split-manifest-v25.json
--out ml/runs/ml2-h1-10k/data --balance-seed 0` (~100 s):

* 10,000 shards → **3,244 unique games** (6,756 duplicate
  shards) — reproduces v25's dedup count exactly.
* **Holdout verified: 324 unique games, manifest match** — the
  recomputed holdout set equals v25's
  `holdout_first_shard_indices` exactly, so the before/after
  comparison is on identical positions.
* Train unique games: **2,920** (Black wins 1,549 / White wins
  1,371 / draws 0) — the game-level skew is 53.0% Black.
* Balanced: **kept 2,742 games** (1,371 Black-win + 1,371
  White-win), **dropped 178 Black-win games** (seed 0;
  dropped indices in the manifest). Data loss 6.1%, as
  predicted (§1.1).
* **118,171 samples** in the train set.
* Position-level check: Black-to-move win 29,362 / loss
  30,409 (49.1% win); White-to-move win 30,409 / loss 27,991
  (52.1% win). Game-level balance is exact, but
  position-level balance is NOT exact: White-win games run
  slightly longer than Black-win games, so they contribute
  more positions of both kinds. The residual 3-point gap is
  much smaller than the original skew and is a game-length
  effect, not a winner effect — recorded honestly, not
  "fixed" by re-weighting positions (that would be a second
  intervention).
* Manifest: `ml/runs/ml2-h1-10k/dedup-balance-manifest.json`
  (gitignored data; kept/dropped first-shard indices listed).
* Memmap bundle built from the train dir
  (`ml/scripts/build_mmap_bundle.py`): 2,742 shards, 118,171
  samples, 914 MB at `ml/runs/ml2-h1-10k/data-mmap/`
  (planes (118171, 8, 15, 15), policy (118171, 225), values
  (118171,)).

### 2.2 BEFORE baseline — holdout accuracy by side to move (v8 and v25 nets)

`ml/scripts/eval_value_holdout.py` on the honest holdout
(324 games, 13,485 positions) and the clean subset (180
games, 7,468 positions — neither net trained on them).
Overall numbers REPRODUCE v25's published values exactly
(v25 honest 0.5720/0.5607, clean 0.5775/0.5599; v8 clean
0.5577/0.5707) — the pipeline is validated end to end.

**Honest holdout (324 games):**

| net | full | no-hist | full: Black-to-move | full: White-to-move | no-hist: Black | no-hist: White |
|---|---|---|---|---|---|---|
| v8 | 0.6390 | 0.5976 | 0.636 (4341/6825) | 0.642 (4276/6660) | 0.5852 | 0.6102 |
| v25 | 0.5720 | 0.5607 | 0.5861 (4000/6825) | 0.5575 (3713/6660) | 0.5458 | 0.5760 |

**Clean subset (180 games, the honest v8-vs-v25 comparison):**

| net | full | no-hist | full: Black-to-move | full: White-to-move | no-hist: Black | no-hist: White |
|---|---|---|---|---|---|---|
| v8 | 0.5577 | 0.5707 | 0.5504 (2081/3781) | 0.5652 (2084/3687) | 0.5551 | 0.5867 |
| v25 | 0.5775 | 0.5599 | **0.6078** (2298/3781) | **0.5465** (2015/3687) | 0.5618 | 0.5579 |

**Reading.** The White-collapse hypothesis is PARTLY
supported, and the clean subset is where it shows: v25's net
has a real White-to-move gap on FULL planes (−0.061 on
clean, −0.029 on honest), while v8 shows NO White gap
anywhere (White is better on all four cells). But on
NO-HISTORY planes the gap vanishes (clean −0.004, honest
+0.030) — and no-history is the leaf condition at depth < 2.
The leaf sees history at depth ≥ 2, so both views matter;
the mechanism test is mixed, not clean. (Caveat: positions
cluster by game, so the effective sample size is ~180 games,
not 7,468 positions — but a 0.061 gap at n≈3,700/side is
~3.9 SE even before clustering, so the full-planes gap is
real.)

**Direction of the error (one-off diagnostic,
`/tmp/v30_side_bias.py`, clean subset, full planes):** the
gap is driven by WHITE WINS, not White losses. When White is
to move and eventually wins, v25's net predicts mean +0.051
(sign acc 0.522) vs +0.139 (0.663) for Black-to-move wins —
it UNDERVALUES White's winning chances. On White-to-move
losses it is fine (mean −0.085, acc 0.571 ≈ Black's 0.552).
So the net cannot see White wins coming — consistent with
the game-level failure (as White, the search cannot find
winning counterplay because the leaf undervalues White's
winning lines). v8 shows no such asymmetry (White-win acc
0.589 vs Black-win 0.578). Prediction for v30: if balancing
works, the White-to-move WIN accuracy should rise toward the
Black level on full planes.

(No-history diagnostic, for completeness: v25's no-history
predictions are shifted strongly positive (means +0.21 to
+0.41) — without history planes the net overvalues the side
to move (Black-win acc 0.861, Black-loss acc 0.257). Both
nets behave oddly on zeroed history; the full-planes view is
the cleaner mechanism signal.)

Raw JSONs: `ml/runs/ml2-h1/holdout-honest-before.json`,
`ml/runs/ml2-h1/holdout-clean-before.json` (text header +
JSON dump, as the script prints).

## 3. Training, holdout, leaf cost, tactical, α sweep (TBD)

## 4. Fast loop (TBD)

## 5. v30 evaluation (FULL VALIDATION BATTERY — final v30 only) (TBD)

## 6. Bar check (v30 vs v8) (TBD)

## 7. What this means (TBD)
