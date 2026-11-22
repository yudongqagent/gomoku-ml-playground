# Iteration 55 — v55: retrain the v48 patch data with 3-ply labels (ML batch 6, H2)

**Status: IN PROGRESS — pre-registered; data regeneration not yet run.**
Comparison target is v44 (the ML track's champion and bar
reference: ladder Elo 98.1, vs v8 0.67, tactical 20/20 +
29/29 — see iter-44.md). This is the H2 workstream of ML
batch 6 (H1 is v54, the targeted White-collapse fix; H3 is
v56, parity-corrected 3-ply labels); work happens on branch
`research/ml-batch-6-h2` in the `gomoku-ml6-h2` worktree.
v55 differs from v48 ONLY in the labels (3-ply instead of
2-ply) on the same data mix — same PUCT search, same v1
priors, same sims (100), same c_puct (1.5), same threat
check, same terminal cases, same architecture and training
recipe — so the battery is a clean A/B of the label depth
on the patch mix, answering iter-48 §10's open question:
was v48's cost the label depth or the data mix?

**Method (carried over):** the fast 10-game loop — play 10
games, read EVERY move, find THE losing move per game
(solver/deeper-search verified, never guessed), patch,
regression-test, repeat. Fast loops run against v0 as well
as v44. Max 4 fast-loop iterations before validation, or
stop when an iteration finds no new patchable weakness
class. The full 4-clause validation battery runs on the
FINAL v55 only. Fast-loop games use development seeds
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
  `5deb26f3df67883f24f04800d5356da6` ✓ (matches iter-44;
  needed for the test suite).
- v48: `ml/runs/ml5-h1/value-net.pt`, md5
  `a8c7a0bcfd749ac774ec9762c829d672` (matches iter-48;
  copied for the §1.5 holdout diagnostic only — v48 is
  NOT a battery opponent; the v48 comparison is against
  its published numbers).

**No code ports needed.** This tree (coordinator branch
038232c: master + the ML batch 6 plan + the fixed
`eval_value_holdout.py`) already carries everything ML
batches 4–5 ported: `_rollout2_leaf_value` /
`_rollout3_leaf_value` in `ml/gomoku_ml/mcts_heuristic.py`,
`generate_distill_data.py label --label-mode
{rollout,rollout2,rollout3}`, the by-side-to-move
`eval_value_holdout.py` (FIXED on this branch — plane-3
side reading, see below), the memmap `train.py`,
`build_mmap_bundle.py`, `sweep_alpha.py`,
`find_losing_moves.py`, `targeted_playout.py`, and the
v37/v42/v43/v44 registry entries. `ml/gomoku_ml/baseline.py`
is FROZEN and untouched.

**The `eval_value_holdout.py` fix (coordinator branch,
038232c):** `load_shards` now reads the side to move from
plane 3 of the stored encoding (all 1.0 iff Black to move)
instead of the index parity `arange(len(planes)) % 2`,
which was wrong on distill shards (iter-48 §0: up to ~60%
attenuation of the true White gap). The fix fails loudly
if plane 3 is not a uniform 0/1 plane. The §1.5 by-side
holdout numbers therefore use TRUE sides, and no separate
`score_shards_by_side.py` port is needed.

**The data-regeneration finding (recorded BEFORE the
pre-registration — it changes the §1.1 plan):** the batch
tasking prescribed regenerating the v48 base data as FRESH
v7/v18 self-play games, on the premise that "the original
game records are gone". That premise is WRONG in one
material respect: v37's shards survive intact at
`gomoku-ml3-h2/ml/runs/ml3-h2/data/` (verified: v7 440
train + 48 holdout shards, 21,595 positions; v18 207 + 22,
9,932; random 90 + 10, 5,700 — total 37,227 positions,
33,849 train / 3,378 holdout, EXACTLY v37's record), and
every shard stores the FULL game move list (`moves`) and
winner. The games JSON files are gone, but the game
records themselves are recoverable from the shards. The
§1.1 plan therefore reconstructs the base games from the
shards instead of playing fresh games — with a pre-registered
verification gate (rollout2 re-labelling must reproduce
all 37,227 stored values with 0 mismatches) and a fallback
to fresh games per the tasking if the gate fails. If the
gate passes, v55's base is v48's base VERBATIM (not merely
in kind), the v48 comparison becomes a clean labels-only
A/B, and the tasking's "different base games" caveat does
not apply. The holdout split is also recoverable exactly:
the holdout shards are games 9, 19, 29, ... (every 10th
game), matching `generate_distill_data.py`'s
`(rec["game"]+1) % 10 == 0` rule — verified per source
before writing this log.

## 1. The question: label depth or data mix?

v48 (ML batch 5, H1) diagnosed a GENERAL White-side
value-head defect in v44's net, patched it with data (all
four battery legs' quiet positions, 2-ply labels, on top of
v37's 37,227-position base), and fixed the played White gap
— but Black fell by at least as much. The patch
redistributed strength across colours; it did not add any.
The open question (iter-48 §10): was the cost the 2-ply
labels or the data mix?

| Leg | v48 | v44 |
|---|---|---|
| vs v44 | 0.44 (41W-53L-6D) | — (bar reference) |
| vs v8 | — | 0.67 (66W-32L-2D) |
| ladder Elo | −34.9 [−141.4, 71.7] | 98.1 [−12.1, 208.2] |
| vs v0 | 0.545 (53W-44L-3D) | 0.58 (55W-39L-6D) |
| vs v7 | 0.265 (23W-70L-7D) | 0.45 (44W-54L-2D) |
| vs v37 | 0.32 (29W-65L-6D) | 0.31 (29W-67L-4D) |
| swap vs v7 | 0.31 (28W-66L-6D) | 0.39 (38W-60L-2D) |
| White vs v0 | 0.52 (gap closed) | 0.59 |
| White vs v7 | 0.27 (nominally better than Black) | 0.40 |

v55 retrains the SAME mix with 3-ply labels (v44's
recipe). The two candidate outcomes, pre-registered (§1.6):
if v55 plays at v44's level, the cost was the label depth;
if it plays at v48's level, the cost was the data mix.
Do NOT retry another 2-ply retrain on this mix (v48 IS that
experiment).

### 1.1 Data regeneration (pre-registered)

**Two sources, exactly as v48's mix:**

**(a) PATCH POSITIONS — from v44's four battery legs**
(tracked in this tree):
`research/iterations/batch-v44-vs-{v0,v7,v8,v37}.json`
(validation seeds 100/200, 100 games each). A new committed
script `ml/scripts/convert_battery_to_games.py` converts
each leg's records to the `generate_distill_data.py` games
format (game index, full `moves` list, winner mapped
black→1 / white→−1 / draw→0). The openings legs' `moves`
include the opening plies (iter-48 §6 verified 100/100
records in both legs; re-verified here: the vs-v37 leg's
record 0 has `opening` [112, 127, 128, 111] and 84 moves),
so the replay is exact. **Verification gate (pre-registered):
the quiet-position counts per leg MUST match iter-48 §7.1
EXACTLY — v0 5,116, v7 3,998, v8 3,753, v37 4,487 (total
17,354)** — because quiet selection depends only on the
moves and `_terminal_leaf_value`, not on the label mode.
If any leg's count differs, STOP and investigate the replay
before training; the labelling must reproduce the diagnosis
positions.

**(b) BASE DATA — reconstructed from v37's shards** (see
§0). A new committed script
`ml/scripts/reconstruct_distill_games.py` reads v37's
shards (train + holdout dirs, all three sources), extracts
the game index from the shard filename, the full `moves`
list, and the winner, and writes games JSONs
(`ml/runs/ml6-h2/games-{v7,v18,random}.json`, gitignored)
in the `generate_distill_data.py` format. The `seed` field
is provenance only (v7 50000+game, v18 60000+game, random
70000+game, per iter-44 §1.2 — the label command does not
use it). **Verification gate (pre-registered): label the
reconstructed games with `--label-mode rollout2` into a
scratch dir and compare all 37,227 values against the
stored shard values — 0 mismatches required** (same code
path, deterministic). If the gate fails, fall back to
fresh games per the tasking (v24's protocol: 488 v7
self-play at 50 sims seeds 50000+, 229 v18 self-play at
50 sims seeds 60000+, 100 random seeds 70000+, via
`generate_distill_data.py games`), the base is then
in-kind rather than verbatim, and the v48 comparison
carries the "different base games" caveat. If the gate
passes, no caveat applies — the base is v48's base
verbatim.

**NOT used for training:** the B1 data
(`~/workspace/projects/gomoku-b3-b1/ml/runs/b3-b1-10k/data/`)
— B1 data collapses White (v25, v31).

### 1.2 Labelling procedure (pre-registered, exact)

- **Label EVERYTHING with 3-ply** (`_rollout3_leaf_value`,
  v44's recipe — iter-44 §1.1): `generate_distill_data.py
  label --label-mode rollout3`. Both the reconstructed base
  games and the patch games. No 2-ply labels anywhere in
  the v55 training data.
- **Quiet positions only** (`_terminal_leaf_value == 0.0`),
  encoded with the real game history exactly as in training
  (`encoding.encode`). Policy targets one-hot on the move
  played (unused at policy_weight=0).
- **Holdout (game-level, v24's protocol):** every 10th game
  of each source routed to a `*-holdout` dir and NEVER
  trained on (`--holdout-every 10`). For the base this
  reproduces v37's exact split (48 v7 / 22 v18 / 10 random
  = 80 games, 3,378 positions); for the patch legs it
  holds out 10 games per leg (40 games, 1,751 positions —
  the same games v48 held out, since the game indices and
  the rule are identical).
- **Expected totals** (if both §1.1 gates pass): base
  37,227 quiet (33,849 train / 3,378 holdout); patch
  17,354 quiet (15,603 train / 1,751 holdout); **total
  train 49,452 positions** (v48's exact count); total
  holdout 5,129.
- **Caveat (recorded, not hidden — same as v48's):** the
  patch positions come from v44's battery games at
  validation seeds 100/200, and v55's battery replays the
  same seeds. v55 ≠ v44, so the games diverge from the
  first differing move; and the v24/v37 precedent is the
  same shape (trained on v7 self-play games, then played
  vs v7). The game-level holdout on the new positions is
  kept for the holdout diagnostic, not for the battery.

### 1.3 Architecture and training (pre-registered)

v44's recipe EXACTLY (iter-44 §1.3), only the data mix
differs:
- **Architecture:** ResNet **16ch/2 blocks** (126,759
  params), trained from scratch (random init — NOT from
  v8's, v24's, v37's, v44's, or v48's checkpoint; the
  labels are a different target on a different mix).
- **Loss:** value MSE only — `policy_weight: 0.0`,
  `value_weight: 1.0`.
- **Optimiser:** AdamW, lr 0.002 cosine, warmup 20 steps,
  weight_decay 1e-4, batch 64, **15 epochs**, augmentation
  on (dihedral), val_fraction 0.05 (random split within
  the train set, for best-checkpoint selection only),
  seed 0.
- **Config:** `ml/configs/ml6-h2.yaml` (v44's `ml4-h3.yaml`
  with run name/dir changed); run dir
  `ml/runs/ml6-h2-train/`; threads pinned to 1
  (`OMP_NUM_THREADS=1 TORCH_THREADS=1`).
  `checkpoint_every_epochs: 1` from the start (v24's
  resume lesson).
- **Memmap path (mandatory):** the train shards are bundled
  with `ml/scripts/build_mmap_bundle.py` into
  `ml/runs/ml6-h2/data-mmap/` (over `ml/runs/ml6-h2/
  data-train/`, symlinks to the seven train dirs) and
  training reads the bundle. The in-RAM load OOM-kills at
  this scale (iter-25 §2.4); do not train from the raw
  shards.
- **Checkpoint selection (pre-registered rule, same as
  v24/v37/v44/v48):** best-by-internal-val-loss
  (`best.pt`), copied to `ml/runs/ml6-h2/value-net.pt`
  (gitignored data, the path v55's registry entry points
  at), verified bit-identical in weights to `best.pt`.
- **Resume rule (v24's lesson):** if training is killed,
  resume from the latest per-epoch checkpoint with
  `--resume` and the REMAINING epochs (cosine schedule
  spans the full 15 via `schedule_epochs`); back up the
  pre-kill `best.pt` first, because the resumed run resets
  `best_val_loss` and would overwrite it at the next epoch
  regardless — the final best is chosen by comparing the
  resumed run's best val loss against the pre-kill best.

### 1.4 α sweep (pre-registered)

α (the `value_weight` shrink on the net's quiet-position
value) is swept over **{0.125, 0.25, 0.5}**, **40 games each
vs v44**, dev seeds 1001/2001 (no openings), via
`ml/scripts/sweep_alpha.py --version 55`. The grid brackets
BOTH relevant winners: v44's 0.125 (3-ply labels, base mix
— monotone decreasing 0.725 / 0.600 / 0.475) and v48's 0.5
(2-ply labels, this mix — U-shaped 0.375 / 0.250 / 0.450).
40 games per arm (not 20): ML batch 5's lesson is that
20-game sweeps rank noisily (winner's regression 0.55 →
0.475 at 100 games), and the plan requires ≥40 per arm.
Pick the best win rate; tie-break: the smaller α
(conservative — less optimizer's-curse exposure).
**No-signal rule:** if all three arms are within 2 wins of
each other (0.05 at 40 games), pick α=0.125 (v44's winner
— v55 uses v44's labels) and say so. The sweep SELECTS α;
the 100-game battery is the verdict. The registry entry's
`value_weight` is updated to the winner after the sweep.
(1.0 is NOT on the grid: v37's sweep already showed 1.0
losing on cooler labels, and v44's monotone shape says
hotter labels want cooler α.)

### 1.5 Holdout diagnostics (pre-registered)

Game-level holdouts (never trained on), targets = 3-ply
labels. v55 net = `ml/runs/ml6-h2/value-net.pt`. All via
the FIXED `ml/scripts/eval_value_holdout.py` (plane-3
TRUE sides — §0), full-planes and no-history views.
**Parity caveat (iter-44 §0): cross-depth sign agreement
is NOT a fit measure** — the +0.12 parity shift alone
drives it to ~0.49. For cross-depth comparisons, Pearson r
(shift-invariant) is the primary measure; sign agreement
is reported with the caveat.
- (a) **v55 net on its own holdout** (120 games, 5,129
  positions: 80 base + 40 patch) — in-distribution; the
  §1.7 stop rule applies here (sign accuracy vs chance).
  Report accuracy BY SIDE TO MOVE (Black/White) on both
  the full-planes and the no-history views — the mechanism
  check: does the 3-ply retrain KEEP v48's White-gap
  closure (pooled −0.008 on unseen patch games) or
  reintroduce v44's White deficit (−0.065…−0.103)?
  ALSO report the no-history view (v37's net collapsed
  without history, 0.195 pooled; v44's and v48's did
  not — check whether v55 inherits the robustness).
- (b) **v48's net on the same 3-ply holdout** — same
  positions, cross-net comparison: how much of the 3-ply
  function does the 2-ply-trained patch net already
  capture? (Pearson r primary.) A large r would say the
  3-ply labels add little information over 2-ply on this
  mix; a small r says the targets genuinely differ.
- (c) **v44's net on the same 3-ply holdout** — v44's net
  trained on 3-ply labels (base mix only): does it fit
  the patch positions as well as v55's net does? The
  patch positions are unseen for v44's net (it never
  trained on them) — a direct measure of what the patch
  data adds to the 3-ply fit.
No decision hinges on (b)–(c); they diagnose whether the
deeper labels were learned and whether the patch data
changes the fit. **Standing caveat: holdout does not
predict played strength — six instances (v31, v38, v42,
v43, v48, v49). The 100-game battery is the verdict.**

### 1.6 Mechanism decision rule (pre-registered)

The question is whether v48's cost was the label depth or
the data mix. The battery arbitrates, read against v48's
and v44's published numbers (§1 table). Primary read: the
vs-v44 clause-1 leg + the ladder Elo; secondary: vs v0,
vs v7, vs v37, swap vs v7; mechanism read: the by-colour
splits (does White stay fixed while Black recovers?).
- **Label depth was the cost** iff v55 plays at v44's
  level: vs v44 ≥ 0.50 (point estimate; the clause-1 CI
  rule is the bar's, not this call's) AND ladder Elo >
  98.1, with the secondary legs at or above v44's numbers
  (vs v0 ≥ 0.58, vs v7 ≥ 0.45, swap ≥ 0.39). The 2-ply
  labels on this mix were the problem; 3-ply labels
  recover v44-level play.
- **Data mix was the cost** iff v55 plays at v48's level:
  vs v44 ≈ 0.44 (within ±0.06), ladder Elo ≈ −34.9
  (within ±50), secondary legs at or near v48's (vs v0
  ≈ 0.545, vs v7 ≈ 0.265, swap ≈ 0.31). The mix itself
  costs the strength regardless of label depth.
- **Partial recovery** (between the two, or mixed legs):
  report the numbers honestly per leg; the mechanism is
  then "both contribute" with the leg pattern as the
  evidence. In particular, if White stays fixed (by-colour
  gap ≤ 0.10 on the heuristic legs, as v48's) while
  Black recovers toward v44's, the 3-ply labels fixed the
  redistribution even if the overall level stays below
  v44's — that is a real finding, not a failure.
- **White-gap check (pre-registered):** v48 closed the
  played White gap on the heuristic legs (v0: gap 0.05;
  v7: White nominally better). If v55's White gap
  re-opens to v44-like levels (≥ 0.15 on v0/v7 legs),
  the 3-ply labels reintroduced the White defect the
  patch data had fixed — record it as a label-parity
  effect, not a mix effect.

### 1.7 What would change the design

- If the distilled net's holdout sign-accuracy on its OWN
  (3-ply-label) holdout is at chance (~0.50), the
  distillation failed to learn the deeper label function
  — stop and report, do not battery a chance-level net.
- If either §1.1 verification gate fails, stop at that
  gate: the patch gate (quiet counts) means the replay is
  wrong — investigate before training; the base gate
  (rollout2 reproduction) means the reconstruction is
  wrong — fall back to fresh games per the tasking.
- If the α sweep's best is within noise of the worst (all
  three within 2 wins), pick α=0.125 per the §1.4
  no-signal rule.
- Fast-loop patches (data/arch/training changes) are
  allowed only as pre-registered-style changes recorded
  in §5 with the specific loss that motivated them. A
  retrain re-runs the α sweep and the battery from scratch
  on the new form. Do NOT retry another 2-ply retrain on
  this mix.

## 2. Eval seeds and battery (pre-registered)

**The ML bar (reference v44, Elo 98.1):** (1) h2h vs v44
≥ 100 games, win rate > 0.50 with Wilson 95% CI lower
bound > 0.50; (2) ladder Elo > 98.1; (3) tactical ≥
20/20 + 29/29 on the corrected suite; (4) all test suites
green. Fail any clause = rejected. Holdout numbers do NOT
predict played strength — the 100-game battery is the
verdict.

**Seeds:** h2h vs v44: v55=100 / v44=200; vs v0: v55=100 /
v0=200; vs v7: v55=100 / v7=200; vs v37: v55=100 / v37=200;
ladder seed 100, 40 games / opponent; swap vs v7, balanced
placement, placement_seed 11, tempo judge; tactical --seed
100. The vs-v44 and vs-v37 legs use the pre-registered
`research/openings-ml4.json` (10 openings) via `--openings`.
Per-game seeding + uniqueness audit on every leg; report
distinct-game counts and de-dup rates.

**Battery runbook** (validation seeds 100/200):
```
# Clause 1: vs v44, 100 games, pre-registered openings
selfplay_batch.py --engine-a v55 --engine-b v44 --games 100 --openings research/openings-ml4.json --seed-a 100 --seed-b 200 --out research/iterations/batch-v55-vs-v44.json
# Clause 2: ladder (40 games/opponent, mcts_sims 100)
eval_version.py --version 55 --ladder-games 40 --out research/iterations/ladder-v55.json
# Clause 3: tactical, both suites, seed 100
eval_tactical_full.py --version 55 --suite both --seed 100
# Clause 4: ml pytest + npm test + tsc
# Secondary: vs v0, vs v7, vs v37 (openings), swap vs v7 (balanced/tempo, placement_seed 11)
selfplay_batch.py --engine-a v55 --engine-b v0 --games 100 --seed-a 100 --seed-b 200 --out research/iterations/batch-v55-vs-v0.json
selfplay_batch.py --engine-a v55 --engine-b v7 --games 100 --seed-a 100 --seed-b 200 --out research/iterations/batch-v55-vs-v7.json
selfplay_batch.py --engine-a v55 --engine-b v37 --games 100 --openings research/openings-ml4.json --seed-a 100 --seed-b 200 --out research/iterations/batch-v55-vs-v37.json
selfplay_batch.py --engine-a v55 --engine-b v7 --games 100 --swap-rule --placement balanced --placement-seed 11 --judge tempo --seed-a 100 --seed-b 200 --out research/iterations/batch-v55-vs-v7-swap.json
```

## 3. Eval order (pre-registered)

(1) Data regeneration + both §1.1 verification gates.
(2) Labelling (§1.2) + memmap bundle + training (§1.3) +
holdout diagnostics (§1.5). (3) Leaf-cost measurement.
(4) α sweep (§1.4) — registry updated to the winner.
(5) FAST LOOP: up to 4 iterations of 10 games vs v44 AND
10 games vs v0 (dev seeds; the winning α's sweep leg
counts as fast-loop iteration 1 vs v44, as in v44/v48).
(6) FULL VALIDATION BATTERY on the final v55 (§2). Any
battery numbers measured before the fast loop completes
are INTERMEDIATE, labelled as such; only the final battery
counts for the bar.

## 4. Data regeneration and labelling (actuals)

**Data pipeline COMPLETE 2026-11-22 ~16:30 PST** (started
15:09 PST, ~81 min, no interruptions; log
`ml/runs/ml6-h2/data-pipeline.log`). Driver
`ml/runs/ml6-h2-data-pipeline.sh` (gitignored run
artifact); both §1.1 gates PASSED.

### 4.1 Base reconstruction — GATE 1 PASSES (verbatim)

`reconstruct_distill_games.py` over v37's shards
(`gomoku-ml3-h2/ml/runs/ml3-h2/data/`):

| Source | Games | Plies | vs iter-44 §2 |
|---|---|---|---|
| v7 | 488 | 25,277 | exact |
| v18 | 229 | 11,511 | exact |
| random | 100 | 11,009 | exact |
| **Total** | **817** | **47,797** | exact |

**Gate 1 (rollout2 reproduction): 0 mismatches / 37,227
positions, max |Δ| 0.00e+00** (v7 0/21,595, v18 0/9,932,
random 0/5,700 — bit-identical values). The reconstruction
is VERBATIM: v55's base is v48's base exactly, not merely
in kind. The tasking's "different base games" caveat does
NOT apply — the v48 comparison is a clean labels-only A/B.
No fresh games were played (the fallback was not needed).

### 4.2 Patch conversion — GATE 2 PASSES (exact counts)

`convert_battery_to_games.py` over v44's four battery legs
(ply counts match iter-48 §7.1 exactly: 6,342 / 4,896 /
4,599 / 5,429). **Gate 2 (quiet counts): 5,116 / 3,998 /
3,753 / 4,487 — EXACT match with iter-48 §7.1.** The
labelling reproduces the diagnosis positions.

### 4.3 Labelling (rollout3 everywhere)

`generate_distill_data.py label --label-mode rollout3`
over all seven games files (game-level holdout every 10th
game):

| Source | Games | Plies | Quiet positions | ±1 labels | Held-out games |
|---|---|---|---|---|---|
| v7 (base) | 488 | 25,277 | 21,595 (19,545 train / 2,050 holdout) | 1,567 (7.3%) | 48 |
| v18 (base) | 229 | 11,511 | 9,932 (9,142 train / 790 holdout) | 842 (8.5%) | 22 |
| random (base) | 100 | 11,009 | 5,700 (5,162 train / 538 holdout) | 1,949 (34.2%) | 10 |
| vs v0 (patch) | 100 | 6,342 | 5,116 (4,546 train / 570 holdout) | 346 (6.8%) | 10 |
| vs v7 (patch) | 100 | 4,896 | 3,998 (3,677 train / 321 holdout) | 336 (8.4%) | 10 |
| vs v8 (patch) | 100 | 4,599 | 3,753 (3,186 train / 567 holdout) | 359 (9.6%) | 10 |
| vs v37 (patch) | 100 | 5,429 | 4,487 (4,194 train / 293 holdout) | 331 (7.4%) | 10 |
| **Total** | **1,217** | **69,063** | **54,581 (49,452 train / 5,129 holdout)** | **5,730 (10.5%)** | **120** |

Cross-checks: the base ±1 counts match iter-44 §2 exactly
(1,567 / 842 / 1,949); the patch train/holdout splits
match iter-48 §7.1 exactly (4,546/570, 3,677/321,
3,186/567, 4,194/293); **total train 49,452 = v48's exact
count**. The 3-ply labels are hotter than v48's 2-ply
labels on the patch legs (±1: 6.8–9.6% vs 5.3–8.0%) — the
5-ply horizon finds more forced wins, as in iter-44.
**The parity shift is in the data:** train label means are
positive on every source (v7 +0.0881, v18 +0.0586, random
+0.2798 — all three EXACTLY iter-44 §2's values; vs-v0
+0.0969, vs-v7 +0.0783, vs-v8 +0.0949, vs-v37 +0.0841);
fraction positive 0.61–0.77. The net will learn a value
function centred near +0.10, like v44's — the α sweep
(§6) is where that shift gets absorbed.

Memmap bundle `ml/runs/ml6-h2/data-mmap/`: 1,097 shards,
49,452 samples, planes (49452, 8, 15, 15) — identical
shard/sample counts to v48's bundle.

## 5. Training (actuals)

**Training COMPLETE 2026-11-22 ~17:49 UTC** (started
16:34 UTC; one runtime restart killed the run at step
10,840/11,025 — epoch 15, ~90 seconds short of finishing
— and the pre-registered resume rule fired: `best.pt`
backed up first (verified bit-identical to
checkpoint-10290.pt, epoch 14, val 0.0919), resumed from
checkpoint-10290.pt with `--epochs 1`; the resumed
epoch-15 val 0.0915 beat the pre-kill best, so the resumed
run's best IS the global best and the `--out` copy is
correct): config `ml/configs/ml6-h2.yaml`, 15 epochs,
batch 64, memmap bundle (49,452 samples; 46,979 train /
2,473 val, seed-0 split). Run dir `ml/runs/ml6-h2-train/`;
log `ml/runs/ml6-h2/train.log`. 11,025 steps, ~190–392
s/epoch (slow — H3's training ran concurrently for most
of the run).

| Epoch | Train loss | Val loss |
|---|---|---|
| 1 | 0.1033 | 0.0997 |
| 2 | 0.1015 | 0.0981 |
| 3 | 0.0999 | 0.0977 |
| 4 | 0.0992 | 0.0978 |
| 5 | 0.0983 | 0.0986 |
| 6 | 0.0977 | 0.0967 |
| 7 | 0.0962 | 0.0957 |
| 8 | 0.0956 | 0.0939 |
| 9 | 0.0944 | 0.0946 |
| 10 | 0.0929 | 0.0945 |
| 11 | 0.0917 | 0.0935 |
| 12 | 0.0906 | 0.0928 |
| 13 | 0.0898 | 0.0924 |
| 14 | 0.0894 | 0.0919 |
| 15 | 0.0891 | **0.0915** |

Val loss decreased overall (small bumps at epochs 4, 5,
9) — no overfitting by epoch 15. **Best checkpoint:
epoch 15 (`best.pt`, val 0.0915)** — the `--out` copy at
`ml/runs/ml6-h2/value-net.pt` is verified bit-identical in
weights to `best.pt` AND to `final.pt` (max |Δ| 0.0 both),
so the registry path holds the best-by-val-loss checkpoint
as pre-registered. (Absolute val loss is not comparable
to v48's 0.0841 or v44's 0.1136 — different targets and
mixes: v55's 3-ply labels on the patch mix sit between
v48's cooler 2-ply target and v44's hotter base-only 3-ply
target.) Architecture verified: ResNet 16ch/2 blocks,
126,759 params, 48 tensors — identical counts and shapes
to v37's, v44's, and v48's checkpoints.
Checkpoint md5: `8ead4de5222d8ef230215b2f0b746b23`.

## 6. Holdout diagnostics (actuals)

Game-level holdouts (never trained on), targets = 3-ply
labels. v55 net = `ml/runs/ml6-h2/value-net.pt`; v48 net =
`ml/runs/ml5-h1/value-net.pt`; v44 net =
`ml/runs/ml4-h3/value-net.pt`. Sign agreement + MSE via the
FIXED `eval_value_holdout.py` (plane-3 TRUE sides — §0);
Pearson r via `ml/scripts/holdout_corr.py`. Raw outputs in
`ml/runs/ml6-h2/holdout.log`. Sanity check: this run
reproduces v44's published 3-ply numbers EXACTLY on its
own base holdout (v7 0.7460, v18 0.7163, random 0.7786 —
full planes; no-history 0.7360/0.7035/0.7880; MSEs
0.0847/0.1057/0.2624) — the eval setup is sound.

### (a) v55 net on its own holdout (in-distribution)

| Holdout | n pos | v55 sign (full) | v55 sign (no-hist) | v55 r | v44 sign (full) | v48 sign (full) |
|---|---|---|---|---|---|---|
| v7 (base) | 2,050 | 0.7350 | 0.7381 | 0.1950 | 0.7460 | 0.2840 |
| v18 (base) | 790 | 0.7047 | 0.6881 | 0.0646 | 0.7163 | 0.3171 |
| random (base) | 538 | 0.7730 | 0.7992 | 0.3620 | 0.7786 | 0.5460 |
| vs-v0 (new) | 570 | 0.7495 | 0.7298 | −0.0197 | 0.7475 | 0.2702 |
| vs-v7 (new) | 321 | 0.7880 | 0.7342 | 0.2074 | 0.7880 | 0.2405 |
| vs-v8 (new) | 567 | 0.6719 | 0.6581 | 0.0762 | 0.6798 | 0.3300 |
| vs-v37 (new) | 293 | 0.7552 | 0.7517 | 0.2564 | 0.7552 | 0.2448 |
| **base pooled** | **3,378** | **0.7339** | — | — | **0.7442** | — |
| **new pooled** | **1,751** | **0.7338** | — | — | **0.7356** | — |
| **all pooled** | **5,129** | **0.7339** | **0.7281** | — | **0.7413** | **0.3164** |

**Verdict: the distillation learned the 3-ply label function
— the §1.7 stop condition (chance-level holdout) does NOT
trigger.** Pooled in-distribution sign agreement 0.7339 vs
0.50 chance. It is slightly LOWER than v44's 0.7442 on the
base holdout and essentially identical on the patch
holdout (0.7338 vs v44's 0.7356) — v44's net, which never
trained on the patch positions, fits them as well as
v55's net does. **The patch data adds nothing to the 3-ply
holdout fit.** If v55 plays differently from v44, it will
not be because it fits these positions better on holdout.
**The v55 net is HISTORY-ROBUST** (pooled no-history drop
0.006: 0.7339 → 0.7281; v7 and random no-history are
BETTER than full) — the v37 collapse is absent, like
v44's and v48's nets.

**By TRUE side (v55 net, full planes):**

| Holdout | Black | White | White − Black |
|---|---|---|---|
| v7 | 0.7730 | 0.6964 | −0.077 |
| v18 | 0.6684 | 0.7410 | +0.073 |
| random | 0.7687 | 0.7774 | +0.009 |
| vs-v0 | 0.6148 | 0.8745 | +0.260 |
| vs-v7 | 0.7310 | 0.8552 | +0.124 |
| vs-v8 | 0.8765 | 0.4829 | −0.394 |
| vs-v37 | 0.7677 | 0.7407 | −0.027 |
| **pooled** | **0.7469** | **0.7208** | **−0.026** |
| **patch pooled** | **0.7466** | **0.7208** | **−0.026** |

The per-leg cells swing wildly (vs-v0 White +0.26, vs-v8
White −0.39) — 10 games per leg is heavily game-clustered,
exactly as iter-48 §7.3 warned; the pooled number is the
read. **The 3-ply retrain did NOT reintroduce a large White
deficit on the holdout:** pooled White gap −0.026, vs
v48's −0.008 (2-ply targets) and v44's diagnosis gap of
−0.065…−0.103 on the full 17,354-position set. (v44's net
on this same holdout subset shows −0.022 pooled — the
holdout's per-leg cells are noise-dominated, and the
pooled holdout gap is smaller than the full-set diagnosis
gap for both nets.) The mechanism check (§1.5a) is
therefore inconclusive on holdout alone — the battery's
played by-colour splits are the real read. **Caveat
(standing): holdout does not predict played strength —
the battery is the verdict.**

**A true-side correction to iter-44 §4:** with TRUE sides
(the fixed eval), v44's net shows a White deficit on v7
positions (0.7070 vs 0.7845, gap −0.078, n≈950/side, ~4
SE) — iter-44's "no White collapse" (Black 0.7463 / White
0.7421, gap 0.004) was measured with the contaminated
index-inferred split. The v18 cell goes the other way
(+0.096), so the pooled base gap is small (−0.016), but
the v7-cell deficit is real and was hidden by the
contamination.

### (b) v48's net on the 3-ply holdout (cross-depth)

Sign agreement 0.24–0.55 (pooled 0.3164) — BELOW chance on
engine positions, exactly as the parity caveat predicted
(v48's net centres near −0.06, the 3-ply targets near
+0.10). **Pearson r (the pre-registered primary measure):
v7 0.0626, v18 0.0680, random 0.4400, vs-v0 −0.0633,
vs-v7 0.0395, vs-v8 −0.1175, vs-v37 −0.0017.** On engine
positions the 2-ply-trained patch net captures almost none
of the 3-ply function in correlation terms (random's 0.44
is the ±1-terminal-dominated exception). **The 3-ply labels
are a genuinely different target on this mix** — the
label-depth change is real, not cosmetic. If v55 plays at
v48's level, it will not be because the labels were
equivalent.

### (c) v44's net on the 3-ply holdout

Sign agreement 0.7442 base / 0.7356 patch pooled (table
(a)); Pearson r v7 0.2027, v18 0.1480, random 0.3508,
vs-v0 0.0176, vs-v7 0.0877, vs-v8 −0.0109, vs-v37 0.1682.
v44's net fits the patch positions about as well as
v55's does on every measure (sign and r) — the patch data
does not improve the 3-ply fit. Note the near-zero r on
vs-v0 for ALL THREE nets (−0.020 / −0.063 / +0.018)
despite sign agreement ~0.75: the nets get the sign right
but the magnitudes don't track (distillation compression —
nets predict small values ~0.08 while the targets carry ±1
spikes), the same pattern iter-44 §4 noted.

**Tooling note:** while fixing `holdout_corr.py` (its first
version called `load_model_from_checkpoint` as if it
returned a bare model; it returns `(model, payload)`), a
runtime security notice fired on a repo-grep exec result.
Per the tasking, this is the known false-positive pattern
(ML batch 5 H2): the flagged item was skipped, the fact
was re-verified by plain file viewing (which runs nothing),
and the work continued — the notice did not stop the task.

## 7. α sweep and fast loops

**Leaf cost (2026-11-22, `ml/runs/ml6-h2/leaf-bench.txt`):**
`bench_value_leaf.py --checkpoint ml/runs/ml6-h2/value-net.pt
--positions 20 --leaf-samples 200 --sims 100`: per-move at 100
sims mean 9,879 ms / median 8,225 ms / p95 22,979 ms; net leaf
evals per move mean 80.7 (17–99); per-leaf net eval median
17.4 ms. Slower than v48's bench (mean 4,120 ms) and v44's
(5,190 ms) — the machine was heavily contended at measure
time (two sibling workstreams + two cross-track matches
running), not a net difference (same 16ch architecture;
80.7 evals/move matches v48's exactly). Battery games run
~2–5 min each under contention; the 100-game matches are
the long pole of this workstream.

### 7.1 α sweep (2026-11-22, ~18:21–22:36 UTC / 10:21–14:36 PST)

`sweep_alpha.py --version 55 --alphas 0.125 0.25 0.5 --games 40
--opponent v44 --seed-a 1001 --seed-b 2001` (dev seeds, no
openings). Two VM restarts killed the run mid-leg; the
driver resumed from the saved JSON each time with per-game
seeding (α=0.125 from game 2 at 1W-0L-1D; α=0.25 from game
17 at 11W-6L-0D) — nothing lost. Early pace was ~5–9
min/game under 5-way contention (two sibling workstreams +
two cross-track matches); after the siblings' phases
finished it fell to ~60–70 s/game. Log:
`ml/runs/ml6-h2/sweep-alpha.log`; per-alpha JSONs in
`research/iterations/alpha-v55-<α>-vs-v44.json`.

| α | W-L-D | win rate | 95% CI | as Black | as White | unique |
|---|-------|----------|--------|----------|----------|--------|
| 0.125 | 20-18-2 | 0.5250 | [0.3750, 0.6706] | 11-8-1 (0.575) | 9-10-1 (0.475) | 40/40 |
| **0.25** | **23-17-0** | **0.5750** | [0.4219, 0.7149] | 11-9-0 (0.550) | 12-8-0 (0.600) | 40/40 |
| 0.5 | 21-19-0 | 0.5250 | [0.3750, 0.6706] | 11-9-0 (0.550) | 10-10-0 (0.500) | 40/40 |

Win rate is inverted-U in α (0.525 / 0.575 / 0.525) —
NOT the monotone-decreasing shape v44's sweep showed
(0.725 / 0.600 / 0.475 vs v8) and NOT v48's U-shape
(0.375 / 0.250 / 0.450 at 20 games vs v44). **Selection
(§1.4 rule, best win rate): α=0.25.** The no-signal rule
does not fire (23 wins vs 20 — a 3-win gap, outside the
within-2-wins band). The 40-game CIs overlap heavily, so
the α ranking is a point-estimate call, not a proven
ordering — but the honest headline is the level, not the
ranking: ALL THREE arms sit ABOVE 0.50 against v44, where
v48's arms all sat below 0.50 (0.375 / 0.250 / 0.450).
First played signal that the 3-ply retrain plays at v44's
level, not v48's — the label-depth reading of the §1.6
mechanism rule, to be confirmed or refuted by the
100-game clause-1 leg (validation seeds, openings).
Registry: `value_weight` was already 0.25 as the
placeholder; the comment now records it as the sweep
winner (no value change).

Two observations: (1) at the winning α=0.25, v55's White
OUTPERFORMS its Black (0.600 vs 0.550) — no White
collapse at any α (worst White 0.475 at α=0.125), consistent
with the §6 holdout finding (pooled true-side gap −0.026).
The 3-ply labels did not reintroduce the White defect the
patch had closed. (2) Two 225-ply full-board marathon
draws occurred in the α=0.125 leg only (games 1 and 24);
the 0.25 and 0.5 legs had none (max 100
plies in the 0.5 leg). The v55-vs-v44 matchup can still
produce marathons, but far less often than v48-vs-v44
(3 of 20 in v48's 0.125 leg).

### 7.2 Fast loop iteration 1 (α=0.25, seeds 1001/2001)

- vs v44: the α=0.25 sweep leg itself (40 games, same dev
  seeds) — **23W-17L-0D (0.5750)**. As Black 11-9-0, as
  White 12-8-0. All 40 games unique.
- vs v0: **3W-7L-0D (0.300)**, CI [0.1078, 0.6032], 160 s
  (`research/iterations/fast-v55-vs-v0-iter1.json`). Colour
  split: as Black 2W-3L-0D (0.400), as White 1W-4L-0D
  (0.200). Well below v48's 0.55 and v44's 0.55 in the
  same loop — but 10 games with a CI spanning [0.11, 0.60],
  so the 100-game battery leg is the real read, not this.
- Loss classification (both matches):
  **24 losses, 24 via open four, 0 missed blocks, 0 clean
  missed preventions in 125 threatened turns** (vs v0:
  7 losses, 38 threatened turns; vs v44: 17 losses, 87
  threatened turns). Defence as measured is clean — the
  losses are positional drift into unstoppable open
  fours, not blundered blocks. Same pattern as v37/v44/v48.
- Losing moves (`find_losing_moves.py`, VCF solver depth
  4; raw `ml/runs/ml6-h2/losing-moves.txt`): **18 of 24
  losses were already lost** at the first VCF moment.
  **SIX genuine blunders:** vs v44 game 5 (White, 65
  plies), ply 59 — (9,10), 2 saves; vs v44 game 6 (Black,
  98 plies), ply 82 — (7,3), 6 saves; vs v44 game 7
  (White, 67 plies), ply 61 — (11,3), 5 saves; vs v44
  game 18 (Black, 40 plies), ply 34 — (11,9), 8 saves;
  vs v44 game 26 (Black, 70 plies), ply 60 — (8,3), 1
  save; vs v0 game 7 (White, 23 plies), ply 15 — (10,10),
  2 saves. The v0 losses are 6-of-7 already lost, at very
  early plies (12, 15, 15, 18, 21, 38) in short games —
  v55 drifts into lost positions against v0's play early,
  it does not blunder late.

**Workup — vs-v0 game 7, ply 15 (VERIFIED):**
`diag_root.py` (raw `ml/runs/ml6-h2/diag-v0-g7-ply15.txt`)
shows the actual (10,10) won on visits (12, our-q −0.114,
prior 0.1238) while both saves have BETTER our-q ((10,5):
6 visits, −0.049; (12,5): 6 visits, −0.051) and a +0.250
terminal leaf (four-creator) against the actual's 0.000.
The search's q ranks the saves first; the prior-driven
visit allocation picks the actual.

**Workup — vs-v44 game 18, ply 34 (VERIFIED):**
`diag_root.py` (raw `ml/runs/ml6-h2/diag-v44-g18-ply34.txt`)
shows the identical pattern: actual (11,9) 15 visits,
our-q +0.033, prior 0.0698; save (12,5) 11 visits, BETTER
our-q +0.057, +0.250 terminal leaf on every leaf variant
against the actual's 0.000 / v22 −0.118. The q gap is
small (+0.024), so the higher prior dominates the visit
allocation.

**Fast-loop verdict: STOP after iteration 1.** Both
workups show the same mechanism — the leaf and backed-up
q rank a save above the played move, but visit-count
selection picks the higher-prior move on a small q gap.
That is the known visit-allocation / blunder-dilution
pattern (v40's finding): the fix would be a q-based root
decision or a backup change, both CLOSED levers, and the
ML track's lever is the net, not the search. No patchable
weakness class remains. The vs-v0 0.300 (10 games) is
noted for the battery leg to adjudicate, not a fast-loop
finding to chase.
