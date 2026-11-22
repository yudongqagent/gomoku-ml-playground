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
