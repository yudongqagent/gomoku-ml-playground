# Iteration 48 — v48: diagnose the White collapse vs heuristic opponents, then patch the mechanism found (ML batch 5, H1)

**Status: IN PROGRESS — pre-registered; diagnosis not yet run.**
Comparison target is v44 (the ML track's champion and bar
reference: ladder Elo 98.1, vs v8 0.67, tactical 20/20 +
29/29 — see iter-44.md). This is the H1 workstream of ML
batch 5 (H2 is v49, 4-ply labels; H3 is v50, the α
ablation); work happens on branch `research/ml-batch-5-h1`
in the `gomoku-ml5-h1` worktree. v48's form is decided by
the diagnosis (§2–§3): it is either a data patch (new net
trained on the mechanism's positions), a search-side patch
that touches no CLOSED lever, or — if the diagnosis finds
no patchable mechanism — the pre-registered α-ablation
fallback (§3.3). Whatever the form, search, priors, sims,
threat check, and terminal cases stay v8/v44 unchanged
unless the §3 decision tree says otherwise, so the battery
is a clean A/B of the patch.

**Method (carried over):** the fast 10-game loop — play 10
games, read EVERY move, find THE losing move per game
(solver/deeper-search verified, never guessed), patch,
regression-test, repeat. Fast loops run against v0 as well
as v44. Max 4 fast-loop iterations before validation, or
stop when an iteration finds no new patchable weakness
class. The full 4-clause validation battery runs on the
FINAL v48 only. Fast-loop games use development seeds
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

**No code ports needed.** This tree (master + the ML batch
5 plan) already carries everything ML batch 4 ported:
`_rollout2_leaf_value` / `_rollout3_leaf_value` in
`ml/gomoku_ml/mcts_heuristic.py`, `generate_distill_data.py
label --label-mode {rollout,rollout2,rollout3}`, the
by-side-to-move `eval_value_holdout.py`, the memmap
`train.py`, `build_mmap_bundle.py`, `sweep_alpha.py`,
`find_losing_moves.py`, `targeted_playout.py`, and the
v37/v42/v43/v44 registry entries. `ml/gomoku_ml/baseline.py`
is FROZEN and untouched.

**A flaw found in the v30 method as applied to distill
holdouts (recorded BEFORE the diagnosis — it shapes §2):**
`eval_value_holdout.py::load_shards` infers the side to
move as `arange(len(planes)) % 2` — correct only when the
shard stores EVERY ply of the game. Distill shards store
only QUIET positions (`generate_distill_data.py cmd_label`
skips plies where `_terminal_leaf_value != 0.0`; v24's
yield was 37,227 quiet of 47,797 plies, ~78%), so after
the first skipped non-quiet ply the inferred side FLIPS,
and every further skip flips it again. The by-side numbers
in iter-37 §4 and iter-44 §4 (e.g. v44's "Black 0.7463 /
White 0.7421 — no White collapse") were computed with
this contaminated split and are unreliable: the true
side gap is attenuated toward zero by the flips. The §2
diagnosis therefore uses TRUE sides (recorded at replay
time, ply t → Black iff t even) and additionally reports
the index-inferred split alongside, to quantify the
contamination. No conclusion of this iteration will rest
on the index-inferred split.

## 1. The problem (ML batch 4's numbers)

The distilled nets collapse as White against heuristic
opponents while holding White against ML opponents:

| Version | White vs v8 | White vs v0 | White vs v7 | White vs v37 | White vs ladder MCTS |
|---|---|---|---|---|---|
| v42 | 0.59 | 0.28 | 0.15 | 0.12 (6W-44L) | 0.25 |
| v43 | 0.60 | 0.21 | — | 0.28 | 3-16-1 |
| v44 | 0.67 | 0.59 | 0.40 | **0.16** (8W-42L) | — (ladder leg 11-9-0 overall) |

Notes: (a) v44's worst White collapse is vs v37 — an ML
opponent, not a heuristic one — so the "vs heuristic"
framing from v42/v43 does not cover v44; the diagnosis
(§2) therefore splits by individual opponent (v0, v7,
v8, v37), not just by family. (b) v44's holdout by-side
numbers showed no White gap (iter-44 §4) — but see §0:
that split was contaminated. (c) v44's fast-loop blunder
workups (iter-44 §5.2) found no value-head failure in
three blunders — but they predated the v37 leg and
covered only 3 of 42 White losses vs v37.

**Two candidate mechanisms:**
- **A — the value head misreads greedy-heuristic
  positions.** The distilled nets trained on v7/v18
  self-play + random positions, labelled by the rollout
  leaf. Positions arising against greedy heuristic
  opponents (v0's pure heuristic, v7's gradient-leaf
  MCTS) may sit off that training distribution in a
  way that specifically corrupts the White-to-move
  value — e.g. the net undervalues White's winning
  chances (v30's pattern: the White gap was driven by
  White-WIN positions) or overvalues White's chances
  and under-defends (iter-44's hypothesis for v44:
  the optimistic 3-ply labels hurt most from behind).
- **B — a search-horizon effect.** The value head is
  fine; the 100-sim search cannot see the forcing
  lines heuristic opponents create as White's
  position deteriorates, so the backed-up root q
  mis-ranks (blunder dilution / visit allocation —
  v40's class, whose leaf-level and backup-level fixes
  were CLOSED in batch 12 as v46/v47).

## 2. Diagnosis (pre-registered)

### 2.1 Positions — no new game generation

Reuse v44's four complete 100-game battery legs from ML
batch 4 (validation seeds 100/200, committed in this
tree): `batch-v44-vs-v0.json`, `batch-v44-vs-v7.json`,
`batch-v44-vs-v8.json`, `batch-v44-vs-v37.json`. Every
record carries the full move list, so positions are
recovered by replay. v44's net never trained on these
positions (it trained on v24's 817 games), so they are
unseen data for both subjects. Expected yield: ~400
games × ~45 plies × ~78% quiet ≈ **14,000 quiet
positions** (~3,500 per opponent, ~1,750 per
opponent × side).

The ladder MCTS is NOT a diagnosis cell: the ladder
JSON records no moves, and the ladder MCTS
(`baseline.HeuristicMCTSPlayer`) is not a
`selfplay_batch` engine. v0 (no-search heuristic) and
v7 (heuristic MCTS) bracket the heuristic family; the
ladder MCTS sits between them. Recorded as a
limitation, not silently dropped.

### 2.2 Subjects, targets, measurements

- **Primary subject: v44's net** (`ml/runs/ml4-h3/
  value-net.pt`), scored against its OWN target, the
  3-ply rollout-leaf label (`_rollout3_leaf_value`).
- **Second subject: v37's net** (`ml/runs/ml3-h2/
  value-net.pt`), scored against its OWN target, the
  2-ply label (`_rollout2_leaf_value`).
- Both labels are computed for every position (the
  2-ply label costs ~10 ms, the 3-ply ~12 ms; the full
  run is ~6 min). Positions are encoded with real
  game history exactly as in training
  (`encoding.encode`, history = boards before each
  earlier ply, last move = previous move).
- **Primary measurement:** sign agreement between the
  net's value and the teacher label, by opponent ×
  side to move (TRUE side), full planes. Also the
  no-history view (planes 2, 4–7 zeroed) for the same
  cells, to check whether any gap is history-mediated.
- **Secondary cuts (pre-registered):**
  (a) accuracy split by eventual game outcome from the
  side-to-move perspective (positions the side
  eventually WINS vs LOSES) — v30's cut: if the White
  gap is driven by White-WIN positions, the net
  undervalues White's winning chances;
  (b) the index-inferred side split alongside the
  true-side split, to quantify the §0 contamination;
  (c) Pearson r (net value vs teacher label) per cell,
  as a magnitude check alongside sign agreement.
- **Tool:** a new script
  `ml/scripts/diag_white_collapse.py` (committed),
  reusing `encoding.encode`,
  `mcts_heuristic._rollout2/3_leaf_value`,
  `_terminal_leaf_value`, and
  `architectures.load_model_from_checkpoint`. It
  writes a JSON artifact
  `ml/runs/ml5-h1/diag-white-collapse.json`
  (gitignored raw numbers) and prints the cell table.
  `eval_value_holdout.py` is NOT used for the
  diagnosis (its side inference is wrong on distill
  shards — §0).

### 2.3 Decision rule (pre-registered thresholds)

Let W_heur = v44 net's White-to-move sign agreement on
heuristic-opponent positions (v0+v7 pooled), W_ml =
same on ML-opponent positions (v8+v37 pooled),
B_heur = Black-to-move sign agreement on the same
heuristic-opponent positions. Per-cell n ≈ 1,750
(pooled ≈ 3,500), so the SE of a difference of two
cell accuracies is ≈ 0.015.

- **Mechanism A** iff W_heur is ≥ 0.05 below W_ml
  **AND** ≥ 0.05 below B_heur (both gaps clear ~3 SE).
  The value head misreads greedy-heuristic positions
  specifically as White.
- **Mechanism B** iff both gaps are < 0.03 (accuracy
  comparable across opponent types and sides) while
  the played White collapse is real (established in
  §1). The value head is fine; the failure is in the
  search. B is then VERIFIED (not assumed) by the
  §3.2 deeper-search check before any patch.
- **Grey zone** (either gap in [0.03, 0.05)): the
  diagnosis is inconclusive on the primary rule; the
  patch decision follows the stronger secondary
  signal — if cut (a) shows the White gap driven by
  White-WIN positions (White-win accuracy ≥ 0.05
  below Black-win accuracy on the same opponent),
  treat as A; otherwise treat as B and run the §3.2
  verification.
- **Per-opponent rule (for the patch data mix, §3.1):**
  a cell (opponent) is a PATCH CELL iff the v44 net's
  White-to-move accuracy on that opponent's positions
  is ≥ 0.05 below its Black-to-move accuracy on the
  same opponent's positions. The patch adds positions
  from every patch cell. (This is what lets v37-game
  positions enter the patch if — as §1 suggests —
  the gap is worst there, even though v37 is an ML
  opponent.)

## 3. Patch decision tree (pre-registered)

### 3.1 If mechanism A — data patch (new net)

Train a new net on the 2-ply-label recipe (v37's —
the strongest known labelling) with the training set
extended by the patch cells' positions:
- **Base:** v37's 2-ply-labelled shards, copied
  VERBATIM from `gomoku-ml3-h2/ml/runs/ml3-h2/data/`
  (train dirs v7/v18/random + holdouts). Verify the
  counts match v37's record (37,227 positions:
  33,849 train / 3,378 holdout) before training.
- **New positions:** ALL quiet positions from v44's
  battery legs against every patch cell (§2.3 rule),
  labelled with `_rollout2_leaf_value` via
  `generate_distill_data.py label --label-mode
  rollout2`, with the same game-level holdout
  protocol (every 10th game held out). Expected
  yield ≈ 3,500 positions per patch cell (≈ 3,150
  train). The exact mix is determined by the §2.3
  rule, not chosen by hand.
- **Recipe:** v37's exactly — ResNet 16ch/2 blocks
  from scratch, 15 epochs, batch 64, lr 0.002
  cosine, warmup 20 steps, weight decay 1e-4,
  value MSE only (policy_weight 0.0), augmentation
  on, val_fraction 0.05, seed 0, memmap path
  (mandatory — in-RAM load OOM-kills at this
  scale), checkpoint selection best-by-internal-val-
  loss. Config `ml/configs/ml5-h1.yaml` (v37's
  `ml3-h2.yaml` with run name/dir changed).
- **α sweep:** {0.125, 0.25, 0.5}, 20 games each vs
  v44, dev seeds 1001/2001, via `sweep_alpha.py
  --version 48`. The grid brackets v37's winner
  (0.25) both ways. Selection: best win rate;
  tie-break the smaller α (conservative — less
  optimizer's-curse exposure); no-signal rule (all
  three arms within one win) → α=0.25 (v37's
  winner). The registry entry's `value_weight` is
  updated to the winner after the sweep.
- **Caveat (recorded, not hidden):** the new
  positions come from v44's battery games at
  validation seeds 100/200, and v48's battery
  replays the same seeds. v48 ≠ v44, so the games
  diverge from the first differing move; and the
  v24/v37 precedent is the same shape (trained on
  v7 self-play games, then played vs v7). The
  game-level holdout on the new positions is kept
  for the holdout diagnostic, not for the battery.

### 3.2 If mechanism B — search-side patch

First VERIFY the horizon effect (do not patch on
the §2.3 negative alone): run `find_losing_moves.py`
on v44's White losses vs v37 (the worst collapse
leg, 42 White losses), work up the genuine blunders
with `diag_root.py`, and run `targeted_playout.py`
at 400 sims on each blunder's position. **Horizon
confirmed** iff deeper search finds a saving move
the 100-sim search missed in ≥ 2 worked-up blunders.
- If confirmed, the patch is **White-only deeper
  labels**: a new net trained on the §3.1 base with
  White-to-move positions labelled by a 4-ply
  rollout-leaf label (`_rollout4_leaf_value`,
  implemented by extending `_rollout3_leaf_value`
  one ply in the same construction, verified by
  manual re-derivation on a 500-position sample
  with 0 mismatches) and Black-to-move positions
  labelled with the 2-ply label. Same recipe and α
  sweep as §3.1. This is distinct from H2's uniform
  4-ply labels (v49): the depth is targeted at the
  collapsing side only. It touches no CLOSED lever
  (no q-based root decision, no tie-break change,
  no VCF check, no history dropout, no sims change).
- If NOT confirmed (deeper search finds saves in
  < 2 blunders), there is no patchable mechanism —
  fall back to §3.3 and say so.

### 3.3 If no patchable mechanism — α-ablation fallback

v48 = v44's net at **α=0.0625** — one step cooler
than v44's sweep winner (0.125), testing whether
the monotone α trend in iter-44 §5.1 (0.725 /
0.600 / 0.475 at α = 0.125 / 0.25 / 0.5) continues
below 0.125. Same checkpoint as v44
(`ml/runs/ml4-h3/value-net.pt`), only
`value_weight` differs. Distinct from H3's v50
(v37's net at α=0.125). No training, no labelling;
the α is pre-registered here, not swept (the
sweep already ran in iter-44; this version tests
one new point below its winner).

## 4. Eval seeds and battery (pre-registered)

**The ML bar (reference v44, Elo 98.1):** (1) h2h vs
v44 ≥ 100 games, win rate > 0.50 with Wilson 95% CI
lower bound > 0.50; (2) ladder Elo > 98.1; (3)
tactical ≥ 20/20 + 29/29 on the corrected suite;
(4) all test suites green. Fail any clause =
rejected. Holdout numbers do NOT predict played
strength — the 100-game battery is the verdict.

**Seeds:** h2h vs v44: v48=100 / v44=200; vs v0:
v48=100 / v0=200; vs v7: v48=100 / v7=200; vs v37:
v48=100 / v37=200; ladder seed 100, 40 games /
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
selfplay_batch.py --engine-a v48 --engine-b v44 --games 100 --openings research/openings-ml4.json --seed-a 100 --seed-b 200 --out research/iterations/batch-v48-vs-v44.json
# Clause 2: ladder (40 games/opponent, mcts_sims 100)
eval_version.py --version 48 --ladder-games 40 --out research/iterations/ladder-v48.json
# Clause 3: tactical, both suites, seed 100
eval_tactical_full.py --version 48 --suite both --seed 100
# Clause 4: ml pytest + npm test + tsc
# Secondary: vs v0, vs v7, vs v37 (openings), swap vs v7 (balanced/tempo, placement_seed 11)
selfplay_batch.py --engine-a v48 --engine-b v0 --games 100 --seed-a 100 --seed-b 200 --out research/iterations/batch-v48-vs-v0.json
selfplay_batch.py --engine-a v48 --engine-b v7 --games 100 --seed-a 100 --seed-b 200 --out research/iterations/batch-v48-vs-v7.json
selfplay_batch.py --engine-a v48 --engine-b v37 --games 100 --openings research/openings-ml4.json --seed-a 100 --seed-b 200 --out research/iterations/batch-v48-vs-v37.json
selfplay_batch.py --engine-a v48 --engine-b v7 --games 100 --swap-rule --placement balanced --placement-seed 11 --judge tempo --seed-a 100 --seed-b 200 --out research/iterations/batch-v48-vs-v7-swap.json
```

## 5. Eval order (pre-registered)

(1) Diagnosis (§2) — measurements + mechanism call
per the §2.3 rule. (2) Patch per the §3 decision
tree (§3.1 data / §3.2 search / §3.3 fallback) +
holdout diagnostics on the new net if one is
trained (by TRUE side — §0). (3) Leaf-cost
measurement. (4) α sweep if a new net is trained
(§3.1/§3.2; not for the §3.3 fallback, whose α is
pre-registered). (5) FAST LOOP: up to 4 iterations
of 10 games vs v44 AND 10 games vs v0 (dev seeds).
(6) FULL VALIDATION BATTERY on the final v48 (§4).
Any battery numbers measured before the fast loop
completes are INTERMEDIATE, labelled as such; only
the final battery counts for the bar.
