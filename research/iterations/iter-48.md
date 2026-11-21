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

## 6. Diagnosis results — mechanism call: TREAT AS A (general White-side value-head defect)

**Diagnosis COMPLETE 2026-11-21 ~01:35 PST** (labelling
18.4 min under load ~7; raw `ml/runs/ml5-h1/diag.log`,
JSON `ml/runs/ml5-h1/diag-white-collapse.json`).
17,354 quiet positions from v44's 400 battery games
(v0: 5,116; v7: 3,998; v8: 3,753; v37: 4,487).
Validity check: the openings legs' `moves` include the
opening plies (100/100 records in both legs — verified
against the driver code, which prepends the opening),
so the replay is exact.

### 6.1 Primary subject: v44 net vs its 3-ply target (TRUE sides, full planes)

| Cell | n | sign agreement | no-hist | Pearson r |
|---|---|---|---|---|
| v0: Black | 2,580 | 0.7820 | 0.7861 | 0.0799 |
| v0: White | 2,536 | 0.7170 | 0.7225 | 0.1433 |
| v7: Black | 2,095 | 0.7651 | 0.7711 | 0.0655 |
| v7: White | 1,903 | 0.6993 | 0.6707 | 0.1208 |
| v8: Black | 1,938 | 0.7805 | 0.7858 | 0.0687 |
| v8: White | 1,815 | 0.6771 | 0.6934 | 0.0200 |
| v37: Black | 2,357 | 0.7562 | 0.7630 | 0.0548 |
| v37: White | 2,130 | 0.6805 | 0.6425 | 0.0744 |
| heuristic-pooled: Black | 4,675 | 0.7743 | 0.7793 | 0.0651 |
| heuristic-pooled: White | 4,439 | 0.7093 | 0.6999 | 0.1309 |
| ml-pooled: Black | 4,295 | 0.7673 | 0.7735 | 0.0598 |
| ml-pooled: White | 3,945 | 0.6789 | 0.6662 | 0.0482 |
| overall | 17,354 | 0.7344 | 0.7320 | 0.0933 |

Per-cell n ≈ 2,000, so the SE of a White-vs-Black
difference is ≈ 0.014; the pooled SE is ≈ 0.009.

**Applying the §2.3 rule:**
- W_heur = 0.7093, W_ml = 0.6789, B_heur = 0.7743.
- Mechanism A (heuristic-specific) requires W_heur
  ≥ 0.05 below W_ml: **FAILS** — W_heur is 0.030
  ABOVE W_ml. The net's White accuracy is BETTER on
  heuristic-opponent positions than on ML-opponent
  positions. The heuristic-specific misreading
  hypothesis is REFUTED.
- Mechanism B requires both gaps < 0.03: **FAILS** —
  B_heur − W_heur = 0.065 (≈ 7 SE).
- Grey zone (gap 1 = 0.030): per the pre-registered
  rule, follow cut (a) — White-win accuracy vs
  Black-win accuracy: v0 −0.059, v7 −0.077, v8
  −0.097, v37 −0.106 — all four clear 0.05. **Treat
  as A.**

**What the mechanism actually is (honest restatement):
a GENERAL White-to-move value-head defect, not a
heuristic-specific one.** The net reproduces its
teacher worse as White than as Black on EVERY
opponent (gaps: v0 −0.065, v7 −0.066, v8 −0.103,
v37 −0.076), and the gap is WORST against the ML
opponents — the opposite of the §1 framing. The
win/loss cut shows both halves of the defect:
White-WIN positions (net undervalues White's
winning chances — v30's pattern: 0.72–0.80 vs
Black's 0.83–0.86) AND White-LOSE positions (net
fails to see White is losing: 0.62–0.66 vs Black's
0.71–0.75 — consistent with iter-44's hypothesis
that the optimistic 3-ply labels make the net
under-defend from behind). A search-horizon
contribution is NOT excluded — the diagnosis shows
a value-head defect exists; it does not prove the
search innocent. The battery arbitrates whether
fixing the value head fixes the played collapse.

**Patch-cell rule (§2.3):** all four cells clear
0.05 (v0 0.065, v7 0.066, v8 0.103, v37 0.076) →
**ALL FOUR LEGS enter the patch** (§7): all quiet
positions from v44's vs-v0, vs-v7, vs-v8, and
vs-v37 battery legs, 2-ply labelled, on top of
v37's 37,227-position base. The mix is determined
by the mechanical rule, not chosen by hand.

### 6.2 Second subject: v37 net vs its 2-ply target

The v37 net shows the INVERSE side pattern: White
BETTER than Black on every opponent (v0 0.6996 vs
0.6093, v7 0.7887 vs 0.6896, v8 0.7499 vs 0.7072,
v37 0.7473 vs 0.6887; overall 0.7056). v37 is the
version that does NOT collapse as White (0.54 as
White vs v44) — the correlation between White-side
accuracy and White played strength holds across
both subjects, which supports the value-head
mechanism. Two further notes: (a) v37's net has a
Black-side defect on v0 positions specifically
(0.6093 — its worst cell by 0.08); (b) v37's
no-history collapse reproduces on these positions
(0.7056 → 0.5039 overall), while v44's net stays
history-robust (0.7344 → 0.7320). Cross-net
overall numbers are not comparable (different
targets).

### 6.3 The §0 contamination, quantified

Index-inferred vs true-side White gaps (v44 net):
v0 0.027 vs 0.065 (attenuated 58%), v7 0.058 vs
0.066, v8 0.044 vs 0.103 (attenuated 57%), v37
0.070 vs 0.076. The index-inferred split attenuates
the true White gap by up to ~60% — iter-44 §4's
"no White collapse" claim (Black 0.7463 / White
0.7421, gap 0.004, on the holdout) was measured
with this contaminated split and is unreliable.
The true-side measurement here finds a real White
gap of 0.065–0.103. (Different position sets, so
this is not a direct replication of iter-44's
holdout numbers — but the method flaw stands, and
any future by-side claim on distill shards must
use true sides.)

## 7. Patch — §3.1 data patch (in progress)

Per the §6 mechanism call: v48 = v8's search + a
new net trained on v37's 2-ply-label recipe with
the training set extended by ALL FOUR patch cells'
positions (17,354 quiet positions from v44's
battery legs, 2-ply labelled, game-level holdout
every 10th game) on top of v37's 37,227-position
base. Same architecture and training recipe as v37
(ResNet 16ch/2 blocks, 15 epochs, batch 64, lr
0.002 cosine, value MSE only, memmap path). α
sweep {0.125, 0.25, 0.5} vs v44, 20 games each,
dev seeds 1001/2001.

### 7.1 Labelling (actuals)

`generate_distill_data.py label --label-mode
rollout2` over the four battery-legs games files
(converted from the battery JSONs; winner mapped
black→1 / white→−1 / draw→0 for the int64 shard
field). 10.0 min total, no interruptions.

| Leg | Games | Plies | Quiet positions | ±1 labels | Held-out games |
|---|---|---|---|---|---|
| vs v0 | 100 | 6,342 | 5,116 (4,546 train / 570 holdout) | 271 (5.3%) | 10 |
| vs v7 | 100 | 4,896 | 3,998 (3,677 train / 321 holdout) | 293 (7.3%) | 10 |
| vs v8 | 100 | 4,599 | 3,753 (3,186 train / 567 holdout) | 300 (8.0%) | 10 |
| vs v37 | 100 | 5,429 | 4,487 (4,194 train / 293 holdout) | 281 (6.3%) | 10 |
| **Total new** | **400** | **21,266** | **17,354 (15,603 train / 1,751 holdout)** | **1,145 (6.6%)** | **40** |

Quiet counts match the §6 diagnosis EXACTLY per
leg (5,116 / 3,998 / 3,753 / 4,487) — the labelling
reproduces the diagnosis positions. Base shards
copied verbatim from `gomoku-ml3-h2` and verified:
33,849 train / 3,378 holdout = 37,227, matching
v37's record. **Total train set: 49,452 positions**
(33,849 base + 15,603 new); total holdout 5,129.
Memmap bundle `ml/runs/ml5-h1/data-mmap/`: 1,097
shards, 49,452 samples, planes (49452, 8, 15, 15).

### 7.2 Training (actuals)

**Training COMPLETE 2026-11-21 ~02:01 PST**
(started ~01:36 PST, ~25 min, no interruptions):
config `ml/configs/ml5-h1.yaml`, 15 epochs, batch
64, memmap bundle (49,452 samples; 46,979 train /
2,473 val, seed-0 split). Run dir
`ml/runs/ml5-h1-train/`; log `ml/runs/ml5-h1/
train.log`. 11,025 steps, ~82–140 s/epoch.

| Epoch | Train loss | Val loss |
|---|---|---|
| 1 | 0.0999 | 0.0986 |
| 2 | 0.0958 | 0.0955 |
| 3 | 0.0943 | 0.0946 |
| 4 | 0.0935 | 0.0937 |
| 5 | 0.0918 | 0.0949 |
| 6 | 0.0907 | 0.0918 |
| 7 | 0.0894 | 0.0906 |
| 8 | 0.0880 | 0.0877 |
| 9 | 0.0863 | 0.0892 |
| 10 | 0.0843 | 0.0874 |
| 11 | 0.0830 | 0.0853 |
| 12 | 0.0812 | 0.0847 |
| 13 | 0.0811 | 0.0844 |
| 14 | 0.0806 | **0.0841** |
| 15 | 0.0804 | 0.0846 |

Val loss decreased overall (small bumps at epochs
5, 9, 15) — no overfitting by epoch 15. **Best
checkpoint: epoch 14 (`best.pt`, val 0.0841)** —
the `--out` copy at `ml/runs/ml5-h1/value-net.pt`
is verified bit-identical in weights to `best.pt`
(max |Δ| 0.0), so the registry path holds the
best-by-val-loss checkpoint as pre-registered.
Architecture verified: ResNet 16ch/2 blocks,
126,759 params, 48 tensors — identical counts to
v24's, v37's, v44's, and v8's checkpoints.
Checkpoint md5: `a8c7a0bcfd749ac774ec9762c829d672`.

### 7.3 Holdout diagnostics (actuals)

Game-level holdouts (never trained on), targets =
2-ply labels. v48 net = `ml/runs/ml5-h1/
value-net.pt`; v37 net = `ml/runs/ml3-h2/
value-net.pt` (same targets, same positions —
direct comparison). Overall sign agreement via
`eval_value_holdout.py` (its by-side split is
contaminated — §0 — so by-side comes from the new
`score_shards_by_side.py`, which recovers TRUE
sides by replay).

| Holdout | n pos | v48 sign (full) | v48 sign (no-hist) | v37 sign (full) | v37 sign (no-hist) |
|---|---|---|---|---|---|
| v7 (base) | 2,050 | 0.7600 | 0.7595 | 0.7395 | 0.5163 |
| v18 (base) | 790 | 0.8321 | 0.8423 | 0.8436 | 0.5846 |
| random (base) | 538 | 0.6942 | 0.5403 | 0.6623 | 0.6604 |
| vs-v0 (new) | 570 | 0.7073 | 0.7132 | 0.6582 | 0.4951 |
| vs-v7 (new) | 321 | 0.8176 | 0.8208 | 0.7453 | 0.5409 |
| vs-v8 (new) | 567 | 0.6923 | 0.7081 | 0.6450 | 0.4852 |
| vs-v37 (new) | 293 | 0.7560 | 0.7560 | 0.7285 | 0.6014 |
| **base pooled** | **3,378** | **0.7666** | — | **0.7519** | — |
| **new pooled** | **1,751** | **0.7329** | — | **0.6837** | — |

v48 beats v37's net on 6 of 7 holdouts (only v18
full is lower, −0.011) and by +0.049 pooled on
the new patch-cell holdouts. **The v48 net is
HISTORY-ROBUST on engine positions** (v7 no-hist
drop 0.0005, v18 no-hist BETTER than full) — the
v37 collapse (0.7395 → 0.5163) is gone, like
v44's net. The exception is random positions
(no-hist 0.5403 vs full 0.6942) — the net leans
on history for random-playout positions, the
least game-like source; engine positions are
what the leaf sees in real games.

**By TRUE side on the patch-cell positions**
(`score_shards_by_side.py`, 2-ply labels):

| Split | v48 Black | v48 White | White − Black | v37 Black | v37 White |
|---|---|---|---|---|---|
| ALL positions (17,354; 90% trained-on) | 0.7399 | 0.7837 | **+0.044** | — | — |
| HOLDOUT only (1,751; unseen) | 0.7368 | 0.7291 | **−0.008** | 0.6581 | 0.7094 |

On unseen games the pooled White gap is −0.008 —
essentially CLOSED, vs v44's −0.065…−0.103 on the
same position distribution (different target, so
the comparison is of the gap construct, not of
absolute accuracy). Per-leg holdout cells are too
noisy to read individually (10 games each,
game-clustered: v0 White 0.5655 vs Black 0.8636
while v8 White 0.8674 vs Black 0.5021 — opposite
swings that cancel in the pooled number). v37's
net on the same unseen positions has a White
ADVANTAGE (+0.051) at lower overall accuracy
(0.684 vs v48's 0.733) — both nets are free of a
White deficit here; v48's is the more accurate
one. **Caveat (standing): holdout does not predict
played strength — the battery is the verdict.**


## 8. α sweep and fast loops

**Leaf cost (2026-11-21, `ml/runs/ml5-h1/leaf-bench.txt`):**
`bench_value_leaf.py --checkpoint ml/runs/ml5-h1/value-net.pt
--positions 20 --leaf-samples 200 --sims 100`: per-move at 100
sims mean 4,120 ms / median 3,458 ms / p95 6,786 ms; net leaf
evals per move mean 80.7 (17–99); per-leaf net eval median
5.94 ms. Comparable to v44's bench (mean 5,190 ms under load
7.45) — same architecture, as expected.

### 8.1 α sweep (2026-11-21, ~02:12–04:06 PST)

`sweep_alpha.py --version 48 --alphas 0.125 0.25 0.5 --games 20
--opponent v44 --seed-a 1001 --seed-b 2001` (dev seeds, no
openings). One runtime restart killed the run during game 19
of the α=0.125 leg; the driver resumed from game 19
(6W-11L-2D) with per-game seeding — nothing lost. Log:
`ml/runs/ml5-h1/sweep-alpha.log`; per-alpha JSONs in
`research/iterations/alpha-v48-<α>-vs-v44.json`.

| α | W-L-D | win rate | 95% CI | as Black | as White | unique |
|---|-------|----------|--------|----------|----------|--------|
| 0.125 | 6-11-3 | 0.3750 | [0.1998, 0.5905] | 2-6-2 (0.300) | 4-5-1 (0.450) | 20/20 |
| 0.25 | 5-15-0 | 0.2500 | [0.1119, 0.4687] | 2-8-0 (0.200) | 3-7-0 (0.300) | 20/20 |
| **0.5** | **9-11-0** | **0.4500** | [0.2582, 0.6579] | 5-5-0 (0.500) | 4-6-0 (0.400) | 20/20 |

Win rate is U-shaped in α (0.375 / 0.250 / 0.450) — NOT the
monotone-decreasing shape v44's sweep showed (0.725 / 0.600 /
0.475), and NOT v37's monotone shape either (0.25 > 0.5 >
1.0 vs v8). **Selection (§3.1 rule, best win rate):
α=0.5.** Registry updated (`scoring.py`,
`value_weight: 0.5`). The no-signal rule does not apply
(arms differ by up to 4 wins). The 20-game CIs overlap
heavily, so the α ranking is a point-estimate call, not a
proven ordering — and ALL THREE arms sit below 0.50
against v44, so the sweep's honest message is that v48
loses to v44 on dev seeds at every α. The 100-game
clause-1 leg (validation seeds, openings) is the real
verdict.

Two observations: (1) at α=0.125 and α=0.25, v48's White
OUTPERFORMS its Black (0.450 vs 0.300; 0.300 vs 0.200) —
the reverse of v44's White collapse, and consistent with
the §7.3 finding that the patch closed the White-side
accuracy gap. The patch may have fixed White at the cost
of overall strength vs v44, or the net is simply weaker
in this matchup. (2) Three of the twenty α=0.125 games
were 225-ply full-board draws (~10 min each) — the
v48-vs-v44 matchup produces marathon draws, which will
make the 100-game battery legs slow.

### 8.2 Fast loop iteration 1 (α=0.5, seeds 1001/2001)

- vs v44: the α=0.5 sweep leg itself (20 games, same dev
  seeds) — **9W-11L-0D (0.4500)**. As Black 5-5-0, as White
  4-6-0. All 20 games unique.
- vs v0: **5W-4L-1D (0.55)**, CI [0.2737, 0.7986], 585 s
  (`research/iterations/fast-v48-vs-v0-iter1.json`). Same
  headline as v44's 0.55 in the same loop. Colour split:
  as Black 1W-3L-1D (0.300), as White 4W-1L-0D (0.800) —
  White is now the STRONG side vs v0, the reverse of the
  original collapse (v42 White 0.28, v43 White 0.21 vs v0).
  Ten games is a small sample, but the direction matches
  the §7.3 holdout finding.
- Loss classification (both matches): **15 losses, 15 via
  open four, 0 missed blocks, 0 clean missed preventions
  in 72 threatened turns.** Defence as measured is clean
  — the losses are positional drift into unstoppable open
  fours, not blundered blocks. Same pattern as v37/v44.
- Losing moves (`find_losing_moves.py`, VCF solver depth
  4; raw `ml/runs/ml5-h1/losing-moves.txt`): **10 of 15
  losses were already lost** at the first VCF moment.
  **FIVE genuine blunders:** vs v44 game 8 (Black, 36
  plies), ply 30 — (8,8), 1 save; vs v44 game 13 (White,
  43 plies), ply 37 — (7,11), 5 saves; vs v44 game 18
  (Black, 38 plies), ply 30 — (11,8), 5 saves; vs v0
  game 2 (Black, 70 plies), ply 50 — (11,4), 7 saves;
  vs v0 game 8 (Black, 28 plies), ply 22 — (9,9), 5
  saves. Four of the five blunders are as Black.

**Workup — vs-v44 game 8, ply 30 (VERIFIED):**
`diag_root.py` (raw `ml/runs/ml5-h1/diag-v44-g8-ply30.txt`)
shows a FLAT root: the actual (8,8) has the best our-q
(−0.047), the most visits (13), and the 1-ply leaf agrees
(actual −0.186 v22, save −0.185 v22 — tied). The VCF is
beyond the 100-sim search's horizon AND beyond the 2-ply
label's 4-ply horizon — the net agrees with the exact
heuristic leaf, and both rank the loser (tied-)first.
Shared search-horizon limit (v44's g12 class), NOT a
value-head failure.

**Workup — vs-v44 game 13, ply 37 (VERIFIED):**
`diag_root.py` (raw `ml/runs/ml5-h1/diag-v44-g13-ply37.txt`)
shows a q-vs-LEAF inversion: the actual (7,11) has our-q
−0.028 vs the saves' −0.115/−0.074/−0.075, but the exact
1-ply leaf ranks ALL FIVE saves clearly first (+0.069 to
+0.149 vs the actual's +0.007). The leaf is RIGHT; the
backed-up root q inverts it through subtree averaging on
a flat root (top move 12 visits over 108 children). One
save, (7,3), even has the best our-q (+0.019) but only 1
visit — the most-visited root decision never sees it.
v40's blunder-dilution / visit-allocation class — a
search problem, not a leaf problem (the leaf-blind
direction is CLOSED; v46/v47's fixes were CLOSED in
batch 12). Not patchable within H1's remit.

**Workup — vs-v44 game 18, ply 30 (VERIFIED):**
`diag_root.py` (raw `ml/runs/ml5-h1/diag-v44-g18-ply30.txt`)
shows a FLAT root: the top six children all have 5
visits; the actual (11,8) has the best our-q (−0.029)
and the 1-ply leaf is essentially tied across actual and
saves (v22 −0.195 vs −0.168…−0.193). Shared-horizon
class, NOT a value-head failure.

**Workup — vs-v0 game 2, ply 50 (VERIFIED):**
`diag_root.py` (raw `ml/runs/ml5-h1/diag-v0-g2-ply50.txt`)
shows the actual (11,4) TIED for best our-q (−0.001, 5
visits) and the 1-ply leaf ranking it (tied-)first
(v22 +0.113, v20 +0.168 — tied with saves on both).
The net agrees with the exact leaf; the saves are
indistinguishable at 1 ply. Shared-horizon class, NOT
a value-head failure. (The prior favours the saves,
0.056 vs 0.027 — but the root decision is
most-visited and the actual got 5 visits vs 4.)

**Workup — vs-v0 game 8, ply 22 (VERIFIED):**
`diag_root.py` (raw `ml/runs/ml5-h1/diag-v0-g8-ply22.txt`)
shows the q nearly right and the prior wrong: the best
save (7,11) has BETTER our-q (+0.006) than the actual
(+0.000) and the v22 leaf agrees (+0.042 vs +0.007),
but only 13 visits vs the actual's 21 — the actual's
prior (0.1226) is 1.4x the save's (0.0852). EXACTLY
v37's game-2 blunder class (and v44's g9): a PRIOR
problem in shared heuristic-track code (`_priors`),
whose change would alter v8/v7/v44 equally — NOT
patchable within H1's remit.

**Fast-loop STOP after iteration 1:** no patchable
weakness class for v48 within H1's design space. The
five blunders are three shared-horizon, one
visit-allocation, one shared-prior — all in shared
search code, none a value-head failure. The §7.3
history-robustness is NOT implicated in any blunder.
No loss-motivated retrain is authorised. The battery
is the verdict.

## 9. Validation battery

### 9.1 Tactical — CLAUSE 3 PASSES

`eval_tactical_full.py --version 48 --suite both --seed 100`
(log `ml/runs/ml5-h1/tactical-v48.log`):

- Old suite: **20/20** ✓
- Hard suite (corrected): **29/29** ✓ — including
  **setup-v1v0s-g92: chose (9,6)**, the quiet setup v31
  failed, and v1bl-g2 (the v40 regression position):
  passed.
- v44 re-verified on the corrected suite first (log
  `ml/runs/ml5-h1/tactical-v44-reverify.log`): 20/20 +
  29/29 — the bar reference's numbers stand.
