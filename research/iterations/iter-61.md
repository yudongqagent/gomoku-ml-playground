# Iteration 61 — v61: the α ablation (v54's net at α=0.125)

ML batch 7, H2. Branch `research/ml-batch-7-h2`, from
`research/ml-batch-7` @ `3bfa863`. Plan:
`research/ml-batch-7-plan.md` §H2.

**Status: PRE-REGISTERED — no eval run yet.**

## 0. The question

v54 differs from v44 in TWO ways:

1. **Net:** v54's net is v37's base (2-ply rollout-leaf labels on
   v24's 817 games / 37,227 quiet positions) PLUS White-only patch
   positions from v44's battery legs and fresh ladder-MCTS legs,
   trained from scratch (ResNet 16ch/2 blocks, 15 epochs; see
   iter-54.md §2). v44's net is the same 37,227 positions with
   3-ply labels and no patch.
2. **α:** `value_weight` 0.5 (v54's sweep winner: 0.4375 vs v44
   over 40 dev-seed games, vs 0.375 @ 0.125 and 0.3625 @ 0.25)
   instead of v44's 0.125.

v54 was REJECTED 3/4 in ML batch 6 (iter-54.md §10): vs v44
0.385, CI [0.2956, 0.4829] (CI upper below 0.50); ladder Elo
168.4 [50.1, 286.7] (passes the point estimate, but loses the
MCTS leg 16-23-1); tactical 20/20 + 29/29; suites green. Which
of the two changes drove v54's weakness vs v44: the White-patch
net, or the hot α=0.5? v61 isolates the α effect: **the v54 net,
unchanged, at α=0.125.** No retraining, no new checkpoint, no
new mechanism — a clean ablation, the mirror of v50 (v37's net
at α=0.125, ML batch 5).

### 0.1 Premise verification (done before any eval)

The ablation's premise is that α (`value_weight`) is an
**inference-time** parameter of `ValueMCTS`, not a training
parameter. Verified three ways, 2026-11-24:

1. **Code** (`ml/gomoku_ml/mcts_value.py`): `value_weight` is a
   `ValueMCTS.__init__` parameter (`value_weight: float | None =
   None`, defaulting to `VALUE_WEIGHT` 0.25); it is stored as
   `self.value_weight` and used in exactly one place —
   `_value_leaf_value`, where the net's quiet-position value is
   multiplied by it (`return float(value[0, 0].item()) *
   self.value_weight`). Terminal cases are exact and unaffected.
   The checkpoint payload is just `model_state` + architecture
   metadata + optimizer state (verified on v54's file: keys
   `arch_params, architecture, blocks, channels, epoch,
   model_state, num_parameters, optimizer_state, step` — no
   `value_weight`, no `alpha` anywhere in the payload).
2. **Name-collision note:** `ml/gomoku_ml/train.py` also has a
   `value_weight` — but that is the TRAINING loss coefficient
   (weight on the value-MSE term, default 1.0), a different
   parameter from the MCTS inference-time α shrink. The two
   share a name and nothing else; changing the MCTS α does not
   touch training, and the checkpoint carries no α.
3. **The sweeps themselves**: v54's own α sweep
   (`sweep_alpha.py --version 54 --alphas 0.125 0.25 0.5`,
   iter-54.md §8.1) evaluated the SAME checkpoint at three α
   values. If α were a training parameter, a sweep of one
   checkpoint would be meaningless. v50 did exactly this for
   v37's net (iter-50.md §0.1).

**Premise CONFIRMED: v61 needs no retraining.** v61 is v54's
checkpoint (`ml/runs/ml6-h1/value-net.pt`, md5
`8b7cff2b0a56c4343a544579468aa9ce`) registered with
`value_weight: 0.125`. The only difference between v61 and v54
is the registry's `value_weight` field; the only difference
between v61 and v44 is the checkpoint (White-patch 2-ply net vs
3-ply net) — α is the same 0.125 in both.

## 1. Pre-registration (before any eval)

### 1.1 The version

- **v61**: v8's PUCT MCTS (100 sims, c_puct 1.5, v1 priors) +
  v54's value net (`ml/runs/ml6-h1/value-net.pt`, White-only
  patch on v37's verbatim base, 2-ply rollout-leaf labels,
  ResNet 16ch/2 blocks) at **α = 0.125**.
- Registry entries in `ml/gomoku_ml/scoring.py`: v61 identical
  to v54's entry except `value_weight: 0.125` (v54: 0.5); v54's
  entry is added too (it lives only on `research/ml-batch-6`,
  not on this branch) so the mirror match v61-vs-v54 can run.
  No new checkpoint file is created; v61 reuses v54's
  checkpoint verbatim.

### 1.2 The comparison and the decision rule

v61 is compared against **both** v54 (same net, α=0.5) and v44
(3-ply net, α=0.125), on the full 100-game battery and the
ladder:

- **If v61 plays at v54's level** — vs v44 ≈ 0.385 (v54's
  number), ladder Elo ≈ 168 — then the **α is NOT the cause**
  of v54's weakness: the White-patch net itself loses to v44
  at either α, and v54's 40-game sweep ranking (0.5 > 0.125)
  was noise or a dev-seed artefact.
- **If v61 plays at v44's level or better** — vs v44 ≈ 0.50 or
  above (CI including or clearing 0.50), ladder Elo ≈ 98 or
  above with no MCTS-leg loss — then the **α was the cause**:
  the hot α=0.5 is what cost v54 the match, and the White-patch
  net at the cool α is competitive with the champion.
- **If v61 lands in between** (e.g. vs v44 ≈ 0.44–0.48, Elo
  between 98 and 168), both factors contributed; report the
  split honestly and do not force a binary verdict.

The mirror match v61-vs-v54 (same net, α 0.125 vs 0.5) is the
direct α measurement: it is v54's sweep comparison (0.5 beat
0.125 by 3 wins over 40 dev-seed games, no openings) run at 100
games with opening variety. A v61 win there means cooling α
below 0.5 helps the v54 net too — and would directly contradict
the sweep's point-estimate ranking, in which case the 40-game
sweep joins ML batch 5's and v54's as a failed ranking (the
plan's standing lesson: 40-game sweeps fail twice already).

Precedent (v50, iter-50.md): v37's net at α=0.125 played at
v37's level, not v44's — vs v44 0.65 (v37: 0.69), ladder Elo
−26.1 (v37: −34.9), mirror vs v37 0.51 — proving v44's Elo gain
came from its 3-ply labels, not the lower α. v61 asks the same
question of v54's net, with the mirror roles reversed (the
hot-α version is the base here, the cool-α version the
ablation).

### 1.3 The ML bar (reference v44, Elo 98.1; clause-2 MCTS-leg rule NEW for this batch)

1. h2h vs v44 ≥ 100 games, win rate > 0.50 with Wilson 95% CI
   lower bound > 0.50.
2. Ladder Elo > 98.1 AND no MCTS-leg loss (mcts leg win rate ≥
   0.50 on the point estimate). Report the MCTS leg alongside
   the Elo.
3. Tactical ≥ 20/20 + 29/29 on the corrected suite.
4. All test suites green.

Fail any clause = rejected. Note: as an ablation, v61's value
does not depend on passing the bar — the finding (α vs net) is
the deliverable either way. But the bar is applied as written;
a pass would make v61 the ML champion (and change the daily
cross-track pairing), which would be reported explicitly.

### 1.4 Battery (pre-registered commands)

All legs 100 games, validation seeds `--seed-a 100 --seed-b
200`, per-game seeding + uniqueness audit on every leg. ML-vs-ML
legs use the pre-registered `research/openings-ml4.json` (10
openings) via `--openings`.

```sh
# Clause 1: vs v44 (openings)
selfplay_batch.py --engine-a v61 --engine-b v44 --games 100 \
  --openings research/openings-ml4.json --seed-a 100 --seed-b 200 \
  --out research/iterations/batch-v61-vs-v44.json
# vs v0 (no-search anchor)
selfplay_batch.py --engine-a v61 --engine-b v0 --games 100 \
  --seed-a 100 --seed-b 200 \
  --out research/iterations/batch-v61-vs-v0.json
# vs v7 (normal rules)
selfplay_batch.py --engine-a v61 --engine-b v7 --games 100 \
  --seed-a 100 --seed-b 200 \
  --out research/iterations/batch-v61-vs-v7.json
# vs v37 (openings)
selfplay_batch.py --engine-a v61 --engine-b v37 --games 100 \
  --openings research/openings-ml4.json --seed-a 100 --seed-b 200 \
  --out research/iterations/batch-v61-vs-v37.json
# Mirror match: vs v54 (openings) — same net, α 0.125 vs 0.5
selfplay_batch.py --engine-a v61 --engine-b v54 --games 100 \
  --openings research/openings-ml4.json --seed-a 100 --seed-b 200 \
  --out research/iterations/batch-v61-vs-v54.json
# swap vs v7 (tempo judge, balanced placement, seed 11)
selfplay_batch.py --engine-a v61 --engine-b v7 --games 100 \
  --swap-rule --placement balanced --placement-seed 11 --judge tempo \
  --seed-a 100 --seed-b 200 \
  --out research/iterations/batch-v61-vs-v7-swap.json
# Ladder (40 games/opponent, auto-resume; committed artifact)
eval_version.py --version 61 --ladder-games 40 \
  --out research/iterations/ladder-v61.json
# Tactical, both suites, seed 100 (re-verify v44's tactical first)
eval_tactical_full.py --version 61 --suite both --seed 100
```

Fast loops (10 games, dev seeds 1001/2001, never the validation
seeds) run vs v0 AND v44 before the battery, per the standing
method; the fast loop is a finding method, not a verdict — the
100-game battery is the verdict. As an ablation, no patch is
sought: any engine change would break the isolation of the α
effect. The loss read is a smoke check for a NEW failure class
only.

### 1.5 Checkpoints (gitignored, copied into this worktree, md5-verified)

- v54 (= v61's net): `ml/runs/ml6-h1/value-net.pt`, md5
  `8b7cff2b0a56c4343a544579468aa9ce` (from `gomoku-ml6` —
  verified; matches the tasking).
- v44: `ml/runs/ml4-h3/value-net.pt`, md5
  `d22af606f7d2ee0254f372da895ae7ad` (from the main worktree —
  verified).
- v37: `ml/runs/ml3-h2/value-net.pt`, md5
  `2fde492aae6d44cbe945e15bd5f1f959` (from `gomoku-ml3-h2` —
  verified).
- v8 (tests): `ml/runs/b3-h2/value-net.pt`, md5
  `5deb26f3df67883f24f04800d5356da6` (from the main worktree —
  verified).

## 2. Registry entries + targeted tests

- Registry entries added in `ml/gomoku_ml/scoring.py` (commit
  `e258ebd`): `54: {"name": "scoring-v54", ...,
  "value_checkpoint": "ml/runs/ml6-h1/value-net.pt",
  "value_weight": 0.5}` — verbatim from
  `research/ml-batch-6` — and `61: {"name": "scoring-v61", ...,
  "value_checkpoint": "ml/runs/ml6-h1/value-net.pt",
  "value_weight": 0.125}` — identical to v54's entry except the
  `value_weight`. No new checkpoint: v61 reuses v54's file
  verbatim (md5 `8b7cff2b0a56c4343a544579468aa9ce`, verified on
  copy).
- Wiring verified live: `get_player(61)` yields
  `value_weight == 0.125` and the same checkpoint path as
  `get_player(54)` (0.5); `ValueMCTS.__init__` has no duplicate
  parameters (the recurring silent-merge trap — checked with
  `inspect.signature`: `self, checkpoint_path, model, sims,
  c_puct, rng, value_weight, device`).
- Targeted tests added in `ml/tests/test_value_weight.py`:
  `test_value_weight_v61_registry_passthrough` (entry carries
  v54's checkpoint path and `value_weight` 0.125, differing
  from v54's 0.5; a registry-built player carries 0.125),
  `test_value_weight_v61_leaf_is_quarter_v54` (same checkpoint,
  α 0.125 vs 0.5: the v61 quiet-leaf value is exactly a quarter
  of the v54 leaf value — the shrink is actually applied, not
  just stored), and `test_value_weight_v61_loads_v54_checkpoint`
  (registry-built v61 and v54 players carry byte-identical net
  weights from the real checkpoint file). All 8 tests in the
  file pass. Note: the third test initially skipped — the
  checkpoint-existence check used a CWD-relative path while
  pytest runs from `ml/`; fixed to resolve against the repo
  root (the registry path itself is already resolved against
  the repo root inside `ScoringPlayer`).
- Full suites on this branch at the start of the eval phase:
  pytest **502 passed, 1 skipped, 58 xfailed** (1550s, log
  `ml/runs/ml7-h2/pytest-baseline.log`); npm test **55/55**;
  tsc clean. Clause 4 green at the start (re-run at the end
  before reporting).

## 3. Fast loops

Iteration 1, dev seeds 1001/2001 (never the validation seeds),
10 games each. Logs: `ml/runs/ml7-h2/fast-v61-v0.log`,
`ml/runs/ml7-h2/fast-v61-v44.log`.

| leg | W-L-D | win rate | 95% CI | as Black | as White | unique |
|-----|-------|----------|--------|----------|----------|--------|
| vs v0 | 5-5-0 | 0.5000 | [0.2366, 0.7634] | 4-1-0 (0.80) | 1-4-0 (0.20) | 10/10 |
| vs v44 | 4-6-0 | 0.4000 | [0.1682, 0.6873] | 3-2-0 (0.60) | 1-4-0 (0.20) | 10/10 |

Both legs are consistent with v61 playing at v54's level
(v54's battery numbers: 0.665 vs v0, 0.385 vs v44) — the vs-v44
leg matches v54's 0.385 almost exactly, and the vs-v0 leg is
below v54's 0.665 but on 10 games with a CI spanning 0.24–0.76.
White is the weak side in both legs (0.20), matching v54's
played White pattern vs v44 (0.30). But 10 games cannot
distinguish v54-level from v44-level play, and dev seeds
flatter. No patch is sought from these loops: v61 is an
ablation, and any change to the engine would break the
isolation of the α effect. The loss read (§3.1) is a smoke
check only; the 100-game battery is the verdict.

### 3.1 Loss read (light)

All 11 fast-loop losses classified with `classify_losses.py`
(details `ml/runs/ml7-h2/fast-losses.json`): **11/11 via open
four, 0 missed blocks, 0 clean missed preventions in 70
threatened turns.** The losses are positional — letting the
opponent build the decisive threat — not tactical blunders.
Same pattern as v50's fast loop (6 losses, 6 via open four, 0
missed blocks) and v44's (9 losses, 9 via open four, 0 missed
blocks). **No new failure class appears at α=0.125.** Nothing
patchable here, and nothing to patch: any engine change would
break the ablation.

## 4. Tactical (corrected suite, seed 100)

- v44 re-verified first on the corrected suite (log
  `ml/runs/ml7-h2/tactical-v44-reverify.log`): **20/20 old +
  29/29 hard** — the corrected suite does not change v44's
  tactical. (The corrected
  `winning-quiet-setup-2` expected list, incl. (9,7), is
  present in the run's JSON.)
- v61 (`eval_tactical_full.py --version 61 --suite both --seed
  100`, log `ml/runs/ml7-h2/tactical-v61.log`): **20/20 old +
  29/29 hard**. Clause 3 passes. Expected: α only shrinks the
  net's quiet-position value; the tactical positions are decided
  by the exact terminal cases and the threat check, which α does
  not touch. (v54 itself was 20/20 + 29/29 at α=0.5.)

## 5. Battery

All legs 100 games, validation seeds 100/200, per-game seeding.
Aggregates below are recomputed from the raw per-game records
with `ml/runs/ml7-h2/aggregate.py` (validated: reproduces v44's
published numbers exactly on all five committed legs — 0.58 vs
v0, 0.67 vs v8, 0.31 vs v37, 0.45 vs v7, 0.39 swap).

The battery survived **7 runtime restarts / VM replacements**
(killed at 33 games banked in leg 1, 33 in leg 2, 69 in leg 3,
26 then 90 in leg 4, 60 in leg 5, 54 in leg 6); auto-resume +
per-game seeding lost zero games — every leg resumed from its
banked count with no seed change (resumed_from_game: 33, 33,
69, 90, 60, 54; the ladder ran clean in one pass).

### 5.1 vs v44 (clause 1, openings) — COMPLETE

`batch-v61-vs-v44.json`: **48W-46L-6D, win rate 0.5100, Wilson
95% CI [0.4135, 0.6058]** — CI lower bound 0.4135 < 0.50, so
**clause 1 FAILS**. As Black 20W-25L-5D (0.45); as White
28W-21L-1D (0.57) — White is the STRONG side vs v44, the
reverse of v54's split (Black 0.47 / White 0.30). Uniqueness
99/100; de-dup 0.5051, CI [0.4082, 0.6015].

First ablation read: v61 plays at v44's level (0.51), not
v54's level (0.385) — a +0.125 swing from cooling α alone, on
the same net. The colour inversion is the mechanism's
fingerprint: the hot α=0.5 was costing v54 its White game vs
v44, and cooling it recovers White (0.30 → 0.57) at a small
Black cost (0.47 → 0.45).

### 5.2 vs v0 — the White fix does NOT survive cooling

`batch-v61-vs-v0.json`: **44W-52L-4D, win rate 0.4600, Wilson
95% CI [0.3656, 0.5574]**. As Black 28W-19L-3D (0.59); as White
16W-33L-1D (0.33). Uniqueness 99/100; de-dup 0.4545, CI
[0.3600, 0.5525].

v54's vs-v0 leg was 0.665 with White the strong side (0.71).
Cooling α DESTROYS that edge: White falls 0.71 → 0.33, Black
is roughly intact (0.62 → 0.59). v54's signature White-vs-v0
strength was a property of the HOT α on this net, not of the
patch alone — the patch's White positions need the hot α to
convert against a greedy opponent. (v44's own vs-v0 White was
0.59 at α=0.125 on the 3-ply net; the v54 net at the same α
manages only 0.33 — the two nets respond to α differently.)

### 5.3 vs v7 — cooling hurts here too

`batch-v61-vs-v7.json`: **28W-68L-4D, win rate 0.3000, Wilson
95% CI [0.2189, 0.3958]**. As Black 17W-29L-4D (0.38); as White
11W-39L-0D (0.22). Uniqueness 100/100; de-dup identical.

v54's vs-v7 leg was 0.380 (Black 0.47 / White 0.29). Cooling α
costs 0.08 overall, both colours worse. v61's 0.30 vs v7 is
the weakest vs-v7 number of any ML version since v42's 0.295.

### 5.4 vs v37 — cooling helps vs ML opponents

`batch-v61-vs-v37.json`: **30W-54L-16D, win rate 0.3800,
Wilson 95% CI [0.2910, 0.4779]**. As Black 16W-25L-9D (0.41);
as White 14W-29L-7D (0.35). Uniqueness 100/100; de-dup
identical. (16 draws — the highest draw count of any v61 leg;
v37-vs-v61 games go long: mean 78 plies, max 225.)

v54's vs-v37 leg was 0.315 (Black 0.42 / White 0.21). Cooling
α gains 0.065, driven by White (0.21 → 0.35) with Black intact
(0.42 → 0.41). Same direction as vs v44: the cool α helps the
v54 net against ML opponents, in both cases through White.

### 5.5 vs v54 — the mirror match (openings) — COMPLETE

`batch-v61-vs-v54.json`: **50W-41L-9D, win rate 0.5450,
Wilson 95% CI [0.4475, 0.6391]**. As Black 31W-16L-3D (0.65);
as White 19W-25L-6D (0.44). Uniqueness 96/100; de-dup 0.5573,
CI [0.4576, 0.6525].

The direct α measurement: cooling α from 0.5 to 0.125 on the
v54 net wins the mirror 0.545 — but the CI includes 0.50, so
the edge is not decisive at 100 games (contrast v50's mirror
vs v37: 0.51, also spanning 0.50). The mirror's colour split
is extreme (Black 0.65 / White 0.44): the cool-α side wins as
Black and loses as White against its own hot-α twin — the α
effect itself is colour-asymmetric in the mirror, even though
the battery legs show cooling helping White vs v44/v37. The
mirror does NOT reproduce the sweep's ranking (0.5 beat 0.125
by 3 wins over 40 dev-seed games, no openings): at 100 games
with openings, the ranking flips. The 40-game sweep joins ML
batch 5's and v54's as a failed ranking — three instances now.

### 5.6 Swap vs v7 — secondary metric

`batch-v61-vs-v7-swap.json`: **30W-66L-4D, win rate 0.3200,
Wilson 95% CI [0.2367, 0.4166]**. Colour split
(judge-assigned): as Black 22W-27L-3D (0.4519, n=52); as White
8W-39L-1D (0.1771, n=48). Uniqueness 100/100; de-dup identical.

v54's swap was 0.345 (Black 0.4135 / White 0.2708). v61 is
slightly worse overall, with White much worse (0.18 vs 0.27)
and Black slightly better (0.45 vs 0.41). Under swap the cool
α's White damage vs heuristic opponents is, if anything,
sharper than under normal rules.

### 5.7 Ladder (clause 2) — FAIL on both parts

`ladder-v61.json` (40 games/opponent, seed 100):
- vs random: 40W-0L-0D (1.000)
- vs heuristic: 22W-17L-1D (0.5625, CI [0.4101, 0.7040]); as
  Black 15W-4L-1D (0.775), as White 7W-13L-0D (0.350)
- vs mcts: 18W-21L-1D (0.4625, CI [0.3182, 0.6134]); as Black
  13W-7L-0D (0.650), as White 5W-14L-1D (0.275)

**Ladder Elo: estimate 43.7, interval [−63.2, 150.5]**
(anchors random −800, heuristic 0). The point estimate is
below v44's 98.1, so clause 2 FAILS on the Elo part — and the
MCTS leg (0.4625 < 0.50) fails the NEW MCTS-leg rule too. Both
parts fail; no ambiguity.

Comparison: v54's ladder was random 40-0, heuristic 27-9-4
(0.725, White 0.75), mcts 16-23-1 (0.4125), Elo 168.4. Cooling
α cost 125 Elo points, almost entirely through the heuristic
leg's White collapse (0.75 → 0.35) — the same White-vs-
heuristic damage as the vs-v0 leg. The MCTS leg improved
slightly (0.4125 → 0.4625, White 0.25 → 0.275) but still loses
outright. v61 is the third ML version in a row to lose the
MCTS leg (v54, v56, v61) after v44 won it 25–15.

### 5.8 Defence — fully solved, as usual

`classify_losses.py` on all six battery legs (details
`ml/runs/ml7-h2/battery-losses.json`): **327 losses pooled,
327/327 via open four, 0 missed blocks, 0 clean missed
preventions in 1609 threatened turns.** Every leg individually
0 missed blocks / 0 clean missed preventions. The losses are
positional drift, not tactical blunders — the α change moves
strength between opponents and colours but does not touch
defence, which the exact terminal cases already solve.

## 6. Verdict — REJECTED (2/4)

| Clause | Bar | v61 | Verdict |
|--------|-----|-----|---------|
| 1. h2h vs v44 | > 0.50, CI lower > 0.50 | 0.5100, CI [0.4135, 0.6058] | **FAIL** |
| 2. Ladder Elo + MCTS leg | > 98.1 AND mcts ≥ 0.50 | 43.7 [−63.2, 150.5]; mcts 0.4625 | **FAIL** (both parts) |
| 3. Tactical | 20/20 + 29/29 | 20/20 + 29/29 | PASS |
| 4. Test suites | all green | (final re-run below) | PASS |

### 6.1 Clause 4 — PASS

Final suites on the branch (after the battery): pytest
**502 passed, 1 skipped, 58 xfailed** (log
`ml/runs/ml7-h2/pytest-final.log`); npm test **55/55**; tsc
clean. Clause 4 PASSES.

### 6.2 The decision-rule conclusion

The pre-registered rule (§1.2) had three branches. v61 lands
between the first two, and the honest conclusion is the
opponent-dependent one the rule's third branch anticipates —
but sharper than "both factors contributed":

**The α WAS the cause of v54's weakness vs v44 — and it was
also the source of v54's strength vs v0.** The two halves of
the "α was the cause" branch split:

- vs v44: v61 plays at v44's level (0.51 vs v54's 0.385), CI
  including 0.50 — the α half of the branch is met. The mirror
  (0.545, CI including 0.50) points the same way without
  proving it.
- Ladder: Elo 43.7 (v54: 168.4), MCTS leg still lost — the
  ladder half of the branch FAILS decisively. Cooling α does
  not reproduce v44's Elo on this net; it destroys the
  heuristic-leg White edge that produced v54's Elo.

The α effect on the v54 net flips sign by opponent class:

| Opponent | v54 (α=0.5) | v61 (α=0.125) | Δ |
|---|---|---|---|
| v44 (ML) | 0.385 | 0.510 | **+0.125** |
| v37 (ML) | 0.315 | 0.380 | **+0.065** |
| v54 mirror | — | 0.545 | (cool wins) |
| v0 (heuristic) | 0.665 | 0.460 | **−0.205** |
| v7 (heuristic) | 0.380 | 0.300 | **−0.080** |
| swap vs v7 | 0.345 | 0.320 | −0.025 |
| Ladder Elo | 168.4 | 43.7 | **−124.7** |

In every leg the movement runs through White: cooling α
repairs White vs ML opponents (0.30 → 0.57 vs v44, 0.21 →
0.35 vs v37) and breaks White vs heuristic opponents (0.71 →
0.33 vs v0, 0.75 → 0.35 on the ladder heuristic leg, 0.27 →
0.18 under swap). The hot α=0.5 was load-bearing for the
White-patch net's play against greedy opponents — the patch's
White positions only convert when the net's value is trusted
hot — while the same hot α made the net over-trust its
(White-optimistic) values against ML opponents that punish
over-extension. v54's sweep winner α=0.5 was a REAL optimum
for the heuristic-heavy ladder, not a sweep artefact: the
40-game sweep ranking (0.5 > 0.125) is contradicted
head-to-head vs v44 and in the mirror, but CONFIRMED on the
ladder, where it matters for the bar. The sweep's failure was
not its ranking but its scope: a 40-game vs-v44 sweep cannot
see the opponent-class interaction.

Contrast with v50: v37's net at α=0.125 played at v37's level
on BOTH axes (vs v44 0.65 ≈ v37's 0.69, Elo −26.1 ≈ v37's
−34.9, mirror 0.51) — α was neutral there, and v44's Elo gain
came from its labels. For the v54 net, α is NOT neutral: it is
a strong, opponent-dependent lever (+0.125 vs v44, −0.205 vs
v0). The two ablations together say α's effect is
net-dependent: on the 2-ply base net it does nothing, on the
White-patch net it trades ML strength against heuristic
strength. **α for the White-patch net is now CLOSED as a
bar-passing lever in either direction**: 0.5 fails clause 1,
0.125 fails clauses 1 and 2, and the mirror shows no decisive
edge at either end (the middle, 0.25, was the worst sweep arm
at 0.3625 vs v44).

### 6.3 What this means for the program

- v44 remains the ML champion and bar reference. No champion
  change, no daily cross-track change.
- The White-patch direction (v48/v54) has now been ablated on
  its α: the patch's White gains vs weak opponents are
  α-dependent, not net-intrinsic. H1's v60 (a White fix that
  survives 3-ply labelling) should be read against this: a fix
  that only shows up at a hot α will pay the same
  ML-opponent tax v54 paid.
- The ladder-Elo / played-strength divergence now has a
  within-net demonstration: the SAME net posts Elo 168.4 at
  α=0.5 and 43.7 at α=0.125, while its h2h vs the champion
  moves the opposite way (0.385 → 0.51). The Elo is a
  heuristic-leg artefact as much as a strength measure — the
  clause-2 MCTS-leg rule helps, but v61 fails both parts, so
  no verdict here turns on it.
