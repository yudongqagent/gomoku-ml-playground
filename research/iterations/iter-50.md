# Iteration 50 — v50: the α ablation (v37's net at α=0.125)

ML batch 5, H3. Branch `research/ml-batch-5-h3`, from
`research/ml-batch-5` @ `7b65a2d`. Plan:
`research/ml-batch-5-plan.md` §H3.

## 0. The question

v44 differs from v37 in TWO ways:

1. **Labels:** 3-ply rollout-leaf labels instead of v37's 2-ply
   labels (same 817 games / 37,227 quiet positions, same
   architecture, same training recipe otherwise).
2. **α:** `value_weight` 0.125 instead of v37's 0.25.

v44 passed the ML bar (vs v8 0.67, ladder Elo 98.1) while v37
failed clause 2 (Elo −34.9) — yet v37 beats v44 head-to-head
0.69–0.31. Which of the two changes drove v44's Elo gain: the
3-ply labels, or the cooler α? v50 isolates the α effect: **the
v37 net, unchanged, at α=0.125.** No retraining, no new
checkpoint, no new mechanism — a clean ablation.

### 0.1 Premise verification (done before any eval)

The ablation's premise is that α (`value_weight`) is an
**inference-time** parameter of `ValueMCTS`, not a training
parameter. Verified three ways, 2026-11-21:

1. **Code** (`ml/gomoku_ml/mcts_value.py`): `value_weight` is a
   `ValueMCTS.__init__` parameter (`value_weight: float | None =
   None`, defaulting to `VALUE_WEIGHT` 0.25); it is stored as
   `self.value_weight` and used in exactly one place —
   `_value_leaf_value`, where the net's quiet-position value is
   multiplied by it (`return float(value[0, 0].item()) *
   self.value_weight`). Terminal cases are exact and unaffected.
   No training code reads `value_weight`: the training scripts
   (`train_value.py`, `generate_distill_data.py`) build the net
   and the labels with no α anywhere — the checkpoint payload is
   just `model_state` + architecture metadata.
2. **iter-44.md §1.4** (pre-registered): "(Note: α is an
   inference-time weight, so holdout sign accuracy cannot select
   it — the sweep is by played win rate, as in v24/v31/v37.)"
3. **The sweeps themselves**: v37's and v44's α sweeps
   (`sweep_alpha.py --version N --alphas ...`) evaluate the SAME
   checkpoint at different α values (v44: 0.125 → 0.725, 0.25 →
   0.600, 0.5 → 0.475 vs v8 over 20 games each). If α were a
   training parameter, a sweep of one checkpoint would be
   meaningless.

**Premise CONFIRMED: v50 needs no retraining.** v50 is v37's
checkpoint (`ml/runs/ml3-h2/value-net.pt`, md5
`2fde492aae6d44cbe945e15bd5f1f959`) registered with
`value_weight: 0.125`. The only difference between v50 and v37
is the registry's `value_weight` field; the only difference
between v50 and v44 is the checkpoint (2-ply vs 3-ply labels) —
α is the same 0.125 in both.

## 1. Pre-registration (before any eval)

### 1.1 The version

- **v50**: v8's PUCT MCTS (100 sims, c_puct 1.5, v1 priors) +
  v37's value net (`ml/runs/ml3-h2/value-net.pt`, 2-ply
  rollout-leaf labels on v24's 817 games / 37,227 positions,
  ResNet 16ch/2 blocks) at **α = 0.125**.
- Registry entry in `ml/gomoku_ml/scoring.py`: identical to
  v37's except `value_weight: 0.125` (v37: 0.25). No new
  checkpoint file is created; v50 reuses v37's checkpoint
  verbatim.

### 1.2 The comparison and the decision rule

v50 is compared against **both** v37 (same net, α=0.25) and
v44 (3-ply net, α=0.125), on the full 100-game battery and the
ladder:

- **If v50 plays at v44's level** — vs v44 ≈ 0.50 (CI including
  0.50, no clear loss), ladder Elo ≈ 98 — then v44's Elo gain
  came from the **lower α**, not the 3-ply labels. (Corollary:
  the 3-ply labels bought nothing over 2-ply at matched α, and
  v44's h2h loss to v37 would then be a label effect at fixed
  α — v37's 2-ply net beating v44's 3-ply net at their
  respective αs is confounded, and v50-vs-v44 isolates it.)
- **If v50 plays at v37's level** — vs v44 ≈ 0.69 (v37's number),
  ladder Elo ≈ −35 — then the **labels drove the gain**: the
  3-ply labels are what lifted v44's Elo, and α=0.125 on the
  2-ply net is neutral or worse.
- **If v50 lands in between** (e.g. vs v44 ≈ 0.55–0.60, Elo
  between −35 and 98), both factors contributed; report the
  split honestly and do not force a binary verdict.

The mirror match v50-vs-v37 (same net, α 0.125 vs 0.25) is the
direct α measurement: it is the α sweep's 20-game comparison
(0.25 beat 0.5 and 1.0 in v37's sweep; 0.125 was never swept on
the 2-ply net) run at 100 games with opening variety. A v50 win
there means cooling α below 0.25 helps the 2-ply net too.

### 1.3 The ML bar (reference v44, Elo 98.1)

1. h2h vs v44 ≥ 100 games, win rate > 0.50 with Wilson 95% CI
   lower bound > 0.50.
2. Ladder Elo > 98.1 (40 games/opponent).
3. Tactical ≥ 20/20 + 29/29 on the corrected suite.
4. All test suites green.

Fail any clause = rejected. Note: as an ablation, v50's value
does not depend on passing the bar — the finding (α vs labels)
is the deliverable either way. But the bar is applied as
written; a pass would make v50 the ML champion (and change the
daily cross-track pairing), which would be reported explicitly.

### 1.4 Battery (pre-registered commands)

All legs 100 games, validation seeds `--seed-a 100 --seed-b
200`, per-game seeding + uniqueness audit on every leg. ML-vs-ML
legs use the pre-registered `research/openings-ml4.json` (10
openings) via `--openings` (ML batch 4: 91–97 unique games in
100 with openings, vs 21 without for v36-vs-v8).

```sh
# Clause 1: vs v44 (openings)
selfplay_batch.py --engine-a v50 --engine-b v44 --games 100 \
  --openings research/openings-ml4.json --seed-a 100 --seed-b 200 \
  --out research/iterations/batch-v50-vs-v44.json
# vs v0 (no-search anchor)
selfplay_batch.py --engine-a v50 --engine-b v0 --games 100 \
  --seed-a 100 --seed-b 200 \
  --out research/iterations/batch-v50-vs-v0.json
# vs v7 (normal rules)
selfplay_batch.py --engine-a v50 --engine-b v7 --games 100 \
  --seed-a 100 --seed-b 200 \
  --out research/iterations/batch-v50-vs-v7.json
# swap vs v7 (tempo judge, balanced placement, seed 11)
selfplay_batch.py --engine-a v50 --engine-b v7 --games 100 \
  --swap-rule --placement balanced --placement-seed 11 --judge tempo \
  --seed-a 100 --seed-b 200 \
  --out research/iterations/batch-v50-vs-v7-swap.json
# Mirror match: vs v37 (openings) — same net, α 0.125 vs 0.25
selfplay_batch.py --engine-a v50 --engine-b v37 --games 100 \
  --openings research/openings-ml4.json --seed-a 100 --seed-b 200 \
  --out research/iterations/batch-v50-vs-v37.json
# Ladder (40 games/opponent, auto-resume; committed artifact)
eval_version.py --version 50 --ladder-games 40 \
  --out research/iterations/ladder-v50.json
# Tactical, both suites, seed 100 (re-verify v44's tactical first)
eval_tactical_full.py --version 50 --suite both --seed 100
```

Fast loops (10 games, dev seeds 1001/2001, never the validation
seeds) run vs v0 AND v44 before the battery, per the standing
method; the fast loop is a finding method, not a verdict — the
100-game battery is the verdict (small-sample smokes mislead;
dev seeds flatter).

### 1.5 Checkpoints (gitignored, copied into this worktree)

- v44: `ml/runs/ml4-h3/value-net.pt`, md5
  `d22af606f7d2ee0254f372da895ae7ad` (from the main worktree —
  verified).
- v37 (= v50's net): `ml/runs/ml3-h2/value-net.pt`, md5
  `2fde492aae6d44cbe945e15bd5f1f959` (from `gomoku-ml3` —
  verified).
- v8 (tests): `ml/runs/b3-h2/value-net.pt`, md5
  `5deb26f3df67883f24f04800d5356da6` (from the main worktree —
  verified).

## 2. Registry entry + targeted tests

- Registry entry added in `ml/gomoku_ml/scoring.py` (commit
  `b0a2dd2`): `50: {"name": "scoring-v50", ...,
  "value_checkpoint": "ml/runs/ml3-h2/value-net.pt",
  "value_weight": 0.125}` — identical to v37's entry except the
  `value_weight`. No new checkpoint: v50 reuses v37's file
  verbatim (md5 `2fde492aae6d44cbe945e15bd5f1f959`, verified on
  copy).
- Wiring verified live: `get_player(50)` yields
  `value_weight == 0.125` and the same checkpoint path as
  `get_player(37)` (0.25); `ValueMCTS.__init__` has no duplicate
  parameters (the recurring silent-merge trap — checked with
  `inspect.signature`).
- Targeted tests added in `ml/tests/test_value_weight.py`:
  `test_value_weight_v50_registry_passthrough` (entry carries
  v37's checkpoint path and `value_weight` 0.125, differing from
  v37's 0.25; a registry-built player carries 0.125) and
  `test_value_weight_v50_leaf_is_half_v37` (same checkpoint, α
  0.125 vs 0.25: the v50 quiet-leaf value is exactly half the
  v37 leaf value — the shrink is actually applied, not just
  stored). Both pass.
- Full suites on this branch: pytest **459 passed, 1 skipped, 58
  xfailed** (570s); npm test **55/55**; tsc clean. Clause 4
  green at the start of the eval phase (re-run at the end before
  reporting).

## 3. Fast loops

Iteration 1, dev seeds 1001/2001 (never the validation seeds),
10 games each. Logs: `ml/runs/ml5-h3/fast-v50-v0.log`,
`ml/runs/ml5-h3/fast-v50-v44.log`.

| leg | W-L-D | win rate | 95% CI | as Black | as White | unique |
|-----|-------|----------|--------|----------|----------|--------|
| vs v0 | 7-3-0 | 0.7000 | [0.3968, 0.8922] | 4-1-0 (0.80) | 3-2-0 (0.60) | 10/10 |
| vs v44 | 6-3-1 | 0.6500 | [0.3537, 0.8631] | 3-1-1 (0.70) | 3-2-0 (0.60) | 10/10 |

Both legs are consistent with v50 playing at roughly v37's
level (v37's battery numbers: 0.60 vs v0, 0.69 vs v44) — but 10
games cannot distinguish v37-level from v44-level play (the vs
v44 CI spans 0.35–0.86), and dev seeds flatter. No patch is
sought from these loops: v50 is an ablation, and any change to
the engine would break the isolation of the α effect. The loss
read (§3.1) is a smoke check only; the 100-game battery is the
verdict.

### 3.1 Loss read (light)

(TODO — filled in while the battery runs.)

## 4. Tactical (corrected suite, seed 100)

- v44 re-verified first on the corrected suite (log
  `ml/runs/ml5-h3/tactical-v44-reverify.log`): **20/20 old +
  29/29 hard** — the corrected suite does not change v44's
  tactical.
- v50 (`eval_tactical_full.py --version 50 --suite both --seed
  100`, log `ml/runs/ml5-h3/tactical-v50.log`): **20/20 old +
  29/29 hard**. Clause 3 passes. Expected: α only shrinks the
  net's quiet-position value; the tactical positions are decided
  by the exact terminal cases and the threat check, which α does
  not touch.

## 5. Battery

(TODO — recomputed aggregates from raw per-game records.)

## 6. Ladder

(TODO)

## 7. Verdict per clause + the ablation finding

(TODO)
