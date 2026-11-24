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
