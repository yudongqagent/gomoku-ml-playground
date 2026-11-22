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

All 6 fast-loop losses replayed with the heuristic scans only
(no MCTS): every loss ends with the opponent creating ≥2 winning
moves (an open four / double threat) 2–3 plies before the end;
v50 missed **no win and no block** in any of the 6 losses. The
losses are positional — letting the opponent build the decisive
threat — not tactical blunders. Same pattern as v44's fast loop
(9 losses, 9 via open four, 0 missed blocks). Nothing patchable
here, and nothing to patch: any engine change would break the
ablation.

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

All legs 100 games, validation seeds 100/200, per-game seeding.
Aggregates below are recomputed from the raw per-game records
with `ml/runs/ml5-h3/aggregate.py` (validated: reproduces the
driver's numbers exactly on v44's legs and the v50 fast loops).
The battery survived 1 runtime restart (VM replacement) during
the vs-v44 leg: the chain was killed at 38 games banked
(23W-9L-6D) and resumed cleanly from game 38 with per-game
seeding — zero games lost, no seed change.

### 5.1 vs v44 (clause 1, openings) — COMPLETE

`batch-v50-vs-v44.json`: **61W-31L-8D, win rate 0.6500, Wilson
95% CI [0.5525, 0.7364]** — CI lower bound 0.5525 > 0.50, so
**clause 1 PASSES**. As Black 33W-14L-3D (0.69); as White
28W-17L-5D (0.61) — no White collapse against v44. Uniqueness
**100/100** (openings worked: better than ML batch 4's 91–97);
de-dup rate identical, 0.6500.

First ablation read: v50 beats v44 0.65, close to v37's 0.69
(v44's own leg vs v37 was 29W-67L-4D, 0.31 from v44's side).
Cooling α to 0.125 on the 2-ply net did NOT bring v50 down to
v44's level — v50 still plays at v37's level head-to-head. That
points to the "labels drove the gain" branch of the §1.2
decision rule; the ladder (§6) is the second half of that call.

### 5.2 vs v0 (no-search anchor) — COMPLETE

`batch-v50-vs-v0.json`: **46W-51L-3D, win rate 0.4750, Wilson
95% CI [0.3799, 0.5720]**. As Black 31W-16L-3D (0.65); as White
15W-35L-0D (**0.30**) — the opponent-dependent White collapse
shows up for v50 too (White 0.61 vs v44, 0.30 vs v0; same
pattern as v42's 0.59/0.28 and v44's White 0.16 vs v37).
Uniqueness 98/100; de-dup 0.4745, CI [0.3785, 0.5724].

Comparison (same validation seeds, same opponent — only α
differs): v37 vs v0 was 0.5850 (56W-39L-5D, CI [0.4870,
0.6767]); v44 vs v0 was 0.5800. v50's 0.4750 is 0.11 below
v37's point estimate (CIs overlap, so not conclusive at 95%,
but the direction is clear). Cooling α to 0.125 on the 2-ply
net appears to HURT against the greedy heuristic even while
keeping the edge over v44 — the cooler net value lets the
priors dominate, and against v0's greedy play the net's attack
gradient matters more. The mirror match (§5.5) tests this
directly.

### 5.3 vs v7 (normal rules) — COMPLETE

`batch-v50-vs-v7.json`: **38W-53L-9D, win rate 0.4250, Wilson
95% CI [0.3327, 0.5229]**. As Black 24W-20L-6D (0.54); as White
14W-33L-3D (**0.31**) — the White collapse vs heuristic
opponents again (White 0.61 vs v44, 0.30 vs v0, 0.31 vs v7).
Uniqueness 99/100; de-dup 0.4192, CI [0.3268, 0.5176]. This leg
survived the second runtime restart (chain killed at 73 games,
resumed cleanly from game 73 — zero games lost).

Comparison: v37 vs v7 was 0.4500 (43W-53L-4D); v44 vs v7 was
0.4500 (44W-54L-2D). v50's 0.4250 sits just below both, with
heavily overlapping CIs — the same direction as the vs-v0 leg,
but within noise on its own. The pattern across the two
heuristic legs (v50 below v37 on both, by 0.11 vs v0 and 0.025
vs v7) is consistent with cooling α costing a little against
heuristic opponents.

### 5.4 swap vs v7 (tempo judge, balanced placement, seed 11) — COMPLETE

`batch-v50-vs-v7-swap.json`: **40W-55L-5D, win rate 0.4250,
Wilson 95% CI [0.3327, 0.5229]**. As Black 23W-26L-3D (0.47);
as White 17W-29L-2D (0.375) — under swap the colour split
narrows sharply vs normal rules (0.54/0.31), as the swap rule
removes the first-player confound. Uniqueness **100/100** (swap
placement varies the openings); de-dup identical, 0.4250. This
leg survived the third runtime restart (chain killed at 5
games, resumed cleanly from game 5 — the marathon game 5
replayed deterministically from its per-game seed).

Comparison: v37 swap vs v7 was 0.4400 (42W-54L-4D); v44 swap vs
v7 was 0.3900 (38W-60L-2D). v50's 0.4250 sits between them,
closest to v37 — again the v37-level read, not the v44-level
read.

### 5.5 vs v37 — the mirror match (openings) — COMPLETE

`batch-v50-vs-v37.json`: **48W-46L-6D, win rate 0.5100,
Wilson 95% CI [0.4135, 0.6058]**. As Black 28W-20L-2D (0.58);
as White 20W-26L-4D (0.44). Uniqueness **100/100** — the
openings did their job even in the mirror (same net on both
sides); de-dup identical, 0.5100. This leg survived the
fourth AND fifth runtime restarts (killed at 21 and at 70
games; resumed cleanly both times, zero games lost). Two
225-ply marathon draws in this leg (games 34 and 77) — the
mirror's two identical engines agree so often that neither
finds a forcing line and the board fills.

This is the direct α measurement: the same checkpoint at
α=0.125 vs α=0.25 is statistically indistinguishable over
100 games (0.51, CI [0.4135, 0.6058] spanning 0.50 by a wide
margin). Cooling α does NOT by itself change the 2-ply net's
played strength — and the vs-v0 (0.475 vs v37's 0.585) and
vs-v7 (0.425 vs v37's 0.450) legs suggest that if anything it
costs a little against heuristic opponents, though both CIs
overlap. The mirror result is the ablation's cleanest single
number: α alone cannot reproduce v44's Elo gain.

## 6. Ladder

`ladder-v50.json` (40 games/opponent, seed 100, mcts_sims=100,
complete=true):

| Opponent | W-L-D | Rate | 95% CI | Black | White |
|---|---|---|---|---|---|
| random | 40-0-0 | 1.0000 | [0.9124, 1.0000] | 20-0-0 | 20-0-0 |
| heuristic | 18-21-1 | 0.4625 | [0.3182, 0.6134] | 15-5-0 | 3-16-1 |
| mcts | 23-17-0 | 0.5750 | [0.4219, 0.7149] | 12-8-0 | 11-9-0 |

**Ladder Elo: −26.1, interval [−132.4, 80.2]** (anchors
random=−800, heuristic=0; the estimate comes from the
heuristic leg, as usual — the mcts leg is reported but does
not enter the Elo).

The heuristic leg is the White collapse in its sharpest form:
v50-as-Black 0.75 (15W-5L-0D) vs v50-as-White 0.175
(3W-16L-1D) against the SAME frozen baseline heuristic. The
mcts leg shows no such split (Black 0.60, White 0.55), so the
collapse is opponent-dependent — it bites against heuristic
opponents, not against searchers. (v48 / H1 is diagnosing
exactly this in this batch.)

Comparison: v44's ladder was random 40-0-0, heuristic
24W-13L-3D (0.6375; Black 0.675, White 0.60), mcts 25W-15L-0D
(0.625) → Elo 98.1 [−12.1, 208.2]. v50's Elo (−26.1) sits
next to v37's (−34.9, 20-game legs), not v44's — and even the
top of v50's interval (80.2) is below the 98.1 bar. The
ladder survived the sixth runtime restart (killed mid-mcts
leg; resumed from the mcts leg, random + heuristic marked
"(resumed)").

## 7. Verdict per clause + the ablation finding

**v50 is REJECTED, 3/4 clauses passed.**

| Clause | Bar | v50 | Verdict |
|---|---|---|---|
| 1. h2h vs v44 (100 games) | > 0.50, CI lower > 0.50 | 0.6500, CI [0.5525, 0.7364] | **PASS** |
| 2. Ladder Elo | > 98.1 | −26.1 [−132.4, 80.2] | **FAIL** |
| 3. Tactical | 20/20 + 29/29 | 20/20 + 29/29 | **PASS** |
| 4. Test suites | all green | pytest 459 passed / 1 skipped / 58 xfailed; npm 55/55; tsc clean | **PASS** |

### The ablation finding

v50 plays at **v37's level, not v44's** — on every measure:

| Measure | v37 | v50 | v44 |
|---|---|---|---|
| vs v44 (100 games) | 0.69 | 0.65 | — |
| Ladder Elo | −34.9 | −26.1 | 98.1 |
| vs v0 | 0.585 | 0.475 | 0.580 |
| vs v7 | 0.450 | 0.425 | 0.450 |
| swap vs v7 | 0.440 | 0.425 | 0.390 |
| vs ladder mcts (40 games) | — | 0.575 | 0.625 |
| Mirror vs v37 | — | 0.51 | — |

The mirror match is the cleanest single number: the same
checkpoint at α=0.125 vs α=0.25 is 48W-46L-6D (0.51, CI
[0.4135, 0.6058]) — statistically indistinguishable. Cooling α
does NOT by itself change the 2-ply net's played strength,
and against heuristic opponents it costs a little (vs v0
0.475 vs v37's 0.585, vs v7 0.425 vs 0.450 — CIs overlap, so
"costs a little" is the weak read; "does nothing" is the
strong read, and the mirror supports the strong read).

Per the pre-registered decision rule (§1.2), the verdict is
the **labels branch: v44's Elo gain came from its 3-ply
labels, not from the lower α.** The α lever is CLOSED for the
2-ply net: α=0.125 vs α=0.25 is a null result, measured
directly. What remains open is the label-depth lever — v49
(H2, 4-ply labels) is testing whether it continues past
3-ply, and the odd-horizon parity shift (3-ply labels
systematically optimistic, mean +0.1094) is the thing to
watch there.

### What the next batch should try

1. **The White collapse vs heuristic opponents is now the
   top ML-track problem, sharpened by v50.** v50-as-White:
   0.175 vs the ladder heuristic, 0.30 vs v0, 0.31 vs v7 —
   but 0.44 in the mirror and 0.55 vs the ladder mcts. The
   collapse is opponent-dependent, not a general White
   weakness: the 2-ply net's White play specifically fails
   against heuristic (non-searching) opponents. v48's
   diagnosis (H1, this batch) is the input; the next batch
   should test a targeted fix (e.g. White-side training
   positions weighted toward heuristic-opponent games, or a
   White-specific α / value correction) rather than another
   global lever.
2. **Do not retry α tuning on the 2-ply net** — the mirror
   shows it is a null lever at this scale. If α is revisited,
   it should be on a 3-ply (or deeper) net, where the
   interaction with label depth is untested.
3. **Label depth remains the live lever** (1-ply 0.515/Elo
   52.5, 2-ply 0.75/−34.9, 3-ply 0.67/98.1 — depth helps
   once, then parity dominates). v49's 4-ply result decides
   whether it continues; if 4-ply also stalls, the next
   lever is the label *quality* (teacher strength / parity
   correction), not more depth.

### Durability notes

- Six runtime restarts across this version's battery (at 38
  games in the vs-v44 leg, 73 in vs-v7, 5 in swap, 21 and 70
  in the mirror, mid-mcts-leg in the ladder). Every resume
  was clean — zero games lost, zero seed changes. The
  auto-resume + per-game-seeding machinery is proven under
  fire.
- The coordinator-flagged execution-risk security notice
  (H2's session) never recurred in this session.
- No new checkpoint: v50 reuses v37's checkpoint
  (`ml/runs/ml3-h2/value-net.pt`, md5
  `2fde492aae6d44cbe945e15bd5f1f959`); the only code change
  is the registry entry (`value_weight: 0.125`) plus two
  targeted tests.
