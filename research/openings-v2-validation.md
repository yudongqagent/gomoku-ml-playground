# Openings v2 validation — normal-rules openings balanced IN PLAY

Date: 2026-11-28. Branch: `research/fix-openings-rebaseline` (phase 1:
fix the openings + the colour/opening confounding; the re-baseline
ladders are a separate later phase and were NOT run here).

## Why v1 was replaced

The eval-audit v1 set (`research/openings-balanced.json`, 24 openings,
attack gap 0 by the v1 score, sha256
`0b3fe4643b87448272b2e7eb6e36bbc7651a5711d692ac8ac0642e1d56ad16e5`)
was NOT balanced in play: White collapsed from it (v64 as White 7–13
vs the ladder heuristic and 3–17 vs the ladder MCTS; v60 as White
6–14 vs heuristic). The literal clause-2 bar (40–0/40–0/31–9, empty
board) was unachievable under it even for v52 itself (re-baseline
legs under v1: 40–0/24–16/18–22, Elo 70.4,
`research/iterations/ladder-v52-balanced.json`). Attack-gap 0 does
not imply balanced in play, so v2 was selected by PLAYED White win
rate, with the gap-0 criterion kept only as a sanity filter.

The SWAP openings (`research/openings-swap.json`) were NOT touched:
they produce exact 20/20 colour splits and are fine.

## Colour/opening confounding fix (same change)

`play_match` in `ml/gomoku_ml/research.py` (and the same rotation in
the battery drivers `ml/scripts/selfplay_batch.py` and
`ml/scripts/fast_loop_match.py`) used `openings[i % len(openings)]`:
with 24 openings and a 40-game leg, opening `j` was used in games
`j` and `j+24` — same parity, so A played the same colour in both,
and each opening was only ever played by A in one colour. Paired
rotation now: games `2k`/`2k+1` share opening `k`
(`openings[(i // 2) % len(openings)]`), so each opening is played
once per colour per leg. The v2 set therefore has 20 openings: each
appears exactly twice per 40-game leg (once per colour); in a
100-game match each appears 4 or 6 times, always colour-balanced
(the 40-game pairing cycle repeats 2.5 times — NOT 5 times each as
an early plan stated; the per-opening colour balance is what the
rotation guarantees). The swap-openings mechanism is UNCHANGED.
Regression tests: `ml/tests/test_research.py`
(`test_play_match_paired_rotation_40_game_leg`,
`test_play_match_paired_rotation_odd_game_count`) and
`ml/tests/test_openings.py`
(`test_openings_batch_paired_rotation_40_game_leg`).

## Candidate pool

Sampler reused, not rewritten:
`ml/scripts/generate_balanced_openings.py::gen_balanced` (same sanity
constraints as v1: attack gap 0, no five, no open four, compact
interacting region within Chebyshev distance 2, distinct openings).
Driver: `ml/scripts/sample_openings_v2.py --n 120 --seed 20261127`
(fresh seed; v1 used 20261125). Pool:
`research/openings-v2-candidates.json`.

- Sampled: 278; gap-0 found: 120; rejected open-four: 0.
- 120 distinct openings, ZERO overlap with the v1 set.
- Mix: 75 quiet (best attack 164/164), 45 sharp (4006/4006).

## Screen: v52 vs v44, 2 games per opening

Driver: `ml/scripts/validate_openings.py` (loads both engines once,
plays each candidate with alternating colours — A Black first, then
A White — per-game reseeding `seed + absolute game index`, flushes
to JSON after every candidate, auto-resumes partial runs; verified
move-for-move identical to `selfplay_batch.py` on a cross-check
candidate). Seeds 41001/42001 (fresh — not the battery 100/200 or
the fast-loop 1001/2001). Output:
`research/iterations/openings-v2-screen-v52-v44-{a,b,c}.json`
(chunks of 40 candidates; 240 games total).

Discriminator: with A = v52 the stronger engine, White wins 1/2
over the pair means v52 won both (neutral); 0/2 means Black won
both (Black-favoured); 2/2 means White won both (White-favoured).

- 120 candidates, 240 games, White 105/240 = **0.4375** overall.
- Distribution: 0/2: 25 (Black-favoured, rejected), 1/2: 85 (pass),
  2/2: 10 (White-favoured, held as balance spares).
- The 2-game screen DID discriminate (29% rejected) — it was not
  saturated by the v52-vs-v44 skill gap — but a single pair is
  noisy, so the final set rests on confirmation, not on the screen
  alone (see below: candidate 3 screened 1/2 but went 2/2 under
  both confirmation pairs).

## Confirmation: 3 further pairs on 40 openings

Confirmation set: `research/openings-v2-confirm.json` — 30
screen-passers (every 3rd passer in candidate order + 1: 18 quiet,
12 sharp) + all 10 White-favouring (2/2) screen candidates as
balance spares (7 quiet, 3 sharp). Each of the 40 was played 2
further games under each of three pairs (6 games/opening, 240
games total), all with fresh seeds:

- v52 vs v27, seeds 43001/44001 →
  `research/iterations/openings-v2-confirm-v52-v27.json` —
  White 26/80 = **0.325**.
- v44 vs v37, seeds 45001/46001 → see the v44/v37 note below.
- v52 vs v44, fresh seeds 47001/48001 (independent of the screen
  seeds) → `research/iterations/openings-v2-confirm-v52-v44.json`
  — White 35/80 = **0.4375**.

6-game totals over the 40 (screen 2 + v52/v27 2 + v52/v44-fresh 2):
White 111/240 = **0.4625**. Distribution of per-opening totals:
1/6: 2 openings (pool 46, 55 — the most Black-leaning), 2/6: 11,
3/6: 21, 4/6: 6. No 0/6, 5/6 or 6/6: no confirmed opening is
decisively biased across the three pair/seed combos. The two
screen-passers that looked White-favouring in the probes
regressed on confirmation (candidate 3: screen 1/2, then 2/2
under both v44/v37-probe and v52/v27-probe seeds — but those
probes used different per-game seeds than the confirmation runs;
under the confirmation seeds it is 4/6).

### The v44/v37 pair (slow, deferred, then run on all 40)

v44-vs-v37 games are long drawn ML-vs-ML games: a single game
from candidate 0 took 134 s alone on this VM (221 plies, draw;
~0.6 s/ply) versus ~30 s for v52-vs-v44 games. The pair was
therefore run LAST and alone on the machine: first on the 20
finalists selected on the 6-game totals (see below), then on the
remaining 20 confirm openings, so the final re-selection could
use 8-game totals for all 40. Seeds 45001/46001 throughout.

- On the 20 six-game finalists:
  `research/iterations/openings-v2-confirm-v44-v37.json` —
  White 12/40 = **0.30**. Per-pair White rates on those 20:
  screen 26/40 = 0.65, v52/v27 17/40 = 0.425, v52/v44-fresh
  17/40 = 0.425, v44/v37 12/40 = 0.30; 8-game total 72/160 =
  0.45. The pair-dependence is real and is reported, not hidden:
  no 20-opening set is 0.50 under every pair at 2 games/pair/
  opening of measurement, and the ML pair systematically sees
  these openings as more Black-leaning than the heuristic pairs
  do. The 6-game selection had hit 0.50 exactly on its own data
  by offsetting a White-leaning screen (0.65) against
  Black-leaning confirmations — i.e. it had fitted pair-level
  noise. That is why the final set was RE-SELECTED on the full
  8-game totals (below) instead of keeping the 6-game set.
- On the remaining 20:
  `research/iterations/openings-v2-confirm-v44-v37-rest.json` —
  White 20/40 = **0.50**. Full v44/v37 over all 40 confirm
  openings: 32/80 = **0.40**.

8-game totals over the 40 (2 games x 4 pair/seed combos):
White 143/320 = **0.447**. Distribution: 1/8: 2 (pool 46, 55 —
the most Black-leaning, consistent across all four combos),
2/8: 3 (pool 38, 82, 116), 3/8: 14, 4/8: 13, 5/8: 7, 6/8: 1
(pool 91 — the most White-leaning). No 0/8 or 7-8/8.

## Final selection (20 openings)

Selection rule (on the 8-game totals, deterministic, no
hand-picking): from the 34 openings with totals in {3,4,5}
(within 1 game of balanced — the 1/8, 2/8 and 6/8 openings are
excluded), take all 13 at 4/8, drop the highest-index sharp four
(pool 42), and add the 4 lowest-index 3/8 (2 quiet + 2 sharp)
and the 4 lowest-index 5/8 (2 quiet + 2 sharp). Result: 20
openings, 8-game White sum **80/160 = 0.50**, 13 quiet / 7 sharp.

Per-opening validation (White wins / 8 games; pairs in order
screen v52/v44, v52/v27, v52/v44-fresh, v44/v37):

| pool | kind | scr | v27 | v44x | v37 | total |
|-----:|------|----:|----:|-----:|----:|------:|
| 0 | sharp | 2 | 1 | 0 | 0 | 3/8 |
| 1 | sharp | 2 | 0 | 1 | 0 | 3/8 |
| 3 | sharp | 1 | 2 | 1 | 0 | 4/8 |
| 4 | sharp | 1 | 1 | 2 | 0 | 4/8 |
| 7 | quiet | 1 | 0 | 1 | 1 | 3/8 |
| 15 | quiet | 1 | 1 | 1 | 1 | 4/8 |
| 18 | quiet | 1 | 2 | 0 | 2 | 5/8 |
| 20 | quiet | 2 | 1 | 0 | 0 | 3/8 |
| 28 | quiet | 2 | 1 | 1 | 1 | 5/8 |
| 29 | quiet | 1 | 1 | 1 | 1 | 4/8 |
| 32 | sharp | 1 | 1 | 1 | 1 | 4/8 |
| 58 | quiet | 1 | 1 | 1 | 1 | 4/8 |
| 70 | quiet | 1 | 0 | 1 | 2 | 4/8 |
| 78 | quiet | 1 | 0 | 1 | 2 | 4/8 |
| 80 | sharp | 2 | 1 | 1 | 1 | 5/8 |
| 94 | sharp | 1 | 1 | 1 | 2 | 5/8 |
| 97 | quiet | 1 | 1 | 1 | 1 | 4/8 |
| 100 | quiet | 1 | 1 | 1 | 1 | 4/8 |
| 113 | quiet | 2 | 0 | 1 | 1 | 4/8 |
| 119 | quiet | 1 | 1 | 1 | 1 | 4/8 |

Per-pair White rates on the final set: screen v52/v44 26/40 =
0.65, v52/v27 17/40 = 0.425, v52/v44-fresh 18/40 = 0.45,
v44/v37 19/40 = 0.475; pooled 80/160 = 0.50. The screen's
0.65 is selection bias, stated plainly: six of the ten
White-favouring screen spares entered the set through their
screen scores (2/2), and the screen pair is the only one that
rates the set White-leaning — the other three pairs cluster
0.425–0.475. The honest summary: the set is balanced at 0.50
on the pooled 8-game evidence with no extreme opening, but
pair-level rates still span 0.425–0.65, so the re-baseline
ladders (phase 2) are the real test of whether White still
collapses — this validation cannot prove that on its own.

The interim 6-game selection
(`research/openings-v2-finalists.json`, 20 openings, White
60/120 = 0.50 on 6-game totals) is SUPERSEDED by this 8-game
re-selection and kept only as an audit trail. The shipped set
is `research/openings-balanced.json` (sha256
`ab866456f0cbf3411adc5092b24d4b4607bc681d4ac0d9ac1ead349aedd0a008`).

Total validation games: 240 (screen) + 160 (v52/v27 +
v52/v44-fresh confirmation) + 80 (v44/v37 over all 40) = 480
games, plus probes/cross-checks. All seeds fresh (screen
41001/42001, v52/v27 43001/44001, v44/v37 45001/46001,
v52/v44-fresh 47001/48001); none is the battery 100/200 or the
fast-loop 1001/2001. Two runtime restarts during the run killed
background workers mid-run; every run flushed per candidate and
auto-resumed, so no games were lost or double-counted (resume
reproduces the uninterrupted run exactly by per-game seeding).

## Files

- `research/openings-balanced.json` — the v2 set (20 openings).
- `research/openings-v2-candidates.json` — the 120-candidate pool.
- `research/openings-v2-confirm.json` — the 40-opening confirmation set.
- `research/iterations/openings-v2-screen-v52-v44-{a,b,c}.json` — screen.
- `research/iterations/openings-v2-confirm-*.json` — confirmation.
- `ml/scripts/sample_openings_v2.py`, `ml/scripts/validate_openings.py`
  — the sampling/validation drivers.
