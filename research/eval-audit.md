# Evaluation-framework audit and fix — 2026-11-25

Yudong's challenge (2026-11-25 07:48 PST): *"double check your evaluation
framework. Did you ensure fair opening for swap rules? Did you also evaluate
free play carefully with enough kinds of games? We want steady improvement
over the iterations and I think the sloppy evaluation will make the progress
useless."*

This document is the audit (read from the code AND the actual records, not
from memory) and the fix, on branch `research/eval-audit` from master
`a8155d8`. No past verdict changes: v52 stays champion, v58/v63 stay
rejected. No full battery was re-run. `baseline.py` is untouched (FROZEN).

## 1. Findings

### 1a. Swap fairness — the comparison is fair, the openings are not

The 3-stone placement is **systematically Black-favoured by a fixed
amount**. In every swap match on record, `opening_best_black` is always
163 or 164 and `opening_best_white` is always 86: the attack gap is always
+77 or +78 in Black's favour (never 0, never White-favoured). The gap
minimisation has a structural floor of 77 because Black has 2 stones vs
White's 1. A 3-stone opening with attack gap 0 is impossible under this
placement.

The tempo judge therefore picks Black ~78–88% of the time:

| source | judge picks Black |
|---|---|
| 100-game swap batteries (v52, v58, v63) | 88/100 each |
| full cross-track matches (100 games) | 82/100 |
| daily 40-game matches | 27–33/40 (67–82%) |
| `swap-judge-split-200.json` (200 openings) | 157 Black / 43 White |

Tempo-margin histogram over those 200 openings: margin 0 in 43, margin 78
in 157. Attack-gap histogram: 77 in 33, 78 in 167. The attack judge picks
Black 200/200. The tempo judge is not biased — it is correctly reading
openings that are Black-favoured. The swap choice is nearly deterministic
(88% Black), so the "choice" is mostly not a choice.

**But the colour assignment is fair in aggregate.** Each engine plays Black
~50% of the time (A: 50/100 in full matches, 52/48 in the 100-game swap
batteries, 16–23/40 in daily matches), because the opener alternates (50/50)
and the judge is engine-independent. The comparison is symmetric. Black wins
46–67% of swap games; the second player wins 46–68% when taking Black. The
colour advantage is real and the second player gets it ~88% of the time —
but it is shared equally.

Opening variety is low: only **24 distinct 3-stone openings** exist in the
placement's reachable set (White stone drawn from 24 cells within Chebyshev
distance 2 of centre; Black's second stone chosen deterministically to
minimise the gap). Most common opening repeats 9/100 times. Swap games
themselves are mostly unique (v52 swap 100/100, v58 swap 99/100, v63 swap
77/100, daily 40/40).

**Verdict:** the swap COMPARISON is fair (symmetric), but the OPENINGS are
not balanced (systematically Black-favoured by a fixed amount) and the
choice is nearly deterministic (88% Black). The swap match tests play from
Black-favoured openings, not from balanced positions.

A further structural fact found while building the fix: of those 24 distinct
placement openings, only **5 have tempo margin 0** and 19 have margin 78
(measured exactly, `_setup_value` for both colours on all 24). So 12
distinct margin-0 openings cannot be drawn from the placement at all — the
first version of the openings generator looped forever on exactly that
(it ran 9+ minutes without producing a file; the loop had no sample cap).

### 1b. Free-play variety — normal-rules clause-1 collapses for deterministic engines

Uniqueness (distinct move sequences / 100 games, de-dup by full move list):

| match | unique / 100 |
|---|---|
| v52 vs v27 (clause-1) | 95 |
| v57 vs v52 | 93 |
| v58 vs v52 | 67 |
| **v63 vs v52 (clause-1)** | **31** (one line ×21, another ×13) |
| v52 vs v0 | 22 |
| **v63 vs v0** | **11** |
| v44 vs v8 / vs v37 / vs v0 / vs v7 (ML legs) | 93–99 |
| daily 2026-11-16, normal rules (v18 vs v8) | 32/40 |

Root cause: all normal-rules games start from the empty board, and the
heuristic MCTS engines are deterministic given the position (RNG only breaks
exact ties). Per-game seeding does not create variety when the move choice is
position-dominated. `evaluate.uniqueness_audit`'s docstring already knew this
("vs-v0 legs collapse to 41–45 unique even with no resume, because the
engines are deterministic and RNG is consumed only on exact ties") — but it
was only a verification practice, not a gate. `research/openings-ml4.json`
has only 10 openings (4 moves each) and is used only for ML-vs-ML legs, not
heuristic clause-1.

The nominal 100-game CI overstates precision when uniqueness is low.
v63 clause-1: nominal CI [0.4329, 0.6249] (width 0.192) vs de-dup CI
[0.4077, 0.7358] (width 0.328) — **1.7x wider**. Effective n is 31, not
100. v52 vs v0 de-dup CI [0.8513, 1.0] vs nominal [0.963, 1.0]; v63 vs v0
de-dup 0.9091, CI [0.6226, 0.9838].

### 1c. Bar robustness — which verdicts are safe, which are fragile

- **v52's bar pass is FRAGILE.** Clause-1 nominal 0.61, CI lower 0.5120
  (passes); de-dup 0.60, CI lower 0.4995 (**fails by 0.0005**). Uniqueness
  is 95/100 (good), so the fragility is because the win rate is close to
  the threshold, not low variety. Ladder is saturated (40-0/40-0/31-9) but
  the mcts leg's uniqueness is unverifiable (no records).
- **v58's rejection is fragile on clause 2.** Clause-1 nominal 0.835, de-dup
  0.7537 — both pass comfortably. Rejection came from clause 2 (mcts leg
  27-13 vs v52's 31-9, a 4-game gap) in a leg with no records — the gap
  could be less reliable than it looks.
- **v63's rejection is ROBUST.** Clause-1 nominal 0.53 (CI lower 0.4329,
  fails) and de-dup 0.5806 (CI lower 0.4077, fails). Both fail.

### 1d. Ladder — no records, unverifiable

Ladder JSONs store only summary stats (W/L/D, CI, as_black/as_white). NO
per-game records. Uniqueness is completely unverifiable for all legs. The
ladder is played from the empty board via `research.play_match`, alternating
colours, with the player's RNG advancing across legs (not per-game
reseeded). The ladder Elo is computed from the heuristic leg only (anchored
on random/heuristic) and is blind to the mcts leg — the leg that
discriminates between strong versions.

## 2. Fixes (branch `research/eval-audit`)

### 2a. Varied openings for ALL matches

Two new fixed sets, generated by `ml/scripts/generate_balanced_openings.py`
(deterministic, seed 20261125):

- **`research/openings-balanced.json`** — 24 four-move openings for
  normal-rules matches. Every opening has attack gap 0 (best Black attack
  == best White attack), no five, no open four, compact interacting region.
  Sampled 57 to find 24 gap-0. Balance classes: 7 sharp (4006/4006),
  16 quiet (164/164), 1 intermediate (406/406). All 24 distinct.
- **`research/openings-swap.json`** — 24 three-stone swap openings, curated
  to a 50/50 tempo-margin mix: 12 with tempo margin 0 (judge ties ->
  White) and 12 with tempo margin 78 (judge picks Black), so the judge's
  pick is 12 Black / 12 White over the set. Sampled 399 from the broad
  compact 3-stone space (Black centre, White within Chebyshev 2 of centre,
  Black 2 within Chebyshev 2 of White or centre) and curated by the exact
  tempo margin. The openings are NOT drawn from the balanced placement
  (only 5 distinct margin-0 openings exist there — see 1a).

The two margin classes are arranged in **pairs [0,0,78,78] x 6, not
strictly alternating [0,78,...]**. Strict alternation has period 2, which
resonates with the opener alternation (also period 2: game i's opener is A
iff i is even) and gives engine A Black in EVERY game — the pick is 50/50
but the colour assignment is maximally unfair. This was caught by the
test `test_default_swap_openings_used_and_picks_balanced`, which failed
with `a_colours == {'black': 4, 'white': 0}` under the interleaved
arrangement. Pairs have even-length blocks, so the opener alternation
averages out within each block and each engine plays Black exactly half
the time (for any battery length that is a multiple of 4), while the pick
stays 50/50.

Drivers now default to these sets (resolve from the script location, not
CWD): normal-rules legs use the balanced set unless `--openings` names
another file or `--no-openings` requests the empty board; swap legs use
the curated set unless `--swap-openings` names another file or
`--no-swap-openings` requests the per-game placement generator. The
auto-resume check includes the openings hashes, so a run never resumes
under a different openings set.

### 2b. Uniqueness quality gate

`evaluate.uniqueness_gate(records, threshold=0.6)` (new, in
`ml/gomoku_ml/evaluate.py`, next to `uniqueness_audit`):
returns `unique_games`, `total_games`, `required_unique =
ceil(threshold * total)`, `pass`, and a note. A battery that FAILS is
flagged INVALID and must be re-run with more varied openings/seeds — its
numbers stand as recorded but no verdict may rest on them.

Threshold justification: below 60% distinct games the effective sample
size is <60% of nominal, so the CI width is understated by more than
sqrt(1/0.6) ≈ 1.29x; at v63's 31% it was understated 1.7x. 0.6 is a
floor, not a target — a healthy battery should be near 100% distinct
(the validation battery below is 12/12 on both legs).

Every battery record now reports the gate: `selfplay_batch.py` and
`fast_loop_match.py` include `uniqueness_gate` in the summary and print
it at the end of a run; `eval_version.py` reports the gate per ladder
leg under `ladder.uniqueness` and prints it. Ladder legs now record
their per-game move lists (`research.ladder_leg_for_player` /
`ladder_for_player` take `openings` and `record_games=True` by default),
so leg variety is verifiable for the first time.

### 2c. Swap fairness recording

Every swap match record now carries the pick distribution and the colours
each engine played: `swap_picks` (judge's Black/White picks) and
`a_colours` (games engine A played as Black/White) in the
`selfplay_batch.py` summary, and `second_colour` per game in
`research.play_match` records. The summary also records
`swap_openings_hash` / `swap_openings_count` alongside the existing
`openings_hash`.

### 2d. Tests

- `test_openings.py`: the shipped balanced set (24, gap 0, distinct, no
  open four), the shipped swap set (24, 3 moves, 2B+1W, 12 margin-0 +
  12 margin-78, picks 12/12, spot-checked against the exact judge on one
  opening of each margin), the defaults (a batch with no `--openings` uses
  the balanced set; a swap batch with no `--swap-openings` uses the curated
  set and records picks 2/2 and colours 2/2 over 4 games).
- `test_per_game_seeding.py`: `uniqueness_gate` at/below threshold and
  the empty-list case; the resume helpers now mirror the driver's defaults
  (balanced / curated openings, correct hashes) — the old helpers built
  partials from the empty board with `None` hashes, which the resume
  check now (correctly) rejects.
- `test_research.py`: `play_match` records openings/full moves/swap
  picks; ladder legs record games.
- `test_eval_version_resume.py`: the ladder partial now uses the balanced
  openings and carries its hash.

Full suite: **503 passed, 58 xfailed** (ml pytest, 871s); **npm test
55/55, `npx tsc --noEmit` clean**. The first full-suite run after the
fix failed 3 resume tests (the helpers above); all three pass after the
helper update, and the re-run is fully green.

## 3. Validation battery (fixed framework)

v52 vs v27, 12 games per leg, seeds 71001/72001, placement-seed 71025,
outputs in `/tmp/eval-audit/` (validation only, not a real battery):

| leg | result | win rate | 95% CI | unique | gate | picks | A colours |
|---|---|---|---|---|---|---|---|
| normal rules (balanced openings) | 9W-3L-0D | 0.75 | [0.4677, 0.9111] | 12/12 | PASS | — | — |
| swap (curated set) | 9W-3L-0D | 0.75 | [0.4677, 0.9111] | 12/12 | PASS | 6 Black / 6 White | 6 Black / 6 White |

Both legs are sane: v52's known 100-game rates vs v27 are 0.61 (normal)
and 0.655 (swap), and 0.75 in 12 games is within noise (both CIs contain
0.61 and 0.655). De-dup win rate equals the nominal rate on both legs
(all games unique), so the nominal CI is no longer overstating precision.
The swap pick is exactly 6/6 and each engine plays Black exactly 6/12 —
the before/after for the pick distribution is 88% Black (uncurated) ->
50% Black (curated), with the colour assignment still exactly 50/50.

## 4. Open decisions (for the parent)

1. **Should v52's clause-1 be re-run under the fixed framework?**
   Its pass is fragile: nominal CI lower 0.5120 passes, de-dup CI lower
   0.4995 fails by 0.0005. A 100-game re-run with the balanced openings
   would settle whether the 0.61 holds when every game starts from a
   distinct balanced opening. Cost: ~100 games × ~50s ≈ 1.5h on this VM
   (v52 is slow). This does NOT change v52's champion status unless the
   re-run fails — and a re-run failure would raise the question of whether
   the bar itself needs re-registration, not just a re-verdict.
2. **Should v58's clause-2 mcts leg be re-run?** Its rejection rests on a
   4-game mcts-leg gap (27-13 vs v52's 31-9) in a leg with no records and
   unverifiable uniqueness. A re-run with the fixed framework (recorded
   legs, balanced openings) would test whether the gap is real. v58 stays
   rejected unless a re-run is authorised AND it passes — do not change
   the verdict on the basis of this audit alone.
3. **v63 needs no re-run on variety grounds.** Its rejection is robust:
   both the nominal and the de-dup clause-1 fail.
4. **The ladder under the fixed framework.** Ladder legs now record games
   and use the balanced openings, but no ladder has been run under the
   fixed framework yet. The next ladder run will be the first with
   verifiable leg variety. The ladder Elo is still computed from the
   heuristic leg only and is blind to the mcts leg — the central
   measurement problem (leg-result reading vs decisive h2h) is unchanged
   and still undecided.
5. **Gate threshold 0.6** is a judgement call (justified in 2b). If a
   future battery legitimately sits at 55–65% distinct (e.g. a very
   strong-vs-weak pairing that converges), the gate will flag it INVALID
   and force a re-run with more openings — that is the intended behaviour,
   but the threshold itself can be revisited with data.
6. **The (406,406) opening.** One of the 24 balanced openings has
   best-attack (406, 406) — a third balance class between quiet (164)
   and sharp (4006). It is balanced (gap 0) and valid; it is simply not
   counted in the file's `sharp_4006` / `quiet_164` meta buckets
   (7 + 16 = 23 of 24). Cosmetic only.

## 5. What changed (files)

- `ml/gomoku_ml/evaluate.py` — `UNIQUENESS_GATE_THRESHOLD = 0.6`,
  `uniqueness_gate()`.
- `ml/gomoku_ml/research.py` — `play_match` takes `openings`,
  `swap_openings`, `record_games` (records opening, full moves,
  `second_colour` for swap); `ladder_leg_for_player` /
  `ladder_for_player` take `openings`, record games by default.
- `ml/scripts/common.py` — `default_openings_path()`,
  `default_swap_openings_path()`.
- `ml/scripts/selfplay_batch.py` — balanced openings default,
  curated swap openings default, `--no-openings` / `--no-swap-openings`,
  `swap_picks` / `a_colours` / `uniqueness_gate` in the summary,
  openings hashes in the auto-resume check.
- `ml/scripts/fast_loop_match.py` — `--openings` (same default),
  `uniqueness_gate` in the output.
- `ml/scripts/eval_version.py` — balanced openings to the ladder,
  per-leg uniqueness gate, openings hash in the resume check.
- `ml/scripts/generate_balanced_openings.py` (new) — the generator for
  both sets, with the broad swap sampler and the pairs arrangement.
- `research/openings-balanced.json`, `research/openings-swap.json` (new).
- Tests: `test_openings.py`, `test_per_game_seeding.py`,
  `test_research.py`, `test_eval_version_resume.py`.
- This report: `research/eval-audit.md`.

`baseline.py` untouched. No past verdict changed. No full battery re-run.
