# Iteration 1 — scoring-v1 (5-window engine) — 2026-11-12

Worker A (Python track), batch 1. Branch master, HEAD 78c733a at start;
nothing committed by this worker (coordinator owns commits).

## What changed

New versioned engine `ml/gomoku_ml/scoring.py` (`scoring-v1`), same
interface as the frozen baseline (`score_move` / `choose_move` /
`ScoringPlayer`). `baseline.py` untouched — it stays the ladder anchor.

1. **5-window scan** (mirrors `src/ai/patterns.ts`): fixes the
   confirmed blind spot — the baseline counts contiguous runs only, so
   a broken four (`XX_XX`) was invisible (baseline scored a
   broken-four-creating move 164; v1 scores it 20,013, a four).
2. **Correct four valuation**: a 4-in-window always has a winning
   square (the empty cell inside the window), so a closed four scores
   20k, never 0; only a contiguous four with the far outside cell
   empty is an open four (200k). (Deliberate deviation from the TS
   engine, which scores a fully-closed four-window 0.)
3. **Max-per-direction, not sum-over-windows**: a single contiguous
   four sits in two overlapping windows; summing (as TS does)
   double-counts one pattern and lets a lone four masquerade as a
   double threat. Each direction contributes its strongest window.
4. **Double-threat bonuses**: +200k for fours in >= 2 directions
   (forced win next move), +50k for open threes in >= 2 directions.
5. **Cheaper win/block**: only radius-1 candidates of the relevant
   colour are scanned (a winning move must be adjacent to a friendly
   stone).
6. **Candidate restriction**: only empty points within Chebyshev
   distance 2 of a stone are scored (centre on an empty board; all
   legal moves as fallback).
7. **No centre preference** in the score (TS parity); attack and
   defence summed at **full value** (baseline discounts defence to
   90%).
8. New `score_attack` (attack-only scorer) — needed by the swap-rule
   colour choice, because the total score is exactly symmetric
   between colours (see swap-rule.md).

Also in this batch: swap-rule helpers in `game.py` (additive only),
research harness `ml/gomoku_ml/research.py` (`play_match`,
`ladder_for_player`, `run_tactical`), benchmark
`ml/scripts/bench_scoring.py`, eval driver
`ml/scripts/eval_scoring.py`, 46 new tests.

## Profile (before / after)

Fixed midgame board (24 stones, 201 legal moves), `bench_scoring.py`,
best of 10 batches x 20 (numbers vary run to run by ~5% on this VM;
an earlier 5-rep run gave 93,345 / 67,584 moves/s and 3.38 / 1.48 ms):

| | baseline | scoring-v1 |
|---|---|---|
| score_all (all legal moves) | 88,891.7 moves/s best (2.26 ms/pass), 87,692.4 mean | 63,713.9 moves/s best (3.15 ms/pass), 63,249.6 mean |
| choose_move latency | 3.47 ms best, 3.53 ms mean | 1.54 ms best, 1.60 ms mean |

Per-move scoring is 1.40x *slower* (a 20-window scan costs more than
8 short walks) but `choose_move` is 2.25x *faster* overall, because
only ~60 candidates are scored (not 201) and the win/block scans
check ~40 radius-1 points (not 201). For self-play throughput the
`choose_move` number is the one that matters.

## Eval numbers (seeds noted; exact, never rounded up)

Head-to-head vs baseline heuristic, 100 games, seeds scoring=100 /
baseline=200: **54W-37L-9D, win rate 0.5850, 95% CI [0.4870,
0.6767]**; as Black 17W-28L-5D, as White 37W-9L-4D. Elo diff +59.6.

Replication, 100 games, seeds 700/800: **46W-43L-11D, win rate
0.515, 95% CI [0.4183, 0.6106]**; as Black 11W-31L-8D, as White
35W-12L-3D. Elo diff +10.4.

Pooled 200 games: 100W-80L-20D, win rate 0.55, 95% CI [0.4808,
0.6174]; as Black 28W-59L-13D (0.345), as White 72W-21L-7D (0.755).

Tactical suite (20 positions): **scoring-v1 20/20, baseline 20/20**
— the suite does not discriminate between these two engines (both
1-ply with win/block); a harder suite is needed for future
iterations.

Ladder for scoring-v1 (10 games/opponent, mcts_sims=100): vs random
10W-0L-0D; vs heuristic 5W-4L-1D (0.55, CI [0.2737, 0.7986]); vs
mcts 4W-4L-2D (0.50, CI [0.2366, 0.7634]). **Ladder Elo estimate
34.9, interval [-169.6, 239.3]** (anchors random=-800, heuristic=0;
relative to THIS ladder only).

Swap-rule match, 20 games, seeds 500/600: **5W-15L-0D, win rate
0.25, CI [0.1119, 0.4687]**; as Black 5W-5L, as White 0W-10L; the
second player took Black in **20/20** games. See swap-rule.md: the
opening-placement policy (each colour's best move) makes Black
systematically stronger, so the swap choice is degenerate. Fix
proposed for iteration 2: balanced placement.

> **Correction (iteration 2, 2026-11-12): the swap-rule numbers above
> are INVALID as swap-rule evidence.** They were measured with the
> wrong turn order (Black to move after the swap, giving Black 3 of
> the first 4 moves). Per the official RIF rules Art. 2.4, White
> plays move 4. The harness was fixed in iteration 2
> (`research.play_swap_game`, `game.choose_swap_color` tie-break now
> White) and the swap match was re-run; see iter-2.md. The
> normal-rules head-to-head, ladder, and tactical numbers in this
> file are unaffected.

## The colour asymmetry (headline finding)

v1 is much stronger as White and much weaker as Black, and it
replicates across seed pairs (pooled: 0.755 as White vs 0.345 as
Black over 200 games). Controls rule out a harness or format
artefact:

- Baseline vs baseline, 100 games: Black wins 39W-9L as Black —
  the normal first-move advantage. So 1-ply play in general does
  NOT favour White; v1 inverts the advantage in this matchup.
- Scoring vs scoring, 100 games: 31W-32L-37D, near-even, but a 37%
  draw rate (vs 6% for baseline-vs-baseline) — v1 defends well
  enough to grind games to draws against itself.

Loss analysis (6 saved losses as Black in
`v1-black-losses.json`, all 28-30 moves): in every loss White won by
creating an **open four** (two winning squares) around move 26-28.
No missed immediate blocks — the win/block scans work. In the
analysed game 1, Black's move 27 correctly blocked an immediate
White five (defence value 10M, legitimate), but White already held a
second independent threat (open-four setup at (10,8)) and won
through it. The game was lost earlier: White built multiple
simultaneous threats while Black's 1-ply play could only answer one
at a time.

**Hypothesis (not yet proven):** v1's full-value defence +
broken-four vision makes it a strong *defender* but a passive
*attacker*. As White it thrives on responding (blocks everything,
counterattacks with double-threat bonuses); as Black it cannot
convert the initiative — its defensive bias lets the opponent
dictate, and the opponent (whose defence is discounted to 90%, so it
attacks more) wins the race to the double threat. Iteration 2
should test attacking-bias variants for the Black role (e.g. defence
discount, or an attack bonus when no immediate danger) and re-run
the colour-split head-to-head.

## Bar verdict (per research/README.md)

1. **Head-to-head: FAIL.** Win rate 0.5850 / 0.515, but the Wilson
   95% CI lower bound is 0.4870 / 0.4183 in the two runs (pooled
   0.4808) — below the required > 0.50 in every case.
2. **Ladder Elo up: point estimate yes (34.9 vs the 0 anchor), but
   the interval [-169.6, 239.3] is wide and includes 0** (only 10
   games/opponent) — weak evidence, not a clear pass.
3. **No tactical regression: PASS** (20/20 vs 20/20).
4. **Test suites: PASS** (ml pytest and npm test green — see the
   batch report for counts).

**Verdict: v1 does NOT meet the "better" bar** (criterion 1 fails).
Per the program rules it should be recorded in `VERSIONS.md` as
rejected and NOT pushed as the new canonical version. The engine,
harness, and swap-rule code remain in the working tree for the
coordinator to decide (the harness and swap helpers are
infrastructure, not the versioned engine). The colour asymmetry is
the thing iteration 2 must fix first: no version with a 0.345
as-Black win rate against the frozen baseline can clear the bar.
