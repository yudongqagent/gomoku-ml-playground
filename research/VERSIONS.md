# Scoring engine versions

Log of scoring-engine versions, their eval numbers, and whether they met
the "better" bar (see `research/README.md`). Only versions that meet the
bar are committed to origin/master.

Ladder Elo is relative to THIS ladder only (anchors random = -800,
heuristic = 0). It is not comparable to human ratings.

| Version | Ladder Elo (95% interval) | Head-to-head vs previous | Tactical | What changed | Commit | Verdict |
|---|---|---|---|---|---|---|
| v0 (baseline) | 0 by definition (anchor) | — | 20/20 (heuristic baseline) | `ml/gomoku_ml/baseline.py`: contiguous-run scoring; broken fours invisible | 78c733a (pre-existing) | anchor |
| v1 (scoring-v1) | 34.9 ([-169.6, 239.3], 10 games/opponent); re-measured 70.4 ([-80.2, 221.1], 20 games/opponent) | vs v0 pooled 200 games: 100W-80L-20D, win rate 0.55, CI [0.4808, 0.6174] | 20/20 | 5-window scan (broken-four fix), max-per-direction, double-threat bonuses, candidate restriction, full-value defence | — (not pushed) | **REJECTED** — head-to-head CI lower bound < 0.50 in both 100-game runs (0.4870, 0.4183) and pooled (0.4808). Colour asymmetry: 0.755 as White vs 0.345 as Black. See `research/iterations/iter-1.md`. |
| v2 (scoring-v2) | -52.5 ([-201.9, 96.9], 20 games/opponent) | vs v1 100 games: 31W-50L-19D, win rate 0.4050, CI [0.3140, 0.5030]; vs v0 100 games: 32W-44L-24D, 0.4400, CI [0.3467, 0.5377]; swap vs v1 100 games: 24W-36L-40D, 0.4400, CI [0.3467, 0.5377] | 20/20 | v1 + defence discount 0.9 (attack bias) | — (not pushed) | **REJECTED** — loses head-to-head to v1 (0.4050) and loses ladder Elo (70.4 -> -52.5 at identical 20 games/opponent). The defence-discount hypothesis is refuted: full-value defence is v1's strength. See `research/iterations/iter-2.md`. |
| v3 (scoring-v3) | 107.5 ([-46.9, 262.0], 20 games/opponent) | vs v1 100 games: 27W-33L-40D, win rate 0.4700, CI [0.3751, 0.5671]; vs v0 100 games: 58W-35L-7D, 0.6150, CI [0.5171, 0.7044]; swap vs v1 100 games: 26W-34L-40D, 0.4600, CI [0.3656, 0.5574] | 20/20 | v1 + 2-ply threat check (prefer a move leaving the opponent with zero live open-four creators, or a forcing move) | — (not pushed to master) | **REJECTED** — loses head-to-head to v1 (0.4700) even though ladder Elo is higher than v1's (107.5 vs 70.4, intervals overlapping). The threat check helps vs weaker opponents but costs vs an equal one (40% draws; the safe move is often a passive block). See `research/iterations/iter-3.md`. |

Notes:
- v0's tactical score (20/20) is the heuristic baseline's score on the
  tactical suite as measured by the eval harness (see `ml/README.md`
  and the round-1/round-2 eval reports).
- v0 is the frozen ladder anchor. It is never modified; new versions
  live in versioned modules and are compared against it.
