# Two tracks — heuristic vs ML model

Yudong's direction (2026-11-15): the heuristic track and the ML-model
track each report improvement against **their own best**, every
iteration is designed to make steady progress, and **once a day** the
two tracks are measured against each other and the relative strength
reported to him. Long-term expectation: one day the ML model surpasses
the heuristic. The work continues autonomously; he runs nothing.

## Track A — heuristic (hand-designed scoring + search)

- **Bar reference: v7** (Elo 190.8) — the only version that has passed
  all 4 clauses of the promotion bar. The bar is unchanged:
  (1) h2h vs v7 ≥100 games, win rate > 0.50 with Wilson 95% CI lower
  bound > 0.50; (2) ladder Elo > 190.8; (3) tactical ≥ 20/20 + 29/29;
  (4) all test suites green.
- **Strength champion: v18** (rollout leaf, both colours) — h2h vs v7
  0.775, ladder Elo 636.4 [240.3, 1032.5], first version ever to beat
  the ladder MCTS outright (14-6). It has NOT passed the full bar
  (tactical 25/29), so it is the strength champion, not the bar
  champion. Lives on `research/batch-7` (`970ecf3`), worktree
  `~/workspace/projects/gomoku-b7/`.
- New heuristic versions are measured against v7 for the bar and
  against v18 for strength. A new strength champion is declared when
  a version beats v18's h2h-vs-v7 rate AND its ladder Elo on the same
  battery (both measures, not one).

## Track B — ML model (learned nets)

- **Champion: v8** (v6's PUCT MCTS + learned value leaf; ResNet
  16ch/2 blocks trained on the B1 dataset — 4,350 recorded games,
  1,449 unique (dedup finding, ML batch 1) — α=0.25) —
  ladder Elo 0.0, vs v0 0.545, vs v1 0.48, tactical 20/20 + 29/29.
  It is the best net-using version by both Elo and vs-v0 (v11: Elo
  0.0, vs v0 0.435; Colab longrun MobileNet: Elo −400.0; CPU round 2
  ResNet: Elo −400.0). Lives on master (`bf866f7`); its checkpoint is
  gitignored data at `ml/runs/b3-h2/value-net.pt` (present in the main
  worktree; copy it into any worktree that needs to run v8).
- **ML track's own bar** (same 4-clause shape, reference v8):
  (1) h2h vs v8 ≥100 games, win rate > 0.50 with Wilson 95% CI lower
  bound > 0.50; (2) ladder Elo > 0.0; (3) tactical ≥ v8's (20/20 +
  29/29); (4) all test suites green.
- A new ML champion is declared when a version passes all 4 clauses
  of the ML bar.

## Daily cross-track measurement

- **What:** the current strength champion of each track plays a
  40-game normal-rules match (alternating colours, 100 sims — the
  program standard, just fewer games than the 100-game battery).
  40 games keeps the daily check to roughly an hour of CPU while the
  gap is large; a full 100-game match runs whenever a champion
  changes in either track.
- **How:** `ml/scripts/fast_loop_match.py --version-a <H> --opponent
  v<M> --games 40 --seed-a <date-seed> --seed-b <date-seed+10000>
  --out <cross-track dir>/YYYY-MM-DD.json`, run in the worktree that
  holds the heuristic champion's code, with the ML champion's
  checkpoint copied in. Seeds are date-derived (YYYYMMDD) so each
  day's games are fresh, never the validation battery's seeds
  (100/200) or the fast loop's (1001/2001).
- **Results:** `~/workspace/goals/gomoku-ml-playground/hidden_files/cross-track/`
  (JSON per day + this file's log below). The daily cron (`gomoku-cross-track-daily`,
  ~08:37 PST) runs the match and reports the result to Yudong in the
  Gomoku side chat.
- **Reading the number:** with the current gap (heuristic Elo 636.4
  vs ML Elo 0.0) the heuristic side should win the large majority.
  The number to watch is the trend: the ML track's win rate climbing
  toward 0.50 is the signal that the model is catching up.

## Batch pattern

- The VM has ~1.5 shared CPUs; heuristic and ML batches run
  **alternating**, not in parallel, so neither starves the other.
  Each track gets an iteration roughly every other batch.
- Batch 8 (heuristic) complete 2026-11-15: 0/3; v23 closest
  (h2h vs v7 0.795, best ever; Elo 511.5 < v18's 636.4); v21
  matched v18's Elo (636.4) but tactical 26/29; the blend is
  CLOSED. ML batch 1 complete 2026-11-16: 0/3; v24 closest
  (3/4, Elo 52.5, first ML version above v8's 0.0); v25 refutes
  "more data"; v26's learned prior collapses. Batch 9
  (heuristic) is next.
- **Dataset-dedup finding (ML batch 1):** the "10k dataset" is
  3,244 unique games, not 7,099 — and v8's "4,350" was 1,449
  unique. The B1 parallel generator seeds worker *w* with
  `seed + w`, so consecutive resume seeds replay each other's
  streams. Future legs must space resume seeds ≥ workers apart,
  and dedup-aware holdouts (by unique game) are now essential.

## Cross-track log

| Date | Heuristic | ML | Score (H–ML) | Win rate (H) | Notes |
|------|-----------|----|--------------|--------------|-------|
| 2026-11-15 | v18 | v8 | 7–3 (10 games) | 0.70 [0.397, 0.892] | Setup probe: v18 as Black 4–1, as White 3–2; CI includes 0.50 (n=10 too small for a verdict); ~126s/game → daily 40-game match ≈1.4h; not a daily measurement |
| 2026-11-16 | v18 | v8 | 22–16–2 (40 games) | 0.575 [0.422, 0.715] | Daily #1: v18 as Black 17–2–1, as White 5–14–1; Black won 31/40; CI still includes 0.50 (n=40 can't prove the edge alone); ML win rate 0.425 vs probe 0.30 (within noise) |
