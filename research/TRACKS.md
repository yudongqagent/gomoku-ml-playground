# Two tracks — heuristic vs ML model

Yudong's direction (2026-11-15): the heuristic track and the ML-model
track each report improvement against **their own best**, every
iteration is designed to make steady progress, and **once a day** the
two tracks are measured against each other and the relative strength
reported to him. Long-term expectation: one day the ML model surpasses
the heuristic. The work continues autonomously; he runs nothing.

## Track A — heuristic (hand-designed scoring + search)

- **Bar reference: v27** (q-gap threshold tie-break on v18's rollout
  leaf; Elo 636.4) — the strongest of the two versions that passed
  all 4 clauses in batch 9 (2026-11-17). The bar is unchanged in
  shape, re-anchored on v27: (1) h2h vs v27 ≥100 games, win rate >
  0.50 with Wilson 95% CI lower bound > 0.50; (2) ladder Elo >
  636.4; (3) tactical ≥ 20/20 + 29/29; (4) all test suites green.
  Note clause 2 is now demanding: 636.4 is the estimate from a
  19W-0L-1D heuristic leg, so beating it effectively requires a
  20-0 leg. v27's own numbers vs v7: 0.7900 (CI [0.7002, 0.8583]),
  vs v0 0.9650, swap vs v7 0.7750 (best swap in the program).
  v7 (Elo 190.8) was the bar reference until batch 9.
- **Also passed the bar (batch 9): v29** (VCF-at-root on v18) — vs
  v7 0.7150, ladder Elo 240.8, tactical 20/20 + 29/29. On master
  alongside v27, but NOT the bar reference: v27 dominates it on
  every measure (h2h vs v7, Elo, vs v0, swap). v29's ladder-vs-MCTS
  leg (15-5) is the best ever. Caveat: v29 is slow (~1.65s/move
  mean on quiet positions, ~7s/move in creator-rich games, vs
  v27's ~35ms) and weak vs v0 (0.665, White 0.60) — its defensive
  check picks preventions by prior noise (verified flaw, iter-29
  §3.4); a VCF-filtered search is the batch-10 direction.
- **Strength champion: v18** (rollout leaf, both colours) — h2h vs v7
  0.775, ladder Elo 636.4 [240.3, 1032.5], first version ever to beat
  the ladder MCTS outright (14-6). It has NOT passed the full bar
  (tactical 25/29 on the original suite, 26/29 on the batch-9
  corrected suite — the (9,7) addition), so it is the strength
  champion, not the bar champion. Lives on `research/batch-7`
  (`970ecf3`), worktree `~/workspace/projects/gomoku-b7/`, and on
  master via the batch-9 merge.
- v27 does NOT displace v18 as strength champion: it beats v18 on
  h2h vs v7 (0.7900 > 0.775) but TIES on ladder Elo (636.4 =
  636.4, exact — same 19W-0L-1D heuristic leg), and the champion
  rule needs a beat on BOTH. v27 is the bar reference (it passed
  the bar; v18 never did) while v18 keeps the strength-champion
  title on the tie. The daily cross-track match pairing is
  therefore unchanged (v18 vs v8).
- New heuristic versions are measured against v27 for the bar and
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
  ResNet: Elo −400.0). Lives on master (`17d08db`); its checkpoint is
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
  40-game match under the official swap rule (opener places 3
  stones, second player picks a side, White plays move 4; tempo
  judge; balanced placement), 100 sims — the program standard,
  just fewer games than the 100-game battery. From 2026-11-17 the
  daily match uses swap rules (Yudong's direction 2026-11-16:
  apply the official rules so White gets a real chance); the
  2026-11-16 match was the last normal-rules one. 40 games keeps
  the daily check to roughly an hour of CPU while the gap is
  large; a full 100-game match runs whenever a champion changes
  in either track.
- **How:** `ml/scripts/selfplay_batch.py --engine-a v<H>
  --engine-b v<M> --games 40 --swap-rule --placement balanced
  --placement-seed <date-seed> --judge tempo --seed-a <date-seed>
  --seed-b <date-seed+10000> --out <cross-track
  dir>/YYYY-MM-DD.json`, run in the worktree that holds the
  heuristic champion's code, with the ML champion's checkpoint
  copied in. Seeds are date-derived (YYYYMMDD) so each day's
  games and openings are fresh, never the validation battery's
  seeds (100/200) or the fast loop's (1001/2001).
- **Results:** `~/workspace/goals/gomoku-ml-playground/hidden_files/cross-track/`
  (JSON per day + this file's log below). The daily cron (`gomoku-cross-track-daily`,
  ~08:37 PST) runs the match and reports the result to Yudong in the
  Gomoku side chat.
- **Reading the number:** with the current gap (heuristic Elo 636.4
  vs ML Elo 0.0) the heuristic side should win the large majority.
  Under swap rules the colour advantage is neutralised per game by
  the swap choice, so the win rate is a cleaner strength signal
  than under normal rules (where Black's advantage dominates the
  variance — Black won 31/40 in the last normal-rules daily
  match). The number to watch is the trend: the ML track's win
  rate climbing toward 0.50 is the signal that the model is
  catching up.

## Batch pattern

- The VM has ~1.5 shared CPUs; heuristic and ML batches run
  **alternating**, not in parallel, so neither starves the other.
  Each track gets an iteration roughly every other batch.
- Batch 8 (heuristic) complete 2026-11-15: 0/3; v23 closest
  (h2h vs v7 0.795, best ever; Elo 511.5 < v18's 636.4); v21
  matched v18's Elo (636.4) but tactical 26/29; the blend is
  CLOSED. ML batch 1 complete 2026-11-16: 0/3; v24 closest
  (3/4, Elo 52.5, first ML version above v8's 0.0); v25 refutes
  "more data"; v26's learned prior collapses. **Batch 9
  (heuristic) complete 2026-11-17: 2/3 PASS — v27 and v29 both
  pass all 4 clauses (the second and third bar passes ever, after
  v7); v28 rejected 2/4** (gap-three demotion now priced in all
  four combinations and CLOSED). v27 (τ=0.05 threshold on v23's
  q-aware tie-break) is the new bar reference; v29 (VCF-at-root
  on v18) also passes but is slower and weaker vs v0. The batch
  also applied the pre-registered suite correction (add (9,7) to
  winning-quiet-setup-2's expected list — v23's verdict stands)
  and proved the theirs-term demotion drives v22's vs-MCTS
  collapse (v28 13-7 vs v22's 5-15). ML batch 2 is next.
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
| 2026-11-16 | v18 | v8 | 22–16–2 (40 games) | 0.575 [0.422, 0.715] | Daily #1 (NORMAL rules — last one; series switches to swap from 2026-11-17): v18 as Black 17–2–1, as White 5–14–1; Black won 31/40; CI still includes 0.50 (n=40 can't prove the edge alone); ML win rate 0.425 vs probe 0.30 (within noise) |
