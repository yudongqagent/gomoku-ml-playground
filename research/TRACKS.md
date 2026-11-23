# Two tracks — heuristic vs ML model

Yudong's direction (2026-11-15): the heuristic track and the ML-model
track each report improvement against **their own best**, every
iteration is designed to make steady progress, and **once a day** the
two tracks are measured against each other and the relative strength
reported to him. Long-term expectation: one day the ML model surpasses
the heuristic. The work continues autonomously; he runs nothing.

## Track A — heuristic (hand-designed scoring + search)

- **Bar reference: v52** (threat-space search at the root; ladder
  Elo unbounded — saturated, interval [407.0, ∞)) — the strongest
  version that passed all 4 clauses in batch 13 (2026-11-22). The
  bar is unchanged in shape, re-anchored on v52: (1) h2h vs v52
  ≥100 games, win rate > 0.50 with Wilson 95% CI lower bound >
  0.50; (2) ladder Elo > v52's (see the leg-result rule below);
  (3) tactical ≥ 20/20 + 29/29; (4) all test suites green.
  **Clause 2 for the v52 reference (pre-registered):** v52's Elo
  is unbounded (its heuristic leg is 40-0-0, the first version ever
  to saturate the 40-game ladder), so clause 2 is judged by the
  leg-result reading — a future version must strictly dominate
  v52's legs (random 40-0, heuristic 40-0, mcts 31-9). In practice
  that means matching the two perfect legs and beating 31-9 on the
  mcts leg. (Or the ladder is re-based with a stronger anchor or
  more games per leg — the saturation problem batch 10 flagged has
  now arrived in full.) v52's own numbers vs v27: 0.6100 (CI
  [0.5120, 0.6998]), vs v0 1.0000 (100-0, the second perfect
  100-game score in the program), swap vs v27 0.6550, challenge vs
  v18 0.6250 (CI [0.5271, 0.7136]). Caveat: v52's clause-1 pass
  rests on the nominal rule (de-dup CI lower 0.4995, misses by
  0.0005 — see iter-52 §9); the challenge match passes on BOTH
  cuts. v27 (Elo 636.4) was the bar reference until batch 13.
- **Also passed the bar (batch 9): v29** (VCF-at-root on v18) — vs
  v7 0.7150, ladder Elo 240.8, tactical 20/20 + 29/29. On master
  alongside v27, but NOT the bar reference: v27 dominates it on
  every measure (h2h vs v7, Elo, vs v0, swap). v29's ladder-vs-MCTS
  leg (15-5) is the best ever. Caveat: v29 is slow (~1.65s/move
  mean on quiet positions, ~7s/move in creator-rich games, vs
  v27's ~35ms) and weak vs v0 (0.665, White 0.60) — its defensive
  check picks preventions by prior noise (verified flaw, iter-29
  §3.4); a VCF-filtered search is the batch-10 direction.
- **Also passed the bar (batch 9): v27** (q-gap threshold tie-break
  on v18's rollout leaf; Elo 636.4) — vs v7 0.7900 (CI [0.7002,
  0.8583]), vs v0 0.9650, swap vs v7 0.7750 (best swap in the
  program until v52). Was the bar reference until batch 13; v52
  dominates it on the ladder (saturated vs 636.4) and the
  challenge (v52 beat v18, which v27 never faced).
- **Strength champion: v52** (threat-space search at the root) —
  h2h vs v27 0.6100, ladder Elo unbounded (saturated), vs v0
  1.0000, swap vs v27 0.6550. It PASSED the full bar (batch 13)
  and beat v18 in the 100-game direct challenge (0.6250, CI
  [0.5271, 0.7136]; de-dup CI lower 0.5106 — passes on both cuts),
  taking the title from v18. Lives on master (`33747e5`).
- **Previous strength champion: v18** (rollout leaf, both colours)
  — h2h vs v7 0.775, ladder Elo 636.4 [240.3, 1032.5], first
  version ever to beat the ladder MCTS outright (14-6). Held the
  title from batch 7 (2026-11-15) until batch 13 (2026-11-22),
  when v52 beat it in the direct challenge. It never passed the
  full bar (tactical 25/29 on the original suite, 26/29 on the
  batch-9 corrected suite — the (9,7) addition). Lives on
  `research/batch-7` (`970ecf3`), worktree
  `~/workspace/projects/gomoku-b7/`, and on master via the
  batch-9 merge.
- The daily cross-track match pairing is now v52 vs v44 (swap
  rules, main worktree).
- New heuristic versions are measured against v52 for the bar and
  against v52 for strength (it holds both titles). A new strength
  champion is declared by DIRECT CHALLENGE: a version that passes
  the bar plays a 100-game match against the current champion
  (v52); it takes the title if it wins the match (win rate > 0.50
  with Wilson 95% CI lower bound > 0.50) AND its ladder Elo beats
  the champion's (by the leg-result reading — see above). A
  bar-passer already clears the Elo clause by passing, so the
  challenge match is the deciding test. (Rule changed 2026-11-17:
  the old rule compared h2h-vs-v7 rates, which no longer appear on
  a battery now that the bar reference is v27; the direct challenge
  measures the same thing without the stale reference. v18 keeps
  the title under both rules — v27 ties it on Elo.)

## Track B — ML model (learned nets)

- **Champion: v44** (3-ply-label distilled value net, ResNet
  16ch/2 blocks, α=0.125) — the first ML version ever to pass all
  4 clauses of the ML bar (ML batch 4, 2026-11-20): vs v8 0.67,
  CI [0.5731, 0.7544]; ladder Elo 98.1 [−12.1, 208.2] (40-game
  legs; the interval includes 0 — the pass is on the point
  estimate, per the pre-registered rule); tactical 20/20 + 29/29;
  vs v0 0.58. First ML version ever to beat the ladder MCTS
  outright (25–15). Lives on master (`6b777e3`); its checkpoint
  is gitignored data at `ml/runs/ml4-h3/value-net.pt` (present in
  the main worktree and the b7 worktree; copy it into any
  worktree that needs to run v44). Caveat: v37 remains the
  strongest ML version by head-to-head (beats v44 0.69–0.31) —
  non-transitivity is the norm in the ML track, so future
  versions are measured against v37 as well as the champion.
- **Previous champion: v8** (v6's PUCT MCTS + learned value leaf;
  ResNet 16ch/2 blocks trained on the B1 dataset — 4,350 recorded
  games, 1,449 unique (dedup finding, ML batch 1) — α=0.25) —
  ladder Elo 0.0, vs v0 0.545, vs v1 0.48, tactical 20/20 + 29/29.
  It held the title from batch 3 (2026-11-14) until v44 passed the
  bar (2026-11-20). Its checkpoint is gitignored data at
  `ml/runs/b3-h2/value-net.pt`.
- **ML track's own bar** (same 4-clause shape, reference v44 —
  re-anchored when v44 passed, 2026-11-20): (1) h2h vs v44 ≥100
  games, win rate > 0.50 with Wilson 95% CI lower bound > 0.50;
  (2) ladder Elo > 98.1; (3) tactical ≥ v44's (20/20 + 29/29);
  (4) all test suites green.
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
  dir>/YYYY-MM-DD.json`, run in the MAIN worktree
  (`~/workspace/projects/gomoku-ml-playground/`, master), which
  holds both champions' code — v18 came onto master via the
  batch-9 merge and v44 via the ML batch 4 merge. (The old b7
  worktree is stale: its registry stops at v20 and does not know
  v44.) The ML champion's checkpoint lives at its registry path
  in the main worktree (`ml/runs/ml4-h3/value-net.pt` for v44). Seeds are date-derived (YYYYMMDD) so each day's
  games and openings are fresh, never the validation battery's
  seeds (100/200) or the fast loop's (1001/2001).
- **Results:** `~/workspace/goals/gomoku-ml-playground/hidden_files/cross-track/`
  (JSON per day + this file's log below). The daily cron (`gomoku-cross-track-daily`,
  ~08:37 PST) runs the match and reports the result to Yudong in the
  Gomoku side chat.
- **Reading the number:** with the current gap (heuristic Elo 636.4
  vs ML Elo 98.1) the heuristic side should win the large majority.
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
  collapse (v28 13-7 vs v22's 5-15). ML batch 2 complete
  2026-11-17: 0/3; v30 closest (0.525 vs v8, a tie); dedup +
  colour-balance CLOSED as a strength lever; scale does not break
  the distillation ceiling (v31 0.40, worse than v8 despite better
  holdout); the VCF check is near-neutral on an already-tactically-
  perfect engine (v32 0.37). v8 stays ML champion. Batch 10
  (heuristic) complete 2026-11-18: 0/3; v33 closest (0.55 vs v27);
  v34 owns the first perfect heuristic leg ever (20-0, Elo
  estimate saturated — clause 2 recorded PASS on the leg-result
  reading: 20-0 strictly dominates v27's 19-0-1, and the plan
  pre-registered 20-0 as the condition for beating 636.4); v35
  ties v27's Elo exactly (636.4). The tie-break lever and
  VCF-prevention for the v27 family are now CLOSED. Two program
  findings: (1) a resume seed-replay bug in selfplay_batch.py —
  resumed segments replay the prefix (v34: 63 unique games in
  100), and engine determinism alone collapses vs-v0 legs to
  41–45 unique; no verdict turns on it (all clause-1 calls stand
  on the de-dup cut; v27's bar pass survives at 0.7766), but
  per-game seeding + uniqueness audit become standard battery
  hygiene from ML batch 3; (2) the ladder is saturated at the top
  (20-0 yields no finite Elo) — batch 11 needs a stronger anchor
  or more games per leg, a re-basing decision. ML batch 3 complete
  2026-11-18: 0/3, but v37 is the strongest ML version ever by
  head-to-head — the first ML version ever to beat v8 with a CI
  clear of 0.50 (0.75, CI [0.657, 0.8245]; de-dup 0.7407) — failing
  only clause 2 on a cold 20-game ladder leg (Elo −34.9; the
  20-game leg's ±~180 Elo noise decided it). The 2-ply label
  effect is real and causal (+0.235 vs v8 over v24 on identical
  positions, only the labels differing). v36 (threshold tie-break
  on the value engine) REJECTED 3/4 — net q-gaps are noise, the
  lever is CLOSED for the value engine. v38 (multi-task value
  net) REJECTED 2/4 — the threat head scores 0% on forcing
  positions; multi-task in this form is CLOSED. History-dependence
  is the top ML-track risk (v37/v38 nets collapse without history
  planes; v24's is robust). Per-game seeding verified in
  production (9+ VM replacements, zero games lost; vs-v0 legs
  now 95–100 unique). Batch 11 (heuristic) complete 2026-11-19:
  0/3; v39 closest (0.495 vs v27 — a coin flip); v40's handed
  hypothesis refuted (the leaf does NOT mis-rank — it ranks the
  unsafe move worst; the mis-ranking lives in backed-up root q /
  visit allocation); v41's White forensics found no patchable
  mechanism (the "White gap" is matchup-dependent). Setup-
  prevention in this form and the leaf-blind direction are now
  CLOSED; the White gap as a single-flaw framing is also closed.
  The 40-game ladder re-basing worked (no saturation, intervals
  narrowed; v39/v40 posted identical legs). Two defects recorded:
  v35's _leaves_opp_vcf has a hole (VCF starts are NOT monotone
  under our move) and exact leaf values break the tie-break
  (v40's v1bl-g2 regression). ML batch 4 complete 2026-11-20:
  1 of 3 — **v44 PASSES all 4 clauses and is the NEW ML
  CHAMPION** (vs v8 0.67, CI clear of 0.50; Elo 98.1; tactical
  29/29; first ML version ever to beat the ladder MCTS outright,
  25–15). Merged to master (6b777e3). Caveat: v44's Elo interval
  [−12.1, 208.2] includes 0 — the pass is on the point estimate,
  per the pre-registered rule (the same rule that failed v37 at
  −34.9); no confirmatory re-run (that would be seed-shopping in
  reverse). v37 remains the strongest ML version by head-to-head
  (beats v44 0.69–0.31). v42 (history dropout on v24's recipe)
  REJECTED 2/4 — the gate passed decisively but the games didn't
  improve; history-plane dropout on v24's recipe is CLOSED. v43
  (history dropout on the v37 net) REJECTED 2/4 — 0.505 vs v8
  (v37's 0.75 became 0.505); the 2-ply labels NEED the history
  planes; history dropout in this form is CLOSED. Holdout
  robustness does not predict played strength — now four
  instances. Opening variety worked (vs-v8 legs now 91–97 unique
  vs 21 for v36). White collapse vs heuristic opponents is now
  the top ML-track problem. Batch 12 (heuristic) is next: v35's
  _leaves_opp_vcf hole (fix-and-measure) and a White-side
  statistics framing. Batch 12 complete 2026-11-21: 0/3; v45
  (fixed VCF check) closest at 0.545 vs v27 — the hole fix found
  0 hole instances in 6,258 turns, so it changed nothing in
  games; its Elo 759.1 (highest in the program) is a draw-count
  artefact (39-0-1 vs 38-0-2), not a fix effect. v46's White
  statistics found no patchable asymmetry (population-level
  confirmation of v41); its fallback (leaf-level top-K min)
  closed a second blunder-dilution variant. v47's minimax backup
  was catastrophic (0.245, tactical 21/29) — q-collapse destroys
  the gradient the mean backup preserved; pure minimax backup is
  CLOSED. Both blunder-dilution fixes are now closed; the mean
  backup's averaging is load-bearing. The White gap is doubly
  confirmed non-patchable in its current framings. ML batch 5
  complete 2026-11-21: 0/3; v44 stays the ML champion. v48's
  White-collapse data patch fixed the played White gap but Black
  fell by at least as much — a redistribution, not an
  improvement. v49's 4-ply labels reverse the 3-ply parity shift
  on every measure but play no better than 2-ply — the label-depth
  lever is now fully mapped at both parities and CLOSED (1-ply
  0.515/52.5, 2-ply 0.75/−34.9, 3-ply 0.67/98.1, 4-ply
  0.475/−137.0). v50's α ablation (v37's net at α=0.125) plays at
  v37's level on every measure — v44's Elo gain came from its
  3-ply labels, not the lower α; the α lever is CLOSED for the
  2-ply net. Holdout does not predict played strength — now six
  instances. Batch 13 complete 2026-11-22: 1 of 3 — **v52
  (threat-space search at the root) PASSES all 4 clauses and is
  the NEW heuristic bar reference AND strength champion** (takes
  the title from v18, which held it since batch 7). v52 vs v27
  0.61 (CI lower 0.5120; de-dup CI lower 0.4995, misses by 0.0005
  — the pass rests on the pre-registered nominal rule, v44
  precedent); ladder Elo unbounded — saturated, the first version
  ever to saturate the 40-game ladder (heuristic leg 40-0-0);
  vs v0 100-0 (second perfect 100-game score); swap vs v27 0.655;
  challenge vs v18 0.625, passes on BOTH cuts. v51 (soft-min
  backup) rejected 1/4 — child-level backup aggregation is now
  CLOSED in both forms (v47 pure minimax, v51 soft-min). v53
  (3-ply rollout leaf, re-scoped — the plan's "2-ply is untested"
  premise was FALSE; v18/v27's leaf is already 2-ply) rejected
  3/4 — beats v27 h2h 0.64 but Elo 511.5 < 636.4 (the clause-2
  paradox: the two losses to v0 as Black in the cold 40-game
  ladder leg cost ~125 Elo). The rollout-depth ladder is fully
  priced (1 rejected, 2 = v18/v27, 3 beats v27 h2h but fails the
  bar, 4 rejected twice). The ladder is saturated at the top —
  terminally: clause 2 for the v52 reference is judged by the
  leg-result reading (dominate 40-0/40-0/31-9). ML batch 6
  complete 2026-11-23: 0/3; v44 stays the ML champion. v54's
  White-only patch is a v0-specialist (White 0.71 vs v0, but
  0.29 vs v7, 0.25 vs the ladder MCTS, 0.21 vs v37) — the White
  fix works vs the weakest opponents only and makes White worse
  vs every stronger opponent; its Elo 168.4 is the highest
  ML-track Elo ever, yet its h2h vs v44 is the worst (0.385).
  v55's 3-ply retrain of the v48 patch data is a partial recovery
  (net-vs-net strength recovered, but the White fix did not
  survive odd-parity 3-ply labelling). v56's symmetric parity
  correction removed 70% of the gradient gap but did not centre
  the labels (sym mean −0.1292) — optimism traded for pessimism;
  the symmetric-label correction is CLOSED. Holdout does not
  predict played strength — now eight instances. **The ladder
  Elo and played strength now openly diverge:** v54 has the best
  Elo (168.4) and the worst h2h (0.385); both v54 and v56 pass
  Elo while losing the MCTS leg outright (16-23-1, 14-25-1) —
  the Elo is computed from the heuristic anchor leg only and is
  blind to the MCTS leg. ML batch 7 should pre-register whether
  the MCTS leg enters clause 2. Tooling: eval_value_holdout.py's
  side-to-move inference is fixed (37.9% of positions were
  mislabelled on distill shards). Batch 14 is next: TSS
  depth/budget sweep, v52's swap/colour interaction, TSS
  speed.
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
| 2026-11-17 | v18 | v8 | 34–6–0 (40 games) | 0.850 [0.709, 0.929] | Daily #2 — FIRST SWAP-RULES daily (new baseline for the swap series; not comparable to 2026-11-16's normal-rules 0.575): v18 as Black 19–1, as White 15–5; second player took Black 30/40, White 10/40; Black won 24/40; ML win rate 0.15. Under swap the colour edge is neutralised, so 0.85 is a cleaner strength signal — the gap is wider than the normal-rules number suggested. CI excludes 0.50: first daily where the heuristic edge is statistically clear. |
| 2026-11-18 | v18 | v8 | 34–5–1 (40 games) | 0.8625 [0.724, 0.938] | Daily #3 — swap rules: v18 as Black 18–1, as White 16–4–1; second player took Black 33/40, White 7/40; Black won 22/40, White 17, 1 draw. ML win rate 0.125 (5/40) vs 0.15 yesterday — no movement toward 0.50; CI still excludes 0.50. ~107 min (ran alongside ML batch 3). |
| 2026-11-19 | v18 | v8 | 31–9–0 (40 games) | 0.7750 [0.625, 0.877] | Daily #4 — swap rules: v18 as Black 16–2, as White 15–7; second player took Black 30/40, White 10/40; Black won 23/40, White 17, 0 draws. ML win rate 0.225 (9/40) vs 0.125 yesterday — up, but CIs overlap ([0.625, 0.877] vs [0.724, 0.938]), so within noise; no clear movement toward 0.50 yet. 40/40 unique games. ~36 min (ran alongside batch 11 v39/v40 batteries). |
| 2026-11-20 | v18 | v44 | 81–15–4 (100 games) | 0.8300 [0.745, 0.891] | FULL MATCH for the ML champion change (v44 replaces v8) — swap rules, offset seeds (independent of the daily match's date seeds): v18 as Black 39–8–3, as White 42–7–1; second player took Black 82/100, White 18/100; Black won 46/100, White 50, 4 draws. ML win rate 0.17 — essentially unchanged from v8's daily average (~0.17 across #2–#4), so v44's Elo gain (98.1 vs 0.0) does not show up against v18 yet. 100/100 unique games. ~48 min, run in the main worktree (b7 is stale). |
| 2026-11-22 | v52 | v44 | 82–16–2 (100 games) | 0.8300 [0.745, 0.891] | FULL MATCH for the heuristic champion change (v52 replaces v18) — swap rules, offset seeds (independent of the daily match's date seeds): v52 as Black 40–8–2, as White 42–8–0; second player took Black 82/100, White 18/100; Black won 48/100, White 50, 2 draws. ML win rate 0.17 — the SAME win rate and CI as v18's full match (0.83, [0.745, 0.891]), so v52's improvement over v18 (challenge 0.625) does not show up against v44 yet — v44 is so much weaker than both that the gap between them is invisible at this level. 100/100 unique games. ~5h05m total wall time (06:58–12:03 PST), interrupted 2× by VM replacements (at 68 and 83 games); per-game seeding auto-resume reproduced the run exactly. Run in the main worktree. |
| 2026-11-20 | v18 | v44 | 23–14–3 (40 games) | 0.6125 [0.458, 0.747] | Daily #5 — swap rules, FIRST daily vs the new ML champion v44 (date seeds, independent of the morning's 100-game full match): v18 as Black 12–4–1 (17 games), as White 11–10–2 (23 games); second player took Black 29/40, White 11/40; Black won 22/40, White 15, 3 draws. ML win rate 0.35 (14/40) vs 0.225 yesterday (vs v8) — up, but CIs overlap ([0.458, 0.747] vs [0.625, 0.877]) and the same-day 100-game full match gave ML 0.17, so no clear movement toward 0.50 yet; CI now includes 0.50 — first swap-series daily where the heuristic edge is not statistically clear. 40/40 unique games. Run interrupted 3× by VM replacements (SIGTERM at 12, 34, 36 games); per-game seeding auto-resume reproduced the uninterrupted run exactly; final segment 456s once batch 12 had stopped. Total wall time ~3h40m (08:37–12:17 PST). |
| 2026-11-21 | v18 | v44 | 34–4–2 (40 games) | 0.8750 [0.7389, 0.9454] | Daily #6 — swap rules: v18 as Black 21–2 (23 games), as White 13–2–2 (17 games); second player took Black 29/40, White 11/40; Black won 23/40, White 15, 2 draws. ML win rate 0.10 (4/40) vs 0.35 yesterday — back down; CIs barely overlap ([0.7389, 0.9454] vs [0.458, 0.747]) and the 2026-11-20 100-game full match gave ML 0.17, so daily #5's 0.35 looks like the outlier; no movement toward 0.50. 40/40 unique games. Run interrupted 2× by VM replacements (at 3 and 37 games); per-game seeding auto-resume reproduced the run exactly. Ran alongside ML batch 5 batteries (v48/v49 vs v0, then v49/v50 vs v44), which slowed it; two games (5, 25) each took ~25 min. Total wall time ~2h18m (08:37–10:55 PST). |
| 2026-11-22 | v52 | v44 | 29–9–2 (40 games) | 0.7500 [0.5981, 0.8581] | Daily #7 — swap rules, FIRST daily vs the new heuristic champion v52 (takes over from v18): v52 as Black 14–1–1 (16 games), as White 15–8–1 (24 games); second player took Black 28/40, White 12/40; Black won 22/40, White 16, 2 draws. ML win rate 0.225 (9/40) vs 0.10 yesterday (vs v18) — up, but the pairing changed (v52, not v18) and the CIs overlap ([0.5981, 0.8581] vs [0.7389, 0.9454]), so within noise; no clear movement toward 0.50. The same-day 100-game full match gave v52 82–16–2 (ML 0.17), the more reliable read — consistent with the daily 0.75. 40/40 unique games. Run interrupted 2× by VM replacements (at 14 and 29 games); per-game seeding auto-resume reproduced the run exactly both times. Two games (14, 34) each took ~27–30 min (v52's TSS in creator-rich positions). Total wall time ~3h28m (08:37–12:05 PST). |
