# Iteration 2 — v2: attack bias (defence discount 0.9)

**Status: v2 REJECTED** — lost head-to-head to v1 and lost ladder Elo.
The defence-discount hypothesis is refuted: v1's full-value defence
is a strength, not a weakness. See the bar check at the end.

## 1. Self-play batches (fixed swap rules, balanced placement)

Two 100-game batches under the CORRECTED swap rules (White to move
after the 3-stone opening; see `swap-rule.md`) with balanced placement
(3 stones in the 5x5 box around the centre, black2 minimising the
attack-score gap). Ran 2026-11-13 on the Linux VM (CPU-only).

| Batch | Result (A = v1) | Win rate | 95% CI | Time |
|---|---|---|---|---|
| v1 vs v1, swap, balanced | 30W-37L-33D | 0.4650 | [0.3704, 0.5622] | 42.5s |
| v1 vs v0 (baseline), swap, balanced | 62W-35L-3D | 0.6350 | [0.5373, 0.7227] | 33.3s |

Records: `batch-v1-vs-v1-swap.json`, `batch-v1-vs-v0-swap.json`
(seed 11 placement, seeds 100/200 players).

### Swap-rule validation (fixed rules)

- **Mirror match is fair:** v1-vs-v1 is 30-37-33 with the CI
  including 0.50, and winners by colour are balanced (Black 32,
  White 35, draws 33). The swap setup no longer systematically
  favours one side (contrast the pre-fix 20/20 Black choices with
  10/10 opener losses).
- **v1 beats the baseline clearly under swap rules:** 0.635, CI
  excludes 0.50. Under normal rules the pooled head-to-head was
  0.55 with CI [0.4808, 0.6174] (includes 0.50). The swap rule
  removes the empty-board opening phase, where v1-as-Black is
  weakest, and the colour confound that masked v1's edge.
  (Note: the pre-registered "better" bar uses normal-rules
  head-to-head, so this does not change v1's rejected verdict —
  but it suggests adopting swap-rule head-to-head for the bar in
  a future batch, since the swap rule exists precisely to remove
  the first-player confound.)
- **The swap choice is still systematically wrong:** the second
  player took Black in 100/100 games in both batches (opening gap
  77-78 in every game, always favouring Black), yet White won
  MORE games overall in v1-vs-v0 (White 56, Black 41, draws 3).
  The attack-gap judge favours Black because Black has 2 stones
  to White's 1, but it ignores tempo: White moves next (move 4),
  and the tempo outweighs the 77-point attack edge. The colour
  choice needs a tempo-aware judge (e.g. compare White's best
  attack against Black's best attack AFTER White's reply) —
  next-batch work. The match comparisons remain fair because the
  opener alternates, so each engine is Black in 50% of games.
- **v1's colour asymmetry persists but is milder under swap
  rules:** v1 as Black 27W-21L (0.5625 excl. draws), as White
  35W-14L (0.714), vs normal rules 0.345 / 0.755. The swap
  opening removes v1's weakest phase (the empty-board opening)
  but not all of the asymmetry.
- **Draw rates:** v1-vs-v1 33% (mean 101.6 plies, max 225 = full
  board — both defend well and grind); v1-vs-v0 3% (mean 59.5
  plies).

## 2. Loss analysis (v1's 35 swap losses vs the baseline)

Replayed all 35 losses and 62 wins with a threat analysis
(`winning_moves` before each move):

- **Every decisive game was decided by an open four:** in 35/35
  losses the winner held >=2 winning moves at the start of a v1
  turn (mean losing-moment ply 25.5); in 62/62 wins v1 held the
  open four. The game at this level is a race to the first open
  four.
- **v1 never misses a block:** 0 missed blocks over 66 single
  threats faced in losses (and 52 in v1-vs-v1 losses). The
  win/block scans work perfectly.
- **v1 never ignores a critical threat:** in 72 losses (both
  batches) v1 faced 432 critical threats (opponent holding an
  open-four creator) and answered every one (win, four, or block).
  The baseline ignored 7 in 62 games — that is how v1 won them.
- **v1 loses fast, wins slow:** v1's losses averaged 27.5 plies;
  its wins averaged 69.5 plies. The opponent wins the open-four
  race early; v1 wins by outlasting, not by winning the race.
- **Double-threat analysis is inconclusive:** the winner held
  >=2 open-four creators at the start of a loser turn in 35/35
  losses AND 62/62 wins — the condition is universal, not
  discriminating. (Mean double-threat ply 11.6 in losses, with
  the game continuing a mean 15.9 plies after — the position
  becomes sharp early and the race continues through forcing
  moves.)

**Hypothesis (from iter-1, tested here):** v1's full-value
defence makes it a strong defender but a passive attacker — it
loses the race to the first open four. If so, discounting
defence (attack bias) should speed up its open-four creation
and improve results. The baseline, which uses a 0.9 defence
discount, wins the race against v1-as-Black.

## 3. The improvement: v2 = v1 + defence discount 0.9

One parameter change in the registry
(`ml/gomoku_ml/scoring.py`, `ENGINE_VERSIONS[2]`):
`defence_weight 1.0 -> 0.9`, mirroring the frozen baseline's
`int(0.9*w)`. Per-direction `int(w*d)`, same mechanism as the
baseline. Two new tests (`test_v2_defence_discount`,
`test_get_player_versions`); 52 tests pass (scoring + research
+ swap + kernel).

Note from testing: there is no purely-attacking move — every
move carries a small defence value (1-2 per direction) from
empty windows, so the discount also shaves a few points off
attacking moves (at most 8).

## 4. v2 evaluation

### Head-to-head (normal rules, 100 games each)

| Match | Result (v2) | Win rate | 95% CI |
|---|---|---|---|
| v2 vs v1 | 31W-50L-19D | 0.4050 | [0.3140, 0.5030] |
| v2 vs v0 | 32W-44L-24D | 0.4400 | [0.3467, 0.5377] |

Colour split vs v1: as Black 18W-26L-6D (0.42), as White
13W-24L-13D (0.39) — v2 loses as BOTH colours. Colour split
vs v0: as Black 11W-23L-16D (0.38), as White 21W-21L-8D
(0.50). Compare v1 vs v0: as Black 0.345, as White 0.755.
The discount bought a small (likely noise) Black gain and a
large White loss. A colour-conditional bias is therefore NOT
supported by this data.

### Swap match (100 games, balanced)

v2 vs v1: 24W-36L-40D, 0.4400, CI [0.3467, 0.5377] — loses
under swap rules too.

### Ladder (20 games/opponent, mcts_sims=100)

| Engine | vs random | vs heuristic | vs mcts | Ladder Elo | 95% interval |
|---|---|---|---|---|---|
| v2 | 20W-0L-0D | 6W-9L-5D (0.425) | 7W-12L-1D (0.375) | **-52.5** | [-201.9, 96.9] |
| v1 (re-measured, same counts) | 20W-0L-0D | 11W-7L-2D (0.600) | 9W-10L-1D (0.475) | **70.4** | [-80.2, 221.1] |

v1's re-measured ladder confirms the colour asymmetry: vs
heuristic as Black 3W-7L-0D (0.30), as White 8W-0L-2D (0.90).
v2's ladder Elo is 122.9 points BELOW v1's.

### Tactical suite

v2: 20/20, same as v1. The suite does not discriminate between
these 1-ply engines (again).

### v2 loss pattern (unchanged)

v2 vs v0: losses mean 26.5 plies, wins mean 88.0 plies — the
same fast-loss / slow-win pattern as v1. The discount did not
change the pattern; it just made v2 lose more often.

## 5. Bar check (v2 vs v1) — REJECTED

1. Head-to-head (100 games): 0.4050, CI [0.3140, 0.5030].
   Need > 0.50 with CI lower bound > 0.50. **FAIL.**
2. Ladder Elo (identical 20 games/opponent): v2 -52.5 vs
   v1 70.4. Need strictly higher. **FAIL.**
3. Tactical: 20/20 = v1's 20/20. Pass (equal).
4. Tests green. Pass.

**v2 is REJECTED. Not committed, not pushed.** The defence
discount hypothesis is refuted: v1's full-value defence is a
strength, and discounting it costs ~123 ladder Elo points and
loses the head-to-head outright.

## 6. What this means for iteration 3

v1's defence is already airtight (0 missed blocks, 0 ignored
critical threats over 432 faced). The problem is not threat
response — it is that v1 loses the open-four race despite
perfect defence. The remaining hypotheses to test:

- **Missed prevention (2-ply gap) — CONFIRMED but small:** on 13
  turns in 9 of v1's 35 swap losses, v1's non-forcing move left an
  opponent open-four creator live when a preventing alternative
  existed (8.2% of the 159 threatened turns where the opponent had
  no immediate winning move). The baseline misses at the same rate
  (23/268 = 8.6%), so this is a shared 1-ply blind spot, not a
  v1-specific weakness — v3's expected gain is modest. (An earlier
  unfiltered count of 60 was inflated by already-decided positions
  where the opponent already held a winning move; the filtered
  count is the correct one.) v3 = v1 + a 2-ply threat check in
  `choose_move` (prefer moves leaving the opponent with zero live
  open-four creators, unless the move wins or makes a four).
- **Opening play:** on an empty board all moves score equally,
  so v1's first move is effectively random (seeded RNG) —
  including corners and edges. A centre-first opening is a
  narrow fix for the Black weakness; it does not explain losses
  at mean 27.5 plies.
- **Candidate restriction:** v1 scores only Chebyshev-2
  candidates. Distant double-threat setups may be missed.
  Expanding to all cells is a clean test (slower, but correct).
- **Not worth testing:** raising the double-open-three bonus
  (58k -> 150k changes nothing — the 58k-200k band is nearly
  empty, and creator blocks at 200k still dominate); a
  colour-conditional defence discount (the v2 data shows the
  Black gain is noise-sized and the White loss is large).
