# Iteration 3 — v3: 2-ply threat check (prevention)

**Status: v3 REJECTED** — lost head-to-head to v1 (0.47) despite a
higher ladder Elo (107.5 vs 70.4). The threat check helps against
weaker opponents but costs against an equal one. See the bar check
in §4.

## 1. Loss analysis for iteration 3

Iteration 3 re-analysed the iteration-1 swap batches (v1-vs-v0,
100 games; v1-vs-v1, 100 games) with a new diagnostic: the
**prevention analysis**. Earlier diagnostics had established that
v1 never misses a block (0 missed blocks over 118 single threats)
and never ignores a critical threat (0 ignored over 432 faced) —
so the losses had to come from something 1-ply scoring cannot see.
The new question: does v1 leave the opponent with an open-four
creator after its move when a different move would have removed
all creators (or won / made a four)?

Key insight for the analysis: a move can only REMOVE opponent
creators (it adds one of our stones, blocking their lines), never
create a new one — so the creators after a move are a subset of
the creators before it, and only the pre-move set needs
re-checking.

### The filter matters (a correction)

The first prevention count (60 missed preventions in 35 losses)
was WRONG: it counted turns where the opponent ALREADY held an
immediate winning move. In those turns the block scan decides the
move (single win: must block) or the game is already lost (two
wins: unblockable), so a "prevention" there is moot. The correct
count excludes those turns:

| | v1's losses (35 games) | Baseline's losses (62 games) |
|---|---|---|
| Threatened turns (opponent holds a creator, no immediate win) | 159 | 268 |
| Non-forcing moves leaving >=1 creator live | 26 | 42 |
| **Clean missed preventions** (a preventing alternative existed) | **13** | **23** |
| Games with >=1 missed prevention | 9 | 20 |
| Miss rate on threatened turns | 8.2% | 8.6% |

**Reading:** the prevention gap is real (13 clean misses in 35
losses) but SMALL, and v1 misses at the same rate as the baseline
(8.2% vs 8.6%) — this is a shared 1-ply blind spot, not a
v1-specific weakness. v3's expected gain is therefore modest: at
most a few games per 100. The head-to-head will decide whether it
is real.

### A concrete case (game 13, ply 22, Black to move)

White holds four open-four creators. v1 plays (5, 4): it blocks
one creator AND creates a double open three — 1-ply score
258,002 — but leaves three creators live, and White makes an
open four next move and wins the race. The preventing move is
(10, 7), a forcing four (1-ply score 20,097): White must block
the four instead of playing a creator. The 1-ply scorer ranks
the double open three (58k) above the forcing four (20k); only
a 2-ply check sees that the four is forcing and the double-three
is not. This position is the regression test
(`test_v3_prevents_where_v1_does_not`).

### Hypotheses NOT pursued (from iter-2.md §6)

- Raising the double-open-three bonus: changes nothing — the
  58k–200k band is nearly empty and creator blocks at 200k still
  dominate.
- Colour-conditional defence: the v2 data shows the Black gain is
  noise-sized and the White loss is large.
- Candidate expansion (Chebyshev-2 -> all cells): no evidence any
  missed move was outside the candidate set — all 13 missed
  preventions had their alternative INSIDE the candidate set.
- Centre-first opening: narrow fix, does not explain losses at
  mean 27.5 plies.

## 2. The improvement: v3 = v1 + 2-ply threat check

One structural change to `choose_move` in
`ml/gomoku_ml/scoring.py` (registry `ENGINE_VERSIONS[3]`,
`threat_check: True`; defence weight stays 1.0):

1. Win scan and block scan unchanged (immediate wins and blocks
   still take priority).
2. Score all candidates in 1-ply as before.
3. **New:** if the opponent has an open-four creator (a move
   leaving them >=2 winning moves), check whether the 1-ply best
   move is *safe* — safe = forcing (wins or makes a four) or
   leaves the opponent with zero live creators. If the 1-ply best
   is safe, take it. Else take the best-scoring safe move. If no
   safe move exists, fall back to the 1-ply best.
4. With no opponent creator, v3 chooses exactly like v1
   (tested: `test_v3_same_as_v1_without_threat`).

Helpers: `_winning_moves` (local, same semantics as
`evaluate.winning_moves`, to avoid the evaluate dependency),
`_opponent_creators`, `_is_forcing`, `_move_is_safe`.

**Cost:** the threat check runs the creator scan (one
`winning_moves` per candidate) plus safety checks when creators
exist. Measured on a random 24-stone board: v3 `choose_move`
68.1 ms/call vs v1 4.6 ms/call (~15x slower, worst case — the
best move is usually safe, so the full safe-move scan rarely
runs). Self-play batches take minutes instead of seconds; the
ladder is unaffected in practice (MCTS games dominate).

## 3. v3 evaluation

### Head-to-head (normal rules, 100 games each)

| Match | Result (v3) | Win rate | 95% CI |
|---|---|---|---|
| v3 vs v1 | 27W-33L-40D | 0.4700 | [0.3751, 0.5671] |
| v3 vs v0 | 58W-35L-7D | 0.6150 | [0.5171, 0.7044] |

v3 beats the baseline MORE convincingly than v1 does (v1 vs v0
pooled: 0.55, CI [0.4808, 0.6174]) — as Black 0.42 (v1: 0.345),
as White 0.81 (v1: 0.755). But v3 loses to v1 head-to-head.
Reading: v3's safety prevents the baseline's creators (the
baseline creates preventable threats), but against v1 — which
rarely leaves a preventable creator and punishes passivity — the
safety preference gives up attacking chances. The bar is against
the previous version, so the v0 result does not save v3; it is
recorded here because it shows the threat check is not useless,
just not a net win against a stronger opponent.

Colour split vs v1: as Black 15W-15L-20D (0.50), as White
12W-18L-20D (0.44) — no colour asymmetry; v3 is just slightly
worse overall, within noise. Draw rate 40% (vs 19% for v2-vs-v1):
v3's safety preference prevents losses but also passes up
winning chances — the safe move is often a passive block instead
of a sharp attack. Win/loss plies are symmetric (43.0 / 42.7),
unlike v1's fast-loss / slow-win pattern.

### Swap match (100 games, balanced)

v3 vs v1: 26W-34L-40D, 0.4600, CI [0.3656, 0.5574] — loses
under swap rules too (same pattern as normal rules: more draws,
fewer wins).

### Ladder (20 games/opponent, mcts_sims=100)

| Engine | vs random | vs heuristic | vs mcts | Ladder Elo | 95% interval |
|---|---|---|---|---|---|
| v3 | 20W-0L-0D | 12W-6L-2D (0.650) | 9W-10L-1D (0.475) | **107.5** | [-46.9, 262.0] |
| v1 (same counts) | 20W-0L-0D | 11W-7L-2D (0.600) | 9W-10L-1D (0.475) | **70.4** | [-80.2, 221.1] |
| v2 (same counts) | 20W-0L-0D | 6W-9L-5D (0.425) | 7W-12L-1D (0.375) | **-52.5** | [-201.9, 96.9] |

v3's ladder Elo is HIGHER than v1's (+37.1), driven by the
heuristic matchup (0.65 vs 0.60; the mcts matchup is identical
at 0.475). But the intervals overlap heavily, so the difference
is within noise — and the ladder measures strength against
WEAKER opponents (random/heuristic/MCTS-100), where the threat
check helps. The head-to-head against v1 itself is the sharper
test, and v3 loses it.

### Tactical suite

v3: 20/20, same as v1 and v2. The suite does not discriminate
between these 1-ply engines (third time).

## 4. Bar check (v3 vs v1) — REJECTED

1. Head-to-head (100 games): 0.4700, CI [0.3751, 0.5671].
   Need > 0.50 with CI lower bound > 0.50. **FAIL.**
2. Ladder Elo (identical 20 games/opponent): v3 107.5 vs
   v1 70.4. Need strictly higher. **PASS** (but intervals
   overlap: [-46.9, 262.0] vs [-80.2, 221.1]).
3. Tactical: 20/20 = v1's 20/20. Pass (equal).
4. Tests green. **PASS** — full suite re-run at end of batch:
   ml pytest 144 passed, npm test 55/55, `npm run build` clean.

**v3 is REJECTED** — clause 1 fails, and the bar requires ALL
clauses. Not committed to master, not pushed to master. The
2-ply threat check is a real fix for a real blind spot (13
clean missed preventions in 35 losses), but its net effect
against an equal opponent is negative: the safe move is often
a passive block that gives up the attack, and v1 exploits the
passivity. The check helps against weaker opponents (ladder
Elo up, v0 head-to-head up) and hurts against equal ones.

## 5. What this means

Batch 1 ends with **0 of 3 versions passing the bar** (v1, v2,
v3 all rejected; master stays at 78c733a). The pattern across
all three iterations:

- The 1-ply engine's defence is already airtight (0 missed
  blocks, 0 ignored critical threats). Scoring tweaks (defence
  discount, threat check) move the needle by less than the
  noise floor of a 100-game match (95% CI width ~±0.10).
- Every decisive game is decided by the open-four race, and
  the race is won by whoever creates the first open four — a
  1-ply scorer can value open fours but cannot PLAN a sequence
  to create one. The next qualitative step is search (2-ply
  minimax over the top candidate moves, or MCTS with the
  heuristic as prior — the project already has MCTS), not more
  scoring-table tuning.
- The tactical suite (20 positions) does not discriminate
  between 1-ply engines — all three versions score 20/20. A
  harder suite (positions requiring 2-ply planning, e.g.
  double-threat creation under a counter-threat) is needed
  before it can gate versions.
- The swap-rule head-to-head is the fairer comparison (it
  removes the opening-phase confound) and should be adopted
  for the bar in a future batch — decided before that batch's
  first eval (see `research/README.md`).
- The kernel still ports v0 (correct — v0 remains canonical
  since no version passed). Re-sync the kernel only when a
  version passes the bar.
