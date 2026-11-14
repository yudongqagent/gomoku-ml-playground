# Swap colour judge v2 — tempo-aware (batch 2, H3)

**Date:** 2026-11-13. **Status:** implemented on branch `research/batch-2`;
decision on the harness default below. Code: `ml/gomoku_ml/game.py`
(`choose_swap_color_tempo`, `choose_swap_color(..., judge=...)`),
`ml/gomoku_ml/research.py` (`play_swap_game` / `play_match` accept
`judge=`). Tests: `ml/tests/test_swap_rule.py` (9 new tests).

## The problem (from batch 1)

The batch-1 judge (`judge="attack"`) compares each colour's best
attack score on the 3-stone opening and takes the higher, ties to
White. It picked **Black in 100/100 games** in both batch-1 swap
batches — yet White won MORE games overall (White 56, Black 41 in
v1-vs-v0). The judge ignores tempo: White moves next (move 4), and
the tempo outweighs Black's ~77-point attack edge. See
`research/iterations/swap-rule.md` (batch-1 + iteration-2 findings).

## The judge (definition)

`game.choose_swap_color_tempo(board, score_move_fn)`:

1. For each colour `c`, compute its **setup value**: among `c`'s best
   immediate attack moves on the opening board (max of the injected
   attack scorer, ties included), take the one with the best
   **follow-up** — `c`'s best attack on the board after that setup
   move. The value is that follow-up score. The max-over-ties makes
   the value independent of which tied setup move is picked (the
   follow-up varies by at most 1 point across tied setups, measured
   on 5 openings).
2. Take the colour with the higher setup value; **ties go to White**
   (White moves next — the tempo breaks the tie).

In words: the colours are compared after **symmetric 1-ply
development** — each colour's value is measured after its own best
setup move, as in the real game where White plays move 4 and Black
move 5 before the race continues — and White's tempo breaks ties.
The judge keeps the attack-only scorer (`scoring.score_attack`); a
symmetric attack+defence scorer would still be vacuous here (see
`swap-rule.md`).

The old behaviour stays available: `choose_swap_color(board, scorer)`
defaults to `judge="attack"`; pass `judge="tempo"` for the new judge.
The harness threads it through: `research.play_swap_game(...,
judge="tempo")` and `research.play_match(..., judge="tempo")`.

## Why this design — the alternatives that failed

The pre-registered H3 plan suggested the naive 1-ply lookahead
(White's best move 4, then compare best attacks) or a tempo handicap.
Both were prototyped before implementing, and both are degenerate.
The root cause is the scorer's tier structure: a gapped three-in-a-
window (3 stones in a 5-window, gaps allowed) scores ~4006, while a
two-in-a-window scores 86–164. Any **sequential** lookahead is
dominated by whichever colour gets the last unanswered setup move:

| Candidate | Design | Result (balanced openings) |
|---|---|---|
| A — naive 1-ply (pre-registered) | White's best move 4, compare best attacks after it | **White 50/50** (seed 11); after-margin mean −3842. White's setup unlocks a 4006 follow-up; Black's non-interacting stones cap at 164. Flips the degeneracy instead of fixing it. |
| B2 — symmetric 2-ply max | value(c) = max over ALL setup moves of the follow-up | **Black 20/20**; margin exactly 77–78 in every game — reproduces the attack judge. Unrestricted maximisation washes out the positional differences. |
| C — 2-ply minimax | White's best move 4, Black replies minimising White's follow-up, compare best attacks | **Black 20/20** by mean +3912. Black's reply blocks White's setup AND sets up its own 4006. Same last-move-wins artifact, opposite direction. |
| **B1-max — symmetric setup (implemented)** | value(c) = best follow-up after c's own best-immediate move (max over ties) | **Black 76 / White 24** (seed 11, n=100); margins exactly 0 or 78. Only the symmetric comparison mixes the split. |

A fixed tempo handicap on the plain attack gap cannot mix the split
either: the gap is 77–78 in **every** balanced opening (see the
200-opening histogram below), so any handicap picks one colour
always. A mixed split needs a judge that looks deeper than the
immediate gap — B1-max is the minimal such deepening.

Note on what the tempo judge can and cannot do: the judge decides
WHO plays which colour, not the openings. With alternating openers
each engine plays each colour 50 times under either judge, so the
judge can only shift winners-by-colour through the pairing of
(engine, colour, opening) — it cannot by itself balance a
positional White edge that lives in the openings. Fully balancing
winners-by-colour needs the PLACER to balance the tempo-aware
values (the current placer minimises the attack gap, which leaves
White favoured by tempo). That is placer work, out of H3's scope;
it is the natural follow-up if the match below still starts
White-favouring.

## Before / after choice split (same 200 openings)

200 balanced openings, placement seed 23 (fresh seed — the design
was selected on seed 11, so this is out-of-sample), both judges on
each opening. Raw data:
`research/iterations/swap-judge-split-200.json`.

| Judge | Black | White |
|---|---|---|
| attack (before) | **200** | **0** |
| tempo (after) | **157** | **43** |

- Attack gap histogram (Black − White): `{77: 33, 78: 167}` — the gap
  is 77–78 in every opening, always favouring Black.
- Tempo margin histogram (Black value − White value): `{0: 43,
  78: 157}` — exactly bimodal. In 43 openings (21.5%) the setup
  values tie at 4006 vs 4006 and White wins on the tempo tie-break;
  in 157 the Black value is one tier-step (78) higher (4084 vs
  4006). No near-ties, no negative margins: Black's setup value is
  never below White's.

The split moves off 100/100 Black: the first clause of the
pre-registered success metric is met.

## Swap matches (100 games each, v1 vs v0)

Setup (identical for both judges, matching batch 1's recorded match):
v1 = `scoring.get_player(1, seed=100)`, v0 =
`baseline.HeuristicPlayer(seed=200)`, swap rule, balanced placement,
`placement_seed=11`, alternating opener. Records:
`batch-v1-vs-v0-swap-attack-rerun.json` and
`batch-v1-vs-v0-swap-tempo.json` in this directory.

| | Attack judge (before) | Tempo judge (after) |
|---|---|---|
| Result (v1) | 62W-35L-3D | 60W-33L-7D |
| Win rate (v1) | 0.6350 | 0.6350 |
| Wilson 95% CI | [0.5373, 0.7227] | [0.5373, 0.7227] |
| v1 as Black | 27W-21L-2D (50 games) | 29W-19L-2D (50 games) |
| v1 as White | 35W-14L-1D (50 games) | 31W-14L-5D (50 games) |
| `swap_choices` | Black 100, White 0 | Black 76, White 24 |
| Winners by colour | Black 41, White 56, draws 3 | Black 43, White 50, draws 7 |
| White's excess (White wins − Black wins) | **+15** | **+7** |

Notes:

- The attack rerun **reproduces batch 1's recorded match exactly**:
  its per-game records (openings, move lists, winners) are identical
  to `batch-v1-vs-v0-swap.json`, and all aggregates match. The
  harness path is bit-for-bit reproducible, and the judge wiring did
  not change the default-path behaviour before the switch.
- The tempo match's 76/24 swap split matches the seed-11 prototype
  split on the same first 100 openings (76/24) — the match openings
  are the seed-11 stream, an independent check of the split.
- v1's overall result is unchanged (0.6350, same CI — 60 + 7/2 =
  62 + 3/2 = 63.5): the judge changes colour assignment, not engine
  strength, and each engine still plays each colour exactly 50 times
  (alternating opener). What changed is WHICH games each engine plays
  as which colour — and draws rose 3 → 7.
- The match still starts White-favouring (White 50 vs Black 43), but
  the excess more than halved (+15 → +7). The residual edge lives in
  the openings (see the placer note above), not in the choice.

## Decision

**SWITCHED.** Both clauses of the pre-registered success metric are
met: (1) the choice split moves off 100/100 Black — 157/43 on the
200-opening set and 76/24 in the match; (2) the winners-by-colour
balance improves — White's excess shrinks from +15 to +7. The
harness default is now the tempo-aware judge:
`research.play_swap_game` and `research.play_match` default to
`judge="tempo"` (pinned by `test_play_match_default_judge_is_tempo`).
The old judge stays available as `judge="attack"`, and
`game.choose_swap_color` keeps its own default of `"attack"` (the
low-level primitive preserves its original behaviour; the harness is
where research batches get their judge).

Caveats, stated plainly:

- All 43 White picks in the 200-opening set are exact ties (margin
  0) decided by the White tie-break — the tempo judge's White picks
  come from the tie-break, not from White ever having a strictly
  higher setup value (Black's value is never below White's). The
  split is a property of the scorer's discrete tiers as much as of
  the positions.
- The match still starts White-favouring (+7). If a future batch
  wants winners-by-colour roughly balanced, the placer must balance
  the tempo-aware values, not the attack gap — that is the natural
  follow-up, and it changes the openings, so it needs its own
  pre-registered run.
- Cost: the tempo judge costs ~100 ms per opening vs ~8 ms for the
  attack judge (CPU time, measured) — about +10 s per 100-game swap
  match. Negligible for research batches.

## Reproduction

- Split: `/tmp/h3_split.py` (200 openings, seed 23) — the script is
  ephemeral; the parameters are: `place_swap_opening(placement=
  "balanced", rng=default_rng(23))`, both judges on each board.
  Per-opening records: `swap-judge-split-200.json` in this directory.
- Matches: `/tmp/h3_matches.py` — v1 `scoring.get_player(1, seed=100)`
  vs v0 `baseline.HeuristicPlayer(seed=200)`, 100 games, swap rule,
  balanced placement, `placement_seed=11`, alternating opener, once
  per judge. Each match was run twice (canonical `play_match` + a
  recording loop calling `play_swap_game`) and the two runs asserted
  identical on W/L/D, per-colour splits and `swap_choices`; every
  recorded game was replayed from its move list and the replayed
  winner asserted equal to the recorded winner.
- Design prototypes (A / B / B2 / C): `/tmp/h3_prototype.py`,
  `/tmp/h3_proto_bc.py`, `/tmp/h3_proto_b2.py`, `/tmp/h3_proto_b1max.py`
  (seed 11; ephemeral — numbers quoted in the table above).
