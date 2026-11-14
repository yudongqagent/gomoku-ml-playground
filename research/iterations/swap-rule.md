# Swap rule — interpretation used by the auto-research program

> **Status: interpretation, awaiting the user's correction.** This note
> records exactly which swap rule the harness implements and which
> policies it uses, so the user can correct it if he meant Swap2 or
> another variant. The code lives in `ml/gomoku_ml/game.py`
> (`build_swap_opening`, `is_swap_opening`, `choose_swap_color`) and
> `ml/gomoku_ml/research.py` (`place_swap_opening`, `play_swap_game`,
> `play_match(..., swap_rule=True)`).

## The rule (standard swap / pie rule)

1. The **opener** places **3 stones** on the empty board: **2 black,
   1 white**, on any distinct empty points, in the order black, white,
   black.
2. The **second player** then **chooses which colour to play**.
3. Play continues with **White to move** — move 4 is White's. The 3
   opening stones ARE moves 1–3 of the game (black, white, black), so
   the game simply continues alternating. Verified against the
   official RIF rules, Article 2.4: *"After this, the player with
   white stones will place a stone and thereafter, they will alternate
   in placing stones until the end of the game."*
   (https://www.renju.net/gomokurules/)

> **Correction (iteration 2, 2026-11-12):** an earlier version of this
> note and of the harness wrongly had Black to move after the swap.
> That gave Black 3 of the first 4 moves and made the swap choice
> degenerate (second player took Black 20/20, opener lost 0/10 as
> White). All swap-rule numbers recorded before this correction
> (including the batch-1 20-game match in iter-1.md) were measured
> under the wrong turn order and are **invalid as swap-rule evidence**;
> the normal-rules head-to-head numbers are unaffected. The fix:
> `research.play_swap_game` now starts the game with White to move,
> and the colour-choice tie-break is White (see policy (ii)).

This is the standard swap rule (pie rule), **not** Swap2. Under Swap2
the second player would have a third option (place 2 more stones and
pass the colour choice back to the opener); that option is **not**
implemented. If the user meant Swap2, the harness needs a second
placement phase and a second choice — the current code would need a
new game phase, not just a tweak.

**Scope note — win condition:** this program uses the project's
existing freestyle rules (a line of **five or more** wins; overline
counts). The RIF rules require *exactly* five. Only the swap
*opening procedure* (3 stones + choice + White to move) is adopted
from the official rules; the exactly-five rule is NOT adopted. If the
user wants exactly-five as well, that is a separate change to
`game.py`.

## Policy (i): how the 3 opening stones are placed in self-play

Two policies exist (`research.place_swap_opening(..., placement=...)`):

- **`"chooser"` (batch-1 policy, now legacy):** the opener uses its
  own move chooser for all three stones, playing each colour's best
  move as it sees it, in order: Black stone, White stone, Black
  stone. Rationale: the opener is one agent placing all three stones,
  so it should place each stone the way it would play that colour.
  Consequence: both current engines open at the centre (7,7) for the
  first black stone. **Found flawed** (see "Findings from batch 1"
  below): it makes Black systematically stronger.
- **`"balanced"` (policy v2, current default for research batches):**
  the opener places stones to BALANCE the position. All three stones
  stay in a compact region (within Chebyshev distance 2 of the
  centre — a 5×5 box): Black's first stone is the centre; White's
  stone is drawn (seeded RNG) from the cells within Chebyshev
  distance 2 of the centre, giving varied openings; Black's second
  stone is the nearby candidate minimising the gap between the two
  colours' best attack scores on the finished opening. The compact
  region matters: an unconstrained gap search finds degenerate
  scattered openings (gap 0 by isolation — stones that do not
  interact), which are not real Gomoku positions. White's stone is
  drawn rather than "White's best attack move" because with no White
  stones on the board every move scores ~equally, so argmax would
  pick an arbitrary edge cell.

## Policy (ii): how the colour choice is made

> **Update (batch 2, H3, 2026-11-13):** the harness default is now
> the tempo-aware judge (`judge="tempo"` — symmetric setup values,
> ties to White; see `swap-judge-v2.md`). The attack-gap judge below
> remains available as `judge="attack"`.

The second player compares the **best available ATTACK score for each
colour** (max over legal moves of the v1 attack-only scorer,
`gomoku_ml.scoring.score_attack`) and takes the colour with the higher
best score. **Ties go to White**: White plays move 4 (RIF Art. 2.4),
so the next-move tempo breaks the tie. (An earlier version tied to
Black, following the wrong Black-to-move turn order — corrected with
the turn-order fix above.)

### Why attack-only — a real trap, found by testing

The naive version of this policy — "compare the best heuristic score
for each colour" using the full `score_move` — is **vacuous**. The v1
score is attack + defence at full value, and at any square the attack
and defence components swap roles between colours, so both colours
score **identically at every square** (verified empirically: max
difference 0 over all legal moves on a 16-stone test board). The
comparison would always tie and the choice would always fall to the
BLACK tie-break, making the swap rule a no-op. The attack-only scorer
breaks the symmetry: it measures whose next move builds more, which is
what the colour choice should turn on.

`game.choose_swap_color` takes the scorer as an injected callable and
documents the asymmetry requirement; the harness passes
`scoring.score_attack` by default. Any future scorer used here must
also be colour-asymmetric (a full-board evaluation from one side's
perspective would also work).

## What the swap choice can and cannot do

- With the current engines the swap choice is a **3-stone heuristic
  judgement**, not a deep evaluation: it sees the best next attack for
  each colour, not the long-term value of the position. A stronger
  opener could in principle place stones that fool the 3-stone
  judgement; that is a known limitation, not a bug.
- The `as_black` / `as_white` breakdown of a swap match is **not
  balanced by construction**: the swap choice decides the colours, so
  one player may end up Black more often. `play_match` reports the
  swap choices (`swap_choices: {black, white}`) alongside the result
  so the imbalance is visible.
- The opener alternates across games in a match (game `i` is opened
  by A when `i` is even), so neither player gets the opener's
  advantage systematically.

## Findings from batch 1 (2026-11-12, Worker A)

A 20-game swap-rule match (scoring-v1 vs baseline heuristic, seeds
500/600) exposed a flaw in policy (i):

- The second player took **Black in 20/20 games** (`swap_choices:
  {black: 20, white: 0}`) — the colour choice never once picked
  White, whichever engine opened.
- The opener lost as White **10/10** when it opened (scoring opened
  10 games, lost all 10 as White; when baseline opened, scoring took
  Black and went 5W-5L). Overall the match was 5W-15L for scoring —
  but the colour split (as Black 5W-5L, as White 0W-10L) shows the
  result is driven by the opening, not by engine strength: in the
  same batch, the non-swap head-to-head had scoring winning 37W-9L
  as White.
- **Diagnosis:** policy (i) places each colour's *best* move, which
  makes Black — who gets two placed stones — systematically stronger
  than White. The second player then correctly takes Black every
  time, and the opener is stuck with the weaker colour. Under the pie
  rule the opener should place **balanced** stones (neither colour
  clearly better), not each colour's best move. The current policy
  defeats the purpose of the swap rule.
- **Recommended iteration-2 change:** balanced placement — choose
  the 3 stones to minimise the gap between the two colours' best
  attack scores (e.g. greedy: centre first, then pick the White
  stone and second Black stone that bring the attack-score gap
  closest to zero). Re-run the swap match afterwards: the second
  player's colour split should move away from 20/20 Black, and the
  opener's as-White record should stop being 0/10. Policy (i) is
  left as implemented for batch 1; this note records the finding so
  the coordinator can schedule the fix.

## Findings from iteration 2 (2026-11-13, fixed rules + balanced placement)

Two 100-game batches under the CORRECTED turn order (White to move
after the opening) with balanced placement (see
`research/iterations/iter-2.md`):

- **The mirror match is now fair:** v1-vs-v1 went 30W-37L-33D
  (win rate 0.4650, CI [0.3704, 0.5622]) with winners by colour
  balanced (Black 32, White 35, draws 33). The pre-fix degeneracy
  (opener losing 10/10 as White) is gone.
- **But the swap choice is STILL 100/100 Black** in both batches.
  The opening gap is 77-78 in every game, always favouring Black:
  with 2 stones against White's 1, Black's best attack is always
  higher, and the gap minimiser cannot find a black2 placement
  (within the compact 5x5 region) that closes it. The balanced
  placement fixed the *placement* degeneracy but not the *choice*
  degeneracy.
- **And the choice is wrong:** in v1-vs-v0, White won MORE games
  overall (White 56, Black 41, draws 3) despite the second player
  always taking Black. The attack-gap judge measures whose next
  move builds more, but ignores tempo: White moves next (move 4),
  and the tempo outweighs the 77-point attack edge. The colour
  choice needs a tempo-aware judge — e.g. compare White's best
  attack against Black's best attack AFTER White's reply (1-ply
  lookahead), or apply a tempo handicap to Black's score.
  Next-batch work.
- **Match comparisons stay fair:** the opener alternates, so each
  engine is the second player (and takes Black) in 50% of games.
  The wrong choice biases the *games* (they start White-favouring)
  but not the *comparison* — both engines face both colours
  equally. The swap rule's purpose (letting the second player pick
  the better colour) is not yet realised, however.

## Findings from batch 2, H3 (2026-11-13, tempo-aware judge)

The tempo-aware judge is implemented
(`game.choose_swap_color_tempo`, `judge="tempo"`) and is now the
**harness default** (`research.play_swap_game` / `play_match`
default to `judge="tempo"`). Full definition, design analysis and
numbers: `research/iterations/swap-judge-v2.md`.

- **Definition:** each colour's value is its best follow-up attack
  after its own best setup move (symmetric 1-ply development; ties
  among best-immediate moves broken by best follow-up). Higher value
  wins; ties go to White (White moves next — the tempo breaks the
  tie). The old attack-gap judge stays available as
  `judge="attack"`; `game.choose_swap_color` keeps its own default
  of `"attack"`.
- **Why not the naive 1-ply lookahead:** simulating White's best
  move 4 and comparing best attacks after it picks White 50/50
  (after-margin mean −3842) — White's setup unlocks a gapped
  three-in-a-window (~4006) while Black's non-interacting stones
  cap at ~164. Any sequential lookahead has this last-move-wins
  artifact (a 2-ply minimax version picks Black 20/20 by ~3900).
  A fixed tempo handicap cannot mix the split either: the attack
  gap is 77–78 in every balanced opening. Only the symmetric
  comparison mixes the split.
- **Choice split (same 200 balanced openings, seed 23):** attack
  judge Black 200 / White 0; tempo judge **Black 157 / White 43**.
  Tempo margins are exactly bimodal (0 in 43 openings, 78 in 157):
  all White picks are exact ties decided by the tempo tie-break.
- **Swap matches (v1 vs v0, 100 games, balanced, placement seed
  11):** attack judge 62W-35L-3D (win rate 0.6350, CI [0.5373,
  0.7227]), swap Black 100 / White 0, winners by colour Black 41 /
  White 56 / draws 3 — this rerun reproduces batch 1's recorded
  match exactly (identical per-game records). Tempo judge
  60W-33L-7D (win rate 0.6350, same CI), swap Black 76 / White 24,
  winners by colour Black 43 / White 50 / draws 7. White's excess
  shrank from +15 to +7.
- **Decision: SWITCHED.** Both pre-registered clauses met (split
  off 100/100 Black; White's excess shrinks). Caveat: the match
  still starts White-favouring (+7) — the residual edge lives in
  the openings, and fully balancing winners-by-colour needs the
  placer to balance the tempo-aware values, not the attack gap
  (follow-up work, not part of H3).

## Open questions for the user

1. **Swap or Swap2?** The harness implements the standard 3-stone
   swap. If Swap2 was meant, say so — it needs a second placement
   phase.
2. **Opening placement:** balanced placement (v2) is now the
   default. Is a fixed set of known-balanced openings preferable?
3. **Colour choice:** the current judge ("best attack score, ties
   to White") systematically picks Black and is empirically wrong
   (White wins more). Should the choice use a deeper evaluation
   (e.g. a short search per colour, or a tempo-aware 1-ply
   lookahead)?
   **Resolved in batch 2 (H3, 2026-11-13):** the tempo-aware judge
   (symmetric setup values, ties to White) is now the harness
   default — split 157/43 on 200 openings, White's match excess
   +15 → +7. See `swap-judge-v2.md`. The residual White edge is a
   placer question now, not a judge question.
