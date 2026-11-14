# Tactical suite v2 — the harder suite (batch 2, Step 0)

Built 2026-11-13 on branch `research/batch-2`, before any batch-2
version was evaluated. File: `ml/gomoku_ml/tactical_positions_hard.json`.
Build script: `ml/scripts/build_tactical_hard.py` (run with
`--assemble` to reproduce the suite from the batch-1 game records).
Tests: `ml/tests/test_tactical_hard.py`.

## Why a new suite

The v1 suite (`tactical_positions.json`, 20 positions) does not
discriminate: scoring v1, v2 and v3 all scored 20/20, and the frozen
baseline also scores 20/20. Every position is solvable by 1-ply
win/block/double-threat scoring. Batch 2's versions are search-based
(H1 2-ply minimax, H2 MCTS with heuristic prior), so the suite must
test 2-ply planning: preventing an open four one move ahead, and
setting one up.

## Positions: 29

| Category | Count | Source |
|---|---|---|
| prevention | 16 | batch-1 lost games, 1 ply before the decisive open four |
| setup | 7 | 6 from batch-1 won games, 2 plies before the decisive open four; 1 hand-constructed control |
| winning | 2 | hand-constructed (2-ply setup and/or VCF first moves) |
| vcf | 4 | hand-constructed (forced win by consecutive fours) |

Colours: 17 Black to move, 12 White to move (prevention 8/8; setup
4/3; winning 2/0; vcf 3/1).

### Prevention (16)

One ply before the decisive open four, the loser to move, the
opponent holding open-four creators and no immediate win. In the
game the loser played a non-preventing move and lost next move.
Expected moves are EXACTLY the preventing moves: an immediate win, a
forcing four, or a move leaving the opponent with zero live
creators (verified exhaustively over all legal moves). Safe-list
sizes are 2-4 (sharp: a random move almost never passes).

Sources: v1-vs-v0-swap (3), v1-vs-v1-swap (3), v2-vs-v1-swap (3),
v2-vs-v1 (2), v3-vs-v0 (2), v2-vs-v0 (2), v1-black-losses (1).
Loser engines: v1 8, v0 4, v2 4 (v3 never missed a clean prevention
in batch 1 — its threat check was built for exactly this). Includes
the iter-3 canonical case (`prev-v1v0s-g13`: v1 played (5,4), the
preventing move is the forcing four (10,7)).

### Setup (7)

Two plies before the decisive open four, the winner to move, holding
no creator yet. In the game the winner played a setup and created
the decisive open four two plies later. Expected moves are EXACTLY
the 2-ply setups: after the move the opponent has no immediate win,
and for every reply an open-four creator survives (verified
exhaustively; only "dangerous" replies need checking — a reply can
only break a setup by occupying a creator, blocking a creator's
winning square, or creating an opponent four). Setup lists are size
1-2. The hand-constructed control (`setup-double-open-three`) is a
double-open-three creation every 1-ply engine finds — it guards
against regressions on easy setups.

### Winning (2)

Hand-constructed positions where genuine winning moves exist of two
kinds: 2-ply setups AND VCF first moves. Listing only one kind
would mark a winning move wrong, so the expected list is exactly
their union. Both are traps for 1-ply scoring: the highest-scoring
moves are double-open-three decoys (gap-threes whose extensions
give only single-ended fours) or a block that scores above the win.

### VCF (4)

Hand-constructed forced wins by consecutive fours (expect type
`vcf`, checked by `evaluate.starts_forced_win`). Two are race
positions (`vcf-race-block-or-win`, Black and White to move): the
side to move holds an open-four creator while the opponent holds an
open three — 1-ply engines block instead of winning. Two are
two-four sequences (`vcf-two-four-sequence`, original and rotated):
the forcing move is a quiet four whose follow-up only becomes a
four after the opponent's forced block; the trap four's block
occupies the square the sequence needs.

## How positions were extracted and verified

Extraction (batch-1 games only — no batch-2 version is judged
against a suite it was tuned on):

1. All 600 decisive games in the 8 batch record files plus
   `v1-black-losses.json` were replayed. Every decisive game was
   won via an open four (0 games skipped for "no creator") — the
   batch-1 finding, confirmed on the full set.
2. The decisive creator is the winner's LAST move before the final
   move that left the winner >= 2 winning moves.
3. Prevention: the board 1 ply before the creator (loser to move),
   kept only if the loser's actual game move was NOT preventing.
   272 candidates; 181 dropped because NO preventing move existed
   (the game was already lost — the race was decided earlier, as
   iter-3 found: only 13 clean missed preventions in 35 losses).
   242 duplicate boards dropped (deterministic engines transpose
   to the same position). 91 usable; 16 selected by fixed quotas
   (loser engine x colour, per-file cap 3, sharp lists preferred).
4. Setup: the board 2 plies before the creator (winner to move),
   kept only if the winner's actual game move IS a setup (and no
   creator pre-existed — otherwise the setup list is meaningless:
   any move leaving the creator live "passes"). 197 candidates;
   6 selected from a 3-per-file shortlist, all CLEAN (no VCF first
   move outside the setup list — 7 of the 22 shortlisted had one
   and were not picked as pure setups).

Verification (all independent of the engines, from
`gomoku_ml.game` / `gomoku_ml.evaluate` only):

* Every expected-move list is EXACTLY the set of moves satisfying
  its property, computed exhaustively (see the property definitions
  in the test file header). The build script asserts this at
  assembly; the test file re-implements the checks independently
  and re-verifies every position.
* The prevention check was cross-checked against
  `scoring._move_is_safe` on 12 sampled boards (0 mismatches), and
  the setup check against a brute-force all-replies version on
  1,645 move-decisions (0 mismatches).
* Fairness: for every position, every move that wins by the checked
  routes (immediate win, VCF, 2-ply setup) is in the expected list
  — a winning move is never marked wrong. Prevention lists include
  all forcing moves (hence all VCF first moves); setup positions
  were required to have no VCF move outside the list; winning
  positions list the union.

Dropped during construction (not in the suite):

* 181 prevention candidates with no preventing move (already lost).
* 242 duplicate boards.
* VCF-ROT (vcf-1 rotated): 8 setup moves exist outside the VCF
  check — a VCT win would be marked wrong. Unfair as a `vcf`
  position; the v1 suite's vcf-1/vcf-2 already cover that shape.
* VCF-TS (from the web engine's `ai.test.ts`): three open-four
  creators already on the board — every engine passes trivially.
* SETUP-HARD v1/v2 designs: the intended quiet setup move was
  joined by extra setup moves scoring 58k (double open three),
  so 1-ply engines passed them. (The v2 board's extra setup moves
  were real — the design was wrong, not the checker.)
* 7 game-setup shortlist entries with VCF moves outside the setup
  list (kept out of the pure-setup category; the union treatment
  exists for the two hand `winning` positions instead).

Known limitation (same narrowness as the v1 suite's VCF
positions): the checks verify wins by consecutive fours (depth 3)
and by 2-ply setup. A move winning by a longer, non-VCF route
would not be listed. No such move is known on any suite position.

## Scores (seed 0)

| Engine | Score | Prevention | Setup | Winning | VCF |
|---|---|---|---|---|---|
| scoring v1 (ScoringPlayer version=1) — the clause-3 reference | **12/29 (0.414)** | 2/16 | 7/7 | 1/2 | 2/4 |
| frozen baseline v0 (HeuristicPlayer) | 11/29 (0.379) | 0/16 | 6/7 | 1/2 | 4/4 |
| scoring v3 (version=3) | 26/29 (0.897) | 16/16 | 7/7 | 1/2 | 2/4 |

v1's 12/29 is stable across seeds 0, 1, 2 (no tie-break sensitivity).
The suite discriminates where the v1 suite did not: v1 12 vs v3 26
(the v1 suite had v1 = v2 = v3 = 20/20). The gap is the prevention
category (2/16 vs 16/16) — v3's threat check was built in batch 1
for exactly this, and it was NOT tuned on this suite.

Honest readings of the categories:

* The 7 game-extracted setups do NOT discriminate v1 (7/7): they
  are positions where a 1-ply engine already played the setup in
  the source game, so 1-ply scoring finds them again. The
  discriminating setups are the two hand `winning` positions (1/2
  for v1, v3 and v0 alike). A future suite revision wanting harder
  setups needs them constructed (like `winning-quiet-setup-2`),
  not extracted from 1-ply games.
* The 3 positions v3 still fails — `winning-quiet-setup-2` and
  both `vcf-race-block-or-win` positions — are the headroom batch-2
  search versions should target. A batch-2 version that only beats
  v1's 12/29 on prevention is not evidence of 2-ply search; the
  `winning`/`vcf` categories are the search test.
* v0 passes all 4 VCF positions (it takes the win in the race) but
  fails all 16 preventions: it has no creator-safety check at all.
  v1 has the check as a scoring penalty, not a filter, so unsafe
  high-scoring moves still beat quiet safe ones — that is the
  batch-1 finding ("v1 loses the open-four race") in suite form.

## The rule for batch 2 (clause 3)

Bar clause 3 (`research/README.md`, unchanged for batch 2):
tactical suite pass rate >= the previous version's. For batch 2,
"the tactical suite" is THIS suite
(`tactical_positions_hard.json`), and "the previous version's" score
is v1's score recorded above — v1 is the reference version for
batch 2 (no batch-1 version passed the bar, so v1 remains the
latest version; its suite is the one to beat). Run it with
`research.run_tactical(chooser, suite)` where `suite` is loaded
from `tactical_positions_hard.json` (the default
`evaluate.load_tactical_suite()` still loads the v1 suite — pass
the path explicitly).

No batch-2 version may be tuned on these positions: the suite is a
test set, not training data. Positions come from batch-1 games
only (plus the hand constructions above, designed from the batch-1
loss analysis, not from any batch-2 version's behaviour).
