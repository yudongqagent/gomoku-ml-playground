# Iteration 64 — fix the v52 `_tss_after_attacker_move` defect

## 0. The problem

A pre-existing v52 defect, found by v59 in batch 14 and NOT fixed
there (fixing it would change move choice, so it belonged to a
future strength batch — this one). From `research/iterations/iter-59.md`
§2 (on branch `research/batch-14-h3`):

> In `_tss_after_attacker_move`, if the attacker's forcing move is
> itself an immediate winning move, the recursion misjudges the
> line: with exactly 1 extension threat the defender "blocks" it,
> `game_over` is True (the attacker's five persists), and the
> function returns False — but the attacker already won. (With >= 2
> extension threats it returns True; with 0 it falls through to the
> open-three check and returns False.) So TSS can miss a forced win
> whose converting move is an immediate win with < 2 extension
> cells. The engine never misses an immediate win at the ROOT (the
> win scan runs before TSS), so the defect only costs TSS forcing
> wins in deeper sequences.

This is a known bug in the champion with a concrete fix — not a
speculative lever. v64 is v52 + this fix, nothing else.

## 1. Design and pre-registration

### 1.1 The fix

In `ml/gomoku_ml/mcts_heuristic.py`, `_tss_after_attacker_move`
gains an entry check, gated behind a flag `fix_won_move`
(default False — v52 behaviour preserved exactly):

```python
if fix_won_move and game.is_winning_stone(board, last_move, attacker):
    return True
```

`game.is_winning_stone` is O(1) (only the four lines through the
stone) and exact: a five through `last_move` must include it, since
the cell was empty before the move was played — so True ⟺ the
just-played forcing move completed five, and the attacker has
already won. The flag threads through `_tss_win` and
`tss_winning_move` (same pattern as v59's `fast` flag), through
`HeuristicMCTS(tss_fix_won_move=...)` and `ScoringPlayer`, and
`ENGINE_VERSIONS[64]` is v52's dict + `"tss_fix_won_move": True`.
Everything else is v52 unchanged: 100 sims, c_puct 1.5, v1 priors,
2-ply rollout leaf, τ=0.05 tie-break, TSS at the root (depth 4,
node budget 3000). `baseline.py` stays FROZEN.

The flag-gating matters: the fix lives in shared TSS code, so an
ungated fix would also change v52's behaviour and the battery
would compare the fixed engine against itself. With the flag,
v52 in the battery is the v52 of record.

### 1.2 Reachability analysis (pre-registered expectation)

Before the battery, the mechanism question: can the defect bite
through `tss_winning_move` at all? Every path into
`_tss_after_attacker_move` plays a forcing move from a board the
attacker holds no winning move on — `tss_winning_move` checks
`scoring._winning_moves` at the root, `_tss_win` checks it at
entry, and the forced-block path in `_tss_win` and the counter-four
branch in `_tss_after_attacker_move` both handle a block that
completes five (`game_over` + `winner == attacker`). A forcing
move that completes five IS a winning move on its pre-move board,
so the entry checks should preclude the defect end-to-end. The
oracle check below (§2.2) confirms `scoring._winning_moves` is
complete. The differential (§2.3) is the empirical test of this
expectation on 1,113 mined/random/mechanism positions.

Pre-registered expectation: **the fix changes no played move on
any reachable position, so the battery should tie v52** — in which
case v64 is REJECTED on clause 1 (and clause 2, which needs strict
domination of the legs) despite the fix being correct at the
function level. A tie is a tie; the bar is the bar. If the
differential DOES find a reachable difference, the expectation is
wrong and the battery decides on its merits.

### 1.3 Battery (pre-registered)

- Clause 1: 100 games v64 vs v52, NORMAL rules, balanced openings
  (the new default — no `--no-openings`), fresh seeds 85001/86001.
  `ml/scripts/selfplay_batch.py --engine-a v64 --engine-b v52
  --games 100 --seed-a 85001 --seed-b 86001 --out
  research/iterations/batch-v64-vs-v52.json`.
- Clause 2: ladder, 40 games/leg, balanced openings, recorded legs,
  seed 100 (same as v52's ladder for comparability).
  `ml/scripts/eval_version.py --version 64 --ladder-games 40
  --out research/iterations/ladder-v64.json`. v64 must strictly
  dominate v52's legs (random 40-0, heuristic 40-0, mcts 31-9).
- Clause 3: tactical, both suites, seed 100.
  `ml/scripts/eval_tactical_full.py --version 64 --suite both
  --seed 100`.
- Clause 4: suites — `ml` pytest, `npm test`, `npx tsc --noEmit`
  (`npm ci` first in this fresh worktree).
- Secondary: 100-game swap match vs v7 (the standard secondary
  metric), balanced placement, tempo judge, placement_seed 11,
  fresh seeds 85002/86002.
  `ml/scripts/selfplay_batch.py --engine-a v64 --engine-b v7
  --games 100 --swap-rule --placement balanced --placement-seed 11
  --seed-a 85002 --seed-b 86002 --out
  research/iterations/batch-v64-vs-v7-swap.json`.
- Challenge match vs v52 (strength champion) ONLY if v64 passes
  all 4 clauses (champion rule v2 — and note clause 1 IS vs the
  champion here, so a clause-1 pass already is a direct win over
  the champion; the separate challenge match is not re-run — see
  §4).

All match drivers auto-resume from partial `--out` files; per-game
seeding with a uniqueness audit on every leg; uniqueness gate 0.6.
Aggregates and Wilson 95% CIs recomputed from raw records with
`ml/scripts/recompute_match.py` (draws count as half a win).

### 1.4 What does NOT change

Everything except the flag-gated entry check. No sims change, no
prior change, no leaf change, no defensive VCF check (CLOSED), no
depth/budget change (TSS depth is priced — v57, batch 14). No
kernel re-sync unless v64 passes the bar.

## 2. Implementation

`ml/gomoku_ml/mcts_heuristic.py`:

* `_tss_after_attacker_move(..., fix_won_move=False)` — the entry
  check above, plus the flag threaded into its three recursive
  calls (the two `_tss_win` calls and the counter-four self-call).
* `_tss_win(..., fix_won_move=False)` — threaded into its two
  `_tss_after_attacker_move` calls.
* `tss_winning_move(..., fix_won_move=False)` — threaded into its
  `_tss_after_attacker_move` call.
* `HeuristicMCTS.__init__` gains `tss_fix_won_move: bool = False`
  (class docstring updated); `choose_move` passes it to
  `tss_winning_move`.

`ml/gomoku_ml/scoring.py` — `ENGINE_VERSIONS[64]` = v52's dict +
`"tss_fix_won_move": True`; `ScoringPlayer.__init__` passes
`tss_fix_won_move` through to `HeuristicMCTS`.

### 2.1 Regression test

`ml/tests/test_v64.py`, 7 tests, all passing on the branch
(24.0s), all 7 FAILING on v52 code (verified against a scratch
copy of `ml/` with the two modified files reverted to HEAD:
TypeError on the unknown `fix_won_move` kwarg, KeyError 64,
ValueError unknown version):

* `test_already_won_one_extension_threat` — the recorded defect:
  Black five (7,5)–(7,9), White blocks the left end (7,4), last
  stone (7,9): exactly 1 extension threat (7,10). Flag off →
  False; flag on → True.
* `test_already_won_zero_extension_threats_edge` — five at the
  board edge (7,10)–(7,14), last stone (7,14): 0 threats. Flag
  off → False; flag on → True.
* `test_already_won_zero_extension_threats_blocked` — five with
  both ends blocked by White: 0 threats. Flag off → False; flag
  on → True.
* `test_two_extension_threats_unchanged` — control: 2 extension
  threats, both flag settings → True (v52 already correct here).
* `test_fix_is_flag_gated_v52_unchanged` — v52 has no fix flag,
  v64 sets it; every other registration key identical to v52.
* `test_fix_changes_nothing_on_mechanism_positions` — on the 5
  v52 mechanism positions the fix changes no TSS move and no node
  count.
* `test_v64_engine_plays_tss_win_on_t1` — the registered v64
  engine plays the TSS win (6,6) on the double-three position,
  exactly like v52.

### 2.2 Oracle check: `scoring._winning_moves` is complete

`/tmp/check_winning_moves.py` (scratch, not committed): brute-force
oracle (for every empty cell, apply + `is_winning_stone`) vs
`scoring._winning_moves` on 3,000 random positions × both colours:
**0 mismatches**. The entry checks the reachability analysis relies
on are sound — a winning move cannot slip past them.

### 2.3 Differential: fixed TSS vs v52 TSS on 1,113 positions

`/tmp/tss_diff.py` (scratch, not committed): `tss_winning_move`
with the flag on vs off on 1,113 positions — every 4th ply ≥ 6 of
30 recorded games each from `batch-v52-vs-v27.json`,
`batch-v52-vs-v0.json`, `batch-v52-vs-v27-swap.json`, plus the 5
mechanism positions, the 49 tactical-suite positions, and 200
random forcing-rich positions. TSS fired on 251 positions under
BOTH settings.

**Result: 0 positions differ — same move AND same node count
everywhere (including all 251 firings).** The fix changes no
played move on any reachable position tested, exactly as the
§1.2 analysis predicted: the entry checks already preclude a
winning move from being played as a forcing move, so the defect
never bites through `tss_winning_move`. The fix is correct at the
function level (§2.1) but has no played-strength effect at the
TSS level — the battery below is therefore expected to tie v52
(a tie fails clause 1 and clause 2), and the battery is run to
confirm that rather than to assume it.

## 3. Battery

All legs use per-game seeding with a uniqueness audit; aggregates
and Wilson 95% CIs recomputed from raw records with
`ml/scripts/recompute_match.py` (draws count as half a win).

### vs v52, 100 games, normal rules (clause 1)

**v64 53W-46L-1D, win rate 0.5350, Wilson 95% CI [0.4378,
0.6296]** — `research/iterations/batch-v64-vs-v52.json` (seeds
85001/86001, balanced openings, 3081s; recomputed, matches the
script summary exactly). **Clause 1 FAILS** — CI lower bound
0.4378 < 0.50. The de-dup cut also fails: 92/100 unique games,
48W-43L-1D, rate 0.5272, CI [0.4261, 0.6260]. Uniqueness gate
PASS (92 ≥ 60). Per-colour: A as Black 34W-16L-0D (0.68), A as
White 19W-30L-1D (0.39) — the same Black/White split v52 itself
shows (Black dominates, White loses), as expected for two engines
that play identically: the difference between the sides is the
seed streams, not the fix.

The result is the tie the §1.2 expectation predicted (0.5350 is
within noise of 0.50 — the exact tie the differential found at the
TSS level, with the MCTS seed streams supplying the ±0.035
wobble). A "win" by 7 games with a CI spanning 0.50 is not a bar
pass, and v64 does not claim it as one.

### Ladder (40 games/opponent, clause 2)

**Random leg: 40W-0L-0D** (matches v52's 40-0). **Heuristic leg:
24W-16L-0D** (as Black 17W-3L, as White 7W-13L) —
`research/iterations/ladder-v64.json` (seed 100, balanced
openings, recorded legs). **Mcts leg: 18W-22L-0D** (as Black
15W-5L, as White 3W-17L), win rate 0.4500, CI [0.3071, 0.6017].
**Ladder Elo: estimate 70.4, interval [-37.7, 178.6].** Uniqueness
gates all PASS (random 40/40, heuristic 37/40, mcts 38/40). The
ladder driver was interrupted by a runtime restart mid-mcts-leg
and resumed from the partial file — random and heuristic legs kept,
RNG state restored; the resume is recorded in the file
(`resumed_from_leg: mcts`).

The mcts leg confirms the framework finding: under balanced
openings White collapses against EVERY opponent (7-13 vs v0,
3-17 vs the ladder MCTS), while Black dominates (17-3, 15-5).
v52's empty-board mcts leg was 31-9 (Black 18-2, White 13-7) —
the openings, not the engine, moved the White numbers. The
re-baseline ladder (below) measures v52's own legs under the
same openings; the clause-2 verdict below is stated on both
readings.

Clause 2, literal pre-registered reading (dominate v52's
empty-board legs 40-0/40-0/31-9): **FAILS** — the heuristic leg
(24-16) does not dominate 40-0 and the mcts leg (18-22) does not
beat 31-9. But that reading compares across frameworks (see the
framework finding above): v52 itself does not achieve 40-0/31-9
under balanced openings, so the literal bar is unachievable
under the new default. Clause 2, same-framework reading (v64's
legs vs v52's legs both under balanced openings): **ALSO FAILS**
— but as a tie, not a loss. The re-baseline ladder
(`research/iterations/ladder-v52-balanced.json`, seed 100,
balanced openings) gives v52's legs as **random 40-0, heuristic
24-16, mcts 18-22, Elo 70.4** — identical to v64's on every leg,
and **all 120 games move-for-move identical** between the two
ladders (verified by record comparison; even the final RNG state
matches). v64 does not strictly dominate v52's legs — it IS v52,
in played games. So clause 2 fails under either reading. The
re-baseline legs (40-0/24-16/18-22, Elo 70.4) are the correct
clause-2 reference for future versions under the new framework;
the old 40-0/40-0/31-9 reference belongs to the empty-board
framework and must not be used for balanced-openings ladders.

**Framework finding (the heuristic leg is NOT a v64 weakness).**
v52's own heuristic leg under the SAME balanced openings and seed
is 27W-13L-0D — and a fresh v64 heuristic leg is 27W-13L-0D with
the IDENTICAL lost games [1, 3, 4, 7, 9, 15, 16, 25, 27, 28, 29,
31, 33] and identical colour splits (Black 17-3, White 10-10).
(`/tmp/check_heuristic_leg.py`, scratch.) The 24-16 vs 27-13
difference between the ladder run and the fresh run is the RNG
stream (the ladder player's RNG advances through the random leg
first). So:

* v64 ≡ v52 move-for-move on the heuristic leg too — the fix
  changes nothing in played games, again.
* The pre-registered clause-2 reference (heuristic 40-0) was
  measured from the EMPTY board (v52's ladder, pre-audit). Under
  the new default balanced openings, v52 itself scores only
  ~27-13 on the heuristic leg. The balanced openings are NOT
  balanced in play: White loses 10-13 of 20 to the greedy
  baseline from them (losses spread across 12 openings, not one
  bad opening — game 1 traced in §5: Black builds an open four
  on the (6,4)–(10,8) diagonal seeded by the opening stone and
  White cannot block both ends; a prevention failure, not a missed
  block). The attack-gap-0 criterion (v1 score) does not capture
  the true colour balance.
* Consequence for clause 2: the literal pre-registered reading
  (dominate 40-0/40-0/31-9) is unachievable under the new
  framework even for v52 itself, so it cannot be the operative
  bar. The honest same-framework comparison is v64's legs vs
  v52's legs BOTH under balanced openings — hence the v52
  re-baseline ladder below. Clause 2 is reported on BOTH readings
  (literal: FAILS; same-framework: see re-baseline). The verdict
  does not turn on this: clause 1 already fails, so v64 is
  rejected either way.

### v52 re-baseline ladder (same framework)

`ml/scripts/eval_version.py --version 52 --ladder-games 40 --out
research/iterations/ladder-v52-balanced.json` (seed 100, balanced
openings) — v52's legs under the new default framework: **random
40-0, heuristic 24-16, mcts 18-22, Elo 70.4** (interval [-37.7,
178.6]; uniqueness 40/40, 37/40, 38/40 — all gates PASS). Every
leg and every game identical to v64's ladder. This is the correct
clause-2 reference for future versions under the new framework;
the old 40-0/40-0/31-9 reference belongs to the empty-board
framework.

### Tactical (clause 3)

v64 at `--seed 100`: **old 20/20, hard 29/29**
(`research/iterations/tactical-v64.txt`). **Clause 3 PASSES.**

### Swap vs v7, 100 games (secondary metric)

[PENDING]

### Suites (clause 4)

* `npm test`: 55 passed (4 files), `npx tsc --noEmit`: clean
  (run after `npm ci` in this fresh worktree).
* `ml` pytest: 503 passed, 7 skipped, 58 xfailed, 0 failures
  (697.66s, re-run on the branch to verify — the earlier run's
  numbers were unrecorded). **Clause 4 PASSES.**

## 4. Verdict

**v64 REJECTED — passes 2/4 clauses (3 and 4 pass; 1 and 2
fail). v52 stays the heuristic bar reference AND strength
champion. No page deployment to Master is needed.** No challenge
match (not a bar-passer; clause 1 was itself vs the champion).

| Clause | Result | Numbers |
|---|---|---|
| 1 — h2h vs v52, 100 games | FAILS | 53W-46L-1D, 0.5350, CI [0.4378, 0.6296]; de-dup 92 unique, 0.5272, CI [0.4261, 0.6260] |
| 2 — ladder, 40 games/leg | FAILS (tie) | random 40-0, heuristic 24-16, mcts 18-22; Elo 70.4 ([-37.7, 178.6]); identical to v52's same-framework re-baseline on all 120 games |
| 3 — tactical | PASSES | 20/20 + 29/29 |
| 4 — suites | PASSES | pytest 503 passed / 7 skipped / 58 xfailed; npm 55/55; tsc clean |

Secondary: swap vs v7 [PENDING — match running].

## 5. Notes

### The fix is correct and untestable-in-play

The defect was real (function-level: v52 returns False where the
fixed code returns True on all four constructed cases, and the
regression test fails on v52 code, passes on v64). But it is
unreachable in played games: every caller checks for a winning
move before playing a forcing move, so no forcing sequence ever
reaches the defective branch. The differential (0 differences on
1,113 positions, 251 TSS firings) and the battery (v64 ≡ v52 on
all 120 ladder games and 53-46-1 on the 100-game h2h — a tie
within noise) agree: the fix changes no played move. A correct
fix to an unreachable defect is a null result, and the battery
measured it as one. The regression test stays in the suite as
protection against future TSS refactors that might route through
the defective branch.

### The balanced openings are not balanced in play

The eval-audit "balanced" openings (`research/openings-balanced.json`,
attack gap 0 by v1 score) are NOT balanced in play: White
collapses against every opponent from them (7-13 vs v0, 3-17 vs
the ladder MCTS; losses spread across 12 openings, not one bad
opening). Game 1 traced: Black builds an open four on the
(6,4)–(10,8) diagonal seeded by the opening stone; White cannot
block both ends — a prevention failure, not a missed block. The
v1-score attack gap does not capture true colour balance. Two
consequences: (1) the literal clause-2 bar (40-0/40-0/31-9,
measured from the empty board) is unachievable under the new
default framework even for v52 itself — the re-baseline legs
(40-0/24-16/18-22, Elo 70.4) are the correct clause-2 reference
for future versions under balanced openings, and the old legs
belong to the empty-board framework; (2) in a 40-game leg with 24
openings, each opening is only ever played by A in one colour
(parity preserved by +24), so colour and opening are confounded
within a leg — a future framework fix should rotate openings
across colours. Neither consequence changes v64's verdict:
clause 1 already fails, so v64 is rejected either way.
