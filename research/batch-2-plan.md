# Batch 2 plan — pre-registered

Status: **approved by Yudong 2026-11-13; batch 2 in progress on branch
`research/batch-2`.** No batch-2 work had run when this plan was
written; the hypotheses, bar, and choices below are unchanged from
the draft. Nothing here changes master or the "better" bar without
his word.
Written 2026-11-13, from the batch-1 findings in `research/VERSIONS.md`,
`research/iterations/iter-1.md`, `iter-2.md`, `iter-3.md`, `swap-rule.md`.

## Direction (from batch 1's conclusion)

1-ply scoring tweaks are below the noise floor of 100-game matches
(v1 Elo 70.4 rejected on CI, v2 refuted at -52.5, v3 107.5 rejected on
h2h vs v1 at 0.47). Batch 2 moves to **search**:

- **H1 — 2-ply minimax over v1's top candidates.** Batch 1's loss
  analysis: every decisive game (35/35) was decided by an open four;
  v1 loses fast (27.5 plies) and wins slow (69.5) — it loses the
  open-four race. A 2-ply lookahead that checks whether the opponent
  gains an open-four creator after our move directly targets that.
- **H2 — MCTS with the heuristic as prior.** If H1's fixed-depth
  search stalls, use v1 scores as the prior in MCTS instead of
  hand-tuning more 1-ply weights.
- **H3 — tempo-aware colour judge for the swap rule.** Batch 1: the
  judge picks Black 100/100 but White won more games (White 56,
  Black 41 in v1-vs-v0) — the judge ignores that White moves next
  (move 4). Fix: compare White's best attack against Black's best
  attack *after* White's reply (1-ply lookahead), or a tempo
  handicap on Black's score. Success metric: the choice split moves
  off 100/100 Black and the swap match stops starting White-favouring.

## Opening rule: standard swap (recommendation)

**Stay with standard swap, not Swap2.** Reasons:

1. Yudong's instruction was the official rule: "3 plays to start,
   then opponent can switch" — that is standard swap (RIF Art. 2.4),
   which batch 1 implemented. Swap2 is a tournament variant, not
   the rule he described.
2. Switching now breaks comparability: all batch-1 swap numbers are
   under standard swap (and the pre-fix ones are already invalid).
3. Swap2 fixes first-player advantage. That is not what is losing
   games — open-four races and search depth are. Swap2 costs a
   second placement phase plus a second choice in the harness, for
   no gain against the identified bottleneck.
4. The swap-rule note's other open items are technical, not
   decisions: balanced placement (v2) is already the default, and
   the colour judge fix is H3 above.

If Yudong meant Swap2, the harness needs a new game phase — say so
and batch 2 starts there instead.

## Harder tactical suite

The current 20-position suite does not discriminate: v1, v2 and v3
all scored 20/20. Before batch 2's first eval, build a harder suite
from batch-1 lost games (the 35 decisive open-four positions) plus
new VCF/VCT positions, and record v1's score on it as the new
baseline. No version is judged against a suite it was tuned on —
suite positions come from batch-1 games only.

## The "better" bar (unchanged, pre-registered)

Same four clauses as batch 1 (`research/README.md`): (1) head-to-head
vs previous version, >= 40 games, alternating colours, win rate
(draws = half a win) > 0.50 with Wilson 95% CI lower bound > 0.50;
(2) ladder Elo strictly higher, identical game counts/opponents;
(3) tactical suite pass rate >= previous version's; (4) all test
suites green (ml pytest, npm test). A version failing any clause is
recorded as rejected and NOT pushed to master.

Pre-registered choice on the README's open item: **clause 1 stays
normal-rules head-to-head**, keeping comparability with batch 1.
Swap-rule head-to-head is reported as a secondary metric, not the
bar. (Batch 1 found v1 beats baseline clearly under swap, 0.635
with CI excluding 0.50, while normal-rules h2h was 0.55 with CI
including 0.50 — the swap number flatters; the normal-rules bar is
the stricter, comparable one.)

## Shape and limits

- Max 3 versions, same as batch 1; each with 100+ games, loss
  analysis with written learnings, Elo check, commit only on pass.
- Kernel re-sync (NumPy/WebGPU/Metal) only when a version passes —
  the kernel ports the winning scorer, not each candidate.
- Honest limits, unchanged: ladder Elo is relative to this ladder
  only and is NOT comparable to human ratings; the "exceeds all human
  masters" end goal is not measurable here without human opponents
  or rated games against strong engines. Batch 2 reports ladder Elo
  plus head-to-head vs the search-based Master engine.
- Freestyle win condition stays five-or-more (overline counts); the
  RIF exactly-five rule is NOT adopted unless Yudong asks.

## The one decision needed

Reply **"go batch 2, standard swap"** (or correct any line above)
and the batch starts. The other open item — the iPhone benchmark —
needs no decision: the Benchmark tab is built into the web app and
verified compiling (see below); it just needs deploying to the
hosted page and a 2–3 minute run in Safari on his iPhone.

## Benchmark tab — deploy readiness (verified 2026-11-13 ~07:54 PST)

- `src/components/BenchmarkView.tsx` is wired into `src/App.tsx`
  (Play / Trainer / Benchmark tabs); instructions in
  `research/iphone-benchmark.md`.
- `npm run build` (tsc + vite) run on branch `research/batch-1`:
  clean, 65 modules, production `dist/` produced. No on-device
  numbers exist yet — only his iPhone run produces them (Safari,
  iOS 26+ for WebGPU; JS-reference fallback on older iOS; Low Power
  Mode off; run 2–3 times and report all runs).
- Remaining step: deploy this `dist/` to the hosted Space
  (slug `gomoku-ml-playground`). The Space was not touched by batch 1.
