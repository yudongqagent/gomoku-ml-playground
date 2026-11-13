# Gomoku auto-research log

Self-play research loop for the heuristic scoring engine. The loop runs in
batches; each batch is reported back before the next starts.

## Canonical scoring function

The canonical scoring function for self-play is the Python heuristic in
`ml/gomoku_ml/baseline.py` (`score_move` / `choose_move`), because it is
what the self-play and evaluation pipeline uses: it is the eval ladder's
`heuristic` opponent and the source of priors for the heuristic-MCTS
player. The web engine (`src/ai/patterns.ts`) is window-based and is the
reference for the 5-window pattern fix, and the deployment target for the
portable scoring kernel (WebGPU / Metal). The kernel ports the same
function; cross-checks against the Python reference are the source of
truth for equivalence.

## Honesty constraints (binding on every iteration)

- **Ladder Elo is relative to THIS ladder only** (anchors: random = -800,
  heuristic = 0 — arbitrary reference points). It is not comparable to
  human ratings, online ratings, or any other system. The program's
  stated end goal ("ELO exceeds all human masters") is NOT measurable
  here: there are no human master opponents or games in this
  environment. Progress is reported against the internal ladder and
  head-to-head vs the search-based Master engine; external validation
  (rated games vs humans or strong engines) is needed for the
  human-master bar.
- **A20 Pro speed numbers come only from on-device measurement.** There
  is no Apple hardware on the training VM. CPU numbers measured here are
  real; on-device numbers come only from Yudong running the benchmark
  harness (`research/iphone-benchmark.md`).
- Exact numbers everywhere: game counts, W/L/D, 95% CIs. Never rounded up.

## The "better" bar (defined BEFORE the first eval — batch 1)

Version vN is "better" than vN-1, and may be committed to
origin/master, iff ALL of these hold:

1. **Head-to-head:** >= 40 games vs vN-1, alternating colors, win rate
   (draws count as half a win) > 0.50 with the Wilson 95% CI lower
   bound > 0.50.
2. **Ladder Elo up:** internal-ladder Elo estimate strictly higher than
   vN-1's, measured with identical game counts and opponents.
3. **No tactical regression:** tactical suite pass rate >= vN-1's.
4. **All test suites green** (ml pytest, npm test).

If the bar is not met, the version is recorded in `VERSIONS.md` as
rejected and is NOT pushed.

> **Future consideration (not a batch-1 change):** iteration 2 found
> that v1 beats the baseline clearly under swap rules (0.635, CI
> excludes 0.50) while the normal-rules head-to-head was 0.55 with
> a CI including 0.50 — the swap rule removes the empty-board
> opening phase and the first-player confound. A future batch may
> adopt swap-rule head-to-head (balanced placement, fixed rules)
> as the bar's clause 1, decided and recorded BEFORE that batch's
> first eval. The batch-1 bar above stands unchanged.

## Layout

- `VERSIONS.md` — version log: version, Elo, head-to-head, what changed, commit SHA.
- `iterations/iter-N.md` — per-iteration log: self-play volume, loss analysis, hypotheses, improvement made, eval numbers, bar verdict.
- `iterations/swap-rule.md` — swap-rule interpretation note.
- `iphone-benchmark.md` — instructions for measuring on-device speed on iPhone.
