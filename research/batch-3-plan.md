# Auto-research batch 3 — plan (pre-registered)

Batch 2 showed the search direction has hit its ceiling on this ladder:
v4 (2-ply minimax), v5 (terminal-aware leaf) and v6 (MCTS with heuristic
prior) all defend near-perfectly and all lose to the frozen v0 anchor.
Defence is solved — 166/166 v6 losses came via open four, with 0 missed
blocks and 1 clean missed prevention in 1,061 threatened turns — and the
gap is attack: the search versions lose the open-four race because their
leaves provide no attack gradient. v1 (Elo 70.4) and v3 (107.5), both
1-ply scorers, remain the strongest versions.

Batch 3 changes the EVALUATION, not the search.

## Hypotheses (in order)

- **H1 — MCTS leaf with an attack gradient.** v6's terminal-only leaf
  values quiet positions at 0, so MCTS has no reason to prefer moves
  that build threats. Build a leaf that scores threat breadth / attack
  potential in quiet positions (e.g. count of live open-four creators,
  double-threat potential, weighted by the portable kernel's per-cell
  attack scores). The leaf must be cheap enough for 100-sim MCTS:
  accelerate it with the portable kernel (NumPy path) and MEASURE the
  per-position cost before committing to it — do not assert a speedup
  number from batch 2's notes without re-measuring it here.
- **H2 — learned value function from self-play.** Train a small value
  net on self-play game outcomes (B2's batched leaf-eval path from
  batch 2 makes CPU data generation ~2x faster) and use it as the MCTS
  leaf / evaluation, replacing the hand-built leaf. Keep the net small
  (the smoke ResNet 16ch/2 blocks is the starting point) and the data
  budget explicit; record games generated, training time, and the
  value net's own hold-out accuracy before any match.
- **H3 — attack-first search.** Prioritise creating threats in quiet
  positions: candidate ordering / search bias toward threat-creating
  moves (open-three makers, double-threat setups) when no forced move
  exists. Only pursued if H1/H2 stall; it is the smallest change and
  the most likely to be noise.

## Rules (unchanged from batch 1/2)

- The 4-clause "better" bar in `research/README.md` is unchanged and
  decides pass/fail for every version. Reference version is v1:
  clause 1 is head-to-head vs v1 (100+ games, alternating colours);
  clause 2 is ladder Elo strictly higher than v1's 70.4 at identical
  game counts/opponents. v3's Elo 107.5 is the current best — beating
  it is the stronger claim and must be recorded as such, but the bar
  itself is vs v1. Clause 3 uses the hard tactical suite (29
  positions, `tactical_positions_hard.json`); reference is v1's 12/29.
  v6's 29/29 is the tactical ceiling to note, not the bar.
- Every version also plays a 100-game match vs v0 (the frozen anchor)
  and a 100-game swap match vs v1 under the tempo judge (now the
  harness default). Batch 2 showed a version can beat v1 and lose to
  v0 — that non-transitivity must be recorded prominently in
  `VERSIONS.md` and the iteration log, not buried.
- Standard swap (RIF Art. 2.4), freestyle win (>= 5). No Swap2.
- Max 3 versions. Each gets a pre-registration note in its iteration
  log BEFORE any eval, 100+ games per match, a loss analysis with
  worked games (the losing move, the best move instead verified by
  solver or deeper search, and exactly what changed in the algorithm
  because of it), and a `VERSIONS.md` entry.
- `ml/gomoku_ml/baseline.py` is FROZEN (ladder anchor) — never modify
  it. New versions live in versioned modules.
- Kernel re-sync (NumPy/WebGPU/Metal) only if a version passes the bar.
- No cloud GPU spend, no money, no wallets, no GitHub releases, no
  Colab driving, no Space edits. Yudong runs nothing. No unbounded
  loop: H1/H2/H3 + max 3 versions, then report.

## Branch and reporting

- Work on branch `research/batch-3`, created from `research/batch-2`.
  Commit this plan file itself first.
- Push to master ONLY if a version passes all 4 clauses; otherwise
  everything stays on `research/batch-3`.
- Final report: per-version numbers (h2h, ladder, tactical, swap),
  which clauses passed/failed, the loss learnings, and what the next
  batch should try. State plainly what was measured vs what is
  unmeasured (especially any GPU claim — there is no GPU here).
