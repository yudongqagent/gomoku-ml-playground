# Batch 3 direction change — 2026-11-13 ~23:00 PST (from Yudong)

Recorded as an addendum; the pre-registered plan
(`research/batch-3-plan.md`) stands unchanged as the H1/H2/H3
direction, and the 4-clause "better" bar is UNCHANGED as the
validation gate. The METHOD changes, and a workstream is added.
Directed mid-batch, while the H1 (v7) and H2 (v8) workstreams were
already running; both were redirected, and B1 was spawned as a
third workstream (branch `research/batch-3-b1`).

## 1. Fast heuristic loop (replaces statistics-first weakness-finding)

Play 10 games against an opponent that produces losses to learn
from (v0, the MCTS ladder opponent, or the current version's
mirror — the matchup that exposes real weaknesses). Read EVERY
move of every game. For each game, find THE losing move: the
specific move where the game turned from drawable/winning to
lost, verified with the solver or a deeper search — never guessed.
Patch the algorithm so that, replayed to that critical position,
it now plays the perfect move, and record each critical position
+ correct move as a regression test the patched engine must pass.
Then play 10 more games and repeat. Think like a human master:
find weak POINTS and weak STRATEGIES — recurring structural
mishandling (specific threat shapes, quiet positions, tempo,
colour-specific patterns) — not just aggregate counts. Report
each iteration concretely: the game, the losing move, the perfect
move instead with the verification behind it, the patch, and the
replay result.

Bound (coordinator): max 4 fast-loop iterations (40 games) per
version before validation, or stop when an iteration finds no new
weakness class. Patches must fix the weakness (the structural
cause), not the single position — the regression test guards the
position; the patch must generalise. The fast loop generates
patches; the 4-clause bar (100-game matches + ladder Elo) still
validates each patch before it counts as "better" — a patch that
fixes 10 games but loses Elo is not progress. The bar battery
must run on the FINAL patched engine; any earlier battery numbers
are intermediate (pre-patch) results.

## 2. B1 parallel self-play (new workstream)

Multiprocessing on top of B2's batched leaf eval (per-worker
shard dirs + merge, as scoped in batch 2's notes). Measure
games/hour at 50 sims before/after on this VM and report the
MEASURED numbers. State plainly what the ceiling is here (4 CPUs)
vs what tens of thousands of games per batch would actually
require (many parallel workers + batched inference + a faster
engine — today the GPU sits idle most of the time because the
Python tree search on few cores is the bottleneck, not the net
eval). Staged data goals: 10,000 games at 50 sims -> train ->
full eval; 50,000 -> eval; 100,000+ -> eval; each stage only
proceeds if Elo improves. The quality bar is the project's "good
level" bar (>=50% vs heuristic-MCTS as both colours over >=40
games AND >=80% tactical, plus beating the 1-ply heuristic). Do
NOT promise Elo numbers at game counts not measured — the staged
evals are what find the curve. H2's learned value function uses
this data path.
