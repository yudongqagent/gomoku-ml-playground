# ML batch 1 plan — distill the rollout leaf, complete the 10k, learn the prior

## Starting point
- The program now runs two tracks (see `research/TRACKS.md`). This is
  the ML track's first batch under its own bar.
- **ML champion and reference: v8** (v6's PUCT MCTS + learned value
  leaf; ResNet 16ch/2 blocks trained on the B1 4,350-game dataset,
  α=0.25) — ladder Elo 0.0, vs v0 0.545, vs v1 0.48, tactical 20/20 +
  29/29. It is the best net-using version by both Elo and vs-v0.
- **ML track's own bar** (same 4-clause shape, reference v8):
  1. Head-to-head vs v8, ≥100 games, alternating colours, win rate
     (draws = half) > 0.50 with Wilson 95% CI lower bound > 0.50.
  2. Ladder Elo strictly higher than v8's 0.0 at identical game
     counts/opponents (20 games/opponent: random, heuristic, mcts).
  3. Tactical pass rate ≥ v8's (20/20 old suite + 29/29 hard suite).
  4. All test suites green (ml pytest, npm test, tsc).
- Every version also plays 100 games vs v7 and 100 games vs v0 (the
  heuristic track's reference and the ladder anchor), plus a 100-game
  swap match vs v7 (tempo judge); record non-transitivity prominently.
- `ml/gomoku_ml/baseline.py` stays FROZEN (ladder anchor). Kernel
  re-sync (NumPy/WebGPU/Metal) only when a version passes the bar.

## Candidates, in priority order
1. **H1 — distill the rollout leaf into a value net.** The rollout
   leaf (v16/v18) is the strongest value signal in the program, and
   the value signal is the binding constraint (v15: the starved-save
   problem is partly a value problem; v13: White's losses are
   beyond-horizon value errors). A net that learns the rollout leaf's
   values could match its value quality at much lower inference cost
   — and might generalise beyond the rollout's 2-ply horizon.
   Method: (a) generate positions from v7 self-play (fast:
   ~1,073 games/hr at 4 workers/50 sims, B1 measured) and from v18
   self-play (slower, stronger distribution); (b) label each sampled
   position with the v18 rollout-leaf value — a single leaf
   evaluation per position, much cheaper than full rollout-leaf
   games; (c) train a value net (start with v8's ResNet 16ch/2
   blocks; go larger only if the data supports it) to predict the
   rollout value; (d) use the trained net as the MCTS leaf (v8-style,
   α weight swept) and run the full battery. Measure the leaf cost
   per call FIRST, then size the dataset to the time budget (target
   ~100k positions, adjust to fit ~6h). Pre-register the dataset
   size, sampling mix, architecture, training config, and α sweep
   before any eval.
2. **H2 — complete the 10k dataset and retrain.** The staged plan's
   next step: 5,650 games remain of the 10k v7-self-play dataset
   (~5.3–7.8h at the measured B1 rate; existing 4,350 games /
   185,264 samples at
   `~/workspace/projects/gomoku-b3-b1/ml/runs/b3-b1-10k/data/`).
   Retrain the v8-style value net on the full 10k with the α lesson
   applied (sweep α; do not assume full weight). Gated on Elo
   improvement, as the staged plan requires.
3. **H3 — learned policy prior from strong self-play.** Train a
   policy net on v18 (or v16) self-play move distributions and use
   it as the MCTS prior, replacing the v1 heuristic prior. The v1
   prior is sharp and load-bearing (v13: flattening is catastrophic),
   so the learned prior must preserve sharpness — pre-register the
   temperature/sharpness handling before any eval. Most ambitious
   (AlphaZero-style policy+value); do it only if H1/H2 leave room.

Max 3 versions, each pre-registered in its iteration log BEFORE any
eval. Version numbers continue the shared numbering after batch 8's
(v21–v23) — confirm the next numbers from `research/VERSIONS.md` at
start (expect v24–v26).

## Standing lessons — ML track
- **Distribution mismatch** (v8/v11): each net was at chance on the
  other's holdout (0.50–0.52). Fixing the distribution alone did NOT
  fix strength. Sample positions broadly (v7 self-play, v18
  self-play, random playouts) — the net sees positions from its own
  search at inference, a different distribution again.
- **Optimizer's curse persists in-distribution** (v11): a net that
  predicts well can still play badly when MCTS exploits its errors.
- **α weight matters** (v8: full weight worse, α=0.25 best; v11:
  α=0.125 best). Always sweep α; never assume full weight.
- **More weak-self-play data is not the fix by itself** (v8: the
  4,350-game net generalised — holdout 0.7165 — but played worse at
  full weight). Data quality (stronger games, better labels) matters
  more than data quantity at this scale.
- **Small-sample smokes mislead** (v8's 0.60 smoke was a first-20
  fluke; v11's sweep 0.575 vs the 100-game 0.435). The 100-game
  battery is the verdict.
- **The value signal is the binding constraint** (v15, v13) — that is
  why H1 distills the strongest value signal we have.
- **q SIGN:** `child.q` is from the OPPONENT's perspective
  (`_select_child` exploits `-child.q`). Our-perspective q is
  `-child.q`.

## Method (carried over)
- The fast 10-game loop stays the finding method: play 10 games, read
  every move, find each game's losing move (solver/deeper-search
  verified, never guessed — for rollout-leaf engines, playouts or
  exact computation, not the v1 oracle alone), patch, regression-test,
  repeat. For ML versions the "patch" may be a data, architecture, or
  training change, pre-registered like any other.
- Fast loops run against v0 as well as v7 (established practice).
- The 4-clause ML bar stays the validation gate.

## Merge discipline (batch-3 lesson)
- Run the full suite PLUS targeted per-version tests after every merge.

## Constraints
- Branch `research/ml-batch-1` from master (`bf866f7`). Commit this
  plan first.
- Use your own git worktree (e.g. `~/workspace/projects/gomoku-ml1/`).
- v18's rollout leaf lives on `research/batch-7`, not on master —
  cherry-pick or port the leaf code (`kernel_v1.py` + the rollout
  leaf) for the distillation labels; do not re-derive. The v1 kernel
  is exact and reusable (0 mismatches).
- v8's checkpoint (`ml/runs/b3-h2/value-net.pt`) is gitignored data
  in the main worktree — copy it into your worktree; a fresh clone
  cannot run v8.
- Training runs on the CPU server. Small ResNets are feasible there.
  The Colab path exists but needs a phone tap per session for Drive
  persistence — do NOT depend on it (Yudong runs nothing).
- No cloud GPU spend, no money, no wallets, no releases, no Colab
  driving, no Space edits. Yudong runs nothing.
- Push the branch when done. Push to master ONLY if a version passes
  all 4 clauses of the ML bar, with the same plain-merge approach as
  batch 3.
