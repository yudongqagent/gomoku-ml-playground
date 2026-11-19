# Batch 11 plan — setup-signature mining, leaf-blind patch, White forensics; ladder re-basing

## Starting point
- The program now runs two tracks (see `research/TRACKS.md`). This is
  the heuristic track's batch after ML batch 3.
- **Bar reference is v27** (Elo 636.4, batch 9). The bar is unchanged
  in shape, re-anchored on v27: (1) h2h vs v27 ≥100 games, win rate
  > 0.50 with Wilson 95% CI lower bound > 0.50; (2) ladder Elo >
  636.4; (3) tactical ≥ 20/20 + 29/29 on the corrected suite; (4)
  all test suites green.
- **Strength champion is v18** (Elo 636.4). **Champion rule v2
  (direct challenge):** a version that passes the bar plays a
  100-game match against the current champion (v18); it takes the
  title if it wins (CI lower > 0.50) AND its ladder Elo beats
  636.4 (automatic for a bar-passer). Only run the challenge match
  for a bar-passer.
- **Versions v39–v41** (ML batch 3 owns v36–v38; confirm the next
  numbers from `research/VERSIONS.md` at start).
- The tactical suite is the CORRECTED suite (batch 9 added (9,7) to
  winning-quiet-setup-2's expected list). All batch-11 versions are
  assessed against it.
- Master contains the v18/v23/v27/v28/v29 code (batch-9 merge) plus
  the ladder auto-resume (batch 10 tooling, cherry-picked) and
  per-game seeding (ML batch 3 tooling, cherry-picked) — the match
  drivers now seed per game (`seed + game_index`) and audit
  uniqueness on every leg. No cherry-picking needed for the base
  engines or the drivers.

## Ladder re-basing (pre-registered scheme change)
- **40 games per opponent per leg** (up from 20). Two reasons:
  (1) batch 10's top saturation — v34's 20-0 heuristic leg yields
  no finite Elo estimate, so the ladder can no longer rank versions
  that beat the anchor outright; a 40-0 leg is far less likely
  (P ≈ 0.13 even at a true 0.95 rate). (2) ML batch 3's v37 —
  the 20-game leg's ±~180 Elo noise decided clause 2 (9-11 vs the
  heuristic against 0.585 over 100 games vs the same opponent;
  P(≤9 wins in 20) ≈ 0.26 under the battery rate). More games
  narrow the intervals without changing the Elo scale — the point
  estimates remain comparable to historical versions (same anchors,
  same formula); only the intervals differ (narrower).
- **Leg-result reading for saturation (pre-registered):** if a leg
  saturates (a perfect score, Elo estimate None), clause 2 is
  judged on the leg result — a 40-0 leg strictly dominates a
  39-0-1 leg (the v34 precedent, iter-34 §3.5). The alternative
  readings would make clause 2 impossible to pass by construction.
- The ladder driver has auto-resume (`eval_version.py --out`, on
  master) — use it; every ladder JSON must be a committed artifact.

## Candidates, in priority order
1. **H1 (v39) — setup-signature mining + patch.** Batch 10 H3's
   forensics found 1 distinct setup position (v0-g4/g32, a
   transpose) where v27's leaf is blind. Mine more instances of
   the signature from v27's games (played and self-play), then
   patch the leaf or search to handle it. Pre-register the
   signature definition and the patch before any eval. If the
   mining finds fewer than ~5 distinct instances, the signature is
   too rare to be a strength lever — say so and fall back to the
   leaf-blind direction (H2's area, with a distinct mechanism).
2. **H2 (v40) — leaf-blind patch.** Batch 10 H2's flavour: the leaf
   ranks the unsafe move best — a LEAF problem, not a search
   problem (distinct from H1's setup signature, which is a
   pattern the leaf doesn't see at all). Diagnose the leaf's
   mis-ranking on the dumped instances (H2's confident misses:
   unique visit leader, leaf ranks the unsafe move best), then
   patch the leaf's threat evaluation. Pre-register the diagnosis
   and the patch before any eval.
3. **H3 (v41) — White-side forensics → targeted patch.** White
   remains the persistent gap across all batch-10 versions (v33
   White 0.42, v35 White 13W-35L-2D vs v27). Read v27's White
   losses specifically (from v35's forensics and fresh White
   games), find any shared mechanism, patch it. Note the CLOSED
   levers: White search/prior/book (batch 5) and more sims (v10)
   are NOT the answer — the mechanism must be a value/leaf or
   horizon effect, verified by playouts or exact computation.
   If the forensics finds nothing patchable, fall back to the
   cheaper-VCF-check idea (batch 10 H3's unused fallback: a
   depth/candidate-limited VCF check for v29 that keeps the
   tactical fixes while reducing the ~1.65s/move cost).

Max 3 versions, each pre-registered in its iteration log BEFORE any
eval.

## Standing lessons — heuristic track
- **q SIGN:** `child.q` is from the OPPONENT's perspective
  (`_select_child` exploits `-child.q`). Our-perspective q is
  `-child.q`.
- **CLOSED levers** — do not reopen without a genuinely new
  mechanism: q-based root decisions as a general mechanism (pure q
  0.20); root-decision for strength (±10/100); the tie-break lever
  (doubly bounded: v23 unconditional hurt vs v0, v27's threshold
  fixed it, v34's conditioning adds nothing); White search/prior/
  book (100 sims optimum; prior-temperature flattening catastrophic);
  exploration tuning (prior floor inert); gap-three demotion in ALL
  FOUR combinations (v14 0.60, v20 0.585, v22 0.655, v28 0.545 vs
  v0 — priced and closed); the blend as a general mechanism (doubly
  bounded: v19, v21); combination-stacking (v17); more sims is NOT
  monotonically better (v10 White 0W-50L at 200 sims); VCF-prevention
  for the v27 family (v35 — the missed-prevention rate was only
  ~1.3% of threatened turns); the defensive check is priced in all
  three forms (prior-pick v29, search-pick v33, search-first v35).
- The leaf is not the binding constraint (v11, v14) — EXCEPT the
  v1-kernel leaf's behaviour against greedy play IS the binding
  constraint for v1-leaf versions (v17, v20), and the leaf's blind
  spots ARE the target of this batch (H1/H2).
- The sharp prior is load-bearing at 100 sims. Defence is solved
  (all pooled losses via open four, 0 missed blocks) — new versions
  must keep it solved. Clean missed preventions are a real
  v27-family property (v34: 6/468 threatened turns, all as White;
  v35 fixed them with search-first at a cost of ~0 games) — report
  the rate for every version.
- Small-sample smokes mislead (the 100-game battery is the verdict);
  single-position diagnostics cannot validate a global search change;
  the v1 walk-back oracle shares the argmax-of-sum flaw (iter-18) —
  every nominated save needs playouts or exact computation.
- **Fast loops run against v0 as well as v27** (the bar reference is
  v27 now, not v7).

## Method (carried over)
- The fast 10-game loop stays the finding method: play 10 games, read
  every move, find each game's losing move (solver/deeper-search
  verified, never guessed — for rollout-leaf engines, playouts or
  exact computation, not the v1 oracle alone), patch, regression-test,
  repeat.
- The 4-clause bar stays the validation gate.
- All match drivers auto-resume from partial `--out` files (ML batch
  1 H1's patch, on master) and now seed per game (`seed +
  game_index`, ML batch 3 tooling, on master) with a uniqueness
  audit on every leg — use that durability and hygiene; batteries
  must survive VM replacements, and every leg reports its
  distinct-game count and de-dup rate.

## Merge discipline (batch-3 lesson)
- Run the full suite PLUS targeted per-version tests after every merge.
- Watch for the recurring traps: the dead-code root-selection trap
  (batch 8, batch 9 H1 — `_pick_root_move` vs `_choose_root_child`),
  the shared-tail trap (batch 8, batch 9 H1/H3), and the silent
  auto-merge breakage on `ValueMCTS.__init__` (ML batch 1, ML batch
  2 — duplicate parameters with NO conflict flag).

## Constraints
- Branch `research/batch-11` from master (current HEAD — verify with
  git rev-parse before branching). Commit this plan first.
- Use your own git worktree (e.g. `~/workspace/projects/gomoku-b11/`).
- `ml/gomoku_ml/baseline.py` stays FROZEN (ladder anchor). Kernel
  re-sync (NumPy/WebGPU/Metal) only when a version passes the bar.
- No cloud GPU spend, no money, no wallets, no releases, no Colab
  driving, no Space edits. Yudong runs nothing.
- Push the branch when done. Push to master ONLY if a version passes
  all 4 clauses of the bar, with the same plain-merge approach as
  batch 3.
- The daily cross-track cron runs each morning ~08:37 PST; do not
  touch it or its output files. The daily pairing is unchanged
  (v18 vs v8, swap rules).
