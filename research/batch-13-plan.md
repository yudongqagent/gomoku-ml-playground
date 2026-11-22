# Batch 13 plan — gradient-preserving backup, threat-space search, 2-ply rollout leaf

## Starting point
- The program now runs two tracks (see `research/TRACKS.md`). This is
  the heuristic track's batch after ML batch 5.
- **Heuristic bar reference is v27** (Elo 636.4; batch 9 re-anchored
  the bar). **Strength champion is v18** (Elo 636.4, h2h vs v7
  0.775, vs v0 0.97, tactical 25/29 original / 26/29 corrected —
  has NOT passed the full bar; lives on `research/batch-7` and on
  master via the batch-9 merge).
- **Champion rule v2** (TRACKS.md): a version that passes the bar
  plays a 100-game DIRECT CHALLENGE match against the current
  champion (v18); it takes the title if it wins the match (win
  rate > 0.50 with Wilson 95% CI lower bound > 0.50) AND its
  ladder Elo beats 636.4 (automatic for a bar-passer). Only run
  the challenge match for a bar-passer.
- **The 4-clause bar** (`research/README.md`): (1) h2h vs v27
  ≥100 games, win rate > 0.50 with Wilson 95% CI lower bound >
  0.50; (2) ladder Elo > 636.4; (3) tactical ≥ 20/20 + 29/29 on
  the corrected suite; (4) all test suites green.
- **Versions v51–v53** (ML batch 5 owns v48–v50; confirm the next
  numbers from `research/VERSIONS.md` at start).
- The tactical suite is the CORRECTED suite (batch 9 added (9,7)
  to winning-quiet-setup-2's expected list). Re-verify v27's
  tactical on it first.
- **The ladder is 40 games per opponent per leg** (batch 11's
  re-basing — it worked: no saturation, intervals narrowed).
  Point estimates remain comparable to historical versions (same
  anchors, same formula). The ladder driver has auto-resume
  (`eval_version.py --out`, on master) — use it; every ladder
  JSON must be a committed artifact.
- **Elo draw-count sensitivity (batch 12 finding):** v45's Elo
  759.1 vs v27's 636.4 came from one fewer draw in the heuristic
  leg (39-0-1 vs 38-0-2) with identical play — a 123-point swing
  from a single draw. Elo point estimates at the top must be
  read with the leg composition alongside, not as a precise
  ranking. Do NOT claim a "highest Elo ever" from a single draw.
  If a version's clause-2 pass rests on a draw-count difference,
  say so explicitly in the iteration log and the report.
- The match drivers now seed per game (`seed + game_index`) with a
  uniqueness audit on every leg (ML batch 3 tooling, on master).

## Candidates, in priority order
1. **H1 (v51) — a backup rule that preserves a gradient across
   all-−1.0 subtrees.** Batch 12's v47 (pure minimax backup) was
   catastrophic (0.245 vs v27, tactical 21/29) because of
   **q-collapse**: in creator-rich positions nearly every subtree
   contains a visited −1.0 line, so minimax sets q = −1.0 for
   almost every root child (g13: ALL 74 visited children at
   −1.0), destroying the "how it loses" gradient the mean backup
   preserved (prevention −0.75 vs non-prevention −1.0). With all
   q gaps exactly 0, v27's τ=0.05 tie-break can never fire; visits
   follow the v1 prior, which prefers attacks (~0.10) over quiet
   preventions (~0.032). The narrow exception: a backup rule
   that preserves a gradient across moves whose subtrees all
   contain a −1.0 line — e.g. a **soft-min** (a smooth minimum
   that weights the worst lines heavily but keeps the others'
   contribution) or a **second-moment backup** (back up both the
   mean and the variance, and rank by a risk-adjusted value).
   Pre-register the rule and the mechanism check (does it
   preserve the prevention/non-prevention gradient that v47's
   minimax destroyed, measured on prev-v1v0s-g13 and
   prev-v1bl-g2?) before any eval. This is a NEW mechanism
   requiring its own pre-registration, not a retry of minimax.
2. **H2 (v52) — threat-space search (TSS) at the root.** Search
   only forcing moves (fours, open threes, double threats) to a
   deeper horizon than the MCTS, and play the winning forcing
   sequence if one exists. This is a genuinely new mechanism,
   distinct from v29's VCF-at-root (which searches only fours)
   and from the MCTS (which searches all moves). TSS is the
   classic Gomoku solver technique: by restricting the move set
   to forcing moves, it can search much deeper than the 100-sim
   MCTS and find deep forcing wins the MCTS can't see. Pre-register
   the move set, the depth, and the mechanism check (does it find
   a forcing win the MCTS misses on a set of test positions?)
   before any eval. Measure the cost first (the move set is
   smaller than the full board, but the depth is larger).
3. **H3 (v53) — 2-ply rollout leaf (on v18/v27).** v18's rollout
   leaf is 1-ply (one random move, then evaluate). The ML
   track's label-depth finding (ML batch 5: 2-ply labels are the
   strongest labelling; 1-ply 0.515, 2-ply 0.75, 3-ply 0.67,
   4-ply 0.475) suggests 2-ply is a sweet spot. A 2-ply rollout
   leaf is untested on the heuristic track (v40 refuted the
   4-ply rollout, which hallucinates a win on a losing safe
   move — but 2-ply is between 1-ply (works) and 4-ply
   (hallucinates)). Pre-register the rollout depth and the cost
   budget before any eval. Measure the cost first (a 2-ply
   rollout roughly doubles the leaf cost).

Max 3 versions, each pre-registered in its iteration log BEFORE any
eval.

## Standing lessons — heuristic track
- **CLOSED levers** — do not reopen without a genuinely new
  mechanism: q-based root decisions as a general mechanism (pure
  q 0.20; v9 tie-by-q 0.45 on v0/v1 leaf) — EXCEPT v23's q-aware
  visit-tie break and v27's threshold refinement, which are the
  narrow exceptions (and batch 10 closed the tie-break lever
  entirely); root-decision for strength (v9/v12 bracket ±10/100);
  White search/prior/book; exploration tuning (v15); prior-
  temperature flattening (catastrophic: 0.775 → 0.05 at T=3.0);
  gap-three demotion in ALL FOUR combinations (batch 9: v14 0.60,
  v20 0.585, v22 0.655, v28 0.545 vs v0 — CLOSED as a strength
  lever); the blend as a general mechanism (v19, v21 — doubly
  bounded); combination-stacking (v17); assuming more sims helps
  (v10 collapsed at 200 sims); the leaf as binding constraint
  (v11, v14) — EXCEPT the v1-kernel leaf's behaviour against
  greedy play IS the binding constraint for v1-leaf versions
  (v17, v20); VCF-prevention for the v27 family (v35 — defensive
  check priced in all three forms: prior-pick v29, search-pick
  v33, search-first v35; re-confirmed v45, batch 12 — the hole
  fix found 0 hole instances in 6,258 turns); the tie-break
  lever (v34 — doubly bounded); setup-prevention in this form
  (v39 — the check fires ~6% of moves while the mechanism
  matters in ~0.3% of turns); the leaf-blind direction (v40 —
  the remaining mis-rankings are averaging/visit-allocation,
  i.e. search problems); the White gap as a single-flaw framing
  (v41 — the gap is matchup-dependent, not a systematic
  v27-White defect; population-statistics confirmation v46,
  batch 12); both blunder-dilution fixes (v46 leaf-level
  pure-min, v47 pure minimax backup — the mean backup's
  averaging is load-bearing, not a bug).
- **Defence is solved** (all pooled losses via open four, 0
  missed blocks) and new versions must keep it solved.
- **q sign:** our-q = −child.q (`_select_child` exploits
  −child.q); q prefers the *losing* move in §2.6's corrected
  reading. Do not re-introduce the sign error.
- **The sharp prior is load-bearing at 100 sims.**
- Small-sample smokes mislead; the 100-game battery is the
  verdict. Single-position diagnostics cannot validate a global
  search change.
- **Fast loops run against v0 as well as v27.**
- The v1 walk-back oracle shares the argmax-of-sum flaw and can
  no longer verify saves alone for rollout-leaf engines — every
  nominated save needs playouts or exact computation.
- **VCF starts are NOT monotone under our move** (batch 11) —
  occupying a setup square can CREATE opponent VCF starts by
  removing one of our own defensive replies. Any future check
  must full-re-check VCF after the candidate move. An immediate
  winning move IS a VCF start but is NOT necessarily a
  four-creator — the fast VCF test must check winning moves
  first.
- Exact leaf values interact with the tie-break (v40) — future
  leaf changes must expect visit-allocation regressions; v40's
  v1bl-g2 failure is the concrete instance.

## Method (carried over)
- The fast 10-game loop stays the finding method: play 10 games, read
  every move, find each game's losing move (solver/deeper-search
  verified, never guessed), patch, regression-test, repeat.
- The 4-clause bar stays the validation gate.
- All match drivers auto-resume from partial `--out` files (ML batch
  1 H1's patch, on master) and now seed per game (`seed +
  game_index`, ML batch 3 tooling, on master) with a uniqueness
  audit on every leg — use that durability and hygiene; batteries
  must survive VM replacements, and every leg reports its
  distinct-game count and de-dup rate.

## Merge discipline (batch-3 lesson)
- Run the full suite PLUS targeted per-version tests after every merge.
- Watch for the recurring traps: the silent auto-merge breakage on
  `ValueMCTS.__init__` (ML batch 1, ML batch 2 — duplicate
  parameters with NO conflict flag; check for duplicate parameters
  before running anything), the dead-code root-selection trap
  (batch 8, batch 9), and the shared-tail trap (batch 8, batch 9,
  ML batch 5).

## Constraints
- Branch `research/batch-13` from master (current HEAD — verify
  with git rev-parse before branching). Commit this plan first.
- Use your own git worktree (e.g. `~/workspace/projects/gomoku-b13/`).
- `ml/gomoku_ml/baseline.py` stays FROZEN (ladder anchor). Kernel
  re-sync (NumPy/WebGPU/Metal) only when a version passes the bar.
- No cloud GPU spend, no money, no wallets, no releases, no Colab
  driving, no Space edits. Yudong runs nothing.
- Push the branch when done. Push to master ONLY if a version passes
  all 4 clauses of the bar, with the same plain-merge approach as
  batch 3. A bar pass changes the heuristic champion (via the
  direct-challenge rule) and the daily cross-track match — say so
  explicitly in the report.
- The daily cross-track cron runs each morning ~08:37 PST; do not
  touch it or its output files. The daily pairing is v18 vs v44
  (swap rules), run in the main worktree (see TRACKS.md).
