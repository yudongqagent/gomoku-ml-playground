# Batch 12 plan — v35's VCF hole, White statistics, blunder dilution

## Starting point
- The program now runs two tracks (see `research/TRACKS.md`). This is
  the heuristic track's batch after ML batch 4 (which produced the
  first ML bar pass: v44 is the new ML champion).
- **Bar reference is v27** (ladder Elo 636.4, tactical 20/20 +
  29/29 on the corrected suite, swap vs v7 0.775). **Strength
  champion is v18** (ladder Elo 636.4, h2h vs v7 0.775, vs v0
  0.97, tactical 26/29 corrected — has NOT passed the full bar;
  lives on master via the batch-9 merge). **Champion rule v2
  (direct challenge, in TRACKS.md):** a version that passes the
  bar plays a 100-game match against v18; it takes the title if
  it wins the match (win rate > 0.50 with Wilson 95% CI lower
  bound > 0.50) AND its ladder Elo beats 636.4 (automatic for a
  bar-passer). Only run the challenge match for a bar-passer.
- **Versions v45–v47** (ML batch 4 owns v42–v44; confirm the next
  numbers from `research/VERSIONS.md` at start).
- The tactical suite is the CORRECTED suite (batch 9 added (9,7) to
  winning-quiet-setup-2's expected list). Re-verify v27's tactical
  on it first (adding an expected move can only raise pass rates,
  but verify).
- **The ladder is 40 games per opponent per leg** (batch 11's
  re-basing — it worked: no saturation, intervals narrowed;
  v39/v40 posted identical legs). Point estimates remain
  comparable to historical versions (same anchors, same formula).
  **Leg-result reading for saturation (pre-registered):** if a leg
  saturates (perfect score, Elo estimate None), clause 2 is judged
  on the leg result — a 40-0 leg strictly dominates a 39-0-1 leg
  (the v34 precedent, iter-34 §3.5). The ladder driver has
  auto-resume (`eval_version.py --out`, on master) — use it;
  every ladder JSON must be a committed artifact.
- The match drivers seed per game (`seed + game_index`) with a
  uniqueness audit on every leg (ML batch 3 tooling, on master).

## Candidates, in priority order
1. **H1 (v45) — fix v35's `_leaves_opp_vcf` hole, then measure.**
   Batch 11 H1's finding: **VCF starts are NOT monotone under our
   move** — occupying a setup square can CREATE opponent VCF starts
   by removing one of our own defensive replies (batch-v33-vs-v27
   g0 ply 29). v35's `_leaves_opp_vcf` checks only the
   original-starts subset, so it has this hole — a known defect in
   a merged-but-unshipped version. Fix it (full-re-check VCF after
   the candidate move), then measure whether the fixed check moves
   strength vs v27. Pre-register the fix and the measurement
   before any eval. Also note batch 11's second finding: an
   immediate winning move IS a VCF start but is NOT necessarily a
   four-creator — the fast VCF test must check winning moves
   first (the verification audit caught 2 false candidates from
   this bug; the fix is in the batch-11 branch).
2. **H2 (v46) — White-side statistics, then patch what the
   statistics point to.** Batch 11 H3's forensics found no
   patchable White mechanism by loss-by-loss reading — the "White
   gap" is matchup-dependent, not a systematic v27-White defect
   (White went 7W-2L-1D in fresh self-play). The White gap as a
   single-flaw framing is CLOSED. This version reframes it:
   collect White-side play statistics across many games
   (win-rate-by-phase, error-rate-by-ply, threat-creation rate by
   colour) and patch the mechanism the statistics point to.
   Pre-register the statistics to collect and the patch rule
   before any eval. If the statistics show no asymmetry worth
   patching, say so and fall back to the blunder-dilution
   direction (H3's area, with a distinct mechanism).
3. **H3 (v47) — blunder-dilution fix (min-aggregation at the
   leaf).** Batch 11 H2's finding: the leaf does NOT mis-rank on
   the dumped instances — it ranks the unsafe move worst. The
   mis-ranking lives in the backed-up root q / visit allocation,
   in three mechanisms. One of them, **blunder dilution** (g61,
   g43), is an aggregation problem, not a leaf problem: the
   unsafe move creates our own open-four race threats, so
   opponent blunder replies return +1 leaves that dilute the
   creator-win average — every leaf value is correct, but the
   average lies. Fix: at the leaf, aggregate the opponent's reply
   values by min (worst-case) instead of mean, or by a min-mean
   blend. Pre-register the aggregation rule before any eval.
   Caution: v0 blunders constantly, so a min-aggregation may hurt
   vs v0 — the battery (vs v0 AND vs v27) will tell. The other
   two mechanisms (visit allocation, VCF undervaluation) are
   NOT this version's remit — visit allocation touches CLOSED
   root-decision levers, and VCF undervaluation was v40's patch
   (worth ~zero at 100 sims).

Max 3 versions, each pre-registered in its iteration log BEFORE any
eval.

## Standing lessons — heuristic track
- q sign: our-q = −child.q (`child.q` is from the opponent's
  perspective).
- **CLOSED levers** — do not reopen without a genuinely new
  mechanism: q-based root decisions as a general mechanism
  (pure q 0.20; v9 tie-by-q 0.45 on v0/v1 leaf) — EXCEPT v23's
  q-aware visit-tie break and its v27 threshold refinement, which
  are the narrow exceptions; root-decision for strength (v9/v12
  bracket ±10/100); White search/prior/book (batch 5); exploration
  tuning (v15); prior-temperature flattening (catastrophic);
  gap-three demotion in ALL FOUR combinations (batch 9: v14 0.60,
  v20 0.585, v22 0.655, v28 0.545 vs v0); the blend as a general
  mechanism (v19, v21 — doubly bounded); combination-stacking
  (v17); more sims is NOT monotonically better (v10); the
  tie-break lever (v34 — doubly bounded); VCF-prevention for the
  v27 family (v35 — defensive check priced in all three forms:
  prior-pick v29, search-pick v33, search-first v35);
  setup-prevention in this form (v39 — the check fires ~6% of
  moves while the mechanism matters in ~0.3% of turns); the
  leaf-blind direction (v40 — the remaining mis-rankings are
  averaging/visit-allocation, i.e. search problems); the White
  gap as a single-flaw framing (v41).
- The leaf is not the binding constraint (v11, v14) — EXCEPT the
  v1-kernel leaf's behaviour against greedy play IS the binding
  constraint for v1-leaf versions (v17, v20), and the leaf's
  blind spots were batch 11's target (now closed).
- Sharp prior is load-bearing at 100 sims (v13's flattening
  catastrophic; v15's floor inert).
- Defence is solved: all pooled losses via open four, 0 missed
  blocks. New versions must keep it solved. **Clean missed
  preventions are a real v27-family property** (v34: 6/468
  threatened turns, 1.3%, all as White; v39: 2/399, 0.5%; v40:
  2/450, 0.44%; v41: 1/654, 0.15%) — report the rate for every
  version.
- Small-sample smokes mislead; the 100-game battery is the
  verdict. Single-position diagnostics cannot validate a global
  search change.
- The v1 walk-back oracle shares the argmax-of-sum flaw and can
  no longer verify saves alone for rollout-leaf engines — every
  nominated save needs playouts or exact computation.
- **Fast loops run against v0 as well as v27.**

## Method (carried over)
- The fast 10-game loop stays the finding method: play 10 games,
  read every move, find each game's losing move (solver/deeper-
  search verified, never guessed), patch, regression-test, repeat.
- The 4-clause bar stays the validation gate.
- All match drivers auto-resume from partial `--out` files (ML
  batch 1 H1's patch, on master) and now seed per game (`seed +
  game_index`, ML batch 3 tooling, on master) with a uniqueness
  audit on every leg — use that durability and hygiene; batteries
  must survive VM replacements, and every leg reports its
  distinct-game count and de-dup rate.

## Merge discipline (batch-3 lesson)
- Run the full suite PLUS targeted per-version tests after every merge.
- Watch for the recurring traps: the dead-code root-selection trap
  (batch 8, batch 9), the shared-tail trap (batch 8, batch 9), and
  the silent auto-merge breakage on `ValueMCTS.__init__` (ML batch
  1, ML batch 2 — duplicate parameters with NO conflict flag;
  check for duplicate parameters before running anything).

## Constraints
- Branch `research/batch-12` from master (current HEAD — verify
  with git rev-parse before branching). Commit this plan first.
- Use your own git worktree (e.g. `~/workspace/projects/gomoku-b12/`).
- `ml/gomoku_ml/baseline.py` stays FROZEN (ladder anchor). Kernel
  re-sync (NumPy/WebGPU/Metal) only when a version passes the bar.
- No cloud GPU spend, no money, no wallets, no releases, no Colab
  driving, no Space edits. Yudong runs nothing.
- Push the branch when done. Push to master ONLY if a version passes
  all 4 clauses of the bar, with the same plain-merge approach as
  batch 3. A bar pass changes the bar reference and may change the
  strength champion (via the direct challenge) — say so explicitly
  in the report.
- The daily cross-track cron runs each morning ~08:37 PST; do not
  touch it or its output files. The daily pairing is now v18 vs
  v44 (swap rules), run in the main worktree (see TRACKS.md).
