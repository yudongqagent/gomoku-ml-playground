"""Improved heuristic Gomoku engine (v1) — 5-window pattern scoring.

Same interface as :mod:`gomoku_ml.baseline` — ``score_move(board, move,
player) -> int`` and ``choose_move(board, player, rng=None) -> int`` —
so it drops into any harness that takes a move chooser. It is a NEW
module: :mod:`gomoku_ml.baseline` is the frozen ladder anchor and is
not modified.

Differences from the baseline (each small, each tested in
``ml/tests/test_scoring.py``):

1. **5-window scan (the broken-four fix).** The baseline counts only
   contiguous runs through a candidate point, so a broken four
   (``XX_XX``) is invisible to it. This engine scores every 5-cell
   window through the point (mirroring the web engine in
   ``src/ai/patterns.ts``): a move completing ``XX_XX`` scores as a
   four, and a move creating a broken four is valued as the threat it
   is.
2. **Correct four valuation.** A 5-window holding four stones of one
   colour always has a winning square — the empty cell inside the
   window — so a four scores as a four (20k) even when both outside
   cells are blocked. Only a *contiguous* four whose far outside cell
   is empty is an open four (200k, two winning squares). (The TS
   engine scores a fully-closed four-window as 0; that undervalues a
   real threat, so this engine deliberately deviates there.)
3. **Max-per-direction, not sum-over-windows.** A single contiguous
   four sits in two overlapping 5-windows; summing all windows (as the
   TS engine does) double-counts one physical pattern and would let a
   lone four masquerade as a double threat. Each direction contributes
   only its strongest window, and the four directions are summed.
4. **Double-threat bonuses.** A move creating fours in two or more
   directions is a forced win next move (the opponent can block only
   one); a move creating open threes in two or more directions forces
   a win in two. Both get explicit bonuses so they beat single
   threats of the same class.
5. **Cheaper win/block scans.** An immediate winning move must be
   adjacent to a friendly stone, so the win and block scans only check
   radius-1 candidates of the relevant colour instead of all ~200
   legal moves.
6. **Candidate restriction for scoring.** Only empty points within
   Chebyshev distance 2 of a stone are scored (centre on an empty
   board; all legal moves as a fallback if the candidate set is
   empty). A move far from all stones is never the best move in
   Gomoku, and this is what makes the engine fast enough for
   large-scale self-play.
7. **No centre preference in the score.** The baseline adds a
   Manhattan-distance term; the window scores carry the signal, and
   the empty-board centre move is handled directly in
   ``choose_move``.

Weights mirror the web engine (orders of magnitude apart so pattern
classes dominate each other): five 10,000,000; open four 200,000;
four 20,000; open three 4,000; three 400; open two 80; two 10; one
1–2. Attack and defence are summed at full value (the baseline
discounts defence to 90%).

Version: v1. Later research iterations add versions to
:data:`ENGINE_VERSIONS` (each version is a parameter set) and use
:func:`get_player` / :class:`ScoringPlayer` with ``version=N``; the
module-level :func:`score_move` / :func:`choose_move` / :class:`ScoringPlayer`
defaults stay v1 forever so old results remain reproducible and old
versions stay importable for A/B comparisons. v2 tests a defence
discount (0.9); v3 adds a 2-ply threat check (prefer a move that
leaves the opponent with no live open-four creator). v4 adds a 2-ply
minimax over the top 1-ply candidates (opponent modelled as v1;
positions evaluated by best-attack difference — see
:func:`_minimax_choose`). v5 keeps v4's search and replaces the leaf
value with a terminal-aware evaluation (see :func:`_leaf_value`):
>= 2 winning moves for a side is terminal, exactly 1 is capped at
four-value.
"""

from __future__ import annotations

import os

import numpy as np

from . import game

VERSION = 1
ENGINE_NAME = "scoring-v1"

# --- version registry -----------------------------------------------------
# Each version is a parameter set for the same engine code. v1 is the
# 5-window engine (full-value defence). Later versions change ONE
# parameter at a time so A/B comparisons stay interpretable.
ENGINE_VERSIONS: dict[int, dict] = {
    1: {"name": "scoring-v1", "defence_weight": 1.0, "threat_check": False},
    # v2 (iteration 2): attack bias — defence discounted to 0.9, the
    # value the frozen baseline uses. Tests the iteration-1
    # hypothesis (see research/iterations/iter-1.md and iter-2.md):
    # v1's full-value defence makes it a strong defender but a
    # passive attacker — it loses the race to the first open four.
    # Every decisive game in the iteration-1 batches was decided by
    # an open four, and v1's losses came fast (mean 27.5 plies) while
    # its wins ground out slowly (mean 69.5 plies).
    # REJECTED (iter-2.md): lost head-to-head to v1 (0.4050) and
    # lost ladder Elo (70.4 -> -52.5).
    2: {"name": "scoring-v2", "defence_weight": 0.9, "threat_check": False},
    # v3 (iteration 3): v1 + a 2-ply threat check. v1 never misses a
    # block and never ignores a critical threat, but it MISSES
    # PREVENTIONS: in 35 swap losses, on 13 turns (in 9 games) its
    # non-forcing move left an opponent open-four creator live when
    # a preventing alternative existed — 8.2% of the 159 threatened
    # turns where the opponent had no immediate winning move (the
    # baseline misses at the same rate, 8.6%, so the gap is real but
    # small and v3's expected gain is modest). The 1-ply scorer
    # cannot see prevention: a forcing four (20k) or a line-block
    # that kills a creator scores far less than a double open three
    # (58k), so the attacking move wins and the creator stays live.
    # v3 prefers a move that leaves the opponent with zero live
    # open-four creators (or a forcing move) whenever the 1-ply best
    # does not.
    3: {"name": "scoring-v3", "defence_weight": 1.0, "threat_check": True},
    # v4 (iteration 4, batch 2): v1 + 2-ply minimax over the top-K
    # 1-ply candidates. Batch 1's loss analysis: every decisive game
    # (35/35) was decided by an open four, and v1 loses the race to
    # the first open four (loses fast, 27.5 plies; wins slow, 69.5).
    # A 1-ply scorer can value open fours but cannot PLAN a sequence
    # to create one while denying the opponent's. v4 searches: for
    # each of the top-K 1-ply moves, it plays the opponent's v1 best
    # reply, then evaluates the resulting position by best-attack
    # difference (attack-only — the TOTAL score is colour-symmetric
    # at defence_weight=1.0, so a total-score difference would always
    # be 0; see _score_parts_flat). Unlike v3's safety filter (which
    # only avoids losing creators), minimax also FINDS attacking
    # sequences: a move that creates a double threat while the
    # opponent's reply is forced scores highly. Cost: ~8 opponent
    # replies + ~16 attack scans per move (~70 ms/call worst case).
    4: {"name": "scoring-v4", "defence_weight": 1.0, "threat_check": False,
        "minimax": True, "minimax_topk": 8},
    # v5 (iteration 5, batch 2): v4 + terminal-aware minimax leaf
    # evaluation. v4's leaf value (best-attack difference) counts a
    # single blockable winning move as -10M — the same as an open
    # four — and lets our attack compensate for the opponent's win,
    # so the minimax overrides the #1-ranked blocking move (iter-4
    # §2.3: game 1 vs v0, verified with the VCF solver; all 77
    # analysed missed preventions follow this pattern). v5 counts
    # winning moves on the leaf board: >= 2 for the opponent is a
    # terminal loss (-WIN_SCORE, no compensation); exactly 1 is
    # capped at four-value (20,000) since the player to move blocks
    # it; symmetric for our winning moves. Search unchanged.
    5: {"name": "scoring-v5", "defence_weight": 1.0, "threat_check": False,
        "minimax": True, "minimax_topk": 8, "terminal_aware": True},
    # v6 (iteration 6, batch 2): H2 — PUCT MCTS with v1 scores as
    # priors and a terminal-aware leaf value (see
    # gomoku_ml.mcts_heuristic). NOT a parameter set of the 1-ply /
    # minimax engine: the "mcts" flag makes ScoringPlayer dispatch
    # to HeuristicMCTS. The 2-ply minimax lineage hit its ceiling
    # in iter-5: the saving move in v5-vs-v0 game 1 ranked 11th-12th
    # by 1-ply score (outside the top-8), and the static leaf
    # values the BEST attack, not the BREADTH of threats. MCTS
    # explores all candidates (guided, not restricted, by priors)
    # and evaluates by searched win rates across many lines. Sims
    # 100 and c_puct 1.5 match the ladder's HeuristicMCTSPlayer, so
    # any edge comes purely from the better priors (v1 vs v0
    # scoring) and the better leaf value (terminal-aware vs crude
    # local attack). The scoring params below are vestigial for
    # this version (the MCTS uses v1's scorer for priors).
    6: {"name": "scoring-v6", "defence_weight": 1.0, "threat_check": False,
        "mcts": True, "mcts_sims": 100},
    # v7 (iteration 7, batch 3): H1 — v6's PUCT MCTS with an ATTACK
    # GRADIENT in quiet leaf positions (see
    # gomoku_ml.mcts_heuristic._gradient_leaf_value and
    # research/iterations/iter-7.md). v6's terminal-only leaf values
    # every quiet position at 0, so the search has no gradient
    # toward building threats and loses the open-four race (iter-6
    # §2: 166/166 losses via open four, 0 missed blocks). v7 keeps
    # v6's terminal values exactly and values quiet leaves by the
    # BREADTH of each side's attack (count of high-attack moves
    # from the kernel's attack-only scores, tanh-squashed) — a
    # count, not a max, because iter-5 §2.2 showed the best attack
    # misleads where threat breadth decides. The ONLY difference
    # between v6 and v7 is the quiet-leaf value: a clean A/B of the
    # gradient. Caveat (stated in iter-7.md): the kernel ports v0
    # (contiguous-run scoring), NOT v1 — the gradient is blind to
    # broken fours. Sims 100 and c_puct 1.5 unchanged.
    7: {"name": "scoring-v7", "defence_weight": 1.0, "threat_check": False,
        "mcts": True, "mcts_sims": 100, "leaf_mode": "gradient"},
    # v8 (iteration 8, batch 3): H2 — v6's PUCT MCTS with a LEARNED
    # value leaf (see gomoku_ml.mcts_value). The "value_checkpoint"
    # key makes ScoringPlayer dispatch to ValueMCTS instead of
    # HeuristicMCTS. The checkpoint path is relative to the repo
    # root (gitignored data under ml/runs/); the GOMOKU_V8_CHECKPOINT
    # env var overrides it. The terminal leaf cases stay exact
    # (+1/-1/-0.25); the net values quiet positions only.
    8: {"name": "scoring-v8", "defence_weight": 1.0, "threat_check": False,
        "mcts": True, "mcts_sims": 100,
        "value_checkpoint": "ml/runs/b3-h2/value-net.pt"},
    # v9 (iteration 9, batch 4): H1 — v7's PUCT MCTS with the ROOT
    # DECISION changed: most-visited, with ties broken by the v1
    # PRIOR instead of the seeded RNG ("tie-prior"; see
    # gomoku_ml.mcts_heuristic._choose_root_child and
    # research/iterations/iter-9.md). Everything else is v7
    # unchanged (gradient leaf, 100 sims, c_puct 1.5). Four
    # variants were pre-registered and smoked 20 games vs v7
    # (seeds 300/400): pure q 0.20 (tactical 22/29), tie-q 0.45,
    # blend 0.30 (tactical 26/29), tie-prior 0.675 — the winner.
    # The q variants failed because iter-7 §2.6's "q prefers the
    # saving move" was SIGN-INVERTED (raw child q is the
    # opponent's value; correctly signed, q prefers the LOSING
    # move — see iter-9.md §1.0). The prior is the only signal
    # that preferred the saving moves in §2.6, and it wins the
    # smoke while keeping tactical 29/29.
    9: {"name": "scoring-v9", "defence_weight": 1.0, "threat_check": False,
        "mcts": True, "mcts_sims": 100, "leaf_mode": "gradient",
        "root_decision": "tie-prior"},
    # v12 (iteration 12, batch 5): H1 — v7's PUCT MCTS with the
    # ROOT DECISION changed to "tie-prior-forcing": most-visited,
    # ties broken by forcing-ness first (moves creating a four),
    # then the v1 prior (see gomoku_ml.mcts_heuristic.
    # _choose_root_child and research/iterations/iter-12.md).
    # Targets iter-9 §2.6's failure mode: v1's 1-ply score
    # undervalues forcing counter-threats (~20k) against
    # occupying a creator (~204k), so tie-prior prefers blocking
    # one of two creators over forcing. Honest limit (iter-12
    # §1.0): the §2.6 position does NOT verify as a single losing
    # move — the rule is a hypothesis to test at match scale, not
    # a verified fix. Everything else is v7 unchanged (gradient
    # leaf, 100 sims, c_puct 1.5).
    12: {"name": "scoring-v12", "defence_weight": 1.0, "threat_check": False,
         "mcts": True, "mcts_sims": 100, "leaf_mode": "gradient",
         "root_decision": "tie-prior-forcing"},
    # v13 (iteration 13, batch 5): H2 — White focus (see
    # gomoku_ml.mcts_white / mcts_v13 and
    # research/iterations/iter-13.md). Base is v7's PUCT MCTS
    # (capped attack-gradient leaf) + v3's 2-ply threat check at
    # the root, with the White sims count chosen by the §1.2
    # sweep (100: White 0.775 vs v0, the peak; 50 -> 0.525,
    # 150 -> 0.55, 200 -> 0.00). The "white_temp" key makes
    # ScoringPlayer dispatch to V13MCTS (the White prior
    # temperature lives there): 2.0 is v7's default — the §1b
    # temperature sweep REFUTED flattening (White 0.775 at 2.0,
    # 0.05 at 3.0, 0.00 at 4.0 vs v0), so v13 ships at 2.0,
    # and the (6,8) opening book was refuted too (0.55 vs the
    # 0.775 control). Versions 9-12 are the parallel
    # batch-4/batch-5 workstreams'.
    13: {"name": "scoring-v13", "defence_weight": 1.0, "threat_check": False,
         "mcts": True, "mcts_sims": 100, "leaf_mode": "gradient",
         "white_sims": 100, "white_temp": 2.0},
    # v14 (iteration 14, batch 5): H3 — v7's PUCT MCTS with the
    # gradient leaf's input scores ported from the v0 kernel to
    # the v1 5-window scorer (see gomoku_ml.kernel_v1 and
    # research/iterations/iter-14.md). The ONLY difference from
    # v7 is the quiet-leaf gradient: same breadth form, same
    # +/-0.2 cap, but the breadth counts come from v1 attack
    # scores (broken-four-aware) with the v1 thresholds
    # (strong >= 4000, developing [80, 4000)) and the re-derived
    # tanh scale 7.5. Terminal values are identical to v7's.
    14: {"name": "scoring-v14", "defence_weight": 1.0, "threat_check": False,
         "mcts": True, "mcts_sims": 100, "leaf_mode": "gradient_v1"},
    # v15 (iteration 15, batch 6): H1 — v7's PUCT MCTS with an
    # UNNORMALIZED prior floor of 0.005 (see
    # gomoku_ml.mcts_heuristic._priors and
    # research/iterations/iter-15.md). Batch 5's recurring failure
    # was saving moves starved at 2-6 visits while the loser held
    # 7-12; no root-decision rule fixes a move the search never
    # visits. The floor (chosen by the pre-registered sweep in
    # iter-15 §1.2/§2.1: F0.005 beat C2.0 on the vs-v7 transfer
    # cells, 0.5875 vs 0.4500) raises below-floor priors to 0.005
    # WITHOUT renormalization — Step 0 measured that
    # renormalization shrinks the very moves the floor is meant to
    # lift (a 0.027 save becomes 0.020 at floor 0.01), because the
    # tail's added mass dominates with ~93 candidates. Above-floor
    # moves keep their exact v7 priors, so the sharp prior is
    # preserved where it matters; a floor is not flattening (v13's
    # temperature probe: flattening is catastrophic). Everything
    # else is v7 unchanged: 100 sims, c_puct 1.5, capped +/-0.2
    # attack-gradient leaf, win/block scans, most-visited root
    # child.
    15: {"name": "scoring-v15", "defence_weight": 1.0, "threat_check": False,
         "mcts": True, "mcts_sims": 100, "leaf_mode": "gradient",
         "prior_floor": 0.005},
    # v16 (iteration 16, batch 6): H2 — v7's PUCT MCTS with a
    # WHITE-ONLY 2-PLY ROLLOUT leaf (see
    # gomoku_ml.mcts_heuristic._rollout_leaf_value and
    # research/iterations/iter-16.md). When the engine plays White,
    # quiet leaves are valued by rolling out exactly 2 plies
    # (opponent reply + our reply, win/block/attack-argmax policy)
    # and evaluating the symmetric v1-kernel gradient at the
    # resulting position — the leaf horizon extends past the forcing
    # exchange, where White's counterplay value lives (iter-13 §2.2:
    # correctly-signed q prefers the losing block over the quiet
    # counterplay save in 4/5 White critical positions, because the
    # save's value materialises several plies out). Terminal values
    # are exact throughout the rollout. When the engine plays Black,
    # v16 IS v7 exactly, so the Black games are a built-in control.
    # Sims (100), c_puct (1.5), priors and root decision are v7
    # unchanged. The v1 kernel is v14's exact port (0 mismatches).
    # Two earlier v16 candidates were measured and refuted before
    # pre-registration (iter-16 §1.1): a White asymmetric gradient
    # (0/5 v13/v14 saves at the search level, saturates) and a
    # symmetric must-block gradient term (regressed v7 from 4/5 to
    # 1/5 saves on its own regression positions). No threat check
    # (minimal, isolable change; iter-13 measured it as worth
    # ~nothing vs v7). Versions 9-15 are the parallel batch-4/5/6
    # workstreams'.
    16: {"name": "scoring-v16", "defence_weight": 1.0, "threat_check": False,
         "mcts": True, "mcts_sims": 100,
         "leaf_mode": "rollout2_v1"},
    # v17 (iteration 17, batch 6): H3 — THE COMBINATION (see
    # gomoku_ml.mcts_v17 and research/iterations/iter-17.md).
    # v7's PUCT MCTS + v12's tie-prior-forcing root rule +
    # v14's v1-kernel gradient leaf (gradient_v1) + v13's 2-ply
    # threat check at the root. Each component passed 3/4 in
    # batch 5 and failed a different clause (v12: clause 1,
    # 0.4950; v13: clause 1, 0.5250; v14: clause 2, Elo -34.9);
    # v17 tests whether they combine to 4/4. 100 sims in both
    # colours (the sims optimum, iter-13 §2.1), c_puct 1.5,
    # sharp v1 priors. The "v17" key makes ScoringPlayer
    # dispatch to V17MCTS (a HeuristicMCTS subclass — WhiteMCTS
    # cannot carry the combination: it hard-codes
    # leaf_mode="gradient" and bypasses _choose_root_child).
    # Versions 15-16 are the parallel batch-6 workstreams'.
    17: {"name": "scoring-v17", "defence_weight": 1.0, "threat_check": False,
         "mcts": True, "mcts_sims": 100, "v17": True},
    # v18 (iteration 18, batch 7): H1 — v16's 2-PLY ROLLOUT leaf
    # applied as Black too (see gomoku_ml.mcts_heuristic and
    # research/iterations/iter-18.md). The rollout function, the v1
    # rollout policy (with open-four priority), the gradient-
    # evaluated rollout end and the v1 gradient are v16's,
    # byte-identical; only the dispatch changed (no colour gate).
    # As White, v18 IS v16 exactly, so the White games are a
    # built-in control. Sims (100), c_puct (1.5), priors and root
    # decision are v7 unchanged. No threat check (minimal, isolable
    # change). Version 17 is the parallel batch-6 H3 workstream's.
    18: {"name": "scoring-v18", "defence_weight": 1.0, "threat_check": False,
         "mcts": True, "mcts_sims": 100,
         "leaf_mode": "rollout2_v1_both"},
    # v20 (iteration 20, batch 7): H3 — v14's v1-kernel gradient
    # leaf with GAP-THREE makers demoted from the strong tier to
    # the developing tier (see gomoku_ml.mcts_heuristic
    # _gradient_v1_gap_leaf_value and research/iterations/
    # iter-20.md). The v1 scorer counts a gap three (XX_X) as an
    # open three (4006), but a gap three is killable by a single
    # opponent move occupying the gap while a true open three
    # cannot be killed in one move — and greedy 1-ply play finds
    # the gap block automatically. Gap-three breadth is phantom
    # against greedy play: it steers v14 into cluster play and
    # away from forcing moves (verified causally in iter-20 §2.0:
    # 6/6 engine-swap continuations flip a v14 Black loss to v0).
    # The v0 kernel filters true open-three makers (v0 >= 1000)
    # from gap-three makers (v0 < 1000); broken-four makers stay
    # strong (a broken four forces a block exactly like a straight
    # four). Tanh scale 5.5 (re-derived by v7's rule from the
    # demoted distribution, iter-20 §1.1); cap ±0.2 unchanged.
    # Everything else is v14 unchanged (100 sims, c_puct 1.5,
    # sharp v1 priors, plain most-visited root). Versions 18-19
    # are the parallel batch-7 workstreams'.
    20: {"name": "scoring-v20", "defence_weight": 1.0, "threat_check": False,
         "mcts": True, "mcts_sims": 100, "leaf_mode": "gradient_v1_gap"},
    # v22 (iteration 22, batch 8): H2 — v14's v1-kernel gradient
    # leaf with gap-three makers demoted ONLY in BLACK's breadth
    # (see gomoku_ml.mcts_heuristic
    # _gradient_v1_gap_black_leaf_value and research/iterations/
    # iter-22.md). v20's GLOBAL demotion traded Black for White
    # (vs v0: Black 0.72, White 0.45): gap-three breadth is
    # phantom for the initiative-holder (Black moves first and
    # holds the tempo) but signal for White's counterplay. Black's
    # breadth uses v20's demoted breadth; White's uses v14's
    # undemoted breadth. Tanh scale re-derived by v7's rule from
    # the conditional distribution (iter-22 §1.1); cap ±0.2
    # unchanged. Everything else is v14 unchanged. Version 21 is
    # the parallel batch-8 workstream's (H1, v18+blend).
    22: {"name": "scoring-v22", "defence_weight": 1.0, "threat_check": False,
         "mcts": True, "mcts_sims": 100, "leaf_mode": "gradient_v1_gap_black"},
    # v23 (iteration 23, batch 8): H3 — v18 + a Q-AWARE VISIT-TIE
    # BREAK at the root (see gomoku_ml.mcts_heuristic.HeuristicMCTS
    # ``tie_break_q`` and research/iterations/iter-23.md). When two
    # or more root children share the maximum visit count, break
    # the tie by correctly-signed q (our-q = -child.q); if q also
    # ties exactly, fall back to v18's RNG/first choice on the
    # q-tied subset. A unique visit leader is untouched. The leaf
    # (v18's rollout leaf, both colours), priors, sims (100),
    # c_puct (1.5) and win/block scans are v18 unchanged — the
    # tie-break is the ONLY change. Narrow, mechanism-specific
    # clause-3 recovery for rollout-leaf engines: v18's three hard
    # tactical failures are q-compression visit-ties where the leaf
    # KNOWS the answer (chosen -1.000 vs expected +0.250) but
    # cannot convert it into visits. This does NOT reopen the
    # CLOSED q-based root-decisions lever (pure q was catastrophic
    # at 0.20; v9's tie-by-q smoked 0.45 on the v0/v1 leaf) — on
    # the rollout leaf, q-compression makes ties far more frequent
    # and the failure mode is documented. Versions 19-22 are the
    # parallel batch-7/8 workstreams'.
    23: {"name": "scoring-v23", "defence_weight": 1.0, "threat_check": False,
         "mcts": True, "mcts_sims": 100,
         "leaf_mode": "rollout2_v1_both", "tie_break_q": True},
    # v27 (iteration 27, batch 9): H1 — v23 + a THRESHOLD on the
    # q-gap tie-break (see gomoku_ml.mcts_heuristic.HeuristicMCTS
    # ``tie_break_q_threshold`` and research/iterations/iter-27.md).
    # The tie-break fires ONLY when the top our-q exceeds the
    # second-highest DISTINCT our-q among the visit-tied children
    # by strictly more than 0.05; below the threshold the choice is
    # v18's (RNG over ALL tied children). Threshold selected by the
    # pre-registered rule (iter-27 §1.2/§1.5): both sweep values
    # (0.05, 0.10) passed the tactical gate 20/20 + 29/29 and
    # scored IDENTICAL 39W-0L-1D (0.9875) in the 40-game vs-v0
    # sweep, so the tie rule selected the smaller threshold (0.05).
    # The temporary sweep entries (90/91) were removed on
    # registration. Everything else is v23 unchanged: 100 sims,
    # c_puct 1.5, v1 priors, rollout leaf both colours.
    27: {"name": "scoring-v27", "defence_weight": 1.0, "threat_check": False,
         "mcts": True, "mcts_sims": 100,
         "leaf_mode": "rollout2_v1_both", "tie_break_q": True,
         "tie_break_q_threshold": 0.05},
    # v28 (iteration 28, batch 9): H2 — v14's v1-kernel gradient
    # leaf with gap-three makers demoted ONLY in BLACK's breadth
    # and ONLY in the OURS term (see gomoku_ml.mcts_heuristic
    # _gradient_v1_gap_ours_leaf_value and research/iterations/
    # iter-28.md). v22's Black-only demotion in BOTH terms priced
    # at +0.33 Black / -0.22 White vs v0: the +0.33 is an
    # ours-term effect (Black's own breadth), the -0.22 a
    # theirs-term effect (White under-fearing Black's threats).
    # At Black-to-move leaves v28 IS v22 (Black's own breadth
    # demoted); at White-to-move leaves v28 IS v14 (nothing
    # demoted — White fears Black's gap threes fully). Tanh scale
    # re-derived by v7's rule from the v28 breadth-difference
    # distribution, both parities pooled (iter-28 §1.1); cap ±0.2
    # unchanged. Everything else is v14 unchanged. Versions 24-26
    # are ML batch 1's, 27 and 29 the parallel batch-9
    # workstreams' (H1 q-gap threshold, H3 v22 forensics).
    28: {"name": "scoring-v28", "defence_weight": 1.0, "threat_check": False,
         "mcts": True, "mcts_sims": 100, "leaf_mode": "gradient_v1_gap_ours"},
    # v29 (iteration 29, batch 9): H3 — Phase-2 fallback (see
    # research/iterations/iter-29.md §1.2, §2.5): v18's 2-ply
    # rollout leaf in both colours PLUS a VCF-at-root check
    # (gomoku_ml.mcts_heuristic _vcf_root_move): an offensive VCF
    # start is played immediately, and a defensive VCF prevention
    # is played when the opponent holds a VCF start; otherwise the
    # engine is exactly v18. Phase 1's forensics found no
    # patchable mechanism in v22's vs-mcts collapse (0 missed
    # blocks, 0 missed forced wins, 6 horizon losses, 1
    # v22-specific root-decision miss, 5 shared search-horizon
    # misses with no exact signature), so the fallback triggered.
    # 100 sims, c_puct 1.5, sharp v1 priors, most-visited root.
    29: {"name": "scoring-v29", "defence_weight": 1.0, "threat_check": False,
         "mcts": True, "mcts_sims": 100,
         "leaf_mode": "rollout2_v1_both", "vcf_root": True},
    # v37 (iteration 37, ML batch 3): H2 — v8's PUCT MCTS with the
    # value net distilled from v18's rollout leaf with DEEPER (2-ply)
    # labels (see research/iterations/iter-37.md). The ONLY
    # differences from v8 are the checkpoint (trained on the SAME
    # 37,227 quiet positions as v24 — v7/v18 self-play + random
    # playouts, same game-level holdout — but labelled with
    # ``_rollout2_leaf_value``: the rollout-leaf value 2 plies
    # ahead under the teacher's own rollout policy, a 4-ply-horizon
    # value vs v24's 2-ply) and the swept α (``value_weight`` —
    # 0.25, the α-sweep winner at 0.75 vs v8 over 20 games; the
    # 2-ply labels are hotter than v24's, mean|v| ~0.20 vs ~0.06,
    # so the smaller shrink wins — and the α effect is almost
    # entirely a White effect: White goes 0.80 → 0.45 → 0.20 as α
    # goes 0.25 → 0.5 → 1.0, while Black stays 0.70–0.80). Search,
    # priors, sims, threat check, terminal cases: v8 unchanged, so
    # the battery is a clean A/B of the deeper-teacher net — and,
    # against v24's numbers, of 2-ply vs 1-ply labels on identical
    # positions. Registered here (ML batch 4) as the secondary
    # head-to-head opponent for v42–v44: v37 is the strongest ML
    # version ever by head-to-head (0.75 vs v8) even though it
    # failed clause 2 (ladder Elo −34.9 on a cold 20-game leg).
    # Checkpoint copied verbatim from gomoku-ml3
    # (md5 2fde492aae6d44cbe945e15bd5f1f959).
    37: {"name": "scoring-v37", "defence_weight": 1.0, "threat_check": False,
         "mcts": True, "mcts_sims": 100,
         "value_checkpoint": "ml/runs/ml3-h2/value-net.pt",
         "value_weight": 0.25},
    # v42 (iteration 42, ML batch 4): H1 — v8's PUCT MCTS with the
    # value net distilled from v18's rollout leaf on v24's recipe
    # (same 37,227 quiet positions, same 1-ply labels, same
    # architecture and training recipe) PLUS history-plane dropout
    # during training (see research/iterations/iter-42.md §1.1):
    # each training sample independently has ALL its history planes
    # (last-move one-hot + stones 2/4 plies ago) zeroed with p=0.5,
    # so the net cannot route its value judgment through history
    # features — the fix for the history-dependence that v37/v38
    # showed (v7 holdout 0.7395 -> 0.5163 / 0.5035 without history)
    # while v24's net is history-robust. The ONLY difference from
    # v24 is the augmentation, so the v42-vs-v24 comparison
    # isolates it. Search, priors, sims, threat check, terminal
    # cases: v8 unchanged. value_weight is the α-sweep winner
    # (0.5: 9W-11L-0D vs v8 over 20 games, best of {0.5, 1.0, 2.0}).
    42: {"name": "scoring-v42", "defence_weight": 1.0, "threat_check": False,
         "mcts": True, "mcts_sims": 100,
         "value_checkpoint": "ml/runs/ml4-h1/value-net.pt",
         "value_weight": 0.5},
    # v43 (iteration 43, ML batch 4): H2 — v8's PUCT MCTS with a
    # value net trained on v37's 2-ply rollout-leaf labels (same
    # 817 games / 37,227 positions as v24/v37) WITH history-plane
    # dropout (p=0.5, planes 2, 4-7 zeroed jointly per sample per
    # epoch) so the net is robust to missing history — v37's net
    # collapses without history planes (v7 holdout 0.7395 ->
    # 0.5163). See research/iterations/iter-43.md. value_weight
    # is the swept α (0.25 — the §1.5 no-signal rule: the sweep's
    # three arms were within one win of each other (10/10/11 wins),
    # so the rule picks v37's winner 0.25; see iter-43.md §5).
    43: {"name": "scoring-v43", "defence_weight": 1.0, "threat_check": False,
         "mcts": True, "mcts_sims": 100,
         "value_checkpoint": "ml/runs/ml4-h2/value-net.pt",
         "value_weight": 0.25},
    # v44 (iteration 44, ML batch 4): H3 — v8's PUCT MCTS with the
    # value net distilled from v18's rollout leaf with 3-ply labels
    # (see research/iterations/iter-44.md). The ONLY differences
    # from v37 are the checkpoint (trained on the SAME 37,227 quiet
    # positions — v7/v18 self-play + random playouts, same
    # game-level holdout — but labelled with
    # ``_rollout3_leaf_value``: the rollout-leaf value 3 plies
    # ahead under the teacher's own rollout policy, a 5-ply-horizon
    # value vs v37's 4-ply) and the swept α (``value_weight`` —
    # 0.125, the §1.4 sweep winner; the 3-ply
    # labels are hotter still than v37's, so the sweep brackets
    # below v37's winner). Search, priors, sims, threat check,
    # terminal cases: v8 unchanged, so the battery is a clean A/B
    # of the 3-ply-label net — and, against v37's numbers, of 3-ply
    # vs 2-ply labels on identical positions.
    44: {"name": "scoring-v44", "defence_weight": 1.0, "threat_check": False,
         "mcts": True, "mcts_sims": 100,
         "value_checkpoint": "ml/runs/ml4-h3/value-net.pt",
         "value_weight": 0.125},  # sweep winner (0.725 vs 0.600 @ 0.25, 0.475 @ 0.5)
}

# --- weights --------------------------------------------------------------
# Orders of magnitude apart so pattern classes dominate each other.
WIN_SCORE = 10_000_000
OPEN_FOUR = 200_000
FOUR = 20_000
OPEN_THREE = 4_000
THREE = 400
OPEN_TWO = 80
TWO = 10
ONE_OPEN = 2
ONE = 1

# A move creating fours in >= 2 directions wins next move outright;
# a move creating open threes in >= 2 directions forces a win in two.
DOUBLE_FOUR_BONUS = 200_000
DOUBLE_OPEN_THREE_BONUS = 50_000

# Pattern kinds (per direction, from the strongest window in it).
_KIND_NONE = 0
_KIND_FOUR = 1
_KIND_OPEN_THREE = 2
_KIND_FIVE = 3

_DIRECTIONS = ((0, 1), (1, 0), (1, 1), (1, -1))


# --- precomputed windows --------------------------------------------------
# All 572 five-cell windows, each as (cells, before, after, dir_index):
# ``cells`` is the 5 flat indices of the window, ``before``/``after``
# the flat index of the cell just outside each end (-1 if off-board),
# ``dir_index`` the index into _DIRECTIONS.

def _build_windows() -> list[tuple[tuple[int, ...], int, int, int]]:
    wins: list[tuple[tuple[int, ...], int, int, int]] = []
    for di, (dy, dx) in enumerate(_DIRECTIONS):
        for y in range(game.SIZE):
            for x in range(game.SIZE):
                ey, ex = y + 4 * dy, x + 4 * dx
                if not (0 <= ey < game.SIZE and 0 <= ex < game.SIZE):
                    continue
                cells = tuple(
                    (y + i * dy) * game.SIZE + (x + i * dx) for i in range(5)
                )
                by, bx = y - dy, x - dx
                ay, ax = ey + dy, ex + dx
                before = (
                    by * game.SIZE + bx
                    if 0 <= by < game.SIZE and 0 <= bx < game.SIZE
                    else -1
                )
                after = (
                    ay * game.SIZE + ax
                    if 0 <= ay < game.SIZE and 0 <= ax < game.SIZE
                    else -1
                )
                wins.append((cells, before, after, di))
    return wins


_WINDOWS = _build_windows()

# For each cell, the windows containing it (at most 20), as
# (window index, position of the cell within the window). The position
# lets the scorer count all five cells without a per-cell
# "is this the move?" check: the move cell is empty on the board, so
# it contributes to neither side's count, and for a four the other
# empty cell's position is (sum of empty positions) - (move position).
_WINDOWS_BY_CELL: list[list[tuple[int, int]]] = [
    [] for _ in range(game.SIZE * game.SIZE)
]
for _wi, (_cells, _b, _a, _d) in enumerate(_WINDOWS):
    for _i, _c in enumerate(_cells):
        _WINDOWS_BY_CELL[_c].append((_wi, _i))


# --- window evaluation ----------------------------------------------------

def _score_from_counts(
    own: int, opp: int, empty_idx: int, open_ends: int,
    flat: list[int], before: int, after: int,
) -> tuple[int, int]:
    """``(score, kind)`` for a window holding ``own`` stones of the
    side that played the candidate move and ``opp`` opponent stones.

    ``empty_idx`` is the position (0-4) of the window's empty cell
    (only meaningful when ``own == 4``, when there is exactly one);
    ``open_ends`` counts the empty cells just outside the window.
    A window containing an opponent stone scores 0 (blocked).
    """
    if opp > 0:
        return 0, _KIND_NONE
    if own == 5:
        return WIN_SCORE, _KIND_FIVE
    if own == 4:
        # The empty cell inside the window is always a winning square,
        # so this is a four whatever the outside looks like. It is an
        # OPEN four (two winning squares) only when the four is
        # contiguous and the outside cell beyond its far end is empty:
        # the gap of a broken four is its only winning square.
        if empty_idx == 0:
            # Four occupies cells 1-4; the far outside cell is `after`.
            if after != -1 and flat[after] == game.EMPTY:
                return OPEN_FOUR, _KIND_FOUR
        elif empty_idx == 4:
            # Four occupies cells 0-3; the far outside cell is `before`.
            if before != -1 and flat[before] == game.EMPTY:
                return OPEN_FOUR, _KIND_FOUR
        return FOUR, _KIND_FOUR
    if own == 3:
        if open_ends == 2:
            return OPEN_THREE, _KIND_OPEN_THREE
        if open_ends == 1:
            return THREE, _KIND_NONE
        return 0, _KIND_NONE
    if own == 2:
        if open_ends == 2:
            return OPEN_TWO, _KIND_NONE
        if open_ends == 1:
            return TWO, _KIND_NONE
        return 0, _KIND_NONE
    if own == 1:
        return (ONE_OPEN if open_ends == 2 else ONE), _KIND_NONE
    return 0, _KIND_NONE


def _score_parts_flat(flat: list[int], move: int, player: int,
                      defence_weight: float = 1.0) -> tuple[int, int]:
    """Core scorer on a flat board (list of 225 ints).

    Returns ``(attack, defence)``: the attack value (what the move
    builds for ``player``, including the double-threat bonuses) and
    the defence value (what it denies the opponent by occupying the
    point), with each direction's defence discounted by
    ``defence_weight`` (per-direction truncation, mirroring the
    baseline's ``int(0.9 * w)``). ``defence_weight=1.0`` is v1
    (full-value defence); later versions test other weights.

    One pass over the ~20 windows through ``move`` counts the
    mover's and the opponent's stones in each window; attack and
    defence follow from the same counts (attack: the move is the
    mover's stone, so own = p+1, opp = o; defence: the move is the
    opponent's stone, so own = o+1, opp = p). Each direction
    contributes only its strongest window (see the module docstring,
    point 3).

    Note the symmetry: at ``defence_weight=1.0``,
    attack(move, player) + defence(move, player) equals
    attack(move, -player) + defence(move, -player) — the total score
    of a square is the same for both colours (attack and defence
    swap roles). With a defence discount the totals differ between
    colours. Either way, anything that must distinguish the colours
    (e.g. the swap-rule colour choice) compares the ATTACK parts,
    not the totals.
    """
    opp = -player
    best_a = [0, 0, 0, 0]
    kinds_a = [_KIND_NONE, _KIND_NONE, _KIND_NONE, _KIND_NONE]
    best_d = [0, 0, 0, 0]
    for wi, move_idx in _WINDOWS_BY_CELL[move]:
        cells, before, after, di = _WINDOWS[wi]
        p = 0
        o = 0
        empty_sum = 0
        # The move cell is empty on the board, so it counts as empty
        # here and contributes to neither side — no per-cell check
        # needed. For a four, the other empty cell's position is
        # empty_sum - move_idx (the empties are the move cell and one
        # other cell).
        for i, c in enumerate(cells):
            v = flat[c]
            if v == player:
                p += 1
            elif v == opp:
                o += 1
            else:
                empty_sum += i
        if p > 0 and o > 0:
            continue  # blocked for both sides
        # Open ends: the cells just outside the window (edge = closed).
        open_ends = 0
        if before != -1 and flat[before] == game.EMPTY:
            open_ends += 1
        if after != -1 and flat[after] == game.EMPTY:
            open_ends += 1
        if o == 0:
            s, kind = _score_from_counts(
                p + 1, 0, empty_sum - move_idx, open_ends,
                flat, before, after,
            )
            if s > best_a[di]:
                best_a[di] = s
                kinds_a[di] = kind
        if p == 0:
            s, _kind = _score_from_counts(
                o + 1, 0, empty_sum - move_idx, open_ends,
                flat, before, after,
            )
            if s > best_d[di]:
                best_d[di] = s
    attack = sum(best_a)
    fours = sum(1 for k in kinds_a if k in (_KIND_FOUR, _KIND_FIVE))
    open_threes = sum(1 for k in kinds_a if k == _KIND_OPEN_THREE)
    if fours >= 2:
        attack += DOUBLE_FOUR_BONUS
    if open_threes >= 2:
        attack += DOUBLE_OPEN_THREE_BONUS
    defence = sum(int(defence_weight * d) for d in best_d)
    return attack, defence


def _score_move_flat(flat: list[int], move: int, player: int,
                     defence_weight: float = 1.0) -> int:
    """Total score (attack + discounted defence) on a flat board."""
    attack, defence = _score_parts_flat(flat, move, player, defence_weight)
    return attack + defence


# --- public interface (mirrors baseline) ----------------------------------

def score_move(board: np.ndarray, move: int, player: int) -> int:
    """Heuristic score for ``player`` playing ``move`` on ``board``
    (v1: attack plus full-value defence, plus double-threat bonuses —
    see the module docstring). Versioned scoring lives on
    :class:`ScoringPlayer` (``.score_move``); this module function
    stays v1 so the interface matches the frozen baseline exactly.
    """
    flat = board.reshape(-1).tolist()
    return _score_move_flat(flat, move, player)


def score_attack(board: np.ndarray, move: int, player: int) -> int:
    """Attack-only score for ``player`` playing ``move`` on ``board``:
    what the move builds for ``player``, including the double-threat
    bonuses, WITHOUT the defence component.

    This is the colour-asymmetric half of :func:`score_move` — the
    total score is identical for both colours at any square (attack
    and defence swap roles), so any comparison between colours (the
    swap-rule colour choice in :func:`gomoku_ml.game.choose_swap_color`)
    must use this, not the total.
    """
    flat = board.reshape(-1).tolist()
    attack, _defence = _score_parts_flat(flat, move, player)
    return attack


def _near_empty(board: np.ndarray, color: int | None, radius: int) -> list[int]:
    """Empty points within Chebyshev distance ``radius`` of any stone
    of ``color`` (``None`` = any stone). Sorted ascending."""
    if color is None:
        stones = np.argwhere(board != game.EMPTY)
    else:
        stones = np.argwhere(board == color)
    if len(stones) == 0:
        return []
    near = np.zeros((game.SIZE, game.SIZE), dtype=bool)
    for y, x in stones:
        y0, y1 = max(0, y - radius), min(game.SIZE, y + radius + 1)
        x0, x1 = max(0, x - radius), min(game.SIZE, x + radius + 1)
        near[y0:y1, x0:x1] = True
    return [int(m) for m in np.flatnonzero((board == game.EMPTY) & near)]


def _candidate_moves(board: np.ndarray) -> list[int]:
    """Moves worth scoring: empty points within Chebyshev distance 2
    of any stone (the centre on an empty board). Falls back to all
    legal moves if the candidate set is empty (near-full board)."""
    stones = int(np.count_nonzero(board))
    if stones == 0:
        return [game.xy_to_move(7, 7)]
    cands = _near_empty(board, None, 2)
    if cands:
        return cands
    return [int(m) for m in game.legal_moves(board)]


# --- 2-ply threat check (v3) ------------------------------------------------
# An "open-four creator" for a player is a move that leaves them with
# >= 2 winning moves (an open four). If the opponent has a creator,
# they make the open four next move and win in two — unless our move
# wins, makes a four (forcing a block), or leaves them with no live
# creator. A 1-ply scorer cannot see this: a line-block that kills a
# creator scores only the opponent's attack at that square, so a
# higher-scoring attacking move beats it and the creator stays live.

def _winning_moves(board: np.ndarray, player: int) -> list[int]:
    """All immediate winning moves for ``player`` (same semantics as
    :func:`gomoku_ml.evaluate.winning_moves`, implemented locally to
    keep this module free of the evaluate dependency)."""
    return [
        m for m in _near_empty(board, player, 1)
        if game.is_winning_move(board, m, player)
    ]


def _opponent_creators(board: np.ndarray, player: int) -> list[int]:
    """Moves by which ``player``'s opponent would create an open four
    (leave themselves >= 2 winning moves). A creator must be adjacent
    to an opponent stone, so it is always in ``_candidate_moves``."""
    opp = -player
    return [
        m for m in _candidate_moves(board)
        if len(_winning_moves(game.apply_move(board, m, opp), opp)) >= 2
    ]


def _is_forcing(board: np.ndarray, move: int, player: int) -> bool:
    """Does ``player``'s ``move`` win immediately or leave them with a
    four (>= 1 winning move)? Either forces the opponent to answer
    instead of playing a creator."""
    b2 = game.apply_move(board, move, player)
    return (
        game.is_winning_stone(b2, move, player)
        or len(_winning_moves(b2, player)) >= 1
    )


def _move_is_safe(board: np.ndarray, move: int, player: int,
                  creators: list[int]) -> bool:
    """Is ``move`` safe against the opponent's ``creators`` (computed
    on the current board)? Safe = forcing (win or four), or no creator
    remains live after the move. A move can only REMOVE creators (it
    adds one of our stones), so the creators after the move are a
    subset of ``creators`` — only those need re-checking."""
    if _is_forcing(board, move, player):
        return True
    b2 = game.apply_move(board, move, player)
    for c in creators:
        if c == move:
            continue  # occupied by our move
        b3 = game.apply_move(b2, c, -player)
        if len(_winning_moves(b3, -player)) >= 2:
            return False
    return True


# --- 2-ply minimax (v4) -----------------------------------------------------
# For each of the top-K 1-ply candidates, play the opponent's v1 best
# reply, then evaluate the resulting position statically. The static
# evaluation is the BEST-ATTACK DIFFERENCE (attack-only scores): the
# total score is colour-symmetric at defence_weight=1.0 (attack and
# defence swap roles between colours — see _score_parts_flat), so a
# total-score difference would always be 0. Attack-only scores are
# colour-asymmetric: a side with a winning move next scores 10M, a
# side facing an open four scores 200k, so the difference encodes
# who wins the open-four race from the resulting position.
#
# The opponent is modelled as v1 (win/block scans + best 1-ply move,
# no threat check, no minimax — the reply call must NOT recurse into
# minimax). Ties on minimax value break toward the better 1-ply
# score, then the RNG (or the first move when no RNG is given).

def _best_attack_on(board: np.ndarray, player: int,
                    defence_weight: float = 1.0,
                    exclude: set[int] | None = None) -> int:
    """Max attack-only score for ``player`` over the candidate moves
    on ``board`` (0 when the board has no candidates), skipping moves
    in ``exclude`` (used by the terminal-aware leaf value to skip
    winning moves, which are valued separately)."""
    flat = board.reshape(-1).tolist()
    best = 0
    for m in _candidate_moves(board):
        if exclude is not None and m in exclude:
            continue
        attack, _defence = _score_parts_flat(flat, m, player, defence_weight)
        if attack > best:
            best = attack
    return best


def _leaf_value(board: np.ndarray, player: int,
                defence_weight: float = 1.0) -> int:
    """Terminal-aware static evaluation of ``board`` from ``player``'s
    perspective (v5 leaf value). Counts each side's winning moves:

    - Opponent with >= 2 winning moves: terminal loss, -WIN_SCORE.
      ``player`` moves next but can block only one square (open
      four) — no attack score can compensate.
    - ``player`` with >= 2 winning moves: terminal win, +WIN_SCORE.
    - Exactly 1 winning move for a side: capped at FOUR — the player
      to move blocks it, so it is a forcing threat, not a win.
    - Otherwise: best-attack difference over non-winning moves.

    The v4 leaf value (raw best-attack difference) scores a single
    blockable winning move at WIN_SCORE, which inverts the ranking
    of blocking vs attacking moves (see iter-4 §2.3)."""
    our_wins = set(_winning_moves(board, player))
    opp_wins = set(_winning_moves(board, -player))
    if len(opp_wins) >= 2:
        return -WIN_SCORE
    if len(our_wins) >= 2:
        return WIN_SCORE
    ours = _best_attack_on(board, player, defence_weight, exclude=our_wins)
    theirs = _best_attack_on(board, -player, defence_weight, exclude=opp_wins)
    if len(our_wins) == 1:
        ours += FOUR
    if len(opp_wins) == 1:
        theirs += FOUR
    return ours - theirs


def _minimax_choose(board: np.ndarray, player: int,
                    flat: list[int],
                    scored: list[tuple[int, int]],
                    rng: np.random.Generator | None,
                    defence_weight: float,
                    topk: int,
                    terminal_aware: bool = False) -> int:
    """Pick the best of the top-``topk`` 1-ply candidates by 2-ply
    minimax. ``scored`` is the 1-ply ``(score, move)`` list from
    :func:`_choose_move` (win/block scans already ran, so no
    candidate wins immediately and the opponent has no immediate win
    on the current board). With ``terminal_aware`` (v5) the leaf is
    evaluated by :func:`_leaf_value`; otherwise by raw best-attack
    difference (v4). Returns a legal move."""
    top = sorted(scored, key=lambda sm: (-sm[0], sm[1]))[:max(1, topk)]
    best_key: tuple[int, int] | None = None
    best_moves: list[int] = []
    for s1, move in top:
        b2 = game.apply_move(board, move, player)
        if game.is_winning_stone(b2, move, player):
            return move  # unreachable (win scan runs first) but safe
        if len(game.legal_moves(b2)) == 0:
            key = (0, s1)  # our move fills the board: draw
        else:
            # Opponent's best reply, modelled as v1 (no minimax —
            # no recursion).
            reply = _choose_move(b2, -player, None, defence_weight,
                                 threat_check=False, minimax=False)
            b3 = game.apply_move(b2, reply, -player)
            if game.is_winning_stone(b3, reply, -player):
                # The opponent wins on the reply: this candidate loses.
                key = (-WIN_SCORE, s1)
            elif terminal_aware:
                key = (_leaf_value(b3, player, defence_weight), s1)
            else:
                ours = _best_attack_on(b3, player, defence_weight)
                theirs = _best_attack_on(b3, -player, defence_weight)
                key = (ours - theirs, s1)
        if best_key is None or key > best_key:
            best_key = key
            best_moves = [move]
        elif key == best_key:
            best_moves.append(move)
    if rng is not None:
        return int(rng.choice(best_moves))
    return best_moves[0]


def _choose_move(board: np.ndarray, player: int,
                 rng: np.random.Generator | None,
                 defence_weight: float,
                 threat_check: bool = False,
                 minimax: bool = False,
                 minimax_topk: int = 8,
                 terminal_aware: bool = False) -> int:
    """Versioned move choice (see :func:`choose_move`).

    With ``threat_check`` (v3): after the win/block scans, if the
    opponent has an open-four creator, prefer the best-scoring move
    that leaves them with no live creator (or a forcing move) over
    the 1-ply best move when the 1-ply best is unsafe. If no safe
    move exists, fall back to the 1-ply best. With no opponent
    creator, v3 chooses exactly like v1.

    With ``minimax`` (v4): after the win/block scans, pick the best
    of the top-``minimax_topk`` 1-ply candidates by 2-ply minimax
    (see :func:`_minimax_choose`). ``minimax`` takes precedence over
    ``threat_check`` (no version sets both).
    """
    legal = game.legal_moves(board)
    if len(legal) == 0:
        raise ValueError("no legal moves")
    # 1. Take the win.
    for move in _near_empty(board, player, 1):
        if game.is_winning_move(board, move, player):
            return move
    # 2. Block the opponent's win.
    for move in _near_empty(board, -player, 1):
        if game.is_winning_move(board, move, -player):
            return move
    # 3. Best heuristic score among candidates.
    flat = board.reshape(-1).tolist()
    scored: list[tuple[int, int]] = [
        (_score_move_flat(flat, move, player, defence_weight), move)
        for move in _candidate_moves(board)
    ]
    best_score = max(s for s, _m in scored)
    best_moves = [m for s, m in scored if s == best_score]

    def _pick(moves: list[int]) -> int:
        if rng is not None:
            return int(rng.choice(moves))
        return moves[0]

    if minimax:
        return _minimax_choose(board, player, flat, scored, rng,
                               defence_weight, minimax_topk,
                               terminal_aware)
    if not threat_check:
        return _pick(best_moves)
    # 3b. 2-ply threat check: if the opponent can create an open four
    # next move, the 1-ply best is only good if it stops them.
    creators = _opponent_creators(board, player)
    if not creators:
        return _pick(best_moves)
    safe_best = [m for m in best_moves
                 if _move_is_safe(board, m, player, creators)]
    if safe_best:
        return _pick(safe_best)
    safe_scored = [(s, m) for s, m in scored
                   if _move_is_safe(board, m, player, creators)]
    if safe_scored:
        best_safe = max(s for s, _m in safe_scored)
        return _pick([m for s, m in safe_scored if s == best_safe])
    return _pick(best_moves)


def choose_move(board: np.ndarray, player: int, rng: np.random.Generator | None = None) -> int:
    """Pick a move for ``player`` on ``board`` (v1).

    1. Immediate win (scanned only among radius-1 candidates of the
       mover's stones — a winning move must be adjacent to a friendly
       stone). 2. Block the opponent's immediate win (same, around the
       opponent's stones). 3. Best 5-window heuristic score among the
       candidate moves (random tie-break with the given rng, or the
       first best move if no rng is given). Versioned move choice
    lives on :class:`ScoringPlayer`; this module function stays v1
    so the interface matches the frozen baseline exactly.
    """
    return _choose_move(board, player, rng, 1.0)


class ScoringPlayer:
    """Stateful wrapper with a seeded RNG for reproducible eval games
    (mirrors :class:`gomoku_ml.baseline.HeuristicPlayer`).

    ``version`` selects the parameter set from
    :data:`ENGINE_VERSIONS` (default 1 = v1 behaviour).
    """

    def __init__(self, seed: int = 0, version: int = 1) -> None:
        if version not in ENGINE_VERSIONS:
            raise ValueError(
                f"unknown scoring engine version: {version} "
                f"(known: {sorted(ENGINE_VERSIONS)})"
            )
        self.version = version
        self.name = ENGINE_VERSIONS[version]["name"]
        self.defence_weight = ENGINE_VERSIONS[version]["defence_weight"]
        self.threat_check = ENGINE_VERSIONS[version].get("threat_check", False)
        self.minimax = ENGINE_VERSIONS[version].get("minimax", False)
        self.minimax_topk = ENGINE_VERSIONS[version].get("minimax_topk", 8)
        self.terminal_aware = ENGINE_VERSIONS[version].get("terminal_aware", False)
        self.mcts = ENGINE_VERSIONS[version].get("mcts", False)
        self.rng = np.random.default_rng(seed)
        self._mcts = None
        if self.mcts:
            value_ckpt = ENGINE_VERSIONS[version].get("value_checkpoint")
            white_sims = ENGINE_VERSIONS[version].get("white_sims")
            white_temp = ENGINE_VERSIONS[version].get("white_temp")
            if ENGINE_VERSIONS[version].get("v17"):
                # v17: the combination (lazy import, mcts_v17
                # imports this module via mcts_heuristic).
                from .mcts_v17 import V17MCTS

                self._mcts = V17MCTS(
                    sims=ENGINE_VERSIONS[version].get("mcts_sims", 100),
                    rng=self.rng,
                )
            elif white_temp is not None:
                # v13: WhiteMCTS + White prior temperature (lazy
                # import, mcts_v13 imports this module via
                # mcts_white/mcts_heuristic).
                from .mcts_v13 import V13MCTS

                self._mcts = V13MCTS(
                    sims=ENGINE_VERSIONS[version].get("mcts_sims", 100),
                    white_sims=white_sims,
                    white_temp=white_temp,
                    rng=self.rng,
                )
            elif white_sims is not None:
                # Colour-asymmetric MCTS (v10 port, v13): lazy import,
                # mcts_white imports this module (via mcts_heuristic).
                from .mcts_white import WhiteMCTS

                self._mcts = WhiteMCTS(
                    sims=ENGINE_VERSIONS[version].get("mcts_sims", 100),
                    white_sims=white_sims,
                    rng=self.rng,
                )
            elif value_ckpt is not None:
                # Learned-value MCTS (v8): lazy import, mcts_value
                # imports this module (via mcts_heuristic). The path
                # is relative to the repo root unless absolute; the
                # GOMOKU_V8_CHECKPOINT env var overrides it.
                from .mcts_value import ValueMCTS

                path = os.environ.get("GOMOKU_V8_CHECKPOINT") or value_ckpt
                if not os.path.isabs(path):
                    repo_root = os.path.dirname(
                        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                    )
                    path = os.path.join(repo_root, path)
                self._mcts = ValueMCTS(
                    checkpoint_path=path,
                    sims=ENGINE_VERSIONS[version].get("mcts_sims", 100),
                    rng=self.rng,
                    value_weight=ENGINE_VERSIONS[version].get("value_weight"),
                )
            else:
                # Lazy import: mcts_heuristic imports this module.
                from .mcts_heuristic import HeuristicMCTS

                self._mcts = HeuristicMCTS(
                    sims=ENGINE_VERSIONS[version].get("mcts_sims", 100),
                    c_puct=ENGINE_VERSIONS[version].get("c_puct", 1.5),
                    prior_floor=ENGINE_VERSIONS[version].get("prior_floor", 0.0),
                    leaf_mode=ENGINE_VERSIONS[version].get("leaf_mode", "terminal"),
                    root_decision=ENGINE_VERSIONS[version].get(
                        "root_decision", "visits"
                    ),
                    vcf_root=ENGINE_VERSIONS[version].get("vcf_root", False),
                    rng=self.rng,
                    tie_break_q=ENGINE_VERSIONS[version].get("tie_break_q", False),
                    tie_break_q_threshold=ENGINE_VERSIONS[version].get(
                        "tie_break_q_threshold", 0.0
                    ),
                )

    def reseed(self, seed: int) -> None:
        """Reset the RNG stream to a fresh generator from ``seed``.

        Per-game seeding (ML batch 3 tooling): the match drivers
        call this before every game with ``base_seed + game_index``,
        so each game is determined by its index alone — a resumed
        run then reproduces the uninterrupted run exactly, instead
        of replaying the prefix (batch 10's resume seed-replay bug:
        the old drivers created the engines once, so on resume the
        RNG streams restarted and the resumed segment replayed the
        prefix when segment-start parity aligned). The inner MCTS
        engine shares this RNG object, so it is reset too.
        """
        self.rng = np.random.default_rng(seed)
        if self._mcts is not None:
            self._mcts.rng = self.rng

    def score_move(self, board: np.ndarray, move: int, player: int) -> int:
        """This version's heuristic score for ``player`` playing
        ``move`` on ``board``."""
        flat = board.reshape(-1).tolist()
        return _score_move_flat(flat, move, player, self.defence_weight)

    def choose_move(self, board: np.ndarray, player: int) -> int:
        if self.mcts:
            return self._mcts.choose_move(board, player)
        return _choose_move(
            board, player, self.rng, self.defence_weight, self.threat_check,
            self.minimax, self.minimax_topk, self.terminal_aware,
        )


def get_player(version: int, seed: int = 0) -> ScoringPlayer:
    """Build a :class:`ScoringPlayer` for engine ``version``."""
    return ScoringPlayer(seed=seed, version=version)
