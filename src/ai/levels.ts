import type { LevelId } from './types';

export type { LevelId } from './types';

export interface LevelMeta {
  id: LevelId;
  name: string;
  tagline: string;
  description: string;
  /**
   * Honest strength note shown to the user. Never claims superhuman play:
   * the Master level is the research v7 engine — the strongest version so
   * far, but still below a strong human amateur.
   */
  strengthNote: string;
  defaultSims: number;
  defaultTimeMs: number;
}

/**
 * AI level metadata. Strength notes are deliberately honest: the Master
 * level is the research v7 engine (MCTS + attack-gradient leaf), the first
 * version to pass the auto-research bar — strong search play, not
 * superhuman. The old AlphaZero-style Master (ONNX net) is retired from
 * play; the bundled demo net was below the 1-ply heuristic.
 */
export const LEVEL_META: Record<LevelId, LevelMeta> = {
  beginner: {
    id: 'beginner',
    name: 'Beginner',
    tagline: 'Learns the patterns with you',
    description:
      'One-ply pattern play: takes immediate wins, blocks immediate losses, and otherwise ' +
      'picks a good-looking pattern move — with an occasional random slip, like a human beginner.',
    strengthNote:
      'Casual beginner strength. Knows the basic patterns (fours, open threes) but has no ' +
      'lookahead beyond one move, so it misses simple two-move tactics. A focused human beginner can beat it.',
    defaultSims: 0,
    defaultTimeMs: 50,
  },
  intermediate: {
    id: 'intermediate',
    name: 'Intermediate',
    tagline: 'Thinks a few moves ahead',
    description:
      'Depth-3 alpha-beta search over a pattern evaluation, with move ordering. Sees simple ' +
      'tactics and two-move combinations.',
    strengthNote:
      'Solid casual-club strength. Spots immediate tactics and short combinations, but its ' +
      '3-ply horizon misses deeper forced wins and it has no real positional understanding.',
    defaultSims: 0,
    defaultTimeMs: 500,
  },
  advanced: {
    id: 'advanced',
    name: 'Advanced',
    tagline: 'Deep search, finds forced wins',
    description:
      'Iterative-deepening alpha-beta with a transposition table, killer/history move ordering, ' +
      'and a threat-space search that finds forced-win lines of consecutive fours (VCF-style).',
    strengthNote:
      'Strong amateur strength. Finds most forced wins within its search horizon and defends ' +
      'tactics well, but its pattern evaluation is hand-written, not learned — a strong human ' +
      'amateur will outplay it positionally.',
    defaultSims: 0,
    defaultTimeMs: 2000,
  },
  expert: {
    id: 'expert',
    name: 'Expert',
    tagline: 'Monte Carlo tree search',
    description:
      'MCTS with UCT selection, pattern-based move priors, and heuristic rollouts. Reasons about ' +
      'whole lines of play rather than a fixed depth.',
    strengthNote:
      'Strong search play, roughly on par with Advanced but with a different style: better at ' +
      'judging messy positions, weaker at exact forced-win calculation. Still no learned knowledge — ' +
      'a strong human amateur will beat it.',
    defaultSims: 1000,
    defaultTimeMs: 2500,
  },
  master: {
    id: 'master',
    name: 'Master',
    tagline: 'Research v7 engine — MCTS with an attack-gradient leaf',
    description:
      'The strongest engine from the auto-research program (v7): PUCT Monte Carlo tree search ' +
      'with heuristic priors and an attack-gradient leaf value. The first version to pass the ' +
      'research bar — it beats the 1-ply heuristic head-to-head (ladder Elo 190.8). No neural ' +
      'network: it runs entirely on search.',
    strengthNote:
      'Strongest level. Beats the 1-ply heuristic decisively and defends tactics perfectly, but ' +
      'it is NOT superhuman — it still loses the direct match to a searching opponent and plays ' +
      'White weakly. A strong human amateur will beat it.',
    defaultSims: 100,
    defaultTimeMs: 5000,
  },
};

export const LEVEL_ORDER: LevelId[] = ['beginner', 'intermediate', 'advanced', 'expert', 'master'];
