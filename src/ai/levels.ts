import type { LevelId } from './types';

export type { LevelId } from './types';

export interface LevelMeta {
  id: LevelId;
  name: string;
  tagline: string;
  description: string;
  /**
   * Honest strength note shown to the user. Never claims superhuman play:
   * the Master level's strength depends entirely on the loaded model, and the
   * bundled model is a small demo net.
   */
  strengthNote: string;
  defaultSims: number;
  defaultTimeMs: number;
}

/**
 * AI level metadata. Strength notes are deliberately honest: only the Master
 * level uses a neural network, and with the bundled small demo net it plays
 * at strong search level — not superhuman. A stronger model (trained longer,
 * on more self-play games) is what would raise its ceiling.
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
    name: 'Master (AlphaZero-style)',
    tagline: 'Neural network + tree search',
    description:
      'PUCT Monte Carlo tree search guided by an ONNX policy+value network running on-device ' +
      '(WebGPU when available, otherwise WASM). Without a loaded model it falls back to Expert-style MCTS.',
    strengthNote:
      'Strength depends entirely on the loaded model. With the bundled small demo net it plays at ' +
      'strong search level — comparable to Expert, NOT superhuman. A larger net trained on extensive ' +
      'self-play would be needed to approach top-human play; the bundled demo net is not that.',
    defaultSims: 400,
    defaultTimeMs: 5000,
  },
};

export const LEVEL_ORDER: LevelId[] = ['beginner', 'intermediate', 'advanced', 'expert', 'master'];
