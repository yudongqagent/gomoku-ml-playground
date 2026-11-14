#!/usr/bin/env node
/**
 * Engine-vs-engine match: new Master (v7) vs old Master (MCTS fallback).
 *
 * The old Master level (alphazero.ts) without a model URL falls back to
 * Expert-style heuristic MCTS (`chooseMctsMove`) — that is the baseline
 * here, at its LEVEL_META defaults (400 sims, 5000 ms/move). The ONNX
 * path is not run: the bundled demo net is Below heuristic (weaker than
 * the fallback), and onnxruntime-web does not run in Node. So this match
 * compares v7 against the STRONGER of the old Master's two behaviours.
 *
 * Both sides play through the same shared pre-check as `src/ai/engine.ts`
 * (take an immediate win; block the opponent's immediate win with the
 * best-scoring block), so the match mirrors real page play. Games
 * alternate colours; draws are full-board games (225 plies).
 *
 * The driver is bundled with esbuild (a transitive vite dependency) and
 * run; it prints per-game results to stderr and one line
 * `MATCH_RESULT {...}` on stdout. A seeded RNG (mulberry32, seed =
 * 1000 + game index) makes the match reproducible.
 *
 * Usage (from the repo root):
 *   node scripts/v7_vs_master_match.mjs [--games 20]
 */

import { spawnSync } from 'node:child_process';
import * as esbuild from 'esbuild';
import * as fs from 'node:fs';
import * as os from 'node:os';
import * as path from 'node:path';
import { fileURLToPath } from 'node:url';

const REPO_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');

// esbuild plugin for Vite's `?raw` imports (the WGSL shader source in
// src/kernel/scoring.ts): load the file as text. The shader string is
// only used by GpuScorer.create, which this driver never calls.
const rawPlugin = {
  name: 'raw-imports',
  setup(build) {
    build.onResolve({ filter: /\.wgsl\?raw$/ }, (args) => ({
      path: path.join(args.resolveDir, args.path.replace(/\?raw$/, '')),
      namespace: 'raw-text',
    }));
    build.onLoad({ filter: /.*/, namespace: 'raw-text' }, async (args) => ({
      contents: await fs.promises.readFile(args.path, 'utf8'),
      loader: 'text',
    }));
  },
};

// NOTE: this template must not contain backticks or `${` sequences — it is
// a JS template literal.
const DRIVER_TS = `
import { chooseMctsMove } from '__REPO_ROOT__/src/ai/mcts';
import { chooseV7Move } from '__REPO_ROOT__/src/ai/v7';
import { evaluateBoard, findWinningMove, findWinningMoves, scoreMove, winRateFromEval } from '__REPO_ROOT__/src/ai/patterns';
import { SIZE, hasFive, type Player, type Point } from '__REPO_ROOT__/src/engine/gomoku';

interface MatchConfig {
  games: number;
  maxMoves: number;
}

const cfg: MatchConfig = JSON.parse(process.env.V7_MATCH_CONFIG as string);

/** Deterministic RNG (mulberry32) so the match is reproducible. */
function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

type EngineId = 'v7' | 'old-master';

/** Shared pre-check (mirrors src/ai/engine.ts): win, else block. Returns a move or null. */
function preCheck(board: Int8Array, player: Player): Point | null {
  const ownWin = findWinningMove(board, player);
  if (ownWin) return ownWin;
  const oppWins = findWinningMoves(board, player === 1 ? -1 : 1);
  if (oppWins.length > 0) {
    let block = oppWins[0];
    let bestScore = -Infinity;
    for (const w of oppWins) {
      const s = scoreMove(board, w.x, w.y, player);
      if (s > bestScore) {
        bestScore = s;
        block = w;
      }
    }
    return block;
  }
  return null;
}

async function playOneTurn(
  board: Int8Array,
  player: Player,
  engine: EngineId,
  rng: () => number,
): Promise<{ move: Point; ms: number }> {
  const t0 = performance.now();
  const forced = preCheck(board, player);
  if (forced) return { move: forced, ms: performance.now() - t0 };
  if (engine === 'v7') {
    const r = chooseV7Move(board, player, { sims: 100, timeMs: 5000, rng });
    return { move: r.move, ms: performance.now() - t0 };
  }
  const r = await chooseMctsMove(board, player, { sims: 400, timeMs: 5000, rng });
  return { move: r.move, ms: performance.now() - t0 };
}

async function playGame(gameIndex: number): Promise<{
  winner: 'v7' | 'old-master' | 'draw';
  plies: number;
  v7Ms: number;
  oldMs: number;
  v7Moves: number;
  oldMoves: number;
}> {
  const rng = mulberry32(1000 + gameIndex);
  const board = new Int8Array(SIZE * SIZE);
  const v7IsBlack = gameIndex % 2 === 0;
  let turn: Player = 1;
  let v7Ms = 0;
  let oldMs = 0;
  let v7Moves = 0;
  let oldMoves = 0;
  for (let ply = 0; ply < cfg.maxMoves; ply++) {
    const engine: EngineId = (turn === 1) === v7IsBlack ? 'v7' : 'old-master';
    const { move, ms } = await playOneTurn(board, turn, engine, rng);
    if (engine === 'v7') {
      v7Ms += ms;
      v7Moves++;
    } else {
      oldMs += ms;
      oldMoves++;
    }
    const idx = move.y * SIZE + move.x;
    if (board[idx] !== 0) throw new Error('illegal move ' + move.x + ',' + move.y + ' by ' + engine);
    board[idx] = turn;
    if (hasFive(board, move.x, move.y)) {
      return { winner: engine, plies: ply + 1, v7Ms, oldMs, v7Moves, oldMoves };
    }
    turn = turn === 1 ? -1 : 1;
  }
  return { winner: 'draw', plies: cfg.maxMoves, v7Ms, oldMs, v7Moves, oldMoves };
}

async function main(): Promise<void> {
  let v7Wins = 0;
  let oldWins = 0;
  let draws = 0;
  let v7AsBlackW = 0;
  let v7AsBlackL = 0;
  let v7AsWhiteW = 0;
  let v7AsWhiteL = 0;
  let totalV7Ms = 0;
  let totalOldMs = 0;
  let totalV7Moves = 0;
  let totalOldMoves = 0;
  for (let g = 0; g < cfg.games; g++) {
    const r = await playGame(g);
    totalV7Ms += r.v7Ms;
    totalOldMs += r.oldMs;
    totalV7Moves += r.v7Moves;
    totalOldMoves += r.oldMoves;
    const v7IsBlack = g % 2 === 0;
    if (r.winner === 'v7') {
      v7Wins++;
      if (v7IsBlack) v7AsBlackW++;
      else v7AsWhiteW++;
    } else if (r.winner === 'old-master') {
      oldWins++;
      if (v7IsBlack) v7AsBlackL++;
      else v7AsWhiteL++;
    } else {
      draws++;
    }
    console.error(
      'game ' + (g + 1) + '/' + cfg.games + ': ' + r.winner + ' in ' + r.plies +
      ' plies (v7 ' + (v7IsBlack ? 'Black' : 'White') + ')',
    );
  }
  const score = v7Wins + 0.5 * draws;
  console.log(
    'MATCH_RESULT ' +
      JSON.stringify({
        games: cfg.games,
        v7Wins,
        oldMasterWins: oldWins,
        draws,
        v7WinRate: score / cfg.games,
        v7AsBlack: { wins: v7AsBlackW, losses: v7AsBlackL },
        v7AsWhite: { wins: v7AsWhiteW, losses: v7AsWhiteL },
        v7MeanMsPerMove: totalV7Moves > 0 ? totalV7Ms / totalV7Moves : 0,
        oldMasterMeanMsPerMove: totalOldMoves > 0 ? totalOldMs / totalOldMoves : 0,
      }),
  );
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
`;

function parseArgs() {
  const args = process.argv.slice(2);
  let games = 20;
  for (let i = 0; i < args.length; i++) {
    if (args[i] === '--games' && i + 1 < args.length) {
      games = parseInt(args[i + 1], 10);
      i++;
    }
  }
  if (!Number.isFinite(games) || games < 1) throw new Error('--games must be a positive integer');
  return { games };
}

async function main() {
  const { games } = parseArgs();
  const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'v7-match-'));
  const driverPath = path.join(tmpDir, 'driver.ts');
  const bundlePath = path.join(tmpDir, 'driver.mjs');
  try {
    fs.writeFileSync(driverPath, DRIVER_TS.replaceAll('__REPO_ROOT__', REPO_ROOT));
    await esbuild.build({
      entryPoints: [driverPath],
      bundle: true,
      platform: 'node',
      format: 'esm',
      outfile: bundlePath,
      plugins: [rawPlugin],
      logLevel: 'error',
    });
    const run = spawnSync(process.execPath, [bundlePath], {
      encoding: 'utf-8',
      env: {
        ...process.env,
        V7_MATCH_CONFIG: JSON.stringify({ games, maxMoves: 225 }),
      },
      maxBuffer: 64 * 1024 * 1024,
      stdio: ['ignore', 'pipe', 'inherit'],
    });
    process.stdout.write(run.stdout ?? '');
    if (run.status !== 0) throw new Error(`match driver exited with status ${run.status}`);
  } finally {
    fs.rmSync(tmpDir, { recursive: true, force: true });
  }
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
