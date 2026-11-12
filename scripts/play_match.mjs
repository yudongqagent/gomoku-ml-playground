#!/usr/bin/env node
/**
 * Play a Python checkpoint against a TypeScript AI level (cross-check).
 *
 * This script is the bridge between the Python ML pipeline and the
 * TypeScript game AI (`src/ai`):
 *
 *   1. It embeds a small TypeScript driver (below) that imports
 *      `chooseMove` from `src/ai/engine` and `hasFive` from
 *      `src/engine/gomoku`, spawns the Python checkpoint player
 *      (`ml/scripts/play_position.py`) over a JSON-lines protocol,
 *      and plays N games alternating colours.
 *   2. It bundles that driver with esbuild (a transitive vite
 *      dependency, `node_modules/.bin/esbuild`) and runs the bundle.
 *   3. The bundle prints one line `MATCH_RESULT {...}` on stdout;
 *      everything else (per-game results, progress) goes to stderr.
 *
 * The driver source is written to a temp dir and deleted afterwards —
 * no extra files are left in the repo. `onnxruntime-web` is marked
 * external: it is only ever dynamically imported by the Master level,
 * which this driver never calls (levels here are `advanced` /
 * `expert`, both pure search).
 *
 * Usage (from the repo root):
 *   node scripts/play_match.mjs --checkpoint ml/runs/<id>/checkpoints/final.pt \
 *       --level advanced --games 4 --sims 50
 *   node scripts/play_match.mjs --checkpoint ... --level expert --games 4 \
 *       --sims 100 --ts-time-ms 500 --ts-sims 200
 *
 * Called by the eval harness via `gomoku_ml.evaluate.run_ts_match`
 * (`ml/scripts/evaluate.py --ts-opponent advanced|expert`).
 */

import { spawn, spawnSync } from 'node:child_process';
import * as fs from 'node:fs';
import * as os from 'node:os';
import * as path from 'node:path';
import { fileURLToPath } from 'node:url';

const REPO_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');

// ---------------------------------------------------------------------------
// Embedded TypeScript driver. Written to a temp file, bundled with esbuild,
// run, then deleted. `__REPO_ROOT__` is replaced with the absolute repo path
// before bundling (the driver lives in a temp dir, so relative imports
// would not resolve). NOTE: this template must not contain backticks or
// `${` sequences — it is a JS template literal.
// ---------------------------------------------------------------------------
const DRIVER_TS = `
import { chooseMove } from '__REPO_ROOT__/src/ai/engine';
import { hasFive } from '__REPO_ROOT__/src/engine/gomoku';
import { spawn, type ChildProcessWithoutNullStreams } from 'node:child_process';
import * as readline from 'node:readline';

interface MatchConfig {
  checkpoint: string;
  python: string;
  playerScript: string;
  level: 'advanced' | 'expert';
  games: number;
  sims: number;
  tsTimeMs: number;
  tsSims: number;
  maxMoves: number;
}

const cfg: MatchConfig = JSON.parse(process.env.PLAY_MATCH_CONFIG as string);

/** Sequential JSON-lines client for ml/scripts/play_position.py. */
class PyPlayer {
  private proc: ChildProcessWithoutNullStreams;
  private waiters: Array<{ resolve: (l: string) => void; reject: (e: Error) => void }> = [];
  private buffered: string[] = [];
  private dead = false;

  constructor(python: string, playerScript: string, checkpoint: string, sims: number) {
    this.proc = spawn(python, [playerScript, '--checkpoint', checkpoint, '--sims', String(sims)], {
      stdio: ['pipe', 'pipe', 'pipe'],
    });
    const rl = readline.createInterface({ input: this.proc.stdout });
    rl.on('line', (line) => {
      const w = this.waiters.shift();
      if (w) w.resolve(line);
      else this.buffered.push(line);
    });
    this.proc.stderr.on('data', (d) => process.stderr.write('[py] ' + d.toString()));
    this.proc.on('exit', (code) => {
      this.dead = true;
      const err = new Error('python player exited with code ' + code);
      for (const w of this.waiters.splice(0)) w.reject(err);
    });
  }

  private nextLine(): Promise<string> {
    const line = this.buffered.shift();
    if (line !== undefined) return Promise.resolve(line);
    if (this.dead) return Promise.reject(new Error('python player is dead'));
    return new Promise((resolve, reject) => this.waiters.push({ resolve, reject }));
  }

  async ready(): Promise<void> {
    const msg = JSON.parse(await this.nextLine()) as { event?: string };
    if (msg.event !== 'ready') throw new Error('python player did not send ready');
  }

  async request(obj: unknown): Promise<any> {
    if (this.dead) throw new Error('python player is dead');
    this.proc.stdin.write(JSON.stringify(obj) + '\\n');
    // Per-move watchdog: a stuck player must fail the match, not hang it.
    const timeout = new Promise<never>((_resolve, reject) =>
      setTimeout(() => reject(new Error('python player timed out (120s) on a move')), 120_000),
    );
    const line = await Promise.race([this.nextLine(), timeout]);
    return JSON.parse(line);
  }

  close(): void {
    try { this.proc.stdin.end(); } catch { /* already closed */ }
    this.proc.kill();
  }
}

type GameResult = 'py' | 'ts' | 'draw';

async function playGame(py: PyPlayer, gameIndex: number): Promise<{ result: GameResult; moves: number }> {
  const board = new Int8Array(225);
  const moves: Array<{ x: number; y: number }> = [];
  const pyIsBlack = gameIndex % 2 === 0;
  let toMove: 1 | -1 = 1;
  while (moves.length < cfg.maxMoves) {
    const pyTurn = (toMove === 1) === pyIsBlack;
    let move: { x: number; y: number };
    if (pyTurn) {
      const res = await py.request({
        board: Array.from(board),
        to_move: toMove,
        moves: moves.map((m) => [m.x, m.y]),
        sims: cfg.sims,
      });
      if (res.error || !res.move) {
        console.error('game ' + (gameIndex + 1) + ': python player error: ' + (res.error || 'no move') + ' — forfeit');
        return { result: 'ts', moves: moves.length };
      }
      move = { x: res.move[0], y: res.move[1] };
    } else {
      const res = await chooseMove({
        board: Array.from(board),
        toMove,
        level: cfg.level,
        options: { timeMs: cfg.tsTimeMs, sims: cfg.tsSims, temperature: 0 },
        history: moves,
      });
      move = res.move;
    }
    if (move.x < 0 || move.x > 14 || move.y < 0 || move.y > 14 || board[move.y * 15 + move.x] !== 0) {
      console.error('game ' + (gameIndex + 1) + ': illegal move (' + move.x + ',' + move.y + ') by ' + (pyTurn ? 'python' : 'ts') + ' — forfeit');
      return { result: pyTurn ? 'ts' : 'py', moves: moves.length };
    }
    board[move.y * 15 + move.x] = toMove;
    moves.push(move);
    if (hasFive(board, move.x, move.y)) {
      return { result: pyTurn ? 'py' : 'ts', moves: moves.length };
    }
    toMove = toMove === 1 ? -1 : 1;
  }
  return { result: 'draw', moves: moves.length };
}

async function main(): Promise<void> {
  const py = new PyPlayer(cfg.python, cfg.playerScript, cfg.checkpoint, cfg.sims);
  try {
    await py.ready();
    let wins = 0;
    let losses = 0;
    let draws = 0;
    const asBlack = { wins: 0, losses: 0, draws: 0 };
    const asWhite = { wins: 0, losses: 0, draws: 0 };
    for (let i = 0; i < cfg.games; i++) {
      const pyIsBlack = i % 2 === 0;
      const { result, moves } = await playGame(py, i);
      const bucket = pyIsBlack ? asBlack : asWhite;
      if (result === 'py') { wins++; bucket.wins++; }
      else if (result === 'ts') { losses++; bucket.losses++; }
      else { draws++; bucket.draws++; }
      const label = result === 'py' ? 'checkpoint wins' : result === 'ts' ? 'TS ' + cfg.level + ' wins' : 'draw';
      console.error('game ' + (i + 1) + '/' + cfg.games + ' (checkpoint ' + (pyIsBlack ? 'Black' : 'White') + '): ' + label + ' in ' + moves + ' moves');
    }
    const summary = {
      level: cfg.level,
      games: cfg.games,
      wins,
      losses,
      draws,
      win_rate: (wins + 0.5 * draws) / cfg.games,
      as_black: asBlack,
      as_white: asWhite,
      sims: cfg.sims,
      ts_time_ms: cfg.tsTimeMs,
      ts_sims: cfg.tsSims,
      checkpoint: cfg.checkpoint,
    };
    console.log('MATCH_RESULT ' + JSON.stringify(summary));
  } finally {
    py.close();
  }
}

main().catch((e) => {
  console.error('match driver failed: ' + (e instanceof Error ? e.message : String(e)));
  process.exit(1);
});
`;

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------
function parseArgs(argv) {
  const args = {
    checkpoint: null,
    level: 'advanced',
    games: 4,
    sims: 50,
    tsTimeMs: 500,
    tsSims: 200,
    maxMoves: 225,
    python: path.join(REPO_ROOT, 'ml', '.venv', 'bin', 'python'),
    json: false,
  };
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    const next = () => {
      if (i + 1 >= argv.length) throw new Error(a + ' needs a value');
      return argv[++i];
    };
    if (a === '--checkpoint') args.checkpoint = next();
    else if (a === '--level') args.level = next();
    else if (a === '--games') args.games = parseInt(next(), 10);
    else if (a === '--sims') args.sims = parseInt(next(), 10);
    else if (a === '--ts-time-ms') args.tsTimeMs = parseInt(next(), 10);
    else if (a === '--ts-sims') args.tsSims = parseInt(next(), 10);
    else if (a === '--max-moves') args.maxMoves = parseInt(next(), 10);
    else if (a === '--python') args.python = next();
    else if (a === '--json') args.json = true;
    else if (a === '--help' || a === '-h') {
      console.log(
        'Usage: node scripts/play_match.mjs --checkpoint <pt> [--level advanced|expert] ' +
          '[--games N] [--sims N] [--ts-time-ms N] [--ts-sims N] [--max-moves N] [--python <path>]',
      );
      process.exit(0);
    } else throw new Error('unknown argument: ' + a);
  }
  if (!args.checkpoint) throw new Error('--checkpoint is required');
  if (args.level !== 'advanced' && args.level !== 'expert') {
    throw new Error("--level must be 'advanced' or 'expert', got " + args.level);
  }
  if (!Number.isInteger(args.games) || args.games < 1) throw new Error('--games must be >= 1');
  if (!Number.isInteger(args.sims) || args.sims < 1) throw new Error('--sims must be >= 1');
  return args;
}

function run(cmd, cmdArgs, opts) {
  const r = spawnSync(cmd, cmdArgs, { encoding: 'utf-8', ...opts });
  if (r.error) throw new Error('failed to run ' + cmd + ': ' + r.error.message);
  return r;
}

async function main() {
  let args;
  try {
    args = parseArgs(process.argv.slice(2));
  } catch (e) {
    console.error('play_match: ' + e.message);
    process.exit(2);
  }
  const checkpointPath = path.resolve(args.checkpoint);
  if (!fs.existsSync(checkpointPath)) {
    console.error('play_match: checkpoint not found: ' + checkpointPath);
    process.exit(2);
  }
  const playerScript = path.join(REPO_ROOT, 'ml', 'scripts', 'play_position.py');
  if (!fs.existsSync(playerScript)) {
    console.error('play_match: python player not found: ' + playerScript);
    process.exit(2);
  }
  const esbuild = path.join(REPO_ROOT, 'node_modules', '.bin', 'esbuild');
  if (!fs.existsSync(esbuild)) {
    console.error(
      'play_match: esbuild not found at ' + esbuild + ' — run `npm install` first.',
    );
    process.exit(2);
  }

  const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'gomoku-match-'));
  const driverPath = path.join(tmpDir, 'driver.ts');
  const bundlePath = path.join(tmpDir, 'driver.cjs');
  try {
    fs.writeFileSync(driverPath, DRIVER_TS.replaceAll('__REPO_ROOT__', REPO_ROOT));
    const bundled = run(esbuild, [
      driverPath,
      '--bundle',
      '--platform=node',
      '--format=cjs',
      '--external:onnxruntime-web',
      '--log-level=warning',
      '--outfile=' + bundlePath,
    ]);
    if (bundled.status !== 0) {
      console.error('play_match: esbuild failed:\n' + (bundled.stderr || bundled.stdout));
      process.exit(1);
    }
    const config = {
      checkpoint: checkpointPath,
      python: args.python,
      playerScript,
      level: args.level,
      games: args.games,
      sims: args.sims,
      tsTimeMs: args.tsTimeMs,
      tsSims: args.tsSims,
      maxMoves: args.maxMoves,
    };
    const code = await new Promise((resolve, reject) => {
      const child = spawn(process.execPath, [bundlePath], {
        stdio: 'inherit',
        env: { ...process.env, PLAY_MATCH_CONFIG: JSON.stringify(config) },
      });
      child.on('error', reject);
      child.on('exit', (c) => resolve(c ?? 1));
    });
    process.exit(code);
  } finally {
    fs.rmSync(tmpDir, { recursive: true, force: true });
  }
}

main().catch((e) => {
  console.error('play_match: ' + (e instanceof Error ? e.message : String(e)));
  process.exit(1);
});
