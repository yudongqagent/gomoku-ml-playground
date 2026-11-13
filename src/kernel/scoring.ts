// Scoring kernel v0 — JavaScript reference + WebGPU (WGSL) wrapper.
//
// The JS reference (scoreAllJs) is an exact port of
// ml/gomoku_ml/baseline.py score_move over all 225 cells. The WebGPU path
// (GpuScorer) runs the same algorithm as a WGSL compute shader
// (scoring.wgsl). Both must produce identical scores; the benchmark view
// self-checks that on the fixed fixture set (fixtures.ts) and also checks
// the JS reference against the frozen Python scores embedded there.
//
// WGSL does not execute in vitest (no WebGPU there), so the vitest suite
// only checks the JS reference. The WGSL path itself WAS executed and
// verified: scripts/check_wgsl_dawn.mjs runs the shader under Dawn +
// SwiftShader in Node and compares it against the frozen Python scores —
// 0 mismatches over 13,500 cells (60 positions), verified 2026-11-12.
// The benchmark mode's self-check remains the on-device execution path:
// it re-verifies WGSL vs JS vs frozen Python on every run, in the browser.
// Nothing in this module runs at import time except table construction,
// so importing it in Node is safe.

import shaderSource from './scoring.wgsl?raw';
import type { KernelFixture } from './fixtures';

export const SIZE = 15;
export const CELLS = SIZE * SIZE;

// Weight table indexed by min(run, 5) * 3 + open_ends — baseline._WEIGHTS.
const W = [
  0, 0, 0,
  0, 1, 10, // run 1
  1, 10, 100, // run 2
  10, 100, 1000, // run 3
  100, 1000, 10000, // run 4
  100000, 100000, 100000, // run 5
] as const;

// Opponent weights: int(0.9 * w) per entry of W, computed with float64
// arithmetic exactly as Python does (Math.trunc(0.9 * w) === int(0.9 * w)
// for these values, since JS numbers are float64 like Python floats).
const W_OPP = W.map((w) => Math.trunc(0.9 * w));

const DIRS: ReadonlyArray<readonly [number, number]> = [
  [1, 0],
  [0, 1],
  [1, 1],
  [1, -1],
];

/** Parse a fixture's 225-char board string into a flat Int8Array. */
export function fixtureToBoard(fixture: KernelFixture): Int8Array {
  const board = new Int8Array(CELLS);
  for (let i = 0; i < CELLS; i++) {
    const c = fixture.board[i];
    board[i] = c === 'B' ? 1 : c === 'W' ? -1 : 0;
  }
  return board;
}

function runAndEnds(
  board: Int8Array,
  x: number,
  y: number,
  dx: number,
  dy: number,
  color: number,
): [number, number] {
  let run = 1;
  let ends = 0;
  let nx = x + dx;
  let ny = y + dy;
  while (nx >= 0 && nx < SIZE && ny >= 0 && ny < SIZE && board[ny * SIZE + nx] === color) {
    run++;
    nx += dx;
    ny += dy;
  }
  if (nx >= 0 && nx < SIZE && ny >= 0 && ny < SIZE && board[ny * SIZE + nx] === 0) ends++;
  nx = x - dx;
  ny = y - dy;
  while (nx >= 0 && nx < SIZE && ny >= 0 && ny < SIZE && board[ny * SIZE + nx] === color) {
    run++;
    nx -= dx;
    ny -= dy;
  }
  if (nx >= 0 && nx < SIZE && ny >= 0 && ny < SIZE && board[ny * SIZE + nx] === 0) ends++;
  return [run, ends];
}

/**
 * Exact JS port of baseline.score_move for every cell: returns an
 * Int32Array of 225 scores (flat index y*15+x) for `player` to move.
 * Scores occupied cells too, exactly as score_move does.
 */
export function scoreAllJs(board: Int8Array, player: 1 | -1): Int32Array {
  const out = new Int32Array(CELLS);
  for (let cell = 0; cell < CELLS; cell++) {
    const x = cell % SIZE;
    const y = Math.floor(cell / SIZE);
    let total = 7 - (Math.abs(x - 7) + Math.abs(y - 7));
    for (const [dx, dy] of DIRS) {
      const [run, ends] = runAndEnds(board, x, y, dx, dy, player);
      total += W[Math.min(run, 5) * 3 + ends];
      const [runO, endsO] = runAndEnds(board, x, y, dx, dy, -player);
      total += W_OPP[Math.min(runO, 5) * 3 + endsO];
    }
    out[cell] = total;
  }
  return out;
}

/** Count cells where two score arrays differ (same length required). */
export function countMismatches(a: ArrayLike<number>, b: ArrayLike<number>): number {
  let n = 0;
  for (let i = 0; i < a.length; i++) {
    if (a[i] !== b[i]) n++;
  }
  return n;
}

// ---------------------------------------------------------------------------
// WebGPU wrapper
// ---------------------------------------------------------------------------

export interface GpuUnavailable {
  available: false;
  reason: string;
}

export interface GpuAvailable {
  available: true;
  adapterInfo: string;
}

export type GpuStatus = GpuAvailable | GpuUnavailable;

/** Probe for WebGPU without creating a device. Never throws. */
export function probeWebGpu(): GpuStatus {
  if (typeof navigator === 'undefined' || !navigator.gpu) {
    return {
      available: false,
      reason:
        'WebGPU is not available in this browser (navigator.gpu is undefined). ' +
        'On iPhone, WebGPU needs Safari 26+ (iOS 26+). The JavaScript reference benchmark below still runs.',
    };
  }
  return { available: true, adapterInfo: '' };
}

const WORKGROUP_SIZE = 64;

/** A prepared batch of positions on the GPU; buffers are created once and reused across runs. */
export class PreparedBatch {
  private readonly scorer: GpuScorer;
  private readonly boardBuffer: GPUBuffer;
  private readonly playerBuffer: GPUBuffer;
  private readonly scoreBuffer: GPUBuffer;
  private readonly readbackBuffer: GPUBuffer;
  private readonly bindGroup: GPUBindGroup;
  private readonly workgroups: number;
  readonly positionCount: number;

  constructor(
    scorer: GpuScorer,
    boards: Int8Array[],
    players: ReadonlyArray<1 | -1>,
  ) {
    this.scorer = scorer;
    this.positionCount = boards.length;
    const device = scorer.device;
    const n = boards.length;
    const boardData = new Int32Array(n * CELLS);
    for (let i = 0; i < n; i++) {
      for (let c = 0; c < CELLS; c++) boardData[i * CELLS + c] = boards[i][c];
    }
    const playerData = new Int32Array(players);

    this.boardBuffer = device.createBuffer({
      label: 'scoring boards',
      size: boardData.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(this.boardBuffer, 0, boardData);
    this.playerBuffer = device.createBuffer({
      label: 'scoring players',
      size: playerData.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(this.playerBuffer, 0, playerData);
    this.scoreBuffer = device.createBuffer({
      label: 'scoring output',
      size: boardData.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
    this.readbackBuffer = device.createBuffer({
      label: 'scoring readback',
      size: boardData.byteLength,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
    this.bindGroup = device.createBindGroup({
      label: 'scoring bind group',
      layout: scorer.pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.boardBuffer } },
        { binding: 1, resource: { buffer: this.playerBuffer } },
        { binding: 2, resource: { buffer: this.scoreBuffer } },
      ],
    });
    this.workgroups = Math.ceil((n * CELLS) / WORKGROUP_SIZE);
  }

  /** Dispatch the shader over the whole batch and read back all scores (flat, N*225). */
  async run(): Promise<Int32Array> {
    const device = this.scorer.device;
    const encoder = device.createCommandEncoder({ label: 'scoring encoder' });
    const pass = encoder.beginComputePass({ label: 'scoring pass' });
    pass.setPipeline(this.scorer.pipeline);
    pass.setBindGroup(0, this.bindGroup);
    pass.dispatchWorkgroups(this.workgroups);
    pass.end();
    encoder.copyBufferToBuffer(
      this.scoreBuffer,
      0,
      this.readbackBuffer,
      0,
      this.scoreBuffer.size,
    );
    device.queue.submit([encoder.finish()]);
    await this.readbackBuffer.mapAsync(GPUMapMode.READ);
    const copy = new Int32Array(this.readbackBuffer.getMappedRange().slice(0));
    this.readbackBuffer.unmap();
    return copy;
  }

  destroy(): void {
    this.boardBuffer.destroy();
    this.playerBuffer.destroy();
    this.scoreBuffer.destroy();
    this.readbackBuffer.destroy();
  }
}

export class GpuScorer {
  readonly device: GPUDevice;
  readonly pipeline: GPUComputePipeline;
  readonly adapterInfo: string;

  private constructor(device: GPUDevice, pipeline: GPUComputePipeline, adapterInfo: string) {
    this.device = device;
    this.pipeline = pipeline;
    this.adapterInfo = adapterInfo;
  }

  /** Create a scorer, or return null with a reason when WebGPU is unavailable. Never throws. */
  static async create(): Promise<{ scorer: GpuScorer } | { scorer: null; reason: string }> {
    const probe = probeWebGpu();
    if (!probe.available) return { scorer: null, reason: probe.reason };
    try {
      const adapter = await navigator.gpu!.requestAdapter({ powerPreference: 'high-performance' });
      if (!adapter) {
        return {
          scorer: null,
          reason: 'WebGPU is present but no adapter could be created on this device.',
        };
      }
      let adapterInfo = 'unknown adapter';
      try {
        if (adapter.requestAdapterInfo) {
          const info = await adapter.requestAdapterInfo();
          adapterInfo = `${info.vendor} ${info.device} (${info.architecture})`.trim();
        }
      } catch {
        // Adapter info is optional; proceed without it.
      }
      const device = await adapter.requestDevice();
      const module = device.createShaderModule({ label: 'scoring kernel', code: shaderSource });
      const pipeline = device.createComputePipeline({
        label: 'scoring pipeline',
        layout: 'auto',
        compute: { module, entryPoint: 'main' },
      });
      return { scorer: new GpuScorer(device, pipeline, adapterInfo) };
    } catch (err) {
      return {
        scorer: null,
        reason: `WebGPU initialisation failed: ${err instanceof Error ? err.message : String(err)}`,
      };
    }
  }

  prepareBatch(boards: Int8Array[], players: ReadonlyArray<1 | -1>): PreparedBatch {
    if (boards.length === 0) throw new Error('prepareBatch needs at least one board');
    if (boards.length !== players.length) {
      throw new Error(`boards (${boards.length}) and players (${players.length}) must match`);
    }
    return new PreparedBatch(this, boards, players);
  }

  destroy(): void {
    this.device.destroy();
  }
}

// ---------------------------------------------------------------------------
// Benchmark
// ---------------------------------------------------------------------------

export interface BenchmarkSelfCheck {
  /** Positions checked (fixture count). */
  positions: number;
  /** Cells compared per check (positions * 225). */
  cells: number;
  /** WGSL vs JS reference mismatches (must be 0). */
  wgslVsJsMismatches: number;
  /** JS reference vs frozen Python scores mismatches (must be 0). */
  jsVsPythonMismatches: number;
}

export interface BenchmarkResult {
  selfCheck: BenchmarkSelfCheck;
  /** Timed WebGPU runs (each run scores all fixture positions once). */
  gpuRuns: number;
  /** Total positions scored in the timed WebGPU runs. */
  gpuPositions: number;
  /** Wall-clock seconds of the timed WebGPU runs (incl. readback). */
  gpuSeconds: number;
  gpuPositionsPerSec: number;
  gpuMsPerPosition: number;
  /** JS reference timing over the same fixtures (same run count). */
  jsPositionsPerSec: number;
  jsMsPerPosition: number;
  adapterInfo: string;
}

export interface BenchmarkProgress {
  phase: 'self-check' | 'gpu-warmup' | 'gpu-timing' | 'js-timing' | 'done';
  done: number;
  total: number;
}

export interface BenchmarkOptions {
  /** Warmup runs before timing (default 3). */
  warmupRuns?: number;
  /** Timed runs (default 20). Each run scores all fixtures once. */
  timedRuns?: number;
  onProgress?: (p: BenchmarkProgress) => void;
}

/**
 * Run the full benchmark: self-check (WGSL vs JS vs frozen Python) then
 * timed WebGPU and JS runs over the fixture set. Throws on GPU errors;
 * the caller (BenchmarkView) turns that into a visible error, never into
 * fake numbers.
 */
export async function runBenchmark(
  scorer: GpuScorer,
  fixtures: KernelFixture[],
  options: BenchmarkOptions = {},
): Promise<BenchmarkResult> {
  const warmupRuns = options.warmupRuns ?? 3;
  const timedRuns = options.timedRuns ?? 20;
  const onProgress = options.onProgress ?? (() => {});
  const boards = fixtures.map(fixtureToBoard);
  const players = fixtures.map((f) => f.toMove);
  const batch = scorer.prepareBatch(boards, players);
  try {
    // -- Self-check: one GPU run, compared cell-by-cell. --
    onProgress({ phase: 'self-check', done: 0, total: 1 });
    const gpuScores = await batch.run();
    let wgslVsJs = 0;
    let jsVsPython = 0;
    for (let i = 0; i < fixtures.length; i++) {
      const js = scoreAllJs(boards[i], players[i]);
      const gpuSlice = gpuScores.subarray(i * CELLS, (i + 1) * CELLS);
      wgslVsJs += countMismatches(gpuSlice, js);
      jsVsPython += countMismatches(js, fixtures[i].scores);
    }
    onProgress({ phase: 'self-check', done: 1, total: 1 });

    // -- Warmup (untimed). --
    for (let r = 0; r < warmupRuns; r++) {
      onProgress({ phase: 'gpu-warmup', done: r, total: warmupRuns });
      await batch.run();
    }

    // -- Timed GPU runs. --
    const t0 = performance.now();
    for (let r = 0; r < timedRuns; r++) {
      onProgress({ phase: 'gpu-timing', done: r, total: timedRuns });
      await batch.run();
    }
    const gpuSeconds = (performance.now() - t0) / 1000;
    const gpuPositions = timedRuns * fixtures.length;

    // -- Timed JS reference runs (same fixtures, same run count). --
    onProgress({ phase: 'js-timing', done: 0, total: timedRuns });
    const jt0 = performance.now();
    for (let r = 0; r < timedRuns; r++) {
      for (let i = 0; i < fixtures.length; i++) scoreAllJs(boards[i], players[i]);
      onProgress({ phase: 'js-timing', done: r + 1, total: timedRuns });
    }
    const jsSeconds = (performance.now() - jt0) / 1000;

    onProgress({ phase: 'done', done: 1, total: 1 });
    return {
      selfCheck: {
        positions: fixtures.length,
        cells: fixtures.length * CELLS,
        wgslVsJsMismatches: wgslVsJs,
        jsVsPythonMismatches: jsVsPython,
      },
      gpuRuns: timedRuns,
      gpuPositions,
      gpuSeconds,
      gpuPositionsPerSec: gpuPositions / gpuSeconds,
      gpuMsPerPosition: (gpuSeconds * 1000) / gpuPositions,
      jsPositionsPerSec: gpuPositions / jsSeconds,
      jsMsPerPosition: (jsSeconds * 1000) / gpuPositions,
      adapterInfo: scorer.adapterInfo,
    };
  } finally {
    batch.destroy();
  }
}

/** JS-reference-only benchmark (used when WebGPU is unavailable — real JS numbers, labelled as such). */
export function runJsOnlyBenchmark(
  fixtures: KernelFixture[],
  runs: number,
): {
  selfCheck: Pick<BenchmarkSelfCheck, 'positions' | 'cells' | 'jsVsPythonMismatches'>;
  positions: number;
  seconds: number;
  positionsPerSec: number;
  msPerPosition: number;
} {
  const boards = fixtures.map(fixtureToBoard);
  const players = fixtures.map((f) => f.toMove);
  let jsVsPython = 0;
  for (let i = 0; i < fixtures.length; i++) {
    jsVsPython += countMismatches(scoreAllJs(boards[i], players[i]), fixtures[i].scores);
  }
  const t0 = performance.now();
  for (let r = 0; r < runs; r++) {
    for (let i = 0; i < fixtures.length; i++) scoreAllJs(boards[i], players[i]);
  }
  const seconds = (performance.now() - t0) / 1000;
  const positions = runs * fixtures.length;
  return {
    selfCheck: {
      positions: fixtures.length,
      cells: fixtures.length * CELLS,
      jsVsPythonMismatches: jsVsPython,
    },
    positions,
    seconds,
    positionsPerSec: positions / seconds,
    msPerPosition: (seconds * 1000) / positions,
  };
}
