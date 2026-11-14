#!/usr/bin/env node
/**
 * Execute the WGSL scoring kernel (src/kernel/scoring.wgsl) under Dawn in
 * Node and compare its output against the frozen Python reference scores
 * (ml/tests/kernel_fixtures.json) on all 60 fixtures x 225 cells.
 *
 * This is a MANUAL verification tool, not part of `npm test`: it needs the
 * `webgpu` npm package (Dawn's Node binding), which is deliberately NOT a
 * project dependency, and a Vulkan driver — on a headless Linux VM the
 * SwiftShader software driver works (it ships with Chromium).
 *
 * Setup (one time, scratch dir outside the repo):
 *   mkdir -p /tmp/wgsl-check && cd /tmp/wgsl-check
 *   npm init -y && npm install webgpu@0.6.2
 *
 * Run (from the repo root; WEBGPU_PKG_PATH points at the installed
 * package — ES modules do not consult NODE_PATH):
 *   VK_ICD_FILENAMES=/opt/meta-chromium/vk_swiftshader_icd.json \
 *     WEBGPU_PKG_PATH=/tmp/wgsl-check/node_modules/webgpu \
 *     node scripts/check_wgsl_dawn.mjs
 *
 * Exit code 0 means 0 mismatches over all 13,500 cells. The timing printed
 * at the end is a SwiftShader software number on the machine it runs on —
 * it is NOT an A20 Pro / on-device number, only a smoke check that dispatch
 * works repeatedly. Real on-device speed comes only from the Benchmark tab
 * in the web app, run on the device itself.
 *
 * Verified 2026-11-12: 0 mismatches over 13,500 cells (60 positions),
 * Dawn via webgpu@0.6.2 + SwiftShader on the Linux VM.
 *
 * The dispatch logic here mirrors src/kernel/scoring.ts exactly (same
 * buffer layout, same workgroup math), so a pass validates the shader AND
 * the wrapper's dispatch code. The browser benchmark mode's self-check
 * remains the on-device execution path.
 */
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

// The `webgpu` package lives in a scratch dir, not in node_modules, so it
// is imported by explicit path (WEBGPU_PKG_PATH, default /tmp/wgsl-check).
const WEBGPU_PKG_PATH =
  process.env.WEBGPU_PKG_PATH ?? '/tmp/wgsl-check/node_modules/webgpu';
const { create, globals } = await import(
  pathToFileURL(path.join(WEBGPU_PKG_PATH, 'index.js')).href
);

Object.assign(globalThis, globals);

const REPO = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const shaderSource = fs.readFileSync(path.join(REPO, 'src/kernel/scoring.wgsl'), 'utf-8');
const fixtures = JSON.parse(
  fs.readFileSync(path.join(REPO, 'ml/tests/kernel_fixtures.json'), 'utf-8'),
).fixtures;

const CELLS = 225;
const WORKGROUP_SIZE = 64;

async function main() {
  const backend = process.argv[2]; // e.g. 'backend=vulkan'
  const gpu = create(backend ? [backend] : []);
  const adapter = await gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) {
    console.log('NO ADAPTER: requestAdapter returned null (no Vulkan driver found?)');
    process.exit(2);
  }
  console.log('adapter obtained');
  const device = await adapter.requestDevice();
  console.log('device obtained');

  const n = fixtures.length;
  const boardData = new Int32Array(n * CELLS);
  const playerData = new Int32Array(n);
  for (let i = 0; i < n; i++) {
    for (let c = 0; c < CELLS; c++) boardData[i * CELLS + c] = fixtures[i].board[c];
    playerData[i] = fixtures[i].to_move;
  }

  const boardBuffer = device.createBuffer({
    size: boardData.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(boardBuffer, 0, boardData);
  const playerBuffer = device.createBuffer({
    size: playerData.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(playerBuffer, 0, playerData);
  const scoreBuffer = device.createBuffer({
    size: boardData.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });
  const readbackBuffer = device.createBuffer({
    size: boardData.byteLength,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  });

  const module = device.createShaderModule({ code: shaderSource });
  const pipeline = device.createComputePipeline({
    layout: 'auto',
    compute: { module, entryPoint: 'main' },
  });
  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: boardBuffer } },
      { binding: 1, resource: { buffer: playerBuffer } },
      { binding: 2, resource: { buffer: scoreBuffer } },
    ],
  });

  async function runOnce() {
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(Math.ceil((n * CELLS) / WORKGROUP_SIZE));
    pass.end();
    encoder.copyBufferToBuffer(scoreBuffer, 0, readbackBuffer, 0, scoreBuffer.size);
    device.queue.submit([encoder.finish()]);
    await readbackBuffer.mapAsync(GPUMapMode.READ);
    const copy = new Int32Array(readbackBuffer.getMappedRange().slice(0));
    readbackBuffer.unmap();
    return copy;
  }

  const gpuScores = await runOnce();
  let mismatches = 0;
  let first = null;
  for (let i = 0; i < n; i++) {
    for (let c = 0; c < CELLS; c++) {
      const got = gpuScores[i * CELLS + c];
      const want = fixtures[i].scores[c];
      if (got !== want) {
        mismatches++;
        if (!first) first = `${fixtures[i].id} cell ${c}: got ${got}, want ${want}`;
      }
    }
  }
  console.log(
    `WGSL vs frozen Python: ${mismatches} mismatches over ${n * CELLS} cells (${n} positions)`,
  );
  if (first) console.log('first mismatch:', first);

  // Timing smoke check only — SwiftShader software on this machine, NOT
  // on-device speed (see the header comment).
  const t0 = performance.now();
  const runs = 10;
  for (let r = 0; r < runs; r++) await runOnce();
  const dt = (performance.now() - t0) / 1000;
  console.log(
    `timed ${runs} runs x ${n} positions in ${dt.toFixed(2)}s: ` +
      `${((runs * n) / dt).toFixed(1)} pos/s (Dawn + SwiftShader on this machine — software path, not device speed)`,
  );
  device.destroy();
  process.exit(mismatches === 0 ? 0 : 1);
}

main().catch((e) => {
  console.error('FAILED:', e);
  process.exit(3);
});
