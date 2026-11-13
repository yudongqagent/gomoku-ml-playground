# Scoring kernel v0 — web side

The v0 scoring function (`ml/gomoku_ml/baseline.py` `score_move`) ported to
the browser, twice: a JavaScript reference and a WGSL compute shader. Both
must produce scores identical to the Python reference on every cell,
including occupied cells (the reference scores all 225 cells
unconditionally).

## Files

| File | What it is |
| --- | --- |
| `scoring.wgsl` | WGSL compute shader: one thread per (position, cell), batches N positions per dispatch. |
| `scoring.ts` | JS reference (`scoreAllJs`), WebGPU wrapper (`GpuScorer`, `PreparedBatch`), benchmark runner (`runBenchmark`, `runJsOnlyBenchmark`). |
| `fixtures.ts` | GENERATED (by `ml/scripts/generate_kernel_fixtures.py`) — 60 fixed positions with frozen Python reference scores. Do not hand-edit. |
| `webgpu.d.ts` | Minimal WebGPU type declarations (TS's DOM lib does not type `navigator.gpu`). Delete if `@webgpu/types` is ever added. |
| `scoring.test.ts` | Vitest: JS reference vs frozen Python scores on all 13,500 cells. |

The benchmark UI is `src/components/BenchmarkView.tsx`, wired in as the
⚡ Benchmark tab in `src/App.tsx`. It self-checks WGSL vs JS vs frozen
Python before timing, and degrades gracefully when WebGPU is unavailable
(clear message, no fake numbers; the JS-reference benchmark still runs).

## Faithfulness notes (v0 quirks are ported, not fixed)

- Runs are contiguous only — the broken-four blind spot (`XX_XX` scores as
  two runs) is in all three implementations, exactly as in `baseline.py`.
- The opponent weight table is `int(0.9 * w)` computed in Python float64
  and hardcoded in WGSL. It is NOT computed in-shader: WGSL `f32`
  `0.9 * 100000` truncates to 89999, while Python gives 90000. The JS
  reference computes `Math.trunc(0.9 * w)` with float64, identical to Python.
- Centre preference `7 - (|x-7| + |y-7|)` is included in all three.

## Verification status (2026-11-12)

| Check | Result |
| --- | --- |
| JS reference vs frozen Python scores (vitest) | 0 mismatches over 13,500 cells (60 positions × 225) |
| WGSL vs frozen Python scores (Dawn + SwiftShader, `scripts/check_wgsl_dawn.mjs`) | 0 mismatches over 13,500 cells |
| WGSL vs JS reference | Re-checked on every benchmark run in the browser (self-check); must be 0 |
| NumPy kernel vs Python reference | 0 mismatches over 13,500 cells — see `ml/tests/test_kernel_numpy.py` |

On-device (A20 Pro) speed is measured only by running the Benchmark tab on
the device itself — see `research/iphone-benchmark.md`. No on-device
numbers exist yet. The SwiftShader timing printed by
`check_wgsl_dawn.mjs` is a software-emulation number, not device speed.
