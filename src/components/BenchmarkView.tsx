// Benchmark view: measures the v0 scoring kernel on this device.
//
// Runs the WGSL compute shader over a fixed set of 60 positions (20
// tactical, 3 structured, 37 random midgame) and reports positions/sec and
// ms/position. Before timing, it self-checks that the WGSL scores match the
// JavaScript reference on every cell, and that the JS reference matches the
// frozen Python reference scores — both mismatch counts must be 0, and the
// run is flagged if they are not.
//
// If WebGPU is unavailable (e.g. an older browser), the view says so
// plainly and offers the JavaScript-reference benchmark instead. It never
// shows GPU numbers without a GPU.

import { useCallback, useState } from 'react';
import { KERNEL_FIXTURES } from '../kernel/fixtures';
import {
  GpuScorer,
  probeWebGpu,
  runBenchmark,
  runJsOnlyBenchmark,
  type BenchmarkProgress,
  type BenchmarkResult,
} from '../kernel/scoring';

type RunState =
  | { status: 'idle' }
  | { status: 'probing' }
  | { status: 'running'; progress: BenchmarkProgress }
  | { status: 'done'; result: BenchmarkResult }
  | { status: 'js-only'; result: ReturnType<typeof runJsOnlyBenchmark> }
  | { status: 'error'; message: string }
  | { status: 'no-webgpu'; reason: string };

const TIMED_RUNS = 20;

function phaseText(p: BenchmarkProgress): string {
  switch (p.phase) {
    case 'self-check':
      return 'Self-checking WGSL vs JS vs Python…';
    case 'gpu-warmup':
      return `GPU warmup (${p.done}/${p.total})…`;
    case 'gpu-timing':
      return `Timing GPU (${p.done}/${p.total} runs)…`;
    case 'js-timing':
      return `Timing JS reference (${p.done}/${p.total} runs)…`;
    case 'done':
      return 'Done.';
  }
}

export default function BenchmarkView() {
  const [state, setState] = useState<RunState>({ status: 'idle' });

  const run = useCallback(async () => {
    setState({ status: 'probing' });
    const probe = probeWebGpu();
    if (!probe.available) {
      setState({ status: 'no-webgpu', reason: probe.reason });
      return;
    }
    const created = await GpuScorer.create();
    if (!created.scorer) {
      setState({ status: 'no-webgpu', reason: created.reason });
      return;
    }
    const scorer = created.scorer;
    try {
      const result = await runBenchmark(scorer, KERNEL_FIXTURES, {
        timedRuns: TIMED_RUNS,
        onProgress: (progress) => setState({ status: 'running', progress }),
      });
      setState({ status: 'done', result });
    } catch (err) {
      setState({
        status: 'error',
        message: err instanceof Error ? err.message : String(err),
      });
    } finally {
      scorer.destroy();
    }
  }, []);

  const runJsOnly = useCallback(() => {
    setState({ status: 'running', progress: { phase: 'js-timing', done: 0, total: TIMED_RUNS } });
    // Let the UI paint before the synchronous JS benchmark blocks the thread.
    window.setTimeout(() => {
      try {
        const result = runJsOnlyBenchmark(KERNEL_FIXTURES, TIMED_RUNS);
        setState({ status: 'js-only', result });
      } catch (err) {
        setState({
          status: 'error',
          message: err instanceof Error ? err.message : String(err),
        });
      }
    }, 50);
  }, []);

  const busy = state.status === 'probing' || state.status === 'running';

  return (
    <div className="benchmark-layout">
      <section className="panel">
        <h2 className="panel-title">⚡ Scoring-kernel benchmark</h2>
        <p className="panel-sub">
          Measures the v0 scoring kernel (the heuristic from{' '}
          <code>ml/gomoku_ml/baseline.py</code>) on this device: the WGSL compute shader scores
          all 225 cells of each position in parallel. Fixed set: {KERNEL_FIXTURES.length}{' '}
          positions (20 tactical, 3 structured, 37 random midgame). Each timed run scores all{' '}
          {KERNEL_FIXTURES.length} positions once; the benchmark runs {TIMED_RUNS} timed runs
          after 3 warmup runs. GPU timing includes the readback of results.
        </p>

        <div className="bench-actions">
          <button type="button" className="btn primary" onClick={() => void run()} disabled={busy}>
            {busy ? 'Running…' : 'Run benchmark'}
          </button>
          <button type="button" className="btn" onClick={runJsOnly} disabled={busy}>
            Run JS-reference benchmark only
          </button>
        </div>

        {state.status === 'running' && (
          <p className="bench-status" role="status">
            {phaseText(state.progress)}
          </p>
        )}

        {state.status === 'no-webgpu' && (
          <div className="bench-notice" role="alert">
            <p>
              <strong>WebGPU is not available here.</strong>
            </p>
            <p>{state.reason}</p>
            <p>
              No GPU numbers are shown because none were measured. The JS-reference benchmark
              above still gives real (slower) numbers for this device.
            </p>
          </div>
        )}

        {state.status === 'error' && (
          <p className="error-text" role="alert">
            Benchmark failed: {state.message} — no numbers are reported.
          </p>
        )}

        {state.status === 'done' && (
          <BenchmarkResults result={state.result} />
        )}

        {state.status === 'js-only' && (
          <div className="bench-results">
            <h3>JS-reference benchmark (no WebGPU)</h3>
            <p className="bench-note">
              These are JavaScript numbers on this device, not GPU numbers.
            </p>
            <table className="bench-table">
              <tbody>
                <tr>
                  <th scope="row">Self-check: JS vs frozen Python scores</th>
                  <td>
                    {state.result.selfCheck.jsVsPythonMismatches} mismatches over{' '}
                    {state.result.selfCheck.cells.toLocaleString()} cells (
                    {state.result.selfCheck.positions} positions)
                  </td>
                </tr>
                <tr>
                  <th scope="row">Positions scored</th>
                  <td>
                    {state.result.positions} ({TIMED_RUNS} runs × {KERNEL_FIXTURES.length}{' '}
                    positions) in {state.result.seconds.toFixed(2)} s
                  </td>
                </tr>
                <tr>
                  <th scope="row">JS reference throughput</th>
                  <td>
                    {state.result.positionsPerSec.toFixed(1)} positions/sec ·{' '}
                    {state.result.msPerPosition.toFixed(3)} ms/position
                  </td>
                </tr>
              </tbody>
            </table>
          </div>
        )}
      </section>

      <section className="panel">
        <h2 className="panel-title">What the numbers mean</h2>
        <p className="panel-sub">
          One “position” is a full 225-cell scoring pass (every empty and occupied point gets a
          score, exactly as the Python reference does). The self-check compares the WGSL shader
          against the JavaScript reference and the JS reference against frozen Python scores on
          all {KERNEL_FIXTURES.length * 225} cells — both mismatch counts must be 0 for the
          benchmark to be trusted. The on-device numbers from this page are the only real
          A20 Pro measurements; nothing here is estimated or simulated.
        </p>
      </section>
    </div>
  );
}

function BenchmarkResults({ result }: { result: BenchmarkResult }) {
  const { selfCheck } = result;
  const trusted = selfCheck.wgslVsJsMismatches === 0 && selfCheck.jsVsPythonMismatches === 0;
  return (
    <div className="bench-results">
      <h3>Results</h3>
      <p className="bench-note">Adapter: {result.adapterInfo}</p>
      {!trusted && (
        <p className="error-text" role="alert">
          Self-check FAILED — the WGSL and JS scores disagree, so these numbers must not be
          trusted or reported.
        </p>
      )}
      <table className="bench-table">
        <tbody>
          <tr>
            <th scope="row">Self-check: WGSL vs JS reference</th>
            <td>
              {selfCheck.wgslVsJsMismatches} mismatches over {selfCheck.cells.toLocaleString()}{' '}
              cells ({selfCheck.positions} positions)
            </td>
          </tr>
          <tr>
            <th scope="row">Self-check: JS vs frozen Python scores</th>
            <td>
              {selfCheck.jsVsPythonMismatches} mismatches over {selfCheck.cells.toLocaleString()}{' '}
              cells
            </td>
          </tr>
          <tr>
            <th scope="row">WebGPU throughput</th>
            <td>
              {result.gpuPositionsPerSec.toFixed(1)} positions/sec ·{' '}
              {result.gpuMsPerPosition.toFixed(3)} ms/position ({result.gpuPositions} positions in{' '}
              {result.gpuSeconds.toFixed(2)} s)
            </td>
          </tr>
          <tr>
            <th scope="row">JS reference throughput (same positions)</th>
            <td>
              {result.jsPositionsPerSec.toFixed(1)} positions/sec ·{' '}
              {result.jsMsPerPosition.toFixed(3)} ms/position
            </td>
          </tr>
        </tbody>
      </table>
      {trusted && (
        <p className="bench-note">
          Copy these two lines back to report on-device speed: WebGPU{' '}
          {result.gpuPositionsPerSec.toFixed(1)} positions/sec (
          {result.gpuMsPerPosition.toFixed(3)} ms/position); JS{' '}
          {result.jsPositionsPerSec.toFixed(1)} positions/sec (
          {result.jsMsPerPosition.toFixed(3)} ms/position). Self-check: 0 mismatches.
        </p>
      )}
    </div>
  );
}
