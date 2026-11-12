// SVG line charts: the win-rate-over-game graph and the training loss curve
// share one small chart primitive (no chart library, no network dependency).

import { useId, useMemo } from 'react';
import type { WinRatePoint } from './types';
import type { EvalHistoryEntry, MetricPoint } from '../hooks/useTrainerApi';
import { evalHistoryWinRates } from '../hooks/useTrainerApi';

export interface ChartPoint {
  x: number;
  y: number;
}

interface SvgLineChartProps {
  points: ChartPoint[];
  width?: number;
  height?: number;
  /** Fixed y domain; defaults to the data range with padding. */
  yMin?: number;
  yMax?: number;
  /** Horizontal reference line (e.g. 50% for win rate). */
  referenceY?: number;
  referenceLabel?: string;
  lineColor?: string;
  areaColor?: string;
  xLabel?: (x: number) => string;
  yLabel?: (y: number) => string;
  emptyText: string;
}

export function SvgLineChart({
  points,
  width = 560,
  height = 160,
  yMin,
  yMax,
  referenceY,
  referenceLabel,
  lineColor = '#2dd4bf',
  areaColor = 'rgba(45, 212, 191, 0.18)',
  xLabel,
  yLabel,
  emptyText,
}: SvgLineChartProps) {
  const gradientId = useId();
  const pad = { left: 40, right: 10, top: 10, bottom: 22 };
  const innerW = width - pad.left - pad.right;
  const innerH = height - pad.top - pad.bottom;

  const { minY, maxY, minX, maxX } = useMemo(() => {
    if (points.length === 0) return { minY: 0, maxY: 1, minX: 0, maxX: 1 };
    const ys = points.map((p) => p.y);
    const xs = points.map((p) => p.x);
    let lo = yMin ?? Math.min(...ys);
    let hi = yMax ?? Math.max(...ys);
    if (hi - lo < 1e-9) {
      hi = lo + 1;
    }
    // A little headroom when the domain comes from the data.
    if (yMin === undefined) lo -= (hi - lo) * 0.08;
    if (yMax === undefined) hi += (hi - lo) * 0.08;
    return { minY: lo, maxY: hi, minX: Math.min(...xs), maxX: Math.max(...xs, minXSafe(xs)) };
    function minXSafe(xsArr: number[]) {
      return Math.min(...xsArr) + 1;
    }
  }, [points, yMin, yMax]);

  if (points.length === 0) {
    return (
      <div className="chart-empty" role="img" aria-label={emptyText}>
        {emptyText}
      </div>
    );
  }

  const xScale = (x: number) => pad.left + ((x - minX) / Math.max(1, maxX - minX)) * innerW;
  const yScale = (y: number) => pad.top + (1 - (y - minY) / (maxY - minY)) * innerH;

  const linePath = points
    .map((p, i) => `${i === 0 ? 'M' : 'L'} ${xScale(p.x).toFixed(1)} ${yScale(p.y).toFixed(1)}`)
    .join(' ');
  const areaPath =
    points.length > 1
      ? `${linePath} L ${xScale(points[points.length - 1].x).toFixed(1)} ${yScale(minY).toFixed(1)} L ${xScale(points[0].x).toFixed(1)} ${yScale(minY).toFixed(1)} Z`
      : '';
  const last = points[points.length - 1];

  const yTicks = [minY, (minY + maxY) / 2, maxY];
  const xTicks = points.length > 1 ? [minX, Math.round((minX + maxX) / 2), maxX] : [minX];

  return (
    <svg
      viewBox={`0 0 ${width} ${height}`}
      className="svg-chart"
      role="img"
      aria-label="Line chart"
      preserveAspectRatio="none"
      style={{ width: '100%', height: 'auto', display: 'block' }}
    >
      <defs>
        <linearGradient id={gradientId} x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor={areaColor} />
          <stop offset="100%" stopColor="rgba(0,0,0,0)" />
        </linearGradient>
      </defs>
      {referenceY !== undefined && referenceY >= minY && referenceY <= maxY && (
        <g>
          <line
            x1={pad.left}
            x2={width - pad.right}
            y1={yScale(referenceY)}
            y2={yScale(referenceY)}
            stroke="rgba(148, 163, 184, 0.45)"
            strokeDasharray="4 4"
            strokeWidth="1"
          />
          {referenceLabel && (
            <text
              x={width - pad.right - 2}
              y={yScale(referenceY) - 4}
              textAnchor="end"
              className="chart-tick"
            >
              {referenceLabel}
            </text>
          )}
        </g>
      )}
      {yTicks.map((t, i) => (
        <text key={i} x={pad.left - 6} y={yScale(t) + 3} textAnchor="end" className="chart-tick">
          {yLabel ? yLabel(t) : t.toFixed(2)}
        </text>
      ))}
      {xTicks.map((t, i) => (
        <text
          key={i}
          x={xScale(t)}
          y={height - 6}
          textAnchor="middle"
          className="chart-tick"
        >
          {xLabel ? xLabel(t) : String(t)}
        </text>
      ))}
      {areaPath && <path d={areaPath} fill={`url(#${gradientId})`} />}
      <path d={linePath} fill="none" stroke={lineColor} strokeWidth="2" strokeLinejoin="round" />
      <circle cx={xScale(last.x)} cy={yScale(last.y)} r="3.5" fill={lineColor} stroke="#0e1116" strokeWidth="1.5" />
    </svg>
  );
}

/** Black's win rate over the game, from the per-move analysis history. */
export function WinRateGraph({ history }: { history: WinRatePoint[] }) {
  const points = useMemo(
    () => history.map((h) => ({ x: h.ply, y: h.blackWinRate })),
    [history],
  );
  return (
    <SvgLineChart
      points={points}
      yMin={0}
      yMax={1}
      referenceY={0.5}
      referenceLabel="50%"
      lineColor="#2dd4bf"
      xLabel={(x) => `Move ${x}`}
      yLabel={(y) => `${Math.round(y * 100)}%`}
      emptyText="No analysis yet — play a few moves with auto-analyze on, or press “Analyze this position”."
    />
  );
}

/** Training loss over steps, from a job's metrics. */
export function LossCurve({ points }: { points: { step: number; loss: number }[] }) {
  const chartPoints = useMemo(
    () => points.map((p) => ({ x: p.step, y: p.loss })),
    [points],
  );
  return (
    <SvgLineChart
      points={chartPoints}
      lineColor="#a78bfa"
      areaColor="rgba(167, 139, 250, 0.18)"
      xLabel={(x) => `Step ${x}`}
      yLabel={(y) => y.toFixed(2)}
      emptyText="No loss data yet — metrics appear once training starts reporting them."
    />
  );
}

// ---------------------------------------------------------------------------
// Multi-series charts (trainer dashboard)
// ---------------------------------------------------------------------------

export interface ChartSeries {
  label: string;
  color: string;
  points: ChartPoint[];
  /** Text shown next to the label in the legend (e.g. the latest value). */
  latestText?: string;
}

interface SvgMultiLineChartProps {
  series: ChartSeries[];
  width?: number;
  height?: number;
  yMin?: number;
  yMax?: number;
  referenceY?: number;
  referenceLabel?: string;
  xLabel?: (x: number) => string;
  yLabel?: (y: number) => string;
  emptyText: string;
}

/** Same primitive as SvgLineChart, but with several coloured series sharing
 * one axis and a legend. Series with a single point render as a dot, not a
 * fake line. */
export function SvgMultiLineChart({
  series,
  width = 560,
  height = 180,
  yMin,
  yMax,
  referenceY,
  referenceLabel,
  xLabel,
  yLabel,
  emptyText,
}: SvgMultiLineChartProps) {
  const pad = { left: 40, right: 10, top: 10, bottom: 22 };
  const innerW = width - pad.left - pad.right;
  const innerH = height - pad.top - pad.bottom;

  const nonEmpty = useMemo(() => series.filter((s) => s.points.length > 0), [series]);
  const totalPoints = nonEmpty.reduce((n, s) => n + s.points.length, 0);

  const { minY, maxY, minX, maxX } = useMemo(() => {
    if (totalPoints === 0) return { minY: 0, maxY: 1, minX: 0, maxX: 1 };
    const ys = nonEmpty.flatMap((s) => s.points.map((p) => p.y));
    const xs = nonEmpty.flatMap((s) => s.points.map((p) => p.x));
    let lo = yMin ?? Math.min(...ys);
    let hi = yMax ?? Math.max(...ys);
    if (hi - lo < 1e-9) hi = lo + 1;
    if (yMin === undefined) lo -= (hi - lo) * 0.08;
    if (yMax === undefined) hi += (hi - lo) * 0.08;
    const loX = Math.min(...xs);
    return { minY: lo, maxY: hi, minX: loX, maxX: Math.max(...xs, loX + 1) };
  }, [nonEmpty, totalPoints, yMin, yMax]);

  if (totalPoints === 0) {
    return (
      <div className="chart-empty" role="img" aria-label={emptyText}>
        {emptyText}
      </div>
    );
  }

  const xScale = (x: number) => pad.left + ((x - minX) / Math.max(1, maxX - minX)) * innerW;
  const yScale = (y: number) => pad.top + (1 - (y - minY) / (maxY - minY)) * innerH;
  const yTicks = [minY, (minY + maxY) / 2, maxY];
  const xTicks = maxX > minX ? [minX, Math.round((minX + maxX) / 2), maxX] : [minX];

  return (
    <div className="multi-chart">
      <svg
        viewBox={`0 0 ${width} ${height}`}
        className="svg-chart"
        role="img"
        aria-label="Multi-series line chart"
        preserveAspectRatio="none"
        style={{ width: '100%', height: 'auto', display: 'block' }}
      >
        {referenceY !== undefined && referenceY >= minY && referenceY <= maxY && (
          <g>
            <line
              x1={pad.left}
              x2={width - pad.right}
              y1={yScale(referenceY)}
              y2={yScale(referenceY)}
              stroke="rgba(148, 163, 184, 0.45)"
              strokeDasharray="4 4"
              strokeWidth="1"
            />
            {referenceLabel && (
              <text
                x={width - pad.right - 2}
                y={yScale(referenceY) - 4}
                textAnchor="end"
                className="chart-tick"
              >
                {referenceLabel}
              </text>
            )}
          </g>
        )}
        {yTicks.map((t, i) => (
          <text key={i} x={pad.left - 6} y={yScale(t) + 3} textAnchor="end" className="chart-tick">
            {yLabel ? yLabel(t) : t.toFixed(2)}
          </text>
        ))}
        {xTicks.map((t, i) => (
          <text key={i} x={xScale(t)} y={height - 6} textAnchor="middle" className="chart-tick">
            {xLabel ? xLabel(t) : String(t)}
          </text>
        ))}
        {nonEmpty.map((s) => {
          const linePath = s.points
            .map((p, i) => `${i === 0 ? 'M' : 'L'} ${xScale(p.x).toFixed(1)} ${yScale(p.y).toFixed(1)}`)
            .join(' ');
          const last = s.points[s.points.length - 1];
          return (
            <g key={s.label}>
              {s.points.length > 1 && (
                <path d={linePath} fill="none" stroke={s.color} strokeWidth="2" strokeLinejoin="round" />
              )}
              <circle
                cx={xScale(last.x)}
                cy={yScale(last.y)}
                r="3.5"
                fill={s.color}
                stroke="#0e1116"
                strokeWidth="1.5"
              />
            </g>
          );
        })}
      </svg>
      <div className="chart-legend">
        {nonEmpty.map((s) => {
          const last = s.points[s.points.length - 1];
          return (
            <span key={s.label} className="chart-legend-item">
              <span className="chart-legend-swatch" style={{ background: s.color }} />
              {s.label}
              {s.latestText ? (
                <strong> {s.latestText}</strong>
              ) : (
                <strong> {last.y.toFixed(3)}</strong>
              )}
            </span>
          );
        })}
      </div>
    </div>
  );
}

/**
 * Policy loss + value loss as two lines on one chart, with the total loss as
 * a third line when it differs from the components. Falls back to the
 * total-loss-only curve when the server reports no per-component losses.
 */
export function LossCurves({ points }: { points: MetricPoint[] }) {
  const series = useMemo<ChartSeries[]>(() => {
    const hasPolicy = points.some((p) => p.policy_loss !== undefined);
    const hasValue = points.some((p) => p.value_loss !== undefined);
    const out: ChartSeries[] = [];
    if (hasPolicy) {
      const pts = points
        .filter((p) => p.policy_loss !== undefined)
        .map((p) => ({ x: p.step, y: p.policy_loss as number }));
      const last = pts[pts.length - 1];
      out.push({
        label: 'Policy loss',
        color: '#a78bfa',
        points: pts,
        latestText: last ? last.y.toFixed(4) : undefined,
      });
    }
    if (hasValue) {
      const pts = points
        .filter((p) => p.value_loss !== undefined)
        .map((p) => ({ x: p.step, y: p.value_loss as number }));
      const last = pts[pts.length - 1];
      out.push({
        label: 'Value loss',
        color: '#2dd4bf',
        points: pts,
        latestText: last ? last.y.toFixed(4) : undefined,
      });
    }
    // Total loss line as well, so the combined trend stays visible.
    const total = points.map((p) => ({ x: p.step, y: p.loss }));
    const lastTotal = total[total.length - 1];
    out.push({
      label: 'Total loss',
      color: '#f59e0b',
      points: total,
      latestText: lastTotal ? lastTotal.y.toFixed(4) : undefined,
    });
    return out;
  }, [points]);

  const useEpochs = points.some((p) => p.epoch !== undefined);
  return (
    <SvgMultiLineChart
      series={series}
      xLabel={(x) => `${useEpochs ? 'Epoch' : 'Step'} ${x}`}
      yLabel={(y) => y.toFixed(2)}
      emptyText="No loss data yet — metrics appear once training starts reporting them."
    />
  );
}

const EVAL_LINE_COLORS = ['#2dd4bf', '#a78bfa', '#f59e0b', '#f472b6', '#60a5fa', '#86efac'];

/**
 * Eval win rate over checkpoints, one line per ladder opponent. With a
 * single eval point there is no trend to draw, so it renders as a table
 * instead of a fake line.
 */
export function EvalWinRateOverTime({ history }: { history: EvalHistoryEntry[] }) {
  const parsed = useMemo(
    () =>
      history.map((entry, i) => ({
        entry,
        index: i,
        rates: evalHistoryWinRates(entry),
      })),
    [history],
  );
  const opponents = useMemo(() => {
    const names: string[] = [];
    for (const p of parsed) {
      for (const name of Object.keys(p.rates)) {
        if (!names.includes(name)) names.push(name);
      }
    }
    return names;
  }, [parsed]);

  if (history.length === 0) {
    return (
      <div className="chart-empty" role="img" aria-label="No eval history yet">
        No eval history yet — evaluations appear here once the run is evaluated.
      </div>
    );
  }
  if (opponents.length === 0) {
    return (
      <div className="chart-empty" role="img" aria-label="Eval history unreadable">
        Eval history exists, but no per-opponent win rates could be read from this
        server&apos;s eval summaries.
      </div>
    );
  }

  const withRates = parsed.filter((p) => Object.keys(p.rates).length > 0);
  if (withRates.length === 1) {
    const only = withRates[0];
    return (
      <table className="eval-table">
        <thead>
          <tr>
            <th>Checkpoint</th>
            <th>Opponent</th>
            <th>Win rate</th>
          </tr>
        </thead>
        <tbody>
          {Object.entries(only.rates).map(([opponent, rate]) => (
            <tr key={opponent}>
              <td>{only.entry.checkpoint}</td>
              <td>{opponent}</td>
              <td>{(rate * 100).toFixed(1)}%</td>
            </tr>
          ))}
        </tbody>
      </table>
    );
  }

  const series: ChartSeries[] = opponents.map((opponent, i) => {
    const pts = withRates
      .filter((p) => p.rates[opponent] !== undefined)
      .map((p) => ({ x: p.index, y: p.rates[opponent] }));
    const last = pts[pts.length - 1];
    return {
      label: `vs ${opponent}`,
      color: EVAL_LINE_COLORS[i % EVAL_LINE_COLORS.length],
      points: pts,
      latestText: last ? `${(last.y * 100).toFixed(1)}%` : undefined,
    };
  });
  const checkpointName = (x: number) => {
    const p = parsed[Math.round(x)];
    return p ? p.entry.checkpoint : `Checkpoint ${x}`;
  };

  return (
    <SvgMultiLineChart
      series={series}
      yMin={0}
      yMax={1}
      referenceY={0.5}
      referenceLabel="50%"
      xLabel={checkpointName}
      yLabel={(y) => `${Math.round(y * 100)}%`}
      emptyText="No eval history yet."
    />
  );
}
