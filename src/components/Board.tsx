// Canvas Gomoku board: crisp grid, star points, coordinates, hover ghost
// stone, last-move marker, winning-line highlight, threat markers, best-move
// marker, preview stone, and a short scale-in animation on each new stone.
// Renders at devicePixelRatio and resizes with its container.

import { useCallback, useEffect, useRef, useState } from 'react';
import { SIZE } from '../engine/gomoku';
import type { Point } from '../engine/gomoku';
import type { ThreatKind } from './analysis';

export interface BoardProps {
  board: Int8Array | number[];
  /** Side to move — drives the hover ghost stone's colour. */
  toMove: 1 | -1;
  lastMove: Point | null;
  winningLine: Point[] | null;
  interactive: boolean;
  onPlay?: (x: number, y: number) => void;
  /** Translucent stone previewed from an analysis suggestion. */
  previewMove?: Point | null;
  bestMove?: Point | null;
  showBestMove?: boolean;
  /** Key threat squares: "x,y" -> kind. */
  threatSquares?: Map<string, ThreatKind> | null;
  showThreats?: boolean;
  flipped?: boolean;
  /** Changes whenever a stone is placed; drives the place animation. */
  animateKey?: number;
  ariaLabel?: string;
}

const STAR_POINTS = [3, 7, 11];
const COL_LABELS = 'abcdefghijklmno';
const ANIM_MS = 180;

const THREAT_COLORS: Record<ThreatKind, string> = {
  'open-four': '#ef4444',
  four: '#f97316',
  'open-three': '#eab308',
};

export default function Board({
  board,
  toMove,
  lastMove,
  winningLine,
  interactive,
  onPlay,
  previewMove = null,
  bestMove = null,
  showBestMove = false,
  threatSquares = null,
  showThreats = false,
  flipped = false,
  animateKey = 0,
  ariaLabel = 'Gomoku board',
}: BoardProps) {
  const wrapRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [sizePx, setSizePx] = useState(520);
  const [hover, setHover] = useState<Point | null>(null);
  const animRef = useRef<{ key: number; start: number } | null>(null);
  const rafRef = useRef<number>(0);

  // Track container size.
  useEffect(() => {
    const el = wrapRef.current;
    if (!el) return;
    const ro = new ResizeObserver((entries) => {
      const w = entries[0]?.contentRect.width;
      if (w && w > 0) setSizePx(Math.floor(w));
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  const geom = useCallback(
    (px: number) => {
      const margin = Math.max(20, Math.round(px * 0.052));
      const cell = (px - margin * 2) / (SIZE - 1);
      return { margin, cell };
    },
    [],
  );

  const toDisplay = useCallback(
    (p: Point): Point => (flipped ? { x: SIZE - 1 - p.x, y: SIZE - 1 - p.y } : p),
    [flipped],
  );
  const fromDisplay = useCallback(
    (p: Point): Point => (flipped ? { x: SIZE - 1 - p.x, y: SIZE - 1 - p.y } : p),
    [flipped],
  );

  const draw = useCallback(
    (animProgress: number) => {
      const canvas = canvasRef.current;
      if (!canvas) return;
      const dpr = window.devicePixelRatio || 1;
      const px = sizePx;
      if (canvas.width !== Math.round(px * dpr)) {
        canvas.width = Math.round(px * dpr);
        canvas.height = Math.round(px * dpr);
      }
      const ctx = canvas.getContext('2d');
      if (!ctx) return;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, px, px);

      const { margin, cell } = geom(px);
      const pos = (dx: number, dy: number) => ({ px: margin + dx * cell, py: margin + dy * cell });
      const stoneR = cell * 0.47;

      // Board surface: warm wood tone with a soft vignette.
      const bg = ctx.createLinearGradient(0, 0, px, px);
      bg.addColorStop(0, '#e9cd92');
      bg.addColorStop(0.5, '#e2bd78');
      bg.addColorStop(1, '#d9ae66');
      ctx.fillStyle = bg;
      const radius = Math.max(8, px * 0.02);
      ctx.beginPath();
      ctx.roundRect(0, 0, px, px, radius);
      ctx.fill();
      // Subtle inner shadow for depth.
      ctx.strokeStyle = 'rgba(90, 60, 20, 0.35)';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.roundRect(0.5, 0.5, px - 1, px - 1, radius);
      ctx.stroke();

      // Grid lines.
      ctx.strokeStyle = 'rgba(74, 48, 16, 0.85)';
      ctx.lineWidth = Math.max(0.8, cell * 0.025);
      ctx.beginPath();
      for (let i = 0; i < SIZE; i++) {
        const a = pos(i, 0);
        const b = pos(i, SIZE - 1);
        ctx.moveTo(a.px, a.py);
        ctx.lineTo(b.px, b.py);
        const c = pos(0, i);
        const d = pos(SIZE - 1, i);
        ctx.moveTo(c.px, c.py);
        ctx.lineTo(d.px, d.py);
      }
      ctx.stroke();

      // Star points.
      ctx.fillStyle = 'rgba(74, 48, 16, 0.9)';
      for (const sx of STAR_POINTS) {
        for (const sy of STAR_POINTS) {
          const { px: cx, py: cy } = pos(sx, sy);
          ctx.beginPath();
          ctx.arc(cx, cy, Math.max(2, cell * 0.09), 0, Math.PI * 2);
          ctx.fill();
        }
      }

      // Coordinates: a–o across, 1–15 down. Labels follow the flip.
      ctx.fillStyle = 'rgba(74, 48, 16, 0.75)';
      ctx.font = `${Math.max(9, Math.round(cell * 0.36))}px ui-sans-serif, system-ui, sans-serif`;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      for (let i = 0; i < SIZE; i++) {
        const labelCol = flipped ? COL_LABELS[SIZE - 1 - i] : COL_LABELS[i];
        const labelRow = flipped ? String(SIZE - i) : String(i + 1);
        const top = pos(i, 0);
        const bottom = pos(i, SIZE - 1);
        const left = pos(0, i);
        const right = pos(SIZE - 1, i);
        ctx.fillText(labelCol, top.px, margin * 0.45);
        ctx.fillText(labelCol, bottom.px, px - margin * 0.45);
        ctx.fillText(labelRow, margin * 0.45, left.py);
        ctx.fillText(labelRow, px - margin * 0.45, right.py);
      }

      const drawStone = (
        dx: number,
        dy: number,
        value: 1 | -1,
        scale: number,
        alpha: number,
      ) => {
        const { px: cx, py: cy } = pos(dx, dy);
        const r = stoneR * scale;
        if (r <= 0) return;
        ctx.save();
        ctx.globalAlpha = alpha;
        // Drop shadow.
        ctx.beginPath();
        ctx.arc(cx + r * 0.08, cy + r * 0.12, r, 0, Math.PI * 2);
        ctx.fillStyle = 'rgba(60, 35, 5, 0.28)';
        ctx.fill();
        const grad = ctx.createRadialGradient(
          cx - r * 0.35,
          cy - r * 0.4,
          r * 0.1,
          cx,
          cy,
          r,
        );
        if (value === 1) {
          grad.addColorStop(0, '#4b4b55');
          grad.addColorStop(0.55, '#232329');
          grad.addColorStop(1, '#0c0c10');
        } else {
          grad.addColorStop(0, '#ffffff');
          grad.addColorStop(0.7, '#f2eee2');
          grad.addColorStop(1, '#cfc8b4');
        }
        ctx.beginPath();
        ctx.arc(cx, cy, r, 0, Math.PI * 2);
        ctx.fillStyle = grad;
        ctx.fill();
        ctx.strokeStyle = value === 1 ? 'rgba(0,0,0,0.5)' : 'rgba(120,100,60,0.55)';
        ctx.lineWidth = 1;
        ctx.stroke();
        ctx.restore();
      };

      // Winning-line glow, drawn under the stones.
      if (winningLine && winningLine.length >= 2) {
        const first = toDisplay(winningLine[0]);
        const last = toDisplay(winningLine[winningLine.length - 1]);
        const a = pos(first.x, first.y);
        const b = pos(last.x, last.y);
        ctx.save();
        ctx.strokeStyle = 'rgba(220, 38, 38, 0.55)';
        ctx.lineWidth = stoneR * 0.5;
        ctx.lineCap = 'round';
        ctx.beginPath();
        ctx.moveTo(a.px, a.py);
        ctx.lineTo(b.px, b.py);
        ctx.stroke();
        ctx.restore();
      }

      // Stones.
      for (let y = 0; y < SIZE; y++) {
        for (let x = 0; x < SIZE; x++) {
          const v = board[y * SIZE + x];
          if (v === 0) continue;
          const d = toDisplay({ x, y });
          const isLast = lastMove !== null && lastMove.x === x && lastMove.y === y;
          const scale = isLast ? 0.55 + 0.45 * animProgress : 1;
          drawStone(d.x, d.y, v as 1 | -1, scale, 1);
        }
      }

      // Threat markers on empty key squares.
      if (showThreats && threatSquares) {
        for (const [key, kind] of threatSquares) {
          const [xs, ys] = key.split(',');
          const x = Number(xs);
          const y = Number(ys);
          if (!Number.isFinite(x) || !Number.isFinite(y)) continue;
          if (board[y * SIZE + x] !== 0) continue;
          const d = toDisplay({ x, y });
          const { px: cx, py: cy } = pos(d.x, d.y);
          ctx.save();
          ctx.fillStyle = THREAT_COLORS[kind];
          ctx.globalAlpha = 0.9;
          ctx.beginPath();
          ctx.arc(cx, cy, Math.max(3, cell * 0.13), 0, Math.PI * 2);
          ctx.fill();
          ctx.globalAlpha = 0.35;
          ctx.beginPath();
          ctx.arc(cx, cy, Math.max(5, cell * 0.26), 0, Math.PI * 2);
          ctx.fill();
          ctx.restore();
        }
      }

      // Preview stone (from an analysis suggestion).
      if (previewMove) {
        const d = toDisplay(previewMove);
        drawStone(d.x, d.y, toMove, 1, 0.45);
        const { px: cx, py: cy } = pos(d.x, d.y);
        ctx.save();
        ctx.strokeStyle = 'rgba(20, 120, 140, 0.9)';
        ctx.setLineDash([4, 3]);
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.arc(cx, cy, stoneR * 1.08, 0, Math.PI * 2);
        ctx.stroke();
        ctx.restore();
      }

      // Best-move marker.
      if (showBestMove && bestMove && board[bestMove.y * SIZE + bestMove.x] === 0) {
        const d = toDisplay(bestMove);
        const { px: cx, py: cy } = pos(d.x, d.y);
        ctx.save();
        ctx.strokeStyle = '#0ea5a4';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.arc(cx, cy, stoneR * 0.72, 0, Math.PI * 2);
        ctx.stroke();
        ctx.fillStyle = '#0ea5a4';
        ctx.beginPath();
        ctx.arc(cx, cy, Math.max(2, cell * 0.07), 0, Math.PI * 2);
        ctx.fill();
        ctx.restore();
      }

      // Hover ghost stone.
      if (interactive && hover && board[hover.y * SIZE + hover.x] === 0) {
        const d = toDisplay(hover);
        drawStone(d.x, d.y, toMove, 1, 0.5);
      }

      // Last-move marker: a small ring in the contrasting colour.
      if (lastMove) {
        const d = toDisplay(lastMove);
        const { px: cx, py: cy } = pos(d.x, d.y);
        const v = board[lastMove.y * SIZE + lastMove.x];
        ctx.save();
        ctx.strokeStyle = v === 1 ? 'rgba(255,255,255,0.9)' : 'rgba(20,20,20,0.75)';
        ctx.lineWidth = Math.max(1.2, cell * 0.045);
        ctx.beginPath();
        ctx.arc(cx, cy, stoneR * 0.45, 0, Math.PI * 2);
        ctx.stroke();
        ctx.restore();
      }

      // Winning stones get a red ring.
      if (winningLine) {
        ctx.save();
        ctx.strokeStyle = '#dc2626';
        ctx.lineWidth = Math.max(1.5, cell * 0.06);
        for (const p of winningLine) {
          const d = toDisplay(p);
          const { px: cx, py: cy } = pos(d.x, d.y);
          ctx.beginPath();
          ctx.arc(cx, cy, stoneR * 0.92, 0, Math.PI * 2);
          ctx.stroke();
        }
        ctx.restore();
      }
    },
    [
      sizePx,
      geom,
      board,
      toMove,
      lastMove,
      winningLine,
      interactive,
      hover,
      previewMove,
      bestMove,
      showBestMove,
      threatSquares,
      showThreats,
      toDisplay,
    ],
  );

  // Redraw on state change; run the place animation when animateKey changes.
  useEffect(() => {
    const prevKey = animRef.current?.key;
    if (prevKey !== animateKey && lastMove) {
      animRef.current = { key: animateKey, start: performance.now() };
      const tick = (now: number) => {
        const started = animRef.current?.start ?? now;
        const t = Math.min(1, (now - started) / ANIM_MS);
        // Ease-out cubic.
        const eased = 1 - Math.pow(1 - t, 3);
        draw(eased);
        if (t < 1) rafRef.current = requestAnimationFrame(tick);
      };
      cancelAnimationFrame(rafRef.current);
      rafRef.current = requestAnimationFrame(tick);
      return () => cancelAnimationFrame(rafRef.current);
    }
    draw(1);
    return undefined;
  }, [draw, animateKey, lastMove]);

  const pointFromEvent = useCallback(
    (e: React.PointerEvent<HTMLCanvasElement>): Point | null => {
      const canvas = canvasRef.current;
      if (!canvas) return null;
      const rect = canvas.getBoundingClientRect();
      const { margin, cell } = geom(rect.width);
      const dx = Math.round((e.clientX - rect.left - margin) / cell);
      const dy = Math.round((e.clientY - rect.top - margin) / cell);
      if (dx < 0 || dx >= SIZE || dy < 0 || dy >= SIZE) return null;
      // Reject taps far from an intersection (more than ~45% of a cell).
      const cx = margin + dx * cell;
      const cy = margin + dy * cell;
      const dist = Math.hypot(e.clientX - rect.left - cx, e.clientY - rect.top - cy);
      if (dist > cell * 0.45) return null;
      return fromDisplay({ x: dx, y: dy });
    },
    [geom, fromDisplay],
  );

  const handlePointerMove = (e: React.PointerEvent<HTMLCanvasElement>) => {
    if (!interactive) {
      setHover(null);
      return;
    }
    setHover(pointFromEvent(e));
  };

  const handlePointerDown = (e: React.PointerEvent<HTMLCanvasElement>) => {
    if (!interactive) return;
    const p = pointFromEvent(e);
    if (p) onPlay?.(p.x, p.y);
  };

  return (
    <div ref={wrapRef} className="board-wrap" style={{ width: '100%', aspectRatio: '1 / 1' }}>
      <canvas
        ref={canvasRef}
        role="img"
        aria-label={ariaLabel}
        style={{
          width: '100%',
          height: '100%',
          display: 'block',
          touchAction: 'manipulation',
          cursor: interactive ? 'pointer' : 'default',
        }}
        onPointerMove={handlePointerMove}
        onPointerLeave={() => setHover(null)}
        onPointerDown={handlePointerDown}
      />
    </div>
  );
}
