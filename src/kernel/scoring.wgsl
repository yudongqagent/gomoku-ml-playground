// Scoring kernel v0 — WGSL compute shader.
//
// Exact port of ml/gomoku_ml/baseline.py score_move, applied to every cell
// of every board in the batch. One thread per (position, cell).
//
// Faithfulness notes:
// * Runs are CONTIGUOUS only (the v0 broken-four blind spot is ported, not
//   fixed): a broken four XX_XX scores as two separate runs.
// * W_OPP is the precomputed int(0.9 * w) table from the Python reference
//   (float64 arithmetic). It is NOT computed in-shader: WGSL f32 0.9 * w
//   would truncate differently (e.g. 0.9f * 100000 = 89999.99... -> 89999
//   instead of 90000).
// * Scores are computed for ALL 225 cells, including occupied ones — the
//   Python score_move never checks legality, and neither does this shader.
// * Board values: 1 = black, -1 = white, 0 = empty. Flat index y*15+x.
//
// Layout:
//   binding 0 (read):  boards,  array<i32>, N * 225, row-major per board.
//   binding 1 (read):  players, array<i32>, N, side to move per board (1/-1).
//   binding 2 (read_write): scores, array<i32>, N * 225, one score per cell.
//
// Dispatch: ceil(N * 225 / 64) workgroups of 64 threads; threads with a
// global index >= N * 225 exit immediately.

// Weight table indexed by min(run, 5) * 3 + open_ends.
// Row 0 is unused (run is always >= 1); rows 1..5 are baseline._WEIGHTS.
const W: array<i32, 18> = array<i32, 18>(
  0, 0, 0,
  0, 1, 10,          // run 1
  1, 10, 100,        // run 2
  10, 100, 1000,     // run 3
  100, 1000, 10000,  // run 4
  100000, 100000, 100000, // run 5
);

// Opponent weight table: int(0.9 * w) per entry of W, computed in Python
// (float64) — see the header note about f32 truncation.
const W_OPP: array<i32, 18> = array<i32, 18>(
  0, 0, 0,
  0, 0, 9,           // run 1
  0, 9, 90,          // run 2
  9, 90, 900,        // run 3
  90, 900, 9000,     // run 4
  90000, 90000, 90000, // run 5
);

// The four line directions (dx, dy): horizontal, vertical, 2 diagonals.
const DX: array<i32, 4> = array<i32, 4>(1, 0, 1, 1);
const DY: array<i32, 4> = array<i32, 4>(0, 1, 1, -1);

@group(0) @binding(0) var<storage, read> boards: array<i32>;
@group(0) @binding(1) var<storage, read> players: array<i32>;
@group(0) @binding(2) var<storage, read_write> scores: array<i32>;

// Board value at (x, y) of the position starting at `base`; 2 (a value
// that is neither a stone nor empty) when off-board, so runs stop and
// off-board ends are not open — matching the Python bounds checks.
fn boardAt(base: i32, x: i32, y: i32) -> i32 {
  if (x < 0 || x >= 15 || y < 0 || y >= 15) {
    return 2;
  }
  return boards[base + y * 15 + x];
}

// (run, open_ends) along direction (dx, dy) if `color` played at (x, y).
// Run counts the played stone itself, so it is >= 1.
fn runAndEnds(base: i32, x: i32, y: i32, dx: i32, dy: i32, color: i32) -> vec2<i32> {
  var run = 1;
  var ends = 0;
  var nx = x + dx;
  var ny = y + dy;
  loop {
    if (boardAt(base, nx, ny) != color) {
      break;
    }
    run += 1;
    nx += dx;
    ny += dy;
  }
  if (boardAt(base, nx, ny) == 0) {
    ends += 1;
  }
  nx = x - dx;
  ny = y - dy;
  loop {
    if (boardAt(base, nx, ny) != color) {
      break;
    }
    run += 1;
    nx -= dx;
    ny -= dy;
  }
  if (boardAt(base, nx, ny) == 0) {
    ends += 1;
  }
  return vec2<i32>(run, ends);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let g = i32(gid.x);
  let total = i32(arrayLength(&scores));
  if (g >= total) {
    return;
  }
  let posIdx = g / 225;
  let cell = g % 225;
  let x = cell % 15;
  let y = cell / 15;
  let base = posIdx * 225;
  let player = players[posIdx];

  // Centre preference, exactly as in baseline.score_move.
  var totalScore = 7 - (abs(x - 7) + abs(y - 7));
  for (var d = 0; d < 4; d++) {
    let re = runAndEnds(base, x, y, DX[d], DY[d], player);
    totalScore += W[min(re.x, 5) * 3 + re.y];
    let reOpp = runAndEnds(base, x, y, DX[d], DY[d], -player);
    totalScore += W_OPP[min(reOpp.x, 5) * 3 + reOpp.y];
  }
  scores[g] = totalScore;
}
