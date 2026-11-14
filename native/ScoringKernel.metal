// Scoring kernel v0 — Metal compute shader (for future native iOS use).
//
// Exact port of ml/gomoku_ml/baseline.py score_move, applied to every cell
// of every board in the batch. One thread per (position, cell). This is the
// same algorithm as src/kernel/scoring.wgsl, transliterated to Metal
// Shading Language; the buffer layout matches the WGSL version plus a
// scalar `total` (Metal kernels cannot query a buffer's length).
//
// ⚠️ NOT COMPILED OR TESTED: there is no Apple hardware and no Metal
// toolchain on the Linux VM where this was written. It must be compiled
// with Xcode (macOS) or on-device before use, and its output must be
// checked against the frozen Python scores in
// ml/tests/kernel_fixtures.json (60 positions x 225 cells = 13,500 cells,
// 0 mismatches required) before it is trusted. See native/README.md.
//
// Faithfulness notes (v0 quirks are ported, not fixed):
// * Runs are CONTIGUOUS only (the v0 broken-four blind spot is ported, not
//   fixed): a broken four XX_XX scores as two separate runs.
// * W_OPP is the precomputed int(0.9 * w) table from the Python reference
//   (float64 arithmetic). It is NOT computed in-shader: float 0.9 * w
//   would truncate differently (e.g. 0.9f * 100000 = 89999.99... -> 89999
//   instead of 90000).
// * Scores are computed for ALL 225 cells, including occupied ones — the
//   Python score_move never checks legality, and neither does this kernel.
// * Board values: 1 = black, -1 = white, 0 = empty. Flat index y*15+x.
//
// Buffers:
//   [[buffer(0)]] boards:  device const int *, N * 225, row-major per board.
//   [[buffer(1)]] players: device const int *, N, side to move per board (1/-1).
//   [[buffer(2)]] scores:  device int *, N * 225, one score per cell (output).
//   [[buffer(3)]] total:   constant int, total thread count = N * 225.
//
// Dispatch: ceil(N * 225 / 256) threadgroups of 256 threads (or any grid
// covering `total` threads); threads with gid >= total exit immediately.

#include <metal_stdlib>
using namespace metal;

// Weight table indexed by min(run, 5) * 3 + open_ends.
// Row 0 is unused (run is always >= 1); rows 1..5 are baseline._WEIGHTS.
constant int W[18] = {
    0, 0, 0,
    0, 1, 10,            // run 1
    1, 10, 100,          // run 2
    10, 100, 1000,       // run 3
    100, 1000, 10000,    // run 4
    100000, 100000, 100000 // run 5
};

// Opponent weight table: int(0.9 * w) per entry of W, computed in Python
// (float64) — see the header note about float truncation.
constant int W_OPP[18] = {
    0, 0, 0,
    0, 0, 9,             // run 1
    0, 9, 90,            // run 2
    9, 90, 900,          // run 3
    90, 900, 9000,       // run 4
    90000, 90000, 90000  // run 5
};

// The four line directions (dx, dy): horizontal, vertical, 2 diagonals.
constant int DX[4] = { 1, 0, 1, 1 };
constant int DY[4] = { 0, 1, 1, -1 };

// Board value at (x, y) of the position starting at `base`; 2 (a value
// that is neither a stone nor empty) when off-board, so runs stop and
// off-board ends are not open — matching the Python bounds checks.
inline int boardAt(device const int *boards, int base, int x, int y) {
    if (x < 0 || x >= 15 || y < 0 || y >= 15) {
        return 2;
    }
    return boards[base + y * 15 + x];
}

// (run, open_ends) along direction (dx, dy) if `color` played at (x, y).
// Run counts the played stone itself, so it is >= 1.
inline int2 runAndEnds(device const int *boards, int base, int x, int y,
                       int dx, int dy, int color) {
    int run = 1;
    int ends = 0;
    int nx = x + dx;
    int ny = y + dy;
    while (boardAt(boards, base, nx, ny) == color) {
        run += 1;
        nx += dx;
        ny += dy;
    }
    if (boardAt(boards, base, nx, ny) == 0) {
        ends += 1;
    }
    nx = x - dx;
    ny = y - dy;
    while (boardAt(boards, base, nx, ny) == color) {
        run += 1;
        nx -= dx;
        ny -= dy;
    }
    if (boardAt(boards, base, nx, ny) == 0) {
        ends += 1;
    }
    return int2(run, ends);
}

kernel void scoreMoves(
    device const int *boards [[buffer(0)]],
    device const int *players [[buffer(1)]],
    device int *scores [[buffer(2)]],
    constant int &total [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    int g = int(gid);
    if (g >= total) {
        return;
    }
    int posIdx = g / 225;
    int cell = g % 225;
    int x = cell % 15;
    int y = cell / 15;
    int base = posIdx * 225;
    int player = players[posIdx];

    // Centre preference, exactly as in baseline.score_move.
    int totalScore = 7 - (abs(x - 7) + abs(y - 7));
    for (int d = 0; d < 4; d++) {
        int2 re = runAndEnds(boards, base, x, y, DX[d], DY[d], player);
        totalScore += W[min(re.x, 5) * 3 + re.y];
        int2 reOpp = runAndEnds(boards, base, x, y, DX[d], DY[d], -player);
        totalScore += W_OPP[min(reOpp.x, 5) * 3 + reOpp.y];
    }
    scores[g] = totalScore;
}
