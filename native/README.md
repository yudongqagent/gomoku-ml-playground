# Native Metal kernel (future iOS use)

`ScoringKernel.metal` is the v0 scoring function
(`ml/gomoku_ml/baseline.py` `score_move`) as a Metal compute kernel — the
same algorithm as the WGSL shader in `src/kernel/scoring.wgsl`,
transliterated to Metal Shading Language. It exists so a future native iOS
app can score boards on the A20 Pro's GPU without going through the browser.

> ⚠️ **Not compiled or tested.** There is no Apple hardware and no Metal
> toolchain on the Linux VM where this was written, so this file has never
> been compiled, let alone run. Do not trust it until it has been compiled
> with Xcode and its output checked against the frozen Python scores in
> `ml/tests/kernel_fixtures.json` (60 positions × 225 cells = 13,500 cells;
> 0 mismatches required). The WGSL twin of this kernel *has* been executed
> and verified (0 mismatches over the same 13,500 cells, under Dawn +
> SwiftShader — see `scripts/check_wgsl_dawn.mjs`), which validates the
> algorithm; the Metal transliteration itself is still unverified.

## What it does

One thread per (position, cell). For each of the 225 cells of each board in
the batch, it sums over the 4 directions (horizontal, vertical, 2
diagonals) the pattern weight for the side to move plus the 90% opponent
weight at the same point, plus the centre preference
`7 - (|x-7| + |y-7|)`. Scores are computed for ALL 225 cells, including
occupied ones, exactly as the Python reference does. Runs are contiguous
only — the v0 broken-four blind spot is ported faithfully, not fixed.

## Buffer layout

All buffers hold 32-bit ints (`int` / `Int32`). `N` is the batch size
(number of positions).

| Buffer | Contents | Size (ints) | Direction |
| --- | --- | --- | --- |
| `[[buffer(0)]]` `boards` | Boards, row-major per board, flat index `y*15+x`. Values: 1 = black, -1 = white, 0 = empty | `N * 225` | read |
| `[[buffer(1)]]` `players` | Side to move per board: 1 (black) or -1 (white) | `N` | read |
| `[[buffer(2)]]` `scores` | Output scores, one per cell, same flat indexing as `boards` | `N * 225` | write |
| `[[buffer(3)]]` `total` | Total thread count = `N * 225` (scalar) | 1 | read |

Dispatch `ceil(N * 225 / 256)` threadgroups of 256 threads (any grid
covering `total` threads works; threads with `gid >= total` exit
immediately). `total` is passed explicitly because Metal kernels cannot
query a buffer's length.

## Wiring it into a Swift app (sketch)

```swift
import Metal

let device = MTLCreateSystemDefaultDevice()!          // A20 Pro GPU on iPhone
let library = try device.makeDefaultLibrary(bundle: .main) // add ScoringKernel.metal to the target
let function = library.makeFunction(name: "scoreMoves")!
let pipeline = try device.makeComputePipelineState(function: function)
let queue = device.makeCommandQueue()!

// boards: [Int32] of N*225, players: [Int32] of N, total = N*225
let boardsBuffer = device.makeBuffer(bytes: boards, length: boards.count * 4, options: .storageModeShared)!
let playersBuffer = device.makeBuffer(bytes: players, length: players.count * 4, options: .storageModeShared)!
let scoresBuffer = device.makeBuffer(length: boards.count * 4, options: .storageModeShared)!
var total = Int32(boards.count)
let totalBuffer = device.makeBuffer(bytes: &total, length: 4, options: .storageModeShared)!

let commandBuffer = queue.makeCommandBuffer()!
let encoder = commandBuffer.makeComputeCommandEncoder()!
encoder.setComputePipelineState(pipeline)
encoder.setBuffer(boardsBuffer, offset: 0, index: 0)
encoder.setBuffer(playersBuffer, offset: 0, index: 1)
encoder.setBuffer(scoresBuffer, offset: 0, index: 2)
encoder.setBuffer(totalBuffer, offset: 0, index: 3)
let threadsPerGroup = MTLSize(width: 256, height: 1, depth: 1)
let groups = MTLSize(width: (boards.count + 255) / 256, height: 1, depth: 1)
encoder.dispatchThreadgroups(groups, threadsPerThreadgroup: threadsPerGroup)
encoder.endEncoding()
commandBuffer.commit()
commandBuffer.waitUntilCompleted()  // or use addCompletedHandler for async readback

let scores = scoresBuffer.contents().bindMemory(to: Int32.self, capacity: boards.count)
// scores[i] is the v0 score for flat cell i % 225 of position i / 225.
```

Notes:

- `.storageModeShared` is correct on iOS (unified memory); on macOS with a
  discrete GPU use `.storageModeManaged` and `didModifyRange` /
  `synchronize`, or a blit to a shared readback buffer.
- For benchmarking, reuse the buffers across runs and time
  `waitUntilCompleted` over many dispatches — the same discipline as the
  web benchmark (`src/kernel/scoring.ts`), which includes the readback in
  its timing.
- Before trusting any Metal numbers, run the fixture cross-check: score
  the 60 positions in `ml/tests/kernel_fixtures.json` and compare against
  their `scores` arrays — 0 mismatches over 13,500 cells required.
