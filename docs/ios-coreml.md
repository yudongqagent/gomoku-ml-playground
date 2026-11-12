# iOS / Core ML integration

How to run a trained Gomoku model on an iPhone with Core ML — including
on the Neural Engine — and how that relates to the browser version.

## The honest headline

- **In a browser, the Neural Engine is NOT directly accessible.**
  Safari/WebKit expose no Neural Engine API. The fastest web path is
  **ONNX Runtime Web with the WebGPU execution provider**, which runs
  the `.onnx` model on the GPU. That is what the web app uses
  (`public/models/*.onnx` + `models.json`).
- **Core ML is the path to the Neural Engine, and it needs a native
  wrapper**: a Swift app (or a Swift package / Capacitor-style native
  shell around the web app) that loads the `.mlpackage` and runs
  inference through Core ML with compute units set to `.all`
  (CPU + GPU + Neural Engine). Core ML decides per-layer where to run;
  on Apple Silicon iPhones the Neural Engine takes the bulk of the
  conv work for a network of this shape.

So: browser = ONNX Runtime Web / WebGPU; native iOS app = Core ML /
Neural Engine. Same trained weights, two export formats.

## Exporting the model (on a Mac)

Core ML conversion is supported on macOS (Apple Silicon recommended).
From the repo root:

```bash
pip install coremltools
python ml/scripts/export_coreml.py \
  --checkpoint ml/runs/<run-id>/checkpoints/final.pt \
  --out GomokuNet.mlpackage
```

Alternative — convert from the already-exported ONNX file:

```bash
python ml/scripts/export_coreml.py \
  --onnx public/models/gomoku-smoke-v1.onnx \
  --out GomokuNet.mlpackage
```

The exported package has:

| | |
|---|---|
| Input | `input` — multi-array, float32, shape `[1, 8, 15, 15]` |
| Output | `policy` — float32 `[1, 225]` logits, row-major (index `y*15+x`) |
| Output | `value` — float32 `[1, 1]`, tanh in `[-1, 1]`, current player's perspective |

The `.mlpackage` itself carries no compute-unit setting — the app
chooses compute units at load time (see below). If `coremltools` is
not installed, the script exits with install instructions instead of
faking an export.

> Note: Core ML export was code-complete but not executed on the Linux
> VM used for development (coremltools not installed there). Run the
> command above on a Mac to produce a real `.mlpackage`.

## Adding the model to an Xcode project

1. Drag `GomokuNet.mlpackage` into the Xcode project (or add it to the
   app target). Xcode generates a type-safe Swift class from the model
   (by default named after the file, e.g. `GomokuNet`).
2. The generated class has an initialiser taking an
   `MLModelConfiguration`:
   `GomokuNet(configuration: MLModelConfiguration())`.
3. The generated input type is `GomokuNetInput`, with an `input`
   property of type `MLMultiArray`; the generated output type is
   `GomokuNetOutput`, with `policy` and `value` properties of type
   `MLMultiArray`.

If you prefer not to use the generated class, load the model directly
with `MLModel(contentsOf:configuration:)` and call
`prediction(from:)` with an `MLFeatureProvider` (e.g. an
`MLDictionaryFeatureProvider` or `MLFeatureValue(multiArray:)`).

## Swift integration sketch

```swift
import CoreML

/// Runs the Gomoku policy+value network via Core ML.
final class GomokuModel {
    private let model: GomokuNet  // generated class from GomokuNet.mlpackage

    init() throws {
        let config = MLModelConfiguration()
        // CPU + GPU + Neural Engine; Core ML places each layer.
        config.computeUnits = .all
        self.model = try GomokuNet(configuration: config)
    }

    /// Runs inference on one position.
    /// - Parameter planes: 8 planes of 15x15 Float32, in the contract order
    ///   (see ml/README.md "Network contract"): own stones, opponent stones,
    ///   last move, side-to-move, own 2 plies ago, opp 2 plies ago,
    ///   own 4 plies ago, opp 4 plies ago — all relative to the player to move.
    /// - Returns: (policyLogits[225], value) with value in [-1, 1] from the
    ///   current player's perspective.
    func predict(planes: [Float]) throws -> (policy: [Float], value: Float) {
        precondition(planes.count == 8 * 15 * 15)
        let inputArray = try MLMultiArray(shape: [1, 8, 15, 15], dataType: .float32)
        // MLMultiArray is row-major; copy the planes in directly.
        let ptr = inputArray.dataPointer.bindMemory(to: Float.self, capacity: planes.count)
        planes.withUnsafeBufferPointer { src in
            ptr.initialize(from: src.baseAddress!, count: planes.count)
        }
        let input = GomokuNetInput(input: inputArray)
        let output = try model.prediction(input: input)
        // policy: [1, 225] logits, row-major, index y*15+x.
        let policyPtr = output.policy.dataPointer.bindMemory(to: Float.self, capacity: 225)
        let policy = Array(UnsafeBufferPointer(start: policyPtr, count: 225))
        // value: [1, 1] tanh.
        let valuePtr = output.value.dataPointer.bindMemory(to: Float.self, capacity: 1)
        return (policy, valuePtr[0])
    }

    /// Best legal move by policy logit (no search — for a quick move picker;
    /// pair with MCTS for real strength, as in the Python `gomoku_ml.mcts`).
    /// - Parameter legalMoves: flat indices (y*15+x) of empty points.
    func bestPolicyMove(policy: [Float], legalMoves: [Int]) -> Int {
        legalMoves.max(by: { policy[$0] < policy[$1] }) ?? legalMoves[0]
    }
}
```

Notes on the sketch:

- `MLModelConfiguration.computeUnits = .all` is the setting that lets
  Core ML use the Neural Engine. The alternatives are `.cpuOnly`,
  `.cpuAndGPU`, and `.cpuAndNeuralEngine`.
- `MLMultiArray(shape:dataType:)` with `.float32` matches the model's
  declared input type. The array is row-major, so the 8 planes can be
  copied in as one flat buffer in plane order.
- `GomokuNetInput` / `GomokuNetOutput` / `GomokuNet` are the classes
  Xcode generates from the `.mlpackage`; the exact names follow the
  file name. If you rename the package, the class names change with it.
- `model.prediction(input:)` is the generated type-safe prediction
  method. The untyped equivalent is
  `model.model.prediction(from: MLFeatureProvider)` on the underlying
  `MLModel`.
- For MCTS, call `predict` once per leaf expansion, exactly as the
  Python `MCTS._evaluate` does — the search logic is identical; only
  the inference call changes.

## Building the 8-plane input on iOS

The input encoding must match `ml/gomoku_ml/encoding.py` exactly:

1. Plane 0: current player's stones (1.0 where present).
2. Plane 1: opponent's stones.
3. Plane 2: last move, one-hot (all zeros if no move played yet).
4. Plane 3: all 1.0 if Black is to move, else all 0.0.
5. Plane 4/5: own/opponent stones as of 2 plies ago (zeros if the game
   is shorter than 2 plies).
6. Plane 6/7: own/opponent stones as of 4 plies ago.

Keep the last few board states in the game controller so planes 4–7
can be filled; they are relative to the player who was to move at that
earlier ply (for even-ply offsets that is the same player as now).

## Performance expectations (honest)

- The smoke demo net (16 channels, 2 blocks) is tiny and runs in well
  under a millisecond per inference on any modern iPhone — but it
  plays poorly; it is a demo, not a strong opponent.
- The full config (128 channels, 10 blocks) is ~3M parameters. On an
  A20 Pro-class chip via the Neural Engine, a single forward pass is
  expected to be a few milliseconds; an MCTS with hundreds of
  simulations per move is therefore feasible in real time, but measure
  it on device — do not quote a number from this document.
- Batching leaf evaluations (evaluating several MCTS leaves in one
  Core ML call) would improve throughput further; the exported model
  has a fixed batch of 1, so batching needs a re-export with a flexible
  batch dimension (future work).

## Checklist for the native wrapper

- [ ] Export `.mlpackage` on a Mac (command above) and verify it loads
      in Xcode's model preview.
- [ ] Confirm the generated class names match the sketch (rename the
      package or the code, not both).
- [ ] Feed the 8-plane input in the exact contract order; write a unit
      test that compares Swift inference output against the Python
      model's output on the same position.
- [ ] Set `computeUnits = .all` and measure per-inference latency on
      device before choosing MCTS simulation counts.
- [ ] Remember the browser build cannot use the Neural Engine — keep
      the ONNX/WebGPU path for the web app.
