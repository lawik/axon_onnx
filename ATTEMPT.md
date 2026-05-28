# ATTEMPT.md — using axon_onnx on real models

Running log of what happens when I try to load and run real ONNX models
through this library. Goal is to find concrete breakage / friction beyond
the synthetic test corpus that `COVERAGE.md` reports against.

Methodology:

1. Start small — a tiny classifier — and confirm import → params → predict
   all work end-to-end against a known input.
2. Climb the size/complexity ladder one rung at a time: small CNN → medium
   CNN → small transformer.
3. Stop at the point where the library can no longer make a model work, and
   capture the failure (op, error message, where in `deserialize.ex` it
   blows up if I can tell).
4. LLMs / diffusion: only if a small *quantized* variant exists, since this
   machine only has ~18 GB free on `/`. Realistically expect to top out
   before that.

For each model I capture:

- Source URL and on-disk size.
- Whether `AxonOnnx.import/2` returns without raising.
- If it raised: the op / message, and (when obvious) the file:line in
  `lib/axon_onnx/deserialize.ex`.
- Whether `Axon.build` + predict produces output.
- Whether the model produces sensible output on a known input (numerics
  check) — not always done; flagged when skipped.

Environment:

- Branch: `coverage-matrix`
- macOS Darwin 25.4.0, Apple Silicon.
- Disk free at start: ~18 GB on `/`.
- Elixir 1.19.5 / OTP 28, `mix compile` warns-but-builds. Four warnings
  about `defp recur_nodes/2` clauses being non-contiguous (cosmetic, not
  blockers).

Scratch files written to `tmp/models/`. The directory is gitignored — er,
isn't — leave a `.gitignore` cleanup as a TODO if this work goes to a PR.

---

## Attempts

### 1. MNIST (CNTK, opset 8, 26 KB) — FAIL: initializer-as-input

- URL: `github.com/onnx/models/.../mnist/model/mnist-8.onnx`
- IR 3, opset 8, ops: `Add Conv MatMul MaxPool Relu Reshape`, 12 nodes.

```elixir
AxonOnnx.import("tmp/models/mnist-8.onnx")
# ** (ArgumentError) unable to build model from ONNX graph, expected
#    value Parameter193 to be a graph input, but it was not present
#    in built graphs
#     lib/axon_onnx/deserialize.ex:4775 AxonOnnx.Deserialize.axon!/2
```

**What's actually wrong:** this model lists every weight tensor in
`graph.input` as well as `graph.initializer`. That's the pre-IR-4
convention (and CNTK is famous for emitting it). axon_onnx treats anything
in `graph.input` as a runtime input — but the param has no producer node,
so when a downstream Conv/MatMul asks for it via `input!`, it can't be
resolved.

**Fix shape:** during graph load, drop any `graph.input` entry whose name
also appears in `graph.initializer`. That's the standard ONNX-spec
behaviour for IR<4 models.

### 2. SqueezeNet 1.0 (opset 12, 5 MB) — FAIL: 2-input Dropout unmatched

- URL: `github.com/onnx/models/.../squeezenet/model/squeezenet1.0-12.onnx`
- IR 7, opset 12, ops: `Concat Conv Dropout GlobalAveragePool MaxPool Relu Softmax`, 66 nodes.
- This one has a clean modern structure (initializers separate from inputs).

```elixir
AxonOnnx.import("tmp/models/squeezenet1.0-12.onnx")
# ** (ArgumentError) unsupported "Dropout"
#    lib/axon_onnx/deserialize.ex:4661 (fallback unsupported handler)
```

**What's actually wrong:** `deserialize.ex:4432` only pattern-matches
`Dropout` with `input: [inp_name]` (single input). Opset ≥ 12 moved
`ratio` from an attribute to a second input, and `training_mode` is an
optional third input. The SqueezeNet node is `inputs: ['fire9/concat_1', '122']`
— two inputs — so the clause doesn't match and the fallback raises.

**Fix shape:** add a 2/3-input clause for opset-12+ Dropout. In inference
mode (`training_mode` absent or 0), Dropout is a pass-through; the
existing single-input branch already handles `ratio == 0 → Axon.nx(identity)`,
so the new clause would mostly extract the ratio constant and reuse that
path.

### 3. MobileNetV2 (opset 12, 14 MB) — FAIL: Clip params passed as parents

- URL: `github.com/onnx/models/.../mobilenet/model/mobilenetv2-12.onnx`
- IR 7, opset 12, ops include `Clip` (the ReLU6 in MBv2's inverted residual blocks).

```elixir
AxonOnnx.import("tmp/models/mobilenetv2-12.onnx", batch_size: 1)
# ** (ArgumentError) invalid input given to layer: %Axon.Node{... op: :conv ...}
#    axon/lib/axon.ex:394 Axon.split_inputs/2
#    axon_onnx/lib/axon_onnx/deserialize.ex:4140  (Clip branch)
#    axon_onnx/lib/axon_onnx/deserialize.ex:1632  (next Conv after Clip)
```

**What's actually wrong:** the Clip handler at `deserialize.ex:4131-4143`,
when fed a normal node plus `Nx.Tensor` min/max, does:

```elixir
min = Axon.param(min_name, fn _ -> Nx.shape(min) end)
max = Axon.param(max_name, fn _ -> Nx.shape(max) end)
layer = Axon.layer(fun, [inp, min, max], name: output_name, op_name: :clip)
```

It passes `%Axon.Parameter{}` values into the `inputs` (parents) list of
`Axon.layer/3`. Parents must be `%Axon{}` graph nodes — parameters belong
in the `:parameters` keyword option. The graph "builds" without raising
because `Axon.layer/3` is lazy, but the next layer that asks for inputs
(`Axon.split_inputs`) chokes when it walks the parent list and finds a
parameter struct.

**Fix shape:** rewrite that branch to either (a) inline `min`/`max` as
`Axon.constant(...)` parents (simplest, since they're known tensors at
build time), or (b) actually pass them via `:parameters` and let the
`fun` receive them via the right calling convention. (a) is preferable —
they're literally constants from the initializer table.

### 4. Score so far

Three real models, three different `deserialize.ex` bugs, none related to
the per-op coverage matrix `COVERAGE.md` reports against. Each is a
graph-shape issue rather than an op-implementation issue:

| # | Model | Failure class |
|---|---|---|
| 1 | MNIST-8 | IR<4 graph convention (initializer also listed as input) |
| 2 | SqueezeNet1.0-12 | Op clause matches wrong arity for opset |
| 3 | MobileNetV2-12 | Op clause produces structurally invalid Axon graph |

Pattern: the per-op tests in the corpus cover *one* shape of each op, so
real models that hit other valid shapes fall off the cliff.

---

_Continuing with more models …_
