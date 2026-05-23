# axon_onnx — Phase 0 Coverage Audit

Status snapshot taken on 2026-05-23 to ground the move toward general-case ONNX
operator coverage. This file is a one-time deliverable; ongoing coverage tracking
lives in `COVERAGE.md` (Phase 1) and the expected-failure registry.

## 1. Harness mechanism

The test runner is a **hand-curated whitelist**, not the full corpus.

- The ONNX backend test data is materialized into `test/cases/` on first `mix
  test` by `test/test_helper.exs:273-300`. It calls `backend-test-tools
  generate-data` (Python) and then copies the contents of `onnx.backend.test`'s
  `data/` directory into `test/cases/`. Subdirectories: `node/`,
  `pytorch-converted/`, `pytorch-operator/`, `simple/`, `real/`, `light/`.
  - With the currently installed `onnx==1.20.1`, `data/node/` contains **1653**
    test cases. `pytorch-converted` 82, `pytorch-operator` 35, `simple` 23,
    `light` 19.
- The corpus is consumed by `test/axon_onnx/deserialize_test.exs` calling
  `check_onnx_test_case!("<category>", "<test_name>")` for each case **the
  authors explicitly listed**. Comments at the top of the file (`# TODO`, `#
  check_onnx_test_case!(...)`) preserve dozens of skipped cases.
  - Active `check_onnx_test_case!` calls: **182**.
  - Commented-out `check_onnx_test_case!` calls: **710**. So we are running
    roughly **20%** of the cases the file itself acknowledges, and a far
    smaller fraction of the 1653-case `node/` corpus.
- The runner only executes ops that have been wired up in the dispatch. There
  is no mechanism to discover newly-supported ops or to flag regressions on
  cases that aren't called out by name. The whitelist hides both gaps and
  progress.

This confirms the open question from the plan: the harness is a whitelist, and
Phase 1 needs to invert it into a coverage matrix backed by an expected-failure
registry.

## 2. Op-type dispatch

`AxonOnnx.Deserialize.recur_nodes/2` (`lib/axon_onnx/deserialize.ex:74-2214`) is
the single dispatch point. It is implemented as Elixir pattern-match clauses
keyed on `%Onnx.NodeProto{op_type: ...}`. Several families are generated via
`for` over module attributes; the rest are individual clauses. The final clause
(`deserialize.ex:2212-2214`) raises `ArgumentError "unsupported #{op_type}"`.

Op_type clause count: **51 unique dispatch heads** in the file, fanning out
through generators to **78 ONNX op_types handled**.

### Generated families

- **Unary Nx ops** (`@nx_op_types`, `deserialize.ex:78-132`, 26 ops): `Abs`,
  `Acos`, `Acosh`, `Asin`, `Asinh`, `Atan`, `Atanh`, `Ceil`, `Cos`, `Cosh`,
  `Erf`, `Floor`, `HardSwish`, `Identity`, `IsInf`, `IsNaN`, `Log`, `Neg`,
  `Not`, `Round`, `Reciprocal`, `Sign`, `Sin`, `Sinh`, `Sqrt`, `Tan`. All lower
  via `Axon.nx/3` to a raw `Nx` function (or fold into a constant if the input
  is constant).
- **Activations** (`@activation_op_types`, `deserialize.ex:134-191`, 13 ops):
  `Celu`, `Elu`, `Exp`, `HardSigmoid`, `LeakyRelu`, `LogSoftmax`, `Relu`,
  `Selu`, `Sigmoid`, `Softmax`, `Softplus`, `Softsign`, `Tanh`. All map to an
  `Axon` activation layer.
- **Reductions** (`@reduction_op_types`, `deserialize.ex:193-266`, 11 ops):
  `ArgMax`, `ArgMin`, `ReduceL1`, `ReduceL2`, `ReduceLogSum`, `ReduceLogSumExp`,
  `ReduceMax`, `ReduceMean`, `ReduceMin`, `ReduceProd`, `ReduceSumSquare`.
  Lower to `Axon.layer/3` wrapping a raw `Nx` reducer. Note `ReduceSum` is
  conspicuously absent.
- **Bias-aware binary ops** (`@builtin_binary_op_types`,
  `deserialize.ex:268-352`, 3 ops): `Add`, `Sub`, `Mul`. `Add` has a
  bias-fusion fast path; otherwise these become `Axon.add`/`subtract`/`multiply`
  or a custom `trainable_binary_layer`.
- **Generic binary ops** (`@binary_op_types`, `deserialize.ex:354-432`, 11
  ops): `And`, `Div`, `Equal`, `Greater`, `GreaterOrEqual`, `Less`,
  `LessOrEqual`, `Mod`, `Or`, `Pow`, `Xor`. Lower to `Axon.layer/3` over a raw
  `Nx` function.

### Individual op handlers

In file order (with primary strategy and starting line):

| Op | Strategy | Line |
|---|---|---|
| `BitShift` | Axon.layer + raw Nx | 434 |
| `Cast` | Axon.nx + `Nx.as_type` | 570 |
| `LRN` | Axon.nx + helper `lrn` | 592 |
| `Gather` | `Axon.embedding` or constant `Nx.take` | 605 |
| `MatMul` | `Axon.dense` or custom matmul layer | 646 |
| `Gemm` | `Axon.dense` + scaling | 683 |
| `MaxPool` | `Axon.max_pool` | 789 |
| `AveragePool` | `Axon.avg_pool` | 853 |
| `Conv` | `Axon.conv` | 914 |
| `BatchNormalization` | `Axon.batch_norm` | 1026 |
| `InstanceNormalization` | `Axon.instance_norm` | 1075 |
| `Concat` | `Axon.concatenate` | 1120 |
| `Split` (1-input) | `Axon.split` | 1139 |
| `Constant` | `Axon.constant` | 1159 |
| `ConstantOfShape` | `Axon.constant` + broadcast | 1205 |
| `Reshape` | `Axon.reshape` (static shapes only) | 1233 |
| `Expand` | Axon.layer + `Nx.multiply`+`Nx.broadcast` | 1285 |
| `Range` | `Axon.constant` | 1325 |
| `Flatten` | `Axon.flatten` | 1344 |
| `Slice` (3/4/5-input) | `slice_layer` helper (raw Nx) | 1353/1367/1384 |
| `Shape` | `Axon.constant` or layer | 1401 |
| `Transpose` | `Axon.transpose` | 1459 |
| `Unsqueeze` (attribute form) | `Axon.nx` + `Nx.new_axis` | 1496 |
| `If` | `Axon.cond` over recursively-built branches | 1540 |
| `Where` | `Axon.layer` + `Nx.select` | 1569 |
| `CumSum` | `Axon.nx` + `Nx.window_sum` | 1631 |
| `Clip` (4 arities) | `Axon.nx`/`Axon.layer` + `Nx.clip` | 1660/1693/1725/1757 |
| `Squeeze` (attribute / input forms) | Axon.layer + `Nx.squeeze` | 1791/1820 |
| `Split` (2-input) | Axon.split | 1846 |
| `EyeLike` | layer/constant + `Nx.eye` | 1868 |
| `RandomUniform` / `Like` | constant `Nx.Random.uniform` | 1897/1920 |
| `RandomNormal` / `Like` | constant `Nx.Random.normal` | 1976/1999 |
| `Dropout` | `Axon.dropout` | 2055 |
| `Pad` (1-input attrs / 2-input dynamic) | `Axon.nx` + `Nx.pad` | 2102/2131 |
| `NonZero` | `Axon.constant` over folded indices | 2176 |
| **Fallback** | raises `ArgumentError` | 2212 |

### Conspicuous absences in dispatch (versus the spec)

These ops have **no handler** today and will crash on the fallback even if the
shape of the input would otherwise be fine: `ReduceSum`, `Tile`, `Trilu`,
`TopK`, `Resize`, `Upsample`, `GatherND`, `GatherElements`, `ScatterND`,
`ScatterElements`, `OneHot`, `LpNormalization`, `LpPool`, `ConvTranspose`,
`ConvInteger`, `MatMulInteger`, `QuantizeLinear`, `DequantizeLinear`,
`QLinearConv`, `QLinearMatMul`, `Loop`, `Scan`, `SequenceAt`, `SequenceConstruct`,
`SequenceErase`, `SequenceInsert`, `SequenceLength`, `Compress`, `Einsum`,
`MeanVarianceNormalization`, `GroupNormalization`, `LayerNormalization`,
`DepthToSpace`, `SpaceToDepth`, `Shrink`, `Mean`, `ReverseSequence`, `Unique`,
`NonMaxSuppression`, `RoiAlign`, `OptionalGetElement`/`OptionalHasElement`,
`StringConcat` and friends.

The full gap will fall out of Phase 1; the list above is illustrative.

## 3. Opset-version awareness

**Opset-blind.** The deserializer never reads `model.opset_import`, never
branches on opset version, and never resolves per-version defaults. Confirmed by
grepping `lib/axon_onnx/deserialize.ex` for `opset`, `ir_version`, `version` —
no matches that affect dispatch (only one comment in `serialize.ex` about
model versions).

This is exactly the canary situation Phase 2 warns about. `Clip` has been
split into four arity-specific clauses to cope with the opset-11 min/max
attribute→input migration; `Squeeze`/`Unsqueeze` have similarly been split into
attribute-form and input-form clauses to cope with the opset-13 migration.
Both currently rely on input shape rather than declared opset version, which is
brittle (correct for now, but the *defaults* of those attributes differ across
versions and aren't honored). Other ops with version-dependent defaults
(`Resize`, `ReduceSum`, `BatchNormalization` training/inference outputs,
`Dropout`'s training mode flag) are not handled at all or are handled with a
single one-size implementation.

## 4. Baseline

- ONNX (Python): **1.20.1**
- onnxruntime: **1.23.2**
- Elixir: **1.19.5** / Erlang/OTP **28**
- `axon`: **0.5.1** (hex)
- `nx`: **0.5.3**
- `exla`: **0.5.1**
- `protox`: **1.6.10**

Numerical pass count was not captured in this audit. The existing suite is
intentionally narrow and serializes ops through Axon, so its count would not
be informative as a coverage baseline — the Phase 1 matrix will produce the
real baseline directly from the 1653-case `node/` corpus (plus
`pytorch-converted`, `pytorch-operator`, `simple`, `light`). Treat that future
number as the starting point.

## 5. Open risks for downstream phases

- `axon ~> 0.5` is fairly old; `Axon.layer/3` and `Axon.nx/3` are the only
  escape hatches available for "lower to raw Nx" today. If newer Axon adds
  cheaper escape hatches or first-class subgraph support we should pick them
  up.
- The current `If` handler builds full Axon subgraphs eagerly. To make `Loop`
  and `Scan` feasible (Phase 4) we will need to recursively build a subgraph
  as an `Nx`-callable closure, not just an Axon model — the deserializer
  already nearly does this for `If` and can be refactored.
- Several "Axon layer" mappings quietly drop attributes the spec defines
  (e.g. `auto_pad: SAME_LOWER` is collapsed to `:same` at
  `padding!/4:2390-2398`, dropping the lower-vs-upper distinction;
  `MaxPool` `dilations` is unsupported via the underlying Axon layer). These
  will surface as failing corpus cases under Phase 1 and need raw-Nx
  re-implementations during Phase 3.
- Tensor dtype coverage in `tensor!/1` already handles f32/f16/bf16/f64,
  u8/16/32/64, s8/16/32/64; `string`/complex types raise. That's a fine
  starting point for Phase 5's type-constraint work.
