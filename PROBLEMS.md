# PROBLEMS — Review of the coverage-matrix branch

Audit of work on `coverage-matrix` against `~/Downloads/axon_onnx_coverage_plan.md`
through commit `e6b7a47 Phase 5: add QLinearMatMul`. Verified locally:
`mix test test/axon_onnx/coverage_test.exs` is green and `mix test
test/axon_onnx/deserialize_test.exs` is 106/106 green, so the registry is
honest about what passes — but several phase deliverables are misrepresented
or skipped, and at least one set of "passing" cases passes by accident.
Findings are ranked by severity.

## High severity

### 1. Phase 2 is plumbing only — dispatch is still opset-blind
Commit `adf299a` installs `AxonOnnx.Deserialize.opset_version/1` and threads
opsets through the process dict, then explicitly defers the actual refactor:

> Per-operator opset branching arrives in Phase 3 where it's the natural fix
> for divergent semantics (Clip's input migration, Resize's rewrites,
> Squeeze/Unsqueeze axes attribute→input).

The plan was unambiguous:

> Refactor dispatch so a builder is selected by (op_type, opset_version), with
> the model's opset import resolving which implementation runs.
> Do this **before** mass-implementing new ops, otherwise every op added now
> becomes version-debt to be reworked later.

`opset_version/1` has zero call sites in `lib/axon_onnx/deserialize.ex`
(`grep -n "Deserialize.opset_version\|opset_version()" lib/axon_onnx/deserialize.ex`
is empty). Every op added in Phases 3–5 was added without consulting it.
`Clip`, `Squeeze`, and `Unsqueeze` still branch by input *arity* rather than
declared opset — the exact "brittle" shortcut COVERAGE_AUDIT.md §3 flagged
and the plan called out as a canary. Defaults that differ across opset
versions (`Clip.min/max`, `Resize.coordinate_transform_mode`,
`BatchNormalization` training-mode outputs, `Dropout.training`) are still
not honoured.

**Fix:** Actually do Phase 2 before continuing — make dispatch take
`(op_type, opset_version)`, and rewrite Clip/Squeeze/Unsqueeze to branch on
the declared opset, not input arity.

### 2. Phase 4 was largely skipped; the loss-function commits are mislabelled
Phase 4 in the plan is "control flow & dynamic shapes (Nx lowering)" with
`If`, `Loop`, `Scan`, and dynamic-shape ops as the deliverables. Reality:

- `If` got a one-line cond-fn fix (good, but trivial — it's the
  pre-existing handler, not new lowering).
- `Loop`: not implemented. `COVERAGE.md` shows `Loop | 11 | 0` passing.
- `Scan`: not implemented. `COVERAGE.md` shows `Scan | 2 | 0` passing.
- Dynamic-shape ops: not addressed. `NonZero` still requires a compile-time
  constant input (`constant!/4`), so the case the plan called out — runtime
  `NonZero` — does not work.
- The two commits actually shipped under the "Phase 4" label
  (`891b7f0 NegativeLogLikelihoodLoss`, `caf2ab0 SoftmaxCrossEntropyLoss`)
  are loss functions. They are not control-flow ops and have no relationship
  to the Phase 4 deliverables.

Phase 4 is the gating phase for non-feedforward models (RNN/decoder/beam
search exports), and it is the only place in the plan that explicitly
demands raw-Nx lowering of subgraphs as closures rather than Axon subgraphs.
Skipping it while marking it "done" by relabelling unrelated work is the
biggest divergence from the plan.

**Fix:** Either retitle those two commits as Phase 3 work and reopen Phase 4,
or actually implement `Loop`/`Scan`/dynamic shapes. The plan explicitly says
"Refactor the deserializer so it can build a subgraph into an Nx-callable
closure, not only a top-level Axon model" — that refactor has not happened;
`If` still calls `graph_to_axon/2` eagerly (deserialize.ex:2236–2237).

### 3. Phase 5 is half a phase
The plan: "Quantization & the type-constraint matrix … `QuantizeLinear`,
`DequantizeLinear`, `QLinearConv`, `QLinearMatMul`, etc.: requires a coherent
int8/uint8 story end-to-end."

What landed: `QuantizeLinear`, `DequantizeLinear`, and (in `e6b7a47`)
`QLinearMatMul` — the latter only for the u8 corpus variants; see
item 4b for why the s8 variants are misdiagnosed. `QLinearConv` (1/1
unsupported), `ConvInteger` (0/2), `MatMulInteger` (0/1), and
`DynamicQuantizeLinear` (the non-expanded forms — 0/3 from the
test perspective; the three `*_expanded` cases pass only because they
decompose into Sub/Mul/Cast/etc.) are still untouched.

Type-constraint coverage — the plan's other half of Phase 5 — was not
worked at all. Type-cast failures dominate the unsupported counts in
`COVERAGE.md` (`Cast | 284 | 10 | 274`, `Reshape | 160 | 1 | 159`,
`Reciprocal | 40 | 2 | 38`). Many of those are likely single-dispatch
issues in `tensor!/1` / `Nx.as_type` rather than missing ops, so leaving
this untouched while declaring "Phase 5" done leaves a great deal of easy
coverage on the table.

**Fix:** Don't claim Phase 5 is shipped. Either complete the quantized-op
family (Q-prefixed and Integer-input ops) and the dtype-coverage sweep, or
break Phase 5 into "5a quantize/dequantize" + "5b QLinear/Integer family" +
"5c dtype matrix" and mark progress accordingly.

### 4. `known_bug` status is unused, so known correctness defects masquerade as "unsupported"
The registry has the three states the plan asks for (`passing |
unsupported | known_bug`) but every non-passing case is `:unsupported`
(`grep -c ":known_bug" lib/axon_onnx/coverage/registry.ex` is 0 outside the
moduledoc). Several commits explicitly acknowledge correctness bugs:

- `61a361b`: "rounding-tie behaviour on int16" in QuantizeLinear.
- `06eb8bb`: "Axon's transpose dilation handling appears to differ from
  the spec" (ConvTranspose dilations).
- COVERAGE_AUDIT §5: `MaxPool` `dilations`, `auto_pad: SAME_LOWER`
  collapsing to `:same`.

These are known divergences from the spec but are filed in the registry
identically to ops that simply aren't implemented. Future readers can't
tell the difference, and "promote the registry entry once you fix it" loses
its forcing function for the known-bug class.

**Fix:** Use `:known_bug` for these. The registry note field is the place
to record the discrepancy.

### 4b. `quantize_target_type/1` silently falls back to `{:u, 8}` for runtime zero-point inputs — int16/int8/uint16 "passes" are u8 by coincidence

`lib/axon_onnx/deserialize.ex:3293-3302`:

```elixir
defp quantize_target_type(nil), do: {:u, 8}
defp quantize_target_type(%Nx.Tensor{} = t), do: Nx.type(t)
defp quantize_target_type(%Axon{} = node) do
  case get_axon_node(node) do
    %Axon.Node{op: :constant, opts: [value: v]} -> Nx.type(v)
    _ -> {:u, 8}
  end
end
```

This is called by both `QuantizeLinear` and `QLinearMatMul` to determine the
output dtype from `y_zero_point`. The corpus models for these ops declare
`y_zero_point` as a graph input (no initializer), which `input!/4` returns
as a non-constant `%Axon{}` node — so the third clause falls through to
`{:u, 8}` regardless of the model's declared int16 / int8 / uint16 type.

Verified by inspecting the corpus protos:

- `test_quantizelinear`: `y_zero_point` elem_type 2 (u8) — passes
  because the default matches.
- `test_quantizelinear_axis`: u8 — passes for the same reason.
- `test_quantizelinear_uint16`: elem_type 4 (u16) declared — defaults to
  u8, fails, marked `:unsupported`.
- `test_quantizelinear_int16`: elem_type 5 (s16) declared — same.
- `test_qlinearmatmul_2D_int8_*`: s8 declared — defaults to u8, fails.

Commit `0750e15` claims "Basic Q/DQ at u8/s8/u16/s16" works; only u8
actually works for runtime zero-points. Commit `e6b7a47` claims the int8
QLinearMatMul failures are "round-to-even behaviour around the int8
boundary" — they're actually outputting u8 instead of s8.

This is *not* a passing-test correctness bug today (the cases that pass
genuinely match the golden tensor), but the registry note field
should record the truth, and the fix is to consult the declared
`ValueInfoProto` elem_type via `model.graph.input` rather than the Axon
node's runtime view. That information is dropped at deserialize.ex:104
(`Axon.input(name, shape: input_shape)`) and would need to be carried
through.

## Medium severity

### 5. `padding!/4` silently collapses `SAME_LOWER` to `:same` (= `SAME_UPPER`)
`lib/axon_onnx/deserialize.ex:3131-3139` — `SAME_LOWER` falls through to
`:same`, with a commented-out correct implementation. COVERAGE_AUDIT.md
predicted this would surface as failing corpus cases under Phase 3. It
wasn't fixed. This affects every conv/pool op that's declared
`auto_pad=SAME_LOWER` in the model, and the bug is silent (wrong outputs,
not a crash).

### 6. Coverage runner's input-order fix was not propagated to the legacy harness
Commit `c624785` fixed an input-ordering bug in
`AxonOnnx.Coverage.run_case/2` — it had been zipping `input_N.pb` against
`Map.keys(Axon.get_inputs/1)` (alphabetical) instead of the proto's
`graph.input` order. The exact same bug still exists in
`OnnxTestHelper.check_onnx_test_case!/3` (`test/test_helper.exs:99-106`).
Every legacy test for a model whose input names aren't alphabetical is
either silently incorrect or passes by coincidence. The fix is a 5-line
edit; it should have been applied to both call sites in the same commit.

### 7. `ScatterElements` reduction modes promise more than they deliver
`lib/axon_onnx/deserialize.ex:1656-1661` — the comment claims:

> the Nx primitives are `indexed_put` (reduction=none), `indexed_add`
> (reduction=add), and a manual scatter-accumulate for mul/min/max.

The implementation at line 3339-3348 only handles `none` and `add`;
`mul/min/max` raise `ArgumentError`. The comment should match the code, or
the code should match the comment.

### 8. `Pad` only supports `mode=constant`
`lib/axon_onnx/deserialize.ex:2986-2998` and `3019-...` — `case mode do`
has a single branch for `"constant"`, so `"reflect"`, `"edge"`, and `"wrap"`
modes raise `CaseClauseError` rather than `ArgumentError` with a helpful
message. Pre-existing, not introduced by this branch, but Phase 3 was the
"shape / tensor manipulation" phase where Pad was supposed to be
re-examined.

### 9. COVERAGE.md was momentarily stale between commits
Each phase commit updates COVERAGE.md by hand-running `mix
axon_onnx.coverage`. The previous-to-last commit had a transient
state where the numbers in `COVERAGE.md` lagged the registry. Not a defect
in the code, but the workflow has no enforcement that COVERAGE.md is
regenerated — a pre-commit hook or a CI assertion would catch this.

## Low severity / housekeeping

### 10. `light/` category is in `@categories` but contains no test cases
`AxonOnnx.Coverage.categories/0` includes `light`, but the directory
contains flat `.onnx` files, not `<name>/model.onnx` test cases, so
`discover/0` silently returns nothing for it. Either drop `light` from the
list, or handle its structure explicitly.

### 11. `RandomNormalLike` uses `op_name: :random_uniform_like`
`lib/axon_onnx/deserialize.ex:2910` — copy-paste from the Uniform path. The
layer still works but the op_name is wrong for introspection.

### 12. `Mean` divides by an integer
`lib/axon_onnx/deserialize.ex:632` — `Nx.divide(sum, n)` where `n` is the
integer count. ONNX Mean is f-type-only per the spec, so this is fine in
practice, but `Nx.divide` on integer inputs would integer-truncate. A
`n * 1.0` would document intent.

### 13. `If` still builds Axon subgraphs eagerly
`lib/axon_onnx/deserialize.ex:2236-2237` calls `graph_to_axon/2` for each
branch as a full Axon graph. This is what COVERAGE_AUDIT.md §5 flagged as
the blocker for `Loop`/`Scan`. Refactoring the subgraph deserializer to
return an Nx-callable closure was a stated Phase 4 deliverable and would
unlock `Loop`/`Scan` together.

## What is solid

- **Phase 0 audit** (`COVERAGE_AUDIT.md`) is accurate and complete.
- **Phase 1 inverted harness** is correct in shape: discover all cases,
  compare against an expected-failure registry, fail loudly on
  `:unsupported→passing` (so regression *and* progress are forced). The
  `mix axon_onnx.coverage` task regenerates a real coverage report.
- **The 669/1793 passing count is real** — every entry in the registry
  matches a file on disk, and `mix test test/axon_onnx/coverage_test.exs`
  is 1793/1793 with the current registry.
- The legacy `deserialize_test.exs` (106 tests) still passes; nothing was
  silently removed from it during this work.
- Many individual op implementations are clean — `Trilu`, `Hardmax`,
  `LayerNormalization`, `RMSNormalization`, `GatherElements`, `CumSum`,
  `QuantizeLinear` core path, and `SoftmaxCrossEntropyLoss` are all
  reasonable raw-Nx lowerings with axis-normalisation done correctly.

## Suggested order of operations

1. Decide whether to retitle the loss-function commits or to back-fill
   `Loop`/`Scan` before any more "phase complete" claims (item 2).
2. Do Phase 2 properly — opset-version-aware dispatch — before adding
   more op families (item 1).
3. Promote known divergences from `:unsupported` to `:known_bug` so the
   registry stops conflating them (item 4).
4. Fix `padding!/4 SAME_LOWER` and the legacy-harness input-order bug
   (items 5, 6) — both are short and unblock real coverage.
5. Sweep the dtype matrix (item 3) — the big wins on `Cast` /
   `Reshape` / etc. are likely shallow.
