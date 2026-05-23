defmodule AxonOnnx.DeserializeOpsetTest do
  @moduledoc """
  Unit tests for the Phase 2 opset-version plumbing.

  These tests are intentionally narrow: they verify that the process-dict
  lifecycle is correct (`opset_version/1` returns `nil` outside an import
  call, the right version inside, and is cleared after the call returns —
  even if the call raises). Per-operator opset branching is exercised
  indirectly through the corpus suite in `AxonOnnx.CoverageTest`.
  """
  use ExUnit.Case, async: false

  alias AxonOnnx.Deserialize

  @opsets_key {AxonOnnx.Deserialize, :opsets}

  setup do
    # Defensive: a prior crashing test must not leak the process dict slot.
    on_exit(fn -> Process.delete(@opsets_key) end)
    :ok
  end

  describe "opset_version/1" do
    test "returns nil when called outside a deserialization call" do
      Process.delete(@opsets_key)
      assert Deserialize.opset_version() == nil
      assert Deserialize.opset_version("") == nil
      assert Deserialize.opset_version("ai.onnx.ml") == nil
    end

    test "returns the version for the default ai.onnx domain when set" do
      Process.put(@opsets_key, %{"" => 13, "ai.onnx.ml" => 2})
      assert Deserialize.opset_version() == 13
      assert Deserialize.opset_version("") == 13
      assert Deserialize.opset_version("ai.onnx.ml") == 2
      assert Deserialize.opset_version("custom.domain") == nil
    end
  end

  describe "process dict lifecycle in AxonOnnx.import/2" do
    test "import sets opset_version during the call and clears it after" do
      # Pick any corpus case that imports successfully.
      model_path = "test/cases/node/test_abs/model.onnx"

      Process.delete(@opsets_key)
      assert Deserialize.opset_version() == nil

      {_model, _params} = AxonOnnx.import(model_path)

      # The dict slot must be cleared on a successful return so the value
      # doesn't bleed into subsequent calls in the same process.
      assert Deserialize.opset_version() == nil
    end

    test "import clears opset_version even when deserialization raises" do
      # Take a real model that decodes cleanly, swap its op_type for one
      # that doesn't dispatch, and re-encode. This exercises the after-clause
      # of `to_axon/2` without depending on hand-building a complete
      # ModelProto from scratch.
      raw = File.read!("test/cases/node/test_abs/model.onnx")
      model = Onnx.ModelProto.decode!(raw)
      [first_node | rest] = model.graph.node
      tampered_node = %{first_node | op_type: "DefinitelyNotARealOp"}
      tampered_graph = %{model.graph | node: [tampered_node | rest]}
      tampered_model = %{model | graph: tampered_graph}
      bytes = Onnx.ModelProto.encode!(tampered_model) |> IO.iodata_to_binary()

      Process.delete(@opsets_key)

      assert_raise ArgumentError, ~r/unsupported "DefinitelyNotARealOp"/, fn ->
        AxonOnnx.load(bytes)
      end

      # After-clause must have cleared the dict even though dispatch raised.
      assert Deserialize.opset_version() == nil
    end

    test "nested imports restore the outer opset on return" do
      # Two corpus cases at potentially different opset versions; we
      # interleave one inside the (notional) other using Process.put as a
      # stand-in for an outer call.
      outer = %{"" => 9}
      Process.put(@opsets_key, outer)
      assert Deserialize.opset_version() == 9

      # An inner import installs its own opsets and restores ours on exit.
      AxonOnnx.import("test/cases/node/test_abs/model.onnx")

      assert Deserialize.opset_version() == 9
      Process.delete(@opsets_key)
    end
  end
end
