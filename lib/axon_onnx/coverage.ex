defmodule AxonOnnx.Coverage do
  @moduledoc """
  Discovery and per-case execution for the ONNX backend test corpus.

  The corpus is materialised under `test/cases/<category>/<test_name>/` by
  `test/test_helper.exs` on first run (invoking the Python
  `backend-test-tools generate-data`). Each case directory holds a `model.onnx`
  plus one or more `test_data_set_N/` directories of `input_*.pb`/`output_*.pb`
  golden tensors produced by `onnxruntime`.

  This module is the inverted harness: rather than naming each case to run,
  callers `discover/0` everything on disk and then `run_case/1` is compared to
  the expected status in `AxonOnnx.Coverage.Registry`.

  All functions are side-effect free apart from `run_case/1`, which imports
  the model and runs the comparison via `AxonOnnx.import/2` + `Axon.predict/4`.
  No ExUnit dependency; this module is callable from a Mix task.
  """

  alias AxonOnnx.Coverage.Registry

  @cases_root Path.join(["test", "cases"])
  @categories ~w(node pytorch-converted pytorch-operator simple light)

  @type case_entry :: %{category: String.t(), name: String.t(), path: String.t()}
  @type status :: :passing | :unsupported | :known_bug
  @type run_result :: :ok | {:error, term()}

  @doc "Filesystem root for the materialised corpus."
  def cases_root, do: @cases_root

  @doc "Categories scanned by `discover/0`. `real/` is excluded — it requires network downloads."
  def categories, do: @categories

  @doc "Walks `test/cases/<category>/` and returns every case directory containing a `model.onnx`."
  @spec discover() :: [case_entry()]
  def discover do
    for category <- @categories,
        dir = Path.join(@cases_root, category),
        File.dir?(dir),
        name <- File.ls!(dir) |> Enum.sort(),
        path = Path.join(dir, name),
        File.exists?(Path.join(path, "model.onnx")),
        do: %{category: category, name: name, path: path}
  end

  @doc "Looks up the registry status for a discovered case."
  @spec expected_status(case_entry()) :: status()
  def expected_status(%{category: cat, name: name}), do: Registry.status({cat, name})

  @doc "Note attached to the case in the registry, or nil."
  @spec note(case_entry()) :: String.t() | nil
  def note(%{category: cat, name: name}), do: Registry.note({cat, name})

  @doc """
  Runs a single discovered case end-to-end.

  Returns `:ok` if `AxonOnnx.import/2` succeeds AND every `test_data_set_*`'s
  predicted outputs are within tolerance of the golden outputs. Otherwise
  returns `{:error, reason}` with a string reason.

  This function never raises — it always returns `:ok` or `{:error, _}` so that
  callers can compare against the registry. Errors from any stage (import,
  inference, golden comparison) are normalised into `{:error, reason}`.
  """
  @spec run_case(case_entry(), keyword()) :: run_result()
  def run_case(%{path: path} = entry, opts \\ []) do
    atol = Keyword.get(opts, :atol, 1.0e-3)
    model_path = Path.join(path, "model.onnx")
    data_paths = path |> Path.join("test_data_set_*") |> Path.wildcard() |> Enum.sort()

    try do
      {model, params} = AxonOnnx.import(model_path)
      inputs = Axon.get_inputs(model)
      input_keys = Map.keys(inputs)

      Enum.each(data_paths, fn data_path ->
        input_paths = data_path |> Path.join("input_*.pb") |> Path.wildcard() |> Enum.sort()
        output_paths = data_path |> Path.join("output_*.pb") |> Path.wildcard() |> Enum.sort()

        inp_tensors =
          input_paths
          |> Enum.map(&pb_to_tensor/1)
          |> Enum.zip(input_keys)
          |> Map.new(fn {v, k} -> {k, v} end)

        out_tensors = Enum.map(output_paths, &pb_to_tensor/1)
        actual_outputs = Axon.predict(model, params, inp_tensors)

        case out_tensors do
          [expected] ->
            assert_close!(actual_outputs, expected, entry, atol)

          many when is_list(many) ->
            actual_list =
              case actual_outputs do
                tuple when is_tuple(tuple) -> Tuple.to_list(tuple)
                list when is_list(list) -> list
                single -> [single]
              end

            Enum.zip(actual_list, many)
            |> Enum.each(fn {a, e} -> assert_close!(a, e, entry, atol) end)
        end
      end)

      :ok
    rescue
      e -> {:error, normalize_error(e)}
    catch
      kind, value -> {:error, "caught #{inspect(kind)}: #{inspect(value)}"}
    end
  end

  defp assert_close!(actual, expected, entry, atol) do
    res = Nx.all_close(actual, expected, atol: atol, equal_nan: true) |> Nx.to_number()

    if res != 1 do
      raise "#{entry.category}/#{entry.name}: outputs diverge from golden (atol=#{atol})"
    end
  end

  defp normalize_error(%{__exception__: true} = e), do: Exception.message(e)
  defp normalize_error(other), do: inspect(other)

  @doc false
  def pb_to_tensor(pb_path) do
    pb_path
    |> File.read!()
    |> Onnx.TensorProto.decode!()
    |> tensor!()
  end

  defp tensor!(%Onnx.TensorProto{data_type: dtype, dims: dims} = tensor) do
    shape = List.to_tuple(dims)

    case dtype do
      1 -> to_nx_tensor(tensor.float_data, tensor.raw_data, {:f, 32}, shape)
      2 -> to_nx_tensor(tensor.int32_data, tensor.raw_data, {:u, 8}, shape)
      3 -> to_nx_tensor(tensor.int32_data, tensor.raw_data, {:s, 8}, shape)
      4 -> to_nx_tensor(tensor.int32_data, tensor.raw_data, {:u, 16}, shape)
      5 -> to_nx_tensor(tensor.int32_data, tensor.raw_data, {:s, 16}, shape)
      6 -> to_nx_tensor(tensor.int32_data, tensor.raw_data, {:s, 32}, shape)
      7 -> to_nx_tensor(tensor.int64_data, tensor.raw_data, {:s, 64}, shape)
      8 -> raise "unsupported Nx tensor type: string"
      9 -> to_nx_tensor(tensor.int32_data, tensor.raw_data, {:u, 8}, shape)
      10 -> to_nx_tensor(tensor.int32_data, tensor.raw_data, {:f, 16}, shape)
      11 -> to_nx_tensor(tensor.double_data, tensor.raw_data, {:f, 64}, shape)
      12 -> to_nx_tensor(tensor.uint64_data, tensor.raw_data, {:u, 32}, shape)
      13 -> to_nx_tensor(tensor.uint64_data, tensor.raw_data, {:u, 64}, shape)
      14 -> raise "unsupported Nx tensor type: C64"
      15 -> raise "unsupported Nx tensor type: C128"
      16 -> to_nx_tensor([], tensor.raw_data, {:bf, 16}, shape)
    end
  end

  defp to_nx_tensor([], <<>>, _, _), do: raise("unsupported empty Nx tensor")

  defp to_nx_tensor([], raw, type, shape) do
    raw |> Nx.from_binary(type) |> Nx.reshape(shape)
  end

  defp to_nx_tensor(data, _, type, shape) do
    data |> Nx.tensor(type: type) |> Nx.reshape(shape)
  end

  @doc """
  Parses the op_type and ONNX opset version out of a case's `model.onnx`.

  Returns `%{op_types: [String.t()], opset: integer() | nil}`. Used for
  reporting; never raises (returns `:error` on parse failure so the COVERAGE
  report can still render).
  """
  @spec inspect_model(case_entry()) :: %{op_types: [String.t()], opset: integer() | nil}
  def inspect_model(%{path: path}) do
    try do
      model =
        path
        |> Path.join("model.onnx")
        |> File.read!()
        |> Onnx.ModelProto.decode!()

      op_types =
        case model.graph do
          %Onnx.GraphProto{node: nodes} ->
            nodes |> Enum.map(& &1.op_type) |> Enum.uniq()

          _ ->
            []
        end

      opset =
        model.opset_import
        |> Enum.find(fn import_id ->
          domain = import_id.domain
          domain == "" or domain == nil or domain == "ai.onnx"
        end)
        |> case do
          nil -> nil
          %{version: v} -> v
        end

      %{op_types: op_types, opset: opset}
    rescue
      _ -> %{op_types: [], opset: nil}
    end
  end
end
