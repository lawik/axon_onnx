defmodule AxonOnnx.Deserialize do
  @moduledoc false

  alias Onnx.ModelProto, as: Model
  alias Onnx.GraphProto, as: Graph
  alias Onnx.ValueInfoProto, as: Value
  alias Onnx.AttributeProto, as: Attribute
  alias Onnx.NodeProto, as: Node
  alias Onnx.TypeProto, as: Type
  alias Onnx.TensorProto, as: Tensor
  alias Onnx.TypeProto.Tensor, as: Placeholder
  alias Onnx.TensorShapeProto, as: Shape
  alias Onnx.TensorShapeProto.Dimension, as: Dimension

  require Logger

  import AxonOnnx.Shared

  @opsets_key {__MODULE__, :opsets}
  @input_types_key {__MODULE__, :input_types}
  @output_shapes_key {__MODULE__, :output_shapes}

  def __load__(binary, opts \\ []) do
    binary
    |> Model.decode!()
    |> to_axon(opts)
  end

  defp to_axon(%Model{graph: %Graph{} = graph, opset_import: opset_imports}, opts) do
    {fold_inputs, dimensions} = pop_fold_inputs(opts)
    opsets = build_opsets(opset_imports)
    input_types = build_input_types(graph)
    output_shapes = build_output_shapes(graph)
    previous_opsets = Process.put(@opsets_key, opsets)
    previous_input_types = Process.put(@input_types_key, input_types)
    previous_output_shapes = Process.put(@output_shapes_key, output_shapes)

    try do
      {graph, params} = graph_to_axon(graph, dimensions, fold_inputs)

      case graph do
        [graph] ->
          # single-output
          {graph, params}

        graph when is_list(graph) ->
          # multi-output
          {Axon.container(List.to_tuple(graph)), params}
      end
    after
      restore_dict(@opsets_key, previous_opsets)
      restore_dict(@input_types_key, previous_input_types)
      restore_dict(@output_shapes_key, previous_output_shapes)
    end
  end

  defp restore_dict(key, nil), do: Process.delete(key)
  defp restore_dict(key, prev), do: Process.put(key, prev)

  # `fold_inputs:` is an opt-in map of `name => Nx.Tensor` that promotes
  # those graph inputs to "phantom initializers" — they're consumed at
  # build time as if they were declared as initializers. Useful for
  # ops like Squeeze/Unsqueeze/Slice whose axes/starts/ends arrive as a
  # graph input in the test corpus but must be constant for static
  # Axon shape inference. Coverage runner uses this to fold static-only
  # parameters declared as inputs in the corpus protos.
  defp pop_fold_inputs(opts) when is_list(opts) do
    case Keyword.pop(opts, :fold_inputs) do
      {nil, rest} -> {%{}, rest}
      {map, rest} when is_map(map) -> {map, rest}
    end
  end

  defp pop_fold_inputs(opts) when is_map(opts), do: pop_fold_inputs(Map.to_list(opts))
  defp pop_fold_inputs(_), do: {%{}, []}

  @doc """
  Returns the ONNX opset version in scope for the current deserialization
  call, for the given domain (default: the core `ai.onnx` domain, encoded as
  the empty string in `opset_import`).

  Returns `nil` if called outside `AxonOnnx.import/2`/`load/2`, or if the
  model does not declare an opset for the requested domain.

  This is the foundation for opset-version-aware dispatch in `recur_nodes/2`.
  An ONNX op's semantics often change across opset versions — `Clip`'s
  min/max moved from attributes to inputs at opset 11; `Squeeze`/`Unsqueeze`
  did the same for `axes` at opset 13; `Resize` has been rewritten multiple
  times — so dispatch clauses that want to honour the spec exactly should
  consult `opset_version/1` rather than rely on argument arity or attribute
  presence alone.

  Subgraphs (e.g. an `If` branch) inherit the parent model's opsets
  automatically: `to_axon/2` installs them in the process dictionary for the
  duration of the call, and the recursive `graph_to_axon/2` for subgraphs
  runs in the same process.
  """
  @spec opset_version(String.t()) :: integer() | nil
  def opset_version(domain \\ "") when is_binary(domain) do
    case Process.get(@opsets_key) do
      nil -> nil
      map -> Map.get(map, domain)
    end
  end

  defp build_opsets(nil), do: %{}

  defp build_opsets(opset_imports) when is_list(opset_imports) do
    Enum.reduce(opset_imports, %{}, fn import_id, acc ->
      Map.put(acc, import_id.domain || "", import_id.version)
    end)
  end

  defp build_input_types(%Graph{input: inputs}) do
    Enum.reduce(inputs, %{}, fn %Value{name: name, type: %Type{value: value}}, acc ->
      case value do
        {:tensor_type, %Placeholder{elem_type: elem_type}} when not is_nil(elem_type) ->
          Map.put(acc, name, onnx_type_to_nx_type(elem_type))

        _ ->
          acc
      end
    end)
  end

  defp build_output_shapes(%Graph{output: outputs}) do
    Enum.reduce(outputs, %{}, fn %Value{name: name, type: %Type{value: value}}, acc ->
      case value do
        {:tensor_type, %Placeholder{shape: %Shape{dim: dims}}} ->
          shape_list =
            Enum.map(dims, fn %Dimension{value: v} ->
              case v do
                {:dim_value, n} -> n
                _ -> nil
              end
            end)

          if Enum.all?(shape_list, &is_integer/1) do
            Map.put(acc, name, List.to_tuple(shape_list))
          else
            acc
          end

        _ ->
          acc
      end
    end)
  end

  @doc """
  Statically-declared shape for a graph output, or `nil`. Available for the
  lifetime of an `AxonOnnx.import/2` / `load/2` call. Useful when a model
  declares its output shape but builds it from runtime inputs (e.g.
  `ConstantOfShape` whose `shape` input is a graph input but whose result
  shape is fixed in `graph.output`).
  """
  @spec output_shape(String.t()) :: tuple() | nil
  def output_shape(name) when is_binary(name) do
    case Process.get(@output_shapes_key) do
      nil -> nil
      map -> Map.get(map, name)
    end
  end

  @doc """
  Looks up the ONNX-declared Nx dtype for a named graph input, or `nil` if
  unknown. Available throughout the lifetime of `AxonOnnx.import/2`/`load/2`.

  Axon 0.5's `Axon.input/2` only carries shape, not dtype, so callers that
  need the declared dtype (e.g. to decide a quantisation output type from
  `y_zero_point`'s declared int8/u16 elem_type) consult this side-channel
  rather than relying on the runtime tensor — which they can't see at
  graph-build time.
  """
  @spec input_type(String.t()) :: Nx.Type.t() | nil
  def input_type(name) when is_binary(name) do
    case Process.get(@input_types_key) do
      nil -> nil
      map -> Map.get(map, name)
    end
  end

  def graph_to_axon(graph, dimensions, fold_inputs \\ %{})

  def graph_to_axon(%Graph{node: nodes} = graph, dimensions, fold_inputs) do
    params = get_params(graph) |> Map.merge(fold_inputs)
    inputs = get_inputs(graph, params, dimensions)
    outputs = get_outputs(graph)
    {nodes, _, params} = get_nodes(nodes, inputs, params, %{})
    {Enum.map(outputs, fn name -> nodes[name] end), params}
  end

  defp get_inputs(%Graph{input: inputs}, params, dimensions) do
    Enum.reduce(inputs, %{}, fn %Value{name: name, type: %Type{value: value}}, acc ->
      if Map.has_key?(params, name) do
        acc
      else
        case value do
          {:tensor_type, %Placeholder{} = tensor} ->
            input_shape = shape!(tensor, dimensions)
            Map.put(acc, name, Axon.input(name, shape: input_shape))

          unsupported ->
            raise ArgumentError, "unsupported input type #{inspect(unsupported)}"
        end
      end
    end)
  end

  defp get_params(%Graph{initializer: initializer}) do
    Enum.reduce(initializer, %{}, fn %Tensor{name: name} = tensor, params ->
      Map.put(params, name, tensor!(tensor))
    end)
  end

  defp get_outputs(%Graph{output: outputs}) do
    Enum.map(outputs, fn %Value{name: name} -> name end)
  end

  defp get_nodes(pruned_nodes, inp, params, used_params) do
    Enum.reduce(pruned_nodes, {inp, params, used_params}, &recur_nodes/2)
  end

  @nx_op_types [
    {"Abs", &Nx.abs/1},
    {"Acos", &Nx.acos/1},
    {"Acosh", &Nx.acosh/1},
    {"Asin", &Nx.asin/1},
    {"Asinh", &Nx.asinh/1},
    {"Atan", &Nx.atan/1},
    {"Atanh", &Nx.atanh/1},
    {"Ceil", &Nx.ceil/1},
    {"Cos", &Nx.cos/1},
    {"Cosh", &Nx.cosh/1},
    {"Erf", &Nx.erf/1},
    {"Floor", &Nx.floor/1},
    {"HardSwish", &hardswish/1},
    {"Identity", &identity/1},
    {"IsNaN", &Nx.is_nan/1},
    {"BitwiseNot", &Nx.bitwise_not/1},
    {"Log", &Nx.log/1},
    {"Neg", &Nx.negate/1},
    {"Not", &Nx.logical_not/1},
    {"Round", &__MODULE__.round_half_to_even/1},
    {"Reciprocal", &reciprocal/1},
    {"Sign", &Nx.sign/1},
    {"Sin", &Nx.sin/1},
    {"Sinh", &Nx.sinh/1},
    {"Sqrt", &Nx.sqrt/1},
    {"Tan", &Nx.tan/1}
  ]

  for {op, fun} <- @nx_op_types do
    defp recur_nodes(
           %Node{op_type: unquote(op), input: [input_name], output: [output_name]},
           {axon, params, used_params}
         ) do
      input = input!(input_name, axon, params, used_params)
      {:name, op_name} = Function.info(unquote(fun), :name)

      output =
        case get_axon_node(input) do
          %Axon.Node{op: :constant, opts: [value: value]} ->
            new_value = apply(unquote(fun), [value])
            Axon.constant(new_value, name: output_name)

          %Axon.Node{} ->
            Axon.nx(input, unquote(fun), name: output_name, op_name: op_name)

          %Nx.Tensor{} = tensor_input ->
            value = apply(unquote(fun), [tensor_input])
            Axon.constant(value, name: output_name)
        end

      updated_axon = Map.put(axon, output_name, output)
      {updated_axon, params, used_params}
    end
  end

  # IsInf honours `detect_positive` / `detect_negative` (both default 1)
  # to selectively flag +Inf, -Inf, or both. `Nx.is_infinity` flags both;
  # combine with sign tests for the selective forms.
  defp recur_nodes(
         %Node{
           op_type: "IsInf",
           attribute: attrs,
           input: [input_name],
           output: [output_name]
         },
         {axon, params, used_params}
       ) do
    input = input!(input_name, axon, params, used_params)
    opts = options!(attrs)
    detect_positive = (opts["detect_positive"] || 1) == 1
    detect_negative = (opts["detect_negative"] || 1) == 1

    fun =
      cond do
        detect_positive and detect_negative ->
          &Nx.is_infinity/1

        detect_positive ->
          fn t -> Nx.logical_and(Nx.is_infinity(t), Nx.greater(t, 0)) end

        detect_negative ->
          fn t -> Nx.logical_and(Nx.is_infinity(t), Nx.less(t, 0)) end

        true ->
          fn t -> Nx.broadcast(Nx.tensor(0, type: {:u, 8}), Nx.shape(t)) end
      end

    output =
      case get_axon_node(input) do
        %Axon.Node{op: :constant, opts: [value: value]} ->
          Axon.constant(fun.(value), name: output_name)

        %Axon.Node{} ->
          Axon.nx(input, fun, name: output_name, op_name: :is_infinity)

        %Nx.Tensor{} = t ->
          Axon.constant(fun.(t), name: output_name)
      end

    {Map.put(axon, output_name, output), params, used_params}
  end

  @activation_op_types [
    {"Celu", :celu, [alpha: {"alpha", 1.0}]},
    {"Elu", :elu, [alpha: {"alpha", 1.0}]},
    {"Exp", :exp, []},
    {"HardSigmoid", :hard_sigmoid, [alpha: {"alpha", 0.2}, beta: {"beta", 0.5}]},
    {"LeakyRelu", :leaky_relu, [alpha: {"alpha", 1.0e-2}]},
    {"LogSoftmax", :log_softmax, [axis: {"axis", -1}]},
    {"Mish", :mish, []},
    {"Relu", :relu, []},
    {"Selu", :selu,
     [alpha: {"alpha", 1.67326319217681884765625}, gamma: {"gamma", 1.05070102214813232421875}]},
    {"Sigmoid", :sigmoid, []},
    {"Softmax", :softmax, [axis: {"axis", -1}]},
    {"Softplus", :softplus, []},
    {"Softsign", :softsign, []},
    {"Tanh", :tanh, []}
  ]

  for {op, act, act_opts} <- @activation_op_types do
    defp recur_nodes(
           %Node{
             op_type: unquote(op),
             attribute: attrs,
             input: [input_name],
             output: [output_name]
           },
           {axon, params, used_params}
         ) do
      input = input!(input_name, axon, params, used_params)
      activation_options = options!(attrs)

      opts =
        Enum.map(unquote(act_opts), fn {k, {name, default}} ->
          if activation_options[name] do
            {k, activation_options[name]}
          else
            {k, default}
          end
        end)

      axon_output =
        case get_axon_node(input) do
          %Axon.Node{op: :constant, opts: [value: value]} ->
            new_value = apply(Axon.Activations, unquote(act), [value] ++ opts)
            Axon.constant(new_value, name: output_name)

          %Axon.Node{} ->
            opts = [name: output_name] ++ opts
            apply(Axon, unquote(act), [input, opts])

          %Nx.Tensor{} = tensor_input ->
            new_value = apply(Axon.Activations, unquote(act), [tensor_input] ++ opts)
            Axon.constant(new_value, name: output_name)
        end

      updated_axon = Map.put(axon, output_name, axon_output)
      {updated_axon, params, used_params}
    end
  end

  @reduction_op_types [
    {"ArgMax", &Nx.argmax/2, :axis, :argmax},
    {"ArgMin", &Nx.argmin/2, :axis, :argmin},
    {"ReduceL1", &l1_norm/2, :axes, :reduce_l1},
    {"ReduceL2", &l2_norm/2, :axes, :reduce_l2},
    {"ReduceLogSum", &logsum/2, :axes, :reduce_log_sum},
    {"ReduceLogSumExp", &logsumexp/2, :axes, :reduce_log_sum_exp},
    {"ReduceMax", &Nx.reduce_max/2, :axes, :reduce_max},
    {"ReduceMean", &Nx.mean/2, :axes, :reduce_mean},
    {"ReduceMin", &Nx.reduce_min/2, :axes, :reduce_min},
    {"ReduceProd", &Nx.product/2, :axes, :reduce_prod},
    {"ReduceSum", &Nx.sum/2, :axes, :reduce_sum},
    {"ReduceSumSquare", &sumsquare/2, :axes, :reduce_sum_square}
  ]

  for {op, reduce_fun, axis_or_axes, op_name} <- @reduction_op_types do
    defp recur_nodes(
           %Node{
             op_type: unquote(op),
             attribute: attrs,
             input: [input_name],
             output: [output_name]
           },
           {axon, params, used_params}
         ) do
      reduce_options = options!(attrs)
      input = input!(input_name, axon, params, used_params)

      keepdims = reduce_options["keepdims"] || 1
      keep_axes = if keepdims == 1, do: true, else: false

      axes =
        if unquote(axis_or_axes) == :axis do
          reduce_options["axis"] || 0
        else
          reduce_options["axes"]
        end

      opts =
        if unquote(axis_or_axes) == :axis do
          last_index = reduce_options["select_last_index"] || 0
          tie_break = if last_index == 0, do: :low, else: :high
          [keep_axis: keep_axes, axis: axes, tie_break: tie_break]
        else
          if axes, do: [keep_axes: keep_axes, axes: axes], else: [keep_axes: keep_axes]
        end

      layer_fun = fn x, opts ->
        opts = Keyword.delete(opts, :mode)
        apply(unquote(reduce_fun), [x, opts])
      end

      layer =
        case get_axon_node(input) do
          %Axon.Node{op: :constant, opts: [value: tensor]} ->
            new_value = layer_fun.(tensor, opts)
            Axon.constant(new_value, name: output_name)

          %Axon.Node{} ->
            Axon.layer(
              layer_fun,
              [input],
              [name: output_name, op_name: unquote(op_name)] ++ opts
            )

          %Nx.Tensor{} = tensor_input ->
            new_value = layer_fun.(tensor_input, opts)
            Axon.constant(new_value, name: output_name)
        end

      updated_axon = Map.put(axon, output_name, layer)

      {updated_axon, params, used_params}
    end
  end

  # 2-input form of axes-driven reductions (opset 13+ for ReduceSum,
  # opset 18+ for the others). The single-input form above handles axes via
  # attribute; this one resolves the axes tensor into a static list at
  # import time when possible, then dispatches into the same layer fn.
  #
  # Resolution paths:
  #   - axes is a Constant op or an initializer → extract values.
  #   - axes is a graph input declared with shape {0} → empty axes; with
  #     noop_with_empty_axes=0 (default) reduce all, with =1 return input
  #     unchanged.
  #   - axes is a runtime graph input with non-empty shape → bail; the
  #     value is needed at trace time and we don't yet have a runtime-axes
  #     reduction path. Surfaces as :unsupported in the registry.
  for {op, reduce_fun, axis_or_axes, op_name} <- @reduction_op_types,
      axis_or_axes == :axes do
    defp recur_nodes(
           %Node{
             op_type: unquote(op),
             attribute: attrs,
             input: [data_name, axes_name],
             output: [output_name]
           },
           {axon, params, used_params}
         ) do
      reduce_options = options!(attrs)
      input = input!(data_name, axon, params, used_params)
      keepdims = reduce_options["keepdims"] || 1
      keep_axes = if keepdims == 1, do: true, else: false
      noop_with_empty = (reduce_options["noop_with_empty_axes"] || 0) == 1

      axes = resolve_reduce_axes!(axes_name, axon, params, used_params)

      layer_fun = fn x, opts ->
        opts = Keyword.delete(opts, :mode)
        apply(unquote(reduce_fun), [x, opts])
      end

      layer =
        cond do
          axes == :empty and noop_with_empty ->
            # Pass input through unchanged; still wrap in a layer so the
            # registered output exists in the axon map.
            Axon.nx(input, & &1, name: output_name, op_name: unquote(op_name))

          axes == :empty ->
            # Reduce all axes — Nx treats omitted axes that way.
            build_reduce_layer(input, layer_fun, [keep_axes: keep_axes], output_name,
              unquote(op_name)
            )

          is_list(axes) ->
            build_reduce_layer(input, layer_fun, [keep_axes: keep_axes, axes: axes],
              output_name, unquote(op_name))
        end

      updated_axon = Map.put(axon, output_name, layer)
      {updated_axon, params, used_params}
    end
  end

  @builtin_binary_op_types [
    {"Add", :add},
    {"Sub", :subtract},
    {"Mul", :multiply}
  ]

  for {op, binary_op} <- @builtin_binary_op_types do
    defp recur_nodes(
           %Node{op_type: unquote(op), input: [inp1, inp2], output: [output_name]},
           {axon, params, used_params}
         ) do
      # There's a potential this is just a bias add, so we check if
      # inp1 is a graph node and inp2 is a parameter and handle
      # accordingly
      if unquote(op) == "Add" and Map.has_key?(axon, inp1) and Map.has_key?(params, inp2) do
        inp1 = input!(inp1, axon, params, used_params)
        inp2 = input!(inp2, axon, params, used_params)

        updated_axon = Map.put(axon, output_name, Axon.bias(inp1, name: output_name))
        updated_params = Map.put(used_params, output_name, %{"bias" => inp2})

        {updated_axon, params, updated_params}
      else
        inp1 = input!(inp1, axon, params, used_params)
        inp2 = input!(inp2, axon, params, used_params)

        {updated_axon, updated_params} =
          case {get_axon_node(inp1), get_axon_node(inp2)} do
            {%Axon.Node{op: :constant, opts: [value: v1]},
             %Axon.Node{op: :constant, opts: [value: v2]}} ->
              new_value = apply(Nx, unquote(binary_op), [v1, v2])
              layer = Axon.constant(new_value, name: output_name)
              updated_axon = Map.put(axon, output_name, layer)
              {updated_axon, used_params}

            {%Axon.Node{op: :constant, opts: [value: v1]}, %Nx.Tensor{} = v2} ->
              new_value = apply(Nx, unquote(binary_op), [v1, v2])
              layer = Axon.constant(new_value, name: output_name)
              updated_axon = Map.put(axon, output_name, layer)
              {updated_axon, used_params}

            {%Nx.Tensor{} = v1, %Axon.Node{op: :constant, opts: [value: v2]}} ->
              new_value = apply(Nx, unquote(binary_op), [v1, v2])
              layer = Axon.constant(new_value, name: output_name)
              updated_axon = Map.put(axon, output_name, layer)
              {updated_axon, used_params}

            {%Axon.Node{}, %Axon.Node{}} ->
              layer = apply(Axon, unquote(binary_op), [inp1, inp2])
              updated_axon = Map.put(axon, output_name, layer)
              {updated_axon, used_params}

            {%Axon.Node{}, %Nx.Tensor{}} ->
              layer =
                trainable_binary_layer(
                  inp1,
                  inp2,
                  unquote(binary_op),
                  output_name,
                  unquote(binary_op)
                )

              updated_axon = Map.put(axon, output_name, layer)
              updated_params = Map.put(used_params, output_name, %{"kernel" => inp2})
              {updated_axon, updated_params}

            {%Nx.Tensor{}, %Axon.Node{}} ->
              layer =
                trainable_binary_layer(
                  inp2,
                  inp1,
                  unquote(binary_op),
                  output_name,
                  unquote(binary_op)
                )

              updated_axon = Map.put(axon, output_name, layer)
              updated_params = Map.put(used_params, output_name, %{"kernel" => inp1})
              {updated_axon, updated_params}
          end

        {updated_axon, params, updated_params}
      end
    end
  end

  @binary_op_types [
    {"And", &Nx.logical_and/2, :logical_and},
    {"BitwiseAnd", &Nx.bitwise_and/2, :bitwise_and},
    {"BitwiseOr", &Nx.bitwise_or/2, :bitwise_or},
    {"BitwiseXor", &Nx.bitwise_xor/2, :bitwise_xor},
    {"Div", &__MODULE__.onnx_div/2, :divide},
    {"Equal", &Nx.equal/2, :equal},
    {"Greater", &Nx.greater/2, :greater},
    {"GreaterOrEqual", &Nx.greater_equal/2, :greater_equal},
    {"Less", &Nx.less/2, :less},
    {"LessOrEqual", &Nx.less_equal/2, :less_or_equal},
    {"Or", &Nx.logical_or/2, :logical_or},
    {"Pow", &Nx.pow/2, :power},
    {"Xor", &Nx.logical_xor/2, :logical_xor}
  ]

  # ONNX random-op `seed` is a FLOAT attribute (optional). Nx.Random.key
  # requires an integer or s64/u64 tensor. Coerce missing / float seeds
  # to a stable integer so the deserialiser doesn't crash on the common
  # `seed=0.0` form. Floats are truncated; nil falls back to 0.
  defp coerce_random_seed(nil), do: 0
  defp coerce_random_seed(seed) when is_integer(seed), do: seed
  defp coerce_random_seed(seed) when is_float(seed), do: trunc(seed)

  # Pool ops accept either a single int (broadcast across spatial axes)
  # or a list with one entry per axis. Normalise to a list.
  defp expand_to_spatial(value, spatial_rank) when is_integer(value),
    do: List.duplicate(value, spatial_rank)

  defp expand_to_spatial(value, _spatial_rank) when is_list(value), do: value

  # ----- RotaryEmbedding --------------------------------------------------

  # Applies rotary embedding to `input` along the head dimension. The
  # 4-D case treats `input` as {batch, num_heads, seq_len, head_size};
  # the 3-D case as {batch, seq_len, num_heads * head_size}.
  defp do_rotary_embedding(x, cos_cache, sin_cache, position_ids, opts) do
    interleaved = opts[:interleaved]
    rotary_dim = opts[:rotary_dim]
    num_heads_opt = opts[:num_heads]
    rank = Nx.rank(x)

    # Reshape 3-D form to 4-D for uniform processing, then reshape back
    # at the end.
    {x_4d, original_3d} =
      cond do
        rank == 4 ->
          {x, false}

        rank == 3 ->
          {batch, seq, hidden} = Nx.shape(x)
          nh = if num_heads_opt > 0, do: num_heads_opt, else: raise(ArgumentError, "RotaryEmbedding 3-D form needs num_heads attribute")
          hs = div(hidden, nh)
          # ONNX 3-D layout is (batch, seq, num_heads * head_size); the
          # 4-D internal layout is (batch, num_heads, seq, head_size).
          {Nx.reshape(x, {batch, seq, nh, hs}) |> Nx.transpose(axes: [0, 2, 1, 3]), true}
      end

    {_batch, _nh, _seq, head_size} = Nx.shape(x_4d)
    effective_rotary = if rotary_dim == 0, do: head_size, else: rotary_dim

    # Look up cos/sin for each position; result shape {batch, seq, rotary/2}.
    cos =
      case position_ids do
        nil ->
          # No position_ids: use cos_cache as-is. Assume it's already in
          # {batch, seq, rotary/2} or compatible-broadcast form.
          cos_cache

        _ ->
          pos = Nx.as_type(position_ids, {:s, 64})
          Nx.take(cos_cache, pos, axis: 0)
      end

    sin =
      case position_ids do
        nil -> sin_cache
        _ -> Nx.take(sin_cache, Nx.as_type(position_ids, {:s, 64}), axis: 0)
      end

    # Broadcast cos/sin to {batch, 1, seq, rotary/2} so they apply to
    # all heads.
    cos_b = Nx.new_axis(cos, 1)
    sin_b = Nx.new_axis(sin, 1)

    # Split the head into the rotated part (first `effective_rotary`
    # entries) and the passthrough part (rest).
    rotated = Nx.slice_along_axis(x_4d, 0, effective_rotary, axis: 3)

    passthrough =
      if effective_rotary < head_size do
        Nx.slice_along_axis(x_4d, effective_rotary, head_size - effective_rotary, axis: 3)
      end

    # Split the rotated half into (x1, x2). For interleaved=0 (default),
    # x1 is the first half and x2 is the second half. For interleaved=1
    # the (even, odd) entries are paired.
    half = div(effective_rotary, 2)

    {x1, x2} =
      if interleaved do
        # Pull even-indexed and odd-indexed entries.
        indices_even = Nx.tensor(Enum.map(0..(half - 1), &(&1 * 2)), type: {:s, 64})
        indices_odd = Nx.tensor(Enum.map(0..(half - 1), &(&1 * 2 + 1)), type: {:s, 64})
        {Nx.take(rotated, indices_even, axis: 3), Nx.take(rotated, indices_odd, axis: 3)}
      else
        {Nx.slice_along_axis(rotated, 0, half, axis: 3),
         Nx.slice_along_axis(rotated, half, half, axis: 3)}
      end

    new_x1 = Nx.subtract(Nx.multiply(x1, cos_b), Nx.multiply(x2, sin_b))
    new_x2 = Nx.add(Nx.multiply(x1, sin_b), Nx.multiply(x2, cos_b))

    rotated_out =
      if interleaved do
        # Interleave new_x1 and new_x2 back into pairs along the head axis.
        stacked = Nx.stack([new_x1, new_x2], axis: 4)
        new_shape =
          stacked
          |> Nx.shape()
          |> Tuple.to_list()
          |> List.delete_at(-1)
          |> List.update_at(-1, &(&1 * 2))
          |> List.to_tuple()

        Nx.reshape(stacked, new_shape)
      else
        Nx.concatenate([new_x1, new_x2], axis: 3)
      end

    final =
      if passthrough do
        Nx.concatenate([rotated_out, passthrough], axis: 3)
      else
        rotated_out
      end

    if original_3d do
      {batch, nh, seq, hs} = Nx.shape(final)
      final |> Nx.transpose(axes: [0, 2, 1, 3]) |> Nx.reshape({batch, seq, nh * hs})
    else
      final
    end
  end

  # ----- Einsum -----------------------------------------------------------

  # Single-input einsum: handles reductions, transposes, and diagonals.
  # `spec_in` is the per-axis subscript string (possibly with "..."),
  # `spec_out` is the output subscript string (may be nil for implicit
  # output).
  defp do_einsum_1(x, spec_in, spec_out) do
    rank = Nx.rank(x)

    # Expand "..." in the input spec to enough single-letter placeholders
    # for the actual rank.
    {in_letters, batch_letters} = einsum_expand_spec(spec_in, rank)

    out_spec =
      case spec_out do
        nil -> einsum_default_output(in_letters, batch_letters)
        _ -> spec_out
      end

    {out_letters, _} = einsum_expand_spec(out_spec, length(batch_letters) + count_non_dots(out_spec))

    # Diagonals: any letter repeated in the input is a diagonal axis. We
    # gather along it before further reductions.
    dup_letters = in_letters |> Enum.frequencies() |> Enum.filter(fn {_, c} -> c > 1 end) |> Enum.map(&elem(&1, 0))

    {x_after_diag, in_after_diag} =
      Enum.reduce(dup_letters, {x, in_letters}, fn letter, {acc_x, acc_letters} ->
        axes = acc_letters |> Enum.with_index() |> Enum.filter(fn {l, _} -> l == letter end) |> Enum.map(&elem(&1, 1))

        [first | rest] = axes
        # Diagonal: take elements where all axes match. We use
        # Nx.take_along_axis with an iota matched to the first axis.
        dim = Nx.axis_size(acc_x, first)
        rest_axes = Enum.sort(rest, :desc)

        {reduced, new_letters} =
          Enum.reduce(rest_axes, {acc_x, acc_letters}, fn ax, {ax_x, ax_letters} ->
            # Index along this axis with the same iota as the first axis.
            # Compute a diagonal slice: for index i in axis `first`, we want
            # the element at position i in axis `ax` too.
            iota =
              Nx.iota({dim}, type: {:s, 64})

            shape_template = Tuple.duplicate(1, Nx.rank(ax_x)) |> put_elem(first, dim)
            iota_b = Nx.reshape(iota, shape_template) |> Nx.broadcast(Nx.shape(ax_x))
            gathered = Nx.take_along_axis(ax_x, iota_b, axis: ax)
            slice = Nx.slice_along_axis(gathered, 0, 1, axis: ax) |> Nx.squeeze(axes: [ax])
            {slice, List.delete_at(ax_letters, ax)}
          end)

        {reduced, new_letters}
      end)

    # Sum over axes whose letters don't appear in the output spec.
    sum_axes =
      in_after_diag
      |> Enum.with_index()
      |> Enum.filter(fn {l, _} -> not Enum.member?(out_letters, l) end)
      |> Enum.map(&elem(&1, 1))
      |> Enum.sort(:desc)

    {x_summed, in_after_sum} =
      Enum.reduce(sum_axes, {x_after_diag, in_after_diag}, fn ax, {ax_x, ax_letters} ->
        {Nx.sum(ax_x, axes: [ax]), List.delete_at(ax_letters, ax)}
      end)

    # Transpose to match the output letter order.
    perm = Enum.map(out_letters, fn letter -> Enum.find_index(in_after_sum, &(&1 == letter)) end)

    if perm == Enum.to_list(0..(length(in_after_sum) - 1)//1) do
      x_summed
    else
      Nx.transpose(x_summed, axes: perm)
    end
  end

  defp do_einsum_2(a, b, spec_a, spec_b, spec_out) do
    rank_a = Nx.rank(a)
    rank_b = Nx.rank(b)
    {a_letters, _} = einsum_expand_spec(spec_a, rank_a)
    {b_letters, _} = einsum_expand_spec(spec_b, rank_b)

    out_spec =
      case spec_out do
        nil ->
          # Implicit output: letters appearing exactly once, in
          # alphabetical order.
          (a_letters ++ b_letters)
          |> Enum.frequencies()
          |> Enum.filter(fn {_, c} -> c == 1 end)
          |> Enum.map(&elem(&1, 0))
          |> Enum.sort()
          |> Enum.join("")

        _ -> spec_out
      end

    {out_letters, _} = einsum_expand_spec(out_spec, count_non_dots(out_spec))

    # Letters classified across the two inputs:
    # * batch: in A, B, and output
    # * contract: in A and B but not in output
    # * a_keep: in A and output (not in B)
    # * b_keep: in B and output (not in A)
    a_set = MapSet.new(a_letters)
    b_set = MapSet.new(b_letters)
    out_set = MapSet.new(out_letters)

    batch_letters =
      a_letters |> Enum.filter(&(MapSet.member?(b_set, &1) and MapSet.member?(out_set, &1)))

    contract_letters =
      a_letters |> Enum.filter(&(MapSet.member?(b_set, &1) and not MapSet.member?(out_set, &1)))

    batch_axes_a = Enum.map(batch_letters, fn l -> Enum.find_index(a_letters, &(&1 == l)) end)
    batch_axes_b = Enum.map(batch_letters, fn l -> Enum.find_index(b_letters, &(&1 == l)) end)
    contract_axes_a = Enum.map(contract_letters, fn l -> Enum.find_index(a_letters, &(&1 == l)) end)
    contract_axes_b = Enum.map(contract_letters, fn l -> Enum.find_index(b_letters, &(&1 == l)) end)

    _ = a_set

    # Use Nx.dot with batch axes + contracting axes. The result's axis
    # order is: batch axes (in their order), then a's remaining axes,
    # then b's remaining axes.
    dotted = Nx.dot(a, contract_axes_a, batch_axes_a, b, contract_axes_b, batch_axes_b)

    # Compute the letter order produced by Nx.dot.
    a_keep_letters =
      a_letters
      |> Enum.with_index()
      |> Enum.filter(fn {l, _} -> not Enum.member?(batch_letters, l) and not Enum.member?(contract_letters, l) end)
      |> Enum.map(&elem(&1, 0))

    b_keep_letters =
      b_letters
      |> Enum.with_index()
      |> Enum.filter(fn {l, _} -> not Enum.member?(batch_letters, l) and not Enum.member?(contract_letters, l) end)
      |> Enum.map(&elem(&1, 0))

    dotted_letters = batch_letters ++ a_keep_letters ++ b_keep_letters

    # Transpose to match output letter order.
    perm = Enum.map(out_letters, fn letter -> Enum.find_index(dotted_letters, &(&1 == letter)) end)

    if perm == Enum.to_list(0..(length(dotted_letters) - 1)//1) do
      dotted
    else
      Nx.transpose(dotted, axes: perm)
    end
  end

  # Expands "..." in a subscript to enough placeholder characters
  # ("\\u0001" + i) to fill the rank. Returns the per-axis letter list and
  # the list of synthetic placeholders representing the "..." span.
  defp einsum_expand_spec(spec, rank) do
    case String.split(spec, "...") do
      [single] ->
        {String.graphemes(single), []}

      [prefix, suffix] ->
        named = String.length(prefix) + String.length(suffix)
        dot_count = max(rank - named, 0)
        placeholders = for i <- 0..(dot_count - 1), do: "<#{i}>"
        {String.graphemes(prefix) ++ placeholders ++ String.graphemes(suffix), placeholders}
    end
  end

  defp count_non_dots(spec) when is_binary(spec) do
    spec |> String.replace("...", "") |> String.length()
  end

  defp count_non_dots(nil), do: 0

  defp einsum_default_output(in_letters, batch_placeholders) do
    # Implicit output (no ->): letters appearing exactly once, in
    # alphabetical order, with "..." prepended if there are batch
    # placeholders.
    rest =
      in_letters
      |> Enum.frequencies()
      |> Enum.filter(fn {l, c} -> c == 1 and l not in batch_placeholders end)
      |> Enum.map(&elem(&1, 0))
      |> Enum.sort()
      |> Enum.join("")

    if batch_placeholders == [], do: rest, else: "..." <> rest
  end

  # Resize helpers — coordinate transformation (output i → input float)
  # and nearest-mode rounding. The corpus exercises five
  # coordinate_transformation_modes; "tf_crop_and_resize" is rejected
  # earlier with a clean error.
  defp resize_coord_transform(out_i, in_dim, out_dim, scale, ctm) do
    case ctm do
      "half_pixel" ->
        (out_i + 0.5) / scale - 0.5

      "half_pixel_symmetric" ->
        adj = in_dim / 2 - out_dim / 2 / scale
        (out_i + 0.5) / scale - 0.5 + adj

      "pytorch_half_pixel" ->
        if out_dim > 1, do: (out_i + 0.5) / scale - 0.5, else: 0.0

      "asymmetric" ->
        out_i / scale

      "align_corners" ->
        if out_dim == 1, do: 0.0, else: out_i * (in_dim - 1) / (out_dim - 1)

      other ->
        raise ArgumentError, "Resize coord transform #{inspect(other)} not supported"
    end
  end

  defp resize_nearest_round(x, "round_prefer_floor") do
    # Round half toward negative infinity.
    floor_x = :math.floor(x) |> trunc()
    frac = x - floor_x
    cond do
      frac > 0.5 -> floor_x + 1
      frac < 0.5 -> floor_x
      true -> floor_x
    end
  end

  defp resize_nearest_round(x, "round_prefer_ceil") do
    floor_x = :math.floor(x) |> trunc()
    frac = x - floor_x
    cond do
      frac > 0.5 -> floor_x + 1
      frac < 0.5 -> floor_x
      true -> floor_x + 1
    end
  end

  defp resize_nearest_round(x, "floor"), do: :math.floor(x) |> trunc()
  defp resize_nearest_round(x, "ceil"), do: :math.ceil(x) |> trunc()

  defp resize_apply_sizes(_zip, _input_shape, axes, "stretch") do
    # `_zip` is the {axis, target_size} list — but the safe shape is just
    # the per-axis target list, with non-resized axes preserved.
    Enum.map(0..(length(_input_shape) - 1)//1, fn ax ->
      case Enum.find_index(axes, &(&1 == ax)) do
        nil -> Enum.at(_input_shape, ax)
        idx -> _zip |> Enum.at(idx) |> elem(1)
      end
    end)
  end

  defp resize_apply_sizes(zip, input_shape, axes, policy)
       when policy in ["not_larger", "not_smaller"] do
    # Pick a single scale that keeps the aspect ratio: smallest scale
    # so no dim exceeds target ("not_larger" → round down so result
    # ≤ scaled input), or largest so no dim falls below target
    # ("not_smaller" → round up so result ≥ scaled input). All resized
    # axes get that scale; un-resized axes keep their input size.
    scales =
      Enum.map(zip, fn {ax, target} ->
        in_dim = Enum.at(input_shape, ax)
        target / in_dim
      end)

    {chosen_scale, rounder} =
      case policy do
        "not_larger" -> {Enum.min(scales), &Float.floor/1}
        "not_smaller" -> {Enum.max(scales), &Float.ceil/1}
      end

    Enum.map(0..(length(input_shape) - 1)//1, fn ax ->
      if ax in axes,
        do: (Enum.at(input_shape, ax) * chosen_scale) |> rounder.() |> trunc(),
        else: Enum.at(input_shape, ax)
    end)
  end

  # Bilinear interpolation along each resized axis: for output index i,
  # compute the floating input coordinate, then interpolate between the
  # two adjacent integer positions with weights (1 - frac) and frac.
  # Implemented as a sequence of per-axis Nx.gather operations followed
  # by weighted Nx.add — gives ONNX bilinear (and N-D linear) by
  # composing 1-D interpolation along each resized axis independently.
  defp resize_linear_apply(x, input_shape, out_shape, per_axis_scales, ctm) do
    rank = length(input_shape)

    Enum.reduce(0..(rank - 1)//1, x, fn ax, acc ->
      in_dim = Enum.at(input_shape, ax)
      out_dim = Enum.at(out_shape, ax)
      s = Enum.at(per_axis_scales, ax)

      if out_dim == in_dim and s == 1.0 do
        acc
      else
        # Build {out_dim} arrays of {lo, hi, weight}
        coords =
          for out_i <- 0..(out_dim - 1) do
            in_f = resize_coord_transform(out_i, in_dim, out_dim, s, ctm)
            in_f_c = in_f |> max(0.0) |> min(in_dim - 1.0)
            lo = in_f_c |> :math.floor() |> trunc()
            hi = min(lo + 1, in_dim - 1)
            frac = in_f_c - lo
            {lo, hi, frac}
          end

        lo_idx = Enum.map(coords, fn {l, _, _} -> l end)
        hi_idx = Enum.map(coords, fn {_, h, _} -> h end)
        weights = Enum.map(coords, fn {_, _, f} -> f end)

        lo_t = Nx.take(acc, Nx.tensor(lo_idx, type: {:s, 64}), axis: ax)
        hi_t = Nx.take(acc, Nx.tensor(hi_idx, type: {:s, 64}), axis: ax)

        w_shape =
          List.to_tuple(
            for i <- 0..(rank - 1)//1, do: if(i == ax, do: out_dim, else: 1)
          )

        w =
          Nx.tensor(weights, type: Nx.type(acc))
          |> Nx.reshape(w_shape)

        Nx.add(Nx.multiply(lo_t, Nx.subtract(Nx.tensor(1.0, type: Nx.type(acc)), w)),
               Nx.multiply(hi_t, w))
      end
    end)
  end

  # Returns the permutation that swaps axes `a` and `b` in a tensor of
  # rank `rank`. e.g. `swap_axes(4, 0, 1)` is `[1, 0, 2, 3]`.
  defp swap_axes(rank, a, b) do
    Enum.map(0..(rank - 1)//1, fn i ->
      cond do
        i == a -> b
        i == b -> a
        true -> i
      end
    end)
  end

  # ONNX ceil_mode=1 rounds the spatial output dim up. Lower it to a
  # two-step adjustment so the floor-mode pool used downstream matches
  # spec output:
  #
  #   1. extra right-padding so the ceil-rounded count of windows fits
  #   2. a trailing slice to drop any window whose start falls entirely
  #      in the pad region ("last window starts on pad")
  #
  # Returns `{padding_config, trim_per_axis}` where trim_per_axis is the
  # number of trailing windows to slice off each spatial axis.
  defp maybe_ceil_mode_adjustment(0, base_padding, spatial_rank, _, _, _, _),
    do: {base_padding, List.duplicate(0, spatial_rank)}

  defp maybe_ceil_mode_adjustment(1, base_padding, spatial_rank, kernel_size, strides, dilations, inp) do
    explicit =
      case base_padding do
        list when is_list(list) -> list
        :valid -> List.duplicate({0, 0}, spatial_rank)
        # :same is shape-preserving already; ceil_mode is moot.
        other -> other
      end

    case explicit do
      list when is_list(list) ->
        input_spatial =
          case kernel_shape_from_axon!(inp) do
            shape when is_tuple(shape) ->
              shape |> Tuple.to_list() |> Enum.take(-spatial_rank)
          end

        stride_list = expand_to_spatial(strides, spatial_rank)
        dilation_list = expand_to_spatial(dilations, spatial_rank)

        list
        |> Enum.zip(input_spatial)
        |> Enum.zip(Enum.zip(Tuple.to_list(kernel_size), Enum.zip(stride_list, dilation_list)))
        |> Enum.map(fn {{{lo, hi}, in_dim}, {k, {s, d}}} ->
          eff_k = (k - 1) * d + 1
          numerator = in_dim + lo + hi - eff_k

          ceil_out =
            if numerator >= 0,
              do: div(numerator + s - 1, s) + 1,
              else: 1

          last_start = (ceil_out - 1) * s
          drop = if last_start >= in_dim + lo, do: 1, else: 0
          kept_out = max(ceil_out - drop, 1)
          required_padded = (ceil_out - 1) * s + eff_k
          extra = max(required_padded - (in_dim + lo + hi), 0)
          {{lo, hi + extra}, ceil_out - kept_out}
        end)
        |> Enum.unzip()
        |> case do
          {padding, trims} -> {padding, trims}
        end

      atom ->
        {atom, List.duplicate(0, spatial_rank)}
    end
  end

  # ONNX Div uses C-style truncation for integers (`Nx.quotient`) and
  # standard floating-point division for floats — `Nx.divide` would
  # promote to float for integer inputs, which differs from the spec.
  @doc false
  def onnx_div(x, y) do
    case Nx.type(x) do
      {kind, _} when kind in [:s, :u] -> Nx.quotient(x, y)
      _ -> Nx.divide(x, y)
    end
  end

  # ONNX Round is half-to-even (banker's rounding), differing from
  # `Nx.round` which rounds half-away-from-zero (e.g. 2.5 → 3 vs ONNX's
  # 2.5 → 2). Implementation: floor + 1 when frac > 0.5 OR (frac == 0.5
  # AND floor is odd). All other ops give the same answer as `Nx.round`.
  @doc false
  def round_half_to_even(x) do
    floor_x = Nx.floor(x)
    frac = Nx.subtract(x, floor_x)
    # floor_x is odd iff floor_x / 2 has a nonzero fractional part
    half_floor = Nx.multiply(Nx.floor(Nx.divide(floor_x, 2.0)), 2.0)
    floor_odd = Nx.not_equal(half_floor, floor_x)
    round_up =
      Nx.logical_or(
        Nx.greater(frac, 0.5),
        Nx.logical_and(Nx.equal(frac, 0.5), floor_odd)
      )

    Nx.add(floor_x, Nx.as_type(round_up, Nx.type(x)))
  end

  for {op, binary_fun, op_name} <- @binary_op_types do
    defp recur_nodes(
           %Node{op_type: unquote(op), input: [inp1, inp2], output: [output_name]},
           {axon, params, used_params}
         ) do
      inp1 = input!(inp1, axon, params, used_params)
      inp2 = input!(inp2, axon, params, used_params)

      fun = fn x, y, _opts -> apply(unquote(binary_fun), [x, y]) end

      {updated_axon, updated_params} =
        case {get_axon_node(inp1), get_axon_node(inp2)} do
          {%Axon.Node{op: :constant, opts: [value: v1]},
           %Axon.Node{op: :constant, opts: [value: v2]}} ->
            new_value = apply(unquote(binary_fun), [v1, v2])
            {Map.put(axon, output_name, Axon.constant(new_value, name: output_name)), used_params}

          {%Axon.Node{op: :constant, opts: [value: v1]}, %Nx.Tensor{} = v2} ->
            new_value = apply(unquote(binary_fun), [v1, v2])
            {Map.put(axon, output_name, Axon.constant(new_value, name: output_name)), used_params}

          {%Nx.Tensor{} = v1, %Axon.Node{op: :constant, opts: [value: v2]}} ->
            new_value = apply(unquote(binary_fun), [v1, v2])
            {Map.put(axon, output_name, Axon.constant(new_value, name: output_name)), used_params}

          {%Nx.Tensor{} = v1, %Nx.Tensor{} = v2} ->
            new_value = apply(unquote(binary_fun), [v1, v2])
            {Map.put(axon, output_name, Axon.constant(new_value, name: output_name)), used_params}

          {%Axon.Node{}, %Axon.Node{}} ->
            layer = Axon.layer(fun, [inp1, inp2], name: output_name, op_name: unquote(op_name))
            {Map.put(axon, output_name, layer), used_params}

          {%Axon.Node{}, %Nx.Tensor{}} ->
            layer =
              trainable_binary_layer(
                inp1,
                inp2,
                unquote(binary_fun),
                output_name,
                unquote(op_name)
              )

            updated_axon = Map.put(axon, output_name, layer)
            updated_params = Map.put(used_params, output_name, %{"kernel" => inp1})
            {updated_axon, updated_params}

          {%Nx.Tensor{}, %Axon.Node{}} ->
            layer =
              trainable_binary_layer(
                inp2,
                inp1,
                unquote(binary_fun),
                output_name,
                unquote(op_name)
              )

            updated_axon = Map.put(axon, output_name, layer)
            updated_params = Map.put(used_params, output_name, %{"kernel" => inp2})
            {updated_axon, updated_params}
        end

      {updated_axon, params, updated_params}
    end
  end

  # ONNX Mod splits on the `fmod` attribute. fmod=1 is C `fmod` (sign of
  # dividend) which matches `Nx.remainder`. fmod=0 (default for integer
  # inputs) is numpy.mod (sign of divisor) — we recover that via
  # `((a % b) + b) % b`. fmod=1 is required for floats per the spec.
  defp recur_nodes(
         %Node{
           op_type: "Mod",
           attribute: attrs,
           input: [inp1_name, inp2_name],
           output: [output_name]
         },
         {axon, params, used_params}
       ) do
    inp1 = input!(inp1_name, axon, params, used_params)
    inp2 = input!(inp2_name, axon, params, used_params)

    fmod = options!(attrs)["fmod"] || 0

    fun = fn x, y, opts ->
      cond do
        opts[:fmod] == 1 ->
          Nx.remainder(x, y)

        match?({k, _} when k in [:s, :u], Nx.type(x)) ->
          Nx.remainder(Nx.add(Nx.remainder(x, y), y), y)

        true ->
          Nx.remainder(x, y)
      end
    end

    apply_fun = &fun.(&1, &2, fmod: fmod)

    {updated_axon, updated_params} =
      case {get_axon_node(inp1), get_axon_node(inp2)} do
        {%Axon.Node{op: :constant, opts: [value: v1]},
         %Axon.Node{op: :constant, opts: [value: v2]}} ->
          {Map.put(axon, output_name, Axon.constant(apply_fun.(v1, v2), name: output_name)),
           used_params}

        {%Axon.Node{op: :constant, opts: [value: v1]}, %Nx.Tensor{} = v2} ->
          {Map.put(axon, output_name, Axon.constant(apply_fun.(v1, v2), name: output_name)),
           used_params}

        {%Nx.Tensor{} = v1, %Axon.Node{op: :constant, opts: [value: v2]}} ->
          {Map.put(axon, output_name, Axon.constant(apply_fun.(v1, v2), name: output_name)),
           used_params}

        {%Nx.Tensor{} = v1, %Nx.Tensor{} = v2} ->
          {Map.put(axon, output_name, Axon.constant(apply_fun.(v1, v2), name: output_name)),
           used_params}

        {%Axon.Node{}, %Axon.Node{}} ->
          layer = Axon.layer(fun, [inp1, inp2], name: output_name, op_name: :mod, fmod: fmod)
          {Map.put(axon, output_name, layer), used_params}

        {%Axon.Node{}, %Nx.Tensor{}} ->
          layer =
            Axon.layer(fun, [inp1, Axon.constant(inp2)],
              name: output_name,
              op_name: :mod,
              fmod: fmod
            )

          {Map.put(axon, output_name, layer), used_params}

        {%Nx.Tensor{}, %Axon.Node{}} ->
          layer =
            Axon.layer(fun, [Axon.constant(inp1), inp2],
              name: output_name,
              op_name: :mod,
              fmod: fmod
            )

          {Map.put(axon, output_name, layer), used_params}
      end

    {updated_axon, params, updated_params}
  end

  defp recur_nodes(
         %Node{
           op_type: "BitShift",
           attribute: attrs,
           input: [inp1_name, inp2_name],
           output: [output_name]
         },
         {axon, params, used_params}
       ) do
    inp1 = input!(inp1_name, axon, params, used_params)
    inp2 = input!(inp2_name, axon, params, used_params)

    bitshift_options = options!(attrs)
    direction = bitshift_options["direction"]

    fun = fn x, y, opts ->
      case opts[:direction] do
        "LEFT" -> Nx.left_shift(Nx.as_type(x, {:s, 64}), Nx.as_type(y, {:s, 64}))
        "RIGHT" -> Nx.right_shift(Nx.as_type(x, {:s, 64}), Nx.as_type(y, {:s, 64}))
      end
    end

    {updated_axon, updated_params} =
      case {get_axon_node(inp1), get_axon_node(inp2)} do
        {%Axon.Node{op: :constant, opts: [value: v1]},
         %Axon.Node{op: :constant, opts: [value: v2]}} ->
          new_value = apply(fun, [v1, v2, [direction: direction]])
          {Map.put(axon, output_name, Axon.constant(new_value, name: output_name)), used_params}

        {%Axon.Node{op: :constant, opts: [value: v1]}, %Nx.Tensor{} = v2} ->
          new_value = apply(fun, [v1, v2, [direction: direction]])
          {Map.put(axon, output_name, Axon.constant(new_value, name: output_name)), used_params}

        {%Nx.Tensor{} = v1, %Axon.Node{op: :constant, opts: [value: v2]}} ->
          new_value = apply(fun, [v1, v2, [direction: direction]])
          {Map.put(axon, output_name, Axon.constant(new_value, name: output_name)), used_params}

        {%Nx.Tensor{} = v1, %Nx.Tensor{} = v2} ->
          new_value = apply(fun, [v1, v2, [direction: direction]])
          {Map.put(axon, output_name, Axon.constant(new_value, name: output_name)), used_params}

        {%Axon.Node{}, %Axon.Node{}} ->
          layer =
            Axon.layer(fun, [inp1, inp2],
              name: output_name,
              op_name: :bitshift,
              direction: direction
            )

          {Map.put(axon, output_name, layer), used_params}

        {%Axon.Node{}, %Nx.Tensor{}} ->
          layer =
            trainable_binary_layer(
              inp1,
              inp2,
              fun,
              output_name,
              :bitshift
            )

          updated_axon = Map.put(axon, output_name, layer)
          updated_params = Map.put(used_params, output_name, %{"kernel" => inp1})
          {updated_axon, updated_params}

        {%Nx.Tensor{}, %Axon.Node{}} ->
          layer =
            trainable_binary_layer(
              inp2,
              inp1,
              fun,
              output_name,
              :bitshift
            )

          updated_axon = Map.put(axon, output_name, layer)
          updated_params = Map.put(used_params, output_name, %{"kernel" => inp2})
          {updated_axon, updated_params}
      end

    {updated_axon, params, updated_params}
  end

  @global_pool_types [
    {"GlobalAveragePool", :global_avg_pool},
    {"GlobalLpPool", :global_lp_pool},
    {"GlobalMaxPool", :global_max_pool}
  ]

  for {op, global_pool_op} <- @global_pool_types do
    defp recur_nodes(
           %Node{op_type: unquote(op), attribute: attrs, input: [input], output: [output_name]},
           {axon, params, used_params}
         ) do
      opts =
        if unquote(op) == "GlobalLpPool" do
          lp_pool_options = options!(attrs)
          [channels: :first, name: output_name, keep_axes: true, norm: lp_pool_options["p"]]
        else
          [channels: :first, name: output_name, keep_axes: true]
        end

      inp = axon!(input, axon)
      layer = apply(Axon, unquote(global_pool_op), [inp, opts])
      updated_axon = Map.put(axon, output_name, layer)

      {updated_axon, params, used_params}
    end
  end

  @variadic_op_types [
    {"Max", &Nx.max/2, :max},
    {"Min", &Nx.min/2, :min},
    {"Sum", &Nx.add/2, :add}
  ]
  for {op, variadic_op, op_name} <- @variadic_op_types do
    defp recur_nodes(
           %Node{op_type: unquote(op), input: inputs, output: [output_name]},
           {axon, params, used_params}
         ) do
      inputs = Enum.map(inputs, &input!(&1, axon, params, used_params))

      fun = fn inputs, _opts ->
        [init | rest] = inputs |> Tuple.to_list()

        Enum.reduce(rest, init, fn x, y ->
          apply(unquote(variadic_op), [x, y])
        end)
      end

      layer =
        Axon.layer(fun, [Axon.container(List.to_tuple(inputs))],
          name: output_name,
          op_name: unquote(op_name)
        )

      updated_axon = Map.put(axon, output_name, layer)

      {updated_axon, params, used_params}
    end
  end

  defp recur_nodes(
         %Node{op_type: "Mean", input: inputs, output: [output_name]},
         {axon, params, used_params}
       ) do
    # ONNX Mean is variadic — true element-wise mean of N tensors, i.e.
    # (x1 + x2 + ... + xN) / N. The legacy binary `mean/2` helper does
    # pairwise (x+y)/2 which gives the wrong answer for N>2 (you'd lose
    # half the weight of the early operands), so we sum-then-divide here.
    n = length(inputs)
    inputs = Enum.map(inputs, &input!(&1, axon, params, used_params))

    fun = fn inputs, _opts ->
      [init | rest] = Tuple.to_list(inputs)
      sum = Enum.reduce(rest, init, &Nx.add/2)
      # Use a float divisor — Nx.divide truncates on integer inputs, and
      # ONNX Mean's type constraint is float-only but corpus-style models
      # sometimes feed integer test data.
      Nx.divide(sum, n * 1.0)
    end

    layer =
      Axon.layer(fun, [Axon.container(List.to_tuple(inputs))],
        name: output_name,
        op_name: :mean
      )

    updated_axon = Map.put(axon, output_name, layer)
    {updated_axon, params, used_params}
  end

  defp recur_nodes(
         %Node{op_type: "Cast", attribute: attrs, input: [input], output: [output_name]},
         {axon, params, used_params}
       ) do
    cast_options = options!(attrs)
    inp = axon!(input, axon)
    nx_type = onnx_type_to_nx_type(cast_options["to"])

    updated_axon =
      case get_axon_node(inp) do
        %Axon.Node{op: :constant, opts: [value: v]} ->
          new_value = Nx.as_type(v, nx_type)
          Map.put(axon, output_name, Axon.constant(new_value, name: output_name))

        %Axon.Node{} ->
          # Stash the target dtype in the layer's opts so the serializer can
          # round-trip Cast — Axon.nx with a captured `&Nx.as_type(&1, t)`
          # buries `t` inside the closure where Axon.Serialize can't see it.
          fun = fn x, opts -> Nx.as_type(x, opts[:to]) end

          layer =
            Axon.layer(fun, [inp], name: output_name, op_name: :cast, to: nx_type)

          Map.put(axon, output_name, layer)
      end

    {updated_axon, params, used_params}
  end

  defp recur_nodes(
         %Node{op_type: "CastLike", input: [inp_name, like_name], output: [output_name]},
         {axon, params, used_params}
       ) do
    inp = input!(inp_name, axon, params, used_params)
    like = input!(like_name, axon, params, used_params)

    target_type =
      case get_axon_node(like) do
        %Axon.Node{op: :constant, opts: [value: v]} -> {:static, Nx.type(v)}
        %Nx.Tensor{} = t -> {:static, Nx.type(t)}
        %Axon.Node{} -> :runtime
      end

    layer =
      case {get_axon_node(inp), target_type} do
        {%Axon.Node{op: :constant, opts: [value: v]}, {:static, t}} ->
          Axon.constant(Nx.as_type(v, t), name: output_name)

        {%Nx.Tensor{} = v, {:static, t}} ->
          Axon.constant(Nx.as_type(v, t), name: output_name)

        {%Axon.Node{}, {:static, t}} ->
          Axon.layer(fn x, _opts -> Nx.as_type(x, t) end, [inp],
            name: output_name,
            op_name: :cast_like
          )

        {%Axon.Node{}, :runtime} ->
          Axon.layer(fn x, like_tensor, _opts -> Nx.as_type(x, Nx.type(like_tensor)) end,
            [inp, like],
            name: output_name,
            op_name: :cast_like
          )
      end

    updated_axon = Map.put(axon, output_name, layer)
    {updated_axon, params, used_params}
  end

  defp recur_nodes(
         %Node{op_type: "LRN", input: [input], attribute: attrs, output: [output_name]},
         {axon, params, used_params}
       ) do
    inp = axon!(input, axon)
    lrn_options = options!(attrs)
    opts = Enum.map(lrn_options, fn {k, v} -> {String.to_atom(k), v} end)

    layer = Axon.nx(inp, &lrn(&1, opts), name: output_name, op_name: :lrn)
    updated_axon = Map.put(axon, output_name, layer)
    {updated_axon, params, used_params}
  end

  defp recur_nodes(
         %Node{op_type: "Gather", input: [x, ind], output: [output_name], attribute: attrs},
         {axon, params, used_params}
       ) do
    x = input!(x, axon, params, used_params)
    ind = input!(ind, axon, params, used_params)
    gather_options = options!(attrs)

    {updated_axon, updated_params} =
      case {get_axon_node(x), get_axon_node(ind)} do
        {%Nx.Tensor{} = kernel, %Axon.Node{}} ->
          {in_size, out_size} = Nx.shape(kernel)
          layer = Axon.embedding(ind, in_size, out_size, name: output_name)

          updated_params = Map.put(used_params, output_name, %{"kernel" => kernel})
          updated_axon = Map.put(axon, output_name, layer)
          {updated_axon, updated_params}

        {%Axon.Node{op: :constant, opts: [value: x]},
         %Axon.Node{op: :constant, opts: [value: ind]}} ->
          new_value = Nx.take(x, Nx.as_type(ind, {:s, 64}))
          layer = Axon.constant(new_value, name: output_name)
          updated_axon = Map.put(axon, output_name, layer)
          {updated_axon, used_params}

        {%Nx.Tensor{} = x, %Nx.Tensor{} = ind} ->
          new_value = Nx.take(x, Nx.as_type(ind, {:s, 64}))
          layer = Axon.constant(new_value, name: output_name)
          updated_axon = Map.put(axon, output_name, layer)
          {updated_axon, used_params}

        {%Axon.Node{}, %Axon.Node{}} ->
          # ONNX Gather default axis is 0; Nx.take rejects nil.
          axis = gather_options["axis"] || 0
          layer = gather_layer(x, ind, axis, output_name)
          updated_axon = Map.put(axon, output_name, layer)
          {updated_axon, used_params}
      end

    {updated_axon, params, updated_params}
  end

  defp recur_nodes(
         %Node{op_type: "MatMul", input: [a, b], output: [output_name]},
         {axon, params, used_params}
       ) do
    a = input!(a, axon, params, used_params)
    b = input!(b, axon, params, used_params)

    # TODO: Constant folding
    {updated_axon, updated_params} =
      case {get_axon_node(a), get_axon_node(b)} do
        {%Axon.Node{}, %Nx.Tensor{} = kernel} ->
          units = Nx.shape(kernel) |> elem(1)

          layer = Axon.dense(a, units, name: output_name, use_bias: false)

          updated_axon = Map.put(axon, output_name, layer)
          updated_params = Map.put(used_params, output_name, %{"kernel" => kernel})
          {updated_axon, updated_params}

        {%Nx.Tensor{} = kernel, %Axon.Node{}} ->
          units = Nx.shape(kernel) |> elem(1)

          layer = Axon.dense(b, units, name: output_name, use_bias: false)

          updated_axon = Map.put(axon, output_name, layer)
          updated_params = Map.put(used_params, output_name, %{"kernel" => kernel})
          {updated_axon, updated_params}

        {%Axon.Node{}, %Axon.Node{}} ->
          layer = numpy_matmul_layer(a, b, output_name)
          updated_axon = Map.put(axon, output_name, layer)
          {updated_axon, used_params}
      end

    {updated_axon, params, updated_params}
  end

  defp recur_nodes(
         %Node{op_type: "Gemm", input: [a, b | maybe_c], attribute: attrs, output: [output_name]},
         {axon, params, used_params}
       ) do
    gemm_options = options!(attrs)

    alpha = Nx.tensor(gemm_options["alpha"] || 1.0)
    beta = Nx.tensor(gemm_options["beta"] || 1.0)
    trans_a = gemm_options["transA"]
    trans_b = gemm_options["transB"]

    a = input!(a, axon, params, used_params)
    b = input!(b, axon, params, used_params)

    c =
      case maybe_c do
        [] ->
          nil

        [c_name] ->
          input!(c_name, axon, params, used_params)
      end

    {updated_axon, updated_params} =
      case {get_axon_node(a), get_axon_node(b), get_axon_node(c)} do
        {%Axon.Node{}, %Nx.Tensor{} = kernel, nil} ->
          inp = if trans_a == 1, do: Axon.transpose(a), else: a
          kernel = if trans_b == 1, do: Nx.transpose(kernel), else: kernel

          units = Nx.shape(kernel) |> elem(1)

          layer =
            inp
            |> Axon.dense(units, name: output_name, use_bias: false)
            |> Axon.multiply(Axon.constant(alpha, name: "gemm_alpha"))

          updated_axon = Map.put(axon, output_name, layer)
          updated_params = Map.put(used_params, output_name, %{"kernel" => kernel})

          {updated_axon, updated_params}

        {%Nx.Tensor{} = kernel, %Axon.Node{}, nil} ->
          inp = if trans_a == 1, do: Axon.transpose(b), else: b
          kernel = if trans_b == 1, do: Nx.transpose(kernel), else: kernel

          units = Nx.shape(kernel) |> elem(1)

          layer =
            inp
            |> Axon.dense(units, name: output_name, use_bias: false)
            |> Axon.multiply(Axon.constant(alpha, name: "gemm_alpha"))

          updated_axon = Map.put(axon, output_name, layer)
          updated_params = Map.put(used_params, output_name, %{"kernel" => kernel})

          {updated_axon, updated_params}

        {%Axon.Node{}, %Axon.Node{}, nil} ->
          a = if trans_a == 1, do: Axon.transpose(a), else: a
          b = if trans_b == 1, do: Axon.transpose(b), else: b

          layer =
            a
            |> numpy_matmul_layer(b, output_name)
            |> Axon.multiply(Axon.constant(alpha, name: "gemm_alpha"))

          updated_axon = Map.put(axon, output_name, layer)
          {updated_axon, used_params}

        {%Axon.Node{}, %Nx.Tensor{} = b, %Nx.Tensor{} = c} ->
          a = if trans_a == 1, do: Axon.transpose(a), else: a
          b = if trans_b == 1, do: Nx.transpose(b), else: b

          layer = dense_with_bias(a, b, alpha, beta, output_name)
          updated_axon = Map.put(axon, output_name, layer)
          updated_params = Map.put(used_params, output_name, %{"kernel" => b, "bias" => c})

          {updated_axon, updated_params}

        {%Nx.Tensor{} = a, %Axon.Node{}, %Nx.Tensor{} = c} ->
          a = if trans_a == 1, do: Nx.transpose(a), else: a
          b = if trans_b == 1, do: Axon.transpose(b), else: b

          layer = dense_with_bias(b, a, alpha, beta, output_name)
          updated_axon = Map.put(axon, output_name, layer)
          updated_params = Map.put(used_params, output_name, %{"kernel" => a, "bias" => c})

          {updated_axon, updated_params}

        {%Axon.Node{}, %Axon.Node{}, %Axon.Node{}} ->
          a = if trans_a == 1, do: Axon.transpose(a), else: a
          b = if trans_b == 1, do: Axon.transpose(b), else: b

          layer =
            a
            |> numpy_matmul_layer(b, output_name)
            |> Axon.multiply(Axon.constant(alpha, name: "gemm_alpha"))
            |> Axon.add(Axon.multiply(c, Axon.constant(beta, name: "gemm_beta")))

          updated_axon = Map.put(axon, output_name, layer)
          {updated_axon, used_params}
      end

    {updated_axon, params, updated_params}
  end

  defp recur_nodes(
         %Node{op_type: "MaxPool", input: [inp], attribute: attrs, output: [output_name | _]},
         {axon, params, used_params}
       ) do
    max_pool_options = options!(attrs)

    kernel_shape = max_pool_options["kernel_shape"]
    ceil_mode = max_pool_options["ceil_mode"] || 0
    auto_pad = max_pool_options["auto_pad"] || "NOTSET"
    storage_order = max_pool_options["storage_order"]
    pads = max_pool_options["pads"]
    strides = max_pool_options["strides"]
    dilations = max_pool_options["dilations"] || 1

    kernel_size = List.to_tuple(kernel_shape)

    if storage_order do
      Logger.warning(
        "Storage order is not supported by Axon and is instead a backend-specific" <>
          " detail. Your model might behave differently from the imported version if" <>
          " the storage order differs"
      )
    end

    strides =
      if strides do
        strides
      else
        List.duplicate(1, tuple_size(kernel_size))
      end

    inp = axon!(inp, axon)

    base_padding = padding!(auto_pad, pads, kernel_size, strides)
    spatial_rank = tuple_size(kernel_size)

    # ceil_mode=1: same as AveragePool — add right-pad to make the
    # ceil-rounded window count fit, then trim trailing windows whose
    # start positions are entirely in the pad region. For MaxPool the
    # "pad" cells need to be the dtype's neg_infinity so they never win
    # max comparisons; we use a custom layer when trimming, otherwise
    # fall back on Axon.max_pool (which is what the serializer can
    # round-trip).
    {padding_config, trim_per_axis} =
      maybe_ceil_mode_adjustment(
        ceil_mode,
        base_padding,
        spatial_rank,
        kernel_size,
        strides,
        dilations,
        inp
      )

    needs_trim = Enum.any?(trim_per_axis, &(&1 > 0))
    pool_name = if needs_trim, do: output_name <> "__pool", else: output_name

    pool_layer =
      Axon.max_pool(inp,
        kernel_size: kernel_size,
        strides: strides,
        padding: padding_config,
        dilations: dilations,
        name: pool_name,
        channels: :first
      )

    layer =
      if needs_trim do
        Axon.nx(
          pool_layer,
          fn x ->
            trim_per_axis
            |> Enum.with_index()
            |> Enum.reduce(x, fn {trim, idx}, acc ->
              if trim > 0 do
                axis = Nx.rank(acc) - spatial_rank + idx
                len = Nx.axis_size(acc, axis) - trim
                Nx.slice_along_axis(acc, 0, len, axis: axis)
              else
                acc
              end
            end)
          end,
          name: output_name,
          op_name: :max_pool_trim
        )
      else
        pool_layer
      end

    updated_axon = Map.put(axon, output_name, layer)
    {updated_axon, params, used_params}
  end

  defp recur_nodes(
         %Node{op_type: "AveragePool", input: [inp], attribute: attrs, output: [output_name]},
         {axon, params, used_params}
       ) do
    avg_pool_options = options!(attrs)

    kernel_shape = avg_pool_options["kernel_shape"]
    ceil_mode = avg_pool_options["ceil_mode"] || 0
    auto_pad = avg_pool_options["auto_pad"] || "NOTSET"
    count_include_pad = avg_pool_options["count_include_pad"] || 0
    pads = avg_pool_options["pads"]
    strides = avg_pool_options["strides"] || 1
    dilations = avg_pool_options["dilations"] || 1

    kernel_size = List.to_tuple(kernel_shape)

    strides =
      if strides do
        strides
      else
        List.duplicate(1, tuple_size(kernel_size))
      end

    inp = axon!(inp, axon)

    base_padding = padding!(auto_pad, pads, kernel_size, strides)

    spatial_rank = tuple_size(kernel_size)

    {padding_config, trim_per_axis} =
      maybe_ceil_mode_adjustment(
        ceil_mode,
        base_padding,
        spatial_rank,
        kernel_size,
        strides,
        dilations,
        inp
      )

    needs_trim = Enum.any?(trim_per_axis, &(&1 > 0))
    pool_name = if needs_trim, do: output_name <> "__pool", else: output_name

    # When no actual padding is needed (or count_include_pad=1), the
    # built-in `Axon.avg_pool` divides by the full window size — which
    # matches both `count_include_pad=1` and the no-pad case (where every
    # window has the same divisor either way). Use the dedicated layer
    # so the serialiser can round-trip these cases via the `:avg_pool`
    # op-type matcher.
    no_explicit_padding =
      case padding_config do
        :valid -> true
        list when is_list(list) -> Enum.all?(list, fn {l, h} -> l == 0 and h == 0 end)
        _ -> false
      end

    use_builtin_avg_pool = count_include_pad == 1 or no_explicit_padding

    pool_layer =
      if use_builtin_avg_pool do
        Axon.avg_pool(inp,
          kernel_size: kernel_size,
          strides: strides,
          padding: padding_config,
          dilations: dilations,
          name: pool_name,
          channels: :first
        )
      else
        # ONNX default: padded zeros don't count toward the divisor.
        # Compute it as `window_sum(x) / window_sum(ones)` with the same
        # padding/strides on both — pad cells contribute 0 to the
        # numerator and 0 to the denominator. The kernel/strides/padding
        # cover only spatial dims; pad with identity (1 / [0,0]) on the
        # batch+channel axes so the window matches the input rank.
        fun = fn x, _opts ->
          rank = Nx.rank(x)
          spatial_rank = tuple_size(kernel_size)
          leading = rank - spatial_rank
          full_kernel = List.duplicate(1, leading) ++ Tuple.to_list(kernel_size)

          full_strides =
            cond do
              is_integer(strides) -> List.duplicate(1, leading) ++ List.duplicate(strides, spatial_rank)
              is_list(strides) -> List.duplicate(1, leading) ++ strides
            end

          full_padding =
            case padding_config do
              atom when is_atom(atom) ->
                atom

              list when is_list(list) ->
                List.duplicate({0, 0}, leading) ++ list
            end

          full_dilations =
            cond do
              is_integer(dilations) -> List.duplicate(1, leading) ++ List.duplicate(dilations, spatial_rank)
              is_list(dilations) -> List.duplicate(1, leading) ++ dilations
            end

          num =
            Nx.window_sum(x, List.to_tuple(full_kernel),
              strides: full_strides,
              padding: full_padding,
              window_dilations: full_dilations
            )

          ones = Nx.broadcast(Nx.tensor(1.0, type: Nx.type(x)), Nx.shape(x))

          den =
            Nx.window_sum(ones, List.to_tuple(full_kernel),
              strides: full_strides,
              padding: full_padding,
              window_dilations: full_dilations
            )

          Nx.divide(num, den)
        end

        Axon.layer(fun, [inp], name: pool_name, op_name: :avg_pool)
      end

    layer =
      if needs_trim do
        # Drop the trailing windows whose starts fall in the pad region
        # (ceil_mode's "last window starts on pad" rule).
        Axon.nx(
          pool_layer,
          fn x ->
            trim_per_axis
            |> Enum.with_index()
            |> Enum.reduce(x, fn {trim, idx}, acc ->
              if trim > 0 do
                axis = Nx.rank(acc) - spatial_rank + idx
                len = Nx.axis_size(acc, axis) - trim
                Nx.slice_along_axis(acc, 0, len, axis: axis)
              else
                acc
              end
            end)
          end,
          name: output_name,
          op_name: :avg_pool_trim
        )
      else
        pool_layer
      end

    {Map.put(axon, output_name, layer), params, used_params}
  end

  defp recur_nodes(
         %Node{op_type: "Conv", attribute: attrs, input: input, output: [output_name]},
         {axon, params, used_params}
       ) do
    conv_options = options!(attrs)

    kernel_shape_options = conv_options["kernel_shape"]
    auto_pad = conv_options["auto_pad"] || "NOTSET"
    dilations = conv_options["dilations"] || 1
    group = conv_options["group"] || 1
    pads = conv_options["pads"]

    [inp_name, kernel_name | maybe_bias] = input

    # ONNX Conv strides default to 1 per spatial axis; Nx.conv rejects nil
    # strides. Fill in once we know the spatial rank from the kernel.
    raw_strides = conv_options["strides"]

    inp = input!(inp_name, axon, params, used_params)
    kernel = input!(kernel_name, axon, params, used_params)

    # Axon.conv requires a static kernel shape. If the kernel is a graph
    # input, the coverage runner's retry loop can fold it from test data —
    # surface a recognisable error so the loop kicks in.
    if match?(%Axon{}, kernel) do
      raise ArgumentError,
            "expected value #{kernel_name} to be a graph input that resolves " <>
              "to a constant tensor — Conv weight must be statically known."
    end

    bias =
      case maybe_bias do
        [] ->
          nil

        [bias_name] ->
          input!(bias_name, axon, params, used_params)
      end

    kernel_shape = Nx.shape(kernel)

    # Kernel shape is a list of integers; If it's not present, infer it
    # from other values.
    kernel_size =
      if kernel_shape_options do
        List.to_tuple(kernel_shape_options)
      else
        Nx.shape(kernel)
        |> Tuple.delete_at(0)
        |> Tuple.delete_at(0)
      end

    strides = raw_strides || List.duplicate(1, tuple_size(kernel_size))
    padding_config = padding!(auto_pad, pads, kernel_size, strides)
    units = elem(kernel_shape, 0)

    {updated_axon, updated_params} =
      case {get_axon_node(inp), get_axon_node(kernel), get_axon_node(bias)} do
        {%Axon.Node{}, %Nx.Tensor{} = kernel, nil} ->
          out_layer =
            Axon.conv(
              inp,
              units,
              kernel_size: kernel_size,
              kernel_dilation: dilations,
              padding: padding_config,
              strides: strides,
              use_bias: false,
              name: output_name,
              feature_group_size: group,
              channels: :first
            )

          updated_axon = Map.put(axon, output_name, out_layer)
          updated_params = Map.put(used_params, output_name, %{"kernel" => kernel})
          {updated_axon, updated_params}

        {%Axon.Node{}, %Nx.Tensor{} = kernel, %Axon.Node{op: :constant, opts: [value: v]}} ->
          # Reshape a 1-D bias of shape {C_out} to {1, C_out, 1, ..., 1} for
          # channels=:first broadcasting. Axon 0.8 reworked
          # Axon.Shape.conv_bias_reshape's signature to take tensors; we
          # inline the shape formula here so we don't need the helper.
          spatial_rank = Nx.rank(kernel) - 2

          shape =
            case Nx.shape(v) do
              {} -> {}
              {c_out} -> List.to_tuple([1, c_out | List.duplicate(1, spatial_rank)])
              other -> other
            end

          out_layer =
            Axon.conv(
              inp,
              units,
              kernel_size: kernel_size,
              kernel_dilation: dilations,
              padding: padding_config,
              strides: strides,
              use_bias: false,
              name: output_name,
              feature_group_size: group,
              channels: :first
            )
            |> Axon.add(Axon.reshape(bias, shape))

          updated_axon = Map.put(axon, output_name, out_layer)
          updated_params = Map.put(used_params, output_name, %{"kernel" => kernel})
          {updated_axon, updated_params}

        {%Axon.Node{}, %Nx.Tensor{} = kernel, %Nx.Tensor{} = bias} ->
          out_layer =
            Axon.conv(
              inp,
              units,
              kernel_size: kernel_size,
              kernel_dilation: dilations,
              padding: padding_config,
              strides: strides,
              use_bias: true,
              name: output_name,
              feature_group_size: group,
              channels: :first
            )

          updated_axon = Map.put(axon, output_name, out_layer)

          updated_params =
            Map.put(used_params, output_name, %{"kernel" => kernel, "bias" => bias})

          {updated_axon, updated_params}
      end

    {updated_axon, params, updated_params}
  end

  defp recur_nodes(
         %Node{op_type: "ConvTranspose", attribute: attrs, input: input, output: [output_name]},
         {axon, params, used_params}
       ) do
    # ONNX ConvTranspose mirrors Conv but with the kernel oriented (C_in,
    # C_out/group, ...spatial...). Axon's conv_transpose handles the
    # spatial inversion and stride/dilation/padding; we keep channels=:first
    # to match ONNX layout.
    options = options!(attrs)
    auto_pad = options["auto_pad"] || "NOTSET"
    group = options["group"] || 1
    pads = options["pads"]
    kernel_shape_attr = options["kernel_shape"]

    if group != 1 do
      raise ArgumentError, "ConvTranspose with group #{group} is not yet supported"
    end

    [inp_name, kernel_name | maybe_bias] = input

    inp = input!(inp_name, axon, params, used_params)
    kernel = input!(kernel_name, axon, params, used_params)

    bias =
      case maybe_bias do
        [] -> nil
        [b] -> input!(b, axon, params, used_params)
      end

    kernel_shape =
      case kernel do
        %Nx.Tensor{} = t -> Nx.shape(t)
        %Axon{} = node -> kernel_shape_from_axon!(node)
      end

    kernel_size =
      if kernel_shape_attr do
        List.to_tuple(kernel_shape_attr)
      else
        kernel_shape |> Tuple.delete_at(0) |> Tuple.delete_at(0)
      end

    spatial_rank = tuple_size(kernel_size)
    dilations = options["dilations"] || List.duplicate(1, spatial_rank)
    strides = options["strides"] || List.duplicate(1, spatial_rank)

    padding_config = padding!(auto_pad, pads, kernel_size, strides)
    units = elem(kernel_shape, 1) * group

    base_opts = [
      kernel_size: kernel_size,
      kernel_dilation: dilations,
      padding: padding_config,
      strides: strides,
      name: output_name,
      channels: :first
    ]

    # ONNX kernel layout is (C_in, C_out/group, ...spatial); Axon expects
    # (C_out, C_in/group, ...spatial). Transpose the first two dims.
    onnx_to_axon_kernel = fn k ->
      perm = [1, 0 | Enum.to_list(2..(Nx.rank(k) - 1)//1)]
      Nx.transpose(k, axes: perm)
    end

    {updated_axon, updated_params} =
      case {get_axon_node(inp), get_axon_node(kernel), get_axon_node(bias)} do
        {%Axon.Node{}, %Nx.Tensor{} = kernel, nil} ->
          out = Axon.conv_transpose(inp, units, [use_bias: false] ++ base_opts)

          {Map.put(axon, output_name, out),
           Map.put(used_params, output_name, %{"kernel" => onnx_to_axon_kernel.(kernel)})}

        {%Axon.Node{}, %Nx.Tensor{} = kernel, %Nx.Tensor{} = bias} ->
          out = Axon.conv_transpose(inp, units, [use_bias: true] ++ base_opts)

          {Map.put(axon, output_name, out),
           Map.put(used_params, output_name, %{
             "kernel" => onnx_to_axon_kernel.(kernel),
             "bias" => bias
           })}

        {%Axon.Node{}, %Axon.Node{}, nil} ->
          # Kernel is a graph input rather than an initialiser — common in
          # the corpus, where every conv-style weight is a runtime input.
          # Drop to Axon.Layers.conv_transpose directly via Axon.layer so we
          # don't need Axon's own parameter creation; transpose the kernel
          # inside the layer fn so traced shapes match.
          fun = fn x, w, _opts ->
            w = onnx_to_axon_kernel.(w)

            Axon.Layers.conv_transpose(x, w, 0,
              strides: strides,
              padding: padding_config,
              kernel_dilation: dilations,
              channels: :first
            )
          end

          out = Axon.layer(fun, [inp, kernel], name: output_name, op_name: :conv_transpose)
          {Map.put(axon, output_name, out), used_params}

        {%Axon.Node{}, %Axon.Node{}, %Axon.Node{}} ->
          fun = fn x, w, b, _opts ->
            w = onnx_to_axon_kernel.(w)

            Axon.Layers.conv_transpose(x, w, b,
              strides: strides,
              padding: padding_config,
              kernel_dilation: dilations,
              channels: :first
            )
          end

          out =
            Axon.layer(fun, [inp, kernel, bias], name: output_name, op_name: :conv_transpose)

          {Map.put(axon, output_name, out), used_params}
      end

    {updated_axon, params, updated_params}
  end

  defp recur_nodes(
         %Node{
           op_type: "BatchNormalization",
           input: [inp, gamma, beta, mean, var],
           output: [output_name],
           attribute: attrs
         },
         {axon, params, used_params}
       ) do
    options = options!(attrs)

    mode = options["training_mode"] || 0
    epsilon = options["epsilon"] || 1.0e-5
    momentum = options["momenutm"] || 0.9

    if mode == 1 do
      Logger.warning("Training mode in batch norm has no effect")
    end

    inp = axon!(inp, axon)

    gamma = param!(gamma, params)
    beta = param!(beta, params)
    mean = param!(mean, params)
    var = param!(var, params)

    updated_axon =
      Map.put(
        axon,
        output_name,
        Axon.batch_norm(inp,
          name: output_name,
          momentum: momentum,
          epsilon: epsilon,
          channel_index: 1
        )
      )

    updated_params =
      Map.put(used_params, output_name, %{
        "gamma" => gamma,
        "beta" => beta,
        "mean" => mean,
        "var" => var
      })

    {updated_axon, params, updated_params}
  end

  defp recur_nodes(
         %Node{
           op_type: "InstanceNormalization",
           attribute: attrs,
           input: [input_name, scale_name, b_name],
           output: [output_name]
         },
         {axon, params, used_params}
       ) do
    options = options!(attrs)

    input = input!(input_name, axon, params, used_params)
    scale = input!(scale_name, axon, params, used_params)
    bias = input!(b_name, axon, params, used_params)

    epsilon = options["epsilon"] || 1.0e-5

    # Hand-roll the normalisation rather than going through
    # Axon.instance_norm — that path defaults to channels=:last while ONNX
    # is channels-first, and its scale/bias parameter shapes don't take
    # well to the corpus's rank-1 weights.
    fun = fn x, scale, bias, opts ->
      eps = opts[:epsilon]
      rank = Nx.rank(x)
      spatial_axes = Enum.to_list(2..(rank - 1)//1)
      mean = Nx.mean(x, axes: spatial_axes, keep_axes: true)
      var = Nx.variance(x, axes: spatial_axes, keep_axes: true)
      normalised = Nx.divide(Nx.subtract(x, mean), Nx.sqrt(Nx.add(var, eps)))

      param_shape = List.to_tuple([1, Nx.axis_size(x, 1) | List.duplicate(1, rank - 2)])
      scale_r = Nx.reshape(scale, param_shape)
      bias_r = Nx.reshape(bias, param_shape)
      Nx.add(Nx.multiply(normalised, scale_r), bias_r)
    end

    out =
      Axon.layer(fun, [input, scale, bias],
        name: output_name,
        op_name: :instance_norm,
        epsilon: epsilon
      )

    updated_axon = Map.put(axon, output_name, out)
    {updated_axon, params, used_params}
  end

  defp recur_nodes(
         %Node{
           op_type: "LayerNormalization",
           attribute: attrs,
           input: inputs,
           output: outputs
         },
         {axon, params, used_params}
       ) do
    # ONNX LayerNormalization (opset 17+) normalises over axes
    # [axis, axis+1, …, rank-1]. axis defaults to -1 (the last axis), which
    # matches Axon.layer_norm's channel_index, but axis=0 means
    # "normalise over the whole tensor" — Axon's layer doesn't directly
    # express that, so we lower to raw Nx ops here. Bias is optional.
    # When the model also requests Mean / InvStdDev outputs (their proto
    # names follow Y), register them as separate Axon layers; XLA's CSE
    # will deduplicate the shared mean/variance compute.
    options = options!(attrs)
    axis = options["axis"] || -1
    epsilon = options["epsilon"] || 1.0e-5

    {input_name, scale_name, bias_name} =
      case inputs do
        [i, s] -> {i, s, nil}
        [i, s, b] -> {i, s, b}
      end

    input = input!(input_name, axon, params, used_params)
    scale = input!(scale_name, axon, params, used_params)
    bias = if bias_name, do: input!(bias_name, axon, params, used_params), else: nil

    {y_fun, y_inputs} =
      case bias do
        nil ->
          {fn x, s, _opts -> do_layer_norm(x, s, nil, axis, epsilon) end, [input, scale]}

        _ ->
          {fn x, s, b, _opts -> do_layer_norm(x, s, b, axis, epsilon) end,
           [input, scale, bias]}
      end

    y_name = hd(outputs)
    y_layer = Axon.layer(y_fun, y_inputs, name: y_name, op_name: :layer_norm)
    axon = Map.put(axon, y_name, y_layer)

    axon =
      outputs
      |> Enum.with_index()
      |> Enum.reduce(axon, fn
        {_y, 0}, acc ->
          acc

        {mean_name, 1}, acc ->
          fun = fn x, _opts -> do_layer_norm_mean(x, axis) end
          layer = Axon.layer(fun, [input], name: mean_name, op_name: :layer_norm_mean)
          Map.put(acc, mean_name, layer)

        {inv_std_name, 2}, acc ->
          fun = fn x, _opts -> do_layer_norm_inv_std(x, axis, epsilon) end
          layer =
            Axon.layer(fun, [input], name: inv_std_name, op_name: :layer_norm_inv_std)

          Map.put(acc, inv_std_name, layer)
      end)

    {axon, params, used_params}
  end

  defp recur_nodes(
         %Node{
           op_type: "SoftmaxCrossEntropyLoss",
           attribute: attrs,
           input: inputs,
           output: outputs
         },
         {axon, params, used_params}
       ) do
    # SoftmaxCrossEntropyLoss = log_softmax(scores, axis=1) → NLLLoss.
    # Optional second output is log_prob (the post-log_softmax tensor).
    options = options!(attrs)
    reduction = options["reduction"] || "mean"
    ignore_index = options["ignore_index"]

    {scores_name, labels_name, weight_name} =
      case inputs do
        [s, l] -> {s, l, nil}
        [s, l, w] -> {s, l, w}
      end

    scores = input!(scores_name, axon, params, used_params)
    labels = input!(labels_name, axon, params, used_params)
    weight = if weight_name, do: input!(weight_name, axon, params, used_params), else: nil

    loss_name = hd(outputs)

    loss_layer =
      case weight do
        nil ->
          fun = fn s, l, _opts ->
            log_prob = Axon.Activations.log_softmax(s, axis: 1)
            do_nll_loss(log_prob, l, nil, ignore_index, reduction)
          end

          Axon.layer(fun, [scores, labels], name: loss_name, op_name: :softmax_cross_entropy)

        _ ->
          fun = fn s, l, w, _opts ->
            log_prob = Axon.Activations.log_softmax(s, axis: 1)
            do_nll_loss(log_prob, l, w, ignore_index, reduction)
          end

          Axon.layer(fun, [scores, labels, weight],
            name: loss_name,
            op_name: :softmax_cross_entropy
          )
      end

    axon = Map.put(axon, loss_name, loss_layer)

    axon =
      case outputs do
        [_only] ->
          axon

        [_loss_name, log_prob_name | _] ->
          log_prob_layer =
            Axon.layer(
              fn s, _opts -> Axon.Activations.log_softmax(s, axis: 1) end,
              [scores],
              name: log_prob_name,
              op_name: :log_softmax
            )

          Map.put(axon, log_prob_name, log_prob_layer)
      end

    {axon, params, used_params}
  end

  defp recur_nodes(
         %Node{
           op_type: "QLinearMatMul",
           input: [a_n, a_scale_n, a_zp_n, b_n, b_scale_n, b_zp_n, y_scale_n, y_zp_n],
           output: [output_name]
         },
         {axon, params, used_params}
       ) do
    # QLinearMatMul = dequantize(a, b) → MatMul → quantize(y_scale, y_zp).
    # All eight inputs are taken; y_zero_point's dtype defines the output
    # type. Supports rank 2 (plain) and rank 3+ (batched) inputs via
    # Nx.dot with explicit batch and contract axes.
    a = input!(a_n, axon, params, used_params)
    a_scale = input!(a_scale_n, axon, params, used_params)
    a_zp = input!(a_zp_n, axon, params, used_params)
    b = input!(b_n, axon, params, used_params)
    b_scale = input!(b_scale_n, axon, params, used_params)
    b_zp = input!(b_zp_n, axon, params, used_params)
    y_scale = input!(y_scale_n, axon, params, used_params)
    y_zp = input!(y_zp_n, axon, params, used_params)

    target_type = quantize_target_type(y_zp_n, y_zp)
    {min_v, max_v} = quantize_range(target_type)

    fun = fn a, a_scale, a_zp, b, b_scale, b_zp, y_scale, y_zp, _opts ->
      work_type = Nx.type(a_scale)

      a_f =
        Nx.multiply(
          Nx.subtract(Nx.as_type(a, work_type), Nx.as_type(a_zp, work_type)),
          a_scale
        )

      b_f =
        Nx.multiply(
          Nx.subtract(Nx.as_type(b, work_type), Nx.as_type(b_zp, work_type)),
          b_scale
        )

      y_f =
        case Nx.rank(a) do
          2 ->
            Nx.dot(a_f, b_f)

          rank ->
            batch_axes = Enum.to_list(0..(rank - 3)//1)
            Nx.dot(a_f, [rank - 1], batch_axes, b_f, [rank - 2], batch_axes)
        end

      # QLinearMatMul per ONNX spec doesn't saturate — the corpus's int8
      # golden wraps around on overflow (e.g. -236 → 20 = -236 + 256). Skip
      # the Nx.clip and let Nx.as_type's modular truncation handle it. The
      # remaining `min_v`/`max_v` are unused here; kept in the outer scope
      # for QuantizeLinear which does saturate.
      _ = {min_v, max_v}

      y_q =
        y_f
        |> Nx.divide(y_scale)
        |> Nx.round()
        |> Nx.add(Nx.as_type(y_zp, work_type))
        |> Nx.as_type(target_type)

      y_q
    end

    layer =
      Axon.layer(fun, [a, a_scale, a_zp, b, b_scale, b_zp, y_scale, y_zp],
        name: output_name,
        op_name: :qlinear_matmul
      )

    updated_axon = Map.put(axon, output_name, layer)
    {updated_axon, params, used_params}
  end

  defp recur_nodes(
         %Node{
           op_type: "QLinearConv",
           attribute: attrs,
           input: inputs,
           output: [output_name]
         },
         {axon, params, used_params}
       ) do
    # QLinearConv = dequantize(x, w) → Conv (+ optional int bias scaled
    # through x_scale*w_scale) → quantize(y_scale, y_zp).
    [x_n, x_scale_n, x_zp_n, w_n, w_scale_n, w_zp_n, y_scale_n, y_zp_n | maybe_b] = inputs

    options = options!(attrs)
    auto_pad = options["auto_pad"] || "NOTSET"
    group = options["group"] || 1
    pads = options["pads"]

    x = input!(x_n, axon, params, used_params)
    x_scale = input!(x_scale_n, axon, params, used_params)
    x_zp = input!(x_zp_n, axon, params, used_params)
    w = input!(w_n, axon, params, used_params)
    w_scale = input!(w_scale_n, axon, params, used_params)
    w_zp = input!(w_zp_n, axon, params, used_params)
    y_scale = input!(y_scale_n, axon, params, used_params)
    y_zp = input!(y_zp_n, axon, params, used_params)
    bias = if maybe_b != [], do: input!(hd(maybe_b), axon, params, used_params), else: nil

    target_type = quantize_target_type(y_zp_n, y_zp)
    {min_v, max_v} = quantize_range(target_type)

    w_shape =
      case w do
        %Nx.Tensor{} = t -> Nx.shape(t)
        %Axon{} = node -> kernel_shape_from_axon!(node)
      end

    kernel_size = w_shape |> Tuple.delete_at(0) |> Tuple.delete_at(0)
    spatial_rank = tuple_size(kernel_size)
    dilations = options["dilations"] || List.duplicate(1, spatial_rank)
    strides = options["strides"] || List.duplicate(1, spatial_rank)
    padding_config = padding!(auto_pad, pads, kernel_size, strides)

    common_inputs = [x, x_scale, x_zp, w, w_scale, w_zp, y_scale, y_zp]

    {fun, layer_inputs} =
      case bias do
        nil ->
          {fn x, x_scale, x_zp, w, w_scale, w_zp, y_scale, y_zp, _opts ->
             qlinear_conv_impl(
               x, x_scale, x_zp, w, w_scale, w_zp, y_scale, y_zp, nil,
               strides, padding_config, dilations, group, target_type, min_v, max_v
             )
           end, common_inputs}

        _ ->
          {fn x, x_scale, x_zp, w, w_scale, w_zp, y_scale, y_zp, b, _opts ->
             qlinear_conv_impl(
               x, x_scale, x_zp, w, w_scale, w_zp, y_scale, y_zp, b,
               strides, padding_config, dilations, group, target_type, min_v, max_v
             )
           end, common_inputs ++ [bias]}
      end

    layer = Axon.layer(fun, layer_inputs, name: output_name, op_name: :qlinear_conv)
    updated_axon = Map.put(axon, output_name, layer)
    {updated_axon, params, used_params}
  end

  defp recur_nodes(
         %Node{op_type: "MatMulInteger", input: inputs, output: [output_name]},
         {axon, params, used_params}
       ) do
    # MatMulInteger(A, B [, a_zero_point [, b_zero_point]]):
    # Y = matmul(A - a_zero_point, B - b_zero_point) at int32.
    {a_n, b_n, a_zp_n, b_zp_n} =
      case inputs do
        [a, b] -> {a, b, nil, nil}
        [a, b, a_zp] -> {a, b, a_zp, nil}
        [a, b, a_zp, b_zp] -> {a, b, a_zp, b_zp}
      end

    a = input!(a_n, axon, params, used_params)
    b = input!(b_n, axon, params, used_params)
    a_zp = if a_zp_n && a_zp_n != "", do: input!(a_zp_n, axon, params, used_params), else: nil
    b_zp = if b_zp_n && b_zp_n != "", do: input!(b_zp_n, axon, params, used_params), else: nil

    {fun, layer_inputs} = integer_matmul_layer(a, b, a_zp, b_zp)
    layer = Axon.layer(fun, layer_inputs, name: output_name, op_name: :matmul_integer)
    updated_axon = Map.put(axon, output_name, layer)
    {updated_axon, params, used_params}
  end

  defp recur_nodes(
         %Node{
           op_type: "ConvInteger",
           attribute: attrs,
           input: inputs,
           output: [output_name]
         },
         {axon, params, used_params}
       ) do
    # ConvInteger(x, w [, x_zp [, w_zp]]):
    # Y = conv(x - x_zp, w - w_zp) at int32. Reuses the Conv attribute
    # mapping (pads/auto_pad/dilations/strides/group).
    {x_n, w_n, x_zp_n, w_zp_n} =
      case inputs do
        [x, w] -> {x, w, nil, nil}
        [x, w, x_zp] -> {x, w, x_zp, nil}
        [x, w, x_zp, w_zp] -> {x, w, x_zp, w_zp}
      end

    options = options!(attrs)
    auto_pad = options["auto_pad"] || "NOTSET"
    group = options["group"] || 1
    pads = options["pads"]

    x = input!(x_n, axon, params, used_params)
    w = input!(w_n, axon, params, used_params)
    x_zp = if x_zp_n && x_zp_n != "", do: input!(x_zp_n, axon, params, used_params), else: nil
    w_zp = if w_zp_n && w_zp_n != "", do: input!(w_zp_n, axon, params, used_params), else: nil

    w_shape =
      case w do
        %Nx.Tensor{} = t -> Nx.shape(t)
        %Axon{} = node -> kernel_shape_from_axon!(node)
      end

    kernel_size = w_shape |> Tuple.delete_at(0) |> Tuple.delete_at(0)
    spatial_rank = tuple_size(kernel_size)
    dilations = options["dilations"] || List.duplicate(1, spatial_rank)
    strides = options["strides"] || List.duplicate(1, spatial_rank)
    padding_config = padding!(auto_pad, pads, kernel_size, strides)

    fun = build_conv_integer_fun(x_zp, w_zp, strides, padding_config, dilations, group)
    layer_inputs = [x, w] ++ Enum.reject([x_zp, w_zp], &is_nil/1)
    layer = Axon.layer(fun, layer_inputs, name: output_name, op_name: :conv_integer)
    updated_axon = Map.put(axon, output_name, layer)
    {updated_axon, params, used_params}
  end

  defp recur_nodes(
         %Node{
           op_type: "DynamicQuantizeLinear",
           input: [x_name],
           output: outputs
         },
         {axon, params, used_params}
       ) do
    # DynamicQuantizeLinear computes a uint8 quantisation from the input
    # tensor's own min/max (always including 0 in the represented range).
    # Spec: opset 11+. Three outputs: y (u8), y_scale (f32), y_zero_point (u8).
    # The output dtype is fixed at u8 by spec — unlike QuantizeLinear /
    # QLinearMatMul there is no zero-point tensor whose declared type
    # could pick s8/s16/etc., so we don't consult `quantize_target_type/1`.
    [y_name, scale_name, zp_name] = outputs

    x = input!(x_name, axon, params, used_params)

    scale_fun = fn x, _opts ->
      max_x = Nx.max(Nx.reduce_max(x), 0.0)
      min_x = Nx.min(Nx.reduce_min(x), 0.0)
      Nx.divide(Nx.subtract(max_x, min_x), 255.0)
    end

    zp_fun = fn x, _opts ->
      max_x = Nx.max(Nx.reduce_max(x), 0.0)
      min_x = Nx.min(Nx.reduce_min(x), 0.0)
      scale = Nx.divide(Nx.subtract(max_x, min_x), 255.0)
      raw = Nx.divide(Nx.negate(min_x), scale) |> Nx.round() |> Nx.clip(0, 255)
      Nx.as_type(raw, {:u, 8})
    end

    y_fun = fn x, _opts ->
      max_x = Nx.max(Nx.reduce_max(x), 0.0)
      min_x = Nx.min(Nx.reduce_min(x), 0.0)
      scale = Nx.divide(Nx.subtract(max_x, min_x), 255.0)
      raw_zp = Nx.divide(Nx.negate(min_x), scale) |> Nx.round() |> Nx.clip(0, 255)
      Nx.divide(x, scale)
      |> Nx.round()
      |> Nx.add(raw_zp)
      |> Nx.clip(0, 255)
      |> Nx.as_type({:u, 8})
    end

    axon =
      axon
      |> Map.put(y_name, Axon.layer(y_fun, [x], name: y_name, op_name: :dynamic_quantize_y))
      |> Map.put(
        scale_name,
        Axon.layer(scale_fun, [x], name: scale_name, op_name: :dynamic_quantize_scale)
      )
      |> Map.put(
        zp_name,
        Axon.layer(zp_fun, [x], name: zp_name, op_name: :dynamic_quantize_zp)
      )

    {axon, params, used_params}
  end

  defp recur_nodes(
         %Node{
           op_type: "DequantizeLinear",
           attribute: attrs,
           input: inputs,
           output: [output_name]
         },
         {axon, params, used_params}
       ) do
    # DequantizeLinear: y = (x - x_zero_point) * x_scale.
    # For per-channel quantisation (1-D scale of size N), scale/zero_point
    # are reshaped to broadcast along x's `axis` dim. Per-tensor (scalar)
    # quantisation needs no reshape. axis defaults to 1 per ONNX opset 13+.
    axis = options!(attrs)["axis"] || 1

    {x_name, scale_name, zp_name} =
      case inputs do
        [x, s] -> {x, s, nil}
        [x, s, z] -> {x, s, z}
      end

    x = input!(x_name, axon, params, used_params)
    scale = input!(scale_name, axon, params, used_params)

    {fun, layer_inputs} =
      case zp_name do
        nil ->
          {fn x, scale, _opts ->
             {scale, _} = broadcast_q_params(scale, nil, x, axis)
             Nx.multiply(Nx.as_type(x, Nx.type(scale)), scale)
           end, [x, scale]}

        _ ->
          zp = input!(zp_name, axon, params, used_params)

          {fn x, scale, zp, _opts ->
             {scale, zp} = broadcast_q_params(scale, zp, x, axis)
             out_type = Nx.type(scale)

             Nx.subtract(Nx.as_type(x, out_type), Nx.as_type(zp, out_type))
             |> Nx.multiply(scale)
           end, [x, scale, zp]}
      end

    layer = Axon.layer(fun, layer_inputs, name: output_name, op_name: :dequantize_linear)
    updated_axon = Map.put(axon, output_name, layer)
    {updated_axon, params, used_params}
  end

  defp recur_nodes(
         %Node{
           op_type: "QuantizeLinear",
           attribute: attrs,
           input: inputs,
           output: [output_name]
         },
         {axon, params, used_params}
       ) do
    # QuantizeLinear: y = saturate(round(x / y_scale) + y_zero_point) cast
    # to zero_point's type. Per-channel quantisation broadcasts a 1-D
    # scale/zero_point along x's `axis` dim (default 1). The saturate clamp
    # is per dtype-range — Nx.as_type truncates on overflow rather than
    # clipping, so we explicitly clip before casting.
    axis = options!(attrs)["axis"] || 1

    {x_name, scale_name, zp_name} =
      case inputs do
        [x, s] -> {x, s, nil}
        [x, s, z] -> {x, s, z}
      end

    x = input!(x_name, axon, params, used_params)
    scale = input!(scale_name, axon, params, used_params)
    zp = if zp_name, do: input!(zp_name, axon, params, used_params), else: nil

    target_type = quantize_target_type(zp_name, zp)

    {min_v, max_v} = quantize_range(target_type)

    {fun, layer_inputs} =
      case zp do
        nil ->
          {fn x, scale, _opts ->
             {scale, _} = broadcast_q_params(scale, nil, x, axis)
             scaled = Nx.divide(x, scale)
             rounded = Nx.round(scaled)
             clipped = Nx.clip(rounded, min_v, max_v)
             Nx.as_type(clipped, target_type)
           end, [x, scale]}

        _ ->
          {fn x, scale, zp, _opts ->
             {scale, zp} = broadcast_q_params(scale, zp, x, axis)
             work_type = Nx.type(scale)
             scaled = Nx.divide(x, scale)
             rounded = Nx.round(scaled)
             shifted = Nx.add(rounded, Nx.as_type(zp, work_type))
             clipped = Nx.clip(shifted, min_v, max_v)
             Nx.as_type(clipped, target_type)
           end, [x, scale, zp]}
      end

    layer = Axon.layer(fun, layer_inputs, name: output_name, op_name: :quantize_linear)
    updated_axon = Map.put(axon, output_name, layer)
    {updated_axon, params, used_params}
  end

  defp recur_nodes(
         %Node{
           op_type: "NegativeLogLikelihoodLoss",
           attribute: attrs,
           input: inputs,
           output: [output_name]
         },
         {axon, params, used_params}
       ) do
    # NLLLoss(input: (N, C, d1, …, dk), target: (N, d1, …, dk),
    #         weight: (C,) optional) → loss
    # Reduction in {none, sum, mean}; mean is weighted by sum-of-used-weights.
    options = options!(attrs)
    reduction = options["reduction"] || "mean"
    ignore_index = options["ignore_index"]

    {input_name, target_name, weight_name} =
      case inputs do
        [i, t] -> {i, t, nil}
        [i, t, w] -> {i, t, w}
      end

    input = input!(input_name, axon, params, used_params)
    target = input!(target_name, axon, params, used_params)
    weight = if weight_name, do: input!(weight_name, axon, params, used_params), else: nil

    layer_inputs = [input, target] ++ if weight, do: [weight], else: []

    {layer, _} =
      case length(layer_inputs) do
        2 ->
          fun = fn i, t, _opts -> do_nll_loss(i, t, nil, ignore_index, reduction) end
          {Axon.layer(fun, layer_inputs, name: output_name, op_name: :nll_loss), nil}

        3 ->
          fun = fn i, t, w, _opts -> do_nll_loss(i, t, w, ignore_index, reduction) end
          {Axon.layer(fun, layer_inputs, name: output_name, op_name: :nll_loss), nil}
      end

    updated_axon = Map.put(axon, output_name, layer)
    {updated_axon, params, used_params}
  end

  defp recur_nodes(
         %Node{
           op_type: "ScatterElements",
           attribute: attrs,
           input: [data_name, indices_name, updates_name],
           output: [output_name]
         },
         {axon, params, used_params}
       ) do
    # ScatterElements writes updates into data at positions derived by
    # combining the per-element indices with the surrounding coordinates of
    # the indices tensor. reduction in {none, add, mul, max, min}.
    options = options!(attrs)
    axis = options["axis"] || 0
    reduction = options["reduction"] || "none"

    data = input!(data_name, axon, params, used_params)
    indices = input!(indices_name, axon, params, used_params)
    updates = input!(updates_name, axon, params, used_params)

    fun = fn d, i, u, _opts ->
      do_scatter_elements(d, i, u, axis, reduction)
    end

    layer =
      Axon.layer(fun, [data, indices, updates], name: output_name, op_name: :scatter_elements)

    updated_axon = Map.put(axon, output_name, layer)
    {updated_axon, params, used_params}
  end

  defp recur_nodes(
         %Node{
           op_type: "ReverseSequence",
           attribute: attrs,
           input: [data_name, lens_name],
           output: [output_name]
         },
         {axon, params, used_params}
       ) do
    # Per-batch reverse of variable-length sequences along time_axis.
    # `sequence_lens` is a 1-D int tensor of per-batch lengths.
    opts = options!(attrs)
    batch_axis = opts["batch_axis"] || 1
    time_axis = opts["time_axis"] || 0

    data = input!(data_name, axon, params, used_params)
    lens = constant!(lens_name, axon, params, used_params) |> Nx.to_flat_list()

    fun = fn x, opts ->
      ba = opts[:batch_axis]
      ta = opts[:time_axis]
      ls = opts[:lens]
      shape = Nx.shape(x)
      rank = tuple_size(shape)
      t_dim = elem(shape, ta)

      # Permute so batch is axis 0 and time is axis 1.
      x_perm =
        cond do
          ba == 0 and ta == 1 ->
            x

          ba == 1 and ta == 0 ->
            Nx.transpose(x, axes: swap_axes(rank, 0, 1))

          true ->
            raise ArgumentError,
                  "ReverseSequence batch_axis=#{ba} time_axis=#{ta} not supported"
        end

      # Build per-batch time index map of shape {N, T}: reverse 0..l-1
      # and keep the rest.
      per_batch =
        Enum.map(ls, fn l ->
          rev = if l > 0, do: Enum.to_list((l - 1)..0//-1), else: []
          tail = Enum.to_list(l..(t_dim - 1)//1)
          rev ++ tail
        end)

      indices_2d = Nx.tensor(per_batch, type: {:s, 64})

      # Broadcast indices to the full perm shape so take_along_axis can
      # gather along axis 1 (time).
      perm_shape = Nx.shape(x_perm)
      trailing_ones = List.duplicate(1, tuple_size(perm_shape) - 2)
      idx_shape = List.to_tuple([elem(perm_shape, 0), t_dim | trailing_ones])
      indices_b = Nx.reshape(indices_2d, idx_shape) |> Nx.broadcast(perm_shape)

      x_rev = Nx.take_along_axis(x_perm, indices_b, axis: 1)

      if ba == 1 and ta == 0,
        do: Nx.transpose(x_rev, axes: swap_axes(rank, 0, 1)),
        else: x_rev
    end

    layer =
      Axon.layer(fun, [data],
        name: output_name,
        op_name: :reverse_sequence,
        batch_axis: batch_axis,
        time_axis: time_axis,
        lens: lens
      )

    {Map.put(axon, output_name, layer), params, used_params}
  end

  defp recur_nodes(
         %Node{
           op_type: "MaxUnpool",
           attribute: attrs,
           input: inputs,
           output: [output_name]
         },
         {axon, params, used_params}
       ) do
    # MaxUnpool scatters the pooled values back into a zero tensor of
    # the original input shape, using the argmax indices produced by a
    # paired MaxPool. We support the (xT, xI) two-input form; the
    # optional output_shape input is consumed when present.
    opts = options!(attrs)
    kernel_shape = opts["kernel_shape"]
    strides = opts["strides"] || kernel_shape
    pads = opts["pads"] || List.duplicate(0, 2 * length(kernel_shape))

    {xt_name, xi_name, output_shape_name} =
      case inputs do
        [xt, xi] -> {xt, xi, nil}
        [xt, xi, os] -> {xt, xi, os}
      end

    xt = input!(xt_name, axon, params, used_params)
    xi = input!(xi_name, axon, params, used_params)

    output_shape =
      cond do
        output_shape_name && output_shape_name != "" ->
          constant!(output_shape_name, axon, params, used_params)
          |> Nx.to_flat_list()

        true ->
          input_spatial = kernel_shape_from_axon!(xt) |> Tuple.to_list()
          spatial_rank = length(kernel_shape)
          spatial_in = Enum.take(input_spatial, -spatial_rank)
          lo_pads = Enum.take(pads, spatial_rank)
          hi_pads = Enum.drop(pads, spatial_rank)

          leading = Enum.take(input_spatial, -spatial_rank * 0)
          _ = leading

          spatial_out =
            Enum.zip([spatial_in, kernel_shape, strides, lo_pads, hi_pads])
            |> Enum.map(fn {in_dim, k, s, lo, hi} ->
              (in_dim - 1) * s + k - lo - hi
            end)

          batch_channel = Enum.take(input_spatial, length(input_spatial) - spatial_rank)
          batch_channel ++ spatial_out
      end

    fun = fn xt, xi, opts ->
      out_shape = List.to_tuple(opts[:output_shape])
      out_size = Enum.reduce(opts[:output_shape], 1, &Kernel.*/2)
      batch_n = elem(out_shape, 0)
      chan_n = elem(out_shape, 1)
      per_batch_channel = div(out_size, batch_n * chan_n)

      xi64 = Nx.as_type(xi, {:s, 64})

      # The indices in xi are flat positions within each (batch, channel)
      # slice. To scatter into a flat {N*C*...} output we add the
      # per-(batch, channel) offset.
      offsets =
        Nx.iota({batch_n * chan_n}, type: {:s, 64})
        |> Nx.multiply(per_batch_channel)
        |> Nx.reshape(List.to_tuple([batch_n, chan_n | List.duplicate(1, tuple_size(Nx.shape(xi)) - 2)]))

      flat_indices = Nx.add(xi64, offsets) |> Nx.flatten()
      flat_values = Nx.flatten(xt)

      zeros = Nx.broadcast(Nx.tensor(0.0, type: Nx.type(xt)), {out_size})

      zeros
      |> Nx.indexed_put(Nx.new_axis(flat_indices, 1), flat_values)
      |> Nx.reshape(out_shape)
    end

    layer =
      Axon.layer(fun, [xt, xi],
        name: output_name,
        op_name: :max_unpool,
        output_shape: output_shape
      )

    {Map.put(axon, output_name, layer), params, used_params}
  end

  defp recur_nodes(
         %Node{
           op_type: "CenterCropPad",
           attribute: attrs,
           input: [data_name, shape_name],
           output: [output_name]
         },
         {axon, params, used_params}
       ) do
    # CenterCropPad: per-axis center-crop OR center-pad to match the
    # target shape. `axes` (optional) restricts which axes are touched.
    target_shape =
      constant!(shape_name, axon, params, used_params)
      |> Nx.to_flat_list()

    axes_attr = options!(attrs)["axes"]
    data = input!(data_name, axon, params, used_params)

    fun = fn x, opts ->
      target = opts[:target_shape]
      axes = opts[:axes]
      rank = Nx.rank(x)

      pos_axes =
        cond do
          is_nil(axes) -> Enum.to_list(0..(rank - 1)//1)
          true -> Enum.map(axes, fn a -> if a < 0, do: a + rank, else: a end)
        end

      target_map = Enum.zip(pos_axes, target) |> Map.new()

      # First crop each axis if input is bigger than target, then pad
      # any axes where target is bigger than (cropped) input.
      cropped =
        Enum.reduce(pos_axes, x, fn axis, acc ->
          in_dim = Nx.axis_size(acc, axis)
          t = Map.fetch!(target_map, axis)
          cond do
            in_dim > t ->
              start = div(in_dim - t, 2)
              Nx.slice_along_axis(acc, start, t, axis: axis)

            true ->
              acc
          end
        end)

      pad_config =
        Enum.map(0..(rank - 1)//1, fn axis ->
          if Enum.member?(pos_axes, axis) do
            in_dim = Nx.axis_size(cropped, axis)
            t = Map.fetch!(target_map, axis)
            if t > in_dim do
              lo = div(t - in_dim, 2)
              hi = t - in_dim - lo
              {lo, hi, 0}
            else
              {0, 0, 0}
            end
          else
            {0, 0, 0}
          end
        end)

      if Enum.any?(pad_config, fn {lo, hi, _} -> lo > 0 or hi > 0 end) do
        Nx.pad(cropped, Nx.tensor(0, type: Nx.type(cropped)), pad_config)
      else
        cropped
      end
    end

    layer =
      Axon.layer(fun, [data],
        name: output_name,
        op_name: :center_crop_pad,
        target_shape: target_shape,
        axes: axes_attr
      )

    {Map.put(axon, output_name, layer), params, used_params}
  end

  defp recur_nodes(
         %Node{
           op_type: "OneHot",
           attribute: attrs,
           input: [indices_name, depth_name, values_name],
           output: [output_name]
         },
         {axon, params, used_params}
       ) do
    # OneHot: for each position in indices, produce a vector of length
    # `depth` along the `axis` dim where on_value (values[1]) sits at
    # index `indices[i]` and off_value (values[0]) elsewhere. Negative
    # indices count from the end; out-of-range indices stay all
    # off_value.
    axis = options!(attrs)["axis"] || -1
    indices = input!(indices_name, axon, params, used_params)
    depth = constant!(depth_name, axon, params, used_params) |> Nx.to_number() |> trunc()
    values = constant!(values_name, axon, params, used_params)

    fun = fn ind, _opts ->
      out_type = Nx.type(values)
      ind_i = Nx.as_type(ind, {:s, 64})
      d = Nx.tensor(depth, type: {:s, 64})
      # Wrap negative indices.
      ind_normalised = Nx.select(Nx.less(ind_i, 0), Nx.add(ind_i, d), ind_i)

      iota = Nx.iota({depth}, type: {:s, 64})
      ind_shape = Nx.shape(ind_normalised)
      ind_rank = tuple_size(ind_shape)
      pos_axis = if axis < 0, do: ind_rank + axis + 1, else: axis

      # Reshape iota to broadcast along `pos_axis`.
      iota_shape =
        List.to_tuple(
          for i <- 0..ind_rank do
            if i == pos_axis, do: depth, else: 1
          end
        )

      iota_b = Nx.reshape(iota, iota_shape)

      # Reshape indices with a 1 inserted at pos_axis.
      ind_expanded_shape =
        ind_shape
        |> Tuple.to_list()
        |> List.insert_at(pos_axis, 1)
        |> List.to_tuple()

      ind_b = Nx.reshape(ind_normalised, ind_expanded_shape)

      mask = Nx.equal(ind_b, iota_b)
      off_value = values[[0]] |> Nx.as_type(out_type)
      on_value = values[[1]] |> Nx.as_type(out_type)
      Nx.select(mask, on_value, off_value)
    end

    layer =
      case get_axon_node(indices) do
        %Axon.Node{op: :constant, opts: [value: v]} ->
          Axon.constant(fun.(v, []), name: output_name)

        %Axon.Node{} ->
          Axon.layer(fun, [indices], name: output_name, op_name: :one_hot)

        %Nx.Tensor{} = t ->
          Axon.constant(fun.(t, []), name: output_name)
      end

    {Map.put(axon, output_name, layer), params, used_params}
  end

  defp recur_nodes(
         %Node{
           op_type: "RotaryEmbedding",
           attribute: attrs,
           input: inputs,
           output: [output_name]
         },
         {axon, params, used_params}
       ) do
    # RotaryEmbedding: rotate (x1, x2) pairs along the head axis using
    # per-position cos/sin caches. Inputs:
    #   * input: {batch, num_heads, seq_len, head_size}  (4D)
    #     or       {batch, seq_len, num_heads * head_size} (3D)
    #   * cos_cache, sin_cache: {max_seq_len, rotary_dim / 2}
    #   * position_ids: {batch, seq_len} or {} (optional)
    opts = options!(attrs)
    interleaved = (opts["interleaved"] || 0) == 1
    rotary_embedding_dim = opts["rotary_embedding_dim"] || 0
    num_heads = opts["num_heads"] || 0

    {input_name, cos_name, sin_name, position_name} =
      case inputs do
        [i, c, s] -> {i, c, s, nil}
        [i, c, s, p] -> {i, c, s, (p != "" && p) || nil}
      end

    input = input!(input_name, axon, params, used_params)
    cos_cache = input!(cos_name, axon, params, used_params)
    sin_cache = input!(sin_name, axon, params, used_params)

    position_ids =
      if position_name do
        input!(position_name, axon, params, used_params)
      end

    fun =
      case position_ids do
        nil ->
          fn x, cos_c, sin_c, _opts ->
            do_rotary_embedding(x, cos_c, sin_c, nil,
              interleaved: interleaved,
              rotary_dim: rotary_embedding_dim,
              num_heads: num_heads
            )
          end

        _ ->
          fn x, cos_c, sin_c, pos, _opts ->
            do_rotary_embedding(x, cos_c, sin_c, pos,
              interleaved: interleaved,
              rotary_dim: rotary_embedding_dim,
              num_heads: num_heads
            )
          end
      end

    layer_inputs =
      [input, cos_cache, sin_cache] ++ (if position_ids, do: [position_ids], else: [])

    layer = Axon.layer(fun, layer_inputs, name: output_name, op_name: :rotary_embedding)

    {Map.put(axon, output_name, layer), params, used_params}
  end

  defp recur_nodes(
         %Node{op_type: "Einsum", attribute: attrs, input: inputs, output: [output_name]},
         {axon, params, used_params}
       ) do
    # Einsum: equation is "lhs -> rhs" where lhs is comma-separated
    # input subscripts. We support single-input reductions/transposes
    # and two-input contractions. Ellipsis "..." for trailing batch
    # dims is recognised.
    equation = options!(attrs)["equation"]

    [lhs, rhs] =
      case String.split(equation, "->", parts: 2) do
        [l, r] -> [String.replace(l, " ", ""), String.replace(r, " ", "")]
        [l] -> [String.replace(l, " ", ""), nil]
      end

    lhs_specs = String.split(lhs, ",")
    input_tensors = Enum.map(inputs, &input!(&1, axon, params, used_params))

    fun =
      case {length(input_tensors), lhs_specs} do
        {1, [in_spec]} ->
          fn x, _opts ->
            do_einsum_1(x, in_spec, rhs)
          end

        {2, [a_spec, b_spec]} ->
          fn a, b, _opts ->
            do_einsum_2(a, b, a_spec, b_spec, rhs)
          end

        _ ->
          raise ArgumentError,
                "Einsum with #{length(input_tensors)} inputs is not yet supported"
      end

    layer =
      case input_tensors do
        [single] ->
          case get_axon_node(single) do
            %Axon.Node{op: :constant, opts: [value: v]} ->
              Axon.constant(fun.(v, []), name: output_name)

            %Nx.Tensor{} = t ->
              Axon.constant(fun.(t, []), name: output_name)

            %Axon.Node{} ->
              Axon.layer(fun, [single], name: output_name, op_name: :einsum)
          end

        [_a, _b] = both ->
          Axon.layer(fun, both, name: output_name, op_name: :einsum)
      end

    {Map.put(axon, output_name, layer), params, used_params}
  end

  defp recur_nodes(
         %Node{op_type: "Resize", attribute: attrs, input: inputs, output: [output_name]},
         {axon, params, used_params}
       ) do
    # Resize: spatial-axis upsample/downsample. We support the modes
    # exercised by the corpus's nearest and linear tests:
    #   * mode: "nearest" (default) and "linear"
    #   * coordinate_transformation_mode: half_pixel (default),
    #     asymmetric, align_corners, pytorch_half_pixel,
    #     half_pixel_symmetric
    #   * nearest_mode: round_prefer_floor (default), round_prefer_ceil,
    #     floor, ceil
    #
    # roi/extrapolation/antialias/cubic stay unsupported and raise so
    # those test cases surface clean errors and remain :unsupported.
    opts = options!(attrs)
    mode = opts["mode"] || "nearest"
    ctm = opts["coordinate_transformation_mode"] || "half_pixel"
    nearest_mode = opts["nearest_mode"] || "round_prefer_floor"
    keep_aspect = opts["keep_aspect_ratio_policy"] || "stretch"
    axes_attr = opts["axes"]
    antialias = opts["antialias"] || 0
    exclude_outside = opts["exclude_outside"] || 0

    if antialias == 1 do
      raise ArgumentError, "Resize antialias=1 is not yet supported"
    end

    if exclude_outside == 1 do
      raise ArgumentError, "Resize exclude_outside=1 is not yet supported"
    end

    if mode not in ["nearest", "linear"] do
      raise ArgumentError, "Resize mode=#{inspect(mode)} is not yet supported"
    end

    if ctm in ["tf_crop_and_resize"] do
      raise ArgumentError, "Resize tf_crop_and_resize is not yet supported"
    end

    {x_name, scales_name, sizes_name} =
      case inputs do
        [x] ->
          {x, nil, nil}

        [x, _roi] ->
          {x, nil, nil}

        [x, _roi, scales] ->
          {x, (scales != "" && scales) || nil, nil}

        [x, _roi, scales, sizes] ->
          {x, (scales != "" && scales) || nil, (sizes != "" && sizes) || nil}
      end

    x = input!(x_name, axon, params, used_params)

    scales =
      if scales_name do
        constant!(scales_name, axon, params, used_params) |> Nx.to_flat_list()
      end

    sizes =
      if sizes_name do
        constant!(sizes_name, axon, params, used_params) |> Nx.to_flat_list()
      end

    input_shape = kernel_shape_from_axon!(x) |> Tuple.to_list()
    rank = length(input_shape)

    axes =
      cond do
        is_nil(axes_attr) -> Enum.to_list(0..(rank - 1)//1)
        true -> Enum.map(axes_attr, fn a -> if a < 0, do: a + rank, else: a end)
      end

    {out_shape, per_axis_scales} =
      cond do
        sizes ->
          sizes_for_axes =
            cond do
              is_nil(axes_attr) -> sizes
              true -> sizes
            end

          new_dims =
            Enum.zip(axes, sizes_for_axes)
            |> resize_apply_sizes(input_shape, axes, keep_aspect)

          scales_list =
            Enum.map(0..(rank - 1)//1, fn ax ->
              new = Enum.at(new_dims, ax)
              old = Enum.at(input_shape, ax)
              new / old
            end)

          {new_dims, scales_list}

        scales ->
          scales_for_axes = scales

          new_dims =
            Enum.map(0..(rank - 1)//1, fn ax ->
              case Enum.find_index(axes, &(&1 == ax)) do
                nil ->
                  Enum.at(input_shape, ax)

                idx ->
                  s = Enum.at(scales_for_axes, idx)
                  trunc(Enum.at(input_shape, ax) * s)
              end
            end)

          full_scales =
            Enum.map(0..(rank - 1)//1, fn ax ->
              case Enum.find_index(axes, &(&1 == ax)) do
                nil -> 1.0
                idx -> Enum.at(scales_for_axes, idx)
              end
            end)

          {new_dims, full_scales}

        true ->
          raise ArgumentError, "Resize requires either scales or sizes"
      end

    # Pre-compute per-axis index arrays mapping each output position to
    # a (clamped) input position. This is independent of x's runtime
    # data so we can do it at build time.
    nearest_indices =
      if mode == "nearest" do
        Enum.map(0..(rank - 1)//1, fn ax ->
          in_dim = Enum.at(input_shape, ax)
          out_dim = Enum.at(out_shape, ax)
          s = Enum.at(per_axis_scales, ax)

          for out_i <- 0..(out_dim - 1) do
            in_f = resize_coord_transform(out_i, in_dim, out_dim, s, ctm)
            in_i = resize_nearest_round(in_f, nearest_mode)
            in_i |> max(0) |> min(in_dim - 1)
          end
        end)
      end

    layer_fun =
      case mode do
        "nearest" ->
          fn x, _opts ->
            Enum.with_index(nearest_indices)
            |> Enum.reduce(x, fn {idxs, ax}, acc ->
              if idxs == Enum.to_list(0..(Enum.at(input_shape, ax) - 1)//1) and
                   length(idxs) == Enum.at(out_shape, ax) do
                acc
              else
                Nx.take(acc, Nx.tensor(idxs, type: {:s, 64}), axis: ax)
              end
            end)
          end

        "linear" ->
          fn x, _opts ->
            resize_linear_apply(x, input_shape, out_shape, per_axis_scales, ctm)
          end
      end

    layer =
      case get_axon_node(x) do
        %Axon.Node{op: :constant, opts: [value: v]} ->
          Axon.constant(layer_fun.(v, []), name: output_name)

        %Axon.Node{} ->
          Axon.layer(layer_fun, [x], name: output_name, op_name: :resize)

        %Nx.Tensor{} = t ->
          Axon.constant(layer_fun.(t, []), name: output_name)
      end

    {Map.put(axon, output_name, layer), params, used_params}
  end

  defp recur_nodes(
         %Node{op_type: "SpaceToDepth", attribute: attrs, input: [input_name], output: [output_name]},
         {axon, params, used_params}
       ) do
    # SpaceToDepth: {N, C, H, W} → {N, C*B*B, H/B, W/B} via
    # reshape → transpose → reshape.
    b = options!(attrs)["blocksize"]
    input = input!(input_name, axon, params, used_params)

    fun = fn x, opts ->
      block = opts[:blocksize]
      {n, c, h, w} = Nx.shape(x)
      x
      |> Nx.reshape({n, c, div(h, block), block, div(w, block), block})
      |> Nx.transpose(axes: [0, 3, 5, 1, 2, 4])
      |> Nx.reshape({n, c * block * block, div(h, block), div(w, block)})
    end

    apply_fun = &fun.(&1, blocksize: b)

    layer =
      case get_axon_node(input) do
        %Axon.Node{op: :constant, opts: [value: v]} ->
          Axon.constant(apply_fun.(v), name: output_name)

        %Axon.Node{} ->
          Axon.layer(fun, [input], name: output_name, op_name: :space_to_depth, blocksize: b)

        %Nx.Tensor{} = t ->
          Axon.constant(apply_fun.(t), name: output_name)
      end

    {Map.put(axon, output_name, layer), params, used_params}
  end

  defp recur_nodes(
         %Node{op_type: "DepthToSpace", attribute: attrs, input: [input_name], output: [output_name]},
         {axon, params, used_params}
       ) do
    # DepthToSpace: inverse of SpaceToDepth. `mode` controls how the
    # depth axis is interpreted before reshape: DCR (default) groups
    # block-row × block-col × channel; CRD groups channel × block-row ×
    # block-col. (Note: we keep this option under `:block_mode` since
    # `Axon.layer/3` claims `:mode` for inference/train selection.)
    opts = options!(attrs)
    b = opts["blocksize"]
    mode = opts["mode"] || "DCR"
    input = input!(input_name, axon, params, used_params)

    fun = fn x, opts ->
      block = opts[:blocksize]
      m = opts[:block_mode]
      {n, c, h, w} = Nx.shape(x)
      c_out = div(c, block * block)

      {reshape_a, transpose_axes} =
        case m do
          "DCR" -> {{n, block, block, c_out, h, w}, [0, 3, 4, 1, 5, 2]}
          "CRD" -> {{n, c_out, block, block, h, w}, [0, 1, 4, 2, 5, 3]}
        end

      x
      |> Nx.reshape(reshape_a)
      |> Nx.transpose(axes: transpose_axes)
      |> Nx.reshape({n, c_out, h * block, w * block})
    end

    apply_fun = &fun.(&1, blocksize: b, block_mode: mode)

    layer =
      case get_axon_node(input) do
        %Axon.Node{op: :constant, opts: [value: v]} ->
          Axon.constant(apply_fun.(v), name: output_name)

        %Axon.Node{} ->
          Axon.layer(fun, [input],
            name: output_name,
            op_name: :depth_to_space,
            blocksize: b,
            block_mode: mode
          )

        %Nx.Tensor{} = t ->
          Axon.constant(apply_fun.(t), name: output_name)
      end

    {Map.put(axon, output_name, layer), params, used_params}
  end

  defp recur_nodes(
         %Node{
           op_type: "Compress",
           attribute: attrs,
           input: [data_name, condition_name],
           output: [output_name]
         },
         {axon, params, used_params}
       ) do
    # Compress: select elements where `condition` is truthy. Without
    # `axis`, the input is flattened first and the result is 1-D. The
    # condition is required as a constant; with fold_inputs it can come
    # from a graph-input bound to test data.
    axis = options!(attrs)["axis"]
    data = input!(data_name, axon, params, used_params)
    condition = constant!(condition_name, axon, params, used_params)

    indices =
      condition
      |> Nx.to_flat_list()
      |> Enum.with_index()
      |> Enum.filter(fn {c, _} -> c != 0 end)
      |> Enum.map(fn {_, i} -> i end)

    fun = fn x, opts ->
      idxs = Nx.tensor(opts[:indices], type: {:s, 64})
      x_to_use = if opts[:axis] == nil, do: Nx.flatten(x), else: x
      a = if opts[:axis] == nil, do: 0, else: opts[:axis]
      Nx.take(x_to_use, idxs, axis: a)
    end

    apply_fun = fn t -> fun.(t, axis: axis, indices: indices) end

    layer =
      case get_axon_node(data) do
        %Axon.Node{op: :constant, opts: [value: v]} ->
          Axon.constant(apply_fun.(v), name: output_name)

        %Nx.Tensor{} = t ->
          Axon.constant(apply_fun.(t), name: output_name)

        %Axon.Node{} ->
          Axon.layer(fun, [data], name: output_name, op_name: :compress,
            axis: axis, indices: indices)
      end

    {Map.put(axon, output_name, layer), params, used_params}
  end

  defp recur_nodes(
         %Node{
           op_type: "GatherND",
           attribute: attrs,
           input: [data_name, indices_name],
           output: [output_name]
         },
         {axon, params, used_params}
       ) do
    # GatherND: indices is `[..., q]`, returns data elements at the
    # addressed q-tuples. `batch_dims` (default 0) treats leading axes as
    # batches that gather independently.
    batch_dims = options!(attrs)["batch_dims"] || 0
    data = input!(data_name, axon, params, used_params)
    indices = input!(indices_name, axon, params, used_params)

    fun = fn d, i, _opts ->
      do_gather_nd(d, Nx.as_type(i, {:s, 64}), batch_dims)
    end

    layer = Axon.layer(fun, [data, indices], name: output_name, op_name: :gather_nd)
    {Map.put(axon, output_name, layer), params, used_params}
  end

  defp recur_nodes(
         %Node{
           op_type: "Unique",
           attribute: attrs,
           input: [data_name],
           output: outputs
         },
         {axon, params, used_params}
       ) do
    # Unique returns up to four tensors: values, indices into input,
    # inverse mapping, counts. With an axis, dedupe along that axis;
    # without, dedupe flat. `sorted=1` (default) sorts by value
    # ascending.
    opts = options!(attrs)
    axis = opts["axis"]
    sorted = (opts["sorted"] || 1) == 1

    if axis do
      raise ArgumentError, "Unique with axis attribute is not yet supported"
    end

    data = constant!(data_name, axon, params, used_params)
    flat = Nx.to_flat_list(data)

    {values, idx, inverse, counts} = unique_with_layouts(flat, sorted)
    type = Nx.type(data)

    materialised =
      outputs
      |> Enum.with_index()
      |> Enum.map(fn {name, i} ->
        tensor =
          case i do
            0 -> Nx.tensor(values, type: type)
            1 -> Nx.tensor(idx, type: {:s, 64})
            2 -> Nx.tensor(inverse, type: {:s, 64})
            3 -> Nx.tensor(counts, type: {:s, 64})
          end

        {name, Axon.constant(tensor, name: name)}
      end)

    updated_axon =
      Enum.reduce(materialised, axon, fn {name, layer}, acc -> Map.put(acc, name, layer) end)

    {updated_axon, params, used_params}
  end

  defp recur_nodes(
         %Node{
           op_type: "ScatterND",
           attribute: attrs,
           input: [data_name, indices_name, updates_name],
           output: [output_name]
         },
         {axon, params, used_params}
       ) do
    # ScatterND: indices is shape `[..., q]` where q <= rank(data); each
    # row addresses a slice of data of shape `data.shape[q:]`. Reductions
    # match ScatterElements: none/add/mul/max/min.
    reduction = options!(attrs)["reduction"] || "none"
    data = input!(data_name, axon, params, used_params)
    indices = input!(indices_name, axon, params, used_params)
    updates = input!(updates_name, axon, params, used_params)

    fun = fn d, i, u, _opts -> do_scatter_nd(d, i, u, reduction) end

    layer =
      Axon.layer(fun, [data, indices, updates], name: output_name, op_name: :scatter_nd)

    {Map.put(axon, output_name, layer), params, used_params}
  end

  defp recur_nodes(
         %Node{op_type: "Shrink", attribute: attrs, input: [input_name], output: [output_name]},
         {axon, params, used_params}
       ) do
    # Shrink(x): if x < -lambd → x + bias ; if x > lambd → x - bias ; else 0.
    # Defaults: bias=0, lambd=0.5.
    options = options!(attrs)
    bias = options["bias"] || 0.0
    lambd = options["lambd"] || 0.5

    input = input!(input_name, axon, params, used_params)

    fun = fn x, _opts ->
      pos = Nx.greater(x, lambd)
      neg = Nx.less(x, -lambd)
      zero = Nx.tensor(0, type: Nx.type(x))

      Nx.select(
        pos,
        Nx.subtract(x, bias),
        Nx.select(neg, Nx.add(x, bias), Nx.broadcast(zero, Nx.shape(x)))
      )
    end

    layer = Axon.layer(fun, [input], name: output_name, op_name: :shrink)
    updated_axon = Map.put(axon, output_name, layer)
    {updated_axon, params, used_params}
  end

  defp recur_nodes(
         %Node{op_type: "LpPool", attribute: attrs, input: [input_name], output: [output_name]},
         {axon, params, used_params}
       ) do
    # LpPool: pooling using L_p norm — sum(|x|^p)^(1/p) over the window.
    # Axon.Layers.lp_pool provides this directly; we map ONNX attributes.
    options = options!(attrs)
    kernel_shape = options["kernel_shape"] |> List.to_tuple()
    p = options["p"] || 2
    auto_pad = options["auto_pad"] || "NOTSET"
    pads = options["pads"]
    strides = options["strides"] || List.duplicate(1, tuple_size(kernel_shape))

    padding_config = padding!(auto_pad, pads, kernel_shape, strides)

    input = input!(input_name, axon, params, used_params)

    fun = fn x, _opts ->
      rank = Nx.rank(x)
      spatial_rank = tuple_size(kernel_shape)
      leading = rank - spatial_rank
      full_kernel = List.duplicate(1, leading) ++ Tuple.to_list(kernel_shape)

      full_strides =
        cond do
          is_integer(strides) -> List.duplicate(1, leading) ++ List.duplicate(strides, spatial_rank)
          is_list(strides) -> List.duplicate(1, leading) ++ strides
        end

      full_padding =
        case padding_config do
          atom when is_atom(atom) -> atom
          list when is_list(list) -> List.duplicate({0, 0}, leading) ++ list
        end

      # ONNX LpPool: (Σ |x|^p)^(1/p) over the window. Axon.Layers.lp_pool
      # uses `Nx.pow(input, p)` without `abs`, which yields NaN for any
      # negative input when `p` is non-integer in defn (it's lowered via
      # log/exp on the negative branch).
      x_abs = Nx.abs(x)

      summed =
        x_abs
        |> Nx.pow(p)
        |> Nx.window_sum(List.to_tuple(full_kernel),
          strides: full_strides,
          padding: full_padding
        )

      Nx.pow(summed, Nx.divide(Nx.tensor(1, type: Nx.type(x)), p))
    end

    layer = Axon.layer(fun, [input], name: output_name, op_name: :lp_pool)
    updated_axon = Map.put(axon, output_name, layer)
    {updated_axon, params, used_params}
  end

  defp recur_nodes(
         %Node{
           op_type: "Hardmax",
           attribute: attrs,
           input: [input_name],
           output: [output_name]
         },
         {axon, params, used_params}
       ) do
    # Hardmax: 1.0 at the argmax position along axis, 0.0 elsewhere. The
    # output has the same dtype as the input. axis is lifted into opts
    # so the serializer can round-trip the attribute.
    axis = options!(attrs)["axis"] || -1
    input = input!(input_name, axon, params, used_params)

    fun = fn x, opts ->
      ax = opts[:axis]
      argmax = Nx.argmax(x, axis: ax, keep_axis: true)
      iota = Nx.iota(Nx.shape(x), axis: ax)
      mask = Nx.equal(iota, argmax)
      Nx.as_type(mask, Nx.type(x))
    end

    layer = Axon.layer(fun, [input], name: output_name, op_name: :hardmax, axis: axis)
    updated_axon = Map.put(axon, output_name, layer)
    {updated_axon, params, used_params}
  end

  defp recur_nodes(
         %Node{
           op_type: "GatherElements",
           attribute: attrs,
           input: [data_name, indices_name],
           output: [output_name]
         },
         {axon, params, used_params}
       ) do
    # GatherElements is per-index take along a single axis. Indices must
    # have the same rank as data and the output adopts the indices' shape.
    # Maps cleanly to Nx.take_along_axis once indices are cast to int64.
    axis = options!(attrs)["axis"] || 0
    data = input!(data_name, axon, params, used_params)
    indices = input!(indices_name, axon, params, used_params)

    fun = fn d, i, _opts ->
      i = Nx.as_type(i, {:s, 64})
      dim = Nx.axis_size(d, axis)
      i = Nx.select(Nx.less(i, 0), Nx.add(i, dim), i)
      Nx.take_along_axis(d, i, axis: axis)
    end

    layer = Axon.layer(fun, [data, indices], name: output_name, op_name: :gather_elements)
    updated_axon = Map.put(axon, output_name, layer)
    {updated_axon, params, used_params}
  end

  defp recur_nodes(
         %Node{
           op_type: "RMSNormalization",
           attribute: attrs,
           input: [input_name, scale_name],
           output: [output_name]
         },
         {axon, params, used_params}
       ) do
    # RMSNormalization (opset 23+): Y = (X * rsqrt(mean(X^2, axes) + epsilon)) * Scale.
    # No mean-centering, no bias — strictly the LLM-favoured normalisation.
    options = options!(attrs)
    axis = options["axis"] || -1
    epsilon = options["epsilon"] || 1.0e-5

    input = input!(input_name, axon, params, used_params)
    scale = input!(scale_name, axon, params, used_params)

    fun = fn x, s, _opts ->
      rank = Nx.rank(x)
      pos_axis = if axis < 0, do: rank + axis, else: axis
      axes = Enum.to_list(pos_axis..(rank - 1)//1)
      mean_sq = Nx.mean(Nx.pow(x, 2), axes: axes, keep_axes: true)
      inv_rms = Nx.rsqrt(Nx.add(mean_sq, epsilon))
      Nx.multiply(Nx.multiply(x, inv_rms), s)
    end

    layer = Axon.layer(fun, [input, scale], name: output_name, op_name: :rms_norm)
    updated_axon = Map.put(axon, output_name, layer)
    {updated_axon, params, used_params}
  end

  defp recur_nodes(
         %Node{op_type: "Concat", attribute: attrs, input: inputs, output: [output_name]},
         {axon, params, used_params}
       ) do
    inputs = for inp <- inputs, do: input!(inp, axon, params, used_params)
    %{"axis" => axis} = options!(attrs)

    updated_axon =
      if Enum.all?(inputs, &constant?(get_axon_node(&1))) do
        vals = Enum.map(inputs, &get_value(get_axon_node(&1)))
        new_value = Nx.concatenate(vals, axis: axis)
        Map.put(axon, output_name, Axon.constant(new_value, name: output_name))
      else
        Map.put(axon, output_name, Axon.concatenate(inputs, axis: axis, name: output_name))
      end

    {updated_axon, params, used_params}
  end

  defp recur_nodes(
         %Node{op_type: "Split", attribute: attrs, input: [inp], output: output_names},
         {axon, params, used_params}
       ) do
    inp = axon!(inp, axon)
    opts = options!(attrs)
    axis = opts["axis"] || 0

    split_sizes =
      cond do
        opts["split"] ->
          opts["split"]

        opts["num_outputs"] ->
          # Opset 18+: split into N parts. ONNX distributes the remainder
          # one-per-output starting from the first — sizes are
          # `ceil(d/N)` for the first `d mod N` outputs and `floor(d/N)`
          # for the rest, so the last partition is the smallest (or
          # empty when N > d).
          n = opts["num_outputs"]

          dim_size =
            case kernel_shape_from_axon!(inp) do
              shape when is_tuple(shape) ->
                pos_axis = if axis < 0, do: tuple_size(shape) + axis, else: axis
                elem(shape, pos_axis)
            end

          base = div(dim_size, n)
          remainder = rem(dim_size, n)

          for i <- 0..(n - 1) do
            if i < remainder, do: base + 1, else: base
          end

        true ->
          # Equal split by output arity — same distribution rule as
          # num_outputs above.
          n = length(output_names)

          dim_size =
            case kernel_shape_from_axon!(inp) do
              shape when is_tuple(shape) ->
                pos_axis = if axis < 0, do: tuple_size(shape) + axis, else: axis
                elem(shape, pos_axis)
            end

          base = div(dim_size, n)
          remainder = rem(dim_size, n)

          for i <- 0..(n - 1) do
            if i < remainder, do: base + 1, else: base
          end
      end

    updated_axon = build_split_layers(inp, axon, split_sizes, axis, output_names)
    {updated_axon, params, used_params}
  end

  # Manual split via Nx.slice_along_axis, since Axon.Layers.split requires
  # rank ≥ 2 and rejects 1-D inputs that ONNX Split happily handles.
  defp build_split_layers(inp, axon, split_sizes, axis, output_names) do
    {_, layers} =
      Enum.reduce(Enum.zip(split_sizes, output_names), {0, []}, fn {size, name}, {offset, acc} ->
        layer =
          Axon.nx(
            inp,
            fn x -> Nx.slice_along_axis(x, offset, size, axis: axis) end,
            name: name,
            op_name: :split
          )

        {offset + size, [{name, layer} | acc]}
      end)

    Enum.reduce(Enum.reverse(layers), axon, fn {name, layer}, acc ->
      Map.put(acc, name, layer)
    end)
  end

  defp recur_nodes(
         %Node{op_type: "Constant", attribute: attrs, output: [output_name]},
         {axon, params, used_params}
       ) do
    constant_options = options!(attrs)

    const =
      cond do
        constant_options["sparse_value"] ->
          raise ArgumentError, "sparse tensors are not supported"

        constant_options["value"] ->
          Axon.constant(tensor!(constant_options["value"]), name: output_name)

        constant_options["value_float"] ->
          Axon.constant(
            Nx.tensor(normalize_special_float(constant_options["value_float"]), type: {:f, 32}),
            name: output_name
          )

        constant_options["value_floats"] ->
          Axon.constant(
            Nx.tensor(Enum.map(constant_options["value_floats"], &normalize_special_float/1),
              type: {:f, 32}
            ),
            name: output_name
          )

        constant_options["value_int"] ->
          Axon.constant(Nx.tensor(constant_options["value_int"], type: {:s, 64}),
            name: output_name
          )

        constant_options["value_ints"] ->
          Axon.constant(Nx.tensor(constant_options["value_ints"], type: {:s, 64}),
            name: output_name
          )

        constant_options["value_string"] or constant_options["value_strings"] ->
          raise ArgumentError, "string tensors are not supported"

        true ->
          raise ArgumentError, "invalid constant tensor type"
      end

    updated_axon = Map.put(axon, output_name, const)

    {updated_axon, params, used_params}
  end

  defp recur_nodes(
         %Node{
           op_type: "ConstantOfShape",
           attribute: attrs,
           input: [shape],
           output: [output_name]
         },
         {axon, params, used_params}
       ) do
    constant_options = options!(attrs)

    # Per spec, `value` defaults to a single f32 zero when omitted.
    value =
      case constant_options["value"] do
        nil -> Nx.tensor(0.0, type: {:f, 32})
        t -> tensor!(t)
      end

    shape =
      cond do
        constant_resolvable?(shape, axon, params, used_params) ->
          shape
          |> constant!(axon, params, used_params)
          |> Nx.to_flat_list()
          |> Enum.map(fn
            -1 -> 1
            x -> x
          end)
          |> List.to_tuple()

        out_shape = output_shape(output_name) ->
          # Shape input is a runtime graph input, but the model declares the
          # output shape — use that to construct the constant.
          out_shape

        true ->
          raise ArgumentError,
                "ConstantOfShape needs either a constant shape input " <>
                  "or a statically-declared output shape; neither is available " <>
                  "for output #{inspect(output_name)}."
      end

    val = Nx.broadcast(value, shape)

    updated_axon = Map.put(axon, output_name, Axon.constant(val, name: output_name))
    {updated_axon, params, used_params}
  end

  defp recur_nodes(
         %Node{op_type: "Reshape", input: [inp, shape], attribute: attrs, output: [output_name]},
         {axon, params, used_params}
       ) do
    reshape_options = options!(attrs)

    allowzero = reshape_options["allowzero"] || 0
    inp = axon!(inp, axon)

    # We currently do not support zero sized dimensions
    if allowzero == 1 do
      Logger.warning(
        "Nx does not support zero-sized dimensions. If your reshape" <>
          " operation contains a zero-sized dimension, it will fail"
      )
    end

    new_shape =
      cond do
        constant_resolvable?(shape, axon, params, used_params) ->
          shape
          |> constant!(axon, params, used_params)
          |> Nx.to_flat_list()
          |> Enum.reduce({[], false}, fn
            0, {cur_shape, already_auto?} -> {cur_shape, already_auto?}
            -1, {cur_shape, false} -> {[:auto | cur_shape], true}
            -1, {cur_shape, true} -> {[1 | cur_shape], true}
            x, {cur_shape, already_auto?} -> {[x | cur_shape], already_auto?}
          end)
          |> elem(0)
          |> Enum.reverse()
          |> List.to_tuple()

        out_shape = output_shape(output_name) ->
          # Shape input is runtime, but the model declares the output shape.
          out_shape

        true ->
          raise ArgumentError,
                "Reshape requires either a constant shape input or a " <>
                  "statically-declared output shape; got neither for " <>
                  "#{inspect(output_name)}."
      end

    updated_axon =
      case get_axon_node(inp) do
        %Axon.Node{op: :constant, opts: [value: v]} ->
          new_value = Nx.reshape(v, new_shape)
          Map.put(axon, output_name, Axon.constant(new_value, name: output_name))

        %Axon.Node{} ->
          Map.put(
            axon,
            output_name,
            Axon.reshape(inp, new_shape, name: output_name)
          )
      end

    {updated_axon, params, used_params}
  end

  defp recur_nodes(
         %Node{op_type: "Expand", input: [inp, shape], output: [output_name]},
         {axon, params, used_params}
       ) do
    inp = input!(inp, axon, params, used_params)

    shape =
      cond do
        constant_resolvable?(shape, axon, params, used_params) ->
          shape
          |> constant!(axon, params, used_params)
          |> Nx.to_flat_list()
          |> Enum.map(fn
            -1 -> 1
            x -> x
          end)
          |> List.to_tuple()

        out_shape = output_shape(output_name) ->
          out_shape

        true ->
          raise ArgumentError,
                "Expand requires either a constant shape input or a " <>
                  "statically-declared output shape; got neither for " <>
                  "#{inspect(output_name)}."
      end

    updated_axon =
      case get_axon_node(inp) do
        %Axon.Node{op: :constant, opts: [value: v]} ->
          new_value = Nx.multiply(v, Nx.broadcast(1, shape))
          layer = Axon.constant(new_value, name: output_name)
          updated_axon = Map.put(axon, output_name, layer)
          updated_axon

        %Nx.Tensor{} = x ->
          new_value = Nx.multiply(x, Nx.broadcast(1, shape))
          layer = Axon.constant(new_value, name: output_name)
          updated_axon = Map.put(axon, output_name, layer)
          updated_axon

        %Axon.Node{} ->
          fun = fn x, _opts -> Nx.multiply(x, Nx.broadcast(1, shape)) end
          layer = Axon.layer(fun, [inp], name: output_name, op_name: :expand)
          updated_axon = Map.put(axon, output_name, layer)
          updated_axon
      end

    {updated_axon, params, used_params}
  end

  defp recur_nodes(
         %Node{op_type: "Range", input: [start, limit, delta], output: [output_name]},
         {axon, params, used_params}
       ) do
    start_t = constant!(start, axon, params, used_params)
    limit_t = constant!(limit, axon, params, used_params)
    delta_t = constant!(delta, axon, params, used_params)

    type = Nx.type(start_t)
    start = Nx.to_number(start_t)
    limit = Nx.to_number(limit_t)
    delta = Nx.to_number(delta_t)

    number_of_elements = max(ceil((limit - start) / delta), 0)

    vals =
      if number_of_elements == 0 do
        []
      else
        for i <- 0..(number_of_elements - 1), do: start + i * delta
      end

    tensor =
      if vals == [] do
        Nx.tensor([], type: type) |> Nx.reshape({0})
      else
        Nx.tensor(vals, type: type)
      end

    updated_axon = Map.put(axon, output_name, Axon.constant(tensor, name: output_name))
    {updated_axon, params, used_params}
  end

  defp recur_nodes(
         %Node{op_type: "Flatten", attribute: attrs, input: [inp], output: [output_name]},
         {axon, params, used_params}
       ) do
    axis = options!(attrs)["axis"] || 1
    input = input!(inp, axon, params, used_params)

    fun = fn x, opts ->
      ax = opts[:axis]
      shape = Nx.shape(x)
      rank = tuple_size(shape)
      pos_axis = if ax < 0, do: rank + ax, else: ax
      dims = Tuple.to_list(shape)
      {prefix, suffix} = Enum.split(dims, pos_axis)
      lead = Enum.reduce(prefix, 1, &Kernel.*/2)
      trail = Enum.reduce(suffix, 1, &Kernel.*/2)
      Nx.reshape(x, {lead, trail})
    end

    output =
      case get_axon_node(input) do
        %Axon.Node{op: :constant, opts: [value: v]} ->
          Axon.constant(fun.(v, axis: axis), name: output_name)

        %Axon.Node{} ->
          Axon.layer(fun, [input], name: output_name, op_name: :flatten, axis: axis)

        %Nx.Tensor{} = t ->
          Axon.constant(fun.(t, axis: axis), name: output_name)
      end

    {Map.put(axon, output_name, output), params, used_params}
  end

  defp recur_nodes(
         %Node{op_type: "TopK", attribute: attrs, input: [x_name, k_name], output: outputs},
         {axon, params, used_params}
       ) do
    # TopK(X, K) → (values, indices). K is a 1-element tensor. axis defaults
    # to -1; largest defaults to 1; sorted defaults to 1.
    options = options!(attrs)
    axis = options["axis"] || -1
    largest = (options["largest"] || 1) == 1
    [values_name, indices_name] = outputs

    x = input!(x_name, axon, params, used_params)

    k =
      cond do
        constant_resolvable?(k_name, axon, params, used_params) ->
          k_name |> constant!(axon, params, used_params) |> Nx.to_number()

        out_shape = output_shape(values_name) ->
          x_rank =
            case x do
              %Nx.Tensor{} = t -> Nx.rank(t)
              %Axon{} = node -> node |> kernel_shape_from_axon!() |> tuple_size()
            end

          pos_axis = if axis < 0, do: x_rank + axis, else: axis
          elem(out_shape, pos_axis)

        true ->
          raise ArgumentError,
                "TopK needs either a constant K input or a statically-" <>
                  "declared output shape; got neither for #{inspect(values_name)}."
      end

    direction = if largest, do: :desc, else: :asc

    values_fun = fn x, _opts ->
      sorted = Nx.argsort(x, axis: axis, direction: direction)
      taken = Nx.take_along_axis(x, sorted, axis: axis)
      Nx.slice_along_axis(taken, 0, k, axis: axis)
    end

    indices_fun = fn x, _opts ->
      sorted = Nx.argsort(x, axis: axis, direction: direction)
      Nx.slice_along_axis(sorted, 0, k, axis: axis)
    end

    values_layer = Axon.layer(values_fun, [x], name: values_name, op_name: :top_k_values)
    indices_layer = Axon.layer(indices_fun, [x], name: indices_name, op_name: :top_k_indices)

    axon =
      axon
      |> Map.put(values_name, values_layer)
      |> Map.put(indices_name, indices_layer)

    {axon, params, used_params}
  end

  defp recur_nodes(
         %Node{op_type: "Tile", input: [inp_name, repeats_name], output: [output_name]},
         {axon, params, used_params}
       ) do
    # Tile repeats `inp` along each axis per `repeats`. The repeats can come
    # in as an initializer (resolved here) or as a graph input; in the
    # latter case the corpus declares the output shape, and we derive
    # repeats from output_shape / input_shape per dim.
    inp = input!(inp_name, axon, params, used_params)

    repeats =
      cond do
        constant_resolvable?(repeats_name, axon, params, used_params) ->
          repeats_name
          |> constant!(axon, params, used_params)
          |> Nx.to_flat_list()

        out_shape = output_shape(output_name) ->
          in_shape =
            case inp do
              %Nx.Tensor{} = t -> Nx.shape(t)
              %Axon{} = node -> kernel_shape_from_axon!(node)
            end

          Enum.zip(Tuple.to_list(in_shape), Tuple.to_list(out_shape))
          |> Enum.map(fn {i, o} -> div(o, max(i, 1)) end)

        true ->
          raise ArgumentError,
                "Tile needs either a constant repeats input or static input " <>
                  "+ output shapes; got neither for #{inspect(output_name)}."
      end

    layer =
      case get_axon_node(inp) do
        %Axon.Node{op: :constant, opts: [value: v]} ->
          Axon.constant(Nx.tile(v, repeats), name: output_name)

        %Nx.Tensor{} = v ->
          Axon.constant(Nx.tile(v, repeats), name: output_name)

        %Axon.Node{} ->
          Axon.layer(fn x, _opts -> Nx.tile(x, repeats) end, [inp],
            name: output_name,
            op_name: :tile
          )
      end

    updated_axon = Map.put(axon, output_name, layer)
    {updated_axon, params, used_params}
  end

  defp recur_nodes(
         %Node{op_type: "Slice", input: [inp, starts, ends], output: [output_name]},
         {axon, params, used_params}
       ) do
    inp = input!(inp, axon, params, used_params)
    starts = constant!(starts, axon, params, used_params) |> Nx.to_flat_list()
    ends = constant!(ends, axon, params, used_params) |> Nx.to_flat_list()

    {updated_axon, updated_params} =
      slice_layer(inp, starts, ends, nil, nil, output_name, axon, used_params)

    {updated_axon, params, updated_params}
  end

  defp recur_nodes(
         %Node{op_type: "Slice", input: [inp, starts, ends, axes], output: [output_name]},
         {axon, params, used_params}
       ) do
    inp = input!(inp, axon, params, used_params)

    starts = constant!(starts, axon, params, used_params) |> Nx.to_flat_list()
    ends = constant!(ends, axon, params, used_params) |> Nx.to_flat_list()
    axes = constant!(axes, axon, params, used_params) |> Nx.to_flat_list()
    steps = List.duplicate(1, length(axes))

    {updated_axon, updated_params} =
      slice_layer(inp, starts, ends, axes, steps, output_name, axon, used_params)

    {updated_axon, params, updated_params}
  end

  defp recur_nodes(
         %Node{op_type: "Slice", input: [inp, starts, ends, axes, steps], output: [output_name]},
         {axon, params, used_params}
       ) do
    inp = input!(inp, axon, params, used_params)

    starts = constant!(starts, axon, params, used_params) |> Nx.to_flat_list()
    ends = constant!(ends, axon, params, used_params) |> Nx.to_flat_list()
    axes = constant!(axes, axon, params, used_params) |> Nx.to_flat_list()
    steps = constant!(steps, axon, params, used_params) |> Nx.to_flat_list()

    {updated_axon, updated_params} =
      slice_layer(inp, starts, ends, axes, steps, output_name, axon, used_params)

    {updated_axon, params, updated_params}
  end

  defp recur_nodes(
         %Node{op_type: "Shape", input: [inp], attribute: attrs, output: [output_name]},
         {axon, params, used_params}
       ) do
    shape_opts = options!(attrs)
    input = input!(inp, axon, params, used_params)
    ends = shape_opts["end"]
    starts = shape_opts["start"] || 0

    fun = fn inp, _opts ->
      shape = Nx.shape(inp)
      rank = Nx.rank(shape)

      starts = max(-rank, min(rank - 1, starts))
      start_axis = Nx.Shape.normalize_axis(shape, starts, List.duplicate(nil, rank))

      end_axis =
        if ends != nil and ends > -rank and ends < rank do
          ends = max(-rank + 1, min(rank - 1, ends))
          Nx.Shape.normalize_axis(shape, ends, List.duplicate(nil, rank))
        else
          rank
        end

      shape_list =
        for i <- start_axis..(end_axis - 1) do
          elem(shape, i) || -1
        end

      Nx.tensor(shape_list)
    end

    layer =
      case get_axon_node(input) do
        %Axon.Node{op: :constant, opts: [value: v]} ->
          new_value = fun.(v, [])
          Axon.constant(new_value, name: output_name)

        %Nx.Tensor{} = t ->
          new_value = fun.(t, [])
          Axon.constant(new_value, name: output_name)

        %Axon.Node{} ->
          layer_inputs =
            input
            |> Axon.get_inputs()
            |> Map.new(fn {k, v} -> {k, Nx.broadcast(0.0, v)} end)

          # Axon 0.8 `get_output_shape/2` returns a template (Nx.Tensor with
          # TemplateBackend), not a shape tuple. Use it directly.
          template = Axon.get_output_shape(input, layer_inputs)
          Axon.constant(fun.(template, []), name: output_name)
      end

    updated_axon = Map.put(axon, output_name, layer)

    {updated_axon, params, used_params}
  end

  defp recur_nodes(
         %Node{op_type: "Size", input: [inp_name], output: [output_name]},
         {axon, params, used_params}
       ) do
    input = input!(inp_name, axon, params, used_params)

    fun = fn t -> Nx.tensor(Nx.size(t), type: {:s, 64}) end

    layer =
      case get_axon_node(input) do
        %Axon.Node{op: :constant, opts: [value: v]} ->
          Axon.constant(fun.(v), name: output_name)

        %Nx.Tensor{} = t ->
          Axon.constant(fun.(t), name: output_name)

        %Axon.Node{} ->
          layer_inputs =
            input
            |> Axon.get_inputs()
            |> Map.new(fn {k, v} -> {k, Nx.broadcast(0.0, v)} end)

          # Axon 0.8 `get_output_shape/2` returns a template (Nx.Tensor with
          # TemplateBackend), not a shape tuple. Use it directly.
          template = Axon.get_output_shape(input, layer_inputs)
          Axon.constant(fun.(template), name: output_name)
      end

    updated_axon = Map.put(axon, output_name, layer)
    {updated_axon, params, used_params}
  end

  defp recur_nodes(
         %Node{op_type: "Transpose", input: [input], attribute: attrs, output: [output_name]},
         {axon, params, used_params}
       ) do
    transpose_options = options!(attrs)

    permutation = transpose_options["perm"]

    inp = input!(input, axon, params, used_params)

    {updated_axon, updated_params} =
      case get_axon_node(inp) do
        %Axon.Node{op: :constant, opts: [value: v]} ->
          new_value =
            if permutation, do: Nx.transpose(v, axes: permutation), else: Nx.transpose(v)

          layer = Axon.constant(new_value, name: output_name)
          updated_axon = Map.put(axon, output_name, layer)
          {updated_axon, used_params}

        %Axon.Node{} ->
          layer = Axon.transpose(inp, permutation, name: output_name)

          updated_axon = Map.put(axon, output_name, layer)
          {updated_axon, used_params}

        %Nx.Tensor{} = inp ->
          new_value =
            if permutation, do: Nx.transpose(inp, axes: permutation), else: Nx.transpose(inp)

          updated_params = Map.put(used_params, output_name, new_value)
          {axon, updated_params}
      end

    {updated_axon, params, updated_params}
  end

  defp recur_nodes(
         %Node{
           op_type: "Unsqueeze",
           input: [input | maybe_axis],
           attribute: attrs,
           output: [output_name]
         },
         {axon, params, used_params}
       ) do
    # Unsqueeze's `axes` migrated from attribute to a second input at opset
    # 13. Same canary check as Squeeze above: catch the malformed-attribute-
    # on-modern-opset combo.
    opset = opset_version()
    unsqueeze_options = options!(attrs)

    if opset && opset >= 13 && maybe_axis == [] && !unsqueeze_options["axes"] do
      raise ArgumentError,
            "Unsqueeze declares opset #{opset} (≥ 13) but axes is missing " <>
              "from both attributes and inputs."
    end

    inp = input!(input, axon, params, used_params)

    axes =
      case maybe_axis do
        [] ->
          unsqueeze_options["axes"]

        [axes] ->
          constant!(axes, axon, params, used_params) |> Nx.to_flat_list()
      end

    fun = fn input ->
      Enum.reduce(axes, input, fn axis, x -> Nx.new_axis(x, axis) end)
    end

    case get_axon_node(inp) do
      %Nx.Tensor{} = tensor ->
        updated_params = Map.put(used_params, output_name, fun.(tensor))
        {axon, params, updated_params}

      %Axon.Node{op: :constant, opts: [value: tensor]} ->
        new_value = fun.(tensor)
        updated_axon = Map.put(axon, output_name, Axon.constant(new_value))
        {updated_axon, params, used_params}

      %Axon.Node{} ->
        updated_axon =
          Map.put(axon, output_name, Axon.nx(inp, fun, name: output_name, op_name: :unsqueeze))

        {updated_axon, params, used_params}
    end
  end

  defp recur_nodes(
         %Node{op_type: "If", input: [input], attribute: attrs, output: outputs},
         {axon, params, used_params}
       ) do
    # ONNX If takes a bool scalar condition plus two subgraphs as attributes.
    # Each subgraph is recursively deserialised via graph_to_axon/2. There
    # are two paths here because Axon.cond under Nx 0.12 raises
    # Nx.Defn.Tree.scope_ids_each on raw Nx.Tensor constants in branches
    # (see test_if).
    #
    # Path A (closed branches — no graph inputs): pre-evaluate both
    # subgraphs to concrete tensors and Nx.select between them inside a
    # plain Axon.layer. This sidesteps Axon.cond's defn-trace issue and
    # covers test_if (both branches are just Constant ops).
    #
    # Path B (open branches): fall back to Axon.cond, which is correct for
    # Nx 0.5 and remains useful for the simpler cond patterns even under
    # 0.12. The proper general-case fix is subgraph-as-closure
    # deserialisation (Phase 4 deliverable), which also unblocks Loop/Scan.
    cond_options = options!(attrs)

    inp = axon!(input, axon)

    else_branch = cond_options["else_branch"]
    then_branch = cond_options["then_branch"]

    {[else_graph], else_params} = graph_to_axon(else_branch, [])
    {[then_graph], then_params} = graph_to_axon(then_branch, [])

    updated_params =
      else_params
      |> Map.merge(then_params)
      |> Map.merge(used_params)

    updated_axon =
      cond do
        closed_subgraph?(else_branch) and closed_subgraph?(then_branch) ->
          # Both branches independent of any graph input. Compute both
          # values eagerly, then pick at runtime via Nx.select.
          else_value = eval_closed_subgraph(else_graph, else_params)
          then_value = eval_closed_subgraph(then_graph, then_params)

          fun = fn pred, _opts ->
            mask = Nx.not_equal(pred, 0)
            Nx.select(mask, then_value, else_value)
          end

          Enum.reduce(outputs, axon, fn out_name, acc ->
            layer = Axon.layer(fun, [inp], name: out_name, op_name: :if_closed)
            Map.put(acc, out_name, layer)
          end)

        true ->
          # Open-branch fallback: Axon.cond. May fail under Nx 0.12 with
          # constants inside branches; the test_if registry entry covers
          # that for now.
          Enum.reduce(outputs, axon, fn out_name, acc ->
            Map.put(
              acc,
              out_name,
              Axon.cond(inp, &Nx.not_equal(&1, 0), then_graph, else_graph)
            )
          end)
      end

    {updated_axon, params, updated_params}
  end

  # True when a named input can be resolved to a concrete tensor at import
  # time (initializer, Constant op, or already-consumed param). Used by
  # shape-driven ops that can fall back on declared output shapes when their
  # shape inputs are runtime.
  defp constant_resolvable?(name, axon, params, used_params) do
    Map.has_key?(params, name) or Map.has_key?(used_params, name) or
      (Map.has_key?(axon, name) and
         match?(%Axon.Node{op: :constant}, get_axon_node(axon[name])))
  end

  # Resolve the axes input of a 2-input reduction (opset 13+/18+ form).
  # Returns a list of axis indices, `:empty` for a declared-shape-{0}
  # graph input (axes absent), or raises if the values aren't statically
  # available.
  defp resolve_reduce_axes!(axes_name, axon, params, used_params) do
    cond do
      Map.has_key?(params, axes_name) ->
        params[axes_name] |> Nx.to_flat_list()

      Map.has_key?(used_params, axes_name) ->
        used_params[axes_name] |> Nx.to_flat_list()

      Map.has_key?(axon, axes_name) ->
        case get_axon_node(axon[axes_name]) do
          %Axon.Node{op: :constant, opts: [value: v]} ->
            Nx.to_flat_list(v)

          %Axon.Node{op: :input, opts: opts} ->
            case Keyword.get(opts, :shape) do
              {0} ->
                :empty

              _other ->
                raise ArgumentError,
                      "Reduction axes via runtime graph input is not yet " <>
                        "supported (axes input #{inspect(axes_name)})."
            end

          _ ->
            raise ArgumentError,
                  "Reduction axes input #{inspect(axes_name)} must resolve to " <>
                    "a constant or empty graph input."
        end

      true ->
        raise ArgumentError, "axes input #{inspect(axes_name)} not found"
    end
  end

  defp build_reduce_layer(input, layer_fun, opts, output_name, op_name) do
    case get_axon_node(input) do
      %Axon.Node{op: :constant, opts: [value: v]} ->
        Axon.constant(layer_fun.(v, opts), name: output_name)

      %Nx.Tensor{} = t ->
        Axon.constant(layer_fun.(t, opts), name: output_name)

      %Axon.Node{} ->
        Axon.layer(layer_fun, [input], [name: output_name, op_name: op_name] ++ opts)
    end
  end

  defp closed_subgraph?(%Onnx.GraphProto{input: inputs}), do: inputs == []
  defp closed_subgraph?(nil), do: false

  defp eval_closed_subgraph(%Axon{} = axon, params) do
    {_init, predict} = Axon.build(axon)
    model_state = Axon.ModelState.new(params)
    predict.(model_state, %{}) |> Nx.backend_copy(Nx.BinaryBackend)
  end

  defp recur_nodes(
         %Node{
           op_type: "GroupNormalization",
           attribute: attrs,
           input: [x_name, scale_name, bias_name],
           output: [output_name]
         },
         {axon, params, used_params}
       ) do
    # GroupNormalization: split the channel axis into `num_groups`, then
    # normalise within each group (channels + spatial dims), then apply
    # the per-channel scale and bias.
    opts = options!(attrs)
    num_groups = opts["num_groups"]
    epsilon = opts["epsilon"] || 1.0e-5

    if is_nil(num_groups) do
      raise ArgumentError, "GroupNormalization requires the num_groups attribute"
    end

    x = input!(x_name, axon, params, used_params)
    scale = input!(scale_name, axon, params, used_params)
    bias = input!(bias_name, axon, params, used_params)

    fun = fn x, scale, bias, opts ->
      g = opts[:num_groups]
      eps = opts[:epsilon]
      shape = Nx.shape(x)
      rank = tuple_size(shape)
      n = elem(shape, 0)
      c = elem(shape, 1)
      spatial = Enum.map(2..(rank - 1)//1, &elem(shape, &1))
      grouped_shape = List.to_tuple([n, g, div(c, g) | spatial])
      reshaped = Nx.reshape(x, grouped_shape)
      norm_axes = Enum.to_list(2..(tuple_size(grouped_shape) - 1)//1)

      mean = Nx.mean(reshaped, axes: norm_axes, keep_axes: true)
      var = Nx.variance(reshaped, axes: norm_axes, keep_axes: true)
      normalised = Nx.divide(Nx.subtract(reshaped, mean), Nx.sqrt(Nx.add(var, eps)))
      flattened = Nx.reshape(normalised, shape)

      scale_shape = List.to_tuple([1, c | List.duplicate(1, rank - 2)])
      scale_r = Nx.reshape(scale, scale_shape)
      bias_r = Nx.reshape(bias, scale_shape)
      Nx.add(Nx.multiply(flattened, scale_r), bias_r)
    end

    layer =
      Axon.layer(fun, [x, scale, bias],
        name: output_name,
        op_name: :group_norm,
        num_groups: num_groups,
        epsilon: epsilon
      )

    {Map.put(axon, output_name, layer), params, used_params}
  end

  defp recur_nodes(
         %Node{op_type: "Gelu", attribute: attrs, input: [input_name], output: [output_name]},
         {axon, params, used_params}
       ) do
    # Gelu: approximate="none" → x * Φ(x) (erf form); approximate="tanh"
    # → tanh-based approximation. Axon.Activations.gelu uses the erf form.
    approximate = options!(attrs)["approximate"] || "none"
    input = input!(input_name, axon, params, used_params)

    fun =
      case approximate do
        "none" ->
          &Axon.Activations.gelu/1

        "tanh" ->
          fn x ->
            # 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
            inner =
              Nx.multiply(
                Nx.sqrt(Nx.tensor(2.0 / :math.pi(), type: Nx.type(x))),
                Nx.add(x, Nx.multiply(Nx.tensor(0.044715, type: Nx.type(x)), Nx.pow(x, 3)))
              )

            Nx.multiply(
              Nx.multiply(Nx.tensor(0.5, type: Nx.type(x)), x),
              Nx.add(Nx.tensor(1.0, type: Nx.type(x)), Nx.tanh(inner))
            )
          end

        other ->
          raise ArgumentError, "Gelu approximate=#{inspect(other)} is not supported"
      end

    output =
      case get_axon_node(input) do
        %Axon.Node{op: :constant, opts: [value: v]} ->
          Axon.constant(fun.(v), name: output_name)

        %Axon.Node{} ->
          Axon.nx(input, fun, name: output_name, op_name: :gelu)

        %Nx.Tensor{} = t ->
          Axon.constant(fun.(t), name: output_name)
      end

    {Map.put(axon, output_name, output), params, used_params}
  end

  defp recur_nodes(
         %Node{op_type: "Swish", attribute: attrs, input: [input_name], output: [output_name]},
         {axon, params, used_params}
       ) do
    # Swish: y = x * sigmoid(alpha * x). alpha default 1.0.
    alpha = options!(attrs)["alpha"] || 1.0
    input = input!(input_name, axon, params, used_params)

    fun = fn x, opts ->
      a = Nx.tensor(opts[:alpha], type: Nx.type(x))
      Nx.multiply(x, Nx.sigmoid(Nx.multiply(a, x)))
    end

    apply_fun = &fun.(&1, alpha: alpha)

    output =
      case get_axon_node(input) do
        %Axon.Node{op: :constant, opts: [value: v]} ->
          Axon.constant(apply_fun.(v), name: output_name)

        %Axon.Node{} ->
          Axon.layer(fun, [input], name: output_name, op_name: :swish, alpha: alpha)

        %Nx.Tensor{} = t ->
          Axon.constant(apply_fun.(t), name: output_name)
      end

    {Map.put(axon, output_name, output), params, used_params}
  end

  defp recur_nodes(
         %Node{
           op_type: "ThresholdedRelu",
           attribute: attrs,
           input: [input_name],
           output: [output_name]
         },
         {axon, params, used_params}
       ) do
    # ThresholdedRelu: y = x if x > alpha, else 0. alpha default 1.0.
    alpha = options!(attrs)["alpha"] || 1.0
    input = input!(input_name, axon, params, used_params)

    fun = fn x, opts ->
      a = Nx.tensor(opts[:alpha], type: Nx.type(x))
      Nx.select(Nx.greater(x, a), x, Nx.tensor(0.0, type: Nx.type(x)))
    end

    apply_fun = &fun.(&1, alpha: alpha)

    output =
      case get_axon_node(input) do
        %Axon.Node{op: :constant, opts: [value: v]} ->
          Axon.constant(apply_fun.(v), name: output_name)

        %Axon.Node{} ->
          Axon.layer(fun, [input], name: output_name, op_name: :thresholded_relu, alpha: alpha)

        %Nx.Tensor{} = t ->
          Axon.constant(apply_fun.(t), name: output_name)
      end

    {Map.put(axon, output_name, output), params, used_params}
  end

  defp recur_nodes(
         %Node{
           op_type: "LpNormalization",
           attribute: attrs,
           input: [input_name],
           output: [output_name]
         },
         {axon, params, used_params}
       ) do
    opts = options!(attrs)
    axis = opts["axis"] || -1
    p = opts["p"] || 2
    input = input!(input_name, axon, params, used_params)

    fun = fn x, opts ->
      ax = opts[:axis]
      p_val = opts[:p]

      norm =
        case p_val do
          1 -> Nx.sum(Nx.abs(x), axes: [ax], keep_axes: true)
          2 -> Nx.sqrt(Nx.sum(Nx.pow(x, 2), axes: [ax], keep_axes: true))
          _ ->
            Nx.pow(Nx.sum(Nx.pow(Nx.abs(x), p_val), axes: [ax], keep_axes: true),
              Nx.divide(Nx.tensor(1, type: Nx.type(x)), p_val))
        end

      Nx.divide(x, norm)
    end

    layer =
      case get_axon_node(input) do
        %Axon.Node{op: :constant, opts: [value: v]} ->
          Axon.constant(fun.(v, axis: axis, p: p), name: output_name)

        %Axon.Node{} ->
          Axon.layer(fun, [input], name: output_name, op_name: :lp_normalization, axis: axis, p: p)

        %Nx.Tensor{} = t ->
          Axon.constant(fun.(t, axis: axis, p: p), name: output_name)
      end

    {Map.put(axon, output_name, layer), params, used_params}
  end

  defp recur_nodes(
         %Node{op_type: "Trilu", attribute: attrs, input: [data_name], output: [output_name]},
         {axon, params, used_params}
       ) do
    upper = (options!(attrs)["upper"] || 1) == 1
    inp = input!(data_name, axon, params, used_params)
    build_trilu_layer(inp, 0, upper, output_name, axon, params, used_params)
  end

  defp recur_nodes(
         %Node{
           op_type: "Trilu",
           attribute: attrs,
           input: [data_name, k_name],
           output: [output_name]
         },
         {axon, params, used_params}
       ) do
    upper = (options!(attrs)["upper"] || 1) == 1
    inp = input!(data_name, axon, params, used_params)
    k = input!(k_name, axon, params, used_params)

    case get_axon_node(k) do
      %Axon.Node{op: :constant, opts: [value: v]} ->
        build_trilu_layer(inp, Nx.to_number(v), upper, output_name, axon, params, used_params)

      %Nx.Tensor{} = t ->
        build_trilu_layer(inp, Nx.to_number(t), upper, output_name, axon, params, used_params)

      %Axon.Node{} ->
        fun = fn x, k_tensor, _opts ->
          do_trilu(x, Nx.as_type(k_tensor, {:s, 64}), upper)
        end

        layer = Axon.layer(fun, [inp, k], name: output_name, op_name: :trilu)
        updated_axon = Map.put(axon, output_name, layer)
        {updated_axon, params, used_params}
    end
  end

  defp recur_nodes(
         %Node{op_type: "Where", input: [c_name, x_name, y_name], output: [output_name]},
         {axon, params, used_params}
       ) do
    condition = input!(c_name, axon, params, used_params)
    x = input!(x_name, axon, params, used_params)
    y = input!(y_name, axon, params, used_params)

    {updated_axon, updated_params} =
      case {get_axon_node(condition), get_axon_node(x), get_axon_node(y)} do
        {%Axon.Node{op: :constant, opts: [value: c]}, %Axon.Node{op: :constant, opts: [value: x]},
         %Axon.Node{op: :constant, opts: [value: y]}} ->
          new_value = Nx.select(c, x, y)
          updated_axon = Map.put(axon, output_name, Axon.constant(new_value, name: output_name))
          {updated_axon, used_params}

        {%Axon.Node{op: :constant, opts: [value: c]}, %Axon.Node{op: :constant, opts: [value: x]},
         %Nx.Tensor{} = y} ->
          new_value = Nx.select(c, x, y)
          updated_axon = Map.put(axon, output_name, Axon.constant(new_value, name: output_name))
          {updated_axon, used_params}

        {%Axon.Node{op: :constant, opts: [value: c]}, %Nx.Tensor{} = x, %Nx.Tensor{} = y} ->
          new_value = Nx.select(c, x, y)
          updated_axon = Map.put(axon, output_name, Axon.constant(new_value, name: output_name))
          {updated_axon, used_params}

        {%Axon.Node{}, %Axon.Node{}, %Axon.Node{}} ->
          fun = fn x, y, z, _opts ->
            # TODO: Nx's shape rules should handle this like a binary broadcast
            # between all operands
            x = Nx.multiply(x, Nx.broadcast(1, y))
            y = Nx.multiply(y, Nx.broadcast(1, x))
            z = Nx.multiply(z, Nx.broadcast(1, y))
            Nx.select(x, y, z)
          end

          layer = Axon.layer(fun, [condition, x, y], name: output_name, op_name: :select)
          updated_axon = Map.put(axon, output_name, layer)
          {updated_axon, used_params}

        {%Axon.Node{}, %Axon.Node{}, %Nx.Tensor{} = y} ->
          # TODO: Nx's shape rules should handle this like a binary broadcast
          # between all operands
          fun = fn x, y, z, _opts ->
            x = Nx.multiply(x, Nx.broadcast(1, y))
            y = Nx.multiply(y, Nx.broadcast(1, x))
            z = Nx.multiply(z, Nx.broadcast(1, y))
            Nx.select(x, y, z)
          end

          param = Axon.param(y_name, fn _, _ -> Nx.shape(y) end)
          layer = Axon.layer(fun, [condition, x, param], name: output_name, op_name: :select)

          updated_axon = Map.put(axon, output_name, layer)
          updated_params = Map.put(used_params, output_name, %{y_name => param})
          {updated_axon, updated_params}
      end

    {updated_axon, params, updated_params}
  end

  defp recur_nodes(
         %Node{
           op_type: "CumSum",
           attribute: attrs,
           input: [x_name, axis_name],
           output: [output_name]
         },
         {axon, params, used_params}
       ) do
    # CumSum opset 11+: takes (x, axis) inputs plus `exclusive` and `reverse`
    # attributes. Lower to Nx.cumulative_sum, which natively supports
    # `reverse`. Exclusive mode is the inclusive cumsum minus x (works for
    # both forward and reverse direction).
    opts = options!(attrs)
    exclusive = (opts["exclusive"] || 0) == 1
    reverse = (opts["reverse"] || 0) == 1

    x = input!(x_name, axon, params, used_params)
    axis = constant!(axis_name, axon, params, used_params) |> Nx.to_number()

    fun = fn x, _opts ->
      cumsum = Nx.cumulative_sum(x, axis: axis, reverse: reverse)
      if exclusive, do: Nx.subtract(cumsum, x), else: cumsum
    end

    layer = Axon.layer(fun, [x], name: output_name, op_name: :cumsum)
    updated_axon = Map.put(axon, output_name, layer)
    {updated_axon, params, used_params}
  end

  defp recur_nodes(
         %Node{op_type: "PRelu", input: [x_name, slope_name], output: [output_name]},
         {axon, params, used_params}
       ) do
    x = input!(x_name, axon, params, used_params)
    slope = input!(slope_name, axon, params, used_params)

    fun = fn x, s, _opts ->
      Nx.select(Nx.less(x, 0), Nx.multiply(s, x), x)
    end

    layer = Axon.layer(fun, [x, slope], name: output_name, op_name: :prelu)
    updated_axon = Map.put(axon, output_name, layer)
    {updated_axon, params, used_params}
  end

  defp recur_nodes(
         %Node{op_type: "Clip", attribute: attrs, input: [inp_name], output: [output_name]},
         {axon, params, used_params}
       ) do
    # Clip's min/max migrated from attributes (pre-opset-11) to inputs at
    # opset 11. The single-input form is the only form that's valid for
    # pre-opset-11 AND for opset 11+ with both bounds omitted. We check the
    # declared opset to surface invalid combinations: an opset ≥ 11 model
    # that still carries min/max attributes is malformed (an exporter bug
    # somewhere) and would silently lose those bounds otherwise. Pre-opset-11
    # treats absent min/max as "no clip" via the dtype-finite fallbacks.
    opset = opset_version()
    inp = input!(inp_name, axon, params, used_params)

    opts = options!(attrs)

    if opset && opset >= 11 && (opts["min"] || opts["max"]) do
      raise ArgumentError,
            "Clip declares opset #{opset} (≥ 11) but supplies min/max as " <>
              "attributes; opset 11+ requires them as inputs."
    end

    min = opts["min"] || Nx.Constants.min_finite({:f, 32})
    max = opts["max"] || Nx.Constants.max_finite({:f, 32})

    updated_axon =
      case get_axon_node(inp) do
        %Axon.Node{op: :constant, opts: [value: v]} ->
          new_value = Nx.clip(v, min, max)
          layer = Axon.constant(new_value, name: output_name)
          Map.put(axon, output_name, layer)

        %Nx.Tensor{} = inp ->
          new_value = Nx.clip(inp, min, max)
          layer = Axon.constant(new_value, name: output_name)
          Map.put(axon, output_name, layer)

        %Axon.Node{} ->
          layer =
            Axon.nx(inp, fn x -> Nx.clip(x, min, max) end, name: output_name, op_name: :clip)

          Map.put(axon, output_name, layer)
      end

    {updated_axon, params, used_params}
  end

  defp recur_nodes(
         %Node{op_type: "Clip", input: [inp_name, min_name], output: [output_name]},
         {axon, params, used_params}
       ) do
    inp = input!(inp_name, axon, params, used_params)
    min = input!(min_name, axon, params, used_params)

    updated_axon =
      case {get_axon_node(inp), get_axon_node(min)} do
        {%Axon.Node{} = inp, %Axon.Node{} = min} ->
          fun = fn x, y, _opts -> Nx.clip(x, y, Nx.Constants.max_finite(Nx.type(x))) end
          layer = Axon.layer(fun, [inp, min], name: output_name)
          Map.put(axon, output_name, layer)

        {%Axon.Node{op: :constant, opts: [value: v]}, %Nx.Tensor{} = min} ->
          new_value = Nx.clip(v, min, Nx.Constants.max_finite(Nx.type(v)))
          layer = Axon.constant(new_value, name: output_name)
          Map.put(axon, output_name, layer)

        {%Axon.Node{}, %Nx.Tensor{} = min} ->
          layer =
            Axon.nx(inp, fn x -> Nx.clip(x, min, Nx.Constants.max_finite(Nx.type(x))) end,
              name: output_name,
              op_name: :clip
            )

          Map.put(axon, output_name, layer)
      end

    {updated_axon, params, used_params}
  end

  defp recur_nodes(
         %Node{op_type: "Clip", input: [inp_name, "", max_name], output: [output_name]},
         {axon, params, used_params}
       ) do
    inp = input!(inp_name, axon, params, used_params)
    max = input!(max_name, axon, params, used_params)

    updated_axon =
      case {inp, max} do
        {%Axon.Node{}, %Axon.Node{}} ->
          fun = fn x, y, _opts -> Nx.clip(x, Nx.Constants.min_finite(Nx.type(x)), y) end
          layer = Axon.layer(fun, [inp, max], name: output_name)
          Map.put(axon, output_name, layer)

        {%Axon.Node{op: :constant, opts: [value: v]}, %Nx.Tensor{} = max} ->
          new_value = Nx.clip(v, Nx.Constants.min_finite(Nx.type(v)), max)
          layer = Axon.constant(new_value, name: output_name)
          Map.put(axon, output_name, layer)

        {%Axon.Node{}, %Nx.Tensor{} = max} ->
          layer =
            Axon.nx(inp, fn x -> Nx.clip(x, Nx.Constants.min_finite(Nx.type(x)), max) end,
              name: output_name,
              op_name: :clip
            )

          Map.put(axon, output_name, layer)
      end

    {updated_axon, params, used_params}
  end

  defp recur_nodes(
         %Node{op_type: "Clip", input: [inp_name, min_name, max_name], output: [output_name]},
         {axon, params, used_params}
       ) do
    inp = input!(inp_name, axon, params, used_params)
    min = input!(min_name, axon, params, used_params)
    max = input!(max_name, axon, params, used_params)

    {updated_axon, used_params} =
      case {get_axon_node(inp), get_axon_node(min), get_axon_node(max)} do
        {%Axon.Node{}, %Axon.Node{}, %Axon.Node{}} ->
          fun = fn x, y, z, _opts -> Nx.clip(x, y, z) end
          layer = Axon.layer(fun, [inp, min, max], name: output_name, op_name: :clip)
          updated_axon = Map.put(axon, output_name, layer)
          {updated_axon, used_params}

        {%Axon.Node{op: :constant, opts: [value: v]}, %Nx.Tensor{} = min, %Nx.Tensor{} = max} ->
          new_value = Nx.clip(v, min, max)
          layer = Axon.constant(new_value, name: output_name)
          Map.put(axon, output_name, layer)

        {%Axon.Node{} = inp, %Nx.Tensor{} = min, %Nx.Tensor{} = max} ->
          fun = fn x, min, max, _opts -> Nx.clip(x, min, max) end
          min = Axon.param(min_name, fn _ -> Nx.shape(min) end)
          max = Axon.param(max_name, fn _ -> Nx.shape(max) end)
          layer = Axon.layer(fun, [inp, min, max], name: output_name, op_name: :clip)
          updated_axon = Map.put(axon, output_name, layer)
          updated_params = Map.put(used_params, output_name, %{min_name => min, max_name => max})
          {updated_axon, updated_params}
      end

    {updated_axon, params, used_params}
  end

  defp recur_nodes(
         %Node{op_type: "Squeeze", attribute: attrs, input: [data], output: [output_name]},
         {axon, params, used_params}
       ) do
    # Squeeze's `axes` migrated from attribute to a second input at opset 13.
    # The single-input form is only valid pre-opset-13, OR opset 13+ with
    # axes omitted (squeeze all size-1 dims). Catch malformed models that
    # supply an `axes` attribute at opset ≥ 13.
    opset = opset_version()
    inp = input!(data, axon, params, used_params)
    squeeze_options = options!(attrs)

    if opset && opset >= 13 && squeeze_options["axes"] do
      raise ArgumentError,
            "Squeeze declares opset #{opset} (≥ 13) but supplies axes as " <>
              "an attribute; opset 13+ requires axes as a second input."
    end

    axes = squeeze_options["axes"]

    fun = fn x, _opts ->
      axes = axes || Nx.axes(x)
      Nx.squeeze(x, axes: axes)
    end

    updated_axon =
      case get_axon_node(inp) do
        %Axon.Node{op: :constant, opts: [value: v]} ->
          new_value = fun.(v, [])
          layer = Axon.constant(new_value, name: output_name)
          Map.put(axon, output_name, layer)

        %Axon.Node{} ->
          layer = Axon.layer(fun, [inp], name: output_name, op_name: :squeeze)
          Map.put(axon, output_name, layer)
      end

    {updated_axon, params, used_params}
  end

  defp recur_nodes(
         %Node{op_type: "Squeeze", input: [data, axes], output: [output_name]},
         {axon, params, used_params}
       ) do
    inp = input!(data, axon, params, used_params)
    axes = constant!(axes, axon, params, used_params) |> Nx.to_flat_list()

    fun = fn x, _params ->
      Nx.squeeze(x, axes: axes)
    end

    updated_axon =
      case get_axon_node(inp) do
        %Axon.Node{op: :constant, opts: [value: v]} ->
          new_value = Nx.squeeze(v, axes: axes)
          layer = Axon.constant(new_value, name: output_name)
          Map.put(axon, output_name, layer)

        %Axon.Node{} ->
          layer = Axon.layer(fun, [inp], name: output_name, op_name: :squeeze)
          Map.put(axon, output_name, layer)

        %Nx.Tensor{} = t ->
          new_value = Nx.squeeze(t, axes: axes)
          layer = Axon.constant(new_value, name: output_name)
          Map.put(axon, output_name, layer)
      end

    {updated_axon, params, used_params}
  end

  defp recur_nodes(
         %Node{op_type: "Split", input: [input, split], attribute: attrs, output: outputs},
         {axon, params, used_params}
       ) do
    split_options = options!(attrs)

    inp = input!(input, axon, params, used_params)
    split = constant!(split, axon, params, used_params) |> Nx.to_flat_list()

    axis = split_options["axis"] || 0
    updated_axon = build_split_layers(inp, axon, split, axis, outputs)

    {updated_axon, params, used_params}
  end

  defp recur_nodes(
         %Node{op_type: "EyeLike", input: [input], attribute: attrs, output: [output_name]},
         {axon, params, used_params}
       ) do
    eye_options = options!(attrs)

    inp = input!(input, axon, params, used_params)

    type = if eye_options["dtype"], do: onnx_type_to_nx_type(eye_options["dtype"]), else: {:f, 32}
    k = eye_options["k"] || 0

    layer =
      case get_axon_node(inp) do
        %Axon.Node{op: :constant, opts: [value: v]} ->
          Axon.constant(eye_like_tensor(Nx.shape(v), type, k))

        %Axon.Node{} ->
          fun = fn x, _opts -> eye_like_tensor(Nx.shape(x), type, k) end
          Axon.layer(fun, [inp], name: output_name, op_name: :eye_like)

        %Nx.Tensor{} = t ->
          Axon.constant(eye_like_tensor(Nx.shape(t), type, k))
      end

    updated_axon = Map.put(axon, output_name, layer)
    {updated_axon, params, used_params}
  end

  # Like `Nx.eye/2` but supports the `k` (diagonal offset) attribute that
  # ONNX EyeLike requires. `k > 0` shifts the diagonal up-right, `k < 0`
  # shifts it down-left.
  defp eye_like_tensor({m, n}, type, 0), do: Nx.eye({m, n}, type: type)

  defp eye_like_tensor({m, n}, type, k) do
    rows = Nx.iota({m, 1})
    cols = Nx.iota({1, n})
    Nx.subtract(cols, rows) |> Nx.equal(k) |> Nx.as_type(type)
  end

  defp recur_nodes(
         %Node{op_type: "RandomUniform", attribute: attrs, output: [output_name]},
         {axon, params, used_params}
       ) do
    random_options = options!(attrs)

    dtype = random_options["dtype"] || 1
    high = random_options["high"] || 1.0
    low = random_options["low"] || 0.0
    seed = random_options["seed"]
    shape = random_options["shape"]
    nx_type = onnx_type_to_nx_type(dtype)

    {tensor, _key} =
      Nx.Random.key(coerce_random_seed(seed))
      |> Nx.Random.uniform(low, high, type: nx_type, shape: List.to_tuple(shape))

    layer = Axon.constant(tensor, name: output_name)
    updated_axon = Map.put(axon, output_name, layer)

    {updated_axon, params, used_params}
  end

  defp recur_nodes(
         %Node{
           op_type: "RandomUniformLike",
           input: [input],
           attribute: attrs,
           output: [output_name]
         },
         {axon, params, used_params}
       ) do
    random_options = options!(attrs)

    inp = input!(input, axon, params, used_params)

    dtype = random_options["dtype"] || 1
    high = random_options["high"] || 1.0
    low = random_options["low"] || 0.0
    seed = random_options["seed"]
    nx_type = onnx_type_to_nx_type(dtype)

    layer =
      case get_axon_node(inp) do
        %Axon.Node{op: :constant, opts: [value: v]} ->
          shape = Nx.shape(v)

          {tensor, _key} =
            Nx.Random.key(coerce_random_seed(seed))
            |> Nx.Random.uniform(low, high, type: nx_type, shape: shape)

          Axon.constant(tensor, name: output_name)

        %Axon.Node{} ->
          fun = fn x, _opts ->
            shape = Nx.shape(x)

            Nx.Random.key(coerce_random_seed(seed))
            |> Nx.Random.uniform(low, high, type: nx_type, shape: shape)
            |> then(fn {tensor, _key} -> tensor end)
          end

          Axon.layer(fun, [inp], name: output_name, op_name: :random_uniform_like)

        %Nx.Tensor{} = t ->
          shape = Nx.shape(t)

          {tensor, _key} =
            Nx.Random.key(coerce_random_seed(seed))
            |> Nx.Random.uniform(low, high, type: nx_type, shape: shape)

          Axon.constant(tensor, name: output_name)
      end

    updated_axon = Map.put(axon, output_name, layer)

    {updated_axon, params, used_params}
  end

  defp recur_nodes(
         %Node{op_type: "RandomNormal", attribute: attrs, output: [output_name]},
         {axon, params, used_params}
       ) do
    random_options = options!(attrs)

    dtype = random_options["dtype"] || 1
    mean = random_options["mean"] || 0.0
    scale = random_options["scale"] || 1.0
    seed = random_options["seed"]
    shape = random_options["shape"]
    nx_type = onnx_type_to_nx_type(dtype)

    {tensor, _key} =
      Nx.Random.key(coerce_random_seed(seed))
      |> Nx.Random.normal(mean, scale, type: nx_type, shape: List.to_tuple(shape))

    layer = Axon.constant(tensor, name: output_name)
    updated_axon = Map.put(axon, output_name, layer)

    {updated_axon, params, used_params}
  end

  defp recur_nodes(
         %Node{
           op_type: "RandomNormalLike",
           input: [input],
           attribute: attrs,
           output: [output_name]
         },
         {axon, params, used_params}
       ) do
    random_options = options!(attrs)

    inp = input!(input, axon, params, used_params)

    dtype = random_options["dtype"] || 1
    mean = random_options["mean"] || 0.0
    scale = random_options["scale"] || 1.0
    seed = random_options["seed"]
    nx_type = onnx_type_to_nx_type(dtype)

    layer =
      case get_axon_node(inp) do
        %Axon.Node{op: :constant, opts: [value: v]} ->
          shape = Nx.shape(v)

          {tensor, _key} =
            Nx.Random.key(coerce_random_seed(seed))
            |> Nx.Random.normal(mean, scale, type: nx_type, shape: shape)

          Axon.constant(tensor, name: output_name)

        %Axon.Node{} ->
          fun = fn x, _opts ->
            shape = Nx.shape(x)

            Nx.Random.key(coerce_random_seed(seed))
            |> Nx.Random.normal(mean, scale, type: nx_type, shape: shape)
            |> then(fn {tensor, _key} -> tensor end)
          end

          Axon.layer(fun, [inp], name: output_name, op_name: :random_normal_like)

        %Nx.Tensor{} = t ->
          shape = Nx.shape(t)

          {tensor, _key} =
            Nx.Random.key(coerce_random_seed(seed))
            |> Nx.Random.normal(mean, scale, type: nx_type, shape: shape)

          Axon.constant(tensor, name: output_name)
      end

    updated_axon = Map.put(axon, output_name, layer)

    {updated_axon, params, used_params}
  end

  defp recur_nodes(
         %Node{
           op_type: "Dropout",
           input: [inp_name],
           attribute: attrs,
           output: [output_name | maybe_mask]
         },
         {axon, params, used_params}
       ) do
    dropout_options = options!(attrs)

    # TODO: Not supported yet in Axon
    _seed = dropout_options["seed"]
    ratio = dropout_options["ratio"] || 0.0
    is_test = dropout_options["is_test"] || 0

    inp = input!(inp_name, axon, params, used_params)

    dropout_layer =
      if is_test == 1 or ratio == 0.0 do
        Axon.nx(inp, & &1, name: output_name, op_name: :dropout)
      else
        Axon.dropout(inp, rate: ratio, name: output_name)
      end

    updated_axon =
      case maybe_mask do
        [] ->
          Map.put(axon, output_name, dropout_layer)

        [mask_name] ->
          layer_inputs =
            inp
            |> Axon.get_inputs()
            |> Map.new(fn {k, v} -> {k, Nx.broadcast(0.0, v)} end)

          # get_output_shape/2 returns a template; pull the shape tuple.
          template = Axon.get_output_shape(inp, layer_inputs)
          mask_layer = Axon.constant(Nx.broadcast(0, Nx.shape(template)))

          axon
          |> Map.put(output_name, dropout_layer)
          |> Map.put(mask_name, mask_layer)
      end

    {updated_axon, params, used_params}
  end

  defp recur_nodes(
         %Node{op_type: "Pad", input: [inp_name], attribute: attrs, output: [output_name]},
         {axon, params, used_params}
       ) do
    pad_options = options!(attrs)

    inp = input!(inp_name, axon, params, used_params)

    mode = pad_options["mode"] || "constant"
    value = pad_options["value"] || 0.0
    pads = pad_options["pads"]

    updated_axon =
      case mode do
        "constant" ->
          config =
            pads
            |> Enum.count()
            |> then(&Enum.chunk_every(pads, div(&1, 2)))
            |> Enum.zip()
            |> Enum.map(fn {x, y} -> {x, y, 0} end)

          pad_layer = Axon.nx(inp, &Nx.pad(&1, value, config), op_name: :pad)
          Map.put(axon, output_name, pad_layer)

        other ->
          raise ArgumentError,
                "Pad mode #{inspect(other)} is not yet supported (only constant is)"
      end

    {updated_axon, params, used_params}
  end

  defp recur_nodes(
         %Node{
           op_type: "Pad",
           input: [inp_name, pad_name | extra_inputs],
           attribute: attrs,
           output: [output_name]
         },
         {axon, params, used_params}
       ) do
    pad_options = options!(attrs)

    inp = input!(inp_name, axon, params, used_params)
    pads_flat = constant!(pad_name, axon, params, used_params) |> Nx.to_flat_list()
    mode = pad_options["mode"] || "constant"

    {value_name, axes_name} =
      case extra_inputs do
        [] -> {nil, nil}
        [""] -> {nil, nil}
        [v] -> {v, nil}
        [v, ""] -> {v, nil}
        ["", a] -> {nil, a}
        [v, a] -> {v, a}
      end

    value =
      cond do
        is_nil(value_name) or value_name == "" -> 0
        true -> constant!(value_name, axon, params, used_params) |> Nx.to_number()
      end

    rank =
      case kernel_shape_from_axon!(inp) do
        shape when is_tuple(shape) -> tuple_size(shape)
      end

    axes =
      cond do
        is_nil(axes_name) or axes_name == "" ->
          Enum.to_list(0..(rank - 1))

        true ->
          constant!(axes_name, axon, params, used_params)
          |> Nx.to_flat_list()
          |> Enum.map(fn a -> if a < 0, do: a + rank, else: a end)
      end

    # ONNX flat-pads layout: [start_axis0, start_axis1, ..., start_axisK,
    # end_axis0, end_axis1, ..., end_axisK]. With an explicit axes list,
    # K = length(axes); otherwise K = rank.
    half = div(length(pads_flat), 2)
    {starts, ends} = Enum.split(pads_flat, half)

    full_pads =
      for ax <- 0..(rank - 1) do
        case Enum.find_index(axes, &(&1 == ax)) do
          nil -> {0, 0}
          idx -> {Enum.at(starts, idx), Enum.at(ends, idx)}
        end
      end

    pad_layer = build_pad_layer(inp, full_pads, value, mode, output_name)
    {Map.put(axon, output_name, pad_layer), params, used_params}
  end

  # Per-axis padding by mode. "constant" lowers to `Nx.pad`; the
  # boundary modes use `Nx.take` with a precomputed index list along
  # each padded axis (edge=clamp, reflect=mirror, wrap=modulo).
  defp build_pad_layer(inp, full_pads, value, "constant", output_name) do
    config = Enum.map(full_pads, fn {a, b} -> {a, b, 0} end)
    Axon.nx(inp, &Nx.pad(&1, value, config), name: output_name, op_name: :pad)
  end

  defp build_pad_layer(inp, full_pads, _value, mode, output_name)
       when mode in ["edge", "reflect", "wrap"] do
    fun = fn x ->
      Enum.with_index(full_pads)
      |> Enum.reduce(x, fn {{lo, hi}, axis}, acc ->
        if lo == 0 and hi == 0 do
          acc
        else
          dim = Nx.axis_size(acc, axis)
          indices = pad_mode_indices(mode, lo, hi, dim)
          Nx.take(acc, Nx.tensor(indices, type: {:s, 64}), axis: axis)
        end
      end)
    end

    Axon.nx(inp, fun, name: output_name, op_name: :pad)
  end

  defp pad_mode_indices("edge", lo, hi, dim) do
    List.duplicate(0, lo) ++ Enum.to_list(0..(dim - 1)) ++ List.duplicate(dim - 1, hi)
  end

  defp pad_mode_indices("reflect", lo, hi, dim) do
    # Reflect around index 0 / index dim-1 without duplicating the edge.
    # `lo` pre-pads with [lo, lo-1, ..., 1] and `hi` post-pads with
    # [dim-2, dim-3, ..., dim-1-hi]. Works for `lo < dim` and `hi < dim`
    # which the ONNX spec guarantees.
    Enum.map(lo..1//-1, & &1) ++
      Enum.to_list(0..(dim - 1)) ++
      Enum.map((dim - 2)..(dim - 1 - hi)//-1, & &1)
  end

  defp pad_mode_indices("wrap", lo, hi, dim) do
    Enum.map(0..(lo - 1), fn i -> Integer.mod(-lo + i, dim) end) ++
      Enum.to_list(0..(dim - 1)) ++
      Enum.map(0..(hi - 1), fn i -> Integer.mod(i, dim) end)
  end

  defp recur_nodes(
         %Node{op_type: "NonZero", input: [inp_name], output: [output_name]},
         {axon, params, used_params}
       ) do
    input = constant!(inp_name, axon, params, used_params)

    rank = Nx.rank(input)

    non_zero_indices =
      Enum.reduce(0..(rank - 1), [], fn axis, indices ->
        before_perm = Enum.to_list(0..(axis - 1)//1)
        after_perm = Enum.to_list((axis + 1)..(rank - 1)//1)
        perm = [axis] ++ before_perm ++ after_perm

        tensor = Nx.transpose(input, axes: perm)

        non_zero_indices =
          tensor
          |> Nx.not_equal(0)
          |> Nx.select(Nx.iota(Nx.shape(tensor), axis: -1), -1)
          |> Nx.to_flat_list()
          |> Enum.filter(&(&1 > -1))

        [non_zero_indices | indices]
      end)

    output =
      non_zero_indices
      |> Nx.tensor()
      |> Axon.constant(name: output_name)

    updated_axon = Map.put(axon, output_name, output)

    {updated_axon, params, used_params}
  end

  defp recur_nodes(%Node{op_type: unsupported}, _) do
    raise ArgumentError, "unsupported #{inspect(unsupported)}"
  end

  def tensor!(%Tensor{data_location: :EXTERNAL, data_type: dtype, dims: dims, external_data: data}) do
    data_options =
      Enum.reduce(data, %{}, fn
        %Onnx.StringStringEntryProto{key: key, value: value}, acc -> Map.put(acc, key, value)
      end)

    location = data_options["location"]

    case File.read(location) do
      {:ok, bytes} ->
        shape = List.to_tuple(dims)
        to_nx_tensor([], bytes, onnx_type_to_nx_type(dtype), shape)

      _error ->
        raise ArgumentError,
              "could not find external data at #{location}," <>
                " you must ensure path to location is correct" <>
                " relative to your current working directory"
    end
  end

  def tensor!(%Tensor{data_type: dtype, dims: dims} = tensor) do
    shape = List.to_tuple(dims)

    case dtype do
      1 ->
        to_nx_tensor(tensor.float_data, tensor.raw_data, {:f, 32}, shape)

      2 ->
        to_nx_tensor(tensor.int32_data, tensor.raw_data, {:u, 8}, shape)

      3 ->
        to_nx_tensor(tensor.int32_data, tensor.raw_data, {:s, 8}, shape)

      4 ->
        to_nx_tensor(tensor.int32_data, tensor.raw_data, {:u, 16}, shape)

      5 ->
        to_nx_tensor(tensor.int32_data, tensor.raw_data, {:s, 16}, shape)

      6 ->
        to_nx_tensor(tensor.int32_data, tensor.raw_data, {:s, 32}, shape)

      7 ->
        to_nx_tensor(tensor.int64_data, tensor.raw_data, {:s, 64}, shape)

      8 ->
        raise "unsupported Nx tensor type: string"

      9 ->
        to_nx_tensor(tensor.int32_data, tensor.raw_data, {:u, 8}, shape)

      10 ->
        to_nx_tensor(tensor.int32_data, tensor.raw_data, {:f, 16}, shape)

      11 ->
        to_nx_tensor(tensor.double_data, tensor.raw_data, {:f, 64}, shape)

      12 ->
        to_nx_tensor(tensor.uint64_data, tensor.raw_data, {:u, 32}, shape)

      13 ->
        to_nx_tensor(tensor.uint64_data, tensor.raw_data, {:u, 64}, shape)

      14 ->
        # TODO(seanmor5): When complex is supported, tensor.float_data
        raise "unsupported Nx tensor type: C64"

      15 ->
        # TODO(seanmor5): When complex is supported, tensor.double_data
        raise "unsupported Nx tensor type: C128"

      16 ->
        to_nx_tensor([], tensor.raw_data, {:bf, 16}, shape)
    end
  end

  defp to_nx_tensor([], <<>>, _, _) do
    raise "unsupported empty Nx tensor"
  end

  defp to_nx_tensor([], raw, type, shape) do
    raw
    |> Nx.from_binary(type)
    |> Nx.reshape(shape)
  end

  defp to_nx_tensor(data, _, type, shape) do
    data
    |> Enum.map(&normalize_special_float/1)
    |> Nx.tensor(type: type)
    |> Nx.reshape(shape)
  end

  # Protobuf decodes IEEE-754 special floats as atoms with dashes
  # (`:"-infinity"`, `:infinity`, `:nan`), but Nx accepts only its own
  # spellings (`:neg_infinity`, `:infinity`, `:nan`). Normalise both here so
  # constants like Attention's `FloatNegInf` round-trip through `Nx.tensor`.
  defp normalize_special_float(:"-infinity"), do: :neg_infinity
  defp normalize_special_float(:"+infinity"), do: :infinity
  defp normalize_special_float(:"-inf"), do: :neg_infinity
  defp normalize_special_float(:"+inf"), do: :infinity
  defp normalize_special_float(:inf), do: :infinity
  defp normalize_special_float(:Infinity), do: :infinity
  defp normalize_special_float(:NaN), do: :nan
  defp normalize_special_float(other), do: other

  defp axon!(name, axon) do
    if Map.has_key?(axon, name) do
      axon[name]
    else
      raise ArgumentError,
            "unable to build model from ONNX graph, expected value #{name}" <>
              " to be a graph input, but it was not present in built" <>
              " graphs"
    end
  end

  defp param!(name, params) do
    if Map.has_key?(params, name) do
      params[name]
    else
      raise ArgumentError,
            "unable to build model from ONNX graph, expected value #{name}" <>
              " to be a parameter input, but it was not present in" <>
              " initializers"
    end
  end

  defp input!(name, axon, params, used_params) do
    cond do
      Map.has_key?(axon, name) ->
        axon[name]

      Map.has_key?(params, name) ->
        params[name]

      Map.has_key?(used_params, name) ->
        used_params[name]

      true ->
        raise ArgumentError, "#{name} was not present in graph or initializers"
    end
  end

  defp constant!(name, axon, params, used_params) do
    cond do
      Map.has_key?(axon, name) ->
        case get_axon_node(axon[name]) do
          %Axon.Node{op: :constant, opts: [value: shape]} ->
            shape

          %Axon.Node{op_name: op} ->
            raise ArgumentError,
                  "unable to build model from ONNX graph, expected value #{name}" <>
                    " to be constant value, but was #{inspect(op)}"
        end

      Map.has_key?(params, name) ->
        params[name]

      Map.has_key?(used_params, name) ->
        used_params[name]

      true ->
        raise ArgumentError,
              "unable to build model from ONNX graph, could not find constant" <>
                " value #{name} in subgraphs or parameters"
    end
  end

  defp padding!(auto_pad, pads, _kernel_size, _strides) do
    case auto_pad do
      val when val == "NOTSET" or val == nil ->
        case pads do
          pads when is_list(pads) ->
            pads
            |> Enum.count()
            |> then(&Enum.chunk_every(pads, div(&1, 2)))
            |> Enum.zip()

          nil ->
            :valid
        end

      val when val == "SAME_UPPER" ->
        :same

      val when val == "SAME_LOWER" ->
        # SAME_LOWER asymmetrically pads the LOWER (start) side when the
        # padding amount is odd; Axon's `:same` is SAME_UPPER. Computing the
        # explicit per-axis padding requires the input shape, which we don't
        # have here, so we raise rather than silently fall back to
        # SAME_UPPER (the prior behaviour produced wrong outputs without any
        # signal). A future fix would plumb the input shape through and
        # build the per-axis {lo, hi} tuple.
        raise ArgumentError,
              "auto_pad=SAME_LOWER is not yet supported; only SAME_UPPER " <>
                "is correctly lowered. Patch deserialize.ex:padding!/4."

      "VALID" ->
        :valid
    end
  end

  defp options!(attrs) when is_list(attrs) do
    Enum.reduce(attrs, %{}, fn %Attribute{type: type, name: name} = attr, options ->
      case type do
        :FLOAT ->
          Map.put(options, name, attr.f)

        :INT ->
          Map.put(options, name, attr.i)

        :STRING ->
          Map.put(options, name, attr.s)

        :TENSOR ->
          Map.put(options, name, attr.t)

        :GRAPH ->
          Map.put(options, name, attr.g)

        :SPARSE_TENSOR ->
          Map.put(options, name, attr.sparse_tensor)

        :TYPE_PROTO ->
          Map.put(options, name, attr.tp)

        :FLOATS ->
          Map.put(options, name, attr.floats)

        :INTS ->
          Map.put(options, name, attr.ints)

        :STRINGS ->
          Map.put(options, name, attr.strings)

        :TENSORS ->
          Map.put(options, name, attr.tensors)

        :GRAPHS ->
          Map.put(options, name, attr.graphs)

        :SPARSE_TENSORS ->
          Map.put(options, name, attr.sparse_tensors)

        :TYPE_PROTOS ->
          Map.put(options, name, attr.type_protos)
      end
    end)
  end

  defp build_trilu_layer(inp, k, upper, output_name, axon, params, used_params)
       when is_integer(k) and is_boolean(upper) do
    case get_axon_node(inp) do
      %Axon.Node{op: :constant, opts: [value: v]} ->
        new_value = do_trilu(v, Nx.tensor(k, type: {:s, 64}), upper)
        layer = Axon.constant(new_value, name: output_name)
        {Map.put(axon, output_name, layer), params, used_params}

      %Nx.Tensor{} = v ->
        new_value = do_trilu(v, Nx.tensor(k, type: {:s, 64}), upper)
        layer = Axon.constant(new_value, name: output_name)
        {Map.put(axon, output_name, layer), params, used_params}

      %Axon.Node{} ->
        fun = fn x, _opts -> do_trilu(x, Nx.tensor(k, type: {:s, 64}), upper) end
        layer = Axon.layer(fun, [inp], name: output_name, op_name: :trilu)
        {Map.put(axon, output_name, layer), params, used_params}
    end
  end

  defp do_nll_loss(input, target, weight, ignore_index, reduction) do
    target_i = Nx.as_type(target, {:s, 64})

    # When ignore_index is a value outside [0, C), Nx.take_along_axis would
    # raise. Substitute a safe in-range value at those positions and rely on
    # the mask to zero them out below.
    safe_target =
      if ignore_index do
        keep = Nx.not_equal(target_i, ignore_index)
        Nx.select(keep, target_i, Nx.tensor(0, type: {:s, 64}))
      else
        target_i
      end

    expanded_target = Nx.new_axis(safe_target, 1)
    gathered = Nx.take_along_axis(input, expanded_target, axis: 1)
    loss_at_target = Nx.squeeze(gathered, axes: [1])
    neg_loss = Nx.negate(loss_at_target)

    {weighted_loss, weight_at_target} =
      if weight do
        w_at = Nx.take(weight, safe_target)
        {Nx.multiply(neg_loss, w_at), w_at}
      else
        {neg_loss, Nx.broadcast(Nx.tensor(1, type: Nx.type(neg_loss)), Nx.shape(neg_loss))}
      end

    {masked_loss, masked_weights} =
      if ignore_index do
        keep = Nx.not_equal(target_i, ignore_index)
        keep_t = Nx.as_type(keep, Nx.type(weighted_loss))
        {Nx.multiply(weighted_loss, keep_t), Nx.multiply(weight_at_target, keep_t)}
      else
        {weighted_loss, weight_at_target}
      end

    case reduction do
      "none" ->
        masked_loss

      "sum" ->
        Nx.sum(masked_loss)

      "mean" ->
        Nx.divide(Nx.sum(masked_loss), Nx.sum(masked_weights))
    end
  end

  defp qlinear_conv_impl(
         x, x_scale, x_zp, w, w_scale, w_zp, y_scale, y_zp, bias,
         strides, padding_config, dilations, group, target_type, min_v, max_v
       ) do
    work_type = Nx.type(x_scale)

    x_f =
      Nx.multiply(
        Nx.subtract(Nx.as_type(x, work_type), Nx.as_type(x_zp, work_type)),
        x_scale
      )

    w_f =
      Nx.multiply(
        Nx.subtract(Nx.as_type(w, work_type), Nx.as_type(w_zp, work_type)),
        w_scale
      )

    y_f =
      Axon.Layers.conv(x_f, w_f, 0,
        strides: strides,
        padding: padding_config,
        kernel_dilation: dilations,
        feature_group_size: group,
        channels: :first
      )

    y_f =
      if bias do
        # Bias is int32 quantised by x_scale * w_scale; convert to float
        # using that combined scale.
        combined_scale = Nx.multiply(x_scale, w_scale)
        bias_f = Nx.multiply(Nx.as_type(bias, work_type), combined_scale)
        Nx.add(y_f, Nx.reshape(bias_f, conv_bias_broadcast_shape(Nx.shape(y_f), Nx.shape(bias_f))))
      else
        y_f
      end

    # Like QLinearMatMul, QLinearConv per spec wraps on output overflow
    # rather than saturating. The min_v / max_v range is unused here but
    # the helper is shared with QuantizeLinear which does saturate.
    _ = {min_v, max_v}

    y_f
    |> Nx.divide(y_scale)
    |> Nx.round()
    |> Nx.add(Nx.as_type(y_zp, work_type))
    |> Nx.as_type(target_type)
  end

  defp conv_bias_broadcast_shape(out_shape, bias_shape) do
    # Convolution bias is (C_out,); broadcast against y of shape
    # (N, C_out, ...spatial) by inserting size-1 dims everywhere except
    # the channel axis.
    case bias_shape do
      {_} ->
        rank = tuple_size(out_shape)
        List.duplicate(1, rank) |> List.to_tuple() |> put_elem(1, elem(bias_shape, 0))

      _ ->
        bias_shape
    end
  end

  defp integer_matmul_layer(a, b, nil, nil) do
    fun = fn a, b, _opts ->
      Nx.dot(Nx.as_type(a, {:s, 32}), Nx.as_type(b, {:s, 32}))
    end

    {fun, [a, b]}
  end

  defp integer_matmul_layer(a, b, a_zp, nil) when not is_nil(a_zp) do
    fun = fn a, b, a_zp, _opts ->
      centered_a =
        Nx.subtract(Nx.as_type(a, {:s, 32}), Nx.as_type(a_zp, {:s, 32}))

      Nx.dot(centered_a, Nx.as_type(b, {:s, 32}))
    end

    {fun, [a, b, a_zp]}
  end

  defp integer_matmul_layer(a, b, nil, b_zp) when not is_nil(b_zp) do
    fun = fn a, b, b_zp, _opts ->
      centered_b =
        Nx.subtract(Nx.as_type(b, {:s, 32}), Nx.as_type(b_zp, {:s, 32}))

      Nx.dot(Nx.as_type(a, {:s, 32}), centered_b)
    end

    {fun, [a, b, b_zp]}
  end

  defp integer_matmul_layer(a, b, a_zp, b_zp) when not is_nil(a_zp) and not is_nil(b_zp) do
    fun = fn a, b, a_zp, b_zp, _opts ->
      centered_a = Nx.subtract(Nx.as_type(a, {:s, 32}), Nx.as_type(a_zp, {:s, 32}))
      centered_b = Nx.subtract(Nx.as_type(b, {:s, 32}), Nx.as_type(b_zp, {:s, 32}))
      Nx.dot(centered_a, centered_b)
    end

    {fun, [a, b, a_zp, b_zp]}
  end

  # The fun closes over the zero-points presence pattern so the runtime
  # layer fn has the right arity.
  defp build_conv_integer_fun(nil, nil, strides, padding, dilations, group) do
    fn x, w, _opts ->
      Axon.Layers.conv(Nx.as_type(x, {:s, 32}), Nx.as_type(w, {:s, 32}), 0,
        strides: strides,
        padding: padding,
        kernel_dilation: dilations,
        feature_group_size: group,
        channels: :first
      )
    end
  end

  defp build_conv_integer_fun(_x_zp, nil, strides, padding, dilations, group) do
    fn x, w, x_zp, _opts ->
      centered_x = Nx.subtract(Nx.as_type(x, {:s, 32}), Nx.as_type(x_zp, {:s, 32}))

      Axon.Layers.conv(centered_x, Nx.as_type(w, {:s, 32}), 0,
        strides: strides,
        padding: padding,
        kernel_dilation: dilations,
        feature_group_size: group,
        channels: :first
      )
    end
  end

  defp build_conv_integer_fun(_x_zp, _w_zp, strides, padding, dilations, group) do
    fn x, w, x_zp, w_zp, _opts ->
      centered_x = Nx.subtract(Nx.as_type(x, {:s, 32}), Nx.as_type(x_zp, {:s, 32}))
      centered_w = Nx.subtract(Nx.as_type(w, {:s, 32}), Nx.as_type(w_zp, {:s, 32}))

      Axon.Layers.conv(centered_x, centered_w, 0,
        strides: strides,
        padding: padding,
        kernel_dilation: dilations,
        feature_group_size: group,
        channels: :first
      )
    end
  end

  defp kernel_shape_from_axon!(%Axon{} = node) do
    layer_inputs =
      node
      |> Axon.get_inputs()
      |> Map.new(fn {k, v} -> {k, Nx.broadcast(0.0, v)} end)

    # Axon 0.8 returns a template tensor; pull the shape.
    Axon.get_output_shape(node, layer_inputs) |> Nx.shape()
  end

  defp broadcast_q_params(scale, zp, x, axis) do
    case Nx.shape(scale) do
      {} ->
        {scale, zp}

      {n} ->
        x_rank = Nx.rank(x)
        pos_axis = if axis < 0, do: x_rank + axis, else: axis

        new_shape =
          List.duplicate(1, x_rank)
          |> List.to_tuple()
          |> put_elem(pos_axis, n)

        zp = if zp, do: Nx.reshape(zp, new_shape), else: nil
        {Nx.reshape(scale, new_shape), zp}

      _ ->
        {scale, zp}
    end
  end

  defp quantize_target_type(zp_name, nil), do: quantize_target_type_fallback(zp_name)

  defp quantize_target_type(_zp_name, %Nx.Tensor{} = t), do: Nx.type(t)

  defp quantize_target_type(zp_name, %Axon{} = node) do
    case get_axon_node(node) do
      %Axon.Node{op: :constant, opts: [value: v]} -> Nx.type(v)
      _ -> quantize_target_type_fallback(zp_name)
    end
  end

  # When zero_point is a runtime graph input, Axon discards its declared
  # dtype. Consult the proto-time input_types side channel (see
  # `input_type/1`); fall back to u8 only if we really know nothing.
  defp quantize_target_type_fallback(nil), do: {:u, 8}

  defp quantize_target_type_fallback(name) do
    case input_type(name) do
      nil -> {:u, 8}
      type -> type
    end
  end

  defp quantize_range({:u, 8}), do: {0, 255}
  defp quantize_range({:s, 8}), do: {-128, 127}
  defp quantize_range({:u, 16}), do: {0, 65_535}
  defp quantize_range({:s, 16}), do: {-32_768, 32_767}
  defp quantize_range({:u, 32}), do: {0, 4_294_967_295}
  defp quantize_range({:s, 32}), do: {-2_147_483_648, 2_147_483_647}

  defp quantize_range(other),
    do: raise(ArgumentError, "QuantizeLinear: unsupported target dtype #{inspect(other)}")

  defp do_scatter_elements(data, indices, updates, axis, reduction) do
    rank = Nx.rank(data)
    pos_axis = if axis < 0, do: rank + axis, else: axis

    indices_shape = Nx.shape(indices)
    indices = Nx.as_type(indices, {:s, 64})
    dim_size = Nx.axis_size(data, pos_axis)
    indices = Nx.select(Nx.less(indices, 0), Nx.add(indices, dim_size), indices)

    coord_tensors =
      for k <- 0..(rank - 1) do
        if k == pos_axis do
          indices
        else
          Nx.iota(indices_shape, axis: k, type: {:s, 64})
        end
      end

    flat_coords =
      coord_tensors
      |> Nx.stack(axis: -1)
      |> Nx.reshape({:auto, rank})

    flat_updates = Nx.flatten(updates)

    case reduction do
      "none" ->
        Nx.indexed_put(data, flat_coords, flat_updates)

      "add" ->
        Nx.indexed_add(data, flat_coords, flat_updates)

      reduction when reduction in ["mul", "max", "min"] ->
        # Nx has no indexed_{mul,max,min}; fold each update sequentially so
        # repeated indices compose correctly per the ONNX spec. The number
        # of updates is statically known from the shape — Enum.reduce here
        # unrolls into N scatter ops at trace time.
        k = Nx.axis_size(flat_updates, 0)
        reduce_op = scatter_reduce_op(reduction)

        Enum.reduce(0..(k - 1), data, fn i, acc ->
          coord_row =
            flat_coords
            |> Nx.slice_along_axis(i, 1, axis: 0)

          update_scalar =
            flat_updates
            |> Nx.slice_along_axis(i, 1, axis: 0)
            |> Nx.reshape({1})

          current = Nx.gather(acc, coord_row) |> Nx.reshape({1})
          new_val = reduce_op.(current, update_scalar)
          Nx.indexed_put(acc, coord_row, new_val)
        end)
    end
  end

  defp scatter_reduce_op("mul"), do: &Nx.multiply/2
  defp scatter_reduce_op("max"), do: &Nx.max/2
  defp scatter_reduce_op("min"), do: &Nx.min/2

  # ScatterND: indices is `[..., q]`, each row addresses the first q axes
  # of `data` and the corresponding update slice has the remaining shape.
  # We flatten the leading "row" axes of indices/updates, expand each
  # q-tuple into per-cell coordinates spanning the addressed slice, and
  # dispatch to indexed_put / indexed_add / a sequential fold for the
  # other reductions (which must compose duplicate indices correctly).
  defp do_scatter_nd(data, indices, updates, reduction) do
    indices = Nx.as_type(indices, {:s, 64})
    rank = Nx.rank(data)
    data_shape = Nx.shape(data)

    indices_shape = Nx.shape(indices)
    indices_rank = tuple_size(indices_shape)
    q = elem(indices_shape, indices_rank - 1)
    leading_dims = for i <- 0..(indices_rank - 2), do: elem(indices_shape, i)
    num_rows = Enum.reduce(leading_dims, 1, &Kernel.*/2)

    flat_indices = Nx.reshape(indices, {num_rows, q})

    slice_dims = for i <- q..(rank - 1), do: elem(data_shape, i)
    slice_size = Enum.reduce(slice_dims, 1, &Kernel.*/2)

    # Build the inner-cell coordinate grid once; each "row" of indices
    # broadcasts across this grid to produce slice_size full coordinates.
    inner_coords =
      case slice_dims do
        [] ->
          # q == rank: each row already addresses a scalar cell.
          nil

        _ ->
          # Tensor of shape {slice_size, rank - q}: every coordinate of
          # the addressed sub-slice, in row-major order.
          coord_tensors =
            for axis_within_slice <- 0..(length(slice_dims) - 1) do
              Nx.iota(List.to_tuple(slice_dims), axis: axis_within_slice, type: {:s, 64})
            end

          coord_tensors
          |> Nx.stack(axis: -1)
          |> Nx.reshape({slice_size, length(slice_dims)})
      end

    # Expand each row of flat_indices to slice_size copies, then
    # concatenate the inner coordinates.
    full_coords =
      case inner_coords do
        nil ->
          flat_indices

        _ ->
          # row_coords: {num_rows, slice_size, q}
          row_coords =
            flat_indices
            |> Nx.new_axis(1)
            |> Nx.broadcast({num_rows, slice_size, q})

          # inner_broadcast: {num_rows, slice_size, rank - q}
          inner_broadcast =
            inner_coords
            |> Nx.new_axis(0)
            |> Nx.broadcast({num_rows, slice_size, length(slice_dims)})

          Nx.concatenate([row_coords, inner_broadcast], axis: -1)
          |> Nx.reshape({num_rows * slice_size, rank})
      end

    flat_updates = Nx.reshape(updates, {num_rows * slice_size})

    case reduction do
      "none" -> Nx.indexed_put(data, full_coords, flat_updates)
      "add" -> Nx.indexed_add(data, full_coords, flat_updates)
      reduction when reduction in ["mul", "max", "min"] ->
        # Sequential fold so duplicate indices compose per spec.
        reduce_op = scatter_reduce_op(reduction)
        n = num_rows * slice_size

        Enum.reduce(0..(n - 1), data, fn i, acc ->
          coord_row = Nx.slice_along_axis(full_coords, i, 1, axis: 0)
          update_scalar =
            flat_updates |> Nx.slice_along_axis(i, 1, axis: 0) |> Nx.reshape({1})

          current = Nx.gather(acc, coord_row) |> Nx.reshape({1})
          new_val = reduce_op.(current, update_scalar)
          Nx.indexed_put(acc, coord_row, new_val)
        end)
    end
  end

  # GatherND: index into the first `q = last_dim(indices)` axes of data
  # (after stripping `batch_dims` leading axes). The result shape is
  # `indices.shape[:-1] ++ data.shape[batch_dims + q:]`.
  defp do_gather_nd(data, indices, batch_dims) do
    data_shape = Nx.shape(data)
    indices_shape = Nx.shape(indices)
    indices_rank = tuple_size(indices_shape)
    q = elem(indices_shape, indices_rank - 1)
    data_rank = tuple_size(data_shape)
    addressed_rank = batch_dims + q

    if batch_dims == 0 do
      # Flatten leading dims of indices to a {N, q} list, then gather
      # each row as a slice of data.
      leading = Tuple.to_list(indices_shape) |> Enum.drop(-1)
      n = Enum.reduce(leading, 1, &Kernel.*/2)
      flat_idx = Nx.reshape(indices, {n, q})

      slice_dims = for i <- addressed_rank..(data_rank - 1)//1, do: elem(data_shape, i)
      slice_size = Enum.reduce(slice_dims, 1, &Kernel.*/2)

      flat_rows =
        Enum.reduce(0..(addressed_rank - 1)//1, 1, fn i, acc -> acc * elem(data_shape, i) end)

      flat_data = Nx.reshape(data, {flat_rows, slice_size})

      # Convert each q-tuple to a flat row-index by computing per-axis
      # strides in row-major order.
      strides =
        for i <- 0..(q - 1)//1 do
          Enum.reduce((i + 1)..(addressed_rank - 1)//1, 1, fn j, acc ->
            acc * elem(data_shape, j)
          end)
        end

      strides_t = Nx.tensor(strides, type: {:s, 64})

      flat_indices = Nx.dot(flat_idx, strides_t)

      gathered = Nx.take(flat_data, flat_indices, axis: 0)

      out_shape = List.to_tuple(leading ++ slice_dims)
      Nx.reshape(gathered, out_shape)
    else
      # With batch_dims, we gather per-batch. The corpus only exercises
      # batch_dims=0; raise a clear error if encountered for now.
      raise ArgumentError, "GatherND with batch_dims=#{batch_dims} is not yet supported"
    end
  end

  defp unique_with_layouts(flat, sorted) do
    # `flat` is the input in flattened first-occurrence order. We need:
    # * `values`: the unique values (optionally sorted)
    # * `indices`: positions of each unique value in the original input
    #   (first occurrence)
    # * `inverse`: for each original position, the index of its value in
    #   `values`
    # * `counts`: how many times each unique value appears
    indexed = Enum.with_index(flat)

    {first_occurrences, _seen} =
      Enum.reduce(indexed, {[], MapSet.new()}, fn {v, i}, {acc, seen} ->
        if MapSet.member?(seen, v) do
          {acc, seen}
        else
          {[{v, i} | acc], MapSet.put(seen, v)}
        end
      end)

    first_occurrences = Enum.reverse(first_occurrences)

    ordered =
      if sorted do
        Enum.sort_by(first_occurrences, fn {v, _i} -> v end)
      else
        first_occurrences
      end

    values = Enum.map(ordered, fn {v, _i} -> v end)
    idx = Enum.map(ordered, fn {_v, i} -> i end)

    value_to_position =
      ordered
      |> Enum.with_index()
      |> Map.new(fn {{v, _i}, pos} -> {v, pos} end)

    inverse = Enum.map(flat, fn v -> Map.fetch!(value_to_position, v) end)
    counts = Enum.map(values, fn v -> Enum.count(flat, &(&1 == v)) end)

    {values, idx, inverse, counts}
  end

  defp do_layer_norm(x, scale, bias, axis, epsilon) do
    {y, _mean, _inv_std} = layer_norm_parts(x, axis, epsilon)
    y = Nx.multiply(y, scale)

    if bias do
      Nx.add(y, bias)
    else
      y
    end
  end

  defp do_layer_norm_mean(x, axis) do
    {_y, mean, _inv_std} = layer_norm_parts(x, axis, 0.0)
    mean
  end

  defp do_layer_norm_inv_std(x, axis, epsilon) do
    {_y, _mean, inv_std} = layer_norm_parts(x, axis, epsilon)
    inv_std
  end

  defp layer_norm_parts(x, axis, epsilon) do
    rank = Nx.rank(x)
    pos_axis = if axis < 0, do: rank + axis, else: axis
    axes = Enum.to_list(pos_axis..(rank - 1)//1)

    mean = Nx.mean(x, axes: axes, keep_axes: true)
    centered = Nx.subtract(x, mean)
    var = Nx.mean(Nx.pow(centered, 2), axes: axes, keep_axes: true)
    inv_std = Nx.rsqrt(Nx.add(var, epsilon))
    {Nx.multiply(centered, inv_std), mean, inv_std}
  end

  defp do_trilu(x, k, upper) do
    shape = Nx.shape(x)
    rank = tuple_size(shape)
    rows = elem(shape, rank - 2)
    cols = elem(shape, rank - 1)

    row_idx = Nx.iota({rows, 1}, type: {:s, 64})
    col_idx = Nx.iota({1, cols}, type: {:s, 64})
    diff = Nx.subtract(col_idx, row_idx)

    mask2d =
      if upper,
        do: Nx.greater_equal(diff, k),
        else: Nx.less_equal(diff, k)

    # For inputs with leading batch dims, broadcast the {rows, cols}
    # mask up to the full input rank — Nx.select doesn't broadcast a
    # lower-rank pred against the data tensor.
    mask = Nx.broadcast(mask2d, shape)
    Nx.select(mask, x, Nx.tensor(0, type: Nx.type(x)))
  end

  defp shape!(%Placeholder{shape: %Shape{dim: dims}}, dim_params) do
    dims
    |> Enum.map(fn %Dimension{value: value} ->
      case value do
        {:dim_value, val} ->
          val

        {:dim_param, key} ->
          param = Keyword.get(dim_params, String.to_atom(key))

          unless param do
            Logger.warning("#{key} has no specified dimension, assuming nil")
          end

          param

        _ ->
          raise ArgumentError, "unsupported dimension type"
      end
    end)
    |> List.to_tuple()
  end
end
