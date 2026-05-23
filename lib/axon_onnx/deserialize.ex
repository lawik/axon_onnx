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

  def __load__(binary, opts \\ []) do
    binary
    |> Model.decode!()
    |> to_axon(opts)
  end

  defp to_axon(%Model{graph: %Graph{} = graph, opset_import: opset_imports}, dimensions) do
    opsets = build_opsets(opset_imports)
    input_types = build_input_types(graph)
    previous_opsets = Process.put(@opsets_key, opsets)
    previous_input_types = Process.put(@input_types_key, input_types)

    try do
      {graph, params} = graph_to_axon(graph, dimensions)

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
    end
  end

  defp restore_dict(key, nil), do: Process.delete(key)
  defp restore_dict(key, prev), do: Process.put(key, prev)

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

  def graph_to_axon(%Graph{node: nodes} = graph, dimensions) do
    params = get_params(graph)
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
    {"IsInf", &Nx.is_infinity/1},
    {"IsNaN", &Nx.is_nan/1},
    {"Log", &Nx.log/1},
    {"Neg", &Nx.negate/1},
    {"Not", &Nx.logical_not/1},
    {"Round", &Nx.round/1},
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

  @activation_op_types [
    {"Celu", :celu, [alpha: {"alpha", 1.0}]},
    {"Elu", :elu, [alpha: {"alpha", 1.0}]},
    {"Exp", :exp, []},
    {"HardSigmoid", :hard_sigmoid, [alpha: {"alpha", 0.2}, beta: {"beta", 0.5}]},
    {"LeakyRelu", :leaky_relu, [alpha: {"alpha", 1.0e-2}]},
    {"LogSoftmax", :log_softmax, [axis: {"axis", -1}]},
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
    {"Div", &Nx.divide/2, :divide},
    {"Equal", &Nx.equal/2, :equal},
    {"Greater", &Nx.greater/2, :greater},
    {"GreaterOrEqual", &Nx.greater_equal/2, :greater_equal},
    {"Less", &Nx.less/2, :less},
    {"LessOrEqual", &Nx.less_equal/2, :less_or_equal},
    {"Mod", &Nx.remainder/2, :mod},
    {"Or", &Nx.logical_or/2, :logical_or},
    {"Pow", &Nx.pow/2, :power},
    {"Xor", &Nx.logical_xor/2, :logical_xor}
  ]

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

    fun = fn x, y, _opts ->
      case bitshift_options["direction"] do
        "LEFT" -> Nx.left_shift(Nx.as_type(x, {:s, 64}), Nx.as_type(y, {:s, 64}))
        "RIGHT" -> Nx.right_shift(Nx.as_type(x, {:s, 64}), Nx.as_type(y, {:s, 64}))
      end
    end

    {updated_axon, updated_params} =
      case {get_axon_node(inp1), get_axon_node(inp2)} do
        {%Axon.Node{op: :constant, opts: [value: v1]},
         %Axon.Node{op: :constant, opts: [value: v2]}} ->
          new_value = apply(fun, [v1, v2, []])
          {Map.put(axon, output_name, Axon.constant(new_value, name: output_name)), used_params}

        {%Axon.Node{op: :constant, opts: [value: v1]}, %Nx.Tensor{} = v2} ->
          new_value = apply(fun, [v1, v2, []])
          {Map.put(axon, output_name, Axon.constant(new_value, name: output_name)), used_params}

        {%Nx.Tensor{} = v1, %Axon.Node{op: :constant, opts: [value: v2]}} ->
          new_value = apply(fun, [v1, v2, []])
          {Map.put(axon, output_name, Axon.constant(new_value, name: output_name)), used_params}

        {%Nx.Tensor{} = v1, %Nx.Tensor{} = v2} ->
          new_value = apply(fun, [v1, v2, []])
          {Map.put(axon, output_name, Axon.constant(new_value, name: output_name)), used_params}

        {%Axon.Node{}, %Axon.Node{}} ->
          layer = Axon.layer(fun, [inp1, inp2], name: output_name, op_name: :bitshift)
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
          axis = gather_options["axis"]
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
         %Node{op_type: "MaxPool", input: [inp], attribute: attrs, output: [output_name]},
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

    # Kernel size is a list of integers
    kernel_size = List.to_tuple(kernel_shape)

    # Axon only supports default ceil_mode right now
    if ceil_mode != 0 do
      raise ArgumentError,
            "invalid ceil_mode #{inspect(ceil_mode)}, Axon only supports" <>
              " ceil_mode of 0"
    end

    # Storage Order is not an Axon concern
    if storage_order do
      Logger.warning(
        "Storage order is not supported by Axon and is instead a backend-specific" <>
          " detail. Your model might behave differently from the imported version if" <>
          " the storage order differs"
      )
    end

    # Axon default strides are equal to the kernel shape (Keras behavior)
    # where as strides default to 1 in ONNX
    strides =
      if strides do
        strides
      else
        List.duplicate(1, tuple_size(kernel_size))
      end

    inp = axon!(inp, axon)

    # Compute padding from auto_pad and pads attributes
    padding_config = padding!(auto_pad, pads, kernel_size, strides)

    updated_axon =
      Map.put(
        axon,
        output_name,
        Axon.max_pool(inp,
          kernel_size: kernel_size,
          strides: strides,
          padding: padding_config,
          dilations: dilations,
          name: output_name,
          channels: :first
        )
      )

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
    _count_include_pad = avg_pool_options["count_include_pad"] || 0
    pads = avg_pool_options["pads"]
    strides = avg_pool_options["strides"] || 1
    dilations = avg_pool_options["dilations"] || 1

    # Kernel size is a list of integers
    kernel_size = List.to_tuple(kernel_shape)

    # Axon only supports default ceil_mode right now
    if ceil_mode != 0 do
      raise ArgumentError,
            "invalid ceil_mode #{inspect(ceil_mode)}, Axon only supports" <>
              " ceil_mode of 0"
    end

    # Axon only supports count_include_pad == 1
    # if count_include_pad != 1 do
    #   raise ArgumentError, "invalid count_include_pad #{inspect(count_include_pad)}," <>
    #                           " Axon only supports mode 1"
    # end

    # Axon default strides are equal to the kernel shape (Keras behavior)
    # where as strides default to 1 in ONNX
    strides =
      if strides do
        strides
      else
        List.duplicate(1, tuple_size(kernel_size))
      end

    inp = axon!(inp, axon)

    # Compute padding from auto_pad and pads attributes
    padding_config = padding!(auto_pad, pads, kernel_size, strides)

    updated_axon =
      Map.put(
        axon,
        output_name,
        Axon.avg_pool(inp,
          kernel_size: kernel_size,
          strides: strides,
          padding: padding_config,
          dilations: dilations,
          name: output_name,
          channels: :first
        )
      )

    {updated_axon, params, used_params}
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
    strides = conv_options["strides"]

    [inp_name, kernel_name | maybe_bias] = input

    inp = input!(inp_name, axon, params, used_params)
    kernel = input!(kernel_name, axon, params, used_params)

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

    {updated_axon, updated_params} =
      case {get_axon_node(input), get_axon_node(scale), get_axon_node(bias)} do
        {%Axon.Node{}, %Nx.Tensor{} = scale, %Nx.Tensor{} = bias} ->
          out = Axon.instance_norm(input, name: output_name, epsilon: options["epsilon"])

          updated_params =
            Map.put(used_params, output_name, %{
              "gamma" => scale,
              "beta" => bias,
              "mean" => Nx.tensor(0.0),
              "var" => Nx.tensor(1.0)
            })

          updated_axon = Map.put(axon, output_name, out)
          {updated_axon, updated_params}

        {%Axon.Node{}, %Axon.Node{}, %Axon.Node{}} ->
          out =
            instance_normalization(input, scale, bias,
              epsilon: options["epsilon"],
              name: output_name
            )

          updated_axon = Map.put(axon, output_name, out)
          {updated_axon, used_params}
      end

    {updated_axon, params, updated_params}
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
    # the indices tensor. We support reduction in {none, add} via
    # Nx.indexed_put / Nx.indexed_add. Reductions mul/min/max are not
    # implemented and raise from do_scatter_elements/5 below — those cases
    # stay :unsupported in the registry.
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
      Axon.Layers.lp_pool(x,
        kernel_size: kernel_shape,
        strides: strides,
        padding: padding_config,
        norm: p,
        channels: :first
      )
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
    # output has the same dtype as the input.
    axis = options!(attrs)["axis"] || -1
    input = input!(input_name, axon, params, used_params)

    fun = fn x, _opts ->
      argmax = Nx.argmax(x, axis: axis, keep_axis: true)
      iota = Nx.iota(Nx.shape(x), axis: axis)
      mask = Nx.equal(iota, argmax)
      Nx.as_type(mask, Nx.type(x))
    end

    layer = Axon.layer(fun, [input], name: output_name, op_name: :hardmax)
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
    %{"axis" => axis, "split" => split_sizes} = options!(attrs)

    split_layers = Axon.split(inp, split_sizes, axis: axis, name: output_names)

    updated_axon =
      split_layers
      |> Tuple.to_list()
      |> Enum.zip(output_names)
      |> Enum.reduce(axon, fn {output, name}, new_axon ->
        Map.put(new_axon, name, output)
      end)

    {updated_axon, params, used_params}
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
          Axon.constant(Nx.tensor(constant_options["value_float"], type: {:f, 32}),
            name: output_name
          )

        constant_options["value_floats"] ->
          Axon.constant(Nx.tensor(constant_options["value_floats"], type: {:f, 32}),
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
    value = tensor!(constant_options["value"])

    shape =
      shape
      |> constant!(axon, params, used_params)
      |> Nx.to_flat_list()
      |> Enum.map(fn
        -1 -> 1
        x -> x
      end)
      |> List.to_tuple()

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

    # Reshape is a constant value input that MUST be known
    # ahead of time so we can build a static graph, we can't
    # support any other reshape types
    shape = constant!(shape, axon, params, used_params)

    # We currently do not support zero sized dimensions
    if allowzero == 1 do
      Logger.warning(
        "Nx does not support zero-sized dimensions. If your reshape" <>
          " operation contains a zero-sized dimension, it will fail"
      )
    end

    new_shape =
      shape
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
    shape = constant!(shape, axon, params, used_params)

    shape =
      shape
      |> Nx.to_flat_list()
      |> Enum.map(fn
        -1 -> 1
        x -> x
      end)
      |> List.to_tuple()

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
    start = constant!(start, axon, params, used_params) |> Nx.to_number()
    limit = constant!(limit, axon, params, used_params) |> Nx.to_number()
    delta = constant!(delta, axon, params, used_params) |> Nx.to_number()

    number_of_elements = max(ceil(div(limit - start, delta)), 0)

    vals =
      for i <- 0..(number_of_elements - 1) do
        start + i * delta
      end

    updated_axon = Map.put(axon, output_name, Axon.constant(Nx.tensor(vals), name: output_name))
    {updated_axon, params, used_params}
  end

  defp recur_nodes(
         %Node{op_type: "Flatten", input: [inp], output: [output_name]},
         {axon, params, used_params}
       ) do
    inp = axon!(inp, axon)

    {Map.put(axon, output_name, Axon.flatten(inp, name: output_name)), params, used_params}
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

  defp closed_subgraph?(%Onnx.GraphProto{input: inputs}), do: inputs == []
  defp closed_subgraph?(nil), do: false

  defp eval_closed_subgraph(%Axon{} = axon, params) do
    {_init, predict} = Axon.build(axon)
    model_state = Axon.ModelState.new(params)
    predict.(model_state, %{}) |> Nx.backend_copy(Nx.BinaryBackend)
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
      case inp do
        %Axon.Node{op: :constant, opts: [value: v]} ->
          new_value = Nx.squeeze(v, axes: axes)
          layer = Axon.constant(new_value, name: output_name)
          Map.put(axon, output_name, layer)

        %Axon.Node{} = inp ->
          layer = Axon.layer(fun, [inp], name: output_name, op_name: :squeeze)
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

    axis = split_options["axis"]

    layers = Axon.split(inp, split, axis: axis, name: outputs)

    updated_axon =
      layers
      |> Tuple.to_list()
      |> Enum.zip(outputs)
      |> Enum.reduce(axon, fn {x, name}, acc -> Map.put(acc, name, x) end)

    {updated_axon, params, used_params}
  end

  defp recur_nodes(
         %Node{op_type: "EyeLike", input: [input], attribute: attrs, output: [output_name]},
         {axon, params, used_params}
       ) do
    eye_options = options!(attrs)

    inp = input!(input, axon, params, used_params)

    type = if eye_options["dtype"], do: onnx_type_to_nx_type(eye_options["dtype"]), else: {:f, 32}

    layer =
      case get_axon_node(inp) do
        %Axon.Node{op: :constant, opts: [value: v]} ->
          shape = Nx.shape(v)
          Axon.constant(Nx.eye(shape, type: type))

        %Axon.Node{} ->
          fun = fn x, _opts -> Nx.eye(Nx.shape(x), type: type) end
          Axon.layer(fun, [inp], name: output_name, op_name: :eye_like)

        %Nx.Tensor{} = t ->
          shape = Nx.shape(t)
          Axon.constant(Nx.eye(shape, type: type))
      end

    updated_axon = Map.put(axon, output_name, layer)
    {updated_axon, params, used_params}
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
      Nx.Random.key(seed)
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
            Nx.Random.key(seed)
            |> Nx.Random.uniform(low, high, type: nx_type, shape: shape)

          Axon.constant(tensor, name: output_name)

        %Axon.Node{} ->
          fun = fn x, _opts ->
            shape = Nx.shape(x)

            Nx.Random.key(seed)
            |> Nx.Random.uniform(low, high, type: nx_type, shape: shape)
            |> then(fn {tensor, _key} -> tensor end)
          end

          Axon.layer(fun, [inp], name: output_name, op_name: :random_uniform_like)

        %Nx.Tensor{} = t ->
          shape = Nx.shape(t)

          {tensor, _key} =
            Nx.Random.key(seed)
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
      Nx.Random.key(seed)
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
            Nx.Random.key(seed)
            |> Nx.Random.normal(mean, scale, type: nx_type, shape: shape)

          Axon.constant(tensor, name: output_name)

        %Axon.Node{} ->
          fun = fn x, _opts ->
            shape = Nx.shape(x)

            Nx.Random.key(seed)
            |> Nx.Random.normal(mean, scale, type: nx_type, shape: shape)
            |> then(fn {tensor, _key} -> tensor end)
          end

          Axon.layer(fun, [inp], name: output_name, op_name: :random_normal_like)

        %Nx.Tensor{} = t ->
          shape = Nx.shape(t)

          {tensor, _key} =
            Nx.Random.key(seed)
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
           input: [inp_name, pad_name | maybe_constant],
           attribute: attrs,
           output: [output_name]
         },
         {axon, params, used_params}
       ) do
    pad_options = options!(attrs)

    inp = input!(inp_name, axon, params, used_params)
    pads = constant!(pad_name, axon, params, used_params) |> Nx.to_flat_list()

    mode = pad_options["mode"] || "constant"

    updated_axon =
      case mode do
        "constant" ->
          value =
            case maybe_constant do
              [] ->
                0

              [""] ->
                0

              [value_name] ->
                constant!(value_name, axon, params, used_params) |> Nx.to_number()
            end

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
    |> Nx.tensor(type: type)
    |> Nx.reshape(shape)
  end

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

      other ->
        raise ArgumentError,
              "ScatterElements reduction=#{inspect(other)} is not yet supported"
    end
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

    mask =
      if upper,
        do: Nx.greater_equal(diff, k),
        else: Nx.less_equal(diff, k)

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
