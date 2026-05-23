defmodule AxonOnnx.Coverage.RoundTripRegistry do
  @moduledoc """
  Expected-failure registry for the bidirectional round-trip test
  (import → export → re-import → predict). Mirrors `AxonOnnx.Coverage.Registry`
  in shape and semantics — `{category, name} => {status, note}`, defaults to
  `:unsupported`, fails loudly on drift in either direction.

  A case is only run through round-trip if its import-side status in
  `AxonOnnx.Coverage.Registry` is `:passing`. Cases that don't import yet
  are not candidates here.

  This registry is the visible measure of how much of the bidirectional
  Nx/Axon ⇄ ONNX path actually works end-to-end.
  """

  @default {:unsupported, nil}

  @entries %{
    {"node", "test_abs"} => {:passing, nil},
    {"node", "test_acos"} => {:passing, nil},
    {"node", "test_acos_example"} => {:passing, nil},
    {"node", "test_acosh"} => {:passing, nil},
    {"node", "test_acosh_example"} => {:passing, nil},
    {"node", "test_asin"} => {:passing, nil},
    {"node", "test_asin_example"} => {:passing, nil},
    {"node", "test_asinh"} => {:passing, nil},
    {"node", "test_asinh_example"} => {:passing, nil},
    {"node", "test_atan"} => {:passing, nil},
    {"node", "test_atan_example"} => {:passing, nil},
    {"node", "test_atanh"} => {:passing, nil},
    {"node", "test_atanh_example"} => {:passing, nil},
    {"node", "test_averagepool_1d_default"} => {:passing, nil},
    {"node", "test_averagepool_2d_default"} => {:passing, nil},
    {"node", "test_averagepool_2d_pads_count_include_pad"} => {:passing, nil},
    {"node", "test_averagepool_2d_precomputed_pads_count_include_pad"} => {:passing, nil},
    {"node", "test_averagepool_2d_precomputed_strides"} => {:passing, nil},
    {"node", "test_averagepool_2d_strides"} => {:passing, nil},
    {"node", "test_averagepool_3d_default"} => {:passing, nil},
    {"node", "test_ceil"} => {:passing, nil},
    {"node", "test_ceil_example"} => {:passing, nil},
    {"node", "test_celu"} => {:passing, nil},
    {"node", "test_clip_default_inbounds_expanded"} => {:passing, nil},
    {"node", "test_clip_default_int8_inbounds_expanded"} => {:passing, nil},
    {"node", "test_constant"} => {:passing, nil},
    {"node", "test_cos"} => {:passing, nil},
    {"node", "test_cos_example"} => {:passing, nil},
    {"node", "test_cosh"} => {:passing, nil},
    {"node", "test_cosh_example"} => {:passing, nil},
    {"node", "test_dropout_random_old"} => {:passing, nil},
    {"node", "test_elu_default"} => {:passing, nil},
    {"node", "test_erf"} => {:passing, nil},
    {"node", "test_exp"} => {:passing, nil},
    {"node", "test_exp_example"} => {:passing, nil},
    {"node", "test_floor"} => {:passing, nil},
    {"node", "test_floor_example"} => {:passing, nil},
    {"node", "test_globalaveragepool"} => {:passing, nil},
    {"node", "test_globalaveragepool_precomputed"} => {:passing, nil},
    {"node", "test_globalmaxpool"} => {:passing, nil},
    {"node", "test_globalmaxpool_precomputed"} => {:passing, nil},
    {"node", "test_hardsigmoid_default"} => {:passing, nil},
    {"node", "test_hardswish"} => {:passing, nil},
    {"node", "test_identity"} => {:passing, nil},
    {"node", "test_isinf"} => {:passing, nil},
    {"node", "test_isinf_float16"} => {:passing, nil},
    {"node", "test_isnan"} => {:passing, nil},
    {"node", "test_isnan_float16"} => {:passing, nil},
    {"node", "test_leakyrelu_default"} => {:passing, nil},
    {"node", "test_log"} => {:passing, nil},
    {"node", "test_log_example"} => {:passing, nil},
    {"node", "test_maxpool_1d_default"} => {:passing, nil},
    {"node", "test_maxpool_2d_default"} => {:passing, nil},
    {"node", "test_maxpool_2d_pads"} => {:passing, nil},
    {"node", "test_maxpool_2d_precomputed_pads"} => {:passing, nil},
    {"node", "test_maxpool_2d_precomputed_same_upper"} => {:passing, nil},
    {"node", "test_maxpool_2d_precomputed_strides"} => {:passing, nil},
    {"node", "test_maxpool_2d_same_upper"} => {:passing, nil},
    {"node", "test_maxpool_2d_strides"} => {:passing, nil},
    {"node", "test_maxpool_2d_uint8"} => {:passing, nil},
    {"node", "test_maxpool_3d_default"} => {:passing, nil},
    {"node", "test_neg"} => {:passing, nil},
    {"node", "test_neg_example"} => {:passing, nil},
    {"node", "test_not_2d"} => {:passing, nil},
    {"node", "test_not_3d"} => {:passing, nil},
    {"node", "test_not_4d"} => {:passing, nil},
    {"node", "test_reciprocal"} => {:passing, nil},
    {"node", "test_reciprocal_example"} => {:passing, nil},
    {"node", "test_relu"} => {:passing, nil},
    {"node", "test_selu_default"} => {:passing, nil},
    {"node", "test_shape"} => {:passing, nil},
    {"node", "test_shape_clip_end"} => {:passing, nil},
    {"node", "test_shape_clip_start"} => {:passing, nil},
    {"node", "test_shape_end_1"} => {:passing, nil},
    {"node", "test_shape_end_negative_1"} => {:passing, nil},
    {"node", "test_shape_example"} => {:passing, nil},
    {"node", "test_shape_start_1"} => {:passing, nil},
    {"node", "test_shape_start_1_end_2"} => {:passing, nil},
    {"node", "test_shape_start_1_end_negative_1"} => {:passing, nil},
    {"node", "test_shape_start_negative_1"} => {:passing, nil},
    {"node", "test_sigmoid"} => {:passing, nil},
    {"node", "test_sigmoid_example"} => {:passing, nil},
    {"node", "test_sign"} => {:passing, nil},
    {"node", "test_sin"} => {:passing, nil},
    {"node", "test_sin_example"} => {:passing, nil},
    {"node", "test_sinh"} => {:passing, nil},
    {"node", "test_sinh_example"} => {:passing, nil},
    {"node", "test_size"} => {:passing, nil},
    {"node", "test_size_example"} => {:passing, nil},
    {"node", "test_softmax_axis_2"} => {:passing, nil},
    {"node", "test_softmax_default_axis"} => {:passing, nil},
    {"node", "test_softmax_example"} => {:passing, nil},
    {"node", "test_softmax_large_number"} => {:passing, nil},
    {"node", "test_softmax_negative_axis"} => {:passing, nil},
    {"node", "test_softplus"} => {:passing, nil},
    {"node", "test_softplus_example"} => {:passing, nil},
    {"node", "test_softsign"} => {:passing, nil},
    {"node", "test_softsign_example"} => {:passing, nil},
    {"node", "test_sqrt"} => {:passing, nil},
    {"node", "test_sqrt_example"} => {:passing, nil},
    {"node", "test_tan"} => {:passing, nil},
    {"node", "test_tan_example"} => {:passing, nil},
    {"node", "test_tanh"} => {:passing, nil},
    {"node", "test_tanh_example"} => {:passing, nil},
    {"pytorch-converted", "test_AvgPool2d"} => {:passing, nil},
    {"pytorch-converted", "test_AvgPool2d_stride"} => {:passing, nil},
    {"pytorch-converted", "test_AvgPool3d"} => {:passing, nil},
    {"pytorch-converted", "test_AvgPool3d_stride"} => {:passing, nil},
    {"pytorch-converted", "test_AvgPool3d_stride1_pad0_gpu_input"} => {:passing, nil},
    {"pytorch-converted", "test_Conv1d"} => {:passing, nil},
    {"pytorch-converted", "test_Conv1d_pad1"} => {:passing, nil},
    {"pytorch-converted", "test_Conv1d_pad1size1"} => {:passing, nil},
    {"pytorch-converted", "test_Conv1d_pad2"} => {:passing, nil},
    {"pytorch-converted", "test_Conv1d_pad2size1"} => {:passing, nil},
    {"pytorch-converted", "test_Conv1d_stride"} => {:passing, nil},
    {"pytorch-converted", "test_Conv2d"} => {:passing, nil},
    {"pytorch-converted", "test_Conv2d_no_bias"} => {:passing, nil},
    {"pytorch-converted", "test_Conv2d_padding"} => {:passing, nil},
    {"pytorch-converted", "test_Conv2d_strided"} => {:passing, nil},
    {"pytorch-converted", "test_Conv3d"} => {:passing, nil},
    {"pytorch-converted", "test_Conv3d_no_bias"} => {:passing, nil},
    {"pytorch-converted", "test_Conv3d_stride"} => {:passing, nil},
    {"pytorch-converted", "test_Conv3d_stride_padding"} => {:passing, nil},
    {"pytorch-converted", "test_LeakyReLU"} => {:passing, nil},
    {"pytorch-converted", "test_MaxPool1d"} => {:passing, nil},
    {"pytorch-converted", "test_MaxPool1d_stride"} => {:passing, nil},
    {"pytorch-converted", "test_MaxPool2d"} => {:passing, nil},
    {"pytorch-converted", "test_MaxPool3d"} => {:passing, nil},
    {"pytorch-converted", "test_MaxPool3d_stride"} => {:passing, nil},
    {"pytorch-converted", "test_MaxPool3d_stride_padding"} => {:passing, nil},
    {"pytorch-converted", "test_ReLU"} => {:passing, nil},
    {"pytorch-converted", "test_SELU"} => {:passing, nil},
    {"pytorch-converted", "test_Sigmoid"} => {:passing, nil},
    {"pytorch-converted", "test_Softmax"} => {:passing, nil},
    {"pytorch-converted", "test_Softmin"} => {:passing, nil},
    {"pytorch-converted", "test_Softplus"} => {:passing, nil},
    {"pytorch-converted", "test_Tanh"} => {:passing, nil},
    {"pytorch-converted", "test_softmax_functional_dim3"} => {:passing, nil},
    {"pytorch-converted", "test_softmax_lastdim"} => {:passing, nil},
    {"pytorch-operator", "test_operator_conv"} => {:passing, nil},
    {"pytorch-operator", "test_operator_exp"} => {:passing, nil},
    {"pytorch-operator", "test_operator_maxpool"} => {:passing, nil},
    {"pytorch-operator", "test_operator_selu"} => {:passing, nil},
    {"simple", "test_sign_model"} => {:passing, nil},
    {"simple", "test_single_relu_model"} => {:passing, nil},
  }

  @doc "Returns the raw `{category, name} => {status, note}` map."
  def entries, do: @entries

  @doc "Status for the given case key. Unlisted cases default to `:unsupported`."
  def status(key) do
    {status, _note} = Map.get(@entries, key, @default)
    status
  end

  @doc "Note for the given case key, or nil."
  def note(key) do
    {_status, note} = Map.get(@entries, key, @default)
    note
  end

  @doc "All keys explicitly present in the registry."
  def keys, do: Map.keys(@entries)
end
