defmodule AxonOnnx.RoundTripTest do
  @moduledoc """
  Bidirectional round-trip test. Walks every corpus case whose import status
  is `:passing` in `AxonOnnx.Coverage.Registry`, exports the imported Axon
  model back to ONNX bytes via `AxonOnnx.dump/4`, re-imports those bytes via
  `AxonOnnx.load/2`, and asserts the second prediction matches the first.

  This is the concrete answer to "we want bidirectional ONNX support" —
  every case here exercises both directions plus consistency. Failures are
  registry-driven the same way the import suite is: unlisted cases default
  to `:unsupported`, a newly-passing case fails so the registry must be
  updated, and a regression on a `:passing` case fails normally.

  Import-only failures are intentionally excluded — this suite assumes the
  import side already works for the case under test (which is why we only
  iterate the import-side `:passing` set).
  """
  use ExUnit.Case, async: false

  alias AxonOnnx.Coverage
  alias AxonOnnx.Coverage.Registry
  alias AxonOnnx.Coverage.RoundTripRegistry

  for entry <- Coverage.discover(),
      Registry.status({entry.category, entry.name}) == :passing do
    category = entry.category
    name = entry.name
    path = entry.path
    {status, _note} =
      Map.get(RoundTripRegistry.entries(), {category, name}, {:unsupported, nil})

    @tag category: category
    @tag round_trip_status: status
    @tag round_trip_case: "#{category}/#{name}"
    @tag timeout: 60_000
    test "#{category}/#{name} round-trip (expect #{status})" do
      entry = %{category: unquote(category), name: unquote(name), path: unquote(path)}
      assert_outcome!(entry, unquote(status), Coverage.run_round_trip(entry))
    end
  end

  defp assert_outcome!(_entry, :passing, :ok), do: :ok

  defp assert_outcome!(entry, :passing, {:error, reason}) do
    flunk("Round-trip regression on #{entry.category}/#{entry.name}: #{reason}")
  end

  defp assert_outcome!(entry, expected, :ok) when expected in [:unsupported, :known_bug] do
    flunk(
      "Round-trip on #{entry.category}/#{entry.name} now passes but registry marks it " <>
        "`#{expected}`. Promote it in AxonOnnx.Coverage.RoundTripRegistry."
    )
  end

  defp assert_outcome!(_entry, expected, {:error, _reason})
       when expected in [:unsupported, :known_bug],
       do: :ok
end
