defmodule AxonOnnx.CoverageTest do
  @moduledoc """
  Inverted-harness coverage test.

  Walks the entire ONNX backend corpus discovered under `test/cases/` and
  compares each case's actual outcome against `AxonOnnx.Coverage.Registry`.

  The semantics, per the Phase 1 contract:

  * registry says `:passing` and the case passes → test passes
  * registry says `:passing` and the case fails → regression, test fails
  * registry says `:unsupported`/unlisted and the case fails → expected fail,
    test passes
  * registry says `:unsupported`/unlisted and the case passes → progress,
    test fails so the registry must be updated
  * registry says `:known_bug` → treated like `:unsupported` for assertion
    purposes (acknowledged failure); a passing run also fails so the bug
    annotation gets retired

  This module is intentionally heavy on tags so subsets are easy to run:

      mix test --include category:node
      mix test --include status:passing --exclude status:unsupported
      mix test test/axon_onnx/coverage_test.exs --only category:simple
  """
  use ExUnit.Case, async: false

  alias AxonOnnx.Coverage
  alias AxonOnnx.Coverage.Registry

  for entry <- Coverage.discover() do
    category = entry.category
    name = entry.name
    path = entry.path
    {status, _note} = Map.get(Registry.entries(), {category, name}, {:unsupported, nil})

    @tag category: category
    @tag status: status
    @tag onnx_case: "#{category}/#{name}"
    @tag timeout: 60_000
    test "#{category}/#{name} (expect #{status})" do
      entry = %{category: unquote(category), name: unquote(name), path: unquote(path)}
      assert_outcome!(entry, unquote(status), Coverage.run_case(entry))
    end
  end

  defp assert_outcome!(_entry, :passing, :ok), do: :ok

  defp assert_outcome!(entry, :passing, {:error, reason}) do
    flunk("Regression on #{entry.category}/#{entry.name}: #{reason}")
  end

  defp assert_outcome!(entry, expected, :ok) when expected in [:unsupported, :known_bug] do
    flunk(
      "Case #{entry.category}/#{entry.name} now passes but registry marks it " <>
        "`#{expected}`. Promote it in AxonOnnx.Coverage.Registry."
    )
  end

  defp assert_outcome!(_entry, expected, {:error, _reason})
       when expected in [:unsupported, :known_bug],
       do: :ok
end
