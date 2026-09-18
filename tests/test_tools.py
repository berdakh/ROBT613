"""Tool schema generation and safe execution."""

from __future__ import annotations

from typing import Literal, Optional

from qwen_workshop.tools import ToolError, ToolRegistry, function_schema, tool


def sample(city: str, unit: Literal["c", "f"] = "c", days: int = 1, tags: list[str] | None = None):
    """Look up the weather.

    Args:
        city: The city name, spelled
            in full.
        unit: Temperature unit.
        days: Days ahead.
        tags: Optional labels.

    Returns:
        A forecast.
    """
    return {"city": city, "unit": unit, "days": days, "tags": tags}


def test_schema_shape():
    schema = function_schema(sample)
    assert schema["type"] == "function"
    fn = schema["function"]
    assert fn["name"] == "sample"
    assert fn["description"] == "Look up the weather."
    assert fn["parameters"]["required"] == ["city"]


def test_schema_types_and_enums():
    props = function_schema(sample)["function"]["parameters"]["properties"]
    assert props["city"]["type"] == "string"
    assert props["days"]["type"] == "integer"
    assert props["unit"]["enum"] == ["c", "f"]
    assert props["tags"]["type"] == "array"
    assert props["tags"]["items"]["type"] == "string"
    assert props["days"]["default"] == 1


def test_multiline_arg_description_is_joined():
    props = function_schema(sample)["function"]["parameters"]["properties"]
    assert props["city"]["description"] == "The city name, spelled in full."


def test_pep604_optional_unwraps():
    """`int | None` must map to "integer", not the default "string"."""

    def f(x: int | None = None):
        """Do a thing.

        Args:
            x: A number.
        """

    props = function_schema(f)["function"]["parameters"]["properties"]
    assert props["x"]["type"] == "integer"


def test_typing_optional_unwraps():
    """The older `Optional[int]` spelling must behave identically.

    Students write both. Under `from __future__ import annotations` these
    arrive as strings, so this also guards the get_type_hints() resolution.
    """
    def f(x: Optional[int] = None):  # noqa: UP045 - testing this spelling on purpose
        """Do a thing.

        Args:
            x: A number.
        """

    props = function_schema(f)["function"]["parameters"]["properties"]
    assert props["x"]["type"] == "integer"


def test_annotation_resolution_survives_missing_names():
    """An unresolvable annotation must degrade, not crash the whole schema."""

    def f(x: "SomeTypeThatDoesNotExist", y: int = 1):  # noqa: F821, UP037
        """Do a thing.

        Args:
            x: Anything.
            y: A number.
        """

    props = function_schema(f)["function"]["parameters"]["properties"]
    assert props["x"]["type"] == "string"  # fell back
    assert props["y"]["type"] == "integer"  # still correct


def test_registry_executes_and_serialises():
    registry = ToolRegistry().add(sample)
    result = registry.call("sample", '{"city": "Astana"}')
    assert result.ok
    assert "Astana" in result.content


def test_registry_reports_unknown_tool_with_alternatives():
    registry = ToolRegistry().add(sample)
    result = registry.call("nope", {})
    assert not result.ok
    assert "no tool named" in result.content
    assert "sample" in result.content  # tells the model what it *can* call


def test_registry_ignores_hallucinated_arguments():
    registry = ToolRegistry().add(sample)
    result = registry.call("sample", {"city": "Almaty", "temperature_unit": "kelvin"})
    assert result.ok
    assert "temperature_unit" in result.content  # the model is told it was dropped
    assert result.arguments == {"city": "Almaty"}


def test_registry_surfaces_tool_error_as_text():
    @tool
    def failing(x: int):
        """Always fails.

        Args:
            x: Anything.
        """
        raise ToolError("x must be positive")

    result = ToolRegistry().add(failing).call("failing", {"x": -1})
    assert not result.ok
    assert "x must be positive" in result.content


def test_registry_never_raises_on_bad_json():
    result = ToolRegistry().add(sample).call("sample", "{not json")
    assert not result.ok
    assert "valid JSON" in result.content


def test_missing_required_argument_is_reported():
    result = ToolRegistry().add(sample).call("sample", {})
    assert not result.ok
    assert "wrong arguments" in result.content


def test_long_output_is_truncated():
    @tool
    def big():
        """Return a lot of text."""
        return "x" * 10_000

    result = ToolRegistry().add(big).call("big", {})
    assert result.ok
    assert "truncated" in result.content
    assert len(result.content) < 5_000


def test_registry_names_are_sorted_and_countable():
    registry = ToolRegistry().add(sample)
    assert registry.names() == ["sample"]
    assert len(registry) == 1
    assert "sample" in registry
