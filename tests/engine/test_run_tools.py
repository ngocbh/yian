"""Per-run tools: `ToolSet` lookup, `ask_user` wrapping, and run isolation."""

from __future__ import annotations

import functools
from typing import Any

import pytest
from langchain_core.tools import StructuredTool

from agent_composer.tools import TOOL_REGISTRY, ToolSet, register_tool, registered_tools


@pytest.fixture(autouse=True)
def _restore_tool_registry():
    """Keep process-global registrations isolated between tests."""
    original = dict(TOOL_REGISTRY)
    yield
    TOOL_REGISTRY.clear()
    TOOL_REGISTRY.update(original)


def _upper(text: str) -> str:
    """Return the text in upper case."""
    return text.upper()


def test_callable_entry_becomes_a_tool_named_by_its_key():
    tools = ToolSet({"shout": _upper})

    (tool,) = tools.resolve(["shout"])

    assert tool.name == "shout"
    assert tool.description == "Return the text in upper case."
    assert tools.invoke("shout", {"text": "hi"}) == "HI"


def test_per_run_entry_replaces_global_tool_for_that_run_only():
    @register_tool("run_tools_probe")
    def probe(text: str) -> str:
        """Global probe."""
        return f"global:{text}"

    replaced = ToolSet({"run_tools_probe": lambda text: f"run:{text}"})

    assert replaced.invoke("run_tools_probe", {"text": "a"}) == "run:a"
    assert ToolSet().invoke("run_tools_probe", {"text": "a"}) == "global:a"


@pytest.mark.parametrize("tools", [{"shout": _upper}, {}])
def test_a_mapping_hides_the_global_registry(tools: dict[str, Any]):
    @register_tool("run_tools_global_only")
    def global_only(text: str) -> str:
        """Global tool."""
        return text

    assert ToolSet().invoke("run_tools_global_only", {"text": "x"}) == "x"
    with pytest.raises(ValueError, match="unknown tool id 'run_tools_global_only'"):
        ToolSet(tools).resolve(["run_tools_global_only"])


def test_registered_tools_is_a_copy_a_run_can_extend():
    @register_tool("run_tools_snapshot")
    def snapshot_probe(text: str) -> str:
        """Global tool."""
        return f"global:{text}"

    registered_tools().pop("run_tools_snapshot")
    assert "run_tools_snapshot" in TOOL_REGISTRY

    tools = ToolSet({**registered_tools(), "shout": _upper})
    assert tools.invoke("run_tools_snapshot", {"text": "a"}) == "global:a"
    assert tools.invoke("shout", {"text": "a"}) == "A"


def test_public_invocation_rejects_unknown_arguments():
    with pytest.raises(ValueError, match="unsupported argument"):
        ToolSet({"shout": _upper}).invoke("shout", {"text": "a", "extra": 1})


def test_base_tool_entry_must_use_its_key_as_name():
    tool = StructuredTool.from_function(_upper, name="upper", description="Upper.")

    assert ToolSet({"upper": tool}).resolve(["upper"]) == [tool]
    with pytest.raises(ValueError, match="must match its key"):
        ToolSet({"shout": tool})


class _Shouter:
    """A callable object; its bound method is the supported form."""

    def __call__(self, text: str) -> str:
        return text.upper()

    def shout(self, text: str) -> str:
        """Return the text in upper case."""
        return text.upper()


def test_bound_method_entry_becomes_a_tool():
    tools = ToolSet({"shout": _Shouter().shout})

    assert tools.invoke("shout", {"text": "hi"}) == "HI"


@pytest.mark.parametrize(
    ("tools", "error"),
    [
        ([("shout", _upper)], "must be a mapping"),
        ({1: _upper}, "non-empty strings"),
        ({"": _upper}, "non-empty strings"),
        ({"group": [_upper]}, "got list"),
        ({"shout": functools.partial(_upper, text="hi")}, r"tools\['shout'\].*got partial"),
        ({"shout": _Shouter()}, r"tools\['shout'\].*got _Shouter"),
    ],
)
def test_invalid_tools_are_rejected(tools: Any, error: str):
    with pytest.raises(TypeError, match=error):
        ToolSet(tools)
