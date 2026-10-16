"""TOOL `retries:` is checked at load, lands on the node, and keeps default flows' identity."""

from __future__ import annotations

import pytest

from yian.compile.manifest import flow_fingerprint
from yian.compose import load_flow
from yian.compose.helpers.errors import LoadError
from yian.defaults import current
from yian.nodes.tool import ToolNode

# A flow recorded before TOOL nodes had `retries:`; its fingerprint is pinned below.
FLOW = """\
id: old_tool_crash
name: old_tool_crash
nodes:
  place:
    kind: tool
    tool_id: place
    input: {symbol: "ACME"}
output: ${place.output}
"""

OLD_FINGERPRINT = "fde8e299e6e345b350315fabc2cc6406390fcb79db37fd3cff4ed3a68d4e0f4e"


def _flow(field_line: str) -> str:
    """Return `FLOW` with `field_line` added to the TOOL node, as line 7."""
    return FLOW.replace("    tool_id: place\n", f"    tool_id: place\n{field_line}\n")


def test_tool_node_retries_defaults_to_the_tool_retries_default() -> None:
    assert ToolNode("t", tool_id="x").retries == current().tool_retries
    assert ToolNode("t", tool_id="x", retries=4).retries == 4


def test_retries_loads_onto_the_node() -> None:
    assert load_flow(_flow("    retries: 2")).compiled.nodes["place"].retries == 2


def test_omitted_retries_uses_the_default(set_defaults) -> None:
    assert load_flow(FLOW).compiled.nodes["place"].retries == current().tool_retries
    set_defaults(TOOL_RETRIES=3)
    assert load_flow(FLOW).compiled.nodes["place"].retries == 3


@pytest.mark.parametrize(
    ("line", "got"),
    [
        ("    retries: -1", "-1"),
        ("    retries: true", "True"),
        ('    retries: "2"', "'2'"),
        ("    retries: 1.5", "1.5"),
        ("    retries: null", "None"),
    ],
)
def test_bad_retries_is_a_load_error_at_its_line(line: str, got: str) -> None:
    message = (
        r"node 'place' \(kind=tool\): `retries:` must be a non-negative integer, got "
        + got
    )
    with pytest.raises(LoadError, match=message) as caught:
        load_flow(_flow(line))
    assert caught.value.diagnostic.frames[-1].range.start_line == 7


def test_zero_retries_keeps_the_flow_fingerprint() -> None:
    assert flow_fingerprint(load_flow(FLOW).compiled) == OLD_FINGERPRINT
    assert flow_fingerprint(load_flow(_flow("    retries: 0")).compiled) == OLD_FINGERPRINT


def test_nonzero_retries_changes_the_flow_fingerprint() -> None:
    assert flow_fingerprint(load_flow(_flow("    retries: 1")).compiled) != OLD_FINGERPRINT
