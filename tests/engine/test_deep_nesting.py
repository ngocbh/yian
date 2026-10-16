"""Walkers over a flow's nested CALL, MAP, and LOOP bodies don't recurse once per level.

Each test loads a chain far past the call nesting limit, then walks it with only a few
frames of headroom, which a walker that recursed per nested flow would overflow.
"""

import pytest

from tests.compose.test_call_nesting_limit import _defs_chain, _mixed_chain
from tests.deep_chains import deep_load, tight_stack
from yian.compile.manifest import expansion_digest, flow_fingerprint
from yian.compose import ComposedFlow, load_flow
from yian.expr import ExpressionError
from yian.nodes.contracts import Binding
from yian.run import _check_templates, _static_nodes

_DEPTH = 300
_HEADROOM = 150


@pytest.fixture(scope="module")
def mixed() -> ComposedFlow:
    """A chain `_DEPTH` levels deep whose levels nest by CALL, MAP, and LOOP in turn."""
    return deep_load(_mixed_chain(_DEPTH))


def _static_nodes_reference(flow, path=()):
    """Return the static nodes `_static_nodes` returns, by recursing per child flow."""
    found = []
    for node_id, node in flow.nodes.items():
        node_path = (*path, node_id)
        found.append(("/".join(node_path), node))
        for child in node.origin_children():
            found.extend(_static_nodes_reference(child, node_path))
    return found


def test_static_nodes_of_a_deep_chain_come_in_pre_order(mixed):
    expected = _static_nodes_reference(mixed.compiled)
    with tight_stack(_HEADROOM):
        found = _static_nodes(mixed.compiled)
    assert [(path, id(node)) for path, node in found] == [
        (path, id(node)) for path, node in expected
    ]
    assert len(found) == 3 * _DEPTH + 3


def test_templates_of_a_deep_chain_check_without_recursing(mixed):
    with tight_stack(_HEADROOM):
        _check_templates(mixed.compiled)


def test_first_bad_span_in_check_order_is_reported():
    """A node's child flows are checked before the next node of its own flow."""
    flow = load_flow(_defs_chain(5))
    first = flow.compiled.nodes["top"].child
    third = first.nodes["step"].child.nodes["step"].child
    third.wiring["step"]["x"] = Binding("${upper(input.x, 1)}")
    first.wiring["__end__"]["result"] = Binding("${upper(step.output, 1)}")
    with pytest.raises(
        ExpressionError,
        match=r"^node 'top/step/step/step' input 'x': expression function 'upper' takes "
        r"\(s\): too many positional arguments$",
    ):
        _check_templates(flow.compiled)


def test_fingerprint_and_expansion_digest_of_a_deep_chain(mixed):
    with tight_stack(_HEADROOM):
        fingerprint = flow_fingerprint(mixed.compiled)
        digest = expansion_digest(mixed.compiled)
    assert len(fingerprint) == len(digest) == 64


def test_fingerprint_of_a_nested_chain_is_unchanged():
    """The digests main computed for this chain, so walking it iteratively keeps identity."""
    flow = load_flow(_mixed_chain(30))
    assert flow_fingerprint(flow.compiled) == (
        "122a83b235f2eec4d186aabba535a83d648b2c55599f054136c34982bebc9e13"
    )
    assert expansion_digest(flow.compiled) == (
        "a956c8a55e2ae28b69abc49cc85b5d2f3c5cac7e1ce9a16fbcd56d7e4bc08b6c"
    )


def test_a_child_flow_cycle_deep_in_a_chain_is_not_manifestable():
    flow = load_flow(_defs_chain(4))
    first = flow.compiled.nodes["top"].child
    third = first.nodes["step"].child.nodes["step"].child
    third.nodes["step"].child = first
    cycle = r"nodes\.top\.child\.nodes\.step\.child\.nodes\.step\.child\.nodes\.step\.child"
    with pytest.raises(
        ValueError, match=rf"^flow\.{cycle}: recursive child-flow cycle is not manifestable$"
    ):
        flow_fingerprint(flow.compiled)
    with pytest.raises(
        ValueError, match=rf"^expansion\.{cycle}: recursive child-flow cycle is not manifestable$"
    ):
        expansion_digest(flow.compiled)


def test_a_bad_field_in_a_grandchild_names_its_manifest_path():
    flow = load_flow(_defs_chain(3))
    grandchild = flow.compiled.nodes["top"].child.nodes["step"].child
    object.__setattr__(grandchild.nodes["step"], "extra", 1)
    node = r"nodes\.top\.child\.nodes\.step\.child\.nodes\.step"
    unknown = r"unknown noncanonical execution field\(s\) for call: \['extra'\]$"
    with pytest.raises(ValueError, match=rf"^flow\.{node}: {unknown}"):
        flow_fingerprint(flow.compiled)
    with pytest.raises(ValueError, match=rf"^expansion\.{node}: {unknown}"):
        expansion_digest(flow.compiled)
