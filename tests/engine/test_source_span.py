"""`SourceSpan` locator + the optional `locator` carried beside `Failure`.

A runtime failure points the CLI at the EXACT YAML sub-location it originates from
(an input binding, an assert expression, a flow parameter). The failure site produces a
`SourceSpan`; `NodeFailed`/`RunFailed` carry it; the CLI resolves it to a line.
"""

from dataclasses import fields

from agent_composer.diagnostics import Failure
from agent_composer.events import NodeFailed, RunFailed, SourceSpan


def test_source_span_is_frozen_and_carries_fields():
    s = SourceSpan(node="report", kind="input", key="as_of")
    assert (s.node, s.kind, s.key) == ("report", "input", "as_of")


def test_node_failed_carries_optional_locator():
    failure = Failure("node.execute", "boom", "n")
    assert NodeFailed(failure).locator is None
    s = SourceSpan(node="n", kind="assert", key="${output} != ''")
    assert NodeFailed(failure, locator=s).locator is s
    assert [field.name for field in fields(NodeFailed)] == [
        "failure",
        "locator",
        "range",
    ]


def test_run_failed_carries_optional_locator():
    failure = Failure("node.execute", "boom", "n")
    assert RunFailed(failure).locator is None
    s = SourceSpan(node=None, kind="input", key="window")
    assert RunFailed(failure, locator=s).locator is s
    assert [field.name for field in fields(RunFailed)] == [
        "failure",
        "locator",
        "range",
    ]
