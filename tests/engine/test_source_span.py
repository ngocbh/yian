"""`SourceSpan` locator + the optional `locator` carried on failure events.

A runtime failure points the CLI at the EXACT YAML sub-location it originates from
(an input binding, an assert expression, a flow parameter). The failure site produces a
`SourceSpan`; `NodeFailed`/`RunFailed` carry it; the CLI resolves it to a line.
"""

from agent_composer.events import NodeFailed, RunFailed, SourceSpan


def test_source_span_is_frozen_and_carries_fields():
    s = SourceSpan(node="report", kind="input", key="as_of")
    assert (s.node, s.kind, s.key) == ("report", "input", "as_of")


def test_node_failed_carries_optional_locator():
    assert NodeFailed("n", "boom").locator is None
    s = SourceSpan(node="n", kind="assert", key="${output} != ''")
    assert NodeFailed("n", "boom", locator=s).locator is s


def test_run_failed_carries_optional_locator():
    assert RunFailed("boom").locator is None
    s = SourceSpan(node=None, kind="input", key="window")
    assert RunFailed("boom", locator=s).locator is s
