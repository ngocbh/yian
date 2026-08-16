"""Temporary `SourceSpan` compatibility metadata carried beside `Failure`.

Failure sites still produce a locator for run-log compatibility, but canonical
presentation resolves `Failure.trace`; the CLI does not inspect this carrier.
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
