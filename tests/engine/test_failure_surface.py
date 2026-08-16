"""Runtime failures expose one authority from evaluation through the public result."""

from __future__ import annotations

from dataclasses import fields
import inspect

import agent_composer.events as events
from agent_composer.diagnostics import Failure
from agent_composer.events import NodeFailed, RunFailed
from agent_composer.nodes.base import NodeAssertFailed
from agent_composer.nodes.binding import BindingError
from agent_composer.run import RunResult
from agent_composer.runtime.eval_node import EvaluationFailure
from agent_composer.runtime.log import NodeFailedRecord


def test_source_span_is_absent_from_runtime_events() -> None:
    """The observer module has no symbolic YAML-location carrier."""
    assert not hasattr(events, "SourceSpan")


def test_failure_carriers_have_no_parallel_location_fields() -> None:
    """Every runtime failure carrier owns only its canonical `Failure`."""
    for carrier in (EvaluationFailure, NodeFailedRecord, NodeFailed, RunFailed):
        assert [field.name for field in fields(carrier)] == ["failure"]

    assert [field.name for field in fields(RunResult)] == [
        "input",
        "status",
        "run_id",
        "log",
        "output",
        "failure",
        "events",
    ]


def test_failure_producers_have_no_location_parameters_or_attributes() -> None:
    """Binding and assertion exceptions carry semantic origins, never YAML coordinates."""
    for exception_type in (BindingError, NodeAssertFailed):
        parameters = inspect.signature(exception_type).parameters
        assert "locator" not in parameters
        assert "range" not in parameters

    binding = BindingError("bad", failure_code="binding.evaluate")
    assertion = NodeAssertFailed("bad")
    for exception in (binding, assertion):
        assert not hasattr(exception, "locator")
        assert not hasattr(exception, "range")


def test_observers_share_the_same_failure_authority() -> None:
    """Failure-only observer construction does not copy or flatten semantics."""
    failure = Failure("node.execute", "boom", "work")

    assert NodeFailed(failure).failure is failure
    assert RunFailed(failure).failure is failure
