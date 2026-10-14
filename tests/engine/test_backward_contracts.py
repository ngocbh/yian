"""Ordinary feedback values and retained backward contract boundaries."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, dataclass, field

import pytest

from yian.diagnostics import Failure, OriginRef
from yian.expr.lineage import has_feedback, validate_feedback
from yian.memory.contracts import (
    LearningFailure,
    LearningResult,
    MemoryOperationFailure,
)
from yian.nodes.base import (
    BackwardContext,
    BackwardResult,
    validate_backward_result,
)
from yian.nodes.contracts import ExecutionPosition, ExecutionScope
from yian.runtime.contracts import BackwardFailure, BackwardOutcome


def test_attributed_backward_result_detaches_messages_and_keeps_reference_coordinates():
    from yian.expr.lineage import SourceRef
    from yian.runtime.contracts import AttributedBackwardResult, BackwardAttribution

    source = SourceRef("run", 1)
    sender = SourceRef("run", 4)
    attribution = BackwardAttribution(sender, "items", 0, 1, (0, 2))
    message = ["first", None]
    result = AttributedBackwardResult(
        source, (message,), (attribution,), BackwardResult(self_feedback=("note",)), True,
    )
    message[0] = "changed"
    result.feedback[0][0] = "also changed"
    assert result.feedback == (["first", None],)
    assert result.attribution == (attribution,)
    assert result.result.self_feedback == ("note",)
    assert not hasattr(attribution, "order")


@pytest.mark.parametrize("args", [
    (None, "input", 0, None, ()),
    (None, None, 1, None, ()),
    (None, None, 0, 0, ()),
    (None, None, 0, None, (0,)),
    ("sender", "input", 0, 0, ()),
])
def test_backward_attribution_rejects_malformed_root_coordinates(args):
    from yian.runtime.contracts import BackwardAttribution

    with pytest.raises((TypeError, ValueError)):
        BackwardAttribution(*args)


@pytest.mark.parametrize("inputs", [{"topic": "reefs"}, None])
def test_declared_variables_supply_local_backward_targets_without_activating_hooks(inputs):
    from yian.nodes.agent import AgentNode
    from tests.compose._declarations import parameters_from_yaml

    node = AgentNode("writer", variables=parameters_from_yaml(
        "variables:\n  tone: !str concise\n", {}, field="variables", require_default=True,
    ))
    context = BackwardContext(
        inputs=inputs,
        input_names=("topic",),
        variable_names=tuple(parameter.name for parameter in node.declared_variables()),
    )
    validate_backward_result(
        BackwardResult(variable_feedback={"tone": ("Use a technical tone.",)}), context,
    )
    for name in ("topic", "other_instruction", "other.tone"):
        with pytest.raises(ValueError, match=name):
            validate_backward_result(
                BackwardResult(variable_feedback={name: ("Be clearer.",)}), context,
            )


def test_feedback_uses_exact_nested_list_lengths() -> None:
    """A whole-list message or a full positional list addresses one recorded value."""
    value = {"summary": "text", "tags": ["a", "b"]}

    validate_feedback("Review the whole result.", value)
    validate_feedback({"tags": "Review these tags."}, value)
    validate_feedback({"tags": [None, "Irrelevant."]}, value)
    validate_feedback({"tags": [None, None]}, value)
    validate_feedback([], [])
    for tags in ([], ["Irrelevant."], [None, None, None]):
        with pytest.raises(ValueError, match="tags"):
            validate_feedback({"tags": tags}, value)

    nested = {"groups": [{"tags": ["a", "b"]}, {"tags": []}]}
    validate_feedback({"groups": [{"tags": ["", None]}, {"tags": []}]}, nested)
    with pytest.raises(ValueError, match=r"groups.*0.*tags"):
        validate_feedback({"groups": [{"tags": [None]}, None]}, nested)


@pytest.mark.parametrize(
    ("feedback", "value"),
    [
        (7, {"summary": "text"}),
        (True, {"summary": "text"}),
        ({"missing": "Bad."}, {"summary": "text"}),
        (1.5, "text"),
        (("Bad.",), ["text"]),
        ({1: "Bad."}, {1: "text"}),
        ({"summary": False}, {"summary": "text"}),
        ({"summary": [None]}, {"summary": "text"}),
        ({"summary": {}}, {"summary": []}),
        ([None], {"summary": "text"}),
        ({}, None),
    ],
)
def test_feedback_rejects_invalid_values_and_unknown_fields(feedback, value) -> None:
    """Feedback remains plain structured selections with exact string field names."""
    with pytest.raises((TypeError, ValueError), match="feedback"):
        validate_feedback(feedback, value)


@pytest.mark.parametrize(
    ("feedback", "expected"),
    [
        (None, False),
        ({}, False),
        ([], False),
        ([None, None], False),
        ({"tags": [None, {"nested": None}]}, False),
        ({"tags": [], "summary": {}}, False),
        ("", True),
        ("Good.", True),
        ({"tags": [None, ""]}, True),
        ({"summary": "Good."}, True),
    ],
)
def test_none_and_empty_selections_have_no_effective_feedback(feedback, expected) -> None:
    """An empty string is still a supplied message; absent selections are not."""
    assert has_feedback(feedback) is expected


def test_backward_result_keeps_message_tuples_distinct_from_item_lists() -> None:
    """Batches preserve independent messages and positional feedback without merging."""
    item_feedback = [None, "Irrelevant."]
    messages = {"tags": (item_feedback, "Review the complete list."), "summary": ("",)}
    result = BackwardResult(
        input_feedback=messages,
        self_feedback=("Be concise.", "Preserve sources."),
        variable_feedback={"instruction": ("Be explicit.", "")},
    )

    assert result.input_feedback == {
        "tags": ([None, "Irrelevant."], "Review the complete list."),
        "summary": ("",),
    }
    assert type(result.input_feedback["tags"]) is tuple
    assert type(result.input_feedback["tags"][0]) is list
    assert result.self_feedback == ("Be concise.", "Preserve sources.")
    assert result.variable_feedback == {"instruction": ("Be explicit.", "")}
    empty = BackwardResult()
    assert empty.input_feedback == {}
    assert empty.self_feedback == ()
    assert empty.variable_feedback == {}

    item_feedback[1] = "Changed by the producer."
    messages.clear()
    view = result.input_feedback
    view["tags"][0][1] = "Changed by the reader."
    view.clear()
    result.variable_feedback.clear()
    assert result.input_feedback["tags"][0] == [None, "Irrelevant."]
    assert result.variable_feedback == {"instruction": ("Be explicit.", "")}
    with pytest.raises(FrozenInstanceError):
        result.self_feedback = ()
    with pytest.raises(TypeError, match="tuple"):
        BackwardResult(input_feedback={"tags": [None, "Irrelevant."]})


def test_result_validates_input_shapes_and_declared_variable_names() -> None:
    """Only this owner's declared inputs and variables can receive feedback."""
    context = BackwardContext(
        inputs={"tags": ["a", "b"], "summary": "text", "reserved": 7},
        input_names=("tags", "summary"),
        variable_names=("instruction",),
    )
    validate_backward_result(
        BackwardResult(
            input_feedback={"tags": ([None, "Irrelevant."], "Review the list.")},
            self_feedback=("Be concise.",),
            variable_feedback={"instruction": ("Make the task explicit.",)},
        ),
        context,
    )
    for name in ("missing", "reserved"):
        with pytest.raises(ValueError, match=name):
            validate_backward_result(
                BackwardResult(input_feedback={name: ("Bad.",)}), context
            )
    with pytest.raises(ValueError, match="tags"):
        validate_backward_result(
            BackwardResult(input_feedback={"tags": (["Bad."],)}), context
        )
    with pytest.raises(ValueError, match="parent_instruction"):
        validate_backward_result(
            BackwardResult(variable_feedback={"parent_instruction": ("Bad.",)}),
            context,
        )
    with pytest.raises(ValueError, match="instruction"):
        validate_backward_result(
            BackwardResult(variable_feedback={"instruction": ("Bad.",)}),
            BackwardContext(inputs={}),
        )

    early_failure = BackwardContext(inputs=None, input_names=("tags",))
    validate_backward_result(
        BackwardResult(self_feedback=("Binding failed.",)), early_failure
    )
    for batch in ((), (None,), ("Bad.",)):
        with pytest.raises(ValueError, match="inputs"):
            validate_backward_result(
                BackwardResult(input_feedback={"tags": batch}), early_failure
            )


@pytest.mark.parametrize("message", [None, 7, True, {}, ["Bad."]])
def test_self_and_variable_feedback_require_text_messages(message) -> None:
    """Behavior and variable channels contain strings, never structured selections."""
    with pytest.raises(TypeError, match="self_feedback"):
        BackwardResult(self_feedback=(message,))
    with pytest.raises(TypeError, match="instruction"):
        BackwardResult(variable_feedback={"instruction": (message,)})


def test_execution_scopes_keep_root_relative_structured_positions() -> None:
    """Source-neutral positions distinguish calls, MAP items, and LOOP drivers."""
    root = ExecutionScope(static_path=(), flow_path=())
    assert root.occurrence_path == ()
    assert root.boundary is None
    positions = (
        ExecutionPosition("call", ("child",)),
        ExecutionPosition("map_item", ("child", "each"), 2),
        ExecutionPosition("loop_body", ("child", "each", "loop"), 3),
    )
    scope = ExecutionScope(
        static_path=("child", "each", "loop", "END"),
        flow_path=("child", "each", "loop"),
        occurrence_path=positions,
        boundary="end",
    )
    assert scope.occurrence_path == positions
    assert positions[-1] != ExecutionPosition("loop_driver", positions[-1].static_path, 3)
    with pytest.raises(FrozenInstanceError):
        scope.boundary = "start"


@pytest.mark.parametrize(
    "factory",
    [
        lambda: ExecutionScope(static_path=["node"]),
        lambda: ExecutionScope(static_path=(True,)),
        lambda: ExecutionScope(static_path=("node",), occurrence_path=[]),
        lambda: ExecutionScope(static_path=("node",), boundary="other"),
        lambda: ExecutionPosition("map_item", ("each",), True),
        lambda: ExecutionPosition("loop_driver", ("loop",), -1),
        lambda: ExecutionPosition("loop_body", ("loop",)),
        lambda: ExecutionPosition("call", ("child",), 2),
    ],
)
def test_execution_scopes_reject_mutable_or_ambiguous_positions(factory) -> None:
    """Occurrence indices are exact non-negative integers and paths are immutable."""
    with pytest.raises((TypeError, ValueError)):
        factory()


def test_backward_context_and_outcome_do_not_expose_mutable_authority() -> None:
    """Producer and reader mutations cannot change retained context or results."""
    inputs = {"tags": ["a", "b"]}
    output = {"summary": ["result"]}
    data = {"prompt": ["Summarize."]}
    downstream = ({"source_path": ["summary", 0]},)
    forward_failure = Failure("node.execute", "Forward failed.", "writer")
    scope = ExecutionScope(static_path=("writer",))
    context = BackwardContext(
        inputs=inputs,
        input_names=("tags",),
        variable_names=("instruction",),
        output=output,
        failure=forward_failure,
        scope=scope,
        run_id="run-1",
        node_id="writer",
        node_started_seq=17,
        node_data=data,
        downstream_context=downstream,
    )
    inputs["tags"].clear()
    output["summary"].clear()
    data["prompt"].clear()
    downstream[0]["source_path"].clear()
    context.inputs["tags"].clear()
    context.output["summary"].clear()
    context.node_data["prompt"].clear()
    context.downstream_context[0]["source_path"].clear()
    assert context.inputs == {"tags": ["a", "b"]}
    assert context.output == {"summary": ["result"]}
    assert context.node_data == {"prompt": ["Summarize."]}
    assert context.downstream_context == ({"source_path": ["summary", 0]},)
    assert (context.run_id, context.node_started_seq) == ("run-1", 17)
    assert context.scope == scope
    assert context.failure == forward_failure

    cause = MemoryOperationFailure("learn", "model.failed", "Model failed.", True)
    failure = BackwardFailure(
        stage="learn",
        code="memory.learn",
        message="Could not compute a learning proposal.",
        scope=scope,
        memory="history",
        cause=cause,
        cause_type="MemoryOperationError",
        traceback="debugging detail",
    )
    learning = LearningResult(
        "failed", (LearningFailure("history", "learn", cause),)
    )
    feedback = BackwardResult(input_feedback={"tags": ([None, "Irrelevant."],)})
    outcome = BackwardOutcome("failed", feedback, failure, learning)
    outcome.feedback.input_feedback["tags"][0].clear()
    feedback.input_feedback.clear()
    assert outcome.feedback.input_feedback == {"tags": ([None, "Irrelevant."],)}
    assert outcome.status == "failed"
    assert outcome.failure.cause == cause
    assert outcome.failure.cause.retryable is True
    assert outcome.failure.traceback == "debugging detail"
    assert outcome.learning == learning
    assert context.failure == forward_failure
    with pytest.raises(FrozenInstanceError):
        context.input_names = ()
    with pytest.raises(FrozenInstanceError):
        outcome.status = "succeeded"


@pytest.mark.parametrize("carrier", ["context", "failure", "outcome"])
@pytest.mark.parametrize("mutation", ["argument", "public_view"])
def test_retained_failure_origins_drop_mutable_subclass_state(carrier, mutation) -> None:
    """Retained diagnostics keep semantic origin keys without subclass aliases."""

    @dataclass(frozen=True)
    class AnnotatedOrigin(OriginRef):
        notes: list[str] = field(default_factory=list)

    origin = AnnotatedOrigin("writer.prompt", ["authored note"])
    failure = Failure(
        code="node.execute",
        message="Forward failed.",
        node_id="writer",
        trace=(origin, OriginRef("writer"), origin),
        cause_type="LookupError",
        traceback="original debugging detail",
    )
    context = BackwardContext(inputs=None, failure=failure)
    backward_failure = BackwardFailure(
        "backward", "node.backward", "Reverse failed.", cause=failure
    )
    outcome = BackwardOutcome("failed", failure=backward_failure)
    retained = {
        "context": context.failure,
        "failure": backward_failure.cause,
        "outcome": outcome.failure.cause,
    }[carrier]

    if mutation == "argument":
        origin.notes.append("changed through the constructor argument")
    else:
        for item in retained.trace:
            if isinstance(item, AnnotatedOrigin):
                item.notes.append("changed through a public diagnostic view")

    assert retained == Failure(
        code="node.execute",
        message="Forward failed.",
        node_id="writer",
        trace=(
            OriginRef("writer.prompt"), OriginRef("writer"), OriginRef("writer.prompt")
        ),
        cause_type="LookupError",
        traceback="original debugging detail",
    )
    assert all(type(item) is OriginRef for item in retained.trace)
    assert failure.trace[0] is origin


def test_backward_outcome_does_not_normalize_its_caller_owned_failure() -> None:
    """Constructing an outcome validates a copy without rewriting its argument."""
    cause = Failure(
        "node.execute", "Forward failed.", "writer", (OriginRef("writer"),)
    )
    failure = BackwardFailure(
        "backward", "node.backward", "Reverse failed.", cause=cause
    )
    original_cause = failure.cause

    outcome = BackwardOutcome("failed", failure=failure)

    assert failure.cause is original_cause
    assert outcome.failure is not failure
    assert outcome.failure == failure


@pytest.mark.parametrize(
    "factory",
    [
        lambda: BackwardContext(inputs=[]),
        lambda: BackwardContext(inputs={1: "text"}),
        lambda: BackwardContext(inputs={"text": object()}),
        lambda: BackwardContext(inputs={}, output=float("inf")),
        lambda: BackwardContext(inputs={}, input_names=["text"]),
        lambda: BackwardContext(inputs={}, input_names=("text", "text")),
        lambda: BackwardContext(inputs={}, variable_names=(7,)),
        lambda: BackwardContext(inputs={}, node_started_seq=True),
        lambda: BackwardContext(inputs={}, node_started_seq=0),
        lambda: BackwardContext(inputs={}, node_data=[]),
        lambda: BackwardContext(inputs={}, downstream_context=({}, {"path": (1,)})),
        lambda: BackwardOutcome("incomplete"),
        lambda: BackwardOutcome("failed"),
        lambda: BackwardOutcome("succeeded", feedback={}),
        lambda: BackwardOutcome("succeeded", failure=BackwardFailure("learn", "x", "x")),
        lambda: BackwardFailure("learn", "", "detail"),
        lambda: BackwardFailure("learn", "x", "detail", node_started_seq=True),
        lambda: BackwardFailure("learn", "x", "detail", cause=ValueError("bad")),
    ],
)
def test_backward_contracts_reject_invalid_snapshots_and_outcomes(factory) -> None:
    """Backward status and source-neutral diagnostic data have explicit contracts."""
    with pytest.raises((TypeError, ValueError)):
        factory()


def test_backward_outcome_can_succeed_without_memory() -> None:
    """A successful reverse computation does not require a learning result from any memory key."""
    result = BackwardResult()
    outcome = BackwardOutcome("succeeded", feedback=result)
    assert outcome.feedback == result
    assert outcome.failure is None
    assert outcome.learning is None


def test_backward_outcome_copy_preserves_ordered_policy_dispositions() -> None:
    from dataclasses import replace
    from yian.memory import PolicyCommitOutcome

    policies = (PolicyCommitOutcome("second", "unknown", "commit-2"),
                PolicyCommitOutcome("first", "unrequested"))
    learning = LearningResult("incomplete", (), policies)
    feedback = BackwardResult()
    outcome = BackwardOutcome("failed", feedback,
                              BackwardFailure("commit", "interrupted", ""), learning)
    copied = replace(outcome)
    assert copied.feedback == feedback
    assert copied.learning == learning
    assert copied.learning is not learning
    assert copied.learning.policies == policies
    assert copied.learning.policies[0] is not policies[0]
