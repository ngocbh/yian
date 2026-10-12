"""Captured structural backward layouts project ordinary feedback values."""

import pytest

from yian.nodes.base import BackwardContext, BackwardResult, Node, validate_backward_result
from yian.nodes.contracts import Parameter
from yian.nodes.end import EndNode
from yian.nodes.start import StartNode


def _context(node, inputs, output):
    return BackwardContext(
        inputs, input_names=node.backward_input_names(), output=output,
        node_data=node.capture_backward_context(),
    )


class _CaptureNode(Node):
    def __init__(self):
        super().__init__("work")
        self.params = [Parameter("topic"), Parameter("limit", has_default=True, default=4)]
        self.variables = [Parameter("instruction")]
        self.material = {"settings": {"active": True}}
        self.captures = 0
        self.ran = False

    def declared_variables(self):
        return tuple(self.variables)

    def capture_backward_context(self):
        self.captures += 1
        return self.material

    def forward(self, inputs):
        from yian.nodes.base import Output

        self.ran = True
        self.params[0] = Parameter("changed_input")
        self.variables[0] = Parameter("changed_variable")
        self.material["settings"]["active"] = False
        self.execution_scope = None
        return Output(inputs["topic"])


def test_behavioral_backward_stops_with_empty_channels_and_static_capture():
    node = Node("ordinary")
    node.params = [Parameter("topic")]
    context = BackwardContext({"topic": "reefs"}, input_names=("topic",), output="answer")

    assert node.backward(context, ("Be more precise.",)) == BackwardResult()
    assert node.capture_backward_context() == {}
    assert node.backward_input_names() == ("topic",)
    assert Node("empty").backward_input_names() == ()


def test_case_backward_inventory_excludes_generated_control_inputs():
    from yian.nodes.case import CaseNode

    node = CaseNode("choose", [])
    node.params = [Parameter(name) for name in ("topic", "__r0", "__on", "__r12", "__remarks")]
    assert node.backward_input_names() == ("topic", "__remarks")


@pytest.mark.parametrize("params", [{"topic": Parameter("topic")}, [object()]])
def test_input_inventory_capture_preserves_canonical_parameter_setup_faults(params):
    from yian.runtime.eval_node import eval_node
    from yian.typesys.pool import VariablePool

    node = Node("invalid")
    node.params = params
    for capture in (False, True):
        with pytest.raises(TypeError, match="params"):
            eval_node(node, None, VariablePool(), capture_bindings=capture)


def test_start_projects_named_delivered_inputs_including_defaults_but_not_absence():
    from yian.runtime.eval_node import eval_node
    from yian.typesys.pool import VariablePool

    node = StartNode("start", params=[
        Parameter("topic"), Parameter("limit", has_default=True, default=4), Parameter("omitted"),
    ])
    evaluated = eval_node(node, None, VariablePool(), wiring_override={"topic": "reefs"}, capture_bindings=True)
    inputs = evaluated.bindings.bound_inputs
    context = _context(node, inputs, evaluated.outcome.value)

    result = node.backward(context, ("Review the request.", {"limit": "Use fewer."}, None))

    assert inputs == {"topic": "reefs", "limit": 4}
    assert result.input_feedback == {
        "topic": ("Review the request.",), "limit": ("Review the request.", "Use fewer."),
    }
    assert context.node_data == {"input_output_paths": {
        "topic": ["topic"], "limit": ["limit"], "omitted": ["omitted"],
    }}
    validate_backward_result(result, context)


@pytest.mark.parametrize("names,inputs,feedback,expected,paths", [
    ([], {}, ("No return value.",), {}, {}),
    (["result"], {"result": {"items": ["a", "b"]}}, ({"items": [None, "Review."]},),
     {"result": ({"items": [None, "Review."]},)}, {"result": []}),
    (["left", "right"], {"left": 1, "right": 2}, ("Review both.", {"right": "Check."}),
     {"left": ("Review both.",), "right": ("Review both.", "Check.")},
     {"left": ["left"], "right": ["right"]}),
])
def test_end_projects_its_zero_one_and_many_output_layouts(names, inputs, feedback, expected, paths):
    node = EndNode.record("end", output_names=names)
    node.params.append(Parameter("assertion_input"))
    delivered = {**inputs, "assertion_input": "diagnostic-only"}
    context = _context(node, delivered, node.forward(delivered).value)

    result = node.backward(context, feedback)

    assert context.input_names == tuple(names)
    assert context.node_data == {"input_output_paths": paths}
    assert result.input_feedback == expected
    validate_backward_result(result, context)


@pytest.mark.parametrize("items", [[], [["same"], ["same"]]])
def test_list_end_keeps_whole_list_text_unchanged_and_records_each_output_position(items):
    node = EndNode.list_("collector", n=len(items))
    inputs = {f"e{index}": item for index, item in enumerate(items)}
    context = _context(node, inputs, node.forward(inputs).value)
    text = "The complete list repeats itself."

    result = node.backward(context, (text, [None] * len(items)))

    assert result.input_feedback == {name: (text,) for name in inputs}
    assert context.node_data == {"input_output_paths": {
        f"e{index}": [index] for index in range(len(items))
    }}
    validate_backward_result(result, context)
    if items:
        positional = node.backward(context, ([None, ["Review the second item."]],))
        assert positional.input_feedback == {"e1": (["Review the second item."],)}
        with pytest.raises(ValueError, match="exactly 2 items"):
            node.backward(context, (["too short"],))


def test_end_backward_uses_captured_layout_after_live_node_fields_change():
    node = EndNode.record("end", output_names=["result"])
    inputs = {"result": {"nested": "answer"}}
    context = _context(node, inputs, node.forward(inputs).value)
    node._names[:] = ["changed", "fields"]
    node._mode = "list"
    node._n = 0
    node.params.clear()

    result = node.backward(context, ({"nested": "Explain."},))

    assert result.input_feedback == {"result": ({"nested": "Explain."},)}
    assert context.node_data == {"input_output_paths": {"result": []}}


def test_stopping_loop_projects_all_carried_fields_without_reevaluating_predicate(monkeypatch):
    import yian.nodes.loop.node as loop_module
    from yian.nodes.loop import LoopNode

    node = LoopNode("loop", flow_id="body", predicate="false", max_iters=2,
                    child_inputs=[Parameter("value")])
    node.params = [Parameter("value"), Parameter("carried_only")]
    inputs = {"value": 7, "carried_only": ["kept"]}
    context = _context(node, inputs, node.forward(inputs).value)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("backward reevaluated the loop predicate")

    monkeypatch.setattr(loop_module, "evaluate_when_record", forbidden)
    node.params.clear()
    result = node.backward(context, ({"carried_only": ["Review."]},))
    assert result.input_feedback == {"carried_only": (["Review."],)}
    validate_backward_result(result, context)
    continuing = BackwardContext(inputs, input_names=context.input_names, node_data=context.node_data)
    assert node.backward(continuing, ("Growth is handled by its connections.",)) == BackwardResult()


@pytest.mark.parametrize("node", [StartNode("start", params=[Parameter("value")]), EndNode.list_("end", n=1)])
def test_failed_boundaries_forward_text_only_when_inputs_and_layout_were_captured(node):
    from yian.diagnostics import Failure

    name, = node.backward_input_names()
    failure = Failure("node.execute", "Failed.", node_id=node.id)
    context = BackwardContext(
        {name: ["delivered"]}, input_names=(name,), failure=failure,
        node_data=node.capture_backward_context(),
    )
    result = node.backward(context, ("Review the supplied value.",))
    assert result.input_feedback == {name: ("Review the supplied value.",)}
    assert context.output is None
    with pytest.raises(ValueError):
        node.backward(context, ({name: "No accepted output shape."},))
    unavailable = BackwardContext(None, input_names=(name,), failure=failure)
    assert node.backward(unavailable, ("No usable binding.",)) == BackwardResult()


def test_successful_null_output_is_a_bare_contributor_but_zero_output_has_none():
    one = EndNode.record("one", output_names=["result"])
    zero = EndNode.record("zero", output_names=[])
    assert one.backward(_context(one, {"result": None}, None), ("Review null.",)).input_feedback == {
        "result": ("Review null.",),
    }
    assert zero.backward(_context(zero, {}, None), ("Review null.",)) == BackwardResult()


def test_stopped_loop_failure_does_not_route_an_unaccepted_carried_output():
    from yian.diagnostics import Failure
    from yian.nodes.loop import LoopNode

    node = LoopNode("loop", flow_id="body", predicate="false", max_iters=2)
    node.params = [Parameter("value")]
    context = BackwardContext(
        {"value": 1}, input_names=("value",), node_data=node.capture_backward_context(),
        failure=Failure("assertion.failed", "Stopped output was rejected.", node_id=node.id),
    )
    assert node.backward(context, ("Review the loop.",)) == BackwardResult()


def test_static_backward_capture_and_names_are_detached_for_the_actual_invocation():
    from yian.nodes.contracts import ExecutionScope
    from yian.runtime.eval_node import eval_node
    from yian.runtime.trace import capture_bindings
    from yian.typesys.pool import VariablePool

    node = _CaptureNode()
    scope = ExecutionScope(("authored",))
    node.execution_scope = scope
    evaluated = eval_node(node, None, VariablePool(), wiring_override={"topic": "reefs"}, capture_bindings=True)
    snapshot = evaluated.backward_capture
    capture = capture_bindings(evaluated.bindings, snapshot)

    assert node.captures == 1 and node.ran
    assert snapshot.input_names == ("topic", "limit")
    assert snapshot.variable_names == ("instruction",)
    assert capture.input_names == snapshot.input_names and capture.variable_names == snapshot.variable_names
    assert capture.scope == scope and node.execution_scope is None
    assert capture.bound_inputs == {"topic": "reefs", "limit": 4}
    assert capture.node_data == {"settings": {"active": True}}
    snapshot.node_data["settings"]["active"] = "changed by caller"
    assert snapshot.node_data == capture.node_data


def test_early_binding_failure_keeps_inventories_without_usable_inputs_or_fabricated_layout():
    from yian.runtime.eval_node import EvaluationFailure, eval_node
    from yian.runtime.trace import capture_bindings
    from yian.typesys.pool import VariablePool

    node = _CaptureNode()
    node.params[0] = Parameter("topic", required=True)
    evaluated = eval_node(node, None, VariablePool(), capture_bindings=True)
    capture = capture_bindings(evaluated.bindings, evaluated.backward_capture)

    assert isinstance(evaluated.outcome, EvaluationFailure)
    assert capture.bound_inputs is None
    assert capture.input_names == ("topic", "limit")
    assert capture.variable_names == ("instruction",)
    assert capture.node_data == {}
    assert node.captures == 0 and not node.ran


@pytest.mark.parametrize("material", [object(), {"bad": object()}, {"bad": float("inf")}, {"bad": "\ud800"}])
def test_static_capture_serialization_faults_are_host_faults_before_the_body(material):
    from yian.runtime.eval_node import eval_node
    from yian.typesys.pool import VariablePool

    node = _CaptureNode()
    node.material = material
    with pytest.raises((TypeError, ValueError)):
        eval_node(node, None, VariablePool(), wiring_override={"topic": "reefs"}, capture_bindings=True)
    assert not node.ran


def test_static_capture_exception_escapes_unchanged_and_capture_off_calls_no_hook():
    from yian.runtime.eval_node import eval_node
    from yian.typesys.pool import VariablePool

    expected = ValueError("capture failed")
    node = _CaptureNode()

    def fail():
        raise expected

    node.capture_backward_context = fail
    with pytest.raises(ValueError) as caught:
        eval_node(node, None, VariablePool(), wiring_override={"topic": "reefs"}, capture_bindings=True)
    assert caught.value is expected and not node.ran
    evaluated = eval_node(node, None, VariablePool(), wiring_override={"topic": "reefs"})
    assert node.ran and evaluated.outcome.value == "reefs"
    assert evaluated.bindings is None and evaluated.backward_capture is None


@pytest.mark.parametrize("output,path", [({"actual": 1}, ["missing"]), ({"actual": 1}, [0]), ([1], [1])])
def test_structural_output_paths_must_exist_in_the_available_accepted_output(output, path):
    node = EndNode.record("end", output_names=["result"])
    context = BackwardContext(
        {"result": 1}, input_names=("result",), output=output,
        node_data={"input_output_paths": {"result": path}},
    )
    with pytest.raises(ValueError, match="output path"):
        node.backward(context, ("Review.",))


def test_static_layout_is_captured_before_a_preassert_failure_after_defaults():
    from yian.nodes.contracts import Assertion
    from yian.runtime.eval_node import EvaluationFailure, eval_node
    from yian.runtime.trace import capture_bindings
    from yian.typesys.pool import VariablePool

    node = EndNode.record("end", output_names=["result"])
    node.params = [Parameter("result", has_default=True, default={"items": [1]})]
    node.pre_asserts = [Assertion("false", "reject before body")]
    evaluated = eval_node(node, None, VariablePool(), capture_bindings=True)
    capture = capture_bindings(evaluated.bindings, evaluated.backward_capture)

    assert isinstance(evaluated.outcome, EvaluationFailure)
    assert evaluated.outcome.failure.code == "assertion.failed"
    assert capture.bound_inputs == {"result": {"items": [1]}}
    assert capture.node_data == {"input_output_paths": {"result": []}}
    node._names.clear()
    context = BackwardContext(
        capture.bound_inputs, input_names=capture.input_names,
        node_data=capture.node_data, failure=evaluated.outcome.failure,
    )
    assert node.backward(context, ("Review the default.",)).input_feedback == {
        "result": ("Review the default.",),
    }


def test_custom_run_uses_its_existing_pre_run_capture_boundary():
    from yian.runtime.eval_node import eval_node
    from yian.typesys.pool import VariablePool

    class CustomRun(_CaptureNode):
        def run(self, inputs, **_caps):
            return self.forward(inputs)

    node = CustomRun()
    evaluated = eval_node(node, None, VariablePool(), wiring_override={"topic": "reefs"}, capture_bindings=True)
    assert evaluated.outcome.value == "reefs"
    assert evaluated.bindings.bound_inputs == {"topic": "reefs", "limit": 4}
    assert evaluated.backward_capture.input_names == ("topic", "limit")
    assert evaluated.backward_capture.node_data == {"settings": {"active": True}}
    assert node.captures == 1


def test_agent_capture_keeps_variable_inventory_separate_from_delivered_inputs(monkeypatch):
    import yian.nodes.agent.node as agent_module
    from yian.compose.helpers.types import read_variables
    from yian.nodes.agent import AgentNode
    from yian.nodes.base import Output
    from yian.runtime.eval_node import eval_node
    from yian.typesys.pool import VariablePool

    node = AgentNode("writer", prompt="${topic}: ${instruction}", variables=read_variables({
        "instruction": "str = Summarize clearly",
    }, {}))
    node.params = [Parameter("topic")]
    monkeypatch.setattr(agent_module, "run_agent", lambda ctx: Output(ctx.prompt))
    evaluated = eval_node(node, None, VariablePool(), wiring_override={"topic": "reefs"}, capture_bindings=True)
    assert evaluated.outcome.value == "reefs: Summarize clearly"
    assert evaluated.backward_capture.variable_names == ("instruction",)
    assert evaluated.backward_capture.node_data["variables"] == {"instruction": "Summarize clearly"}
    assert evaluated.backward_capture.node_data["rendered_prompt"] == "reefs: Summarize clearly"
    assert evaluated.bindings.bound_inputs == {"topic": "reefs"}
