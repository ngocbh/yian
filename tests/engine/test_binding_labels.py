"""A node's binding labels replace a param's name in its runtime binding errors.

Without a label each message keeps its default ``input '<p>'`` / ``binding '<p>'`` text.
"""

from types import SimpleNamespace

import pytest

from yian.compile.model import START_ID
from yian.nodes.binding import BindingError, bind_params
from yian.nodes.code import CodeNode
from yian.nodes.contracts import Binding, Parameter, ValueContract, ValueTemplate
from yian.runtime.eval_node import EvaluationFailure, eval_node
from yian.typesys import Type, ValueKind
from yian.typesys.pool import VariablePool
from tests.engine._fakes import FuncNode


INTEGER = Type.scalar(ValueKind.INTEGER)


def _pool(**inputs):
    """Return a pool whose flow input holds ``inputs``."""
    pool = VariablePool()
    pool.set(START_ID, inputs)
    return pool


def _fail(source, pool, *, param=None, labels=None) -> BindingError:
    """Bind one parameter (``p`` by default) to ``source`` and return the error raised."""
    parameter = param if param is not None else Parameter("p")
    with pytest.raises(BindingError) as info:
        bind_params([parameter], {parameter.name: source}, pool, labels=labels)
    return info.value


_INT = ValueContract(INTEGER)

_CASES = [
    pytest.param(
        "${input.x * 10}", {"x": 1e308}, None,
        "flow output: non-finite number inf is not a yian value",
        "input 'p': non-finite number inf is not a yian value",
        "binding.evaluate",
        id="nonfinite",
    ),
    pytest.param(
        "${[1.0, input.x * 10]}", {"x": 1e308}, None,
        "flow output at [1]: non-finite number inf is not a yian value",
        "input 'p' at [1]: non-finite number inf is not a yian value",
        "binding.evaluate",
        id="nonfinite-at-path",
    ),
    pytest.param(
        Binding("${input.s}", contract=_INT), {"s": "x"}, None,
        "flow output: 'x' does not match declared type int",
        "binding 'p': 'x' does not match declared type int",
        "binding.type",
        id="typed-binding",
    ),
    pytest.param(
        ValueTemplate({"a": "${input.x / input.z}"}), {"x": 1.0, "z": 0.0}, None,
        "flow output at a: division by zero: 1.0 / 0.0",
        "binding 'p.a': division by zero: 1.0 / 0.0",
        "binding.evaluate",
        id="template-leaf",
    ),
    pytest.param(
        ValueTemplate({"a": Binding("${input.s}", contract=_INT)}), {"s": "x"}, None,
        "flow output at a: 'x' does not match declared type int",
        "binding 'p.a': 'x' does not match declared type int",
        "binding.type",
        id="nested-typed-binding",
    ),
    pytest.param(
        "${input.s}", {"s": "x"}, Parameter("p", contract=_INT),
        "flow output: 'x' does not match declared type int",
        "input 'p': 'x' does not match declared type int",
        "contract.input",
        id="contract-input",
    ),
]


@pytest.mark.parametrize(
    ("source", "inputs", "param", "labeled", "default", "code"), _CASES
)
def test_label_replaces_the_param_name(source, inputs, param, labeled, default, code):
    exc = _fail(source, _pool(**inputs), param=param, labels={"p": "flow output"})

    assert (str(exc), exc.failure_code) == (labeled, code)


@pytest.mark.parametrize(
    ("source", "inputs", "param", "labeled", "default", "code"), _CASES
)
def test_without_a_label_the_default_text_stays(
    source, inputs, param, labeled, default, code
):
    unlabeled = _fail(source, _pool(**inputs), param=param)
    other = _fail(source, _pool(**inputs), param=param, labels={"q": "flow output"})

    assert (str(unlabeled), unlabeled.failure_code) == (default, code)
    assert str(other) == default


def test_fresh_node_has_no_binding_labels():
    assert CodeNode("n", code="return 1").binding_labels == {}


def test_eval_node_uses_the_node_binding_labels():
    node = FuncNode("work", lambda inputs: inputs["p"], reads={"p": "${input.x * 10}"})
    node.binding_labels = {"p": "X"}
    flow = SimpleNamespace(wiring={node.id: node._wiring_src})

    outcome = eval_node(node, flow, _pool(x=1e308)).outcome

    assert isinstance(outcome, EvaluationFailure)
    assert outcome.failure.message == "X: non-finite number inf is not a yian value"
