import pytest

from yian.nodes.base import NodeKind, Output
from yian.nodes.contracts import Assertion, Parameter
from yian.nodes.end import EndNode


def test_end_kind():
    assert EndNode.record("__end__", output_names=["a"]).kind == NodeKind.END


# --- RECORD mode (a flow's `output: {...}`) — always a dict keyed by the declared names.
def test_end_record_zero_outputs_returns_empty_record():
    out = EndNode.record("__end__", output_names=[]).run({})
    assert isinstance(out, Output) and out.value == {}


def test_end_record_one_output_returns_record():
    out = EndNode.record("__end__", output_names=["only"]).run({"only": 42})
    assert out.value == {"only": 42}


def test_end_record_one_output_captures_a_keyed_path():
    node = EndNode.record("__end__", output_names=["only"])
    assert node.capture_backward_context() == {"input_output_paths": {"only": ["only"]}}


def test_end_record_two_outputs_returns_keyed_object():
    node = EndNode.record("__end__", output_names=["report", "n"])
    out = node.run({"report": "r", "n": 4})
    assert out.value == {"report": "r", "n": 4}


def test_end_record_params_are_output_names():
    node = EndNode.record("__end__", output_names=["report", "n"])
    assert [p.name for p in node.params] == ["report", "n"]


# --- VALUE mode (a flow's `output: ${x}`, or no output) — returns the one value or None.
def test_end_value_returns_the_bare_value():
    node = EndNode.value("__end__", output_name="only")
    assert [p.name for p in node.params] == ["only"]
    assert node.returns_record is False
    assert node.run({"only": 42}).value == 42


def test_end_value_without_a_name_has_no_params_and_returns_none():
    node = EndNode.value("__end__", output_name=None)
    assert node.params == []
    assert node.run({}).value is None


def test_end_value_captures_the_whole_output_path():
    node = EndNode.value("__end__", output_name="only")
    assert node.backward_input_names() == ("only",)
    assert node.capture_backward_context() == {"input_output_paths": {"only": []}}


def test_end_record_returns_record():
    assert EndNode.record("__end__", output_names=["a", "b"]).returns_record is True


def test_end_value_rejects_more_than_one_name():
    with pytest.raises(ValueError):
        EndNode("__end__", output_names=["a", "b"], value=True)
    with pytest.raises(ValueError):
        EndNode("__end__", n=1, value=True)


# --- LIST mode (the MAP fan-in) — replaces COLLECTOR; join in over order (index).
def test_end_list_joins_in_over_order():
    node = EndNode.list_("__end__", n=3)
    out = node.run({"e0": "ACME", "e1": "BETA", "e2": "GOOG"})
    assert out.value == ["ACME", "BETA", "GOOG"]


def test_end_list_n_zero_returns_empty_list():
    out = EndNode.list_("__end__", n=0).run({})
    assert out.value == []


def test_end_list_params_are_e0_to_en():
    node = EndNode.list_("__end__", n=2)
    assert [p.name for p in node.params] == ["e0", "e1"]


def test_end_post_asserts_fire_record_scoped_through_eval_node():
    """An END post-assert fires RECORD-scoped through eval_node's `run` wrapper: the producer
    ref is LOWERED to a synth param (`__assert_emit`) fed by a `${emit.output.n:-null}` wiring, so
    bind_params binds it INTO the record and the assert reads `${__assert_emit}` node-locally. This
    is the production shape after ref lowering (the pool-fallback in eval_node was retired)."""
    from types import SimpleNamespace

    from yian.runtime.eval_node import EvaluationFailure, eval_node
    from yian.typesys.pool import VariablePool

    pool = VariablePool()
    pool.set("emit", {"n": 4})  # the producer value bound into the record via the synth param

    node = EndNode.value("__end__", output_name="n")
    node.post_asserts = [Assertion("${__assert_emit} > 0", "emit must be positive")]
    node.params.append(Parameter(name="__assert_emit"))
    flow = SimpleNamespace(wiring={"__end__": {"n": "${emit.output.n}",
                                              "__assert_emit": "${emit.output.n:-null}"}})
    assert eval_node(node, flow, pool).outcome == Output(4)

    bad = EndNode.value("__end__", output_name="n")
    bad.post_asserts = [Assertion("${__assert_emit} > 100", "emit must exceed 100")]
    bad.params.append(Parameter(name="__assert_emit"))
    failed = eval_node(bad, flow, pool).outcome
    assert isinstance(failed, EvaluationFailure)
    assert "post-assert failed" in failed.failure.message


def test_end_post_asserts_resolve_namespaced_cross_node_ref_via_synth_param():
    """A flow terminal's END post-assert reading a NAMESPACED cross-node ref
    (`${each#0/n.output.X}`, the shape expand produces on a fanned-in child END) is LOWERED to a
    synth param bound into the record: the assert holds/fails record-scoped with no END special
    case and no `node.kind` read (the pool-fallback firing path is gone)."""
    from types import SimpleNamespace

    from yian.runtime.eval_node import EvaluationFailure, eval_node
    from yian.typesys.pool import VariablePool

    pool = VariablePool()
    pool.set("each#0/n", {"score": 7})           # a namespaced child value bound in via the param
    wiring = {"__end__": {"n": "${each#0/n.output.score}",
                          "__assert_score": "${each#0/n.output.score:-null}"}}
    flow = SimpleNamespace(wiring=wiring)

    ok_node = EndNode.value("__end__", output_name="n")
    ok_node.post_asserts = [Assertion("${__assert_score} > 0", "score must be positive")]
    ok_node.params.append(Parameter(name="__assert_score"))
    assert eval_node(ok_node, flow, pool).outcome == Output(7)

    bad_node = EndNode.value("__end__", output_name="n")
    bad_node.post_asserts = [Assertion("${__assert_score} > 100", "score must exceed 100")]
    bad_node.params.append(Parameter(name="__assert_score"))
    failed = eval_node(bad_node, flow, pool).outcome
    assert isinstance(failed, EvaluationFailure)
    assert "post-assert failed" in failed.failure.message
