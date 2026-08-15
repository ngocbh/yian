import pytest

from agent_composer.compile.model import START_ID
from agent_composer.compose.helpers.types import InputDecl
from agent_composer.nodes.base import NodeKind, Output
from agent_composer.nodes.contracts import Parameter
from agent_composer.nodes.start import StartNode
from agent_composer.runtime.eval_node import EvaluationFailure, eval_node
from agent_composer.typesys.pool import VariablePool
from agent_composer.typesys.values import ValueKind, Type


def _decl(name, type_, default=None, required=False, typ=None):
    return InputDecl(name, type_, default, required,
                     typ or Type.scalar(ValueKind.STRING))


def test_start_kind_and_params_are_input_names():
    node = StartNode("__start__", input_decls=[_decl("topic", "str"),
                                               _decl("window", "int")])
    assert node.kind == NodeKind.START
    # params = the input NAMES (so eval_node->bind_params reads its wired sources).
    assert [p.name for p in node.params] == ["topic", "window"]
    assert all(type(p) is Parameter for p in node.params)


def test_start_run_coerces_defaults_into_a_record():
    # integer default coerced ("30" -> 30); a caller-passed string int coerced ("7" -> 7).
    node = StartNode("__start__", input_decls=[
        _decl("topic", "str"),
        _decl("window", "int", default="30",
              typ=Type.scalar(ValueKind.INTEGER)),
        _decl("offset", "int", typ=Type.scalar(ValueKind.INTEGER)),
    ])
    out = node.run({"topic": "ACME", "offset": "7"})
    assert isinstance(out, Output)
    assert out.value == {"topic": "ACME", "offset": 7, "window": 30}  # OBJECT keyed by input name


def test_start_e08_type_mismatch_raises_located_message():
    # a value violating the declared type raises the byte-stable located string.
    node = StartNode("__start__", input_decls=[
        _decl("n", "int", required=True, typ=Type.scalar(ValueKind.INTEGER)),
    ])
    with pytest.raises(Exception, match=r"input `n` — .*does not match declared type int"):
        node.run({"n": ["not", "an", "int"]})


def test_start_through_eval_node_binds_from_wiring():
    # START_ID runs THROUGH the engine seam: bind_params reads its sources from flow.wiring[START_ID],
    # run coerces, and eval_node returns the raw Output record.
    from types import SimpleNamespace

    node = StartNode("__start__", input_decls=[_decl("topic", "str")])
    pool = VariablePool()
    pool.set(START_ID, {"topic": "ACME"})  # the seed-stand-in source the wiring points at
    flow = SimpleNamespace(wiring={"__start__": {"topic": "${input.topic}"}})
    assert eval_node(node, flow, pool) == Output({"topic": "ACME"})


def test_start_through_eval_node_fills_omitted_default():
    # when a child input is OMITTED by the caller (no wiring edge), the START_ID
    # node ITSELF fills the declared default — not the REF/MAP driver. START_ID's params carry
    # `default`/`required`, so bind_params fills the absent input before run() coerces it.
    # (Pre-fix the bare params bound it as present-None, so apply_defaults could not fill it.)
    from types import SimpleNamespace

    node = StartNode("__start__", input_decls=[
        _decl("topic", "str"),
        _decl("window", "int", default="30", typ=Type.scalar(ValueKind.INTEGER)),
    ])
    pool = VariablePool()
    pool.set(START_ID, {"topic": "ACME"})
    # wiring binds ONLY topic — `window` has NO edge (the parent omitted it).
    flow = SimpleNamespace(wiring={"__start__": {"topic": "${input.topic}"}})
    assert eval_node(node, flow, pool) == Output(
        {"topic": "ACME", "window": 30}
    )


def test_start_through_eval_node_required_unbound_fails():
    # START_ID's params carry `required`, so a required input that resolves unbound at the boundary
    # fails loudly (a runtime backstop to the load-time _check_ref_bindings gate).
    from types import SimpleNamespace

    node = StartNode("__start__", input_decls=[
        _decl("n", "int", required=True, typ=Type.scalar(ValueKind.INTEGER)),
    ])
    pool = VariablePool()
    flow = SimpleNamespace(wiring={"__start__": {}})  # nothing bound
    assert isinstance(eval_node(node, flow, pool), EvaluationFailure)
