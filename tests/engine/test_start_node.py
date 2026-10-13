from yian.compile.model import START_ID
from yian.nodes.base import NodeKind, Output
from yian.nodes.contracts import Parameter, ValueContract
from yian.nodes.start import StartNode
from yian.runtime.eval_node import EvaluationFailure, eval_node
from yian.typesys.pool import VariablePool
from yian.typesys.values import ValueKind, Type


_MISSING = object()


def _parameter(name, *, default=_MISSING, required=False, typ=None):
    return Parameter(
        name=name,
        contract=ValueContract(typ or Type.scalar(ValueKind.STRING)),
        required=required,
        has_default=default is not _MISSING,
        default=None if default is _MISSING else default,
        coerce_strings=True,
    )


def test_start_kind_and_params_are_input_names():
    node = StartNode(
        "__start__",
        params=[_parameter("topic"), _parameter("window")],
    )
    assert node.kind == NodeKind.START
    # params = the input NAMES (so eval_node->bind_params reads its wired sources).
    assert [p.name for p in node.params] == ["topic", "window"]
    assert all(type(p) is Parameter for p in node.params)


def test_start_through_eval_node_defaults_into_a_record_without_converting_text():
    from types import SimpleNamespace

    node = StartNode(
        "__start__",
        params=[
            _parameter("topic"),
            _parameter("window", default=30, typ=Type.scalar(ValueKind.INTEGER)),
            _parameter("offset", typ=Type.scalar(ValueKind.INTEGER)),
        ],
    )
    pool = VariablePool()
    pool.set(START_ID, {"topic": "ACME", "offset": 7})
    flow = SimpleNamespace(
        wiring={
            "__start__": {
                "topic": "${input.topic}",
                "offset": "${input.offset}",
            }
        }
    )

    assert eval_node(node, flow, pool).outcome == Output(
        {"topic": "ACME", "offset": 7, "window": 30}
    )

    # Text is converted only by `run_flow` before START is seeded, never by START itself.
    pool.set(START_ID, {"topic": "ACME", "offset": "7"})
    failure = eval_node(node, flow, pool).outcome.failure
    assert failure.code == "contract.input"
    assert "input 'offset': '7' does not match declared type int" in failure.message


def test_start_e08_type_mismatch_returns_generic_input_failure():
    from types import SimpleNamespace

    node = StartNode(
        "__start__",
        params=[
            _parameter("n", required=True, typ=Type.scalar(ValueKind.INTEGER))
        ],
    )
    pool = VariablePool()
    pool.set(START_ID, {"n": ["not", "an", "int"]})
    flow = SimpleNamespace(wiring={"__start__": {"n": "${input.n}"}})

    result = eval_node(node, flow, pool).outcome

    assert isinstance(result, EvaluationFailure)
    assert result.failure.code == "contract.input"
    assert "does not match declared type int" in result.failure.message
    assert result.failure.trace == ()


def test_start_through_eval_node_binds_from_wiring():
    # START_ID runs THROUGH the engine seam: bind_params reads its sources from flow.wiring[START_ID],
    # bind_params coerces, and eval_node returns the raw Output record.
    from types import SimpleNamespace

    node = StartNode("__start__", params=[_parameter("topic")])
    pool = VariablePool()
    pool.set(START_ID, {"topic": "ACME"})  # the seed-stand-in source the wiring points at
    flow = SimpleNamespace(wiring={"__start__": {"topic": "${input.topic}"}})
    assert eval_node(node, flow, pool).outcome == Output({"topic": "ACME"})


def test_start_through_eval_node_fills_omitted_default():
    # when a child input is OMITTED by the caller (no wiring edge), the START_ID
    # node itself fills the declared default, not the CALL/MAP driver. START_ID's params carry
    # `default`/`required`, so bind_params fills the absent input before run().
    from types import SimpleNamespace

    node = StartNode(
        "__start__",
        params=[
            _parameter("topic"),
            _parameter("window", default=30, typ=Type.scalar(ValueKind.INTEGER)),
        ],
    )
    pool = VariablePool()
    pool.set(START_ID, {"topic": "ACME"})
    # wiring binds ONLY topic — `window` has NO edge (the parent omitted it).
    flow = SimpleNamespace(wiring={"__start__": {"topic": "${input.topic}"}})
    assert eval_node(node, flow, pool).outcome == Output(
        {"topic": "ACME", "window": 30}
    )


def test_start_through_eval_node_required_unbound_fails():
    # START_ID's params carry `required`, so a required input that resolves unbound at the boundary
    # fails loudly (a runtime backstop to the load-time _check_ref_bindings gate).
    from types import SimpleNamespace

    node = StartNode(
        "__start__",
        params=[
            _parameter("n", required=True, typ=Type.scalar(ValueKind.INTEGER))
        ],
    )
    pool = VariablePool()
    flow = SimpleNamespace(wiring={"__start__": {}})  # nothing bound
    assert isinstance(eval_node(node, flow, pool).outcome, EvaluationFailure)
