"""A flow's `output:` shape decides its result: `${x}` returns x, `{...}` always a record."""

import pytest

from yian.compile.model import END_ID, START_ID, FlowOutput
from yian.compose import LoadError, load_flow
from yian.compose.helpers.output_contracts import combine_output_contracts
from yian.compose.nodes.resolver import ChildSignature, child_signature
from yian.nodes.base import BackwardResult
from yian.nodes.contracts import ValueContract
from yian.nodes.end import EndNode
from yian.run import run_flow
from yian.runtime.learning import validate_attributed_backward
from yian.typesys.values import Type, ValueKind
from tests.engine._backward_fakes import RecordingBackwardNode
from tests.engine._graph_builder import _engine, _graph


_STEP = """
  step:
    kind: code
    output: str
    code: |
      return "filled"
"""

_FLOW = """
id: shape
name: shape
nodes:""" + _STEP + "%s"

# A def whose `output:` is the given section, indented under `defs:`.
_DEF = """
  child:
    nodes:
      step:
        kind: code
        output: str
        code: |
          return "filled"
%s"""


def _run(text, inputs=None):
    return run_flow(load_flow(text), inputs or {}).output


def _parent(def_output, nodes, output):
    return (
        "id: parent\nname: parent\ndefs:"
        + _DEF % def_output
        + "nodes:\n"
        + nodes
        + f"output: {output}\n"
    )


_ONE_FIELD_DEF_OUTPUT = "    output:\n      order: ${step.output}\n"


@pytest.mark.parametrize(
    ("section", "expected"),
    [
        ("output:\n  order: ${step.output}\n", {"order": "filled"}),
        (
            "output:\n  order: ${step.output}\n  note: ${step.output}\n",
            {"order": "filled", "note": "filled"},
        ),
        ("output: ${step.output}\n", "filled"),
        ("", None),
        ("output: {}\n", {}),
    ],
    ids=["one-field", "two-fields", "value", "absent", "empty-record"],
)
def test_flow_result_follows_the_output_shape(section, expected):
    assert _run(_FLOW % section) == expected


@pytest.mark.parametrize(
    ("output", "expected"),
    [("${c.output.order}", "filled"), ("${c.output}", {"order": "filled"})],
)
def test_call_over_a_one_field_record_def_returns_the_record(output, expected):
    text = _parent(_ONE_FIELD_DEF_OUTPUT, "  c:\n    kind: call\n    call: child\n", output)

    assert _run(text) == expected


def test_call_over_a_flat_single_node_def_returns_the_bare_value():
    text = """
id: parent
name: parent
defs:
  greet:
    kind: code
    output: str
    code: |
      return "hi"
nodes:
  c:
    kind: call
    call: greet
output: ${c.output}
"""
    assert _run(text) == "hi"


def test_inline_call_body_with_a_one_field_output_returns_the_record():
    text = """
id: parent
name: parent
nodes:
  c:
    kind: call
    nodes:
      step:
        kind: code
        output: str
        code: |
          return "filled"
    output:
      order: ${step.output}
output: ${c.output}
"""
    assert _run(text) == {"order": "filled"}


_MAP_NODES = """  each:
    kind: map
    call: child
    over: ${input.items}
"""


def _map_parent(def_output):
    return "input:\n  items: list[str]\n" + _parent(def_output, _MAP_NODES, "${each.output}")


def test_map_over_a_one_field_record_def_returns_a_list_of_records():
    text = _map_parent(_ONE_FIELD_DEF_OUTPUT)
    loaded = load_flow(text)

    contract = loaded.compiled.nodes["each"].output_contract
    assert contract.type.kind is ValueKind.LIST_OBJECT
    assert contract.type.element.fields == {"order": Type.scalar(ValueKind.STRING)}
    assert run_flow(loaded, {"items": ["a", "b"]}).output == [
        {"order": "filled"},
        {"order": "filled"},
    ]


@pytest.mark.parametrize("def_output", ["", "    output: {}\n"], ids=["absent", "empty-record"])
def test_map_over_a_def_without_fields_is_rejected(def_output):
    with pytest.raises(LoadError, match="must declare exactly one output value"):
        load_flow(_map_parent(def_output))


def test_nested_value_re_export_of_a_record_def_keeps_the_record():
    text = """
id: parent
name: parent
defs:
  b:
    nodes:
      step:
        kind: code
        output: str
        code: |
          return "filled"
    output:
      order: ${step.output}
  a:
    nodes:
      inner:
        kind: call
        call: b
    output: ${inner.output}
nodes:
  c:
    kind: call
    call: a
output: ${c.output.order}
"""
    assert _run(text) == "filled"


def test_case_then_call_over_a_one_field_record_def_returns_the_record():
    nodes = """  fallback:
    kind: code
    output: str
    code: |
      return "never"
  route:
    kind: case
    cases:
      - when: "${input.go}"
        then: call(child)
    else: fallback
"""
    text = "input:\n  go: bool\n" + _parent(_ONE_FIELD_DEF_OUTPUT, nodes, "${route.output.order}")

    assert _run(text, {"go": True}) == "filled"


def test_child_signature_of_a_one_field_record_is_its_exact_record():
    sig = child_signature(load_flow(_FLOW % "output:\n  order: ${step.output}\n"))

    assert sig.returns_record is True
    assert type(sig.output) is ValueContract
    assert sig.output.type.kind is ValueKind.OBJECT
    assert sig.output.type.fields == {"order": Type.scalar(ValueKind.STRING)}


def test_child_signature_of_a_value_output_is_the_value_contract():
    sig = child_signature(load_flow(_FLOW % "output: ${step.output}\n"))

    assert sig.returns_record is False
    assert type(sig.output) is ValueContract
    assert sig.output.type == Type.scalar(ValueKind.STRING)


def test_child_signature_of_an_empty_record_is_the_empty_record_contract():
    sig = child_signature(load_flow(_FLOW % "output: {}\n"))

    assert sig.returns_record is True
    assert sig.output == combine_output_contracts({})


@pytest.mark.parametrize(
    "def_output", [_ONE_FIELD_DEF_OUTPUT, "    output: {}\n"], ids=["one-field", "empty-record"]
)
def test_reading_a_missing_field_of_a_record_child_is_rejected_at_load(def_output):
    # Records the current behavior: a field read on a child's statically known record
    # contract is already checked at load, for one field and for none.
    text = _parent(def_output, "  c:\n    kind: call\n    call: child\n", "${c.output.missing}")

    with pytest.raises(LoadError, match="reads unknown field 'missing' on record"):
        load_flow(text)


def test_direct_child_signature_must_say_it_returns_a_record():
    text = ValueContract(Type.scalar(ValueKind.STRING))
    with pytest.raises(ValueError):
        ChildSignature(
            inputs=[],
            output=combine_output_contracts({"a": text, "b": text}),
            declared_outputs=(("a", text), ("b", text)),
            returns_record=False,
        )


def test_loop_body_with_an_empty_record_output_is_rejected():
    text = """
id: l
name: l
nodes:
  lp:
    kind: loop
    times: 2
    input:
      count: 0
    nodes:
      bump:
        kind: code
        output: int
        code: |
          return 1
    output: {}
output: ${lp.output}
"""
    with pytest.raises(
        LoadError, match="whose field names are a subset of the carried record"
    ):
        load_flow(text)


def test_loop_body_with_a_one_field_record_output_runs():
    text = """
id: l
name: l
nodes:
  lp:
    kind: loop
    times: 3
    input:
      count: 0
    nodes:
      bump:
        kind: code
        input:
          n: ${input.count}
        output: int
        code: |
          return inputs["n"] + 1
    output:
      count: ${bump.output}
output: ${lp.output}
"""
    assert _run(text) == {"count": 3}


def test_root_feedback_on_a_one_field_record_reaches_its_producer():
    from yian.runtime.backward import compute_backward

    def passthrough(_context, feedback):
        return BackwardResult({"x": feedback})

    producer = RecordingBackwardNode(
        "producer", lambda inputs: inputs["x"], passthrough, reads={"x": "${input.topic}"}
    )
    end = EndNode.record(END_ID, output_names=["order"])
    flow = _graph(
        [producer, end],
        [(START_ID, "producer")],
        outputs=[FlowOutput("order", "${producer.output}")],
    )
    engine = _engine(flow, {"topic": "reefs"}, learnable=True)
    list(engine.run())
    assert engine.state.trace is not None

    feedback = {"order": "Review the order."}
    root, results = compute_backward(engine.state.trace, feedback)
    validate_attributed_backward(engine.state.trace, feedback, root, results)

    assert len(producer.calls) == 1
    assert producer.calls[0][1] == ("Review the order.",)
    assert root.input_feedback == {"topic": ("Review the order.",)}
