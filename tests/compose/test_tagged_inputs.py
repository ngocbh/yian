"""Flow inputs declared with YAML tags (`qty: !int`, `history: !list(Message) []`)."""

import re
from textwrap import indent

import pytest

from tests.compose._declarations import parameters_from_yaml
from yian.compose import LoadError, load_flow, read_flow_inputs
from yian.compose.source import SourceDocument
from yian.run import run_flow
from yian.typesys.types import read_typedefs

REG = read_typedefs(
    {
        "Cfg": {"kind": "Literal(cfg)", "a": "int"},
        "Message": {"role": "str", "content": "str"},
    }
)

_DECLARATIONS = """\
qty: !int
history: !list(Message) []
model: !Optional(str)
note: !Optional(str) null
cfg: !Cfg {kind: cfg, a: 1}
s: !str ${x}
"""

_EXPECTED = {
    "qty": 3,
    "history": [],
    "model": None,
    "note": None,
    "cfg": {"kind": "cfg", "a": 1},
    "s": "${x}",
}

_HEADER = """\
id: tagged_inputs
name: tagged_inputs
typedefs:
  Cfg: {kind: Literal(cfg), a: int}
  Message: {role: str, content: str}
"""

_WORKER = """\
kind: code
output: object
code: return dict(inputs)
"""


def _worker_bindings(declarations: str) -> str:
    """Bind every declared input by name, as a multi-node body must."""
    names = [line.split(":", 1)[0] for line in declarations.splitlines() if line]
    return "input:\n" + "".join(f"  {name}: ${{input.{name}}}\n" for name in names)


def _flow(placement: str, declarations: str) -> str:
    """Spell one flow whose `input:` holds `declarations` at `placement`."""
    block = indent(declarations, "  ")
    if placement == "root":
        worker = _WORKER + _worker_bindings(declarations)
        return (
            _HEADER + "input:\n" + block + "nodes:\n  worker:\n"
            + indent(worker, "    ") + "output: ${worker.output}\n"
        )
    if placement == "compact":
        return _HEADER + "input:\n" + block + _WORKER
    caller = (
        "input:\n  qty: !int\nnodes:\n  invoke:\n    kind: call\n    call: child\n"
        "    input:\n      qty: ${input.qty}\noutput: ${invoke.output}\n"
    )
    if placement == "definition":
        worker = _WORKER + _worker_bindings(declarations)
        child = (
            "input:\n" + block + "nodes:\n  worker:\n" + indent(worker, "    ")
            + "output: ${worker.output}\n"
        )
    else:
        child = "input:\n" + block + _WORKER
    return _HEADER + "defs:\n  child:\n" + indent(child, "    ") + caller


PLACEMENTS = ["root", "definition", "flat-definition", "compact"]


@pytest.mark.parametrize("placement", PLACEMENTS)
def test_tagged_inputs_load_and_run_with_their_defaults(placement):
    flow = load_flow(_flow(placement, _DECLARATIONS))
    result = run_flow(flow, {"qty": 3})
    assert result.status == "succeeded", result.failure
    assert result.output == _EXPECTED


@pytest.mark.parametrize("placement", ["root", "compact"])
def test_a_tagged_input_without_a_value_is_required(placement):
    flow = load_flow(_flow(placement, _DECLARATIONS))
    result = run_flow(flow, {})
    assert result.status == "failed"
    assert result.failure.code == "binding.evaluate"
    assert "required input 'qty' is unbound" in result.failure.message


@pytest.mark.parametrize("placement", PLACEMENTS)
@pytest.mark.parametrize(
    "declaration, fragment",
    [
        (
            "cfg: !Cfg {a: 1}",
            "'cfg': the value is not a valid Cfg: record missing required fields: ['kind']",
        ),
        ("n: !int(1..12) 13", "'n': `13` is not a valid int(1..12)"),
        (
            "d: !str 2026-01-01",
            "'d': `2026-01-01` is not a valid str; quote it to keep it as text: "
            '`d: !str "2026-01-01"`',
        ),
    ],
)
def test_a_tagged_default_must_fit_its_tag(placement, declaration, fragment):
    with pytest.raises(LoadError, match=re.escape(fragment)):
        load_flow(_flow(placement, f"qty: !int\n{declaration}\n"))


@pytest.mark.parametrize("placement", PLACEMENTS)
def test_a_tagged_default_with_a_non_string_key_is_a_load_error(placement):
    with pytest.raises(
        LoadError,
        match=re.escape("the key `true` was read by YAML as a boolean"),
    ):
        load_flow(_flow(placement, "qty: !int\nflags: !object {true: 1}\n"))


def test_a_tag_does_not_fill_in_a_missing_field():
    with pytest.raises(LoadError, match=re.escape("missing required fields: ['kind']")):
        parameters_from_yaml("input:\n  cfg: !Cfg {a: 1}\n", REG)
    (parameter,) = parameters_from_yaml("input:\n  cfg: !Cfg {kind: cfg, a: 1}\n", REG)
    assert parameter.default == {"kind": "cfg", "a": 1}


def test_tagged_parameters_read_required_and_default_from_the_value():
    parameters = parameters_from_yaml("input:\n" + indent(_DECLARATIONS, "  "), REG)
    shape = {
        parameter.name: (parameter.required, parameter.has_default, parameter.default)
        for parameter in parameters
    }
    assert shape == {
        "qty": (True, False, None),
        "history": (False, True, []),
        "model": (False, False, None),
        "note": (False, True, None),
        "cfg": (False, True, {"kind": "cfg", "a": 1}),
        "s": (False, True, "${x}"),
    }
    types = {parameter.name: parameter.contract.type for parameter in parameters}
    assert types["model"].nullable and types["history"].kind.is_list()
    assert all(parameter.coerce_strings for parameter in parameters)


def test_tagged_parameter_origins_point_at_the_declaration():
    text = _HEADER + "input:\n  qty: !int\n  history: !list(Message) []\n"
    flow = load_flow(text + _WORKER)
    cursor = SourceDocument.parse(text).root.child("input")
    resolver = flow.origin_resolver
    parameters = {
        parameter.name: parameter
        for parameter in flow.compiled.nodes["__start__"].params
    }
    history = parameters["history"]
    assert resolver.resolve(history.origin).range == cursor.child("history").key_range
    for origin in (history.contract.origins.default, history.default_origin):
        assert resolver.resolve(origin).range == cursor.child("history").range
    qty = parameters["qty"]
    assert resolver.resolve(qty.contract.origins.default).range == cursor.child("qty").range
    assert qty.default_origin is None


def test_source_free_inputs_accept_the_parenthesized_spelling():
    (parameter,) = read_flow_inputs({"x": "Optional(int)"}, {})
    assert parameter.contract.type.nullable and not parameter.required


@pytest.mark.parametrize(
    "text, prefix",
    [
        (
            "id: f\nname: f\ninput: [topic]\n" + _WORKER,
            "compact (single-node) flow: ",
        ),
        (
            "id: f\nname: f\ndefs:\n  child:\n    input: [topic]\n" + indent(_WORKER, "    ")
            + "nodes: {}\n",
            "defs entry 'child': single-node ",
        ),
    ],
    ids=["compact", "flat-definition"],
)
def test_a_single_node_input_list_shows_the_tagged_spelling(text, prefix):
    message = prefix + "`input:` must be a param map (`name: !TYPE`, e.g. `topic: !str`)"
    with pytest.raises(LoadError, match=re.escape(message)):
        load_flow(text)
