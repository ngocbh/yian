"""`call` build + child-signature resolution — LOAD/COMPILE facts.

The loader resolves a callable to a `ComposedFlow`, derives its signature, re-exports the
codomain, and name/arity-checks bindings. (End-to-end RUN lives in test_ref_run.py /
test_map.py; the cross-flow type check e06 in test_errors.py.) These assert:

- seed 03 (`research-one`) loads as a child flow (its `ComposedFlow` exposes a
  resolvable child SIGNATURE — canonical `Parameter`s + the single codomain `Type`);
- seed 04 (a plain `call` over 03) loads+compiles; the built `CallNode` (REF)
  `output_type` re-exports the child's single codomain `Type` (a `{report, asof}` record)
  via the test-local child resolver;
- seed 05 (a `map` over 03) loads+compiles; the built `MapNode` (`kind: map`)
  `output_type == list[<child codomain>]` (a `LIST_OBJECT` of the record) and `parallel=True`;
- a `call` binding to a non-declared callable input → loud `LoadError`;
- a `map` whose callable has no codomain (≠1 declared output value) → loud `LoadError`;
- a flow with a `call` but no resolver → loud `LoadError`.
"""

from pathlib import Path

import pytest

from agent_composer.nodes.call import CallNode   # nodes/call/__init__.py
from agent_composer.nodes.contracts import Binding, Parameter, ValueContract
from agent_composer.nodes.map import MapNode
from agent_composer.typesys.values import ValueKind
from agent_composer.compose import ComposedFlow, LoadError, load_flow
from agent_composer.compose.nodes.resolver import ChildSignature, child_signature

_SEEDS = Path(__file__).resolve().parents[2] / "tests" / "seeds"


def _text(name: str) -> str:
    return (_SEEDS / name).read_text()


# A test-local child resolver: child flow id -> its loaded `ComposedFlow` (the loader
# derives the signature + bakes the compiled child). Seeds 04/05 ref `research-one`
# (= seed 03's id); we load seed 03 once and hand it back by id.
def _make_resolver():
    children = {"research-one": load_flow(_text("03-research-one.yaml"))}

    def resolver(flow_id: str, version=None):
        try:
            return children[flow_id]
        except KeyError as exc:
            raise LoadError(f"unknown child flow {flow_id!r}") from exc

    return resolver


# --------------------------------------------------------------------------- #
# seed 03 — the child flow loads + exposes a signature
# --------------------------------------------------------------------------- #


def test_seed03_loads_as_child_flow():
    loaded = load_flow(_text("03-research-one.yaml"))
    assert isinstance(loaded, ComposedFlow)
    sig = child_signature(loaded)
    assert isinstance(sig, ChildSignature)
    # declared inputs: topic (required str) + as_of (typed date default -> not required).
    parameters = {parameter.name: parameter for parameter in sig.inputs}
    assert set(parameters) == {"topic", "as_of"}
    assert all(type(parameter) is Parameter for parameter in sig.inputs)
    assert all(type(parameter.contract) is ValueContract for parameter in sig.inputs)
    assert parameters["topic"].required is True
    assert parameters["topic"].contract.type.kind is ValueKind.STRING
    assert parameters["as_of"].required is False
    assert parameters["as_of"].has_default is True
    assert parameters["as_of"].default == "2026-01-01"
    assert parameters["as_of"].contract.type.kind is ValueKind.DATE
    assert all(parameter.coerce_strings for parameter in sig.inputs)
    # the single codomain value: a {report, asof} record (>=2 outputs -> closed record).
    assert type(sig.output) is ValueContract
    assert sig.output.type.kind == ValueKind.OBJECT
    assert set(sig.output.type.fields) == {"report", "asof"}


# --------------------------------------------------------------------------- #
# seed 04 — REF loads + the REF node re-exports the child's codomain Type
# --------------------------------------------------------------------------- #


def test_seed04_ref_loads_and_reexports_child_output():
    loaded = load_flow(_text("04-call.yaml"), search_paths=[_SEEDS])
    assert isinstance(loaded, ComposedFlow)
    research = loaded.compiled.nodes["research"]
    assert isinstance(research, CallNode)               # REF -> CallNode
    assert not hasattr(research, "over")               # REF carries no over/parallel
    assert not hasattr(research, "parallel")
    assert research.flow_id == "research-one"
    # output_type re-exports the child's single codomain (the {report, asof} record).
    typ = research.output_type
    assert typ is not None
    assert typ.kind == ValueKind.OBJECT
    assert set(typ.fields) == {"report", "asof"}
    # the REF sources live on flow.wiring; the node carries only the param names.
    assert loaded.compiled.wiring["research"] == {
        "topic": Binding("${input.topic}")
    }
    assert [p.name for p in research.params] == ["topic"]


def test_seed04_downstream_dotted_ref_validates_against_child_record():
    # `take` reads ${research.output.report}/${research.output.asof}; the REF node's
    # stamped output_type (a CHECKED record) lets the dotted-field walk pass (not loud).
    loaded = load_flow(_text("04-call.yaml"), search_paths=[_SEEDS])
    assert loaded.compiled.wiring["take"] == {
        "brief": Binding("${research.output.report}"),
        "asof": Binding("${research.output.asof}"),
    }


# --------------------------------------------------------------------------- #
# seed 05 — MAP loads + output_type == list[<child codomain>], parallel=True
# --------------------------------------------------------------------------- #


def test_seed05_map_loads_with_list_output_and_parallel():
    loaded = load_flow(_text("05-call-map.yaml"), search_paths=[_SEEDS])
    assert isinstance(loaded, ComposedFlow)
    research_each = loaded.compiled.nodes["research_each"]
    assert isinstance(research_each, MapNode)          # kind: map -> MapNode
    assert not hasattr(research_each, "over")          # the SOURCE rides flow.wiring, not the node
    assert research_each.flow_id == "research-one"
    assert research_each.parallel is True
    assert research_each.title == "Research each topic"  # node_name -> title
    # output_type = list[<child codomain>] -> a LIST_OBJECT (element = the record).
    typ = research_each.output_type
    assert typ is not None
    assert typ.kind == ValueKind.LIST_OBJECT
    assert typ.element.kind == ValueKind.OBJECT
    assert set(typ.element.fields) == {"report", "asof"}
    # the MAP sources (incl the reserved `over`, over-first) live on flow.wiring; the node
    # carries only the per-element param names (`over:` is the iteration source, not a param).
    assert loaded.compiled.wiring["research_each"] == {
        "over": Binding("${input.topics}"),
        "topic": Binding("${item}"),
        "as_of": Binding('${input.as_of:-"today"}'),
    }
    assert [p.name for p in research_each.params] == ["topic", "as_of"]


def test_map_input_named_over_is_reserved():
    # a `map`'s `over` is the iteration source (the reserved wiring key);
    # an `inputs:` param named `over` collides and is rejected at load.
    text = (
        "id: bad-over\nname: bad\ninput:\n  xs: list[str]\n"
        "nodes:\n  m:\n    kind: map\n    call: child\n    over: ${input.xs}\n"
        "    inputs:\n      over: ${item}\n"
        "output: ${m.output}\n"
    )
    with pytest.raises(LoadError, match="'over' is reserved"):
        load_flow(text)


# --------------------------------------------------------------------------- #
# name/arity validation — loud on a bad binding / a bad MAP child
# --------------------------------------------------------------------------- #

_REF_BAD_NAME = """
id: bad-ref
name: bad_ref
input:
  topic: str
uses:
  research-one: research-one
nodes:
  research:
    kind: call
    call: research-one
    input:
      symbol: ${input.topic}     # `symbol` is NOT a declared callable input
output:
  out: ${research.output}
"""


def test_ref_binding_to_undeclared_child_input_is_loud():
    with pytest.raises(LoadError) as exc:
        load_flow(_REF_BAD_NAME, child_resolver=_make_resolver())
    msg = str(exc.value)
    assert "symbol" in msg
    assert "research-one" in msg


_CHILD_NO_OUTPUT = """
id: no-output
name: no_output
input:
  topic: str
nodes:
  emit:
    kind: code
    input:
      topic: ${input.topic}
    code: m:f
"""

_MAP_OVER_NO_OUTPUT = """
id: bad-map
name: bad_map
input:
  topics: list[str]
nodes:
  each:
    kind: map
    call: no-output
    over: ${input.topics}
    input:
      topic: ${item}
output:
  out: ${each.output}
"""


def test_map_child_with_no_output_is_loud():
    child = load_flow(_CHILD_NO_OUTPUT)
    assert child_signature(child).output is None  # no codomain -> MAP cannot form list[U]

    def resolver(flow_id, version=None):
        assert flow_id == "no-output"
        return child

    with pytest.raises(LoadError) as exc:
        load_flow(_MAP_OVER_NO_OUTPUT, child_resolver=resolver)
    assert "exactly one" in str(exc.value) or "output" in str(exc.value).lower()


# --------------------------------------------------------------------------- #
# no resolver -> loud (mirror the legacy validate_flow REF/MAP resolver requirement)
# --------------------------------------------------------------------------- #


def test_ref_map_without_resolver_is_loud():
    with pytest.raises(LoadError) as exc:
        load_flow(_text("04-call.yaml"))
    assert "resolver" in str(exc.value).lower()


def test_refless_flow_still_loads_without_resolver():
    # a refless flow (no call) must still load with NO resolver.
    loaded = load_flow(_text("01-structured-agent.yaml"))
    assert isinstance(loaded, ComposedFlow)
