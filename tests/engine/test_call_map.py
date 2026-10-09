"""`call` build + child-signature resolution — LOAD/COMPILE facts.

The loader resolves a callable to a `ComposedFlow`, derives its signature, re-exports the
codomain, and name/arity-checks bindings. (End-to-end RUN lives in test_call_run.py /
test_map.py; the cross-flow type check e06 in test_errors.py.) These assert:

- seed 03 (`research-one`) loads as a child flow (its `ComposedFlow` exposes a
  resolvable child SIGNATURE — canonical `Parameter`s + one codomain `ValueContract`);
- seed 04 (a plain `call` over 03) loads+compiles; the built `CallNode`
  re-exports the child's exact codomain contract (a `{report, asof}` record) via the
  test-local child resolver;
- seed 05 (a `map` over 03) loads+compiles; the built `MapNode` (`kind: map`)
  has `output_contract.type == list[<child codomain>]` (a `LIST_OBJECT` of the record)
  and `parallel=True`;
- a `call` binding to a non-declared callable input → loud `LoadError`;
- a `map` whose callable has no codomain (≠1 declared output value) → loud `LoadError`;
- a flow with a `call` but no resolver → loud `LoadError`.
"""

from pathlib import Path

import pytest

from yian.nodes.call import CallNode   # nodes/call/__init__.py
from yian.nodes.contracts import Parameter, ValueContract
from yian.nodes.map import MapNode
from yian.typesys.values import ValueKind
from yian.compose import ComposedFlow, LoadError, load_flow
from yian.compose.helpers.errors import LoadDiagnostics
from yian.compose.nodes.resolver import ChildSignature, child_signature

_SEEDS = Path(__file__).resolve().parents[2] / "tests" / "seeds"


def _text(name: str) -> str:
    return (_SEEDS / name).read_text()


# A test-local child resolver: child flow id -> its loaded `ComposedFlow` (the loader
# derives the signature + bakes the compiled child). Seeds 04/05 call `research-one`
# (= seed 03's id); we load seed 03 once and hand it back by id.
def _make_resolver():
    children = {"research-one": load_flow(_text("03-research-one.yaml"))}

    def resolver(flow_id: str, version=None):
        try:
            return children[flow_id]
        except KeyError as exc:
            raise LoadDiagnostics().error(
                f"unknown child flow {flow_id!r}"
            ) from exc

    return resolver


# --------------------------------------------------------------------------- #
# seed 03 — the child flow loads + exposes a signature
# --------------------------------------------------------------------------- #


def test_seed03_loads_as_child_flow():
    loaded = load_flow(_text("03-research-one.yaml"))
    assert isinstance(loaded, ComposedFlow)
    sig = child_signature(loaded)
    assert isinstance(sig, ChildSignature)
    # Declared input parameters: topic (required str) + as_of (typed date default -> not required).
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
# seed 04 — CALL loads and re-exports the child's codomain contract
# --------------------------------------------------------------------------- #


def test_seed04_call_loads_and_reexports_child_output():
    loaded = load_flow(_text("04-call.yaml"), search_paths=[_SEEDS])
    assert isinstance(loaded, ComposedFlow)
    research = loaded.compiled.nodes["research"]
    assert isinstance(research, CallNode)
    assert not hasattr(research, "over")               # CALL carries no over/parallel
    assert not hasattr(research, "parallel")
    assert research.flow_id == "research-one"
    # The exact contract re-exports the child's single codomain ({report, asof}).
    contract = research.output_contract
    assert contract is not None
    typ = contract.type
    assert typ.kind == ValueKind.OBJECT
    assert set(typ.fields) == {"report", "asof"}
    # CALL sources live on flow.wiring; the node carries only the parameter names.
    wiring = loaded.compiled.wiring["research"]
    assert {name: binding.source for name, binding in wiring.items()} == {
        "topic": "${input.topic}",
    }
    assert wiring["topic"].origin is not None
    assert [p.name for p in research.params] == ["topic"]
    assert research.params[0].origin is not None


def test_seed04_downstream_dotted_ref_validates_against_child_record():
    # `take` reads ${research.output.report}/${research.output.asof}; the CALL node's
    # stamped output contract (a checked record) lets the dotted-field walk pass.
    loaded = load_flow(_text("04-call.yaml"), search_paths=[_SEEDS])
    wiring = loaded.compiled.wiring["take"]
    assert {name: binding.source for name, binding in wiring.items()} == {
        "brief": "${research.output.report}",
        "asof": "${research.output.asof}",
    }
    assert all(binding.origin is not None for binding in wiring.values())
    assert all(
        parameter.origin is not None
        for parameter in loaded.compiled.nodes["take"].params
    )


# --------------------------------------------------------------------------- #
# seed 05 — MAP loads + list[<child codomain>] output contract, parallel=True
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
    # The contract type is list[<child codomain>] (element = the record).
    contract = research_each.output_contract
    assert contract is not None
    typ = contract.type
    assert typ.kind == ValueKind.LIST_OBJECT
    assert typ.element.kind == ValueKind.OBJECT
    assert set(typ.element.fields) == {"report", "asof"}
    # the MAP sources (incl the reserved `over`, over-first) live on flow.wiring; the node
    # carries only the per-element param names (`over:` is the iteration source, not a param).
    wiring = loaded.compiled.wiring["research_each"]
    assert {name: binding.source for name, binding in wiring.items()} == {
        "over": "${input.topics}",
        "topic": "${item}",
        "as_of": '${input.as_of:-"today"}',
    }
    assert wiring["over"].origin is not None
    assert wiring["topic"].origin is not None
    assert wiring["as_of"].origin is not None
    assert [p.name for p in research_each.params] == ["topic", "as_of"]
    assert all(parameter.origin is not None for parameter in research_each.params)


def test_map_input_named_over_is_reserved():
    # a `map`'s `over` is the iteration source (the reserved wiring key);
    # an `input:` parameter named `over` collides and is rejected at load.
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

_CALL_BAD_NAME = """
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


def test_call_binding_to_undeclared_child_input_is_loud():
    with pytest.raises(LoadError) as exc:
        load_flow(_CALL_BAD_NAME, child_resolver=_make_resolver())
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
    signature = child_signature(child)
    assert signature.output is None
    assert signature.output_names == ()  # no declaration -> MAP cannot form list[U]
    assert signature.declared_outputs == ()

    def resolver(flow_id, version=None):
        assert flow_id == "no-output"
        return child

    with pytest.raises(LoadError) as exc:
        load_flow(_MAP_OVER_NO_OUTPUT, child_resolver=resolver)
    assert "exactly one" in str(exc.value) or "output" in str(exc.value).lower()


_NULL_TO_NULLABLE_CALL = """
id: null-to-nullable-call
name: null_to_nullable_call
defs:
  echo:
    input:
      maybe: Optional[str]
    nodes: {}
    output: ${input.maybe}
nodes:
  invoke:
    kind: call
    call: echo
    input:
      maybe: ${null}
output: ${invoke.output}
"""


def test_call_accepts_proven_null_for_nullable_child_input():
    """Static binding compatibility mirrors runtime nullable coercion."""
    loaded = load_flow(_NULL_TO_NULLABLE_CALL)

    contract = loaded.compiled.nodes["invoke"].output_contract
    assert contract is not None
    assert contract.type.kind is ValueKind.STRING
    assert contract.type.nullable is True


# --------------------------------------------------------------------------- #
# A composed call/map requires a child resolver.
# --------------------------------------------------------------------------- #


def test_call_map_without_resolver_is_loud():
    with pytest.raises(LoadError) as exc:
        load_flow(_text("04-call.yaml"))
    assert "resolver" in str(exc.value).lower()


def test_flow_without_calls_loads_without_resolver():
    # A flow with no call or map loads without a resolver.
    loaded = load_flow(_text("01-structured-agent.yaml"))
    assert isinstance(loaded, ComposedFlow)
