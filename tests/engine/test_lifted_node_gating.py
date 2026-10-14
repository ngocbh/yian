"""A node lifted out of an authored node runs on its host's routes and ordering.

Each lifting desugar pass records the node it lifted from as the host: an inline `call(...)`
in a node's binding, the inner calls of a nested branch `call(...)`, and an adaptive gate's
composer AGENT. The lifted node carries the host's CASE route edges (the else route included),
`depends_on:` (co-skip), and `runs_after:` (order). It also waits for the host's other data
inputs and is skipped when one of them would skip the host; a defaulted input (`:-`) only makes
it wait, and sibling calls lifted from one host still run in parallel. So no call, side effect,
or LLM call happens on a branch the host isn't on, and the taken branch runs as before.
"""

import pytest

import yian.llm_clients as llm_clients_mod
from tests.engine import _compose_codefns
from yian.compose import LoadError, load_flow
from yian.events import NodeSkipped, NodeStarted, NodeSucceeded
from yian.nodes.case import ELSE_HANDLE
from yian.run import run_flow

# `side` records its tag when it runs; `wrap` and `items` feed a branch call and a MAP;
# `pair` joins two strings; `meet` is `side` that runs only alongside another `meet`.
_DEFS = """
defs:
  pair:
    input:
      x: !str
      y: !str
    nodes:
      j:
        kind: code
        input:
          x: ${input.x}
          y: ${input.y}
        output: str
        code: return inputs["x"] + inputs["y"]
    output: ${j.output}
  meet:
    input:
      tag: !str
    nodes:
      fx:
        kind: code
        code: tests.engine._compose_codefns:meet
        input:
          tag: ${input.tag}
        output: str
    output: ${fx.output}
  side:
    input:
      tag: !str
    nodes:
      fx:
        kind: code
        code: tests.engine._compose_codefns:record
        input:
          tag: ${input.tag}
        output: str
    output: ${fx.output}
  wrap:
    input:
      x: !str
    nodes:
      w:
        kind: code
        input:
          x: ${input.x}
        output: str
        code: return "wrapped " + inputs["x"]
    output: ${w.output}
  items:
    input:
      tag: !str
    nodes:
      fx:
        kind: code
        code: tests.engine._compose_codefns:record
        input:
          tag: ${input.tag}
        output: str
      ls:
        kind: code
        input:
          t: ${fx.output}
        output: list(str)
        code: return [inputs["t"]]
    output: ${ls.output}
  echo:
    input:
      item: !str
    nodes:
      e:
        kind: code
        input:
          v: ${input.item}
        output: str
        code: return inputs["v"]
    output: ${e.output}
"""

_HEAD = "id: t\nname: t\ninput:\n  go: !bool\n  other: !bool\n"

# `prep` records `prep`; `pick` routes `maybe` only when `other` is true.
_PREP = """
  prep:
    kind: code
    code: tests.engine._compose_codefns:record
    input:
      tag: prep
    output: str
"""
_MAYBE = """
  pick:
    kind: case
    cases:
      - when: ${input.other}
        then: maybe
  maybe:
    kind: code
    output: str
    code: return "m"
"""


def _flow(nodes: str, output: str = "  r: ${gate.output}\n") -> str:
    """A flow with the shared inputs and defs, ``nodes``, and ``output``."""
    return _HEAD + _DEFS + "nodes:\n" + nodes + "output:\n" + output


def _run(text: str, go: bool = False, other: bool = False, **kwargs):
    """Load and run a flow with the recorder cleared; return the flow and the result."""
    _compose_codefns.RECORDED.clear()
    flow = load_flow(text)
    result = run_flow(flow, {"go": go, "other": other}, **kwargs)
    assert result.status == "succeeded", result.failure
    return flow, result


def _edge(flow, edge_id: str):
    """The compiled edge named ``edge_id``."""
    [edge] = [e for e in flow.compiled.edges if e.id == edge_id]
    return edge


def _never_ran(result, node_id: str) -> bool:
    """Whether ``node_id`` was skipped and never started."""
    started = {e.node_id for e in result.events if isinstance(e, NodeStarted)}
    skipped = {e.node_id for e in result.events if isinstance(e, NodeSkipped)}
    return node_id not in started and node_id in skipped


def _starts_after(result, first: str, then: str) -> bool:
    """Whether ``then`` started after ``first`` succeeded."""
    events = list(result.events)
    done = next(
        i for i, e in enumerate(events) if isinstance(e, NodeSucceeded) and e.node_id == first
    )
    start = next(
        i for i, e in enumerate(events) if isinstance(e, NodeStarted) and e.node_id == then
    )
    return done < start


def _index(result, kind, node_id: str) -> int:
    """The position of the first ``kind`` event for ``node_id``."""
    return next(
        i for i, e in enumerate(result.events) if isinstance(e, kind) and e.node_id == node_id
    )


# --------------------------------------------------------------------------- #
# inline_calls: `call(...)` in a node's binding.
# --------------------------------------------------------------------------- #

_PUBLISH = """
  publish:
    kind: code
    input:
      draft: call(side, tag="publish")
    output: str
    code: return "published " + inputs["draft"]
"""
_SKIP_IT = """
  skip_it:
    kind: code
    output: str
    code: return "skipped"
"""
_GATE = """
  gate:
    kind: case
    cases:
      - when: ${input.go}
        then: publish
    else: skip_it
"""


def test_inline_call_does_not_run_on_an_untaken_route():
    flow, result = _run(_flow(_GATE + _PUBLISH + _SKIP_IT), go=False)
    assert _compose_codefns.RECORDED == []
    assert _never_ran(result, "__call_0")
    assert result.output == {"r": "skipped"}
    assert _edge(flow, "gate->__call_0#0").source_handle == "publish"


def test_inline_call_runs_on_the_taken_route():
    _, result = _run(_flow(_GATE + _PUBLISH + _SKIP_IT), go=True)
    assert _compose_codefns.RECORDED == ["publish"]
    assert result.output == {"r": "published publish"}


def test_inline_call_runs_after_the_host_runs_after():
    publish = _PUBLISH.replace("    kind: code\n", "    kind: code\n    runs_after: [prep]\n", 1)
    flow, result = _run(_flow(_GATE + _PREP + publish + _SKIP_IT), go=True)
    edge = _edge(flow, "prep~>__call_0#0")
    assert edge.ordering and edge.optional
    assert _compose_codefns.RECORDED == ["prep", "publish"]
    assert _starts_after(result, "prep", "__call_0")


def test_inline_call_is_skipped_with_a_host_depends_on():
    publish = _PUBLISH.replace("    kind: code\n", "    kind: code\n    depends_on: [maybe]\n", 1)
    flow, result = _run(_flow(_GATE + _MAYBE + publish + _SKIP_IT), go=True, other=False)
    edge = _edge(flow, "maybe~>__call_0#0")
    assert edge.ordering and not edge.optional
    assert _compose_codefns.RECORDED == []
    assert _never_ran(result, "__call_0")


@pytest.mark.parametrize("go, ran", [(False, ["publish"]), (True, [])])
def test_inline_call_follows_the_else_route(go, ran):
    gate = _GATE.replace("then: publish", "then: skip_it").replace("else: skip_it", "else: publish")
    flow, _ = _run(_flow(gate + _PUBLISH + _SKIP_IT), go=go)
    assert _edge(flow, "gate->__call_0#0").source_handle == ELSE_HANDLE
    assert _compose_codefns.RECORDED == ran


@pytest.mark.parametrize("go", [False, True])
def test_inline_call_of_a_then_and_else_host_runs_on_both_routes(go):
    gate = _GATE.replace("else: skip_it", "else: publish")
    flow, _ = _run(_flow(gate + _PUBLISH, output="  r: ${publish.output}\n"), go=go)
    handles = [
        e.source_handle for e in flow.compiled.edges if e.to == "__call_0" and e.source_handle
    ]
    assert handles == ["publish", ELSE_HANDLE]
    assert _compose_codefns.RECORDED == ["publish"]


@pytest.mark.parametrize("go, other, ran", [(False, False, []), (False, True, ["publish"])])
def test_inline_call_of_a_join_target_runs_when_any_route_reaches_it(go, other, ran):
    gates = """
  c1:
    kind: case
    cases:
      - when: ${input.go}
        then: publish
  c2:
    kind: case
    cases:
      - when: ${input.other}
        then: publish
"""
    flow, _ = _run(
        _flow(gates + _PUBLISH, output='  r: ${publish.output :- "none"}\n'), go=go, other=other
    )
    sources = [e.from_ for e in flow.compiled.edges if e.to == "__call_0" and e.source_handle]
    assert sources == ["c1", "c2"]
    assert _compose_codefns.RECORDED == ran


def test_map_over_call_does_not_run_on_an_untaken_route():
    nodes = """
  gate:
    kind: case
    cases:
      - when: ${input.go}
        then: m
    else: skip_it
  m:
    kind: map
    call: echo
    over: call(items, tag="items")
    input:
      item: ${item}
""" + _SKIP_IT
    _run(_flow(nodes), go=False)
    assert _compose_codefns.RECORDED == []
    _run(_flow(nodes), go=True)
    assert _compose_codefns.RECORDED == ["items"]


def test_loop_seed_call_does_not_run_on_an_untaken_route():
    nodes = """
  gate:
    kind: case
    cases:
      - when: ${input.go}
        then: lp
    else: skip_it
  lp:
    kind: loop
    times: 1
    input:
      s: call(side, tag="seed")
    nodes:
      k:
        kind: code
        input:
          s: ${input.s}
        output:
          s: str
        code: |
          return {"s": inputs["s"]}
    output: ${k.output}
""" + _SKIP_IT
    _run(_flow(nodes, output='  r: ${skip_it.output :- "none"}\n'), go=False)
    assert _compose_codefns.RECORDED == []
    _run(_flow(nodes, output='  r: ${skip_it.output :- "none"}\n'), go=True)
    assert _compose_codefns.RECORDED == ["seed"]


# The h1 pattern as a child flow, called per run and per MAP element.
_CHILD_DEF = """
  child:
    input:
      go: !bool
    nodes:
      gate:
        kind: case
        cases:
          - when: ${input.go}
            then: publish
        else: skip_it
      publish:
        kind: code
        input:
          draft: call(side, tag="child")
        output: str
        code: return inputs["draft"]
      skip_it:
        kind: code
        output: str
        code: return "skipped"
    output: ${gate.output}
"""


def test_inline_call_in_a_called_child_flow_follows_the_child_route():
    text = (
        _HEAD + _DEFS + _CHILD_DEF
        + "nodes:\n  c:\n    kind: call\n    call: child\n    input:\n      go: ${input.go}\n"
        + "output:\n  r: ${c.output}\n"
    )
    _, result = _run(text, go=False)
    assert _compose_codefns.RECORDED == []
    assert result.output == {"r": "skipped"}


def test_inline_call_in_a_mapped_child_flow_runs_once_per_taken_element():
    text = (
        _HEAD + _DEFS + _CHILD_DEF
        + "nodes:\n  m:\n    kind: map\n    call: child\n"
        + "    over: ${[input.go, input.other]}\n    input:\n      go: ${item}\n"
        + "output:\n  r: ${m.output}\n"
    )
    _, result = _run(text, go=True, other=False)
    assert _compose_codefns.RECORDED == ["child"]
    assert result.output == {"r": ["child", "skipped"]}


def test_flow_output_call_has_no_host_and_runs():
    nodes = _GATE + _PUBLISH + _SKIP_IT
    _, result = _run(_flow(nodes, output='  r: ${gate.output}\n  o: call(side, tag="out")\n'))
    assert "out" in _compose_codefns.RECORDED
    assert result.output["o"] == "out"


_TYPED_DEFS = """
typedefs:
  Buy: {action: 'Literal(buy)', qty: int}
  Hold: {action: 'Literal(hold)'}
  Decision: Union(Buy, Hold)
"""
_PLACE_DEF = """
  place:
    input:
      order: !Buy
    nodes:
      p:
        kind: code
        code: tests.engine._compose_codefns:record
        input:
          tag: ${input.order.action}
        output: str
    output: ${p.output}
"""


@pytest.mark.parametrize("action, ran", [("buy", ["buy"]), ("hold", [])])
def test_inline_call_in_a_routed_node_reads_the_narrowed_member(action, ran):
    # Under `when: buy` the call gets `b`'s route, so it may pass `d` where a `Buy` is needed.
    nodes = """
  d:
    kind: code
    input:
      a: ${input.action}
    output: Decision
    code: |
      return {"action": inputs["a"], "qty": 2} if inputs["a"] == "buy" else {"action": "hold"}
  route:
    kind: case
    on: ${d.output.action}
    cases:
      - when: buy
        then: b
    else: e
  b:
    kind: code
    input:
      r: call(place, order=${d.output})
    output: str
    code: return inputs["r"]
  e:
    kind: code
    output: str
    code: return "held"
"""
    text = (
        "id: t\nname: t\ninput:\n  action: !str\n" + _TYPED_DEFS + _DEFS + _PLACE_DEF
        + "nodes:\n" + nodes + "output:\n  r: ${route.output}\n"
    )
    _compose_codefns.RECORDED.clear()
    result = run_flow(load_flow(text), {"action": action})
    assert result.status == "succeeded", result.failure
    assert _compose_codefns.RECORDED == ran
    assert result.output == {"r": "buy" if ran else "held"}


# --------------------------------------------------------------------------- #
# case_targets: the inner calls of a nested branch `call(...)`.
# --------------------------------------------------------------------------- #

# Calls mint inner-first: the inner `side` call is `__call_0`, the outer `wrap` is `__call_1`.
_NESTED = """
  gate:
    kind: case
    cases:
      - when: ${input.go}
        then: call(wrap, x=call(side, tag="inner"))
    else: b
  b:
    kind: code
    output: str
    code: return "B"
"""


def test_nested_branch_call_does_not_run_its_inner_call_on_an_untaken_route():
    flow, result = _run(_flow(_NESTED), go=False)
    assert _compose_codefns.RECORDED == []
    assert _never_ran(result, "__call_0")
    assert result.output == {"r": "B"}
    assert _edge(flow, "gate->__call_0#0").source_handle == "__call_1"
    assert not [e for e in flow.compiled.edges if e.from_ == "__start__" and e.to == "__call_0"]


def test_nested_branch_call_runs_on_the_taken_route():
    _, result = _run(_flow(_NESTED), go=True)
    assert _compose_codefns.RECORDED == ["inner"]
    assert result.output == {"r": "wrapped inner"}


def test_nested_branch_call_is_skipped_with_the_case_depends_on():
    nested = _NESTED.replace("    kind: case\n", "    kind: case\n    depends_on: [maybe]\n", 1)
    _, result = _run(_flow(_MAYBE + nested), go=True, other=False)
    assert _compose_codefns.RECORDED == []
    assert _never_ran(result, "__call_0")


def test_nested_branch_call_runs_after_the_case_runs_after():
    nested = _NESTED.replace("    kind: case\n", "    kind: case\n    runs_after: [prep]\n", 1)
    _, result = _run(_flow(_PREP + nested), go=True)
    assert _compose_codefns.RECORDED == ["prep", "inner"]
    assert _starts_after(result, "prep", "__call_0")


# --------------------------------------------------------------------------- #
# adaptive_questions: the composer AGENT `<gate>__compose`.
# --------------------------------------------------------------------------- #

_COMPOSED = [
    {
        "question": "Which database?",
        "header": "DB",
        "options": [{"label": "Postgres", "description": "Relational."}],
        "multi_select": False,
    }
]


class _ComposeChat:
    """A fake structured-output chat that counts composer calls, or fails them."""

    def __init__(self, calls: list, fail: bool = False):
        self.calls = calls
        self.fail = fail

    def bind_tools(self, tools):
        return self

    def invoke(self, messages):
        raise AssertionError("structured path expected, not a plain-text invoke")

    def with_structured_output(self, schema):
        chat = self

        class _Bound:
            def invoke(self, messages):
                chat.calls.append("compose")
                if chat.fail:
                    raise RuntimeError("the composer must not run here")
                return schema.model_validate({"items": _COMPOSED})

        return _Bound()


_ADAPTIVE = """
  router:
    kind: case
    cases:
      - when: ${input.go}
        then: ask
    else: done
  ask:
    kind: human_input
    input:
      ctx: ${input.go}
    adaptive_questions:
      prompt: "Design one question for: ${ctx}"
  done:
    kind: code
    output: str
    code: return "done"
"""


def _adaptive(monkeypatch, nodes: str, go: bool, other: bool = False, fail: bool = False):
    """Run an adaptive flow; return the flow, result, composer calls, and human asks."""
    calls: list = []
    asked: list = []
    monkeypatch.setattr(
        llm_clients_mod, "model_from_config", lambda cfg: _ComposeChat(calls, fail)
    )
    flow, result = _run(
        _flow(nodes, output="  r: ${router.output}\n"),
        go=go,
        other=other,
        tools={"ask_user": lambda **request: asked.append(request) or {"DB": "Postgres"}},
    )
    return flow, result, calls, asked


def test_adaptive_composer_does_not_run_on_an_untaken_route(monkeypatch):
    flow, result, calls, asked = _adaptive(monkeypatch, _ADAPTIVE, go=False)
    assert (calls, asked) == ([], [])
    assert result.output == {"r": "done"}
    assert _edge(flow, "router->ask__compose#0").source_handle == "ask"


def test_adaptive_composer_failure_on_an_untaken_route_does_not_fail_the_run(monkeypatch):
    _, result, calls, _ = _adaptive(monkeypatch, _ADAPTIVE, go=False, fail=True)
    assert calls == []
    assert result.output == {"r": "done"}


def test_adaptive_composer_runs_on_the_taken_route(monkeypatch):
    _, result, calls, asked = _adaptive(monkeypatch, _ADAPTIVE, go=True)
    assert calls == ["compose"]
    assert [q["header"] for q in asked[0]["questions"]] == ["DB"]
    assert result.output == {"r": {"DB": "Postgres"}}


def test_adaptive_composer_runs_after_the_gate_runs_after(monkeypatch):
    nodes = _PREP + _ADAPTIVE.replace(
        "    kind: human_input\n", "    kind: human_input\n    runs_after: [prep]\n"
    )
    flow, result, calls, _ = _adaptive(monkeypatch, nodes, go=True)
    edge = _edge(flow, "prep~>ask__compose#0")
    assert edge.ordering and edge.optional
    assert calls == ["compose"]
    assert _starts_after(result, "prep", "ask__compose")


def test_adaptive_composer_is_skipped_with_the_gate_depends_on(monkeypatch):
    nodes = _MAYBE + _ADAPTIVE.replace(
        "    kind: human_input\n", "    kind: human_input\n    depends_on: [maybe]\n"
    )
    _, result, calls, asked = _adaptive(monkeypatch, nodes, go=True, other=False)
    assert (calls, asked) == ([], [])
    assert _never_ran(result, "ask__compose")


def test_adaptive_gate_with_an_inline_call_gates_both_lifted_nodes(monkeypatch):
    nodes = _PREP + _ADAPTIVE.replace(
        "      ctx: ${input.go}\n", '      ctx: call(side, tag="ctx")\n'
    ).replace("    kind: human_input\n", "    kind: human_input\n    runs_after: [prep]\n")
    flow, result, calls, asked = _adaptive(monkeypatch, nodes, go=False)
    assert (calls, asked, _compose_codefns.RECORDED) == ([], [], ["prep"])
    assert _never_ran(result, "__call_0") and _never_ran(result, "ask__compose")
    assert _edge(flow, "router->__call_0#0").source_handle == "ask"
    assert _edge(flow, "prep~>__call_0#0").optional


def test_reading_a_lifted_node_in_the_hosts_runs_after_is_a_cycle():
    # `audit` reads the composer, and the gate orders after `audit`: the composer now orders
    # after `audit` too, so the two wait on each other.
    nodes = _ADAPTIVE.replace(
        "    kind: human_input\n", "    kind: human_input\n    runs_after: [audit]\n"
    ) + """
  audit:
    kind: code
    input:
      q: ${ask__compose.output}
    output: int
    code: return len(inputs["q"])
"""
    with pytest.raises(LoadError, match="cycle"):
        load_flow(_flow(nodes, output="  r: ${router.output}\n"))


# --------------------------------------------------------------------------- #
# data gates: the host's other data inputs.
# --------------------------------------------------------------------------- #

_PUBLISH_NOTES = """
  publish:
    kind: code
    input:
      draft: call(side, tag="draft")
      notes: ${maybe.output}
    output: str
    code: return inputs["draft"] + inputs["notes"]
"""
_PUBLISH_OUT = "  r: ${publish.output}\n"


def test_inline_call_is_skipped_with_a_host_data_input():
    flow, result = _run(_flow(_MAYBE + _PUBLISH_NOTES, output=_PUBLISH_OUT), other=False)
    assert _never_ran(result, "__call_0")
    assert _compose_codefns.RECORDED == []
    assert result.output == {"r": None}
    edge = _edge(flow, "maybe->__call_0#0")
    assert edge.input_group == "publish.notes"
    assert edge.optional is False


def test_inline_call_waits_for_a_host_data_input_on_the_taken_path():
    _, result = _run(_flow(_MAYBE + _PUBLISH_NOTES, output=_PUBLISH_OUT), other=True)
    assert _compose_codefns.RECORDED == ["draft"]
    assert result.output == {"r": "draftm"}
    assert _starts_after(result, "maybe", "__call_0")


def test_inline_call_only_waits_for_a_defaulted_host_input():
    publish = _PUBLISH_NOTES.replace("${maybe.output}", '${maybe.output :- "none"}')
    flow, result = _run(_flow(_MAYBE + publish, output=_PUBLISH_OUT), other=False)
    assert _compose_codefns.RECORDED == ["draft"]
    assert result.output == {"r": "draftnone"}
    assert _edge(flow, "maybe->__call_0#0").optional is True
    assert _index(result, NodeSkipped, "maybe") < _index(result, NodeStarted, "__call_0")



# Each string nested in a host's value is its own input group, so a lifted node gets one
# gate group per nested read, named after the host and the nested position.
_PUBLISH_NESTED_VALUE = _PUBLISH_NOTES.replace(
    "      notes: ${maybe.output}\n", '      notes: {text: "${maybe.output}", n: 1}\n'
).replace('inputs["notes"]', 'inputs["notes"]["text"]')


def test_inline_call_is_skipped_with_a_host_nested_data_input():
    text = _flow(_MAYBE + _PUBLISH_NESTED_VALUE, output=_PUBLISH_OUT)
    flow, result = _run(text, other=False)
    assert _never_ran(result, "__call_0")
    assert _compose_codefns.RECORDED == []
    assert result.output == {"r": None}
    edge = _edge(flow, "maybe->__call_0#0")
    assert edge.input_group == "publish.notes/text"
    assert edge.optional is False


def test_inline_call_only_waits_for_a_defaulted_host_nested_input():
    publish = _PUBLISH_NESTED_VALUE.replace("${maybe.output}", "${maybe.output :- 'none'}")
    flow, result = _run(_flow(_MAYBE + publish, output=_PUBLISH_OUT), other=False)
    assert _compose_codefns.RECORDED == ["draft"]
    assert result.output == {"r": "draftnone"}
    assert _edge(flow, "maybe->__call_0#0").optional is True


_PUBLISH_TWO = """
  publish:
    kind: code
    input:
      a: call(meet, tag="a")
      b: call(meet, tag="b")
      notes: ${maybe.output}
    output: str
    code: return inputs["a"] + inputs["b"] + inputs["notes"]
"""


def test_sibling_calls_from_one_host_run_in_parallel():
    _compose_codefns.MEET.reset()
    flow, _ = _run(_flow(_MAYBE + _PUBLISH_TWO, output=_PUBLISH_OUT), other=True)
    assert sorted(_compose_codefns.RECORDED) == ["a", "b"]
    assert not [
        e for e in flow.compiled.edges if {e.from_, e.to} == {"__call_0", "__call_1"}
    ]


def test_sibling_calls_from_one_host_are_skipped_with_a_host_data_input():
    _compose_codefns.MEET.reset()
    _run(_flow(_MAYBE + _PUBLISH_TWO, output=_PUBLISH_OUT), other=False)
    assert _compose_codefns.RECORDED == []


# Calls mint inner-first: the inner `side` call is `__call_0`, the outer `pair` is `__call_1`.
_PUBLISH_NESTED = """
  publish:
    kind: code
    input:
      draft: call(pair, x=call(side, tag="inner"), y=${maybe.output})
    output: str
    code: return inputs["draft"]
"""


@pytest.mark.parametrize("other, ran, out", [(False, [], None), (True, ["inner"], "innerm")])
def test_nested_inline_call_looks_through_to_its_siblings_inputs(other, ran, out):
    _, result = _run(_flow(_MAYBE + _PUBLISH_NESTED, output=_PUBLISH_OUT), other=other)
    assert _compose_codefns.RECORDED == ran
    assert result.output == {"r": out}
    if not other:
        assert _never_ran(result, "__call_0") and _never_ran(result, "__call_1")


_NESTED_PAIR = """
  gate:
    kind: case
    cases:
      - when: ${input.go}
        then: call(pair, x=call(side, tag="inner"), y=${maybe.output})
    else: b
  b:
    kind: code
    output: str
    code: return "B"
"""


@pytest.mark.parametrize("other, ran, out", [(False, [], None), (True, ["inner"], "innerm")])
def test_nested_branch_inner_call_is_skipped_with_the_branch_calls_data_input(other, ran, out):
    _, result = _run(_flow(_MAYBE + _NESTED_PAIR), go=True, other=other)
    assert _compose_codefns.RECORDED == ran
    assert result.output == {"r": out}
    if not other:
        assert _never_ran(result, "__call_0")


_ADAPTIVE_NOTE = _MAYBE + _ADAPTIVE.replace(
    "      ctx: ${input.go}\n", '      ctx: call(side, tag="ctx")\n      note: ${maybe.output}\n'
)


def test_adaptive_lifted_nodes_are_skipped_with_a_gate_data_input(monkeypatch):
    _, result, calls, asked = _adaptive(monkeypatch, _ADAPTIVE_NOTE, go=True, other=False)
    assert (calls, asked, _compose_codefns.RECORDED) == ([], [], [])
    assert _never_ran(result, "__call_0") and _never_ran(result, "ask__compose")


def test_adaptive_lifted_nodes_wait_for_a_gate_data_input(monkeypatch):
    _, _, calls, _ = _adaptive(monkeypatch, _ADAPTIVE_NOTE, go=True, other=True)
    assert _compose_codefns.RECORDED == ["ctx"]
    assert calls == ["compose"]


def test_reading_a_lifted_node_from_a_host_input_is_a_cycle():
    # `spy` reads the call, and `publish` reads `spy`: the call now waits on `spy`.
    nodes = """
  spy:
    kind: code
    input:
      v: ${__call_0.output}
    output: str
    code: return inputs["v"]
  publish:
    kind: code
    input:
      draft: call(side, tag="d")
      notes: ${spy.output}
    output: str
    code: return inputs["draft"] + inputs["notes"]
"""
    with pytest.raises(LoadError, match="cycle") as caught:
        load_flow(_flow(nodes, output=_PUBLISH_OUT))
    assert "__call_0" in str(caught.value) and "spy" in str(caught.value)
    assert "__call_0 depends on spy (gate: publish.input.notes)" in caught.value.diagnostic.notes
