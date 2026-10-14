"""Runs of flows with value entries in `nodes:` (`exit_result: !object {exited: true}`).

A tagged entry lowers to a CODE node that returns its value, so a flow output, another
node, or a CASE route reads it as `${<id>.output}`. These runs cover literal and
template entries (a span in a nested list too), named and nested tags, keys a node body
would treat specially, `$$` escapes, top-level `call(...)` entries, value entries inside
`defs:`, inline CALL, LOOP and MAP bodies, and YAML anchors, aliases and merge keys on
value entries.

An entry is skipped like any node: when a CASE routes away from it, or when a whole-span
read names a skipped node, while an embedded span renders a skipped read as empty text
and an unrouted entry with no reads always runs. A routed `call(...)` entry skips its
lifted call with it, and a resumed log replays an entry's recorded output. A routed entry
reads a union member field its branch selects; a field its branch excludes is a load
error.
"""

import json
from pathlib import Path

import pytest

from yian.compose import LoadError, load_flow
from yian.events import NodeSkipped, NodeStarted
from yian.nodes.base import _HostStop
from yian.run import resume_flow, run_flow


# The graph's own boundary nodes, which start in every run.
_BOUNDARY = {"__start__", "__end__"}


def _flow_node(node_id: str) -> bool:
    """Whether `node_id` is a node of the flow itself: not a boundary, not inside a call."""
    return node_id not in _BOUNDARY and "/" not in node_id


def _started(result) -> set[str]:
    """Return the ids of the flow's nodes that started in a run."""
    return {
        e.node_id
        for e in result.events
        if isinstance(e, NodeStarted) and _flow_node(e.node_id)
    }


def _skipped(result) -> set[str]:
    """Return the ids of the flow's nodes skipped in a run."""
    return {
        e.node_id
        for e in result.events
        if isinstance(e, NodeSkipped) and _flow_node(e.node_id)
    }


def _succeeded_values(path: Path) -> dict[str, object]:
    """Return each recorded node success in a run log, keyed by node id."""
    values: dict[str, object] = {}
    for line in path.read_text().splitlines():
        record = json.loads(line)
        if record["kind"] == "node_succeeded":
            payload = record["payload"]
            values[payload["node_id"]] = payload["value"]
    return values


def _run(text: str, inputs: dict | None = None):
    """Load and run `text`, expecting success, and return its output."""
    result = run_flow(load_flow(text), inputs or {})
    assert result.status == "succeeded", result.failure
    return result.output


def _flow(entries: str, *, output: str = "${x.output}", head: str = "") -> str:
    """Return a flow with a `name: str` input, int producers `a` and `b`, and `entries`."""
    return (
        "id: f\n"
        "name: f\n"
        f"{head}"
        "input:\n"
        "  name: !str\n"
        "nodes:\n"
        "  a:\n"
        "    kind: code\n"
        "    output: int\n"
        "    code: |\n"
        "      return 2\n"
        "  b:\n"
        "    kind: code\n"
        "    output: int\n"
        "    code: |\n"
        "      return 3\n"
        f"{entries}"
        f"output: {output}\n"
    )


_BUY = "typedefs:\n  Buy:\n    action: Literal(buy)\n    conviction: float\n"


@pytest.mark.parametrize(
    ("entry", "expected"),
    [
        ("!int 3", 3),
        ("!float 1.5", 1.5),
        ("!bool true", True),
        ("!Optional(str) null", None),
        ("!list(str) [a, b]", ["a", "b"]),
        ("!int ${a.output + b.output}", 5),
        ("!str Hello, ${input.name}", "Hello, Ada"),
        ("!object {exited: true, reply: ''}", {"exited": True, "reply": ""}),
        ("!str costs $$5", "costs $5"),
        ("!object {a: '$${x}'}", {"a": "${x}"}),
        ("!str 'call(f)'", "call(f)"),
    ],
)
def test_a_value_entry_runs_to_its_value(entry, expected):
    output = _run(_flow(f"  x: {entry}\n"), {"name": "Ada"})

    assert output == expected
    assert type(output) is type(expected)


def test_a_nested_list_with_a_span_runs():
    text = _flow('  x: !object\n    items:\n      - "${input.name}"\n      - x\n')

    assert _run(text, {"name": "Ada"}) == {"items": ["Ada", "x"]}


def test_a_named_type_entry_runs_with_a_span():
    text = (
        "id: f\n"
        "name: f\n"
        f"{_BUY}"
        "nodes:\n"
        "  s:\n"
        "    kind: code\n"
        "    output: float\n"
        "    code: |\n"
        "      return 0.5\n"
        "  d: !Buy\n"
        "    action: buy\n"
        "    conviction: ${s.output}\n"
        "output: ${d.output}\n"
    )

    assert _run(text) == {"action": "buy", "conviction": 0.5}


def test_a_nested_tag_runs():
    text = _flow(
        "  x: !object\n    d: !Buy {action: buy, conviction: 1}\n    n: ${a.output}\n",
        head=_BUY,
    )

    assert _run(text, {"name": "Ada"}) == {
        "d": {"action": "buy", "conviction": 1},
        "n": 2,
    }


def test_node_body_keys_in_a_value_are_kept():
    value = {
        "input": 1,
        "output": 2,
        "inputs": 3,
        "kind": "call",
        "call": "f",
        "nodes": {},
    }
    text = _flow(
        "  x: !object {input: 1, output: 2, inputs: 3, kind: call, call: f, nodes: {}}\n"
    )

    assert _run(text, {"name": "Ada"}) == value


_F_DEF = (
    "defs:\n"
    "  f:\n"
    "    input:\n"
    "      x: !str\n"
    "    nodes:\n"
    "      up:\n"
    "        kind: code\n"
    "        input:\n"
    "          x: ${input.x}\n"
    "        output: str\n"
    "        code: |\n"
    "          return inputs['x'].upper()\n"
    "    output: ${up.output}\n"
)


def test_a_top_level_flow_call_entry_runs_the_flow():
    text = _flow("  x: !str call(f, x=${input.name})\n", head=_F_DEF)

    assert _run(text, {"name": "Ada"}) == "ADA"


def test_a_value_entry_in_a_defs_body():
    head = (
        "defs:\n"
        "  g:\n"
        "    input:\n"
        "      n: !int\n"
        "    nodes:\n"
        "      v: !object\n"
        "        n: ${input.n}\n"
        "        where: def\n"
        "    output: ${v.output}\n"
    )
    text = _flow("  x: !object call(g, n=${a.output})\n", head=head)

    assert _run(text, {"name": "Ada"}) == {"n": 2, "where": "def"}


def test_a_value_entry_in_an_inline_call_body():
    text = _flow(
        "  x:\n"
        "    kind: call\n"
        "    input:\n"
        "      who: ${input.name}\n"
        "    nodes:\n"
        "      hi: !str hi ${input.who}\n"
        "    output: ${hi.output}\n"
    )

    assert _run(text, {"name": "Ada"}) == "hi Ada"


def test_a_value_entry_in_an_inline_loop_body():
    text = _flow(
        "  x:\n"
        "    kind: loop\n"
        "    times: 3\n"
        "    input:\n"
        "      n: 0\n"
        "    nodes:\n"
        "      bump: !Count\n"
        "        n: ${input.n + 1}\n"
        "    output: ${bump.output}\n",
        head="typedefs:\n  Count:\n    n: int\n",
    )

    assert _run(text, {"name": "Ada"}) == {"n": 3}


def test_a_value_entry_in_a_map_body():
    text = _flow(
        "  items: !list(str) [a, b]\n"
        "  x:\n"
        "    kind: map\n"
        "    over: ${items.output}\n"
        "    input:\n"
        "      v: ${item}\n"
        "    nodes:\n"
        "      wrap: !str got ${input.v}\n"
        "    output: ${wrap.output}\n"
    )

    assert _run(text, {"name": "Ada"}) == ["got a", "got b"]


def test_an_anchored_value_entry_and_its_aliases():
    text = _flow(
        "  base: &b !object {a: 1}\n"
        "  copy: *b\n"
        "  x: !object\n"
        "    inner: *b\n",
        output="\n  base: ${base.output}\n  copy: ${copy.output}\n  x: ${x.output}",
    )

    assert _run(text, {"name": "Ada"}) == {
        "base": {"a": 1},
        "copy": {"a": 1},
        "x": {"inner": {"a": 1}},
    }


def test_a_merge_key_inside_a_value_entry():
    text = _flow(
        "  x: !object\n    <<: *shared\n    c: 3\n",
        head="x-shared: &shared {a: 1, b: 2}\n",
    )

    assert _run(text, {"name": "Ada"}) == {"a": 1, "b": 2, "c": 3}


def test_a_runtime_type_failure_points_at_the_entry():
    text = (
        "id: f\n"
        "name: f\n"
        "nodes:\n"
        "  p:\n"
        "    kind: code\n"
        "    code: |\n"
        "      return 'five'\n"
        "  x: !int ${p.output}\n"
        "output: ${x.output}\n"
    )
    flow = load_flow(text)

    result = run_flow(flow, {})

    assert result.status == "failed"
    assert result.failure.code == "binding.type"
    assert result.failure.node_id == "x"
    frame = flow.diagnose(result.failure).frames[-1]
    assert frame.range.covered_lines(text) == [8]


def _producer(output: str) -> str:
    """Return a CODE node `p` returning text, declaring `output` unless it is empty."""
    declared = f"    output: {output}\n" if output else ""
    return f"  p:\n    kind: code\n{declared}    code: |\n      return 'hi'\n"


@pytest.mark.parametrize(
    ("entries", "output", "expected"),
    [
        ("  r: !object {exited: true, reply: ''}\n", "${r.output.reply}", ""),
        (
            "  r: !object {exited: true, reply: ''}\n",
            "${r.output}",
            {"exited": True, "reply": ""},
        ),
        (
            _producer("str") + "  r: !object\n    a: ${p.output}\n    b: 1\n",
            "${r.output}",
            {"a": "hi", "b": 1},
        ),
        (
            _producer("") + "  r: !object\n    a: ${p.output}\n    b: 1\n",
            "${r.output.a}",
            "hi",
        ),
        (
            _producer("str") + "  r: !object\n    a: ${p.output}\n    p: {q: 1}\n",
            "${r.output.p.q}",
            1,
        ),
        (
            _producer("") + "  r: !object\n    a: ${p.output}\n    p: !object {q: 1}\n",
            "${r.output.p}",
            {"q": 1},
        ),
        ("  r: !object\n    p:\n      q: [1, 2]\n", "${r.output.p.q}", [1, 2]),
        ("  r: !object {}\n", "${r.output}", {}),
        ('  r: !object {a: "$${input.name}"}\n', "${r.output}", {"a": "${input.name}"}),
        (
            "  x:\n"
            "    kind: code\n"
            "    output: {q: int}\n"
            "    code: |\n"
            "      return {'q': 1}\n"
            "  r: !object ${x.output}\n",
            "${r.output}",
            {"q": 1},
        ),
    ],
)
def test_a_written_out_object_entry_runs_with_its_record(entries, output, expected):
    assert _run(_flow(entries, output=output), {"name": "Ada"}) == expected


_ROUTE_FLOW = """
id: f
name: f
input:
  cmd: !str
nodes:
  route:
    kind: case
    on: ${input.cmd}
    cases:
      - when: /exit
        then: exit_result
    else: reply
  exit_result: !object {exited: true, reply: ""}
  reply:
    kind: code
    input:
      text: ${input.cmd}
    code: |
      return {'exited': False, 'reply': inputs['text']}
output: ${route.output}
"""


@pytest.mark.parametrize(
    ("cmd", "taken", "untaken", "expected"),
    [
        ("/exit", "exit_result", "reply", {"exited": True, "reply": ""}),
        ("hi", "reply", "exit_result", {"exited": False, "reply": "hi"}),
    ],
)
def test_a_case_routes_to_a_value_entry_or_a_node(cmd, taken, untaken, expected):
    result = run_flow(load_flow(_ROUTE_FLOW), {"cmd": cmd})

    assert result.status == "succeeded", result.failure
    assert result.output == expected
    assert _started(result) == {"route", taken}
    assert _skipped(result) == {untaken}


_SKIP_FLOW = """
id: f
name: f
input:
  cmd: !str
nodes:
  route:
    kind: case
    on: ${input.cmd}
    cases:
      - when: go
        then: maybe
  maybe:
    kind: code
    output: str
    code: |
      return 'm'
  whole: !object
    a: ${maybe.output}
  embedded: !str "got ${maybe.output}"
  lone: !int 7
output:
  whole: ${whole.output}
  embedded: ${embedded.output}
  lone: ${lone.output}
"""


def test_entries_reading_a_routed_away_node():
    result = run_flow(load_flow(_SKIP_FLOW), {"cmd": "stop"})

    assert result.status == "succeeded", result.failure
    assert result.output == {"whole": None, "embedded": "got ", "lone": 7}
    assert _started(result) == {"route", "embedded", "lone"}
    assert _skipped(result) == {"maybe", "whole"}


def test_entries_reading_a_routed_node():
    result = run_flow(load_flow(_SKIP_FLOW), {"cmd": "go"})

    assert result.status == "succeeded", result.failure
    assert result.output == {"whole": {"a": "m"}, "embedded": "got m", "lone": 7}
    assert _started(result) == {"route", "maybe", "whole", "embedded", "lone"}
    assert _skipped(result) == set()


def _union_flow(*, buy: str, other: str) -> str:
    """Return a flow whose CASE on a `Union(Buy, Sell)` tag routes `buy` to `buy`."""
    return (
        "id: f\n"
        "name: f\n"
        "typedefs:\n"
        "  Buy:\n"
        "    action: Literal(buy)\n"
        "    qty: int\n"
        "  Sell:\n"
        "    action: Literal(sell)\n"
        "    price: float\n"
        "input:\n"
        "  kind: !str\n"
        "nodes:\n"
        "  d:\n"
        "    kind: code\n"
        "    input:\n"
        "      kind: ${input.kind}\n"
        "    code: |\n"
        '      if inputs["kind"] == "buy":\n'
        '          return {"action": "buy", "qty": 3}\n'
        '      return {"action": "sell", "price": 1.5}\n'
        "    output: Union(Buy, Sell)\n"
        "  route:\n"
        "    kind: case\n"
        "    on: ${d.output.action}\n"
        "    cases:\n"
        "      - when: buy\n"
        "        then: buy\n"
        "    else: other\n"
        f"  buy: !object\n    qty: {buy}\n"
        f"  other: !object\n    qty: {other}\n"
        "output: ${route.output}\n"
    )


def test_a_routed_entry_reads_a_union_member_field_its_branch_selects():
    flow = load_flow(_union_flow(buy='"${d.output.qty}"', other="0"))

    assert run_flow(flow, {"kind": "buy"}).output == {"qty": 3}
    assert run_flow(flow, {"kind": "sell"}).output == {"qty": 0}


def test_a_routed_entry_reading_a_field_its_branch_excludes_is_a_load_error():
    text = _union_flow(buy="0", other='"${d.output.qty}"')

    with pytest.raises(LoadError) as caught:
        load_flow(text)

    assert str(caught.value) == (
        "flow has unresolved references:\n  value entry 'other' from: reference "
        "${d.output.qty} reads unknown field 'qty' on record"
    )
    frame = caught.value.diagnostic.frames[-1]
    line = text.splitlines().index('    qty: "${d.output.qty}"') + 1
    assert frame.range.start_line == line


_CALL_FLOW = _F_DEF + """
id: f
name: f
input:
  cmd: !str
  t: !str
nodes:
  route:
    kind: case
    on: ${input.cmd}
    cases:
      - when: go
        then: e
    else: other
  other:
    kind: code
    code: |
      return 'other'
  e: !str call(f, x=${input.t})
output: ${route.output}
"""


def test_a_routed_flow_call_entry_runs_its_call_only_when_taken():
    flow = load_flow(_CALL_FLOW)
    (call_id,) = [node_id for node_id in flow.compiled.nodes if node_id.startswith("__call_")]

    taken = run_flow(flow, {"cmd": "go", "t": "ada"})
    untaken = run_flow(flow, {"cmd": "no", "t": "ada"})

    assert taken.status == "succeeded", taken.failure
    assert taken.output == "ADA"
    assert _started(taken) == {"route", "e", call_id}
    assert _skipped(taken) == {"other"}
    assert untaken.status == "succeeded", untaken.failure
    assert untaken.output == "other"
    assert _started(untaken) == {"route", "other"}
    assert _skipped(untaken) == {"e", call_id}


_GATE_FLOW = """
id: f
name: f
input:
  t: !str
nodes:
  ask:
    kind: human_input
    prompt: "Command?"
  route:
    kind: case
    on: ${ask.output}
    cases:
      - when: exit
        then: exit_result
    else: other
  exit_result: !object
    reply: ${input.t}
    exited: true
  other:
    kind: code
    code: |
      return {'reply': 'still here', 'exited': False}
output: ${route.output}
"""


def test_a_resumed_run_routes_to_a_value_entry_and_replays_it(tmp_path: Path):
    path = tmp_path / "gate.jsonl"
    interruption = _HostStop("interrupt before human answer")

    def interrupt(**request):
        raise interruption

    with pytest.raises(_HostStop) as caught:
        run_flow(load_flow(_GATE_FLOW), {"t": "bye"}, log=path, tools={"ask_user": interrupt})
    assert caught.value is interruption
    assert "exit_result" not in _succeeded_values(path)

    answered = resume_flow(
        load_flow(_GATE_FLOW), log=path, tools={"ask_user": lambda **request: "exit"}
    )

    assert answered.status == "succeeded", answered.failure
    assert answered.output == {"reply": "bye", "exited": True}
    assert _started(answered) == {"ask", "route", "exit_result"}
    assert _skipped(answered) == {"other"}
    assert _succeeded_values(path)["exit_result"]["value"] == answered.output
    before = path.read_bytes()

    replayed = resume_flow(load_flow(_GATE_FLOW), log=path)

    assert replayed.status == "succeeded", replayed.failure
    assert replayed.output == answered.output
    assert not [e for e in replayed.events if isinstance(e, NodeStarted)]
    assert path.read_bytes() == before
