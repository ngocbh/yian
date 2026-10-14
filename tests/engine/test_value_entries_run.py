"""Runs of flows with value entries in `nodes:` (`exit_result: !object {exited: true}`).

A tagged entry lowers to a CODE node that returns its value, so a flow output, another
node, or a CASE route reads it as `${<id>.output}`. These runs cover literal and
template entries, named and nested tags, keys a node body would treat specially, `$$`
escapes, top-level `call(...)` entries, value entries inside `defs:`, inline CALL, LOOP
and MAP bodies, and YAML anchors, aliases and merge keys on value entries.
"""

import pytest

from yian.compose import load_flow
from yian.run import run_flow


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
