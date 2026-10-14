"""A flow output that reads a node CASE routing may skip returns null when it is skipped."""

from __future__ import annotations

from pathlib import Path
import textwrap

import pytest

import tests.engine._compose_codefns as codefns
from yian.compose import LoadError, load_flow
from yian.run import resume_flow, run_flow
from yian.typesys.values import ValueKind


def _flow(
    nodes: str,
    output: str,
    *,
    inputs: str = "approve: !bool",
    defs: str = "",
    typedefs: str = "",
) -> str:
    """Return a flow text with the given `typedefs:`, `input:`, `defs:`, `nodes:` and `output:` blocks."""
    return (
        f"id: f\nname: f\n{typedefs}input:\n  {inputs}\n{defs}nodes:\n{nodes}output:{output}\n"
    )


def _code(node: str, value: str, output: str = "str", extra: str = "") -> str:
    """Return an inline CODE node returning the Python expression `value`."""
    return (
        f"  {node}:\n    kind: code\n    output: {output}\n{extra}"
        f"    code: |\n      return {value}\n"
    )


def _gate(then: str, else_: str | None = None, name: str = "gate", when: str = "approve") -> str:
    """Return a CASE that routes to `then` when `input.<when>` holds, else to `else_`."""
    text = (
        f"  {name}:\n    kind: case\n    cases:\n"
        f"      - when: ${{input.{when}}}\n        then: {then}\n"
    )
    return text + (f"    else: {else_}\n" if else_ is not None else "")


# then: order / else: rejected
_R1 = _gate("order", "rejected") + _code("order", '"filled"') + _code("rejected", '"no"')
# two `when:` cases and no `else:`: neither matches when `approve` and `other` are false
_R2 = (
    "  gate:\n    kind: case\n    cases:\n"
    "      - when: ${input.approve}\n        then: a\n"
    "      - when: ${input.other}\n        then: b\n"
    + _code("a", '"A"')
    + _code("b", '"B"')
)
_R2_INPUTS = "approve: !bool\n  other: !bool"


def _run(text: str, inputs: dict, **kw):
    """Load and run `text`, returning the output; the run must succeed."""
    result = run_flow(load_flow(text, **kw), inputs)
    assert result.status == "succeeded", result.failure
    return result.output


def _sources(text: str, **kw) -> list:
    """Return the loaded flow's output sources in declaration order."""
    return [output.from_ for output in load_flow(text, **kw).compiled.outputs]


def test_record_output_of_a_skipped_branch_is_null():
    """The skipped branch target gives null; the taken one gives its value."""
    text = _flow(_R1, "\n  order: ${order.output}")
    assert _sources(text) == ["${order.output:-null}"]
    assert _run(text, {"approve": False}) == {"order": None}
    assert _run(text, {"approve": True}) == {"order": "filled"}


def test_coalesce_over_a_routing_only_case_is_null():
    """A `|` over targets of a CASE that may match nothing gives null."""
    text = _flow(_R2, "\n  v: ${a.output | b.output}", inputs=_R2_INPUTS)
    assert _sources(text) == ["${(a.output | b.output) :- null}"]
    assert _run(text, {"approve": False, "other": False}) == {"v": None}
    assert _run(text, {"approve": True, "other": False}) == {"v": "A"}
    assert _run(text, {"approve": False, "other": True}) == {"v": "B"}


def test_nested_case_target_is_null_when_either_case_skips_it():
    """A target of an inner CASE gives null when the inner or the outer CASE skips it."""
    nodes = (
        _gate("inner", "rejected", name="outer")
        + _gate("order", "hold", name="inner", when="inner")
        + _code("order", '"filled"')
        + _code("hold", '"held"')
        + _code("rejected", '"no"')
    )
    text = _flow(nodes, "\n  order: ${order.output}", inputs="approve: !bool\n  inner: !bool")
    assert _run(text, {"approve": True, "inner": False}) == {"order": None}
    assert _run(text, {"approve": False, "inner": True}) == {"order": None}
    assert _run(text, {"approve": True, "inner": True}) == {"order": "filled"}


def test_value_mode_output_is_null():
    """A value-mode `output: ${order.output}` returns null."""
    text = _flow(_R1, " ${order.output}")
    assert _sources(text) == ["${order.output:-null}"]
    assert _run(text, {"approve": False}) is None


def test_dotted_output_is_null():
    """A field read `${order.output.status}` of a skipped node returns null."""
    nodes = (
        _gate("order", "rejected")
        + _code("order", '{"status": "filled"}', output="\n      status: str")
        + _code("rejected", '"no"')
    )
    text = _flow(nodes, "\n  s: ${order.output.status}")
    assert _sources(text) == ["${order.output.status:-null}"]
    assert _run(text, {"approve": False}) == {"s": None}
    assert _run(text, {"approve": True}) == {"s": "filled"}


def test_ref_default_group_over_a_routing_only_case_is_null():
    """`${a.output :- b.output}` over a CASE that may match nothing returns null."""
    text = _flow(_R2, "\n  v: ${a.output :- b.output}", inputs=_R2_INPUTS)
    assert _sources(text) == ["${(a.output :- b.output) :- null}"]
    assert _run(text, {"approve": False, "other": False}) == {"v": None}
    assert _run(text, {"approve": True, "other": False}) == {"v": "A"}


def test_depends_on_a_skipped_target_is_null():
    """A node with `depends_on` a skipped branch target is skipped too, and gives null."""
    nodes = _R1 + _code("after", '"after"', extra="    depends_on: [order]\n")
    text = _flow(nodes, "\n  a: ${after.output}")
    assert _run(text, {"approve": False}) == {"a": None}
    assert _run(text, {"approve": True}) == {"a": "after"}


def test_depends_on_a_case_runs_on_every_branch():
    """`depends_on` a CASE runs whichever branch the CASE takes, so its output needs no null."""
    nodes = _R1 + _code("after", '"after"', extra="    depends_on: [gate]\n")
    text = _flow(nodes, "\n  a: ${after.output}")
    assert _sources(text) == ["${after.output}"]
    assert _run(text, {"approve": True}) == {"a": "after"}
    assert _run(text, {"approve": False}) == {"a": "after"}


_TAKE = """defs:
  take:
    input:
      stance: !str
    nodes:
      x:
        kind: code
        output: str
        input:
          stance: ${input.stance}
        code: |
          return inputs["stance"]
    output: ${x.output}
"""


def test_then_call_target_is_null():
    """A `then: call(...)` target gives null when an outer CASE skips its CASE."""
    # Reading `${inner.output}` makes `inner` produce a value, so it must be exhaustive;
    # the skip comes from the routing-only `outer`, which may match nothing.
    nodes = (
        _gate("inner", name="outer")
        + _gate('call(take, stance="pro")', 'call(take, stance="con")', name="inner", when="inner")
    )
    text = _flow(
        nodes, "\n  v: ${inner.output}", inputs="approve: !bool\n  inner: !bool", defs=_TAKE
    )
    assert _run(text, {"approve": False, "inner": True}) == {"v": None}
    assert _run(text, {"approve": True, "inner": True}) == {"v": "pro"}
    assert _run(text, {"approve": True, "inner": False}) == {"v": "con"}


def test_call_node_target_is_null():
    """A `kind: call` node behind a CASE gives null when skipped."""
    nodes = (
        _gate("c", "rejected")
        + '  c:\n    kind: call\n    call: take\n    input:\n      stance: "pro"\n'
        + _code("rejected", '"no"')
    )
    text = _flow(nodes, "\n  v: ${c.output}", defs=_TAKE)
    assert _run(text, {"approve": False}) == {"v": None}
    assert _run(text, {"approve": True}) == {"v": "pro"}


_BUMP = """defs:
  bump:
    input:
      n: !int
    nodes:
      step:
        kind: code
        input:
          n: ${input.n}
        output:
          n: int
        code: |
          return {"n": inputs["n"] + 1}
    output: ${step.output}
"""


def test_loop_node_target_is_null():
    """A `kind: loop` node behind a CASE gives null when skipped."""
    nodes = (
        _gate("lp", "rejected")
        + "  lp:\n    kind: loop\n    call: bump\n    input:\n      n: 0\n"
        "    while: n < 2\n    max: 5\n"
        + _code("rejected", '"no"')
    )
    text = _flow(nodes, "\n  v: ${lp.output}", defs=_BUMP)
    assert _run(text, {"approve": False}) == {"v": None}
    assert _run(text, {"approve": True}) == {"v": {"n": 2}}


def test_map_node_target_is_null():
    """A `kind: map` node behind a CASE gives null when skipped."""
    nodes = (
        _gate("m", "rejected")
        + "  m:\n    kind: map\n    call: take\n    over: ${input.xs}\n"
        "    input:\n      stance: ${item}\n"
        + _code("rejected", '"no"')
    )
    text = _flow(nodes, "\n  v: ${m.output}", inputs="approve: !bool\n  xs: !list(str)", defs=_TAKE)
    assert _run(text, {"approve": False, "xs": ["p"]}) == {"v": None}
    assert _run(text, {"approve": True, "xs": ["p"]}) == {"v": ["p"]}


def test_defs_child_returns_null_to_its_caller():
    """A `defs:` child whose output reads a node behind its own CASE returns null."""
    defs = (
        "defs:\n  child:\n    input:\n      approve: !bool\n    nodes:\n"
        "      gate:\n        kind: case\n        cases:\n"
        "          - when: ${input.approve}\n            then: order\n"
        "      order:\n        kind: code\n        output: str\n"
        '        code: |\n          return "filled"\n'
        "    output: ${order.output}\n"
    )
    nodes = "  c:\n    kind: call\n    call: child\n    input:\n      approve: ${input.approve}\n"
    text = _flow(nodes, " ${c.output}", defs=defs)
    assert _sources(text) == ["${c.output}"]
    assert _run(text, {"approve": False}) is None
    assert _run(text, {"approve": True}) == "filled"


# then: pro / else: caut (always takes a branch)
_JOIN = _gate("pro", "caut") + _code("pro", '"pro"') + _code("caut", '"caut"')


def test_coalesce_over_an_exhaustive_case_is_unchanged():
    """`${pro.output | caut.output}` over then/else can never be skipped."""
    text = _flow(_JOIN, "\n  v: ${pro.output | caut.output}")
    assert _sources(text) == ["${pro.output | caut.output}"]
    assert _run(text, {"approve": True}) == {"v": "pro"}
    assert _run(text, {"approve": False}) == {"v": "caut"}


def test_case_output_of_an_exhaustive_case_is_unchanged():
    """`${gate.output}` of a then/else CASE keeps its coalesce."""
    text = _flow(_JOIN, "\n  v: ${gate.output}")
    assert _sources(text) == ["${pro.output | caut.output}"]
    assert _run(text, {"approve": False}) == {"v": "caut"}


# A typed CASE with a value branch: `then: "pro"` / else: caut (always takes a branch)
_TYPED_JOIN = (
    "  route:\n    kind: case\n    output: str\n    cases:\n"
    '      - when: ${input.approve}\n        then: "pro"\n'
    "    else: caut\n" + _code("caut", '"caut"')
)


def test_case_output_of_a_typed_exhaustive_case_is_unchanged():
    """`${route.output}` of a typed exhaustive CASE reads its result node, never null."""
    text = _flow(_TYPED_JOIN, "\n  v: ${route.output}")
    assert _sources(text) == ["${__case_route.output}"]
    assert _run(text, {"approve": True}) == {"v": "pro"}
    field = _call_field_type(_TYPED_JOIN, "\n  v: ${route.output}", "v")
    assert field.kind is ValueKind.STRING
    assert not field.nullable


def test_case_output_over_a_typed_inner_case_is_unchanged():
    """An untyped exhaustive CASE whose branch is a typed CASE keeps a plain coalesce."""
    nodes = (
        _gate("route", "hold", name="gate", when="other")
        + _TYPED_JOIN
        + _code("hold", '"held"')
    )
    text = _flow(nodes, "\n  v: ${gate.output}", inputs=_R2_INPUTS)
    assert _sources(text) == ["${__case_route.output | hold.output}"]
    assert _run(text, {"approve": False, "other": True}) == {"v": "caut"}
    assert _run(text, {"approve": True, "other": False}) == {"v": "held"}


def test_case_output_of_a_typed_case_over_a_typed_inner_case_is_unchanged():
    """A typed exhaustive CASE whose branch is a typed CASE reads its result node."""
    nodes = (
        "  gate:\n    kind: case\n    output: str\n    cases:\n"
        "      - when: ${input.other}\n        then: route\n"
        '    else: "held"\n' + _TYPED_JOIN
    )
    text = _flow(nodes, "\n  v: ${gate.output}", inputs=_R2_INPUTS)
    assert _sources(text) == ["${__case_gate.output}"]
    assert _run(text, {"approve": True, "other": True}) == {"v": "pro"}
    assert _run(text, {"approve": True, "other": False}) == {"v": "held"}
    field = _call_field_type(nodes, "\n  v: ${gate.output}", "v", other=True)
    assert field.kind is ValueKind.STRING
    assert not field.nullable


def test_root_node_output_is_unchanged():
    """An output reading a node no CASE routes keeps its source."""
    text = _flow(_R1 + _code("root", '"r"'), "\n  r: ${root.output}")
    assert _sources(text) == ["${root.output}"]
    assert _run(text, {"approve": False}) == {"r": "r"}


def test_reader_of_an_exhaustive_join_is_unchanged():
    """A node that reads an exhaustive join always runs."""
    extra = "    input:\n      v: ${pro.output | caut.output}\n"
    text = _flow(_JOIN + _code("j", 'inputs["v"]', extra=extra), "\n  j: ${j.output}")
    assert _sources(text) == ["${j.output}"]
    assert _run(text, {"approve": False}) == {"j": "caut"}


def test_literal_output_is_unchanged():
    """A literal output is left alone."""
    text = _flow(_R1, "\n  k: 5")
    assert _sources(text) == [5]
    assert _run(text, {"approve": False}) == {"k": 5}


def test_embedded_template_is_unchanged():
    """Embedded text renders a skipped reference as an empty string, as before."""
    text = _flow(_R1, "\n  t: '${order.output} and ${rejected.output}'")
    assert _sources(text) == ["${order.output} and ${rejected.output}"]
    assert _run(text, {"approve": False}) == {"t": " and no"}


def test_comparison_is_unchanged():
    """A comparison over a skipped reference is false, as before."""
    text = _flow(_R1, '\n  c: ${order.output == "filled"}')
    assert _sources(text) == ['${order.output == "filled"}']
    assert _run(text, {"approve": False}) == {"c": False}
    assert _run(text, {"approve": True}) == {"c": True}


@pytest.mark.parametrize(("tag", "expected"), [(None, None), ('"a"', "A")])
def test_on_over_a_nullable_literal_may_match_nothing(tag, expected):
    """A null `on:` value matches no tag, so the targets' join may be skipped."""
    nodes = (
        _code("cls", tag or "None", output="Optional(Literal(a, b))")
        + "  route:\n    kind: case\n    on: ${cls.output}\n    cases:\n"
        "      - when: a\n        then: na\n      - when: b\n        then: nb\n"
        + _code("na", '"A"')
        + _code("nb", '"B"')
    )
    text = _flow(nodes, "\n  v: ${na.output | nb.output}")
    assert _sources(text) == ["${(na.output | nb.output) :- null}"]
    assert _run(text, {"approve": True}) == {"v": expected}


def _child_defs(
    nodes: str, output: str, name: str = "child", inputs: str = "approve: !bool"
) -> str:
    """Return a `defs:` block with one child taking `inputs` (one per line) over `nodes`."""
    return (
        f"defs:\n  {name}:\n    input:\n{textwrap.indent(inputs, '      ')}\n    nodes:\n"
        f"{textwrap.indent(nodes, '    ')}    output:{textwrap.indent(output, '    ')}\n"
    )


_CALL_C1 = "  c1:\n    kind: call\n    call: child\n    input:\n      approve: ${input.approve}\n"


def _call_field_type(nodes: str, output: str, field: str, *, other: bool = False):
    """Return the type of `field` in the output contract of a CALL to a child flow.

    With `other`, the child also takes `other: !bool` and the CALL passes `false`.
    """
    call, inputs = _CALL_C1, "approve: !bool"
    if other:
        call, inputs = call + "      other: false\n", inputs + "\nother: !bool"
    text = _flow(call, " ${c1.output}", defs=_child_defs(nodes, output, inputs=inputs))
    contract = load_flow(text).compiled.nodes["c1"].output_contract
    return contract.type.fields[field]


def test_call_output_of_a_skipped_branch_is_optional():
    """A CALL to a child whose output reads a CASE target sees `Optional(str)`."""
    field = _call_field_type(_R1, "\n  order: ${order.output}", "order")
    assert field.kind is ValueKind.STRING
    assert field.nullable


def test_call_output_of_a_coalesce_over_a_routing_only_case_is_optional():
    """A coalesce of `str` targets of a CASE that may match nothing is `Optional(str)`."""
    field = _call_field_type(_R2, "\n  v: ${a.output | b.output}", "v", other=True)
    assert field.kind is ValueKind.STRING
    assert field.nullable


def test_call_output_of_a_dotted_skipped_read_is_optional():
    """A field read `${order.output.status}` of a CASE target is Optional of its type."""
    nodes = (
        _gate("order", "rejected")
        + _code("order", '{"status": 1}', output="\n      status: int")
        + _code("rejected", '"no"')
    )
    field = _call_field_type(nodes, "\n  s: ${order.output.status}", "s")
    assert field.kind is ValueKind.INTEGER
    assert field.nullable


def test_typed_output_over_a_skipped_branch_fails_at_runtime():
    """`order: !str ${order.output}` loads; a skip fails the declared type at runtime."""
    text = _flow(_R1, "\n  order: !str ${order.output}")
    assert _run(text, {"approve": True}) == {"order": "filled"}
    result = run_flow(load_flow(text), {"approve": False})
    assert result.status == "failed"
    assert result.failure.code == "binding.type"


def test_optional_typed_output_over_a_skipped_branch_is_null():
    """`order: !Optional(str) ${order.output}` returns null when skipped."""
    text = _flow(_R1, "\n  order: !Optional(str) ${order.output}")
    assert _run(text, {"approve": False}) == {"order": None}
    assert _run(text, {"approve": True}) == {"order": "filled"}


_USE_STR = """  use:
    input:
      s: !str
    nodes:
      x:
        kind: code
        output: str
        input:
          s: ${input.s}
        code: |
          return inputs["s"]
    output: ${x.output}
"""


def _pass_on(binding: str) -> str:
    """Return a parent passing `c1`'s skipped output to a `str` input of another child."""
    defs = _child_defs(_R1, "\n  order: ${order.output}") + _USE_STR
    nodes = _CALL_C1 + f"  c2:\n    kind: call\n    call: use\n    input:\n      {binding}\n"
    return _flow(nodes, " ${c2.output}", defs=defs)


def test_passing_a_skipped_child_output_to_a_str_input_is_a_load_error():
    """An untyped argument passing an `Optional(str)` child output to `str` is rejected."""
    with pytest.raises(LoadError, match=r"Optional\(str\)"):
        load_flow(_pass_on("s: ${c1.output.order}"))


def test_typed_argument_narrows_a_skipped_child_output_at_runtime():
    """`s: !str ${c1.output.order}` loads; a skip fails the declared type at runtime."""
    text = _pass_on("s: !str ${c1.output.order}")
    assert _run(text, {"approve": True}) == "filled"
    result = run_flow(load_flow(text), {"approve": False})
    assert result.status == "failed"
    assert result.failure.code == "binding.type"


# A LOOP body that makes a note only when `flip` holds (a CASE with no `else:`).
_FLIP_BODY = """      gate:
        kind: case
        cases:
          - when: ${input.flip}
            then: mk
      mk:
        kind: code
        output: str
        code: |
          return "made"
      bump:
        kind: code
        input:
          n: ${input.n}
        output: int
        code: |
          return inputs["n"] + 1
"""


def _inline_loop(note_seed: str, times: int) -> str:
    """Return a flow whose inline LOOP body returns `note: ${mk.output}`."""
    nodes = (
        f"  repeat:\n    kind: loop\n    times: {times}\n    input:\n      n: 0\n"
        f"      note: {note_seed}\n      flip: ${{input.flip}}\n    nodes:\n"
        f"{textwrap.indent(_FLIP_BODY, '  ')}"
        "    output:\n      n: ${bump.output}\n      note: ${mk.output}\n"
    )
    return _flow(nodes, " ${repeat.output}", inputs="flip: !bool")


def test_inline_loop_over_an_untyped_seed_asks_for_its_type():
    """The body field is `Optional(str)`, so an untyped `"start"` seed must be typed."""
    with pytest.raises(LoadError, match="declare the seed's type"):
        load_flow(_inline_loop('"start"', 1))


def test_inline_loop_sets_a_skipped_field_to_null():
    """A skipped body field sets the carried field to null, not its previous value."""
    text = _inline_loop('!str "start"', 1)
    assert _run(text, {"flip": False}) == {"n": 1, "note": None, "flip": False}


def test_inline_loop_over_a_typed_str_seed_rejects_the_null_next_iteration():
    """Over a typed `str` seed, the null fails the next iteration's parameter check."""
    result = run_flow(load_flow(_inline_loop('!str "start"', 2)), {"flip": False})
    assert result.status == "failed"
    assert result.failure.code == "contract.input"


@pytest.mark.parametrize(("flip", "note"), [(False, None), (True, "made")])
def test_inline_loop_over_an_optional_seed_carries_the_null(flip, note):
    """Over an `Optional(str)` seed the null is carried; a taken branch gives its value."""
    text = _inline_loop('!Optional(str) "start"', 2)
    assert _run(text, {"flip": flip}) == {"n": 2, "note": note, "flip": flip}


def test_named_loop_body_sets_a_skipped_field_to_null():
    """A `defs:` LOOP body returning a skipped node's output carries null."""
    defs = (
        "defs:\n  body:\n    input:\n      n: !int\n      note: !Optional(str)\n"
        "      flip: !bool\n    nodes:\n"
        f"{_FLIP_BODY}"
        "    output:\n      n: ${bump.output}\n      note: ${mk.output}\n"
    )
    nodes = (
        "  repeat:\n    kind: loop\n    call: body\n    times: 2\n    input:\n"
        '      n: 0\n      note: "start"\n      flip: ${input.flip}\n'
    )
    text = _flow(nodes, " ${repeat.output}", inputs="flip: !bool", defs=defs)
    assert _run(text, {"flip": False}) == {"n": 2, "note": None, "flip": False}


def test_map_inline_body_returns_null_per_skipped_element():
    """A MAP over an inline body gives null for each element whose branch was skipped."""
    nodes = (
        "  each:\n    kind: map\n    over: [true, false]\n    input:\n      flag: ${item}\n"
        "    nodes:\n"
        "      gate:\n        kind: case\n        cases:\n"
        "          - when: ${input.flag}\n            then: mk\n"
        "      mk:\n        kind: code\n        output: str\n"
        '        code: |\n          return "made"\n'
        "    output: ${mk.output}\n"
    )
    text = _flow(nodes, " ${each.output}", inputs="approve: !bool")
    assert _run(text, {"approve": True}) == ["made", None]


_REPLAY_R1 = _gate("order", "rejected") + "".join(
    f"  {node}:\n    kind: code\n    output: str\n"
    f"    code: tests.engine._compose_codefns:skipped_{node}\n"
    for node in ("order", "rejected")
)


def test_replay_returns_the_same_null(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Resuming a finished run that returned a skipped null replays it without running code."""
    seen: list[str] = []

    def skipped_order(inputs: dict) -> str:
        seen.append("order")
        return "filled"

    def skipped_rejected(inputs: dict) -> str:
        seen.append("rejected")
        return "no"

    for fn in (skipped_order, skipped_rejected):
        monkeypatch.setattr(codefns, fn.__name__, fn, raising=False)
    text = _flow(_REPLAY_R1, "\n  order: ${order.output}")
    path = tmp_path / "r.jsonl"

    first = run_flow(load_flow(text), {"approve": False}, log=path)
    assert first.status == "succeeded", first.failure
    assert first.output == {"order": None}
    assert seen == ["rejected"]

    seen.clear()
    resumed = resume_flow(load_flow(text), log=path)
    assert resumed.status == first.status
    assert resumed.output == first.output
    assert seen == []


# then: order / else: rejected, where `order` returns a record with an `int` field
_QTY = (
    _gate("order", "rejected")
    + _code("order", '{"qty": 2}', output="\n      qty: int")
    + _code("rejected", '"no"')
)


def _load_error(text: str) -> LoadError:
    """Load `text`, which must fail, and return the load error."""
    with pytest.raises(LoadError) as caught:
        load_flow(text)
    return caught.value


def _error_lines(error: LoadError, text: str) -> list[int]:
    """Return the lines of `text` the error's innermost frame covers."""
    return error.diagnostic.frames[-1].range.covered_lines(text)


def test_arithmetic_on_a_skipped_output_is_a_load_error():
    """`${order.output.qty + 1}` in an output asks for a `:-` default, at the output line."""
    text = _flow(_QTY, "\n  q: ${order.output.qty + 1}")
    error = _load_error(text)
    message = str(error)
    assert "output 'q'" in message
    assert "order.output.qty" in message
    assert ":-" in message
    line = text.splitlines().index("  q: ${order.output.qty + 1}") + 1
    assert _error_lines(error, text) == [line]


def test_arithmetic_in_a_node_input_that_still_runs_is_a_load_error():
    """A node that still runs when `order` is skipped can't do arithmetic on its output."""
    extra = "    input:\n      q: ${order.output.qty * 2}\n"
    use = _code("use", 'inputs["q"]', output="int", extra=extra)
    text = _flow(_QTY + use, "\n  u: ${use.output}")
    error = _load_error(text)
    assert "node 'use' input 'q'" in str(error)
    assert "order.output.qty" in str(error)
    line = text.splitlines().index("      q: ${order.output.qty * 2}") + 1
    assert _error_lines(error, text) == [line]


def test_arithmetic_in_an_embedded_template_is_a_load_error():
    """Arithmetic inside surrounding text is checked too."""
    error = _load_error(_flow(_QTY, "\n  t: 'total ${order.output.qty + 1}'"))
    assert "output 't'" in str(error)


def test_arithmetic_in_a_case_on_is_a_load_error():
    """A CASE `on:` that does arithmetic on a skipped value is rejected, naming `on:`."""
    nodes = (
        _QTY
        + "  route:\n    kind: case\n    on: ${order.output.qty + 1}\n    cases:\n"
        "      - when: 3\n        then: three\n    else: other\n"
        + _code("three", '"3"')
        + _code("other", '"?"')
    )
    error = _load_error(_flow(nodes, "\n  v: ${three.output | other.output}"))
    assert "node 'route' on: ${order.output.qty + 1}" in str(error)
    assert "order.output.qty" in str(error)


@pytest.mark.parametrize("expr", ["-order.output.qty", "order.output.qty ** 2"])
def test_unary_minus_and_power_on_a_skipped_output_are_load_errors(expr):
    """Unary minus and `**` are arithmetic too."""
    error = _load_error(_flow(_QTY, f"\n  q: ${{{expr}}}"))
    assert "order.output.qty" in str(error)


def test_arithmetic_on_a_coalesce_of_routing_only_targets_is_a_load_error():
    """`${(a.output | b.output) + 1}` is null when the CASE matches nothing."""
    nodes = (
        "  gate:\n    kind: case\n    cases:\n"
        "      - when: ${input.approve}\n        then: a\n"
        "      - when: ${input.other}\n        then: b\n"
        + _code("a", "1", output="int")
        + _code("b", "2", output="int")
    )
    text = _flow(nodes, "\n  v: ${(a.output | b.output) + 1}", inputs=_R2_INPUTS)
    error = _load_error(text)
    assert "nodes 'a' and 'b'" in str(error)


def test_arithmetic_in_a_node_that_co_skips_loads():
    """A node that also reads `${order.output}` plainly is skipped with it."""
    extra = "    input:\n      q: ${order.output.qty * 2}\n      o: ${order.output}\n"
    use = _code("use", 'inputs["q"]', output="int", extra=extra)
    text = _flow(_QTY + use, "\n  u: ${use.output}")
    assert _run(text, {"approve": True}) == {"u": 4}
    assert _run(text, {"approve": False}) == {"u": None}


def test_arithmetic_in_a_node_routed_by_a_case_on_the_value_loads():
    """A node behind a CASE whose `when:` reads `order` never runs when `order` is skipped."""
    extra = "    input:\n      q: ${order.output.qty * 2}\n"
    nodes = (
        _QTY
        + "  check:\n    kind: case\n    cases:\n"
        "      - when: ${order.output.qty > 1}\n        then: use\n"
        + _code("use", 'inputs["q"]', output="int", extra=extra)
    )
    text = _flow(nodes, "\n  u: ${use.output}")
    assert _run(text, {"approve": True}) == {"u": 4}
    assert _run(text, {"approve": False}) == {"u": None}


def test_arithmetic_on_a_node_that_depends_on_a_case_loads():
    """A node with `depends_on` a CASE runs on every branch, so its output is never null."""
    note = _code("note", "1", output="int", extra="    depends_on: [gate]\n")
    text = _flow(_QTY + note, "\n  q: ${order.output.qty :- (note.output + 1)}")
    assert _run(text, {"approve": True}) == {"q": 2}
    assert _run(text, {"approve": False}) == {"q": 2}


def test_arithmetic_on_a_defaulted_value_loads():
    """`(order.output.qty :- 0) + 1` gives an explicit default."""
    text = _flow(_QTY, "\n  q: ${(order.output.qty :- 0) + 1}")
    assert _run(text, {"approve": False}) == {"q": 1}
    assert _run(text, {"approve": True}) == {"q": 3}


def test_comparison_on_a_skipped_output_loads():
    """A comparison is not arithmetic."""
    load_flow(_flow(_QTY, "\n  c: ${order.output.qty > 1}"))


def test_arithmetic_on_an_exhaustive_join_loads():
    """`(pro.output | caut.output) + 1` over then/else is never null from a skip."""
    nodes = (
        _gate("pro", "caut") + _code("pro", "1", output="int") + _code("caut", "2", output="int")
    )
    text = _flow(nodes, "\n  v: ${(pro.output | caut.output) + 1}")
    assert _run(text, {"approve": True}) == {"v": 2}
    assert _run(text, {"approve": False}) == {"v": 3}


def test_arithmetic_on_an_input_loads():
    """A flow input is never skipped."""
    text = _flow(_QTY, "\n  v: ${input.x + 1}", inputs="approve: !bool\n  x: !int")
    assert _run(text, {"approve": False, "x": 1}) == {"v": 2}


def test_arithmetic_on_a_node_that_cannot_be_skipped_runs():
    """Arithmetic over a root node in a flow with a CASE loads and runs as before."""
    nodes = _QTY + _code("root", "5", output="int")
    text = _flow(nodes, "\n  v: ${root.output * 2 + 1}\n  w: ${-root.output}")
    assert _run(text, {"approve": False}) == {"v": 11, "w": -5}


def test_rewritten_output_fingerprints_as_the_explicit_null_default(
    monkeypatch: pytest.MonkeyPatch,
):
    """The rewrite changes the flow's fingerprint to that of the authored `:-null` form."""
    from yian.compile.manifest import flow_fingerprint
    import yian.compose.build.pipeline as pipeline

    rewritten = flow_fingerprint(load_flow(_flow(_R1, "\n  order: ${order.output}")).compiled)
    explicit = load_flow(_flow(_R1, "\n  order: ${order.output:-null}")).compiled
    assert rewritten == flow_fingerprint(explicit)
    monkeypatch.setattr(pipeline, "null_skipped_outputs", lambda outputs, model: outputs)
    plain = load_flow(_flow(_R1, "\n  order: ${order.output}")).compiled
    assert plain.outputs[0].from_ == "${order.output}"
    assert rewritten != flow_fingerprint(plain)


def test_hint_names_a_placeholder_default():
    """The hint asks for a default without guessing its value."""
    message = str(_load_error(_flow(_QTY, "\n  q: ${order.output.qty + 1}")))
    assert message.endswith(
        "give it a default with :-, e.g. (order.output.qty :- <default>)"
    )


def test_hint_replaces_a_null_default():
    """An operand already defaulted to null gets the placeholder in place of `null`."""
    message = str(_load_error(_flow(_QTY, "\n  q: ${(order.output.qty :- null) + 1}")))
    assert message.endswith("e.g. (order.output.qty :- <default>)")


# `a` is skippable (then: a / else: rejected); `gate2` is a searched case that reads it.
_A = _gate("a", "rejected") + _code("a", '{"n": 2}', output="\n      n: int") + _code(
    "rejected", '"no"'
)


def _searched_gate(when: str) -> str:
    """Return the flow where searched case `gate2` routes to `hot` when `when` holds."""
    nodes = (
        _A
        + f'  gate2:\n    kind: case\n    cases:\n      - when: "{when}"\n        then: hot\n'
        "    else: cold\n"
        + _code("hot", '"hot"')
        + _code("cold", '"cold"')
    )
    return _flow(nodes, "\n  v: ${hot.output | cold.output}")


def test_arithmetic_in_a_searched_when_under_a_default_is_a_load_error():
    """A searched `when:` runs with `a` skipped under `:-`, so arithmetic inside is rejected."""
    when = "((a.output.n + 1) :- 0) > 1"
    text = _searched_gate(when)
    error = _load_error(text)
    assert str(error) == (
        "node 'gate2' cases[0].when: ((a.output.n + 1) :- 0) > 1 does arithmetic on "
        "a.output.n, which is null when node 'a' is skipped; give it a default with :-, "
        "e.g. (a.output.n :- <default>)"
    )
    line = text.splitlines().index(f'      - when: "{when}"') + 1
    assert _error_lines(error, text) == [line]


def test_arithmetic_on_a_null_default_in_a_searched_when_is_a_load_error():
    """`(a.output.n :- null) + 1` in a searched `when:` asks for a real default."""
    message = str(_load_error(_searched_gate("(a.output.n :- null) + 1 > 1")))
    assert message.startswith("node 'gate2' cases[0].when: (a.output.n :- null) + 1 > 1 ")
    assert message.endswith("e.g. (a.output.n :- <default>)")


def test_arithmetic_on_a_defaulted_value_in_a_searched_when_loads():
    """`(a.output.n :- 0) + 1` in a searched `when:` gives the default and runs."""
    text = _searched_gate("(a.output.n :- 0) + 1 > 1")
    assert _run(text, {"approve": True}) == {"v": "hot"}
    assert _run(text, {"approve": False}) == {"v": "cold"}


def test_plain_arithmetic_in_a_searched_when_loads():
    """A plain read skips `gate2` with `a`, so its arithmetic never sees a skipped null."""
    text = _searched_gate("a.output.n + 1 > 1")
    assert _run(text, {"approve": True}) == {"v": "hot"}
    assert _run(text, {"approve": False}) == {"v": None}


_UNION_FLOW = """id: f
name: f
typedefs:
  Buy: {action: 'Literal(buy)', qty: int}
  Sell: {action: 'Literal(sell)', qty: int}
  Decision: Union(Buy, Sell)
input:
  approve: !bool
nodes:
  gate:
    kind: case
    cases:
      - when: ${input.approve}
        then: d
    else: rejected
  d:
    kind: code
    output: Decision
    code: |
      return {"action": "sell", "qty": 9}
  rejected:
    kind: code
    output: str
    code: |
      return "no"
  gate2:
    kind: case
    cases:
      - when: "isinstance(d.output, Buy) and ((d.output.qty + 1) :- 0) > 5"
        then: hot
    else: cold
  hot:
    kind: code
    output: str
    code: |
      return "hot"
  cold:
    kind: code
    output: str
    code: |
      return "cold"
output:
  v: ${hot.output | cold.output}
"""


def test_isinstance_guard_in_a_searched_when_over_a_skippable_union_routes():
    """Arithmetic on a narrowed, defaulted field loads and routes a Sell to `else:`.

    The plain `isinstance(d.output, Buy)` read skips `gate2` with `d`, so the arithmetic
    never sees a skipped null.
    """
    assert _run(_UNION_FLOW, {"approve": True}) == {"v": "cold"}
    assert _run(_UNION_FLOW, {"approve": False}) == {"v": None}


_CHILD = (
    "defs:\n  child:\n    input:\n      n: !int\n    nodes:\n      k:\n        kind: code\n"
    "        input:\n          n: ${input.n}\n        output: int\n        code: |\n"
    "          return inputs['n']\n    output: ${k.output}\n"
)
# then: order / else: rejected, where `order` returns a list field and a str field
_ITEMS = (
    _gate("order", "rejected")
    + _code("order", '{"items": [1], "s": "2020"}', output="\n      items: list(int)\n      s: str")
    + _code("rejected", '"no"')
)


def test_arithmetic_in_a_map_over_names_map_over():
    """A MAP `over:` that does arithmetic on a skipped value is rejected as `map over`."""
    over = "    over: ${order.output.items + [3]}\n"
    nodes = _ITEMS + "  m:\n    kind: map\n" + over + "    call: child\n    input:\n      n: ${item}\n"
    text = _flow(nodes, "\n  r: ${rejected.output}", defs=_CHILD)
    error = _load_error(text)
    assert "node 'm' map over: ${order.output.items + [3]}" in str(error)
    assert _error_lines(error, text) == [text.splitlines().index(over.rstrip("\n")) + 1]


def test_arithmetic_in_a_wait_until_names_wait_until():
    """A WAIT `until:` that does arithmetic on a skipped value is rejected as `wait until`."""
    until = "    until: ${order.output.s + '-01-01'}\n"
    text = _flow(_ITEMS + "  w:\n    kind: wait\n" + until, "\n  r: ${rejected.output}")
    error = _load_error(text)
    assert "node 'w' wait until: ${order.output.s + '-01-01'}" in str(error)
    assert _error_lines(error, text) == [text.splitlines().index(until.rstrip("\n")) + 1]


def test_arithmetic_in_a_default_is_checked_as_if_it_always_runs():
    """`a.output :- (b.output + 1)` is rejected though `b` runs whenever `a` is skipped."""
    nodes = _gate("a", "b") + _code("a", "1", output="int") + _code("b", "2", output="int")
    error = _load_error(_flow(nodes, "\n  v: ${a.output :- (b.output + 1)}"))
    assert "does arithmetic on b.output" in str(error)


# The Literal input type `_on_input` routes on; a two-member Literal can't sit in a tag.
_K = "typedefs:\n  K: Literal(x, y)\n"


def _on_input(tags: str = "x y") -> str:
    """Return a CASE routing on `input.k` to `a`/`b` for each tag in `tags`."""
    cases = "".join(
        f"      - when: {tag}\n        then: {target}\n" for tag, target in zip(tags.split(), "ab")
    )
    return (
        "  sw:\n    kind: case\n    on: ${input.k}\n    cases:\n"
        + cases
        + _code("a", "1", output="int")
        + _code("b", "2", output="int")
    )


@pytest.mark.parametrize(
    ("tags", "inputs", "source", "expected"),
    [
        ("x y", "k: !K", "${a.output | b.output}", "${a.output | b.output}"),
        ("x", "k: !str", "${a.output}", "${a.output:-null}"),
        (
            "x y",
            "k: !Optional(K)",
            "${a.output | b.output}",
            "${(a.output | b.output) :- null}",
        ),
    ],
)
def test_on_over_a_literal_input(tags, inputs, source, expected):
    """A fully listed non-nullable Literal input always takes a branch; a `str` or nullable
    input may not, so its join defaults to null."""
    text = _flow(_on_input(tags), f"\n  v: {source}", inputs=inputs, typedefs=_K)
    assert _sources(text) == [expected]
    assert _run(text, {"k": "x"}) == {"v": 1}


def test_arithmetic_on_a_join_over_a_literal_input_loads():
    """The join of a fully listed Literal input is never null from a skip."""
    text = _flow(
        _on_input(), "\n  v: ${(a.output | b.output) + 1}", inputs="k: !K", typedefs=_K
    )
    assert _run(text, {"k": "y"}) == {"v": 3}


def test_child_routing_on_a_literal_input_passes_to_a_str_input():
    """A child whose output is `${sw.output}` over a Literal input returns `str` to its parent."""
    nodes = (
        "  sw:\n    kind: case\n    on: ${input.k}\n    cases:\n"
        "      - when: x\n        then: a\n      - when: y\n        then: b\n"
        + _code("a", '"A"')
        + _code("b", '"B"')
    )
    defs = _child_defs(nodes, " ${sw.output}", inputs="k: !K") + _USE_STR
    calls = (
        "  c1:\n    kind: call\n    call: child\n    input:\n      k: y\n"
        "  c2:\n    kind: call\n    call: use\n    input:\n      s: ${c1.output}\n"
    )
    assert _run(_flow(calls, " ${c2.output}", defs=defs, typedefs=_K), {"approve": True}) == "B"


def test_map_body_routing_on_a_literal_item_is_not_rewritten():
    """A MAP body routing on a Literal element always takes a branch, so its join stays."""
    body = textwrap.indent(_on_input(), "    ")

    def each(output: str) -> str:
        return _flow(
            "  each:\n    kind: map\n    over: ${input.ks}\n    input:\n      k: ${item}\n"
            f"    nodes:\n{body}    output: {output}\n",
            " ${each.output}",
            inputs="ks: !list(K)",
            typedefs=_K,
        )

    child = load_flow(each("${a.output | b.output}")).compiled.nodes["each"].child
    assert [output.from_ for output in child.outputs] == ["${a.output | b.output}"]
    assert _run(each("${(a.output | b.output) + 1}"), {"ks": ["x", "y"]}) == [2, 3]
