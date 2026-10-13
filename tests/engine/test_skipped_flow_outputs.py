"""A flow output that reads a node CASE routing may skip returns null when it is skipped."""

from __future__ import annotations

import textwrap

import pytest

from yian.compose import LoadError, load_flow
from yian.run import run_flow
from yian.typesys.values import ValueKind


def _flow(nodes: str, output: str, *, inputs: str = "approve: bool", defs: str = "") -> str:
    """Return a flow text with the given `input:`, `defs:`, `nodes:` and `output:` blocks."""
    return (
        f"id: f\nname: f\ninput:\n  {inputs}\n{defs}nodes:\n{nodes}output:{output}\n"
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
# two `when:` cases and no `else:`: neither matches when `approve` is false
_R2 = (
    "  gate:\n    kind: case\n    cases:\n"
    "      - when: ${input.approve}\n        then: a\n"
    "      - when: ${input.approve == false and false}\n        then: b\n"
    + _code("a", '"A"')
    + _code("b", '"B"')
)


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
    text = _flow(_R2, "\n  v: ${a.output | b.output}")
    assert _sources(text) == ["${(a.output | b.output) :- null}"]
    assert _run(text, {"approve": False}) == {"v": None}
    assert _run(text, {"approve": True}) == {"v": "A"}


def test_nested_case_target_is_null_when_either_case_skips_it():
    """A target of an inner CASE gives null when the inner or the outer CASE skips it."""
    nodes = (
        _gate("inner", "rejected", name="outer")
        + _gate("order", "hold", name="inner", when="inner")
        + _code("order", '"filled"')
        + _code("hold", '"held"')
        + _code("rejected", '"no"')
    )
    text = _flow(nodes, "\n  order: ${order.output}", inputs="approve: bool\n  inner: bool")
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
    """`${a.output :- ${b.output}}` over a CASE that may match nothing returns null."""
    text = _flow(_R2, "\n  v: ${a.output :- ${b.output}}")
    assert _sources(text) == ["${(a.output :- ${b.output}) :- null}"]
    assert _run(text, {"approve": False}) == {"v": None}
    assert _run(text, {"approve": True}) == {"v": "A"}


def test_depends_on_a_skipped_target_is_null():
    """A node with `depends_on` a skipped branch target is skipped too, and gives null."""
    nodes = _R1 + _code("after", '"after"', extra="    depends_on: [order]\n")
    text = _flow(nodes, "\n  a: ${after.output}")
    assert _run(text, {"approve": False}) == {"a": None}
    assert _run(text, {"approve": True}) == {"a": "after"}


def test_depends_on_a_case_is_null_when_it_takes_a_real_branch():
    """`depends_on` a CASE is skipped when the CASE takes a `when:` branch, not `else:`."""
    nodes = _R1 + _code("after", '"after"', extra="    depends_on: [gate]\n")
    text = _flow(nodes, "\n  a: ${after.output}")
    assert _sources(text) == ["${after.output:-null}"]
    assert _run(text, {"approve": True}) == {"a": None}
    assert _run(text, {"approve": False}) == {"a": "after"}


_TAKE = """defs:
  take:
    input:
      stance: str
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
    """A `then: call(...)` target of a CASE that may match nothing gives null."""
    nodes = '  gate:\n    kind: case\n    cases:\n      - when: ${input.approve}\n' \
        '        then: call(take, stance="pro")\n'
    text = _flow(nodes, "\n  v: ${gate.output}", defs=_TAKE)
    assert _run(text, {"approve": False}) == {"v": None}
    assert _run(text, {"approve": True}) == {"v": "pro"}


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
      n: int
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
        "    while: ${n} < 2\n    max: 5\n"
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
    text = _flow(nodes, "\n  v: ${m.output}", inputs="approve: bool\n  xs: list[str]", defs=_TAKE)
    assert _run(text, {"approve": False, "xs": ["p"]}) == {"v": None}
    assert _run(text, {"approve": True, "xs": ["p"]}) == {"v": ["p"]}


def test_defs_child_returns_null_to_its_caller():
    """A `defs:` child whose output reads a node behind its own CASE returns null."""
    defs = (
        "defs:\n  child:\n    input:\n      approve: bool\n    nodes:\n"
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
        _code("cls", tag or "None", output="Optional[Literal[a, b]]")
        + "  route:\n    kind: case\n    on: ${cls.output}\n    cases:\n"
        "      - when: a\n        then: na\n      - when: b\n        then: nb\n"
        + _code("na", '"A"')
        + _code("nb", '"B"')
    )
    text = _flow(nodes, "\n  v: ${na.output | nb.output}")
    assert _sources(text) == ["${(na.output | nb.output) :- null}"]
    assert _run(text, {"approve": True}) == {"v": expected}


def _child_defs(nodes: str, output: str, name: str = "child") -> str:
    """Return a `defs:` block with one child taking `approve: bool` over `nodes`."""
    return (
        f"defs:\n  {name}:\n    input:\n      approve: bool\n    nodes:\n"
        f"{textwrap.indent(nodes, '    ')}    output:{textwrap.indent(output, '    ')}\n"
    )


_CALL_C1 = "  c1:\n    kind: call\n    call: child\n    input:\n      approve: ${input.approve}\n"


def _call_field_type(nodes: str, output: str, field: str):
    """Return the type of `field` in the output contract of a CALL to a child flow."""
    text = _flow(_CALL_C1, " ${c1.output}", defs=_child_defs(nodes, output))
    contract = load_flow(text).compiled.nodes["c1"].output_contract
    return contract.type.fields[field]


def test_call_output_of_a_skipped_branch_is_optional():
    """A CALL to a child whose output reads a CASE target sees `Optional[str]`."""
    field = _call_field_type(_R1, "\n  order: ${order.output}", "order")
    assert field.kind is ValueKind.STRING
    assert field.nullable


def test_call_output_of_a_coalesce_over_a_routing_only_case_is_optional():
    """A coalesce of `str` targets of a CASE that may match nothing is `Optional[str]`."""
    field = _call_field_type(_R2, "\n  v: ${a.output | b.output}", "v")
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
    """`order: str = ${order.output}` loads; a skip fails the declared type at runtime."""
    text = _flow(_R1, "\n  order: str = ${order.output}")
    assert _run(text, {"approve": True}) == {"order": "filled"}
    result = run_flow(load_flow(text), {"approve": False})
    assert result.status == "failed"
    assert result.failure.code == "binding.type"


def test_optional_typed_output_over_a_skipped_branch_is_null():
    """`order: Optional[str] = ${order.output}` returns null when skipped."""
    text = _flow(_R1, "\n  order: Optional[str] = ${order.output}")
    assert _run(text, {"approve": False}) == {"order": None}
    assert _run(text, {"approve": True}) == {"order": "filled"}


_USE_STR = """  use:
    input:
      s: str
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
    """An untyped argument passing an `Optional[str]` child output to `str` is rejected."""
    with pytest.raises(LoadError, match=r"Optional\[str\]"):
        load_flow(_pass_on("s: ${c1.output.order}"))


def test_typed_argument_narrows_a_skipped_child_output_at_runtime():
    """`s: str = ${c1.output.order}` loads; a skip fails the declared type at runtime."""
    text = _pass_on("s: str = ${c1.output.order}")
    assert _run(text, {"approve": True}) == "filled"
    result = run_flow(load_flow(text), {"approve": False})
    assert result.status == "failed"
    assert result.failure.code == "binding.type"
