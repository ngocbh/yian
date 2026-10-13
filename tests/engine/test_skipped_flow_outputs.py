"""A flow output that reads a node CASE routing may skip returns null when it is skipped."""

from __future__ import annotations

import pytest

from yian.compose import load_flow
from yian.run import run_flow


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
