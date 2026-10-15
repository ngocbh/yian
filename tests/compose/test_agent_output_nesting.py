"""An AGENT's `output:` type nests at most `MAX_AGENT_OUTPUT_NESTING` levels (16) at load.

Building a model's structured reply schema recurses once per type level, so a deeper
type is refused at load rather than overflowing the stack at run. Levels count as
`type_levels` counts them: each record and list is a level, a union counts as its
deepest member, and `Optional` adds none. Every authored AGENT is checked: in `nodes:`, a
compact flow, a `defs:` entry, or an inline body. An AGENT the engine builds with its own
output type, such as an `adaptive_questions:` block's, is not. Other kinds keep only the
type nesting limit.

The message reads `node '<id>': this AGENT's `output:` nests <n> levels, deeper than
<limit>, the limit for a model's structured reply (each record and list is a level);
flatten it, or build the deeper value in a CODE node`, located at the `output:` field.
"""

from __future__ import annotations

import pytest

from yian.compose import LoadError, load_flow
from yian.defaults import DefaultsError, read


def _message(node: str, levels: int, limit: int = 16) -> str:
    return (
        f"node {node!r}: this AGENT's `output:` nests {levels} levels, deeper than "
        f"{limit}, the limit for a model's structured reply (each record and list is a "
        "level); flatten it, or build the deeper value in a CODE node"
    )


def _written(levels: int) -> str:
    """Return a record written out as nested mappings, `levels` records deep."""
    return "{k: " * levels + "int" + "}" * levels


def _typedef_chain(levels: int) -> tuple[str, str]:
    """Return `(typedefs, output)` for a chain of typedefs `levels` records deep."""
    lines = ["typedefs:", "  R1: {k: int}"]
    lines += [f"  R{i}: {{k: R{i - 1}}}" for i in range(2, levels + 1)]
    return "\n".join(lines) + "\n", f"R{levels}"


def _list_of_records(levels: int) -> tuple[str, str]:
    """Return `(typedefs, output)` alternating records and lists, `levels` levels deep."""
    # R_i is a record whose field is a list of R_(i-1), so it nests 2*i - 1 levels.
    records = (levels + 1) // 2
    lines = ["typedefs:", "  R1: {k: int}"]
    lines += [f"  R{i}: {{k: 'list(R{i - 1})'}}" for i in range(2, records + 1)]
    output = f"R{records}" if levels % 2 else f"'list(R{records})'"
    return "\n".join(lines) + "\n", output


def _optional_unions(levels: int) -> tuple[str, str]:
    """Return `(typedefs, output)` with an `Optional(Union(A, S))` field at each level."""
    lines = ["typedefs:", "  S: {tag: Literal(s), n: int}", "  A1: {tag: Literal(a), k: int}"]
    lines += [
        f"  A{i}: {{tag: Literal(a), k: 'Optional(Union(A{i - 1}, S))'}}"
        for i in range(2, levels + 1)
    ]
    return "\n".join(lines) + "\n", f"A{levels}"


def _agent_flow(output: str, typedefs: str = "") -> str:
    """Return a flow whose AGENT `a` declares `output:` on line 7."""
    return (
        "id: f\nname: f\nnodes:\n  a:\n    kind: agent\n    prompt: hi\n"
        f"    output: {output}\noutput: ${{a.output}}\n{typedefs}"
    )


def _assert_refused(
    text: str, node: str, levels: int, line: int, limit: int = 16, *, prefix: str = ""
) -> None:
    with pytest.raises(LoadError) as caught:
        load_flow(text)
    assert str(caught.value) == prefix + _message(node, levels, limit)
    assert caught.value.diagnostic.frames[-1].range.start_line == line


def test_a_written_out_record_past_the_limit_fails_at_load() -> None:
    _assert_refused(_agent_flow(_written(17)), "a", 17, 7)


def test_a_written_out_record_at_the_limit_loads() -> None:
    load_flow(_agent_flow(_written(16)))


@pytest.mark.parametrize(
    "shape", [_typedef_chain, _list_of_records, _optional_unions],
    ids=["typedef-chain", "list-of-records", "optional-union"],
)
def test_each_shape_past_the_limit_fails_and_at_the_limit_loads(shape) -> None:
    typedefs, output = shape(17)
    _assert_refused(_agent_flow(output, typedefs), "a", 17, 7)

    typedefs, output = shape(16)
    load_flow(_agent_flow(output, typedefs))


def test_a_compact_agent_flow_past_the_limit_fails() -> None:
    text = f"id: f\nname: f\nkind: agent\nprompt: hi\noutput: {_written(17)}\n"
    _assert_refused(text, "f", 17, 5)


def test_an_agent_in_a_multi_node_def_past_the_limit_fails() -> None:
    text = (
        "id: f\nname: f\ndefs:\n  d:\n    nodes:\n      u:\n        kind: agent\n"
        f"        prompt: hi\n        output: {_written(17)}\n"
        "      v:\n        kind: code\n        code: tests.engine._compose_codefns:echo\n"
        "        input:\n          topic: x\n        output: str\n"
        "    output: ${u.output}\n"
        "nodes:\n  use:\n    kind: call\n    call: d\noutput: ${use.output}\n"
    )
    _assert_refused(text, "u", 17, 9, prefix="defs entry 'd': ")


def test_an_agent_in_an_inline_map_body_past_the_limit_fails() -> None:
    text = (
        "id: f\nname: f\nnodes:\n  m:\n    kind: map\n    over: [1]\n"
        "    nodes:\n      keep:\n        kind: agent\n        prompt: hi\n"
        f"        output: {_written(17)}\n    output: ${{keep.output}}\n"
        "output: ${m.output}\n"
    )
    _assert_refused(text, "keep", 17, 11)


def test_the_limit_follows_the_variable(set_defaults) -> None:
    set_defaults(MAX_AGENT_OUTPUT_NESTING=3)

    _assert_refused(_agent_flow(_written(4)), "a", 4, 7, limit=3)
    load_flow(_agent_flow(_written(3)))


def test_an_engine_built_agent_output_skips_the_limit(set_defaults) -> None:
    # The questions AGENT an `adaptive_questions:` block builds has a 4-level output.
    set_defaults(MAX_AGENT_OUTPUT_NESTING=3)

    load_flow(
        "id: gate\nname: gate\nnodes:\n  ask:\n    kind: human_input\n"
        "    adaptive_questions:\n      prompt: Ask what you need.\noutput: ${ask.output}\n"
    )


def test_a_fifty_level_output_fails_at_load_without_recursion_error() -> None:
    # Fifty levels load under the type nesting limit but overflow schema generation.
    with pytest.raises(LoadError) as caught:
        load_flow(_agent_flow(_written(50)))

    assert str(caught.value) == _message("a", 50)
    error: BaseException | None = caught.value
    while error is not None:
        assert not isinstance(error, RecursionError)
        error = error.__cause__ or error.__context__


def test_a_code_node_output_past_the_limit_still_loads() -> None:
    load_flow(
        "id: f\nname: f\nnodes:\n  c:\n    kind: code\n"
        "    code: tests.engine._compose_codefns:echo\n    input:\n      topic: x\n"
        f"    output: {_written(17)}\noutput: ${{c.output}}\n"
    )


def test_the_variable_can_only_lower_the_limit() -> None:
    assert read({"YIAN_MAX_AGENT_OUTPUT_NESTING": "16"}).max_agent_output_nesting == 16
    assert read({"YIAN_MAX_AGENT_OUTPUT_NESTING": "1"}).max_agent_output_nesting == 1
    with pytest.raises(DefaultsError, match="YIAN_MAX_AGENT_OUTPUT_NESTING"):
        read({"YIAN_MAX_AGENT_OUTPUT_NESTING": "17"})
