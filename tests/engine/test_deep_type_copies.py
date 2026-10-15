"""A run's copy of its flow shares the flow's types, so a deep `output:` record runs.

A `Type` is never changed after it is built, so `copy.deepcopy` returns the type itself,
and copying a compiled flow costs stack for its nodes, not for how deep a type nests. A
100-level `output:` record on a CODE node then runs, runs learnably, and resumes under 10
nested `defs:` calls, and runs under 50; a union nested at every level runs; and a deeper
record allowed by a raised type limit fails its output contract instead of overflowing.
"""

import copy
import inspect
import json
import sys
from contextlib import contextmanager
from pathlib import Path

from yian import load_flow, run_flow
from yian.nodes.contracts import output_contract_type
from yian.run import resume_flow
from yian.typesys.values import Type, ValueKind

_DEEP_TREE = "tests.engine._compose_codefns:deep_tree"


@contextmanager
def _stack_headroom(frames: int):
    """Pin the recursion limit to `frames` past the current stack depth, restoring it after.

    This usually lowers the limit, so a deep flow is shown to fit in that headroom.
    """
    old = sys.getrecursionlimit()
    sys.setrecursionlimit(len(inspect.stack(0)) + frames)
    try:
        yield
    finally:
        sys.setrecursionlimit(old)


def _tree_record(levels: int) -> str:
    """Return a `{tree: {a: ... "str"}}` record type nesting `levels` records in total."""
    return "{tree: " + "{a: " * (levels - 1) + '"str"' + "}" * levels


def _tree_value(levels: int) -> dict:
    """Return the value `deep_tree` makes for `n = levels - 1`, matching `_tree_record`."""
    tree: object = "leaf"
    for _ in range(levels - 1):
        tree = {"a": tree}
    return {"tree": tree}


def _code_node(indent: str, output: str) -> list[str]:
    """Return the lines of a `u` CODE node passing `n` through, with `output:` type `output`."""
    return [
        f"{indent}u:",
        f"{indent}  kind: code",
        f"{indent}  code: {_DEEP_TREE}",
        f"{indent}  input:",
        f"{indent}    n: ${{input.n}}",
        f"{indent}  output: {output}",
    ]


def _flow(levels: int, calls: int) -> str:
    """Return a flow whose CODE node has a `levels`-level `output:` record under `calls` calls.

    With `calls=0` the CODE node sits at the root. Otherwise the root calls `d0`, each
    `di` calls `d(i+1)`, and the last def holds the CODE node.
    """
    output = _tree_record(levels)
    lines = ["id: deep", "name: deep", "input:", "  n: !int"]
    if calls == 0:
        lines += ["nodes:", *_code_node("  ", output), "output: ${u.output}"]
        return "\n".join(lines) + "\n"
    lines.append("defs:")
    for i in range(calls):
        lines += [f"  d{i}:", "    input:", "      n: !int", "    nodes:"]
        if i < calls - 1:
            lines += [
                "      u:",
                "        kind: call",
                f"        call: d{i + 1}",
                "        input:",
                "          n: ${input.n}",
            ]
        else:
            lines += _code_node("      ", output)
        lines.append("    output: ${u.output}")
    lines += [
        "nodes:",
        "  use:",
        "    kind: call",
        "    call: d0",
        "    input:",
        "      n: ${input.n}",
        "output: ${use.output}",
    ]
    return "\n".join(lines) + "\n"


def _record(fields: dict[str, Type]) -> Type:
    """Return a record `Type` requiring every field in `fields`."""
    return Type(kind=ValueKind.OBJECT, fields=fields, required=frozenset(fields))


def test_a_type_copies_as_itself() -> None:
    leaf = Type.scalar(ValueKind.STRING)
    record = _record({"a": leaf})
    typed_list = Type(kind=ValueKind.LIST_OBJECT, element=record)
    union = Type(
        kind=ValueKind.OBJECT, variants=(record, _record({"b": leaf})), nullable=True
    )

    for typ in (leaf, record, typed_list, union):
        assert copy.deepcopy(typ) is typ
        assert copy.deepcopy({"a": typ})["a"] is typ


def test_copying_a_compiled_flow_does_not_recurse_through_a_deep_record() -> None:
    compiled = load_flow(_flow(100, 0)).compiled

    with _stack_headroom(120):
        copied = copy.deepcopy(compiled)

    assert copied is not compiled


def test_a_compiled_flow_copy_has_its_own_nodes_and_shares_their_types() -> None:
    original = load_flow(_flow(3, 0)).compiled

    copied = copy.deepcopy(original)

    assert copied.nodes["u"] is not original.nodes["u"]
    typ = output_contract_type(original.nodes["u"].output_contract)
    assert typ is not None
    assert output_contract_type(copied.nodes["u"].output_contract) is typ


def test_a_deep_record_under_nested_calls_runs() -> None:
    flow = load_flow(_flow(100, 10))

    with _stack_headroom(400):
        result = run_flow(flow, {"n": 99}, num_workers=0)

    assert result.status == "succeeded", result.failure
    assert result.output == _tree_value(100)


def test_a_deep_record_under_nested_calls_runs_learnably_and_resumes(tmp_path: Path) -> None:
    text = _flow(100, 10)
    path = tmp_path / "deep.jsonl"

    with _stack_headroom(400):
        first = run_flow(load_flow(text), {"n": 99}, log=path, learnable=True, num_workers=0)
        assert first.status == "succeeded", first.failure
        assert first.output == _tree_value(100)
        # Cut right after the last expansion so resume replays every call and runs the leaf.
        lines = path.read_text().splitlines(keepends=True)
        expanded = [
            i for i, line in enumerate(lines) if json.loads(line)["kind"] == "node_expanded"
        ]
        assert len(expanded) == 10
        path.write_text("".join(lines[: expanded[-1] + 1]))

        resumed = resume_flow(load_flow(text), log=path)

    assert resumed.status == "succeeded", resumed.failure
    assert resumed.output == _tree_value(100)


def test_a_deep_record_at_the_call_nesting_limit_runs() -> None:
    flow = load_flow(_flow(100, 50))

    with _stack_headroom(650):
        result = run_flow(flow, {"n": 99}, num_workers=0)

    assert result.status == "succeeded", result.failure
    assert result.output == _tree_value(100)


def test_a_union_nested_at_every_level_runs() -> None:
    typedefs = ["typedefs:", "  S: {k: Literal(s)}", "  T0: {k: Literal(r), a: int}"]
    typedefs += [
        f"  T{i}: {{k: Literal(r), a: 'Optional(Union(T{i - 1}, S))'}}" for i in range(1, 100)
    ]
    text = "\n".join([
        "id: deep",
        "name: deep",
        "input:",
        "  n: !int",
        "nodes:",
        "  u:",
        "    kind: code",
        "    input:",
        "      n: ${input.n}",
        "    output: T99",
        "    code: |",
        "      v = {'k': 's'}",
        "      for _ in range(inputs['n']):",
        "          v = {'k': 'r', 'a': v}",
        "      return v",
        "output: ${u.output}",
        *typedefs,
    ]) + "\n"
    expected: dict = {"k": "s"}
    for _ in range(99):
        expected = {"k": "r", "a": expected}
    flow = load_flow(text)

    with _stack_headroom(650):
        result = run_flow(flow, {"n": 99}, num_workers=0)

    assert result.status == "succeeded", result.failure
    assert result.output == expected


def test_a_record_past_the_value_limit_fails_its_output_contract(set_defaults) -> None:
    set_defaults(MAX_TYPE_NESTING=125)
    flow = load_flow(_flow(125, 10))

    with _stack_headroom(400):
        result = run_flow(flow, {"n": 124}, num_workers=0)

    assert result.status == "failed"
    assert result.failure.code == "contract.output"
    assert "nests deeper than" in result.failure.message

