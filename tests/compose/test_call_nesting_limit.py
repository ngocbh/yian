"""A flow's call, map, and loop bodies nest at most 50 levels; a deeper chain is a located load error.

Each body is one level, whether it is a `defs:` entry, an inline body, a `call(...)`
directive, or a `uses:` flow, and whether a call, map, or loop reaches it. A chain at the
limit loads, runs, and resumes. A chain one past it fails at the root call that reaches 51
levels; a much longer chain fails at the first call past the limit, before the loader
recurses any deeper. Imported files load before `defs:`, so a file loads at
its importer's base stack and its own uncalled defs don't count the importing chain.

A flow's chain of `uses:` imports is held to the same limit, counted apart from call depth:
the root importing `u0` is one file deep, and an import that would make the chain 51 files
deep fails at its alias, before the loader loads the file, whether or not anything calls it.
A file already loaded (from the cache or a memoizing resolver) is checked by the import
depth it carries, so the order of a flow's `uses:` entries does not change the result.
"""

import dataclasses
import inspect
import json
import sys
import textwrap
from contextlib import contextmanager
from pathlib import Path

import pytest

from yian.compose import LoadError, load_flow
from yian.compose.helpers.errors import LoadDiagnostics
from yian.compose.nodes.resolver import call_nesting
from yian.compose.source import SourceDocument
from yian.events import NodeStarted
from yian.run import resume_flow, run_flow

_PAST_51 = "nests calls 51 levels deep, past the call nesting limit of 50"
_PAST_DEEP = "more than 50 levels deep, past the call nesting limit of 50"
_IMPORT_51 = "makes a chain of uses: files 51 deep, past the call nesting limit of 50"
_LEAF = (
    "    input:\n"
    "      x: !str\n"
    "    nodes:\n"
    "      step:\n"
    "        kind: code\n"
    "        code: tests.engine._compose_codefns:echo\n"
    "        input:\n"
    "          topic: ${input.x}\n"
    "        output: str\n"
    "    output: ${step.output}\n"
)
_ROOT_LEAF = (
    "  step:\n"
    "    kind: code\n"
    "    code: tests.engine._compose_codefns:echo\n"
    "    input:\n"
    "      topic: ${input.x}\n"
    "    output: str\n"
    "output: ${step.output}\n"
)


def _calling(target: str) -> str:
    """Return a def body whose one `step` node calls `target` with `x`."""
    return (
        "    input:\n"
        "      x: !str\n"
        "    nodes:\n"
        "      step:\n"
        "        kind: call\n"
        f"        call: {target}\n"
        "        input:\n"
        "          x: ${input.x}\n"
        "    output: ${step.output}\n"
    )


def _defs_chain(
    n: int,
    *,
    reverse: bool = False,
    prefix: str = "c",
    last_calls: str | None = None,
    uses: dict[str, str] | None = None,
) -> str:
    """Return a flow whose root `top` calls `c0` and each `ci` calls `c(i+1)`.

    The last def is a CODE leaf echoing `x`, or calls `last_calls` when given.
    `reverse` writes the defs last-first.
    """
    entries = []
    for i in range(n):
        if i < n - 1:
            body = _calling(f"{prefix}{i + 1}")
        elif last_calls is not None:
            body = _calling(last_calls)
        else:
            body = _LEAF
        entries.append(f"  {prefix}{i}:\n{body}")
    if reverse:
        entries.reverse()
    head = f"id: chain{n}\nname: chain{n}\ninput:\n  x: !str\n"
    if uses:
        head += "uses:\n" + "".join(f"  {alias}: {ref}\n" for alias, ref in uses.items())
    return (
        head
        + "defs:\n"
        + "".join(entries)
        + "nodes:\n"
        "  top:\n"
        "    kind: call\n"
        f"    call: {prefix}0\n"
        "    input:\n"
        "      x: ${input.x}\n"
        "output: ${top.output}\n"
    )


def _inline_chain(n: int) -> str:
    """Return a flow whose root `top` nests `n` inline call bodies around a CODE leaf."""
    body = (
        "kind: code\n"
        "code: tests.engine._compose_codefns:echo\n"
        "input:\n"
        "  topic: ${input.x}\n"
        "output: str\n"
    )
    for _ in range(n):
        body = (
            "kind: call\n"
            "input:\n"
            "  x: ${input.x}\n"
            "nodes:\n"
            "  step:\n" + textwrap.indent(body, "    ") + "output: ${step.output}\n"
        )
    return (
        f"id: inline{n}\nname: inline{n}\ninput:\n  x: !str\nnodes:\n  top:\n"
        + textwrap.indent(body, "    ")
        + "output: ${top.output}\n"
    )


def _mixed_chain(n: int) -> str:
    """Return a flow whose defs reach one another by CALL, MAP, and LOOP in turn.

    Root `top` calls `c0`, each `ci` reaches `c(i+1)`, and the last def is a CODE leaf. A LOOP body returns the record it carries, so each def right under a LOOP outputs
    `{v: ...}`.
    """
    entries = []
    for i in range(n):
        under_loop = i > 0 and i % 3 == 0  # the CALL levels sit right under a LOOP
        if i == n - 1:
            node = (
                "        kind: code\n"
                "        code: tests.engine._compose_codefns:echo\n"
                "        input:\n"
                "          topic: ${input.v}\n"
                "        output: str\n"
            )
            result = "${step.output}"
        else:
            callee = f"c{i + 1}"
            node = {
                0: f"        kind: call\n        call: {callee}\n"
                "        input:\n          v: ${input.v}\n",
                1: f"        kind: map\n        over: ${{[input.v]}}\n        call: {callee}\n"
                "        input:\n          v: ${item}\n",
                2: f"        kind: loop\n        times: 1\n        call: {callee}\n"
                "        input:\n          v: ${input.v}\n",
            }[i % 3]
            result = "${input.v}" if under_loop else "${step.output}"
        record_output = f"\n      v: {result}\n"
        value_output = f" {result}\n"
        output = record_output if under_loop else value_output
        entries.append(
            f"  c{i}:\n    input:\n      v: !str\n    nodes:\n      step:\n{node}"
            f"    output:{output}"
        )
    return (
        f"id: mixed{n}\nname: mixed{n}\ninput:\n  v: !str\ndefs:\n"
        + "".join(entries)
        + "nodes:\n"
        "  top:\n"
        "    kind: call\n"
        "    call: c0\n"
        "    input:\n"
        "      v: ${input.v}\n"
        "output: ${top.output}\n"
    )


def _directive_chain(n: int) -> str:
    """Return a flow whose `call(...)` directives chain through every def.

    The root `step` echoes `call(c0, x=${input.x})`, each `ci` echoes `call(c(i+1), ...)`,
    and the last def is a CODE leaf.
    """

    def echo_call(target: str, indent: str) -> str:
        return (
            f"{indent}step:\n"
            f"{indent}  kind: code\n"
            f"{indent}  code: tests.engine._compose_codefns:echo\n"
            f"{indent}  input:\n"
            f"{indent}    topic: call({target}, x=${{input.x}})\n"
            f"{indent}  output: str\n"
        )

    entries = []
    for i in range(n):
        if i < n - 1:
            body = (
                "    input:\n      x: !str\n    nodes:\n"
                + echo_call(f"c{i + 1}", "      ")
                + "    output: ${step.output}\n"
            )
        else:
            body = _LEAF
        entries.append(f"  c{i}:\n{body}")
    return (
        f"id: directive{n}\nname: directive{n}\ninput:\n  x: !str\ndefs:\n"
        + "".join(entries)
        + "nodes:\n"
        + echo_call("c0", "  ")
        + "output: ${step.output}\n"
    )


def _importing(name: str, uses: dict[str, str]) -> str:
    """Return a flow `name` that imports `uses` without calling them; its node is a CODE leaf."""
    head = f"id: {name}\nname: {name}\ninput:\n  x: !str\n"
    if uses:
        head += "uses:\n" + "".join(f"  {alias}: {ref}\n" for alias, ref in uses.items())
    return head + "nodes:\n" + _ROOT_LEAF


def _chain_texts(
    n: int, *, prefix: str = "u", calls: bool = True, last_imports: str | None = None
) -> dict[str, str]:
    """Return flows `u0`..`u(n-1)` by name, each importing the next as `nxt`.

    With `calls`, each file but the last calls `nxt`; otherwise every node is a CODE
    leaf. The last file is a CODE leaf that imports `last_imports` as `nxt` when given.
    """
    texts = {}
    for i in range(n):
        name = f"{prefix}{i}"
        target = f"{prefix}{i + 1}" if i < n - 1 else last_imports
        if calls and i < n - 1:
            texts[name] = (
                f"id: {name}\nname: {name}\ninput:\n  x: !str\n"
                f"uses:\n  nxt: {target}\nnodes:\n"
                "  step:\n"
                "    kind: call\n"
                "    call: nxt\n"
                "    input:\n"
                "      x: ${input.x}\n"
                "output: ${step.output}\n"
            )
        else:
            texts[name] = _importing(name, {"nxt": target} if target else {})
    return texts


def _write_files(directory: Path, texts: dict[str, str]) -> None:
    """Write each flow text to `<directory>/<name>.yaml`."""
    for name, text in texts.items():
        (directory / f"{name}.yaml").write_text(text)


def _uses_files(directory: Path, n: int, *, calls: bool = True) -> str:
    """Write a chain of flow files and return a root flow that imports the first.

    Files `u0`..`u(n-1)` each import and call the next, the last is a CODE leaf, and the
    root imports `u0` as `first`. Without `calls`, the files and root only import.
    """
    _write_files(directory, _chain_texts(n, calls=calls))
    if not calls:
        return _importing("root", {"first": "u0"})
    return (
        "id: root\nname: root\ninput:\n  x: !str\nuses:\n  first: u0\nnodes:\n"
        "  top:\n"
        "    kind: call\n"
        "    call: first\n"
        "    input:\n"
        "      x: ${input.x}\n"
        "output: ${top.output}\n"
    )


def _import_message(n: int, limit: int = 50) -> str:
    """Return the error for `u(n-2)`'s import of `u(n-1)`, making the chain `n` files deep."""
    return (
        f"callable 'nxt' (uses: 'u{n - 1}'): importing it makes a chain of uses: files "
        f"{n} deep, past the call nesting limit of {limit} (a chain of uses: imports is "
        "held to the same limit as nested call, map, and loop bodies); flatten the chain"
    )


def _assert_refused_at_alias(error: LoadError, label: str, text: str, target: str) -> None:
    """Assert a load error's innermost frame is file `label`'s `nxt: <target>` alias."""
    frame = error.diagnostic.frames[-1]
    assert frame.label == label
    assert frame.note == "uses alias 'nxt'"
    alias = SourceDocument.parse(text).root.child("uses").child("nxt")
    assert frame.range == alias.range
    assert frame.range.start_line == _line_of(text, f"nxt: {target}")


def _line_of(text: str, line: str) -> int:
    """Return the 1-based number of the first line whose stripped text is `line`."""
    return next(
        number
        for number, authored in enumerate(text.splitlines(), start=1)
        if authored.strip() == line
    )


def _error_line(error: LoadError) -> int:
    """Return the start line of a load error's innermost frame."""
    frame = error.diagnostic.frames[-1]
    assert frame.range is not None
    return frame.range.start_line


def _causes(error: BaseException) -> list[BaseException]:
    """Return `error` and every exception in its cause and context chain."""
    found: list[BaseException] = []
    current: BaseException | None = error
    while current is not None and current not in found:
        found.append(current)
        current = current.__cause__ or current.__context__
    return found


def _assert_no_recursion_error(error: LoadError) -> None:
    """Assert no `RecursionError` is in a load error's cause chain."""
    assert not any(isinstance(exc, RecursionError) for exc in _causes(error))


@contextmanager
def _stack_headroom(frames: int = 650):
    """Pin the recursion limit to `frames` past the current stack depth, restoring it after.

    This usually lowers the limit, so an at-limit chain is shown to fit in that headroom.
    """
    old = sys.getrecursionlimit()
    sys.setrecursionlimit(len(inspect.stack(0)) + frames)
    try:
        yield
    finally:
        sys.setrecursionlimit(old)


def test_chain_at_the_limit_loads_runs_and_resumes(tmp_path: Path) -> None:
    text = _defs_chain(50)
    path = tmp_path / "chain.jsonl"
    leaf = "/".join(["top", *["step"] * 50])

    with _stack_headroom():
        first = run_flow(load_flow(text), {"x": "hi"}, log=path, num_workers=0)
        assert first.status == "succeeded", first.failure
        assert first.output == "hi"
        # Cut right after the last expansion so resume replays all 50 and runs the leaf.
        lines = path.read_text().splitlines(keepends=True)
        expanded = [
            i for i, line in enumerate(lines) if json.loads(line)["kind"] == "node_expanded"
        ]
        assert len(expanded) == 50
        path.write_text("".join(lines[: expanded[-1] + 1]))

        resumed = resume_flow(load_flow(text), log=path)

    assert resumed.status == "succeeded", resumed.failure
    assert resumed.output == "hi"
    assert leaf in {e.node_id for e in resumed.events if isinstance(e, NodeStarted)}


def test_chain_past_the_limit_is_located_at_the_root_call() -> None:
    text = _defs_chain(51)

    with pytest.raises(LoadError) as caught:
        load_flow(text)

    assert _PAST_51 in str(caught.value)
    assert "call node 'top': calling callable 'c0'" in str(caught.value)
    _assert_no_recursion_error(caught.value)
    assert len(caught.value.diagnostic.frames) == 1
    assert _error_line(caught.value) == _line_of(text, "call: c0")


@pytest.mark.parametrize("n", [97, 1000])
def test_long_chain_fails_at_the_first_call_past_the_limit(n: int) -> None:
    text = _defs_chain(n)

    with pytest.raises(LoadError) as caught:
        load_flow(text)

    assert _PAST_DEEP in str(caught.value)
    assert "call node 'step': calling callable 'c51'" in str(caught.value)
    _assert_no_recursion_error(caught.value)
    assert _error_line(caught.value) == _line_of(text, "call: c51")


def test_chain_written_callee_first_is_caught_from_built_callables() -> None:
    text = _defs_chain(51, reverse=True)

    with pytest.raises(LoadError) as caught:
        load_flow(text)

    assert _PAST_51 in str(caught.value)
    assert _error_line(caught.value) == _line_of(text, "call: c0")
    load_flow(_defs_chain(50, reverse=True))


class _FakeNode:
    """A node whose only static child is `child`, counting how often it is asked."""

    def __init__(self, child: object) -> None:
        self.child = child
        self.calls = 0

    def origin_children(self) -> tuple[object, ...]:
        self.calls += 1
        return (self.child,) if self.child is not None else ()


class _FakeFlow:
    """A flow with the given nodes keyed by position."""

    def __init__(self, nodes: list[_FakeNode]) -> None:
        self.nodes = {f"n{i}": node for i, node in enumerate(nodes)}


def test_call_nesting_counts_levels_not_paths() -> None:
    # Each flow's two nodes share the next flow, so there are 2**40 paths but 40 levels.
    flow = _FakeFlow([])
    nodes: list[_FakeNode] = []
    for _ in range(40):
        pair = [_FakeNode(flow), _FakeNode(flow)]
        nodes.extend(pair)
        flow = _FakeFlow(pair)

    assert call_nesting(flow) == 40
    assert sum(node.calls for node in nodes) == 80
    assert all(node.calls == 1 for node in nodes)


def test_imports_load_before_defs(tmp_path: Path) -> None:
    # A file at the limit called from 49 levels down is caught when the call returns.
    deep = tmp_path / "deep"
    deep.mkdir()
    (deep / "lib.yaml").write_text(_defs_chain(50, prefix="k"))
    text = _defs_chain(49, last_calls="lib", uses={"lib": "lib"})
    with _stack_headroom(), pytest.raises(LoadError) as caught:
        load_flow(text, search_paths=[deep])
    assert "nests calls 99 levels deep, past the call nesting limit of 50" in str(
        caught.value
    )
    assert _error_line(caught.value) == _line_of(text, "call: lib")

    # A one-level file's uncalled defs nest 50 on their own, not under the importer.
    legal = tmp_path / "legal"
    legal.mkdir()
    lib = _defs_chain(51, prefix="k").split("nodes:\n  top:\n")[0] + "nodes:\n" + _ROOT_LEAF
    (legal / "lib.yaml").write_text(lib)
    with _stack_headroom():
        load_flow(
            _defs_chain(49, last_calls="lib", uses={"lib": "lib"}), search_paths=[legal]
        )


@pytest.mark.parametrize(
    "definition",
    [
        """    nodes:
      invoke:
        kind: call
        call: external
    output: ${invoke.output}
""",
        """    kind: call
    call: external
""",
    ],
)
def test_broken_import_is_reported_at_its_alias_before_defs(definition: str) -> None:
    text = f"""id: def-inner-line
name: def_inner_line
uses:
  external: child.yaml
defs:
  local:
{definition}nodes:
  invoke:
    kind: call
    call: local
output: ${{invoke.output}}
"""

    def fail_resolver(_path: str, _version: str | None = None):
        raise LoadDiagnostics().error("external failure")

    with pytest.raises(LoadError) as caught:
        load_flow(text, child_resolver=fail_resolver)

    assert str(caught.value) == "external failure"
    frames = caught.value.diagnostic.frames
    assert len(frames) == 1
    assert frames[0].note == "uses alias 'external'"
    assert frames[0].range is not None
    assert frames[0].range.start_line == _line_of(text, "external: child.yaml")


def test_alias_collision_wins_over_a_broken_target() -> None:
    text = (
        "id: f\nname: f\ninput:\n  x: !str\nuses:\n  step: hub:x/y@1\nnodes:\n" + _ROOT_LEAF
    )

    with pytest.raises(LoadError, match=r"uses: alias \['step'\] collides with a node id"):
        load_flow(text)


@pytest.mark.parametrize(
    "text",
    [
        # A def calls the alias.
        "id: f\nname: f\ninput:\n  x: !str\nuses:\n  ext: child\n"
        "defs:\n  d:\n" + _calling("ext") + "nodes:\n"
        "  top:\n    kind: call\n    call: d\n    input:\n      x: ${input.x}\n"
        "output: ${top.output}\n",
        # The root calls the alias.
        "id: f\nname: f\ninput:\n  x: !str\nuses:\n  ext: child\nnodes:\n"
        "  top:\n    kind: call\n    call: ext\n    input:\n      x: ${input.x}\n"
        "output: ${top.output}\n",
        # Nothing calls the alias.
        "id: f\nname: f\ninput:\n  x: !str\nuses:\n  ext: child\nnodes:\n" + _ROOT_LEAF,
    ],
    ids=["def-calls", "root-calls", "uncalled"],
)
def test_resolver_exceptions_become_load_errors_at_the_alias(text: str) -> None:
    def raising_resolver(_path: str, _version: str | None = None):
        raise KeyError("boom")

    with pytest.raises(LoadError) as caught:
        load_flow(text, child_resolver=raising_resolver)

    assert str(caught.value) == "callable 'ext' (uses: 'child'): could not resolve: 'boom'"
    frames = caught.value.diagnostic.frames
    assert len(frames) == 1
    assert frames[0].note == "uses alias 'ext'"
    assert frames[0].range is not None
    assert frames[0].range.start_line == _line_of(text, "ext: child")


def test_inline_bodies_count() -> None:
    with _stack_headroom():
        result = run_flow(load_flow(_inline_chain(50)), {"x": "hi"}, num_workers=0)
    assert result.status == "succeeded", result.failure
    assert result.output == "hi"

    text = _inline_chain(51)
    with pytest.raises(LoadError) as caught:
        load_flow(text)

    assert _PAST_DEEP in str(caught.value)
    _assert_no_recursion_error(caught.value)
    # An inline body is the callable its node calls, so the error sits on the innermost
    # call node's `nodes:` body.
    innermost = SourceDocument.parse(text).root.child("nodes").child("top")
    for _ in range(50):
        innermost = innermost.child("nodes").child("step")
    assert caught.value.diagnostic.frames[-1].range == innermost.range_of("nodes")


def test_map_and_loop_bodies_count() -> None:
    with _stack_headroom():
        result = run_flow(load_flow(_mixed_chain(50)), {"v": "hi"}, num_workers=0)
    assert result.status == "succeeded", result.failure
    assert result.output == [{"v": "hi"}]
    # The CODE leaf, 50 bodies down, ran.
    assert any(
        e.node_id.count("/") == 50 and e.node_id.endswith("/step")
        for e in result.events
        if isinstance(e, NodeStarted)
    )

    text = _mixed_chain(51)
    with pytest.raises(LoadError) as caught:
        load_flow(text)

    assert _PAST_51 in str(caught.value)
    _assert_no_recursion_error(caught.value)
    assert _error_line(caught.value) == _line_of(text, "call: c0")


def test_call_directive_counts() -> None:
    with _stack_headroom():
        result = run_flow(load_flow(_directive_chain(50)), {"x": "hi"}, num_workers=0)
    assert result.status == "succeeded", result.failure
    assert result.output == "hi"

    with pytest.raises(LoadError, match="past the call nesting limit of 50") as caught:
        load_flow(_directive_chain(51))
    _assert_no_recursion_error(caught.value)


def test_uses_files_count(tmp_path: Path) -> None:
    legal = tmp_path / "legal"
    legal.mkdir()
    with _stack_headroom():
        flow = load_flow(_uses_files(legal, 50), search_paths=[legal])
        result = run_flow(flow, {"x": "hi"}, num_workers=0)
    assert result.status == "succeeded", result.failure
    assert result.output == "hi"

    deep = tmp_path / "deep"
    deep.mkdir()
    text = _uses_files(deep, 51)
    with pytest.raises(LoadError) as caught:
        load_flow(text, search_paths=[deep])

    # The import is refused before the call check runs on the built chain.
    assert str(caught.value) == _import_message(51)
    _assert_no_recursion_error(caught.value)
    _assert_refused_at_alias(caught.value, "u49.yaml", (deep / "u49.yaml").read_text(), "u50")


def test_long_uses_chain_fails_at_the_import_past_the_limit(tmp_path: Path) -> None:
    text = _uses_files(tmp_path, 200)

    with pytest.raises(LoadError) as caught:
        load_flow(text, search_paths=[tmp_path])

    assert str(caught.value) == _import_message(51)
    assert _IMPORT_51 in str(caught.value)
    assert "maximum recursion depth" not in str(caught.value)
    _assert_no_recursion_error(caught.value)
    frames = caught.value.diagnostic.frames
    assert len(frames) == 51
    _assert_refused_at_alias(caught.value, "u49.yaml", (tmp_path / "u49.yaml").read_text(), "u50")
    assert frames[0].label == "<flow>"
    assert frames[0].range is not None
    assert frames[0].range.start_line == _line_of(text, "first: u0")


def test_import_only_chain_counts(tmp_path: Path) -> None:
    legal = tmp_path / "legal"
    legal.mkdir()
    with _stack_headroom():
        load_flow(_uses_files(legal, 50, calls=False), search_paths=[legal])

    deep = tmp_path / "deep"
    deep.mkdir()
    text = _uses_files(deep, 60, calls=False)
    with pytest.raises(LoadError) as caught:
        load_flow(text, search_paths=[deep])

    assert str(caught.value) == _import_message(51)
    _assert_no_recursion_error(caught.value)
    _assert_refused_at_alias(caught.value, "u49.yaml", (deep / "u49.yaml").read_text(), "u50")


def _shared_file_texts(under_x: int) -> dict[str, str]:
    """Return a file `x` over an `under_x`-file chain, and a 40-file chain ending in `x`."""
    texts = _chain_texts(under_x, prefix="c", calls=False)
    texts["x"] = _importing("x", {"nxt": "c0"})
    texts.update(_chain_texts(40, prefix="a", calls=False, last_imports="x"))
    return texts


_ORDERS = [{"x": "x", "a": "a0"}, {"a": "a0", "x": "x"}]


@pytest.mark.parametrize("uses", _ORDERS, ids=["x-first", "chain-first"])
def test_import_depth_does_not_depend_on_load_order(tmp_path: Path, uses) -> None:
    # Through the 40-file chain, x's 10-file chain ends 51 files deep; loading x first
    # caches it, so the chain then reaches x from the cache.
    deep = tmp_path / "deep"
    deep.mkdir()
    _write_files(deep, _shared_file_texts(10))
    with pytest.raises(LoadError) as caught:
        load_flow(_importing("root", uses), search_paths=[deep])
    assert _IMPORT_51 in str(caught.value)
    _assert_no_recursion_error(caught.value)

    legal = tmp_path / "legal"
    legal.mkdir()
    _write_files(legal, _shared_file_texts(9))
    with _stack_headroom():
        load_flow(_importing("root", uses), search_paths=[legal])


def _memoizing_resolver(texts: dict[str, str]):
    """Return a resolver loading each named text once, through itself, and reusing it."""
    memo = {}

    def resolve(path: str, version: str | None = None):
        if path not in memo:
            memo[path] = load_flow(
                texts[path], child_resolver=resolve, source_label=f"{path}.yaml"
            )
        return memo[path]

    return resolve


@pytest.mark.parametrize("uses", _ORDERS, ids=["x-first", "chain-first"])
def test_import_depth_holds_through_a_memoizing_resolver(uses) -> None:
    root = _importing("root", uses)
    with pytest.raises(LoadError) as caught:
        load_flow(root, child_resolver=_memoizing_resolver(_shared_file_texts(10)))
    assert _IMPORT_51 in str(caught.value)
    _assert_no_recursion_error(caught.value)

    with _stack_headroom(frames=800):
        load_flow(root, child_resolver=_memoizing_resolver(_shared_file_texts(9)))


def test_resolver_that_calls_load_flow_is_bounded() -> None:
    texts = _chain_texts(200)

    def resolve(path: str, version: str | None = None):
        return load_flow(texts[path], child_resolver=resolve, source_label=f"{path}.yaml")

    with pytest.raises(LoadError) as caught:
        load_flow(_importing("root", {"first": "u0"}), child_resolver=resolve)

    assert str(caught.value) == _import_message(51)
    _assert_no_recursion_error(caught.value)
    _assert_refused_at_alias(caught.value, "u49.yaml", texts["u49"], "u50")


def test_import_chain_follows_the_configured_limit(tmp_path: Path, set_defaults) -> None:
    set_defaults(MAX_CALL_NESTING=5)
    legal = tmp_path / "legal"
    legal.mkdir()
    load_flow(_uses_files(legal, 5, calls=False), search_paths=[legal])

    deep = tmp_path / "deep"
    deep.mkdir()
    text = _uses_files(deep, 6, calls=False)
    with pytest.raises(LoadError) as caught:
        load_flow(text, search_paths=[deep])

    assert str(caught.value) == _import_message(6, limit=5)
    _assert_refused_at_alias(caught.value, "u4.yaml", (deep / "u4.yaml").read_text(), "u5")


@pytest.mark.parametrize("depth", [-1, "1", True, 1.0, None])
def test_resolved_flow_with_a_bad_import_depth_is_refused(depth) -> None:
    text = _importing("f", {"ext": "child"})
    leaf = load_flow(_importing("child", {}))

    def resolve(_path: str, _version: str | None = None):
        return dataclasses.replace(leaf, import_depth=depth)

    with pytest.raises(LoadError) as caught:
        load_flow(text, child_resolver=resolve)

    assert str(caught.value) == (
        f"callable 'ext' (uses: 'child'): resolved flow has import_depth {depth!r}; "
        "expected a non-negative int"
    )
    frames = caught.value.diagnostic.frames
    assert len(frames) == 1
    assert frames[0].note == "uses alias 'ext'"
    assert frames[0].range is not None
    assert frames[0].range.start_line == _line_of(text, "ext: child")


def test_loaded_flow_records_its_import_depth(tmp_path: Path) -> None:
    flow = load_flow(_uses_files(tmp_path, 3, calls=False), search_paths=[tmp_path])
    assert flow.import_depth == 3
    assert load_flow(_importing("f", {})).import_depth == 0


def test_imports_and_a_calling_defs_chain_count_apart(tmp_path: Path) -> None:
    # The import chain and the call chain in its last file are each 30 deep, 60 together.
    texts = _chain_texts(30, calls=False)
    texts["u29"] = _defs_chain(30)
    _write_files(tmp_path, texts)
    with _stack_headroom():
        load_flow(_importing("root", {"first": "u0"}), search_paths=[tmp_path])


_UNCALLED_DEFS = (
    _defs_chain(51, prefix="k").split("nodes:\n  top:\n")[0] + "nodes:\n" + _ROOT_LEAF
)


# A last file that calls its chain is the deepest accepted load, since importing it
# rebases a compiled graph 50 levels deep.
@pytest.mark.parametrize("last", [_defs_chain(50), _UNCALLED_DEFS], ids=["called", "uncalled"])
def test_imports_at_the_limit_over_defs_at_the_limit_load(tmp_path: Path, last: str) -> None:
    texts = _chain_texts(50, calls=False)
    texts["u49"] = last
    _write_files(tmp_path, texts)
    with _stack_headroom(frames=800):
        load_flow(_importing("root", {"first": "u0"}), search_paths=[tmp_path])


def test_long_cross_file_cycle_reports_the_depth_limit(tmp_path: Path) -> None:
    # A cycle longer than the limit is 51 files deep before it closes, so the depth
    # error, not the cycle error, is what the author sees.
    _write_files(tmp_path, _chain_texts(60, calls=False, last_imports="u0"))
    with pytest.raises(LoadError) as caught:
        load_flow(_importing("root", {"first": "u0"}), search_paths=[tmp_path])

    assert str(caught.value) == _import_message(51)
    _assert_no_recursion_error(caught.value)
