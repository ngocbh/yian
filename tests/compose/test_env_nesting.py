"""An `env:` value nested past 100 levels is a located load error.

Each `env:` value is checked at load with the counting the run-time check applies to a
value, whether YAML text or aliases built it, so a 1000-link alias chain fails at load
instead of raising `RecursionError` when the run walks it. A value at the limit loads
and runs.

Message prefix per site, each at the `env:` entry's value:

- flow `env:`: `flow env '<key>'`.
- a multi-node `defs:` entry's `env:`: `defs entry '<def>': env '<key>'`.
- a node's `env:`: `node '<id>' env '<key>'`; a single-node `defs:` entry's reads
  `defs entry '<def>': node '<def>' env '<key>'`, and a compact flow's names the flow id
  as the node.
"""

from __future__ import annotations

from typing import Callable, Iterator

import pytest

from yian.compose import LoadError, load_flow
from yian.defaults import Defaults
from yian.run import run_flow

MAX_VALUE_NESTING = Defaults().max_value_nesting

_LIMIT = f"value nests deeper than {MAX_VALUE_NESTING} levels"

_ECHO = "    kind: code\n    code: tests.engine._compose_codefns:echo\n"

# --------------------------------------------------------------------------- #
# Value forms: each returns `(header, value)`, where `header` holds the top-level
# `x-anchors:` line the value's aliases name (empty for written-out text).
# --------------------------------------------------------------------------- #


def _list(levels: int) -> tuple[str, str]:
    """Return a list written out as text, nesting `levels` levels."""
    return "", "[" * levels + "1" + "]" * levels


def _anchors(count: int, *, record: bool) -> str:
    """Return an `x-anchors:` line whose anchor `a<i>` names a value nesting i+1 levels."""
    if record:
        items = ["&a0 {a: 1}"] + [f"&a{i} {{a: *a{i - 1}}}" for i in range(1, count)]
    else:
        items = ["&a0 [1]"] + [f"&a{i} [*a{i - 1}]" for i in range(1, count)]
    return f"x-anchors: [{', '.join(items)}]\n"


def _list_alias(levels: int) -> tuple[str, str]:
    """Return a one-item list over an alias chain, nesting exactly `levels` levels."""
    return _anchors(levels - 1, record=False), f"[*a{levels - 2}]"


def _record_alias(levels: int) -> tuple[str, str]:
    """Return a one-key mapping over an alias chain, nesting exactly `levels` levels."""
    return _anchors(levels - 1, record=True), f"{{a: *a{levels - 2}}}"


# --------------------------------------------------------------------------- #
# Flows: each puts `value` at `env: {K: value}` on one site.
# --------------------------------------------------------------------------- #


def _flow_env(header: str, value: str) -> str:
    """Return a flow whose flow `env:` sets `K` to `value`."""
    return (
        f"{header}id: f\nname: f\nenv:\n  K: {value}\nnodes:\n  use:\n{_ECHO}"
        "    input:\n      topic: x\noutput: ${use.output}\n"
    )


def _node_env(header: str, value: str) -> str:
    """Return a flow whose node `use` has `env:` setting `K` to `value`."""
    return (
        f"{header}id: f\nname: f\nnodes:\n  use:\n{_ECHO}    env:\n      K: {value}\n"
        "    input:\n      topic: x\noutput: ${use.output}\n"
    )


def _multi_def_env(header: str, value: str) -> str:
    """Return a flow that calls multi-node def `d`, whose `env:` sets `K` to `value`."""
    return (
        f"{header}id: f\nname: f\ndefs:\n  d:\n    env:\n      K: {value}\n"
        "    nodes:\n      keep:\n        kind: code\n"
        "        code: tests.engine._compose_codefns:echo\n"
        "        input:\n          topic: x\n    output: ${keep.output}\n"
        "nodes:\n  use:\n    kind: call\n    call: d\noutput: ${use.output}\n"
    )


def _single_def_env(header: str, value: str) -> str:
    """Return a flow that calls single-node def `d`, whose `env:` sets `K` to `value`."""
    return (
        f"{header}id: f\nname: f\ndefs:\n  d:\n{_ECHO}    env:\n      K: {value}\n"
        "    input:\n      topic: !str\n"
        "nodes:\n  use:\n    kind: call\n    call: d\n    input:\n      topic: x\n"
        "output: ${use.output}\n"
    )


def _compact_env(header: str, value: str) -> str:
    """Return a compact (single-node) flow whose `env:` sets `K` to `value`."""
    return (
        f"{header}id: f\nname: f\nenv:\n  K: {value}\n"
        "kind: code\ncode: tests.engine._compose_codefns:echo\ninput:\n  topic: !str x\n"
    )


_Flow = Callable[[str, str], str]

# (flow builder, message prefix, located line)
_SITES: dict[str, tuple[_Flow, str, str]] = {
    "flow": (_flow_env, "flow env 'K' at ", "  K:"),
    "node": (_node_env, "node 'use' env 'K' at ", "      K:"),
    "multi-node-def": (_multi_def_env, "defs entry 'd': env 'K' at ", "      K:"),
    "single-node-def": (
        _single_def_env, "defs entry 'd': node 'd' env 'K' at ", "      K:",
    ),
    "compact": (_compact_env, "node 'f' env 'K' at ", "  K:"),
}

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _line_of(text: str, prefix: str) -> int:
    """Return the 1-based number of the only line of `text` starting with `prefix`."""
    (number,) = [
        index
        for index, line in enumerate(text.splitlines(), start=1)
        if line.startswith(prefix)
    ]
    return number


def _chain(error: BaseException) -> Iterator[BaseException]:
    """Yield `error` and every exception reachable through `__cause__`/`__context__`."""
    seen: set[int] = set()
    pending: list[BaseException | None] = [error]
    while pending:
        current = pending.pop()
        if current is None or id(current) in seen:
            continue
        seen.add(id(current))
        yield current
        pending.extend([current.__cause__, current.__context__])


def _assert_limit_error(text: str, where: str, line_prefix: str) -> None:
    """Assert loading `text` fails with the limit error for `where` at `line_prefix`."""
    with pytest.raises(LoadError) as caught:
        load_flow(text)
    error = caught.value
    message = error.diagnostic.message
    assert message.startswith(where) and _LIMIT in message, message
    assert error.diagnostic.frames[-1].range.start_line == _line_of(text, line_prefix)
    assert not any(isinstance(link, RecursionError) for link in _chain(error))


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("site", list(_SITES))
@pytest.mark.parametrize("form", [_list_alias, _record_alias], ids=["list", "record"])
def test_env_value_over_a_1000_link_alias_chain_is_a_located_load_error(
    site: str, form: Callable[[int], tuple[str, str]]
) -> None:
    flow, where, line = _SITES[site]

    _assert_limit_error(flow(*form(1000)), where, line)


def test_flow_env_value_past_the_limit_as_text_is_a_located_load_error() -> None:
    _assert_limit_error(_flow_env(*_list(MAX_VALUE_NESTING + 1)), "flow env 'K' at ", "  K:")


def test_env_value_follows_the_configured_limit(set_defaults) -> None:
    set_defaults(MAX_VALUE_NESTING=5)

    load_flow(_flow_env(*_list(5)))
    with pytest.raises(LoadError) as caught:
        load_flow(_flow_env(*_list(6)))
    assert "value nests deeper than 5 levels" in caught.value.diagnostic.message


def test_flow_env_nesting_is_checked_before_max_tool_iterations() -> None:
    header, value = _list_alias(1000)
    text = (
        f"{header}id: f\nname: f\nenv:\n  max_tool_iterations: {value}\nnodes:\n"
        f"  use:\n{_ECHO}    input:\n      topic: x\noutput: ${{use.output}}\n"
    )

    _assert_limit_error(text, "flow env 'max_tool_iterations' at ", "  max_tool_iterations:")


@pytest.mark.parametrize("site", list(_SITES))
@pytest.mark.parametrize("form", [_list, _list_alias], ids=["text", "alias"])
def test_env_value_at_the_limit_loads_and_runs(
    site: str, form: Callable[[int], tuple[str, str]]
) -> None:
    flow, _, _ = _SITES[site]

    result = run_flow(load_flow(flow(*form(MAX_VALUE_NESTING))), {})

    assert result.status == "succeeded", result.failure
