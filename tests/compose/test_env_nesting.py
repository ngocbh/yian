"""An `env:` value nested past 100 levels is a located load error.

Each `env:` value is checked at load with the counting the run-time check applies to a
value, whether YAML text or aliases built it, so a 1000-link alias chain fails at load
instead of raising `RecursionError` when the run walks it (under a raised
`YIAN_MAX_YAML_NODES`: at the default, the chain's anchors exceed the node limit first).
A value at the limit loads and runs.

An `env:` value that holds itself through a recursive YAML alias (`K: &a [*a]`) is a
located load error too, rather than a `ValueError` from the run's fingerprint: the
message reads `<prefix>env '<key>' at <path>: cannot descend through a recursive alias
occurrence`, where `<path>` names the first entry that refers back to a container
enclosing it. Lists, records, and the lists of pairs `!!pairs` builds all count. The
depth check runs first, so a cyclic value nested past the limit gets the depth message.
A value that repeats an alias without holding itself loads and runs.

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


# A listed chain's anchors cost quadratically many nodes: a 1000-link chain holds over
# 500,000, past the default YIAN_MAX_YAML_NODES. These tests exercise the nesting checks
# behind the node limit, so they raise it; test_yaml_node_limit.py covers the default.
@pytest.fixture(autouse=True)
def _raised_yaml_node_limit(set_defaults) -> None:
    """Raise the YAML node limit so a 1000-link alias chain reaches the nesting checks."""
    set_defaults(MAX_YAML_NODES=10_000_000)


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


# --------------------------------------------------------------------------- #
# Recursive aliases
# --------------------------------------------------------------------------- #

_CYCLE = "cannot descend through a recursive alias occurrence"


def _assert_cycle_error(text: str, message_start: str, line_prefix: str) -> None:
    """Assert loading `text` fails with the cycle error starting `message_start`."""
    with pytest.raises(LoadError) as caught:
        load_flow(text)
    error = caught.value
    message = error.diagnostic.message
    assert message == f"{message_start}: {_CYCLE}", message
    assert error.diagnostic.frames[-1].range.start_line == _line_of(text, line_prefix)
    assert not any(
        isinstance(link, (RecursionError, ValueError)) and not isinstance(link, LoadError)
        for link in _chain(error)
    )


@pytest.mark.parametrize("site", list(_SITES))
def test_env_value_holding_itself_is_a_located_load_error(site: str) -> None:
    flow, where, line = _SITES[site]

    _assert_cycle_error(flow("", "&a [*a]"), f"{where}[0]", line)


@pytest.mark.parametrize(
    ("value", "path"),
    [
        ("&a {b: *a}", "b"),
        ("[1, &a [2, *a]]", "[1][1]"),
        ("&a [&b [*a]]", "[0][0]"),
        ("&a !!pairs [x: *a]", "[0][1]"),
        ("&a {b: 1, <<: {c: *a}}", "c"),
    ],
    ids=["record", "inner-list", "outer-alias", "pairs", "merge-key"],
)
def test_flow_env_cycle_names_the_entry_that_refers_back(value: str, path: str) -> None:
    _assert_cycle_error(_flow_env("", value), f"flow env 'K' at {path}", "  K:")


def _flow_env_inline(env: str) -> str:
    """Return a flow whose flow `env:` is the one-line mapping `env`."""
    return (
        f"id: f\nname: f\nenv: {env}\nnodes:\n  use:\n{_ECHO}"
        "    input:\n      topic: x\noutput: ${use.output}\n"
    )


def _node_env_inline(env: str) -> str:
    """Return a flow whose node `use` has the one-line `env:` mapping `env`."""
    return (
        f"id: f\nname: f\nnodes:\n  use:\n{_ECHO}    env: {env}\n"
        "    input:\n      topic: x\noutput: ${use.output}\n"
    )


@pytest.mark.parametrize(
    ("flow", "where", "line"),
    [
        (_flow_env_inline, "flow env 'K'", "env:"),
        (_node_env_inline, "node 'use' env 'K'", "    env:"),
    ],
    ids=["flow", "node"],
)
@pytest.mark.parametrize(
    ("env", "path"), [("&e {K: *e}", "K"), ("&e {K: [*e]}", "[0].K")], ids=["direct", "list"]
)
def test_env_mapping_holding_itself_is_a_located_load_error(
    flow: _Flow, where: str, line: str, env: str, path: str
) -> None:
    _assert_cycle_error(flow(env), f"{where} at {path}", line)


def _cyclic_list_alias(levels: int) -> tuple[str, str]:
    """Return a one-item list over an alias chain ending in a self-alias, `levels` deep."""
    items = ["&a0 [*a0]"] + [f"&a{i} [*a{i - 1}]" for i in range(1, levels - 1)]
    return f"x-anchors: [{', '.join(items)}]\n", f"[*a{levels - 2}]"


def test_env_nesting_is_checked_before_the_cycle() -> None:
    _assert_limit_error(
        _flow_env(*_cyclic_list_alias(MAX_VALUE_NESTING + 1)), "flow env 'K' at ", "  K:"
    )
    with pytest.raises(LoadError) as caught:
        load_flow(_flow_env(*_cyclic_list_alias(MAX_VALUE_NESTING)))
    message = caught.value.diagnostic.message
    assert message.startswith("flow env 'K' at [0]") and message.endswith(_CYCLE), message


def test_env_value_repeating_an_alias_loads_and_runs() -> None:
    text = "x-anchors: [&s [1, 2]]\n" + _flow_env("", "[*s, *s]")

    result = run_flow(load_flow(text), {})

    assert result.status == "succeeded", result.failure


def test_env_scalar_value_loads() -> None:
    load_flow(_flow_env("", "1"))
