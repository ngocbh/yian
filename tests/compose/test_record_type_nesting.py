"""A record type in `output:` counts its levels against the type nesting limit at load.

A record written as a mapping, or as a list of single-key maps, in a node's `output:`
is one level, and so is every record inside it. A field's type text adds the levels of
its resolved type: each record and list is a level, a union counts as its deepest
member, a scalar adds none, and a typedef name counts as deep as its definition. The
total must stay within `MAX_TYPE_NESTING` (100 by default), however YAML text or aliases
built the record, so a 1000-link alias chain is a located load error instead of a
`RecursionError`. A bare type text or typedef name at the top of `output:` keeps the
type parser's own checks.

The message reads `type at <path> nests deeper than <limit> levels, the limit for a type
(each record and list is a level)`, located at the record or field that crosses the
limit; when a field's type pushes it over, it ends `; its type '<text>' adds <d>`. A
CASE's `output:` is rejected as a CASE `output:` whatever its depth.
"""

from __future__ import annotations

from typing import Callable, Iterator

import pytest

from yian.compose import LoadError, load_flow
from yian.defaults import Defaults
from yian.run import run_flow

MAX_TYPE_NESTING = Defaults().max_type_nesting

_LIMIT = f"nests deeper than {MAX_TYPE_NESTING} levels, the limit for a type"

_ECHO = "    kind: code\n    code: tests.engine._compose_codefns:echo\n"

# A typedef 99 lists deep.
_T99 = "list(" * 99 + "int" + ")" * 99

# --------------------------------------------------------------------------- #
# Record forms: each returns `(header, value)`, where `header` holds the top-level
# `x-anchors:` line the value's aliases name (empty for written-out text).
# --------------------------------------------------------------------------- #


def _records(levels: int, leaf: str) -> tuple[str, str]:
    """Return a record nesting exactly `levels` records, the innermost field typed `leaf`.

    `levels=1` is `{k: <leaf>}`. Past 10 levels the record is built over an alias chain.
    """
    if levels <= 10:
        return "", "{k: " * levels + f'"{leaf}"' + "}" * levels
    items = [f'&m0 {{k: "{leaf}"}}'] + [f"&m{i} {{k: *m{i - 1}}}" for i in range(1, levels - 1)]
    return f"x-anchors: [{', '.join(items)}]\n", f"{{k: *m{levels - 2}}}"


def _field_list_records(levels: int, leaf: str) -> tuple[str, str]:
    """Return `_records(levels, leaf)` spelled as lists of fields, each level `[{k: ...}]`."""
    if levels <= 10:
        return "", "[{k: " * levels + f'"{leaf}"' + "}]" * levels
    items = [f'&m0 [{{k: "{leaf}"}}]'] + [
        f"&m{i} [{{k: *m{i - 1}}}]" for i in range(1, levels - 1)
    ]
    return f"x-anchors: [{', '.join(items)}]\n", f"[{{k: *m{levels - 2}}}]"


# --------------------------------------------------------------------------- #
# Flows: each puts `value` at one node's `output:`.
# --------------------------------------------------------------------------- #


def _code(header: str, value: str, typedefs: str = "") -> str:
    """Return a flow whose CODE node `use` declares `output: value`."""
    return (
        f"{header}id: f\nname: f\n{typedefs}nodes:\n  use:\n{_ECHO}"
        f"    input:\n      topic: x\n    output: {value}\noutput: ${{use.output}}\n"
    )


def _agent(header: str, value: str) -> str:
    """Return a flow whose AGENT node `use` declares `output: value`."""
    return (
        f"{header}id: f\nname: f\nnodes:\n  use:\n    kind: agent\n    prompt: hi\n"
        f"    output: {value}\noutput: ${{use.output}}\n"
    )


def _human_input(header: str, value: str) -> str:
    """Return a flow whose HUMAN_INPUT node `use` declares `output: value`."""
    return (
        f"{header}id: f\nname: f\nnodes:\n  use:\n    kind: human_input\n    prompt: hi\n"
        f"    output: {value}\noutput: ${{use.output}}\n"
    )


def _tool(header: str, value: str) -> str:
    """Return a flow whose TOOL node `use` declares `output: value`."""
    return (
        f"{header}id: f\nname: f\nnodes:\n  use:\n    kind: tool\n    tool_id: http_get\n"
        f"    input:\n      url: x\n    output: {value}\noutput: ${{use.output}}\n"
    )


def _compact(header: str, value: str) -> str:
    """Return a compact (single-node) flow declaring `output: value`."""
    return (
        f"{header}id: f\nname: f\nkind: code\ncode: tests.engine._compose_codefns:echo\n"
        f"input:\n  topic: !str\noutput: {value}\n"
    )


_CALL = (
    "nodes:\n  use:\n    kind: call\n    call: d\n    input:\n      topic: x\n"
    "output: ${use.output}\n"
)


def _single_def(header: str, value: str) -> str:
    """Return a flow calling single-node def `d`, which declares `output: value`."""
    return (
        f"{header}id: f\nname: f\ndefs:\n  d:\n{_ECHO}    input:\n      topic: !str\n"
        f"    output: {value}\n{_CALL}"
    )


def _multi_def(header: str, value: str) -> str:
    """Return a flow calling multi-node def `d`, whose node `u` declares `output: value`."""
    return (
        f"{header}id: f\nname: f\ndefs:\n  d:\n    input:\n      topic: !str\n"
        "    nodes:\n      u:\n        kind: code\n"
        "        code: tests.engine._compose_codefns:echo\n"
        f"        input:\n          topic: x\n        output: {value}\n"
        f"    output: ${{u.output}}\n{_CALL}"
    )


_SITES: dict[str, Callable[[str, str], str]] = {
    "code": _code,
    "agent": _agent,
    "human_input": _human_input,
    "tool": _tool,
    "compact": _compact,
    "single-node-def": _single_def,
    "multi-node-def": _multi_def,
}

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _chain(error: BaseException) -> Iterator[BaseException]:
    """Yield `error` and every exception reachable through `__cause__`/`__context__`."""
    seen: set[int] = set()
    pending = [error]
    while pending:
        current = pending.pop()
        if current is None or id(current) in seen:
            continue
        seen.add(id(current))
        yield current
        pending.extend([current.__cause__, current.__context__])


def _limit_error(text: str, limit: str = _LIMIT) -> str:
    """Assert loading `text` fails with a located limit error, and return its message."""
    with pytest.raises(LoadError) as caught:
        load_flow(text)
    error = caught.value
    message = error.diagnostic.message
    assert limit in message, message
    assert error.diagnostic.frames[-1].range is not None
    assert not any(isinstance(link, RecursionError) for link in _chain(error))
    return message


# --------------------------------------------------------------------------- #
# A 1000-link alias chain
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("site", list(_SITES))
def test_a_record_output_over_a_1000_link_alias_chain_is_a_located_load_error(
    site: str,
) -> None:
    _limit_error(_SITES[site](*_records(1000, "str")))


def test_a_list_of_fields_output_over_a_1000_link_alias_chain_is_a_located_load_error() -> None:
    _limit_error(_code(*_field_list_records(1000, "str")))


def test_a_case_output_over_a_1000_link_alias_chain_is_rejected_as_a_case_output() -> None:
    header, value = _records(1000, "str")
    text = (
        f"{header}id: f\nname: f\ninput:\n  cmd: !str\nnodes:\n  route:\n    kind: case\n"
        f"    on: ${{input.cmd}}\n    output: {value}\n    cases:\n      - when: a\n"
        "        then: reply\n    else: reply\n  reply:\n"
        f"{_ECHO}    input:\n      topic: replied\n    output: str\n"
        "output: ${route.output}\n"
    )
    with pytest.raises(LoadError) as caught:
        load_flow(text)
    assert str(caught.value) == (
        "case node 'route': a `case` routes and has no `output:`; tag its result where "
        "you read it, e.g. `x: !<type> ${route.output}`"
    )
    assert not any(isinstance(link, RecursionError) for link in _chain(caught.value))


# --------------------------------------------------------------------------- #
# Records and field types counted together
# --------------------------------------------------------------------------- #


def test_one_record_past_the_limit_is_a_load_error() -> None:
    message = _limit_error(_code(*_records(MAX_TYPE_NESTING + 1, "str")))

    assert f"type at {'.'.join(['k'] * MAX_TYPE_NESTING)} nests" in message
    assert "its type" not in message


def test_a_field_type_counts_its_lists() -> None:
    message = _limit_error(_code(*_records(99, "list(list(int))")))

    assert message.endswith(
        "(each record and list is a level); its type 'list(list(int))' adds 2"
    ), message


def test_records_and_a_deep_list_field_are_counted_together() -> None:
    leaf = "list(" * 50 + "int" + ")" * 50

    _limit_error(_code(*_records(51, leaf)))


def test_a_typedef_name_counts_as_deep_as_its_definition() -> None:
    message = _limit_error(_code(*_records(2, "T"), typedefs=f'typedefs:\n  T: "{_T99}"\n'))

    assert message.endswith("its type 'T' adds 99"), message


def test_record_nesting_follows_the_configured_limit(set_defaults) -> None:
    set_defaults(MAX_TYPE_NESTING=5)

    load_flow(_code(*_records(5, "str")))
    _limit_error(_code(*_records(6, "str")), "nests deeper than 5 levels, the limit for a type")


# --------------------------------------------------------------------------- #
# Guards
# --------------------------------------------------------------------------- #


def test_a_record_at_the_limit_loads_and_runs() -> None:
    result = run_flow(load_flow(_code(*_records(MAX_TYPE_NESTING, "str"))), {})

    assert result is not None


def test_types_at_the_limit_load() -> None:
    load_flow(_code(*_records(98, "list(list(int))")))
    load_flow(_code(*_records(50, "list(" * 50 + "int" + ")" * 50)))
    load_flow(_code(*_records(1, "T"), typedefs=f'typedefs:\n  T: "{_T99}"\n'))


def test_a_record_at_the_highest_configurable_limit_loads_and_runs(set_defaults) -> None:
    set_defaults(MAX_TYPE_NESTING=125)

    result = run_flow(load_flow(_code(*_records(125, "str"))), {})

    assert result is not None


@pytest.mark.parametrize("output", ["&R {a: *R}", "&R [{a: *R}]"])
def test_a_recursive_record_alias_keeps_its_error(output: str) -> None:
    with pytest.raises(LoadError) as caught:
        load_flow(_code("", output))
    assert "cannot descend through a recursive alias occurrence" in str(caught.value)
