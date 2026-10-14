"""A written-out value nested past 100 levels is a located load error.

Each value written out in a flow file at a value position is checked at load with the
counting the run-time check applies to it, whether YAML text or aliases built it. A
value at the limit loads (and runs); one past it fails at load, located, and a
1000-link alias chain fails the same way instead of raising `RecursionError`.

Message prefix and range per site:

- node `input:` key, LOOP seed: `node '<id>' input '<key>'`, at the input key.
- MAP `over:`, WAIT `until:`: `node '<id>' field 'over'` / `'until'`, at the field.
- value entry: `value entry '<id>'`, at the entry.
- flow, def, and compact-flow input default: `input '<name>' default` (a def's message
  starts `defs entry '<def>': `), at the declaration; AGENT variable default:
  `variable '<name>' default`, at the declaration.
- flow output: `flow output '<name>'` for a record field (one level in), at the field;
  `flow output` for a tagged section or a single value, at the section.
- a binding inside an inline body: as for any node input, at the inner binding.
- CASE value branch: `node '<id>' then[<i>] value` / `node '<id>' else value`, at the
  branch.
"""

from __future__ import annotations

from typing import Callable, Iterator

import pytest

from yian.compose import LoadError, load_flow
from yian.run import run_flow
from yian.typesys.values import MAX_VALUE_NESTING

_LIMIT = (
    f"value nests deeper than {MAX_VALUE_NESTING} levels, the limit for a yian value"
)

# --------------------------------------------------------------------------- #
# Value forms: each returns `(header, value)`, where `header` holds the top-level
# `x-anchors:` line the value's aliases name (empty for written-out text).
# --------------------------------------------------------------------------- #


def _list(levels: int) -> tuple[str, str]:
    """Return a list written out as text, nesting `levels` levels."""
    return "", "[" * levels + "1" + "]" * levels


def _record(levels: int) -> tuple[str, str]:
    """Return a mapping written out as text, nesting `levels` levels."""
    return "", "{a: " * levels + "1" + "}" * levels


def _anchors(count: int, *, record: bool) -> str:
    """Return an `x-anchors:` line whose anchor `a<i>` names a value nesting i+1 levels."""
    if record:
        items = ["&a0 {a: 1}"] + [f"&a{i} {{a: *a{i - 1}}}" for i in range(1, count)]
    else:
        items = ["&a0 [1]"] + [f"&a{i} [*a{i - 1}]" for i in range(1, count)]
    return f"x-anchors: [{', '.join(items)}]\n"


def _list_alias(levels: int) -> tuple[str, str]:
    """Return a one-item list over an alias chain, nesting exactly `levels` levels.

    YAML can't tag an alias, so the value wraps the last alias in one written level.
    """
    return _anchors(levels - 1, record=False), f"[*a{levels - 2}]"


def _record_alias(levels: int) -> tuple[str, str]:
    """Return a one-key mapping over an alias chain, nesting exactly `levels` levels."""
    return _anchors(levels - 1, record=True), f"{{a: *a{levels - 2}}}"


def _record_list(levels: int) -> tuple[str, str]:
    """Return a one-item list of a mapping written out as text, nesting `levels` levels."""
    return "", "[" + _record(levels - 1)[1] + "]"


def _record_list_alias(levels: int) -> tuple[str, str]:
    """Return a one-item list of a mapping over an alias chain, nesting `levels` levels."""
    return _anchors(levels - 2, record=True), f"[{{a: *a{levels - 3}}}]"


# --------------------------------------------------------------------------- #
# Flows
# --------------------------------------------------------------------------- #

_ECHO = "    kind: code\n    code: tests.engine._compose_codefns:echo\n"


def _input_flow(header: str, value: str) -> str:
    """Return a flow whose CODE node `use` binds `topic` to `value` and echoes it."""
    return (
        f"id: f\nname: f\n{header}nodes:\n  use:\n{_ECHO}"
        f"    input:\n      topic: {value}\noutput: ${{use.output}}\n"
    )


def _tagged_input_flow(header: str, value: str) -> str:
    """Return `_input_flow` with `topic` tagged `!object`."""
    return _input_flow(header, f"!object {value}")


def _loop_flow(header: str, value: str) -> str:
    """Return a flow whose LOOP `l` seeds its carried `s` with `value`."""
    return (
        f"id: f\nname: f\n{header}defs:\n  body:\n    input:\n      s: !object\n"
        "    nodes:\n      keep:\n        kind: code\n"
        "        code: tests.engine._compose_codefns:echo\n"
        "        input:\n          topic: ${input.s}\n"
        "    output:\n      s: ${keep.output}\n"
        "nodes:\n  l:\n    kind: loop\n    call: body\n    times: 2\n"
        f"    input:\n      s: {value}\noutput: ${{l.output.s}}\n"
    )


def _map_flow(header: str, value: str) -> str:
    """Return a flow whose MAP `m` echoes each item of `over: value`."""
    return (
        f"id: f\nname: f\n{header}defs:\n  wrap:\n    input:\n      v: !object\n"
        "    nodes:\n      keep:\n        kind: code\n"
        "        code: tests.engine._compose_codefns:echo\n"
        "        input:\n          topic: ${input.v}\n"
        "    output: ${keep.output}\n"
        f"nodes:\n  m:\n    kind: map\n    call: wrap\n    over: {value}\n"
        "    input:\n      v: ${item}\noutput: ${m.output}\n"
    )


def _wait_flow(header: str, value: str) -> str:
    """Return a flow whose WAIT `w` waits `until: value`."""
    return (
        f"id: f\nname: f\n{header}nodes:\n  w:\n    kind: wait\n    until: {value}\n"
        f"  use:\n{_ECHO}    input:\n      topic: x\n    runs_after: [w]\n"
        "output: ${use.output}\n"
    )


def _value_entry_flow(header: str, value: str) -> str:
    """Return a flow whose value entry `v` holds `!object value`, echoed by `use`."""
    return (
        f"id: f\nname: f\n{header}nodes:\n  v: !object {value}\n  use:\n{_ECHO}"
        "    input:\n      topic: ${v.output}\noutput: ${use.output}\n"
    )


def _flow_default_flow(header: str, value: str) -> str:
    """Return a flow whose input `topic` defaults to `!object value`, echoed by `use`."""
    return (
        f"id: f\nname: f\n{header}input:\n  topic: !object {value}\nnodes:\n  use:\n"
        f"{_ECHO}    input:\n      topic: ${{input.topic}}\noutput: ${{use.output}}\n"
    )


def _def_default_flow(header: str, value: str) -> str:
    """Return a flow that calls def `child`, whose input `topic` defaults to `value`."""
    return (
        f"id: f\nname: f\n{header}defs:\n  child:\n    input:\n"
        f"      topic: !object {value}\n    nodes:\n      keep:\n        kind: code\n"
        "        code: tests.engine._compose_codefns:echo\n"
        "        input:\n          topic: ${input.topic}\n    output: ${keep.output}\n"
        "nodes:\n  c:\n    kind: call\n    call: child\noutput: ${c.output}\n"
    )


def _compact_default_flow(header: str, value: str) -> str:
    """Return a compact (single-node) flow whose input `topic` defaults to `value`."""
    return (
        f"id: f\nname: f\n{header}input:\n  topic: !object {value}\n"
        "kind: code\ncode: tests.engine._compose_codefns:echo\n"
    )


def _agent_variable_flow(header: str, value: str) -> str:
    """Return a flow whose AGENT `writer` declares variable `t` defaulting to `value`."""
    return (
        f"id: f\nname: f\n{header}llm_config: {{provider: ollama, model: test}}\n"
        "nodes:\n  writer:\n    kind: agent\n    variables:\n"
        f"      t: !object {value}\n    prompt: go\n    output: str\n"
        "output: ${writer.output}\n"
    )


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
    pending = [error]
    while pending:
        current = pending.pop()
        if current is None or id(current) in seen:
            continue
        seen.add(id(current))
        yield current
        pending.extend([current.__cause__, current.__context__])


def _load_error(text: str) -> LoadError:
    """Load `text`, expecting a `LoadError`, and return it."""
    with pytest.raises(LoadError) as caught:
        load_flow(text)
    return caught.value


def _assert_limit_error(text: str, where: str, line_prefix: str) -> None:
    """Assert loading `text` fails with the limit error for `where` at `line_prefix`."""
    error = _load_error(text)
    message = error.diagnostic.message
    assert where in message and _LIMIT in message, message
    assert error.diagnostic.frames[-1].range.start_line == _line_of(text, line_prefix)
    assert not any(isinstance(link, RecursionError) for link in _chain(error))


# --------------------------------------------------------------------------- #
# Node bindings
# --------------------------------------------------------------------------- #

_Flow = Callable[[str, str], str]
_Form = Callable[[int], tuple[str, str]]

# (flow builder, written-out form, aliased form, message prefix, located line, runs)
_BINDING_SITES: dict[str, tuple[_Flow, _Form, _Form, str, str, bool]] = {
    "input": (
        _input_flow, _list, _list_alias, "node 'use' input 'topic' at ", "      topic:", True,
    ),
    "tagged-input": (
        _tagged_input_flow,
        _record,
        _record_alias,
        "node 'use' input 'topic' at ",
        "      topic:",
        True,
    ),
    # The body's record output wraps the seed, so a 100-level seed fails when it runs.
    "loop-seed": (
        _loop_flow, _record, _record_alias, "node 'l' input 's' at ", "      s: {", False,
    ),
    # 100 levels in `over:` means items nesting 99.
    "map-over": (
        _map_flow,
        _record_list,
        _record_list_alias,
        "node 'm' field 'over' at ",
        "    over:",
        True,
    ),
    # A literal `until:` isn't a date, so it fails when it runs.
    "wait-until": (
        _wait_flow, _list, _list_alias, "node 'w' field 'until' at ", "    until:", False,
    ),
    "value-entry": (
        _value_entry_flow, _record, _record_alias, "value entry 'v' at ", "  v:", True,
    ),
    # A default is checked as the binding it fills.
    "flow-default": (
        _flow_default_flow,
        _record,
        _record_alias,
        "input 'topic' default at ",
        "  topic:",
        True,
    ),
    "def-default": (
        _def_default_flow,
        _record,
        _record_alias,
        "defs entry 'child': input 'topic' default at ",
        "      topic: !object",
        True,
    ),
    "compact-default": (
        _compact_default_flow,
        _record,
        _record_alias,
        "input 'topic' default at ",
        "  topic:",
        True,
    ),
    # Load only: running it would call a model.
    "agent-variable": (
        _agent_variable_flow,
        _record,
        _record_alias,
        "variable 't' default at ",
        "      t:",
        False,
    ),
}


@pytest.mark.parametrize("site", list(_BINDING_SITES))
@pytest.mark.parametrize("aliased", [False, True], ids=["text", "alias"])
def test_binding_at_the_limit_loads(site: str, aliased: bool) -> None:
    flow, text_form, alias_form, _, _, runs = _BINDING_SITES[site]
    header, value = (alias_form if aliased else text_form)(MAX_VALUE_NESTING)

    loaded = load_flow(flow(header, value))

    if runs:
        result = run_flow(loaded, {})
        assert result.status == "succeeded", result.failure


@pytest.mark.parametrize("site", list(_BINDING_SITES))
@pytest.mark.parametrize("aliased", [False, True], ids=["text", "alias"])
def test_binding_past_the_limit_is_a_located_load_error(site: str, aliased: bool) -> None:
    flow, text_form, alias_form, where, line, _ = _BINDING_SITES[site]
    header, value = (alias_form if aliased else text_form)(MAX_VALUE_NESTING + 1)

    _assert_limit_error(flow(header, value), where, line)


@pytest.mark.parametrize("site", list(_BINDING_SITES))
def test_binding_over_a_1000_link_alias_chain_is_a_located_load_error(site: str) -> None:
    flow, _, alias_form, where, line, _ = _BINDING_SITES[site]

    _assert_limit_error(flow(*alias_form(1000)), where, line)


def test_issue_repro_alias_chain_is_a_located_load_error() -> None:
    """The reported 1000-link chain, written straight into the input, fails at load."""
    items = ", ".join(f"&a{i} [*a{i - 1}]" if i else "&a0 [1]" for i in range(1000))
    text = (
        "id: f\nname: f\nnodes:\n  use:\n    kind: code\n"
        "    code: tests.engine._compose_codefns:echo\n"
        f"    input:\n      topic: [{items}]\n    output: str\noutput: ${{use.output}}\n"
    )

    _assert_limit_error(text, "node 'use' input 'topic' at ", "      topic:")


def _untagged_value_entry_flow(header: str, value: str) -> str:
    """Return a flow whose untagged list entry `v` sits beside a node `use`."""
    return (
        f"id: f\nname: f\n{header}nodes:\n  v: {value}\n  use:\n{_ECHO}"
        "    input:\n      topic: x\noutput: ${use.output}\n"
    )


# An untagged list entry is always a load error that suggests a tag, so there is no
# loading case at the limit; past it, the limit is reported instead.
@pytest.mark.parametrize(
    ("form", "levels"),
    [(_list, MAX_VALUE_NESTING + 1), (_list_alias, MAX_VALUE_NESTING + 1), (_list_alias, 1000)],
    ids=["text-101", "alias-101", "alias-1000"],
)
def test_untagged_value_entry_past_the_limit_reports_the_limit(form: _Form, levels: int) -> None:
    _assert_limit_error(_untagged_value_entry_flow(*form(levels)), "value entry 'v' at ", "  v:")


# --------------------------------------------------------------------------- #
# CASE value branches
# --------------------------------------------------------------------------- #


def _case_flow(header: str, then: str, else_: str) -> str:
    """Return a flow whose CASE `r` has one `then:` branch and an `else:` branch."""
    return (
        f"id: f\nname: f\n{header}input:\n  b: !bool\nnodes:\n  r:\n    kind: case\n"
        f"    cases:\n      - when: ${{input.b}}\n        then: {then}\n    else: {else_}\n"
        f"  use:\n{_ECHO}    input:\n      topic: x\noutput: ${{use.output}}\n"
    )


# A value branch is always a load error, so there is no loading case at the limit.
@pytest.mark.parametrize(
    ("form", "levels"),
    [(_list, MAX_VALUE_NESTING + 1), (_list_alias, MAX_VALUE_NESTING + 1), (_list_alias, 1000)],
    ids=["text-101", "alias-101", "alias-1000"],
)
@pytest.mark.parametrize("branch", ["then", "else"])
def test_case_value_branch_past_the_limit_reports_the_limit(
    branch: str, form: _Form, levels: int
) -> None:
    header, value = form(levels)
    if branch == "then":
        text = _case_flow(header, value, "use")
        where, line = "node 'r' then[0] value at ", "        then:"
    else:
        text = _case_flow(header, "use", value)
        where, line = "node 'r' else value at ", "    else:"

    _assert_limit_error(text, where, line)


# --------------------------------------------------------------------------- #
# Flow outputs
# --------------------------------------------------------------------------- #


def _record_output_flow(header: str, value: str) -> str:
    """Return a flow whose record output has field `r: value` beside `u`."""
    return (
        f"id: f\nname: f\n{header}nodes:\n  use:\n{_ECHO}    input:\n      topic: x\n"
        f"output:\n  r: {value}\n  u: ${{use.output}}\n"
    )


def _tagged_output_flow(header: str, value: str) -> str:
    """Return a flow whose whole output section is `!object value`."""
    return (
        f"id: f\nname: f\n{header}nodes:\n  use:\n{_ECHO}    input:\n      topic: x\n"
        f"output: !object {value}\n"
    )


def _single_output_flow(header: str, value: str) -> str:
    """Return a flow whose single output is the untagged `value`."""
    return (
        f"id: f\nname: f\n{header}nodes:\n  use:\n{_ECHO}    input:\n      topic: x\n"
        f"output: {value}\n"
    )


# (flow builder, written-out form, aliased form, message prefix, located line, limit).
# An untagged mapping is a record whose fields sit one level in; a tagged section or a
# single value is one value.
_OUTPUT_SITES: dict[str, tuple[_Flow, _Form, _Form, str, str, int]] = {
    "record": (
        _record_output_flow,
        _record,
        _record_alias,
        "flow output 'r' at ",
        "  r:",
        MAX_VALUE_NESTING - 1,
    ),
    "tagged-section": (
        _tagged_output_flow,
        _record,
        _record_alias,
        "flow output at ",
        "output:",
        MAX_VALUE_NESTING,
    ),
    "single": (
        _single_output_flow, _list, _list_alias, "flow output at ", "output:", MAX_VALUE_NESTING,
    ),
}


@pytest.mark.parametrize("site", list(_OUTPUT_SITES))
@pytest.mark.parametrize("aliased", [False, True], ids=["text", "alias"])
def test_flow_output_at_the_limit_loads_and_runs(site: str, aliased: bool) -> None:
    flow, text_form, alias_form, _, _, limit = _OUTPUT_SITES[site]

    loaded = load_flow(flow(*(alias_form if aliased else text_form)(limit)))

    result = run_flow(loaded, {})
    assert result.status == "succeeded", result.failure


@pytest.mark.parametrize("site", list(_OUTPUT_SITES))
@pytest.mark.parametrize("aliased", [False, True], ids=["text", "alias"])
def test_flow_output_past_the_limit_is_a_located_load_error(site: str, aliased: bool) -> None:
    flow, text_form, alias_form, where, line, limit = _OUTPUT_SITES[site]
    header, value = (alias_form if aliased else text_form)(limit + 1)

    _assert_limit_error(flow(header, value), where, line)


@pytest.mark.parametrize("site", list(_OUTPUT_SITES))
def test_flow_output_over_a_1000_link_alias_chain_is_a_located_load_error(site: str) -> None:
    flow, _, alias_form, where, line, _ = _OUTPUT_SITES[site]

    _assert_limit_error(flow(*alias_form(1000)), where, line)


# --------------------------------------------------------------------------- #
# Inline bodies
# --------------------------------------------------------------------------- #


def test_binding_in_an_inline_map_body_over_a_1000_link_alias_chain_is_located() -> None:
    """The inline body's own load reports the inner binding, not a resolve failure."""
    header, value = _list_alias(1000)
    text = (
        f"id: f\nname: f\n{header}nodes:\n  m:\n    kind: map\n    over: [1]\n"
        "    nodes:\n      keep:\n        kind: code\n"
        "        code: tests.engine._compose_codefns:echo\n"
        f"        input:\n          topic: {value}\n    output: ${{keep.output}}\n"
        "output: ${m.output}\n"
    )

    _assert_limit_error(text, "node 'keep' input 'topic' at ", "          topic:")
