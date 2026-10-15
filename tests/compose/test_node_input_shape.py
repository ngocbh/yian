"""A node's `input:` must be a mapping; any other value is a located load error.

Every kind with an `input:` binding map (LOOP, CALL, MAP, CODE, TOOL, AGENT, HUMAN_INPUT)
rejects a whole-record reference, other text, a number, or a list at parse, at the
`input:` field, with a message that names the kind's binding noun and shows a fix.
"""

import re

import pytest

from yian.compose import LoadError, load_flow
from yian.run import run_flow


_FLOW = """\
id: input-shape
name: input_shape
input:
  z: !int
defs:
  step:
    input:
      n: !int
    nodes:
      b:
        kind: code
        input:
          n: ${input.n}
        output:
          n: int
        code: |
          return {"n": inputs["n"] + 1}
    output: ${b.output}
nodes:
  seed:
    kind: code
    input:
      z: ${input.z}
    output:
      n: int
    code: |
      return {"n": inputs["z"]}
NODE
"""

_LOOP = """\
  repeat:
    kind: loop
    times: 2
    call: step
    input: INPUT
output: ${repeat.output}"""


def _loop_flow(input_text: str) -> str:
    """Return the shared flow with a named LOOP `repeat` whose `input:` is `input_text`."""
    return _FLOW.replace("NODE", _LOOP.replace("INPUT", input_text))


def _load_error(text: str) -> LoadError:
    """Load `text`, assert it raises a `LoadError`, and return the error."""
    with pytest.raises(LoadError) as excinfo:
        load_flow(text)
    return excinfo.value


def test_whole_record_loop_seed_is_a_located_load_error():
    """A LOOP seeded with a whole record names the reference and points at `input:`."""
    text = _loop_flow("${seed.output}")
    with pytest.raises(
        LoadError,
        match=re.escape(
            "node 'repeat' (kind=loop): `input:` must be a mapping of carried fields "
            "(got the reference '${seed.output}'); list each carried field, such as "
            "`<field>: ${seed.output.<field>}`"
        ),
    ) as excinfo:
        load_flow(text)
    line = text.splitlines().index("    input: ${seed.output}") + 1
    assert excinfo.value.diagnostic.frames[-1].range.start_line == line


@pytest.mark.parametrize(
    ("node", "noun", "verb"),
    [
        ("kind: call\n    call: step", "the child's arguments", "bind each argument"),
        (
            "kind: map\n    over: [1, 2]\n    call: step",
            "the child's arguments",
            "bind each argument",
        ),
        ("kind: code\n    code: return 1", "parameter names to bindings", "bind each parameter"),
        (
            "kind: tool\n    tool_id: http_get",
            "parameter names to bindings",
            "bind each parameter",
        ),
        ("kind: agent\n    prompt: hi", "parameter names to bindings", "bind each parameter"),
        (
            "kind: human_input\n    prompt: hi",
            "parameter names to bindings",
            "bind each parameter",
        ),
    ],
    ids=["call", "map", "code", "tool", "agent", "human_input"],
)
def test_whole_reference_input_is_a_load_error_for_every_kind(node, noun, verb):
    """Every kind with an `input:` map rejects a whole-record reference with its own noun."""
    text = _FLOW.replace("NODE", f"  under_test:\n    {node}\n    input: ${{seed.output}}")
    message = str(_load_error(text))
    assert f"`input:` must be a mapping of {noun} (got the reference '${{seed.output}}')" in (
        message
    )
    assert f"; {verb}, such as `<field>: ${{seed.output.<field>}}`" in message


@pytest.mark.parametrize(
    ("input_text", "got", "example"),
    [
        (
            "${seed.output.n}",
            "(got the reference '${seed.output.n}')",
            "`<field>: ${seed.output.n.<field>}`",
        ),
        (
            "${seed.output.n + 1}",
            "(got the expression '${seed.output.n + 1}')",
            "`<field>: ${...}`",
        ),
        ("x ${seed.output}", "(got 'x ${seed.output}')", "`<field>: ${...}`"),
    ],
    ids=["nested-reference", "expression", "embedded"],
)
def test_non_reference_spans_get_a_generic_example(input_text, got, example):
    """Only a lone reference gets a path-specific example; other spans get `${...}`."""
    with pytest.raises(
        LoadError,
        match=re.escape(
            f"`input:` must be a mapping of carried fields {got}; list each carried field, "
            f"such as {example}"
        ),
    ):
        load_flow(_loop_flow(input_text))


@pytest.mark.parametrize(
    ("input_text", "tail"),
    [
        ("5", "(got 5); list each carried field, such as `<field>: ...`"),
        ("[1]", "(got [1]); list each carried field, such as `<field>: ...`"),
        ("abc", "(got 'abc'); list each carried field, such as `<field>: ...`"),
        ("[]", "(got []); write `input: {}` or remove `input:`"),
        ("''", "(got ''); write `input: {}` or remove `input:`"),
        (
            "!object ${seed.output}",
            "(got the reference '${seed.output}'); list each carried field, such as "
            "`<field>: ${seed.output.<field>}`",
        ),
    ],
    ids=["number", "list", "text", "empty-list", "empty-text", "tagged-reference"],
)
def test_non_mapping_input_is_a_load_error(input_text, tail):
    """A number, list, plain text, or tagged reference as `input:` is a load error."""
    with pytest.raises(
        LoadError,
        match=re.escape(f"`input:` must be a mapping of carried fields {tail}"),
    ):
        load_flow(_loop_flow(input_text))


_DEFS_FLOW = """\
id: input-shape-defs
name: input_shape_defs
input:
  z: !int
defs:
  step:
    input:
      n: !int
    nodes:
      b:
        kind: code
        input:
          n: ${input.n}
        output:
          n: int
        code: |
          return {"n": inputs["n"] + 1}
      c:
        kind: code
        input: ${b.output}
        code: return 1
    output: ${b.output}
nodes:
  run:
    kind: call
    call: step
    input:
      n: ${input.z}
output: ${run.output}
"""


def test_non_mapping_input_in_a_defs_body_is_located():
    """A non-mapping `input:` inside a `defs:` body names the defs entry and the node."""
    message = str(_load_error(_DEFS_FLOW))
    assert "defs entry 'step': " in message
    assert "node 'c' (kind=code): `input:` must be a mapping" in message


@pytest.mark.parametrize("input_text", ["", " {}"], ids=["null", "empty-mapping"])
def test_null_or_empty_mapping_input_still_loads(input_text):
    """A CODE node with a null or empty-mapping `input:` still loads."""
    load_flow(
        _FLOW.replace(
            "NODE",
            f"  c:\n    kind: code\n    code: return 1\n    input:{input_text}",
        )
    )


def test_listed_carried_fields_load_and_run():
    """A LOOP that lists each carried field loads and runs to the expected record."""
    flow = load_flow(_loop_flow("\n      n: ${seed.output.n}"))
    result = run_flow(flow, {"z": 1})
    assert result.status == "succeeded", result.failure
    assert result.output == {"n": 3}


_INLINE_LOOP = """\
  repeat:
    kind: loop
    times: 2
    input: ${seed.output}
    nodes:
      b:
        kind: code
        input:
          n: ${input.n}
        output:
          n: int
        code: |
          return {"n": inputs["n"] + 1}
    output: ${b.output}
output: ${repeat.output}"""


def test_inline_loop_whole_reference_keeps_its_message():
    """An inline LOOP keeps the inline-body message for a non-mapping `input:`."""
    with pytest.raises(
        LoadError, match=re.escape("inline `input:` must be a mapping with string keys")
    ):
        load_flow(_FLOW.replace("NODE", _INLINE_LOOP))
