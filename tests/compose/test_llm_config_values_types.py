"""The sources of `llm_config` values must be typed, and their types must fit the field."""

from __future__ import annotations

import pytest

from yian.compose.helpers.errors import LoadError
from yian.compose.loader import load_flow

_TYPEDEFS = """\
typedefs:
  WithKey:
    model: str
    api_key: str
  WithInherit:
    model: str
    inherit: bool
  WithBogus:
    model: str
    bogus: int
  WithUrl:
    model: str
    base_url: str
  WithHeaders:
    model: str
    headers: object
  Fine:
    provider: Optional(str)
    model: str
  BadBlock:
    bogus: int
  GoodBlock:
    max_tokens: int
  BadProvider: Literal(nope, other)
  SomeProvider: Literal(openai, nope)
"""

_INPUTS = """\
input:
  s: !str
  i: !int
  b: !int
  bad_p: !BadProvider
  some_p: !SomeProvider
  key: !WithKey
  inh: !WithInherit
  bogus: !WithBogus
  url: !WithUrl
  headers: !WithHeaders
  fine: !Fine
  obj: !object
  bad_block: !BadBlock
  good_block: !GoodBlock
"""


def _flow(llm_lines: str, *, extra_nodes: str = "", flow_layer: str = "") -> str:
    """Return a flow whose agent `chat` has the node lines `llm_lines` (indented 4)."""
    return f"""\
id: f
name: f
{_TYPEDEFS}{_INPUTS}{flow_layer}nodes:
{extra_nodes}  chat:
    kind: agent
    prompt: Go.
{llm_lines}output: ${{chat.output}}
"""


def _loads(text: str) -> None:
    load_flow(text, source_label="f.yaml")


def _error(text: str) -> str:
    with pytest.raises(LoadError) as caught:
        load_flow(text, source_label="f.yaml")
    return caught.value.diagnostic.message


def _line(text: str) -> int:
    with pytest.raises(LoadError) as caught:
        load_flow(text, source_label="f.yaml")
    return caught.value.diagnostic.frames[-1].range.start_line


def _line_of(text: str, needle: str) -> int:
    return next(n for n, line in enumerate(text.splitlines(), 1) if needle in line)


_UNTYPED = """\
  pick:
    kind: code
    code: |
      return "gpt-x"
"""


def test_an_untyped_producer_must_be_tagged():
    text = _flow("    llm_config: {model: '${pick.output}'}\n", extra_nodes=_UNTYPED)
    message = _error(text)
    assert message.startswith("node 'chat' llm_config.model: "), message
    assert "tag it" in message
    assert _line(text) == _line_of(text, "${pick.output}")
    _loads(_flow("    llm_config: {model: !str '${pick.output}'}\n", extra_nodes=_UNTYPED))


def test_an_int_into_model_is_disjoint():
    message = _error(_flow("    llm_config: {model: '${input.i}'}\n"))
    assert message.startswith("node 'chat' llm_config.model: "), message
    assert "must be str" in message and "int" in message


def test_a_provider_with_no_known_member_fails_naming_them():
    message = _error(_flow("    llm_config: {provider: '${input.bad_p}'}\n"))
    assert message.startswith("node 'chat' llm_config.provider: "), message
    assert "nope" in message and "other" in message


def test_a_provider_with_a_known_member_or_any_text_loads():
    _loads(_flow("    llm_config: {provider: '${input.some_p}'}\n"))
    _loads(_flow("    llm_config: {provider: '${input.s}'}\n"))


def test_a_block_field_takes_its_annotation_type():
    template = (
        "    llm_config: {{anthropic: {{thinking: {{type: enabled, budget_tokens: '{}'}}}}}}\n"
    )
    message = _error(_flow(template.format("${input.s}")))
    assert message.startswith(
        "node 'chat' llm_config.anthropic.thinking.budget_tokens: "
    ), message
    assert "must be int" in message
    # The bound (at least 1024) is checked when the value resolves.
    _loads(_flow(template.format("${input.b}")))


@pytest.mark.parametrize(
    ("source", "field"),
    [
        ("${input.key}", "api_key"),
        ("${input.url}", "base_url"),
        ("${input.headers}", "headers"),
        ("${input.inh}", "inherit"),
        ("${input.bogus}", "bogus"),
    ],
)
def test_a_whole_record_type_must_declare_only_llm_config_fields(source, field):
    message = _error(_flow(f"    llm_config: {source}\n"))
    assert message.startswith("node 'chat' llm_config: "), message
    assert f"`{field}`" in message


def test_a_whole_record_of_allowed_fields_or_object_loads():
    _loads(_flow("    llm_config: ${input.fine}\n"))
    _loads(_flow("    llm_config: ${input.obj}\n"))


def test_a_whole_record_that_is_not_a_record_fails():
    message = _error(_flow("    llm_config: ${input.s}\n"))
    assert message.startswith("node 'chat' llm_config: "), message


def test_a_whole_block_type_must_declare_only_block_fields():
    message = _error(_flow("    llm_config: {anthropic: '${input.bad_block}'}\n"))
    assert message.startswith("node 'chat' llm_config.anthropic: "), message
    assert "`bogus`" in message
    _loads(_flow("    llm_config: {anthropic: '${input.good_block}'}\n"))


def test_a_constructor_fails():
    text = _flow("    llm_config: {model: '${Fine(input.fine).model}'}\n")
    message = _error(text)
    assert message.startswith("node 'chat' llm_config.model: "), message
    assert "Fine(...)" in message


@pytest.mark.parametrize(
    ("llm_lines", "flow_layer", "subject"),
    [
        ("    llm_config: {model: '${isinstance(input.s, str)}'}\n", "", "node 'chat'"),
        (
            "    llm_config: {model: m0}\n",
            "llm_config: {model: '${isinstance(input.s, str)}'}\n",
            "flow",
        ),
    ],
)
def test_isinstance_fails(llm_lines, flow_layer, subject):
    message = _error(_flow(llm_lines, flow_layer=flow_layer))
    assert message.startswith(f"{subject} llm_config.model: "), message
    assert "isinstance is not supported" in message


@pytest.mark.parametrize(
    ("llm_lines", "flow_layer", "subject"),
    [
        ("    llm_config: {model: !str '${input.i}'}\n", "", "node 'chat' llm_config.model"),
        ("", "llm_config: {model: !str '${input.i}'}\n", "flow llm_config.model"),
        (
            "    llm_config: {model: m0}\n",
            "llm_config: {model: !str '${input.i}'}\n",
            "flow llm_config.model",
        ),
        ("    llm_config: !Fine ${input.s}\n", "", "node 'chat' llm_config"),
    ],
    ids=["node", "flow-carried", "flow-shielded", "whole-record"],
)
def test_a_tag_the_source_cannot_fit_fails_at_the_field(llm_lines, flow_layer, subject):
    text = _flow(llm_lines, flow_layer=flow_layer)
    message = _error(text)
    assert message.startswith(f"{subject}: declared "), message
    assert "__llm_config" not in message
    assert _line(text) == _line_of(text, "${input.")


def test_a_flow_layer_value_no_agent_carries_is_still_checked():
    text = _flow(
        "    llm_config: {model: m0}\n", flow_layer="llm_config: {model: '${input.i}'}\n"
    )
    message = _error(text)
    assert message.startswith("flow llm_config.model: "), message
    assert "must be str" in message
    assert _line(text) == _line_of(text, "${input.i}")
    unknown = _flow(
        "    llm_config: {model: m0}\n", flow_layer="llm_config: {model: '${input.nope}'}\n"
    )
    assert "unknown flow input 'nope'" in _error(unknown)


@pytest.mark.parametrize(
    ("llm_lines", "flow_layer"),
    [
        ("    llm_config: !WithInherit ${input.obj}\n", ""),
        ("    llm_config: {model: m0}\n", "llm_config: ${input.inh}\n"),
        ("    llm_config: {model: m0}\n", "llm_config: !WithInherit ${input.obj}\n"),
    ],
)
def test_a_whole_record_type_declaring_inherit_is_refused(llm_lines, flow_layer):
    message = _error(_flow(llm_lines, flow_layer=flow_layer))
    assert "llm_config" in message and "`inherit`" in message, message
