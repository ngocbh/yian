"""A load error in an adaptive gate's composer names the gate and points at the authored text."""

from __future__ import annotations

import pytest

from yian.compose import LoadError, load_flow

# The gate's `llm_config:` value starts on line 10.
_FLOW = """\
id: f
name: f
input: {{proj: str}}
nodes:
  ask:
    kind: human_input
    input: {{proj: '${{input.proj}}'}}
    adaptive_questions:
      prompt: "Q ${{proj}}"
{llm_config}output: ${{ask.output}}
"""

_LLM_SITE = "node 'ask' adaptive_questions.llm_config"


def _flow(*, llm_config: str = "") -> str:
    return _FLOW.format(llm_config=llm_config)


def _load_error(text: str) -> LoadError:
    with pytest.raises(LoadError) as caught:
        load_flow(text)
    return caught.value


def _line(error: LoadError) -> int:
    return error.diagnostic.frames[-1].range.start_line


@pytest.mark.parametrize(
    ("config", "fragments"),
    [
        (
            "{inherit: 'false'}",
            ["adaptive_questions.llm_config `inherit:` must be true or false, got 'false'"],
        ),
        ("{model: 5}", ["adaptive_questions.llm_config: 1 validation error for LLMConfig"]),
        ("{temparature: 1}", ["temparature", "extra_forbidden"]),
        ("5", ["adaptive_questions.llm_config must be a mapping"]),
        ("{api_key: x}", ["adaptive_questions.llm_config.api_key is a host credential"]),
    ],
    ids=["inherit", "model-type", "unknown-key", "not-mapping", "api-key"],
)
def test_bad_llm_config_is_named_and_located_at_the_llm_config(config, fragments):
    error = _load_error(_flow(llm_config=f"      llm_config: {config}\n"))
    message = str(error)
    assert message.startswith(_LLM_SITE)
    for fragment in fragments:
        assert fragment in message
    assert "ask__compose" not in message
    assert _line(error) == 10


def test_block_llm_config_error_is_located_at_its_first_key():
    error = _load_error(
        _flow(llm_config="      llm_config:\n        model: m\n        bogus: 1\n")
    )
    message = str(error)
    assert message.startswith(_LLM_SITE)
    assert "bogus" in message
    assert "ask__compose" not in message
    assert _line(error) == 11


def test_compact_flow_llm_config_error_names_the_flow_node():
    error = _load_error(
        """\
id: f
name: f
input: {proj: str}
kind: human_input
adaptive_questions:
  prompt: "Q ${proj}"
  llm_config: {temparature: 1}
"""
    )
    message = str(error)
    assert message.startswith("node 'f' adaptive_questions.llm_config")
    assert "f__compose" not in message
    assert _line(error) == 7


def test_valid_llm_config_loads():
    load_flow(_flow(llm_config="      llm_config: {model: m}\n"))
