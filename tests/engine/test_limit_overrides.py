"""The language safety limits follow `YIAN_<NAME>`, and each maximum fits the stack.

A test sets a limit with the `set_defaults` fixture. The proofs at each entry's
maximum run with 300 fewer frames of recursion headroom, because a caller may already
sit deep in its own stack.
"""

import json
import sys
from pathlib import Path

import httpx
import pytest
from langchain_core.messages import AIMessage

from yian import defaults, llm_clients
from yian.compose import LoadError, load_flow
from yian.run import resume_flow, run_flow
from yian.typesys import TypeCheckError
from yian.typesys.types import parse_type
from yian.typesys.values import reject_deep_nesting
from tests.compose.test_call_nesting_limit import (
    _chain_texts,
    _defs_chain,
    _importing,
    _stack_headroom,
    _write_files,
)
from tests.compose.test_skip_model import _many_cases
from tests.engine._memory_fakes import _Memory
from tests.engine.test_value_nesting_boundaries import (
    _assert_learns,
    _code_flow,
    _list_body,
    _memory_flow,
)


def _nest(levels: int) -> object:
    """Return `1` wrapped in `levels` single-key dicts."""
    value: object = 1
    for _ in range(levels):
        value = {"a": value}
    return value


def _upper(levels: int, ref: str) -> str:
    """Return `levels` nested `upper(...)` calls around `ref`."""
    return "upper(" * levels + ref + ")" * levels


def _with_less_stack(call, *args, **kwargs):
    """Return `call(*args, **kwargs)` run with 300 fewer frames of recursion headroom."""
    limit = sys.getrecursionlimit()
    sys.setrecursionlimit(limit - 300)
    try:
        return call(*args, **kwargs)
    finally:
        sys.setrecursionlimit(limit)


def _maximum(name: str) -> int:
    """Return the largest value the entry `name` accepts from the environment."""
    return next(entry.maximum for entry in defaults.ENTRIES if entry.name == name)


def _binding_flow(binding: str) -> str:
    """Return a flow whose one CODE node binds and returns `topic`."""
    return (
        "id: f\nname: f\ninput:\n  s: !str\nnodes:\n  use:\n    kind: code\n    input:\n"
        "      topic: ${" + binding + "}\n    output: str\n"
        "    code: |\n      return inputs['topic']\noutput: ${use.output}\n"
    )


def _yaml_flow(topic: str) -> str:
    """Return a flow whose CODE node's `topic` input is written out as `topic`."""
    return (
        "id: f\nname: f\nnodes:\n  use:\n    kind: code\n"
        "    code: tests.engine._compose_codefns:echo\n"
        "    input:\n      topic: " + topic + "\n    output: str\noutput: ${use.output}\n"
    )


def test_value_nesting_follows_the_variable(set_defaults):
    set_defaults(MAX_VALUE_NESTING=3)

    reject_deep_nesting(_nest(3), "output")
    with pytest.raises(TypeCheckError) as caught:
        reject_deep_nesting(_nest(4), "output")

    assert str(caught.value).startswith("output at a.a.a: ")
    assert "deeper than 3 levels" in str(caught.value)


def test_a_run_input_over_the_set_value_limit_is_refused(set_defaults):
    flow = load_flow(
        "id: f\nname: f\ninput:\n  x: !object\nnodes:\n  echo:\n    kind: code\n"
        "    input:\n      x: ${input.x}\n    code: |\n      return inputs['x']\n"
        "output: ${echo.output}\n"
    )
    set_defaults(MAX_VALUE_NESTING=3)

    with pytest.raises(ValueError, match="deeper than 3 levels"):
        run_flow(flow, {"x": _nest(4)})


def test_expression_nesting_follows_the_value_limit(set_defaults):
    set_defaults(MAX_VALUE_NESTING=3)

    load_flow(_binding_flow(_upper(3, "input.s")))
    with pytest.raises(LoadError, match="expression nests deeper than 3 levels"):
        load_flow(_binding_flow(_upper(4, "input.s")))


def test_a_written_out_value_over_the_set_value_limit_fails_at_load(set_defaults):
    set_defaults(MAX_VALUE_NESTING=3)

    load_flow(_yaml_flow("[[[1]]]"))
    with pytest.raises(LoadError, match="value nests deeper than 3 levels"):
        load_flow(_yaml_flow("[[[[1]]]]"))


def test_type_nesting_follows_the_variable(set_defaults):
    set_defaults(MAX_TYPE_NESTING=2)

    parse_type("list(list(int))")
    with pytest.raises(TypeCheckError, match="at most 2 levels"):
        parse_type("list(list(list(int)))")


def test_yaml_nesting_follows_the_variable_and_names_the_value_limit(set_defaults):
    set_defaults(MAX_YAML_NESTING=5, MAX_VALUE_NESTING=4)

    # The root, nodes, use, and input mappings are levels 1-4.
    load_flow(_yaml_flow("[1]"))
    with pytest.raises(LoadError) as caught:
        load_flow(_yaml_flow("[[1]]"))

    message = str(caught.value)
    assert "this list nests deeper than 5 levels" in message
    assert "a yian value nests at most 4)" in message
    frame_range = caught.value.diagnostic.frames[-1].range
    assert (frame_range.start_line, frame_range.start_col) == (8, 15)


def test_call_nesting_follows_the_variable(set_defaults):
    set_defaults(MAX_CALL_NESTING=2)

    load_flow(_defs_chain(2))
    with pytest.raises(LoadError) as caught:
        load_flow(_defs_chain(3))

    assert "nests calls 3 levels deep, past the call nesting limit of 2" in str(
        caught.value
    )


def test_skip_worlds_follow_the_variable(set_defaults):
    assert not _many_cases().may_all_skip({"a", "b"})

    set_defaults(MAX_SKIP_WORLDS=1)

    assert _many_cases().may_all_skip({"a", "b"})


def test_value_nesting_maximum_learns_with_less_stack(set_defaults):
    levels = _maximum("MAX_VALUE_NESTING")
    set_defaults(MAX_VALUE_NESTING=levels)

    _assert_learns(_memory_flow(_code_flow(_list_body(levels))), {}, "good")


@pytest.mark.parametrize("num_workers", [0, 2])
def test_expression_nesting_maximum_runs_with_less_stack(set_defaults, num_workers):
    levels = _maximum("MAX_VALUE_NESTING")
    set_defaults(MAX_VALUE_NESTING=levels)

    flow = _with_less_stack(load_flow, _binding_flow(_upper(levels, "input.s")))
    result = _with_less_stack(run_flow, flow, {"s": "a"}, num_workers=num_workers)

    assert result.status == "succeeded" and result.output == "A"


def test_condition_at_the_expression_maximum_runs_thirty_calls_deep(set_defaults):
    levels = _maximum("MAX_VALUE_NESTING")
    set_defaults(MAX_VALUE_NESTING=levels)
    condition = _upper(levels - 1, "input.s") + " == 'A'"  # the comparison is a level
    lines = ["id: f", "name: f", "input:", "  s: !str", "defs:"]
    for i in range(30):
        lines += [f"  c{i}:", "    input:", "      s: !str", "    nodes:"]
        if i < 29:
            lines += [
                "      sub:", "        kind: call", f"        call: c{i + 1}",
                "        input:", "          s: ${input.s}", "    output: ${sub.output}",
            ]
        else:
            lines += [
                "      route:", "        kind: case", "        cases:",
                f"          - when: \"{condition}\"", "            then: yes",
                "        else: no",
                "      yes:", "        kind: code", "        output: str", "        code: |",
                "          return 'yes'",
                "      no:", "        kind: code", "        output: str", "        code: |",
                "          return 'no'",
                "    output: ${yes.output | no.output}",
            ]
    lines += [
        "nodes:", "  top:", "    kind: call", "    call: c0", "    input:",
        "      s: ${input.s}", "output: ${top.output}",
    ]

    flow = _with_less_stack(load_flow, "\n".join(lines) + "\n")

    assert _with_less_stack(run_flow, flow, {"s": "a"}).output == "yes"
    assert _with_less_stack(run_flow, flow, {"s": "b"}).output == "no"


def test_learnable_run_at_the_expression_maximum_logs_and_resumes(set_defaults, tmp_path):
    levels = _maximum("MAX_VALUE_NESTING")
    set_defaults(MAX_VALUE_NESTING=levels)
    text = (
        "id: deep-call\nname: deep_call\nlong_term_memory: test\ninput:\n  s: !str\n"
        "defs:\n  echo:\n    input:\n      s: !str\n    nodes: {}\n    output: ${input.s}\n"
        "nodes:\n  outer:\n    kind: call\n    call: echo\n    input:\n"
        "      s: ${" + _upper(levels, "input.s") + "}\n"
        "output: ${outer.output}\n"
    )
    flow = _with_less_stack(load_flow, text)
    backend = _Memory()
    path = tmp_path / "deep.jsonl"

    result = _with_less_stack(
        run_flow, flow, {"s": "a"}, log=path, learnable=True, memory={"test": backend}
    )
    assert result.status == "succeeded" and result.output == "A"
    replayed = _with_less_stack(resume_flow, flow, log=path, memory={"test": backend})

    assert replayed.status == "succeeded" and replayed.output == "A"


@pytest.mark.parametrize("num_workers", [0, 2])
def test_call_nesting_maximum_runs_with_less_stack(set_defaults, num_workers):
    levels = _maximum("MAX_CALL_NESTING")
    set_defaults(MAX_CALL_NESTING=levels)

    flow = _with_less_stack(load_flow, _defs_chain(levels))
    result = _with_less_stack(run_flow, flow, {"x": "hi"}, num_workers=num_workers)

    assert result.status == "succeeded" and result.output == "hi"


def test_type_nesting_maximum_parses_with_less_stack(set_defaults):
    levels = _maximum("MAX_TYPE_NESTING")
    set_defaults(MAX_TYPE_NESTING=levels)
    text = "list(" * levels + "int" + ")" * levels

    _with_less_stack(parse_type, text)
    with pytest.raises(TypeCheckError, match=f"at most {levels} levels"):
        parse_type("list(" + text + ")")


def _union_chain(levels: int) -> tuple[str, object]:
    """Return `(typedefs, reply)` for `U<levels>`, an `Optional(Union(...))` at each level.

    This is the costliest shape per level for structured-output schema generation.
    """
    lines = [
        "typedefs:",
        "  A1: {tag: Literal(a), k: int}",
        "  B1: {tag: Literal(b), n: int}",
        "  U1: Union(A1, B1)",
    ]
    reply: object = 1
    for i in range(2, levels + 1):
        lines += [
            f"  A{i}: {{tag: Literal(a), k: 'Optional(U{i - 1})'}}",
            f"  B{i}: {{tag: Literal(b), n: int}}",
            f"  U{i}: Union(A{i}, B{i})",
        ]
    for _ in range(levels):
        reply = {"tag": "a", "k": reply}
    return "\n".join(lines) + "\n", reply


def _deep_agent_flow(levels: int, llm_config: str) -> tuple[str, object]:
    """Return `(flow, reply)`: an AGENT with the worst-shape output 50 calls deep.

    `llm_config` holds the AGENT's `llm_config:` entries, one `key: value` per line.
    """
    typedefs, reply = _union_chain(levels)
    config = "".join(f"          {line}\n" for line in llm_config.splitlines())
    leaf = (
        "  leaf:\n    input:\n      x: !str\n    nodes:\n      writer:\n"
        "        kind: agent\n        prompt: hi\n        retries: 0\n"
        f"        llm_config:\n{config}        output: U{levels}\n"
        "    output: ${writer.output}\n"
    )
    chain = _defs_chain(49, last_calls="leaf")
    return chain.replace("defs:\n", typedefs + "defs:\n" + leaf, 1), reply


class _JsonReply:
    """A chat model with no structured binding that answers with `reply` as JSON."""

    def __init__(self, reply):
        self.reply = reply

    def invoke(self, _messages):
        return AIMessage(content=json.dumps(self.reply))


@pytest.mark.parametrize("num_workers", [0, 2])
def test_agent_output_nesting_maximum_runs_with_less_stack_prompted(
    set_defaults, monkeypatch, num_workers
):
    levels = _maximum("MAX_AGENT_OUTPUT_NESTING")
    set_defaults(MAX_AGENT_OUTPUT_NESTING=levels)
    # This model has no native structured output, so the reply is prompted for.
    text, reply = _deep_agent_flow(levels, "provider: openai\nmodel: no-structured-sentinel")
    # A union root is wrapped in a `value` field.
    monkeypatch.setattr(
        llm_clients, "model_from_config", lambda _config: _JsonReply({"value": reply})
    )

    flow = _with_less_stack(load_flow, text)
    result = _with_less_stack(run_flow, flow, {"x": "hi"}, num_workers=num_workers)

    assert result.status == "succeeded", result.failure
    assert result.output == reply


@pytest.mark.parametrize("num_workers", [0, 2])
def test_agent_output_nesting_maximum_runs_with_less_stack_on_azure(
    set_defaults, monkeypatch, num_workers
):
    pytest.importorskip("langchain_openai")
    from tests.test_structured_open_schema import _ToolCallServer, _azure

    levels = _maximum("MAX_AGENT_OUTPUT_NESTING")
    set_defaults(MAX_AGENT_OUTPUT_NESTING=levels)
    monkeypatch.setenv("OPENAI_API_VERSION", "2025-03-01-preview")
    monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)
    # langchain sends `gpt-4` structured output as a tool call, after the client's
    # `has_open_schema` check walks the schema. The deployment must stay `gpt-4`: a
    # `gpt-4o` deployment takes the strict json_schema branch, which `_ToolCallServer`
    # doesn't answer.
    text, reply = _deep_agent_flow(
        levels, "provider: azure\nmodel: gpt-4\nbase_url: https://x.openai.azure.com/"
    )
    server = _ToolCallServer({"value": reply})
    with httpx.Client(transport=httpx.MockTransport(server)) as http_client:
        monkeypatch.setattr(
            llm_clients,
            "model_from_config",
            lambda _config: _azure("gpt-4", http_client=http_client, max_retries=0),
        )
        flow = _with_less_stack(load_flow, text)
        result = _with_less_stack(run_flow, flow, {"x": "hi"}, num_workers=num_workers)

    assert result.status == "succeeded", result.failure
    assert result.output == reply
    (body,) = server.bodies
    assert "tools" in body


def test_agent_output_nesting_maximum_loads_under_the_deepest_import_chain(tmp_path: Path):
    # The worst accepted load: 50 imports whose last file calls a 50-level chain, here
    # ending in an AGENT whose output schema is built at load.
    levels = _maximum("MAX_AGENT_OUTPUT_NESTING")
    texts = _chain_texts(50, calls=False)
    texts["u49"], _ = _deep_agent_flow(levels, "provider: openai\nmodel: gpt-5.5")
    _write_files(tmp_path, texts)

    with _stack_headroom(900):
        load_flow(_importing("root", {"first": "u0"}), search_paths=[tmp_path])
