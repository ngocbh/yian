"""A too-deep expression is a located load error at every expression site; a long operator chain loads and runs.

Each call, list, unary operator, comparison, nested `${`, and chain of same-precedence
operators is one level, and an expression may nest 100. A deeper one fails the load
with a message naming the limit, located at the field that holds it (an `asserts:`
item at its `- <expr>` line, a prompt at its `prompt:` line). A chain of thousands
of `+` or `*` terms is one level, so it loads, runs, logs, and resumes.
"""

from pathlib import Path

import pytest

from yian.compose import LoadError, load_flow
from yian.run import resume_flow, run_flow
from tests.engine._memory_fakes import _Memory

_SEEDS = Path(__file__).resolve().parents[1] / "seeds"
_TOO_DEEP = "expression nests deeper than 100 levels"
_CHAIN = 3000


def _line_of(text: str, needle: str) -> int:
    """Return the 1-based line number of the first line containing `needle`."""
    return next(
        number
        for number, line in enumerate(text.splitlines(), start=1)
        if needle in line
    )


def _not(n: int) -> str:
    """Return a condition of `n` nested `not`s."""
    return "not " * n + "true"


def _neg(n: int) -> str:
    """Return a number of `n` nested negations."""
    return "-(" * n + "1" + ")" * n


def _upper(n: int, ref: str) -> str:
    """Return `n` nested `upper(...)` calls around `ref`."""
    return "upper(" * n + ref + ")" * n


def _list(n: int) -> str:
    """Return a list literal nested `n` deep."""
    return "[" * n + "1" + "]" * n


def _code_flow(binding: str, *, node_extra: str = "", output: str = "${use.output}", tail: str = "") -> str:
    """Return a flow whose one CODE node binds `topic` to `binding`."""
    return (
        "id: f\nname: f\ninput:\n  x: !int\n  s: !str\n"
        + "nodes:\n  use:\n    kind: code\n    input:\n      topic: " + binding + "\n"
        + "    output: str\n" + node_extra
        + "    code: |\n      return str(inputs['topic'])\n"
        + "output: " + output + "\n" + tail
    )


def _routed(case_fields: str) -> str:
    """Return a flow whose CASE `route` (fields `case_fields`) picks node `yes` or `no`."""
    return (
        "id: f\nname: f\ninput:\n  x: !int\n  s: !str\nnodes:\n  route:\n    kind: case\n"
        + case_fields
        + "    else: no\n"
        + "  yes:\n    kind: code\n    output: str\n    code: |\n      return 'yes'\n"
        + "  no:\n    kind: code\n    output: str\n    code: |\n      return 'no'\n"
        + "output: ${yes.output | no.output}\n"
    )


def _searched(condition: str) -> str:
    """Return a routed flow with one searched branch on `condition`."""
    return _routed("    cases:\n      - when: \"" + condition + "\"\n        then: yes\n")


def _human(n: int) -> str:
    """Return a HUMAN_INPUT flow whose question text holds an `n`-level span."""
    return (
        "id: f\nname: f\ninput:\n  proj: !str\nnodes:\n  ask:\n    kind: human_input\n"
        "    input:\n      proj: ${input.proj}\n    questions:\n"
        "      - question: \"Which framework for ${" + _upper(n, "proj") + "}?\"\n"
        "        header: Framework\n"
        "output: ${ask.output}\n"
    )


def _seed(name: str) -> str:
    return (_SEEDS / name).read_text()


_CALL_DEF = (
    "defs:\n  g:\n    input:\n      d: !int\n    nodes:\n      run:\n        kind: code\n"
    "        output: str\n        code: |\n          return 'x'\n    output: ${run.output}\n"
)

# site -> builder of (flow text, needle on the line the error is located at)
_SITES = {
    "searched when": lambda n: (_searched(_not(n)), "- when:"),
    "searched when span": lambda n: (_searched("${" + _not(n) + "}"), "- when:"),
    "case on": lambda n: (
        _routed("    on: \"${" + _neg(n) + "}\"\n    cases:\n      - when: 1\n        then: yes\n"),
        "on:",
    ),
    "whole binding": lambda n: (_code_flow("\"${" + _neg(n) + "}\""), "topic:"),
    "binding in text": lambda n: (_code_flow("\"v=${" + _neg(n) + "}\""), "topic:"),
    "flow output": lambda n: (_code_flow("${input.s}", output="\"${" + _neg(n) + "}\""), "output: \"${"),
    "flow assert": lambda n: (_code_flow("${input.s}", tail="asserts:\n  - " + _not(n) + "\n"), "- not"),
    "node assert": lambda n: (
        _code_flow("${input.s}", node_extra="    asserts:\n      - " + _not(n) + "\n"),
        "- not",
    ),
    "loop until": lambda n: (
        _seed("28-refine-loop.yaml").replace("until: score >= 0.8", "until: " + _not(n)),
        "until: not",
    ),
    "map over": lambda n: (
        _seed("05-call-map.yaml").replace("over: ${input.topics}", "over: \"${input.topics :- " + _list(n) + "}\""),
        "over: \"${",
    ),
    "prompt span": lambda n: (
        _seed("00-hello-agent.yaml").replace("on ${topic}.", "on ${" + _upper(n, "topic") + "}."),
        "prompt: |-",
    ),
    "call argument": lambda n: (
        _code_flow("call(g, d=${" + _neg(n) + "})").replace("nodes:\n", _CALL_DEF + "nodes:\n", 1),
        "topic: call(",
    ),
    "typed binding": lambda n: (_code_flow("!int ${" + _neg(n) + "}"), "topic:"),
    "human question": lambda n: (_human(n), "- question:"),
}


@pytest.mark.parametrize("depth", [101, _CHAIN])
@pytest.mark.parametrize("site", list(_SITES))
def test_a_too_deep_expression_is_a_located_load_error(site: str, depth: int) -> None:
    text, needle = _SITES[site](depth)
    with pytest.raises(LoadError) as caught:
        load_flow(text, search_paths=[str(_SEEDS)])
    assert _TOO_DEEP in str(caught.value)
    assert caught.value.diagnostic.frames[-1].range.start_line == _line_of(text, needle)


def test_thousands_of_nested_spans_are_a_located_load_error() -> None:
    text = _searched("${" * _CHAIN + "true" + "}" * _CHAIN)
    with pytest.raises(LoadError) as caught:
        load_flow(text)
    assert "`${...}` can't appear inside an expression" in str(caught.value)
    assert caught.value.diagnostic.frames[-1].range.start_line == _line_of(text, "- when:")


def test_a_deep_case_on_match_value_is_text_and_loads() -> None:
    # A match value under `on:` is compared as written, never evaluated.
    text = _routed(
        "    on: ${input.s}\n    cases:\n      - when: \"${" + _neg(_CHAIN) + "}\"\n        then: yes\n"
    )
    flow = load_flow(text)
    assert run_flow(flow, {"x": 1, "s": "a"}).output == "no"


def _sum(n: int, term: str = "input.x") -> str:
    return " + ".join([term] * n)


def test_a_long_sum_binding_loads_and_runs() -> None:
    text = (
        "id: f\nname: f\ninput:\n  x: !int\nnodes:\n  use:\n    kind: code\n    input:\n"
        "      total: ${" + _sum(_CHAIN) + "}\n    output: int\n"
        "    code: |\n      return inputs['total']\noutput: ${use.output}\n"
    )
    result = run_flow(load_flow(text), {"x": 7})
    assert result.status == "succeeded" and result.output == _CHAIN * 7


@pytest.mark.parametrize(("x", "taken"), [(1, "yes"), (-1, "no")])
def test_a_long_sum_condition_routes(x: int, taken: str) -> None:
    flow = load_flow(_searched(_sum(_CHAIN) + " > 0"))
    assert run_flow(flow, {"x": x, "s": "a"}).output == taken


def test_a_long_product_binding_runs() -> None:
    text = (
        "id: f\nname: f\ninput:\n  x: !int\nnodes:\n  use:\n    kind: code\n    input:\n"
        "      total: ${input.x * " + " * ".join(["1"] * _CHAIN) + "}\n    output: int\n"
        "    code: |\n      return inputs['total']\noutput: ${use.output}\n"
    )
    result = run_flow(load_flow(text), {"x": 5})
    assert result.status == "succeeded" and result.output == 5


def _concat_flow(terms: int):
    """Return a learnable flow passing a `terms`-term list concatenation through a CALL."""
    return load_flow(
        "id: long-concat\nname: long_concat\nlong_term_memory: test\ninput:\n  values: !list(str)\n"
        "defs:\n  identity:\n    input:\n      values: !list(str)\n    nodes: {}\n    output: ${input.values}\n"
        "nodes:\n  outer:\n    kind: call\n    call: identity\n    input:\n"
        "      values: ${" + _sum(terms, "input.values") + "}\n"
        "output: ${outer.output}\n"
    )


def test_a_learnable_run_of_a_long_list_chain_logs_and_resumes(tmp_path) -> None:
    flow = _concat_flow(_CHAIN)
    backend = _Memory()
    path = tmp_path / "long.jsonl"
    result = run_flow(flow, {"values": ["v"]}, log=path, learnable=True, memory={"test": backend})
    assert result.status == "succeeded" and result.output == ["v"] * _CHAIN
    replayed = resume_flow(flow, log=path, memory={"test": backend})
    assert replayed.status == "succeeded" and replayed.output == result.output


def test_feedback_on_a_list_chain_passes_back(tmp_path) -> None:
    # Backward over a list chain costs about the square of its length, so this chain is short.
    flow = _concat_flow(24)
    backend = _Memory()
    result = run_flow(flow, {"values": ["v"]}, log=tmp_path / "short.jsonl", learnable=True,
                      memory={"test": backend})
    assert result.status == "succeeded" and result.output == ["v"] * 24
    assert result.backward("good").status == "succeeded"


_SUM_DEF = (
    "defs:\n  add:\n    input:\n      x: !int\n    nodes:\n      use:\n        kind: code\n"
    "        input:\n          total: ${" + _sum(_CHAIN) + "}\n        output: int\n"
    "        code: |\n          return inputs['total']\n    output: ${use.output}\n"
)


def test_a_long_chain_inside_a_call_runs() -> None:
    text = (
        "id: f\nname: f\ninput:\n  x: !int\n" + _SUM_DEF
        + "nodes:\n  top:\n    kind: call\n    call: add\n    input:\n      x: ${input.x}\n"
        "output: ${top.output}\n"
    )
    result = run_flow(load_flow(text), {"x": 2})
    assert result.status == "succeeded" and result.output == _CHAIN * 2


def test_a_long_chain_inside_a_map_body_runs() -> None:
    text = (
        "id: f\nname: f\ninput:\n  xs: !list(int)\n" + _SUM_DEF
        + "nodes:\n  each:\n    kind: map\n    over: ${input.xs}\n    call: add\n    input:\n      x: ${item}\n"
        "output: ${each.output}\n"
    )
    result = run_flow(load_flow(text), {"xs": [1, 3]})
    assert result.status == "succeeded" and result.output == [_CHAIN, _CHAIN * 3]


def test_a_condition_at_the_limit_runs_thirty_calls_deep() -> None:
    condition = _upper(99, "input.s") + " == 'A'"  # 99 calls and a comparison
    lines = ["id: f", "name: f", "input:", "  s: !str", "defs:"]
    for i in range(30):
        lines += [f"  c{i}:", "    input:", "      s: !str", "    nodes:"]
        if i < 29:
            lines += ["      sub:", "        kind: call", f"        call: c{i + 1}", "        input:",
                      "          s: ${input.s}", "    output: ${sub.output}"]
        else:
            lines += ["      route:", "        kind: case", "        cases:",
                      f"          - when: \"{condition}\"", "            then: yes", "        else: no",
                      "      yes:", "        kind: code", "        output: str", "        code: |",
                      "          return 'yes'",
                      "      no:", "        kind: code", "        output: str", "        code: |",
                      "          return 'no'",
                      "    output: ${yes.output | no.output}"]
    lines += ["nodes:", "  top:", "    kind: call", "    call: c0", "    input:", "      s: ${input.s}",
              "output: ${top.output}"]
    flow = load_flow("\n".join(lines) + "\n")
    assert run_flow(flow, {"s": "a"}).output == "yes"
    assert run_flow(flow, {"s": "b"}).output == "no"


def test_a_binding_at_the_limit_runs() -> None:
    text = (
        "id: f\nname: f\ninput:\n  s: !str\nnodes:\n  use:\n    kind: code\n    input:\n"
        "      topic: ${" + _upper(100, "input.s") + "}\n    output: str\n"
        "    code: |\n      return inputs['topic']\noutput: ${use.output}\n"
    )
    result = run_flow(load_flow(text), {"s": "a"})
    assert result.status == "succeeded" and result.output == "A"
