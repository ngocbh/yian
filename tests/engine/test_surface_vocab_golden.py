"""Surface-vocabulary golden tests for the singular ref syntax.

Each test asserts ONE contract of the new surface vocabulary: section keywords
(`input:`/`output:`), node-first ref strings (`${<node>.output}`, `${input.k}`),
and the boundary ids / config carriers that came with them.
"""

from __future__ import annotations

import importlib
import pytest

from agent_composer.compose import load_flow
from agent_composer.run import run_flow
from agent_composer.compose.helpers.errors import LoadError


# --------------------------------------------------------------------------- #
# Section keywords + resolver heads
# --------------------------------------------------------------------------- #


# parser dual-arm model_validator normalizes both inputs:/input:
def test_section_keywords_load() -> None:
    """Both `inputs:`/`outputs:` (legacy) and `input:`/`output:` (new) load identically."""
    legacy = """
id: f
name: f
input:
  x: str
nodes:
  n:
    kind: code
    input:
      v: ${input.x}
    output: str
    code: tests.engine._fakes:passthrough
output:
  result: ${n.output}
"""
    loaded = load_flow_from_string(legacy)
    rec = run_flow(loaded, {"x": "hi"})
    assert rec.status == "succeeded"
    assert rec.output == "hi"


# pool.resolve node-id arm + singular input arm
def test_resolver_node_first_ref() -> None:
    """`${<node>.output.k}` resolves to `store[<node>][k]` — the new node-first head."""
    from agent_composer.typesys.pool import VariablePool

    pool = VariablePool()
    pool.set("score", {"rating": 0.7, "rationale": "ok"})
    assert pool.resolve("score", ["output", "rating"]) == 0.7
    assert pool.resolve("score", ["output", "rationale"]) == "ok"
    # 2-segment whole-value form: ${<node>.output} returns the bound value.
    assert pool.resolve("score", ["output"]) == {"rating": 0.7, "rationale": "ok"}


def test_resolver_singular_input() -> None:
    """`${input.k}` resolves to `store[start_id][k]` — singular spelling, alias of `inputs`."""
    from agent_composer.typesys.pool import VariablePool

    pool = VariablePool(start_id="__start__")
    pool.set("__start__", {"topic": "META", "window": 30})
    assert pool.resolve("input", ["topic"]) == "META"
    assert pool.resolve("input", ["window"]) == 30


# _rens_internal emits the new shape (singular-only)
def test_namespace_ref_node_first() -> None:
    """Child-cloning rewrites internal `${<X>.output}` and `${input.k}` to namespaced
    node-first references."""
    from agent_composer.compile.expand import _rens_internal

    # ${<X>.output} (singular) → namespaced node-first
    assert _rens_internal("${foo.output}", callsite="each#0") == "${each#0/foo.output}"
    assert _rens_internal("${foo.output.bar}", callsite="each#0") == "${each#0/foo.output.bar}"
    # ${input.k} → namespaced START_ID's output
    assert (
        _rens_internal("${input.x}", callsite="each#0")
        == "${each#0/__start__.output.x}"
    )


# --------------------------------------------------------------------------- #
# Section keywords + ref-string constructors
# --------------------------------------------------------------------------- #


# every internal constructor emits the new shape
def test_constructors_emit_new_shape() -> None:
    """Every ref-string a loader/expand mints uses `<node>.output` / `input.k`, not the old shape."""
    # Load a flow that exercises every constructor site (defs auto-wire, inline call, case, MAP).
    src = """
id: f
name: f
input:
  x: str
nodes:
  echo:
    kind: code
    input:
      v: ${input.x}
    output: str
    code: tests.engine._fakes:passthrough
output:
  result: ${echo.output}
"""
    loaded = load_flow_from_string(src)
    # Every wiring entry must be the new shape.
    for nid, wiring in loaded.compiled.wiring.items():
        for param, source in wiring.items():
            if not isinstance(source, str):
                continue
            assert "${outputs." not in source, f"{nid}.{param}: legacy outputs head: {source!r}"
            assert "${inputs." not in source, f"{nid}.{param}: legacy inputs head: {source!r}"


# expand_case_outputs rewrite
def test_case_value_shorthand_node_first() -> None:
    """`${<case>.output.field}` coalesces over branch targets — the case-shorthand expansion."""
    src = """
id: f
name: f
input:
  x: int
nodes:
  gate:
    kind: case
    cases:
      - when: "${input.x} > 0"
        then: pos
    else: neg
  pos:
    kind: code
    input:
      v: ${input.x}
    output:
      tag: str
      val: int
    code: tests.engine._fakes:tagged_pos
  neg:
    kind: code
    input:
      v: ${input.x}
    output:
      tag: str
      val: int
    code: tests.engine._fakes:tagged_neg
output:
  result: ${gate.output.tag}
"""
    loaded = load_flow_from_string(src)
    rec_pos = run_flow(loaded, {"x": 5})
    rec_neg = run_flow(loaded, {"x": -5})
    assert rec_pos.output == "pos"
    assert rec_neg.output == "neg"


# --------------------------------------------------------------------------- #
# Seeds use the new surface
# --------------------------------------------------------------------------- #


# every seed YAML uses the new singular surface
def test_all_seeds_use_new_shape() -> None:
    """Every `tests/seeds/*.yaml` uses `input:`/`output:` sections and `${input.X}`/`${<node>.output.X}` refs."""
    import tests.seeds
    from pathlib import Path

    seeds_dir = Path(tests.seeds.__file__).parent
    for yaml_path in seeds_dir.rglob("*.yaml"):
        text = yaml_path.read_text()
        # Section keywords: forbid plural at any indentation.
        for line in text.splitlines():
            stripped = line.lstrip()
            assert not stripped.startswith("inputs:"), f"{yaml_path}: legacy `inputs:` section"
            assert not stripped.startswith("outputs:"), f"{yaml_path}: legacy `outputs:` section"
        # Ref-string literals: forbid plural heads.
        assert "${inputs." not in text, f"{yaml_path}: legacy ${{inputs.X}} ref"
        assert "${outputs." not in text, f"{yaml_path}: legacy ${{outputs.X}} ref"


# --------------------------------------------------------------------------- #
# Test fixtures use the new surface
# --------------------------------------------------------------------------- #


# every test fixture uses the new shape
def test_all_test_fixtures_use_new_shape() -> None:
    """Every `tests/**/*.py` and inline YAML fixture uses the new ref shape."""
    from pathlib import Path

    tests_root = Path(__file__).parent.parent
    offenders: list[str] = []
    for py_path in tests_root.rglob("*.py"):
        if py_path.name == "test_surface_vocab_golden.py":
            continue  # this file mentions the OLD shape literals on purpose, ignore
        text = py_path.read_text()
        if "${inputs." in text or "${outputs." in text:
            offenders.append(str(py_path.relative_to(tests_root)))
    assert not offenders, f"Legacy ref-strings remain in: {offenders}"


# --------------------------------------------------------------------------- #
# Node-local pool-head set + END_ID post-assert output injection
# --------------------------------------------------------------------------- #


# `_POOL_HEADS` is singular
def test_pool_heads_singular() -> None:
    """`_POOL_HEADS = {input, system, item}` (no plural); a node-local assert reading `${inputs.X}` raises."""
    from agent_composer.compose.validate.asserts import _POOL_HEADS

    assert _POOL_HEADS == frozenset({"input", "system", "item"})
    # `inputs`/`outputs` are NOT in the pool-head set; they would be caught by the typo hint.
    assert "inputs" not in _POOL_HEADS
    assert "outputs" not in _POOL_HEADS


# END_ID post-asserts route record-scoped (`${output}` injection).
# Previously, an END_ID `${output}` ref silently resolved to None via the pool (no `output`
# head), so any assert with `${output}` silently held. The fix: `${output}` reads
# the END_ID's committed value through an injected `{"output": end_value}` record.
def test_end_post_assert_reads_output() -> None:
    """Direct EndNode with `post_asserts = ["${output} > 0"]` fires correctly through
    eval_node's END_ID branch.

    Before the fix: pool-scoped `first_failing_assert(["${output} > 0"], pool)` silently resolved
    `${output}` to None (the pool has no `output` head), then `None > 0` raised — caught
    as `evaluate_when` -> False -> assert SILENTLY held as failure even on a positive
    output. After the fix: `${output}` reads through the injected `{"output": end_value}`
    record so the comparison evaluates correctly.
    """
    from agent_composer.compile.model import CompiledFlow, END_ID, START_ID, Edge
    from agent_composer.nodes.end import EndNode
    from agent_composer.nodes.start import StartNode
    from agent_composer.runtime.engine import FlowEngine
    from agent_composer.typesys.pool import VariablePool
    from tests.engine._fakes import FuncNode, derive_wiring

    def _make_flow(post_assert: str) -> CompiledFlow:
        start = StartNode(START_ID, input_decls=[])
        n = FuncNode("n", lambda p: 5)
        end = EndNode.record(END_ID, output_names=["result"])
        end.post_asserts = [post_assert]
        nodes = {START_ID: start, "n": n, END_ID: end}
        edges = [
            Edge("e0", START_ID, "n"),
            Edge("e1", "n", END_ID, input_group="result"),
        ]
        wiring = derive_wiring(nodes)
        # END_ID input wiring: `result` <- n's output
        wiring[END_ID] = {"result": "${n.output}"}
        return CompiledFlow.from_parts(nodes, edges, wiring=wiring)

    # Pass-case: `${output} > 0` holds since END_ID value = 5 (single-output → bare).
    g_pass = _make_flow("${output} > 0")
    events_pass = list(FlowEngine(g_pass).run())
    assert type(events_pass[-1]).__name__ == "RunSucceeded", events_pass[-1]
    assert events_pass[-1].output == 5

    # Fail-case: `${output} > 100` fires LOUD (pre-fix would have silently held).
    g_fail = _make_flow("${output} > 100")
    events_fail = list(FlowEngine(g_fail).run())
    assert type(events_fail[-1]).__name__ == "RunFailed", events_fail[-1]
    assert "post-assert failed" in events_fail[-1].error


# --------------------------------------------------------------------------- #
# Python carriers expose `.input`
# --------------------------------------------------------------------------- #


# ComposedFlow/RunResult carry .input
def test_python_carriers_singular() -> None:
    """All Python carriers expose `.input` (singular); `.inputs` no longer exists."""
    from agent_composer.compose.loader import ComposedFlow
    from agent_composer.run import RunResult

    assert "input" in ComposedFlow.__dataclass_fields__
    assert "inputs" not in ComposedFlow.__dataclass_fields__
    # RunResult is a dataclass-style class with `inputs: Dict[str, Any]` field today.
    assert "input" in RunResult.__dataclass_fields__
    assert "inputs" not in RunResult.__dataclass_fields__


# --------------------------------------------------------------------------- #
# No spec/common grab-bag package: ids on the node classes, LLMConfig with the clients
# --------------------------------------------------------------------------- #


def test_no_spec_or_common_package() -> None:
    """Neither `agent_composer.spec` nor `agent_composer.common` exists.
    The boundary ids live on the node classes (StartNode.ID/EndNode.ID,
    re-exported by compile.model); LLMConfig lives with the llm clients."""
    for gone in ("agent_composer.spec", "agent_composer.common"):
        with pytest.raises(ImportError):
            importlib.import_module(gone)

    from agent_composer.compile.model import END_ID, START_ID
    from agent_composer.llm_clients import LLMConfig
    from agent_composer.nodes.end import EndNode
    from agent_composer.nodes.start import StartNode

    assert START_ID == StartNode.ID == "__start__"
    assert END_ID == EndNode.ID == "__end__"
    assert LLMConfig.__module__ == "agent_composer.llm_clients.config"


# --------------------------------------------------------------------------- #
# llm_config end-to-end dict carrier
# --------------------------------------------------------------------------- #


# AgentNode.llm_config carries a plain dict end-to-end
def test_llm_config_dict_carrier() -> None:
    """`AgentNode.llm_config` is a plain dict (or None), not an `LLMConfig` instance."""
    from agent_composer.nodes.agent.node import AgentNode

    src = """
id: f
name: f
input:
  x: str
nodes:
  a:
    kind: agent
    input:
      topic: ${input.x}
    output: str
    prompt: "About ${topic}."
    llm_config:
      provider: anthropic
      model: claude-opus-4-7
output:
  result: ${a.output}
"""
    loaded = load_flow_from_string(src)
    agent_node = loaded.compiled.nodes["a"]
    assert isinstance(agent_node, AgentNode)
    # a plain dict, not an LLMConfig instance.
    assert isinstance(agent_node.llm_config, dict)


# LLMConfig has extra=forbid; a typo'd llm_config: is a LoadError
def test_llm_config_extra_forbid() -> None:
    """A typo in `llm_config:` (e.g. `temparature` instead of `temperature`) fails at LOAD."""
    src = """
id: f
name: f
input:
  x: str
nodes:
  a:
    kind: agent
    input:
      topic: ${input.x}
    output: str
    prompt: "About ${topic}."
    llm_config:
      temparature: 0.5
output:
  result: ${a.output}
"""
    with pytest.raises(LoadError):
        load_flow_from_string(src)


# --------------------------------------------------------------------------- #
# Legacy keywords are rejected after the alias delete
# --------------------------------------------------------------------------- #


# parser model_validator rejects old keywords with a bespoke LoadError
def test_legacy_shape_loud_after_alias_delete() -> None:
    """After the alias delete (parser rejector flip), `inputs:`/`outputs:` is a LoadError
    with the bespoke migration message."""
    legacy = """
id: f
name: f
inputs:
  x: str
nodes:
  n:
    kind: code
    input:
      v: ${input.x}
    output: str
    code: tests.engine._fakes:passthrough
output:
  result: ${n.output}
"""
    with pytest.raises(LoadError, match="rename the section"):
        load_flow_from_string(legacy)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def load_flow_from_string(src: str):
    """Load a flow directly from a YAML string (load_flow already accepts text)."""
    return load_flow(src)
