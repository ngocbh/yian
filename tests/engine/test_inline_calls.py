"""Inline `call(...)` directive calls — end-to-end.

An inline call is a whole-value `call(<flow-id>, kw=...)` directive that DESUGARS at
load into an anonymous `call` node, with the host binding rewritten to
`${<synth>.output}` — pure sugar, no new runtime kind. Two layers:

- `compose.desugar.inline_calls.desugar_call_directives` (pure string→data
  recognition) — exercised in `test_call_directive.py`.
- `compose.desugar.inline_calls.desugar_inline_calls` + the loader wiring — exercised
  end-to-end (load + run, edge inference, defs callee, flow output) in the loader-wiring
  block below.
- the `${item}`-capture + synth-id-collision guards — the guards block.
"""

import pytest
# --------------------------------------------------------------------------- #
# `desugar_inline_calls` + the loader wiring (end-to-end load + run).
# CODE-only children (Ollama-free), resolver-free via in-file `defs:`.
# --------------------------------------------------------------------------- #

from yian.compose import LoadError, load_flow  # noqa: E402
from yian.run import run_flow  # noqa: E402

# An in-file `enrich` def that echoes its `topic` — the callee for the inline calls.
_ENRICH_DEF = """
defs:
  enrich:
    input:
      topic: str
    nodes:
      x:
        kind: code
        code: tests.engine._compose_codefns:echo
        input:
          topic: ${input.topic}
        output: str
    output: ${x.output}
"""


def test_inline_call_in_node_input_runs():
    text = f"""
id: ic1
name: ic1
input:
  topic: str
{_ENRICH_DEF}
nodes:
  use:
    kind: code
    code: tests.engine._compose_codefns:echo
    input:
      topic: call(enrich, topic=${{input.topic}})
    output: str
output: ${{use.output}}
"""
    loaded = load_flow(text)
    assert "__call_0" in loaded.compiled.nodes  # the inline call became a synth node
    result = run_flow(loaded, {"topic": "ACME"})
    assert result.status == "succeeded"
    assert result.output == "ACME"


def test_inline_call_in_human_input_synthesizes_and_wires_call():
    text = f"""
id: ic_human_input
name: ic_human_input
input:
  topic: str
{_ENRICH_DEF}
nodes:
  ask:
    kind: human_input
    input:
      context: call(enrich, topic=${{input.topic}})
    prompt: "Review ${{context}}"
    output: str
output: ${{ask.output}}
"""
    loaded = load_flow(text)
    edges = {(edge.from_, edge.to) for edge in loaded.compiled.edges}

    assert "__call_0" in loaded.compiled.nodes
    assert ("__call_0", "ask") in edges


def test_inline_call_in_loop_input_synthesizes_wires_and_runs():
    text = """
id: ic_loop
name: ic_loop
input:
  start: int
defs:
  seed:
    input:
      n: int
    nodes:
      echo:
        kind: code
        code: tests.engine._compose_codefns:echo
        input:
          topic: ${input.n}
        output: int
    output: ${echo.output}
  body:
    input:
      n: int
    nodes:
      step:
        kind: code
        code: tests.engine._compose_codefns:loop_countdown
        input:
          n: ${input.n}
        output:
          n: int
    output: ${step.output}
nodes:
  repeat:
    kind: loop
    call: body
    input:
      n: call(seed, n=${input.start})
    times: 1
output: ${repeat.output}
"""
    loaded = load_flow(text)
    edges = {(edge.from_, edge.to) for edge in loaded.compiled.edges}

    assert "__call_0" in loaded.compiled.nodes
    assert ("__call_0", "repeat") in edges
    result = run_flow(loaded, {"start": 3})
    assert result.status == "succeeded"
    assert result.output == {"n": 2}


def test_inline_call_literal_arg_runs():
    # a bare literal arg (string) flows like the named form's `input: {topic: HARD}`.
    text = f"""
id: ic_lit
name: ic_lit
input:
  topic: str
{_ENRICH_DEF}
nodes:
  use:
    kind: code
    code: tests.engine._compose_codefns:echo
    input:
      topic: call(enrich, topic="HARDCODED")
    output: str
output: ${{use.output}}
"""
    result = run_flow(load_flow(text), {"topic": "ACME"})
    assert result.output == "HARDCODED"


def test_inline_call_fan_in_infers_edges():
    text = f"""
id: ic_fanin
name: ic_fanin
input:
  topic: str
{_ENRICH_DEF}
nodes:
  y:
    kind: code
    code: tests.engine._compose_codefns:echo
    input:
      topic: ${{input.topic}}
    output: str
  use:
    kind: code
    code: tests.engine._compose_codefns:echo
    input:
      topic: call(enrich, topic=${{y.output}})
    output: str
output: ${{use.output}}
"""
    loaded = load_flow(text)
    edges = {(e.from_, e.to) for e in loaded.compiled.edges}
    assert ("y", "__call_0") in edges          # y -> synth call (the fan-in)
    assert ("__call_0", "use") in edges        # synth call -> consumer
    result = run_flow(loaded, {"topic": "ACME"})
    assert result.status == "succeeded"
    assert result.output == "ACME"


def test_nested_inline_calls_run():
    text = """
id: ic_nested
name: ic_nested
input:
  topic: str
defs:
  inner:
    input:
      topic: str
    nodes:
      x:
        kind: code
        code: tests.engine._compose_codefns:echo
        input:
          topic: ${input.topic}
        output: str
    output: ${x.output}
  outer:
    input:
      topic: str
    nodes:
      x:
        kind: code
        code: tests.engine._compose_codefns:echo
        input:
          topic: ${input.topic}
        output: str
    output: ${x.output}
nodes:
  use:
    kind: code
    code: tests.engine._compose_codefns:echo
    input:
      topic: call(outer, topic=call(inner, topic=${input.topic}))
    output: str
output: ${use.output}
"""
    loaded = load_flow(text)
    assert "__call_0" in loaded.compiled.nodes  # inner (minted first)
    assert "__call_1" in loaded.compiled.nodes  # outer
    result = run_flow(loaded, {"topic": "ACME"})
    assert result.status == "succeeded"
    assert result.output == "ACME"


def test_inline_call_in_flow_output_runs():
    text = f"""
id: ic_out
name: ic_out
input:
  topic: str
{_ENRICH_DEF}
nodes:
  seed:
    kind: code
    code: tests.engine._compose_codefns:echo
    input:
      topic: ${{input.topic}}
    output: str
output: call(enrich, topic=${{seed.output}})
"""
    loaded = load_flow(text)
    assert "__call_0" in loaded.compiled.nodes
    result = run_flow(loaded, {"topic": "ACME"})
    assert result.status == "succeeded"
    assert result.output == "ACME"


def test_inline_call_resolves_through_external_resolver():
    external = """
id: ext_flow
name: ext_flow
input:
  topic: str
nodes:
  x:
    kind: code
    code: tests.engine._compose_codefns:echo
    input:
      topic: ${input.topic}
    output: str
output: ${x.output}
"""
    text = """
id: ic_ext
name: ic_ext
input:
  topic: str
uses:
  ext_flow: ext_flow
nodes:
  use:
    kind: code
    code: tests.engine._compose_codefns:echo
    input:
      topic: call(ext_flow, topic=${input.topic})
    output: str
output: ${use.output}
"""

    def resolver(flow_id, version=None):
        assert flow_id == "ext_flow"
        return load_flow(external)

    loaded = load_flow(text, child_resolver=resolver)
    result = run_flow(loaded, {"topic": "ACME"})
    assert result.status == "succeeded"
    assert result.output == "ACME"


# --------------------------------------------------------------------------- #
# Guards: `${item}` capture + reserved synth-id prefix + malformed call.
# --------------------------------------------------------------------------- #


def test_inline_call_capturing_item_is_loud():
    # an inline call lifts to a top-level synth node with no map-element scope, so an
    # arg reading ${item} is rejected — use a named `call` node instead.
    text = """
id: ic_item
name: ic_item
input:
  topics: list[str]
defs:
  one:
    input:
      topic: str
    nodes:
      x:
        kind: code
        code: tests.engine._compose_codefns:echo
        input:
          topic: ${input.topic}
        output: str
    output: ${x.output}
  wrap:
    input:
      t: str
    nodes:
      x:
        kind: code
        code: tests.engine._compose_codefns:echo
        input:
          topic: ${input.t}
        output: str
    output: ${x.output}
nodes:
  each:
    kind: map
    call: one
    over: ${input.topics}
    input:
      topic: call(wrap, t=${item})
output: ${each.output}
"""
    with pytest.raises(LoadError) as exc:
        load_flow(text)
    msg = str(exc.value)
    assert "item" in msg
    assert "named `call`" in msg
    assert "`input:` may read" in msg


def test_user_node_with_synth_prefix_is_loud():
    text = """
id: ic_collide
name: ic_collide
input:
  topic: str
nodes:
  __call_0:
    kind: code
    code: tests.engine._compose_codefns:echo
    input:
      topic: ${input.topic}
    output: str
output: ${__call_0.output}
"""
    with pytest.raises(LoadError) as exc:
        load_flow(text)
    msg = str(exc.value)
    assert "__call_" in msg
    assert "reserved" in msg.lower()


def test_inline_call_positional_arg_is_loud_and_located():
    text = f"""
id: ic_pos
name: ic_pos
input:
  topic: str
{_ENRICH_DEF}
nodes:
  use:
    kind: code
    code: tests.engine._compose_codefns:echo
    input:
      topic: call(enrich, 30)
    output: str
output: ${{use.output}}
"""
    with pytest.raises(LoadError) as exc:
        load_flow(text)
    assert "keyword" in str(exc.value)
    assert exc.value.diagnostic.frames[-1].range.start_line is not None


def test_inline_call_seed_21_loads():
    from pathlib import Path

    seeds = Path(__file__).resolve().parents[2] / "tests" / "seeds"
    loaded = load_flow((seeds / "21-inline-call.yaml").read_text())
    # the inline call in `summary`'s input desugared into a synth call node.
    assert any(nid.startswith("__call_") for nid in loaded.compiled.nodes)


# --------------------------------------------------------------------------- #
# Named-form parity, located flow-output errors, and supported binding positions:
# TOOL `input:`, mapped-call `over:`, defs bodies, multi-site ids, unknown callees,
# and runtime failure over a synthesized binding.
# --------------------------------------------------------------------------- #

from yian.compose import desugar_inline_calls  # noqa: E402
from yian.compose.parse.file import ToolDescriptor  # noqa: E402


def test_quoted_interpolated_arg_strips_quotes_like_named_form():
    # a quoted arg with interpolation `"hi ${name}"` unwraps to the template `hi ${name}`
    # (quotes stripped), matching the named form's YAML quoted scalar — no literal quotes.
    text = f"""
id: ic_qi
name: ic_qi
input:
  name: str
{_ENRICH_DEF}
nodes:
  use:
    kind: code
    code: tests.engine._compose_codefns:echo
    input:
      topic: call(enrich, topic="hi ${{input.name}}")
    output: str
output: ${{use.output}}
"""
    result = run_flow(load_flow(text), {"name": "ACME"})
    assert result.status == "succeeded"
    assert result.output == "hi ACME"  # NOT '"hi ACME"' — no spurious quote chars


def test_malformed_inline_call_in_outputs_is_located():
    text = f"""
id: ic_outloc
name: ic_outloc
input:
  topic: str
{_ENRICH_DEF}
nodes:
  seed:
    kind: code
    code: tests.engine._compose_codefns:echo
    input:
      topic: ${{input.topic}}
    output: str
output: call(enrich, 30)
"""
    with pytest.raises(LoadError) as exc:
        load_flow(text)
    assert "keyword" in str(exc.value)
    assert exc.value.diagnostic.frames[-1].range.start_line is not None


def test_inline_call_in_tool_inputs_desugars():
    # the ToolDescriptor's internal `inputs` binding field desugars too. Unit-level
    # (a real TOOL run needs a registered tool); asserts the synth node + rewrite.
    descriptors = {
        "t": ToolDescriptor(id="t", tool_id="some_tool", inputs={"q": "call(enrich, x=${input.k})"})
    }
    new_descriptors, _, _ = desugar_inline_calls(descriptors, "${t.output}")
    assert new_descriptors["t"].inputs["q"] == "${__call_0.output}"
    synth = new_descriptors["__call_0"]
    assert synth.call == "enrich"
    assert synth.inputs == {"x": "${input.k}"}
    assert synth.over is None


def test_inline_call_in_mapped_call_over_runs():
    text = """
id: ic_over
name: ic_over
input:
  topics: list[str]
defs:
  one:
    input:
      topic: str
    nodes:
      x:
        kind: code
        code: tests.engine._compose_codefns:echo
        input:
          topic: ${input.topic}
        output: str
    output: ${x.output}
  producelist:
    input:
      xs: list[str]
    nodes:
      x:
        kind: code
        code: tests.engine._compose_codefns:echo
        input:
          topic: ${input.xs}
        output: list[str]
    output: ${x.output}
nodes:
  each:
    kind: map
    call: one
    over: call(producelist, xs=${input.topics})
    input:
      topic: ${item}
output: ${each.output}
"""
    loaded = load_flow(text)
    edges = {(e.from_, e.to) for e in loaded.compiled.edges}
    assert ("__call_0", "each") in edges  # the synth-over producer -> the mapped call
    result = run_flow(loaded, {"topics": ["A", "B"]})
    assert result.status == "succeeded"
    assert result.output == ["A", "B"]


def test_two_host_sites_mint_unique_ids_and_both_run():
    text = f"""
id: ic_two
name: ic_two
input:
  topic: str
{_ENRICH_DEF}
nodes:
  a:
    kind: code
    code: tests.engine._compose_codefns:echo
    input:
      topic: call(enrich, topic=${{input.topic}})
    output: str
  b:
    kind: code
    code: tests.engine._compose_codefns:echo
    input:
      topic: call(enrich, topic=${{input.topic}})
    output: str
output:
  ra: ${{a.output}}
  rb: ${{b.output}}
"""
    loaded = load_flow(text)
    assert {"__call_0", "__call_1"} <= set(loaded.compiled.nodes)  # distinct, no collision
    result = run_flow(loaded, {"topic": "ACME"})
    assert result.status == "succeeded"
    assert result.output == {"ra": "ACME", "rb": "ACME"}


def test_inline_call_inside_defs_body_runs():
    # the desugar runs for def bodies too (shared _assemble); the def's synth ids live in
    # their own namespace. Here `enrich`'s own node hosts an inline call to `helper`.
    text = """
id: ic_defbody
name: ic_defbody
input:
  topic: str
defs:
  helper:
    input:
      topic: str
    nodes:
      x:
        kind: code
        code: tests.engine._compose_codefns:echo
        input:
          topic: ${input.topic}
        output: str
    output: ${x.output}
  enrich:
    input:
      topic: str
    nodes:
      y:
        kind: code
        code: tests.engine._compose_codefns:echo
        input:
          topic: call(helper, topic=${input.topic})
        output: str
    output: ${y.output}
nodes:
  go:
    kind: call
    call: enrich
    input:
      topic: ${input.topic}
output: ${go.output}
"""
    result = run_flow(load_flow(text), {"topic": "ACME"})
    assert result.status == "succeeded"
    assert result.output == "ACME"


def test_inline_call_unknown_callee_is_loud():
    text = """
id: ic_unk
name: ic_unk
input:
  topic: str
nodes:
  use:
    kind: code
    code: tests.engine._compose_codefns:echo
    input:
      topic: call(nope, x=${input.topic})
    output: str
output: ${use.output}
"""
    with pytest.raises(LoadError) as exc:
        load_flow(text)
    msg = str(exc.value)
    assert "nope" in msg
    assert "uses:" in msg


def test_inline_call_e06_type_mismatch_is_loud():
    # a synth call binding flows through the same cross-flow type check as a named call.
    text = """
id: ic_e06
name: ic_e06
input:
  topic: str
defs:
  enrich:
    input:
      topic: str
    nodes:
      x:
        kind: code
        code: tests.engine._compose_codefns:echo
        input:
          topic: ${input.topic}
        output: str
    output: ${x.output}
nodes:
  num:
    kind: code
    code: tests.engine._compose_codefns:echo
    input:
      topic: ${input.topic}
    output: int
  use:
    kind: code
    code: tests.engine._compose_codefns:echo
    input:
      topic: call(enrich, topic=${num.output})
    output: str
output: ${use.output}
"""
    with pytest.raises(LoadError) as exc:
        load_flow(text)
    msg = str(exc.value)
    assert "__call_0" in msg
    assert "child expects" in msg  # the cross-flow type-check message
