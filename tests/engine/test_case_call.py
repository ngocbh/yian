"""`case` `then:/else: call(...)` inline-call branch targets.

A branch target may be an inline call (a fresh owned call) instead of a placed node id; it
desugars to a synth `__call_<n>` node and the then:/else: is rewritten to that synth id. Sound
because the case veto skip-floods the non-chosen synth branch. Accepted grammar: a whole-value
`call(...)` directive (a route target is one node id). A `${...}` branch is a value, so a
flow call inside its span is the retired inline `${flow(args)}` form, a LoadError.
"""

import pytest

from yian.compose import LoadError, load_flow
from yian.compose.desugar.inline_bodies import INLINE_FLOW_PREFIX
from yian.run import run_flow

# The retired inline `${flow(args)}` form's message.
_SPAN_FLOW_CALL = r"a flow call 'take' inside \$\{\.\.\.\} is not supported"

# An in-file `take` def (the branch callable): echoes its stance via the `took` CODE fn.
_TAKE_DEF = """
defs:
  take:
    input:
      stance: str
    nodes:
      x:
        kind: code
        code: tests.engine._compose_codefns:took
        input:
          stance: ${input.stance}
        output: str
    output: ${x.output}
"""


def test_non_bare_then_call_is_rejected():
    # a `${...}` branch is a value; a flow call inside its span is the retired inline
    # `${flow(args)}` form -> loud, pointing at the call(...) directive.
    text = f"""
id: bad
name: bad
input:
  score: float
{_TAKE_DEF}
nodes:
  gate:
    kind: case
    output: str
    cases:
      - when: "${{input.score}} >= 0.5"
        then: ${{ take(stance="a") | take(stance="b") }}
    else: call(take, stance="c")
output: ${{gate.output}}
"""
    with pytest.raises(LoadError, match=_SPAN_FLOW_CALL):
        load_flow(text)


def test_inline_binding_and_then_call_share_minter_no_collision():
    # an inline-binding call(...) AND a then: call(...) in one flow must get DISTINCT synth ids.
    text = f"""
id: both
name: both
input:
  score: float
  topic: str
defs:
  take:
    input:
      stance: str
    nodes:
      x:
        kind: code
        code: tests.engine._compose_codefns:took
        input:
          stance: ${{input.stance}}
        output: str
    output: ${{x.output}}
  enrich:
    input:
      topic: str
    nodes:
      y:
        kind: code
        code: tests.engine._compose_codefns:echo
        input:
          topic: ${{input.topic}}
        output: str
    output: ${{y.output}}
nodes:
  use:
    kind: code
    code: tests.engine._compose_codefns:echo
    input:
      topic: call(enrich, topic=${{input.topic}})
    output: str
  gate:
    kind: case
    cases:
      - when: "${{input.score}} >= 0.5"
        then: call(take, stance="pro")
    else: use
output: ${{gate.output}}
"""
    loaded = load_flow(text)
    synth = [nid for nid in loaded.compiled.nodes if nid.startswith("__call_")]
    assert len(synth) == len(set(synth)) == 2  # distinct ids, no collision


# --------------------------------------------------------------------------- #
# The `call(...)` directive spelling of a then:/else: branch target — the only
# supported inline-call branch form (the legacy `${ take(...) }` span is retired).
# --------------------------------------------------------------------------- #


def _flow_directive():
    return f"""
id: cc
name: cc
input:
  score: float
{_TAKE_DEF}
nodes:
  gate:
    kind: case
    cases:
      - when: "${{input.score}} >= 0.5"
        then: call(take, stance="pro")
    else: call(take, stance="con")
output: ${{gate.output}}
"""


def test_then_else_call_directive_desugars_to_synth_branch_nodes():
    loaded = load_flow(_flow_directive())
    # both then: and else: call(...) directives became synth call nodes
    synth = [nid for nid in loaded.compiled.nodes if nid.startswith("__call_")]
    assert len(synth) == 2
    # the gate's control edges target the synth ids (not a placed user node)
    control_targets = {
        e.to for e in loaded.compiled.edges if e.source_handle is not None and e.from_ == "gate"
    }
    assert control_targets == set(synth)


def test_then_call_directive_branch_runs_taken_value():
    out = run_flow(load_flow(_flow_directive()), {"score": 0.9})
    assert out.status == "succeeded"
    assert out.output == "took:pro"


def test_else_call_directive_branch_runs_taken_value():
    out = run_flow(load_flow(_flow_directive()), {"score": 0.2})
    assert out.status == "succeeded"
    assert out.output == "took:con"


def test_non_bare_then_call_directive_is_rejected():
    # a route target must resolve to ONE node id: a coalesce of call(...) directives is loud.
    text = f"""
id: bad
name: bad
input:
  score: float
{_TAKE_DEF}
nodes:
  gate:
    kind: case
    cases:
      - when: "${{input.score}} >= 0.5"
        then: call(take, stance="a") | call(take, stance="b")
    else: call(take, stance="c")
output: ${{gate.output}}
"""
    with pytest.raises(LoadError, match="a branch target must be a node id or a single call"):
        load_flow(text)


@pytest.mark.parametrize("label", ["then", "else"])
@pytest.mark.parametrize(
    "target, problem",
    [
        ('call(take, stance="a", stance="b")', "keyword 'stance' more than once"),
        ("call(take, 30)", "passes a positional argument '30'"),
        ("call(take", "`call(` is never closed"),
        ('call(take, stance="a") tail', "text follows the closing `)`: 'tail'"),
        ('call(take stance="a")', "must start with a flow id (a bare name) and a comma"),
    ],
)
def test_malformed_branch_call_directive_is_a_located_load_error(label, target, problem):
    # an unquoted then:/else: starting with `call(` must be exactly one directive.
    then_target, else_target = (target, "nope") if label == "then" else ("nope", target)
    text = f"""
id: bad
name: bad
input:
  score: float
{_TAKE_DEF}
nodes:
  gate:
    kind: case
    cases:
      - when: "${{input.score}} >= 0.5"
        then: {then_target}
    else: {else_target}
  nope:
    kind: code
    code: tests.engine._compose_codefns:took
    input:
      stance: x
    output: str
output: ${{gate.output}}
"""
    with pytest.raises(LoadError) as caught:
        load_flow(text)
    message = str(caught.value)
    assert f"case node 'gate' {label}:" in message
    assert problem in message
    assert "If you meant the text, quote it:" in message
    line = next(
        number
        for number, source in enumerate(text.splitlines(), start=1)
        if f"{label}: call(" in source
    )
    assert caught.value.diagnostic.frames[-1].range.start_line == line


def test_quoted_malformed_branch_call_is_a_text_value():
    text = f"""
id: ok
name: ok
input:
  score: float
nodes:
  gate:
    kind: case
    output: str
    cases:
      - when: "${{input.score}} >= 0.5"
        then: 'call(take'
    else: "call(take) tail"
output: ${{gate.output}}
"""
    flow = load_flow(text)
    assert run_flow(flow, {"score": 0.9}).output == "call(take"
    assert run_flow(flow, {"score": 0.1}).output == "call(take) tail"


def test_inline_binding_and_then_call_directive_share_minter_no_collision():
    # an inline-binding call(...) AND a then: call(...) in one flow -> DISTINCT synth ids.
    text = f"""
id: both
name: both
input:
  score: float
  topic: str
defs:
  take:
    input:
      stance: str
    nodes:
      x:
        kind: code
        code: tests.engine._compose_codefns:took
        input:
          stance: ${{input.stance}}
        output: str
    output: ${{x.output}}
  enrich:
    input:
      topic: str
    nodes:
      y:
        kind: code
        code: tests.engine._compose_codefns:echo
        input:
          topic: ${{input.topic}}
        output: str
    output: ${{y.output}}
nodes:
  use:
    kind: code
    code: tests.engine._compose_codefns:echo
    input:
      topic: call(enrich, topic=${{input.topic}})
    output: str
  gate:
    kind: case
    cases:
      - when: "${{input.score}} >= 0.5"
        then: call(take, stance="pro")
    else: use
output: ${{gate.output}}
"""
    loaded = load_flow(text)
    synth = [nid for nid in loaded.compiled.nodes if nid.startswith("__call_")]
    assert len(synth) == len(set(synth)) == 2  # distinct ids, no collision


def test_case_output_over_call_directive_then_taken():
    # ${gate.output} over a call(...)-desugared then: — the taken then value comes back.
    out = run_flow(load_flow(_flow_directive()), {"score": 0.9})
    assert out.status == "succeeded"
    assert out.output == "took:pro"


def test_case_output_over_call_directive_else_taken():
    # ${gate.output} over a call(...)-desugared else: — the taken else value comes back.
    out = run_flow(load_flow(_flow_directive()), {"score": 0.2})
    assert out.status == "succeeded"
    assert out.output == "took:con"


def _flow_nested_directive():
    # a NESTED branch target: the then: call(...) has an inner call(...) as a kwarg, so the
    # single target lifts TWO synth nodes (inner-first) — exercises the multi-node lift loop.
    return f"""
id: cc
name: cc
input:
  score: float
{_TAKE_DEF}
nodes:
  gate:
    kind: case
    cases:
      - when: "${{input.score}} >= 0.5"
        then: call(take, stance=call(take, stance="pro"))
    else: call(take, stance="con")
output: ${{gate.output}}
"""


def test_nested_then_call_directive_lifts_two_synth_nodes():
    # the nested then: target lifts BOTH the inner and outer call as synth nodes; the else:
    # lifts one more — three synth call nodes in all, the gate's then-edge on the outer.
    loaded = load_flow(_flow_nested_directive())
    synth = [nid for nid in loaded.compiled.nodes if nid.startswith("__call_")]
    assert len(synth) == len(set(synth)) == 3


def test_nested_then_call_directive_branch_runs_taken_value():
    # inner take(stance="pro") -> "took:pro"; outer take(stance="took:pro") -> "took:took:pro".
    out = run_flow(load_flow(_flow_nested_directive()), {"score": 0.9})
    assert out.status == "succeeded"
    assert out.output == "took:took:pro"


@pytest.mark.parametrize("branch", ["then", "else"])
def test_private_callee_in_case_target_is_rejected_at_branch_before_resolution(
    branch, monkeypatch
):
    private_name = INLINE_FLOW_PREFIX + "c" * 64
    then_target = f"call({private_name})" if branch == "then" else "safe"
    else_target = f"call({private_name})" if branch == "else" else "safe"
    text = f"""id: private_case
name: private_case
nodes:
  safe:
    kind: code
    code: tests.engine._compose_codefns:echo
    input:
      topic: safe
    output: str
  gate:
    kind: case
    cases:
      - when: "true"
        then: {then_target}
    else: {else_target}
output: ${{gate.output}}
"""
    resolved = []

    def resolve(*args, **kwargs):
        resolved.append((args, kwargs))
        raise AssertionError("the private callee reached child resolution")

    monkeypatch.setattr("yian.compose.nodes.call._resolve_child", resolve)

    with pytest.raises(LoadError) as caught:
        load_flow(text)

    expected_line = next(
        number
        for number, source_line in enumerate(text.splitlines(), start=1)
        if private_name in source_line
    )
    assert caught.value.diagnostic.frames[-1].range is not None
    assert caught.value.diagnostic.frames[-1].range.start_line == expected_line
    assert INLINE_FLOW_PREFIX in str(caught.value)
    assert private_name not in str(caught.value)
    assert resolved == []


def test_then_span_flow_call_target_is_rejected():
    # the retired inline `${flow(args)}` case branch is a LoadError pointing at call(...).
    text = f"""
id: bad
name: bad
input:
  score: float
{_TAKE_DEF}
nodes:
  gate:
    kind: case
    output: str
    cases:
      - when: "${{input.score}} >= 0.5"
        then: ${{ take(stance="pro") }}
    else: call(take, stance="con")
output: ${{gate.output}}
"""
    with pytest.raises(LoadError, match=_SPAN_FLOW_CALL):
        load_flow(text)


def test_malformed_branch_call_holding_a_span_offers_no_quoting_fix():
    # quoting a span-bearing call is its own load error, so the message does not suggest it.
    text = f"""
id: bad
name: bad
input:
  score: float
{_TAKE_DEF}
nodes:
  gate:
    kind: case
    cases:
      - when: "${{input.score}} >= 0.5"
        then: call(take, stance=${{input.score}}
    else: call(take, stance="c")
output: ${{gate.output}}
"""
    with pytest.raises(LoadError) as caught:
        load_flow(text)
    message = str(caught.value)
    assert "case node 'gate' then: a branch target must be a node id or a single call" in message
    assert "`call(` is never closed" in message
    assert "If you meant the text" not in message
