"""Tests for the desugar phase: flow-level rewrites that run before the build loop.

Each desugar pass rewrites a flow's parsed descriptors into ordinary descriptors +
bindings. A synth node a pass mints locates at the SITE it desugared from — the
`call(...)` directive's own binding line, not the host node header — so its downstream
errors caret the author's actual source. The exception is `adaptive_questions`, whose
synth composer genuinely SHARES the gate's inputs verbatim and so correctly inherits the
gate's `.range`/`field_ranges`.
"""

from agent_composer.compose.desugar.adaptive_questions import desugar_adaptive_questions
from agent_composer.compose.desugar.case_targets import desugar_case_call_targets
from agent_composer.compose.desugar.inline_calls import (
    _synth_id_gen,
    desugar_inline_calls,
)
from agent_composer.compose.parse.nodes import parse_nodes
from agent_composer.compose.source import SourceDocument


def _nodes_reader(text):
    """Return the physical cursor over a flow document's ``nodes:`` mapping."""

    return SourceDocument.parse(text).root.child("nodes")


def test_desugar_inline_calls_locate_at_their_binding_site():
    # A whole-value `call(...)` in a node's `input:` binding lifts to a synth `call` node.
    # The synth CallDescriptor locates at its OWN binding line (the `x:` entry the directive
    # was written on), NOT the host node header — and carries an empty `field_ranges` (it owns
    # no authored fields), so a bad ref inside the directive carets the directive.
    text = (
        "id: f\nname: f\nnodes:\n"
        "  writer:\n    kind: code\n    code: m:f\n"
        "    input:\n      x: call(helper, n=3)\n    output: str\n"
        "output: ${writer.output}\n"
    )
    descriptors = parse_nodes(_nodes_reader(text))
    host = descriptors["writer"]
    new_descriptors, _outputs, _asserts = desugar_inline_calls(
        descriptors, None, next_id=_synth_id_gen()
    )
    synth = [d for nid, d in new_descriptors.items() if nid.startswith("__call_")]
    assert len(synth) == 1
    call_desc = synth[0]
    assert call_desc.call == "helper"
    assert call_desc.range == host.field_ranges[("input", "x")]  # the `x:` binding, not the header
    assert call_desc.range != host.range                         # ... which is strictly the header
    assert call_desc.field_ranges == {}                          # owns no authored fields


def test_desugar_adaptive_questions_preserves_ranges():
    # An adaptive human_input gate lifts a synth composer agent (`<gate>__compose`).
    # The synth agent must inherit the gate node's range/field_ranges (no node_lines map).
    text = (
        "id: f\nname: f\nnodes:\n"
        "  ask:\n    kind: human_input\n"
        "    adaptive_questions:\n      prompt: Ask about ${input.topic}\n"
        "    input:\n      topic: ${input.topic}\n"
        "output: ${ask.output}\n"
    )
    descriptors = parse_nodes(_nodes_reader(text))
    gate = descriptors["ask"]
    new_descriptors = desugar_adaptive_questions(descriptors)
    composer = new_descriptors["ask__compose"]
    assert composer.range == gate.range
    assert composer.field_ranges == gate.field_ranges


def test_desugar_case_call_targets_locate_at_their_target_line():
    # A `case` then:/else: that is a whole-value `call(...)` directive lifts a synth `call`
    # node. That synth node locates at its OWN `then:` line (captured positionally in the case
    # descriptor's `then_ranges`), NOT the host case header, and carries an empty `field_ranges`.
    text = (
        "id: f\nname: f\nnodes:\n"
        "  route:\n    kind: case\n    on: ${score.output}\n"
        "    cases:\n      - when: high\n        then: call(helper, n=3)\n"  # `then:` on line 9
        "    else: cold\n"
        "output: ${route.output}\n"
    )
    descriptors = parse_nodes(_nodes_reader(text))
    host = descriptors["route"]
    new_descriptors = desugar_case_call_targets(descriptors, _synth_id_gen())
    synth = [d for nid, d in new_descriptors.items() if nid.startswith("__call_")]
    assert len(synth) == 1
    call_desc = synth[0]
    assert call_desc.call == "helper"
    assert call_desc.range == host.then_ranges[0]  # the `then:` line, not the case header
    assert call_desc.range != host.range
    assert call_desc.field_ranges == {}            # owns no authored fields
