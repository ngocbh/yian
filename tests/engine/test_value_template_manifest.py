"""Value templates and their nested typed bindings in flow manifests.

A template encodes apart from every authored literal, a nested binding keeps its
declared type, every part of a template is identity (credential-named slots too, as
for literals), and flows without templates keep their fingerprints byte for byte.
"""

import copy
import json

import pytest

from yian.compile.manifest import flow_fingerprint, flow_manifest
from yian.compile.model import CompiledFlow, Edge, Flow, FlowOutput
from yian.compose import load_flow
from yian.diagnostics import OriginRef
from yian.nodes.code import CodeNode
from yian.nodes.contracts import (
    Binding,
    ContractOrigins,
    Parameter,
    ValueContract,
    ValueTemplate,
)
from yian.nodes.end import EndNode
from yian.nodes.start import StartNode
from yian.typesys.values import Type, ValueKind


INTEGER = Type.scalar(ValueKind.INTEGER)


def _fragment(**wiring) -> Flow:
    """Build a START -> CODE -> END fragment whose CODE node takes ``wiring``."""
    start = StartNode("entry", params=[])
    body = CodeNode("body", code='return inputs["x"]')
    body.params = [Parameter(name=name) for name in wiring]
    end = EndNode.value("terminal", output_name="result")
    return Flow(
        nodes={"entry": start, "body": body, "terminal": end},
        edges=[
            Edge("entry->body", "entry", "body", ordering=True),
            Edge("body->terminal", "body", "terminal", input_group="result"),
        ],
        wiring={
            "entry": {},
            "body": {name: Binding(source) for name, source in wiring.items()},
            "terminal": {"result": Binding("${body.output}")},
        },
        start_id="entry",
        end_id="terminal",
    )


def _wired(flow: Flow, param: str):
    """Return one parameter's canonical wiring value from the body node."""
    wiring = flow_manifest(flow)["wiring"]["body"]
    [value] = [
        entry["value"]
        for entry in wiring["entries"]
        if entry["key"] == {"type": "str", "value": param}
    ]
    return value


def test_a_value_template_encodes_as_a_tagged_tree():
    flow = _fragment(x=ValueTemplate({"a": "${n.output}", "b": [1]}))

    encoded = _wired(flow, "x")

    assert encoded["type"] == "value_template"
    assert encoded["tree"]["type"] == "mapping"
    assert {"key": {"type": "str", "value": "a"},
            "value": {"type": "str", "value": "${n.output}"}} in encoded["tree"]["entries"]


def test_a_nested_binding_encodes_its_source_and_declared_type():
    typed = _fragment(
        x=ValueTemplate({"a": Binding("${n.output}", contract=ValueContract(INTEGER))})
    )
    untyped = _fragment(x=ValueTemplate({"a": Binding("${n.output}")}))

    typed_value = _wired(typed, "x")["tree"]["entries"][0]["value"]
    untyped_value = _wired(untyped, "x")["tree"]["entries"][0]["value"]

    assert typed_value["type"] == "binding"
    assert typed_value["source"] == {"type": "str", "value": "${n.output}"}
    assert typed_value["contract"]["kind"] == "int"
    assert untyped_value == {
        "contract": None,
        "source": {"type": "str", "value": "${n.output}"},
        "type": "binding",
    }
    assert flow_fingerprint(typed) != flow_fingerprint(untyped)


def test_template_identity_differs_from_every_authored_literal():
    template = _fragment(x=ValueTemplate({"a": "${n.output}"}))
    plain = _fragment(x={"a": "${n.output}"})
    shaped = _fragment(x=_wired(template, "x"))

    fingerprints = {flow_fingerprint(flow) for flow in (template, plain, shaped)}

    assert len(fingerprints) == 3


def test_a_nested_binding_is_not_a_bare_string_or_untyped_template_part():
    bare = _fragment(x=ValueTemplate({"a": "${n.output}"}))
    nested = _fragment(x=ValueTemplate({"a": Binding("${n.output}")}))

    assert flow_fingerprint(bare) != flow_fingerprint(nested)


def test_template_and_nested_binding_origins_are_not_identity():
    first = _fragment(
        x=ValueTemplate(
            {"a": "${n.output}", "b": Binding(
                "${m.output}",
                OriginRef("first/b"),
                ValueContract(INTEGER, ContractOrigins(default=OriginRef("first/c"))),
            )},
            ((("a",), OriginRef("first/a")),),
        )
    )
    second = _fragment(
        x=ValueTemplate(
            {"a": "${n.output}", "b": Binding(
                "${m.output}",
                OriginRef("second/b"),
                ValueContract(INTEGER, ContractOrigins(default=OriginRef("second/c"))),
            )},
            ((("a",), OriginRef("second/a")),),
        )
    )

    assert flow_fingerprint(first) == flow_fingerprint(second)


def test_template_parts_change_identity():
    first = _fragment(x=ValueTemplate({"a": "${n.output}", "b": [1, "${m.output}"]}))
    changed = _fragment(x=ValueTemplate({"a": "${n.output}", "b": [1, "${k.output}"]}))

    assert flow_fingerprint(first) != flow_fingerprint(changed)


def test_a_nested_binding_must_have_only_canonical_fields():
    nested = Binding("${n.output}")
    object.__setattr__(nested, "extra", 1)
    flow = _fragment(x=ValueTemplate({"a": nested}))

    with pytest.raises(ValueError, match="unknown noncanonical field"):
        flow_fingerprint(flow)


def test_a_template_under_a_credential_name_keeps_every_part_as_identity():
    first = _fragment(
        password=ValueTemplate({"primary": "${input.first}", "note": "first", "pin": 1})
    )
    edited_text = _fragment(
        password=ValueTemplate({"primary": "${input.first}", "note": "second", "pin": 1})
    )
    edited_number = _fragment(
        password=ValueTemplate({"primary": "${input.first}", "note": "first", "pin": 2})
    )
    rerouted = _fragment(
        password=ValueTemplate({"primary": "${input.other}", "note": "first", "pin": 1})
    )

    text = json.dumps(flow_manifest(first), sort_keys=True)
    assert "redacted" not in text
    assert "input.first" in text
    fingerprints = {
        flow_fingerprint(flow) for flow in (first, edited_text, edited_number, rerouted)
    }
    assert len(fingerprints) == 4


def test_a_plain_mapping_under_a_credential_key_is_identity():
    first = _fragment(request={"headers": {"Authorization": "Bearer ${input.a}"}})
    edited = _fragment(request={"headers": {"Authorization": "Bearer ${input.b}"}})

    assert flow_fingerprint(first) != flow_fingerprint(edited)
    assert "redacted" not in json.dumps(flow_manifest(first), sort_keys=True)


def test_a_flow_output_template_encodes_as_a_tagged_tree():
    compiled = load_flow(
        """
id: outputs
name: Outputs
nodes:
  step:
    kind: code
    output: int
    code: |
      return 1
output: ${step.output}
"""
    ).compiled
    changed = copy.deepcopy(compiled)
    template = ValueTemplate({"n": "${step.output}"})
    changed.outputs = [FlowOutput(compiled.outputs[0].name, Binding(template))]
    plain = copy.deepcopy(compiled)
    plain.outputs = [FlowOutput(compiled.outputs[0].name, Binding({"n": "${step.output}"}))]

    manifest = flow_manifest(changed)

    assert manifest["outputs"][0]["from"]["type"] == "value_template"
    assert flow_fingerprint(changed) != flow_fingerprint(plain)
    assert isinstance(changed, CompiledFlow)


_UNTYPED_FLOW = """
id: plain
name: Plain
input:
  x: !int
  token: !str
nodes:
  step:
    kind: code
    input:
      x: ${input.x}
      request: {headers: {Authorization: "Bearer $${input.token}"}, items: [1, "a"]}
      password: "${input.token}"
    output: int
    code: |
      return inputs["x"] + 1
output: ${step.output}
"""


def test_a_flow_without_templates_keeps_its_fingerprint():
    # The fingerprint main gives the same flow written before value templates, with
    # untagged inputs and the nested text as `Bearer ${input.token}` (nested strings
    # were literal there, so `$$` was not unescaped); a template-free flow must not move.
    assert flow_fingerprint(load_flow(_UNTYPED_FLOW).compiled) == (
        "c5ea81e947585f9000040c8446b8c14ed3114e12a14c3584338d1b4db3b3d5e7"
    )


_LLM_FLOW = """\
id: f
name: f
input: {m: !str , n: !str , cfg: !object }
FLOW_LAYER
nodes:
  a:
    kind: agent
    prompt: Go.
    llm_config: NODE_LAYER
output: ${a.output}
"""


def _llm_fingerprint(
    node_layer: str,
    flow_layer: str = "llm_config: {provider: anthropic, anthropic: {max_tokens: 900}}",
) -> str:
    """Fingerprint `_LLM_FLOW` with the node and flow `llm_config` layers given."""
    text = _LLM_FLOW.replace("NODE_LAYER", node_layer).replace("FLOW_LAYER", flow_layer)
    return flow_fingerprint(load_flow(text).compiled)


def test_a_literal_llm_config_flow_keeps_its_fingerprint():
    # The fingerprint main gives this flow, from before `llm_config` read values.
    text = """\
id: f
name: f
llm_config: {provider: anthropic, anthropic: {max_tokens: 900}}
nodes:
  a:
    kind: agent
    prompt: Go.
    llm_config: {model: claude-x, inherit: true, anthropic: {temperature: 0.2}}
output: ${a.output}
"""
    assert flow_fingerprint(load_flow(text).compiled) == (
        "0f2c73ac336010ede82bbbc2f9a6584bbe70a5c7453285d4a2eedbdb81e4376e"
    )


def test_llm_config_values_are_fingerprinted_as_written():
    assert _llm_fingerprint("{model: '${input.m}'}") != _llm_fingerprint(
        "{model: '${input.n}'}"
    )
    assert _llm_fingerprint("{model: '${input.m}'}") != _llm_fingerprint("{model: m}")


def test_a_tag_on_an_llm_config_value_changes_the_fingerprint():
    assert _llm_fingerprint("{model: !str '${input.m}'}") != _llm_fingerprint(
        "{model: '${input.m}'}"
    )


def test_whole_record_llm_config_values_are_fingerprinted():
    node = _llm_fingerprint("${input.cfg}")
    assert node != _llm_fingerprint("!object ${input.cfg}")
    assert node != _llm_fingerprint("{model: '${input.m}'}")
    flow = _llm_fingerprint("{model: m}", flow_layer="llm_config: ${input.cfg}")
    assert flow != _llm_fingerprint("{model: m}", flow_layer="llm_config: !object ${input.cfg}")
    assert flow != _llm_fingerprint("{model: m}", flow_layer="llm_config: {model: x}")


def test_a_flow_llm_config_value_is_fingerprinted_as_written():
    assert _llm_fingerprint(
        "{model: m}", flow_layer="llm_config: {provider: '${input.m}'}"
    ) != _llm_fingerprint("{model: m}", flow_layer="llm_config: {provider: '${input.n}'}")


def test_a_literal_llm_config_with_a_doubled_dollar_reads_unescaped():
    # A literal layer is read as a template would be, so `$$` is one `$` and the
    # fingerprint is that of the unescaped text.
    assert _llm_fingerprint('{model: "a$$b"}') == _llm_fingerprint('{model: "a$b"}')
    flow = load_flow(
        _LLM_FLOW.replace("NODE_LAYER", '{model: "a$$b"}').replace("FLOW_LAYER", "")
    )
    assert flow_manifest(flow.compiled)["nodes"]["a"]["own_llm_config"] == {
        "entries": [{"key": {"type": "str", "value": "model"},
                     "value": {"type": "str", "value": "a$b"}}],
        "type": "mapping",
    }
