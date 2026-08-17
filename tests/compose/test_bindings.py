"""`compose.nodes.bindings.map_binding_strings`: the kind-blind binding-field walk.

Each kind's `BINDING_FIELDS` string values are rewritten by the mapper, non-string values
are left as-is, and a kind with no binding fields (case/wait) passes through unchanged
as the same object. HUMAN_INPUT and LOOP expose their ordinary `inputs` mappings."""

from agent_composer.compose.nodes.bindings import map_binding_strings
from agent_composer.compose.nodes.agent import AgentDescriptor
from agent_composer.compose.nodes.call import CallDescriptor
from agent_composer.compose.nodes.case import CaseDescriptor
from agent_composer.compose.nodes.human_input import HumanInputDescriptor
from agent_composer.compose.nodes.loop import LoopDescriptor
from agent_composer.compose.nodes.map import MapDescriptor
from agent_composer.compose.nodes.tool import ToolDescriptor
from agent_composer.compose.nodes.wait import WaitDescriptor


def _up(_field, _key, value):
    return value.upper()


def test_agent_inputs_rewritten_non_strings_left():
    desc = AgentDescriptor(id="a", inputs={"x": "${src}", "n": 5})
    out = map_binding_strings(desc, _up)
    assert out.inputs == {"x": "${SRC}", "n": 5}
    # A fresh descriptor (not the same object) — a rewrite happened.
    assert out is not desc


def test_tool_inputs_rewritten():
    desc = ToolDescriptor(id="t", tool_id="fetch", inputs={"url": "${u}", "n": 3})
    out = map_binding_strings(desc, _up)
    assert out.inputs == {"url": "${U}", "n": 3}


def test_call_inputs_and_over_rewritten():
    desc = CallDescriptor(id="c", call="sub", inputs={"seed": "${s}"}, over="${xs}")
    out = map_binding_strings(desc, _up)
    assert out.inputs == {"seed": "${S}"}
    assert out.over == "${XS}"
    assert out.kind == "call"


def test_map_kind_inputs_and_over_rewritten():
    desc = MapDescriptor(
        id="m", call="sub", inputs={"seed": "${s}"}, over="${xs}"
    )
    out = map_binding_strings(desc, _up)
    assert out.inputs == {"seed": "${S}"}
    assert out.over == "${XS}"


def test_mapper_distinguishes_input_key_from_scalar_field():
    desc = MapDescriptor(
        id="m",
        call="sub",
        inputs={"over": "${input_value}"},
        over="${items}",
    )
    sites = []

    def record(field, key, value):
        sites.append((field, key, value))
        return value

    map_binding_strings(desc, record)

    assert sites == [
        ("inputs", "over", "${input_value}"),
        ("over", "over", "${items}"),
    ]


def test_call_over_none_left_untouched():
    desc = CallDescriptor(id="c", call="sub", inputs={}, over=None)
    out = map_binding_strings(desc, _up)
    assert out.over is None


def test_no_binding_fields_pass_through_unchanged():
    # CASE and WAIT have no authored input-binding mapping -> identity (same object).
    for desc in (
        CaseDescriptor(id="k", on="${sel}", cases=[]),
        WaitDescriptor(id="w", until="${t}"),
    ):
        assert map_binding_strings(desc, _up) is desc


def test_human_input_and_loop_inputs_rewritten():
    for desc in (
        HumanInputDescriptor(id="h", inputs={"q": "${qs}"}, prompt="hi"),
        LoopDescriptor(id="l", call="body", inputs={"q": "${qs}"}),
    ):
        out = map_binding_strings(desc, _up)
        assert out.inputs == {"q": "${QS}"}
        assert out is not desc
