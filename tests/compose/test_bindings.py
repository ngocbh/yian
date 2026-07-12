"""6.12 — `compose.bindings.map_binding_strings`: the kind-blind binding-field walk that
replaced the `calls.map_binding_strings_in_descriptor` isinstance ladder.

Asserts parity with the retired ladder's behavior: each kind's `BINDING_FIELDS` string
values are rewritten by the mapper, non-string values are left as-is, and a kind with no
binding fields (case/wait/loop/human_input) passes through UNCHANGED (same object)."""

from agent_composer.compose.bindings import map_binding_strings
from agent_composer.compose.kinds.agent import AgentDescriptor
from agent_composer.compose.kinds.call import CallDescriptor
from agent_composer.compose.kinds.case import CaseDescriptor
from agent_composer.compose.kinds.human_input import HumanInputDescriptor
from agent_composer.compose.kinds.loop import LoopDescriptor
from agent_composer.compose.kinds.map import MapDescriptor
from agent_composer.compose.kinds.tool import ToolDescriptor
from agent_composer.compose.kinds.wait import WaitDescriptor


def _up(value):
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


def test_call_over_none_left_untouched():
    desc = CallDescriptor(id="c", call="sub", inputs={}, over=None)
    out = map_binding_strings(desc, _up)
    assert out.over is None


def test_no_binding_fields_pass_through_unchanged():
    # case / wait / loop / human_input list no BINDING_FIELDS -> identity (same object).
    for desc in (
        CaseDescriptor(id="k", on="${sel}", cases=[]),
        WaitDescriptor(id="w", until="${t}"),
        LoopDescriptor(id="l", call="body", inputs={"acc": "${seed}"}),
        HumanInputDescriptor(id="h", inputs={"q": "${qs}"}, prompt="hi"),
    ):
        assert map_binding_strings(desc, _up) is desc
