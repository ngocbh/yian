"""Agent variables declared with YAML tags (`num_sentences: !int(1..12) 4`)."""

import re
from textwrap import indent

import pytest

import yian.nodes.agent.node as agent_module
from tests.compose._declarations import parameters_from_yaml
from yian.compose import LoadError, load_flow
from yian.nodes.base import Output
from yian.run import run_flow
from yian.typesys.types import read_typedefs

REG = read_typedefs({"Tone": "Literal(concise, detailed)"})

_VARIABLES = """\
num_sentences: !int(1..12) 4
tone: !Tone concise
instruction: !str "Summarize clearly"
"""

_HEADER = """\
id: tagged_variables
name: tagged_variables
typedefs:
  Tone: Literal(concise, detailed)
"""


def _with(declaration: str) -> str:
    """Return the shared variables with `declaration` replacing (or added to) its name."""
    name = declaration.split(":", 1)[0]
    kept = [line for line in _VARIABLES.splitlines(True) if not line.startswith(f"{name}:")]
    return "".join(kept) + declaration


def _agent(variables: str) -> str:
    """Spell one AGENT body whose prompt renders every variable."""
    return (
        "kind: agent\nvariables:\n"
        + indent(variables, "  ")
        + 'prompt: "${instruction} in ${num_sentences} sentences, ${tone}."\n'
        + "output: str\n"
    )


def _flow(placement: str, variables: str) -> str:
    """Spell one flow holding the AGENT at `placement`."""
    body = _agent(variables)
    caller = "nodes:\n  invoke:\n    kind: call\n    call: child\noutput: ${invoke.output}\n"
    if placement == "compact":
        return _HEADER + body
    if placement == "flat-definition":
        return _HEADER + "defs:\n  child:\n" + indent(body, "    ") + caller
    if placement == "definition":
        child = "nodes:\n  writer:\n" + indent(body, "    ") + "output: ${writer.output}\n"
        return _HEADER + "defs:\n  child:\n" + indent(child, "    ") + caller
    return (
        _HEADER + "nodes:\n  invoke:\n    kind: call\n    nodes:\n      writer:\n"
        + indent(body, "        ") + "    output: ${writer.output}\noutput: ${invoke.output}\n"
    )


PLACEMENTS = ["compact", "definition", "flat-definition", "inline"]


@pytest.mark.parametrize("placement", PLACEMENTS)
def test_tagged_variables_render_their_defaults(placement, monkeypatch):
    monkeypatch.setattr(agent_module, "run_agent", lambda ctx: Output(ctx.prompt))
    result = run_flow(load_flow(_flow(placement, _VARIABLES)), {})
    assert result.status == "succeeded", result.failure
    assert result.output == "Summarize clearly in 4 sentences, concise."


@pytest.mark.parametrize("placement", PLACEMENTS)
@pytest.mark.parametrize(
    "variables, fragment",
    [
        (
            "x: !int\n",
            "variable 'x': a variable needs a default: `x: !int <default>`",
        ),
        ("num_sentences: !int(1..12) 13\n", "'num_sentences': `13` is not a valid int(1..12)"),
        ("tone: !Tone loud\n", "'tone': `loud` is not a valid Tone"),
    ],
)
def test_a_tagged_variable_needs_a_default_that_fits(placement, variables, fragment):
    with pytest.raises(LoadError, match=re.escape(fragment)):
        load_flow(_flow(placement, _with(variables)))


def test_tagged_variables_read_their_types_and_defaults():
    variables = parameters_from_yaml(
        "variables:\n" + indent(_VARIABLES, "  "), REG, field="variables", require_default=True
    )
    assert {variable.name: variable.default for variable in variables} == {
        "num_sentences": 4,
        "tone": "concise",
        "instruction": "Summarize clearly",
    }
    assert all(variable.has_default and not variable.required for variable in variables)
    assert variables[0].contract.type.bounds == (1, 12)
    assert variables[1].contract.type.tags == frozenset({"concise", "detailed"})
