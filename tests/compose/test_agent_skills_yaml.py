"""`skills:` on AGENT: YAML shape, located errors, and flow identity."""

from __future__ import annotations

import pytest

from yian.compile.manifest import flow_fingerprint
from yian.compose import load_flow
from yian.compose.helpers.errors import LoadError


def _flow(skills_line: str) -> str:
    return f"""\
id: skills
name: skills
nodes:
  review:
    kind: agent
    prompt: Review the change.
{skills_line}
    output: str
output: ${{review.output}}
"""


def test_skills_list_and_all_reach_the_agent_node() -> None:
    listed = load_flow(_flow("    skills: [release-notes, code-review]"))
    assert listed.compiled.nodes["review"].skills == ["release-notes", "code-review"]
    everything = load_flow(_flow("    skills: all"))
    assert everything.compiled.nodes["review"].skills == "all"
    assert load_flow(_flow("")).compiled.nodes["review"].skills == []


@pytest.mark.parametrize(
    ("skills_line", "message"),
    [
        ("    skills: code-review", "`skills:` must be a list of skill names or `all`"),
        ("    skills: [1]", "`skills:` must be a list of skill names or `all`"),
        ("    skills: [Code_Review]", "invalid skill name"),
    ],
)
def test_bad_skills_point_at_the_skills_line(skills_line: str, message: str) -> None:
    with pytest.raises(LoadError, match=message) as caught:
        load_flow(_flow(skills_line))
    assert caught.value.diagnostic.frames[-1].range.start_line == 7


def test_skills_are_identity_but_empty_and_order_are_not() -> None:
    base = flow_fingerprint(load_flow(_flow("")).compiled)
    assert flow_fingerprint(load_flow(_flow("    skills: []")).compiled) == base
    ab = flow_fingerprint(load_flow(_flow("    skills: [a, b]")).compiled)
    assert ab != base
    assert flow_fingerprint(load_flow(_flow("    skills: [b, a]")).compiled) == ab
    assert flow_fingerprint(load_flow(_flow("    skills: all")).compiled) not in {base, ab}
