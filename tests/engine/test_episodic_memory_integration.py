"""Public flow integration for the built-in episodic-memory backend."""

from __future__ import annotations

import json

from agent_composer import load_flow, run_flow
from agent_composer.memory.episodic import EpisodicMemory
from agent_composer.nodes.agent.modes import MODES, AgentRunContext
from agent_composer.nodes.base import Output

_FLOW = """id: episodic-history
name: episodic_history
input:
  topic: str
nodes:
  answer:
    kind: agent
    mode: plain
    long_term_memory:
      name: episodic
      config: {max_episodes: 4, recall_limit: 2}
    input: {topic: "${input.topic}"}
    prompt: Explain ${topic}.
    output: str
output: ${answer.output}
"""


def test_backward_commits_episode_recalled_by_a_fresh_run(monkeypatch) -> None:
    """A public learnable run writes only on backward for later recall."""
    observed_sections: list[tuple[str, ...]] = []

    def answer(context: AgentRunContext) -> Output:
        observed_sections.append(
            tuple(section.text for section in context.memory_sections)
        )
        return Output("Use a queue to preserve arrival order.")

    monkeypatch.setitem(MODES, "plain", answer)
    flow = load_flow(_FLOW)
    memory = EpisodicMemory()

    first = run_flow(
        flow,
        {"topic": "queue ordering"},
        memory={"episodic": memory},
        learnable=True,
    )
    assert first.status == "succeeded"
    assert observed_sections == [()]

    # Forward retains the episode but does not update memory.
    before_backward = run_flow(
        flow,
        {"topic": "queue ordering"},
        memory={"episodic": memory},
    )
    assert before_backward.status == "succeeded"
    assert observed_sections[-1] == ()

    outcome = first.backward()
    assert outcome.status == "succeeded"
    assert outcome.learning is not None
    assert outcome.learning.status == "succeeded"

    later = run_flow(
        flow,
        {"topic": "queue ordering"},
        memory={"episodic": memory},
    )
    assert later.status == "succeeded"
    recalled = observed_sections[-1]
    assert len(recalled) == 1
    episode = json.loads(recalled[0])["episode"]
    assert episode["run_id"] == first.run_id
    assert episode["inputs"] == {"topic": "queue ordering"}
    assert episode["output"] == "Use a queue to preserve arrival order."
