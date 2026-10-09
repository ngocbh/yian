"""HUMAN_INPUT — one host-owned blocking interaction leaf.

The node package renders and validates authored prompts/questions, calls the
run's ``ask_user`` tool through the trusted metadata path, and returns the
handler's answer as ``Output``. It owns no terminal UI and creates no durable run
state; hosts pass the interaction handler per run with ``tools=``.

Knows about: nodes.base, expr, tools, peer HUMAN_INPUT modules, and Pydantic.
Never imports: compile, compose, runtime, the run boundary, CLI, or
               provider SDKs.
"""

from yian.nodes.human_input.node import HumanInputNode

__all__ = ["HumanInputNode"]
