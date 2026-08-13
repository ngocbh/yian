"""HUMAN_INPUT — one host-owned blocking interaction leaf.

The node package renders and validates authored prompts/questions, calls the
reserved ``ask_user`` registry entry through the trusted metadata path, and
returns the handler's answer as ``Output``. It owns no terminal UI and creates no
checkpoint pause; hosts install the interaction handler outside the node layer.

Knows about: nodes.base, expr, tools, peer HUMAN_INPUT modules, and Pydantic.
Never imports: compile, compose, runtime, suspension, the run boundary, CLI, or
               provider SDKs.
"""

from agent_composer.nodes.human_input.node import HumanInputNode

__all__ = ["HumanInputNode"]
