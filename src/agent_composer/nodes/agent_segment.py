"""Canonical seed contract for temporary AGENT continuation growth.

This leaf module owns the serializable descriptor pair emitted by the AGENT node and consumed by
compile-owned growth construction plus durable replay. It lives directly under `nodes` so importing
the contract does not execute `nodes.agent` mode registration or reach into `compile`, `runtime`,
or `suspension`.
"""

from typing import Any

from pydantic import BaseModel


class AgentSegment(BaseModel):
    """Validated seed for one temporary AGENT continuation expansion.

    `hi_desc` describes the synthesized human-input leaf (`node_id`, `slot`, and `prompt`).
    `resume_desc` carries the resume-agent frame (`memo`, `iterations`, `pending`, and optional
    model/tool/control configuration) as serializable data.
    """

    hi_desc: dict[str, Any]
    resume_desc: dict[str, Any]
