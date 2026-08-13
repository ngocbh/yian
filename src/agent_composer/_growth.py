"""Dependency-free seed schemas shared by graph construction and durability.

This internal leaf prevents the compile and suspension packages from importing
each other solely to agree on the temporary AGENT continuation seed shape.
"""

from typing import Any

from pydantic import BaseModel

__all__ = ["AgentSegment"]


class AgentSegment(BaseModel):
    """Validated seed for one temporary AGENT continuation expansion.

    ``hi_desc`` describes the synthesized human-input leaf (node id, slot, and
    prompt). ``resume_desc`` carries the resume-agent frame as serializable data.
    """

    hi_desc: dict[str, Any]
    resume_desc: dict[str, Any]
