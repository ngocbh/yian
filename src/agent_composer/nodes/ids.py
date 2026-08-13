"""Deterministic identifiers shared by graph-expanding nodes and compilers."""


def ns(callsite: str, child_id: str) -> str:
    """Namespace a child node/edge id under its callsite: `<callsite>/<child_id>`.
    `callsite` = spawner id (REF/agent) or `f"{spawner}#{i}"` (MAP element i); nests."""
    return f"{callsite}/{child_id}"


def map_callsite(spawner_id: str, i: int) -> str:
    """The per-element callsite for MAP element `i`: `f"{spawner}#{i}"`."""
    return f"{spawner_id}#{i}"
