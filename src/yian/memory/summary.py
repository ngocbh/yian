"""Pure whole-turn selection and private summary-cache lineage evidence.

Projections and summaries are opaque text. This module owns no model, mutable
history, lease, or public snapshot representation.
"""

from dataclasses import dataclass
import hashlib
import json

from yian.memory.contracts import ConversationInvocation, ConversationView


def admissions(view: ConversationView) -> tuple[ConversationInvocation, ...]:
    """Read turn order from admissions, including turns with no effect or item."""
    result = []
    for entry in view.applied:
        run_id, sequence, ordinal = json.loads(entry.operation_id)
        if ordinal == 0:
            result.append(ConversationInvocation(run_id, sequence))
    return tuple(result)


def prefix_digest(view: ConversationView, count: int) -> str:
    """Prove the complete raw prefix preceding the next admitted invocation."""
    seen = 0
    prefix = []
    for entry in view.applied:
        if json.loads(entry.operation_id)[2] == 0:
            if seen == count:
                break
            seen += 1
        prefix.append(
            (
                entry.operation_id,
                entry.event_digest,
                entry.before_digest,
                entry.after_digest,
            )
        )
    else:
        raise ValueError("a summary prefix must precede the current admission")
    return hashlib.sha256(
        json.dumps(prefix, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True)
class Summary:
    """Derived text bound to an immutable raw prefix and projection identity."""

    namespace: str
    generation: int
    projection_version: str
    covered: tuple[ConversationInvocation, ...]
    digest: str
    text: str

    def __post_init__(self) -> None:
        if type(self.text) is not str:
            raise TypeError("summary must be a string")
        if not self.text.strip():
            raise ValueError("summary must contain nonempty text")

    def matches(
        self,
        view: ConversationView,
        prior: tuple[ConversationInvocation, ...],
        version: str,
    ) -> bool:
        """Check ownership and complete prefix evidence before reusing text."""
        return (
            (self.namespace, self.generation, self.projection_version)
            == (view.namespace, view.generation, version)
            and bool(self.covered)
            and prior[: len(self.covered)] == self.covered
            and self.digest == prefix_digest(view, len(self.covered))
        )
