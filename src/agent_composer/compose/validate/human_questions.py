"""Graph-stage: validate each built human_input gate's `questions:` surface at load time.

A loader post-build pass over the built node map. A LITERAL `questions:` list is structurally
checked via `parse_questions` (rejects bad count/shape/duplicate headers); a ref-form
`questions_input` must name one of the node's declared inputs (mirrors the AGENT prompt-scope
rule). It relocates `validate.validate_human_questions` verbatim, adding a `.range` anchor off
the built gate's `field_ranges[("field", "questions")]` (the coarse `questions:` span,
else the node's own range).

Imports flow down only: `nodes` (HumanInputNode + its questions parser), `compose.helpers.errors`,
`events` (SourceRange). Only the loader (and the compose package facade) import this back.
"""

from __future__ import annotations

from typing import Optional

from agent_composer.compose.helpers.errors import LoadDiagnostics
from agent_composer.events import SourceRange
from agent_composer.nodes.base import Node


def validate_human_questions(
    nodes: dict[str, Node],
    node_lines: "dict[str, int] | None" = None,
    *,
    diagnostics: LoadDiagnostics | None = None,
) -> None:
    """Validate each built human_input gate's questions surface at load time.

    - A LITERAL questions list -> parse_questions (rejects bad count/shape/dup headers),
      re-raised as a located LoadError.
    - A ref-form questions_input -> the name MUST be one of the node's declared inputs
      (mirrors the AGENT prompt-scope rule); otherwise LoadError with a hint.
    Legacy (no questions) gates pass through. Each error anchors on the gate's `questions:`
    field range (`field_ranges[("field", "questions")]`, else the node's own range).
    `node_lines` is accepted for transitional caller compatibility but is not location
    authority. Accumulates all errors into one LoadError.
    """
    diagnostics = diagnostics or LoadDiagnostics()
    from agent_composer.nodes.human_input.node import HumanInputNode
    from agent_composer.nodes.human_input.questions import (
        QuestionSpecError,
        parse_questions,
    )

    errors: list[tuple[str, Optional[SourceRange]]] = []
    for node_id, node in nodes.items():
        if not isinstance(node, HumanInputNode):
            continue
        rng = node.field_ranges.get(("field", "questions")) or node.range
        if node.questions is not None:
            # A literal list baked into the node: structurally check it at LOAD,
            # so a bad count/shape/duplicate header fails here rather than at run.
            try:
                parse_questions(node.questions)
            except QuestionSpecError as exc:
                errors.append((f"node {node_id!r} (kind=human_input): {exc}", rng))
        elif node.questions_input is not None:
            # A ref-form gate reads its questions from a declared input — the name
            # must be wired in via `input:` (mirrors the AGENT prompt-scope rule).
            declared = {p.name for p in (node.params or [])}
            if node.questions_input not in declared:
                errors.append((
                    f"node {node_id!r} (kind=human_input): questions reference "
                    f"${{{node.questions_input}}} is not a declared input — feed dynamic "
                    f"questions through `input:` wiring and reference the declared name "
                    f"(questions: ${{name}}), or use adaptive_questions:",
                    rng,
                ))
    if errors:
        first_range = next(
            (error_range for _, error_range in errors if error_range is not None),
            None,
        )
        raise diagnostics.error(
            "flow has invalid human_input questions:\n  "
            + "\n  ".join(msg for msg, _ in errors),
            range=first_range,
        )
