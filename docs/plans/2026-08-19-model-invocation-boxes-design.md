# Model Invocation Boxes Design

## Goal

Make the provider-level model invocation visually structural instead of showing
its contents as bracketed text.

## Decision

Replace the single `Model invocation` Mermaid node with a labelled subgraph.
Inside it, render three bordered sibling nodes:

1. `provider-defined context tokens`
2. `tool/function schemas`
3. `output controls`

Arrange the nodes left-to-right. Connect them only with Mermaid's invisible
`~~~` layout links so the diagram does not claim a processing order between
parallel invocation components. Preserve the existing edge from
`Provider/API serialization` to the outer `Model invocation` container.

## Alternatives considered

- One record-shaped node with compartments is compact, but Mermaid support and
  styling are less consistent with the rest of this guide.
- Three uncontained nodes are clear individually, but lose the important fact
  that all three belong to one model invocation.
- A vertically stacked subgraph is narrower but unnecessarily increases the
  diagram height; it remains a fallback if the horizontal layout renders poorly.

## Validation

Render `docs/agent.md` with the repository's pinned Mermaid versions and build
the documentation with `mkdocs build --strict`. Visually confirm that the outer
container and all three inner borders are visible.
