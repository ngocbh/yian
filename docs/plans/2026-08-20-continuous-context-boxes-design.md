# Continuous Context Boxes Design

## Goal

Show message and model-invocation context boundaries as divider lines inside
continuous rectangles rather than as bracketed text or separated child boxes.

## Decision

Represent each segmented object as one Mermaid flowchart node whose HTML label
is a collapsed table. The table supplies the true outer border and shared cell
borders; the redundant Mermaid node border is transparent.

The ordered-messages table has:

1. a `messages (ordered)` header;
2. a SystemMessage row split into system prompt, root-flow memory, child-flow
   memory, and AGENT-local memory in outer-to-local order;
3. a HumanMessage row containing the rendered prompt;
4. an optional-additional-messages row split into tool history and structured
   fallback/retry correction.

The Model invocation table has one header and three adjacent cells: provider-
defined context tokens, tool/function schemas, and output controls. The cells
are parallel request channels; their placement does not claim flattened token
order or execution order.

Use `currentColor` for theme-compatible borders and `white-space: nowrap` for
stable compartments. The existing documentation scroller handles the diagram
on narrow viewports.

## Alternatives considered

- Nested Mermaid subgraphs and adjacent nodes leave gaps, so they do not show a
  continuous boundary.
- `block-beta` and `packet-beta` cannot express the nested SystemMessage
  compartments within this flowchart and still leave visual gaps.
- A custom flex layout with diagram-local `themeCSS` works, but adds more CSS
  and Mermaid configuration than a semantic collapsed table.

## Compatibility

This relies on Mermaid HTML labels (`foreignObject`), already used by the
pinned browser renderer. Keep each HTML table on one physical Mermaid source
line because whitespace inside a quoted label is preserved. Validate with
Mermaid 11.16.1 / Mermaid CLI 11.16.0 and the browser-based documentation job.
