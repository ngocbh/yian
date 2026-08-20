# Agent Request Cards Design

## Goal

Explain an AGENT node's model requests with a clear, responsive visual that
separates request structure from how the request evolves across model turns.

## Decision

Replace the Agent page's Mermaid diagram with two semantic HTML/CSS figures:

1. **One request contains** — a two-lane card with ordered messages on the left
   and turn-dependent request fields on the right, followed by a small provider
   adapter → model invocation path.
2. **One AGENT node may make several requests** — a vertical ordered timeline
   for initial construction, optional/repeatable tool rounds, optional structured
   generation/correction, and the final node outcome.

This separates three ideas that the current diagram conflates:

- message order;
- request fields that sit alongside messages;
- later model requests that accumulate interaction history or corrections.

Stop the structural visual at `Model invocation`. Providers own their wire and
token representations, so the guide must not invent a universal flattened
context order.

## Long-term memory vocabulary

Inside `SystemMessage`, label policy recall **Recalled long-term memory
(outer → local)** and show three frames:

1. root-flow frame;
2. child-flow frame(s);
3. AGENT-local frame.

Keep the rendered prompt under **Current task**, and assistant tool calls/results
under **In-node interaction history**. Neither is long-term memory. Add a note
that “long-term” means recallable across runs; actual storage lifetime and
persistence are supplied by the host memory implementation. Built-in
`ReflectiveMemory` is process-local and crosses fresh runs only when reused.

## Presentation

Use semantic `<figure>`, `<figcaption>`, `<section>`, and ordered-list markup in
`docs/agent.md`. Put narrowly namespaced `.ac-request-*` styles in the existing
`docs/stylesheets/extra.css`; do not add a new asset package.

At desktop width, ordered messages and parallel request fields occupy two
columns. Below 48rem they stack without horizontal scrolling. Use Material theme
variables for foreground, background, border, and accent colors. Optional states
must be labelled in text and not communicated by color alone.

## Verification

Keep strict MkDocs and repository tests. Update the existing Puppeteer docs job:

- Mermaid runtime checks cover only pages that still contain Mermaid;
- assert semantic order and sibling relationships in the new figures;
- at desktop and mobile widths, assert no component overflow and the expected
  two-column/stacked layout;
- in light and dark schemes, assert readable representative contrast and visible
  borders/connectors.

Avoid pixel-golden screenshots; structural, geometry, and contrast assertions
are deterministic across browser/font updates.
