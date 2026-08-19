# Plan: Draw Model Invocation Components as Boxes

**Goal**: Render the three provider-level invocation components as distinct
bordered boxes inside one labelled Mermaid container.

**Architecture**: Change only the Mermaid source in `docs/agent.md`; preserve
the existing request-to-provider-to-model flow and use invisible layout links
between sibling invocation components.

**Tech Stack**: Markdown, Mermaid 11.16.1, Mermaid CLI 11.16.0, MkDocs.

## Step 1: Establish the current render

**File**: `docs/agent.md`

1. Render the current Mermaid document with the pinned Mermaid CLI.
2. Confirm the invocation content is one text node rather than three bordered
   components.

## Step 2: Replace the invocation text node

**File**: `docs/agent.md`

1. Define a `Model invocation` subgraph with left-to-right layout.
2. Add bordered nodes for context tokens, tool/function schemas, and output
   controls.
3. Add `~~~` layout links and connect the serializer to the subgraph.
4. Run `git diff --check`.

## Step 3: End-to-end verification

**File**: `docs/agent.md`

1. Render the document with Mermaid 11.16.1 / CLI 11.16.0.
2. Inspect the rendered image for three visible child boxes inside the outer
   invocation container.
3. Run `conda run --no-capture-output -n agentcomposer mkdocs build --strict`.
4. Commit the documentation change and open an issue-linked pull request.

## Dependencies

| Group | Steps | Can parallelize |
| --- | --- | --- |
| 1 | Step 1 | No |
| 2 | Step 2 | No; depends on Step 1 |
| 3 | Step 3 | No; depends on Step 2 |
