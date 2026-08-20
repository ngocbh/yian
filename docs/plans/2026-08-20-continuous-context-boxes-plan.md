# Plan: Render Continuous Context Boundaries

**Goal**: Replace separated or textual context components with two continuous,
segmented boxes in the Agent guide.

**Architecture**: Use collapsed HTML tables inside Mermaid nodes. Table borders
own the visual boundary; transparent Mermaid shells preserve graph edges without
adding a second border.

**Tech Stack**: Markdown, Mermaid 11.16.1, Mermaid CLI 11.16.0, MkDocs.

## Step 1: Capture the current visual failure

**File**: `docs/agent.md`

1. Render the current diagram with the pinned Mermaid CLI.
2. Confirm the invocation compartments have gaps and the messages node uses
   bracketed text without visible message/memory boundaries.

## Step 2: Implement the segmented messages box

**File**: `docs/agent.md`

1. Replace the messages label with one collapsed HTML table.
2. Add shared borders for the message rows and SystemMessage memory sections.
3. Preserve system → human → optional-additional message order and outer-to-
   local memory order.
4. Make the Mermaid shell transparent so only the continuous table boundary is
   visible.

## Step 3: Implement the segmented invocation box

**File**: `docs/agent.md`

1. Replace the invocation subgraph with one collapsed HTML-table node.
2. Add adjacent cells for context tokens, tool/function schemas, and output
   controls.
3. Preserve the serializer-to-invocation edge without edges between cells.

## Step 4: End-to-end verification

**File**: `docs/agent.md`

1. Run `git diff --check`.
2. Render the Markdown file with Mermaid 11.16.1 / CLI 11.16.0.
3. Inspect the rendered image for continuous outer borders and touching divider
   lines.
4. Run `conda run --no-capture-output -n agentcomposer mkdocs build --strict`.
5. Run the repository test suite, open an issue-linked PR, and merge only after
   CI passes.

## Dependencies

| Group | Steps | Can parallelize |
| --- | --- | --- |
| 1 | Step 1 | No |
| 2 | Steps 2–3 | No; both edit the same Mermaid block |
| 3 | Step 4 | No; depends on Steps 2–3 |
