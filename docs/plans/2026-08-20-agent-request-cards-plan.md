# Plan: Replace the AGENT Request Diagram with Responsive Cards

**Goal**: Present request structure and multi-turn evolution as two clear,
responsive, accessible documentation visuals.

**Architecture**: Semantic HTML lives with the Agent prose, styles stay in the
existing documentation stylesheet, and the existing Puppeteer CI harness checks
structure, responsiveness, and theme readability.

**Tech Stack**: Markdown/HTML, CSS Grid/Flexbox, MkDocs Material, Puppeteer.

## Step 1: Replace the Mermaid request diagram

**File**: `docs/agent.md`

1. Remove the request Mermaid fence and its diagram-specific explanation.
2. Add a request `<figure>` containing an ordered message stack and sibling
   request-field cards.
3. Label recalled cross-run context as long-term memory and show root-flow →
   child-flow → AGENT-local frame order.
4. Add the host-lifetime caveat.
5. Add a separate ordered timeline for tool rounds and structured corrections.

## Step 2: Add responsive, theme-aware styles

**File**: `docs/stylesheets/extra.css`

1. Define narrowly scoped request-card variables from Material theme variables.
2. Add desktop two-column request layout and bordered message rows.
3. Add memory-frame chips, request-field cards, adapter path, and vertical
   timeline markers.
4. At 48rem and below, stack request lanes and prevent horizontal overflow.
5. Avoid color-only meaning and preserve visible focus/text contrast.

## Step 3: Update browser documentation checks

**File**: `.github/workflows/docs.yml`

1. Stop expecting a Mermaid host on the Agent page.
2. Assert the request and timeline figures exist with ordered semantic content.
3. Assert request fields are siblings of, not children of, messages.
4. At 1440px assert two request columns; at 390px assert one stacked column and
   no component overflow.
5. Exercise default and slate schemes and assert representative 4.5:1 text
   contrast plus visible borders/connectors.

## Step 4: End-to-end verification

**Files**: `docs/agent.md`, `docs/stylesheets/extra.css`,
`.github/workflows/docs.yml`

1. Run `git diff --check`.
2. Run `conda run --no-capture-output -n agentcomposer mkdocs build --strict`.
3. Serve `site/` and run the browser assertions locally with the pinned
   Puppeteer/Chromium stack.
4. Inspect desktop/mobile screenshots in light and dark modes.
5. Run `conda run --no-capture-output -n agentcomposer env PYTHONPATH=src python -m pytest`.
6. Request independent review, open an issue-linked PR, and merge only after CI
   passes.

## Dependencies

| Group | Steps | Can parallelize |
| --- | --- | --- |
| 1 | Step 1 | No |
| 2 | Step 2 | No; it styles Step 1's exact markup |
| 3 | Step 3 | No; it asserts Steps 1–2 |
| 4 | Step 4 | No; it verifies all prior steps |
