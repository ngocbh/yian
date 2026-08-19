# Plan: Rewrite the Agent model-context guide

**Goal**: Make `docs/agent.md` explain agents as boundary plus behavior and show
the exact AGENT model-request structure without inventing provider token order.

**Architecture**: Documentation-only change. Preserve the current runtime and
YAML surface; distinguish current behavior from planned issues #259 and #260.

**Tech stack**: Markdown, Mermaid, MkDocs Material, Agent Composer's YAML loader,
pytest.

## Dependencies

| Group | Steps | Can parallelize |
| --- | --- | --- |
| 1 | Steps 1–2 | No; both edit `docs/agent.md` |
| 2 | Step 3 | No; validates the completed page |
| 3 | Step 4 | No; final review and GitHub workflow |

## Step 1: Explain boundary and behavior top-down

**File**: `docs/agent.md`

1. Confirm the old structure is present:

   ```bash
   rg -n 'What an agent contains|flowchart TB|Capability' docs/agent.md
   ```

2. Keep the first two paragraphs, remove the redundant opening diagram, and
   replace the capability inventory with:
   - boundary;
   - behavior;
   - flow behavior as a fixed graph;
   - atomic node behavior;
   - the LLM-backed AGENT node as one concrete atomic behavior.
3. Keep the terminology precise: Agent, Node, AGENT node, Flow, and Child flow.
4. Verify the old heading and opening diagram are absent and the new hierarchy
   is present.
5. Commit the logical documentation step.

## Step 2: Show the actual AGENT model request and memory stack

**File**: `docs/agent.md`

1. Add one Mermaid diagram with:
   - ordered `messages`;
   - the system message's system-prompt and outer-to-local memory segments;
   - the rendered human prompt;
   - tool schemas and response schema as parallel request fields;
   - provider-owned serialization into model context tokens.
2. Explain tool-loop history without placing tool descriptions in the prompt.
3. Map `input:`, `prompt:`, `memory_policy:`, `tools:`, `output:`, `mode:`,
   `llm_config:`, `retries:`, and `env:` to either message content, request
   attachments, or execution controls.
4. Add a compact YAML example of enclosing-flow and AGENT-local memory.
5. Add a short planned-extensions note linking #259 and #260 without documenting
   either field as implemented.
6. Retain and tighten the existing compact, multi-node, and child-flow examples.
7. Commit the completed page rewrite.

## Step 3: Validate executable examples and factual contracts

**Files**: `docs/agent.md`, existing tests under `tests/engine/`

1. Extract every `yaml` fence from `docs/agent.md` and load it with
   `agent_composer.load_flow`; assert the expected fence count.
2. Run:

   ```bash
   conda run -n agentcomposer env PYTHONPATH=src python -m pytest -q \
     tests/engine/test_agent_memory_recall.py \
     tests/engine/test_memory_layout.py \
     tests/engine/test_memory_authoring.py
   conda run -n agentcomposer mkdocs build --strict
   git diff --check
   ```

3. Correct only claims or examples disproved by these checks.

## Step 4: Review, integrate, and merge

1. Request an independent review against issue #261 and the design document.
2. Fix every Critical or Important finding and rerun the relevant checks.
3. Run the full suite:

   ```bash
   conda run -n agentcomposer env PYTHONPATH=src python -m pytest
   ```

4. Push `dev/docs/agent-model-context`, open a PR with `Fixes #261`, watch CI,
   merge into `main`, and verify the issue closes.
