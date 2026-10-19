/**
 * Retired authoring fields stay rejected: a port of `tests/engine/test_retired_runtime_surface.py`'s
 * load rows, through `assembleText` where Python loads the flow.
 *
 * Changed: `test_retired_agent_authoring_fields_are_rejected` keeps its load half; its
 * `AgentNode(...)` half is not ported, since the TS options type has no such keys.
 *
 * Not ported: `test_checkpoint_and_suspension_modules_are_gone` and
 * `test_agent_continuation_packages_and_builders_are_gone` (Python modules and attributes; the
 * descriptor-field half is the load row here), `test_shipped_blocking_node_packages_cannot_construct_pause`
 * (an AST scan of Python packages), `test_shipped_blocking_node_bodies_are_typed_to_output`
 * (Python type hints; TS checks this statically), and `test_final_log_only_runtime_and_public_surface`
 * (Python run and runtime signatures). `test_effect_nodes_drop_scratch_keys` and
 * `test_wait_event_mode_and_the_watchdog_are_gone` are recorded in `resolve_until.test.ts`.
 */

import { expect, test } from "bun:test";

import { assembleText, type CaseRow, loadError, parsedNodes, runCase } from "../compose/declarations.ts";

test.each(["entry", "controls"])("retired agent authoring field %s is rejected", (field) => {
  const text = `
id: retired-agent-${field}
name: retired_agent_${field}
input: {}
nodes:
  agent:
    kind: agent
    prompt: hi
    output: str
    ${field}: retired
output: \${agent.output}
`;

  expect(loadError(() => assembleText(text)).message).toContain(`field '${field}' is not allowed`);
});

test.each(["event", "event_spec", "poll", "is_timed"])("retired wait authoring field %s is rejected", (field) => {
  const text = `
id: retired-wait-${field}
name: retired_wait_${field}
input: {}
nodes:
  wait:
    kind: wait
    until: 2000-01-01
    ${field}: retired
output: \${wait.output}
`;

  expect(loadError(() => assembleText(text)).message).toContain(`field '${field}' is not allowed`);
});

// ---------- harvested conformance cases ----------

test.each<CaseRow>([{ id: "load-core/engine_retired_runtime_surface__retired_wait_field", call: parsedNodes }])(
  "parseNodes: $id",
  runCase,
);
