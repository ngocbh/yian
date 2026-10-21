/**
 * Retired authoring fields stay rejected: a port of `tests/engine/test_retired_runtime_surface.py`'s
 * load rows, through `assembleText` where Python loads the flow.
 *
 * Changed: `test_retired_agent_authoring_fields_are_rejected` keeps its load half; its
 * `AgentNode(...)` half is not ported, since the TS options type has no such keys.
 *
 * Not ported: `test_checkpoint_and_suspension_modules_are_gone` and the module and attribute rows
 * of `test_agent_continuation_packages_and_builders_are_gone` (Python modules and attributes; the
 * descriptor-field half is the load row here), `test_shipped_blocking_node_packages_cannot_construct_pause`
 * (an AST scan of Python packages) and `test_shipped_blocking_node_bodies_are_typed_to_output`
 * (Python type hints; TS checks this statically). `test_effect_nodes_drop_scratch_keys` and
 * `test_wait_event_mode_and_the_watchdog_are_gone` are recorded in `resolve_until.test.ts`.
 *
 * Ported: the `build_grow` row of `test_agent_continuation_packages_and_builders_are_gone`
 * (`buildGrow` refuses an AGENT), pinning the whole message where Python matches a fragment.
 * Ported (adapted): `test_final_log_only_runtime_and_public_surface`: the retired events, `Pause`,
 * engine methods and `resumeCommand` are absent; `FlowEngine` isn't on the package root; the
 * `run_flow` and `resume_flow` parameter rows check `RunFlowOptions`'s and `ResumeFlowOptions`'s
 * keys (TS-only: `signal`, `env`, `scheduler`, `random`, `models`, and the `@internal`
 * `checkLlmConfigs` of Python's `_resume_flow_controlled`) and the `RunResult` row its own fields
 * and getters (TS-only: `interruptedAttempts`; `_backward_handle` is a private field).
 *
 * TS-only: the package root (`index.ts`) exports exactly its documented root list plus the memory
 * storage names and `ToolAttemptsFailed`, each the owning module's export, and `yian/run`
 * (`run.ts`) re-exports `BackwardOutcome` and `Steering`.
 */

import { expect, test } from "bun:test";

import { buildGrow } from "../../src/compile/expand.ts";
import { CompiledFlow } from "../../src/compile/model.ts";
import { FlowValidationError } from "../../src/compile/validation.ts";
import { ComposedFlow } from "../../src/compose/build/pipeline.ts";
import { LoadError } from "../../src/compose/helpers/errors.ts";
import { loadFlow } from "../../src/compose/loader.ts";
import { Diagnostic, Failure } from "../../src/diagnostics.ts";
import * as events from "../../src/events.ts";
import { registerTemplateFn } from "../../src/expr/builtins.ts";
import { ExpressionError, evaluateWhen } from "../../src/expr/expressions.ts";
import * as yian from "../../src/index.ts";
import {
  FileMarkdownStore,
  InMemoryMarkdownStore,
  type MarkdownStore,
  MarkdownVersionMismatch,
  markdownVersion,
} from "../../src/memory/storage/markdown.ts";
import { checkStoragePath, SQLiteStorage } from "../../src/memory/storage/sqlite.ts";
import { buildMemories, MemoryRegistry } from "../../src/memory_host.ts";
import { AgentNode } from "../../src/nodes/agent/node.ts";
import * as base from "../../src/nodes/base.ts";
import { ToolAttemptsFailed } from "../../src/nodes/tool/node.ts";
import * as run from "../../src/run.ts";
import { BackwardOutcome } from "../../src/runtime/contracts.ts";
import { FlowEngine } from "../../src/runtime/engine.ts";
import { ToolContext } from "../../src/tools/context.ts";
import { PermanentToolError, ReportedToolError, TransientToolError } from "../../src/tools/errors.ts";
import { FunctionTool, tool } from "../../src/tools/function_tools.ts";
import { VariablePool } from "../../src/typesys/pool.ts";
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

test("buildGrow refuses an AGENT: agent continuations are gone", () => {
  expect(() => buildGrow(new AgentNode("agent"), {})).toThrow(
    new RangeError("node 'agent' (agent) cannot build graph growth"),
  );
});

// test_final_log_only_runtime_and_public_surface
test("only the final log-only runtime and run API remain", () => {
  for (const name of ["PauseRequested", "RunPaused", "RunAborted", "NodeStreamEvent"]) {
    expect(name in events).toBe(false);
  }
  expect("Pause" in base).toBe(false);

  for (const method of ["snapshot", "restore", "resume", "requestAbort", "request_abort"]) {
    expect(method in FlowEngine.prototype).toBe(false);
  }
  expect("FlowEngine" in yian).toBe(false);

  expect("resumeCommand" in run).toBe(false);
  // The options' keys, exactly: a missing or extra key fails `tsc`.
  const options: Record<keyof run.RunFlowOptions, true> = {
    log: true,
    runId: true,
    numWorkers: true,
    onEvent: true,
    onStream: true,
    llmConfig: true,
    memory: true,
    learnable: true,
    tools: true,
    skills: true,
    steering: true,
    context: true,
    signal: true,
    env: true,
    scheduler: true,
    random: true,
    models: true,
  };
  expect(Object.keys(options)).toHaveLength(17);
  expect(run.runFlow.length).toBe(2);
  const resumeOptions: Record<keyof run.ResumeFlowOptions, true> = {
    log: true,
    numWorkers: true,
    onEvent: true,
    onStream: true,
    memory: true,
    tools: true,
    skills: true,
    steering: true,
    apiKey: true,
    headers: true,
    context: true,
    signal: true,
    env: true,
    scheduler: true,
    random: true,
    models: true,
    checkLlmConfigs: true,
  };
  expect(Object.keys(resumeOptions)).toHaveLength(17);
  expect(run.resumeFlow.length).toBe(2);

  const result = new run.RunResult({ input: {}, status: "succeeded", runId: "r1", log: null });
  expect(Object.keys(result)).toEqual([
    "input",
    "status",
    "runId",
    "log",
    "output",
    "failure",
    "events",
    "interruptedAttempts",
  ]);
  const getters = Object.entries(Object.getOwnPropertyDescriptors(run.RunResult.prototype))
    .filter(([, descriptor]) => descriptor.get !== undefined)
    .map(([name]) => name);
  expect(getters).toEqual(["learning", "backwardOutcome"]);
});

test("the package root exports exactly its documented root list plus the memory storage names and ToolAttemptsFailed", () => {
  const expected: Record<string, unknown> = {
    loadFlow,
    runFlow: run.runFlow,
    resumeFlow: run.resumeFlow,
    buildMemories,
    MemoryRegistry,
    SQLiteStorage,
    checkStoragePath,
    FileMarkdownStore,
    InMemoryMarkdownStore,
    MarkdownVersionMismatch,
    markdownVersion,
    Failure,
    Diagnostic,
    ComposedFlow,
    CompiledFlow,
    VariablePool,
    evaluateWhen,
    LoadError,
    FlowValidationError,
    ExpressionError,
    ToolContext,
    FunctionTool,
    tool,
    PermanentToolError,
    TransientToolError,
    ReportedToolError,
    registerTemplateFn,
    ToolAttemptsFailed,
  };
  expect(Object.keys(yian).sort()).toEqual(Object.keys(expected).sort());
  const exported = yian as Record<string, unknown>;
  expect(Object.keys(expected).filter((name) => exported[name] !== expected[name])).toEqual([]);
  // The type export, checked by `tsc`.
  const store: yian.MarkdownStore = new InMemoryMarkdownStore() satisfies MarkdownStore;
  expect(store).toBeInstanceOf(InMemoryMarkdownStore);
});

test("yian/run re-exports BackwardOutcome and Steering", () => {
  expect(run.BackwardOutcome).toBe(BackwardOutcome);
  // The type re-export, checked by `tsc`.
  const steering = (inbox: base.Steering): run.Steering => inbox;
  expect(typeof steering).toBe("function");
});

// ---------- harvested conformance cases ----------

test.each<CaseRow>([{ id: "load-core/engine_retired_runtime_surface__retired_wait_field", call: parsedNodes }])(
  "parseNodes: $id",
  runCase,
);
