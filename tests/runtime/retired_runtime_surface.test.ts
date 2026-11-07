/**
 * Retired authoring fields stay rejected at load, retired events and engine methods stay absent,
 * and the package root and `yian/run` export exactly their documented names.
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
import { ProfileMemory } from "../../src/memory/profile.ts";
import {
  FileMarkdownStore,
  InMemoryMarkdownStore,
  type MarkdownStore,
  MarkdownVersionMismatch,
  markdownVersion,
} from "../../src/memory/storage/markdown.ts";
import { checkStoragePath, SQLiteStorage } from "../../src/memory/storage/sqlite.ts";
import { buildMemories, MemoryRegistry } from "../../src/memory_host.ts";
import { checkConversation, checkMessage, ConversationError } from "../../src/nodes/agent/messages.ts";
import { AgentNode } from "../../src/nodes/agent/node.ts";
import * as base from "../../src/nodes/base.ts";
import { ToolAttemptsFailed } from "../../src/nodes/tool/node.ts";
import * as run from "../../src/run.ts";
import { FlowEngine } from "../../src/runtime/engine.ts";
import { pending, ToolContext } from "../../src/tools/context.ts";
import { PermanentToolError, ReportedToolError, TransientToolError } from "../../src/tools/errors.ts";
import { FunctionTool, tool } from "../../src/tools/function_tools.ts";
import { VariablePool } from "../../src/typesys/pool.ts";
import { InboxRef } from "../../src/typesys/values.ts";
import { assembleText, loadError } from "../helpers/declarations.ts";

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

test("buildGrow refuses an AGENT: agent continuations are gone", () => {
  expect(() => buildGrow(new AgentNode("agent"), {})).toThrow(
    new RangeError("node 'agent' (agent) cannot build graph growth"),
  );
});

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
  expect("resumeFlow" in run).toBe(false);
  expect("resumeFlow" in yian).toBe(false);
  // The options' keys, exactly: a missing or extra key fails `tsc`.
  const options: Record<keyof run.RunFlowOptions, true> = {
    log: true,
    runId: true,
    numWorkers: true,
    onEvent: true,
    onStream: true,
    live: true,
    llmConfig: true,
    memory: true,
    tools: true,
    skills: true,
    context: true,
    signal: true,
    env: true,
    scheduler: true,
    random: true,
    models: true,
    parent: true,
  };
  expect(Object.keys(options)).toHaveLength(17);
  expect(run.runFlow.length).toBe(2);
  const advanceOptions: Record<keyof run.AdvanceOptions, true> = {
    log: true,
    flow: true,
    codeRoot: true,
    numWorkers: true,
    onEvent: true,
    onStream: true,
    live: true,
    memory: true,
    tools: true,
    skills: true,
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
  expect(Object.keys(advanceOptions)).toHaveLength(19);
  expect(run.advance.length).toBe(1);

  const result = new run.RunResult({ input: {}, status: "succeeded", runId: "r1", log: null });
  expect(Object.keys(result)).toEqual([
    "input",
    "status",
    "runId",
    "log",
    "output",
    "failure",
    "events",
    "crashedAttempts",
    "waits",
    "unread",
  ]);
  expect(Object.getOwnPropertyNames(run.RunResult.prototype)).toEqual(["constructor"]);
});

test("the package root exports exactly its documented root list plus the memory storage names, ProfileMemory, ToolAttemptsFailed, pending, the message names, InboxRef and the host calls", () => {
  const expected: Record<string, unknown> = {
    loadFlow,
    runFlow: run.runFlow,
    advance: run.advance,
    send: run.send,
    withdraw: run.withdraw,
    complete: run.complete,
    close: run.close,
    interrupt: run.interrupt,
    cancel: run.cancel,
    pendingCalls: run.pendingCalls,
    RunBusyError: run.RunBusyError,
    DeliveryRefused: run.DeliveryRefused,
    buildMemories,
    MemoryRegistry,
    SQLiteStorage,
    checkStoragePath,
    FileMarkdownStore,
    InMemoryMarkdownStore,
    MarkdownVersionMismatch,
    markdownVersion,
    ProfileMemory,
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
    pending,
    FunctionTool,
    tool,
    PermanentToolError,
    TransientToolError,
    ReportedToolError,
    registerTemplateFn,
    ToolAttemptsFailed,
    checkMessage,
    checkConversation,
    ConversationError,
    InboxRef,
  };
  expect(Object.keys(yian).sort()).toEqual(Object.keys(expected).sort());
  const exported = yian as Record<string, unknown>;
  expect(Object.keys(expected).filter((name) => exported[name] !== expected[name])).toEqual([]);
  // The type export, checked by `tsc`.
  const store: yian.MarkdownStore = new InMemoryMarkdownStore() satisfies MarkdownStore;
  expect(store).toBeInstanceOf(InMemoryMarkdownStore);
  const m: yian.Message = { role: "user", parts: [] };
  expect(m.parts).toEqual([]);
});

test("yian/run has no BackwardOutcome; Steering is gone, replaced by interrupt(run, inbox)", () => {
  expect("BackwardOutcome" in run).toBe(false);
  // `Steering` was a type, so its removal is checked by `tsc`: nothing here may name it.
  expect(typeof run.interrupt).toBe("function");
});
