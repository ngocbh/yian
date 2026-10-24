/**
 * Builders shared by the inbox and wait reducer tests (`waits_reducer.test.ts`,
 * `wait_scenarios.test.ts`): a flow whose START declares an inbox, exact TOOL nodes for `take`,
 * `sleep` and `ask`, and a JSONL log and state past the run start. A `*.test.ts` imported by
 * another test file registers its tests twice, so these live here.
 */
import { mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { applyExecutionLayout, resolveExecutionLayout } from "../../src/compile/execution_layout.ts";
import { flowFingerprint } from "../../src/compile/manifest.ts";
import { CompiledFlow, Edge, END_ID, START_ID } from "../../src/compile/model.ts";
import { ComposedFlow } from "../../src/compose/build/pipeline.ts";
import type { Node } from "../../src/nodes/base.ts";
import { CodeNode } from "../../src/nodes/code/node.ts";
import { Parameter, ValueContract } from "../../src/nodes/contracts.ts";
import { EndNode } from "../../src/nodes/end/node.ts";
import { StartNode } from "../../src/nodes/start/node.ts";
import { ToolNode } from "../../src/nodes/tool/node.ts";
import { JsonlRunLog, type RunLog } from "../../src/runtime/log.ts";
import {
  commitInitialPool,
  commitPrepared,
  prepareNodeStarted,
  prepareNodeSucceeded,
  preparePoolCommit,
  prepareRunStarted,
  RunState,
} from "../../src/runtime/state.ts";
import { toolCallId } from "../../src/tools/context.ts";
import { VariablePool } from "../../src/typesys/pool.ts";
import { typeFor } from "../../src/typesys/types.ts";
import { InboxRef, inboxType, Type, ValueKind } from "../../src/typesys/values.ts";
import { FINGERPRINT, FIXED_CLOCK, forwardCapture, RUN_ID } from "./run_state_fixtures.ts";

/** The START's inbox param `msgs: Inbox(str)`. */
export const MSGS = new Parameter("msgs", { contract: new ValueContract(inboxType(Type.scalar(ValueKind.STRING))) });

/** `node`'s TOOL call id in the fixtures' run. */
export const call = (node: string) => toolCallId(RUN_ID, node);

/** The take output type `{events: list(str), closed: bool}`, built in code (the type grammar has no inline record). */
export const TAKE_OUTPUT = new Type({
  kind: ValueKind.OBJECT,
  fields: new Map([
    ["events", typeFor("list(str)", new Map())],
    ["closed", Type.scalar(ValueKind.BOOLEAN)],
  ]),
  required: new Set(["events", "closed"]),
});

/** An exact ToolNode for `toolId`; `take` gets the take output contract. */
export function tool(id: string, toolId: string): ToolNode {
  const node = new ToolNode(id, { toolId });
  if (toolId === "take") node.outputContract = new ValueContract(TAKE_OUTPUT);
  return node;
}

/** A side branch: an exact CodeNode (never run; the reducer only sees records). */
export const work = () => new CodeNode("work", { code: "return 1" });

/** START(msgs: Inbox(str), ...params) -> each node -> END (END joins them all). */
export function waitFlow(...nodes: Node[]): CompiledFlow {
  return paramFlow([MSGS], ...nodes);
}

/** START(...params) -> each node -> END (END joins them all). */
export function paramFlow(params: readonly Parameter[], ...nodes: Node[]): CompiledFlow {
  return CompiledFlow.fromParts(
    new Map<string, Node>([
      [START_ID, new StartNode(START_ID, { params })],
      ...nodes.map((node) => [node.id, node] as const),
      [END_ID, EndNode.value(END_ID, { outputName: null })],
    ]),
    [
      ...nodes.map((node) => new Edge(`start-${node.id}`, START_ID, node.id)),
      ...nodes.map((node) => new Edge(`${node.id}-end`, node.id, END_ID)),
    ],
  );
}

/** Commit `nodeId`'s output and success; a learnable state records a literal capture. */
export function succeed(log: RunLog, state: RunState, nodeId: string, output: unknown): void {
  const node = state.flow.nodes.get(nodeId)!;
  commitPrepared(log, state, preparePoolCommit(state, nodeId, output));
  commitPrepared(
    log,
    state,
    prepareNodeSucceeded(state, nodeId, { forwardCapture: state.learnable ? forwardCapture(node, {}) : null }),
  );
}

/**
 * A JSONL log and state of `flow` after run start and START's success (holding each inbox's
 * handle). `fingerprint` defaults to `flowFingerprint(flow)`; a flow holding a test double passes
 * `FINGERPRINT` and is never replayed through `replayRunLog`. A learnable run gets the execution
 * layout after fingerprinting, as `learnableStarted` does. `numWorkers` defaults to 0.
 */
export function started(
  flow: CompiledFlow,
  options: { learnable?: boolean; defaults?: Record<string, bigint>; fingerprint?: string; numWorkers?: number } = {},
) {
  const fingerprint = options.fingerprint ?? flowFingerprint(flow);
  if (options.learnable) applyExecutionLayout(flow, resolveExecutionLayout(flow));
  const path = join(mkdtempSync(join(tmpdir(), "waits-")), "log.jsonl");
  const log = JsonlRunLog.create(path, RUN_ID, { clock: FIXED_CLOCK });
  const state = new RunState(flow);
  commitPrepared(
    log,
    state,
    prepareRunStarted(state, {
      runId: RUN_ID,
      flowFingerprint: fingerprint,
      startWiringOverride: {},
      effectiveLlmByStaticPath: {},
      numWorkers: options.numWorkers ?? 0,
      learnable: options.learnable ?? false,
      defaults: options.defaults ?? {},
    }),
  );
  const initial = new VariablePool(START_ID);
  initial.set(START_ID, {});
  initial.addSystem("run_id", RUN_ID);
  commitInitialPool(log, state, initial);
  commitPrepared(log, state, prepareNodeStarted(state, START_ID));
  const start = state.flow.nodes.get(START_ID)!;
  const handles = (start.params ?? []).filter((param) => param.contract?.type.kind === ValueKind.INBOX);
  succeed(log, state, START_ID, Object.fromEntries(handles.map((param) => [param.name, new InboxRef(param.name)])));
  return { path, log, state, composed: new ComposedFlow(flow) };
}
export { FINGERPRINT };

/** Reopen a closed scenario log for an offline append, as a host does. */
export const reopen = (path: string) => JsonlRunLog.resume(path, { clock: FIXED_CLOCK });
