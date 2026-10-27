/**
 * The public direct-engine contract. A direct caller prepares and commits `RunStarted` first, then
 * lends that state and log to `FlowEngine`, which drives only new work and borrows the log.
 */

import { afterEach, beforeEach, expect, test } from "bun:test";

import { buildGrow } from "../../src/compile/expand.ts";
import { resolveLlmCascade } from "../../src/compile/llm_cascade.ts";
import { CompiledFlow, Edge, END_ID, FlowOutput, START_ID } from "../../src/compile/model.ts";
import { loadFlow } from "../../src/compose/loader.ts";
import {
  NodeExpanded,
  NodeStarted,
  NodeSucceeded,
  type RunEvent,
  RunStarted,
  RunSucceeded,
  StreamChunk,
} from "../../src/events.ts";
import * as yian from "../../src/index.ts";
import { Message, Response, TextPart } from "../../src/llm/messages.ts";
import { ScriptedModel } from "../../src/llm/testing.ts";
import { Node, type NodeCaps, NodeKind, type NodeResult } from "../../src/nodes/base.ts";
import { EndNode } from "../../src/nodes/end/node.ts";
import { StartNode } from "../../src/nodes/start/node.ts";
import { FlowEngine, type FlowEngineOptions } from "../../src/runtime/engine.ts";
import { MemoryRunLog, NodeStartedRecord, RunStoppedRecord } from "../../src/runtime/log.ts";
import { commitPrepared, prepareNodeFailed, prepareNodeStarted, RunState } from "../../src/runtime/state.ts";
import { ToolSet } from "../../src/tools/registry.ts";
import {
  engine,
  FuncNode,
  graph,
  NO_ENV,
  nodeFailure,
  noGrowth,
  rejected,
  RUN_ID,
  StreamNode,
} from "./run_state_fixtures.ts";

/** The invocations of the action flow's node. */
const probe: string[] = [];

/** Return `START -> work -> END`, whose `work` records each invocation in `probe`. */
function actionFlow(): CompiledFlow {
  const work = new FuncNode("work", () => {
    probe.push("ran");
    return "done";
  });
  return graph([work], [
    [START_ID, "work"],
    ["work", END_ID],
  ], { outputs: [new FlowOutput("output", "${work.output}")] });
}

/** Return the run's rejection message after checking its class. */
async function refused(run: Promise<unknown>, kind: ErrorConstructor): Promise<string> {
  const error = await rejected(run);
  expect(error).toBeInstanceOf(kind);
  return (error as Error).message;
}

test("growth builder is a required constructor dependency", () => {
  const { state, log } = engine(actionFlow());
  expect(() => new FlowEngine(state, log, { numWorkers: 0 } as FlowEngineOptions)).toThrow(
    new TypeError("grow_builder must be callable"),
  );
});

test("unprepared state is rejected before any node executes", async () => {
  probe.length = 0;
  const state = new RunState(actionFlow());
  const log = new MemoryRunLog(RUN_ID);
  const run = new FlowEngine(state, log, { growBuilder: noGrowth, numWorkers: 0 }).run();
  expect(await refused(run, RangeError)).toBe("FlowEngine requires an initialized RunState");
  expect(probe).toEqual([]);
  expect(log.nextSequence).toBe(0);
});

test("mismatched state and log are rejected before execution", async () => {
  probe.length = 0;
  const { state, log } = engine(actionFlow());
  const other = new MemoryRunLog("another-run");
  const run = new FlowEngine(state, other, { growBuilder: noGrowth, numWorkers: 0 }).run();
  expect(await refused(run, RangeError)).toBe("run-state/log run_id mismatch: state 'run-123', log 'another-run'");
  expect(probe).toEqual([]);
  // RunStarted and the run start's two commits.
  expect(log.nextSequence).toBe(3);
  expect(other.nextSequence).toBe(0);
});

test("mismatched durable sequence is rejected before execution", async () => {
  probe.length = 0;
  const { state, log } = engine(actionFlow());
  log.append(new NodeStartedRecord("unapplied"));
  const run = new FlowEngine(state, log, { growBuilder: noGrowth, numWorkers: 0 }).run();
  expect(await refused(run, RangeError)).toBe("run-state/log sequence mismatch: state 3, log 4");
  expect(probe).toEqual([]);
  expect(state.nextSequence).toBe(3);
  expect(log.nextSequence).toBe(4);
});

test("mismatched durable worker count is rejected before execution", async () => {
  probe.length = 0;
  const { state, log } = engine(actionFlow(), { numWorkers: 1 });
  const run = new FlowEngine(state, log, { growBuilder: noGrowth, numWorkers: 0 }).run();
  expect(await refused(run, RangeError)).toBe("run-state/engine worker mismatch: state 1, engine 0");
  expect(probe).toEqual([]);
  expect(state.numWorkers).toBe(1);
  expect(state.nextSequence).toBe(3);
  expect(log.nextSequence).toBe(3);
});

test("the constructor checks its arguments", () => {
  const { state, log } = engine(actionFlow());
  const build = (options: Record<string, unknown>, stateArg: unknown = state, logArg: unknown = log) => () =>
    new FlowEngine(stateArg as RunState, logArg as MemoryRunLog, {
      growBuilder: noGrowth,
      numWorkers: 0,
      ...options,
    } as FlowEngineOptions);

  expect(build({}, {})).toThrow(new TypeError("FlowEngine expects a RunState"));
  expect(build({}, state, { runId: RUN_ID, nextSequence: 1 })).toThrow(new TypeError("FlowEngine expects a RunLog"));
  expect(build({ numWorkers: 1.5 })).toThrow(new TypeError("num_workers must be a non-negative integer"));
  expect(build({ numWorkers: "1" })).toThrow(new TypeError("num_workers must be a non-negative integer"));
  expect(build({ numWorkers: -1 })).toThrow(new RangeError("num_workers must be a non-negative integer"));
  expect(build({ onEvent: 1 })).toThrow(new TypeError("on_event must be callable or null"));
  expect(build({ onStream: 1 })).toThrow(new TypeError("on_stream must be callable or null"));
  expect(build({ memoryContextOpener: 1 })).toThrow(new TypeError("memory_context_opener must be callable or null"));
  expect(build({ skills: {} })).toThrow(new TypeError("skills must be a map or null"));
  expect(build({ models: { env: NO_ENV, createModel: 1 } })).toThrow(
    new TypeError("models must provide env(name) and createModel(config, options), or be null"),
  );
});

test("the run's tool set is bound to the run id and the run signal", () => {
  const { state, log } = engine(actionFlow());
  const tools = new ToolSet({});
  const built = new FlowEngine(state, log, { growBuilder: noGrowth, numWorkers: 0, tools });
  expect(built.tools).toBeInstanceOf(ToolSet);
  expect(built.tools).not.toBe(tools);
  expect(built.tools.runId).toBe(RUN_ID);
  expect(built.tools.signal).toBe(built.signal);
  expect(built.signal.aborted).toBe(false);
  expect(engine(actionFlow()).tools.runId).toBe(RUN_ID);
});

test("a terminal state returns no events, and a second run is refused", async () => {
  const { state, log } = engine(actionFlow());
  commitPrepared(log, state, prepareNodeStarted(state, START_ID));
  commitPrepared(log, state, prepareNodeFailed(state, nodeFailure(START_ID)));
  expect(state.result).not.toBeNull();
  const size = log.nextSequence;
  const terminal = new FlowEngine(state, log, { growBuilder: noGrowth, numWorkers: 0 });
  expect(await terminal.run()).toEqual([]);
  expect(log.nextSequence).toBe(size);
  expect(await refused(terminal.run(), Error)).toBe("FlowEngine.run() was already called");
});

test("unfinished starts must be requeued before execution", async () => {
  probe.length = 0;
  const { state, log } = engine(actionFlow());
  commitPrepared(log, state, prepareNodeStarted(state, START_ID));
  const run = new FlowEngine(state, log, { growBuilder: noGrowth, numWorkers: 0 }).run();
  expect(await refused(run, RangeError)).toBe(
    "FlowEngine requires unfinished starts to be requeued with RunResumed before execution",
  );
  expect(probe).toEqual([]);
  expect(log.nextSequence).toBe(4);
});

/** Return `START -> twice -> END`, where `twice` doubles the input `x`. */
function staticFlow(): CompiledFlow {
  const twice = new FuncNode("twice", (inputs) => (inputs["n"] as bigint) * 2n, { reads: { n: "${input.x}" } });
  return graph([twice], [
    [START_ID, "twice"],
    ["twice", END_ID],
  ], { outputs: [new FlowOutput("output", "${twice.output}")] });
}

test("static run yields only new post durable observers", async () => {
  const callbacks: [RunEvent, number][] = [];
  const onEvent = (event: RunEvent): void => {
    callbacks.push([event, built.log.nextSequence]);
  };
  const built = engine(staticFlow(), { inputs: { x: 4n }, onEvent });
  const events = await built.run();

  expect(callbacks.map(([event]) => event)).toEqual([...events]);
  callbacks.forEach(([event], index) => expect(event).toBe(events[index]!));
  expect(events.some((event) => event instanceof RunStarted)).toBe(false);
  expect(events.map((event) => event.constructor)).toEqual([
    NodeStarted,
    NodeSucceeded,
    NodeStarted,
    NodeSucceeded,
    NodeStarted,
    NodeSucceeded,
    RunSucceeded,
  ]);
  // Each node's observers follow its record: after the run start's three records, START's start is
  // the fourth record, and its success the sixth, after its commit.
  expect(callbacks.slice(0, 2).map(([, sequence]) => sequence)).toEqual([4, 6]);
  expect((events.at(-1) as RunSucceeded).output).toBe(8n);
  expect(built.state.result).toEqual(events.at(-1) as RunSucceeded);
});

test("stream chunks are callback only and never durable", async () => {
  const stream = new StreamNode("stream", ["hel", "lo"]);
  const flow = CompiledFlow.fromParts(
    new Map<string, Node>([
      [START_ID, new StartNode(START_ID)],
      ["stream", stream],
      [END_ID, EndNode.value(END_ID, { outputName: "output" })],
    ]),
    [
      new Edge("start-stream", START_ID, "stream", { optional: true, ordering: true }),
      new Edge("stream-end", "stream", END_ID, { inputGroup: "output" }),
    ],
    {
      wiring: new Map([
        [START_ID, new Map()],
        ["stream", new Map()],
        [END_ID, new Map([["output", "${stream.output.output}"]])],
      ]),
    },
  );
  const chunks: StreamChunk[] = [];
  const built = engine(flow, { onStream: (chunk) => chunks.push(chunk) });
  const events = await built.run();

  expect(chunks).toEqual([
    new StreamChunk({ nodeId: "stream", key: "output", chunk: "hel" }),
    new StreamChunk({ nodeId: "stream", key: "output", chunk: "lo" }),
  ]);
  expect(events.some((event) => event instanceof StreamChunk)).toBe(false);
  expect(built.log.records().some((envelope) => envelope.record instanceof StreamChunk)).toBe(false);
  // The run start's three records, then three successes of three records each.
  expect(built.log.nextSequence).toBe(12);
  expect(built.state.result).toEqual(events.at(-1) as RunSucceeded);
  expect(events.at(-1)).toEqual(new RunSucceeded({ output: "hello" }));
});

test("engine borrows and never closes the run log", async () => {
  const built = engine(staticFlow(), { inputs: { x: 3n } });
  let closes = 0;
  const close = built.log.close.bind(built.log);
  built.log.close = () => {
    closes += 1;
    close();
  };
  await built.run();
  expect(closes).toBe(0);
  built.log.close();
  expect(closes).toBe(1);
});

const AGENT_FLOW = `
id: direct-agent
name: direct_agent
nodes:
  answer:
    kind: agent
    prompt: answer directly
output: \${answer.output}
`;

test("direct agent uses the run's model factory", async () => {
  const configs: Record<string, unknown>[] = [];
  const message = new Message({ role: "assistant", parts: [new TextPart("direct answer")] });
  const reply = new Response({ message, finishReason: "stop" });
  const model = new ScriptedModel([reply], { provider: "openai", model: "fake-direct" });
  const flow = (await loadFlow(AGENT_FLOW)).compiled;
  const effective = resolveLlmCascade(flow, {}, { defaults: { provider: "openai", model: "fake-direct" } });
  const models = {
    env: NO_ENV,
    createModel: (config: Readonly<Record<string, unknown>>) => {
      configs.push({ ...config });
      return model;
    },
  };
  const events = await engine(flow, { effectiveLlmByStaticPath: effective, growBuilder: buildGrow, models }).run();

  expect(configs).toEqual([{ provider: "openai", model: "fake-direct" }]);
  expect(model.requests[0]!.messages).toHaveLength(2);
  expect(events.at(-1)).toEqual(new RunSucceeded({ output: "direct answer" }));
});

const CHILD_FLOW = `
id: direct-child
name: direct_child
input:
  topic: !str
nodes:
  echo:
    kind: code
    code: return inputs["topic"]
    input:
      topic: \${input.topic}
    output: str
output: \${echo.output}
`;

const CALL_FLOW = `
id: direct-call
name: direct_call
input:
  topic: !str
uses:
  direct-child: direct-child
nodes:
  invoke:
    kind: call
    call: direct-child
    input:
      topic: \${input.topic}
output: \${invoke.output}
`;

const MAP_FLOW = `
id: direct-map
name: direct_map
input:
  topics: !list(str)
uses:
  direct-child: direct-child
nodes:
  invoke:
    kind: map
    call: direct-child
    over: \${input.topics}
    input:
      topic: \${item}
output: \${invoke.output}
`;

// The body increments `n` and sets `exited` once `n` reaches 3: three iterations from the seed.
const LOOP_FLOW = `
id: direct-loop
name: direct_loop
defs:
  bump:
    input:
      n: !int
      exited: !bool
    nodes:
      step:
        kind: code
        code: |
          const n = inputs["n"] + 1;
          return { n, exited: n >= 3 };
        input:
          n: \${input.n}
          exited: \${input.exited}
        output:
          n: int
          exited: bool
    output: \${step.output}
nodes:
  repeat:
    kind: loop
    call: bump
    input:
      n: 0
      exited: false
    while: not exited
    max: 5
output: \${repeat.output}
`;

/** Resolve the one test child the direct CALL and MAP runs use. */
function childResolver(flowId: string) {
  if (flowId !== "direct-child") throw new Error(`unknown child '${flowId}'`);
  return loadFlow(CHILD_FLOW);
}

/** Load one real authorable spawner graph for the named combinator, with its run inputs. */
async function loadDynamic(kind: string): Promise<[CompiledFlow, Record<string, unknown>]> {
  if (kind === "call") return [(await loadFlow(CALL_FLOW, { childResolver })).compiled, { topic: "Ada" }];
  if (kind === "map") return [(await loadFlow(MAP_FLOW, { childResolver })).compiled, { topics: ["Ada", "Grace"] }];
  return [(await loadFlow(LOOP_FLOW)).compiled, {}];
}

test.each([
  ["call", "Ada"],
  ["map", ["Ada", "Grace"]],
  ["loop", { n: 3n, exited: true }],
])("direct engine executes the real dynamic combinator %s", async (kind, expected) => {
  const [flow, inputs] = await loadDynamic(kind);
  const events = await engine(flow, { inputs, growBuilder: buildGrow }).run();

  expect(events.some((event) => event instanceof NodeExpanded)).toBe(true);
  expect(events.at(-1)).toBeInstanceOf(RunSucceeded);
  expect((events.at(-1) as RunSucceeded).output).toEqual(expected);
});

const unhandled: unknown[] = [];
const collect = (reason: unknown): void => {
  unhandled.push(reason);
};
beforeEach(() => {
  unhandled.length = 0;
  process.on("unhandledRejection", collect);
});
afterEach(() => {
  process.off("unhandledRejection", collect);
});

/** A node that waits until its signal aborts, then rejects with the reason. */
class WaitNode extends Node {
  readonly kind = NodeKind.CODE;

  override forward(_inputs: Record<string, unknown>, caps: NodeCaps): Promise<NodeResult> {
    return new Promise((_resolve, reject) => {
      caps.signal!.addEventListener("abort", () => reject(caps.signal!.reason), { once: true });
    });
  }
}

test("an onEvent promise that rejects before the terminal result rejects run with its error", async () => {
  const error = new Error("host listener failed");
  const flow = graph([new WaitNode("work")], [
    [START_ID, "work"],
    ["work", END_ID],
  ]);
  const onEvent = (event: RunEvent): Promise<void> | undefined =>
    event instanceof NodeStarted && event.nodeId === "work" ? Promise.reject(error) : undefined;
  const built = engine(flow, { onEvent });

  expect(await rejected(built.run())).toBe(error);
  expect(built.signal.aborted).toBe(true);
  expect(built.signal.reason).toBe(error);
  expect(built.state.result).toBeNull();
  const work = built.log.records().filter((envelope) => (envelope.record as { nodeId?: string }).nodeId === "work");
  expect(work.map((envelope) => envelope.record.constructor)).toEqual([NodeStartedRecord]);
  await new Promise((resolve) => setTimeout(resolve, 0));
  expect(unhandled).toEqual([]);
});

test("stopForCallback stops the run with that error", async () => {
  const error = new Error("boundary listener failed");
  const flow = graph([new WaitNode("work")], [
    [START_ID, "work"],
    ["work", END_ID],
  ]);
  const built = engine(flow, {
    onEvent: (event: RunEvent) => {
      if (event instanceof NodeStarted && event.nodeId === "work") built.stopForCallback(error);
    },
  });

  expect(await rejected(built.run())).toBe(error);
  expect(built.signal.reason).toBe(error);
  const stops = built.log.records().flatMap(({ record }) => (record instanceof RunStoppedRecord ? [record] : []));
  expect(stops.map((record) => [record.reason, record.message, record.nodeIds])).toEqual([
    ["callback", "boundary listener failed", ["work"]],
  ]);
});

test("stopForCallback after the terminal result is ignored", async () => {
  const built = engine(staticFlow(), { inputs: { x: 1n } });

  expect((await built.run()).at(-1)).toEqual(new RunSucceeded({ output: 2n }));
  built.stopForCallback(new Error("late"));
  expect(built.signal.aborted).toBe(false);
  expect(built.log.records().some(({ record }) => record instanceof RunStoppedRecord)).toBe(false);
});

test("an onEvent promise that rejects after the terminal result is ignored", async () => {
  const onEvent = (event: RunEvent): Promise<void> | undefined =>
    event instanceof RunSucceeded ? Promise.reject(new Error("late")) : undefined;
  const built = engine(staticFlow(), { inputs: { x: 1n }, onEvent });

  expect((await built.run()).at(-1)).toEqual(new RunSucceeded({ output: 2n }));
  await new Promise((resolve) => setTimeout(resolve, 0));
  expect(built.signal.aborted).toBe(false);
  expect(unhandled).toEqual([]);
});

test("FlowEngine is not exported from the package root", () => {
  expect("FlowEngine" in yian).toBe(false);
});
