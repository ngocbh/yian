/**
 * A loaded flow is a reusable program; each run owns its mutable live graph: a port of
 * `tests/engine/test_run_reuse.py`.
 *
 * Ported: `test_loaded_recursive_flow_can_be_run_twice_without_graph_or_llm_leaks` and
 * `test_log_resume_restores_into_a_private_graph`.
 *
 * Changed: an AGENT's model comes from the `models` option, which records the config it is built
 * from (Python patches `model_from_config`); the interruption is the run's `signal`, which the
 * `ask_user` handler aborts, and the run rejects with its reason (Python: the handler raises
 * `_HostStop`); a child flow's identity is compared by reference. CODE references resolve from the
 * conformance fixtures, and every run sees only the test's credential store in its environment.
 *
 * Covered by a case: none.
 */

import { afterEach, beforeEach, expect, test } from "bun:test";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";

import type { Flow } from "../../src/compile/model.ts";
import type { ComposedFlow } from "../../src/compose/build/pipeline.ts";
import { loadFlow } from "../../src/compose/loader.ts";
import { ScriptedModel } from "../../src/llm/testing.ts";
import type { ModelAccess } from "../../src/llm_clients/factory.ts";
import { AgentNode } from "../../src/nodes/agent/node.ts";
import { resumeFlow, runFlow } from "../../src/run.ts";

import { hostEnv } from "./mcp_helpers.ts";
import { rejected, textReply } from "./run_state_fixtures.ts";

const FIXTURES = fileURLToPath(new URL("../../../conformance/fixtures", import.meta.url));
let dir: string;

beforeEach(() => {
  dir = mkdtempSync(join(tmpdir(), "yian-run-reuse-"));
});

afterEach(() => {
  rmSync(dir, { recursive: true, force: true });
});

const RECURSIVE_FLOW = `
id: reusable
name: reusable
llm_config: {openai: {temperature: 0.2}}
input:
  topics: !list(str)
defs:
  call_child:
    nodes:
      answer:
        kind: agent
        prompt: call
        llm_config: {model: call-model}
    output: \${answer.output}
  map_child:
    input:
      topic: !str
    nodes:
      inspect:
        kind: agent
        prompt: inspect
        llm_config: {model: map-model}
      emit:
        kind: code
        depends_on: [inspect]
        input:
          topic: \${input.topic}
        output: str
        code: ./compose_codefns.ts:echo
    output: \${emit.output}
  loop_child:
    input:
      n: !int
      exited: !bool
    nodes:
      inspect:
        kind: agent
        prompt: inspect
        llm_config: {model: loop-model}
      step:
        kind: code
        depends_on: [inspect]
        input:
          n: \${input.n}
          exited: \${input.exited}
        output:
          n: int
          exited: bool
        code: ./compose_codefns.ts:loop_bump
    output: \${step.output}
nodes:
  call_once:
    kind: call
    call: call_child
  map_each:
    kind: map
    call: map_child
    over: \${input.topics}
    input:
      topic: \${item}
  repeat:
    kind: loop
    call: loop_child
    input:
      n: 0
      exited: false
    while: not exited
    max: 5
  finish:
    kind: agent
    depends_on: [call_once, map_each, repeat]
    prompt: finish
    llm_config: {model: finish-model}
output:
  called: \${call_once.output}
  mapped: \${map_each.output}
  repeated: \${repeat.output}
  finished: \${finish.output}
`;

const BLOCKING_CALL = `
id: blocking-call
name: blocking_call
defs:
  approval:
    nodes:
      ask:
        kind: human_input
        prompt: approve?
        output: str
    output: \${ask.output}
nodes:
  gate:
    kind: call
    call: approval
output: \${gate.output}
`;

/** The static topology of `flow`, recursively, with each child flow by reference. */
function graphSnapshot(flow: Flow): Record<string, unknown> {
  const agents: unknown[] = [];
  const children: unknown[] = [];
  for (const [nodeId, node] of flow.nodes) {
    if (node instanceof AgentNode) {
      agents.push([nodeId, structuredClone(node.ownLlmConfig), structuredClone(node.llmConfig)]);
    }
    const child = (node as { child?: Flow | null }).child ?? null;
    if (child !== null) children.push([nodeId, child, graphSnapshot(child)]);
  }
  return {
    nodes: [...flow.nodes.keys()],
    edges: [...flow.edges],
    wiring: [...flow.wiring].map(([nodeId, wiring]) => [nodeId, [...wiring]]),
    agents,
    children,
  };
}

/** Expect `after` to equal `before`, each child flow the same object. */
function expectSameGraph(after: Record<string, unknown>, before: Record<string, unknown>): void {
  expect(after).toEqual(before);
  const childrenOf = (snapshot: Record<string, unknown>) => (snapshot["children"] as [string, Flow][]).map(([, c]) => c);
  childrenOf(after).forEach((child, index) => expect(child).toBe(childrenOf(before)[index]!));
}

/** A model access that records each config it builds from and answers `ok`. */
function scripted(seen: Record<string, unknown>[]): ModelAccess {
  return {
    env: hostEnv(dir),
    createModel: (config) => {
      seen.push({ ...config });
      return new ScriptedModel([textReply("ok")], {
        provider: config["provider"] as string,
        model: config["model"] as string,
      });
    },
  };
}

test("test_loaded_recursive_flow_can_be_run_twice_without_graph_or_llm_leaks", async () => {
  const seen: Record<string, unknown>[] = [];
  const loaded = await loadFlow(RECURSIVE_FLOW, { sourceLabel: "flow.yaml", baseDir: FIXTURES });
  const original = graphSnapshot(loaded.compiled);

  const first = await runFlow(loaded, { topics: ["alpha", "beta"] }, {
    env: hostEnv(dir),
    models: scripted(seen),
    llmConfig: { provider: "openai", openai: { reasoning: { effort: "low" } } },
  });
  expect(first.failure).toBeNull();
  expect(first.status).toBe("succeeded");
  expectSameGraph(graphSnapshot(loaded.compiled), original);
  const firstConfigs = [...seen];
  seen.length = 0;

  const second = await runFlow(loaded, { topics: ["alpha", "beta"] }, {
    env: hostEnv(dir),
    models: scripted(seen),
    llmConfig: { provider: "google", google: { generationConfig: { thinkingConfig: { thinkingLevel: "HIGH" } } } },
  });
  expect(second.failure).toBeNull();
  expect(second.status).toBe("succeeded");
  expect(second.output).toEqual(first.output);
  expectSameGraph(graphSnapshot(loaded.compiled), original);

  expect(firstConfigs.length).toBeGreaterThan(0);
  for (const config of firstConfigs) {
    expect(config["provider"]).toBe("openai");
    expect(config["openai"]).toEqual({ temperature: 0.2, reasoning: { effort: "low" } });
  }
  expect(seen.length).toBeGreaterThan(0);
  for (const config of seen) {
    expect(config["provider"]).toBe("google");
    expect(config["google"]).toEqual({ generationConfig: { thinkingConfig: { thinkingLevel: "HIGH" } } });
  }
});

test("test_log_resume_restores_into_a_private_graph", async () => {
  const path = join(dir, "private-recovery-graph.jsonl");
  const interruption = new Error("interrupt before human answer");
  const host = new AbortController();
  const source: ComposedFlow = await loadFlow(BLOCKING_CALL);
  const sourceStatic = graphSnapshot(source.compiled);
  const interrupt = () => {
    host.abort(interruption);
    throw interruption;
  };

  const caught = await rejected(
    runFlow(source, {}, { log: path, env: hostEnv(dir), signal: host.signal, tools: { ask_user: interrupt } }),
  );
  expect(caught).toBe(interruption);
  expectSameGraph(graphSnapshot(source.compiled), sourceStatic);

  const fresh = await loadFlow(BLOCKING_CALL);
  const original = graphSnapshot(fresh.compiled);
  const resumed = await resumeFlow(fresh, { log: path, env: hostEnv(dir), tools: { ask_user: () => "yes" } });

  expect(resumed.failure).toBeNull();
  expect(resumed.status).toBe("succeeded");
  expect(resumed.output).toBe("yes");
  expectSameGraph(graphSnapshot(fresh.compiled), original);
});
