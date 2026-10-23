/**
 * A port of `tests/engine/test_backward_composition.py`: public backward through loaded flows
 * composing CALL, CASE, MAP and LOOP, owned memories, and a stale reflective snapshot.
 *
 * Ported: all 6 tests, the parametrized ones whole. Models are `ScriptedModel`s built by a model
 * access (`models`) whose `pick` a test reassigns where Python re-patches `model_from_config`; a
 * run that must build no model checks the access built none. Records are read from the run's
 * JSONL log; an AGENT backward prompt is its request's user message read as JSON.
 *
 * Changed:
 * - `test_concat_repeat_and_opaque_routes_cross_public_call_without_reevaluation`: list repetition
 *   is a type error (decision 7), so `input.values * 2` becomes two more concatenated copies. The
 *   opaque builtin is registered with `registerTemplateFn` and put back after the test.
 * - `test_owned_memories_learn_complete_history_and_reject_invalid_proposals`: the invalid learn
 *   result is `null` (Python's is a tuple where a list is required); learn-context feedback holds
 *   its ints as `bigint`s.
 * - `test_two_public_runs_reject_the_stale_reflective_snapshot_after_any_commit`: the reflector is
 *   given as `ReflectiveMemory`'s `createModel`; Python's one `with_structured_output` binding is
 *   the one request asking for structured output.
 * - Python's `ValueError` is a `RangeError`; CODE bodies are JavaScript (decision 11).
 */

import { mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { afterAll, describe, expect, test } from "bun:test";

import type { ComposedFlow } from "../../src/compose/build/pipeline.ts";
import { loadFlow } from "../../src/compose/loader.ts";
import { NodeSucceeded } from "../../src/events.ts";
import { registerTemplateFn, TEMPLATE_FNS } from "../../src/expr/builtins.ts";
import type { ChatModel } from "../../src/llm/events.ts";
import { Message, Response, TextPart } from "../../src/llm/messages.ts";
import { ScriptedModel } from "../../src/llm/testing.ts";
import { MemoryMutation, type LearnContext } from "../../src/memory/contracts.ts";
import { ReflectiveMemory } from "../../src/memory/reflective.ts";
import { BackwardResult } from "../../src/nodes/base.ts";
import { CodeNode } from "../../src/nodes/code/node.ts";
import { legacyConnectionConfig } from "../../src/llm_clients/connection.ts";
import type { ModelAccess } from "../../src/llm_clients/factory.ts";
import { replayRunLog, type ResumeFlowOptions, resumeFlow, type RunFlowOptions, runFlow } from "../../src/run.ts";
import {
  BackwardFinishedRecord,
  BackwardReadyRecord,
  BackwardStartedRecord,
  evidencePayload,
  JsonlRunLog,
  MemoryCommitRequestedRecord,
  NodeStartedRecord,
  NodeSucceededRecord,
  type RecordEnvelope,
  type RunStartedRecord,
} from "../../src/runtime/log.ts";
import { logJson } from "../../src/typesys/json.ts";
import { type Value, writeAsText } from "../../src/typesys/values.ts";
import { hostEnv } from "./mcp_helpers.ts";
import { FakeMemory } from "./memory_fakes.ts";
import { feedbackEntries, rejected, textReply } from "./run_state_fixtures.ts";

const dir = mkdtempSync(join(tmpdir(), "yian-backward-composition-"));
afterAll(() => rmSync(dir, { recursive: true, force: true }));
const env = hostEnv(dir);

let files = 0;

/** A fresh path in the file's temp dir. */
const fresh = (name: string) => join(dir, `${files++}-${name}`);

type Pick = (config: Readonly<Record<string, unknown>>) => ChatModel;

/** A pick for a run that must build no model. */
const NO_MODEL: Pick = () => {
  throw new Error("no model may be built");
};

/**
 * Model access over the file's env: each build records its config and serves `pick(config)`; a
 * test reassigns `pick` where Python re-patches `model_from_config`.
 */
function models(pick: Pick = NO_MODEL): ModelAccess & { pick: Pick; configs: Readonly<Record<string, unknown>>[] } {
  const access = {
    pick,
    configs: [] as Readonly<Record<string, unknown>>[],
    env,
    createModel: (config: Readonly<Record<string, unknown>>) => {
      access.configs.push(config);
      return access.pick(config);
    },
  };
  return access;
}

/** Load `text` in the file's env. */
const load = (text: string) => loadFlow(text, { env });

/** `runFlow` in the file's env. */
const run = (flow: ComposedFlow, inputs: Readonly<Record<string, unknown>>, options: RunFlowOptions = {}) =>
  runFlow(flow, inputs, { env, ...options });

/** `resumeFlow` in the file's env. */
const resume = (flow: ComposedFlow, options: ResumeFlowOptions) => resumeFlow({ env, ...options });

/** An AGENT backward reply: Python's `_reply`, empty channels unless `channels` sets them. */
function feedbackReply(channels: Record<string, unknown> = {}): Response {
  const text = JSON.stringify({ input_feedback: {}, self_feedback: [], variable_feedback: {}, ...channels });
  return new Response({ message: new Message({ role: "assistant", parts: [new TextPart(text)] }) });
}

/** The material of `model`'s backward request `index` (its user message, read as JSON). */
function material(model: ScriptedModel, index = 0): Record<string, any> {
  return JSON.parse(model.requests[index]!.messages[1]!.text());
}

/** `value` as the prompt material's text form shows it, read back as JSON. */
const asJson = (value: unknown) => JSON.parse(writeAsText({ value } as Value)).value;

/** The envelopes of the closed durable log at `path`. */
function envelopes(path: string): readonly RecordEnvelope[] {
  const log = JsonlRunLog.resume(path);
  try {
    return log.records();
  } finally {
    log.close();
  }
}

test("nested call case selects partial feedback without promoting child defaults", async () => {
  const flow = await load(`id: selected-feedback
name: selected_feedback
input:
  selected: !bool
  payload: !object
defs:
  leaf:
    input:
      payload: !object
      local: !str child default
    nodes: {}
    output:
      payload: \${input.payload}
      local: \${input.local}
  wrapper:
    input:
      selected: !bool
      payload: !object
    nodes:
      gate:
        kind: case
        cases:
          - when: input.selected
            then: chosen
        else: unselected
      chosen:
        kind: call
        call: leaf
        input:
          payload: \${input.payload}
      unselected:
        kind: call
        call: leaf
        input:
          payload: \${input.payload}
    output: \${chosen.output | unselected.output}
nodes:
  outer:
    kind: call
    call: wrapper
    input:
      selected: \${input.selected}
      payload: \${input.payload}
output: \${outer.output}
`);
  const access = models();
  const path = fresh("selected.jsonl");
  const payload = { title: "unchanged", rows: [["a"], ["b", "c"]] };
  const result = await run(flow, { selected: true, payload }, { log: path, learnable: true, models: access });
  expect(result.status).toBe("succeeded");
  expect(result.output).toEqual({ payload, local: "child default" });
  const before = readFileSync(path);
  const error = await rejected(result.backward({ payload: { rows: [null, [null]] } }));
  expect(error).toBeInstanceOf(RangeError);
  expect((error as Error).message).toContain("exactly 2");
  expect(readFileSync(path)).toEqual(before);
  expect(result.backwardOutcome).toBeNull();

  const selected = { rows: [null, ["Revise b.", null]] };
  const outcome = await result.backward({ payload: selected, local: "Keep the default." });
  expect(outcome.status).toBe("succeeded");
  expect(outcome.feedback).toEqual(new BackwardResult({ inputFeedback: { payload: [selected] } }));
  const all = envelopes(path);
  const starts = new Map(
    all.flatMap((envelope) =>
      envelope.record instanceof NodeStartedRecord ? [[envelope.seq, envelope.record.nodeId] as const] : [],
    ),
  );
  const ready = all.map((envelope) => envelope.record).filter((record) => record instanceof BackwardReadyRecord);
  expect(ready).toHaveLength(1);
  const byNode = new Map(ready[0]!.attributedResults.map((item) => [starts.get(item.source.executionSeq)!, item]));
  expect([...byNode.keys()].some((name) => name.includes("unselected"))).toBe(false);
  expect(byNode.get("outer/gate")!.invoked).toBe(false);
  expect(byNode.get("outer/gate")!.feedback).toEqual([]);
  expect(byNode.get("outer/chosen/__start__")!.result.inputFeedback).toEqual({
    payload: [selected],
    local: ["Keep the default."],
  });
  expect(result.output).toEqual({ payload, local: "child default" });
  expect(access.configs).toEqual([]);
});

describe("public map preserves equal item positions after reversed completion", () => {
  for (const feedback of ["Review every item.", [null, ""]] as const) {
    test(JSON.stringify(feedback), async () => {
      const flow = await load(`id: ordered-map-feedback
name: ordered_map_feedback
llm_config: {provider: ollama, model: local-test}
input:
  values: !list(str)
defs:
  item_flow:
    input:
      value: !str
    nodes:
      gate:
        kind: code
        code: return true
        output: bool
      answer:
        kind: agent
        depends_on: [gate]
        input:
          value: \${input.value}
        prompt: \${value}
    output: \${answer.output}
nodes:
  each:
    kind: map
    call: item_flow
    over: \${input.values}
    input:
      value: \${item}
output: \${each.output}
`);
      let secondDone!: () => void;
      const ready = new Promise<void>((resolve) => (secondDone = resolve));
      const observe = (event: unknown) => {
        if (event instanceof NodeSucceeded && event.nodeId === "each#1/__end__") secondDone();
      };
      const forwardChats: ScriptedModel[] = [];
      const access = models(() => {
        const chat = new ScriptedModel([textReply("same")]);
        forwardChats.push(chat);
        return chat;
      });
      const path = fresh("map.jsonl");
      const original = CodeNode.prototype.forward;
      // The gate is held in the node's main-thread `forward`; only the CODE body runs in a worker.
      CodeNode.prototype.forward = async function (this: CodeNode, inputs, caps) {
        const occurrences = this.executionScope!.occurrencePath;
        expect(occurrences).toHaveLength(1);
        if (occurrences[0]!.index === 0) await ready;
        return original.call(this, inputs, caps);
      };
      let result;
      try {
        result = await run(flow, { values: ["same", "same"] }, {
          log: path,
          learnable: true,
          numWorkers: 4,
          onEvent: observe,
          models: access,
        });
      } finally {
        CodeNode.prototype.forward = original;
      }
      expect(result.status).toBe("succeeded");
      expect(result.output).toEqual(["same", "same"]);
      expect(forwardChats).toHaveLength(2);
      expect(forwardChats.every((chat) => chat.requests.length === 1)).toBe(true);
      const forwardRecords = envelopes(path).map((envelope) => envelope.record);
      const succeeded = forwardRecords.filter((record) => record instanceof NodeSucceededRecord);
      const itemEnds = succeeded
        .filter((record) => record.forwardCapture!.scope!.staticPath.join("/") === "each/__end__")
        .map((record) => record.forwardCapture!.scope!.occurrencePath[0]!.index);
      expect(itemEnds).toEqual([1, 0]);
      const before = readFileSync(path);
      const error = await rejected(result.backward([null]));
      expect(error).toBeInstanceOf(RangeError);
      expect((error as Error).message).toContain("exactly 2");
      expect(readFileSync(path)).toEqual(before);
      expect(result.backwardOutcome).toBeNull();

      const backwardChats: ScriptedModel[] = [];
      const message = typeof feedback === "string" ? feedback : "";
      access.pick = () => {
        const chat = new ScriptedModel([feedbackReply({ input_feedback: { value: [message] } })]);
        backwardChats.push(chat);
        return chat;
      };
      const outcome = await result.backward(feedback);
      const expectedIndices = typeof feedback === "string" ? [0, 1] : [1];
      expect(outcome.status).toBe("succeeded");
      expect(outcome.feedback).toEqual(
        new BackwardResult({
          inputFeedback: {
            values: expectedIndices.map((index) => [0, 1].map((position) => (position === index ? message : null))),
          },
        }),
      );
      expect(backwardChats).toHaveLength(expectedIndices.length);
      const agentRecords = new Map(
        succeeded
          .filter((record) => record.forwardCapture!.scope!.staticPath.join("/") === "each/answer")
          .map((record) => [record.nodeId, record]),
      );
      const observedPositions: number[] = [];
      for (const chat of backwardChats) {
        expect(chat.requests).toHaveLength(1);
        expect(chat.requests[0]!.tools).toEqual([]);
        const prompt = material(chat);
        expect(prompt["feedback"]).toEqual([message]);
        expect(prompt["forward"]["inputs"]).toEqual({ value: "same" });
        const details = (prompt["downstream_context"] as any[]).flatMap((entry) => entry["contexts"]);
        const index = agentRecords.get(prompt["forward"]["node_id"])!.forwardCapture!.scope!.occurrencePath[0]!.index!;
        expect(
          details.some(
            (detail) =>
              detail["role"] === "output" &&
              JSON.stringify(detail["result"]) === JSON.stringify(["same", "same"]) &&
              JSON.stringify(detail["positions"]) === JSON.stringify([[index]]),
          ),
        ).toBe(true);
        observedPositions.push(index);
      }
      expect(observedPositions.sort()).toEqual(expectedIndices);
    });
  }
});

test("call and map loop calls retain each agent after pruning and terminal resume", async () => {
  const flow = await load(`id: nested-agent-history
name: nested_agent_history
input:
  n: !int
  values: !list(int)
defs:
  agent_body:
    input:
      n: !int
    nodes:
      author:
        kind: agent
        variables:
          tone: !str concise
        input:
          n: \${input.n}
        prompt: '\${tone}: \${n}'
        output:
          n: int
    output: \${author.output}
  loop_body:
    input:
      n: !int
    nodes:
      invoke:
        kind: call
        call: agent_body
        input:
          n: \${input.n}
    output: \${invoke.output}
  direct:
    llm_config: {provider: openai, model: no-structured-sentinel, openai: {temperature: 0.1}}
    input:
      n: !int
    nodes:
      loop:
        kind: loop
        call: loop_body
        input:
          n: \${input.n}
        times: 2
    output: \${loop.output}
  mapped:
    llm_config: {provider: deepseek, model: no-structured-sentinel, deepseek: {temperature: 0.8}}
    input:
      n: !int
    nodes:
      loop:
        kind: loop
        call: loop_body
        input:
          n: \${input.n}
        times: 2
    output: \${loop.output}
nodes:
  one:
    kind: call
    call: direct
    input:
      n: \${input.n}
  many:
    kind: map
    call: mapped
    over: \${input.values}
    input:
      n: \${item}
output:
  single: \${one.output}
  mapped: \${many.output}
`);
  const forwardChats: ScriptedModel[] = [];
  const access = models(() => {
    const chat = new ScriptedModel([textReply('{"n": 1}')]);
    forwardChats.push(chat);
    return chat;
  });
  const path = fresh("nested.jsonl");
  const result = await run(flow, { n: 3, values: [10, 20] }, { log: path, learnable: true, models: access });
  expect(result.status).toBe("succeeded");
  expect(result.output).toEqual({ single: { n: 1n }, mapped: [{ n: 1n }, { n: 1n }] });
  expect(forwardChats).toHaveLength(6);
  expect(forwardChats.every((chat) => chat.requests.length === 1)).toBe(true);
  const forwardConfigs = [...access.configs];
  const log = JsonlRunLog.resume(path);
  let all: readonly RecordEnvelope[];
  let replay: ReturnType<typeof replayRunLog>;
  try {
    all = log.records();
    replay = replayRunLog(flow, log);
  } finally {
    log.close();
  }
  expect(all.some((envelope) => envelope.record instanceof BackwardStartedRecord)).toBe(false);
  const trace = replay.trace!;
  const acceptedAgents = new Map(
    trace.occurrences
      .filter((item) => item.record.forwardCapture!.nodeData?.["prompt_template"] === "${tone}: ${n}")
      .map((item) => [item.nodeId, item]),
  );
  expect(acceptedAgents.size).toBe(6);
  expect(forwardConfigs.map(logJson).sort()).toEqual(
    [...acceptedAgents.values()]
      .map((item) => logJson(item.record.forwardCapture!.nodeData!["effective_llm_config"]))
      .sort(),
  );
  const expectedMap = (all[0]!.record as RunStartedRecord).effectiveLlmByStaticPath;
  for (const occurrence of acceptedAgents.values()) {
    const capture = occurrence.record.forwardCapture!;
    const scope = capture.scope!;
    const positions = scope.occurrencePath;
    const bodies = positions.filter((position) => position.kind === "loop_body");
    expect(bodies).toHaveLength(1);
    const body = bodies[0]!;
    expect([0, 1]).toContain(body.index!);
    const direct = scope.staticPath[0] === "one";
    expect(positions.map((position) => position.kind)).toEqual([direct ? "call" : "map_item", "loop_body", "call"]);
    expect(capture.boundInputs!["n"]).toBe(body.index === 1 ? 1n : direct ? 3n : [10n, 20n][positions[0]!.index!]!);
    const data = capture.nodeData!;
    expect(data["variables"]).toEqual({ tone: "concise" });
    expect(data["variable_domains"]).toEqual({ tone: "str" });
    expect(data["rendered_prompt"]).toBe(`concise: ${capture.boundInputs!["n"]}`);
    const selected = expectedMap[scope.staticPath.join("/")]!;
    expect(data["effective_llm_config"]).toEqual(selected);
    expect(selected["provider"]).toBe(direct ? "openai" : "deepseek");
    expect(selected[selected["provider"] as string]).toEqual({ temperature: direct ? 0.1 : 0.8 });
    expect(occurrence.record.evidence).not.toBeNull();
    expect(occurrence.record.evidence!.prompt).toBe(data["rendered_prompt"] as string);
  }

  const priorBody = trace.occurrences.filter((item) => {
    const scope = item.record.forwardCapture!.scope!;
    const owner = scope.staticPath.slice(0, -1).join("/");
    return (
      (owner === "one/loop" || owner === "many/loop") &&
      scope.occurrencePath.some((position) => position.kind === "loop_body" && position.index === 0)
    );
  });
  expect(priorBody).toHaveLength(9);
  expect(priorBody.every((item) => !replay.flow.nodes.has(item.nodeId))).toBe(true);

  access.pick = NO_MODEL;
  const before = readFileSync(path);
  const inspected = await resume(flow, {
    log: path,
    models: access,
    onEvent: () => {
      throw new Error("terminal resume emitted event");
    },
  });
  expect(inspected.events).toEqual([]);
  expect(inspected.output).toEqual(result.output);
  expect(inspected.backwardOutcome).toBeNull();
  expect(readFileSync(path)).toEqual(before);
  expect(access.configs).toHaveLength(6);
  const backwardChats: [Readonly<Record<string, unknown>>, ScriptedModel][] = [];
  access.pick = (config) => {
    const chat = new ScriptedModel([
      feedbackReply({
        input_feedback: { n: ["Refine n."] },
        self_feedback: ["Keep the record."],
        variable_feedback: { tone: ["Keep concise."] },
      }),
    ]);
    backwardChats.push([config, chat]);
    return chat;
  };
  const outcome = await inspected.backward("Review every iteration.");
  expect(outcome.status).toBe("succeeded");
  expect(outcome.feedback).toEqual(
    new BackwardResult({
      inputFeedback: {
        n: ["Refine n."],
        values: [
          ["Refine n.", null],
          [null, "Refine n."],
        ],
      },
    }),
  );
  expect(backwardChats).toHaveLength(acceptedAgents.size);
  const seen: number[] = [];
  for (const [config, chat] of backwardChats) {
    expect(chat.requests).toHaveLength(1);
    expect(chat.requests[0]!.tools).toEqual([]);
    const forward = material(chat)["forward"];
    const occurrence = acceptedAgents.get(forward["node_id"])!;
    const capture = occurrence.record.forwardCapture!;
    expect(forward["node_started_seq"]).toBe(occurrence.source.executionSeq);
    expect(forward["inputs"]).toEqual(asJson(capture.boundInputs));
    expect(forward["output"]).toEqual({ n: 1 });
    for (const key of ["prompt_template", "rendered_prompt", "variables", "variable_domains"]) {
      expect(forward[key]).toEqual(asJson(capture.nodeData![key]));
    }
    expect(forward["evaluation_evidence"]).toEqual(asJson(evidencePayload(occurrence.record.evidence)));
    expect(config).toEqual(legacyConnectionConfig(capture.nodeData!["effective_llm_config"] as Record<string, unknown>));
    seen.push(occurrence.source.executionSeq);
  }
  expect(new Set(seen).size).toBe(acceptedAgents.size);
  const ready = envelopes(path)
    .map((envelope) => envelope.record)
    .filter((record) => record instanceof BackwardReadyRecord);
  expect(ready).toHaveLength(1);
  expect(ready[0]!.attributedResults).toHaveLength(trace.occurrences.length);
  expect(
    ready[0]!.attributedResults.filter((item) => seen.includes(item.source.executionSeq)).every((item) => item.invoked),
  ).toBe(true);
  const after = readFileSync(path);
  access.pick = NO_MODEL;
  const recovered = await resume(flow, { log: path, models: access });
  expect(recovered.events).toEqual([]);
  expect(recovered.backwardOutcome).toEqual(outcome);
  expect(recovered.output).toEqual(result.output);
  expect(readFileSync(path)).toEqual(after);
  expect(access.configs).toHaveLength(12);
});

test("concat repeat and opaque routes cross public call without reevaluation", async () => {
  const calls: unknown[] = [];
  const saved = TEMPLATE_FNS.get("composition_opaque");
  registerTemplateFn("composition_opaque", {
    params: [{ name: "value" }],
    call: (value) => {
      calls.push(structuredClone(value));
      return value;
    },
  });
  try {
    // Repetition (`input.values * 2`) is a type error, so the two repeated copies are concatenated.
    const flow = await load(`id: composed-expressions
name: composed_expressions
long_term_memory: test
input:
  values: !list(str)
defs:
  identity:
    input:
      values: !list(str)
    nodes: {}
    output: \${input.values}
  combine:
    input:
      values: !list(str)
    nodes:
      invoke:
        kind: call
        call: identity
        input:
          values: \${input.values + input.values + input.values + ['literal'] + composition_opaque(input.values)}
    output: \${invoke.output}
nodes:
  outer:
    kind: call
    call: combine
    input:
      values: \${input.values}
output: \${outer.output}
`);
    const backend = new FakeMemory();
    const memory = new Map([["test", backend]]);
    const path = fresh("expressions.jsonl");
    const result = await run(flow, { values: ["same"] }, { log: path, learnable: true, memory });
    expect(result.status).toBe("succeeded");
    expect(result.output).toEqual(["same", "same", "same", "literal", "same"]);
    expect(calls).toEqual([["same"]]);
    const outcome = await result.backward(["first", "second", "third", "literal feedback", "opaque feedback"]);
    expect(outcome.status).toBe("succeeded");
    expect(outcome.feedback).toEqual(
      new BackwardResult({ inputFeedback: { values: [["first"], ["second"], ["third"]] } }),
    );
    expect(backend.learnContexts).toHaveLength(1);
    const stopped = feedbackEntries(backend.learnContexts[0]!).filter((entry) => entry["channel"] === "stopped");
    expect(stopped.map((entry) => entry["value"])).toEqual([["opaque feedback"]]);
    expect(stopped.every((entry) => entry["target_name"] === "values")).toBe(true);
    expect(
      stopped.some((entry) =>
        ((entry["contribution_context"] as any)["contexts"] as any[]).some(
          (detail) =>
            String(detail["expression"]).includes("composition_opaque(") &&
            JSON.stringify(detail["result"]) === JSON.stringify(["same"]),
        ),
      ),
    ).toBe(true);
    expect(calls).toEqual([["same"]]);
    const before = readFileSync(path);
    const inspected = await resume(flow, { log: path, memory });
    expect(inspected.backwardOutcome).toEqual(outcome);
    expect(inspected.events).toEqual([]);
    expect(readFileSync(path)).toEqual(before);
    expect(calls).toEqual([["same"]]);
  } finally {
    if (saved === undefined) TEMPLATE_FNS.delete("composition_opaque");
    else TEMPLATE_FNS.set("composition_opaque", saved);
  }
});

describe("owned memories learn complete history and reject invalid proposals", () => {
  for (const invalidProposal of [false, true]) {
    test(String(invalidProposal), async () => {
      const flow = await load(`id: owned-composition
name: owned_composition
long_term_memory: outer
llm_config: {provider: ollama, model: local-test}
input:
  topic: !str
defs:
  child:
    long_term_memory: inner
    input:
      topic: !str
    nodes:
      work:
        kind: code
        code: return 7
        output: int
      author:
        kind: agent
        long_term_memory: local
        variables:
          tone: !str concise
        input:
          topic: \${input.topic}
        prompt: '\${topic}: \${tone}'
    output: \${author.output}
nodes:
  gate:
    kind: case
    cases:
      - when: 'true'
        then: invoke
    else: unused
  invoke:
    kind: call
    call: child
    input:
      topic: \${input.topic}
  unused:
    kind: call
    call: child
    input:
      topic: \${input.topic}
output: \${invoke.output | unused.output}
`);
      const backends = new Map(
        ["inner", "local", "outer"].map((key) => [key, new FakeMemory({ recallText: "Preserve detail." })]),
      );
      const forwardChat = new ScriptedModel([textReply("answer")]);
      const access = models(() => forwardChat);
      const path = fresh("owned.jsonl");
      const result = await run(flow, { topic: "reefs" }, { learnable: true, log: path, memory: backends, models: access });
      expect(result.status).toBe("succeeded");
      expect(result.output).toBe("answer");
      expect(forwardChat.requests).toHaveLength(1);
      for (const backend of backends.values()) {
        expect(backend.learnContexts).toEqual([]);
        expect(backend.commitCalls).toEqual([]);
      }
      const forward = envelopes(path);
      expect((forward[0]!.record as RunStartedRecord).memories!.map((spec) => spec.key)).toEqual(["inner", "local", "outer"]);
      const authors = forward.filter(
        (item) => item.record instanceof NodeSucceededRecord && item.record.nodeId === "invoke/author",
      );
      expect(authors).toHaveLength(1);
      const agentStarts = forward.filter(
        (item) => item.record instanceof NodeStartedRecord && item.record.nodeId === "invoke/author",
      );
      expect(agentStarts).toHaveLength(1);
      const agentSequence = agentStarts[0]!.seq;
      expect(authors[0]!.seq).toBeGreaterThan(agentSequence);
      expect((authors[0]!.record as NodeSucceededRecord).evidence!.memorySections).toHaveLength(3);

      // Each learner receives only its own memory's context despite inherited recall/evidence.
      let backwardChat!: ScriptedModel;
      const propose = (context: LearnContext) => {
        expect(backwardChat.requests).toHaveLength(1);
        for (const backend of backends.values()) expect(backend.commitCalls).toEqual([]);
        return [new MemoryMutation(context.memory, { seen: context.episodes.length })];
      };
      for (const [key, backend] of backends) {
        // TS learn results are arrays; Python's invalid result is a tuple, here `null`.
        backend.learnOutcomes.push(invalidProposal && key === "local" ? () => null : propose);
      }
      backwardChat = new ScriptedModel([
        feedbackReply({
          input_feedback: { topic: ["Refine topic."] },
          self_feedback: ["Keep context.", "Avoid guessing."],
          variable_feedback: { tone: ["Stay concise.", "Be precise."] },
        }),
      ]);
      access.pick = () => backwardChat;
      const outcome = await result.backward("Review answer.");
      expect(outcome.feedback).toEqual(new BackwardResult({ inputFeedback: { topic: ["Refine topic."] } }));
      expect(backwardChat.requests).toHaveLength(1);
      expect(backwardChat.requests[0]!.tools).toEqual([]);
      const records = envelopes(path).map((item) => item.record);
      if (invalidProposal) {
        expect(outcome.status).toBe("failed");
        expect(outcome.failure!.stage).toBe("learn");
        expect(outcome.failure!.code).toBe("memory.invalid_result");
        expect(outcome.failure!.memory).toBe("local");
        expect(
          records.some((record) => record instanceof BackwardReadyRecord || record instanceof MemoryCommitRequestedRecord),
        ).toBe(false);
        for (const backend of backends.values()) expect(backend.commitCalls).toEqual([]);
        expect(records.at(-1)).toBeInstanceOf(BackwardFinishedRecord);
        return;
      }

      expect(outcome.status).toBe("succeeded");
      expect(outcome.learning!.status).toBe("succeeded");
      for (const [key, backend] of backends) {
        expect(backend.learnContexts).toHaveLength(1);
        const context = backend.learnContexts[0]!;
        expect(context.memory).toBe(key);
        expect(backend.commitCalls).toHaveLength(1);
        expect(backend.commitCalls[0]![1].map((mutation) => mutation.mutationId)).toEqual([key]);
        const episodes = context.episodes.filter((episode) => episode.executionSeq === agentSequence);
        expect(episodes).toHaveLength(1);
        expect(episodes[0]!.ownerStaticPath).toEqual(["invoke", "author"]);
        expect(episodes[0]!.output).toBe("answer");
        const entries = feedbackEntries(context).filter((entry) => entry["execution_seq"] === BigInt(agentSequence));
        expect(entries.every((entry) => JSON.stringify(entry["owner_static_path"]) === '["invoke","author"]')).toBe(true);
        const values = (channel: string) =>
          entries.filter((entry) => entry["channel"] === channel).map((entry) => entry["value"]);
        expect(values("output")).toEqual(["Review answer."]);
        expect(values("self")).toEqual(["Keep context.", "Avoid guessing."]);
        expect(values("variable")).toEqual(["Stay concise.", "Be precise."]);
        for (const [channel, target] of [
          ["self", null],
          ["variable", "tone"],
        ] as const) {
          const messages = entries.filter((entry) => entry["channel"] === channel);
          expect(messages.map((entry) => entry["message_index"])).toEqual([0n, 1n]);
          for (const entry of messages) {
            expect(entry["target_name"]).toBe(target);
            expect(entry["contribution_context"]).toEqual({});
          }
        }
      }
      expect(backends.get("local")!.learnContexts[0]!.episodes).toHaveLength(1);
      for (const key of ["inner", "outer"]) {
        expect(
          backends
            .get(key)!
            .learnContexts[0]!.episodes.some((episode) => episode.ownerStaticPath.join("/") === "invoke/work"),
        ).toBe(true);
      }
    });
  }
});

describe("two public runs reject the stale reflective snapshot after any commit", () => {
  for (const action of ["noop", "add"]) {
    test(action, async () => {
      const flow = await load(`id: reflective-composition
name: reflective_composition
long_term_memory: reflective
input:
  text: !str
nodes: {}
output: \${input.text}
`);
      const payload = {
        action,
        lesson_id: action === "noop" ? null : "one",
        text: action === "noop" ? null : "Keep the original detail.",
      };
      const learner = new ScriptedModel([textReply(JSON.stringify({ actions: [payload] }))]);
      const configurations: Readonly<Record<string, unknown>>[] = [];
      const backend = new ReflectiveMemory({
        env,
        llmConfig: { provider: "ollama", model: "local-reflector" },
        createModel: (config) => {
          configurations.push(structuredClone(config));
          return learner;
        },
      });
      const memory = new Map([["reflective", backend]]);
      const access = models();
      const [firstPath, secondPath] = [fresh("a.jsonl"), fresh("b.jsonl")];
      const first = await run(flow, { text: "A" }, {
        runId: "snapshot-a",
        log: firstPath,
        learnable: true,
        memory,
        models: access,
      });
      const second = await run(flow, { text: "B" }, {
        runId: "snapshot-b",
        log: secondPath,
        learnable: true,
        memory,
        models: access,
      });
      const firstStart = envelopes(firstPath)[0]!.record as RunStartedRecord;
      const secondStart = envelopes(secondPath)[0]!.record as RunStartedRecord;
      expect(firstStart.memorySnapshots).toEqual(secondStart.memorySnapshots);
      const initial = firstStart.memorySnapshots![0]!.snapshot as Record<string, any>;
      expect(learner.requests).toHaveLength(0);
      expect(configurations).toEqual([]);

      const winner = await second.backward("B feedback");
      expect(winner.status).toBe("succeeded");
      expect(winner.learning!.status).toBe("succeeded");
      expect(winner.feedback).toEqual(new BackwardResult({ inputFeedback: { text: ["B feedback"] } }));
      const requests = envelopes(secondPath)
        .map((item) => item.record)
        .filter((record) => record instanceof MemoryCommitRequestedRecord);
      expect(requests).toHaveLength(1);
      expect(requests[0]!.mutations).toHaveLength(action === "noop" ? 0 : 1);
      const session = backend.openRun();
      const committed = session.snapshot as Record<string, any>;
      session.close();
      expect(committed["revision"]).toBe(initial["revision"] + 1n);
      expect(committed["lessons"]).toHaveLength(action === "noop" ? 0 : 1);
      expect(learner.requests).toHaveLength(1);
      expect(configurations).toHaveLength(1);
      // Python's one `with_structured_output` binding: the one call asks for structured output.
      expect(learner.requests[0]!.outputFormat).not.toBeNull();

      const forwardState = structuredClone([first.status, first.output, first.failure]);
      expect(forwardState).toEqual(["succeeded", "A", null]);
      const stale = await first.backward("A feedback");
      expect([first.status, first.output, first.failure]).toEqual(forwardState);
      expect(stale.status).toBe("failed");
      expect(stale.failure!.stage).toBe("learn");
      expect(stale.failure!.code).toBe("reflective.stale_generation");
      expect(stale.feedback).toEqual(new BackwardResult({ inputFeedback: { text: ["A feedback"] } }));
      expect(stale.learning!.status).toBe("failed");
      expect(stale.learning!.policies.map((policy) => policy.status)).toEqual(["unrequested"]);
      const records = envelopes(firstPath).map((item) => item.record);
      expect(records.at(-1)).toBeInstanceOf(BackwardFinishedRecord);
      expect(
        records.some((record) => record instanceof BackwardReadyRecord || record instanceof MemoryCommitRequestedRecord),
      ).toBe(false);
      expect(learner.requests).toHaveLength(1);
      const reopened = backend.openRun();
      expect(reopened.snapshot).toEqual(committed);
      reopened.close();
      const before = readFileSync(firstPath);
      const inspected = await resume(flow, {
        log: firstPath,
        memory: new Map([["reflective", {} as ReflectiveMemory]]),
        models: access,
      });
      expect([inspected.status, inspected.output, inspected.failure]).toEqual(forwardState);
      expect(inspected.backwardOutcome).toEqual(stale);
      expect(inspected.events).toEqual([]);
      expect(readFileSync(firstPath)).toEqual(before);
      expect(learner.requests).toHaveLength(1);
      expect(access.configs).toEqual([]);
    });
  }
});
