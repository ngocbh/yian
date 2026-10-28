/** Public backward failures, durable attempt prefixes, and whole-pass retries. */

import fs, { mkdtempSync, readFileSync, rmSync, statSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { afterAll, describe, expect, spyOn, test } from "bun:test";

import type { ComposedFlow } from "../../src/compose/build/pipeline.ts";
import { loadFlow } from "../../src/compose/loader.ts";
import type { ChatModel } from "../../src/llm/events.ts";
import { Message, Response, TextPart } from "../../src/llm/messages.ts";
import { ScriptedModel } from "../../src/llm/testing.ts";
import type { ModelAccess } from "../../src/llm_clients/factory.ts";
import { LearnContext, MemoryMutation, MemoryOperationError } from "../../src/memory/contracts.ts";
import { BackwardResult } from "../../src/nodes/base.ts";
import { CodeNode } from "../../src/nodes/code/node.ts";
import { type AdvanceOptions, advance, type RunFlowOptions, type RunResult, runFlow } from "../../src/run.ts";
import {
  BackwardFinishedRecord,
  BackwardReadyRecord,
  BackwardStartedRecord,
  JsonlRunLog,
  MemoryCommitFailedRecord,
  MemoryCommitRequestedRecord,
  MemoryCommitSucceededRecord,
  NodeFailedRecord,
  type RunRecord,
  SCHEMA_VERSION,
} from "../../src/runtime/log.ts";
import { logJson, parseJson } from "../../src/typesys/json.ts";
import { hostEnv } from "../helpers/mcp_helpers.ts";
import { FakeMemory } from "../helpers/memory_fakes.ts";
import { CrashAt, crashes, rejected, textReply } from "../helpers/run_state_fixtures.ts";

const dir = mkdtempSync(join(tmpdir(), "yian-backward-recovery-"));
afterAll(() => rmSync(dir, { recursive: true, force: true }));
const env = hostEnv(dir);

let files = 0;

/** A fresh path in the file's temp dir. */
const fresh = (name: string) => join(dir, `${files++}-${name}`);

type Pick = (config: Readonly<Record<string, unknown>>) => ChatModel;

/** A pick for a call that must build no model. */
const NO_MODEL: Pick = () => {
  throw new Error("no model may be built");
};

/**
 * Model access over the file's env: each build records its config and serves `pick(config)`; a
 * test may reassign `pick`.
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

/** `advance` in the file's env. */
const resume = (flow: ComposedFlow, options: AdvanceOptions) => advance({ env, ...options });

/** A `Map` serving the memory key `test` with `backend`. */
const testMemory = (backend: unknown) => new Map([["test", backend as FakeMemory]]);

/** An AGENT backward reply: empty channels unless `channels` sets them. */
function feedbackReply(channels: Record<string, unknown> = {}): Response {
  const text = JSON.stringify({ input_feedback: {}, self_feedback: [], variable_feedback: {}, ...channels });
  return new Response({ message: new Message({ role: "assistant", parts: [new TextPart(text)] }) });
}

/** The material of `model`'s backward request `index` (its user message, read as JSON). */
function material(model: ScriptedModel, index = 0): Record<string, any> {
  return JSON.parse(model.requests[index]!.messages[1]!.text());
}

/** The records of the closed durable log at `path`. */
function records(path: string): RunRecord[] {
  const log = JsonlRunLog.resume(path);
  try {
    return log.records().map((envelope) => envelope.record);
  } finally {
    log.close();
  }
}

/** The attempt ids of the `BackwardStarted` records at `path`. */
function startedAttempts(path: string): number[] {
  return records(path).flatMap((record) => (record instanceof BackwardStartedRecord ? [record.attemptId] : []));
}

/** The lines of the log at `path`, each with its newline. */
const linesOf = (path: string) => readFileSync(path, "utf8").split(/(?<=\n)/);

/** The wire kind of one log line. */
const kindOf = (line: string) => (JSON.parse(line) as { kind: string }).kind;

/** The forward fields compared before and after each call. */
const forwardOf = (result: RunResult): unknown[] => [
  result.input,
  result.status,
  result.output,
  result.failure,
  [...result.events],
];

/** The backend's open snapshots, closes, learn contexts and commit calls, counted. */
const counters = (backend: FakeMemory) => [
  backend.openSnapshots.length,
  backend.closeCount,
  backend.learnContexts.length,
  backend.commitCalls.length,
];

/** A thrower for a call that must not happen. */
const fail = (what: string) =>
  function () {
    throw new Error(what);
  };

/**
 * Replace `owner.prototype[name]` with `replacement(original)` while `fn` runs, restoring it after.
 */
async function patched<T>(
  owner: { prototype: object },
  name: string,
  replacement: (original: Function) => Function,
  fn: () => Promise<T>,
): Promise<T> {
  const prototype = owner.prototype as Record<string, unknown>;
  const own = Object.hasOwn(prototype, name);
  const original = prototype[name] as Function;
  prototype[name] = replacement(original);
  try {
    return await fn();
  } finally {
    if (own) prototype[name] = original;
    else delete prototype[name];
  }
}

const CHAIN = `id: backward-recovery
name: backward_recovery
long_term_memory: test
llm_config: {provider: ollama, model: local-test}
input:
  topic: !str
nodes:
  first:
    kind: agent
    input:
      topic: \${input.topic}
    prompt: 'Draft \${topic}'
  second:
    kind: agent
    long_term_memory: test
    input:
      draft: \${first.output}
    prompt: 'Review \${draft}'
output: \${second.output}
`;

const RECORDED_BACKWARD_FLOW = `id: historical-backward
name: historical_backward
long_term_memory: test
nodes:
  work:
    kind: code
    code: return 3
    output: int
output: \${work.output}
`;

/**
 * The record kinds each historical log is cut at. Schema 13 adds
 * the run start's four commits after `run_started` and a `pool_commit` before each success.
 */
const RUN_START_KINDS = ["run_started", "pool_commit", "pool_commit", "pool_commit", "pool_commit"];
const SUCCESS_KINDS = ["node_started", "pool_commit", "node_succeeded"];
const FORWARD_KINDS = [...RUN_START_KINDS, ...SUCCESS_KINDS, ...SUCCESS_KINDS, ...SUCCESS_KINDS];
const BACKWARD_KINDS = [...FORWARD_KINDS, "backward_started", "backward_ready", "memory_commit_requested"];
const RECORDED_BACKWARD_KINDS = [...BACKWARD_KINDS, "memory_commit_succeeded", "backward_finished"];
const RECORDED_BACKWARD_FAILED_KINDS = [...BACKWARD_KINDS, "memory_commit_failed", "backward_finished"];
const RECORDED_FAILURE_KINDS = [...RUN_START_KINDS, ...SUCCESS_KINDS, "node_started", "node_failed"];

/**
 * The lines of a log of the historical flow:
 * its forward run (`"failure"`: the CODE body returns text for its int), and for `"backward"` and
 * `"backward_failed"` one whole attempt whose commit succeeds or is declined.
 */
async function recordedLines(kind: "forward" | "failure" | "backward" | "backward_failed"): Promise<string[]> {
  const source =
    kind === "failure" ? RECORDED_BACKWARD_FLOW.replace("return 3", 'return "bad"') : RECORDED_BACKWARD_FLOW;
  const declined = new MemoryOperationError("store", "declined", "Declined.", false);
  const backend = new FakeMemory({ commitOutcomes: kind === "backward_failed" ? [declined] : [] });
  const path = fresh("recorded.jsonl");
  const result = await run(await load(source), {}, { log: path, learnable: true, memory: testMemory(backend) });
  if (kind === "backward" || kind === "backward_failed") await result.backward();
  const lines = linesOf(path);
  expect(lines.map(kindOf)).toEqual(
    {
      forward: FORWARD_KINDS,
      failure: RECORDED_FAILURE_KINDS,
      backward: RECORDED_BACKWARD_KINDS,
      backward_failed: RECORDED_BACKWARD_FAILED_KINDS,
    }[kind],
  );
  return lines;
}

/** Run the same real two-AGENT public flow before installing backward replies. */
async function chain(options: { workers?: number; backend?: FakeMemory } = {}) {
  const flow = await load(CHAIN);
  const backend = options.backend ?? new FakeMemory();
  const chat = new ScriptedModel([textReply("draft"), textReply("answer")]);
  const access = models(() => chat);
  const path = fresh("chain.jsonl");
  const result = await run(flow, { topic: "reefs" }, {
    log: path,
    learnable: true,
    numWorkers: options.workers ?? 0,
    memory: testMemory(backend),
    models: access,
  });
  expect(result.status).toBe("succeeded");
  expect(result.output).toBe("answer");
  expect(chat.requests).toHaveLength(2);
  expect(backend.learnContexts).toEqual([]);
  expect(backend.commitCalls).toEqual([]);
  return { flow, path, backend, result, forwardChat: chat, access };
}

describe("recorded terminal inspection is inert", () => {
  for (const failed of [false, true]) {
    test(failed ? "failure" : "success", async () => {
      const flow = await load(
        failed ? RECORDED_BACKWARD_FLOW.replace("return 3", 'return "bad"') : RECORDED_BACKWARD_FLOW,
      );
      const wires = await recordedLines(failed ? "failure" : "forward");
      const path = fresh("terminal.jsonl");
      const original = wires.join("");
      writeFileSync(path, original);
      const backend = new FakeMemory();
      const access = models();

      const [result, repeated] = await patched(CodeNode, "forward", () => fail("terminal forward"), () =>
        patched(CodeNode, "backward", () => fail("terminal backward"), () =>
          patched(JsonlRunLog, "append", () => fail("terminal append"), async () => [
            await resume(flow, { log: path, memory: testMemory(backend), models: access }),
            await resume(flow, { log: path, memory: testMemory(backend), models: access }),
          ]),
        ),
      );

      expect(result.status).toBe(failed ? "failed" : "succeeded");
      expect(result.output).toBe(failed ? null : 3n);
      expect(result.events).toEqual([]);
      expect(repeated.events).toEqual([]);
      expect(result.backwardOutcome).toBeNull();
      if (failed) {
        const last = records(path).at(-1) as NodeFailedRecord;
        expect(result.failure).toEqual(last.failure);
        expect(repeated.failure).toEqual(last.failure);
      }
      expect(backend.openSnapshots).toEqual([]);
      expect(backend.learnContexts).toEqual([]);
      expect(backend.commitCalls).toEqual([]);
      expect(backend.closeCount).toBe(0);
      expect(access.configs).toEqual([]);
      expect(readFileSync(path, "utf8")).toBe(original);
    });
  }
});

describe("recorded backward history admits a later attempt at every boundary", () => {
  for (const [failedCommit, boundary, disposition] of [
    // Boundaries past the run start's four commits and three success commits.
    [false, 14, "unrequested"],
    [false, 15, "unrequested"],
    [false, 16, "unknown"],
    [false, 17, "succeeded"],
    [false, 18, "succeeded"],
    [true, 17, "failed"],
    [true, 18, "failed"],
  ] as const) {
    test(`${failedCommit}, ${boundary}, ${disposition}`, async () => {
      const flow = await load(RECORDED_BACKWARD_FLOW);
      const wires = (await recordedLines(failedCommit ? "backward_failed" : "backward")).slice(0, boundary + 1);
      const path = fresh("attempt.jsonl");
      const original = wires.join("");
      writeFileSync(path, original);
      const backend = new FakeMemory();
      const appended: RunRecord[] = [];

      await patched(CodeNode, "forward", () => fail("historical forward reran"), () =>
        patched(CodeNode, "backward", () => fail("empty backward called node"), () =>
          patched(
            JsonlRunLog,
            "append",
            (append) =>
              function (this: JsonlRunLog, record: RunRecord) {
                appended.push(record);
                return append.call(this, record);
              },
            async () => {
              const result = await resume(flow, { log: path, memory: testMemory(backend), models: models() });
              expect(result.status).toBe("succeeded");
              expect(result.output).toBe(3n);
              expect(result.events).toEqual([]);
              expect(result.backwardOutcome!.learning!.policies[0]!.status).toBe(disposition);
              expect(appended).toEqual([]);
              expect(readFileSync(path, "utf8")).toBe(original);
              expect(backend.openSnapshots).toEqual([]);
              expect(backend.learnContexts).toEqual([]);
              expect(backend.commitCalls).toEqual([]);

              const outcome = await result.backward();
              expect(outcome.status).toBe("succeeded");
              expect(backend.openSnapshots).toEqual([{ namespace: "test" }]);
              expect(backend.learnContexts).toHaveLength(1);
              expect(backend.commitCalls).toHaveLength(1);
              expect(
                appended.flatMap((record) => (record instanceof BackwardStartedRecord ? [record.attemptId] : [])),
              ).toEqual([2]);
              const current = readFileSync(path, "utf8");
              expect(current.startsWith(original)).toBe(true);
              const added = linesOf(path).slice(wires.length);
              expect(added.length).toBeGreaterThan(0);
              expect(added.every((line) => (JSON.parse(line) as { schema: number }).schema === SCHEMA_VERSION)).toBe(
                true,
              );
              const beforeReopen = appended.length;
              const inspected = await resume(flow, { log: path, memory: testMemory({}), models: models() });
              expect(inspected.backwardOutcome).toEqual(outcome);
              expect(inspected.events).toEqual([]);
              expect(appended).toHaveLength(beforeReopen);
              expect(readFileSync(path, "utf8")).toBe(current);
            },
          ),
        ),
      );
    });
  }
});

describe("invalid root shapes preserve the public forward result and unopened attempt", () => {
  for (const [failed, value, outputType, feedback] of [
    [false, '{"items": [1, 2]}', "object", { items: [null] }],
    [false, "[[1], [2, 3]]", "list(list(int))", [null, [null]]],
    [false, "[[1], [2, 3]]", "list(list(int))", [null, [null, null, null]]],
    [false, '{"known": 1}', "object", { missing: null }],
    [false, "[1, 2]", "list(int)", { "0": "Review." }],
    [true, "0", "int", {}],
    [true, "0", "int", []],
  ] as const) {
    test(`${failed}, ${value}, ${outputType}, ${JSON.stringify(feedback)}`, async () => {
      const code = failed ? `if (true) throw new Error("forward stopped");\nreturn ${value}` : `return ${value}`;
      const flow = await load(`id: invalid-root-feedback
name: invalid_root_feedback
long_term_memory: test
nodes:
  work:
    kind: code
    code: |
${code.replace(/^/gm, "      ")}
    output: ${outputType}
output: \${work.output}
`);
      const backend = new FakeMemory();
      const path = fresh("invalid.jsonl");
      const access = models();
      const result = await run(flow, {}, { log: path, learnable: true, memory: testMemory(backend), models: access });
      expect(result.status).toBe(failed ? "failed" : "succeeded");
      const forward = forwardOf(result);
      const [previousOutcome, previousLearning] = [result.backwardOutcome, result.learning] as const;
      const before = readFileSync(path, "utf8");
      const counted = counters(backend);

      const error = await rejected(result.backward(feedback));
      expect(error instanceof TypeError || error instanceof RangeError).toBe(true);

      expect(forwardOf(result)).toEqual(forward);
      expect(result.backwardOutcome).toBe(previousOutcome);
      expect(result.learning).toBe(previousLearning);
      expect(readFileSync(path, "utf8")).toBe(before);
      expect(counters(backend)).toEqual(counted);
      expect(records(path).some((record) => record instanceof BackwardStartedRecord)).toBe(false);
      const inspected = await resume(flow, { log: path, memory: testMemory(backend), models: access });
      expect(inspected.events).toEqual([]);
      expect(inspected.backwardOutcome).toBeNull();
      expect([inspected.input, inspected.status, inspected.output, inspected.failure]).toEqual(forward.slice(0, 4));
      expect(readFileSync(path, "utf8")).toBe(before);
      expect(counters(backend)).toEqual(counted);
      expect(access.configs).toEqual([]);
    });
  }
});

describe("real agent failure discards partial reverse results and retries every callback", () => {
  for (const invalidResult of [false, true]) {
    for (const workers of [0, 2]) {
      test(`${invalidResult}, ${workers}`, async () => {
        const { flow, path, backend, result, forwardChat, access } = await chain({ workers });
        const forward = forwardOf(result);
        const forwardBytes = readFileSync(path, "utf8");
        const count = records(path).length;
        const bad = invalidResult
          ? feedbackReply({ input_feedback: { topic: ["valid sibling"], foreign: ["reject"] } })
          : textReply("not JSON");
        const reverse = new ScriptedModel([feedbackReply({ input_feedback: { draft: ["temporary"] } }), bad]);
        access.pick = () => reverse;

        const outcome = await result.backward("First review.");

        expect(outcome.status).toBe("failed");
        expect(outcome.failure!.stage).toBe("backward");
        expect(outcome.failure!.code).toBe(invalidResult ? "backward.result" : "backward.execute");
        expect(outcome.failure!.nodeId).toBe("first");
        expect(outcome.feedback).toBeNull();
        expect(outcome.learning!.status).toBe("incomplete");
        expect(outcome.learning!.policies.map((policy) => policy.status)).toEqual(["unrequested"]);
        const accepted = records(path).slice(count);
        expect(accepted.map((record) => record.constructor)).toEqual([BackwardStartedRecord, BackwardFinishedRecord]);
        expect((accepted.at(-1) as BackwardFinishedRecord).attributedResults).toEqual([]);
        expect(readFileSync(path, "utf8").startsWith(forwardBytes)).toBe(true);
        expect(reverse.requests).toHaveLength(2);
        expect(forwardChat.requests).toHaveLength(2);
        expect([0, 1].map((index) => material(reverse, index)["forward"]["node_id"])).toEqual(["second", "first"]);
        expect(backend.learnContexts).toEqual([]);
        expect(backend.commitCalls).toEqual([]);
        expect(backend.closeCount).toBe(2);
        expect(forwardOf(result)).toEqual(forward);
        const before = readFileSync(path, "utf8");
        const inspected = await resume(flow, { log: path, memory: testMemory(backend), models: access });
        expect(inspected.events).toEqual([]);
        expect(inspected.backwardOutcome).toEqual(outcome);
        expect([inspected.input, inspected.status, inspected.output, inspected.failure]).toEqual(forward.slice(0, 4));
        expect(readFileSync(path, "utf8")).toBe(before);
        expect(reverse.requests).toHaveLength(2);
        expect(backend.openSnapshots).toHaveLength(2);

        const retry = new ScriptedModel([
          feedbackReply({ input_feedback: { draft: ["fresh draft"] } }),
          feedbackReply({ input_feedback: { topic: ["fresh topic"] } }),
        ]);
        access.pick = () => retry;
        const completed = await inspected.backward("Second review.");
        expect(completed.status).toBe("succeeded");
        expect(completed.feedback).toEqual(new BackwardResult({ inputFeedback: { topic: ["fresh topic"] } }));
        expect(retry.requests).toHaveLength(2);
        expect(reverse.requests).toHaveLength(2);
        expect(forwardChat.requests).toHaveLength(2);
        expect(material(retry)["feedback"]).toEqual(["Second review."]);
        expect(startedAttempts(path)).toEqual([1, 2]);
        expect(backend.learnContexts).toHaveLength(1);
        expect(backend.commitCalls).toHaveLength(1);
        const requests = records(path).filter((record) => record instanceof MemoryCommitRequestedRecord);
        expect(requests).toHaveLength(1);
        expect(requests[0]!.mutations).toEqual([]);
        expect(backend.commitCalls).toEqual([[requests[0]!.commitId, requests[0]!.mutations]]);
        expect(backend.openSnapshots).toEqual([null, { namespace: "test" }, { namespace: "test" }]);
        expect(backend.closeCount).toBe(3);
        expect(backend.sessions.every((session) => session.closed)).toBe(true);
        expect(forwardOf(result)).toEqual(forward);
        expect(forwardOf(inspected)).toEqual([...forward.slice(0, 4), []]);
      });
    }
  }
});

describe("terminal resume inspects actual public attempt prefixes without reopening memory", () => {
  for (const [lastKind, disposition] of [
    ["backward_ready", "unrequested"],
    ["memory_commit_requested", "unknown"],
    ["memory_commit_succeeded", "succeeded"],
  ] as const) {
    test(lastKind, async () => {
      const { flow, path, backend, result, forwardChat, access } = await chain();
      const forward = forwardOf(result);
      backend.learnOutcomes.push((context: LearnContext) => [
        new MemoryMutation(`own/${context.memory}`, { retain: true }),
      ]);
      const reverse = new ScriptedModel([
        feedbackReply({ input_feedback: { draft: ["draft review"] } }),
        feedbackReply({ input_feedback: { topic: ["topic review"] } }),
      ]);
      access.pick = () => reverse;
      const completed = await result.backward("Review.");
      expect(completed.status).toBe("succeeded");
      const requests = records(path).filter((record) => record instanceof MemoryCommitRequestedRecord);
      expect(requests).toHaveLength(1);
      expect(requests[0]!.mutations).toHaveLength(1);
      expect(backend.commitCalls).toEqual([[requests[0]!.commitId, requests[0]!.mutations]]);
      const completeBytes = readFileSync(path, "utf8");
      const lines = linesOf(path);
      const boundary = lines.findIndex((line) => kindOf(line) === lastKind);
      const prefixPath = fresh(`${lastKind}.jsonl`);
      const prefixBytes = lines.slice(0, boundary + 1).join("");
      writeFileSync(prefixPath, prefixBytes);
      const counted = counters(backend);
      access.pick = NO_MODEL;
      const built = access.configs.length;

      const inspected = await resume(flow, { log: prefixPath, memory: testMemory({}), models: access });

      const outcome = inspected.backwardOutcome!;
      expect(outcome.status).toBe("failed");
      expect(outcome.failure!.code).toBe("backward.unfinished");
      expect(outcome.feedback).toEqual(completed.feedback);
      expect(outcome.feedback).toEqual(new BackwardResult({ inputFeedback: { topic: ["topic review"] } }));
      expect(outcome.learning!.status).toBe("incomplete");
      expect(outcome.learning!.policies).toHaveLength(1);
      const policy = outcome.learning!.policies[0]!;
      expect(policy.memory).toBe("test");
      expect(policy.status).toBe(disposition);
      expect(policy.failure).toBeNull();
      expect(policy.commitId === null).toBe(disposition === "unrequested");
      expect(inspected.events).toEqual([]);
      expect([inspected.input, inspected.status, inspected.output, inspected.failure]).toEqual(forward.slice(0, 4));
      expect(forwardOf(result)).toEqual(forward);
      expect(readFileSync(prefixPath, "utf8")).toBe(prefixBytes);
      expect(readFileSync(path, "utf8")).toBe(completeBytes);
      expect(counters(backend)).toEqual(counted);
      expect(access.configs).toHaveLength(built);
      expect(reverse.requests).toHaveLength(2);
      expect(forwardChat.requests).toHaveLength(2);
      const ready = records(prefixPath).filter((record) => record instanceof BackwardReadyRecord);
      expect(ready).toHaveLength(1);
      expect(ready[0]!.rootResult).toEqual(completed.feedback!);
      expect(ready[0]!.attributedResults).toHaveLength(4);
      expect(ready[0]!.proposals).toHaveLength(1);
      expect(ready[0]!.proposals.every((proposal) => proposal.mutations.length === 1)).toBe(true);
    });
  }
});

describe("late learning failure keeps completed feedback but retry rebuilds every proposal", () => {
  for (const failureKind of ["learner", "duplicate_proposal"]) {
    test(failureKind, async () => {
      const { flow, path, backend, result, forwardChat, access } = await chain();
      const forward = forwardOf(result);
      let before = readFileSync(path, "utf8");
      const count = records(path).length;
      backend.learnOutcomes.push(
        failureKind === "duplicate_proposal"
          ? () => [new MemoryMutation("duplicate", { attempt: 1 }), new MemoryMutation("duplicate", { attempt: 1 })]
          : new MemoryOperationError("learner", "declined", "Learner declined.", false),
      );
      const reverse = new ScriptedModel([
        feedbackReply({ input_feedback: { draft: ["old draft"] } }),
        feedbackReply({ input_feedback: { topic: ["old topic"] } }),
      ]);
      access.pick = () => reverse;
      const outcome = await result.backward("First review.");
      expect(outcome.status).toBe("failed");
      expect(outcome.failure!.stage).toBe("learn");
      expect(outcome.failure!.code).toBe(failureKind === "learner" ? "declined" : "memory.invalid_result");
      expect(outcome.feedback).toEqual(new BackwardResult({ inputFeedback: { topic: ["old topic"] } }));
      expect(outcome.learning!.status).toBe("failed");
      expect(outcome.learning!.policies[0]!.status).toBe("unrequested");
      const accepted = records(path).slice(count);
      expect(accepted.map((record) => record.constructor)).toEqual([BackwardStartedRecord, BackwardFinishedRecord]);
      expect((accepted.at(-1) as BackwardFinishedRecord).attributedResults).toHaveLength(4);
      expect(readFileSync(path, "utf8").startsWith(before)).toBe(true);
      expect(reverse.requests).toHaveLength(2);
      expect(backend.learnContexts).toHaveLength(1);
      expect(backend.commitCalls).toEqual([]);
      expect(outcome.failure!.memory).toBe("test");
      expect(backend.learnContexts[0]!.memory).toBe("test");
      expect(backend.closeCount).toBe(2);
      expect(forwardOf(result)).toEqual(forward);
      before = readFileSync(path, "utf8");
      const inspected = await resume(flow, { log: path, memory: testMemory(backend), models: access });
      expect(inspected.events).toEqual([]);
      expect(inspected.backwardOutcome).toEqual(outcome);
      expect([inspected.input, inspected.status, inspected.output, inspected.failure]).toEqual(forward.slice(0, 4));
      expect(readFileSync(path, "utf8")).toBe(before);
      expect(reverse.requests).toHaveLength(2);
      expect(backend.openSnapshots).toHaveLength(2);

      backend.learnOutcomes.push((context: LearnContext) => [
        new MemoryMutation(`fresh/${context.memory}`, { attempt: 2 }),
      ]);
      const retry = new ScriptedModel([
        feedbackReply({ input_feedback: { draft: ["fresh draft"] } }),
        feedbackReply({ input_feedback: { topic: ["fresh topic"] } }),
      ]);
      access.pick = () => retry;
      const completed = await inspected.backward("Second review.");
      expect(completed.status).toBe("succeeded");
      expect(completed.feedback).toEqual(new BackwardResult({ inputFeedback: { topic: ["fresh topic"] } }));
      expect(retry.requests).toHaveLength(2);
      expect(reverse.requests).toHaveLength(2);
      expect(forwardChat.requests).toHaveLength(2);
      expect(backend.learnContexts).toHaveLength(2);
      expect(backend.commitCalls).toHaveLength(1);
      const [original, repeated] = backend.learnContexts as [LearnContext, LearnContext];
      expect(original.memory).toBe(repeated.memory);
      expect(original.episodes).toEqual(repeated.episodes);
      expect(original.feedback).not.toEqual(repeated.feedback);
      const committed = backend.commitCalls[0]![1];
      expect(committed).toHaveLength(1);
      expect(committed.every((mutation) => mutation.mutationId.startsWith("fresh/"))).toBe(true);
      expect(committed.every((mutation) => JSON.stringify(mutation.payload) === '{"attempt":2}')).toBe(true);
      const requests = records(path).filter((record) => record instanceof MemoryCommitRequestedRecord);
      expect(requests).toHaveLength(1);
      expect(requests[0]!.mutations).toHaveLength(1);
      expect(requests[0]!.attemptId).toBe(2);
      expect(backend.commitCalls).toEqual([[requests[0]!.commitId, requests[0]!.mutations]]);
      expect(backend.openSnapshots).toEqual([null, { namespace: "test" }, { namespace: "test" }]);
      expect(backend.closeCount).toBe(3);
      expect(backend.sessions.every((session) => session.closed)).toBe(true);
      expect(forwardOf(result)).toEqual(forward);
      expect(forwardOf(inspected)).toEqual([...forward.slice(0, 4), []]);
    });
  }
});

/**
 * Run `fn` with `LearnContext.fromFrozen` throwing `RangeError("context too deep")` on its first
 * call (START's occurrence), passing later calls through.
 */
async function withFailingFirstContext<T>(fn: () => Promise<T>): Promise<T> {
  const fromFrozen = LearnContext.fromFrozen;
  let calls = 0;
  const spy = spyOn(LearnContext, "fromFrozen").mockImplementation((...args) => {
    calls += 1;
    if (calls === 1) throw new RangeError("context too deep");
    return fromFrozen.apply(LearnContext, args);
  });
  try {
    return await fn();
  } finally {
    spy.mockRestore();
  }
}

test("unmaterializable owner context is a complete known failure on each explicit retry", async () => {
  const flow = await load(`id: deep-owner-context
name: deep_owner_context
long_term_memory: test
input:
  payload: !object
nodes: {}
output: \${input.payload}
`);
  let value: Record<string, unknown> = { leaf: "retained" };
  for (let level = 0; level < 20; level++) value = { next: value };
  const backend = new FakeMemory();
  const path = fresh("deep.jsonl");
  const access = models();
  const result = await run(flow, { payload: value }, { log: path, learnable: true, memory: testMemory(backend), models: access });
  expect(result.status).toBe("succeeded");
  expect(result.output).toEqual(value);
  const forward = forwardOf(result);
  let before = readFileSync(path, "utf8");
  const first = await withFailingFirstContext(() => result.backward("First review."));
  expect(first.failure!.stage).toBe("learn");
  expect(first.failure!.code).toBe("memory.invalid_context");
  // START's episode is the first material the context construction refuses.
  expect([first.failure!.nodeId, first.failure!.nodeStartedSeq]).toEqual(["__start__", 5]);
  expect(first.failure!.message).toBe(
    "memory learn context could not be constructed for node '__start__': context too deep",
  );
  expect(first.feedback).toEqual(new BackwardResult({ inputFeedback: { payload: ["First review."] } }));
  expect(first.learning!.status).toBe("failed");
  expect(first.learning!.policies[0]!.status).toBe("unrequested");
  expect(readFileSync(path, "utf8").startsWith(before)).toBe(true);
  expect(forwardOf(result)).toEqual(forward);
  before = readFileSync(path, "utf8");
  const inspected = await resume(flow, { log: path, memory: testMemory(backend), models: access });
  expect(inspected.events).toEqual([]);
  expect(inspected.backwardOutcome).toEqual(first);
  expect([inspected.input, inspected.status, inspected.output, inspected.failure]).toEqual(forward.slice(0, 4));
  expect(readFileSync(path, "utf8")).toBe(before);
  expect(backend.openSnapshots).toHaveLength(2);
  const second = await withFailingFirstContext(() => inspected.backward("Second review."));
  expect(second.failure).toEqual(first.failure);
  expect(second.feedback).toEqual(new BackwardResult({ inputFeedback: { payload: ["Second review."] } }));
  expect(backend.learnContexts).toEqual([]);
  expect(backend.commitCalls).toEqual([]);
  const all = records(path);
  expect(startedAttempts(path)).toEqual([1, 2]);
  const finished = all.filter((record) => record instanceof BackwardFinishedRecord);
  expect(finished).toHaveLength(2);
  expect(finished.every((record) => record.attributedResults.length === 2)).toBe(true);
  expect(
    all.some((record) => record instanceof BackwardReadyRecord || record instanceof MemoryCommitRequestedRecord),
  ).toBe(false);
  expect(backend.openSnapshots).toEqual([null, { namespace: "test" }, { namespace: "test" }]);
  expect(backend.closeCount).toBe(3);
  expect(backend.sessions.every((session) => session.closed)).toBe(true);
  expect(forwardOf(result)).toEqual(forward);
  expect(forwardOf(inspected)).toEqual([...forward.slice(0, 4), []]);
  expect(access.configs).toEqual([]);
});

test("reverse cancellation retains started only and resume recomputes the entire pass", async () => {
  const { flow, path, backend, result, forwardChat, access } = await chain();
  const forward = forwardOf(result);
  let before = readFileSync(path, "utf8");
  const count = records(path).length;
  // The process is lost while the second reverse call fails: no later record is accepted.
  const reverse = new CrashAt(
    [feedbackReply({ input_feedback: { draft: ["temporary"] } }), new Error("stop reverse work")],
    2,
  );
  access.pick = () => reverse;
  await crashes(reverse, () => result.backward("First review."));
  expect(records(path).slice(count).map((record) => record.constructor)).toEqual([BackwardStartedRecord]);
  expect(readFileSync(path, "utf8").startsWith(before)).toBe(true);
  const outcome = result.backwardOutcome!;
  expect(outcome.failure!.code).toBe("backward.unfinished");
  expect(outcome.feedback).toBeNull();
  expect(outcome.learning!.status).toBe("incomplete");
  expect(outcome.learning!.policies[0]!.status).toBe("unrequested");
  expect(reverse.requests).toHaveLength(2);
  expect(forwardChat.requests).toHaveLength(2);
  expect(backend.learnContexts).toEqual([]);
  expect(backend.commitCalls).toEqual([]);
  expect(backend.closeCount).toBe(2);
  expect(forwardOf(result)).toEqual(forward);

  before = readFileSync(path, "utf8");
  const inspected = await resume(flow, { log: path, memory: testMemory(backend), models: access });
  expect(inspected.events).toEqual([]);
  expect(inspected.backwardOutcome).toEqual(outcome);
  expect([inspected.input, inspected.status, inspected.output, inspected.failure]).toEqual(forward.slice(0, 4));
  expect(readFileSync(path, "utf8")).toBe(before);
  expect(reverse.requests).toHaveLength(2);
  expect(backend.openSnapshots).toHaveLength(2);

  const retry = new ScriptedModel([
    feedbackReply({ input_feedback: { draft: ["fresh draft"] } }),
    feedbackReply({ input_feedback: { topic: ["fresh topic"] } }),
  ]);
  access.pick = () => retry;
  expect((await inspected.backward("Second review.")).feedback).toEqual(
    new BackwardResult({ inputFeedback: { topic: ["fresh topic"] } }),
  );
  const accepted = records(path).slice(count);
  expect(accepted.slice(0, 3).map((record) => record.constructor)).toEqual([
    BackwardStartedRecord,
    BackwardFinishedRecord,
    BackwardStartedRecord,
  ]);
  expect((accepted[1] as BackwardFinishedRecord).outcome.feedback).toBeNull();
  expect((accepted[1] as BackwardFinishedRecord).attributedResults).toEqual([]);
  expect((accepted[2] as BackwardStartedRecord).attemptId).toBe(2);
  expect(retry.requests).toHaveLength(2);
  expect(reverse.requests).toHaveLength(2);
  expect(forwardChat.requests).toHaveLength(2);
  expect(backend.learnContexts).toHaveLength(1);
  expect(backend.commitCalls).toHaveLength(1);
  const requests = records(path).filter((record) => record instanceof MemoryCommitRequestedRecord);
  expect(requests).toHaveLength(1);
  expect(requests[0]!.mutations).toEqual([]);
  expect(backend.commitCalls).toEqual([[requests[0]!.commitId, requests[0]!.mutations]]);
  expect(backend.openSnapshots).toEqual([null, { namespace: "test" }, { namespace: "test" }]);
  expect(backend.closeCount).toBe(3);
  expect(backend.sessions.every((session) => session.closed)).toBe(true);
  expect(forwardOf(result)).toEqual(forward);
  expect(forwardOf(inspected)).toEqual([...forward.slice(0, 4), []]);
});

describe("public append rejection keeps exact authority and retries after every phase", () => {
  for (const [phase, rejectedType] of [
    ["started", BackwardStartedRecord],
    ["ready", BackwardReadyRecord],
    ["request", MemoryCommitRequestedRecord],
    ["partial_request", MemoryCommitRequestedRecord],
    ["success_observation", MemoryCommitSucceededRecord],
    ["failure_observation", MemoryCommitFailedRecord],
    ["finished", BackwardFinishedRecord],
    ["failed_finished", BackwardFinishedRecord],
  ] as const) {
    test(phase, async () => {
      const declined = phase === "failure_observation" || phase === "failed_finished";
      const { flow, path, backend, result, forwardChat, access } = await chain({
        backend: new FakeMemory({
          commitOutcomes: declined ? [new MemoryOperationError("store", "declined", "Store declined.", false)] : [],
        }),
      });
      const forward = forwardOf(result);
      const forwardBytes = readFileSync(path, "utf8");
      const count = records(path).length;
      let generation = 1;
      const propose = (context: LearnContext) => [
        new MemoryMutation(`pass-${generation}/${context.memory}`, { generation }),
      ];
      backend.learnOutcomes.push(propose, propose);
      const reverse = new ScriptedModel([
        feedbackReply({ input_feedback: { draft: ["old draft"] } }),
        feedbackReply({ input_feedback: { topic: ["old topic"] } }),
      ]);
      access.pick = () => reverse;
      const error = new Error("rejected append");
      const rejectedPrefixes: string[] = [];
      const partialWrites: Uint8Array[] = [];

      const caught = await patched(
        JsonlRunLog,
        "append",
        (append) =>
          function (this: JsonlRunLog, record: RunRecord) {
            if (!(record instanceof rejectedType)) return append.call(this, record);
            rejectedPrefixes.push(readFileSync(path, "utf8"));
            if (phase !== "partial_request") throw error;
            // Half the line reaches the file before the write fails; the append rolls it back.
            const write = fs.writeSync;
            const spy = spyOn(fs, "writeSync").mockImplementation(((
              fd: number,
              buffer: Uint8Array,
              offset: number,
              length: number,
              position: number,
            ) => {
              const partial = buffer.subarray(offset, offset + Math.floor(length / 2));
              write(fd, partial, 0, partial.length, position);
              expect(statSync(path).size).toBeGreaterThan(Buffer.byteLength(rejectedPrefixes.at(-1)!));
              partialWrites.push(partial);
              throw error;
            }) as never);
            try {
              return append.call(this, record);
            } finally {
              spy.mockRestore();
            }
          },
        () => rejected(result.backward("First review.")),
      );
      expect(caught).toBe(error);
      expect(rejectedPrefixes).toEqual([readFileSync(path, "utf8")]);
      expect(partialWrites.length > 0).toBe(phase === "partial_request");
      expect(readFileSync(path, "utf8").startsWith(forwardBytes)).toBe(true);
      const expected: Function[] = phase === "started" ? [] : [BackwardStartedRecord];
      if (phase !== "started" && phase !== "ready") expected.push(BackwardReadyRecord);
      const committedPhase = ["success_observation", "failure_observation", "finished", "failed_finished"].includes(phase);
      if (committedPhase) expected.push(MemoryCommitRequestedRecord);
      if (phase === "finished") expected.push(MemoryCommitSucceededRecord);
      if (phase === "failed_finished") expected.push(MemoryCommitFailedRecord);
      const accepted = records(path).slice(count);
      expect(accepted.map((record) => record.constructor)).toEqual(expected);
      const expectedCallbacks = phase === "started" ? 0 : 2;
      const expectedLearns = phase === "started" ? 0 : 1;
      const expectedCommits = committedPhase ? 1 : 0;
      expect(reverse.requests).toHaveLength(expectedCallbacks);
      expect(forwardChat.requests).toHaveLength(2);
      expect(backend.learnContexts).toHaveLength(expectedLearns);
      expect(backend.commitCalls).toHaveLength(expectedCommits);
      if (expectedCommits > 0) {
        const requests = accepted.filter((record) => record instanceof MemoryCommitRequestedRecord);
        expect(requests).toHaveLength(1);
        expect(requests[0]!.mutations).toHaveLength(1);
        expect(backend.commitCalls).toEqual([[requests[0]!.commitId, requests[0]!.mutations]]);
      }
      expect(backend.closeCount).toBe(2);
      expect(backend.sessions.every((session) => session.closed)).toBe(true);
      expect(forwardOf(result)).toEqual(forward);
      const outcome = result.backwardOutcome;
      if (phase === "started") {
        expect(outcome).toBeNull();
        expect(result.learning).toBeNull();
      } else {
        expect(outcome!.failure!.code).toBe("backward.unfinished");
        if (phase === "ready") expect(outcome!.feedback).toBeNull();
        else expect(outcome!.feedback).toEqual(new BackwardResult({ inputFeedback: { topic: ["old topic"] } }));
        expect(outcome!.learning!.status).toBe(phase === "failed_finished" ? "failed" : "incomplete");
        const disposition = ({
          ready: "unrequested",
          request: "unrequested",
          partial_request: "unrequested",
          success_observation: "unknown",
          failure_observation: "unknown",
          finished: "succeeded",
          failed_finished: "failed",
        } as const)[phase];
        expect(outcome!.learning!.policies[0]!.status).toBe(disposition);
      }

      const before = readFileSync(path, "utf8");
      const inspected = await resume(flow, { log: path, memory: testMemory(backend), models: access });
      expect(inspected.events).toEqual([]);
      expect(inspected.backwardOutcome).toEqual(outcome);
      expect([inspected.input, inspected.status, inspected.output, inspected.failure]).toEqual(forward.slice(0, 4));
      expect(readFileSync(path, "utf8")).toBe(before);
      expect(reverse.requests).toHaveLength(expectedCallbacks);
      expect(backend.openSnapshots).toHaveLength(2);

      generation = 2;
      const retry = new ScriptedModel([
        feedbackReply({ input_feedback: { draft: ["fresh draft"] } }),
        feedbackReply({ input_feedback: { topic: ["fresh topic"] } }),
      ]);
      access.pick = () => retry;
      const completed = await inspected.backward("Second review.");
      expect(completed.status).toBe("succeeded");
      expect(completed.feedback).toEqual(new BackwardResult({ inputFeedback: { topic: ["fresh topic"] } }));
      expect(retry.requests).toHaveLength(2);
      expect(reverse.requests).toHaveLength(expectedCallbacks);
      expect(forwardChat.requests).toHaveLength(2);
      expect(backend.learnContexts).toHaveLength(expectedLearns + 1);
      expect(backend.commitCalls).toHaveLength(expectedCommits + 1);
      const all = records(path);
      const requests = all.filter((record) => record instanceof MemoryCommitRequestedRecord);
      expect(requests).toHaveLength(expectedCommits + 1);
      const request = requests.at(-1)!;
      expect(request.mutations).toHaveLength(1);
      expect(backend.commitCalls.at(-1)).toEqual([request.commitId, request.mutations]);
      expect(
        backend.commitCalls
          .at(-1)![1]
          .every(
            (mutation) =>
              mutation.mutationId.startsWith("pass-2/") && JSON.stringify(mutation.payload) === '{"generation":2}',
          ),
      ).toBe(true);
      expect(startedAttempts(path)).toEqual(phase === "started" ? [1] : [1, 2]);
      if (phase !== "started") {
        const prior = all.filter((record) => record instanceof BackwardFinishedRecord && record.attemptId === 1);
        expect(prior).toHaveLength(1);
        expect((prior[0] as BackwardFinishedRecord).outcome.failure!.code).toBe("backward.unfinished");
        expect((prior[0] as BackwardFinishedRecord).outcome.feedback).toEqual(outcome!.feedback);
      }
      expect(backend.openSnapshots).toEqual([null, { namespace: "test" }, { namespace: "test" }]);
      expect(backend.closeCount).toBe(3);
      expect(backend.sessions.every((session) => session.closed)).toBe(true);
      expect(forwardOf(result)).toEqual(forward);
      expect(forwardOf(inspected)).toEqual([...forward.slice(0, 4), []]);
    });
  }
});

test("three memory failure preserves ordered dispositions and admits a public retry", async () => {
  let source = CHAIN.replace("long_term_memory: test", "long_term_memory: A");
  source = source.replace("  first:\n    kind: agent", "  first:\n    kind: agent\n    long_term_memory: B");
  const flow = await load(source.replace("    long_term_memory: test", "    long_term_memory: C"));
  const backends = new Map(["A", "B", "C"].map((name) => [name, new FakeMemory()]));
  backends.get("B")!.commitOutcomes.push(new MemoryOperationError("store", "backward.unfinished", "Store declined.", true));
  const forwardChat = new ScriptedModel([textReply("draft"), textReply("answer")]);
  const access = models(() => forwardChat);
  const path = fresh("policies.jsonl");
  const result = await run(flow, { topic: "reefs" }, { log: path, learnable: true, memory: backends, models: access });
  expect(result.status).toBe("succeeded");
  expect(result.output).toBe("answer");
  const forward = forwardOf(result);
  const forwardBytes = readFileSync(path, "utf8");
  const count = records(path).length;
  const learned: string[] = [];
  for (const [name, backend] of backends) {
    backend.learnOutcomes.push(() => {
      for (const each of backends.values()) expect(each.commitCalls).toEqual([]);
      learned.push(name);
      return [new MemoryMutation("same-id", { policy: name })];
    });
  }
  const reverse = new ScriptedModel([
    feedbackReply({ input_feedback: { draft: ["draft review"] } }),
    feedbackReply({ input_feedback: { topic: ["topic review"] } }),
  ]);
  access.pick = () => reverse;
  const outcome = await result.backward("Review.");

  expect(learned).toEqual(["A", "B", "C"]);
  expect(outcome.status).toBe("failed");
  expect(outcome.failure!.stage).toBe("commit");
  expect(outcome.failure!.code).toBe("backward.unfinished");
  expect(outcome.failure!.memory).toBe("B");
  expect(outcome.feedback).toEqual(new BackwardResult({ inputFeedback: { topic: ["topic review"] } }));
  expect(outcome.learning!.status).toBe("failed");
  expect(outcome.learning!.policies.map((policy) => [policy.memory, policy.status])).toEqual([
    ["A", "succeeded"],
    ["B", "failed"],
    ["C", "unrequested"],
  ]);
  expect(outcome.learning!.policies[1]!.failure).toEqual(outcome.failure!.cause as never);
  expect((outcome.failure!.cause as { retryable: boolean }).retryable).toBe(true);
  const accepted = records(path).slice(count);
  expect(accepted.map((record) => record.constructor)).toEqual([
    BackwardStartedRecord,
    BackwardReadyRecord,
    MemoryCommitRequestedRecord,
    MemoryCommitSucceededRecord,
    MemoryCommitRequestedRecord,
    MemoryCommitFailedRecord,
    BackwardFinishedRecord,
  ]);
  const requests = accepted.filter((record) => record instanceof MemoryCommitRequestedRecord);
  expect(requests.map((record) => record.memory)).toEqual(["A", "B"]);
  for (const request of requests) {
    expect(request.mutations).toHaveLength(1);
    expect(backends.get(request.memory)!.commitCalls).toEqual([[request.commitId, request.mutations]]);
  }
  const ready = accepted[1] as BackwardReadyRecord;
  expect(ready.proposals.map((proposal) => proposal.memory)).toEqual(
    [...backends.values()].map((backend) => backend.learnContexts[0]!.memory),
  );
  expect(ready.proposals.map((proposal) => proposal.mutations[0]!.payload)).toEqual(
    [...backends.keys()].map((name) => ({ policy: name })),
  );
  expect([...backends.values()].map((backend) => backend.commitCalls.length)).toEqual([1, 1, 0]);
  expect(readFileSync(path, "utf8").startsWith(forwardBytes)).toBe(true);
  expect(reverse.requests).toHaveLength(2);
  expect(forwardChat.requests).toHaveLength(2);
  for (const backend of backends.values()) {
    expect(backend.learnContexts).toHaveLength(1);
    expect(backend.closeCount).toBe(2);
    expect(backend.sessions.every((session) => session.closed)).toBe(true);
  }
  expect(forwardOf(result)).toEqual(forward);
  const before = readFileSync(path, "utf8");
  const retry = new ScriptedModel([
    feedbackReply({ input_feedback: { draft: ["draft again"] } }),
    feedbackReply({ input_feedback: { topic: ["topic again"] } }),
  ]);
  access.pick = () => retry;

  const inspected = await resume(flow, { log: path, memory: backends, models: access });
  expect(inspected.events).toEqual([]);
  expect(inspected.backwardOutcome).toEqual(outcome);
  expect([inspected.input, inspected.status, inspected.output, inspected.failure]).toEqual(forward.slice(0, 4));
  const completed = await inspected.backward("Second review.");
  expect(completed.status).toBe("succeeded");
  expect(completed.learning!.status).toBe("succeeded");
  expect(completed.learning!.policies.map((policy) => policy.status)).toEqual(["succeeded", "succeeded", "succeeded"]);
  expect(readFileSync(path, "utf8").startsWith(before)).toBe(true);
  expect(retry.requests).toHaveLength(2);
  expect([...backends.values()].map((backend) => backend.commitCalls.length)).toEqual([2, 2, 1]);
  expect(startedAttempts(path)).toEqual([1, 2]);
  expect(forwardOf(result)).toEqual(forward);
  expect(forwardOf(inspected)).toEqual([...forward.slice(0, 4), []]);
});

describe("public replay rejects complete record corruption before any new effects", () => {
  for (const corruption of ["root_projection", "requested_batch"]) {
    test(corruption, async () => {
      const { flow, path, backend, result, forwardChat, access } = await chain();
      const forward = forwardOf(result);
      backend.learnOutcomes.push((context: LearnContext) => [
        new MemoryMutation(`own/${context.memory}`, { retain: true }),
      ]);
      const reverse = new ScriptedModel([
        feedbackReply({ input_feedback: { draft: ["draft review"] } }),
        feedbackReply({ input_feedback: { topic: ["topic review"] } }),
      ]);
      access.pick = () => reverse;
      expect((await result.backward("Review.")).status).toBe("succeeded");
      const requests = records(path).filter((record) => record instanceof MemoryCommitRequestedRecord);
      expect(requests).toHaveLength(1);
      expect(requests[0]!.mutations).toHaveLength(1);
      expect(backend.commitCalls).toEqual([[requests[0]!.commitId, requests[0]!.mutations]]);
      const [previousOutcome, previousLearning] = [result.backwardOutcome, result.learning] as const;
      const counted = counters(backend);
      const lines = linesOf(path);
      const kind = corruption === "root_projection" ? "backward_ready" : "memory_commit_requested";
      const index = lines.findIndex((line) => kindOf(line) === kind);
      // Read and written in the log's codec, so ints and floats keep their spelling.
      const record = parseJson(lines[index]!, "codec") as Record<string, any>;
      if (corruption === "root_projection") {
        record["payload"]["root_result"]["input_feedback"]["topic"] = ["forged projection"];
      } else {
        record["payload"]["mutations"][0]["payload"] = { forged: true };
      }
      lines[index] = `${logJson(record)}\n`;
      const corrupted = lines.join("");
      writeFileSync(path, corrupted);
      access.pick = NO_MODEL;
      const built = access.configs.length;

      const message = corruption === "root_projection" ? "root result must match" : "batch and identity must match";
      const error = await rejected(result.backward("Review after tampering."));
      expect(error).toBeInstanceOf(RangeError);
      expect((error as Error).message).toContain(message);
      expect(result.backwardOutcome).toBe(previousOutcome);
      expect(result.learning).toBe(previousLearning);
      const resumeError = await rejected(resume(flow, { log: path, memory: testMemory(backend), models: access }));
      expect(resumeError).toBeInstanceOf(RangeError);
      expect((resumeError as Error).message).toContain(message);

      expect(readFileSync(path, "utf8")).toBe(corrupted);
      expect(counters(backend)).toEqual(counted);
      expect(access.configs).toHaveLength(built);
      expect(reverse.requests).toHaveLength(2);
      expect(forwardChat.requests).toHaveLength(2);
      expect(forwardOf(result)).toEqual(forward);
    });
  }
});
