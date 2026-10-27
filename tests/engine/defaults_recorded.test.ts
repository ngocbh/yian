/**
 * Recorded defaults: the `RunStartedRecord.defaults` a run starts with, its codec, and resume and
 * replay checking them against the process's `YIAN_*` values.
 */

import { existsSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { afterAll, afterEach, beforeEach, expect, test } from "bun:test";

import { buildGrow } from "../../src/compile/expand.ts";
import { flowFingerprint } from "../../src/compile/manifest.ts";
import { copyGraph } from "../../src/compile/model.ts";
import { loadFlow } from "../../src/compose/loader.ts";
import {
  BUILTIN,
  DefaultsError,
  DefaultsMismatchError,
  forget,
  install,
  processEnv,
  withEnv,
} from "../../src/defaults.ts";
import { ScriptedModel } from "../../src/llm/testing.ts";
import { advance, runFlow } from "../../src/run.ts";
import { NodeExpandedRecord, readRunLog } from "../../src/runtime/log.ts";
import { RunState, replayRecords } from "../../src/runtime/state.ts";
import { hostEnv } from "./mcp_helpers.ts";
import { rejected, serving, textReply } from "./run_state_fixtures.ts";

import {
  decodeRecord,
  encodeRecord,
  RecordEnvelope,
  RunStartedRecord,
  SCHEMA_VERSION,
} from "../../src/runtime/log.ts";
import { type JsonObject, logJson, parseJson } from "../../src/typesys/json.ts";

// Start each test from the built-in values, whatever `YIAN_*` variables the shell sets.
beforeEach(() => install(BUILTIN));
afterEach(forget);

const dir = mkdtempSync(join(tmpdir(), "yian-defaults-recorded-"));
afterAll(() => rmSync(dir, { recursive: true, force: true }));

/** Assert `fn` throws an error of exactly class `cls` with exactly `message`. */
function expectError(fn: () => unknown, cls: new (...args: never[]) => Error, message: string): void {
  let caught: unknown;
  try {
    fn();
  } catch (error) {
    caught = error;
  }
  expect((caught as Error | undefined)?.constructor).toBe(cls);
  expect((caught as Error).message).toBe(message);
}

/** Return a run-start record with the given recorded defaults. */
function runStarted(defaults?: unknown): RunStartedRecord {
  return new RunStartedRecord({
    flowFingerprint: "a".repeat(64),
    startWiringOverride: {},
    effectiveLlmByStaticPath: {},
    numWorkers: 0,
    defaults: defaults as Record<string, bigint>,
  });
}

const NOT_AN_INT = "RunStarted defaults value for 'MAX_TOTAL_NODES' must be an int";

test.each<[string, unknown, new (...args: never[]) => Error, string]>([
  ["string", { MAX_TOTAL_NODES: "5000" }, TypeError, NOT_AN_INT],
  ["bool", { MAX_TOTAL_NODES: true }, TypeError, NOT_AN_INT],
  ["float", { MAX_TOTAL_NODES: 5000 }, TypeError, NOT_AN_INT],
  ["empty-name", { "": 1n }, RangeError, "RunStarted defaults names must be non-empty strings"],
])("the record rejects a bad recorded value: %s", (_name, recorded, cls, message) => {
  expectError(() => runStarted(recorded), cls, message);
});

test("the record rejects a list of pairs as a type error", () => {
  expectError(() => runStarted([["MAX_TOTAL_NODES", 5000n]]), TypeError, "RunStarted defaults must be a mapping");
  expectError(
    () => runStarted(new Map([["MAX_TOTAL_NODES", 5000n]])),
    TypeError,
    "RunStarted defaults must be a mapping",
  );
});

test("the record keeps its own copy of the values", () => {
  const recorded = { MAX_TOTAL_NODES: 5000n };
  const record = runStarted(recorded);

  recorded.MAX_TOTAL_NODES = 6000n;

  expect(record.defaults).toEqual({ MAX_TOTAL_NODES: 5000n });
  expect(Object.isFrozen(record.defaults)).toBe(true);
  expect(runStarted().defaults).toEqual({});
});

/** Encode `record` as the first line of run `direct-engine-test`. */
function encode(record: RunStartedRecord): string {
  return encodeRecord(new RecordEnvelope(SCHEMA_VERSION, "direct-engine-test", 0, record));
}

/** The decoded payload of `line`, ints as `bigint`s. */
function payloadOf(line: string): JsonObject {
  return (parseJson(line, "codec") as JsonObject)["payload"] as JsonObject;
}

test("recorded values encode and decode", () => {
  const encoded = encode(runStarted({ MAX_TOTAL_NODES: 5000n }));

  expect(payloadOf(encoded)["defaults"]).toEqual({ MAX_TOTAL_NODES: 5000n });
  expect((decodeRecord(encoded).record as RunStartedRecord).defaults).toEqual({ MAX_TOTAL_NODES: 5000n });
});

test("recorded values are sorted by name", () => {
  const encoded = encode(runStarted({ MAX_TOTAL_NODES: 5000n, AGENT_RETRIES: 0n }));

  expect(Object.keys(payloadOf(encoded)["defaults"] as JsonObject)).toEqual(["AGENT_RETRIES", "MAX_TOTAL_NODES"]);
  expect(encoded).toContain('"defaults":{"AGENT_RETRIES":0,"MAX_TOTAL_NODES":5000}');
});

test.each<[string, unknown, new (...args: never[]) => Error, string]>([
  ["string", { MAX_TOTAL_NODES: "5000" }, TypeError, NOT_AN_INT],
  ["bool", { MAX_TOTAL_NODES: true }, TypeError, NOT_AN_INT],
  // A JS number is written as a float token, `5.0`.
  ["float", { MAX_TOTAL_NODES: 5 }, TypeError, NOT_AN_INT],
  ["empty-name", { "": 1n }, RangeError, "RunStarted defaults names must be non-empty strings"],
])("decode rejects a bad recorded value: %s", (_name, recorded, cls, message) => {
  const envelope = parseJson(encode(runStarted()), "codec") as { payload: Record<string, unknown> };
  envelope.payload["defaults"] = recorded;
  expectError(() => decodeRecord(logJson(envelope)), cls, message);
});

test("the default run log writes empty defaults", () => {
  const encoded = encode(runStarted());
  const decoded = decodeRecord(encoded);

  expect(encoded).toContain('"defaults":{}');
  expect((decoded.record as RunStartedRecord).defaults).toEqual({});
  expect(encodeRecord(decoded)).toBe(encoded);
});

// --- runs and resumes --- //

const TWO_STEPS = `
id: two
name: two
nodes:
  first:
    kind: code
    code: |
      return {n: 1}
  second:
    kind: code
    input:
      n: \${first.output.n}
    code: |
      return {n: inputs["n"] + 1}
output: \${second.output}
`;

const MAP = `
id: fan
name: fan
input:
  xs: !list(int)
defs:
  one:
    input:
      v: !int
    nodes:
      echo:
        kind: code
        input:
          v: \${input.v}
        code: |
          return {v: inputs["v"]}
    output: \${echo.output}
nodes:
  each:
    kind: map
    over: \${input.xs}
    call: one
    input:
      v: \${item}
output: \${each.output}
`;

const AGENT = `
id: talk
name: talk
nodes:
  ask:
    kind: agent
    prompt: say hi
output: \${ask.output}
`;

/** An environment holding only the test's credential store and `vars`. */
const envWith = (vars: Readonly<Record<string, string>> = {}) => hostEnv(dir, vars);
const AT_5000 = { YIAN_MAX_TOTAL_NODES: "5000" };

let logs = 0;
/** A fresh log path. */
const freshLog = () => join(dir, `run-${++logs}.jsonl`);

/** The lines of `path`, each with its newline. */
const linesOf = (path: string) => readFileSync(path, "utf8").split(/(?<=\n)/);

/** The `run_started` payload of the log at `path`. */
const runStartedPayload = (path: string) => JSON.parse(linesOf(path)[0]!).payload;

/** Apply `edit` to the `run_started` payload of the log at `path` and write it back canonically. */
function rewriteRunStarted(path: string, edit: (payload: Record<string, unknown>) => void): void {
  const lines = linesOf(path);
  const envelope = parseJson(lines[0]!, "codec") as { payload: Record<string, unknown> };
  edit(envelope.payload);
  lines[0] = encodeRecord(decodeRecord(logJson(envelope))) + "\n";
  writeFileSync(path, lines.join(""));
}

/** Run the two-step flow in `env` into `path`, then keep only the log up to START's success. */
async function interruptedRun(path: string, env = envWith()): Promise<void> {
  const result = await runFlow(await loadFlow(TWO_STEPS), {}, { log: path, env });
  expect(result.status).toBe("succeeded");
  // The run start (RunStarted and four commits), then START's start, commit and success.
  writeFileSync(path, linesOf(path).slice(0, 8).join(""));
}

test("the default run log writes empty defaults (run)", async () => {
  const path = freshLog();
  expect((await runFlow(await loadFlow(TWO_STEPS), {}, { log: path, env: envWith() })).status).toBe("succeeded");
  expect(runStartedPayload(path).defaults).toEqual({});
});

test("resume with a different value fails", async () => {
  const path = freshLog();
  await interruptedRun(path, envWith(AT_5000));
  const before = readFileSync(path);

  const error = await rejected(
    advance({ log: path, env: envWith({ YIAN_MAX_TOTAL_NODES: "6000" }) }),
  );

  expect(error).toBeInstanceOf(DefaultsMismatchError);
  expect((error as Error).message).toContain("YIAN_MAX_TOTAL_NODES is 6000 here");
  expect((error as Error).message).toContain("started with 5000");
  expect(readFileSync(path).equals(before)).toBe(true);
});

test("resume without the variable fails", async () => {
  const path = freshLog();
  await interruptedRun(path, envWith(AT_5000));

  const error = await rejected(advance({ log: path, env: envWith() }));

  expect(error).toBeInstanceOf(DefaultsMismatchError);
  expect((error as Error).message).toContain("set YIAN_MAX_TOTAL_NODES=5000");
});

test("resume of a default run with a variable set fails", async () => {
  const path = freshLog();
  await interruptedRun(path);

  const error = await rejected(advance({ log: path, env: envWith(AT_5000) }));

  expect(error).toBeInstanceOf(DefaultsMismatchError);
  expect((error as Error).message).toContain("unset YIAN_MAX_TOTAL_NODES");
});

test("an old log resumes as all defaults: a plain log", async () => {
  const plain = freshLog();
  await interruptedRun(plain);
  expect((await advance({ log: plain, env: envWith() })).output).toEqual({ n: 2n });
});

test("an old log resumes as all defaults: a stripped log is refused (Changed)", async () => {
  const stripped = freshLog();
  await interruptedRun(stripped, envWith(AT_5000));
  // Written raw: the codec would refuse the stripped record.
  const lines = linesOf(stripped);
  const envelope = parseJson(lines[0]!, "codec") as { payload: Record<string, unknown> };
  delete envelope.payload.defaults;
  lines[0] = logJson(envelope) + "\n";
  writeFileSync(stripped, lines.join(""));
  const before = readFileSync(stripped);

  const error = await rejected(advance({ log: stripped, env: envWith() }));

  expect(error).toBeInstanceOf(RangeError);
  expect((error as Error).message).toContain('invalid run_started payload fields; missing=["defaults"]');
  expect(readFileSync(stripped).equals(before)).toBe(true);
});

test("a mismatch is reported before the fingerprint", async () => {
  const path = freshLog();
  const zeroRetriesEnv = envWith({ YIAN_AGENT_RETRIES: "0" });
  const zeroRetries = await withEnv(zeroRetriesEnv, () => loadFlow(AGENT));
  const result = await runFlow(zeroRetries, {}, {
    log: path,
    env: zeroRetriesEnv,
    llmConfig: { provider: "anthropic", model: "claude-test" },
    models: serving(new ScriptedModel([textReply("hi")])),
  });
  expect(result.status, result.failure?.message).toBe("succeeded");
  const defaultRetries = await loadFlow(AGENT);
  expect(flowFingerprint(defaultRetries.compiled)).not.toBe(flowFingerprint(zeroRetries.compiled));

  // Resuming a finished run is enough: the check runs before anything else in replay.
  const error = await rejected(advance({ log: path, env: envWith() }));

  expect(error).toBeInstanceOf(DefaultsMismatchError);
  expect((error as Error).message).toContain("YIAN_AGENT_RETRIES");
  expect((error as Error).message).not.toContain("fingerprint");
});

test("an unknown recorded name fails", async () => {
  const path = freshLog();
  await interruptedRun(path);
  rewriteRunStarted(path, (payload) => {
    payload.defaults = { MAX_NOT_A_THING: 1n };
  });

  const error = await rejected(advance({ log: path, env: envWith() }));

  expect(error).toBeInstanceOf(DefaultsMismatchError);
  expect((error as Error).message).toContain("YIAN_MAX_NOT_A_THING");
});

/** Run `fn` with `process.env[name]` set to `value`, restoring it after. */
async function withProcessVariable<T>(name: string, value: string, fn: () => Promise<T>): Promise<T> {
  const had = Object.hasOwn(process.env, name);
  const old = process.env[name];
  process.env[name] = value;
  try {
    return await fn();
  } finally {
    if (had) process.env[name] = old;
    else delete process.env[name];
  }
}

test("a variable changed after the first read is ignored", async () => {
  // The process's values are already read (installed); the later variable is never read.
  const path = freshLog();
  await withProcessVariable("YIAN_MAX_TOTAL_NODES", "5000", async () => {
    expect((await runFlow(await loadFlow(TWO_STEPS), {}, { log: path, env: processEnv })).status).toBe("succeeded");
  });
  expect(runStartedPayload(path).defaults).toEqual({});
});

test("a bad variable fails before load", async () => {
  const flow = await loadFlow(TWO_STEPS);
  await withProcessVariable("YIAN_MAX_TOTAL_NODES", "abc", async () => {
    forget();
    const error = await rejected(loadFlow(TWO_STEPS));
    expect(error).toBeInstanceOf(DefaultsError);
    expect((error as Error).message).toContain("YIAN_MAX_TOTAL_NODES");
  });
  forget();

  const path = freshLog();
  const error = await rejected(runFlow(flow, {}, { log: path, env: envWith({ YIAN_MAX_TOTAL_NODES: "abc" }) }));
  expect(error).toBeInstanceOf(DefaultsError);
  expect((error as Error).message).toContain("YIAN_MAX_TOTAL_NODES");
  expect(existsSync(path)).toBe(false);
});

test("a node budget over the limit fails live", async () => {
  const result = await runFlow(await loadFlow(MAP), { xs: [1, 2, 3, 4, 5] }, {
    env: envWith({ YIAN_MAX_TOTAL_NODES: "3" }),
  });

  expect(result.status).toBe("failed");
  expect(result.failure!.message).toContain("expansion exceeded node budget (3)");
});

test("replay takes the node budget from the record", async () => {
  const flow = await loadFlow(MAP);
  const path = freshLog();
  expect((await runFlow(flow, { xs: [1, 2, 3, 4, 5] }, { log: path, env: envWith() })).status).toBe("succeeded");
  const envelopes = readRunLog(path);
  expect(envelopes.some((envelope) => envelope.record instanceof NodeExpandedRecord)).toBe(true);
  install({ ...BUILTIN, maxTotalNodes: 2 });

  const state = replayRecords(new RunState(copyGraph(flow.compiled)), envelopes, { growBuilder: buildGrow });

  expect(state.flow.nodes.size).toBeGreaterThan(2);
  expect(state.runDefaults).toEqual(BUILTIN);
});

test("replay takes a recorded node budget from the record", async () => {
  const flow = await loadFlow(MAP);
  const path = freshLog();
  expect((await runFlow(flow, { xs: [1, 2, 3, 4, 5] }, { log: path, env: envWith(AT_5000) })).status).toBe(
    "succeeded",
  );
  const envelopes = readRunLog(path);
  install({ ...BUILTIN, maxTotalNodes: 2 });

  const state = replayRecords(new RunState(copyGraph(flow.compiled)), envelopes, { growBuilder: buildGrow });

  expect(state.runDefaults.maxTotalNodes).toBe(5000);
});
