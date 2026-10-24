/**
 * TS-only: every write to the variable pool is a `pool_commit` record. The run start writes the
 * system values and the START inputs as commits right after `run_started`; a node's output is a
 * commit its `node_succeeded` names by seq. An orphan commit (its node never succeeded) is ignored
 * on replay and its node reruns; a log cut inside the run start can't be resumed.
 *
 * Also checks the public JSON form hosts read `pool_commit` values with (`TypedValue`, `logJson`,
 * `parseJson` from `yian/run`), on Bun and, with `YIAN_TEST_NODE` set to a Node binary, on the
 * built package.
 */

import { execFileSync, spawnSync } from "node:child_process";
import { existsSync, mkdtempSync, readdirSync, readFileSync, realpathSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";

import { afterAll, expect, test } from "bun:test";

import type { ComposedFlow } from "../../src/compose/build/pipeline.ts";
import { loadFlow } from "../../src/compose/loader.ts";
import { StartNode } from "../../src/nodes/start/node.ts";
import {
  FileRef,
  inspectRun,
  iterRunRecords,
  JSONDecodeError,
  type JsonObject,
  logJson,
  parseJson,
  replayRunLog,
  advance,
  runFlow,
  runRecordOffsets,
  TypedValue,
} from "../../src/run.ts";
import {
  decodeRecord,
  encodeRecord,
  JsonlRunLog,
  MemoryRunLog,
  NodeSucceededRecord,
  PoolCommitRecord,
  readRunLog,
  RecordEnvelope,
  SCHEMA_VERSION,
} from "../../src/runtime/log.ts";
import { TypedValue as RuntimeTypedValue } from "../../src/typesys/values.ts";
import { crashAfter, NO_ENV, rejected } from "./run_state_fixtures.ts";

const tmp = mkdtempSync(join(tmpdir(), "yian-pool-commit-"));
afterAll(() => rmSync(tmp, { recursive: true, force: true }));

let logs = 0;
/** A fresh log path under the test's temp folder. */
const freshLog = () => join(tmp, `log-${++logs}.jsonl`);

/** A loop whose body CALLs another def, so child ENDs commit under their spawners. */
const CALL_IN_LOOP = `
id: count
name: count
defs:
  add1:
    input: {x: !int}
    nodes:
      inc: {kind: code, input: {x: "\${input.x}"}, output: {y: int}, code: "return {y: inputs.x + 1}"}
    output: {y: "\${inc.output.y}"}
  body:
    input: {x: !int}
    nodes:
      step: {kind: call, call: add1, input: {x: "\${input.x}"}}
    output: {x: "\${step.output.y}"}
nodes:
  count: {kind: loop, call: body, input: {x: 0}, times: 2}
output: \${count.output.x}
`;

/** A MAP over three items. */
const DOUBLES = `
id: doubles
name: doubles
input: {xs: !list(int)}
defs:
  double:
    input: {x: !int}
    nodes:
      times2: {kind: code, input: {x: "\${input.x}"}, output: {y: int}, code: "return {y: inputs.x * 2}"}
    output: \${times2.output.y}
nodes:
  each: {kind: map, over: "\${input.xs}", call: double, input: {x: "\${item}"}}
output: \${each.output}
`;

const load = (text: string) => loadFlow(text, { sourceLabel: "flow.yaml" });

/** Run `flow` serially (or with `numWorkers`) into `log`. */
const run = (flow: ComposedFlow, log: string, inputs: Record<string, unknown> = {}, numWorkers = 0) =>
  runFlow(flow, inputs, { env: NO_ENV, log, runId: "r1", numWorkers });

/** The log's lines, each without its newline. */
const lines = (log: string) => readFileSync(log, "utf8").split("\n").slice(0, -1);

/** The seq of every `pool_commit` a `node_succeeded` names, by naming record seq. */
function named(envelopes: readonly RecordEnvelope[]): Map<number, number> {
  const commits = new Map<number, number>();
  for (const { kind, seq, payload } of envelopes) {
    if (kind === "node_succeeded") commits.set(seq, Number(payload["commit"]));
  }
  return commits;
}

/**
 * Return `line`, a `pool_commit` record, re-encoded through the public JSON form: its payload's
 * value decoded with `TypedValue.fromWire` and written back with `toWire`.
 */
function roundTrip(line: string): string {
  const envelope = parseJson(line, "codec") as JsonObject & { payload: JsonObject };
  const { payload } = envelope;
  const typed = TypedValue.fromWire({ kind: payload["kind"], value: payload["value"] });
  return logJson({ ...envelope, payload: { ...payload, ...typed.toWire() } });
}

/** A flow whose commits hold big ints, infinities, tag-shaped records, a date and a datetime. */
const VALUES = `
id: values
name: values
input: {when: !date}
nodes:
  huge: {kind: code, output: int, code: "return 2n ** 80n"}
  negative: {kind: code, output: float, code: "return -Infinity"}
  nums: {kind: code, output: list(float), code: "return [1n, 0.5, Infinity]"}
  make:
    kind: code
    input: {when: "\${input.when}"}
    output: {big: int, inf: float, tag: object, esc: object, when: date}
    code: |
      return {big: -(2n ** 80n), inf: Infinity, tag: {$float: "inf"}, esc: {$record: {a: 1n}}, when: inputs.when}
output: {make: "\${make.output}", whole: "\${0.5 + 1.5}"}
`;

const ROOT = join(import.meta.dirname, "../..");
const BUILT = join(ROOT, "dist/src/run.js");
/**
 * The public entry. It resolves by self-reference from inside ROOT (this file, and children whose
 * cwd is ROOT). Typed string so tsc doesn't resolve it to dist/, missing before a build.
 */
const ENTRY: string = "@ngocbh/yian/run";

/**
 * The Node binary named by `YIAN_TEST_NODE`, or `undefined` when it's unset. A set
 * value that can't run, a Node older than the `engines` floor, or a missing build
 * fails the whole file, so a typo can't pass as a run with the Node case skipped.
 * A copy of `tests/llm_clients.test.ts`'s: importing a test file would run its tests.
 */
const NODE = ((): string | undefined => {
  const node = process.env.YIAN_TEST_NODE;
  if (!node) return undefined;
  let version: string;
  try {
    version = execFileSync(node, ["--version"], { encoding: "utf8" }).trim();
  } catch (error) {
    throw new Error(`YIAN_TEST_NODE=${node} doesn't run`, { cause: error });
  }
  const [major = 0, minor = 0] = version.replace(/^v/, "").split(".").map(Number);
  if (major < 22 || (major === 22 && minor < 18)) {
    throw new Error(`YIAN_TEST_NODE=${node} is Node ${version}; the Node checks need 22.18 or later`);
  }
  if (!existsSync(BUILT)) throw new Error(`${BUILT} is missing; run bun run build first`);
  return node;
})();

test("the run start writes its system values, then the START inputs, as commits after run_started", async () => {
  const log = freshLog();
  await run(await load(CALL_IN_LOOP), log);
  const envelopes = readRunLog(log);
  expect(envelopes[0]!.kind).toBe("run_started");
  expect(Object.hasOwn(envelopes[0]!.payload, "initial_pool")).toBe(false);
  const start = envelopes.slice(1, 5).map(({ kind, payload }) => [kind, payload["node_id"], payload["key"]]);
  expect(start).toEqual([
    ["pool_commit", StartNode.ID, "system.now"],
    ["pool_commit", StartNode.ID, "system.run_id"],
    ["pool_commit", StartNode.ID, "system.today"],
    ["pool_commit", StartNode.ID, StartNode.ID],
  ]);
  expect(envelopes[2]!.payload["value"]).toBe("r1");
  expect(envelopes[4]!.payload).toEqual({ node_id: StartNode.ID, key: StartNode.ID, kind: "object", value: {} });
  expect(envelopes[5]!.kind).toBe("node_started");
});

test("every node_succeeded names an earlier commit of its node, or a child START its expansion's, and each commit is named once", async () => {
  const log = freshLog();
  await run(await load(CALL_IN_LOOP), log);
  const envelopes = readRunLog(log);
  const commits = named(envelopes);
  expect(commits.size).toBeGreaterThan(5);
  for (const [seq, commit] of commits) {
    const named = envelopes[commit]!;
    const nodeId = envelopes[seq]!.payload["node_id"];
    expect(named.kind).toBe("pool_commit");
    expect(commit).toBeLessThan(seq);
    // A child START's commit is its spawner's, under the START's key.
    if (named.payload["node_id"] !== nodeId) expect([nodeId, named.payload["key"]]).toEqual([nodeId, nodeId]);
  }
  const nodeCommits = envelopes.filter(({ kind, seq }) => kind === "pool_commit" && seq > 4).map(({ seq }) => seq);
  expect([...commits.values()].sort((a, b) => a - b)).toEqual(nodeCommits);
});

test("each commit's key is its target: a CALL's child END and the stopping LOOP driver commit under their spawner, an expansion under its child START", async () => {
  const log = freshLog();
  await run(await load(CALL_IN_LOOP), log);
  const envelopes = readRunLog(log);
  const moved = envelopes
    .filter(({ kind, seq, payload }) => kind === "pool_commit" && seq > 4 && payload["key"] !== payload["node_id"])
    .map(({ payload }) => [payload["node_id"], payload["key"]]);
  // Schema 12 recorded the END and driver keys as the successes' `commit_target`s.
  expect(moved).toEqual([
    ["count", "count#0/__start__"],
    ["count#0/step", "count#0/step/__start__"],
    ["count#0/step/__end__", "count#0/step"],
    ["count~1", "count#1/__start__"],
    ["count#1/step", "count#1/step/__start__"],
    ["count#1/step/__end__", "count#1/step"],
    ["count~2", "count"],
  ]);
});

test("a cut between a node's commit and its success reruns the node, leaving the orphan unread", async () => {
  const flow = await load(CALL_IN_LOOP);
  const uncutLog = freshLog();
  const uncut = await run(flow, uncutLog);
  const uncutEnvelopes = readRunLog(uncutLog);
  const cuts = [...named(uncutEnvelopes).values()];
  for (const k of cuts) {
    const node = uncutEnvelopes[k]!.payload["node_id"] as string;
    const key = uncutEnvelopes[k]!.payload["key"] as string;
    const log = freshLog();
    await crashAfter(k + 1, () => run(flow, log));
    const cutEnvelopes = readRunLog(log);
    expect(cutEnvelopes.at(-1)!.kind).toBe("pool_commit");

    // The orphan is ignored on replay: the pool holds what it held before the commit.
    const inspection = inspectRun(log, { flow, env: NO_ENV });
    expect(inspection.status).toBe("crashed");
    expect([...inspection.inFlight]).toContain(node);
    const replayed = (count: number) => replayRunLog(flow, MemoryRunLog.fromRecords(cutEnvelopes.slice(0, count)));
    const before = replayed(k).pool.store.get(key);
    const after = replayed(k + 1).pool.store.get(key);
    expect(after === undefined ? undefined : logJson(after.toWire())).toBe(
      before === undefined ? undefined : logJson(before.toWire()),
    );

    const orphan = lines(log)[k];
    const resumed = await advance({ log, env: NO_ENV });
    expect(resumed.status).toBe("succeeded");
    expect(resumed.output).toEqual(uncut.output);
    expect(lines(log)[k]).toBe(orphan);
    const envelopes = readRunLog(log);
    const starts = envelopes.filter(({ kind, payload }) => kind === "node_started" && payload["node_id"] === node);
    expect(starts).toHaveLength(2);
    const commits = envelopes.filter(({ kind, payload }) => kind === "pool_commit" && payload["node_id"] === node);
    const rerun = commits.find(({ seq }) => seq > k)!;
    expect(rerun).toBeDefined();
    // An expansion commit is named by its child START's success, any other by its node's.
    const success = envelopes.find(({ kind, payload }) => kind === "node_succeeded" && Number(payload["commit"]) === rerun.seq);
    expect(success!.payload["node_id"]).toBe(key.endsWith("/__start__") ? key : node);
  }
}, 60_000);

/**
 * Return the log text `envelopes` would have with the record at `from` moved to just after the
 * record at `to`, cut after the record that followed `to`, every seq renumbered.
 */
function interleaved(text: readonly string[], to: number, from: number): string {
  const records = text.map((line) => parseJson(line, "codec") as JsonObject);
  const moved = records.splice(from, 1)[0]!;
  records.splice(to + 1, 0, moved);
  return records
    .slice(0, to + 3)
    .map((record, seq) => `${logJson({ ...record, seq: BigInt(seq) })}\n`)
    .join("");
}

test("parallel nodes whose commits interleave: only the node without its success reruns", async () => {
  const flow = await load(DOUBLES);
  const uncutLog = freshLog();
  await run(flow, uncutLog, { xs: [1, 2, 3] }, 2);
  const envelopes = readRunLog(uncutLog);
  // The first success whose commit comes after another node's start, and a later node's commit.
  const successes = envelopes.filter(({ kind }) => kind === "node_succeeded");
  let pair: [number, number] | null = null;
  for (const success of successes) {
    const node = success.payload["node_id"] as string;
    const commit = Number(success.payload["commit"]);
    const other = envelopes.find(
      ({ kind, seq, payload }) => kind === "pool_commit" && seq > success.seq && payload["node_id"] !== node &&
        envelopes.some((e) => e.kind === "node_started" && e.seq < commit && e.payload["node_id"] === payload["node_id"]),
    );
    if (other !== undefined) {
      pair = [commit, other.seq];
      break;
    }
  }
  expect(pair).not.toBeNull();
  const [first, second] = pair!;
  const finished = envelopes[first]!.payload["node_id"] as string;
  const rerun = envelopes[second]!.payload["node_id"] as string;
  const log = freshLog();
  writeFileSync(log, interleaved(lines(uncutLog), first, second));
  const cut = readRunLog(log);
  expect(cut.slice(-3).map(({ kind, payload }) => [kind, payload["node_id"]])).toEqual([
    ["pool_commit", finished],
    ["pool_commit", rerun],
    ["node_succeeded", finished],
  ]);

  const resumed = await advance({ log, env: NO_ENV });
  expect(resumed.output).toEqual([2n, 4n, 6n]);
  const after = readRunLog(log);
  const startsOf = (node: string) => after.filter(({ kind, payload }) => kind === "node_started" && payload["node_id"] === node);
  expect(startsOf(finished)).toHaveLength(1);
  expect(startsOf(rerun)).toHaveLength(2);
  const success = after.find(({ kind, payload }) => kind === "node_succeeded" && payload["node_id"] === finished)!;
  expect(Number(success.payload["commit"])).toBe(first);
});

/** Replace the `commit` of the record at `seq` in the log `text` and return the new text. */
function withCommit(text: readonly string[], seq: number, commit: number): string {
  const record = parseJson(text[seq]!, "codec") as { payload: JsonObject };
  record.payload["commit"] = BigInt(commit);
  return [...text.slice(0, seq), logJson(record), ...text.slice(seq + 1)].map((line) => `${line}\n`).join("");
}

test("a node_succeeded naming a commit that isn't its node's pending one is refused on replay", async () => {
  const flow = await load(CALL_IN_LOOP);
  const log = freshLog();
  await run(flow, log);
  const text = lines(log);
  const envelopes = readRunLog(log);
  const successes = [...named(envelopes)];
  const [seq, own] = successes[2]!;
  const node = envelopes[seq]!.payload["node_id"] as string;
  const other = successes[1]![1];
  const started = envelopes.find(({ kind, payload }) => kind === "node_started" && payload["node_id"] === node)!.seq;
  for (const commit of [other, 4, started]) {
    expect(commit).not.toBe(own);
    const edited = freshLog();
    writeFileSync(edited, withCommit(text, seq, commit));
    const replay = () => replayRunLog(flow, MemoryRunLog.fromRecords(readRunLog(edited)));
    expect(replay).toThrow(RangeError);
    expect(replay).toThrow(`node_succeeded of '${node}' names commit ${commit}, not its pending pool_commit ${own}`);
  }
});

test("a pool_commit for a node that isn't running is refused on replay", async () => {
  const flow = await load(CALL_IN_LOOP);
  const log = freshLog();
  await run(flow, log);
  const text = lines(log);
  const envelopes = readRunLog(log);
  const first = envelopes.find(({ kind, seq }) => kind === "pool_commit" && seq > 4)!;
  const record = parseJson(text[first.seq]!, "codec") as { payload: JsonObject };
  record.payload["node_id"] = "count";
  const edited = freshLog();
  writeFileSync(edited, [...text.slice(0, first.seq), logJson(record)].map((line) => `${line}\n`).join(""));
  const replay = () => replayRunLog(flow, MemoryRunLog.fromRecords(readRunLog(edited)));
  expect(replay).toThrow(RangeError);
  expect(replay).toThrow("pool_commit node 'count' is not in flight");
});

test("a log cut inside its run start can't be resumed, and stays as it is", async () => {
  const flow = await load(CALL_IN_LOOP);
  for (const k of [1, 2, 3, 4]) {
    const log = freshLog();
    await crashAfter(k, () => run(flow, log));
    const before = readFileSync(log);
    const error = await rejected(advance({ log, env: NO_ENV }));
    expect(error).toBeInstanceOf(RangeError);
    expect((error as Error).message).toBe(`run log ${log} ends inside its run start; start the run again`);
    expect(readFileSync(log).equals(before)).toBe(true);
    // `inspectRun` replays the log too, and refuses it the same way.
    expect(() => inspectRun(log, { flow, env: NO_ENV })).toThrow(RangeError);
    expect(() => inspectRun(log, { flow, env: NO_ENV })).toThrow("ends inside its run start; start the run again");
  }
});

test("runRecordOffsets maps each seq to its line's byte offset, without the writer's lock", async () => {
  const log = freshLog();
  await run(await load(CALL_IN_LOOP), log);
  const text = readFileSync(log);
  const expected: number[] = [];
  for (let offset = 0; offset < text.length; offset = text.indexOf(0x0a, offset) + 1) expected.push(offset);
  const writer = JsonlRunLog.resume(log);
  try {
    const offsets = runRecordOffsets(log);
    expect(offsets).toEqual(expected);
    for (const k of [0, 1, 5, offsets.length - 1]) {
      const first = iterRunRecords(log, { afterOffset: offsets[k]! }).next().value!;
      expect(first.seq).toBe(k);
    }
  } finally {
    writer.close();
  }
});

test("every pool_commit line of the conformance golden logs round-trips byte for byte through the public JSON form", () => {
  const cases = join(import.meta.dirname, "../../../conformance/cases");
  let checked = 0;
  const kinds = new Set<string>();
  for (const name of readdirSync(cases, { recursive: true, encoding: "utf8" })) {
    if (!name.endsWith(".log.jsonl")) continue;
    for (const line of lines(join(cases, name))) {
      const { kind, payload } = parseJson(line, "codec") as { kind: string; payload: JsonObject };
      // The conformance masks `system.now` and `system.today`; a derived commit holds no value, nor
      // does a take commit, which names the events it took by seq.
      const valueless = Object.hasOwn(payload, "from") || Object.hasOwn(payload, "events");
      if (kind !== "pool_commit" || payload["value"] === "<masked>" || valueless) continue;
      expect(roundTrip(line)).toBe(line);
      kinds.add(payload["kind"] as string);
      checked++;
    }
  }
  // So a moved fixture folder can't pass empty.
  expect(checked).toBeGreaterThanOrEqual(80);
  expect([...kinds]).toEqual(expect.arrayContaining(["object", "str", "int", "float", "list[int]"]));
});

test("a run's pool_commit values with big ints, infinities and tag-shaped records round-trip through the public JSON form", async () => {
  const log = freshLog();
  const result = await run(await load(VALUES), log, { when: "2026-10-23" });
  expect(result.status).toBe("succeeded");
  const kinds = new Set<string>();
  const values = new Map<string, unknown>();
  let end: string | undefined;
  for (const line of lines(log)) {
    const { kind, payload } = parseJson(line, "codec") as { kind: string; payload: JsonObject };
    if (kind !== "pool_commit") continue;
    expect(roundTrip(line)).toBe(line);
    kinds.add(payload["kind"] as string);
    values.set(payload["node_id"] as string, TypedValue.fromWire({ kind: payload["kind"], value: payload["value"] }).value);
    if (payload["node_id"] === "__end__") end = line;
  }
  expect([...kinds]).toEqual(expect.arrayContaining(["datetime", "date", "int", "float", "list[float]", "object"]));
  expect(values.get("huge")).toBe(2n ** 80n);
  expect(values.get("negative")).toBe(-Infinity);
  const make = values.get("make") as Record<string, unknown>;
  expect(make["big"]).toBe(-(2n ** 80n));
  expect(make["tag"]).toEqual({ $float: "inf" });
  expect(end).toContain('"whole":2.0');
  expect((values.get("__end__") as Record<string, unknown>)["whole"]).toBe(2);
});

test("iterRunRecords hands a host the pool_commit payload the public JSON form decodes", async () => {
  const log = freshLog();
  expect((await run(await load(VALUES), log, { when: "2026-10-23" })).status).toBe("succeeded");
  let checked = 0;
  for (const { envelope } of iterRunRecords(log, { kinds: ["pool_commit"] })) {
    const { payload } = envelope;
    const typed = TypedValue.fromWire({ kind: payload["kind"], value: payload["value"] });
    expect(logJson({ ...payload, ...typed.toWire() })).toBe(logJson(payload));
    checked++;
  }
  const commits = lines(log).filter((line) => (parseJson(line, "codec") as { kind: string }).kind === "pool_commit");
  expect(checked).toBe(commits.length);
});

test("the public JSON form reads a file as a FileRef and refuses text the log refuses", () => {
  const file = TypedValue.fromWire({ kind: "file", value: { uri: "s3://b/k", mime: null, name: "k" } });
  expect(FileRef.isFileRef(file.value)).toBe(true);
  expect(logJson(file.toWire())).toBe('{"kind":"file","value":{"mime":null,"name":"k","uri":"s3://b/k"}}');
  expect(() => parseJson("Infinity", "codec")).toThrow(JSONDecodeError);
});

test.if(process.versions.bun !== undefined)("on Bun the run entry is the source module", async () => {
  expect(realpathSync(fileURLToPath(import.meta.resolve(ENTRY)))).toBe(realpathSync(join(ROOT, "src/run.ts")));
  const entry = (await import(ENTRY)) as typeof import("../../src/run.ts");
  expect(entry.TypedValue).toBe(RuntimeTypedValue);
});

/**
 * A `pool_commit` line as the log writes it, holding a big int, both infinities, an escaped record
 * and an integral float: fixed text, so a change to the spelling the writer and reader share fails.
 */
const WRITTEN_LINE =
  '{"at":"2026-10-23T20:22:41.126Z","kind":"pool_commit","payload":{"key":"make","kind":"object","node_id":"make",' +
  '"value":{"big":1208925819614629174706176,"esc":{"$record":{"$record":{"a":1}}},"inf":{"$float":"inf"},' +
  '"ninf":{"$float":"-inf"},"whole":2.0}},"run_id":"r1","schema":21,"seq":9}';

test("a fixed pool_commit line with a big int, infinities and an escaped record round-trips through the public JSON form", () => {
  expect(roundTrip(WRITTEN_LINE)).toBe(WRITTEN_LINE);
  const { payload } = parseJson(WRITTEN_LINE, "codec") as { payload: JsonObject };
  const value = TypedValue.fromWire({ kind: payload["kind"], value: payload["value"] }).value as Record<string, unknown>;
  expect(value).toEqual({ big: 2n ** 80n, esc: { $record: { a: 1n } }, inf: Infinity, ninf: -Infinity, whole: 2 });
});

test.skipIf(!NODE)("on Node the built run entry round-trips a pool_commit line with a big int and infinities", () => {
  const script = `
const { FileRef, JSONDecodeError, TypedValue, logJson, parseJson } = await import(${JSON.stringify(ENTRY)});
const line = ${JSON.stringify(WRITTEN_LINE)};
const envelope = parseJson(line, "codec");
const { payload } = envelope;
const typed = TypedValue.fromWire({ kind: payload.kind, value: payload.value });
const value = typed.value;
const again = logJson({ ...envelope, payload: { ...payload, ...typed.toWire() } });
const file = TypedValue.fromWire({ kind: "file", value: { uri: "s3://b/k", mime: null, name: "k" } }).value;
let decodeError = false;
try {
  parseJson("NaN", "codec");
} catch (error) {
  decodeError = error instanceof JSONDecodeError;
}
console.log(JSON.stringify({
  resolved: import.meta.resolve(${JSON.stringify(ENTRY)}),
  same: again === line,
  big: typeof value.big,
  inf: value.inf === Infinity,
  ninf: value.ninf === -Infinity,
  esc: Object.keys(value.esc),
  whole: [typeof value.whole, value.whole],
  file: FileRef.isFileRef(file),
  decodeError,
}));
`;
  const result = spawnSync(NODE!, ["--input-type=module", "-e", script], {
    cwd: ROOT,
    encoding: "utf8",
    timeout: 60_000,
  });
  expect(result.status, `signal: ${result.signal}\nstderr: ${result.stderr}`).toBe(0);
  expect(result.stderr).toBe("");
  const { resolved, ...checks } = JSON.parse(result.stdout);
  expect(realpathSync(fileURLToPath(resolved))).toBe(realpathSync(BUILT));
  expect(checks).toEqual({
    same: true,
    big: "bigint",
    inf: true,
    ninf: true,
    esc: ["$record"],
    whole: ["number", 2],
    file: true,
    decodeError: true,
  });
}, 70_000);

test("a messages commit holds steps and nothing else", () => {
  const record = new PoolCommitRecord("reply", "reply.new_messages", null, { steps: [5, 6, 9] });
  expect(record.steps).toEqual([5, 6, 9]);
  expect(Object.isFrozen(record.steps)).toBe(true);
  expect([record.value, record.from, record.take]).toEqual([null, null, null]);
  const text = encodeRecord(new RecordEnvelope(SCHEMA_VERSION, "r1", 10, record, "2026-10-24T00:00:00.000Z"));
  expect(text).toContain('"payload":{"key":"reply.new_messages","node_id":"reply","steps":[5,6,9]}');
  expect((decodeRecord(text).record as PoolCommitRecord).steps).toEqual([5, 6, 9]);
});

test("a messages commit may name no steps; other commits have steps null", () => {
  expect(new PoolCommitRecord("reply", "reply.new_messages", null, { steps: [] }).steps).toEqual([]);
  expect(new PoolCommitRecord("x", "x", { kind: "str", value: "v" }).steps).toBeNull();
});

test.each([[[3, 3]], [[4, 2]], [[0]], [[1.5]], [["1"]]])("steps %j is refused", (steps) => {
  expect(() => new PoolCommitRecord("reply", "reply.new_messages", null, { steps: steps as number[] })).toThrow(
    "steps must be strictly increasing positive integers",
  );
});

test("a pool_commit with steps and another form is refused", () => {
  expect(() => new PoolCommitRecord("reply", "reply", { kind: "str", value: "x" }, { steps: [1] })).toThrow(
    "a pool_commit holds exactly one of value, from, take and steps",
  );
});

test("node_succeeded names its messages commit, or null", () => {
  const withMessages = new NodeSucceededRecord("reply", 9, { messages: 10 });
  const text = encodeRecord(new RecordEnvelope(SCHEMA_VERSION, "r1", 11, withMessages, "2026-10-24T00:00:00.000Z"));
  expect(text).toContain('"payload":{"commit":9,"forward_capture":null,"messages":10,"node_id":"reply"}');
  expect((decodeRecord(text).record as NodeSucceededRecord).messages).toBe(10);
  const without = encodeRecord(new RecordEnvelope(SCHEMA_VERSION, "r1", 4, new NodeSucceededRecord("x", 3), null));
  expect(without).toContain('"payload":{"commit":3,"forward_capture":null,"messages":null,"node_id":"x"}');
  expect((decodeRecord(without).record as NodeSucceededRecord).messages).toBeNull();
  expect(() => new NodeSucceededRecord("x", 3, { messages: 0 })).toThrow("messages must be a positive integer");
});
