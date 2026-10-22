/**
 * Values stored for `Any` across a durable run log: replay and resume keep them unchanged: a port
 * of `tests/engine/test_any_replay.py`.
 *
 * A value typed `Any` keeps its natural carrier (an int stays an int, a bool stays a bool, a list
 * is stored as `list[Any]`). Replay re-checks each recorded value against its declared type and
 * requires the same canonical wire back, so that check must be idempotent; a node that runs after
 * resume must see the same values a straight run sees.
 *
 * Ported: every row. The CODE bodies are written in JS: `big` returns `2n ** 80n`, and `seen`
 * names each input's JS `typeof` (`null` for null), where Python names its class. A host stop is
 * the `ask_user` handler aborting the run's signal and throwing its reason, where Python raises
 * `_HostStop`; the run rejects with that reason. The committed values are read with the TS log
 * codec's `decodeRecord`, which also checks every record, where Python reads the raw JSON after
 * `decode_record`.
 *
 * Changed: ints read as `bigint`s (decision 4a), and a CODE body receives an int within ±2^53 as a
 * `number` (decision 45), so `seen` names an int and a float alike, and `2**80` as `bigint`. A
 * whole number a CODE body returns reads back as an int (decision 9A), so the floats are `2.5`
 * and `1.5` where Python's are `2.0` and `1.0`.
 */

import { afterAll, expect, test } from "bun:test";
import { mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { loadFlow } from "../../src/compose/loader.ts";
import { NodeStarted } from "../../src/events.ts";
import { type RunResult, resumeFlow, runFlow } from "../../src/run.ts";
import { decodeRecord } from "../../src/runtime/log.ts";
import { ValueKind } from "../../src/typesys/values.ts";
import { hostEnv } from "./mcp_helpers.ts";
import { rejection } from "./run_state_fixtures.ts";

const dir = mkdtempSync(join(tmpdir(), "yian-any-replay-"));
afterAll(() => rmSync(dir, { recursive: true, force: true }));
const env = hostEnv(dir);

const ANY_FLOW = `
id: any-replay
name: any_replay
input:
  x: !Any
nodes:
  mixed:
    kind: code
    output: Any
    code: |
      return [1, 2.5, true, null]
  obj:
    kind: code
    output: Any
    code: |
      return {"a": [1, null]}
  big:
    kind: code
    output: Any
    code: |
      return 2n ** 80n
  flt:
    kind: code
    output: Any
    code: |
      return 1.5
  items:
    kind: code
    output: list(Any)
    code: |
      return [1, "two"]
  nothing:
    kind: code
    output: Optional(Any)
    code: |
      return null
  rec:
    kind: code
    output:
      v: Any
    code: |
      return {"v": [1, true]}
  ask:
    kind: human_input
    input:
      mixed: \${mixed.output}
      obj: \${obj.output}
      big: \${big.output}
      flt: \${flt.output}
      items: \${items.output}
      nothing: \${nothing.output}
      rec: \${rec.output}
    prompt: "Go on?"
    output: str
  seen:
    kind: code
    input:
      x: \${input.x}
      mixed: \${mixed.output}
      obj: \${obj.output}
      big: \${big.output}
      flt: \${flt.output}
      items: \${items.output}
      nothing: \${nothing.output}
      rec: \${rec.output}
      answer: \${ask.output}
    code: |
      const names = (v) => {
        if (Array.isArray(v)) return v.map(names)
        if (v === null) return "null"
        if (typeof v === "object") return Object.fromEntries(Object.entries(v).map(([k, i]) => [k, names(i)]))
        return typeof v
      }
      return Object.fromEntries(Object.entries(inputs).filter(([k]) => k !== "answer").map(([k, v]) => [k, names(v)]))
output:
  x: \${input.x}
  mixed: \${mixed.output}
  obj: \${obj.output}
  big: \${big.output}
  flt: \${flt.output}
  items: \${items.output}
  nothing: \${nothing.output}
  rec: \${rec.output}
  seen: \${seen.output}
`;

const INPUT = { x: [1n, 2n] };

const ANY_NODES = ["mixed", "obj", "big", "flt", "items", "nothing", "rec"];

const EXPECTED_OUTPUT = {
  x: [1n, 2n],
  mixed: [1n, 2.5, true, null],
  obj: { a: [1n, null] },
  big: 2n ** 80n,
  flt: 1.5,
  items: [1n, "two"],
  nothing: null,
  rec: { v: [1n, true] },
  seen: {
    x: ["number", "number"],
    mixed: ["number", "number", "boolean", "null"],
    obj: { a: ["number", "null"] },
    big: "bigint",
    flt: "number",
    items: ["number", "string"],
    nothing: "null",
    rec: { v: ["number", "boolean"] },
  },
};

const EXPECTED_KINDS = {
  mixed: ValueKind.LIST_ANY,
  obj: ValueKind.OBJECT,
  big: ValueKind.INTEGER,
  flt: ValueKind.NUMBER,
  items: ValueKind.LIST_ANY,
  nothing: ValueKind.NONE,
  rec: ValueKind.OBJECT,
};

/** Answer the gate. */
const answer = () => "yes";

/** Return each Any node's committed value (`{kind, value}`) from the log at `path`, decoded. */
function committed(path: string): Record<string, { kind: string; value: unknown }> {
  const values: Record<string, { kind: string; value: unknown }> = {};
  for (const line of readFileSync(path, "utf8").split("\n")) {
    if (line === "") continue;
    const record = decodeRecord(line) as unknown as {
      kind: string;
      payload: { node_id: string; value: { kind: string; value: unknown } };
    };
    if (record.kind === "node_succeeded" && ANY_NODES.includes(record.payload.node_id)) {
      values[record.payload.node_id] = record.payload.value;
    }
  }
  return values;
}

/** Run the flow to completion with a log at `path`. */
async function straightRun(path: string): Promise<RunResult> {
  const result = await runFlow(await loadFlow(ANY_FLOW, { env }), INPUT, { env, log: path, tools: { ask_user: answer } });
  expect(result.failure).toBeNull();
  expect(result.status).toBe("succeeded");
  return result;
}

/** The ids of the nodes whose `NodeStarted` `result`'s events hold. */
const startedIds = (result: RunResult) =>
  new Set(result.events.flatMap((event) => (event instanceof NodeStarted ? [event.nodeId] : [])));

test("a straight run stores any values in their natural carriers", async () => {
  const path = join(dir, "straight-1.jsonl");
  const result = await straightRun(path);

  expect(result.output).toStrictEqual(EXPECTED_OUTPUT);
  const values = committed(path);
  expect(Object.fromEntries(Object.entries(values).map(([node, value]) => [node, value.kind]))).toStrictEqual(EXPECTED_KINDS);
  expect(values.mixed!.value).toStrictEqual([1n, 2.5, true, null]);
  expect(values.items!.value).toStrictEqual([1n, "two"]);
  expect(values.rec!.value).toStrictEqual({ v: [1n, true] });
  expect(values.obj!.value).toStrictEqual({ a: [1n, null] });
  expect(values.big!.value).toBe(2n ** 80n);
  const started = JSON.parse(readFileSync(path, "utf8").split("\n")[0]!).payload.initial_pool;
  expect(started.store.__start__).toStrictEqual({ kind: ValueKind.OBJECT, value: { x: [1, 2] } });
});

test("resuming a completed any run returns the same output", async () => {
  const path = join(dir, "done.jsonl");
  const first = await straightRun(path);
  const before = readFileSync(path);

  const resumed = await resumeFlow(await loadFlow(ANY_FLOW, { env }), { log: path, env, tools: { ask_user: answer } });

  expect(resumed.failure).toBeNull();
  expect(resumed.status).toBe("succeeded");
  expect(resumed.output).toStrictEqual(first.output);
  expect(resumed.input).toStrictEqual(INPUT);
  expect(startedIds(resumed).size).toBe(0);
  expect(readFileSync(path).equals(before)).toBe(true);
});

test("a node after resume reads replayed any values unchanged", async () => {
  const straight = join(dir, "straight-2.jsonl");
  const expected = await straightRun(straight);

  const path = join(dir, "stopped.jsonl");
  const interruption = new Error("interrupt before human answer");
  const host = new AbortController();
  const interrupt = () => {
    host.abort(interruption);
    throw interruption;
  };

  const caught = await rejection(
    runFlow(await loadFlow(ANY_FLOW, { env }), INPUT, { env, log: path, signal: host.signal, tools: { ask_user: interrupt } }),
  );
  expect(caught).toBe(interruption);
  expect(committed(path)).toStrictEqual(committed(straight));

  const resumed = await resumeFlow(await loadFlow(ANY_FLOW, { env }), { log: path, env, tools: { ask_user: answer } });

  expect(resumed.failure).toBeNull();
  expect(resumed.status).toBe("succeeded");
  expect(resumed.output).toStrictEqual(expected.output);
  const started = startedIds(resumed);
  expect(ANY_NODES.filter((node) => started.has(node))).toEqual([]);
  expect(started.has("seen")).toBe(true);
  expect(committed(path)).toStrictEqual(committed(straight));
});
