/**
 * A `loop` whose `call:` names a tool runs it each turn through a one-node body yian generates at
 * load. The tool gets every carried field as an argument and returns the carried fields that
 * change: with no `output:`, the whole carried record. `retries:`, `output:` and
 * `adaptive_questions:` move to the body's node; the loop keeps `asserts:` and its driver.
 */

import { afterAll, beforeAll, expect, test } from "bun:test";
import { mkdtempSync, realpathSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { flowFingerprint } from "../../src/compile/manifest.ts";
import { loadFlow } from "../../src/compose/loader.ts";
import { outputContractType } from "../../src/nodes/contracts.ts";
import { ToolNode } from "../../src/nodes/tool/node.ts";
import { runFlow, runStartTools, staticNodes } from "../../src/run.ts";
import { tool } from "../../src/tools/function_tools.ts";
import { canonicalTypeLabel } from "../../src/typesys/values.ts";

import { loadErrorAsync } from "../helpers/declarations.ts";
import { hostEnv } from "../helpers/mcp_helpers.ts";

let tmp: string;

beforeAll(() => {
  tmp = realpathSync(mkdtempSync(join(tmpdir(), "yian-loop-tools-")));
});

afterAll(() => rmSync(tmp, { recursive: true, force: true }));

/** A flow with the input `job`, the memory `lessons` and `inputs` (lines under `input:`), and `nodes`. */
const app = (nodes: string, inputs = "") =>
  `id: app\nname: app\ninput:\n  job: !str\n  lessons: !FileMemory\n${inputs}nodes:\n${nodes}`;

/** `app` whose loop `poll` calls `call` with the node's other `lines`. */
const loopApp = (call: string, lines: string, inputs = "") =>
  app(`  poll:\n    kind: loop\n    call: ${call}\n${lines}`, inputs);

/** The lines of a loop carrying `job` and a literal `done`, while not done. */
const POLL = "    input:\n      job: ${input.job}\n      done: false\n    while: not done\n    max: 3\n";

/** The node at `path` of the flow `text` loads. */
async function nodeAt(text: string, path: string) {
  return staticNodes(await loadFlow(text)).find(([each]) => each === path)![1];
}

/** The message and the source text at the range loading `text` fails with. */
async function failure(text: string): Promise<[string, string]> {
  const error = await loadErrorAsync(loadFlow(text));
  const range = error.diagnostic.frames.at(-1)!.range!;
  const lines = text.split("\n");
  const at =
    range.startLine === range.endLine
      ? lines[range.startLine - 1]!.slice(range.startCol - 1, range.endCol - 1)
      : `${lines[range.startLine - 1]!.slice(range.startCol - 1)}...`;
  return [error.message, at];
}

/** A host tool `status`, which finishes the job. */
const status = tool({
  name: "status",
  description: "A job's status.",
  parameters: { type: "object", properties: { job: { type: "string" }, done: { type: "boolean" } } },
  execute: (args: Record<string, unknown>) => ({ job: args["job"], done: true }),
});

test("every carried field is a body input the tool gets, a literal seed and the while: flag included", async () => {
  const text = loopApp("status", POLL);
  const start = await nodeAt(text, "poll/__start__");
  expect(start.params!.map((parameter) => parameter.name)).toEqual(["job", "done"]);
  expect((await nodeAt(text, "poll/poll")).params!.map((parameter) => parameter.name)).toEqual(["job", "done"]);
});

test("with no output:, the tool's result is checked against the whole carried record", async () => {
  const node = (await nodeAt(loopApp("status", POLL), "poll/poll")) as ToolNode;
  expect(canonicalTypeLabel(outputContractType(node.outputContract)!)).toBe('{"done":bool,"job":str}');
});

test("retries: and output: move to the body's node; the loop keeps asserts: and its driver", async () => {
  const lines = `${POLL}    retries: 2\n    output: {done: !bool}\n    asserts: [output.done]\n    node_name: Poll\n`;
  const node = (await nodeAt(loopApp("status", lines), "poll/poll")) as ToolNode;
  expect(node.retries).toBe(2);
  expect(canonicalTypeLabel(outputContractType(node.outputContract)!)).toBe('{"done":bool}');
  expect(node.postAsserts).toEqual([]);

  const loop = (await loadFlow(loopApp("status", lines))).compiled.nodes.get("poll")!;
  expect(loop.title).toBe("Poll");
  expect(loop.postAsserts.map((assertion) => assertion.display)).toEqual(["output.done"]);
});

test("with no output:, an untagged null seed is refused at the seed; a tagged one loads", async () => {
  const lines = (seed: string) => `    input:\n      job: \${input.job}\n      cursor: ${seed}\n    times: 2\n`;
  expect(await failure(loopApp("status", lines("null")))).toEqual([
    "loop node 'poll': carried field 'cursor' is typed None from its seed, and tool `status` returns every " +
      "carried field, so it could only return null there; declare the seed's type, e.g. " +
      "`cursor: !Optional(str) null`",
    "null",
  ]);
  const node = (await nodeAt(loopApp("status", lines("!Optional(str) null")), "poll/poll")) as ToolNode;
  expect(canonicalTypeLabel(outputContractType(node.outputContract)!)).toBe('{"cursor":Optional(str),"job":str}');
});

// Each row: the loop's `call:` and other lines, the load error, and the text it points at.
test.each([
  ["status", `${POLL}    retries: -1\n`, "node 'poll' (kind=loop): `retries:` must be a non-negative integer, got -1", "-1"],
  [
    "{crm: sleep}",
    POLL,
    "node 'poll' (kind=loop): `call: {crm: sleep}`: sleep is a built-in, not a group's tool; write `call: sleep`",
    "{crm: sleep}",
  ],
  ["{lessons: raed}", POLL, "node 'poll' (kind=loop) grants {lessons: [raed]}: `lessons` has no tool `raed`", "{lessons: raed}"],
  [
    "{lessons: read}",
    "    input:\n      path: a.md\n    while: path == lessons\n    max: 2\n",
    "loop node 'poll': `while:` reads `lessons`, a memory; a loop predicate can't read a memory",
    "poll",
  ],
  [
    "status",
    `${POLL}    adaptive_questions: {prompt: hi}\n`,
    "node 'poll' (call: status): `adaptive_questions:` applies only to call: ask",
    "adaptive_questions",
  ],
  [
    "status",
    `${POLL}    retry: 2\n`,
    "node 'poll' (kind=loop): field 'retry' is not allowed (allowed: adaptive_questions, asserts, call, depends_on, " +
      "env, input, kind, max, node_name, on_max_exceeded, output, retries, runs_after, times, until, while)",
    "2",
  ],
  [
    "status",
    "    input:\n      max-results: 5\n    times: 1\n",
    "loop node 'poll': carried field 'max-results' must be an identifier (ASCII letters, digits, and _, not " +
      "starting with a digit) so ${input.max-results} can read it; try 'max_results'",
    "max-results",
  ],
])("call: %s with %j fails at the author's field", async (call, lines, message, at) => {
  const [got, where] = await failure(loopApp(call, lines));
  expect(got).toContain(message);
  expect(where).toBe(at);
});

test("a tool other than take given an inbox gets the loop's inbox rule, naming the tool", async () => {
  const text = loopApp("status", "    input:\n      box: ${input.box}\n    times: 1\n", "  box: !Inbox(str)\n");
  expect((await failure(text))[0]).toBe(
    "loop node 'poll': tool `status` declares inbox input 'box'; only `take` reads an inbox (a loop passes every " +
      "carried field to its tool)",
  );
});

test("take named in a group and given an inbox gets the call node's refusal of the group", async () => {
  const text = loopApp("{crm: take}", "    input:\n      from: ${input.box}\n    times: 1\n", "  box: !Inbox(str)\n");
  expect((await failure(text))[0]).toBe(
    "node 'poll' (kind=loop): `call: {crm: take}`: take is a built-in, not a group's tool; write `call: take`",
  );
});

test("take carrying the fields it returns fails its own input rule, saying a loop passes them all", async () => {
  const lines = "    input:\n      from: ${input.box}\n      events: !list(str) []\n      closed: false\n    until: closed\n";
  expect((await failure(loopApp("take", lines, "  box: !Inbox(str)\n")))[0]).toBe(
    "node 'poll' (call: take): take takes from, take and timeout; got 'events' (a loop passes " +
      "every carried field to its tool)",
  );
});

// Each row: an ask's lines and the loop's refusal of what it answers.
test.each([
  [
    "    output: !str\n",
    "loop node 'poll': tool `ask` returns str, not a record of carried fields (a loop's tool returns the carried " +
      "fields that change); declare the record it returns with `output:`",
  ],
  [
    "",
    "loop node 'poll': tool `ask` returns an untyped value, not a record of carried fields (a loop's tool returns the " +
      "carried fields that change); declare the record it returns with `output:`",
  ],
])("an ask answering %j is refused as no record of carried fields", async (lines, message) => {
  const text = loopApp("ask", `    input:\n      prompt: Name?\n    times: 1\n${lines}`);
  expect((await failure(text))[0]).toBe(message);
});

// Each row: the lines of a loop over `status` carrying `job` and `done`, and its type or end error,
// which names the tool.
test.each([
  [
    "    output: {done: !str}\n    while: not done\n    max: 3\n",
    "loop node 'poll': tool `status` returns 'done' as str, but the carried field is typed bool from its seed;",
  ],
  [
    "    output: {job: !str}\n    while: not done\n",
    "loop node 'poll': this loop can never end: `while: not done` reads `done`, which tool `status` never " +
      "returns; add `done` to its `output:`, or add `max:`",
  ],
])("a loop's tool with %j gets the loop's error in the tool's words", async (lines, message) => {
  const text = loopApp("status", `    input:\n      job: \${input.job}\n      done: false\n${lines}`);
  expect((await failure(text))[0]).toContain(message);
});

// Each row: a loop's `call:` and lines, and its refusal of what the tool returns, which lists the
// carried fields but the memory yian passes, and offers `output:` only when a field is carried.
test.each([
  [
    "{lessons: read}",
    "    input:\n      path: a.md\n    times: 1\n    output: {path: !str, x: !int}\n",
    "loop node 'poll': tool `{lessons: read}` returns {\"path\":str,\"x\":int}, but `x` isn't a carried field " +
      "[\"path\"] (a loop's tool returns the carried fields that change)",
  ],
  [
    "{lessons: read}",
    "    times: 1\n",
    "loop node 'poll': tool `{lessons: read}` returns an untyped value, not a record of carried fields (a loop's " +
      "tool returns the carried fields that change)",
  ],
  [
    "tick",
    "    times: 1\n",
    "loop node 'poll': tool `tick` returns an untyped value, not a record of carried fields (a loop's tool returns " +
      "the carried fields that change)",
  ],
])("a loop over %s with %j words its refusal by the carried fields", async (call, lines, message) => {
  expect((await failure(loopApp(call, lines)))[0]).toBe(message);
});

test("run start names a loop's tool by the loop", async () => {
  const text =
    "id: r\nname: r\ndefs:\n  c:\n    nodes:\n" +
    "      poll: {kind: loop, call: statsu, input: {job: a, done: false}, while: not done, max: 2}\n" +
    "    output: ${poll.output}\nnodes:\n  run: {kind: call, call: c}\n";
  const flow = await loadFlow(text);

  expect(runStartTools(flow)).toEqual([{ node: "run/poll", site: "loop node 'poll'", name: "statsu", hint: null }]);
  const run = runFlow(flow, {}, { env: hostEnv(tmp), tools: { status } });
  expect(((await run.catch((caught: unknown) => caught)) as Error).message).toBe(
    "loop node 'poll' in 'run' names tool 'statsu', which isn't in this run's tools; known tools: [\"status\"]; " +
      "did you mean 'status'?",
  );
});

// Each row: the field on a loop naming the flow `step`, and the refusal.
test.each([
  ["retries: 2", "`retries:` doesn't apply to a flow call: a flow retries inside its own nodes"],
  ["output: !int", "`output:` doesn't apply to a flow call: a flow's result type comes from the flow"],
  ["adaptive_questions: {prompt: hi}", "`adaptive_questions:` applies only to `call: ask`"],
])("a loop naming a flow refuses %s", async (field, problem) => {
  const text =
    'id: r\nname: r\ndefs:\n  step: {input: {n: !int}, kind: code, code: "return {n: 1}", output: {n: !int}}\n' +
    `nodes:\n  l: {kind: loop, call: step, input: {n: 0}, times: 1, ${field}}\n`;
  expect((await failure(text))[0]).toBe(`node 'l' (kind=loop): ${problem}`);
});

test("a loop over a flow keeps its fingerprint", async () => {
  const text =
    "id: r\nname: r\n" +
    'defs:\n  step: {input: {n: !int}, kind: code, code: "return {n: inputs.n - 1}", output: {n: !int}}\n' +
    "nodes:\n" +
    "  named: {kind: loop, call: step, input: {n: 3}, while: n > 0}\n" +
    "  inline:\n    kind: loop\n    input: {n: 3}\n    while: n > 0\n" +
    '    nodes:\n      d: {kind: code, code: "return inputs.n - 1", input: {n: "${input.n}"}, output: !int}\n' +
    "    output:\n      n: ${d.output}\n" +
    "output:\n  named: ${named.output}\n  inline: ${inline.output}\n";
  expect(flowFingerprint((await loadFlow(text)).compiled)).toBe("da0dfbeb0dede2b6a347b580d8ab827b27aa6e1859c914c13011cc99bc9c95bf");
});
