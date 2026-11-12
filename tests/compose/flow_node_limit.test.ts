/**
 * The composed graph is bounded at load (`YIAN_MAX_FLOW_NODES`): each call of a def or `uses:` flow
 * counts as a copy of its body, a MAP or LOOP body counts once, and a flow past the limit is
 * refused at the alias or call that crosses it.
 */

import { expect, test } from "bun:test";

import { ComposedFlow } from "../../src/compose/build/pipeline.ts";
import { loadFlow } from "../../src/compose/loader.ts";
import { BUILTIN, type Env } from "../../src/defaults.ts";
import { runFlow } from "../../src/run.ts";
import { loadErrorAsync } from "../helpers/declarations.ts";
import { NO_ENV } from "../helpers/run_state_fixtures.ts";

/** `NO_ENV` with `YIAN_MAX_FLOW_NODES` set to `limit`. */
const limited = (limit: number): Env => (name) => (name === "YIAN_MAX_FLOW_NODES" ? String(limit) : NO_ENV(name));

/** The 1-based line of the first line of `text` holding `needle`. */
const lineOf = (text: string, needle: string) => text.split("\n").findIndex((line) => line.includes(needle)) + 1;

/** A chain of `depth` files: `l<i>` imports `l<i+1>` as `a` and `b` and calls both; the last is a leaf. */
function diamond(depth: number): Map<string, string> {
  const files = new Map<string, string>();
  for (let i = 0; i < depth; i++) {
    files.set(
      `l${i}`,
      `id: l${i}\nname: l${i}\nuses:\n  a: l${i + 1}\n  b: l${i + 1}\n` +
        "nodes:\n  x:\n    kind: call\n    call: a\n  y:\n    kind: call\n    call: b\noutput: ${x.output}\n",
    );
  }
  files.set(`l${depth}`, `id: l${depth}\nname: l${depth}\nnodes:\n  v: !str leaf\noutput: \${v.output}\n`);
  return files;
}

/** Load `l0` of `files`, each file loaded once through an in-memory resolver. */
function loadDiamond(files: Map<string, string>, env: Env): Promise<ComposedFlow> {
  const loads = new Map<string, Promise<ComposedFlow>>();
  const resolver = (path: string): Promise<ComposedFlow> => {
    let load = loads.get(path);
    if (load === undefined) {
      load = loadFlow(files.get(path)!, { childResolver: resolver, sourceLabel: `${path}.yaml`, env });
      loads.set(path, load);
    }
    return load;
  };
  return loadFlow(files.get("l0")!, { childResolver: resolver, sourceLabel: "l0.yaml", env });
}

/** One file whose defs `d0` to `d<depth - 1>` each call the next twice; `d<depth>` is a value leaf. */
function ladder(depth: number): string {
  let text = "id: lad\nname: lad\ninput:\n  x: !str\ndefs:\n";
  for (let i = 0; i < depth; i++) {
    text +=
      `  d${i}:\n    input:\n      x: !str\n    nodes:\n` +
      `      a:\n        kind: call\n        call: d${i + 1}\n        input:\n          x: \${input.x}\n` +
      `      b:\n        kind: call\n        call: d${i + 1}\n        input:\n          x: \${a.output}\n` +
      "    output: ${b.output}\n";
  }
  text += `  d${depth}:\n    input:\n      x: !str\n    nodes:\n      v: !str \${input.x}\n    output: \${v.output}\n`;
  return text + "nodes:\n  top:\n    kind: call\n    call: d0\n    input:\n      x: ${input.x}\noutput: ${top.output}\n";
}

const MESSAGE = (limit: number) =>
  `this flow expands to more than ${limit} nodes, counting each call of a def or uses: flow as a ` +
  `copy; the limit for a flow is ${limit} (YIAN_MAX_FLOW_NODES)`;

test("the limit defaults to 100,000", () => {
  expect(BUILTIN.maxFlowNodes).toBe(100_000);
});

test("a deep uses: diamond is refused at the alias of the first file past the limit", async () => {
  const files = diamond(20);
  const error = await loadErrorAsync(loadDiamond(files, limited(2000)));

  expect(error.message).toContain(MESSAGE(2000));
  const frames = error.diagnostic.frames;
  expect(frames[0]!.label).toBe("l0.yaml");
  expect(frames[0]!.range.startLine).toBe(lineOf(files.get("l0")!, "a: l1"));
  // The innermost frame is the call that crossed the limit, in the file whose copies passed it.
  const inner = frames.at(-1)!;
  expect(inner.range.startLine).toBe(lineOf(files.get("l0")!, "y:"));
  expect(inner.label).not.toBe("l0.yaml");
});

test("a deep in-file defs ladder is refused at the call that crosses the default limit", async () => {
  const text = ladder(20);
  const error = await loadErrorAsync(loadFlow(text, { sourceLabel: "lad.yaml", env: NO_ENV }));

  expect(error.message).toContain(MESSAGE(100_000));
  const frame = error.diagnostic.frames.at(-1)!;
  expect(frame.label).toBe("lad.yaml");
  // With its START and END, d<i> holds 4 + 2 * (d<i+1>'s count) nodes and d20 holds 3: d6 holds
  // 114,684, over the limit, and d7 57,340, so d6's second call `b` crosses it.
  expect(frame.range.startLine).toBe(lineOf(text, "call: d7") + 3);
});

test("a def called twice counts twice, and a MAP body once", async () => {
  // `pair`'s body is its START, `v` and END; each CALL adds one node and a copy of the body.
  const body = "    nodes:\n      v: !str x\n    output: ${v.output}\n";
  const twice = `id: t\nname: t\ndefs:\n  pair:\n${body}nodes:\n  p:\n    kind: call\n    call: pair\n  q:\n    kind: call\n    call: pair\noutput: \${q.output}\n`;
  const mapped =
    `id: m\nname: m\ndefs:\n  pair:\n    input:\n      i: !int\n${body}` +
    "nodes:\n  m:\n    kind: map\n    over: [1, 2, 3, 4, 5, 6, 7, 8]\n    call: pair\n    input:\n      i: ${item}\noutput: ${m.output}\n";
  // START, p, q, END and two copies of the 3-node body: 10 nodes.
  await loadFlow(twice, { env: limited(10) });
  expect((await loadErrorAsync(loadFlow(twice, { env: limited(9) }))).message).toContain(MESSAGE(9));
  // START, m, END and one copy of the body, however many items: 6 nodes.
  await loadFlow(mapped, { env: limited(6) });
  expect((await loadErrorAsync(loadFlow(mapped, { env: limited(5) }))).message).toContain(MESSAGE(5));
});

test("a 10-level defs ladder under the limit loads and runs", async () => {
  const flow = await loadFlow(ladder(10), { env: NO_ENV });
  const result = await runFlow(flow, { x: "a" }, { env: NO_ENV });

  expect([result.status, result.output]).toEqual(["succeeded", "a"]);
}, 60_000);
