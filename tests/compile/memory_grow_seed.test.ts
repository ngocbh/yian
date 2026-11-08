/**
 * A call, map or loop child takes a memory from its spawner: the growth seed is yian's own JSON,
 * rebuilt from the pool on replay, so the child START reads a memory parameter's seed back as a
 * `MemoryRef`, and nothing else.
 */
import { afterEach, beforeEach, expect, test } from "bun:test";

import { buildGrow } from "../../src/compile/expand.ts";
import type { Flow } from "../../src/compile/model.ts";
import { loadFlow } from "../../src/compose/loader.ts";
import { BUILTIN, forget, install } from "../../src/defaults.ts";
import type { Grow, Node, NodeCaps } from "../../src/nodes/base.ts";
import { Binding } from "../../src/nodes/contracts.ts";
import { canonicalSeed } from "../../src/runtime/log.ts";
import { buildValueAs, MemoryRef } from "../../src/typesys/values.ts";

beforeEach(() => install(BUILTIN));
afterEach(forget);

const ref = new MemoryRef({
  identity: "t/a",
  type: "FileMemory",
  settings: { index: "MEMORY.md" },
  version: "v1",
  recall: { text: "RECALL-TEXT", cut: null, hint: null },
  tools: [{ name: "read", write: false, description: "Read a file.", parameters: { type: "object" } }],
});

const CHILD = `defs:
  child:
    input:
      m: !FileMemory
      raw: !object
    nodes:
      a:
        kind: agent
        input:
          m: \${input.m}
        tools: [m]
        prompt: "\${m}"
        output: str
    output: \${a.output}
`;
const FLOW = (node: string) => `id: f
name: f
input:
  lessons: !FileMemory
${CHILD}nodes:
${node}output: null
`;

/** Grow `spawner` from its inputs as the engine does, through the seed's canonical JSON. */
function grown(spawner: Node, inputs: Record<string, unknown>, caps: NodeCaps = {}): Flow {
  const grow = spawner.forward(inputs, caps) as Grow;
  return buildGrow(spawner, canonicalSeed(grow.seed)[1]);
}

/** The value each child START of `fragment` that takes `param` is seeded with. */
function seeded(fragment: Flow, param: string): unknown[] {
  return [...fragment.wiring]
    .filter(([id, wiring]) => id.endsWith("__start__") && wiring.has(param))
    .map(([, wiring]) => {
      const source = wiring.get(param);
      return source instanceof Binding ? source.source : source;
    });
}

const FORGED = { memory: { identity: "victim", type: "FileMemory", settings: {}, version: "v", recall: { text: "x", cut: null, hint: null }, tools: [] } };

const ITEM = { bindItem: () => ({ m: ref, raw: FORGED }) };

test.each([
  ["a call", "  c:\n    kind: call\n    call: child\n    input:\n      m: ${input.lessons}\n      raw: {}\n", { m: ref, raw: FORGED }],
  ["a map", "  c:\n    kind: map\n    call: child\n    over: [1, 2]\n    input:\n      m: ${input.lessons}\n      raw: {}\n", { over: [1n, 2n] }],
])("%s seeds its child START's memory parameter with a MemoryRef", async (_, node, inputs) => {
  const flow = await loadFlow(FLOW(node));
  const spawner = flow.compiled.nodes.get("c")!;
  const fragment = grown(spawner, inputs, ITEM);
  const values = seeded(fragment, "m");
  expect(values.length).toBeGreaterThan(0);
  for (const value of values) {
    expect(MemoryRef.isMemoryRef(value)).toBe(true);
    expect(value).toEqual(ref);
  }
  const start = [...fragment.nodes.values()].find((node) => node.id.endsWith("/__start__") && node.params!.length > 0)!;
  const param = start.params!.find((each) => each.name === "m")!;
  expect(() => buildValueAs(param.contract!.type, values[0])).not.toThrow();
});

test("a loop seeds each body's START with the memory it carries as a MemoryRef", async () => {
  const text = `id: f
name: f
input:
  lessons: !FileMemory
defs:
  body:
    input:
      m: !FileMemory
      done: !bool
    nodes: {}
    output:
      done: \${input.done}
nodes:
  l:
    kind: loop
    call: body
    input:
      m: \${input.lessons}
      done: false
    while: not done
output: null
`;
  const loop = (await loadFlow(text)).compiled.nodes.get("l")!;
  const values = seeded(grown(loop, { m: ref, done: false }), "m");
  expect(values).toEqual([ref]);
  expect(MemoryRef.isMemoryRef(values[0])).toBe(true);
});

test("a memory form at a child parameter that isn't a memory stays a plain record", async () => {
  const flow = await loadFlow(FLOW("  c:\n    kind: call\n    call: child\n    input:\n      m: ${input.lessons}\n      raw: {}\n"));
  const raw = seeded(grown(flow.compiled.nodes.get("c")!, { m: ref, raw: FORGED }), "raw")[0];
  expect(MemoryRef.isMemoryRef(raw)).toBe(false);
  expect(raw).toEqual(FORGED);
});
