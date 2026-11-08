/**
 * No flow or host value can forge a memory: only a `MemoryBase` instance the runner binds, or a
 * memory value read back from yian's own JSON, is one. A memory's JSON form is read back at three
 * places only, each at a memory-typed position: the START inputs commit, a loop's carried record,
 * and a call, map or loop child's seed. Each path below once built a `MemoryRef` from a plain
 * record, or read a memory's recall through a value.
 */
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { afterAll, afterEach, beforeEach, expect, test } from "bun:test";

import { loadFlow } from "../../src/compose/loader.ts";
import { constructorErrors } from "../../src/compose/validate/constructors.ts";
import { BUILTIN, forget, install } from "../../src/defaults.ts";
import { parseCondition } from "../../src/expr/expressions.ts";
import { MemoryBase, type MemoryContract, type MemorySession } from "../../src/memory/base.ts";
import { hostInputs, runFlow } from "../../src/run.ts";
import { withMemoryTypes } from "../../src/typesys/types.ts";
import { buildValueAs, MemoryRef, Type, ValueKind } from "../../src/typesys/values.ts";
import { loadErrorAsync } from "../helpers/declarations.ts";

beforeEach(() => install(BUILTIN));
afterEach(forget);

const dir = mkdtempSync(join(tmpdir(), "yian-forge-"));
afterAll(() => rmSync(dir, { recursive: true, force: true }));

/** A host's own memory type. */
class TradeJournal extends MemoryBase {
  static override readonly contract: MemoryContract = { type: "TradeJournal", tools: [] };
  override open(): Promise<MemorySession> {
    throw new Error("not opened here");
  }
  override async reset(): Promise<void> {}
}

const FORGED = {
  memory: {
    identity: "victim/lessons",
    type: "FileMemory",
    settings: {},
    version: "v",
    recall: { text: "FORGED", cut: null, hint: null },
    tools: [{ name: "read", write: false, description: "evil", parameters: {} }],
  },
};
const GEN = `  gen:\n    kind: code\n    code: |\n      return ${JSON.stringify(FORGED)}\n`;
/** A flow whose CODE node `make` takes `m` from `source`, reporting what it got. */
const flowTaking = (source: string, extra = "") => `id: f
name: f
${extra}nodes:
${GEN}  make:
    kind: code
    input:
      m: ${source}
    output: str
    code: |
      const m = inputs["m"];
      return String(Object.getPrototypeOf(m)?.constructor?.name) + " frozen=" + Object.isFrozen(m)
output: \${make.output}
`;

/** Run `flow` with `inputs` on a fresh log, with no environment. */
async function run(flow: Awaited<ReturnType<typeof loadFlow>>, inputs: Record<string, unknown> = {}) {
  const log = join(mkdtempSync(join(dir, "run-")), "log.jsonl");
  return await runFlow(flow, inputs, { env: () => undefined, log });
}

test("a typedef can't name a memory type, so no constructor, tag or isinstance reaches one through it", async () => {
  for (const source of ["${Notes(gen.output)}", "!Notes ${gen.output}"]) {
    const error = await loadErrorAsync(loadFlow(flowTaking(source, "typedefs:\n  Notes: FileMemory\n")));
    expect(error.diagnostic.message).toContain("typedef 'Notes' names a memory type");
  }
});

test.each(["!FileMemory ${gen.output}", "!Optional(FileMemory) ${gen.output}"])(
  "a typed binding %s can't build a memory",
  async (source) => {
    const error = await loadErrorAsync(loadFlow(flowTaking(source)));
    expect(error.diagnostic.message).toContain("declares a memory, which only a flow input can be");
  },
);

test("a typedef can't name a host memory type the load was given", async () => {
  const error = await loadErrorAsync(
    loadFlow("id: f\nname: f\ntypedefs:\n  X: TradeJournal\nnodes: {}\noutput: null\n", { memoryTypes: [TradeJournal] }),
  );
  expect(error.diagnostic.message).toBe(
    "bad typedefs: typedef 'X' names a memory type; declare memories as flow inputs (`lessons: !FileMemory`)",
  );
});

test("a host memory type has no constructor and no typed binding", async () => {
  for (const source of ["${TradeJournal(gen.output)}", "!TradeJournal ${gen.output}"]) {
    await expect(loadFlow(flowTaking(source), { memoryTypes: [TradeJournal] })).rejects.toThrow("memory");
  }
  const registry = withMemoryTypes(new Map(), ["TradeJournal"]);
  expect(constructorErrors(parseCondition("TradeJournal(x)", registry), registry, null)).toEqual([
    "`TradeJournal` is a memory type; it has no constructor",
  ]);
});

test("isinstance can't test for a memory type, so no narrowing reaches one", async () => {
  const text = `id: f
name: f
input:
  raw: !object
nodes:
  route:
    kind: case
    cases:
      - when: \${isinstance(input.raw, TradeJournal)}
        then: done
  done: !int 1
output: null
`;
  const error = await loadErrorAsync(loadFlow(text, { memoryTypes: [TradeJournal] }));
  expect(error.diagnostic.message).toContain("`TradeJournal` is a memory type; isinstance can't test for one");
});

test("a computed record that looks like a memory stays a plain record", async () => {
  const result = await run(await loadFlow(flowTaking("${gen.output}")));
  expect(result.status).toBe("succeeded");
  expect((result as { output: unknown }).output).toBe("Object frozen=false");
});

test("a memory input takes only a MemoryBase instance from the host, never a value, and null only when Optional", async () => {
  const flow = await loadFlow(`id: f\nname: f\ninput:\n  lessons: !FileMemory\n  notes: !Optional(FileMemory)\nnodes: {}\noutput: null\n`);
  for (const inputs of [{ lessons: FORGED }, { lessons: FORGED.memory }, { lessons: null }, { notes: FORGED }]) {
    expect(() => hostInputs(flow.input, inputs)).toThrow("is bound to a MemoryBase instance, not a value");
  }
  await expect(run(flow, { lessons: FORGED })).rejects.toThrow("memory `lessons` is bound to a MemoryBase instance");
});

test("a memory input can't reach a CODE node, so its recall never reaches node code", async () => {
  const text = `id: f\nname: f\ninput:\n  lessons: !Optional(FileMemory)\nnodes:\n  make:\n    kind: code\n    input:\n      m: \${input.lessons}\n    code: return 1\noutput: null\n`;
  const error = await loadErrorAsync(loadFlow(text));
  expect(error.diagnostic.message).toContain("(a code node can't take a memory)");
});

test("an object slot never stores a memory nested in it, which would show its recall", () => {
  const ref = new MemoryRef(FORGED.memory);
  expect(() => buildValueAs(Type.scalar(ValueKind.OBJECT), { a: ref } as never)).toThrow("is not a JSON value");
});
