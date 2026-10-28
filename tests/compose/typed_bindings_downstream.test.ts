/**
 * Downstream checks see a typed binding's declared type.
 *
 * A binding tagged with a type on a CALL, MAP or LOOP node declares the type its child receives, so
 * the cross-flow check compares that declared type with the child's parameters instead of the type
 * inferred from the value.
 */

import { afterAll, afterEach, expect, test } from "bun:test";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import type { ComposedFlow } from "../../src/compose/build/pipeline.ts";
import { forget } from "../../src/defaults.ts";
import type { CallNode } from "../../src/nodes/call/node.ts";
import { type Parameter, ValueContract } from "../../src/nodes/contracts.ts";
import type { LoopNode } from "../../src/nodes/loop/node.ts";
import type { MapNode } from "../../src/nodes/map/node.ts";
import { runFlow } from "../../src/run.ts";
import { typeFor } from "../../src/typesys/types.ts";
import { hostEnv } from "../helpers/mcp_helpers.ts";
import { assembled, assembleText, loadConformanceFlow, loadError, runCase } from "../helpers/declarations.ts";

const dir = mkdtempSync(join(tmpdir(), "yian-typed-bindings-downstream-"));
afterAll(() => rmSync(dir, { recursive: true, force: true }));
const env = hostEnv(dir);

afterEach(() => forget());

const STRICT_CHILD = `defs:
  child:
    input:
      x: !int
    nodes:
      twice:
        kind: code
        input:
          x: \${input.x}
        output: int
        code: |
          return inputs["x"] * 2
    output: \${twice.output}
`;

/** A flow that calls the strict `child` with `x` bound as `binding`, after `producer`'s nodes. */
function callFlow(binding: string, producer: string): string {
  return `id: downstream
name: downstream
${STRICT_CHILD}nodes:
${producer}  run:
    kind: call
    call: child
    input:
      x: ${binding}
output: \${run.output}
`;
}

/** A `maybe` CODE producer declaring `Optional(int)` that returns `value`. */
const maybe = (value: string) =>
  `  maybe:\n    kind: code\n    output: Optional(int)\n    code: |\n      return ${value}\n`;

/** An `i` CODE producer declaring `int`. */
const INT_PRODUCER = "  i:\n    kind: code\n    output: int\n    code: |\n      return 3\n";

/** An `opaque` CODE producer with no declared output that returns `value`. */
const opaque = (value: string) => `  opaque:\n    kind: code\n    code: |\n      return ${value}\n`;

/** Assemble `text`. */
const load = (text: string) => assembleText(text);

/** Load `text` and run it with no inputs. */
const run = async (text: string) => runFlow(await loadConformanceFlow(text), {}, { env });

// ---------- CALL ----------

test("an untyped nullable source into a strict child fails to load", () => {
  const error = loadError(() => load(callFlow("${maybe.output}", maybe("3"))));

  expect(error.message).toBe("call node 'run': binding 'x' — child expects 'int', source is 'Optional(int)'");
});

test.each([
  ["narrows a nullable source into a strict child", "3"],
  ["rejects a null at runtime", "null"],
])("a typed binding %s (loads)", (_name, value) => {
  load(callFlow("!int ${maybe.output}", maybe(value)));
});

test("a typed binding narrows a nullable source into a strict child (runs)", async () => {
  const result = await run(callFlow("!int ${maybe.output}", maybe("3")));

  expect(result.failure).toBeNull();
  expect(result.status).toBe("succeeded");
  expect(result.output).toBe(6n);
});

test("a typed binding rejects a null at runtime (runs)", async () => {
  const result = await run(callFlow("!int ${maybe.output}", maybe("null")));

  expect(result.status).toBe("failed");
  expect(result.failure!.code).toBe("binding.type");
  expect(result.failure!.nodeId).toBe("run");
});

test("a typed nullable binding into a strict child fails to load", () => {
  const error = loadError(() => load(callFlow("!Optional(int) ${maybe.output}", maybe("3"))));

  expect(error.message).toBe("call node 'run': binding 'x' — child expects 'int', source is 'Optional(int)'");
});

test("a declared float does not fit a strict int child", () => {
  const error = loadError(() => load(callFlow("!float ${i.output}", INT_PRODUCER)));

  expect(error.message).toBe("call node 'run': binding 'x' — child expects 'int', source is 'float'");
});

test("a map binding uses the declared type", () => {
  const text = `id: downstream-map
name: downstream_map
${STRICT_CHILD}nodes:
${INT_PRODUCER}  each:
    kind: map
    over: [1, 2]
    call: child
    input:
      x: !float \${i.output}
output: \${each.output}
`;

  const error = loadError(() => load(text));

  expect(error.message).toBe("map node 'each': binding 'x' — child expects 'int', source is 'float'");
});

// ---------- LOOP ----------

/** A flow whose LOOP seeds `n` as `seed` into a named body `bump` that returns `n: int`. */
function loopFlow(seed: string, producer: string, bodyInput = "!int"): string {
  return `id: downstream-loop
name: downstream_loop
defs:
  bump:
    input:
      n: ${bodyInput}
    nodes:
      step:
        kind: code
        input:
          n: \${input.n}
        output: int
        code: |
          return inputs["n"] + 1
    output:
      n: \${step.output}
nodes:
${producer}  loop:
    kind: loop
    times: 2
    input:
      n: ${seed}
    call: bump
output: \${loop.output}
`;
}

test("a declared seed type is checked against the body input", () => {
  const error = loadError(() => load(loopFlow("!str ${opaque.output}", opaque("1"))));

  expect(error.message).toBe("loop node 'loop': binding 'n' — child expects 'int', source is 'str'");
});

test("a declared seed type is compared with the returned field", () => {
  // The producer is untyped, so only the declaration gives the seed a type: the same seed written
  // without one loads. The body takes `n: Any`, so both the seed and the returned `int` fit its input.
  load(loopFlow("${opaque.output}", opaque("1"), "!Any"));

  const error = loadError(() => load(loopFlow("!str ${opaque.output}", opaque("1"), "!Any")));

  expect(error.message).toBe(
    "loop node 'loop': carried field 'n' — body output is 'int', carried source is 'str' " +
      "(a returned field must keep its carried type)",
  );
});

test("a declared seed type types an opaque producer (loads)", () => {
  load(loopFlow("!int ${opaque.output}", opaque("1")));
});

test("a declared seed type types an opaque producer (runs)", async () => {
  const result = await run(loopFlow("!int ${opaque.output}", opaque("1")));

  expect(result.failure).toBeNull();
  expect(result.status).toBe("succeeded");
  expect(result.output).toEqual({ n: 3n });
});

test.each([
  ["untyped", "${opaque.output}"],
  ["declared int", "!int ${opaque.output}"],
])("a declared seed type is checked on the loop node (loads): %s", (_name, seed) => {
  load(loopFlow(seed, opaque('"1"')));
});

test("a declared seed type is checked on the loop node (runs)", async () => {
  // Untyped, the text reaches the body's own `int` parameter, which does not convert it across
  // the LOOP edge, so the body's start fails.
  const untyped = await run(loopFlow("${opaque.output}", opaque('"1"')));
  expect(untyped.status).toBe("failed");
  expect(untyped.failure!.code).toBe("contract.input");
  expect(untyped.failure!.nodeId).toBe("loop#0/__start__");

  const result = await run(loopFlow("!int ${opaque.output}", opaque('"1"')));

  expect(result.status).toBe("failed");
  expect(result.failure!.code).toBe("binding.type");
  expect(result.failure!.nodeId).toBe("loop");
});

// ---------- inline bodies ----------

/** The parameters of the inline body of CALL, MAP or LOOP `nodeId` in `flow`, keyed by name. */
function bodyInputs(flow: Pick<ComposedFlow, "compiled">, nodeId: string): Map<string, Parameter> {
  const node = flow.compiled.nodes.get(nodeId) as CallNode | MapNode | LoopNode;
  return new Map(node.childInputs.map((param) => [param.name, param]));
}

test("typed inline binding skips inference of an opaque expression", async () => {
  // A declared type types the inline parameter even when its value cannot be inferred.
  const loaded = await loadConformanceFlow(`id: inline-typed
name: inline_typed
nodes:
  a:
    kind: code
    code: |
      return 4
  wrap:
    kind: call
    input:
      n: !int \${a.output + 1}
    nodes:
      twice:
        kind: code
        input:
          n: \${input.n}
        output: int
        code: |
          return inputs["n"] * 2
    output: \${twice.output}
output: \${wrap.output}
`);

  const param = bodyInputs(loaded, "wrap").get("n")!;
  expect(param.contract).toStrictEqual(new ValueContract(typeFor("int", new Map())));
  expect(param.required).toBe(true);

  const result = await runFlow(loaded, {}, { env });
  expect(result.failure).toBeNull();
  expect(result.status).toBe("succeeded");
  expect(result.output).toBe(10n);
});

test("typed inline binding types an empty list", async () => {
  // An empty list literal takes its element type from the declaration.
  const loaded = await loadConformanceFlow(`id: inline-empty
name: inline_empty
nodes:
  wrap:
    kind: call
    input:
      items: !list(str) []
    nodes:
      count:
        kind: code
        input:
          items: \${input.items}
        output: int
        code: |
          return inputs["items"].length
    output: \${count.output}
output: \${wrap.output}
`);

  expect(bodyInputs(loaded, "wrap").get("items")!.contract!.type).toStrictEqual(typeFor("list(str)", new Map()));

  const result = await runFlow(loaded, {}, { env });
  expect(result.failure).toBeNull();
  expect(result.status).toBe("succeeded");
  expect(result.output).toBe(0n);
});

const FLOAT_LOOP = `id: inline-float-loop
name: inline_float_loop
nodes:
  loop:
    kind: loop
    times: 2
    input:
      r: !float 0
    nodes:
      half:
        kind: code
        input:
          r: \${input.r}
        output: float
        code: |
          return inputs["r"] + 0.5
    output:
      r: \${half.output}
output: \${loop.output}
`;

test("a typed inline loop seed types the body parameter (loads)", () => {
  // A `float` seed lets an inline body carry a float past its int literal seed.
  const { flow } = assembleText(FLOAT_LOOP);

  expect(bodyInputs(flow, "loop").get("r")!.contract!.type).toStrictEqual(typeFor("float", new Map()));
});

test("a typed inline loop seed types the body parameter (runs)", async () => {
  const result = await run(FLOAT_LOOP);

  expect(result.failure).toBeNull();
  expect(result.status).toBe("succeeded");
  expect(result.output).toEqual({ r: 1n });
});

const OPTIONAL_SEED = `id: inline-optional-seed
name: inline_optional_seed
nodes:
  p:
    kind: code
    code: |
      return "seed"
  loop:
    kind: loop
    times: 2
    input:
      n: 0
      label: !Optional(str) \${p.output}
    nodes:
      step:
        kind: code
        input:
          n: \${input.n}
        output: int
        code: |
          return inputs["n"] + 1
    output:
      n: \${step.output}
output: \${loop.output}
`;

test("a typed inline loop seed types an opaque producer (loads)", () => {
  // An `Optional(str)` seed over an untyped producer loads.
  const { flow } = assembleText(OPTIONAL_SEED);

  const param = bodyInputs(flow, "loop").get("label")!;
  expect(param.contract!.type).toStrictEqual(typeFor("Optional(str)", new Map()));
  expect(param.required).toBe(false);
});

test("a typed inline loop seed types an opaque producer (runs)", async () => {
  const result = await run(OPTIONAL_SEED);

  expect(result.failure).toBeNull();
  expect(result.status).toBe("succeeded");
  expect(result.output).toEqual({ n: 2n, label: "seed" });
});

test("typed inline map item binding runs", async () => {
  // A typed `${item}` binding types an inline MAP body over an untyped producer.
  const loaded = await loadConformanceFlow(`id: inline-map-item
name: inline_map_item
nodes:
${opaque('["a", "b"]')}  each:
    kind: map
    over: \${opaque.output}
    input:
      t: !str \${item}
    nodes:
      shout:
        kind: code
        input:
          t: \${input.t}
        output: str
        code: |
          return inputs["t"].toUpperCase()
    output: \${shout.output}
output: \${each.output}
`);

  expect(bodyInputs(loaded, "each").get("t")!.contract!.type).toStrictEqual(typeFor("str", new Map()));

  const result = await runFlow(loaded, {}, { env });
  expect(result.failure).toBeNull();
  expect(result.status).toBe("succeeded");
  expect(result.output).toEqual(["A", "B"]);
});

// ---------- harvested conformance cases ----------

test.each([
  {
    id: "load-types/compose_typed_bindings_downstream__seed_type_is_compared_with_the_returned_field",
    call: assembled,
  },
])("assemble: $id", runCase);
