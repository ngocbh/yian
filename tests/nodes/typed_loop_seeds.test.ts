/**
 * Typed LOOP seeds, `name: !Type value` on a loop's `input:` typing its carried record.
 *
 * A loop's `input:` is the carried record's turn-0 seed, so a typed seed is checked when the
 * driver binds it, and a field the body omits carries the declared type into the loop's merged
 * output contract.
 */

import { expect, test } from "bun:test";

import { Binding, ValueContract } from "../../src/nodes/contracts.ts";
import type { LoopNode } from "../../src/nodes/loop/node.ts";
import { runFlow } from "../../src/run.ts";
import { typeFor } from "../../src/typesys/types.ts";
import { Type, typeEquals, ValueKind } from "../../src/typesys/values.ts";
import { loadConformanceFlow } from "../helpers/declarations.ts";

const TYPED_COUNTER = `
id: typed-counter
name: typed_counter
nodes:
  loop:
    kind: loop
    times: 3
    input:
      rounds: !int 0
    nodes:
      bump:
        kind: code
        input:
          n: \${input.rounds}
        output: int
        code: |
          return inputs["n"] + 1
    output:
      rounds: \${bump.output}
output: \${loop.output}
`;

test("typed literal seed binds its value and runs", async () => {
  // A typed literal seed carries its declared contract and seeds the first turn.
  const loaded = await loadConformanceFlow(TYPED_COUNTER);
  const binding = loaded.compiled.wiring.get("loop")!.get("rounds")!;
  expect(binding.constructor).toBe(Binding);
  expect(binding.source).toBe(0n);
  expect(typeEquals(binding.contract!.type, typeFor("int", new Map()))).toBe(true);

  const result = await runFlow(loaded, {});
  expect(result.status).toBe("succeeded");
  expect(result.output).toEqual({ rounds: 3n });
});

// `p` is an untyped CODE node, so the `label` seed's inferred contract is opaque; only its
// declared type can prove what the omitted field carries. The body is a named `defs:` flow
// so its own `n: int` parameter, not the seed declarations, types what it receives.
const PARTIAL_TYPED_SEED = `
id: partial-typed-seed
name: partial_typed_seed
defs:
  bump:
    input:
      n: !int
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
    call: bump
output: \${loop.output}
`;

test("omitted typed seed carries its declared type", async () => {
  // A field the body omits keeps its declared seed type in the loop's output contract.
  const loaded = await loadConformanceFlow(PARTIAL_TYPED_SEED);
  const loop = loaded.compiled.nodes.get("loop") as LoopNode;
  expect(loop.returnedFields).toEqual(["n"]);
  const contract = loop.outputContract as ValueContract;
  const registry = new Map();
  expect(contract.constructor).toBe(ValueContract);
  const expected = new Type({
    kind: ValueKind.OBJECT,
    fields: new Map([
      ["label", typeFor("Optional(str)", registry)],
      ["n", typeFor("int", registry)],
    ]),
    required: new Set(["label", "n"]),
  });
  expect(typeEquals(contract.type, expected)).toBe(true);

  const result = await runFlow(loaded, {});
  expect(result.status).toBe("succeeded");
  expect(result.output).toEqual({ n: 2n, label: "seed" });
});

const MISFIT_SEED = `
id: misfit-seed
name: misfit_seed
defs:
  bump:
    input:
      n: !int
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
  p:
    kind: code
    code: |
      return "x"
  loop:
    kind: loop
    times: 2
    input:
      n: !int \${p.output}
    call: bump
output: \${loop.output}
`;

test("typed seed rejects a value that does not fit", async () => {
  // A seed value outside its declared type fails the loop on that binding.
  const result = await runFlow(await loadConformanceFlow(MISFIT_SEED), {});
  expect(result.status).toBe("failed");
  expect(result.failure!.nodeId).toBe("loop");
  expect(result.failure!.code).toBe("binding.type");
  expect(result.failure!.message).toContain("binding 'n'");
});
