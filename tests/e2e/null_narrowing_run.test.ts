/**
 * Running a CASE that routes on a null check: a set value takes the branch, null the `else:`.
 *
 * The branch passes the value, narrowed to not null, to a def's strict input (a CALL) or as a MAP's
 * per-item argument; these runs show the narrowing matches what the value does at run.
 */

import { afterAll, expect, test } from "bun:test";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { runFlow } from "../../src/run.ts";
import { loadConformanceFlow } from "../helpers/declarations.ts";
import { hostEnv } from "../helpers/mcp_helpers.ts";

const dir = mkdtempSync(join(tmpdir(), "yian-null-narrowing-run-"));
afterAll(() => rmSync(dir, { recursive: true, force: true }));
const env = hostEnv(dir);

const ECHO = "./fixtures/compose_codefns.ts:echo";

const PRODUCER_FLOW = `
id: p
name: p
input:
  s: !Optional(str)
  items: !list(int)
nodes:
  x:
    kind: code
    code: ${ECHO}
    input:
      topic: \${input.s}
    output: !Optional(str)
  gate:
    kind: case
    cases:
      - when: x.output != null
        then: use
    else: skip
  USE
  skip:
    kind: code
    code: ${ECHO}
    input:
      topic: skipped
    output: !str
output: \${use.output :- skip.output}
defs:
  take:
    input:
      v: !str
    nodes:
      d:
        kind: code
        code: ${ECHO}
        input:
          topic: \${input.v}
        output: !str
    output: \${d.output}
  take_each:
    input:
      i: !int
      v: !str
    nodes:
      d:
        kind: code
        code: ${ECHO}
        input:
          topic: \${input.v}
        output: !str
    output: \${d.output}
`;

const CALL = `use:
    kind: call
    call: take
    input:
      v: \${x.output}`;

const MAP = `use:
    kind: map
    call: take_each
    over: \${input.items}
    input:
      i: \${item}
      v: \${x.output}`;

const INPUT_FLOW = `
id: p
name: p
typedefs:
  Item:
    qty: int
input:
  item: !Optional(Item)
defs:
  take:
    input:
      item: !Item
    nodes:
      d:
        kind: code
        code: ${ECHO}
        input:
          topic: \${input.item.qty}
        output: !int
    output: \${d.output}
nodes:
  gate:
    kind: case
    cases:
      - when: input.item != null
        then: use
    else: skip
  use:
    kind: call
    call: take
    input:
      item: \${input.item}
  skip:
    kind: code
    code: ${ECHO}
    input:
      topic: 0
    output: !int
output: \${use.output :- skip.output}
`;

/** Load `text` and run it on `inputs`; return the output of the run, which must succeed. */
async function output(text: string, inputs: Record<string, unknown>): Promise<unknown> {
  const result = await runFlow(await loadConformanceFlow(text, { env }), inputs, { env });
  expect(result.failure).toBeNull();
  expect(result.status).toBe("succeeded");
  return result.output;
}

test.each([
  ["call-set", CALL, "hi", "hi"],
  ["call-null", CALL, null, "skipped"],
  ["map-set", MAP, "hi", ["hi", "hi"]],
  ["map-null", MAP, null, "skipped"],
] as const)("a null checked producer output runs down the branch it takes (%s)", async (_, use, s, expected) => {
  expect(await output(PRODUCER_FLOW.replace("USE", use), { s, items: [1, 2] })).toEqual(expected);
});

test.each([
  [{ qty: 3 }, 3n],
  [null, 0n],
] as const)("a null checked input runs down the branch it takes (%o)", async (item, expected) => {
  expect(await output(INPUT_FLOW, { item })).toBe(expected);
});
