/**
 * A MAP item field named `output` reads like any other field (`${item.output}`): a port of
 * `tests/compose/test_item_output_field.py`.
 *
 * Ported: `test_map_reads_an_item_field_named_output` (both rows) and
 * `test_map_inside_a_called_def_reads_an_item_field_named_output`. The CODE body is JS
 * (`toUpperCase()`, decision 11).
 *
 * Covered by `run-map-call/compose_item_output_field__inline_map_body_reads_an_item_field_named_output`:
 * `test_inline_map_body_reads_an_item_field_named_output`.
 */

import { expect, test } from "bun:test";

import { loadFlow } from "../../src/compose/loader.ts";
import { runFlow } from "../../src/run.ts";

const LEAF = `  leaf:
    input:
      x: !str
    nodes:
      up:
        kind: code
        input:
          x: \${input.x}
        output: str
        code: return inputs.x.toUpperCase()
    output: \${up.output}
`;

const ROWS = { rows: [{ output: "a" }, { output: "b" }] };

/** Load and run `text` over `ROWS`, expecting success, and return its output. */
async function run(text: string): Promise<unknown> {
  const result = await runFlow(await loadFlow(text), ROWS);
  expect(result.status).toBe("succeeded");
  return result.output;
}

test.each([
  ["${item.output}", ["A", "B"]],
  ["'${item.output}!'", ["A!", "B!"]],
])("map reads an item field named output (%s)", async (itemRef, expected) => {
  const text = `id: p
name: p
typedefs:
  Row: {output: str}
input:
  rows: !list(Row)
defs:
${LEAF}nodes:
  m:
    kind: map
    over: \${input.rows}
    call: leaf
    input:
      x: ${itemRef}
output: \${m.output}
`;
  expect(await run(text)).toEqual(expected);
});

test("map inside a called def reads an item field named output", async () => {
  const text = `id: p
name: p
typedefs:
  Row: {output: str}
input:
  rows: !list(Row)
defs:
${LEAF}  mid:
    input:
      rows: !list(Row)
    nodes:
      m:
        kind: map
        over: \${input.rows}
        call: leaf
        input:
          x: \${item.output}
    output: \${m.output}
nodes:
  c:
    kind: call
    call: mid
    input:
      rows: \${input.rows}
output: \${c.output}
`;
  expect(await run(text)).toEqual(["A", "B"]);
});
