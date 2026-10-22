/**
 * Typed bindings that narrow a union, checked when a run binds the value: a port of
 * `tests/engine/test_typed_bindings_unions_run.py`.
 *
 * Ported: `test_a_member_value_passes_and_the_consumer_reads_its_fields` and
 * `test_a_narrowed_child_output_reaches_a_union_typed_parent_input`. The CODE bodies are Python's,
 * which read the same in JS (decision 11), but for `String(...)` in place of `str(...)`, as the
 * harvested cases write it. The int output reads as a `bigint` (decision 4a).
 *
 * Covered by `run-core/engine_typed_bindings_unions_run__another_member_fails_the_consumer`:
 * `test_another_member_fails_the_consumer_on_its_binding`; by
 * `run-core/engine_typed_bindings_unions_run__another_member_fails_the_childs_output`:
 * `test_another_member_fails_the_childs_narrowed_output`.
 */

import { afterAll, expect, test } from "bun:test";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { loadFlow } from "../../src/compose/loader.ts";
import { runFlow } from "../../src/run.ts";
import { hostEnv } from "./mcp_helpers.ts";

const dir = mkdtempSync(join(tmpdir(), "yian-typed-bindings-unions-run-"));
afterAll(() => rmSync(dir, { recursive: true, force: true }));
const env = hostEnv(dir);

const HEAD = `
id: narrow
name: narrow
typedefs:
  Buy: {action: 'Literal(buy)', qty: int}
  Sell: {action: 'Literal(sell)', qty: int, price: float}
  Decision: Union(Buy, Sell)
input:
  decision: !object
`;

const NODE_FLOW =
  HEAD +
  `
nodes:
  decide:
    kind: code
    input:
      decision: \${input.decision}
    code: return inputs['decision']
    output: Decision
  place:
    kind: code
    input:
      buy: !Buy \${decide.output}
    code: return 'buy ' + String(inputs['buy']['qty'])
    output: str
output: \${place.output}
`;

const CALL_FLOW =
  HEAD +
  `
defs:
  pick:
    input:
      decision: !object
    nodes:
      decide:
        kind: code
        input:
          decision: \${input.decision}
        code: return inputs['decision']
        output: Decision
    output: !Buy \${decide.output}
nodes:
  picked:
    kind: call
    call: pick
    input:
      decision: \${input.decision}
  widen:
    kind: code
    input:
      d: !Decision \${picked.output}
    code: return inputs['d']['qty']
    output: int
output: \${widen.output}
`;

const BUY = { action: "buy", qty: 3 };

/** Load `text`, run it on `decision`, and return the output of the run, which must succeed. */
async function output(text: string, decision: unknown): Promise<unknown> {
  const result = await runFlow(await loadFlow(text, { env }), { decision }, { env });
  expect(result.failure).toBeNull();
  expect(result.status).toBe("succeeded");
  return result.output;
}

test("a member value passes and the consumer reads its fields", async () => {
  expect(await output(NODE_FLOW, BUY)).toBe("buy 3");
});

test("a narrowed child output reaches a union typed parent input", async () => {
  expect(await output(CALL_FLOW, BUY)).toBe(3n);
});
