/**
 * Nested `${...}` in input and output values is evaluated, end to end: a port of
 * `tests/engine/test_nested_templates_run.py`.
 *
 * A mapping or list binding value is a template at every depth: a nested whole `${...}` keeps its
 * value, a nested interpolation renders text, `$$` unescapes, and a nested tag checks the value at
 * its position without filling anything in.
 *
 * Ported: every row but those covered below. The CODE bodies are Python's, which read the same in
 * JS (decision 11). Ints read as `bigint`s (decision 4a).
 *
 * Covered by `run-core/engine_nested_templates_run__nested_reference_in_a_flow_output_field`:
 * `test_nested_reference_in_a_flow_output_field_is_evaluated`; by
 * `run-core/engine_nested_templates_run__nested_escaped_span`:
 * `test_nested_escaped_span_unescapes_to_text`; by
 * `run-core/engine_nested_templates_run__nested_escape_without_a_span`:
 * `test_nested_escape_without_a_span_unescapes_to_text`; by
 * `run-core/engine_nested_templates_run__nested_interpolation_renders_text`:
 * `test_nested_interpolation_renders_text`; by
 * `run-core/engine_nested_templates_run__nested_tag_inside_a_list`:
 * `test_nested_tag_inside_a_list_checks_its_item`.
 */

import { afterAll, expect, test } from "bun:test";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { loadFlow } from "../../src/compose/loader.ts";
import { type RunResult, runFlow } from "../../src/run.ts";
import { loadErrorAsync } from "../compose/declarations.ts";
import { hostEnv } from "./mcp_helpers.ts";

const dir = mkdtempSync(join(tmpdir(), "yian-nested-templates-run-"));
afterAll(() => rmSync(dir, { recursive: true, force: true }));
const env = hostEnv(dir);

const PRODUCER = `  n:
    kind: code
    output: str
    code: |
      return "hi"
`;

/** Return a flow with the `n` producer, `nodes` and `output`. */
const flow = (nodes: string, output: string, head = "") =>
  `${head}id: nested\nname: nested\nnodes:\n${PRODUCER}${nodes}output:${output}\n`;

/** Return a CODE node `c` that returns its input `p`, bound as `binding`. */
const echo = (binding: string) => `  c:\n    kind: code\n    input:\n      p:${binding}\n    code: |\n      return inputs["p"]\n`;

/** Load and run `text`; the run must succeed. */
async function run(text: string, inputs: Record<string, unknown> = {}): Promise<RunResult> {
  const result = await runFlow(await loadFlow(text, { env }), inputs, { env });
  expect(result.failure).toBeNull();
  expect(result.status).toBe("succeeded");
  return result;
}

test("reference in a flow output list is evaluated", async () => {
  const text = flow("", "\n  - ${n.output}\n  - 1");

  expect((await run(text)).output).toEqual(["hi", 1n]);
});

test("nested reference in a node input is evaluated", async () => {
  const text = flow(echo("\n        q: ${n.output}"), " ${c.output}");

  expect((await run(text)).output).toEqual({ q: "hi" });
});

test("nested item read in a map input is evaluated per item", async () => {
  const text = `id: nested_map
name: nested_map
input:
  rows: !list(str)
nodes:
  m:
    kind: map
    over: \${input.rows}
    input:
      v:
        i: \${item}
    nodes:
      echo:
        kind: code
        input:
          x: \${input.v}
        code: |
          return inputs["x"]
    output: \${echo.output}
output: \${m.output}
`;

  expect((await run(text, { rows: ["a", "b"] })).output).toEqual([{ i: "a" }, { i: "b" }]);
});

test("nested reference in an inline body argument is evaluated", async () => {
  const text = flow(
    `  c:
    kind: call
    input:
      payload:
        q: \${n.output}
        tags: [x, y]
    nodes:
      echo:
        kind: code
        input:
          x: \${input.payload}
        code: |
          return inputs["x"]
    output: \${echo.output}
`,
    " ${c.output}",
  );

  expect((await run(text)).output).toEqual({ q: "hi", tags: ["x", "y"] });
});

const BUY_TYPEDEFS = `typedefs:
  Buy:
    action: Literal(buy)
    conviction: float
`;

const SCORE = `  score:
    kind: code
    output: float
    code: |
      return 0.75
`;

/** Return a flow whose `c` node binds `decision` as a `!Buy` block of `fields`. */
const decisionFlow = (fields: string) =>
  flow(
    SCORE + "  c:\n    kind: code\n    input:\n      decision: !Buy\n" + fields + '    code: |\n      return inputs["decision"]\n',
    " ${c.output}",
    BUY_TYPEDEFS,
  );

test("nested tagged block checks and runs", async () => {
  const text = decisionFlow("        action: buy\n        conviction: ${score.output}\n");

  expect((await run(text)).output).toEqual({ action: "buy", conviction: 0.75 });
});

test("nested tagged block without a required field names it", async () => {
  const text = decisionFlow("        conviction: ${score.output}\n");

  const error = await loadErrorAsync(loadFlow(text, { env }));

  expect(error.message).toContain("(missing required `action`)");
});

test("nested tag inside a list rejects a mismatched item", async () => {
  const text = flow(echo("\n        - !int ${n.output}"), " ${c.output}");

  const error = await loadErrorAsync(loadFlow(text, { env }));

  expect(error.message).toContain("declared int, but the value is str");
});
