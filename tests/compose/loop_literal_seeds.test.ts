/**
 * A `while:` loop false on its seeds runs its body zero times and outputs the seeds, so a literal
 * seed of a field the body returns must fit the body's type for it, as a reference seed must. A
 * loop that runs at least once never outputs a seed as is: an `until:` or `times:` loop, or a
 * `while:` true on its literal seeds.
 */

import { expect, test } from "bun:test";

import { runFlow } from "../../src/run.ts";
import { assembled, assembleText, type CaseRow, loadConformanceFlow, loadError, runCase } from "../helpers/declarations.ts";
import { outputTag } from "../helpers/output_tags.ts";

/**
 * A flow whose LOOP `pages` calls the `defs:` body `step`, carrying `cursor` and `n`.
 *
 * @param driver The LOOP's driver lines.
 * @param cursor The `cursor:` seed.
 * @param options.in The body's input type for `cursor` (default `Optional(str)`).
 * @param options.out The body's output type for `cursor` (default `str`).
 * @param options.n The `n:` seed (default `0`).
 */
function pages(driver: string, cursor: string, options: { in?: string; out?: string; n?: string } = {}): string {
  const { in: input = "Optional(str)", out = "str", n = "0" } = options;
  return `
id: pages
name: pages
input:
  z: !int
defs:
  step:
    input:
      cursor: !${input}
      n: !int
    nodes:
      next:
        kind: code
        input:
          n: \${input.n}
        output:
          cursor: ${outputTag(out)}
          n: !int
        code: |
          return {"cursor": "page", "n": inputs.n + 1}
    output: \${next.output}
nodes:
  pages:
    kind: loop
    ${driver}
    max: 3
    call: step
    input:
      cursor: ${cursor}
      n: ${n}
output: \${pages.output}
`;
}

test("a while: false on a literal null seed is refused at load", () => {
  expect(loadError(() => assembleText(pages("while: cursor != null", "null"))).message).toBe(
    "loop node 'pages': carried field 'cursor' — body output is 'str', literal seed is 'None' " +
      "(a `while:` false at the start outputs the seeds; a returned field must keep its carried type)",
  );
});

test("a mismatched literal record seed is refused", () => {
  const text = pages("while: cursor == null", '{page: "x"}', { in: "Optional(object)", out: "{page: int}" });
  expect(loadError(() => assembleText(text)).message).toBe(
    "loop node 'pages': carried field 'cursor' — body output is '{\"page\":int}', literal seed is " +
      "'{\"page\":Literal(x)}' (a `while:` false at the start outputs the seeds; a returned field must keep " +
      "its carried type)",
  );
});

test("a while: reading a reference seed checks the literal seeds", () => {
  const text = pages("while: cursor != null or n < 1", "null", { n: "${input.z}" });
  expect(loadError(() => assembleText(text)).message).toContain("literal seed is 'None'");
});

test.each([
  ["a while: true on the literal seeds", "while: cursor == null"],
  ["an until: loop", "until: cursor != null"],
  ["a times: loop", "times: 1"],
])("%s runs at least once, so its literal seed loads and runs", async (_id, driver) => {
  const text = pages(driver, "null").replace("    max: 3\n", driver.startsWith("times") ? "" : "    max: 3\n");
  const result = await runFlow(await loadConformanceFlow(text), { z: 0n });
  expect(result.status, result.failure?.message).toBe("succeeded");
  expect(result.output).toEqual({ cursor: "page", n: 1n });
});

test("a literal seed that fits the body's type loads under a while: false on it", () => {
  assembleText(pages("while: cursor == null", '"a"'));
});

test.each<CaseRow>([{ id: "load-loop/compose_loop_literal_seeds__a_while_false_on_a_null_seed", call: assembled }])(
  "$id",
  runCase,
);
