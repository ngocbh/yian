/**
 * A read of a node a CASE may skip, while its reader still runs, is a load error wherever null
 * fails: an operand of arithmetic, an argument of a shipped builtin, the right side of `in` or
 * `not in`, and a comprehension's list. A builtin parameter whose default is null takes null as
 * left out, so a read there loads, as do a read with a default, a guarded read, the left side of
 * `in`, `isinstance`, a host function and a reader skipped with its producer.
 */

import { afterEach, expect, test } from "bun:test";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { loadFlow } from "../../src/compose/loader.ts";
import { registerTemplateFn, TEMPLATE_FNS } from "../../src/expr/builtins.ts";
import { runFlow } from "../../src/run.ts";
import { assembleText, loadError } from "../helpers/declarations.ts";
import { hostEnv } from "../helpers/mcp_helpers.ts";

/** Return the j5 flow: `gate` runs `tags` only when `input.go`, and `count` reads it as `value`. */
const j5 = (value: string, type = "int") =>
  "id: j5\nname: j5\ninput:\n  go: !bool\nnodes:\n" +
  "  gate:\n    kind: case\n    cases:\n      - when: ${input.go}\n        then: tags\n" +
  "  tags: !list(str) \"${['a', 'b']}\"\n" +
  `  count: !${type} '${value.replaceAll("'", "''")}'\n` +
  "output: ${count.output}\n";

/** Return the message for `source`, which does `doing` with `operand`, null when node 'tags' is skipped. */
const refused = (source: string, doing: string, operand = "tags.output") =>
  `value entry 'count': ${source} ${doing}, which is null when node 'tags' is skipped; give it a default ` +
  `with :-, e.g. (${operand} :- <default>)`;

test.each([
  ["${len(tags.output)}", "int", "passes tags.output to len()"],
  ['${join(tags.output, ",")}', "str", "passes tags.output to join()"],
  ['${join(["a"], sep=tags.output[0])}', "str", "passes tags.output to join()"],
  ["${render_as_json(tags.output)}", "str", "passes tags.output to render_as_json()"],
  ["${upper(tags.output[0])}", "str", "passes tags.output to upper()"],
  ["${len(tags.output) :- 0}", "int", "passes tags.output to len()"],
  ['${3 in tags.output}', "bool", "uses tags.output as the right side of `in`"],
  ['${"a" not in tags.output}', "bool", "uses tags.output as the right side of `not in`"],
  ["${[x for x in tags.output]}", "list(str)", "iterates over tags.output"],
  ["${[y for x in ['a'] for y in tags.output]}", "list(str)", "iterates over tags.output"],
])("%s is a load error naming the read", (value, type, doing) => {
  const error = loadError(() => assembleText(j5(value, type)));
  expect(error.message).toBe(refused(value, doing));
});

test("a searched when: that passes a maybe-skipped read to a builtin is a load error", () => {
  const text =
    "id: s\nname: s\ninput:\n  go: !bool\nnodes:\n" +
    "  gate:\n    kind: case\n    cases:\n      - when: ${input.go}\n        then: quick\n" +
    "  quick: !list(int) \"${[1, 2]}\"\n" +
    "  pick:\n    kind: case\n    cases:\n      - when: (len(quick.output) :- 0) > 1\n        then: many\n" +
    "    else: few\n" +
    "  many: !str many\n  few: !str few\n" +
    "output: ${many.output | few.output}\n";
  expect(loadError(() => assembleText(text)).message).toBe(
    "node 'pick' cases[0].when: (len(quick.output) :- 0) > 1 passes quick.output to len(), which is null " +
      "when node 'quick' is skipped; give it a default with :-, e.g. (quick.output :- <default>)",
  );
});

test.each([
  ["${len(tags.output :- [])}", "int"],
  ["${tags.output != null and len(tags.output) > 0}", "bool"],
  ["${len(tags.output) if tags.output != null else 0}", "int"],
  ["${tags.output in [['a', 'b']]}", "bool"],
  ["${isinstance(tags.output, list)}", "bool"],
  ['${split("a b", sep=tags.output[0])}', "list(str)"],
  ['${strip(" a ", tags.output[0])}', "str"],
  ["${[x for x in (tags.output :- [])]}", "list(str)"],
])("%s loads and runs on both branches", async (value, type) => {
  const flow = await loadFlow(j5(value, type));
  for (const go of [true, false]) {
    const result = await runFlow(flow, { go });
    expect(result.failure).toBeNull();
  }
});

test("a reader skipped with its producer loads", () => {
  // `n` reads `tags.output` whole too, so it is skipped with `tags` and never sees null.
  const text =
    "id: j5\nname: j5\ninput:\n  go: !bool\nnodes:\n" +
    "  gate:\n    kind: case\n    cases:\n      - when: ${input.go}\n        then: tags\n" +
    "  tags: !list(str) \"${['a', 'b']}\"\n" +
    "  n:\n    kind: code\n    input:\n      all: ${tags.output}\n      count: ${len(tags.output)}\n" +
    "    output: !int\n    code: return inputs.count\n" +
    "output: ${n.output}\n";
  assembleText(text);
});

test("a reader on the producer's own branch loads", () => {
  const text =
    "id: j5\nname: j5\ninput:\n  go: !bool\nnodes:\n" +
    "  gate:\n    kind: case\n    cases:\n      - when: ${input.go}\n        then: tags\n" +
    "  tags: !list(str) \"${['a', 'b']}\"\n" +
    "  check:\n    kind: case\n    cases:\n      - when: ${len(tags.output) > 1}\n        then: done\n" +
    "  done: !str done\n" +
    "output: ${done.output}\n";
  assembleText(text);
});

const dir = mkdtempSync(join(tmpdir(), "yian-skipped-null-reads-"));
const saved = TEMPLATE_FNS.get("count_or_zero");
afterEach(() => {
  if (saved === undefined) TEMPLATE_FNS.delete("count_or_zero");
  else TEMPLATE_FNS.set("count_or_zero", saved);
});

test("a host function may take a maybe-skipped read", async () => {
  registerTemplateFn("count_or_zero", {
    params: [{ name: "xs" }],
    call: (xs) => BigInt(Array.isArray(xs) ? xs.length : 0),
  });
  const env = hostEnv(dir);
  try {
    const flow = await loadFlow(j5("${count_or_zero(tags.output)}"), { env });
    expect((await runFlow(flow, { go: false }, { env })).output).toBe(0n);
    expect((await runFlow(flow, { go: true }, { env })).output).toBe(2n);
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});
