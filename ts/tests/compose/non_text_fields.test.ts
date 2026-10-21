/**
 * A text or expression field YAML reads as non-text is a load error at that value: a port of
 * `tests/compose/test_non_text_fields.py`. Values are as TS YAML gives them: ints are `bigint`.
 * The flow rows read through `parseFile`, the node rows through `parseFile`, `flowNodesCursor`
 * and `parseNodes` with the registered kinds.
 *
 * Changed: a flow with no `nodes:` says "missing required top-level key 'nodes'", where Python's
 * text is pydantic's. The null and text `node_name:` rows, the null optional flow field rows and
 * the null and text `on:` rows check the parsed value, where Python loads the flow; the `on:` rows
 * also assemble it (`assembleText`), the null one failing on its unresolved `when: x`.
 * `test_text_code_and_tool_id_still_load` assembles the root flow with `assembleText`, where
 * Python loads it. `test_non_text_call_is_a_load_error` and `test_null_call_is_empty` parse the
 * flow's nodes, where Python loads it; `test_text_call_still_loads` checks the parsed `call:`, then
 * loads the flow with `loadFlow`. The while and until rows of
 * `test_already_located_fields_are_unchanged` parse the flow's nodes too, since the LOOP parse
 * raises their error. The searched-when and on-when rows of that test, whose errors the CASE build
 * raises, and its then and else rows, whose errors the case-branch check raises, load with
 * `assembleText`, as the loader assembles the root flow.
 *
 * The `llm_config` rows (`test_non_text_llm_config_field_is_a_load_error`,
 * `test_nested_llm_config_text_field`, `test_other_llm_config_errors_are_unchanged`) load each of
 * the node, flow, def and `adaptive_questions.llm_config` sites with `loadFlow`, as do the def
 * `asserts:` rows (`test_non_text_def_assert_is_a_load_error`,
 * `test_non_list_def_asserts_is_a_load_error`, `test_null_def_asserts_loads`).
 *
 * The harvested `load-core/compose_non_text_fields__*` cases are pinned at the end of each block,
 * by the call that raises their error. The inline-body rows of
 * `test_code_in_def_compact_merge_and_inline_body` and
 * `test_non_text_kind_in_a_def_and_inline_body` load with `assembleText` (the error comes from the
 * child's assembly) and pin the whole message; their def, compact and merge rows load with
 * `loadFlow`.
 *
 * Dropped: the date rows (TS reads an unquoted date as text) and
 * `test_not_text_detail_reads_the_first_string_error` (`not_text_detail` reads a pydantic error).
 */

import { afterEach, describe, expect, test } from "bun:test";

import type { LoadError } from "../../src/compose/helpers/errors.ts";
import { notTextError } from "../../src/compose/nodes/base.ts";
import type { CallDescriptor } from "../../src/compose/nodes/call.ts";
import type { CaseDescriptor } from "../../src/compose/nodes/case.ts";
import type { LoopDescriptor } from "../../src/compose/nodes/loop.ts";
import { loadFlow } from "../../src/compose/loader.ts";
import { parseFile } from "../../src/compose/parse/file.ts";
import { forget } from "../../src/defaults.ts";
import {
  assembled,
  assembleText,
  fileOf,
  loadError,
  loadErrorAsync,
  nodesOf,
  parsedNodes,
  runCase,
  runTextCase,
} from "./declarations.ts";

afterEach(() => forget());

test("notTextError names the field", () => {
  expect(notTextError("prompt", 5n)).toBe(
    "`prompt:` must be text, but YAML read it as an integer; put it in quotes",
  );
  expect(notTextError("questions[0].question", [1n])).toBe(
    "`questions[0].question:` must be text, but YAML read it as a list; put it in quotes",
  );
  expect(notTextError("questions[0].header", null)).toBe(
    "`questions[0].header:` must be text, but it is empty; write the text",
  );
});

test("notTextError takes a fix", () => {
  expect(notTextError("kind", [1n], "write one of a, b")).toBe(
    "`kind:` must be text, but YAML read it as a list; write one of a, b",
  );
  expect(notTextError("kind", null, "write one of a, b")).toBe(
    "`kind:` must be text, but it is empty; write one of a, b",
  );
});

// ---------- the parse rows ----------

/** Non-text values with what YAML read each as; the date row drops, since TS reads it as text. */
const NON_TEXT = [
  ["5", "an integer"],
  ["0", "an integer"],
  ["true", "a boolean"],
  ["false", "a boolean"],
  ["1.5", "a float"],
  ["[a]", "a list"],
  ["[]", "a list"],
  ["{x: 1}", "a mapping"],
  ["{}", "a mapping"],
] as const;

/** `NON_TEXT` with each phrase as the message words it, plus the empty `null`. */
const NON_TEXT_OR_NULL = [
  ...NON_TEXT.map(([value, phrase]) => [value, `YAML read it as ${phrase}`] as const),
  ["null", "it is empty"] as const,
];

/** The 1-based line of `text` holding `marker` (the first match). */
function lineOf(text: string, marker: string): number {
  const index = text.split("\n").findIndex((line) => line.includes(marker));
  if (index < 0) throw new Error(`${marker} not in text`);
  return index + 1;
}

/** Return the start line of the last frame of `error`'s diagnostic. */
function startLine(error: LoadError): number {
  return error.diagnostic.frames.at(-1)!.range.startLine;
}

/** Return the `LoadError` loading `text` with `loadFlow` rejects with. */
const loadFails = (text: string) => loadErrorAsync(loadFlow(text));

/** Return the `LoadError` `parseFile` throws for `text`. */
function fileError(text: string): LoadError {
  return loadError(() => parseFile(text));
}

/** A flow whose `nodes:` entry `a` has `kind: <v>`. */
function kindFlow(v: string): string {
  return `id: f
name: f
nodes:
  a:
    kind: ${v}
    prompt: hi
output: \${a.output}
`;
}

/** A compact (single-node) flow with `kind: <v>`. */
function compactKind(v: string): string {
  return `id: f\nname: f\noutput: int\nkind: ${v}\ncode: return 1\n`;
}

/** A compact (single-node) flow with `id: <v>`, or no `id:` when `v` is null. */
function compactId(v: string | null): string {
  const head = v === null ? "" : `id: ${v}\n`;
  return `${head}name: f\noutput: int\nkind: code\ncode: return 1\n`;
}

/** A flow whose code node `a` has `node_name: <v>`. */
function nodeNameFlow(v: string): string {
  return `id: f
name: f
nodes:
  a:
    kind: code
    node_name: ${v}
    output: str
    code: "return 'x'"
output: \${a.output}
`;
}

/** A flow whose CODE node `a` has `code: <v>`. */
function codeFlow(v: string): string {
  return `id: f
name: f
nodes:
  a:
    kind: code
    output: str
    code: ${v}
output: \${a.output}
`;
}

/** A flow whose TOOL node `a` has `tool_id: <v>`. */
function toolFlow(v: string): string {
  return `id: f
name: f
nodes:
  a:
    kind: tool
    tool_id: ${v}
    input: {symbol: x}
output: \${a.output}
`;
}

/** A flow whose CASE node `a` has `on: <v>`, routing to CODE nodes `b` and `c`. */
function caseOn(v: string): string {
  return `id: f
name: f
input:
  proj: !str
nodes:
  a:
    kind: case
    on: ${v}
    cases:
      - when: x
        then: b
    else: c
  b:
    kind: code
    output: str
    code: "return 'b'"
  c:
    kind: code
    output: str
    code: "return 'c'"
output: \${b.output | c.output}
`;
}

/** The fields each composite kind needs before its `call:`, so only `call:` is wrong. */
const CALL_HEADS: ReadonlyMap<string, string> = new Map([
  ["call", ""],
  ["map", "    over: ${input.xs}\n"],
  ["loop", "    times: 2\n"],
]);

/** A flow whose `kind` node `a` has `call: <v>`, beside a `defs:` entry `body`. */
function callFlow(kind: string, v: string): string {
  return `id: f
name: f
input:
  xs: !list(int)
defs:
  body:
    input:
      v: !int
    nodes:
      one:
        kind: code
        output: {v: int}
        code: "return {'v': 1}"
    output: \${one.output}
nodes:
  a:
    kind: ${kind}
${CALL_HEADS.get(kind)!}    call: ${v}
    input: {v: 1}
output: \${a.output}
`;
}

/** `caseOn`'s flow without `on:`: a searched CASE whose case reads `when: x`. */
const SEARCHED_CASE = caseOn("null").replace("    on: null\n", "");

const FLOW = `id: f
name: f
nodes:
  a:
    kind: code
    output: str
    code: "return 'x'"
output: \${a.output}
`;

/** A flow whose top-level `<field>:` is `<v>`. */
function flowField(field: string, v: string): string {
  const text = `id: f
name: f
description: d
version: '1'
nodes:
  a:
    kind: code
    output: str
    code: "return 'x'"
output: \${a.output}
`;
  return text
    .split("\n")
    .map((line) => (line.startsWith(`${field}: `) ? `${field}: ${v}` : line))
    .join("\n");
}

describe("parseFile", () => {
  test.each(NON_TEXT_OR_NULL)("non-text compact kind is a load error: %s", (value, phrase) => {
    const text = compactKind(value);

    const error = fileError(text);

    expect(error.message).toContain(
      `compact (single-node) flow: \`kind:\` must be text, but ${phrase}; ` +
        "write one of agent, code, human_input, tool",
    );
    expect(startLine(error)).toBe(lineOf(text, "kind: "));
  });

  test.each(NON_TEXT)("non-text compact id is a load error: %s", (value, phrase) => {
    const text = compactId(value);

    const error = fileError(text);

    expect(error.message).toContain(
      `compact (single-node) flow: \`id:\` must be text, but YAML read it as ${phrase}; ` +
        "put it in quotes",
    );
    expect(startLine(error)).toBe(lineOf(text, "id: "));
  });

  test.each([
    ["null", "null"],
    ["empty", "''"],
    ["absent", null],
  ] as const)("null, empty and absent compact id are required: %s", (_, value) => {
    expect(fileError(compactId(value)).message).toContain(
      "compact (single-node) flow: a top-level `id:` is required",
    );
  });

  test.each(
    ["id", "name", "description", "version"].flatMap((field) =>
      (
        [
          ["5", "an integer"],
          ["true", "a boolean"],
          ["1.5", "a float"],
          ["[a]", "a list"],
        ] as const
      ).map(([value, phrase]) => [field, value, phrase] as const),
    ),
  )("non-text flow field is a load error: %s %s", (field, value, phrase) => {
    const text = flowField(field, value);

    const error = fileError(text);

    expect(
      error.message.startsWith(
        `\`${field}:\` must be text, but YAML read it as ${phrase}; put it in quotes`,
      ),
    ).toBe(true);
    expect(startLine(error)).toBe(lineOf(text, `${field}: `));
  });

  test.each(["id", "name"])("null flow field: %s", (field) => {
    expect(fileError(flowField(field, "null")).message).toContain(
      `\`${field}:\` must be text, but it is empty; write the text`,
    );
  });

  test.each(["description", "version"])("null optional flow field parses: %s", (field) => {
    expect(parseFile(flowField(field, "null"))[field as "description" | "version"]).toBeNull();
  });

  test.each(NON_TEXT_OR_NULL)("non-text uses ref is a load error: %s", (value, phrase) => {
    const text = `${FLOW}uses:\n  lib: ${value}\n`;

    const error = fileError(text);

    expect(error.message).toContain(`\`uses.lib:\` must be text, but ${phrase}`);
    expect(startLine(error)).toBe(lineOf(text, "lib: "));
  });

  test.each(NON_TEXT)("non-text flow assert is a load error: %s", (value, phrase) => {
    const text = `${FLOW}asserts:\n  - "true"\n  - ${value}\n`;

    const error = fileError(text);

    expect(error.message).toContain(`\`asserts[1]:\` must be text, but YAML read it as ${phrase}`);
    expect(startLine(error)).toBe(text.split("\n").length - 1);
  });

  test("a missing nodes section names it", () => {
    // Changed: Python's text is pydantic's "validation error for ComposeFile".
    expect(fileError("id: f\nname: f\noutput: x\n").message).toBe(
      "missing required top-level key 'nodes'",
    );
  });

  test.each([
    "load-core/compose_non_text_fields__missing_nodes_keeps_the_schema_error",
    "load-core/compose_non_text_fields__non_text_compact_id",
    "load-core/compose_non_text_fields__non_text_compact_kind",
    "load-core/compose_non_text_fields__non_text_flow_assert",
    "load-core/compose_non_text_fields__non_text_flow_version",
    "load-core/compose_non_text_fields__null_compact_id",
    "load-core/compose_non_text_fields__null_compact_kind",
    "load-core/compose_non_text_fields__null_flow_id",
  ])("case %s", (id) => runTextCase({ id, call: fileOf }));
});

describe("parseNodes", () => {
  test.each(NON_TEXT)("non-text node name is a load error: %s", (value, phrase) => {
    const text = nodeNameFlow(value);

    const error = loadError(() => nodesOf(text));

    expect(error.message).toContain(
      `node 'a' (kind=code): \`node_name:\` must be text, but YAML read it as ${phrase}; ` +
        "put it in quotes",
    );
    expect(startLine(error)).toBe(lineOf(text, "node_name: "));
  });

  test.each([
    ["null", null],
    ["'5'", "5"],
  ])("null and text node name still parse: %s", (value, expected) => {
    expect(nodesOf(nodeNameFlow(value)).get("a")!.nodeName).toBe(expected);
  });

  test.each(NON_TEXT)("non-text kind is a load error: %s", (value, phrase) => {
    const text = kindFlow(value);

    const error = loadError(() => nodesOf(text));

    expect(error.message).toContain(
      `node 'a': \`kind:\` must be text, but YAML read it as ${phrase}; ` +
        "write one of agent, call, case, code, human_input, loop, map, tool, wait",
    );
    expect(startLine(error)).toBe(lineOf(text, "kind: "));
  });

  test.each([
    ["null", "add `kind:` for a node"],
    ["llm", "unknown kind"],
    ["''", "unknown kind"],
  ])("null and text kinds are unchanged: %s", (value, fragment) => {
    expect(loadError(() => nodesOf(kindFlow(value))).message).toContain(fragment);
  });

  test.each(NON_TEXT)("non-text case on is a load error: %s", (value, phrase) => {
    const text = caseOn(value);

    const error = loadError(() => nodesOf(text, []));

    expect(error.message).toContain(
      `node 'a' (kind=case): \`on:\` must be text, but YAML read it as ${phrase}; ` +
        "write a reference such as `on: ${input.x}`",
    );
    expect(startLine(error)).toBe(lineOf(text, "on: "));
  });

  describe.each([...CALL_HEADS.keys()])("%s", (kind) => {
    test.each(NON_TEXT)("non-text call is a load error: %s", (value, phrase) => {
      const text = callFlow(kind, value);

      const error = loadError(() => nodesOf(text));

      expect(error.message).toContain(
        `node 'a' (kind=${kind}): \`call:\` must be text, but YAML read it as ${phrase}; put it in quotes`,
      );
      expect(startLine(error)).toBe(lineOf(text, "call: "));
    });

    test("null call is empty", () => {
      const text = callFlow(kind, "null");

      const error = loadError(() => nodesOf(text));

      expect(error.message).toContain(
        `node 'a' (kind=${kind}): \`call:\` must be text, but it is empty; write the text`,
      );
      expect(startLine(error)).toBe(lineOf(text, "call: "));
    });

    test("text call still loads", async () => {
      expect((nodesOf(callFlow(kind, "body")).get("a") as CallDescriptor | LoopDescriptor).call).toBe("body");
      await loadFlow(callFlow(kind, "body"));
    });
  });

  test.each([
    ["while", "while: 5"],
    ["until", "until: 5"],
  ])("a non-string %s is located as before", (key, marker) => {
    const text = callFlow("loop", "body").replace("    times: 2\n", `    times: 2\n    ${marker}\n`);

    const error = loadError(() => nodesOf(text));

    expect(error.message).toContain("must be an expression string over the carried record");
    expect(error.message).toContain(`\`${key}:\``);
    expect(startLine(error)).toBe(lineOf(text, marker));
  });

  test("null case on is absent", () => {
    // `on: null` is the searched form.
    expect((nodesOf(caseOn("null"), []).get("a") as CaseDescriptor).on).toBeNull();
    expect(loadError(() => assembleText(caseOn("null"))).message).toContain("unresolved references");
  });

  test("text case on still loads", () => {
    expect((nodesOf(caseOn("${input.proj}"), []).get("a") as CaseDescriptor).on).toBe("${input.proj}");
    assembleText(caseOn("${input.proj}"));
  });

  test("a node's asserts must be a list of strings, located", () => {
    const text = FLOW.replace("    output: str\n", '    output: str\n    asserts: ["true", 5]\n');

    const error = loadError(() => nodesOf(text));

    expect(error.message).toContain("`asserts:` must be a list of strings");
    expect(startLine(error)).toBe(lineOf(text, "asserts: "));
  });

  test.each(NON_TEXT)("non-text code is a load error: %s", (value, phrase) => {
    const text = codeFlow(value);

    const error = loadError(() => nodesOf(text));

    expect(error.message).toContain(
      `node 'a' (kind=code): \`code:\` must be text, but YAML read it as ${phrase}; put it in quotes`,
    );
    expect(startLine(error)).toBe(lineOf(text, "code: "));
  });

  test.each(NON_TEXT)("non-text tool id is a load error: %s", (value, phrase) => {
    const text = toolFlow(value);

    const error = loadError(() => nodesOf(text));

    expect(error.message).toContain(
      `node 'a' (kind=tool): \`tool_id:\` must be text, but YAML read it as ${phrase}; put it in quotes`,
    );
    expect(startLine(error)).toBe(lineOf(text, "tool_id: "));
  });

  test.each([
    [codeFlow("null"), "code", "code"],
    [toolFlow("null"), "tool", "tool_id"],
  ])("null code and tool id are empty: %#", (text, kind, key) => {
    const error = loadError(() => nodesOf(text));

    expect(error.message).toContain(
      `node 'a' (kind=${kind}): \`${key}:\` must be text, but it is empty; write the text`,
    );
    expect(startLine(error)).toBe(lineOf(text, `${key}: `));
  });

  test.each([codeFlow(`"return 'x'"`), toolFlow("'5'")])("text code and tool id still load: %#", (text) => {
    expect(() => assembleText(text)).not.toThrow();
  });

  test("tagged non-text tool id", () => {
    // `!str 5` still reads as an integer here; the tag's own error comes once it is quoted.
    expect(loadError(() => nodesOf(toolFlow("!str 5"))).message).toContain(
      "`tool_id:` must be text, but YAML read it as an integer; put it in quotes",
    );
  });

  test.each(["load-core/compose_non_text_fields__non_text_kind", "load-core/compose_non_text_fields__null_kind"])(
    "case %s",
    (id) => runTextCase({ id, call: (text, label) => nodesOf(text, [], label) }),
  );

  test.each([
    "load-core/compose_non_text_fields__node_asserts_holding_a_number",
    "load-core/compose_non_text_fields__non_text_call_on_a_call_node",
    "load-core/compose_non_text_fields__non_text_call_on_a_loop_node",
    "load-core/compose_non_text_fields__non_text_call_on_a_map_node",
    "load-core/compose_non_text_fields__non_text_case_on",
    "load-core/compose_non_text_fields__non_text_code",
    "load-core/compose_non_text_fields__non_text_loop_while",
    "load-core/compose_non_text_fields__non_text_node_name",
    "load-core/compose_non_text_fields__non_text_tool_id",
    "load-core/compose_non_text_fields__null_call",
    "load-core/compose_non_text_fields__null_tool_id",
  ])("case %s", (id) => runCase({ id, call: parsedNodes }));
});

describe("the CASE build", () => {
  test("a searched-when error is located as before", () => {
    const text = SEARCHED_CASE.replace("when: x", "when: 5");

    const error = loadError(() => assembleText(text));

    expect(error.message).toContain("each searched case needs a string `when:` expression");
    expect(startLine(error)).toBe(lineOf(text, "when: 5"));
  });

  test("an on-when error is located as before", () => {
    const text = caseOn("${input.proj}").replace("when: x", "when: [a]");

    const error = loadError(() => assembleText(text));

    expect(error.message).toContain("takes text, a number, or a bool");
    expect(startLine(error)).toBe(lineOf(text, "when: [a]"));
  });

  test.each([
    ["then", "then: b", "then: 5"],
    ["else", "else: c", "else: 5"],
  ])("a non-text %s is located as before", (key, written, marker) => {
    const text = caseOn("${input.proj}").replace(written, marker);

    const error = loadError(() => assembleText(text));

    expect(error.message).toContain("a branch names a node or calls a flow");
    expect(startLine(error)).toBe(lineOf(text, marker));
  });

  test.each([
    "load-core/compose_non_text_fields__list_on_when",
    "load-core/compose_non_text_fields__non_text_searched_when",
    "load-core/compose_non_text_fields__non_text_then",
  ])("case %s", (id) => runCase({ id, call: assembled }));
});

describe("an inline body", () => {
  const codeInInlineBody = `id: f
name: f
nodes:
  a:
    kind: call
    nodes:
      echo:
        kind: code
        output: str
        code: 5
    output: \${echo.output}
output: \${a.output}
`;
  const kindInInlineBody = `id: f
name: f
nodes:
  a:
    kind: call
    nodes:
      echo:
        kind: [a]
        output: str
        code: "return 'x'"
    output: \${echo.output}
output: \${a.output}
`;

  test("non-text code in an inline body is a load error", () => {
    const error = loadError(() => assembleText(codeInInlineBody));

    expect(error.message).toBe(
      "node 'echo' (kind=code): `code:` must be text, but YAML read it as an integer; put it in quotes",
    );
    expect(startLine(error)).toBe(lineOf(codeInInlineBody, "code: 5"));
  });

  test("non-text kind in an inline body is a load error", () => {
    const error = loadError(() => assembleText(kindInInlineBody));

    expect(error.message).toBe(
      "node 'echo': `kind:` must be text, but YAML read it as a list; " +
        "write one of agent, call, case, code, human_input, loop, map, tool, wait",
    );
    expect(startLine(error)).toBe(lineOf(kindInInlineBody, "kind: [a]"));
  });

  test.each(["load-core/compose_non_text_fields__non_text_kind_in_an_inline_body"])("case %s", (id) =>
    runCase({ id, call: assembled }),
  );
});

describe("a def, compact or merged node", () => {
  const codeInDef = `id: f
name: f
defs:
  body:
    nodes:
      one:
        kind: code
        output: str
        code: "return 'x'"
      two:
        kind: code
        output: str
        code: 5
    output: \${two.output}
nodes:
  a:
    kind: call
    call: body
output: \${a.output}
`;
  const codeCompact = "id: f\nname: f\noutput: str\nkind: code\ncode: 5\n";
  const codeMerged = `id: f
name: f
x-base: &base {code: 5}
nodes:
  a:
    <<: *base
    kind: code
    output: str
output: \${a.output}
`;
  const kindInDef = `id: f
name: f
defs:
  sub:
    kind: [a]
    output: str
    code: "return 'x'"
nodes:
  a:
    kind: call
    call: sub
output: \${a.output}
`;

  test.each([
    ["def", codeInDef, "two", lineOf(codeInDef, "code: 5")],
    ["compact", codeCompact, "f", lineOf(codeCompact, "code: 5")],
    ["merge", codeMerged, "a", lineOf(codeMerged, "x-base:")],
  ])("non-text code in a %s is a load error", async (_site, text, node, line) => {
    const error = await loadFails(text);

    expect(error.message).toContain(`node '${node}' (kind=code): \`code:\` must be text`);
    expect(startLine(error)).toBe(line);
  });

  test("non-text kind in a def is a load error", async () => {
    const error = await loadFails(kindInDef);

    expect(error.message).toContain("`kind:` must be text");
    expect(startLine(error)).toBe(lineOf(kindInDef, "kind: [a]"));
  });
});

describe("llm_config", () => {
  /** A flow whose agent node `a` has `llm_config: <v>`. */
  const llmOnNode = (v: string) => `id: f
name: f
nodes:
  a:
    kind: agent
    prompt: hi
    llm_config: ${v}
output: \${a.output}
`;
  /** A flow whose adaptive gate `ask` has `adaptive_questions.llm_config: <v>`. */
  const llmOnGate = (v: string) => `id: f
name: f
input:
  proj: !str
nodes:
  ask:
    kind: human_input
    input: {proj: '\${input.proj}'}
    adaptive_questions:
      prompt: "Q \${proj}"
      llm_config: ${v}
output: \${ask.output}
`;
  /** A flow with the flow-level `llm_config: <v>`. */
  const llmOnFlow = (v: string) => `id: f
name: f
llm_config: ${v}
nodes:
  a:
    kind: agent
    prompt: hi
output: \${a.output}
`;
  /** A flow whose `defs:` entry `sub` has `llm_config: <v>`. */
  const llmOnDef = (v: string) => `id: f
name: f
defs:
  sub:
    llm_config: ${v}
    nodes:
      b:
        kind: agent
        prompt: hi
    output: \${b.output}
nodes:
  a:
    kind: call
    call: sub
output: \${a.output}
`;
  const SITES: [string, (v: string) => string, string][] = [
    ["node", llmOnNode, "node 'a': llm_config: "],
    ["flow", llmOnFlow, "flow: llm_config: "],
    ["def", llmOnDef, "defs entry 'sub': llm_config: "],
    ["adaptive", llmOnGate, "adaptive_questions.llm_config: "],
  ];

  test.each(SITES)("a non-text llm_config field is a load error: %s", async (_site, builder, prefix) => {
    const text = builder("{model: 5}");

    const error = await loadFails(text);

    expect(error.message).toContain(
      prefix + "`model:` must be text, but YAML read it as an integer; put it in quotes",
    );
    expect(error.message).not.toContain("validation error for LLMConfig");
    expect(startLine(error)).toBe(lineOf(text, "llm_config: "));
  });

  test.each(SITES)("a nested llm_config text field: %s", async (_site, builder, prefix) => {
    const error = await loadFails(builder("{provider: openai, openai: {user: 5}}"));

    expect(error.message).toContain(prefix + "`openai.user:` must be text");
  });

  describe.each(SITES)("other llm_config errors are unchanged: %s", (_site, builder, prefix) => {
    test.each([
      ["{provider: 5}", "1 validation error"],
      ["{provider: azure, azure: {stop: 5}}", "2 validation errors"],
      ["{provider: azure, azure: {stop: [5]}}", "2 validation errors"],
    ])("%s", async (value, count) => {
      expect((await loadFails(builder(value))).message).toContain(prefix + count + " for LLMConfig");
    });
  });

  test.each(["load-core/compose_non_text_fields__non_text_node_llm_config_field"])("case %s", (id) =>
    runCase({ id, call: assembled }),
  );
});

describe("def asserts", () => {
  /** A flow whose multi-node `defs:` entry `b` has `asserts: <v>`. */
  const defAsserts = (v: string) => `id: f
name: f
defs:
  b:
    nodes:
      c:
        kind: code
        output: int
        code: return 1
    output: \${c.output}
    asserts: ${v}
nodes:
  a:
    kind: call
    call: b
output: \${a.output}
`;

  test.each(NON_TEXT)("a non-text def assert is a load error: %s", async (value, phrase) => {
    const text = defAsserts(`\n      - "true"\n      - ${value}`);

    const error = await loadFails(text);

    expect(error.message).toContain(
      `defs entry 'b': \`asserts[1]:\` must be text, but YAML read it as ${phrase}; put it in quotes`,
    );
    expect(startLine(error)).toBe(lineOf(text, `      - ${value}`));
  });

  test.each(["5", "false", "0", "''", "{}", "{x: 1}", "output > 0"])(
    "a non-list def asserts is a load error: %s",
    async (value) => {
      const text = defAsserts(value);

      const error = await loadFails(text);

      expect(error.message).toContain("defs entry 'b': `asserts:` must be a list of strings");
      expect(startLine(error)).toBe(lineOf(text, "asserts: "));
    },
  );

  test("null def asserts loads", async () => {
    await loadFlow(defAsserts("null"));
  });
});
