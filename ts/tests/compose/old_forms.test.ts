/**
 * Each old type form of a flow input or agent variable is a located load error that shows the
 * tag spelling: a port of the declaration tests of `tests/compose/test_old_forms.py`, reading
 * the declarations with `parametersFromYaml` (`readSourceParameters`) where Python loads a flow.
 *
 * Ported: `test_untagged_flow_input_shows_the_tag`, `test_untagged_agent_variable_shows_the_tag`,
 * `test_source_free_declarations_take_a_type_only`,
 * `test_source_free_input_rejects_a_record_map`, and `test_typed_flow_output_text_shows_the_tag`
 * and `test_typed_single_flow_output_text_shows_the_tag`, which assemble the flow with
 * `assembleText`.
 *
 * TS-only: an untagged float default's example, spelled as Python's `repr` spells a finite
 * float and as `.inf` for infinity, which YAML reads back as a float (Python writes `inf`,
 * which reads back as text); a type with a default that doesn't parse (`n: int = [`), kept as
 * written where Python reads an unbound name; and the harvested-case table, comparing the
 * message and the last frame: its flow-input rows read the declarations, and its agent-variable
 * and flow-output rows assemble the flow as the loader calls `assemble`.
 *
 * Ported through `loadFlow`: `test_bracket_typedef_shows_the_parenthesized_spelling`,
 * `test_bracket_node_output_shows_the_parenthesized_spelling`,
 * `test_typed_binding_text_shows_the_tag`, `test_typed_binding_text_that_reads_as_words_offers_quoting`
 * and `test_untagged_input_in_a_definition_shows_the_tag`; and, run with `runFlow`,
 * `test_text_without_a_type_on_the_left_is_plain_text`.
 */

import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { afterAll, afterEach, beforeEach, describe, expect, test } from "bun:test";

import { readFlowInputs, readVariables } from "../../src/compose/helpers/types.ts";
import { loadFlow } from "../../src/compose/loader.ts";
import { BUILTIN, forget, install } from "../../src/defaults.ts";
import { runFlow } from "../../src/run.ts";
import { readTypedefs, type TypeRegistry } from "../../src/typesys/types.ts";
import {
  assembled,
  assembleText,
  type CaseRow,
  inputsAt,
  loadError,
  loadErrorAsync,
  parametersFromYaml,
  runCase,
} from "./declarations.ts";
import { hostEnv } from "../engine/mcp_helpers.ts";

// Pinned before `MESSAGE_REGISTRY` is read too, so neither depends on the shell's `YIAN_*` limits.
install(BUILTIN);
beforeEach(() => install(BUILTIN));
afterEach(forget);

const EMPTY: TypeRegistry = new Map();

/** Read `text`'s declarations, expecting `message` located at the line spelled `line`. */
function errorAt(
  text: string,
  message: string,
  line: string,
  read: (text: string) => unknown,
): void {
  const error = loadError(() => read(text));
  expect(error.message).toContain(message);
  expect(error.diagnostic.frames.at(-1)!.range.startLine).toBe(text.split("\n").indexOf(line) + 1);
}

/** Load `text` with `loadFlow`, expecting `message` located at the line spelled `line`. */
async function loadErrorAt(text: string, message: string, line: string): Promise<void> {
  const error = await loadErrorAsync(loadFlow(text));
  expect(error.message).toContain(message);
  expect(error.diagnostic.frames.at(-1)!.range.startLine).toBe(text.split("\n").indexOf(line) + 1);
}

const NODE_HEADER = "id: old_forms\nname: old_forms\ninput:\n  n: !int\n";

// ---------- bracket types ----------

test("a bracket typedef shows the parenthesized spelling", async () => {
  const text =
    "id: f\nname: f\ntypedefs:\n  Buy:\n    action: Literal[buy]\n" +
    "input:\n  b: !Buy\nkind: code\ncode: return inputs['b']\noutput: Buy\n";
  await loadErrorAt(text, "types use parentheses: `Literal(buy)`", "    action: Literal[buy]");
});

test.each([
  ["Optional[str]", "Optional(str)"],
  ["int[1, 12]", "int(1..12)"],
  ["dict[str, int]", "object"],
])("a bracket node output %p shows the parenthesized spelling", async (declared, spelled) => {
  const text =
    NODE_HEADER +
    "nodes:\n  step:\n    kind: code\n    code: return null\n" +
    `    output: ${declared}\noutput: \${step.output}\n`;
  await loadErrorAt(text, `types use parentheses: \`${spelled}\``, `    output: ${declared}`);
});

// ---------- `name: T = value` bindings ----------

/** A flow whose `use` node binds `name` to `value`. */
const bindingFlow = (name: string, value: string) => `\
id: old_bindings
name: old_bindings
typedefs:
  Buy: {action: Literal(buy)}
  Sell: {action: Literal(sell)}
input:
  n: !int
nodes:
  size:
    kind: code
    code: "return {'qty': inputs['n']}"
    input:
      n: \${input.n}
    output: {qty: int}
  use:
    kind: code
    code: return inputs['${name}']
    input:
      ${name}: ${value}
    output: str
output: \${use.output}
`;

/** The advice a binding error ends with, for text meant literally. */
const quoteHint = (name: string, text: string) => `. If you meant the text, quote it: \`${name}: ${text}\``;

test.each([
  ["qty", "int = ${size.output.qty}", "`qty: !int ${size.output.qty}`"],
  ["q", "Optional[int] = ${size.output.qty}", "`q: !Optional(int) ${size.output.qty}`"],
  ["d", "Union[Buy, Sell] = ${size.output}", "name `Union(Buy, Sell)` in `typedefs:` and tag with its name"],
  ["s", 'str = "30"', '`s: !str "30"`'],
  ["s", "str = 30", '`s: !str "30"`'],
  ["b", "bool = yes", "`b: !bool true` (`yes` is text; write `true`)"],
  ["n", "int = 5", "`n: !int 5`"],
  ["b", "Buy = ${size.output}", "`b: !Buy ${size.output}`"],
  ["e", "int =", "`e: !int ${...}`"],
  ["t", "int = call(f, x = 1)", "`t: !int call(f, x = 1)`"],
])("the typed binding text %p: %p shows the tag", async (name, value, spelled) => {
  await loadErrorAt(
    bindingFlow(name, value),
    `binding '${name}': write the type as a tag: ${spelled}` +
      quoteHint(name, value.includes('"') ? `'${value}'` : `"${value}"`),
    `      ${name}: ${value}`,
  );
});

test("typed binding text that reads as words offers quoting", async () => {
  await loadErrorAt(
    bindingFlow("note", "date = tomorrow"),
    "binding 'note': write the type as a tag: `note: !date tomorrow`" +
      '. If you meant the text, quote it: `note: "date = tomorrow"`',
    "      note: date = tomorrow",
  );
});

describe("text without a type on the left is plain text", () => {
  const dir = mkdtempSync(join(tmpdir(), "yian-old-forms-"));
  afterAll(() => rmSync(dir, { recursive: true, force: true }));

  test.each([
    ["a = b", "a = b"],
    ['"int = 5"', "int = 5"],
    ["itn = 5", "itn = 5"],
    ["x = ${input.n}", "x = 3"],
  ])("%p", async (value, text) => {
    const result = await runFlow(await loadFlow(bindingFlow("note", value)), { n: 3 }, { env: hostEnv(dir) });

    expect(result.status).toBe("succeeded");
    expect(result.output).toBe(text);
  });
});

// ---------- flow inputs and agent variables ----------

test("an untagged input in a definition shows the tag", async () => {
  const text =
    "id: f\nname: f\ndefs:\n  child:\n    input:\n      qty: int\n" +
    "    kind: code\n    code: return 1\n    output: int\n" +
    "nodes:\n  c:\n    kind: call\n    call: child\n    input:\n      qty: 1\n" +
    "output: ${c.output}\n";
  await loadErrorAt(text, "flow input 'qty': write the type as a tag: `qty: !int`", "      qty: int");
});

const MESSAGE_REGISTRY = readTypedefs({ ChatMsg: { role: "str", content: "str" } });

/** Read `text`'s `input:` declarations against the `ChatMsg` typedef. */
const flowInputs = (text: string) => parametersFromYaml(text, MESSAGE_REGISTRY);

/** Read `text`'s `variables:` declarations as agent variables. */
const agentVariables = (text: string) =>
  parametersFromYaml(text, EMPTY, { field: "variables", requireDefault: true });

describe("untagged declarations", () => {
  test.each([
    ["qty: int", "flow input 'qty': write the type as a tag: `qty: !int`"],
    [
      "history: list[ChatMsg] = []",
      "flow input 'history': write the type as a tag: `history: !list(ChatMsg) []`",
    ],
    ["n: int = 3", "flow input 'n': write the type as a tag: `n: !int 3`"],
    ["o: Optional[str]", "flow input 'o': write the type as a tag: `o: !Optional(str)`"],
    [
      "cfg: {a: int}",
      "flow input 'cfg': name the record in `typedefs:` and tag the input with it, " +
        "e.g. `cfg: !Cfg`",
    ],
    [
      "user_cfg: {a: int}",
      "flow input 'user_cfg': name the record in `typedefs:` and tag the input with " +
        "it, e.g. `user_cfg: !UserCfg`",
    ],
    [
      "x: 5",
      "flow input 'x': an input is a tag with an optional default, e.g. `x: !int` or `x: !int 5`",
    ],
    [
      "x: two words",
      "flow input 'x': an input is a tag with an optional default, " +
        "e.g. `x: !str` or `x: !str two words`",
    ],
    // Text whose leading name is no type and close to none is a value, not a type.
    [
      "topic: hello",
      "flow input 'topic': an input is a tag with an optional default, " +
        "e.g. `topic: !str` or `topic: !str hello`",
    ],
    [
      "topic: hello = world",
      "flow input 'topic': an input is a tag with an optional default, " +
        'e.g. `topic: !str` or `topic: !str "hello = world"`',
    ],
    [
      "x: '30'",
      "flow input 'x': an input is a tag with an optional default, " +
        'e.g. `x: !str` or `x: !str "30"`',
    ],
    // A leading name that is a type keeps its tag even when an argument names no type, so
    // pasting it back reaches the tag reader's unknown-type error.
    [
      "history: list[message] = []",
      "flow input 'history': write the type as a tag: `history: !list(message) []`",
    ],
    // A leading name close to a type name is spelled as that name.
    ["x: strr", "flow input 'x': write the type as a tag: `x: !str`"],
    ["x: Strr = hi", "flow input 'x': write the type as a tag: `x: !str hi`"],
    ["x: Optinal(int)", "flow input 'x': write the type as a tag: `x: !Optional(int)`"],
    ["x: Optional(Foo)", "flow input 'x': write the type as a tag: `x: !Optional(Foo)`"],
    [
      "u: Union[A, B]",
      "flow input 'u': write the type as a tag: " +
        "name `Union(A, B)` in `typedefs:` and tag with its name",
    ],
    ["s: str = 30", "flow input 's': write the type as a tag: `s: !str \"30\"`"],
    [
      "f: bool = no",
      "flow input 'f': write the type as a tag: `f: !bool false` (`no` is text; write `false`)",
    ],
  ])("the untagged flow input %p shows the tag", (declaration, message) => {
    errorAt(`input:\n  ${declaration}\n`, message, `  ${declaration}`, flowInputs);
  });

  test.each([
    ["n: int[1, 12] = 4", "variable 'n': write the type as a tag: `n: !int(1..12) 4`"],
    ["n: int = 4", "variable 'n': write the type as a tag: `n: !int 4`"],
    ["n: int", "variable 'n': write the type as a tag: `n: !int <default>`"],
    ["n: 4", "variable 'n': a variable is a tag with a default, e.g. `n: !int 4`"],
    [
      "tone: concise",
      "variable 'tone': a variable is a tag with a default, e.g. `tone: !str concise`",
    ],
  ])("the untagged agent variable %p shows the tag", (declaration, message) => {
    errorAt(`variables:\n  ${declaration}\n`, message, `  ${declaration}`, agentVariables);
  });

  test.each([
    ["x: 2.5", "2.5"],
    ["x: 2.0", "2.0"],
    ["x: 1.0e-7", "1e-07"],
    ["x: 1e21", "1e+21"],
    ["x: .inf", ".inf"],
    ["x: -.inf", "-.inf"],
  ])("the untagged float default %p is spelled %p", (declaration, spelled) => {
    errorAt(
      `input:\n  ${declaration}\n`,
      "flow input 'x': an input is a tag with an optional default, " +
        `e.g. \`x: !float\` or \`x: !float ${spelled}\``,
      `  ${declaration}`,
      flowInputs,
    );
  });

  test("a default that doesn't parse is shown as written", () => {
    const message = "flow input 'n': write the type as a tag: `n: !int [`";
    errorAt("input:\n  n: int = [\n", message, "  n: int = [", flowInputs);
  });
});

describe("source-free declarations", () => {
  test.each([
    ["readFlowInputs", readFlowInputs],
    ["readVariables", readVariables],
  ] as const)("%s takes a type only", (_name, reader) => {
    expect(loadError(() => reader({ x: "int = 3" }, EMPTY)).message).toContain(
      "source-free declarations take a type only; write defaults in YAML with a tag",
    );
  });

  test("a source-free input rejects a record map", () => {
    expect(loadError(() => readFlowInputs({ x: { a: "int" } }, EMPTY)).message).toContain(
      "input 'x': a source-free declaration is type text",
    );
  });
});

// ---------- typed flow outputs ----------

/** A flow whose `use` node binds `v` to `${input.n}`, with the flow `output:` lines `output`. */
const outputFlow = (output: string) =>
  bindingFlow("v", "${input.n}").replace("output: ${use.output}\n", `${output}\n`);

describe("typed flow outputs", () => {
  test("a typed flow output field's text shows the tag", () => {
    errorAt(
      outputFlow("output:\n  r: int = ${use.output}"),
      "binding 'r': write the type as a tag: `r: !int ${use.output}`" +
        '. If you meant the text, quote it: `r: "int = ${use.output}"`',
      "  r: int = ${use.output}",
      assembleText,
    );
  });

  test("a typed single flow output's text shows the tag", () => {
    errorAt(
      outputFlow("output: Optional[int] = ${use.output}"),
      "binding 'output': write the type as a tag: `output: !Optional(int) ${use.output}`" +
        '. If you meant the text, quote it: `output: "Optional[int] = ${use.output}"`',
      "output: Optional[int] = ${use.output}",
      assembleText,
    );
  });
});

// ---------- harvested conformance cases ----------

const rootInput = inputsAt(["input"]);
const childInput = inputsAt(["defs", "child", "input"], ["def", "child"]);
const OLD_FORMS = "load-core/compose_old_forms__";

test.each<CaseRow>([
  { id: `${OLD_FORMS}typed_flow_output_text_shows_the_tag`, call: assembled },
  { id: `${OLD_FORMS}untagged_agent_variable`, call: assembled },
  { id: `${OLD_FORMS}untagged_agent_variable_value`, call: assembled },
  { id: `${OLD_FORMS}untagged_bool_input_with_a_yaml11_word`, call: rootInput },
  { id: `${OLD_FORMS}untagged_flow_input`, call: rootInput },
  { id: `${OLD_FORMS}untagged_flow_input_value`, call: rootInput },
  { id: `${OLD_FORMS}untagged_record_flow_input`, call: rootInput },
  { id: `${OLD_FORMS}untagged_union_flow_input`, call: rootInput },
  {
    id: `${OLD_FORMS}untagged_input_in_a_definition_shows_the_tag`,
    call: childInput,
    prefix: "defs entry 'child': ",
  },
])("$id", runCase);
