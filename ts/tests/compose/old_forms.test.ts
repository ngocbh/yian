/**
 * Each old type form of a flow input or agent variable is a located load error that shows the
 * tag spelling: a port of the declaration tests of `tests/compose/test_old_forms.py`, reading
 * the declarations with `parametersFromYaml` (`readSourceParameters`) where Python loads a flow.
 *
 * Ported: `test_untagged_flow_input_shows_the_tag`, `test_untagged_agent_variable_shows_the_tag`,
 * `test_source_free_declarations_take_a_type_only` and
 * `test_source_free_input_rejects_a_record_map`.
 *
 * TS-only: an untagged float default's example, spelled as Python's `repr` spells a finite
 * float and as `.inf` for infinity, which YAML reads back as a float (Python writes `inf`,
 * which reads back as text); a type with a default that doesn't parse (`n: int = [`), kept as
 * written where Python reads an unbound name; and the harvested-case table, which makes the
 * loader's call on the declarations of the conformance cases these readers decide, comparing
 * the message and the last frame.
 *
 * Later (T3-22): the `load_flow` tests: the bracket typedef and node output, the typed binding
 * text, the typed flow output text, text without a type that stays plain text, and the
 * untagged definition input.
 */

import { afterEach, beforeEach, describe, expect, test } from "bun:test";

import { LoadDiagnostics, LoadError } from "../../src/compose/helpers/errors.ts";
import {
  readFlowInputs,
  readSourceFlowInputs,
  readSourceParameters,
  readVariables,
} from "../../src/compose/helpers/types.ts";
import { SourceCatalogBuilder } from "../../src/compose/source/catalog.ts";
import type { SourceCursor, SourceDocument } from "../../src/compose/source/index.ts";
import { BUILTIN, forget, install } from "../../src/defaults.ts";
import { readTypedefs, type TypeRegistry } from "../../src/typesys/types.ts";
import type { Case } from "../conformance.ts";
import { caseDocument, parametersFromYaml } from "./declarations.ts";

// Pinned before `MESSAGE_REGISTRY` is read too, so neither depends on the shell's `YIAN_*` limits.
install(BUILTIN);
beforeEach(() => install(BUILTIN));
afterEach(forget);

const EMPTY: TypeRegistry = new Map();

/** Return the `LoadError` `run` throws. */
function loadError(run: () => unknown): LoadError {
  try {
    run();
  } catch (error) {
    if (error instanceof LoadError) return error;
    throw error;
  }
  throw new Error("expected a LoadError");
}

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

// ---------- flow inputs and agent variables ----------

const MESSAGE_REGISTRY = readTypedefs({ Message: { role: "str", content: "str" } });

/** Read `text`'s `input:` declarations against the `Message` typedef. */
const flowInputs = (text: string) => parametersFromYaml(text, MESSAGE_REGISTRY);

/** Read `text`'s `variables:` declarations as agent variables. */
const agentVariables = (text: string) =>
  parametersFromYaml(text, EMPTY, { field: "variables", requireDefault: true });

describe("untagged declarations", () => {
  test.each([
    ["qty: int", "flow input 'qty': write the type as a tag: `qty: !int`"],
    [
      "history: list[Message] = []",
      "flow input 'history': write the type as a tag: `history: !list(Message) []`",
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

// ---------- harvested conformance cases ----------

/** The call the loader makes on a case's document. */
type Call = (document: SourceDocument, registry: TypeRegistry, diagnostics: LoadDiagnostics) => unknown;

interface Row {
  /** The case, `group/name`. */
  id: string;
  call: Call;
  /** The loader's prefix on the expected message. */
  prefix?: string;
}

/** Return the cursor at `path` from the document root. */
function at(document: SourceDocument, path: string[]): SourceCursor {
  return path.reduce((cursor: SourceCursor, step) => cursor.child(step), document.root);
}

/** Read the `input:` declarations at `path` as the loader reads `owner`'s (the root's: `[]`). */
const declared =
  (path: string[], owner: string[] = []): Call =>
  (document, registry, diagnostics) =>
    readSourceFlowInputs(at(document, path), registry, {
      sitePath: owner,
      catalogBuilder: new SourceCatalogBuilder(),
      diagnostics,
    });

/** Read the agent variables at `path` as the loader reads them, under `sitePath`. */
const variables =
  (path: string[], sitePath: string[]): Call =>
  (document, registry, diagnostics) =>
    readSourceParameters(at(document, path), registry, {
      sitePath,
      catalogBuilder: new SourceCatalogBuilder(),
      diagnostics,
      fieldName: "variables",
      declarationLabel: "variable",
      requireDefault: true,
    });

const rootInput = declared(["input"]);
const childInput = declared(["defs", "child", "input"], ["def", "child"]);
const askVariables = variables(["nodes", "ask", "variables"], ["node", "ask"]);

/** Make the row's call on its case and compare the result with the case's `expect`. */
function runRow(row: Row): void {
  const { case_, document } = caseDocument(row.id);
  const root = document.root;
  const registry = root.has("typedefs") ? readTypedefs(root.mapping("typedefs")) : new Map();
  const diagnostics = LoadDiagnostics.fromDocument(document);
  const expected = (case_["expect"] as Case)["load"];
  if (expected === "ok") {
    row.call(document, registry, diagnostics);
    return;
  }
  const error = loadError(() => row.call(document, registry, diagnostics));
  const frame = error.diagnostic.frames.at(-1)!;
  const range = frame.range;
  const diagnostic = ((expected as Case)["diagnostics"] as Case[])[0]!;
  const expectedFrame = (diagnostic["frames"] as Case[]).at(-1)!;
  const message = diagnostic["message"] as string;
  const prefix = row.prefix ?? "";
  expect(message.startsWith(prefix)).toBe(true);
  expect([
    error.message,
    frame.label,
    [range.startLine, range.startCol, range.endLine, range.endCol],
  ]).toEqual([
    message.slice(prefix.length),
    expectedFrame["label"] as string,
    (expectedFrame["range"] as bigint[]).map(Number),
  ]);
}

const OLD_FORMS = "load-core/compose_old_forms__";

test.each<Row>([
  { id: `${OLD_FORMS}untagged_agent_variable`, call: askVariables },
  { id: `${OLD_FORMS}untagged_agent_variable_value`, call: askVariables },
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
])("$id", runRow);
