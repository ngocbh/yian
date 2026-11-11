/**
 * Reading one Compose type declaration into a `Type` through `readType`, and the declared types
 * of the conformance cases that the type readers and load validators reject.
 */

import { afterEach, beforeEach, describe, expect, test } from "bun:test";

import { LoadDiagnostics } from "../../src/compose/helpers/errors.ts";
import { readTaggedValue } from "../../src/compose/helpers/tags.ts";
import { readSourceParameters, readType, readValueContract } from "../../src/compose/helpers/types.ts";
import { encodeSitePath, SourceCatalogBuilder } from "../../src/compose/source/catalog.ts";
import { SourceDocument } from "../../src/compose/source/index.ts";
import { BUILTIN, forget, install, reload } from "../../src/defaults.ts";
import { OriginRef, SourceRange } from "../../src/diagnostics.ts";
import type { ValueContract } from "../../src/nodes/contracts.ts";
import { readTypedefs, type TypeRegistry } from "../../src/typesys/types.ts";
import { canonicalTypeLabel, type Type, ValueKind } from "../../src/typesys/values.ts";
import {
  assembled,
  type CaseCall,
  caseDocument,
  caseRegistry,
  type CaseRow,
  cursorAt,
  inputsAt,
  loadError,
  runCase,
} from "../helpers/declarations.ts";
import type { Case } from "../conformance/runner.ts";

const EMPTY: TypeRegistry = new Map();

/** The load error for a type written as a YAML list. */
const LIST_TYPE =
  "a type can't be a YAML list: write a record as one mapping of its fields, as in {a: int, b: str}, " +
  "and a list type as list(T)";

const LIMIT_TEXT = "the limit for a type (each record and list is a level)";

// Start each test from the built-in values, whatever `YIAN_*` variables the shell sets.
beforeEach(() => install(BUILTIN));
afterEach(forget);

/** Return the field `name` of record type `typ`. */
function field(typ: Type, name: string): Type {
  return typ.fields!.get(name)!;
}

describe("readType", () => {
  test("scalar", () => {
    const sh = readType("float", EMPTY);
    expect(sh.kind).toBe(ValueKind.NUMBER);
    expect(sh.fields).toBeUndefined();
    expect(sh.element).toBeUndefined();
  });

  test("list", () => {
    const sh = readType("list(str)", EMPTY);
    expect(sh.kind).toBe(ValueKind.LIST_STRING);
    expect(sh.element?.kind).toBe(ValueKind.STRING);
  });

  test("flat map", () => {
    const sh = readType({ rating: "float", rationale: "str" }, EMPTY);
    expect(sh.kind).toBe(ValueKind.OBJECT);
    expect(field(sh, "rating").kind).toBe(ValueKind.NUMBER);
    expect(field(sh, "rationale").kind).toBe(ValueKind.STRING);
    expect(sh.required).toEqual(new Set(["rating", "rationale"]));
  });

  test("nested map", () => {
    const sh = readType({ summary: { count: "int", meta: { as_of: "date" } } }, EMPTY);
    expect(sh.kind).toBe(ValueKind.OBJECT);
    const summary = field(sh, "summary");
    expect(summary.kind).toBe(ValueKind.OBJECT);
    expect(field(summary, "count").kind).toBe(ValueKind.INTEGER);
    const meta = field(summary, "meta");
    expect(meta.kind).toBe(ValueKind.OBJECT);
    expect(field(meta, "as_of").kind).toBe(ValueKind.DATE);
  });

  test("a list of single-key maps is refused with the mapping to write", () => {
    expect(loadError(() => readType([{ decision: "str" }, { why: "str" }], EMPTY)).message).toBe(LIST_TYPE);
  });

  test("optional field excluded from required", () => {
    const sh = readType({ score: "float", note: "Optional(str)" }, EMPTY);
    expect(field(sh, "note").nullable).toBe(true);
    expect(field(sh, "score").nullable).toBe(false);
    expect(sh.required).toEqual(new Set(["score"]));
  });

  test("registry name", () => {
    const registry = readTypedefs({ Rating: { category: "str", score: "float" } });
    const sh = readType("Rating", registry);
    expect(sh.kind).toBe(ValueKind.OBJECT);
    expect(field(sh, "category").kind).toBe(ValueKind.STRING);
    expect(field(sh, "score").kind).toBe(ValueKind.NUMBER);
  });

  test("malformed leaf raises", () => {
    expect(loadError(() => readType("list(", EMPTY)).message).toContain("list(");
  });
});

describe("type nesting", () => {
  test("a record nesting past a configured limit is refused at its path", () => {
    reload((name) => (name === "YIAN_MAX_TYPE_NESTING" ? "2" : undefined));
    expect(loadError(() => readType({ a: { b: { c: "int" } } }, EMPTY)).message).toBe(
      `type at a.b nests deeper than 2 levels, ${LIMIT_TEXT}`,
    );
    expect(loadError(() => readType({ a: { b: "list(int)" } }, EMPTY)).message).toBe(
      `type at a.b nests deeper than 2 levels, ${LIMIT_TEXT}; its type 'list(int)' adds 1`,
    );
    expect(readType({ a: { b: "int" } }, EMPTY).kind).toBe(ValueKind.OBJECT);
  });

  test("a record is measured before its entries are read, however aliases built it", () => {
    reload((name) => (name === "YIAN_MAX_TYPE_NESTING" ? "3" : undefined));
    // A record holding itself would recurse forever if its entries were read first.
    const cyclic = SourceDocument.parse("x: &a {k: *a}\n", "flow.yaml").root.child("x").value;
    expect(loadError(() => readType(cyclic, EMPTY)).message).toBe(
      `type at k.k.k nests deeper than 3 levels, ${LIMIT_TEXT}`,
    );

    const text = "a: [&a0 {k: int}, &a1 {k: *a0}, &a2 {k: *a1}, &a3 {k: *a2}]\nx: *a3\n";
    const document = SourceDocument.parse(text, "flow.yaml");
    const error = loadError(() =>
      readValueContract(document.root.child("x"), EMPTY, {
        sitePath: ["x"],
        catalogBuilder: new SourceCatalogBuilder(),
      }),
    );
    expect(error.message).toBe(`type at k.k.k nests deeper than 3 levels, ${LIMIT_TEXT}`);
    expect(error.diagnostic.frames.at(-1)!.range).toEqual(new SourceRange(1, 5, 1, 17));
  });
});

describe("record field names", () => {
  const EMPTY_NAME = "record field names must be non-empty strings";
  const SURROGATE_NAME = "record field name '\\ud800': field names must not contain a Unicode surrogate";

  test("an empty or surrogate name is refused without a source, unranged", () => {
    const empty = loadError(() => readType({ "": "int" }, EMPTY));
    expect(empty.message).toBe(EMPTY_NAME);
    expect(empty.diagnostic.frames).toEqual([]);
    const surrogate = loadError(() => readType({ "\uD800": "int" }, EMPTY));
    expect(surrogate.message).toBe(SURROGATE_NAME);
    expect(surrogate.diagnostic.frames).toEqual([]);
  });

  test.each([
    ['x:\n  "": int\n', EMPTY_NAME, new SourceRange(2, 3, 2, 5)],
    ['x:\n  "\\uD800": int\n', SURROGATE_NAME, new SourceRange(2, 3, 2, 11)],
  ])("an empty or surrogate name in %p is refused at its key", (text, message, range) => {
    const document = SourceDocument.parse(text, "flow.yaml");
    const error = loadError(() =>
      readValueContract(document.root.child("x"), EMPTY, {
        sitePath: ["x"],
        catalogBuilder: new SourceCatalogBuilder(),
      }),
    );
    expect(error.message).toBe(message);
    expect(error.diagnostic.frames.at(-1)!.range).toEqual(range);
  });
});

test("any YAML list is refused as a type", () => {
  for (const node of [[{ a: "int", b: "str" }], ["str"], []]) {
    expect(loadError(() => readType(node, EMPTY)).message).toBe(LIST_TYPE);
  }
});

test("readValueContract registers the root and each authored field in canonical order", () => {
  const text = "output:\n  b: int\n  a:\n    y: str\n    x: int\n";
  const document = SourceDocument.parse(text, "flow.yaml");
  const catalogBuilder = new SourceCatalogBuilder();
  const prefix = ["node", "n", "output", "contract"];
  const contract = readValueContract(document.root.child("output"), EMPTY, {
    sitePath: prefix,
    catalogBuilder,
  });
  expect([...contract.type.fields!.keys()]).toEqual(["b", "a"]);

  expect(contract.origins.default).toEqual(new OriginRef(encodeSitePath([...prefix, "root"])));
  expect(contract.origins.entries.map(([path]) => path)).toEqual([["a"], ["a", "x"], ["a", "y"], ["b"]]);
  for (const [path, origin] of contract.origins.entries) {
    expect(origin).toEqual(new OriginRef(encodeSitePath([...prefix, "path", ...path])));
  }

  const ranges = [contract.origins.default!, ...contract.origins.entries.map(([, origin]) => origin)].map(
    (origin) => catalogBuilder.resolveRegistered(origin)!.range,
  );
  expect(ranges).toEqual([
    new SourceRange(2, 3, 6, 1),
    new SourceRange(4, 5, 6, 1),
    new SourceRange(5, 8, 5, 11),
    new SourceRange(4, 8, 4, 11),
    new SourceRange(2, 6, 2, 9),
  ]);
});

// ---------- harvested conformance cases ----------

/** Read the `output:` declaration at `path` as the loader reads node `nodeId`'s output. */
function output(path: readonly string[], nodeId: string): CaseCall {
  return (document, registry, diagnostics, builder) =>
    readValueContract(cursorAt(document, path), registry, {
      sitePath: ["node", nodeId, "output", "contract"],
      catalogBuilder: builder,
      diagnostics,
    });
}

const RECORD_NESTING = "types/compose_record_type_nesting__";
const UNION_TYPEDEFS = "types/engine_union_typedefs__";
const OUTPUT_ORIGINS = "load-core/compose_node_output_contract_origins__";
const compact = output(["output"], "compact");
const use = output(["nodes", "use", "output"], "use");
const emit = output(["nodes", "emit", "output"], "emit");

test.each<CaseRow>([
  { id: `${RECORD_NESTING}a_typedef_name_counts_as_deep_as_its_definition`, call: use },
  { id: `${RECORD_NESTING}record_past_a_configured_limit`, call: use },
  { id: `${RECORD_NESTING}record_at_a_configured_limit_loads`, call: use },
  { id: `${RECORD_NESTING}list_typedef_at_the_limit_loads`, call: use },
  { id: "types/compose_type_name_checks__surrogate_literal_member_in_an_output", call: compact },
  { id: "types/engine_typedef_nesting__type_wrapping_a_deep_typedef", call: compact },
  { id: "types/engine_types__type_text_brackets_hint", call: compact },
  { id: `${UNION_TYPEDEFS}bar_operator_hints_union`, call: compact },
  { id: `${UNION_TYPEDEFS}member_must_be_a_record`, call: compact },
  { id: `${UNION_TYPEDEFS}none_member_hints_optional`, call: compact },
  { id: `${UNION_TYPEDEFS}nullable_member_rejected`, call: compact },
  { id: "load-types/seed_e13_bad_type_expression", call: output(["nodes", "calc", "output"], "calc") },
  { id: `${OUTPUT_ORIGINS}compact_unbalanced_output_type`, call: compact },
  { id: `${OUTPUT_ORIGINS}invalid_output_list_item`, call: emit },
  { id: `${OUTPUT_ORIGINS}non_text_nested_output_type`, call: emit },
  { id: `${OUTPUT_ORIGINS}unknown_nested_output_type`, call: emit },
  { id: `${UNION_TYPEDEFS}load_flow_accepts_union_typedef`, call: inputsAt(["input"]) },
])("$id", runCase);

/** Return the label of input `name`'s type in the `input:` declarations at `path`. */
function inputLabel(document: SourceDocument, registry: TypeRegistry, path: string[], name: string): string {
  const parameters = readSourceParameters(cursorAt(document, path), registry, {
    sitePath: [],
    catalogBuilder: new SourceCatalogBuilder(),
  });
  return canonicalTypeLabel(parameters.find((parameter) => parameter.name === name)!.contract!.type);
}

/** Return the label of node `nodeId`'s declared `output:` type. */
function outputLabel(document: SourceDocument, registry: TypeRegistry, nodeId: string): string {
  const read = output(["nodes", nodeId, "output"], nodeId);
  const diagnostics = LoadDiagnostics.fromDocument(document);
  const contract = read(document, registry, diagnostics, new SourceCatalogBuilder()) as ValueContract;
  return canonicalTypeLabel(contract.type);
}

/** Return the label of the type tagged on the slot at `path`. */
function taggedLabel(document: SourceDocument, registry: TypeRegistry, path: string[]): string {
  const catalogBuilder = new SourceCatalogBuilder();
  const slot = readTaggedValue(cursorAt(document, path), registry, {
    label: path.at(-1)!,
    sitePath: path,
    catalogBuilder,
    diagnostics: LoadDiagnostics.fromDocument(document),
    ledger: catalogBuilder.consumedTags,
  });
  return canonicalTypeLabel(slot.contract!.type);
}

const OPEN_RANGE = "types/compose_open_range_flows__";

test.each<{ id: string; fragment: (document: SourceDocument, registry: TypeRegistry) => string }>([
  {
    id: `${OPEN_RANGE}call_with_a_wider_open_range_source`,
    fragment: (document, registry) =>
      `child expects '${inputLabel(document, registry, ["defs", "child", "input"], "t")}', ` +
      `source is '${outputLabel(document, registry, "a")}'`,
  },
  {
    id: `${OPEN_RANGE}case_when_outside_an_open_range`,
    fragment: (document, registry) => `(\`${inputLabel(document, registry, ["input"], "n")}\`)`,
  },
  {
    id: `${OPEN_RANGE}tagged_binding_disjoint_from_an_open_range`,
    fragment: (document, registry) =>
      `declared ${taggedLabel(document, registry, ["nodes", "use", "input", "x"])}, ` +
      `but the value is ${outputLabel(document, registry, "a")}`,
  },
])("the compared types of $id", ({ id, fragment }) => {
  const { case_, document } = caseDocument(id);
  const registry = caseRegistry(document);
  const wanted = (((case_["expect"] as Case)["load"] as Case)["diagnostics"] as Case[])[0]!;
  expect(wanted["message"] as string).toContain(fragment(document, registry));
});

test(`assembled: ${OPEN_RANGE}case_when_outside_an_open_range`, () => {
  runCase({ id: `${OPEN_RANGE}case_when_outside_an_open_range`, call: assembled });
});
