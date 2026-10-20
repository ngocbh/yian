/**
 * The types of a file's dotted reads, a port of
 * `tests/compose/test_file_field_reads.py::test_file_field_type_gives_the_type_of_each_file_field`
 * and `::test_project_output_contract_types_the_fields_of_a_file`.
 *
 * Through `assembleText`:
 * `test_another_field_of_a_file_in_an_inline_body_is_a_located_load_error`, pinning the whole
 * message, and its case
 * `load-core/compose_file_field_reads__another_field_of_a_file_in_an_inline_body`; and the case
 * `load-core/compose_file_field_reads__the_uri_of_a_file_is_str`, which the typed-binding check
 * decides.
 *
 * Later: the other load-time checks of `uri`, `mime` and `name` reads (with the loader) and the
 * flow runs (with the runner).
 */

import { expect, test } from "bun:test";

import { projectOutputContract } from "../../src/compose/helpers/output_contracts.ts";
import { isExactly, ValueContract } from "../../src/nodes/contracts.ts";
import { fileFieldType, Type, typeEquals, ValueKind } from "../../src/typesys/values.ts";
import { assembled, assembleText, type CaseRow, loadError, runCase } from "./declarations.ts";

const STR = Type.scalar(ValueKind.STRING);
const OPTIONAL_STR = new Type({ kind: ValueKind.STRING, nullable: true });

test.each<[string, Type | null]>([
  ["uri", STR],
  ["mime", OPTIONAL_STR],
  ["name", OPTIONAL_STR],
  ["urii", null],
  ["__class__", null],
  ["", null],
  // TS-only: a file has no inherited JS property as a field.
  ["constructor", null],
  ["__proto__", null],
])("fileFieldType gives the type of each file field: %p", (name, expected) => {
  expect(fileFieldType(name)).toEqual(expected);
});

test("projectOutputContract types the fields of a file", () => {
  const record = new Type({
    kind: ValueKind.OBJECT,
    fields: new Map([
      ["f", Type.scalar(ValueKind.FILE)],
      ["g", new Type({ kind: ValueKind.FILE, nullable: true })],
    ]),
    required: new Set(["f", "g"]),
  });
  const contract = new ValueContract(record);

  /** The projected type of `fields`, requiring the path to project to an exact `ValueContract`. */
  const projected = (...fields: string[]): Type => {
    const result = projectOutputContract(contract, fields);
    expect(isExactly(result, ValueContract)).toBe(true);
    return (result as ValueContract).type;
  };

  expect(typeEquals(projected("f", "uri"), STR)).toBe(true);
  expect(typeEquals(projected("f", "mime"), OPTIONAL_STR)).toBe(true);
  expect(typeEquals(projected("g", "uri"), OPTIONAL_STR)).toBe(true);
  expect(projectOutputContract(contract, ["f", "urii"])).toBe(null);
});

// ---------- inline bodies ----------

/** A flow whose inline CALL `b` binds `s` to `binding`, over a file input `f`. */
function inlineCall(binding: string): string {
  return (
    "id: p\nname: p\ninput:\n  f: !file\nnodes:\n" +
    `  b:\n    kind: call\n    input:\n      s: ${binding}\n    nodes:\n` +
    '      k: {kind: code, input: {s: "${input.s}"}, code: "return inputs[\'s\']"}\n' +
    "    output: ${k.output}\noutput: ${b.output}\n"
  );
}

/** A flow whose inline MAP `b` over `xs` (typed `over`) binds `s` to `binding`. */
function inlineMap(over: string, binding: string): string {
  return (
    `id: p\nname: p\ntypedefs:\n  Item: {f: file}\ninput:\n  xs: !${over}\nnodes:\n` +
    `  b:\n    kind: map\n    over: \${input.xs}\n    input:\n      s: ${binding}\n` +
    "    nodes:\n" +
    '      k: {kind: code, input: {s: "${input.s}"}, code: "return inputs[\'s\']"}\n' +
    "    output: ${k.output}\noutput: ${b.output}\n"
  );
}

test.each([
  ["call-input", inlineCall("${input.f.urii}"), "${input.f.urii}"],
  ["map-record-item", inlineMap("list(Item)", "${item.f.urii}"), "${item.f.urii}"],
  ["map-file-item", inlineMap("list(file)", "${item.urii}"), "${item.urii}"],
])("another field of a file in an inline body is a located load error: %s", (_name, text, reference) => {
  const error = loadError(() => assembleText(text));

  expect(error.message).toBe(
    `cannot infer type for inline binding 'b.input.s': reference ${reference} reads field 'urii' on a ` +
      "file, which has only uri, mime, and name; declare its type with a tag (`s: !<type> ...`) or use an " +
      "explicitly typed named `defs:` body",
  );
  expect(error.diagnostic.frames.at(-1)!.range.startLine).toBe(
    text.split("\n").findIndex((line) => line.includes("s: ${")) + 1,
  );
});

test.each<CaseRow>([
  { id: "load-core/compose_file_field_reads__another_field_of_a_file_in_an_inline_body", call: assembled },
  { id: "load-core/compose_file_field_reads__the_uri_of_a_file_is_str", call: assembled },
])("assemble: $id", runCase);
