/**
 * The types of a file's dotted reads, a port of
 * `tests/compose/test_file_field_reads.py::test_file_field_type_gives_the_type_of_each_file_field`
 * and `::test_project_output_contract_types_the_fields_of_a_file`.
 *
 * Later: the load-time checks of `uri`, `mime` and `name` reads (with the loader) and the
 * flow runs (with the runner).
 */

import { expect, test } from "bun:test";

import { projectOutputContract } from "../../src/compose/helpers/output_contracts.ts";
import { ValueContract } from "../../src/nodes/contracts.ts";
import { fileFieldType, Type, typeEquals, ValueKind } from "../../src/typesys/values.ts";

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

  /** The projected type of `fields`, or `null` when the path doesn't project. */
  const projected = (...fields: string[]): Type | null => {
    const result = projectOutputContract(contract, fields);
    return result === null ? null : (result as ValueContract).type;
  };

  expect(typeEquals(projected("f", "uri")!, STR)).toBe(true);
  expect(typeEquals(projected("f", "mime")!, OPTIONAL_STR)).toBe(true);
  expect(typeEquals(projected("g", "uri")!, OPTIONAL_STR)).toBe(true);
  expect(projected("f", "urii")).toBe(null);
});
