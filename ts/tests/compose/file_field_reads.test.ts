/**
 * The types of a file's dotted reads, a port of
 * `tests/compose/test_file_field_reads.py::test_file_field_type_gives_the_type_of_each_file_field`.
 *
 * Later: the load-time checks of `uri`, `mime` and `name` reads (with the loader) and the
 * flow runs (with the runner).
 */

import { expect, test } from "bun:test";

import { fileFieldType, Type, ValueKind } from "../../src/typesys/values.ts";

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
