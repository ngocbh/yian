/**
 * YAML 1.1 boolean spellings, which YAML 1.2 core rules read as text, and their fix; and what
 * YAML read a value as, a port of `tests/compose/test_yaml_words.py`.
 *
 * Not ported: the date, datetime and binary `yamlReadAs` rows; the loader keeps dates as text
 * and builds no bytes.
 */

import { describe, expect, test } from "bun:test";

import { formerBooleanFix, yamlReadAs } from "../../src/compose/helpers/yaml_words.ts";
import { type TypeRegistry, typeFor } from "../../src/typesys/index.ts";

describe("formerBooleanFix", () => {
  test.each([
    ["yes", [true, "`yes` is text; write `true`"]],
    ["Yes", [true, "`Yes` is text; write `true`"]],
    ["YES", [true, "`YES` is text; write `true`"]],
    ["on", [true, "`on` is text; write `true`"]],
    ["ON", [true, "`ON` is text; write `true`"]],
    ["y", [true, "`y` is text; write `true`"]],
    ["no", [false, "`no` is text; write `false`"]],
    ["N", [false, "`N` is text; write `false`"]],
    ["Off", [false, "`Off` is text; write `false`"]],
    [" off ", [false, "`off` is text; write `false`"]],
  ])("maps the former boolean spelling %p to its boolean and fix", (value, expected) => {
    expect(formerBooleanFix(value)).toEqual(expected as [boolean, string]);
  });

  test.each(["true", "false", "ture", "", "yes please", true, 1n, null])(
    "has no fix for %p",
    (value) => {
      expect(formerBooleanFix(value)).toBeNull();
    },
  );

  test.each([
    ["bool", [true, "`yes` is text; write `true`"]],
    ["Optional(bool)", [true, "`yes` is text; write `true`"]],
    ["int", null],
    ["Literal(a, b)", null],
  ])("offers the fix only when the boolean fits %p", (typeText, expected) => {
    const declared = typeFor(typeText, new Map());
    expect(formerBooleanFix("yes", declared)).toEqual(expected as [boolean, string] | null);
  });

  const registry: TypeRegistry = new Map([
    [
      "Rec",
      {
        kind: "record",
        fields: new Map([
          ["ok", "bool"],
          ["name", "str"],
        ]),
      },
    ],
  ]);

  test.each([
    ["list(bool)", ["yes", "no"], [true, "`yes` is text; write `true`"]],
    ["list(bool)", [true, "Off"], [false, "`Off` is text; write `false`"]],
    ["Rec", { ok: "on", name: "yes" }, [true, "`on` is text; write `true`"]],
    ["list(str)", ["yes"], null],
    ["list(bool)", ["maybe"], null],
    ["list(bool)", [1n], null],
    ["Rec", { ok: true, name: 3n }, null],
  ])("finds the rejected word inside a %p", (typeText, value, expected) => {
    const declared = typeFor(typeText, registry);
    expect(formerBooleanFix(value, declared)).toEqual(expected as [boolean, string] | null);
  });
});

describe("yamlReadAs", () => {
  class Foo {}

  test.each([
    [true, "a boolean"],
    [5n, "an integer"],
    [1.5, "a float"],
    [5, "a float"],
    [null, "null"],
    [[1n], "a list"],
    [{ x: 1n }, "a mapping"],
    [new Foo(), "a `Foo`"],
  ])("names %p as %p", (value, phrase) => {
    expect(yamlReadAs(value)).toBe(phrase);
  });
});
