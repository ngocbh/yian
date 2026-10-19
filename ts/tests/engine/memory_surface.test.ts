/**
 * The `long_term_memory:` declaration parser: a port of the parser half of
 * `tests/engine/test_memory_surface.py`.
 *
 * Ported: `test_sugar_is_one_memory_named_after_its_type`,
 * `test_mapping_and_long_form_keep_authored_order` and the rows of
 * `test_malformed_values_are_load_errors` (ints written as bigints). The messages of the
 * harvested `load-agent/engine_memory_surface__empty_flow_declaration` and
 * `load-agent/engine_memory_authoring__{reserved_memory_name,config_not_a_mapping,empty_memory_name}`
 * are pinned.
 *
 * Changed: unknown long-form keys are listed with `writeAsText` over the names in code-unit
 * order (`unknown key(s) ["extra"]`), where Python writes a sorted list of reprs.
 *
 * Not ported: the `{3: "episodic"}` row (a record key is always text).
 *
 * Later (T3-22): `test_root_flow_declarations_are_parsed_and_stamped`,
 * `test_null_flow_declaration_is_the_same_as_absent`,
 * `test_flow_shape_errors_point_at_the_value`, `test_def_flow_declarations_carry_the_def_owner`,
 * `test_agent_declarations_are_parsed_and_stamped_with_the_node`,
 * `test_agent_null_declaration_is_absent_and_empty_mapping_fails` and
 * `test_resolver_errors_become_located_load_errors` need the loader.
 */

import { describe, expect, test } from "bun:test";

import { LoadDiagnostics, LoadError } from "../../src/compose/helpers/errors.ts";
import { parseMemoryDeclarations } from "../../src/compose/helpers/memory.ts";
import { SourceDocument } from "../../src/compose/source/index.ts";
import { DeclSite, type MemoryDecl } from "../../src/memory/contracts.ts";
import type { Value } from "../../src/typesys/values.ts";

const SITE = new DeclSite([], ["root"], null);

/** Parse `value` with an unlocated context, no range and `SITE`. */
function parse(value: Value): readonly MemoryDecl[] {
  return parseMemoryDeclarations(value, {
    diagnostics: new LoadDiagnostics(),
    sourceRange: null,
    site: SITE,
  });
}

/** Return each declaration's name, type, config and site. */
function shape(decls: readonly MemoryDecl[]): unknown[] {
  return decls.map((decl) => [decl.name, decl.type, decl.config, decl.site]);
}

/** Return the `LoadError` parsing `value` throws. */
function loadError(value: Value): LoadError {
  try {
    parse(value);
  } catch (error) {
    expect(error).toBeInstanceOf(LoadError);
    return error as LoadError;
  }
  throw new Error("expected a LoadError");
}

describe("parseMemoryDeclarations", () => {
  test("sugar is one memory named after its type", () => {
    const decls = parse("episodic");
    expect(shape(decls)).toEqual([["episodic", "episodic", {}, SITE]]);
    expect(decls[0]!.site).toBe(SITE);
    expect(Object.isFrozen(decls)).toBe(true);
  });

  test("mapping and long form keep authored order", () => {
    const value = {
      lessons: "reflective",
      notes: { type: "episodic", config: { max_episodes: 50n } },
      plain: { type: "vector" },
    };
    expect(shape(parse(value))).toEqual([
      ["lessons", "reflective", {}, SITE],
      ["notes", "episodic", { max_episodes: 50n }, SITE],
      ["plain", "vector", {}, SITE],
    ]);
    expect(parse(value).every((decl) => decl.origin === null)).toBe(true);
  });

  test.each<[Value, string]>([
    [{}, "declares no memory"],
    [3n, "type name or a mapping"],
    [["episodic"], "type name or a mapping"],
    [{ name: "reflective", config: { max_items: 3n } }, "reserved"],
    [{ name: "reflective" }, "reserved"],
    [{ type: "episodic" }, "reserved"],
    [{ notes: { type: "episodic", extra: 1n } }, "unknown key"],
    [{ notes: { config: {} } }, "requires 'type'"],
    [{ notes: { type: 3n } }, "type must be a string"],
    [{ notes: { type: "episodic", config: [] } }, "config must be a mapping"],
    [{ notes: 3n }, "type name or a mapping"],
    [{ "9notes": "episodic" }, "memory name"],
    [{ notes: "my.type" }, "memory type"],
    ["", "memory name"],
    ["name", "reserved"],
    ["type", "reserved"],
    ["config", "reserved"],
    [{ notes: { type: "episodic", config: { path: "x.sqlite" } } }, "is not allowed"],
    [{ notes: { type: "episodic", config: { max_items: 3n } } }, "unknown key"],
  ])("malformed row %# is a load error", (value, message) => {
    const { diagnostic } = loadError(value);
    expect(diagnostic.code).toBe("compose.load");
    expect(diagnostic.message).toContain(message);
  });

  test.each<[Value, string]>([
    [{}, "long_term_memory: {} declares no memory; remove the key or name one"],
    [{ name: "reflective" }, "long_term_memory name 'name' is reserved (reserved: config, name, type)"],
    [{ notes: { type: "episodic", config: [] } }, "long_term_memory 'notes' config must be a mapping"],
    ["", "long_term_memory '': memory name '' must match [A-Za-z_][A-Za-z0-9_-]*"],
    [
      { notes: { type: "episodic", extra: 1n } },
      'long_term_memory \'notes\' has unknown key(s) ["extra"] (allowed: type, config)',
    ],
    [3n, "long_term_memory must be a type name or a mapping of memory names to types"],
    [
      { notes: 3n },
      "long_term_memory 'notes' must be a type name or a mapping with 'type' and optional 'config'",
    ],
    [{ notes: { config: {} } }, "long_term_memory 'notes' requires 'type'"],
    [{ notes: { type: 3n } }, "long_term_memory 'notes' type must be a string"],
  ])("pinned message row %#", (value, message) => {
    expect(loadError(value).diagnostic.message).toBe(message);
  });

  test("unknown keys are listed in code-unit order", () => {
    const value = { notes: { type: "episodic", b: 1n, Z: 1n, a: 1n } };
    expect(loadError(value).diagnostic.message).toBe(
      'long_term_memory \'notes\' has unknown key(s) ["Z","a","b"] (allowed: type, config)',
    );
  });

  test("a wrapped declaration error keeps the MemoryDecl error as its cause", () => {
    const error = loadError({ notes: "my.type" });
    expect(error.cause).toBeInstanceOf(Error);
    expect(error.cause).not.toBeInstanceOf(LoadError);
    expect(error.diagnostic.message).toBe(`long_term_memory 'notes': ${(error.cause as Error).message}`);
  });

  test("a located error has the value's range as its one frame", () => {
    const document = SourceDocument.parse("long_term_memory: {}\n");
    const sourceRange = document.root.rangeOf("long_term_memory");
    expect(sourceRange).not.toBeNull();
    let caught: unknown = null;
    try {
      parseMemoryDeclarations(document.root.get("long_term_memory"), {
        diagnostics: LoadDiagnostics.fromDocument(document),
        sourceRange,
        site: SITE,
      });
    } catch (error) {
      caught = error;
    }
    expect(caught).toBeInstanceOf(LoadError);
    const { frames } = (caught as LoadError).diagnostic;
    expect(frames).toHaveLength(1);
    expect(frames[0]!.range).toEqual(sourceRange!);
  });
});
