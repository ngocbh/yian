/**
 * The `uses:` ref grammar `[<scheme>:]<path>[@<version>]`: a port of the `parse_uses_ref`
 * rows of `tests/engine/test_uses.py`.
 *
 * Ported: the seven `test_parse_*` tests and `test_malformed_rejected`, each malformed value
 * pinned to its `compose.load` message; the messages of the harvested
 * `load-core/engine_uses__malformed_ref_{empty,without_a_flow_path,with_an_empty_version}`.
 *
 * Later (T3-22): the `parse_file` `uses:`/`system:` section tests, the `system:` shape tests,
 * the `_CallableScope` resolver tests and the other `load_flow` tests need the loader.
 */

import { describe, expect, test } from "bun:test";

import { LoadError } from "../../src/compose/helpers/errors.ts";
import { parseUsesRef, UsesRef } from "../../src/compose/helpers/uses.ts";

/** Return the `LoadError` parsing `value` throws. */
function loadError(value: string): LoadError {
  try {
    parseUsesRef(value);
  } catch (error) {
    expect(error).toBeInstanceOf(LoadError);
    return error as LoadError;
  }
  throw new Error("expected a LoadError");
}

describe("parseUsesRef", () => {
  test("local with version", () => {
    expect(parseUsesRef("library/relevance@v1")).toEqual(new UsesRef(null, "library/relevance", "v1"));
  });

  test("local no version", () => {
    expect(parseUsesRef("helper")).toEqual(new UsesRef(null, "helper", null));
  });

  test("nested path", () => {
    expect(parseUsesRef("a/b/c")).toEqual(new UsesRef(null, "a/b/c", null));
  });

  test("hub scheme", () => {
    expect(parseUsesRef("hub:quant/fancy@2.0")).toEqual(new UsesRef("hub", "quant/fancy", "2.0"));
  });

  test("hub scheme no version", () => {
    expect(parseUsesRef("hub:quant/fancy")).toEqual(new UsesRef("hub", "quant/fancy", null));
  });

  test("multi @: the last wins", () => {
    expect(parseUsesRef("a@b@c")).toEqual(new UsesRef(null, "a@b", "c"));
  });

  test("extracts a drive letter as the scheme", () => {
    // Parsing only extracts the scheme (lowercased); the resolver rejects an unknown one, so
    // a Windows-drive-looking ref never touches the disk.
    expect(parseUsesRef("C:/x")).toEqual(new UsesRef("c", "/x", null));
  });

  test("a colon after a slash is not a scheme", () => {
    expect(parseUsesRef("a/b:c")).toEqual(new UsesRef(null, "a/b:c", null));
  });

  test("the ref is frozen", () => {
    expect(Object.isFrozen(parseUsesRef("helper"))).toBe(true);
  });

  test.each([
    ["", "uses: ref must be a non-empty string"],
    ["  ", "uses: ref must be a non-empty string"],
    ["@v1", "uses: ref '@v1' has no flow path"],
    ["hub:", "uses: ref 'hub:' has no flow path"],
    ["hub:@v1", "uses: ref 'hub:@v1' has no flow path"],
    ["name@", "uses: ref 'name@' has an empty version (trailing '@')"],
  ])("malformed %p is rejected", (value, message) => {
    const { diagnostic } = loadError(value);
    expect(diagnostic.code).toBe("compose.load");
    expect(diagnostic.message).toBe(message);
  });
});
