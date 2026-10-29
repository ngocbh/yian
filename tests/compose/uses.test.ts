/**
 * External callables: the `uses:` ref grammar `[<scheme>:]<path>[@<version>]`, the `uses:` and
 * `system:` sections, and resolving a `uses:` alias at load. A resolver returns a promise or a flow.
 */

import { describe, expect, test } from "bun:test";

import type { ComposedFlow } from "../../src/compose/build/pipeline.ts";
import { LoadDiagnostics, LoadError } from "../../src/compose/helpers/errors.ts";
import { parseUsesRef, UsesRef } from "../../src/compose/helpers/uses.ts";
import { parseFile } from "../../src/compose/parse/file.ts";
import { SourceDocument } from "../../src/compose/source/index.ts";

import { loadConformanceFlow, loadErrorAsync } from "../helpers/declarations.ts";

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

  test("an @ in a directory is part of the path", () => {
    expect(parseUsesRef("./vendor/@acme/flow.yaml")).toEqual(new UsesRef(null, "./vendor/@acme/flow.yaml", null));
    expect(parseUsesRef("./vendor/@acme/flow.yaml@v1")).toEqual(new UsesRef(null, "./vendor/@acme/flow.yaml", "v1"));
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

// A minimal valid single-node flow; each test appends its own `uses:`/`system:` sections.
const BASE = `id: f
name: f
input:
  x: !str
nodes:
  a: {kind: code, code: ./fixtures/compose_codefns.ts:echo, input: {x: "\${input.x}"}}
output: {y: "\${a.output}"}
`;

describe("the uses: and system: sections", () => {
  test("parseFile accepts uses: and system:", () => {
    const f = parseFile(BASE + "uses:\n  mom: library/relevance@v1\nsystem:\n  paths: [../shared]\n");
    expect(f.uses).toEqual({ mom: "library/relevance@v1" });
    expect(f.system).toEqual({ paths: ["../shared"] });
  });

  test("parseFile defaults them to empty", () => {
    const f = parseFile(BASE);
    expect(f.uses).toEqual({});
    expect(f.system).toEqual({});
  });

  test("an unknown top-level key is still refused", () => {
    expect(() => parseFile(BASE + "bogus: 1\n")).toThrow(/unknown top-level key/);
  });

  test("system: refuses an unknown key", async () => {
    const error = await loadErrorAsync(loadConformanceFlow(BASE + "system:\n  bogus: 1\n"));
    expect(error.message).toBe('system: unknown key(s) ["bogus"] (allowed: paths)');
  });

  test("system: paths must be a list of strings", async () => {
    const error = await loadErrorAsync(loadConformanceFlow(BASE + "system:\n  paths: 5\n"));
    expect(error.message).toBe("system: paths must be a list of strings");
  });

  test("system: paths as a list of strings loads", async () => {
    // Inert without searchPaths; only its shape is checked.
    await loadConformanceFlow(BASE + "system:\n  paths: [a, b]\n");
  });

  test("${system.paths} is not a reference", async () => {
    const text = BASE.replace('"${input.x}"', '"${system.paths}"') + "system:\n  paths: [.]\n";
    const error = await loadErrorAsync(loadConformanceFlow(text));
    expect(error.message).toContain("reference ${system.paths} uses unknown system ambient 'paths'");
  });
});

// A child flow taking `x: str`; fake resolvers load it.
const CHILD = `id: child
name: child
input:
  x: !str
nodes:
  c: {kind: code, code: ./fixtures/compose_codefns.ts:echo, input: {x: "\${input.x}"}}
output: {y: "\${c.output}"}
`;

/** A flow whose node `a` calls `alias` with `x`. */
const withCall = (alias: string) =>
  "id: f\nname: f\ninput:\n  x: !str\n" +
  `nodes:\n  a: {kind: call, call: ${alias}, input: {x: "\${input.x}"}}\n` +
  'output: {y: "${a.output}"}\n';

/** Return `CHILD` declaring `version`, or as written when `version` is `null`. */
const childAt = (version: string | null) =>
  version === null ? CHILD : CHILD.replace("name: child\n", `name: child\nversion: "${version}"\n`);

/** A resolver recording each `(path, version)` in `seen` and loading `CHILD` at that version. */
const fakeResolver =
  (seen: [string, string | null][]) =>
  (path: string, version: string | null): Promise<ComposedFlow> => {
    seen.push([path, version]);
    return loadConformanceFlow(childAt(version));
  };

/** A resolver refusing every path. */
const reject = (path: string): ComposedFlow => {
  throw new LoadDiagnostics().error(`no such flow '${path}'`);
};

const EIGHT = ["p8", "p7", "p6", "p5", "p4", "p3", "p2", "p1"];

describe("resolving a uses: alias", () => {
  test("an alias resolves through the resolver with its version", async () => {
    const seen: [string, string | null][] = [];
    await loadConformanceFlow(withCall("mom") + "uses:\n  mom: library/relevance@v1\n", {
      childResolver: fakeResolver(seen),
    });
    expect(seen).toEqual([["library/relevance", "v1"]]);
  });

  test("an alias whose path has an @ in a directory", async () => {
    const seen: [string, string | null][] = [];
    await loadConformanceFlow(withCall("c") + "uses:\n  c: ./vendor/@acme/child.yaml\n", {
      childResolver: fakeResolver(seen),
    });
    expect(seen).toEqual([["./vendor/@acme/child.yaml", null]]);
  });

  test("an alias without a version", async () => {
    const seen: [string, string | null][] = [];
    await loadConformanceFlow(withCall("mom") + "uses:\n  mom: helper\n", { childResolver: fakeResolver(seen) });
    expect(seen).toEqual([["helper", null]]);
  });

  test("the hub: scheme is deferred", async () => {
    const text = withCall("m") + "uses:\n  m: hub:q/f@1\n";
    const error = await loadErrorAsync(loadConformanceFlow(text, { childResolver: fakeResolver([]) }));
    expect(error.message).toBe("callable 'm' (uses: 'hub:q/f@1'): marketplace not supported yet (hub: scheme)");
  });

  test("an unknown scheme is refused when resolving", async () => {
    const text = withCall("m") + "uses:\n  m: git:q/f@1\n";
    const error = await loadErrorAsync(loadConformanceFlow(text, { childResolver: fakeResolver([]) }));
    expect(error.message).toBe(
      "callable 'm' (uses: 'git:q/f@1'): unknown scheme 'git': (recognized: a local ref, or hub:)",
    );
  });

  test("a name in both defs: and uses: is refused", async () => {
    const text =
      withCall("dup") +
      "uses:\n  dup: child\n" +
      "defs:\n  dup:\n    input:\n      x: !str\n" +
      '    nodes:\n      c: {kind: code, code: ./fixtures/compose_codefns.ts:echo, input: {x: "${x}"}}\n' +
      '    output: {y: "${c.output}"}\n';
    const error = await loadErrorAsync(loadConformanceFlow(text, { childResolver: fakeResolver([]) }));
    expect(error.message).toBe('name(s) ["dup"] declared in both defs: and uses: (a callable has one namespace)');
  });

  test("an alias colliding with a node id is refused", async () => {
    const text = withCall("a") + "uses:\n  a: child\n";
    const error = await loadErrorAsync(loadConformanceFlow(text, { childResolver: fakeResolver([]) }));
    expect(error.message).toBe('uses: alias ["a"] collides with a node id');
  });

  test("an uncalled broken alias is loud", async () => {
    const error = await loadErrorAsync(
      loadConformanceFlow(BASE + "uses:\n  dead: hub:x/y@1\n", { childResolver: fakeResolver([]) }),
    );
    expect(error.message).toContain("marketplace not supported yet");
  });

  test("an uncalled missing alias is loud, at its alias", async () => {
    const text = BASE + "uses:\n  dead: nope\n";
    const expected = SourceDocument.parse(text).root.child("uses").child("dead").range;
    const error = await loadErrorAsync(loadConformanceFlow(text, { childResolver: reject }));
    expect(error.diagnostic.frames.map((frame) => frame.range)).toEqual([expected!]);
  });

  test("aliases resolve in source order", async () => {
    const seen: [string, string | null][] = [];
    const text = BASE + "uses:\n" + EIGHT.map((alias) => `  ${alias}: lib_${alias}\n`).join("");
    await loadConformanceFlow(text, { childResolver: fakeResolver(seen) });
    expect(seen).toEqual(EIGHT.map((alias) => [`lib_${alias}`, null]));
  });

  test("the first broken alias in source order is reported", async () => {
    const text = BASE + "uses:\n" + EIGHT.map((alias) => `  ${alias}: lib_${alias}\n`).join("");
    const expected = SourceDocument.parse(text).root.child("uses").child("p8").range;
    const error = await loadErrorAsync(loadConformanceFlow(text, { childResolver: reject }));
    expect(error.message).toBe("no such flow 'lib_p8'");
    expect(error.diagnostic.frames[0]!.range).toEqual(expected!);
  });

  test("an inline call(...) routes through the alias", async () => {
    const seen: [string, string | null][] = [];
    const text =
      "id: f\nname: f\ninput:\n  x: !str\nnodes:\n" +
      "  a: {kind: code, code: ./fixtures/compose_codefns.ts:echo, " +
      'input: {x: "call(mom, x=inputs.x)"}}\n' +
      'output: {y: "${a.output}"}\nuses:\n  mom: library/relevance@v1\n';
    await loadConformanceFlow(text, { childResolver: fakeResolver(seen) });
    expect(seen).toContainEqual(["library/relevance", "v1"]);
  });

  test("the cross-flow type check runs through an alias", async () => {
    // The child takes `x: int`; the parent binds text through the alias.
    const childInt =
      "id: c\nname: c\ninput:\n  x: !int\n" +
      'nodes:\n  c: {kind: code, code: ./fixtures/compose_codefns.ts:echo, input: {x: "${input.x}"}}\n' +
      'output: {y: "${c.output}"}\n';
    const text =
      "id: f\nname: f\ninput:\n  s: !str\n" +
      'nodes:\n  a: {kind: call, call: mom, input: {x: "${input.s}"}}\n' +
      'output: {y: "${a.output}"}\nuses:\n  mom: c\n';
    const error = await loadErrorAsync(
      loadConformanceFlow(text, { childResolver: () => loadConformanceFlow(childInt) }),
    );
    expect(error.message).toBe("call node 'a': binding 'x' — child expects 'int', source is 'str'");
  });

  test("a host resolver's child is held to the ref's version pin", async () => {
    // The file resolver and a resume check the pin too, so a flow that loads here also resumes.
    const text = withCall("c") + "uses:\n  c: child@1\n";
    const expected = SourceDocument.parse(text).root.child("uses").child("c").range;
    const error = await loadErrorAsync(
      loadConformanceFlow(text, { childResolver: () => loadConformanceFlow(childAt("2")) }),
    );
    expect(error.message).toBe(
      'uses: ref pinned child@1 but the resolved file declares version: "2" ' +
        '(file-per-version: name a file that declares "1")',
    );
    expect(error.diagnostic.frames.map((frame) => frame.range)).toEqual([expected!]);
  });

  test("a bare call without uses: or defs: is a located error that never asks the resolver", async () => {
    const seen: [string, string | null][] = [];
    const error = await loadErrorAsync(loadConformanceFlow(withCall("ghost"), { childResolver: fakeResolver(seen) }));
    expect(error.message).toBe(
      "callable 'ghost' is neither an in-file defs: callable nor a uses: alias — external flows " +
        "are reachable only through a uses: alias (declare `uses: {ghost: <ref>}`)",
    );
    expect(error.diagnostic.frames.length).toBeGreaterThan(0);
    expect(seen).toEqual([]);
  });
});
