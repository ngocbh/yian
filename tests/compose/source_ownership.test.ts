/**
 * Ownership lints for the compose source package.
 *
 * Only `compose/source` touches the `yaml` package, it imports no other compose module, and the
 * rest of compose never reaches into the document graph's private slots. Each check is a function
 * of `{path, text}` sources, so snippets prove the rules the real tree is held to.
 */

import { describe, expect, spyOn, test } from "bun:test";
import { existsSync, readdirSync, readFileSync } from "node:fs";
import { join, posix, sep } from "node:path";
import { fileURLToPath } from "node:url";

import { parseFile, parseSourceDocument } from "../../src/compose/parse/file.ts";
import { SourceDocument } from "../../src/compose/source/index.ts";
import { blank, importStatements } from "../helpers/source_scan.ts";
import { loadConformanceFlow, loadError, loadErrorAsync } from "../helpers/declarations.ts";

const SRC = fileURLToPath(new URL("../../src/", import.meta.url));

/** One source file: its path below `src` in `/` form, and its text. */
interface Source {
  path: string;
  text: string;
}

/** Return every `.ts` file below `src`, or `[]` while `src` doesn't exist. */
function sources(): Source[] {
  if (!existsSync(SRC)) return [];
  return readdirSync(SRC, { recursive: true, encoding: "utf8" })
    .filter((name) => name.endsWith(".ts"))
    .sort()
    .map((name) => ({
      path: name.split(sep).join("/"),
      text: readFileSync(join(SRC, name), "utf8"),
    }));
}

const below = (dir: string, path: string) => path.startsWith(`${dir}/`);

/** Return `path:line` for each import of the `yaml` package outside `compose/source`. */
function yamlImports(files: Source[]): string[] {
  return files
    .filter((file) => !below("compose/source", file.path))
    .flatMap((file) =>
      importStatements(file.text)
        .filter(({ specifier }) => specifier === "yaml" || specifier.startsWith("yaml/"))
        .map(({ line }) => `${file.path}:${line}`),
    );
}

/**
 * Return `path:line: module` for each relative import in `compose/source` whose
 * module id (its path below `src` without `.ts`/`.js`) is in compose but outside
 * `compose/source`.
 */
function upwardImports(files: Source[]): string[] {
  return files
    .filter((file) => below("compose/source", file.path))
    .flatMap((file) =>
      importStatements(file.text)
        .filter(({ specifier }) => specifier.startsWith("."))
        .map(({ line, specifier }) => ({
          line,
          id: posix.join(posix.dirname(file.path), specifier).replace(/\.[jt]s$/, ""),
        }))
        .filter(({ id }) => below("compose", id) && !below("compose/source", id))
        .map(({ line, id }) => `${file.path}:${line}: ${id}`),
    );
}

const PRIVATE_SLOT = /\._(?:value|definitions|rootDefinition)\b/g;

/**
 * Return `path:line: slot` for each read of a document graph slot in compose
 * outside `compose/source`, in code, not in comments or string contents.
 */
function slotReads(files: Source[]): string[] {
  return files
    .filter((file) => below("compose", file.path) && !below("compose/source", file.path))
    .flatMap((file) =>
      [...blank(file.text).matchAll(PRIVATE_SLOT)].map(
        (m) => `${file.path}:${file.text.slice(0, m.index).split("\n").length}: ${m[0]}`,
      ),
    );
}

describe("source ownership", () => {
  const files = sources();

  test("only compose/source imports yaml", () => {
    expect(yamlImports(files)).toEqual([]);
  });

  test("compose/source has no upward compose imports", () => {
    expect(upwardImports(files)).toEqual([]);
  });

  test("the document graph slots are private to compose/source", () => {
    expect(slotReads(files)).toEqual([]);
  });
});

describe("source ownership snippets", () => {
  test("a yaml import outside compose/source is reported with its line", () => {
    const text = 'const a = 1;\nimport { parse } from "yaml";\nexport * from "yaml/util";';
    expect(yamlImports([{ path: "compose/helpers/yaml_words.ts", text }])).toEqual([
      "compose/helpers/yaml_words.ts:2",
      "compose/helpers/yaml_words.ts:3",
    ]);
    expect(yamlImports([{ path: "runtime/log.ts", text }])).toHaveLength(2);
    expect(yamlImports([{ path: "compose/source/document.ts", text }])).toEqual([]);
    const other = 'import "yamlish"; import "./yaml.ts"; // import "yaml"';
    expect(yamlImports([{ path: "runtime/log.ts", text: other }])).toEqual([]);
  });

  test("a compose/source import of another compose module is reported", () => {
    const path = "compose/source/document.ts";
    const text = [
      'import { a } from "../helpers/yaml_words.ts";',
      'import type { B } from "../index.js";',
      'export * from "../load.ts";',
    ].join("\n");
    expect(upwardImports([{ path, text }])).toEqual([
      `${path}:1: compose/helpers/yaml_words`,
      `${path}:2: compose/index`,
      `${path}:3: compose/load`,
    ]);
    expect(upwardImports([{ path: "compose/load.ts", text }])).toEqual([]);
    const allowed = [
      'import { c } from "./cursor.ts";',
      'import { d } from "../../diagnostics.ts";',
      'import { e } from "yaml";',
      'import { f } from "../source/common.ts";',
    ].join("\n");
    expect(upwardImports([{ path, text: allowed }])).toEqual([]);
  });

  test("a document graph slot read in compose outside compose/source is reported", () => {
    const path = "compose/helpers/yaml_words.ts";
    const text = "const v = cursor._value;\nconst d = doc._definitions[doc._rootDefinition];";
    expect(slotReads([{ path, text }])).toEqual([
      `${path}:1: ._value`,
      `${path}:2: ._definitions`,
      `${path}:2: ._rootDefinition`,
    ]);
    expect(slotReads([{ path: "compose/source/cursor.ts", text }])).toEqual([]);
    expect(slotReads([{ path: "runtime/log.ts", text }])).toEqual([]);
    const other = [
      'const s = "doc._value";',
      "// cursor._definitions",
      "const v = cursor.value + cursor._values + cursor._rootDefinitionId;",
    ].join("\n");
    expect(slotReads([{ path, text: other }])).toEqual([]);
  });
});

const FLOW = `id: owned
name: owned
nodes:
  echo:
    kind: code
    code: ./fixtures/compose_codefns.ts:echo
    input:
      value: hello
    output: str
output: \${echo.output}
`;

const INLINE_FLOW = `id: owned-inline
name: owned_inline
nodes:
  wrapper:
    kind: call
    nodes:
      echo:
        kind: code
        code: ./fixtures/compose_codefns.ts:echo
        input:
          value: hello
        output: str
    output: \${echo.output}
output: \${wrapper.output}
`;

const DEFS_FLOW = `id: owned-defs
name: owned_defs
defs:
  echo_child:
    input:
      value: !str
    nodes:
      echo:
        kind: code
        code: ./fixtures/compose_codefns.ts:echo
        input:
          value: \${input.value}
        output: str
    output: \${echo.output}
nodes:
  invoke:
    kind: call
    call: echo_child
    input:
      value: hello
output: \${invoke.output}
`;

describe("parse ownership", () => {
  test("parseFile reuses a supplied document", () => {
    const document = SourceDocument.parse(FLOW);
    const parse = spyOn(SourceDocument, "parse");
    try {
      const parsed = parseFile(document);

      expect(parse).not.toHaveBeenCalled();
      expect(parsed.id).toBe("owned");
      expect((parsed.nodes["echo"] as Record<string, unknown>)["kind"]).toBe("code");
    } finally {
      parse.mockRestore();
    }
  });

  test("parseSourceDocument translates a syntax error", () => {
    const error = loadError(() => parseSourceDocument("id: [", "broken.yaml"));

    expect(error.message).toContain("invalid YAML");
    expect(error.diagnostic.frames.at(-1)!.range!.startLine).toBe(1);
  });

  test.each([
    ["plain", FLOW],
    ["inline", INLINE_FLOW],
    ["defs", DEFS_FLOW],
  ])("root variants use one document: %s", async (_id, source) => {
    const parse = spyOn(SourceDocument, "parse");
    try {
      await loadConformanceFlow(source);

      expect(parse).toHaveBeenCalledTimes(1);
    } finally {
      parse.mockRestore();
    }
  });

  test("a root load does not mutate the captured document", async () => {
    const captured: [SourceDocument, SourceDocument["root"]["value"]][] = [];
    const original = SourceDocument.parse.bind(SourceDocument);
    const parse = spyOn(SourceDocument, "parse").mockImplementation((text, label) => {
      const document = original(text, label);
      captured.push([document, document.root.value]);
      return document;
    });
    try {
      await loadConformanceFlow(INLINE_FLOW);
    } finally {
      parse.mockRestore();
    }

    expect(captured).toHaveLength(1);
    const [document, originalValue] = captured[0]!;
    expect(document.root.value).toEqual(originalValue);
  });

  test("loadFlow of a malformed root is a located load error", async () => {
    const error = await loadErrorAsync(loadConformanceFlow("id: ["));

    expect(error.message).toContain("invalid YAML");
    expect(error.diagnostic.frames.at(-1)!.range!.startLine).toBe(1);
  });

  test("a recursive boundary alias does not escape as a source access error", async () => {
    const source = `&root
id: recursive-boundary
name: recursive_boundary
input: *root
nodes:
  echo:
    kind: code
    code: ./fixtures/compose_codefns.ts:echo
    input: {value: hello}
    output: str
output: \${echo.output}
`;

    await loadErrorAsync(loadConformanceFlow(source));
  });
});
