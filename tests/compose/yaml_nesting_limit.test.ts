/**
 * YAML nested past the loader's limit is a located error. The first problem in the text wins over a
 * later one, and aliases resolve in the composed tree.
 */

import { afterEach, describe, expect, test } from "bun:test";
import {
  type Alias,
  isAlias,
  isSeq,
  LineCounter,
  type Node,
  YAMLError,
  type YAMLMap,
  type YAMLSeq,
} from "yaml";

import { parseSourceDocument } from "../../src/compose/parse/file.ts";
import { SourceSyntaxError } from "../../src/compose/source/common.ts";
import { SourceDocument } from "../../src/compose/source/document.ts";
import { parseYamlLiteral } from "../../src/compose/source/literals.ts";
import { composeFlowDocument, FlowLoaderError } from "../../src/compose/source/scalars.ts";
import { BUILTIN, install } from "../../src/defaults.ts";
import { SourceRange } from "../../src/diagnostics.ts";
import type { Value } from "../../src/typesys/index.ts";
import { assembled, runCase, runTextCase } from "../helpers/declarations.ts";

const MAX_YAML_NESTING = BUILTIN.maxYamlNesting;
const MAX_VALUE_NESTING = BUILTIN.maxValueNesting;

const MESSAGE = `nests deeper than ${MAX_YAML_NESTING} levels, the limit for a YAML document`;

afterEach(() => install(BUILTIN));

/** Return the `FlowLoaderError` composing `text` throws, failing if it throws none. */
function refusal(text: string): FlowLoaderError {
  try {
    composeFlowDocument(text);
  } catch (error) {
    expect(error).toBeInstanceOf(FlowLoaderError);
    return error as FlowLoaderError;
  }
  throw new Error("expected a FlowLoaderError");
}

describe("the YAML nesting limit", () => {
  test("is twice the value limit, and the message names both", () => {
    expect(MAX_YAML_NESTING).toBe(2 * MAX_VALUE_NESTING);
    const depth = MAX_YAML_NESTING + 1;
    const error = refusal("[".repeat(depth) + "]".repeat(depth));

    expect(error.message).toContain(`this list nests deeper than ${MAX_YAML_NESTING} levels`);
    expect(error.message).toContain(`a yian value nests at most ${MAX_VALUE_NESTING})`);
  });

  test.each([
    ["[", "]", "list"],
    ["{a: ", "}", "mapping"],
  ])("refuses a flow collection past it at its opener: %p", (opener, closer, kind) => {
    const depth = MAX_YAML_NESTING + 1;
    const error = refusal(opener.repeat(depth) + "1" + closer.repeat(depth));

    expect(error.message).toContain(`this ${kind} ${MESSAGE}`);
    expect(error.node?.range?.[0]).toBe(opener.length * MAX_YAML_NESTING);
  });

  test.each([
    ["a:", "b: 1", "mapping"],
    ["-", "- 1", "list"],
  ])("refuses a block collection past it at its first line: %p", (line, last, kind) => {
    const depth = MAX_YAML_NESTING;
    let text = "";
    for (let i = 0; i < depth; i++) text += "  ".repeat(i) + line + "\n";
    const lastLine = text.length + 2 * depth;
    text += "  ".repeat(depth) + last + "\n";

    const error = refusal(text);

    expect(error.message).toContain(`this ${kind} ${MESSAGE}`);
    expect(error.node?.range?.[0]).toBe(lastLine);
  });

  test("lets a document at the limit compose", () => {
    const depth = MAX_YAML_NESTING;
    let value = composeFlowDocument("[".repeat(depth) + "]".repeat(depth)).document.contents;
    for (let i = 0; i < depth - 1; i++) {
      expect(isSeq(value)).toBe(true);
      expect((value as YAMLSeq).items).toHaveLength(1);
      value = (value as YAMLSeq).items[0] as typeof value;
    }
    expect(isSeq(value) && value.items.length === 0).toBe(true);
  });

  test("doesn't add up sibling collections", () => {
    const deep = "[".repeat(MAX_YAML_NESTING - 1) + "]".repeat(MAX_YAML_NESTING - 1);
    composeFlowDocument("[" + Array(3).fill(deep).join(", ") + "]");
    composeFlowDocument("[" + Array(5000).fill("[1]").join(", ") + "]");
  });

  test("counts an implicit pair mapping as a level", () => {
    install({ ...BUILTIN, maxYamlNesting: 2 });
    composeFlowDocument("[a: 1]");

    expect(refusal("[a: [1]]").message).toContain("this list nests deeper than 2 levels");
  });

  test("refuses a 100,000-level list with the nesting error, not the library's", () => {
    const depth = 100_000;
    const error = refusal("[".repeat(depth) + "]".repeat(depth));

    expect(error.message).toContain(`this list ${MESSAGE}`);
  });

  test("an isolated literal past the limit is a syntax error", () => {
    const text = "[".repeat(600) + "]".repeat(600);

    expect(() => parseYamlLiteral(text)).toThrow(SourceSyntaxError);
    expect(() => parseYamlLiteral(text)).toThrow(MESSAGE);
  });

  test("reports the first problem in the text", () => {
    const deep = 300;
    let caught: unknown;
    try {
      composeFlowDocument("[," + "[".repeat(deep) + "]".repeat(deep) + "]");
    } catch (error) {
      caught = error;
    }
    expect(caught).toBeInstanceOf(YAMLError);
    expect(caught).not.toBeInstanceOf(FlowLoaderError);
    expect((caught as YAMLError).pos[0]).toBe(1);

    const error = refusal("[".repeat(deep) + "}" + "]".repeat(deep));
    expect(error.message).toContain(`this list ${MESSAGE}`);
  });
});

/** Return the `SourceSyntaxError` parsing `text` throws, failing if it throws none. */
function documentRefusal(text: string): SourceSyntaxError {
  try {
    SourceDocument.parse(text);
  } catch (error) {
    expect(error).toBeInstanceOf(SourceSyntaxError);
    return error as SourceSyntaxError;
  }
  throw new Error("expected a SourceSyntaxError");
}

describe("a source document", () => {
  test("refuses a document past the limit with the message naming both limits", () => {
    const depth = MAX_YAML_NESTING + 1;
    const error = documentRefusal("[".repeat(depth) + "]".repeat(depth));

    expect(error.message).toContain(`this list nests deeper than ${MAX_YAML_NESTING} levels`);
    expect(error.message).toContain(`a yian value nests at most ${MAX_VALUE_NESTING})`);
  });

  test.each([
    ["[", "]", "list"],
    ["{a: ", "}", "mapping"],
  ])("locates a flow collection past the limit: %p", (opener, closer, kind) => {
    const depth = MAX_YAML_NESTING + 1;
    const error = documentRefusal(opener.repeat(depth) + "1" + closer.repeat(depth));

    expect(error.message).toContain(`this ${kind} ${MESSAGE}`);
    const column = 1 + opener.length * MAX_YAML_NESTING;
    expect(error.range).toEqual(new SourceRange(1, column, 1, column + 1));
  });

  test.each([
    ["a:", "b: 1", "mapping"],
    ["-", "- 1", "list"],
  ])("locates a block collection past the limit: %p", (line, last, kind) => {
    const depth = MAX_YAML_NESTING;
    let text = "";
    for (let i = 0; i < depth; i++) text += "  ".repeat(i) + line + "\n";
    text += "  ".repeat(depth) + last + "\n";

    const error = documentRefusal(text);

    expect(error.message).toContain(`this ${kind} ${MESSAGE}`);
    expect(error.range).toEqual(
      new SourceRange(depth + 1, 2 * depth + 1, depth + 1, 2 * depth + 2),
    );
  });

  test("parses a document at the limit", () => {
    const depth = MAX_YAML_NESTING;
    let value: Value = SourceDocument.parse("[".repeat(depth) + "]".repeat(depth))._value;
    for (let i = 0; i < depth - 1; i++) {
      expect(Array.isArray(value) && value.length === 1).toBe(true);
      value = (value as Value[])[0]!;
    }
    expect(value).toEqual([]);
  });

  test("doesn't add up sibling collections", () => {
    const deep = "[".repeat(MAX_YAML_NESTING - 1) + "]".repeat(MAX_YAML_NESTING - 1);
    SourceDocument.parse("[" + Array(3).fill(deep).join(", ") + "]");
    SourceDocument.parse("[" + Array(5000).fill("[1]").join(", ") + "]");
  });
});

describe("aliases", () => {
  test("refuses an undefined alias at the alias", () => {
    const error = refusal("[*nope]");

    expect(error.message).toBe("found undefined alias 'nope'");
    expect(isAlias(error.node)).toBe(true);
    expect(error.node?.range?.[0]).toBe(1);
  });

  test("refuses an alias whose anchor comes later", () => {
    expect(refusal("[*a, &a 1]").message).toBe("found undefined alias 'a'");
  });

  test("resolve a redefined anchor to its latest definition", () => {
    const { document, aliasTargets } = composeFlowDocument("[&a 1, &a 2, *a]");
    const [, second, alias] = (document.contents as YAMLSeq<Node>).items;

    expect(aliasTargets.size).toBe(1);
    expect(isAlias(alias)).toBe(true);
    expect<Node | undefined>(aliasTargets.get(alias as Alias)).toBe(second);
  });

  test("resolve an alias inside its own anchored node to that node", () => {
    const { document, aliasTargets } = composeFlowDocument("&a [*a]");
    const [alias] = (document.contents as YAMLSeq<Node>).items;

    expect(isAlias(alias)).toBe(true);
    expect<Node | undefined>(aliasTargets.get(alias as Alias)).toBe(document.contents as Node);
  });

  test("resolve an alias used as a mapping key", () => {
    // `*k:` would read `k:` as the alias name, so a space comes before the colon.
    const { document, aliasTargets } = composeFlowDocument("[&k a, {*k : 1}]");
    const [scalar, mapping] = (document.contents as YAMLSeq<Node>).items;
    const key = (mapping as YAMLMap<Node, Node>).items[0]?.key;

    expect(isAlias(key)).toBe(true);
    expect<Node | undefined>(aliasTargets.get(key as Alias)).toBe(scalar);
  });
});

test("the options reach the parser", () => {
  const lineCounter = new LineCounter();
  composeFlowDocument("a: 1\nb: 2\n", { lineCounter });

  expect(lineCounter.linePos(5)).toEqual({ line: 2, col: 1 });
});

test("parseSourceDocument: yaml/compose_yaml_nesting_limit__600_levels_is_located", () => {
  runTextCase({
    id: "yaml/compose_yaml_nesting_limit__600_levels_is_located",
    call: parseSourceDocument,
    message:
      "invalid YAML: this list nests deeper than 200 levels, the limit for a YAML document " +
      "(each mapping and list is a level; a yian value nests at most 100); flatten it",
  });
});

test("assemble: yaml/compose_yaml_nesting_limit__value_at_the_document_limit_parses", () => {
  runCase({ id: "yaml/compose_yaml_nesting_limit__value_at_the_document_limit_parses", call: assembled });
});
