/**
 * YAML nested past the loader's limit is a located error: a port of
 * `tests/compose/test_yaml_nesting_limit.py`.
 *
 * The flow and `SourceDocument` cases come with `document.ts`. TS-era additions: the first
 * problem in the text wins over a later one, and aliases resolve in the composed tree.
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

import { SourceSyntaxError } from "../../src/compose/source/common.ts";
import { parseYamlLiteral } from "../../src/compose/source/literals.ts";
import { composeFlowDocument, FlowLoaderError } from "../../src/compose/source/scalars.ts";
import { BUILTIN, install } from "../../src/defaults.ts";

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
