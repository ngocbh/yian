/**
 * A YAML document that expands past `maxYamlNodes` nodes is a located error: a port of
 * `tests/compose/test_yaml_node_limit.py`.
 *
 * Every mapping, list, and scalar node (mapping keys included) counts once per place it
 * appears, so an alias, and a `<<:` merge source, counts every node it names again. The
 * `load_flow` cases, with their frames, come with the loader; here each text is composed
 * directly, and the error node's line stands in for the frame's line.
 */

import { afterEach, describe, expect, test } from "bun:test";
import { isAlias, isMap, isNode, isSeq, type Node, type YAMLSeq } from "yaml";

import { SourceSyntaxError } from "../../src/compose/source/common.ts";
import { SourceDocument } from "../../src/compose/source/document.ts";
import { parseYamlLiteral, readPlainScalar } from "../../src/compose/source/literals.ts";
import { composeFlowDocument, FlowLoaderError } from "../../src/compose/source/scalars.ts";
import { BUILTIN, install } from "../../src/defaults.ts";
import { isRecord } from "../../src/typesys/index.ts";

const MAX_YAML_NODES = BUILTIN.maxYamlNodes;

afterEach(() => install(BUILTIN));

/** Return the start of the node-limit message for `limit`. */
function message(limit: number): string {
  return `holds more than ${limit} YAML nodes`;
}

/** Use `limit` as the node limit until the test ends. */
function limitNodes(limit: number): void {
  install({ ...BUILTIN, maxYamlNodes: limit });
}

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

/** Return the `FlowLoaderError` composing `text` throws and the seconds it took. */
function timedRefusal(text: string): { error: FlowLoaderError; seconds: number } {
  const started = performance.now();
  const error = refusal(text);
  return { error, seconds: (performance.now() - started) / 1000 };
}

/** Return the 1-based line of `text` that `node` starts on. */
function lineOf(text: string, node: Node | null): number {
  const start = node?.range?.[0];
  if (start === undefined) throw new Error("expected a node with a range");
  return text.slice(0, start).split("\n").length;
}

/** Return the seconds `run` takes. */
function seconds(run: () => void): number {
  const started = performance.now();
  run();
  return (performance.now() - started) / 1000;
}

/** Return a one-node flow whose `topic` input is `topic`, after a top-level `header`. */
function flow(topic: string, header = ""): string {
  return (
    `id: f\nname: f\n${header}nodes:\n  use:\n    kind: code\n` +
    "    code: tests.engine._compose_codefns:echo\n" +
    `    input:\n      topic: ${topic}\n    output: str\noutput: \${use.output}\n`
  );
}

/**
 * Return an `x-anchors:` block list, one anchor per line, each twice the last.
 *
 * Line 4 holds `a0`, so `a<i>` is on line `4 + i` of a flow built by `flow`.
 */
function doublingAnchors(levels: number): string {
  const lines = ["x-anchors:", "  - &a0 [1]"];
  for (let i = 1; i < levels; i++) lines.push(`  - &a${i} [*a${i - 1}, *a${i - 1}]`);
  return lines.join("\n") + "\n";
}

/** Return an `x-anchors:` line whose anchor `a<i>` names a list nesting i+1 levels. */
function linearAnchors(count: number): string {
  const items = ["&a0 [1]"];
  for (let i = 1; i < count; i++) items.push(`&a${i} [*a${i - 1}]`);
  return `x-anchors: [${items.join(", ")}]\n`;
}

/** Return `count` copies of `item`, comma-separated. */
function repeated(item: string, count: number): string {
  return Array(count).fill(item).join(", ");
}

describe("the YAML node limit", () => {
  test("the message names the variable and what a node is", () => {
    limitNodes(50);

    expect(refusal("{a: [" + repeated("1", 50) + "]}").message).toBe(
      "this list holds more than 50 YAML nodes, counting each alias as a copy of what " +
        "it names; the limit for a YAML document is 50 (YIAN_MAX_YAML_NODES; each " +
        "mapping, list, and scalar is a node)",
    );
  });

  test("exact boundary", () => {
    limitNodes(50);

    // One list and 49 scalars are 50 nodes.
    const { document } = composeFlowDocument("[" + repeated("1", 49) + "]");
    expect((document.contents as YAMLSeq).items).toHaveLength(49);
    expect(refusal("[" + repeated("1", 50) + "]").message).toContain(message(50));
  });

  test("mapping keys count", () => {
    limitNodes(50);
    const pairs = (count: number) =>
      Array.from({ length: count }, (_, i) => `k${i}: 1`).join(", ");

    // One mapping, 24 keys, and 24 values are 49 nodes; 25 pairs are 51.
    composeFlowDocument("{" + pairs(24) + "}");
    expect(refusal("{" + pairs(25) + "}").message).toContain(`this mapping ${message(50)}`);
  });

  test("each alias counts as a copy", () => {
    limitNodes(70);
    const ten = "[" + repeated("1", 10) + "]";
    const aliased = (copies: number) => `x: &t ${ten}\ny: [` + repeated("*t", copies) + "]\n";

    // The root, x, the 11-node list, y, and y's list are 15 nodes; each alias adds 11.
    composeFlowDocument(aliased(0));
    composeFlowDocument(aliased(5));
    expect(refusal(aliased(6)).message).toContain(message(70));
  });

  test("a merge source counts in full", () => {
    limitNodes(1000);
    const lines = ["m0: &m0 {k0: 1}"];
    for (let i = 1; i < 8; i++) {
      lines.push(`m${i}: &m${i} {<<: [*m${i - 1}, *m${i - 1}], k${i}: 1}`);
    }
    const text = lines.join("\n") + "\n";

    // m<i> holds 2 * m<i-1> + 5 nodes, so m6 holds 507 and m7's merge list
    // [*m6, *m6] holds 1015, although merged m7 has only 8 keys.
    const error = refusal(text);
    expect(error.message).toContain(`this list ${message(1000)}`);
    expect(lineOf(text, error.node)).toBe(8);
    const merged = parseYamlLiteral(lines.slice(0, 5).join("\n") + "\n");
    expect(isRecord(merged) ? merged["m4"] : null).toEqual({
      k0: 1n,
      k1: 1n,
      k2: 1n,
      k3: 1n,
      k4: 1n,
    });
  });

  test("doubling aliases fail fast at the first anchor over the limit", () => {
    // a<i> holds 3 * 2**i - 1 nodes: a15 (98,303) fits and a16 (196,607) does not.
    expect(3 * 2 ** 15 - 1 <= MAX_YAML_NODES && MAX_YAML_NODES < 3 * 2 ** 16 - 1).toBe(true);
    const text = flow("*a29", doublingAnchors(30));
    expect(text.split("\n")[3 + 16]).toBe("  - &a16 [*a15, *a15]");

    const { error, seconds: took } = timedRefusal(text);
    expect(took).toBeLessThan(5);
    expect(error.message).toContain(`this list ${message(MAX_YAML_NODES)}`);
    expect(lineOf(text, error.node)).toBe(4 + 16);
    expect(error.node?.anchor).toBe("a16");
  });

  test("an isolated literal past the limit is a syntax error", () => {
    const items = ["&a0 [1]"];
    for (let i = 1; i < 30; i++) items.push(`&a${i} [*a${i - 1}, *a${i - 1}]`);

    const text = `[${items.join(", ")}]`;

    expect(() => parseYamlLiteral(text)).toThrow(SourceSyntaxError);
    expect(() => parseYamlLiteral(text)).toThrow(message(MAX_YAML_NODES));
  });

  test("a listed 1000-link chain meets the node limit first", () => {
    // a<i> holds i + 2 nodes, so the listed chain holds about 500,000.
    const text = flow("[*a998]", linearAnchors(999));
    const error = refusal(text);

    expect(error.message).toContain(message(MAX_YAML_NODES));
    expect(lineOf(text, error.node)).toBe(3);
  });

  test("a count a recursive alias cut short is not reused outside the cycle", () => {
    // Inside r, s's aliases of r are recursive and count 1 each, but t0 reaches s from
    // outside r, where each *r expands r's 12,287-node a12 again: t0 holds over 98,000
    // nodes. With the anchors before it, the count passes the limit inside t0, so the
    // error names the deepest node on the path already over it, the x-anchors list.
    const lines = ["  - &r [*a12, &s [*r, *r, *r, *r]]", "  - &t0 [*s, *s]"];
    for (let j = 1; j < 30; j++) lines.push(`  - &t${j} [*t${j - 1}, *t${j - 1}]`);
    const text = flow("*t29", doublingAnchors(13) + lines.join("\n") + "\n");
    expect(text.split("\n")[3]).toBe("  - &a0 [1]");

    const { error, seconds: took } = timedRefusal(text);
    expect(took).toBeLessThan(5);
    expect(error.message).toContain(`this list ${message(MAX_YAML_NODES)}`);
    expect(lineOf(text, error.node)).toBe(4);
  });

  test("a node on its own cycle is counted again inside it", () => {
    // M is on its own cycle, so each *M inside D counts 1 even after M's first visit:
    // the block holds about 1,200 nodes, not 400 copies of M's 402.
    const anchors = "x-anchors:\n  - &M [&D [" + repeated("*M", 400) + "]]\n  - [*D]\n";

    composeFlowDocument(anchors);
    composeFlowDocument(flow("hi", anchors));
  });

  const doubling: string[] = ["&s0 [*r, *r]"];
  for (let i = 1; i < 30; i++) doubling.push(`&s${i} [*s${i - 1}, *s${i - 1}]`);
  const wide: string[] = ["&u [" + repeated("*r", 1000) + "]", "&s1 [*u, *u]"];
  for (let i = 2; i < 30; i++) wide.push(`&s${i} [*s${i - 1}, *s${i - 1}]`);
  const repeatedNodes: string[] = ["&s0 [*r]"];
  for (let i = 1; i < 16; i++) repeatedNodes.push(`&s${i} [*s${i - 1}, *s${i - 1}]`);
  for (let i = 0; i < 300; i++) repeatedNodes.push("*s15", "[*s15]");

  test.each([
    // Each s<i> depends on r, so the walk cannot memoize it; s<i> doubles s<i-1>.
    ["doubling", doubling, "mapping", 1],
    // u's thousand recursive aliases each count, so recounting u stays bounded.
    ["wide", wide, "list", 3],
    // s15 depends on r and holds about 65,000 nodes; r lists it 300 more times,
    // bare and wrapped, so a walk that recounted each would take minutes.
    ["repeated", repeatedNodes, "mapping", 1],
  ] as const)("exponential nodes inside a cycle fail fast: %s", (_, items, kind, line) => {
    // Once the document has counted past the limit, a node a recursive alias made
    // path-dependent is not counted again: the error names the deepest node on the
    // path already over the limit, which here is often the document itself.
    const text = flow("*r", `x-anchors: [&r [${items.join(", ")}]]\n`);

    const { error, seconds: took } = timedRefusal(text);
    expect(took).toBeLessThan(5);
    expect(error.message).toContain(`this ${kind} ${message(MAX_YAML_NODES)}`);
    expect(lineOf(text, error.node)).toBe(line);
  });

  test.each(["&x [*x, *x, [*x]]", "&m {<<: *m}"])("a pure cycle ends: %p", (value) => {
    expect(seconds(() => composeFlowDocument(flow(value)))).toBeLessThan(5);
  });

  test("the count matches a full expansion", () => {
    // Below the limit the count is exact, recursive aliases or not: a document composes
    // at a limit of its full expansion and fails one below it.
    const random = mulberry32(646);
    for (let index = 0; index < 300; index++) {
      const text = randomAliasGraph(random, index % 2 === 1);
      limitNodes(Number.MAX_SAFE_INTEGER);
      const expanded = expandedCount(text);
      if (expanded > 5000) continue;

      limitNodes(expanded - 1);
      expect([text, overNodeLimit(text)]).toEqual([text, true]);
      limitNodes(expanded);
      expect([text, overNodeLimit(text)]).toEqual([text, false]);
    }
  });

  test("reading a plain scalar is unaffected", () => {
    limitNodes(0);

    expect(readPlainScalar("1e3")).toBe(1000);
    expect(readPlainScalar("~")).toBeNull();
  });

  test("a plain 20,000-element list composes", () => {
    const { document } = composeFlowDocument("[" + repeated("1", 20_000) + "]");

    expect((document.contents as YAMLSeq).items).toHaveLength(20_000);
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
  test("the message names the variable and what a node is", () => {
    limitNodes(50);

    expect(documentRefusal("{a: [" + repeated("1", 50) + "]}").message).toBe(
      "this list holds more than 50 YAML nodes, counting each alias as a copy of what " +
        "it names; the limit for a YAML document is 50 (YIAN_MAX_YAML_NODES; each " +
        "mapping, list, and scalar is a node)",
    );
  });

  test("exact boundary", () => {
    limitNodes(50);

    // One list and 49 scalars are 50 nodes.
    expect(SourceDocument.parse("[" + repeated("1", 49) + "]")._value).toEqual(
      Array(49).fill(1n),
    );
    expect(documentRefusal("[" + repeated("1", 50) + "]").message).toContain(message(50));
  });

  test("mapping keys count", () => {
    limitNodes(50);
    const pairs = (count: number) =>
      Array.from({ length: count }, (_, i) => `k${i}: 1`).join(", ");

    // One mapping, 24 keys, and 24 values are 49 nodes; 25 pairs are 51.
    SourceDocument.parse("{" + pairs(24) + "}");
    expect(documentRefusal("{" + pairs(25) + "}").message).toContain(
      `this mapping ${message(50)}`,
    );
  });

  test("each alias counts as a copy", () => {
    limitNodes(70);
    const ten = "[" + repeated("1", 10) + "]";
    const aliased = (copies: number) => `x: &t ${ten}\ny: [` + repeated("*t", copies) + "]\n";

    // The root, x, the 11-node list, y, and y's list are 15 nodes; each alias adds 11.
    SourceDocument.parse(aliased(0));
    SourceDocument.parse(aliased(5));
    expect(documentRefusal(aliased(6)).message).toContain(message(70));
  });

  test("a merge source counts in full", () => {
    limitNodes(1000);
    const lines = ["m0: &m0 {k0: 1}"];
    for (let i = 1; i < 8; i++) {
      lines.push(`m${i}: &m${i} {<<: [*m${i - 1}, *m${i - 1}], k${i}: 1}`);
    }

    // m<i> holds 2 * m<i-1> + 5 nodes, so m6 holds 507 and m7's merge list
    // [*m6, *m6] holds 1015, although merged m7 has only 8 keys.
    const error = documentRefusal(lines.join("\n") + "\n");
    expect(error.message).toContain(`this list ${message(1000)}`);
    expect(error.range?.startLine).toBe(8);
    const merged = SourceDocument.parse(lines.slice(0, 5).join("\n") + "\n")._value;
    expect(isRecord(merged) ? merged["m4"] : null).toEqual({
      k0: 1n,
      k1: 1n,
      k2: 1n,
      k3: 1n,
      k4: 1n,
    });
  });

  test("a node on its own cycle is counted again inside it", () => {
    const anchors = "x-anchors:\n  - &M [&D [" + repeated("*M", 400) + "]]\n  - [*D]\n";

    SourceDocument.parse(anchors);
  });

  test("a plain 20,000-element list loads", () => {
    const value = SourceDocument.parse("[" + repeated("1", 20_000) + "]")._value;

    expect(Array.isArray(value) && value.length).toBe(20_000);
  });
});

/** Return a seeded generator of floats in [0, 1): mulberry32. */
function mulberry32(seed: number): () => number {
  let state = seed;
  return () => {
    state = (state + 0x6d2b79f5) | 0;
    let t = Math.imul(state ^ (state >>> 15), 1 | state);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Return a flow list of nested anchored lists and aliases, recursive if asked. */
function randomAliasGraph(random: () => number, recursive: boolean): string {
  const openNames: string[] = [];
  const doneNames: string[] = [];
  const randint = (low: number, high: number) => low + Math.floor(random() * (high - low + 1));

  const value = (depth: number): string => {
    const roll = random();
    const pool = [...doneNames, ...(recursive ? openNames : [])];
    if (pool.length > 0 && roll < 0.45) return "*" + pool[Math.floor(random() * pool.length)];
    if (depth > 4 || roll < 0.6) return "1";
    const name = `a${openNames.length + doneNames.length}`;
    openNames.push(name);
    const items = Array.from({ length: randint(1, 4) }, () => value(depth + 1));
    openNames.splice(openNames.indexOf(name), 1);
    doneNames.push(name);
    return `&${name} [${items.join(", ")}]`;
  };

  return "[" + Array.from({ length: randint(2, 6) }, () => value(0)).join(", ") + "]";
}

/** Return how many nodes `text` expands to, counting a recursive alias as 1. */
function expandedCount(text: string): number {
  const { document, aliasTargets } = composeFlowDocument(text);
  const path = new Set<Node>();

  const count = (child: Node | null): number => {
    if (child === null) return 1;
    const node = isAlias(child) ? aliasTargets.get(child)! : child;
    if (path.has(node)) return 1;
    path.add(node);
    let items: unknown[] = [];
    if (isSeq(node)) items = node.items;
    else if (isMap(node)) items = node.items.flatMap((pair) => [pair.key, pair.value]);
    let total = 1;
    for (const each of items) total += count(isNode(each) ? each : null);
    path.delete(node);
    return total;
  };

  return count(document.contents);
}

/** Return whether composing `text` meets the node limit. */
function overNodeLimit(text: string): boolean {
  try {
    composeFlowDocument(text);
  } catch (error) {
    expect(error).toBeInstanceOf(FlowLoaderError);
    expect((error as Error).message).toContain("YAML nodes");
    return true;
  }
  return false;
}
