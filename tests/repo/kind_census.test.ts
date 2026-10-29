/**
 * Kind-dispatch census guarding the kind-agnostic engine core.
 *
 * The engine core (`runtime/engine.ts` + `runtime/eval_node.ts`) knows only an
 * abstract node contract and the closed `Outcome` sum; it must not branch on a
 * node's kind nor on a concrete `*Expansion` type. The census finds the lines
 * that do and holds their count at zero. It reads the source, so it survives
 * line-number drift.
 *
 * A kind-dispatch line holds any of, outside imports, comments and strings:
 *   - a `NodeKind` member (`NodeKind.AGENT`);
 *   - `SPAWNER_KINDS` or `_SPAWNER_KINDS`, the spawner-kind set;
 *   - `instanceof <Name>Expansion`, also qualified (`instanceof x.CallExpansion`),
 *     a branch on a concrete expansion type;
 *   - `kind` compared with a node-kind string literal, on either side
 *     (`node.kind === "agent"`, `"map" !== n.kind`);
 *   - a `case "<node kind>":` clause of a `switch` whose head mentions `kind`.
 * With no `enum` (`erasableSyntaxOnly`), `NodeKind` is a const object plus a
 * string union, so the last two are the natural ways to branch on a kind.
 * Literals that aren't node kinds, like the `Outcome` sum's `"output"`, `"route"`
 * and `"grow"`, don't count, and neither does a diagnostic read of `node.kind`.
 *
 * Known blind spots (don't route dispatch through them to dodge the census; they
 * break the kind-agnostic contract too): indirect access (`NodeKind[name]`,
 * `node["kind"]`), membership tests (`["call", "map"].includes(node.kind)`),
 * kind-keyed tables built in another file, and type-name compares
 * (`desc.constructor.name === "CallExpansion"`), and comparisons through
 * indexing, parentheses or casts (`"map" === nodes[0].kind`,
 * `(node.kind as string) === "agent"`). The `*Expansion` check relies on every
 * expansion type's name ending in `Expansion`.
 *
 * Known over-count: the `case` clauses of a `switch` nested inside a kind
 * `switch` are counted too, whatever the nested switch's head.
 *
 * Outside the core the literal check would misread `ExecutionPosition.kind ===
 * "call"`, a position kind rather than a node kind; the census doesn't scan there.
 */

import { describe, expect, test } from "bun:test";
import { existsSync, readFileSync } from "node:fs";

import { blank, closing, importStatements } from "../helpers/source_scan.ts";

const SRC = new URL("../../src/", import.meta.url);

/** The two modules, below `src`, that make up the engine core the census guards. */
const CORE_MODULES = ["runtime/engine.ts", "runtime/eval_node.ts"];

/** The values of `NodeKind`. */
const NODE_KINDS = new Set([
  "agent",
  "code",
  "tool",
  "case",
  "loop",
  "start",
  "end",
  "call",
  "map",
]);

/** Kind references that count wherever they appear in code. */
const KIND_REFERENCES = [
  /\bNodeKind\.[A-Za-z_]/g,
  /\b_?SPAWNER_KINDS\b/g,
  /\binstanceof\s+(?:[\w$]+\.)*[A-Za-z_$][\w$]*Expansion\b/g,
];

// Over blanked text a literal is its quotes around spaces; group 2 spans its contents.
const KIND_IS_LITERAL = /\bkind\s*[!=]==?\s*(["'`])( *)\1/dg;
const LITERAL_IS_KIND = /(["'`])( *)\1\s*[!=]==?\s*(?:[\w$]+[?!]?\.)*kind(?![\w$])/dg;
const CASE_LITERAL = /\bcase\s*(["'`])( *)\1\s*:/dg;

/**
 * Return the 1-based numbers of the distinct lines of `text` that dispatch on a
 * node's kind, leaving out imports, comments and string contents.
 */
function dispatchLines(text: string): Set<number> {
  let code = blank(text);
  for (const { start, end } of importStatements(text)) {
    code = code.slice(0, start) + code.slice(start, end).replace(/[^\r\n]/g, " ") + code.slice(end);
  }
  const lines = new Set<number>();
  const hit = (offset: number) => lines.add(code.slice(0, offset).split("\n").length);
  // The blanked literal keeps its offsets, so its text is read from the source;
  // `shift` is where the matched text starts in `code`.
  const isKind = (m: RegExpMatchArray, shift = 0) => {
    const [start, end] = m.indices![2]!;
    return NODE_KINDS.has(text.slice(shift + start, shift + end));
  };

  for (const pattern of KIND_REFERENCES) {
    for (const m of code.matchAll(pattern)) hit(m.index);
  }
  for (const pattern of [KIND_IS_LITERAL, LITERAL_IS_KIND]) {
    for (const m of code.matchAll(pattern)) if (isKind(m)) hit(m.index);
  }
  for (const m of code.matchAll(/\bswitch\s*\(/g)) {
    const headEnd = closing(code, m.index + m[0].length - 1);
    const open = code.indexOf("{", headEnd);
    if (!/\bkind\b/.test(code.slice(m.index, headEnd)) || open < 0) continue;
    const body = code.slice(open, closing(code, open));
    for (const c of body.matchAll(CASE_LITERAL)) if (isKind(c, open)) hit(open + c.index);
  }
  return lines;
}

describe("kind census", () => {
  test("the engine core exists and holds no kind dispatch", () => {
    // A missing core module fails, so a rename can't switch the census off.
    const missing = CORE_MODULES.filter((path) => !existsSync(new URL(path, SRC)));
    expect(missing, "engine core modules missing from src").toEqual([]);
    const sites: Record<string, number[]> = {};
    for (const path of CORE_MODULES) {
      const lines = [...dispatchLines(readFileSync(new URL(path, SRC), "utf8"))];
      if (lines.length > 0) sites[path] = lines;
    }
    // The engine core dispatches only on node traits and hooks and the closed `Outcome` sum.
    expect(sites, "kind-dispatch lines in the engine core; move kind-specific work into nodes/").toEqual({});
  });

  test("the detector counts dispatch sites", () => {
    const count = (text: string) => dispatchLines(text).size;
    expect(count("if (node.kind === NodeKind.AGENT) {}")).toBe(1);
    expect(count("if (SPAWNER_KINDS.has(k)) {}")).toBe(1);
    expect(count("if (_SPAWNER_KINDS.has(k)) {}")).toBe(1);
    expect(count("if (x instanceof CallExpansion) {}")).toBe(1);
    expect(count("if (x instanceof expansions.CallExpansion) {}")).toBe(1);
    expect(count('if (node.kind === "agent") {}')).toBe(1);
    expect(count('if ("map" !== n.kind) {}')).toBe(1);
    expect(count('const { kind } = node;\nif (kind === "agent") {}')).toBe(1);
    expect(dispatchLines('switch (node.kind) {\n  case "case":\n    break;\n}')).toEqual(new Set([2]));
    expect(count("if (NodeKind.MAP === k || x instanceof MapExpansion) {}")).toBe(1);
    expect(count('a = NodeKind.CALL;\nb = SPAWNER_KINDS;\nif (kind == "loop") {}')).toBe(3);
  });

  test("the detector skips imports, comments, strings and other sums", () => {
    const quiet = [
      'import { NodeKind } from "../nodes/contracts.ts";',
      'import {\n  NodeKind,\n  SPAWNER_KINDS,\n} from "./x.ts";',
      'export { NodeKind } from "./y.ts";',
      "// NodeKind.AGENT",
      "/* NodeKind.CASE */",
      'const s = "NodeKind.AGENT";',
      "const s = `kind ${node.kind}`;",
      'if (outcome.kind === "output") {}',
      'switch (outcome.kind) { case "route": }',
      'switch (loc.boundary) { case "start": }',
    ];
    for (const text of quiet) expect([text, dispatchLines(text).size]).toEqual([text, 0]);
  });
});
