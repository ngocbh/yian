/**
 * Kind-dispatch census guarding the kind-agnostic engine core, a port of
 * `tests/engine/test_kind_census.py`.
 *
 * The engine core (`runtime/engine.ts` + `runtime/eval_node.ts`) knows only an
 * abstract node contract and the closed `Outcome` sum; it must not branch on a
 * node's kind nor on a concrete `*Expansion` type. The census counts the lines
 * that do and holds the count at the ceiling, which is zero. It reads the source,
 * so it survives line-number drift.
 *
 * A kind-dispatch line holds any of, outside imports, comments and strings:
 *   - a `NodeKind` member (`NodeKind.AGENT`);
 *   - `SPAWNER_KINDS` or `_SPAWNER_KINDS`, the spawner-kind set;
 *   - `instanceof <Name>Expansion`, a branch on a concrete expansion type;
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
 * (`desc.constructor.name === "CallExpansion"`). The `*Expansion` check relies on
 * every expansion type's name ending in `Expansion`.
 *
 * Outside the core the literal check would misread `ExecutionPosition.kind ===
 * "call"`, a position kind rather than a node kind; the census doesn't scan there.
 */

import { describe, expect, test } from "bun:test";
import { existsSync, readFileSync } from "node:fs";

import { blank, importStatements } from "./source_scan.ts";

const SRC = new URL("../../src/", import.meta.url);

/** The two modules, below `src`, that make up the engine core the census guards. */
const CORE_MODULES = ["runtime/engine.ts", "runtime/eval_node.ts"];

/** The engine core dispatches only on node traits and hooks and the closed `Outcome` sum. */
const KIND_DISPATCH_CEILING = 0;

/** The values of `NodeKind`. */
const NODE_KINDS = new Set([
  "agent",
  "code",
  "tool",
  "case",
  "human_input",
  "wait",
  "loop",
  "start",
  "end",
  "call",
  "map",
]);

/**
 * Return the core modules that are missing and must exist. Before the engine
 * lands (no core module and no `run.ts`) none must; after, both must, so a
 * rename can't switch the census off.
 */
function missingCore(exists: (path: string) => boolean): string[] {
  if (!exists("run.ts") && !CORE_MODULES.some((path) => exists(path))) return [];
  return CORE_MODULES.filter((path) => !exists(path));
}

/** Kind references that count wherever they appear in code. */
const KIND_REFERENCES = [
  /\bNodeKind\.[A-Za-z_]/g,
  /\b_?SPAWNER_KINDS\b/g,
  /\binstanceof\s+[A-Za-z_$][\w$]*Expansion\b/g,
];

// Over blanked text a literal is its quotes around spaces; group 2 spans its contents.
const KIND_IS_LITERAL = /\bkind\s*[!=]==?\s*(["'`])( *)\1/g;
const LITERAL_IS_KIND = /(["'`])( *)\1\s*[!=]==?\s*(?:[\w$]+[?!]?\.)*kind(?![\w$])/g;
const CASE_LITERAL = /\bcase\s*(["'`])( *)\1\s*:/g;

/** Return the index of the bracket closing the `(` or `{` at `open` in `code`. */
function closing(code: string, open: number): number {
  const [opener, closer] = code[open] === "(" ? ["(", ")"] : ["{", "}"];
  let depth = 0;
  for (let i = open; i < code.length; i++) {
    if (code[i] === opener) depth++;
    else if (code[i] === closer && --depth === 0) return i;
  }
  return code.length;
}

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
  // The blanked literal keeps its offsets, so its text is read from the source.
  const isKind = (quote: number, length: number) => NODE_KINDS.has(text.slice(quote + 1, quote + 1 + length));

  for (const pattern of KIND_REFERENCES) {
    for (const m of code.matchAll(pattern)) hit(m.index);
  }
  for (const m of code.matchAll(KIND_IS_LITERAL)) {
    const length = m[2]!.length;
    if (isKind(m.index + m[0].length - length - 2, length)) hit(m.index);
  }
  for (const m of code.matchAll(LITERAL_IS_KIND)) {
    if (isKind(m.index, m[2]!.length)) hit(m.index);
  }
  for (const m of code.matchAll(/\bswitch\s*\(/g)) {
    const headEnd = closing(code, m.index + m[0].length - 1);
    const open = code.indexOf("{", headEnd);
    if (!/\bkind\b/.test(code.slice(m.index, headEnd)) || open < 0) continue;
    const body = code.slice(open, closing(code, open));
    for (const c of body.matchAll(CASE_LITERAL)) {
      const quote = open + c.index + c[0].indexOf(c[1]!);
      if (isKind(quote, c[2]!.length)) hit(open + c.index);
    }
  }
  return lines;
}

const coreExists = (path: string) => existsSync(new URL(path, SRC));

/** Return each existing core module's path mapped to its sorted kind-dispatch lines. */
function census(): Record<string, number[]> {
  return Object.fromEntries(
    CORE_MODULES.filter(coreExists).map((path) => [
      path,
      [...dispatchLines(readFileSync(new URL(path, SRC), "utf8"))].sort((a, b) => a - b),
    ]),
  );
}

/** Return `""` when no core module is missing, else a message naming the missing ones. */
function missingReport(): string {
  const missing = missingCore(coreExists);
  return missing.length === 0 ? "" : `engine core modules missing from src: ${missing.join(", ")}`;
}

const total = (sites: Record<string, number[]>) =>
  Object.values(sites).reduce((sum, lines) => sum + lines.length, 0);

describe("kind census", () => {
  test("kind dispatch stays at or below the ceiling", () => {
    expect(missingReport()).toBe("");
    const sites = census();
    const count = total(sites);
    expect(
      count <= KIND_DISPATCH_CEILING
        ? ""
        : `kind-dispatch census rose to ${count} (ceiling ${KIND_DISPATCH_CEILING}). ` +
            `The engine core must not gain kind dispatch. Sites: ${JSON.stringify(sites)}`,
    ).toBe("");
  });

  test("the ceiling matches the observed count", () => {
    expect(missingReport()).toBe("");
    const sites = census();
    const count = total(sites);
    expect(
      count === KIND_DISPATCH_CEILING
        ? ""
        : `kind-dispatch census is ${count} but KIND_DISPATCH_CEILING is ${KIND_DISPATCH_CEILING}. ` +
            `Set the ceiling to ${count} to keep the ratchet tight. Sites: ${JSON.stringify(sites)}`,
    ).toBe("");
  });

  test("missing core files are allowed only before the engine exists", () => {
    const over = (paths: string[]) => missingCore((path) => paths.includes(path));
    expect(over([])).toEqual([]);
    expect(over(["run.ts"])).toEqual(["runtime/engine.ts", "runtime/eval_node.ts"]);
    expect(over(["runtime/engine.ts"])).toEqual(["runtime/eval_node.ts"]);
    expect(over(["runtime/engine.ts", "runtime/eval_node.ts"])).toEqual([]);
  });

  test("the detector counts dispatch sites", () => {
    const count = (text: string) => dispatchLines(text).size;
    expect(count("if (node.kind === NodeKind.AGENT) {}")).toBe(1);
    expect(count("if (SPAWNER_KINDS.has(k)) {}")).toBe(1);
    expect(count("if (_SPAWNER_KINDS.has(k)) {}")).toBe(1);
    expect(count("if (x instanceof CallExpansion) {}")).toBe(1);
    expect(count('if (node.kind === "agent") {}')).toBe(1);
    expect(count('if ("map" !== n.kind) {}')).toBe(1);
    expect(count('const { kind } = node;\nif (kind === "agent") {}')).toBe(1);
    expect(dispatchLines('switch (node.kind) {\n  case "case":\n    break;\n}')).toEqual(new Set([2]));
    expect(count("if (NodeKind.MAP === k || x instanceof MapExpansion) {}")).toBe(1);
    expect(count('a = NodeKind.CALL;\nb = SPAWNER_KINDS;\nif (kind == "wait") {}')).toBe(3);
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
