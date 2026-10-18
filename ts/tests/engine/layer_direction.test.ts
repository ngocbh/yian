/**
 * Import-direction ratchets and source lints for `ts/src`, a port of
 * `tests/engine/test_layer_direction.py`.
 *
 * These tests pin the source scanner the rules read imports through.
 */

import { describe, expect, test } from "bun:test";

import { blank, importStatements } from "./source_scan.ts";

/** Return the indexes of every newline in `text`. */
function newlines(text: string): number[] {
  return [...text.matchAll(/\n/g)].map((m) => m.index);
}

describe("blank", () => {
  test("keeps offsets and newline positions", () => {
    const text = [
      'const a = "x // y"; // tail',
      "/* block",
      "   comment */ const b = `t ${a} u`;",
      "const r = /[\"']/g; const s = 'q';",
    ].join("\n");
    const blanked = blank(text);
    expect(blanked.length).toBe(text.length);
    expect(newlines(blanked)).toEqual(newlines(text));
  });

  test("blanks comments and the contents of strings, templates and regex literals", () => {
    expect(blank('a; // c\nb; /* d\ne */ f')).toBe("a;     \nb;     \n     f");
    expect(blank("x = \"ab\" + 'cd';")).toBe("x = \"  \" + '  ';");
    expect(blank("x = `ab`;")).toBe("x = `  `;");
    expect(blank("x = /ab/g;")).toBe("x = /  /g;");
  });

  test("regex literals with quotes or slashes open no string or comment", () => {
    for (const regex of ["/[\"']/", "/[\\]\"']/", "/\\/\\//"]) {
      const text = `const r = ${regex};\nimport { a } from "./a.ts";`;
      expect(importStatements(text).map((i) => i.specifier)).toEqual(["./a.ts"]);
    }
    const afterKeyword = 'return /"/;\nimport { a } from "./a.ts";';
    expect(importStatements(afterKeyword).map((i) => i.specifier)).toEqual(["./a.ts"]);
  });

  test("a slash after an operand is division", () => {
    for (const code of ["x = a / b / c;", "y = a[0]! / 2 / 3;", "z = i++ / n / m;", "w = (x) / 2 / 3;"]) {
      expect(blank(code)).toBe(code);
    }
  });

  test("comment openers inside a string start no comment", () => {
    expect(blank('"a // b"; c')).toBe('"      "; c');
    expect(blank('"a /* b"; c')).toBe('"      "; c');
  });

  test("an escaped quote does not end a string", () => {
    expect(blank('"a\\"b"; c')).toBe('"    "; c');
    expect(blank("'a\\'b'; c")).toBe("'    '; c");
  });

  test("code inside a template substitution stays code, also when nested", () => {
    expect(blank('`a ${b + "c"} d`')).toBe('`  ${b + " "}  `');
    expect(blank("`a ${`b ${c}`} d`")).toBe("`  ${`  ${c}`}  `");
    expect(blank("`a ${ {k: 1}.k } d`")).toBe("`  ${ {k: 1}.k }  `");
  });

  test("an unterminated string or regex ends at its line's end", () => {
    expect(blank('x = "abc\ny = 1;')).toBe('x = "   \ny = 1;');
    expect(blank("x = /abc\ny = 2;")).toBe("x = /   \ny = 2;");
  });
});

describe("importStatements", () => {
  test("finds each import and export form with its specifier and line", () => {
    const forms = [
      'import { expand } from "../../compile/expand.ts";',
      'import type { X } from "./types.ts";',
      'import * as e from "./e.ts";',
      'export { x } from "./x.ts";',
      'export * from "./all.ts";',
      'export type { T } from "./t.ts";',
      'import "./side.ts";',
      'const m = await import("./dynamic.ts");',
      'type T = import("./typepos.ts").Y;',
    ];
    const found = importStatements(forms.join("\n"));
    expect(found.map((i) => [i.line, i.specifier])).toEqual([
      [1, "../../compile/expand.ts"],
      [2, "./types.ts"],
      [3, "./e.ts"],
      [4, "./x.ts"],
      [5, "./all.ts"],
      [6, "./t.ts"],
      [7, "./side.ts"],
      [8, "./dynamic.ts"],
      [9, "./typepos.ts"],
    ]);
  });

  test("start and end span the statement up to the closing quote", () => {
    const text = 'const z = 1;\nimport { expand } from "../../compile/expand.ts";\n';
    const [found] = importStatements(text);
    expect(found).toBeDefined();
    expect(text.slice(found!.start, found!.end)).toBe('import { expand } from "../../compile/expand.ts"');
  });

  test("a multi-line import reports the specifier's line and spans every line", () => {
    const text = 'const z = 1;\nimport {\n  a,\n  b,\n} from "./ab.ts";\n';
    const found = importStatements(text);
    expect(found).toHaveLength(1);
    expect(found[0]!.line).toBe(5);
    expect(found[0]!.specifier).toBe("./ab.ts");
    expect(text.slice(found[0]!.start, found[0]!.end)).toBe('import {\n  a,\n  b,\n} from "./ab.ts"');
  });

  test("two statements on one line give two entries", () => {
    const found = importStatements('import { a } from "./a.ts"; import { b } from "./b.ts";');
    expect(found.map((i) => [i.line, i.specifier])).toEqual([
      [1, "./a.ts"],
      [1, "./b.ts"],
    ]);
  });

  test("imports inside comments, strings and templates are ignored", () => {
    const text = [
      '// import { a } from "./a.ts";',
      '/* import { b } from "./b.ts"; */',
      "const s = \"import x from './y.ts'\";",
      'const t = `import { c } from "./c.ts"`;',
    ].join("\n");
    expect(importStatements(text)).toEqual([]);
  });
});
