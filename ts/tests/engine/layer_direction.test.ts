/**
 * Import-direction ratchets and source lints for `ts/src`, a port of
 * `tests/engine/test_layer_direction.py`.
 *
 * Lower layers never import the layers that drive them, a few leaves import only
 * declared modules, and engine-dependent or platform-specific APIs stay out of
 * `src` (the platform ones outside `platform.ts`). Each check is a function of
 * `{path, text}` sources, so snippets prove the rules the real tree is held to.
 */

import { describe, expect, test } from "bun:test";
import { existsSync, readdirSync, readFileSync } from "node:fs";
import { join, posix, sep } from "node:path";
import { fileURLToPath } from "node:url";

import { blank, importStatements } from "./source_scan.ts";

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
    .map((name) => ({ path: name.split(sep).join("/"), text: readFileSync(join(SRC, name), "utf8") }));
}

/** One import of a source file. */
interface ImportTarget {
  line: number;
  target: string;
  /** The trimmed source line of the specifier. */
  statement: string;
  /** Whether the specifier is relative, so `target` is a yian module id. */
  local: boolean;
}

/**
 * Return each import of `file`. A relative specifier becomes a module id: its path
 * below `src` without `.ts`/`.js`, keeping a trailing `/index` (`memory/index` is the
 * memory package's facade, `index` the package root's). The package's own name,
 * `@ngocbh/yian`, is `index` too. Others stay as written.
 */
function importTargets({ path, text }: Source): ImportTarget[] {
  const lines = text.split("\n");
  return importStatements(text).map(({ line, specifier }) => {
    const own = specifier === "@ngocbh/yian";
    const local = own || specifier.startsWith(".");
    const relative = () => posix.join(posix.dirname(path), specifier).replace(/\.[jt]s$/, "");
    return {
      line,
      target: own ? "index" : local ? relative() : specifier,
      statement: lines[line - 1]!.trim(),
      local,
    };
  });
}

/** One import-direction rule for the files `files` selects. */
interface Rule {
  name: string;
  files: (path: string) => boolean;
  /** Modules the files must not import, nor any module below them. */
  forbid?: string[];
  /** If set, the only modules (and modules below them) the files may import besides `node:` builtins. */
  allow?: string[];
  /** With `allow`, `false` bans `node:` builtins too. */
  builtins?: false;
  why: string;
  /** For the snippet tests: a file the rule covers, and module ids it must reject and accept. */
  samples: { file: string; bad: string[]; good: string[] };
}

const below = (dir: string) => (path: string) => path.startsWith(`${dir}/`);
const only = (file: string) => (path: string) => path === file;

const RULES: Rule[] = [
  {
    name: "nodes never import compile",
    files: below("nodes"),
    forbid: ["compile"],
    why: "nodes must not import compile; return a canonical Grow seed and let the injected compile-layer builder interpret it",
    samples: {
      file: "nodes/call/node.ts",
      bad: ["compile", "compile/expand", "compile/index"],
      good: ["nodes/contracts", "diagnostics", "skills", "compiled"],
    },
  },
  {
    name: "nodes never import runtime",
    files: below("nodes"),
    forbid: ["runtime"],
    why: "nodes must not import runtime; direct-driver fallbacks belong on a lower seam",
    samples: {
      file: "nodes/call/node.ts",
      bad: ["runtime", "runtime/log"],
      good: ["run", "nodes/contracts", "skills"],
    },
  },
  {
    name: "nodes/contracts.ts imports only diagnostics and typesys",
    files: only("nodes/contracts.ts"),
    allow: ["diagnostics", "typesys"],
    builtins: false,
    why: "nodes/contracts.ts may import only diagnostics and typesys",
    samples: {
      file: "nodes/contracts.ts",
      bad: ["nodes/call/node", "nodes/index", "index", "compile", "runtime", "skills", "defaults"],
      good: ["diagnostics", "typesys", "typesys/json"],
    },
  },
  {
    name: "compile never imports compose or runtime",
    files: below("compile"),
    forbid: ["compose", "runtime"],
    why: "compile must stay representation-neutral and execution-free",
    samples: {
      file: "compile/expand.ts",
      bad: ["compose", "compose/load", "runtime", "runtime/log"],
      good: ["nodes/contracts", "diagnostics", "run"],
    },
  },
  {
    name: "runtime never imports compile/expand",
    files: below("runtime"),
    forbid: ["compile/expand"],
    why: "runtime must not import compile/expand; the run boundary injects the growth builder",
    samples: {
      file: "runtime/log.ts",
      bad: ["compile/expand"],
      good: ["compile/graph", "compile/expander", "nodes/contracts"],
    },
  },
  {
    name: "runtime never imports memory_host or cli",
    files: below("runtime"),
    forbid: ["memory_host", "cli"],
    why: "runtime must not import memory_host or cli",
    samples: {
      file: "runtime/log.ts",
      bad: ["memory_host", "cli", "cli/main"],
      good: ["memory/contracts", "nodes/contracts"],
    },
  },
  {
    name: "memory/contracts.ts imports only diagnostics and defaults",
    files: only("memory/contracts.ts"),
    allow: ["diagnostics", "defaults"],
    why: "lower memory imports undeclared dependencies",
    samples: {
      file: "memory/contracts.ts",
      bad: [
        ...["nodes", "compile", "compose", "runtime", "run", "llm_clients", "index"],
        ...["memory/episodic", "memory/session", "memory/conversation", "memory/index"],
      ],
      good: ["diagnostics", "defaults"],
    },
  },
  {
    name: "memory/conversation.ts imports only memory/contracts",
    files: only("memory/conversation.ts"),
    allow: ["memory/contracts"],
    why: "lower memory imports undeclared dependencies",
    samples: {
      file: "memory/conversation.ts",
      bad: [
        ...["nodes", "compile", "compose", "runtime", "run", "llm_clients", "index", "diagnostics"],
        ...["memory/episodic", "memory/session", "memory/index"],
      ],
      good: ["memory/contracts"],
    },
  },
  {
    name: "skills imports no engine or front-end layer",
    files: below("skills"),
    forbid: ["nodes", "compile", "compose", "runtime", "run", "cli"],
    why: "skills must stay a leaf",
    samples: {
      file: "skills/load.ts",
      bad: ["nodes", "nodes/call/node", "compile", "compose", "runtime", "run", "cli"],
      good: ["diagnostics", "defaults"],
    },
  },
  {
    name: "memory never imports engine, host or cli layers",
    files: below("memory"),
    forbid: ["runtime", "compose", "compile", "run", "memory_host", "cli", "index"],
    why: "memory must not import runtime, compose, compile, run, memory_host, cli, or the package root",
    samples: {
      file: "memory/session.ts",
      bad: ["runtime", "compose", "compile", "run", "memory_host", "cli", "index"],
      good: ["memory/contracts", "memory/index", "diagnostics"],
    },
  },
  {
    name: "defaults.ts imports only node builtins",
    files: only("defaults.ts"),
    allow: [],
    why: "defaults must stay a node: builtin leaf",
    samples: {
      file: "defaults.ts",
      bad: ["diagnostics", "typesys", "index"],
      good: [],
    },
  },
  {
    name: "llm imports only defaults, platform, typesys/json and llm",
    files: below("llm"),
    allow: ["defaults", "platform", "typesys/json", "llm"],
    why: "llm may import only node: builtins, defaults, platform, typesys/json and llm",
    samples: {
      file: "llm/client.ts",
      bad: ["llm_clients", "nodes", "runtime", "index", "typesys", "typesys/other", "diagnostics"],
      good: ["defaults", "platform", "typesys/json", "llm/errors", "llm/index"],
    },
  },
];

/** Whether module id `target` is `entry` or a module below it. */
const within = (target: string, entry: string) => target === entry || target.startsWith(`${entry}/`);

/** Whether `rule` rejects importing `target`. */
function breaks(rule: Rule, { target, local }: ImportTarget): boolean {
  if (!local) return rule.allow !== undefined && (rule.builtins === false || !target.startsWith("node:"));
  if (rule.forbid?.some((entry) => within(target, entry))) return true;
  return rule.allow !== undefined && !rule.allow.some((entry) => within(target, entry));
}

/** Return `path:line: statement -> target` for each import in `files` that `rule` rejects. */
function ruleViolations(rule: Rule, files: Source[]): string[] {
  return files
    .filter((file) => rule.files(file.path))
    .flatMap((file) =>
      importTargets(file)
        .filter((t) => breaks(rule, t))
        .map((t) => `${file.path}:${t.line}: ${t.statement} -> ${t.target}`),
    );
}

/** One banned source form in the files `files` selects. */
interface Lint {
  name: string;
  files: (path: string) => boolean;
  /** Return the line and text of each banned form in `file`. */
  find: (file: Source) => { line: number; text: string }[];
  fix: string;
}

/** Return a `find` that matches the global `pattern` in code, not in comments or string contents. */
const inCode =
  (pattern: RegExp) =>
  ({ text }: Source) =>
    [...blank(text).matchAll(pattern)].map((m) => ({ line: text.slice(0, m.index).split("\n").length, text: m[0] }));

const outsidePlatform = (path: string) => path !== "platform.ts";

const BARE_SORT: Lint = {
  name: "src sorts with a comparator",
  files: () => true,
  find: inCode(/\.(?:sort|toSorted)\s*\(\s*\)/g),
  fix:
    "a bare sort compares values converted to text by UTF-16 code units, so numbers sort as text " +
    "and text doesn't sort in Python's code-point order; pass a comparator (compareCodeUnits for text)",
};

const ENGINE_DEPENDENT: Lint = {
  name: "src uses no engine-dependent API",
  files: () => true,
  // `nextTick` also counts bare, for `const { nextTick } = process` and its import.
  find: inCode(
    /(?<![\w$])(?:localeCompare|Intl|toLocale[\w$]*|Date\.parse|process\.nextTick|setImmediate)(?![\w$])|(?<![\w$.])nextTick(?![\w$])/g,
  ),
  fix:
    "results differ between engines; use no localeCompare, Intl, toLocale*, Date.parse, " +
    "nextTick or setImmediate",
};

const BARE_FETCH: Lint = {
  name: "only platform.ts references fetch",
  files: outsidePlatform,
  // Any bare `fetch` counts, a method named fetch too, so no fetch reference hides behind a
  // method of that name; a key (`{ fetch: safeFetch }`, `fetch?: F`) names no global. Blind
  // spots, all deliberate dodges: `const { fetch: f } = globalThis`, `globalThis["fetch"]`,
  // `c ? fetch : g`.
  find: inCode(/(?<![\w$.])fetch(?![\w$]|\s*\??\s*:)|(?<![\w$])globalThis\.fetch(?![\w$])/g),
  fix: "use safeFetch from platform.ts; don't name a method fetch",
};

const BUN_GLOBAL: Lint = {
  name: "only platform.ts references Bun",
  files: outsidePlatform,
  find: inCode(/(?<![\w$])Bun(?![\w$])/g),
  fix: "reach Bun through a platform.ts adapter",
};

const BUN_IMPORT: Lint = {
  name: "only platform.ts imports bun modules",
  files: outsidePlatform,
  find: (file) =>
    importTargets(file)
      .filter(({ target }) => target === "bun" || target.startsWith("bun:"))
      .map(({ line, statement }) => ({ line, text: statement })),
  fix: "reach bun modules through a platform.ts adapter",
};

const ENV_READ: Lint = {
  name: "only defaults.ts reads the process environment",
  files: (path) => path !== "defaults.ts",
  // `Bun.env` counts here too, since platform.ts may reference Bun but not read the environment.
  // A computed key on `process` counts whatever it names, as it hides which property it reads,
  // and a property named `process` (`child.process.env`) counts too.
  // Blind spots, all deliberate dodges: an alias (`const p = process; p.env`,
  // `import * as p from "node:process"`, `export * from "node:process"`),
  // `globalThis["process"]`, `(await import("node:process")).env`, and a destructure nesting
  // braces before `env` (`const { a: { b }, env } = process`).
  find: (file) => {
    const blanked = blank(file.text);
    const imports = importStatements(file.text)
      .filter(({ specifier }) => specifier === "process" || specifier === "node:process")
      .filter(({ start, end }) => /(?<![\w$])env(?![\w$])/.test(blanked.slice(start, end)))
      .map(({ start, end, line }) => ({ line, text: file.text.slice(start, end).replace(/\s+/g, " ") }));
    const reads = inCode(
      /(?<![\w$])(?:(?:process\s*\??\.|import\s*\.\s*meta\s*\.|Deno\s*\.|Bun\s*\.)\s*env(?![\w$])|process\s*\[)|\{[^{}]*(?<![\w$.])env(?![\w$])[^{}]*\}\s*=\s*(?:globalThis\s*\??\.\s*)?process(?![\w$])/g,
    )(file).map((hit) => ({ ...hit, text: hit.text.replace(/\s+/g, " ") }));
    return [...reads, ...imports].sort((a, b) => a.line - b.line);
  },
  fix:
    "adapters and transport read no environment; take an injected Env accessor (defaults.ts's " +
    "processEnv reads process.env) and document each variable read",
};

const LINTS = [BARE_SORT, ENGINE_DEPENDENT, BARE_FETCH, BUN_GLOBAL, BUN_IMPORT, ENV_READ];

/** Return each violation in `files` as `path:line: text`. */
function lintViolations(lint: Lint, files: Source[]): string[] {
  return files
    .filter((file) => lint.files(file.path))
    .flatMap((file) => lint.find(file).map((hit) => `${file.path}:${hit.line}: ${hit.text}`));
}

/** Return `""` when there are no violations, else `why` followed by one violation per line. */
function report(why: string, violations: string[]): string {
  return violations.length === 0 ? "" : `${why}:\n${violations.join("\n")}`;
}

/** Return the indexes of every match of the global `pattern` in `text`. */
function indexes(text: string, pattern: RegExp): number[] {
  return [...text.matchAll(pattern)].map((m) => m.index);
}

describe("blank", () => {
  test("keeps offsets and newline positions", () => {
    const text = [
      'const a = "x // y"; // tail',
      "/* block",
      "   comment */ const b = `t ${a} u`;",
      "const r = /[\"']/g; const s = 'q';",
    ].join("\n");
    for (const sample of [text, text.replaceAll("\n", "\r\n")]) {
      const blanked = blank(sample);
      expect(blanked.length).toBe(sample.length);
      expect(indexes(blanked, /\n/g)).toEqual(indexes(sample, /\n/g));
      expect(indexes(blanked, /\r/g)).toEqual(indexes(sample, /\r/g));
    }
  });

  test("blanks comments and the contents of strings, templates and regex literals", () => {
    expect(blank('a; // c\nb; /* d\ne */ f')).toBe("a;     \nb;     \n     f");
    expect(blank("x = \"ab\" + 'cd';")).toBe("x = \"  \" + '  ';");
    expect(blank("x = `ab`;")).toBe("x = `  `;");
    expect(blank("x = /ab/g;")).toBe("x = /  /g;");
  });

  test("regex literals with quotes or slashes open no string or comment", () => {
    expect(blank("x = /[/]'/; y = 'z';")).toBe("x = /    /; y = ' ';");
    expect(blank("x = /\\/\\//; y = 'z';")).toBe("x = /    /; y = ' ';");
    expect(blank("return /'/; y = 'z';")).toBe("return / /; y = ' ';");
    for (const regex of ["/[\"']/", "/[\\]\"']/", "/\\/\\//"]) {
      const text = `const r = ${regex};\nimport { a } from "./a.ts";`;
      expect(importStatements(text).map((i) => i.specifier)).toEqual(["./a.ts"]);
    }
    const afterKeyword = 'return /"/;\nimport { a } from "./a.ts";';
    expect(importStatements(afterKeyword).map((i) => i.specifier)).toEqual(["./a.ts"]);
  });

  test("a slash after an operand is division", () => {
    for (const code of [
      "x = a / b / c;",
      "y = a[0]! / 2 / 3;",
      "z = i++ / n / m;",
      "w = (x) / 2 / 3;",
      "v = 10 / 2 / 3;",
      "u = obj.return / 2 / 3;",
      "t = a.of / 2 / 3;",
    ]) {
      expect(blank(code)).toBe(code);
    }
    expect(blank("s = `a` / 2 / 3;")).toBe("s = ` ` / 2 / 3;");
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

  test("a semicolon-less export does not swallow the next import", () => {
    const text = 'export { a }\nimport { b } from "./b.ts";';
    const found = importStatements(text);
    expect(found).toHaveLength(1);
    expect(text.slice(found[0]!.start, found[0]!.end)).toBe('import { b } from "./b.ts"');
  });

  test("an import method on an object is no import", () => {
    expect(importStatements('foo.import("./no.ts");')).toEqual([]);
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

describe("importTargets", () => {
  const targets = (path: string, text: string) => importTargets({ path, text }).map((t) => t.target);

  test("normalizes relative spellings to module ids", () => {
    const fromNode = [
      'import { expand } from "../../compile/expand.ts";',
      'export { expand } from "../../compile/expand.js";',
      'import "../../index.ts";',
    ].join("\n");
    expect(targets("nodes/call/node.ts", fromNode)).toEqual(["compile/expand", "compile/expand", "index"]);
    const fromMemory = 'import "./index.ts";\nimport type { C } from "./contracts.ts";';
    expect(targets("memory/conversation.ts", fromMemory)).toEqual(["memory/index", "memory/contracts"]);
    expect(targets("defaults.ts", 'import "./typesys/json.ts";')).toEqual(["typesys/json"]);
  });

  test("keeps other specifiers as written", () => {
    const text = 'import { readFileSync } from "node:fs";\nimport YAML from "yaml";\nimport { Database } from "bun:sqlite";';
    expect(targets("runtime/log.ts", text)).toEqual(["node:fs", "yaml", "bun:sqlite"]);
  });

  test("gives each import's line and trimmed source line", () => {
    const text = 'const a = 1;\n  import { b } from "./b.ts";\n';
    expect(importTargets({ path: "runtime/log.ts", text })).toEqual([
      { line: 2, target: "runtime/b", statement: 'import { b } from "./b.ts";', local: true },
    ]);
  });
});

describe("layer rules", () => {
  const files = sources();
  for (const rule of RULES) {
    test(rule.name, () => {
      expect(report(rule.why, ruleViolations(rule, files))).toBe("");
    });
  }
});

/** The import spellings every rule must see. */
const FORMS: ((specifier: string) => string)[] = [
  (s) => `import { x } from "${s}";`,
  (s) => `import type { X } from "${s}";`,
  (s) => `export { x } from "${s}";`,
  (s) => `export * from "${s}";`,
  (s) => `import "${s}";`,
  (s) => `const m = await import("${s}");`,
  (s) => `type T = import("${s}").X;`,
];

/** Return the relative specifier, ending in `extension`, from `file` to module `id`. */
function spell(file: string, id: string, extension: string): string {
  const path = posix.relative(posix.dirname(file), id);
  return `${path.startsWith(".") ? "" : "./"}${path}${extension}`;
}

describe("layer rule snippets", () => {
  for (const rule of RULES) {
    test(rule.name, () => {
      const { file, bad, good } = rule.samples;
      const reported = (text: string) => ruleViolations(rule, [{ path: file, text }]).length > 0;
      const texts = (specifiers: string[]) => specifiers.flatMap((s) => FORMS.map((form) => form(s)));
      const local = (ids: string[]) => ids.flatMap((id) => [spell(file, id, ".ts"), spell(file, id, ".js")]);
      // An allow-list admits only `node:` builtins outside src (none with `builtins: false`); a
      // forbid list ignores packages.
      const packages = ["openai", "yaml", "bun:sqlite", "fs"];
      const banned = rule.allow ? (rule.builtins === false ? [...packages, "node:fs"] : packages) : [];
      const rejected = [...local(bad), ...banned];
      const accepted = [...local(good), ...["node:fs", ...packages].filter((s) => !banned.includes(s))];
      expect(texts(rejected).filter((text) => !reported(text))).toEqual([]);
      expect(texts(accepted).filter(reported)).toEqual([]);
    });
  }

  test("a violation names the file, line, statement and target", () => {
    const rule = RULES.find((r) => r.name === "nodes never import compile")!;
    const text = 'const a = 1;\nimport { expand } from "../../compile/expand.ts";';
    expect(ruleViolations(rule, [{ path: "nodes/call/node.ts", text }])).toEqual([
      'nodes/call/node.ts:2: import { expand } from "../../compile/expand.ts"; -> compile/expand',
    ]);
    expect(ruleViolations(rule, [{ path: "compile/graph.ts", text }])).toEqual([]);
  });

  test("an allowed import never exempts another on the same line", () => {
    const conversation = RULES.find((r) => r.name === "memory/conversation.ts imports only memory/contracts")!;
    const contracts = RULES.find((r) => r.name === "memory/contracts.ts imports only diagnostics and defaults")!;
    const check = (rule: Rule, path: string, text: string) =>
      ruleViolations(rule, [{ path, text }]).map((v) => v.slice(v.lastIndexOf(" -> ") + 4));
    const shared = 'import { a } from "./contracts.ts"; import { b } from "./session.ts";';
    expect(check(conversation, "memory/conversation.ts", shared)).toEqual(["memory/session"]);
    const facade = 'import "./index.ts"; export * from "./contracts.ts";';
    expect(check(conversation, "memory/conversation.ts", facade)).toEqual(["memory/index"]);
    const root = 'import "../index.ts"; import { f } from "../diagnostics.ts";';
    expect(check(contracts, "memory/contracts.ts", root)).toEqual(["index"]);
  });

  test("nodes/contracts.ts imports no node: builtin", () => {
    const rule = RULES.find((r) => r.name === "nodes/contracts.ts imports only diagnostics and typesys")!;
    const text = 'import { readFileSync } from "node:fs";';
    expect(ruleViolations(rule, [{ path: "nodes/contracts.ts", text }])).toEqual([
      'nodes/contracts.ts:1: import { readFileSync } from "node:fs"; -> node:fs',
    ]);
  });

  test("the package's own name is the root facade", () => {
    const rule = RULES.find((r) => r.name === "memory never imports engine, host or cli layers")!;
    const text = 'import { run } from "@ngocbh/yian";';
    expect(ruleViolations(rule, [{ path: "memory/session.ts", text }])).toEqual([
      'memory/session.ts:1: import { run } from "@ngocbh/yian"; -> index',
    ]);
  });

  test("entries match whole path segments of relative imports only", () => {
    const rule = (entries: Pick<Rule, "forbid" | "allow">): Rule => ({
      name: "t",
      files: () => true,
      why: "",
      samples: { file: "x.ts", bad: [], good: [] },
      ...entries,
    });
    const reported = (r: Rule, text: string) => ruleViolations(r, [{ path: "x.ts", text }]).length > 0;
    expect(reported(rule({ forbid: ["run"] }), 'import "./runtime.ts";')).toBe(false);
    expect(reported(rule({ forbid: ["runtime"] }), 'import "./run.ts";')).toBe(false);
    expect(reported(rule({ forbid: ["run"] }), 'import "./run/flow.ts";')).toBe(true);
    // An npm package named like a yian module is no yian module.
    expect(reported(rule({ forbid: ["run"] }), 'import "run";')).toBe(false);
    expect(reported(rule({ allow: ["defaults"] }), 'import "defaults";')).toBe(true);
    expect(reported(rule({ allow: ["llm"] }), 'import "./llm_clients.ts";')).toBe(true);
  });
});

describe("lints", () => {
  const files = sources();
  for (const lint of LINTS) {
    test(lint.name, () => {
      expect(report(lint.fix, lintViolations(lint, files))).toBe("");
    });
  }
});

describe("runner lints", () => {
  // The conformance runner also runs on Node, so it holds to src's Bun rules.
  const path = fileURLToPath(new URL("../conformance.ts", import.meta.url));
  const runner = { path: "../tests/conformance.ts", text: readFileSync(path, "utf8") };
  for (const lint of [BUN_GLOBAL, BUN_IMPORT]) {
    test(lint.name, () => {
      expect(report(lint.fix, lintViolations(lint, [runner]))).toBe("");
    });
  }
});

describe("lint snippets", () => {
  const hits = (lint: Lint, text: string, path = "runtime/log.ts") => lintViolations(lint, [{ path, text }]);

  test("a bare sort is reported with its line, also in platform.ts", () => {
    const forms = ["xs.sort()", "xs.toSorted( )", "xs.sort ()", "xs\n  .sort()"];
    expect(hits(BARE_SORT, forms.join(";\n"))).toEqual([
      "runtime/log.ts:1: .sort()",
      "runtime/log.ts:2: .toSorted( )",
      "runtime/log.ts:3: .sort ()",
      "runtime/log.ts:5: .sort()",
    ]);
    expect(hits(BARE_SORT, "xs.sort()", "platform.ts")).toEqual(["platform.ts:1: .sort()"]);
    for (const text of ["xs.sort(compareCodeUnits);", "xs.sort((a, b) => a - b);", "xs.toSorted(compareCodeUnits);"]) {
      expect(hits(BARE_SORT, text)).toEqual([]);
    }
  });

  test("each engine-dependent form is reported with its line", () => {
    const forms = [
      "a.localeCompare(b)",
      "new Intl.Collator()",
      "n.toLocaleString()",
      "d.toLocaleDateString()",
      "Date.parse(s)",
      "process.nextTick(f)",
      "setImmediate(f)",
      "const { nextTick } = process",
      'import { nextTick } from "node:process"',
    ];
    expect(hits(ENGINE_DEPENDENT, forms.join(";\n"))).toEqual([
      "runtime/log.ts:1: localeCompare",
      "runtime/log.ts:2: Intl",
      "runtime/log.ts:3: toLocaleString",
      "runtime/log.ts:4: toLocaleDateString",
      "runtime/log.ts:5: Date.parse",
      "runtime/log.ts:6: process.nextTick",
      "runtime/log.ts:7: setImmediate",
      "runtime/log.ts:8: nextTick",
      "runtime/log.ts:9: nextTick",
    ]);
    expect(hits(ENGINE_DEPENDENT, "Date.parse(s)", "platform.ts")).toEqual(["platform.ts:1: Date.parse"]);
  });

  test("strings, comments and other names pass; template code doesn't", () => {
    for (const text of [
      'const s = "use localeCompare";',
      "// Intl",
      "/* Date.parse(s) */",
      "const t = `toLocaleString`;",
      "const parse = myDate.parse(s);",
      "queue.nextTick(f);",
      "const nextTicks = 1;",
    ]) {
      expect(hits(ENGINE_DEPENDENT, text)).toEqual([]);
    }
    expect(hits(ENGINE_DEPENDENT, "const s = `n: ${x.toLocaleString()}`;")).toEqual([
      "runtime/log.ts:1: toLocaleString",
    ]);
  });

  test("a bare fetch reference is reported; safeFetch, x.fetch and a fetch key pass", () => {
    const text = [
      "await fetch(url);",
      "await fetch (url);",
      "globalThis.fetch(url);",
      "const f = globalThis.fetch;",
      "const g = fetch;",
      "class C { fetch(u) {} }",
      "interface I { fetch(u: string): void }",
      "function fetch(u) {}",
    ];
    expect(hits(BARE_FETCH, text.join("\n"))).toEqual([
      "runtime/log.ts:1: fetch",
      "runtime/log.ts:2: fetch",
      "runtime/log.ts:3: globalThis.fetch",
      "runtime/log.ts:4: globalThis.fetch",
      "runtime/log.ts:5: fetch",
      "runtime/log.ts:6: fetch",
      "runtime/log.ts:7: fetch",
      "runtime/log.ts:8: fetch",
    ]);
    // Only the key: in `fetch?: typeof fetch` the `typeof fetch` is still reported.
    const keys = ["new Transport(url, { fetch: safeFetch });", "interface O { fetch?: F }", "const o = { fetch : f };"];
    const others = ["safeFetch(url);", "client.fetch(x);", "client?.fetch(x);", 'const s = "fetch(url)";', "prefetch(x);"];
    for (const ok of [...keys, ...others]) {
      expect(hits(BARE_FETCH, ok)).toEqual([]);
    }
  });

  test("fetch, Bun and bun modules pass in platform.ts and fail elsewhere", () => {
    for (const [lint, text] of [
      [BARE_FETCH, "await fetch(url);"],
      [BUN_GLOBAL, "const f = Bun.file(path);"],
      [BUN_IMPORT, 'import { Database } from "bun:sqlite";'],
      [BUN_IMPORT, 'import { file } from "bun";'],
    ] as const) {
      expect(hits(lint, text, "platform.ts")).toEqual([]);
      expect(hits(lint, text)).toHaveLength(1);
    }
    expect(hits(BUN_IMPORT, 'const a = 1;\nimport { Database } from "bun:sqlite";')).toEqual([
      'runtime/log.ts:2: import { Database } from "bun:sqlite";',
    ]);
    for (const ok of ['type F = BunFile;', 'const s = "Bun.file";', 'import { x } from "./bun.ts";']) {
      expect([...hits(BUN_GLOBAL, ok), ...hits(BUN_IMPORT, ok)]).toEqual([]);
    }
  });

  test("each process environment read is reported with its line, also in platform.ts", () => {
    const forms = [
      "const a = process.env.YIAN_X;",
      "const b = process?.env;",
      'const c = process["env"];',
      "const d = globalThis.process.env;",
      "const e = import.meta.env.YIAN_X;",
      'const f = Deno.env.get("YIAN_X");',
      "const g = Bun.env.YIAN_X;",
      "const { env } = process;",
      'import { env } from "node:process";',
      'import { argv, env as vars } from "process";',
      'export { env } from "node:process";',
      'import {\n  env,\n} from "node:process";',
      "const { env: vars } = globalThis.process;",
      "const {\n  argv,\n  env,\n} = process;",
    ];
    expect(hits(ENV_READ, forms.join("\n"))).toEqual([
      "runtime/log.ts:1: process.env",
      "runtime/log.ts:2: process?.env",
      "runtime/log.ts:3: process[",
      "runtime/log.ts:4: process.env",
      "runtime/log.ts:5: import.meta.env",
      "runtime/log.ts:6: Deno.env",
      "runtime/log.ts:7: Bun.env",
      "runtime/log.ts:8: { env } = process",
      'runtime/log.ts:9: import { env } from "node:process"',
      'runtime/log.ts:10: import { argv, env as vars } from "process"',
      'runtime/log.ts:11: export { env } from "node:process"',
      'runtime/log.ts:14: import { env, } from "node:process"',
      "runtime/log.ts:15: { env: vars } = globalThis.process",
      "runtime/log.ts:16: { argv, env, } = process",
    ]);
    expect(hits(ENV_READ, "const a = process.env.YIAN_X;", "platform.ts")).toEqual(["platform.ts:1: process.env"]);
  });

  test("defaults.ts reads the process environment; comments, strings and other names pass", () => {
    const read = 'import process from "node:process";\nconst a = process.env.YIAN_X;';
    expect(hits(ENV_READ, read, "defaults.ts")).toEqual([]);
    expect(hits(ENV_READ, read)).toEqual(["runtime/log.ts:2: process.env"]);
    for (const ok of [
      "// reads process.env",
      "/* import.meta.env */",
      'const s = "process.env.YIAN_X";',
      "const t = `Deno.env`;",
      'import { argv } from "node:process";',
      'import process from "node:process";',
      'import { env } from "./env.ts";',
      "const { env } = options;",
      "const v = config.env;",
      'const w = processEnv("YIAN_X");',
    ]) {
      expect(hits(ENV_READ, ok)).toEqual([]);
    }
  });
});
