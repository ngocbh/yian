/**
 * A code root on disk and the routes a module inside it could take to load code from outside it,
 * for the tests that hold a tool module's and a CODE step's imports to the root: each route as the
 * module text that takes it and the refusal it meets on each runtime, and the imports that stay
 * inside, each with the value it binds.
 *
 * `codeRootTree` builds, in a fresh folder: `root/`, the code root, holding `inner.ts`, a JSON and a
 * text file, a file named like an addon, a tsconfig mapping `@out/*` to `outside/`, a symlink to
 * `inner.ts`, symlinks out to `outside/` (one an extensionless addon's), `lib/a.ts` importing
 * outside, and two packages in `root/node_modules` (an ES module importing its own file, and a
 * CommonJS one); beside it, `outside/`, a `node_modules` holding packages the root finds only by
 * walking up (one an addon), and `rootlink`, a symlink to the root.
 */

import { mkdirSync, mkdtempSync, realpathSync, symlinkSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { pathToFileURL } from "node:url";

/** The folders of a tree `codeRootTree` built. */
export interface CodeRootTree {
  /** The folder holding the others; its removal removes the tree. */
  readonly base: string;
  /** The code root. */
  readonly root: string;
  /** `outside/evil.ts`, outside the root. */
  readonly evil: string;
}

/** Write each file of `files`, by its path under `dir`, making its folders. */
function writeAll(dir: string, files: Readonly<Record<string, string>>): void {
  for (const [path, text] of Object.entries(files)) {
    mkdirSync(dirname(join(dir, path)), { recursive: true });
    writeFileSync(join(dir, path), text);
  }
}

/** Build the tree in a fresh folder under the system's temp folder, all paths real. */
export function codeRootTree(): CodeRootTree {
  const base = realpathSync(mkdtempSync(join(tmpdir(), "yian-code-root-")));
  writeAll(base, {
    "outside/evil.ts": 'export const x = "OUTSIDE";\n',
    "outside/types.ts": 'export type Foo = string;\nthrow new Error("types.ts was loaded");\n',
    "outside/dir/e.ts": 'export const x = "OUTSIDE-DIR";\n',
    "outside/note.txt": "OUTSIDE-TEXT",
    "outside/evil.cjs": 'module.exports = { x: "OUTSIDE-CJS" };\n',
    // Not a real addon: a refused one never reaches `dlopen`, and one let through fails in it.
    "outside/addon.node": "not an addon\n",
    "node_modules/yian-walkup/package.json": '{"name":"yian-walkup","type":"module","main":"index.js"}\n',
    "node_modules/yian-walkup/index.js": 'export const x = "WALKUP";\n',
    "node_modules/yian-addon/package.json": '{"name":"yian-addon","main":"addon.node"}\n',
    "node_modules/yian-addon/addon.node": "not an addon\n",
    "root/inner.ts": 'export const x = "INNER";\n',
    "root/data.json": '{"x":"JSON"}\n',
    "root/note.txt": "TEXT",
    "root/inside.node": "not an addon\n",
    // Bun maps these for `require` and `import`; Node reads no tsconfig.
    "root/tsconfig.json": '{"compilerOptions":{"paths":{"@out/*":["../outside/*"]}}}\n',
    "root/lib/a.ts": 'import { x } from "../../outside/evil.ts";\nexport const got = x;\n',
    "root/node_modules/yian-inner/package.json": '{"name":"yian-inner","type":"module","main":"index.js"}\n',
    "root/node_modules/yian-inner/index.js": 'export { x } from "./lib.js";\n',
    "root/node_modules/yian-inner/lib.js": 'export const x = "PACKAGE";\n',
    "root/node_modules/yian-cjs/package.json": '{"name":"yian-cjs","main":"index.js"}\n',
    "root/node_modules/yian-cjs/index.js": 'module.exports = { x: require("./lib.js") };\n',
    "root/node_modules/yian-cjs/lib.js": 'module.exports = "COMMONJS";\n',
  });
  symlinkSync("inner.ts", join(base, "root/alias.ts"));
  symlinkSync("../outside/evil.ts", join(base, "root/link.ts"));
  symlinkSync("../outside/dir", join(base, "root/linkdir"));
  symlinkSync("../outside/addon.node", join(base, "root/nlink.node"));
  symlinkSync("../outside/evil.cjs", join(base, "root/link.cjs"));
  symlinkSync("root", join(base, "rootlink"));
  return { base, root: join(base, "root"), evil: join(base, "outside/evil.ts") };
}

/** How a route's import is refused on each runtime; `null` where the runtime has no such route. */
export interface Refusal {
  readonly bun: string | RegExp | null;
  readonly node: string | RegExp | null;
}

/** One route out of the root: the module text that takes it, binding `got`, and its refusal. */
export interface EscapeRoute {
  readonly name: string;
  readonly source: (tree: CodeRootTree) => string;
  readonly refusal: Refusal;
}

/** What a refusal of `entry.ts` importing `specifier` reads, on both runtimes. */
function named(specifier: string): Refusal {
  const message = `entry.ts imports '${specifier}', which resolves outside the code root`;
  return { bun: message, node: message };
}

/** A refusal of `entry.ts` importing `outside/evil.ts` by its absolute path, as a path or a `file:` URL. */
const ABSOLUTE = /entry\.ts imports '\/[^']*\/outside\/evil\.ts', which resolves outside the code root/;
const FILE_URL = /entry\.ts imports 'file:\/\/\/[^']*\/outside\/evil\.ts', which resolves outside the code root/;

/** What a refusal of a URL of `scheme` reads on each runtime: Bun can't tell who imported it. */
function url(scheme: string): Refusal {
  const why = `a '${scheme}:' URL, which isn't a file inside the code root`;
  return { bun: `the code imports ${why}`, node: `entry.ts imports ${why}` };
}

/** A computed specifier, so no static scan could see it. */
const computed = (path: string) => `${JSON.stringify(path.slice(0, 3))} + ${JSON.stringify(path.slice(3))}`;

/** What a refusal of loading `outside`'s file `name` reads, when no importer or import is known. */
const loads = (name: string) => `the code loads '../outside/${name}', which is outside the code root`;

/** What calling `name`, which could undo the guard, throws under a code root. */
export const LOCKED = (name: string) => `${name} is off under a code root: it could undo the guard on imports`;

/** Module text that loads `outside/evil.cjs` through a fresh `Module` and `Module._extensions[".js"]`. */
const moduleExtensions = (tree: CodeRootTree) =>
  `import Module from "node:module";\nconst f = ${JSON.stringify(join(tree.base, "outside/evil.cjs"))};\n` +
  "const m: any = new (Module as any)(f, null);\nm.filename = f;\nm.paths = [];\n" +
  '(Module as any)._extensions[".js"](m, f);\nconst got = m.exports.x;\n';

/**
 * Module text that runs `attempt` with a built-in patched by `patch`, which defines `restore`, and
 * restores it whatever happens, so a pooled CODE thread's later calls see the built-in again.
 */
const patched = (patch: string, attempt: string, got = "PATCHED") =>
  `${patch}\ntry {\n${attempt}\n} finally {\n  restore();\n}\nconst got = ${JSON.stringify(got)};\n`;

/** Module text binding `req` (a `require` of this module) and `M` (`node:module`). */
const REQUIRE = 'import { createRequire } from "node:module";\nconst req = createRequire(import.meta.url);\nconst M: any = req("node:module");\n';

/** Module text that patches `Reflect.apply` to keep each function it is handed in `seen`. */
const CATCH_APPLY =
  "const apply = Reflect.apply;\nconst seen: Function[] = [];\n" +
  "Reflect.apply = function (f: Function, self: unknown, args: ArrayLike<unknown>) { seen.push(f); return apply(f, self, args); } as typeof Reflect.apply;\n" +
  "const restore = () => { Reflect.apply = apply; };";

/** Module text that patches `fs.realpathSync`, `.native` too, to return its path as it is. */
const SAME_PATH =
  'import fs from "node:fs";\nconst realpath = fs.realpathSync;\n' +
  "(fs as any).realpathSync = Object.assign((p: string) => p, { native: (p: string) => p });\n" +
  "const restore = () => { (fs as any).realpathSync = realpath; };";

/** Module text that patches `String.prototype.startsWith` to say no path starts with `../`. */
const NO_PARENT =
  "const startsWith = String.prototype.startsWith;\n" +
  'String.prototype.startsWith = function (this: string, prefix: string, at?: number) { return prefix === "../" || prefix === "..\\\\" ? false : startsWith.call(this, prefix, at); };\n' +
  "const restore = () => { String.prototype.startsWith = startsWith; };";

/** Module text that patches `node:module`'s export `name` of `node:path` or `node:url` and syncs it into named imports. */
const SYNCED = (module: string, name: string, value: string) =>
  REQUIRE +
  `const patchedModule: any = req(${JSON.stringify(module)});\nconst original = patchedModule.${name};\n` +
  `patchedModule.${name} = ${value};\nM.syncBuiltinESMExports();\n` +
  "const restore = () => { patchedModule." + name + " = original; M.syncBuiltinESMExports(); };";

/** Module text that tries to take the guard away, on either runtime, and goes on when it can't. */
const UNDO = 'try {\n  if (typeof Bun !== "undefined") Bun.plugin.clearAll();\n' +
  '  else (await import("node:module")).registerHooks({ resolve: (s: string, c: any, n: any) => n(s, c) });\n} catch {}\n';

export const ESCAPE_ROUTES: readonly EscapeRoute[] = [
  {
    name: "a static ../ import",
    source: () => 'import { x } from "../outside/evil.ts";\nexport const got = x;\n',
    refusal: named("../outside/evil.ts"),
  },
  {
    name: "a re-export",
    source: () => 'export { x } from "../outside/evil.ts";\nconst got = "RE-EXPORTED";\n',
    refusal: named("../outside/evil.ts"),
  },
  {
    name: "a computed import()",
    source: () => `export const got = (await import(${computed("../outside/evil.ts")})).x;\n`,
    refusal: named("../outside/evil.ts"),
  },
  {
    name: "createRequire",
    source: () =>
      'import { createRequire } from "node:module";\n' +
      `export const got = createRequire(import.meta.url)(${computed("../outside/evil.ts")}).x;\n`,
    refusal: named("../outside/evil.ts"),
  },
  {
    name: "Bun's global require",
    source: () => `export const got = require(${computed("../outside/evil.ts")}).x;\n`,
    refusal: { ...named("../outside/evil.ts"), node: null },
  },
  {
    name: "Bun's import.meta.require",
    source: () => `export const got = import.meta.require(${computed("../outside/evil.ts")}).x;\n`,
    refusal: { ...named("../outside/evil.ts"), node: null },
  },
  {
    name: "a file symlink to outside",
    source: () => 'import { x } from "./link.ts";\nexport const got = x;\n',
    refusal: named("./link.ts"),
  },
  {
    name: "a folder symlink to outside",
    source: () => 'import { x } from "./linkdir/e.ts";\nexport const got = x;\n',
    refusal: named("./linkdir/e.ts"),
  },
  {
    name: "an absolute path outside",
    source: (tree) => `export const got = (await import(${JSON.stringify(tree.evil)})).x;\n`,
    refusal: { bun: ABSOLUTE, node: ABSOLUTE },
  },
  {
    name: "a file: URL outside",
    source: (tree) => `export const got = (await import(${JSON.stringify(pathToFileURL(tree.evil).href)})).x;\n`,
    // Bun hands a plugin the URL as its path.
    refusal: { bun: ABSOLUTE, node: FILE_URL },
  },
  {
    name: "a text import outside",
    source: () => 'import note from "../outside/note.txt" with { type: "text" };\nconst got = note;\n',
    refusal: named("../outside/note.txt"),
  },
  {
    name: "an addon in a parent folder's node_modules, by require",
    source: () =>
      'import { createRequire } from "node:module";\nconst addon = createRequire(import.meta.url)("yian-addon");\nconst got = typeof addon;\n',
    // Bun's CommonJS loader opens an addon by `process.dlopen`, past its plugins.
    refusal: {
      bun: "the code loads '../node_modules/yian-addon/addon.node', which is outside the code root",
      node: "entry.ts imports 'yian-addon', which resolves outside the code root",
    },
  },
  {
    name: "an addon through an extensionless symlink",
    source: () =>
      'import { createRequire } from "node:module";\nconst addon = createRequire(import.meta.url)("./nlink");\nconst got = typeof addon;\n',
    refusal: { bun: loads("addon.node"), node: named("./nlink").node },
  },
  {
    name: "an addon through tsconfig paths",
    source: () => 'const addon = require("@out/addon.node");\nconst got = typeof addon;\n',
    refusal: { bun: loads("addon.node"), node: null },
  },
  {
    name: "process.dlopen of a file outside",
    source: (tree) =>
      `process.dlopen({ exports: {} } as any, ${JSON.stringify(join(tree.base, "outside/addon.node"))});\nconst got = "OPENED";\n`,
    refusal: { bun: loads("addon.node"), node: loads("addon.node") },
  },
  {
    name: "Module._extensions on a fresh Module",
    source: moduleExtensions,
    refusal: { bun: loads("evil.cjs"), node: loads("evil.cjs") },
  },
  {
    name: "Module.prototype.load",
    source: (tree) => moduleExtensions(tree).replace('(Module as any)._extensions[".js"](m, f);', "m.load(f);"),
    // Bun's `Module` has no `load`.
    refusal: { bun: null, node: loads("evil.cjs") },
  },
  {
    name: "Bun.plugin.clearAll",
    source: () => 'Bun.plugin.clearAll();\nconst got = "CLEARED";\n',
    refusal: { bun: LOCKED("Bun.plugin.clearAll"), node: null },
  },
  {
    name: "a resolve hook of the module's own",
    source: () =>
      'import { registerHooks } from "node:module";\nregisterHooks({ resolve: (s: string, c: any, n: any) => n(s, c) });\nconst got = "HOOKED";\n',
    refusal: { bun: null, node: LOCKED("module.registerHooks") },
  },
  {
    name: "an async loader of the module's own",
    source: () => 'import { register } from "node:module";\nregister("data:text/javascript,");\nconst got = "REGISTERED";\n',
    refusal: { bun: null, node: LOCKED("module.register") },
  },
  {
    name: "a ../ import after trying to take the guard away",
    source: () => UNDO + `const got = (await import(${computed("../outside/evil.ts")})).x;\n`,
    refusal: named("../outside/evil.ts"),
  },
  {
    name: "a patched Reflect.apply handed what process.dlopen calls",
    source: (tree) => {
      const inside = JSON.stringify(join(tree.root, "inside.node"));
      const out = JSON.stringify(join(tree.base, "outside/addon.node"));
      // A native dlopen that leaked would open the file outside: "file too short".
      return patched(CATCH_APPLY, `  try { process.dlopen({ exports: {} } as any, ${inside}); } catch {}`) +
        `for (const f of seen) f.call(process, { exports: {} }, ${out});\nprocess.dlopen({ exports: {} } as any, ${out});\n`;
    },
    refusal: { bun: loads("addon.node"), node: loads("addon.node") },
  },
  {
    name: "a patched Reflect.apply handed what Module._extensions calls",
    source: (tree) => {
      const out = JSON.stringify(join(tree.base, "outside/evil.cjs"));
      const fresh = `const m: any = new M(${out}, null);\n  m.filename = ${out};\n  m.paths = [];`;
      return REQUIRE +
        patched(CATCH_APPLY, '  delete req.cache[req.resolve("yian-cjs")];\n  req("yian-cjs");') +
        `for (const load of seen) {\n  ${fresh}\n  try { load.call(M._extensions, m, ${out}); } catch { continue; }\n` +
        '  if (m.exports?.x === "OUTSIDE-CJS") throw new Error("a leaked loader loaded outside/evil.cjs");\n}\n' +
        `{\n  ${fresh}\n  M._extensions[".js"](m, ${out});\n}\n`;
    },
    refusal: { bun: loads("evil.cjs"), node: null },
  },
  {
    name: "a patched fs.realpathSync, then process.dlopen of a symlink out",
    source: (tree) => patched(SAME_PATH, `  process.dlopen({ exports: {} } as any, ${JSON.stringify(join(tree.root, "nlink.node"))});`),
    refusal: { bun: loads("addon.node"), node: loads("addon.node") },
  },
  {
    name: "a patched fs.realpathSync, then a require of a symlink out",
    source: () => REQUIRE + patched(SAME_PATH, '  req("./link.cjs");'),
    refusal: named("./link.cjs"),
  },
  {
    name: "a patched fs.realpathSync, then Module._extensions on a symlink out",
    source: (tree) => {
      const link = JSON.stringify(join(tree.root, "link.cjs"));
      return REQUIRE + patched(SAME_PATH, `  const m: any = new M(${link}, null);\n  m.filename = ${link};\n  m.paths = [];\n  M._extensions[".js"](m, ${link});`);
    },
    refusal: { bun: loads("evil.cjs"), node: loads("evil.cjs") },
  },
  {
    name: "a patched String.prototype.startsWith, then process.dlopen outside",
    source: (tree) => patched(NO_PARENT, `  process.dlopen({ exports: {} } as any, ${JSON.stringify(join(tree.base, "outside/addon.node"))});`),
    refusal: { bun: loads("addon.node"), node: loads("addon.node") },
  },
  {
    name: "a patched String.prototype.startsWith, then a ../ import",
    source: () => patched(NO_PARENT, `  await import(${computed("../outside/evil.ts")});`),
    refusal: named("../outside/evil.ts"),
  },
  {
    name: "a patched RegExp.prototype.exec and test, then a ../ import",
    source: () =>
      patched(
        "const { exec, test } = RegExp.prototype;\nRegExp.prototype.exec = () => null;\nRegExp.prototype.test = () => false;\n" +
          "const restore = () => { RegExp.prototype.exec = exec; RegExp.prototype.test = test; };",
        `  await import(${computed("../outside/evil.ts")});`,
      ),
    refusal: named("../outside/evil.ts"),
  },
  {
    name: "node:path's relative patched and synced into named imports, then a ../ import",
    source: () => patched(SYNCED("node:path", "relative", '() => "inner.ts"'), `  await import(${computed("../outside/evil.ts")});`),
    refusal: named("../outside/evil.ts"),
  },
  {
    name: "node:url's fileURLToPath patched and synced into named imports, then a file: URL out",
    source: (tree) =>
      patched(
        SYNCED("node:url", "fileURLToPath", `() => ${JSON.stringify(join(tree.root, "inner.ts"))}`),
        `  await import(${JSON.stringify(pathToFileURL(tree.evil).href)});`,
      ),
    refusal: { bun: ABSOLUTE, node: FILE_URL },
  },
  {
    name: "a nested file importing out",
    source: () => 'import { got } from "./lib/a.ts";\n',
    refusal: {
      bun: "lib/a.ts imports '../../outside/evil.ts', which resolves outside the code root",
      node: "lib/a.ts imports '../../outside/evil.ts', which resolves outside the code root",
    },
  },
  {
    name: "a package in a parent folder's node_modules",
    source: () => 'import { x } from "yian-walkup";\nexport const got = x;\n',
    // Bun resolves a package name without its plugins, so only the file it loads is known.
    refusal: {
      bun: "the code loads '../node_modules/yian-walkup/index.js', which is outside the code root",
      node: "entry.ts imports 'yian-walkup', which resolves outside the code root",
    },
  },
  {
    name: "a data: URL",
    source: () => `export const got = (await import(${computed("data:text/javascript,export const x = 'DATA'")})).x;\n`,
    refusal: url("data"),
  },
  {
    name: "a blob: URL",
    source: () =>
      'const blob = new Blob(["export const x = \'BLOB\'"], { type: "text/javascript" });\n' +
      "export const got = (await import(URL.createObjectURL(blob))).x;\n",
    refusal: url("blob"),
  },
  {
    name: "an https: URL",
    source: () => 'export const got = (await import("https://example.invalid/x.js")).x;\n',
    // Bun has no network imports: it reads the URL as a missing file.
    refusal: { bun: /ENOENT reading "https:\/\/example\.invalid\/x\.js"/, node: url("https").node },
  },
];

/** One import that stays inside the root, and the value `got` it binds; `bunOnly` when Node can't import it. */
export interface AllowedImport {
  readonly name: string;
  readonly source: (tree: CodeRootTree) => string;
  readonly got: string;
  readonly bunOnly?: boolean;
}

export const ALLOWED_IMPORTS: readonly AllowedImport[] = [
  {
    name: "a file inside, directly and through a symlink inside",
    source: (tree) =>
      'import { x } from "./inner.ts";\nimport { x as y } from "./alias.ts";\n' +
      `const z = (await import(${JSON.stringify(join(tree.root, "inner.ts"))})).x;\n` +
      'export const got = [x, y, z].join(",");\n',
    got: "INNER,INNER,INNER",
  },
  {
    name: "packages in the root's node_modules, an ES module and a CommonJS one, with their own imports",
    source: () =>
      'import { x } from "yian-inner";\nimport cjs from "yian-cjs";\nexport const got = x + "," + cjs.x;\n',
    got: "PACKAGE,COMMONJS",
  },
  {
    name: "a JSON file inside",
    source: () => 'import data from "./data.json" with { type: "json" };\nexport const got = data.x;\n',
    got: "JSON",
  },
  {
    name: "a text file inside",
    source: () => 'import note from "./note.txt" with { type: "text" };\nexport const got = note;\n',
    got: "TEXT",
    bunOnly: true,
  },
  {
    name: "built-ins, by import, import() and require",
    source: () =>
      'import { readFileSync } from "node:fs";\nimport path from "path";\nimport { createRequire } from "node:module";\n' +
      'const os = await import("node:os");\nconst util = createRequire(import.meta.url)("util");\n' +
      'const bun = typeof Bun === "undefined" ? "function" : typeof (await import("bun:sqlite")).Database;\n' +
      "export const got = [typeof readFileSync, typeof path.join, typeof os.cpus, typeof util.format, bun].join(\",\");\n",
    got: "function,function,function,function,function",
  },
  {
    name: "type-only imports, of yian and of a file outside",
    source: () =>
      'import type { ToolOptions } from "@ngocbh/yian/tools";\nimport type { Foo } from "../outside/types.ts";\n' +
      'const shown: Foo = "TYPES";\nexport const got: string = shown;\nexport type Options = ToolOptions;\n',
    got: "TYPES",
  },
];

/** Whether `message` holds `refusal`: a string as it is, or text a pattern matches. */
export function refuses(refusal: string | RegExp, message: string): boolean {
  return typeof refusal === "string" ? message.includes(refusal) : refusal.test(message);
}
