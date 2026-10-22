// Ported from tests/test_packaging.py: the package ships what it needs.
// `test_package_data_covers_every_runtime_data_file` becomes "every file under `src/` is a `.ts`
// file", since `files` ships `src` and its `dist/src` build whole. `test_python_floor_is_3_11`
// becomes the `engines` check and `test_mcp_extra_pins_the_mcp_sdk` the MCP SDK's optional peer.
// Added here: the npm fields (`files`, `bin`, `exports`, `prepack`, `publishConfig`), exact
// dependency pins, and the names each entry point exports, Python's `yian.__all__`, `yian.run`,
// `yian.tools` and `yian.skills`; and `LICENSES`, the text `yian --licenses` prints, against
// `NOTICE`, Bun's license and the license files of the packages the binaries bundle. With
// `YIAN_UPDATE_LICENSES=1` the test writes `src/cli/licenses.ts` instead, taking Bun's license from
// the file `YIAN_BUN_LICENSE` names, or else from the current `LICENSES`.

import { describe, expect, test } from "bun:test";
import { existsSync, readdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, join, relative } from "node:path";
import { pathToFileURL } from "node:url";

import { LICENSES } from "../src/cli/licenses.ts";
import type { MarkdownStore } from "../src/index.ts";
import type { Steering } from "../src/run.ts";

const ROOT = join(import.meta.dirname, "..");
const PACKAGE = JSON.parse(readFileSync(join(ROOT, "package.json"), "utf8"));
const MCP_SDK = "@modelcontextprotocol/client";

/** Each export path and the module under `src/` it names, without the `.ts`. */
const ENTRIES: Record<string, string> = {
  ".": "index",
  "./run": "run",
  "./tools": "tools/index",
  "./skills": "compose/skills",
};

/** The runtime names each export path must carry. */
const NAMES: Record<string, string[]> = {
  ".": [
    "loadFlow",
    "runFlow",
    "resumeFlow",
    "buildMemories",
    "MemoryRegistry",
    "Failure",
    "Diagnostic",
    "ComposedFlow",
    "CompiledFlow",
    "VariablePool",
    "evaluateWhen",
    "LoadError",
    "FlowValidationError",
    "ExpressionError",
    "ToolContext",
    "FunctionTool",
    "tool",
    "PermanentToolError",
    "TransientToolError",
    "ReportedToolError",
    "ToolAttemptsFailed",
    "registerTemplateFn",
    "SQLiteStorage",
    "checkStoragePath",
    "FileMarkdownStore",
    "InMemoryMarkdownStore",
    "MarkdownVersionMismatch",
    "markdownVersion",
  ],
  "./run": ["RunResult", "BackwardOutcome", "inspectRun", "runFlow", "resumeFlow"],
  "./tools": [
    "registerTool",
    "registeredTools",
    "invokeTool",
    "mcpServer",
    "McpServer",
    "McpError",
    "McpCallCancelled",
    "McpConnectError",
    "McpInvalidRequestError",
    "McpRequestError",
    "McpServerClosed",
    "McpTimeoutError",
    "McpToolError",
    "McpTransportError",
  ],
  "./skills": ["loadSkills", "loadSkillDir", "defaultSkillRoots", "Skill", "SkillError"],
};

/** Every file below `directory`, relative to `ROOT`. */
function filesUnder(directory: string): string[] {
  return readdirSync(directory, { recursive: true, withFileTypes: true })
    .filter((entry) => entry.isFile())
    .map((entry) => relative(ROOT, join(entry.parentPath, entry.name)));
}

describe("package.json", () => {
  test("ships the build, the sources and the notices, publicly", () => {
    expect(PACKAGE.files).toEqual(["dist/src", "src", "LICENSE", "NOTICE", "README.md"]);
    expect(PACKAGE.publishConfig).toEqual({ access: "public" });
    expect(PACKAGE.scripts.prepack).toBe("cp ../LICENSE LICENSE && bun run build");
  });

  test("the yian bin is the built CLI, run by node", () => {
    expect(PACKAGE.bin).toEqual({ yian: "./dist/src/cli/index.js" });
    expect(readFileSync(join(ROOT, "src", "cli", "index.ts"), "utf8").split("\n")[0]).toBe(
      "#!/usr/bin/env node",
    );
  });

  test("the runtime floors are Node 22.18 and Bun 1.4", () => {
    expect(PACKAGE.engines).toEqual({ node: ">=22.18", bun: ">=1.4" });
  });

  test("every dependency is pinned exactly", () => {
    const groups = ["dependencies", "peerDependencies", "devDependencies"];
    const loose = groups.flatMap((group) =>
      Object.entries(PACKAGE[group] as Record<string, string>)
        .filter(([, version]) => !/^\d+\.\d+\.\d+$/.test(version))
        .map(([name, version]) => `${group}: ${name}@${version}`),
    );
    expect(loose).toEqual([]);
  });

  test("the MCP SDK is an optional peer, installed for development", () => {
    expect(PACKAGE.dependencies[MCP_SDK]).toBeUndefined();
    expect(PACKAGE.peerDependencies[MCP_SDK]).toBe(PACKAGE.devDependencies[MCP_SDK]);
    expect(PACKAGE.peerDependenciesMeta[MCP_SDK]).toEqual({ optional: true });
  });

  test("each export maps Bun to the source and other runtimes to the build", () => {
    const expected: Record<string, unknown> = { "./package.json": "./package.json" };
    for (const [path, module] of Object.entries(ENTRIES)) {
      expected[path] = {
        bun: `./src/${module}.ts`,
        types: `./dist/src/${module}.d.ts`,
        default: `./dist/src/${module}.js`,
      };
    }
    expect(PACKAGE.exports).toEqual(expected);
    // Resolution takes the first matching condition, so Bun must see its own before `default`.
    for (const path of Object.keys(ENTRIES)) expect(Object.keys(PACKAGE.exports[path])[0]).toBe("bun");
  });
});

describe("entry points", () => {
  for (const [path, module] of Object.entries(ENTRIES)) {
    test(`${path} exports its runtime names`, async () => {
      const exported = await import(pathToFileURL(join(ROOT, "src", `${module}.ts`)).href);
      expect(NAMES[path]!.filter((name) => exported[name] === undefined)).toEqual([]);
    });
  }

  test("the type-only names are exported", () => {
    // The check is `bun run typecheck`: the imports above fail to compile without the names.
    const names = [] as [Steering?, MarkdownStore?];
    expect(names).toEqual([]);
  });
});

describe("package data", () => {
  test("every file under src is TypeScript, so src and dist/src carry the whole runtime", () => {
    expect(filesUnder(join(ROOT, "src")).filter((path) => !path.endsWith(".ts"))).toEqual([]);
  });
});

const UPDATE = process.env.YIAN_UPDATE_LICENSES === "1";
/** The line above and below each section header in `LICENSES`. */
const RULE = "-".repeat(79);
/** Bun's section in `LICENSES`: its header and its text, up to the next section. */
const BUN_SECTION = new RegExp(`\\n${RULE}\\n(bun@[^\\n]*)\\n${RULE}\\n\\n([\\s\\S]*?)(?=\\n${RULE}\\n|$)`);
const LICENSE_FILE = /^licen[cs]e(\.(md|txt))?$/i;

/** Return the parsed `package.json` of the package in `dir`. */
function packageJson(dir: string) {
  return JSON.parse(readFileSync(join(dir, "package.json"), "utf8"));
}

/** Return the directory of package `name` as Node finds it from `from`: `node_modules` there, then up. */
function resolvePackage(name: string, from: string): string {
  for (let dir = from; ; dir = dirname(dir)) {
    const candidate = join(dir, "node_modules", name);
    if (existsSync(join(candidate, "package.json"))) return candidate;
    if (dirname(dir) === dir) throw new Error(`cannot resolve ${name} from ${from}`);
  }
}

/**
 * Return the directories of the packages the binaries bundle: the `dependencies` and
 * `peerDependencies` of `package.json`, and then each one's own `dependencies`.
 */
function bundled(): string[] {
  const found = new Set<string>();
  const visit = (names: string[], from: string) => {
    for (const name of names) {
      const dir = resolvePackage(name, from);
      if (found.has(dir)) continue;
      found.add(dir);
      visit(Object.keys(packageJson(dir).dependencies ?? {}), dir);
    }
  };
  visit([...Object.keys(PACKAGE.dependencies), ...Object.keys(PACKAGE.peerDependencies)], ROOT);
  return [...found];
}

/** Return a section of `LICENSES`: `header` between rules, then `text`, ending in a newline. */
function section(header: string, text: string): string {
  return `\n${RULE}\n${header}\n${RULE}\n\n${text.endsWith("\n") ? text : `${text}\n`}`;
}

/** Return Bun's header and text: from `YIAN_BUN_LICENSE` when updating, else from `LICENSES`. */
function bunLicense(): [string, string] {
  const path = process.env.YIAN_BUN_LICENSE;
  if (UPDATE && path) {
    const version = process.versions.bun;
    if (version === undefined) throw new Error("YIAN_BUN_LICENSE names Bun's license: update under Bun");
    return [`bun@${version} (MIT; statically links JavaScriptCore, LGPL-2)`, readFileSync(path, "utf8")];
  }
  const match = BUN_SECTION.exec(LICENSES);
  if (match === null) throw new Error("LICENSES has no Bun section; set YIAN_BUN_LICENSE to Bun's LICENSE.md");
  return [match[1]!, match[2]!];
}

/** Return `LICENSES` rebuilt: `NOTICE`, Bun's section, then each bundled package's, by name. */
function buildLicenses(): string {
  const packages = bundled()
    .map((dir) => ({ dir, json: packageJson(dir) }))
    .sort((a, b) => (a.json.name < b.json.name ? -1 : a.json.name > b.json.name ? 1 : 0));
  let text = readFileSync(join(ROOT, "NOTICE"), "utf8") + section(...bunLicense());
  for (const { dir, json } of packages) {
    const file = readdirSync(dir).sort().find((entry) => LICENSE_FILE.test(entry));
    if (file === undefined) throw new Error(`${json.name} has no license file`);
    text += section(`${json.name}@${json.version} (${json.license})`, readFileSync(join(dir, file), "utf8"));
    // amaro's WebAssembly build of swc carries swc's own license.
    if (json.name === "amaro") {
      const swc = readFileSync(join(dir, "dist", "LICENSE"), "utf8");
      text += section(`amaro@${json.version} dist/LICENSE (swc, Apache-2.0)`, swc);
    }
  }
  return text;
}

/** Return the source of `src/cli/licenses.ts` holding `text`. */
function licensesModule(text: string): string {
  return (
    "/**\n" +
    " * The text `yian --licenses` prints: `NOTICE`, then the licenses of Bun and of the npm packages\n" +
    " * the binaries bundle. Generated by `YIAN_UPDATE_LICENSES=1 bun test tests/packaging.test.ts`;\n" +
    " * not edited by hand.\n" +
    " */\n\n" +
    `export const LICENSES: string =\n  ${JSON.stringify(text)};\n`
  );
}

describe("LICENSES", () => {
  test("is NOTICE, then Bun's license and each bundled package's", () => {
    const expected = buildLicenses();
    if (UPDATE) {
      writeFileSync(join(ROOT, "src", "cli", "licenses.ts"), licensesModule(expected));
      return;
    }
    expect(LICENSES).toBe(expected);
  });

  test("the bundled packages are the runtime and peer dependencies and theirs", () => {
    const names = bundled().map((dir) => packageJson(dir).name).sort();
    expect(names).toEqual([
      "@modelcontextprotocol/client",
      "@modelcontextprotocol/core",
      "amaro",
      "cross-spawn",
      "eventsource",
      "eventsource-parser",
      "isexe",
      "jose",
      "path-key",
      "pkce-challenge",
      "shebang-command",
      "shebang-regex",
      "which",
      "yaml",
      "zod",
    ]);
  });

  test("Bun's section names the running Bun's version", () => {
    const version = /^bun@(\d+\.\d+\.\d+) /.exec(bunLicense()[0])?.[1];
    expect(version).toBeDefined();
    // Node runs the check too, with no Bun version to compare.
    if (process.versions.bun !== undefined) expect(version).toBe(process.versions.bun);
  });
});
