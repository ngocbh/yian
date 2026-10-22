// Ported from tests/test_packaging.py: the package ships what it needs.
// `test_package_data_covers_every_runtime_data_file` becomes "every file under `src/` is a `.ts`
// file", since `files` ships `src` and its `dist/src` build whole. `test_python_floor_is_3_11`
// becomes the `engines` check and `test_mcp_extra_pins_the_mcp_sdk` the MCP SDK's optional peer.
// Added here: the npm fields (`files`, `bin`, `exports`, `prepack`, `publishConfig`), exact
// dependency pins, and the names each entry point exports, Python's `yian.__all__`, `yian.run`,
// `yian.tools` and `yian.skills`.

import { describe, expect, test } from "bun:test";
import { readdirSync, readFileSync } from "node:fs";
import { join, relative } from "node:path";
import { pathToFileURL } from "node:url";

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
