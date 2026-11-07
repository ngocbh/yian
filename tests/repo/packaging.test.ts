// The package ships what it needs: the npm fields in package.json, the README that is the npm page,
// the names each entry point exports, and LICENSES against the notices (`YIAN_UPDATE_LICENSES=1`
// rewrites it). Opt-in runs: `YIAN_TEST_BINARY` a `bun run build:bin` binary, `YIAN_TEST_NODE` the
// built `./code-worker` on that Node, and `YIAN_TEST_PACK=1` the packed, installed npm tarball.

import { afterAll, afterEach, beforeAll, describe, expect, test } from "bun:test";
import { execFileSync, spawn, spawnSync } from "node:child_process";
import {
  existsSync,
  mkdirSync,
  mkdtempSync,
  readdirSync,
  readFileSync,
  realpathSync,
  rmSync,
  statSync,
  symlinkSync,
  writeFileSync,
} from "node:fs";
import { createServer } from "node:http";
import type { AddressInfo } from "node:net";
import { tmpdir } from "node:os";
import { dirname, isAbsolute, join, relative, sep } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

import { LICENSES } from "../../src/cli/licenses.ts";
import { AuthStore } from "../../src/llm_clients/auth/index.ts";
import type {
  AuthRequest,
  AuthRequestOptions,
  AuthState,
  AuthStoreOptions,
  Credentials,
  Issue,
  ListModelsOptions,
  LLMProvider,
} from "../../src/llm_clients/index.ts";
import type {
  FileLock,
  FileLockReason,
  Sqlite,
  SqliteParam,
  SqliteRow,
  SqliteRunResult,
  SqliteStatement,
  SqliteValue,
} from "../../src/platform.ts";
import type { Message as RootMessage } from "../../src/index.ts";
import type {
  JsonMode,
  JsonObject,
  JsonValue,
  Message,
  MessagePart,
  StorageKind,
  Value,
  ValueRecord,
} from "../../src/run.ts";
import { stdioOptions } from "../helpers/mcp_helpers.ts";
import { testNode } from "../helpers/test_node.ts";

const ROOT = join(import.meta.dirname, "..", "..");
const PACKAGE = JSON.parse(readFileSync(join(ROOT, "package.json"), "utf8"));
const MCP_SDK = "@modelcontextprotocol/client";

/** Each export path and the module under `src/` it names, without the `.ts`. */
const ENTRIES: Record<string, string> = {
  ".": "index",
  "./run": "run",
  "./tools": "tools/index",
  "./skills": "compose/skills",
  "./platform": "platform",
  "./llm_clients": "llm_clients/index",
  "./code-worker": "nodes/code/worker",
};

/** The runtime names each export path must carry. */
const NAMES: Record<string, string[]> = {
  ".": [
    "loadFlow",
    "runFlow",
    "advance",
    "send",
    "withdraw",
    "complete",
    "close",
    "interrupt",
    "cancel",
    "pendingCalls",
    "RunBusyError",
    "DeliveryRefused",
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
    "pending",
    "FunctionTool",
    "tool",
    "PermanentToolError",
    "TransientToolError",
    "ReportedToolError",
    "ToolAttemptsFailed",
    "registerTemplateFn",
    "checkMessage",
    "checkConversation",
    "ConversationError",
    "InboxRef",
  ],
  "./run": [
    "RunResult",
    "inspectRun",
    "agentCalls",
    "agentCall",
    "runFlow",
    "advance",
    "send",
    "withdraw",
    "complete",
    "close",
    "interrupt",
    "cancel",
    "pendingCalls",
    "RunBusyError",
    "DeliveryRefused",
    "loadRecordedFlow",
    "runLogPath",
    "updateRunIndex",
    "readRunIndex",
    "RunIndexBusyError",
    "TypedValue",
    "FileRef",
    "InboxRef",
    "logJson",
    "parseJson",
    "JSONDecodeError",
    "checkMessage",
    "checkConversation",
    "ConversationError",
  ],
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
    "McpOutputError",
    "McpRequestError",
    "McpSdkMissingError",
    "McpServerClosed",
    "McpTimeoutError",
    "McpToolError",
    "McpTransportError",
    "pending",
    "isPending",
    "TAKE",
    "SLEEP",
    "ASK",
  ],
  "./skills": ["loadSkills", "loadSkillDir", "defaultSkillRoots", "Skill", "SkillError"],
  "./platform": ["lockFile", "waitForLock", "isLocked", "FileLockError", "openSqlite", "sqliteErrorCode", "syncFolder"],
  "./llm_clients": [
    "LLMConfig",
    "LLM_CONFIG_KEYS",
    "SUPPORTED_LLM_PROVIDERS",
    "ValidationError",
    "azureResourceEndpoint",
    "listModels",
    "ModelListError",
    "AuthStore",
    "OpenAIAuth",
    "AuthError",
    "HttpResponse",
    "OAuthCredentials",
    "ApiKeyCredentials",
    "ProviderSelection",
  ],
  // Imported for its side effect, as a bundling host's compile entry; its named exports are internal.
  "./code-worker": [],
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
    expect(PACKAGE.scripts.prepack).toBe("npm run build");
  });

  test("links only the docs site while the repository is private", () => {
    expect(PACKAGE.homepage).toBe("https://yian.readthedocs.io/");
    expect(PACKAGE).not.toHaveProperty("repository");
    expect(PACKAGE).not.toHaveProperty("bugs");
  });

  test("the yian bin is the built CLI entry, run by node", () => {
    // Without a leading `./`: `npm publish` drops a bin path written with one.
    expect(PACKAGE.bin).toEqual({ yian: "dist/src/cli/bin.js" });
    expect(readFileSync(join(ROOT, "src", "cli", "bin.ts"), "utf8").split("\n")[0]).toBe(
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

  test("the README, which is the npm page, links only the docs site", () => {
    const readme = readFileSync(join(ROOT, "README.md"), "utf8");
    expect(readme).toContain("https://yian.readthedocs.io/");
    expect(readme).toContain("npm install @ngocbh/yian");
    expect(readme).not.toContain("github.com/ngocbh/yian");
    // npm resolves relative links against `repository`, which is unset while the repo is private.
    expect(readme.match(/\]\((?!https:\/\/)[^)]*\)/g) ?? []).toEqual([]);
    expect(readme).not.toMatch(/python|pytest|pip install/i);
    expect(readme.split("\n").length).toBeLessThanOrEqual(80);
  });
});

/** The `yian` bin's source, which Bun and Node (22.18 and later) run as is. */
const BIN = join(ROOT, "src", "cli", "bin.ts");
/** The error the bin prints, after `this is`, on a runtime older than the floor. */
const FLOOR_ERROR = "Error: yian needs Node 22.18 or later, or Bun 1.4 or later; this is ";

/**
 * The source of a preload module that makes `process.versions` report `runtime` at `version`:
 * for Node, it also removes `bun`, so Bun passes for an older Node too.
 */
function reportAs(runtime: "node" | "bun", version: string): string {
  const set = `Object.defineProperty(process.versions, ${JSON.stringify(runtime)}, { value: ${JSON.stringify(version)}, configurable: true, enumerable: true });\n`;
  return runtime === "bun" ? set : `delete process.versions.bun;\n${set}`;
}

/**
 * Run `bin` with `args` on this runtime in a fresh temp dir, first loading a module whose source is
 * `preload` when given.
 */
function runBin(bin: string, args: string[], preload?: string) {
  const dir = mkdtempSync(join(tmpdir(), "yian-bin-"));
  try {
    const flags: string[] = [];
    if (process.versions.bun !== undefined) flags.push("--no-install", "--no-env-file");
    if (preload !== undefined) {
      const file = join(dir, "preload.mjs");
      writeFileSync(file, preload);
      flags.push(...(process.versions.bun !== undefined ? ["--preload", file] : ["--import", pathToFileURL(file).href]));
    }
    return spawnSync(process.execPath, [...flags, bin, ...args], {
      cwd: dir,
      env: { HOME: dir, PATH: process.env.PATH ?? "" },
      encoding: "utf8",
      timeout: 60_000,
    });
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
}

describe("the yian bin", () => {
  test("runs the CLI on this runtime", () => {
    const result = runBin(BIN, ["--version"]);
    expect(result.status, result.stderr).toBe(0);
    expect(result.stdout.trim()).toBe(PACKAGE.version);
  });

  // 22.9 sorts after 22.18 as text, so the parts must compare as numbers.
  for (const [runtime, version] of [
    ["bun", "1.3.9"],
    ["node", "22.17.1"],
    ["node", "22.9.0"],
    ["node", "20.19.0"],
  ] as const) {
    test(`refuses ${runtime} ${version}, naming the floor, before loading the CLI`, () => {
      const result = runBin(BIN, ["--version"], reportAs(runtime, version));
      const name = runtime === "bun" ? "Bun" : "Node";
      expect(result.stderr).toBe(`${FLOOR_ERROR}${name} ${version}.\n`);
      expect(result.stdout).toBe("");
      expect(result.status).toBe(1);
    });
  }

  test("the floor it names is package.json's engines", () => {
    expect(FLOOR_ERROR).toContain(`Node ${PACKAGE.engines.node.slice(2)} or later, or Bun ${PACKAGE.engines.bun.slice(2)} or later`);
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
    const names = [] as [
      StorageKind?,
      Value?,
      ValueRecord?,
      JsonValue?,
      JsonObject?,
      JsonMode?,
      FileLock?,
      FileLockReason?,
      Sqlite?,
      SqliteStatement?,
      SqliteParam?,
      SqliteValue?,
      SqliteRow?,
      SqliteRunResult?,
      LLMProvider?,
      ListModelsOptions?,
      AuthRequest?,
      AuthRequestOptions?,
      AuthStoreOptions?,
      Credentials?,
      AuthState?,
      Issue?,
      Message?,
      MessagePart?,
      RootMessage?,
    ];
    expect(names).toEqual([]);
  });
});

describe("package data", () => {
  test("every file under src is TypeScript, so src and dist/src carry the whole runtime", () => {
    // The bundled flows' sources: the CLI runs them from `src/cli/bundled.ts`, which embeds them.
    const bundledFlow = (path: string) => path.startsWith(join("src", "cli", "flows") + sep) && path.endsWith(".yaml");
    const other = filesUnder(join(ROOT, "src")).filter((path) => !path.endsWith(".ts") && !bundledFlow(path));
    expect(other).toEqual([]);
  });
});

const UPDATE = process.env.YIAN_UPDATE_LICENSES === "1";
/** The line above and below each section header in `LICENSES`. */
const RULE = "-".repeat(79);
/** A package's license and notice files, which `LICENSES` carries in order of name. */
const LICENSE_FILE = /^(licen[cs]e|notice|third_party_notices)(\.(md|txt))?$/i;
/**
 * A `//#region` comment naming a file pnpm installed, `.pnpm/<name>@<version>[_<peers>]/`, with a
 * scoped name written `@scope+name`: the mark a bundled `dist/` file leaves where it copied a package.
 */
const VENDORED_REGION = /^\/\/#region \S*node_modules\/\.pnpm\/((?:@[^/+]+\+)?[^/@]+)@([^/_]+)[^/]*\/node_modules\//gm;

/** Return the parsed `package.json` of the package in `dir`. */
function packageJson(dir: string) {
  return JSON.parse(readFileSync(join(dir, "package.json"), "utf8"));
}

/** Return the directory of package `name` as Node finds it from `from`: `node_modules` there, then up. */
function resolvePackage(name: string, from: string): string {
  for (let dir = from; ; dir = dirname(dir)) {
    const candidate = join(dir, "node_modules", name);
    // Node resolves a package through its symlinks.
    if (existsSync(join(candidate, "package.json"))) return realpathSync(candidate);
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

/** A package copied into the `dist/` of a bundled package, `carrier`. */
interface Vendored {
  name: string;
  version: string;
  carrier: string;
}

/**
 * Return the packages copied into the `dist/` files of the bundled packages, by name: the ones their
 * `//#region` comments name, each with the first bundled package found carrying it.
 */
function vendored(): Vendored[] {
  const found = new Map<string, Vendored>();
  for (const dir of bundled()) {
    const dist = join(dir, "dist");
    if (!existsSync(dist)) continue;
    const carrier = packageJson(dir).name;
    for (const entry of readdirSync(dist, { recursive: true, withFileTypes: true })) {
      if (!entry.isFile()) continue;
      const source = readFileSync(join(entry.parentPath, entry.name), "utf8");
      for (const [, scoped, version] of source.matchAll(VENDORED_REGION)) {
        const name = scoped!.replace("+", "/");
        if (!found.has(`${name}@${version}`)) found.set(`${name}@${version}`, { name, version: version!, carrier });
      }
    }
  }
  return [...found.values()].sort((a, b) => (a.name < b.name ? -1 : a.name > b.name ? 1 : 0));
}

/** Return a section of `LICENSES`: `header` between rules, then `text`, ending in a newline. */
function section(header: string, text: string): string {
  return `\n${RULE}\n${header}\n${RULE}\n\n${text.endsWith("\n") ? text : `${text}\n`}`;
}

/** Return the header and text of the section of the current `LICENSES` whose header `matches`. */
function currentSection(matches: (header: string) => boolean): [string, string] | undefined {
  // `NOTICE`, then a header and a blank line and text for each section.
  const parts = LICENSES.split(`\n${RULE}\n`);
  for (let i = 1; i + 1 < parts.length; i += 2) {
    if (matches(parts[i]!)) return [parts[i]!, parts[i + 1]!.slice(1)];
  }
  return undefined;
}

/**
 * Return the header and text of the package in `dir`: its name, version and license, then `suffix`;
 * and its license and notice files, joined.
 */
function packageLicense(dir: string, suffix = ""): [string, string] {
  const json = packageJson(dir);
  const files = readdirSync(dir).sort().filter((entry) => LICENSE_FILE.test(entry));
  if (files.length === 0) throw new Error(`${json.name} has no license file`);
  if (typeof json.license !== "string") throw new Error(`${json.name} has no license field`);
  const texts = files.map((file) => readFileSync(join(dir, file), "utf8"));
  return [
    `${json.name}@${json.version} (${json.license})${suffix}`,
    texts.map((text) => (text.endsWith("\n") ? text : `${text}\n`)).join("\n"),
  ];
}

/** Return Bun's header and text: from `YIAN_BUN_LICENSE` when updating, else from `LICENSES`. */
function bunLicense(): [string, string] {
  const path = process.env.YIAN_BUN_LICENSE;
  if (UPDATE && path) {
    const version = process.versions.bun;
    if (version === undefined) throw new Error("YIAN_BUN_LICENSE names Bun's license: update under Bun");
    return [`bun@${version} (MIT; statically links JavaScriptCore, LGPL-2)`, readFileSync(path, "utf8")];
  }
  const kept = currentSection((header) => header.startsWith("bun@"));
  if (kept === undefined) throw new Error("LICENSES has no Bun section; set YIAN_BUN_LICENSE to Bun's LICENSE.md");
  return kept;
}

/** The Bun version Bun's section of `LICENSES` names, or `undefined` when it names none. */
function bunLicenseVersion(): string | undefined {
  return /^bun@(\d+\.\d+\.\d+) /.exec(bunLicense()[0])?.[1];
}

/**
 * Return the header and text of `pkg`'s section. When updating with `YIAN_VENDORED_LICENSES` set,
 * they come from the npm package extracted there (`<dir>/package/`); else from `LICENSES`.
 */
function vendoredLicense(pkg: Vendored): [string, string] {
  const id = `${pkg.name}@${pkg.version}`;
  const suffix = `, copied into ${pkg.carrier}`;
  const root = process.env.YIAN_VENDORED_LICENSES;
  if (UPDATE && root) {
    for (const entry of readdirSync(root)) {
      const dir = join(root, entry, "package");
      if (!existsSync(join(dir, "package.json"))) continue;
      const json = packageJson(dir);
      if (`${json.name}@${json.version}` === id) return packageLicense(dir, suffix);
    }
    throw new Error(`YIAN_VENDORED_LICENSES=${root} has no extracted ${id}`);
  }
  const kept = currentSection((header) => header.startsWith(`${id} (`) && header.endsWith(suffix));
  if (kept === undefined) {
    throw new Error(`LICENSES has no section for ${id}${suffix}; set YIAN_VENDORED_LICENSES to its extracted npm package`);
  }
  return kept;
}

/**
 * Return `LICENSES` rebuilt: `NOTICE`, Bun's section, each bundled package's by name, then each
 * package copied into their `dist/` by name.
 */
function buildLicenses(): string {
  const packages = bundled()
    .map((dir) => ({ dir, json: packageJson(dir) }))
    .sort((a, b) => (a.json.name < b.json.name ? -1 : a.json.name > b.json.name ? 1 : 0));
  let text = readFileSync(join(ROOT, "NOTICE"), "utf8") + section(...bunLicense());
  for (const { dir, json } of packages) {
    text += section(...packageLicense(dir));
    // amaro's WebAssembly build of swc carries swc's own license.
    if (json.name === "amaro") {
      const swc = readFileSync(join(dir, "dist", "LICENSE"), "utf8");
      text += section(`amaro@${json.version} dist/LICENSE (swc, Apache-2.0)`, swc);
    }
  }
  for (const pkg of vendored()) text += section(...vendoredLicense(pkg));
  return text;
}

/** Return the source of `src/cli/licenses.ts` holding `text`. */
function licensesModule(text: string): string {
  return (
    "/**\n" +
    " * The text `yian --licenses` prints: `NOTICE`, then the licenses of Bun, of the npm packages the\n" +
    " * binaries bundle and of the packages copied into their `dist/`. Generated by\n" +
    " * `YIAN_UPDATE_LICENSES=1 bun test tests/repo/packaging.test.ts`; not edited by hand.\n" +
    " */\n\n" +
    `export const LICENSES: string =\n  ${JSON.stringify(text)};\n`
  );
}

describe("LICENSES", () => {
  test("is NOTICE, then Bun's license, each bundled package's and each copied-in package's", () => {
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

  test("the copied-in packages are the ones the MCP client's dist carries", () => {
    expect(vendored()).toEqual([
      { name: "@cfworker/json-schema", version: "4.1.1", carrier: MCP_SDK },
      { name: "ajv", version: "8.18.0", carrier: MCP_SDK },
      { name: "ajv-formats", version: "3.0.1", carrier: MCP_SDK },
      { name: "content-type", version: "1.0.5", carrier: MCP_SDK },
      { name: "fast-deep-equal", version: "3.1.3", carrier: MCP_SDK },
      { name: "fast-uri", version: "3.1.0", carrier: MCP_SDK },
      { name: "jose", version: "6.2.2", carrier: MCP_SDK },
      { name: "json-schema-traverse", version: "1.0.0", carrier: MCP_SDK },
      { name: "json-schema-typed", version: "8.0.2", carrier: MCP_SDK },
    ]);
  });

  test("Bun's section names a Bun version", () => {
    expect(bunLicenseVersion()).toBeDefined();
  });
});

/**
 * The compiled binary named by `YIAN_TEST_BINARY`, or `undefined` when it's unset. A set value
 * that isn't an absolute path to a file fails the whole file, so a typo can't pass as a run with
 * the binary checks skipped.
 */
const BINARY = ((): string | undefined => {
  const binary = process.env.YIAN_TEST_BINARY;
  if (!binary) return undefined;
  if (!isAbsolute(binary) || !existsSync(binary) || !statSync(binary).isFile()) {
    throw new Error(`YIAN_TEST_BINARY=${binary} is not an absolute path to a file`);
  }
  return binary;
})();

/** How long one spawn of the binary may take before it is killed, so a hang can't stall the run. */
const SPAWN_LIMIT = 60_000;
/** Each binary test's limit, above `SPAWN_LIMIT` so the spawn's own failure reports first. */
const BINARY_TEST_LIMIT = 90_000;

const CODE_FLOW = `id: compiled
name: compiled
input:
  n: !int
nodes:
  double:
    kind: code
    input: {n: "\${input.n}"}
    output: int
    code: |
      const two: number = 2
      return inputs.n * two
  triple:
    kind: code
    input: {n: "\${double.output}"}
    output: int
    code: ./helpers.ts:triple
output: \${triple.output}
`;

const BLOCKED_FLOW = `id: blocked
name: blocked
nodes:
  spin:
    kind: code
    output: int
    code: |
      console.log("ready")
      while (true) {}
output: \${spin.output}
`;

const TOOL_FLOW = `id: mcp_tool
name: mcp_tool
nodes:
  call:
    kind: tool
    tool_id: echo
    input: {text: "hi"}
output: \${call.output}
`;

/** The temp dirs the current test made, removed after it. */
const workdirs: string[] = [];
/** The process groups the current test started, killed after it. */
const groups: number[] = [];

/** A fresh private temp dir holding an empty `home/` and the `files` given, by name. */
function workdir(files: Record<string, string> = {}): string {
  const dir = realpathSync(mkdtempSync(join(tmpdir(), "yian-binary-")));
  workdirs.push(dir);
  mkdirSync(join(dir, "home"));
  for (const [name, text] of Object.entries(files)) writeFileSync(join(dir, name), text);
  return dir;
}

/** The CODE flow and the helper module it references, which imports a sibling module. */
const CODE_FILES = {
  "flow.yaml": CODE_FLOW,
  "helpers.ts":
    'import { three } from "./lib.ts";\n\n' +
    "export function triple(inputs: { n: number }): number {\n  return inputs.n * three;\n}\n",
  "lib.ts": "export const three: number = 3;\n",
};

interface BinaryResult {
  code: number | null;
  signal: NodeJS.Signals | null;
  stdout: string;
  stderr: string;
  /** Whether the spawn hit `SPAWN_LIMIT` and its process group was killed. */
  timedOut: boolean;
}

/** Kill process group `pid` with SIGKILL, if any process of it is left. */
function killGroup(pid: number): void {
  try {
    process.kill(-pid, "SIGKILL");
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code !== "ESRCH") throw error;
  }
}

/**
 * Run the binary (`binary`, else `BINARY`) with `args` in `dir`, with only `PATH`, a `HOME` and an
 * auth file in `dir`, and `env`, in its environment, as the leader of a new process group. With
 * `interrupt`, send it one SIGINT once a line `ready` appears on stdout or stderr. On `SPAWN_LIMIT`,
 * kill the group.
 */
function runBinary(
  dir: string,
  args: string[],
  options: { env?: Record<string, string>; interrupt?: boolean; binary?: string } = {},
): Promise<BinaryResult> {
  return new Promise((resolve, reject) => {
    const child = spawn(options.binary ?? BINARY!, args, {
      cwd: dir,
      env: {
        PATH: process.env.PATH ?? "",
        HOME: join(dir, "home"),
        YIAN_AUTH_FILE: join(dir, "auth.json"),
        ...options.env,
      },
      // Its own group, so a timeout or the test's cleanup also ends what it started: the `run
      // --json` child and MCP servers hold the output pipes, and outlive a killed binary.
      detached: true,
    });
    if (child.pid !== undefined) groups.push(child.pid);
    let stdout = "";
    let stderr = "";
    let interrupted = false;
    let timedOut = false;
    const timer = setTimeout(() => {
      timedOut = true;
      if (child.pid !== undefined) killGroup(child.pid);
    }, SPAWN_LIMIT);
    const seen = () => {
      if (options.interrupt && !interrupted && /^ready\r?\n/m.test(stdout + stderr)) {
        interrupted = true;
        // The binary's own process only: a run must forward the stop to whatever it started.
        child.kill("SIGINT");
      }
    };
    // Decoded across chunks, so a multi-byte character split between two reads stays whole.
    child.stdout.setEncoding("utf8");
    child.stderr.setEncoding("utf8");
    child.stdout.on("data", (chunk) => {
      stdout += chunk;
      seen();
    });
    child.stderr.on("data", (chunk) => {
      stderr += chunk;
      seen();
    });
    child.on("error", (error) => {
      clearTimeout(timer);
      reject(error);
    });
    child.on("close", (code, signal) => {
      clearTimeout(timer);
      resolve({ code, signal, stdout, stderr, timedOut });
    });
  });
}

/** The result's code, signal and output, for a failed assertion's message. */
function context(result: BinaryResult): string {
  const limit = result.timedOut ? ` (killed after ${SPAWN_LIMIT} ms)` : "";
  return `code: ${result.code}, signal: ${result.signal}${limit}\nstdout: ${result.stdout}\nstderr: ${result.stderr}`;
}

/** The last non-empty line of `text`. */
function lastLine(text: string): string {
  return text.trim().split("\n").at(-1)!;
}

describe.skipIf(!BINARY)("the compiled binary", () => {
  afterEach(() => {
    // Groups first, so nothing still writes into a dir being removed.
    for (const pid of groups.splice(0)) killGroup(pid);
    for (const dir of workdirs.splice(0)) rmSync(dir, { recursive: true, force: true });
  });

  // The binary embeds a Bun, so its license must name that one. Without a binary the version is
  // only what CI pins, and any Bun may run the suite.
  test(
    "Bun's section names the Bun the binary embeds",
    async () => {
      const result = await runBinary(workdir(), ["--version"], { env: { BUN_BE_BUN: "1" } });
      expect(result.code, context(result)).toBe(0);
      expect(bunLicenseVersion()).toBe(result.stdout.trim());
    },
    BINARY_TEST_LIMIT,
  );

  test(
    "--licenses prints LICENSES",
    async () => {
      const result = await runBinary(workdir(), ["--licenses"]);
      expect(result.code, context(result)).toBe(0);
      expect(result.stdout).toBe(LICENSES);
    },
    BINARY_TEST_LIMIT,
  );

  test(
    "--version prints package.json's version",
    async () => {
      const result = await runBinary(workdir(), ["--version"]);
      expect(result.code, context(result)).toBe(0);
      expect(result.stdout).toBe(`${PACKAGE.version}\n`);
    },
    BINARY_TEST_LIMIT,
  );

  test(
    "runs an inline CODE node and a reference whose module imports a sibling",
    async () => {
      const result = await runBinary(workdir(CODE_FILES), ["run", "flow.yaml", "-w", "0", "--input", "n=2"]);
      expect(result.code, context(result)).toBe(0);
      expect(lastLine(result.stdout)).toBe("12");
    },
    BINARY_TEST_LIMIT,
  );

  test(
    "one SIGINT stops a blocked CODE body with exit 130",
    async () => {
      const dir = workdir({ "blocked.yaml": BLOCKED_FLOW });
      const result = await runBinary(dir, ["run", "blocked.yaml", "-w", "0"], { interrupt: true });
      expect(result.code, context(result)).toBe(130);
      expect(result.stderr).toContain("cancelled by user");
    },
    BINARY_TEST_LIMIT,
  );

  test(
    "run --json prints one yian.cli.v1 document",
    async () => {
      const args = ["run", "flow.yaml", "-w", "0", "--input", "n=2", "--json"];
      const result = await runBinary(workdir(CODE_FILES), args);
      expect(result.code, context(result)).toBe(0);
      expect(JSON.parse(result.stdout)).toEqual({ format: "yian.cli.v1", output: 12, status: "succeeded" });
    },
    BINARY_TEST_LIMIT,
  );

  test(
    "run --json on a blocked CODE body stops on one SIGINT with exit 130",
    async () => {
      const dir = workdir({ "blocked.yaml": BLOCKED_FLOW });
      const result = await runBinary(dir, ["run", "blocked.yaml", "-w", "0", "--json"], { interrupt: true });
      expect(result.code, context(result)).toBe(130);
    },
    BINARY_TEST_LIMIT,
  );

  test(
    "auth saves an ollama login after listing the stub server's models",
    async () => {
      const requests: string[] = [];
      const server = createServer((request, response) => {
        requests.push(`${request.method} ${request.url}`);
        if (request.url === "/api/tags") {
          response.writeHead(200, { "content-type": "application/json" });
          response.end(JSON.stringify({ models: [{ name: "m" }] }));
        } else {
          response.writeHead(404);
          response.end();
        }
      });
      await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
      const { port } = server.address() as AddressInfo;
      const dir = workdir();
      const file = join(dir, "a.json");
      let result: BinaryResult;
      try {
        result = await runBinary(dir, ["auth", "ollama", "--model", "m", "--auth-file", file], {
          env: { OLLAMA_BASE_URL: `http://127.0.0.1:${port}` },
        });
      } finally {
        await new Promise((resolve) => server.close(resolve));
      }
      expect(result.code, context(result)).toBe(0);
      expect(result.stdout).toContain("Saved ollama login (none) with model m\n");
      expect(requests).toEqual(["GET /api/tags"]);
      const selected = new AuthStore(file).selection("ollama")!;
      expect([selected.method, selected.model]).toEqual(["none", "m"]);
    },
    BINARY_TEST_LIMIT,
  );

  test(
    "ignores a .env and a bunfig.toml in the cwd",
    async () => {
      const dir = workdir({
        ...CODE_FILES,
        ".env": "YIAN_MAX_VALUE_NESTING=nope\n",
        "bunfig.toml": 'preload = ["./preload.ts"]\n',
        "preload.ts":
          'import { writeFileSync } from "node:fs";\n\nwriteFileSync(`${import.meta.dir}/marker`, "loaded");\n',
      });
      const result = await runBinary(dir, ["run", "flow.yaml", "-w", "0", "--input", "n=2"]);
      expect(result.code, context(result)).toBe(0);
      expect(lastLine(result.stdout)).toBe("12");
      expect(existsSync(join(dir, "marker"))).toBe(false);
    },
    BINARY_TEST_LIMIT,
  );

  test(
    "checks every example",
    async () => {
      const examples = join(ROOT, "examples");
      const flows = readdirSync(examples)
        .filter((name) => name.endsWith(".yaml"))
        .map((name) => join(examples, name));
      expect(flows.length).toBeGreaterThan(0);
      const result = await runBinary(workdir(), ["check", ...flows]);
      expect(result.code, context(result)).toBe(0);
    },
    BINARY_TEST_LIMIT,
  );

  test(
    "a TOOL node calls an MCP server from --mcp-config",
    async () => {
      const dir = workdir({ "flow.yaml": TOOL_FLOW });
      const config = join(dir, "mcp.json");
      // The fake server runs on the runtime running this test, the same way the MCP CLI tests start it.
      writeFileSync(config, JSON.stringify({ mcpServers: { fake: stdioOptions() } }));
      const result = await runBinary(dir, ["run", "flow.yaml", "-w", "0", "--mcp-config", config]);
      expect(result.code, context(result)).toBe(0);
      expect(lastLine(result.stdout)).toBe("hi");
    },
    BINARY_TEST_LIMIT,
  );
});

/** The `./code-worker` entry's import name. */
const CODE_WORKER = "@ngocbh/yian/code-worker";
const WORKER_BUILT = join(ROOT, "dist/src/nodes/code/worker.js");

const NODE = testNode(WORKER_BUILT);

/** How long compiling a host may take before it is killed. */
const COMPILE_LIMIT = 30_000;
/** Each host test's limit, above `COMPILE_LIMIT + SPAWN_LIMIT` so a step's own failure reports first. */
const HOST_TEST_LIMIT = 100_000;

/**
 * Compile, with the running Bun, a host that imports yian through `node_modules` and runs the
 * flow file its argument names with `{ n: 2 }`, printing the status, output and failure message.
 * With `withShim`, a second entry `src/nodes/code/worker.ts` imports `@ngocbh/yian/code-worker`.
 * Return the binary's path.
 */
function compileHost(withShim: boolean): string {
  const host = workdir();
  mkdirSync(join(host, "src/nodes/code"), { recursive: true });
  mkdirSync(join(host, "node_modules/@ngocbh"), { recursive: true });
  writeFileSync(join(host, "package.json"), '{"name":"host","private":true}');
  symlinkSync(ROOT, join(host, "node_modules/@ngocbh/yian"));
  writeFileSync(
    join(host, "src/main.ts"),
    'import { readFileSync } from "node:fs";\n' +
      'import { loadFlow, runFlow } from "@ngocbh/yian";\n' +
      'const flow = await loadFlow(readFileSync(process.argv[2]!, "utf8"), { baseDir: process.cwd() });\n' +
      "const result = await runFlow(flow, { n: 2 });\n" +
      'console.log(result.status, String(result.output), result.failure?.message ?? "");\n',
  );
  if (withShim) writeFileSync(join(host, "src/nodes/code/worker.ts"), `import ${JSON.stringify(CODE_WORKER)};\n`);
  const args = ["build", "--compile", "--no-compile-autoload-dotenv", "--no-compile-autoload-bunfig", "src/main.ts"];
  if (withShim) args.push("src/nodes/code/worker.ts");
  // No `--target`: the running Bun, so nothing is downloaded.
  const result = spawnSync(process.execPath, [...args, "--outfile", "out/host"], {
    cwd: host,
    encoding: "utf8",
    timeout: COMPILE_LIMIT,
    killSignal: "SIGKILL",
  });
  expect(result.status, `error: ${result.error}, signal: ${result.signal}\nstderr: ${result.stderr}`).toBe(0);
  return join(host, "out/host");
}

// After `lastLine`, so the names above are set when Node runs this body.
describe("the code-worker entry", () => {
  afterEach(() => {
    for (const pid of groups.splice(0)) killGroup(pid);
    for (const dir of workdirs.splice(0)) rmSync(dir, { recursive: true, force: true });
  });

  test.skipIf(!NODE)("on Node it resolves to the build and imports without running a job", () => {
    // `node_modules/@ngocbh` doesn't exist, so this resolves only by self-reference through `exports`.
    const name = JSON.stringify(CODE_WORKER);
    const script = `await import(${name}); console.log(import.meta.resolve(${name}));`;
    const result = spawnSync(NODE!, ["--input-type=module", "-e", script], {
      cwd: ROOT,
      encoding: "utf8",
      timeout: 60_000,
    });
    expect(result.status, `signal: ${result.signal}\nstderr: ${result.stderr}`).toBe(0);
    expect(result.stderr).toBe("");
    expect(realpathSync(fileURLToPath(result.stdout.trim()))).toBe(realpathSync(WORKER_BUILT));
  });

  test.if(process.versions.bun !== undefined)(
    "a host compile with a code-worker shim entry runs CODE nodes",
    async () => {
      const result = await runBinary(workdir(CODE_FILES), ["flow.yaml"], { binary: compileHost(true) });
      expect(result.code, context(result)).toBe(0);
      expect(lastLine(result.stdout)).toBe("succeeded 12");
    },
    HOST_TEST_LIMIT,
  );

  test.if(process.versions.bun !== undefined)(
    "a host compile without the shim entry fails its CODE nodes",
    async () => {
      const result = await runBinary(workdir(CODE_FILES), ["flow.yaml"], { binary: compileHost(false) });
      expect(lastLine(result.stdout), context(result)).toStartWith("failed");
      expect(result.stdout).toContain("CODE worker failed: Cannot find module");
    },
    HOST_TEST_LIMIT,
  );
});

/** Whether to pack the package and test the tarball: slow, and the install needs the network. */
const PACK = process.env.YIAN_TEST_PACK === "1";
/** The export paths a consumer imports (`./package.json` is data). */
const IMPORTS = Object.keys(PACKAGE.exports).filter((path) => path !== "./package.json");

describe.if(PACK)("the npm tarball", () => {
  let dir = "";
  let tarball = "";
  let files: string[] = [];

  beforeAll(() => {
    dir = mkdtempSync(join(tmpdir(), "yian-pack-"));
    // Build first and pack with --ignore-scripts: a `prepack` run under `npm pack` prints npm's
    // script banners to stdout, which would break the JSON parse.
    execFileSync("npm", ["run", "build"], { cwd: ROOT, stdio: ["ignore", "ignore", "inherit"] });
    const [packed] = JSON.parse(
      execFileSync("npm", ["pack", "--json", "--ignore-scripts", "--pack-destination", dir], {
        cwd: ROOT,
        encoding: "utf8",
        stdio: ["ignore", "pipe", "inherit"],
      }),
    ) as { filename: string; files: { path: string }[] }[];
    tarball = join(dir, packed!.filename);
    files = packed!.files.map((file) => file.path).sort();
    execFileSync("tar", ["-xzf", tarball, "-C", dir]);
  }, 300_000);

  // Removing the extracted tree and the consumer's node_modules can outlast the 5 s hook default.
  afterAll(() => {
    if (dir !== "") rmSync(dir, { recursive: true, force: true });
  }, 60_000);

  test("holds only the build, the sources, the notices and package.json", () => {
    const top = new Set(["LICENSE", "NOTICE", "README.md", "package.json"]);
    const stray = files.filter(
      (path) =>
        !(path.startsWith("dist/src/") || path.startsWith("src/") || top.has(path)) ||
        /(^|\/)(tests?|fixtures|conformance)\//.test(path) ||
        /\.test\./.test(path),
    );
    expect(stray).toEqual([]);
  });

  test("carries every export target and the bin", () => {
    const targets = Object.values(PACKAGE.exports as Record<string, string | Record<string, string>>)
      .flatMap((target) => (typeof target === "string" ? [target] : Object.values(target)))
      .map((target) => target.replace(/^\.\//, ""));
    expect([...targets, PACKAGE.bin.yian].filter((target) => !files.includes(target))).toEqual([]);
  });

  test("the bin starts with a node shebang", () => {
    expect(readFileSync(join(dir, "package", PACKAGE.bin.yian), "utf8").startsWith("#!/usr/bin/env node\n")).toBe(true);
  });

  test("installs, runs and typechecks for a consumer on plain Node", () => {
    const app = join(dir, "app");
    mkdirSync(app);
    writeFileSync(join(app, "package.json"), JSON.stringify({ name: "consumer", private: true, type: "module" }));
    execFileSync("npm", ["install", "--no-audit", "--no-fund", tarball], { cwd: app, stdio: "pipe" });
    const bin = join(app, "node_modules", ".bin", "yian");
    const version = execFileSync(bin, ["--version"], { cwd: app, encoding: "utf8" });
    expect(version.trim()).toBe(PACKAGE.version);
    // On a Node older than the floor, the installed bin says so instead of exiting 0 silently.
    const preload = join(dir, "old-node.mjs");
    writeFileSync(preload, reportAs("node", "22.17.0"));
    const old = spawnSync(bin, ["--version"], {
      cwd: app,
      env: { ...process.env, NODE_OPTIONS: `--import=${pathToFileURL(preload).href}` },
      encoding: "utf8",
    });
    expect([old.status, old.stdout, old.stderr]).toEqual([1, "", `${FLOOR_ERROR}Node 22.17.0.\n`]);
    const specifier = (path: string) => `@ngocbh/yian${path.slice(1)}`;
    const imports = IMPORTS.map((path) => `await import(${JSON.stringify(specifier(path))});`).join("\n");
    execFileSync("node", ["--input-type=module", "-e", imports], { cwd: app, stdio: "pipe" });
    writeFileSync(
      join(app, "index.ts"),
      IMPORTS.map((path, i) => `import * as m${i} from ${JSON.stringify(specifier(path))};\nvoid m${i};`).join("\n"),
    );
    writeFileSync(
      join(app, "tsconfig.json"),
      JSON.stringify({
        compilerOptions: {
          module: "NodeNext",
          moduleResolution: "NodeNext",
          target: "ES2022",
          strict: true,
          noEmit: true,
          skipLibCheck: false,
          types: ["node"],
          typeRoots: [join(ROOT, "node_modules", "@types")],
        },
        include: ["index.ts"],
      }),
    );
    execFileSync(join(ROOT, "node_modules", ".bin", "tsc"), ["-p", app], { cwd: app, stdio: "pipe" });
  }, 300_000);
});
