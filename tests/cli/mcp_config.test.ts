/**
 * Reading `mcpServers` files into unopened MCP servers (`cli/mcp_config.ts`).
 *
 * Ported from tests/cli/test_mcp_config.py, all 23 tests under their names, parametrized rows as
 * `test.each`; the bad-entries table adds the row `{"command": "srv", "disabled": null}`.
 *
 * Changed:
 * - Python's `UserWarning` is the `warn` option; the tests collect its messages in an array.
 * - Python's `isolated_cli_home` fixture is an env with `HOME`.
 * - A value path quotes its key as JSON: `env["A"]`, not Python's `env['A']`.
 * - A built server's options are read through its private `spec` and `label`; Python's
 *   `server._args == ()` is `[]`, and `server._owner is None` is `server.isOpen === false`.
 * - The `mcpServer` validation message is TS's (`mcpServer url must be an http or https URL with a
 *   host`), still matching "http or https".
 *
 * TS-only: the exact messages of a few bad entries (a `null` `args` and `disabled` among them); a
 * `__proto__` server name and a `__proto__` `env` key are kept as data; a duplicate
 * server name and a `NaN` token are not valid JSON; a file that isn't UTF-8 is `not valid JSON: not
 * UTF-8 text`; `userMcpConfig` falls back to the account's home when `HOME` is unset or empty; with
 * `YIAN_TEST_NODE` set, the built module loads the same entries on Node.
 */

import { afterEach, beforeEach, describe, expect, test } from "bun:test";
import { execFileSync, spawnSync } from "node:child_process";
import { existsSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir, userInfo } from "node:os";
import { join } from "node:path";
import { pathToFileURL } from "node:url";

import { loadMcpConfig, McpConfigError, userMcpConfig } from "../../src/cli/mcp_config.ts";
import type { Env } from "../../src/defaults.ts";
import { McpServer, type McpServerSpec } from "../../src/tools/mcp.ts";

let tmp: string;

beforeEach(() => {
  tmp = mkdtempSync(join(tmpdir(), "yian-cli-mcp-config-"));
});

afterEach(() => {
  rmSync(tmp, { recursive: true, force: true });
});

/** An `Env` reading only `vars`. */
function envOf(vars: Record<string, string>): Env {
  return (name) => (Object.hasOwn(vars, name) ? vars[name] : undefined);
}

/** Write `{"mcpServers": servers}` to `name` in the temp dir; return its path. */
function write(name: string, servers: unknown): string {
  const path = join(tmp, name);
  writeFileSync(path, JSON.stringify({ mcpServers: servers }));
  return path;
}

/** The checked options a server was built with. */
function spec(server: McpServer | undefined): McpServerSpec {
  return (server as unknown as { spec: McpServerSpec }).spec;
}

/** Return the `McpConfigError` `loadMcpConfig` throws. */
function loadError(
  paths: readonly string[],
  vars: Record<string, string> = {},
  options: { reserved?: Iterable<string> } = {},
): McpConfigError {
  try {
    loadMcpConfig(paths, envOf(vars), { ...options, warn: () => {} });
  } catch (error) {
    expect(error).toBeInstanceOf(McpConfigError);
    return error as McpConfigError;
  }
  throw new Error("loadMcpConfig didn't throw");
}

test("test_user_config_is_under_the_home_yian_directory", () => {
  expect(userMcpConfig(envOf({ HOME: "/home/me" }))).toBe(join("/home/me", ".yian", "mcp.json"));
});

test.each<Record<string, string>>([{}, { HOME: "" }])("userMcpConfig falls back to the account's home (%j)", (vars) => {
  expect(userMcpConfig(envOf(vars))).toBe(join(userInfo().homedir, ".yian", "mcp.json"));
});

test("test_stdio_entry_expands_env_and_keeps_args", () => {
  const path = write("mcp.json", {
    files: { type: "stdio", command: "npx", args: ["-y", "server-files", "."], env: { T: "${TOK}" }, cwd: "/srv" },
  });

  const servers = loadMcpConfig([path], envOf({ TOK: "abc" }));

  expect(Object.keys(servers)).toEqual(["files"]);
  const server = servers.files;
  expect(server).toBeInstanceOf(McpServer);
  expect(spec(server).command).toBe("npx");
  expect(spec(server).args).toEqual(["-y", "server-files", "."]);
  expect(spec(server).env).toEqual({ T: "abc" });
  expect(spec(server).cwd).toBe("/srv");
  expect(spec(server).url).toBeUndefined();
  expect(server!.label).toBe("files");
  expect(server!.isOpen).toBe(false);
});

test.each(["http", "streamable-http"])("test_http_entry_expands_headers (%s)", (kind) => {
  const path = write("mcp.json", {
    remote: { type: kind, url: "https://tools.example.com/mcp", headers: { Authorization: "Bearer ${TOKEN}" } },
  });

  const server = loadMcpConfig([path], envOf({ TOKEN: "s3cret" })).remote;

  expect(spec(server).url).toBe("https://tools.example.com/mcp");
  expect(spec(server).headers).toEqual({ Authorization: "Bearer s3cret" });
  expect(spec(server).command).toBeUndefined();
  expect(server!.label).toBe("remote");
});

test("test_sse_type_is_rejected", () => {
  const path = write("mcp.json", { old: { type: "sse", url: "https://x.example.com/sse" } });

  const error = loadError([path]);

  expect(error.message).toContain("the SSE transport isn't supported; use streamable HTTP");
  expect(error.message).toContain(path);
  expect(error.message).toContain("'old'");
});

test.each([
  [
    { type: "websocket", url: "https://x.example.com" },
    "'type' must be one of stdio, http, streamable-http; got 'websocket'",
  ],
  [{ type: "stdio", url: "https://x.example.com" }, "'type' 'stdio' needs 'command', not 'url'"],
  [{ type: "http", command: "srv" }, "'type' 'http' needs 'url', not 'command'"],
  [{ type: 3, command: "srv" }, "'type' must be one of stdio, http, streamable-http; got 3"],
  [{ type: ["stdio"], command: "srv" }, `'type' must be one of stdio, http, streamable-http; got ["stdio"]`],
])("test_type_must_be_known_and_agree_with_command_or_url (%j)", (entry, reason) => {
  const path = write("mcp.json", { s: entry });

  expect(loadError([path]).message).toBe(`${path}: server 's': ${reason}`);
});

test("test_missing_variable_names_file_server_and_field", () => {
  const path = write("mcp.json", { files: { command: "srv", args: ["--key", "${MISSING}"] } });

  expect(loadError([path]).message).toBe(
    `${path}: server 'files': args[1]: \${MISSING} is not set; set it or write \${MISSING:-default}`,
  );
});

test.each([
  ["${MISSING:-d}", "d"],
  ["${SET:-d}", "v"],
  ["${MISSING:-}", ""],
  ["x${SET}y", "xvy"],
  ["a$$b", "a$b"],
  ["$${SET}", "${SET}"],
  ["cost $5", "cost $5"],
  ["plain", "plain"],
])("test_expansion_forms (%s)", (text, expected) => {
  const path = write("mcp.json", { s: { command: "srv", args: [text] } });

  const server = loadMcpConfig([path], envOf({ SET: "v" })).s;

  expect(spec(server).args).toEqual([expected]);
});

test("test_expansion_covers_command_cwd_url_and_header_values", () => {
  const path = write("mcp.json", {
    local: { command: "${BIN}/srv", cwd: "${HOME_DIR}" },
    remote: { url: "https://${HOST}/mcp", headers: { "X-Key": "${KEY}" } },
  });
  const vars = { BIN: "/opt/bin", HOME_DIR: "/home/me", HOST: "tools.example.com", KEY: "k" };

  const servers = loadMcpConfig([path], envOf(vars));

  expect(spec(servers.local).command).toBe("/opt/bin/srv");
  expect(spec(servers.local).cwd).toBe("/home/me");
  expect(spec(servers.remote).url).toBe("https://tools.example.com/mcp");
  expect(spec(servers.remote).headers).toEqual({ "X-Key": "k" });
});

test("test_unterminated_reference_is_an_error", () => {
  const path = write("mcp.json", { s: { command: "srv", env: { A: "${OPEN" } } });

  expect(loadError([path], { OPEN: "x" }).message).toBe(
    `${path}: server 's': env["A"]: '\${' must start a \${VAR} or \${VAR:-default} reference;` +
      " write '$$' for a literal '$'",
  );
});

test("test_a_reference_inside_a_default_is_an_error", () => {
  const path = write("mcp.json", { s: { command: "srv", args: ["${A:-${B}}"] } });

  expect(loadError([path], { B: "b" }).message).toContain("args[0]: '${' must start");
});

test("test_a_reserved_server_name_is_an_error_naming_the_file", () => {
  const first = write("first.json", { ok: { command: "srv" } });
  const second = write("second.json", { ask_user: { command: "srv" } });

  const error = loadError([first, second], {}, { reserved: new Set(["ask_user", "load_skill"]) });

  expect(error.message).toBe(
    `${second}: server 'ask_user': the name is reserved (a registered tool or a yian built-in: take,` +
      " sleep, ask, ask_user, load_skill); rename the server",
  );
});

test("test_a_reserved_name_a_later_file_disables_is_no_error", () => {
  const first = write("first.json", { ask_user: { command: "srv" } });
  const second = write("second.json", { ask_user: { disabled: true } });

  expect(Object.keys(loadMcpConfig([first, second], envOf({}), { reserved: ["ask_user"] }))).toEqual([]);
});

test("test_later_file_replaces_same_named_server", () => {
  const first = write("a.json", { s: { command: "old", args: ["1"] }, keep: { command: "k" } });
  const second = write("b.json", { s: { url: "https://new.example.com/mcp" } });

  const servers = loadMcpConfig([first, second], envOf({}));

  expect(Object.keys(servers).sort()).toEqual(["keep", "s"]);
  expect(spec(servers.s).url).toBe("https://new.example.com/mcp");
  expect(spec(servers.s).command).toBeUndefined();
  expect(spec(servers.s).args).toEqual([]);
});

test("test_later_disabled_entry_removes_server", () => {
  const first = write("a.json", { s: { command: "srv" }, keep: { command: "k" } });
  const second = write("b.json", { s: { disabled: true } });

  expect(Object.keys(loadMcpConfig([first, second], envOf({})))).toEqual(["keep"]);
});

test("test_disabled_entry_with_command_is_skipped", () => {
  const path = write("mcp.json", {
    off: { command: "srv", disabled: true },
    on: { command: "srv", disabled: false },
  });

  expect(Object.keys(loadMcpConfig([path], envOf({})))).toEqual(["on"]);
});

test("test_unknown_keys_warn_once_each_and_server_loads", () => {
  const path = write("mcp.json", { s: { command: "srv", timeout: 30, autoApprove: [], description: "x" } });
  const messages: string[] = [];

  const servers = loadMcpConfig([path], envOf({}), { warn: (message) => messages.push(message) });

  expect(messages).toEqual([
    `${path}: server 's': ignoring unknown key 'timeout'`,
    `${path}: server 's': ignoring unknown key 'autoApprove'`,
    `${path}: server 's': ignoring unknown key 'description'`,
  ]);
  expect(spec(servers.s).command).toBe("srv");
});

test("test_unknown_key_warning_is_a_user_warning", () => {
  const path = write("mcp.json", { s: { command: "srv", alwaysAllow: [] } });
  const messages: string[] = [];

  loadMcpConfig([path], envOf({}), { warn: (message) => messages.push(message) });

  expect(messages).toEqual([`${path}: server 's': ignoring unknown key 'alwaysAllow'`]);
});

test("test_not_json_names_the_file", () => {
  const path = join(tmp, "mcp.json");
  writeFileSync(path, "{not json");

  expect(loadError([path]).message).toBe(
    `${path}: the MCP config is not valid JSON:` +
      " Expecting property name enclosed in double quotes: line 1 column 2 (char 1)",
  );
});

test.each([
  ['{"mcpServers": {"s": {"command": "a"}, "s": {"command": "b"}}}', "a duplicate server name"],
  ['{"mcpServers": {"s": {"command": "srv", "timeout": NaN}}}', "a NaN token"],
])("%s is not valid JSON (%s)", (text) => {
  const path = join(tmp, "mcp.json");
  writeFileSync(path, text);

  expect(loadError([path]).message).toStartWith(`${path}: the MCP config is not valid JSON: `);
});

test("a file that isn't UTF-8 is not valid JSON", () => {
  const path = join(tmp, "mcp.json");
  writeFileSync(path, Buffer.from([0x7b, 0xff, 0x7d]));

  expect(loadError([path]).message).toBe(`${path}: the MCP config is not valid JSON: not UTF-8 text`);
});

// Each row is wrapped, since `test.each` spreads an array row into arguments.
test.each([[{}], [{ servers: {} }], [[]], [{ mcpServers: [] }], [{ mcpServers: "x" }]])(
  "test_missing_or_wrong_mcp_servers_names_the_file (%j)",
  (document) => {
    const path = join(tmp, "mcp.json");
    writeFileSync(path, JSON.stringify(document));

    expect(loadError([path]).message).toBe(
      `${path}: the MCP config needs an "mcpServers" object mapping names to servers`,
    );
  },
);

test.each([
  ["srv", null],
  [["srv"], null],
  [{ command: "srv", url: "https://x.example.com/mcp" }, "command"],
  [{}, "command"],
  [{ args: ["a"] }, "command"],
  [{ command: "srv", args: "a b" }, "args"],
  [{ command: "srv", args: ["a", 1] }, "args"],
  [{ command: "srv", env: { A: 1 } }, "env"],
  [{ command: "srv", env: ["A=1"] }, "env"],
  [{ url: "https://x.example.com/mcp", headers: { A: true } }, "headers"],
  [{ command: 5 }, "command"],
  [{ command: "srv", cwd: 1 }, "cwd"],
  [{ url: 5 }, "url"],
  [{ command: "srv", disabled: "yes" }, "disabled"],
  [{ command: "srv", disabled: null }, "disabled"],
])("test_bad_entries_name_file_server_and_field (%j)", (entry, field) => {
  const path = write("mcp.json", { s: entry });

  const message = loadError([path]).message;

  expect(message).toContain(path);
  expect(message).toContain("'s'");
  if (field !== null) expect(message).toContain(field);
});

test("bad entries' messages", () => {
  const cases: [unknown, string][] = [
    ["srv", "the entry must be an object, got a string"],
    [{ command: "srv", args: null }, "'args' must be a list of strings"],
    [{ command: "srv", env: { A: 1 } }, `env["A"] must be a string, got a number`],
    [{ url: "https://x.example.com/mcp", headers: ["A"] }, "'headers' must be an object of strings, got an array"],
    [{ command: "srv", disabled: null }, "'disabled' must be true or false, got null"],
  ];
  for (const [entry, reason] of cases) {
    const path = write("mcp.json", { s: entry });
    expect(loadError([path]).message).toBe(`${path}: server 's': ${reason}`);
  }
});

test("test_mcp_server_validation_errors_are_reraised", () => {
  const path = write("mcp.json", { s: { url: "ftp://x.example.com/mcp" }, t: { command: "" } });

  const error = loadError([path]);

  expect(error.message).toBe(`${path}: server 's': mcpServer url must be an http or https URL with a host`);
  expect(error.cause).toBeInstanceOf(RangeError);
});

test("test_missing_file_names_it", () => {
  const path = join(tmp, "absent.json");

  expect(loadError([path]).message).toBe(`${path}: can't read the MCP config: No such file or directory`);
});

test("test_no_files_gives_no_servers", () => {
  expect(Object.keys(loadMcpConfig([], envOf({})))).toEqual([]);
});

test("null counts as absent for type, env, headers, cwd and url", () => {
  const path = write("mcp.json", {
    local: { command: "srv", type: null, env: null, cwd: null, url: null },
    remote: { url: "https://x.example.com/mcp", headers: null, command: null },
  });

  const servers = loadMcpConfig([path], envOf({}));

  expect(spec(servers["local"])).toMatchObject({ command: "srv", env: undefined, cwd: undefined, url: undefined });
  expect(spec(servers["remote"])).toMatchObject({ url: "https://x.example.com/mcp", headers: undefined });
});

test("a __proto__ server name and a __proto__ env key stay data", () => {
  const path = join(tmp, "mcp.json");
  writeFileSync(path, '{"mcpServers": {"__proto__": {"command": "srv", "env": {"__proto__": "x"}}}}');

  const servers = loadMcpConfig([path], envOf({}));

  expect(Object.getPrototypeOf(servers)).toBeNull();
  expect(Object.keys(servers)).toEqual(["__proto__"]);
  const env = spec(servers["__proto__"]).env!;
  expect(Object.keys(env)).toEqual(["__proto__"]);
  expect(Object.getOwnPropertyDescriptor(env, "__proto__")?.value).toBe("x");
});

/** The `ts/` directory. */
const ROOT = join(import.meta.dirname, "../..");
const BUILT = join(ROOT, "dist/src/cli/mcp_config.js");

/** This process's environment without `YIAN_*` keys, so a child reads no ambient yian settings. */
function childEnv(): NodeJS.ProcessEnv {
  return Object.fromEntries(Object.entries(process.env).filter(([key]) => !key.startsWith("YIAN_")));
}

/**
 * The Node binary named by `YIAN_TEST_NODE`, or `undefined` when it's unset. A set
 * value that can't run, a Node older than the `engines` floor, or a missing build
 * fails the whole file, so a typo can't pass as a run with the Node case skipped.
 * A copy of `tests/engine/profile_memory.test.ts`'s: importing a test file would run its tests.
 */
const NODE = ((): string | undefined => {
  const node = process.env.YIAN_TEST_NODE;
  if (!node) return undefined;
  let version: string;
  try {
    version = execFileSync(node, ["--version"], { encoding: "utf8" }).trim();
  } catch (error) {
    throw new Error(`YIAN_TEST_NODE=${node} doesn't run`, { cause: error });
  }
  const [major = 0, minor = 0] = version.replace(/^v/, "").split(".").map(Number);
  if (major < 22 || (major === 22 && minor < 18)) {
    throw new Error(`YIAN_TEST_NODE=${node} is Node ${version}; the Node checks need 22.18 or later`);
  }
  if (!existsSync(BUILT)) throw new Error(`${BUILT} is missing; run bun run build first`);
  return node;
})();

describe.skipIf(!NODE)("the built module on Node", () => {
  test("Node loads a stdio and an HTTP entry with expansion as Bun does", () => {
    const path = write("mcp.json", {
      local: { command: "${BIN}/srv", args: ["-v", "${MISSING:-d}"], env: { T: "${TOK}" }, cwd: "/srv" },
      remote: { type: "http", url: "https://${HOST}/mcp", headers: { "X-Key": "Bearer ${TOK}" } },
    });
    const vars = { BIN: "/opt/bin", TOK: "abc", HOST: "tools.example.com" };
    const specs = (servers: Record<string, McpServer>) =>
      JSON.stringify(Object.entries(servers).map(([name, server]) => [name, server.label, spec(server)]));
    const script = `
const { loadMcpConfig } = await import(${JSON.stringify(pathToFileURL(BUILT).href)});
const vars = ${JSON.stringify(vars)};
const servers = loadMcpConfig([${JSON.stringify(path)}], (name) => vars[name]);
console.log(JSON.stringify(Object.entries(servers).map(([name, server]) => [name, server.label, server.spec])));
`;
    const result = spawnSync(NODE!, ["--input-type=module", "-e", script], {
      encoding: "utf8",
      timeout: 20_000,
      env: childEnv(),
    });
    expect(result.stderr).toBe("");
    expect(result.status).toBe(0);
    expect(result.stdout).toBe(`${specs(loadMcpConfig([path], envOf(vars)))}\n`);
  });
});
