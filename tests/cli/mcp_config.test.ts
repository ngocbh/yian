/**
 * Reading `mcpServers` files into unopened MCP servers (`cli/mcp_config.ts`): the accepted and
 * refused entries, their warnings (collected through the `warn` option) and the user config path.
 * A value path quotes its key as JSON (`env["A"]`).
 */

import { afterEach, beforeEach, describe, expect, spyOn, test } from "bun:test";
import { spawnSync } from "node:child_process";
import { chmodSync, mkdtempSync, rmSync, symlinkSync, writeFileSync } from "node:fs";
import { tmpdir, userInfo } from "node:os";
import { join } from "node:path";
import { pathToFileURL } from "node:url";

import { loadMcpConfig, McpConfigError, userMcpConfig } from "../../src/cli/mcp_config.ts";
import type { Env } from "../../src/defaults.ts";
import { McpServer, type McpServerSpec } from "../../src/tools/mcp.ts";
import { testNode } from "../helpers/test_node.ts";

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
  options: { reserved?: Iterable<string>; userConfig?: string } = {},
): McpConfigError {
  try {
    loadMcpConfig(paths, envOf(vars), { ...options, warn: () => {} });
  } catch (error) {
    expect(error).toBeInstanceOf(McpConfigError);
    return error as McpConfigError;
  }
  throw new Error("loadMcpConfig didn't throw");
}

test("user config is under the home yian directory", () => {
  expect(userMcpConfig(envOf({ HOME: "/home/me" }))).toBe(join("/home/me", ".yian", "mcp.json"));
});

test.each<Record<string, string>>([{}, { HOME: "" }])("userMcpConfig falls back to the account's home (%j)", (vars) => {
  expect(userMcpConfig(envOf(vars))).toBe(join(userInfo().homedir, ".yian", "mcp.json"));
});

describe.skipIf(process.platform === "win32")("the user config's owner and mode", () => {
  const servers = { files: { command: "npx" } };

  test("a user config others may write is refused, naming the fix; one they may only read is read", () => {
    const path = write("mcp.json", servers);
    for (const mode of [0o620, 0o602, 0o666]) {
      chmodSync(path, mode);
      expect(loadError([path], {}, { userConfig: path }).message).toBe(
        `${path}: the MCP config can be changed by other users, and its commands run as you; fix it with: chmod go-w ${path}`,
      );
    }
    for (const mode of [0o600, 0o644, 0o444, 0o755]) {
      chmodSync(path, mode);
      expect(Object.keys(loadMcpConfig([path], envOf({}), { userConfig: path }))).toEqual(["files"]);
    }
  });

  test("a user config another user owns is refused", () => {
    const path = write("mcp.json", servers);
    const uid = spyOn(process, "getuid").mockReturnValue(process.getuid!() + 1);
    try {
      expect(loadError([path], {}, { userConfig: path }).message).toBe(
        `${path}: the MCP config belongs to another user, and its commands would run as you; keep your own copy`,
      );
    } finally {
      uid.mockRestore();
    }
  });

  test("a symlinked user config is checked at its target", () => {
    const target = write("dotfiles.json", servers);
    const path = join(tmp, "mcp.json");
    symlinkSync(target, path);
    expect(Object.keys(loadMcpConfig([path], envOf({}), { userConfig: path }))).toEqual(["files"]);
    chmodSync(target, 0o666);
    expect(loadError([path], {}, { userConfig: path }).message).toStartWith(`${path}: the MCP config can be changed by other users`);
  });

  test("an --mcp-config file is read whatever its owner and mode", () => {
    const user = write("mcp.json", servers);
    const flag = write("flag.json", { other: { command: "npx" } });
    chmodSync(flag, 0o666);
    expect(Object.keys(loadMcpConfig([user, flag], envOf({}), { userConfig: user }))).toEqual(["files", "other"]);
  });
});

test("stdio entry expands env and keeps args", () => {
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

test("sse type is rejected", () => {
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

test("missing variable names file server and field", () => {
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

test("expansion covers command cwd url and header values", () => {
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

test("unterminated reference is an error", () => {
  const path = write("mcp.json", { s: { command: "srv", env: { A: "${OPEN" } } });

  expect(loadError([path], { OPEN: "x" }).message).toBe(
    `${path}: server 's': env["A"]: '\${' must start a \${VAR} or \${VAR:-default} reference;` +
      " write '$$' for a literal '$'",
  );
});

test("a reference inside a default is an error", () => {
  const path = write("mcp.json", { s: { command: "srv", args: ["${A:-${B}}"] } });

  expect(loadError([path], { B: "b" }).message).toContain("args[0]: '${' must start");
});

test("a reserved server name is an error naming the file", () => {
  const first = write("first.json", { ok: { command: "srv" } });
  const second = write("second.json", { ask_user: { command: "srv" } });

  const error = loadError([first, second], {}, { reserved: new Set(["ask_user", "load_skill"]) });

  expect(error.message).toBe(
    `${second}: server 'ask_user': the name is reserved (a registered tool or a yian built-in: take,` +
      " sleep, ask, ask_user, load_skill); rename the server",
  );
});

test("a reserved name a later file disables is no error", () => {
  const first = write("first.json", { ask_user: { command: "srv" } });
  const second = write("second.json", { ask_user: { disabled: true } });

  expect(Object.keys(loadMcpConfig([first, second], envOf({}), { reserved: ["ask_user"] }))).toEqual([]);
});

test("later file replaces same named server", () => {
  const first = write("a.json", { s: { command: "old", args: ["1"] }, keep: { command: "k" } });
  const second = write("b.json", { s: { url: "https://new.example.com/mcp" } });

  const servers = loadMcpConfig([first, second], envOf({}));

  expect(Object.keys(servers).sort()).toEqual(["keep", "s"]);
  expect(spec(servers.s).url).toBe("https://new.example.com/mcp");
  expect(spec(servers.s).command).toBeUndefined();
  expect(spec(servers.s).args).toEqual([]);
});

test("later disabled entry removes server", () => {
  const first = write("a.json", { s: { command: "srv" }, keep: { command: "k" } });
  const second = write("b.json", { s: { disabled: true } });

  expect(Object.keys(loadMcpConfig([first, second], envOf({})))).toEqual(["keep"]);
});

test("disabled entry with command is skipped", () => {
  const path = write("mcp.json", {
    off: { command: "srv", disabled: true },
    on: { command: "srv", disabled: false },
  });

  expect(Object.keys(loadMcpConfig([path], envOf({})))).toEqual(["on"]);
});

test("unknown keys warn once each and server loads", () => {
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

test("unknown key warning is a user warning", () => {
  const path = write("mcp.json", { s: { command: "srv", alwaysAllow: [] } });
  const messages: string[] = [];

  loadMcpConfig([path], envOf({}), { warn: (message) => messages.push(message) });

  expect(messages).toEqual([`${path}: server 's': ignoring unknown key 'alwaysAllow'`]);
});

test("not json names the file", () => {
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
    [{ command: "srv", args: "a" }, "'args' must be a list of strings"],
    [{ command: "srv", env: { A: 1 } }, `env["A"] must be a string, got a number`],
    [{ url: "https://x.example.com/mcp", headers: ["A"] }, "'headers' must be an object of strings, got an array"],
    [{ command: "srv", disabled: "yes" }, "'disabled' must be true or false, got a string"],
  ];
  for (const [entry, reason] of cases) {
    const path = write("mcp.json", { s: entry });
    expect(loadError([path]).message).toBe(`${path}: server 's': ${reason}`);
  }
});

test("an http server off this machine names the file and server", () => {
  const path = write("mcp.json", { remote: { url: "http://tools.example.com/mcp", headers: { Authorization: "Bearer t" } } });
  expect(loadError([path]).message).toBe(
    `${path}: server 'remote': mcpServer url must use https; http is allowed only for localhost, 127.0.0.1, and ::1`,
  );
});

test("mcp server validation errors are reraised", () => {
  const path = write("mcp.json", { s: { url: "ftp://x.example.com/mcp" }, t: { command: "" } });

  const error = loadError([path]);

  expect(error.message).toBe(`${path}: server 's': mcpServer url must be an http or https URL with a host`);
  expect(error.cause).toBeInstanceOf(RangeError);
});

test("missing file names it", () => {
  const path = join(tmp, "absent.json");

  expect(loadError([path]).message).toBe(`${path}: can't read the MCP config: No such file or directory`);
});

test("no files gives no servers", () => {
  expect(Object.keys(loadMcpConfig([], envOf({})))).toEqual([]);
});

test("concurrencySafe marks all or some of a server's tools, without a warning", () => {
  const path = write("mcp.json", {
    all: { command: "srv", concurrencySafe: true },
    none: { command: "srv", concurrencySafe: false },
    some: { url: "https://x.example.com/mcp", concurrencySafe: ["search", "fetch"] },
    unset: { command: "srv" },
  });
  const messages: string[] = [];

  const servers = loadMcpConfig([path], envOf({}), { warn: (message) => messages.push(message) });

  expect(messages).toEqual([]);
  expect(spec(servers["all"]).concurrencySafe).toBe(true);
  expect(spec(servers["none"]).concurrencySafe).toBe(false);
  expect(spec(servers["some"]).concurrencySafe).toEqual(["search", "fetch"]);
  expect(spec(servers["unset"]).concurrencySafe).toBe(false);
});

test.each<[unknown, string]>([
  ["yes", "a string"],
  [[1], "an array holding a number"],
  [{ search: true }, "an object"],
])("concurrencySafe %j names the file and the server", (value, got) => {
  const path = write("mcp.json", { s: { command: "srv", concurrencySafe: value } });
  expect(loadError([path]).message).toBe(
    `${path}: server 's': 'concurrencySafe' must be true, false or a list of tool names, got ${got}`,
  );
});

test("null counts as absent for every key", () => {
  const path = write("mcp.json", {
    local: { command: "srv", args: null, type: null, env: null, cwd: null, url: null, disabled: null, concurrencySafe: null },
    remote: { url: "https://x.example.com/mcp", headers: null, command: null },
  });

  const servers = loadMcpConfig([path], envOf({}));

  expect(spec(servers["local"])).toMatchObject({
    command: "srv",
    args: [],
    env: undefined,
    cwd: undefined,
    url: undefined,
    concurrencySafe: false,
  });
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

/** The package root. */
const ROOT = join(import.meta.dirname, "../..");
const BUILT = join(ROOT, "dist/src/cli/mcp_config.js");

/** This process's environment without `YIAN_*` keys, so a child reads no ambient yian settings. */
function childEnv(): NodeJS.ProcessEnv {
  return Object.fromEntries(Object.entries(process.env).filter(([key]) => !key.startsWith("YIAN_")));
}

const NODE = testNode(BUILT);

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
