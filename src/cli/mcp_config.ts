/**
 * Read `mcpServers` JSON files into unopened MCP servers for `yian run`.
 *
 * The CLI reads the user's `~/.yian/mcp.json` when it exists, plus each file named with
 * `--mcp-config`; nothing is discovered from the working directory. Each file holds the common
 * `{"mcpServers": {name: entry}}` shape. `${VAR}` and `${VAR:-default}` in string values expand
 * from the given environment at load.
 */

import { closeSync, constants, fstatSync, openSync, readFileSync } from "node:fs";
import { join } from "node:path";
import process from "node:process";

import type { Env } from "../defaults.ts";
import { reprText, shellQuote } from "../diagnostics.ts";
import { homeDir, isSystemError, warn as platformWarn } from "../platform.ts";
import { type McpServer, mcpServer } from "../tools/mcp.ts";
import { RESERVED_TOOL_NAMES } from "../tools/registry.ts";
import { JSONDecodeError, type JsonObject, type JsonValue, parseJson } from "../typesys/json.ts";
import { type Value, writeAsJson, writeAsText } from "../typesys/values.ts";
import { osReason } from "./utils.ts";

const KNOWN_KEYS: ReadonlySet<string> = new Set([
  "command",
  "args",
  "env",
  "cwd",
  "url",
  "headers",
  "type",
  "disabled",
]);
const TYPES: readonly string[] = ["stdio", "http", "streamable-http"];
// `$$` first so `$${X}` is a literal `${X}`; any other `${` that doesn't match the reference form
// falls to the last branch and is an error. A default can't hold `$`, so a nested `${A:-${B}}` is
// that error rather than the literal text `${B}`.
const REFERENCE = /\$\$|\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}$]*))?\}|\$\{/g;

/** An MCP config file can't be read; the message names the file and server (and field). */
export class McpConfigError extends RangeError {
  override name = "McpConfigError";
}

/**
 * Return the path of the user's MCP config file: `~/.yian/mcp.json`, whether or not it exists,
 * under `HOME` from `env` when set and non-empty, else the account's home directory.
 */
export function userMcpConfig(env: Env): string {
  return join(homeDir(env), ".yian", "mcp.json");
}

/**
 * Read `mcpServers` files, in order, into unopened MCP servers keyed by name.
 *
 * A later file's entry replaces an earlier entry of the same name whole; an entry with
 * `"disabled": true` drops the server, so `{"disabled": true}` alone turns off a server an earlier
 * file defines. Entry keys read are `command`, `args`, `env`, `cwd`, `url`, `headers`, `type`
 * (`"stdio"`, `"http"` or `"streamable-http"`) and `disabled`, each absent when `null`; any other
 * key is ignored with a warning. In `command`, each `args` item, `cwd`, `url`, and the values of
 * `env` and `headers`, `${VAR}` expands to `env(VAR)`, `${VAR:-default}` to `default` when `VAR`
 * is unset or empty, and `$$` to `$`; a default can't contain `$`, and any other `$` stays as
 * written.
 *
 * The user's own config, `userConfig`, holds commands that run as this user, so it is read only
 * when this user owns it (its target, for a symlink) and no one else may write it. Other files,
 * which the command line names each time, aren't checked.
 *
 * @param paths The config files to read, each of which must exist. Later files win.
 * @param env The variables `${VAR}` references expand from.
 * @param options `reserved` names no enabled server may take, such as the run's other tool ids
 *   (default none); `warn` receives each unknown-key warning (default yian's `warn`);
 *   `userConfig` is the path in `paths`, if any, that is the user's own config (`userMcpConfig`).
 * @returns The enabled servers by name, in a null-prototype record, each labelled with its name
 *   and not yet open. As in any JS record, integer-like names come first, then the rest in file order.
 * @throws {McpConfigError} When a file can't be read or isn't JSON, has no `mcpServers` object,
 *   or an entry isn't an object, has both or neither of `command` and `url`, has a value of the
 *   wrong type, a `type` that is unknown, `"sse"`, or disagrees with `command`/`url`, references an
 *   unset variable without a default or holds a `$` in a default, fails `mcpServer`'s checks, or
 *   when an enabled server's name is in `reserved`; and when `userConfig` belongs to another user
 *   or has a group or other write bit (not on Windows, which has neither).
 *
 * @example
 * const user = userMcpConfig(processEnv);
 * const paths = statSync(user, { throwIfNoEntry: false })?.isFile() ? [user] : [];
 * const servers = loadMcpConfig(paths, processEnv, { userConfig: user });
 */
export function loadMcpConfig(
  paths: readonly string[],
  env: Env,
  {
    reserved = [],
    warn = platformWarn,
    userConfig,
  }: { reserved?: Iterable<string>; warn?: (message: string) => void; userConfig?: string } = {},
): Record<string, McpServer> {
  // Null-prototype records, so a server named `__proto__` stays data.
  const servers: Record<string, McpServer> = Object.create(null);
  const sources: Record<string, string> = Object.create(null);
  for (const path of paths) {
    for (const [name, entry] of Object.entries(readEntries(path, path === userConfig))) {
      const server = build(path, name, entry, env, warn);
      if (server === null) {
        delete servers[name];
      } else {
        servers[name] = server;
        sources[name] = path;
      }
    }
  }
  const taken = new Set(reserved);
  for (const name of Object.keys(servers)) {
    if (taken.has(name)) {
      throw new McpConfigError(
        `${sources[name]}: server ${reprText(name)}: the name is reserved (a registered tool or a yian` +
          ` built-in: ${[...RESERVED_TOOL_NAMES].join(", ")}); rename the server`,
      );
    }
  }
  return servers;
}

/**
 * Return the `mcpServers` object of the file at `path`; with `own`, only when this user owns the
 * file and no one else may write it.
 */
function readEntries(path: string, own: boolean): JsonObject {
  let bytes: Buffer;
  try {
    bytes = own ? readOwnFile(path) : readFileSync(path);
  } catch (error) {
    if (!isSystemError(error)) throw error;
    throw new McpConfigError(`${path}: can't read the MCP config: ${osReason(error)}`, { cause: error });
  }
  let text: string;
  try {
    text = new TextDecoder("utf-8", { fatal: true, ignoreBOM: true }).decode(bytes);
  } catch (error) {
    if (!(error instanceof TypeError)) throw error;
    throw new McpConfigError(`${path}: the MCP config is not valid JSON: not UTF-8 text`, { cause: error });
  }
  let document: JsonValue;
  try {
    document = parseJson(text, "plain");
  } catch (error) {
    if (!(error instanceof JSONDecodeError)) throw error;
    throw new McpConfigError(`${path}: the MCP config is not valid JSON: ${error.message}`, { cause: error });
  }
  const servers =
    typeof document === "object" &&
    document !== null &&
    !Array.isArray(document) &&
    Object.hasOwn(document, "mcpServers")
      ? document.mcpServers
      : null;
  if (typeof servers !== "object" || servers === null || Array.isArray(servers)) {
    throw new McpConfigError(`${path}: the MCP config needs an "mcpServers" object mapping names to servers`);
  }
  return servers;
}

/**
 * Return the bytes of the file at `path`, which this user must own and no one else may write.
 * The checks read the opened file, so a file swapped in after them is never read.
 *
 * @throws {McpConfigError} When another user owns the file or it has a group or other write bit.
 * @throws the `fs` error of a failed open or read.
 */
function readOwnFile(path: string): Buffer {
  // Non-blocking, so a FIFO swapped in can't hang the open; its read fails instead.
  const fd = openSync(path, constants.O_RDONLY | (constants.O_NONBLOCK ?? 0));
  try {
    // Windows has no uid and no POSIX mode bits to check.
    if (process.getuid !== undefined) {
      const info = fstatSync(fd);
      if (info.uid !== process.getuid()) {
        throw new McpConfigError(
          `${path}: the MCP config belongs to another user, and its commands would run as you; keep your own copy`,
        );
      }
      if ((info.mode & 0o022) !== 0) {
        throw new McpConfigError(
          `${path}: the MCP config can be changed by other users, and its commands run as you; fix it with: ` +
            `chmod go-w ${shellQuote(path)}`,
        );
      }
    }
    return readFileSync(fd);
  } finally {
    closeSync(fd);
  }
}

/** Return the server one entry describes, or `null` when it is disabled. */
function build(
  path: string,
  name: string,
  entry: JsonValue,
  env: Env,
  warn: (message: string) => void,
): McpServer | null {
  const where = `${path}: server ${reprText(name)}`;
  if (typeof entry !== "object" || entry === null || Array.isArray(entry)) {
    throw new McpConfigError(`${where}: the entry must be an object, got ${kind(entry)}`);
  }
  for (const key of Object.keys(entry)) {
    if (!KNOWN_KEYS.has(key)) warn(`${where}: ignoring unknown key ${reprText(key)}`);
  }
  const disabled = Object.hasOwn(entry, "disabled") ? ((entry.disabled as JsonValue) ?? false) : false;
  if (typeof disabled !== "boolean") {
    throw new McpConfigError(`${where}: 'disabled' must be true or false, got ${kind(disabled)}`);
  }
  if (disabled) return null;

  const command = optionalString(where, entry, "command");
  const url = optionalString(where, entry, "url");
  if ((command === undefined) === (url === undefined)) {
    throw new McpConfigError(`${where}: give exactly one of 'command' (a stdio server) and 'url' (an HTTP server)`);
  }
  const type = Object.hasOwn(entry, "type") ? (entry.type as JsonValue) : null;
  if (type !== null) {
    if (type === "sse") {
      throw new McpConfigError(`${where}: 'type': the SSE transport isn't supported; use streamable HTTP`);
    }
    if (typeof type !== "string" || !TYPES.includes(type)) {
      const shown = typeof type === "string" ? reprText(type) : writeAsText(type as Value);
      throw new McpConfigError(`${where}: 'type' must be one of ${TYPES.join(", ")}; got ${shown}`);
    }
    const [needed, other] = type === "stdio" ? (["command", "url"] as const) : (["url", "command"] as const);
    if ((needed === "command" ? command : url) === undefined) {
      throw new McpConfigError(`${where}: 'type' ${reprText(type)} needs ${reprText(needed)}, not ${reprText(other)}`);
    }
  }

  const args = Object.hasOwn(entry, "args") ? (entry.args ?? []) : [];
  if (!Array.isArray(args) || !args.every((arg) => typeof arg === "string")) {
    throw new McpConfigError(`${where}: 'args' must be a list of strings`);
  }
  const vars = optionalStringRecord(where, entry, "env");
  const headers = optionalStringRecord(where, entry, "headers");
  const cwd = optionalString(where, entry, "cwd");

  const expandText = (field: string, text: string | undefined): string | undefined =>
    text === undefined ? undefined : expand(where, field, text, env);
  // `Object.fromEntries` defines each key, so a `__proto__` key stays data.
  const expandRecord = (field: string, value: Record<string, string> | undefined) =>
    value === undefined
      ? undefined
      : Object.fromEntries(
          Object.entries(value).map(([key, item]) => [
            key,
            expand(where, `${field}[${writeAsJson(key)}]`, item, env),
          ]),
        );

  const fields = {
    command: expandText("command", command),
    args: args.map((arg, index) => expand(where, `args[${index}]`, arg, env)),
    env: expandRecord("env", vars),
    cwd: expandText("cwd", cwd),
    url: expandText("url", url),
    headers: expandRecord("headers", headers),
  };
  try {
    return mcpServer({ ...fields, label: name });
  } catch (error) {
    if (!(error instanceof TypeError || error instanceof RangeError)) throw error;
    throw new McpConfigError(`${where}: ${error.message}`, { cause: error });
  }
}

/** Return `entry[key]` when it is a string, `undefined` when absent or `null`, else throw. */
function optionalString(where: string, entry: JsonObject, key: string): string | undefined {
  const value = Object.hasOwn(entry, key) ? (entry[key] as JsonValue) : null;
  if (value === null) return undefined;
  if (typeof value !== "string") {
    throw new McpConfigError(`${where}: ${reprText(key)} must be a string, got ${kind(value)}`);
  }
  return value;
}

/**
 * Return `entry[key]` when it is an object of strings, `undefined` when absent or `null`, else
 * throw.
 */
function optionalStringRecord(where: string, entry: JsonObject, key: string): Record<string, string> | undefined {
  const value = Object.hasOwn(entry, key) ? (entry[key] as JsonValue) : null;
  if (value === null) return undefined;
  if (typeof value !== "object" || Array.isArray(value)) {
    throw new McpConfigError(`${where}: ${reprText(key)} must be an object of strings, got ${kind(value)}`);
  }
  for (const [itemKey, item] of Object.entries(value)) {
    if (typeof item !== "string") {
      throw new McpConfigError(`${where}: ${key}[${writeAsJson(itemKey)}] must be a string, got ${kind(item)}`);
    }
  }
  return value as Record<string, string>;
}

/** Return `text` with `$$`, `${VAR}` and `${VAR:-default}` expanded from `env`. */
function expand(where: string, field: string, text: string, env: Env): string {
  if (!text.includes("$")) return text;
  return text.replace(REFERENCE, (whole: string, variable: string | undefined, fallback: string | undefined) => {
    if (whole === "$$") return "$";
    if (variable === undefined) {
      throw new McpConfigError(
        `${where}: ${field}: '\${' must start a \${VAR} or \${VAR:-default} reference;` +
          " write '$$' for a literal '$'",
      );
    }
    const value = env(variable);
    if (fallback !== undefined) return value ? value : fallback;
    if (value === undefined) {
      throw new McpConfigError(
        `${where}: ${field}: \${${variable}} is not set; set it or write \${${variable}:-default}`,
      );
    }
    return value;
  });
}

/** Return the JSON name of `value`'s type, for error messages. */
function kind(value: JsonValue): string {
  if (value === null) return "null";
  if (typeof value === "boolean") return "a boolean";
  if (typeof value === "number" || typeof value === "bigint") return "a number";
  if (typeof value === "string") return "a string";
  if (Array.isArray(value)) return "an array";
  return "an object";
}
