/**
 * Read `mcpServers` JSON files into unopened MCP servers for `yian run`: the user's
 * `~/.yian/mcp.json` when it exists, plus each `--mcp-config` file. Not ported to TypeScript yet.
 */

import type { Env } from "../defaults.ts";
import type { McpServer } from "../tools/mcp.ts";

/** An MCP config file can't be read; the message names the file and server (and field). */
export class McpConfigError extends RangeError {
  override name = "McpConfigError";
}

/**
 * Return the path of the user's MCP config file, `~/.yian/mcp.json` under `env`'s `HOME`. Not
 * ported to TypeScript yet.
 *
 * @throws {Error} Always, until it is ported.
 */
export function userMcpConfig(env: Env): string {
  throw new Error("userMcpConfig is not ported to TypeScript yet");
}

/**
 * Read `mcpServers` files, in order, into unopened MCP servers keyed by name, `${VAR}` expanded
 * from `env`; no enabled server may take a `reserved` name. Not ported to TypeScript yet.
 *
 * @throws {Error} Always, until it is ported.
 */
export function loadMcpConfig(
  paths: readonly string[],
  env: Env,
  options: { reserved?: Iterable<string> } = {},
): Record<string, McpServer> {
  throw new Error("loadMcpConfig is not ported to TypeScript yet");
}
