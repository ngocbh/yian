/**
 * Helpers for tests that run the fake MCP server in `tests/fixtures/mcp/`, a port of yian's
 * `tests/engine/mcp_helpers.py`. They import no `bun:test`, so they also run built on Node.
 */

import { execFileSync, spawnSync } from "node:child_process";
import { existsSync, readFileSync } from "node:fs";
import { type AddressInfo, createServer } from "node:net";
import { join } from "node:path";
import { fileURLToPath } from "node:url";

// Built, this file runs as `dist/tests/engine/mcp_helpers.js`, one directory deeper.
const BUILT = import.meta.url.endsWith(".js");
/** The `ts/` directory. */
export const ROOT = fileURLToPath(new URL(BUILT ? "../../../" : "../../", import.meta.url));
const FAKE_SOURCE = join(ROOT, "tests", "fixtures", "mcp", "fake_server.ts");
const FAKE_BUILT = join(ROOT, "dist", "tests", "fixtures", "mcp", "fake_server.js");

/** The fake server script: the source on Bun, the built file on Node. */
export const FAKE = BUILT ? FAKE_BUILT : FAKE_SOURCE;

/**
 * The Node binary named by `YIAN_TEST_NODE`, or `undefined` when it's unset. A set value that
 * can't run, a Node older than the `engines` floor, or a missing build fails the importing test
 * file, so a typo can't pass as a run with the Node rows skipped. A copy of
 * `function_tool.test.ts`'s.
 */
export const NODE = ((): string | undefined => {
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
  const built = join(ROOT, "dist", "src", "tools", "index.js");
  if (!existsSync(built)) throw new Error(`${built} is missing; run bun run build first`);
  return node;
})();

/** The command and arguments that start the fake server over stdio with `flags`. */
export function stdioOptions(...flags: string[]): { command: string; args: string[] } {
  // Without `--no-install`, Bun may try to install the fixture's imports from the network.
  const runtimeFlags = process.versions.bun === undefined ? [] : ["--no-install"];
  return { command: process.execPath, args: [...runtimeFlags, FAKE, ...flags] };
}

/** A TCP port on 127.0.0.1 that was free a moment ago. */
export function freePort(): Promise<number> {
  return new Promise((resolve, reject) => {
    const server = createServer();
    server.once("error", reject);
    server.listen(0, "127.0.0.1", () => {
      const { port } = server.address() as AddressInfo;
      server.close(() => resolve(port));
    });
  });
}

/** Whether process `pid` exists and is not a zombie. */
export function childAlive(pid: number): boolean {
  try {
    process.kill(pid, 0);
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === "EPERM") return true;
    if ((error as NodeJS.ErrnoException).code === "ESRCH") return false;
    throw error;
  }
  // A zombie still answers signal 0; only its parent can reap it.
  const stat = spawnSync("ps", ["-o", "stat=", "-p", String(pid)], { encoding: "utf8" }).stdout.trim();
  return stat !== "" && !stat.startsWith("Z");
}

/** The pids listed in a marker or grandchild file, one per line; a missing file lists none. */
export function pids(file: string): number[] {
  if (!existsSync(file)) return [];
  return readFileSync(file, "utf8")
    .split("\n")
    .filter((line) => /^\d+$/.test(line.trim()))
    .map(Number);
}

/** Poll `predicate` every 20 ms for up to `ms`; return whether it became true. */
export async function waitFor(predicate: () => boolean | Promise<boolean>, ms: number): Promise<boolean> {
  const deadline = Date.now() + ms;
  for (;;) {
    if (await predicate()) return true;
    if (Date.now() >= deadline) return false;
    await new Promise((resolve) => setTimeout(resolve, 20));
  }
}

/** The `ps` lines of running fake servers, from the source or the built file. */
export function strayProcesses(): string[] {
  const listing = spawnSync("ps", ["-eo", "pid,command"], { encoding: "utf8" }).stdout;
  return listing.split("\n").filter((line) => line.includes(FAKE_SOURCE) || line.includes(FAKE_BUILT));
}

/** Servers a test opened, to close in `afterEach`. */
export interface ServerTracker {
  /** Remember `server` and return it. */
  track<T extends { close(): Promise<void> }>(server: T): T;
  /** Close every server remembered since the last call. */
  closeAll(): Promise<void>;
}

/** A new {@link ServerTracker}. */
export function trackServers(): ServerTracker {
  const open: { close(): Promise<void> }[] = [];
  return {
    track(server) {
      open.push(server);
      return server;
    },
    async closeAll() {
      await Promise.all(open.splice(0).map((server) => server.close()));
    },
  };
}

/** SIGKILL every pid listed in `files`, ignoring pids already gone. */
export function killMarkers(files: Iterable<string>): void {
  for (const file of files) {
    for (const pid of pids(file)) {
      try {
        process.kill(pid, "SIGKILL");
      } catch {
        // Already gone.
      }
    }
  }
}
