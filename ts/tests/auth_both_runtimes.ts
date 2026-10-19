/**
 * The credential store's cross-process scenarios, written once for both runtimes: plain async
 * functions using `node:assert/strict`, so the same checks run as Bun tests
 * (`provider_auth.test.ts`) and, built, on Node, where this module is the main one and prints one
 * JSON line per scenario. Run with `child` as its first argument, it is one of the processes a
 * scenario spawns.
 */

import assert from "node:assert/strict";
import { type ChildProcess, spawn } from "node:child_process";
import fs from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";

import { OAuthCredentials } from "../src/llm_clients/auth/common.ts";
import { AuthStore } from "../src/llm_clients/auth/store.ts";
import { realScheduler } from "../src/platform.ts";

/** A spawned child, with everything it printed so far and its exit. */
interface Child {
  process: ChildProcess;
  stdout: () => string;
  stderr: () => string;
  exited: Promise<number | null>;
}

/** Spawn this module as a child resolving `credentialId` from the store at `path`. */
function spawnChild(path: string, credentialId: string, counterFile: string): Child {
  const child = spawn(process.execPath, [fileURLToPath(import.meta.url), "child", path, credentialId, counterFile], {
    stdio: "pipe",
  });
  let stdout = "";
  let stderr = "";
  child.stdout.setEncoding("utf8").on("data", (chunk: string) => {
    stdout += chunk;
  });
  child.stderr.setEncoding("utf8").on("data", (chunk: string) => {
    stderr += chunk;
  });
  const exited = new Promise<number | null>((resolve, reject) => {
    child.on("error", reject);
    child.on("close", (code) => resolve(code));
  });
  return { process: child, stdout: () => stdout, stderr: () => stderr, exited };
}

/** Resolve once `child` has printed `ready`; reject if it exits first. */
function ready(child: Child): Promise<void> {
  return new Promise((resolve, reject) => {
    const check = () => {
      if (child.stdout().split("\n").includes("ready")) resolve();
    };
    child.process.stdout!.on("data", check);
    check();
    child.exited.then(
      () => reject(new Error(`a child exited before it was ready; stderr:\n${child.stderr()}`)),
      reject,
    );
  });
}

/** Three processes refresh one expiring login at once: exactly one refresh runs. */
async function refreshIsSerializedAcrossProcesses(): Promise<void> {
  const tmp = fs.mkdtempSync(join(tmpdir(), "yian-auth-"));
  const path = join(tmp, "auth", "auth.json");
  const counterFile = join(tmp, "refreshes");
  fs.writeFileSync(counterFile, "");
  const children: Child[] = [];
  let timer: ReturnType<typeof setTimeout> | undefined;
  try {
    const expiring = new OAuthCredentials({ accessToken: "old", refreshToken: "refresh", expiresAt: 900, accountId: "acct" });
    const selection = await new AuthStore(path).save("openai", expiring, "model");
    for (let i = 0; i < 3; i++) children.push(spawnChild(path, selection.credentialId, counterFile));
    const limit = new Promise<never>((_, reject) => {
      timer = setTimeout(() => reject(new Error("the children didn't finish within 20 s")), 20_000);
    });
    await Promise.race([Promise.all(children.map(ready)), limit]);
    for (const child of children) child.process.stdin!.end("go\n");
    const codes = await Promise.race([Promise.all(children.map((child) => child.exited)), limit]);
    for (const [i, child] of children.entries()) {
      assert.equal(codes[i], 0, `a child failed; stderr:\n${child.stderr()}`);
      assert.deepEqual(child.stdout().trim().split("\n"), ["ready", "new"]);
    }
    assert.equal(fs.readFileSync(counterFile, "utf8"), "refresh\n");
    assert.deepEqual(new AuthStore(path).selection(), selection);
  } finally {
    clearTimeout(timer);
    for (const child of children) child.process.kill("SIGKILL");
    fs.rmSync(tmp, { recursive: true, force: true });
  }
}

/** The child side: print `ready`, wait for `go`, resolve the login, print its access token. */
async function child(path: string, credentialId: string, counterFile: string): Promise<void> {
  console.log("ready");
  let input = "";
  for await (const chunk of process.stdin) {
    input += String(chunk);
    if (input.includes("go\n")) break;
  }
  const store = new AuthStore(path, { scheduler: { now: () => 1_000_000, sleep: realScheduler.sleep } });
  const credentials = await store.credentials("openai", credentialId, {
    refresh: async () => {
      fs.appendFileSync(counterFile, "refresh\n");
      await realScheduler.sleep(200);
      return new OAuthCredentials({ accessToken: "new", refreshToken: "rotated", expiresAt: 5000, accountId: "acct" });
    },
  });
  console.log((credentials as OAuthCredentials).accessToken);
}

/** The scenarios by name, in the order `main` runs them. */
export const SCENARIOS: Readonly<Record<string, () => Promise<void>>> = {
  test_refresh_is_serialized_across_processes: refreshIsSerializedAcrossProcesses,
};

/** Run every scenario, printing `{"name", "ok", "error"?}` per line; return how many failed. */
async function main(): Promise<number> {
  let failures = 0;
  for (const [name, run] of Object.entries(SCENARIOS)) {
    try {
      await run();
      console.log(JSON.stringify({ name, ok: true }));
    } catch (error) {
      failures++;
      const text = error instanceof Error ? (error.stack ?? error.message) : String(error);
      console.log(JSON.stringify({ name, ok: false, error: text }));
    }
  }
  return failures;
}

// `import.meta.main` doesn't exist on Node 22.
if (process.argv[1] === fileURLToPath(import.meta.url)) {
  if (process.argv[2] === "child") {
    const [path, credentialId, counterFile] = process.argv.slice(3) as [string, string, string];
    await child(path, credentialId, counterFile);
  } else {
    // No process.exit: a leaked timer or socket then shows as a hang.
    process.exitCode = (await main()) ? 1 : 0;
  }
}
