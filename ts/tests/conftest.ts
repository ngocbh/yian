/**
 * The suite's preload, as Python's `tests/conftest.py` and `tests/cli/conftest.py`: every test
 * starts from built-in yian defaults, an empty private auth store and an empty home, and from an
 * environment that an earlier test's changes no longer reach.
 *
 * At load it clears `XDG_CONFIG_HOME`, every defaults variable, `YIAN_DEFAULT_PROVIDER` and
 * `YIAN_DEFAULT_MODEL` (never `YIAN_TEST_*`, which switch test groups on) and keeps that
 * environment as the baseline. Before each test, `process.env` is put back to the baseline, with
 * `YIAN_AUTH_FILE` at `auth.json` (not created) in a fresh dir inside one per-process `yian-test-*`
 * dir and `HOME` at its `home/`, and the cached defaults are dropped, so the next `current()` reads
 * the built-in ones.
 *
 * Changed: Python sets `YIAN_DEFAULT_PROVIDER`/`YIAN_DEFAULT_MODEL` for every test; TS clears them
 * (no TS test relied on a suite-wide default model). Python points `HOME` at a fresh directory for
 * CLI tests only; TS does it for every test.
 *
 * Node also loads `--import` preloads in each worker thread, such as a CODE body's, which shares
 * the test's environment; there this module does nothing, since a reset would change the
 * environment under the running test and a hook would start a `node:test` run in the worker,
 * whose handlers take the worker's uncaught errors.
 */

import fs from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import process from "node:process";
import { isMainThread } from "node:worker_threads";

import { afterAll, beforeEach } from "bun:test";

import { ENTRIES, forget } from "../src/defaults.ts";
import { runtime } from "../src/platform.ts";

const CLEARED = [
  "XDG_CONFIG_HOME",
  ...ENTRIES.map((each) => each.variable),
  "YIAN_DEFAULT_PROVIDER",
  "YIAN_DEFAULT_MODEL",
];
if (isMainThread) for (const name of CLEARED) delete process.env[name];
const BASELINE: Readonly<Record<string, string | undefined>> = { ...process.env };

/** The per-process parent of the tests' dirs, made on first use. */
let root: string | undefined;

/**
 * Put `process.env` back to the baseline, with a fresh empty auth store and home, and drop the
 * cached defaults.
 */
function isolate(): void {
  for (const name of Object.keys(process.env)) {
    if (!Object.hasOwn(BASELINE, name)) delete process.env[name];
  }
  Object.assign(process.env, BASELINE);
  // Lazy: `--rerun-each` runs the preload's `afterAll` before a rerun, so later tests need a new
  // parent.
  root ??= fs.mkdtempSync(join(tmpdir(), "yian-test-"));
  const dir = fs.mkdtempSync(join(root, "test-"));
  fs.mkdirSync(join(dir, "home"));
  process.env.YIAN_AUTH_FILE = join(dir, "auth.json");
  process.env.HOME = join(dir, "home");
  forget();
}

/** Remove the per-process parent with every test's dir in it. */
function removeRoot(): void {
  if (root !== undefined) fs.rmSync(root, { recursive: true, force: true });
  root = undefined;
}

// Removed once at the end rather than after each test: Bun and Node order a file's `afterEach`
// hooks differently against the preload's, and a test's own cleanup may still need its dir.
// Bun runs every file in one process and never runs `exit` handlers under `bun test`, but runs the
// preload's `afterAll` after the last file's own hooks.
// Node runs each file in its own process and runs the preload's `afterAll` before the file's own,
// so there removal waits for `exit`.
// A killed run, or a Bun run stopped by `--bail` (which runs neither hook), leaves that one dir.
if (isMainThread) {
  if (runtime === "bun") afterAll(removeRoot);
  else process.on("exit", removeRoot);

  // Module-level code and `beforeAll` hooks run before any `beforeEach`. Bun loads the preload
  // once per process, so this call covers only the first test file; later files start with the
  // previous test's store and home, which are temp dirs but not necessarily empty.
  isolate();
  beforeEach(isolate);
}
