// The CLI names its command: the help with no arguments and for a group alone, and a scan of
// `src`, `tests` and `README.md` for the old `ac` name.

import { describe, expect, test } from "bun:test";
import { readdirSync } from "node:fs";
import { readFile } from "node:fs/promises";
import { join, relative } from "node:path";
import { PassThrough } from "node:stream";
import { fileURLToPath } from "node:url";

import { main } from "../../src/cli/index.ts";
import type { CliIo, Writer } from "../../src/cli/utils.ts";
import { NO_ENV } from "../helpers/run_state_fixtures.ts";

/** Return a `CliIo` with an empty environment whose streams are collected. */
function testIo() {
  const out: string[] = [];
  const err: string[] = [];
  const writer = (chunks: string[]): Writer => ({ write: async (text) => void chunks.push(text) });
  const io: CliIo = {
    stdout: writer(out),
    stderr: writer(err),
    stdin: new PassThrough(),
    stderrStream: new PassThrough(),
    env: NO_ENV,
  };
  return { io, stdout: () => out.join(""), stderr: () => err.join("") };
}

describe("yian help", () => {
  test("help names composer", async () => {
    const { io, stdout } = testIo();
    expect(await main(["--help"], io)).toBe(0);
    expect(stdout()).toContain("Usage: yian ");
  });

  test("no arguments print the root help and exit 2", async () => {
    const { io, stdout } = testIo();
    expect(await main([], io)).toBe(2);
    expect(stdout()).toContain("Usage: yian [OPTIONS] COMMAND [ARGS]...");
    expect(stdout()).toContain("Run agent flows from the command line.");
    expect(stdout()).toMatch(/^ {2}skills +Check skill directories\.$/m);
    expect(stdout()).toMatch(/^ {2}chat +Chat with yian's assistant: it helps you find, read, check, write and run/m);
  });

  test("yian chat --help prints its usage and options", async () => {
    const { io, stdout } = testIo();
    expect(await main(["chat", "--help"], io)).toBe(0);
    expect(stdout()).toContain("Usage: yian chat [OPTIONS] [FLOW]");
    expect(stdout()).toContain("With FLOW, the chat is about that flow.");
    for (const option of ["--log", "--provider", "--model", "--engine-trace"]) expect(stdout()).toContain(option);
    // The flow's inputs and inbox are `yian run`'s: the chat runs its own assistant.
    for (const option of ["--inbox", "--input", "--inputs", "--num-workers"]) expect(stdout()).not.toContain(option);
  });

  test("a group alone prints its help and exits 2", async () => {
    for (const [group, summary] of [
      ["skills", "Check skill directories."],
      ["config", "Check an llm_config layer."],
    ]) {
      const { io, stdout } = testIo();
      expect(await main([group!], io)).toBe(2);
      expect(stdout()).toContain(`Usage: yian ${group} [OPTIONS] COMMAND [ARGS]...`);
      expect(stdout()).toContain(summary!);
      expect(stdout()).toMatch(/^ {2}check +\S/m);
    }
  });
});

const TS_ROOT = fileURLToPath(new URL("../..", import.meta.url));
const STALE = /\bac (?:(?:run|chat|auth)\b|--help\b|\.\.\.)|`ac`|\bThe ac CLI\b/;

/** Return every file under `dir`, recursively. */
function files(dir: string): string[] {
  return readdirSync(dir, { recursive: true, withFileTypes: true })
    .filter((entry) => entry.isFile())
    .map((entry) => join(entry.parentPath, entry.name));
}

// Reading the ~3,000 files in parallel batches takes 2-5 s on a loaded machine (load 50), against about 30 s
// one at a time.
test("docs and sources name the composer command", async () => {
  const self = fileURLToPath(import.meta.url);
  const scanned = [join(TS_ROOT, "README.md"), ...files(join(TS_ROOT, "src")), ...files(join(TS_ROOT, "tests"))];
  // Batches keep the open files under a default descriptor limit (256 on macOS).
  const texts: string[] = [];
  for (let at = 0; at < scanned.length; at += 64) {
    texts.push(...(await Promise.all(scanned.slice(at, at + 64).map((path) => readFile(path, "utf8")))));
  }
  const stale = scanned.flatMap((path, at) =>
    path === self || !STALE.test(texts[at]!)
      ? []
      : texts[at]!
          .split("\n")
          .flatMap((line, index) =>
            STALE.test(line) ? [`${relative(TS_ROOT, path)}:${index + 1}: ${line.trim()}`] : [],
          ),
  );
  expect(stale).toEqual([]);
}, 20_000);
