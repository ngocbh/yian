// Ported from tests/cli/test_cli_name.py: `test_help_names_composer` and
// `test_docs_and_sources_name_the_composer_command`, the scan over `ts/src`, `ts/tests` and
// `ts/README.md`. Added here: the help printed with no arguments and for a group alone.

import { describe, expect, test } from "bun:test";
import { readdirSync, readFileSync, realpathSync } from "node:fs";
import { join, relative } from "node:path";
import { PassThrough } from "node:stream";
import { fileURLToPath } from "node:url";

import { main } from "../../src/cli/index.ts";
import type { CliIo, Writer } from "../../src/cli/utils.ts";
import { NO_ENV } from "../engine/run_state_fixtures.ts";

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
    expect(stdout()).toMatch(/^ {2}chat +Chat with a flow: each line you type is sent to its Inbox\(str\) or Inbox\(Message\) input\./m);
  });

  test("yian chat --help prints its usage and options", async () => {
    const { io, stdout } = testIo();
    expect(await main(["chat", "--help"], io)).toBe(0);
    expect(stdout()).toContain("Usage: yian chat [OPTIONS] FLOW");
    expect(stdout()).toContain("Opens no MCP servers and loads no skills.");
    for (const option of ["--inbox", "--input", "--inputs", "--log", "--num-workers", "--provider", "--model"]) {
      expect(stdout()).toContain(option);
    }
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

test("docs and sources name the composer command", () => {
  const self = realpathSync(fileURLToPath(import.meta.url));
  const scanned = [join(TS_ROOT, "README.md"), ...files(join(TS_ROOT, "src")), ...files(join(TS_ROOT, "tests"))];
  const stale = scanned
    .filter((path) => realpathSync(path) !== self)
    .flatMap((path) =>
      readFileSync(path, "utf8")
        .split("\n")
        .flatMap((line, index) =>
          STALE.test(line) ? [`${relative(TS_ROOT, path)}:${index + 1}: ${line.trim()}`] : [],
        ),
    );
  expect(stale).toEqual([]);
});
