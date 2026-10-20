// Ported from tests/cli/test_run.py: `test_bad_kv_is_rejected`, `test_inputs_json_must_be_an_object`,
// `test_inputs_file_must_be_utf8` and `test_flag_overrides_json`, on the helpers those runs go
// through (`parseKv`, `readInputsJson`, `mergeInputs`). Added here: the other `cli/run.ts` helpers
// (`diagnosticJson`, `jsonValue`, `flowBanner`, `stopOnSignals`), `skillsCheck`, and `main`'s
// command table: the defaults check, usage errors and unported commands. The CLI-run tests come
// with `yian run`.

import { afterEach, beforeEach, describe, expect, test } from "bun:test";
import { EventEmitter } from "node:events";
import { spawnSync } from "node:child_process";
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { PassThrough } from "node:stream";
import { fileURLToPath } from "node:url";

import {
  diagnosticJson,
  flowBanner,
  HostStop,
  jsonValue,
  mergeInputs,
  parseKv,
  readInputsJson,
  skillsCheck,
  stopOnSignals,
} from "../../src/cli/run.ts";
import { main } from "../../src/cli/index.ts";
import { type CliIo, printJson, UsageError, type Writer } from "../../src/cli/utils.ts";
import { Diagnostic, DiagnosticFrame, SourceRange } from "../../src/diagnostics.ts";
import { FileRef } from "../../src/typesys/values.ts";

let tmp: string;

beforeEach(() => {
  tmp = mkdtempSync(join(tmpdir(), "yian-cli-run-"));
});

afterEach(() => {
  rmSync(tmp, { recursive: true, force: true });
});

/** Write `content` to a fresh `--inputs` file and return its path. */
function inputsFile(content: string | Uint8Array): string {
  const path = join(tmp, "in.json");
  writeFileSync(path, content);
  return path;
}

/** A writer collecting what it is given. */
function sink() {
  const chunks: string[] = [];
  const out: Writer = { write: async (text) => void chunks.push(text) };
  return { out, text: () => chunks.join("") };
}

describe("parseKv", () => {
  test("bad kv is rejected", () => {
    expect(() => parseKv(["nokv"])).toThrow(new UsageError("Invalid value: --input must be k=v (got 'nokv')"));
  });

  test("splits at the first '=' and strips the key", () => {
    const parsed = parseKv([" topic =a=b", "__proto__=x", "empty="]);
    expect(Object.entries(parsed)).toEqual([
      ["topic", "a=b"],
      ["__proto__", "x"],
      ["empty", ""],
    ]);
  });
});

describe("readInputsJson", () => {
  test("inputs json must be an object", () => {
    for (const [text, kind] of [
      ["null", "null"],
      ["[1, 2]", "array"],
      ['[["topic", "rivers"]]', "array"],
      ['"rivers"', "string"],
      ["3", "number"],
      ["2.5", "number"],
      ["true", "boolean"],
    ]) {
      expect(() => readInputsJson(inputsFile(text!))).toThrow(
        new UsageError(`Invalid value for --inputs: must contain a JSON object, got ${kind}`),
      );
    }
  });

  test("invalid JSON names its line and column", () => {
    expect(() => readInputsJson(inputsFile("{bad"))).toThrow(
      new UsageError(
        "Invalid value for --inputs: invalid JSON at line 1, column 2: " +
          "Expecting property name enclosed in double quotes",
      ),
    );
    // Lines count as Python's text-mode read counts them, `\r\n` and `\r` as one break each.
    expect(() => readInputsJson(inputsFile('{\r\n"a": 1,\r"b" 2}'))).toThrow(
      new UsageError("Invalid value for --inputs: invalid JSON at line 3, column 5: Expecting ':' delimiter"),
    );
    // A BOM is kept, so it is refused as Python refuses it.
    expect(() => readInputsJson(inputsFile("\ufeff{\"a\": 1}"))).toThrow(
      new UsageError(
        "Invalid value for --inputs: invalid JSON at line 1, column 1: " +
          "Unexpected UTF-8 BOM (decode using utf-8-sig)",
      ),
    );
  });

  test("inputs file must be utf8", () => {
    // A UTF-16 export is a usage error, not a decoding crash.
    const utf16 = Buffer.concat([Buffer.from([0xff, 0xfe]), Buffer.from('{"topic": "rivers"}', "utf16le")]);
    expect(() => readInputsJson(inputsFile(utf16))).toThrow(
      new UsageError("Invalid value for --inputs: is not UTF-8 text"),
    );
  });

  test("reads an object, ints as bigints", () => {
    expect(readInputsJson(inputsFile('{"topic": "rivers", "n": 3, "x": 1.5}'))).toEqual({
      topic: "rivers",
      n: 3n,
      x: 1.5,
    });
  });
});

describe("mergeInputs", () => {
  test("flag overrides json", () => {
    const path = inputsFile('{"topic": "rivers", "n": 1}');
    expect(mergeInputs(path, ["topic=mountains"])).toEqual({ topic: "mountains", n: 1n });
    expect(mergeInputs(undefined, ["topic=mountains"])).toEqual({ topic: "mountains" });
    expect(mergeInputs(undefined, [])).toEqual({});
  });
});

describe("diagnosticJson", () => {
  test("ranges as ints; empty frames, notes and note left out", async () => {
    const frame = new DiagnosticFrame({ label: "flow.yaml", text: "x\n", range: new SourceRange(6, 9, 6, 23) });
    const noted = new DiagnosticFrame({
      label: "child.yaml",
      text: "y\n",
      range: new SourceRange(1, 1, 2, 1),
      note: "inside leaf",
    });
    const diag = new Diagnostic({ code: "compose.load", message: "bad", frames: [frame, noted], notes: ["see docs"] });
    expect(diagnosticJson(diag)).toEqual({
      code: "compose.load",
      message: "bad",
      frames: [
        { label: "flow.yaml", range: [6n, 9n, 6n, 23n] },
        { label: "child.yaml", range: [1n, 1n, 2n, 1n], note: "inside leaf" },
      ],
      notes: ["see docs"],
    });
    expect(diagnosticJson(new Diagnostic({ code: "run.setup", message: "m" }))).toEqual({
      code: "run.setup",
      message: "m",
    });
    const { out, text } = sink();
    await printJson({ error: diagnosticJson(new Diagnostic({ code: "c", message: "m", frames: [frame] })) }, out);
    expect(text()).toContain('"range":[6,9,6,23]');
  });
});

describe("jsonValue", () => {
  test("a file becomes its mapping, in lists and records too", () => {
    const file = new FileRef({ uri: "file:///a.txt", mime: "text/plain", name: "a.txt" });
    const plain = { uri: "file:///a.txt", mime: "text/plain", name: "a.txt" };
    expect(jsonValue(file)).toEqual(plain);
    expect(jsonValue([file, { inner: [file], n: 1n }])).toEqual([plain, { inner: [plain], n: 1n }]);
    expect(jsonValue("text")).toBe("text");
    expect(jsonValue(null)).toBe(null);
  });

  test("a record shaped like the escape is escaped once", async () => {
    const { out, text } = sink();
    await printJson({ output: jsonValue({ $record: 1n }) }, out);
    expect(text()).toBe('{"format":"yian.cli.v1","output":{"$record":{"$record":1}}}\n');
  });
});

describe("flowBanner", () => {
  test("boxes the name, version and description", () => {
    const { out } = sink();
    const env = () => undefined;
    expect(flowBanner({ name: "demo", description: "does things", version: "1.0" }, out, env)).toBe(
      [
        "╭────────────────────────────────────╮",
        "│ Running flow: demo  (version: 1.0) │",
        "│ Description: does things           │",
        "╰────────────────────────────────────╯",
        "",
      ].join("\n"),
    );
    expect(flowBanner({ name: null, description: "d", version: null }, out, env)).toBe(
      [
        "╭─────────────────────────╮",
        "│ Running flow: (unnamed) │",
        "│ Description: d          │",
        "╰─────────────────────────╯",
        "",
      ].join("\n"),
    );
    expect(flowBanner({ name: null, description: null, version: null }, out, env)).toBeNull();
    expect(flowBanner({ name: "", description: "", version: "" }, out, env)).toBeNull();
  });
});

describe("stopOnSignals", () => {
  const SIGNALS = ["SIGINT", "SIGTERM", "SIGHUP"] as const;
  const listeners = (target: EventEmitter) => SIGNALS.map((name) => target.listenerCount(name));

  test("each signal aborts with a HostStop naming it and removes all three listeners", () => {
    for (const name of SIGNALS) {
      const target = new EventEmitter();
      const controller = new AbortController();
      stopOnSignals(controller, target);
      expect(listeners(target)).toEqual([1, 1, 1]);
      target.emit(name);
      expect(controller.signal.reason).toBeInstanceOf(HostStop);
      expect(controller.signal.reason.signal).toBe(name);
      expect(listeners(target)).toEqual([0, 0, 0]);
    }
  });

  test("the disposer removes the listeners", () => {
    const target = new EventEmitter();
    const controller = new AbortController();
    const dispose = stopOnSignals(controller, target);
    dispose();
    expect(listeners(target)).toEqual([0, 0, 0]);
    expect(controller.signal.aborted).toBe(false);
  });

  test("an already aborted controller keeps its reason and still removes them", () => {
    const target = new EventEmitter();
    const controller = new AbortController();
    const first = new HostStop("SIGINT");
    controller.abort(first);
    stopOnSignals(controller, target);
    target.emit("SIGTERM");
    expect(controller.signal.reason).toBe(first);
    expect(listeners(target)).toEqual([0, 0, 0]);
  });
});

/** Return a `CliIo` over `env` whose streams are collected. */
function testIo(env: Record<string, string> = {}) {
  const out = sink();
  const err = sink();
  const io: CliIo = {
    stdout: out.out,
    stderr: err.out,
    stdin: new PassThrough(),
    env: (name) => (Object.hasOwn(env, name) ? env[name] : undefined),
  };
  return { io, stdout: out.text, stderr: err.text };
}

/** Write a skill directory at `tmp/skills/<name>` holding `skillMd`, and return its path. */
function skillDir(name: string, skillMd: string): string {
  const dir = join(tmp, "skills", name);
  mkdirSync(dir, { recursive: true });
  writeFileSync(join(dir, "SKILL.md"), skillMd);
  return dir;
}

const GOOD_SKILL = "---\nname: good\ndescription: Does good things.\n---\nDo good things.\n";

describe("skillsCheck", () => {
  test("text: ok lines on stdout, errors on stderr, exit 1 when one fails", async () => {
    const good = skillDir("good", GOOD_SKILL);
    const { io, stdout, stderr } = testIo();
    expect(await skillsCheck([good], { json: false }, io)).toBe(0);
    expect(stdout()).toBe(`${good}: ok (good)\n`);
    expect(stderr()).toBe("");

    const missing = join(tmp, "skills", "missing");
    const failed = testIo();
    expect(await skillsCheck([good, missing], { json: false }, failed.io)).toBe(1);
    expect(failed.stdout()).toBe(`${good}: ok (good)\n`);
    expect(failed.stderr()).toBe(`${missing}: missing SKILL.md\n`);
  });

  test("json: a missing and a bad SKILL.md are skill.invalid, each path as given", async () => {
    const good = skillDir("good", GOOD_SKILL);
    skillDir("bad", "---\nname: bad\n---\nNo description.\n");
    const bad = join(tmp, "skills", "..", "skills", "bad") + "/";
    const empty = join(tmp, "skills", "empty");
    mkdirSync(empty);
    const { io, stdout, stderr } = testIo();
    expect(await skillsCheck([good, bad, empty], { json: true }, io)).toBe(1);
    expect(stderr()).toBe("");
    const document = JSON.parse(stdout());
    expect(document.format).toBe("yian.cli.v1");
    expect(document.skills[0]).toEqual({ path: good, ok: true, name: "good", description: "Does good things." });
    expect(document.skills.map((s: { path: string }) => s.path)).toEqual([good, bad, empty]);
    for (const result of document.skills.slice(1)) {
      expect(result.ok).toBe(false);
      expect(result.error.code).toBe("skill.invalid");
      expect(typeof result.error.message).toBe("string");
    }
  });
});

describe("main", () => {
  const BAD_DEFAULTS = { YIAN_MAX_TOTAL_NODES: "abc" };

  test("a bad YIAN_ default: the error on stderr, a document only under --json, exit 1", async () => {
    const flow = join(tmp, "flow.yaml");
    writeFileSync(flow, "id: f\n");
    const text = testIo(BAD_DEFAULTS);
    expect(await main(["check", flow], text.io)).toBe(1);
    expect(text.stdout()).toBe("");
    expect(text.stderr()).toContain("YIAN_MAX_TOTAL_NODES");

    const json = testIo(BAD_DEFAULTS);
    expect(await main(["check", "--json", flow], json.io)).toBe(1);
    expect(json.stderr()).toBe(text.stderr());
    expect(JSON.parse(json.stdout())).toEqual({
      format: "yian.cli.v1",
      error: { code: "defaults.invalid", message: text.stderr().trimEnd() },
    });
  });

  test("an unknown command or option exits 2 with nothing on stdout", async () => {
    for (const [argv, message] of [
      [["nope"], "No such command 'nope'."],
      [["--nope"], "No such option: --nope"],
      [["skills", "nope"], "No such command 'nope'."],
      [["skills", "check", "--nope", "x"], "No such option: --nope"],
    ] as const) {
      const { io, stdout, stderr } = testIo();
      expect(await main(argv, io)).toBe(2);
      expect(stdout()).toBe("");
      expect(stderr()).toContain(`Error: ${message}\n`);
    }
  });

  test("an unknown command is a usage error before the defaults are checked", async () => {
    const { io, stderr } = testIo(BAD_DEFAULTS);
    expect(await main(["nope"], io)).toBe(2);
    expect(stderr()).toContain("No such command 'nope'.");
  });

  test("cli rejects negative workers", async () => {
    const flow = join(tmp, "flow.yaml");
    writeFileSync(flow, "id: f\n");
    const { io, stdout, stderr } = testIo();
    expect(await main(["run", "-w", "-1", flow], io)).toBe(2);
    expect(stdout()).toBe("");
    expect(stderr()).toStartWith("Usage: yian run [OPTIONS] FLOW\nTry 'yian run --help' for help.\n");
  });

  test("a missing flow, resume log or --root directory exits 2", async () => {
    const flow = join(tmp, "flow.yaml");
    writeFileSync(flow, "id: f\n");
    const missing = join(tmp, "missing.yaml");
    const log = join(tmp, "missing.jsonl");
    const nodir = join(tmp, "nodir");
    for (const [argv, message] of [
      [["run", missing], `Invalid value for 'FLOW': File '${missing}' does not exist.`],
      [["run", flow, "--resume", log], `Invalid value for '--resume': File '${log}' does not exist.`],
      [["check", "--root", nodir, "x.yaml"], `Invalid value for '--root': Directory '${nodir}' does not exist.`],
    ] as const) {
      const { io, stdout, stderr } = testIo();
      expect(await main(argv, io)).toBe(2);
      expect(stdout()).toBe("");
      expect(stderr()).toEndWith(`Error: ${message}\n`);
    }
  });

  test("an unported command exits 1 with its line", async () => {
    for (const [argv, name] of [
      [["providers"], "providers"],
      [["config", "check"], "config check"],
      [["auth", "openai"], "auth"],
    ] as const) {
      const { io, stdout, stderr } = testIo();
      expect(await main(argv, io)).toBe(1);
      expect(stdout()).toBe("");
      expect(stderr()).toBe(`yian ${name} is not ported to TypeScript yet\n`);
    }
  });

  test("a command's --help prints its help and exits 0", async () => {
    const { io, stdout } = testIo();
    expect(await main(["skills", "check", "--help"], io)).toBe(0);
    expect(stdout()).toStartWith("Usage: yian skills check [OPTIONS] DIRS...\n");
  });
});

// The entry as the conformance runner runs it: `.ts` from source, `.js` when built.
const CLI_ENTRY = fileURLToPath(
  new URL(`../../src/cli/index.${import.meta.url.endsWith(".ts") ? "ts" : "js"}`, import.meta.url),
);

// Guards truncation at exit through the real entry, not `flush` itself, which progress writes need.
test("the entry writes a large --json document whole to piped stdout", () => {
  const dirs = Array.from({ length: 3000 }, (_, i) => `missing-skill-directory-${String(i).padStart(4, "0")}`);
  const runtime = process.versions.bun !== undefined ? ["--no-install"] : [];
  const ran = spawnSync(process.execPath, [...runtime, CLI_ENTRY, "skills", "check", "--json", ...dirs], {
    cwd: tmp,
    env: { HOME: tmp },
    encoding: "utf8",
    maxBuffer: 64 * 1024 * 1024,
    timeout: 30_000,
  });
  expect(ran.status).toBe(1);
  expect(ran.stdout.length).toBeGreaterThan(128 * 1024);
  const document = JSON.parse(ran.stdout);
  expect(document.skills.map((s: { path: string }) => s.path)).toEqual(dirs);
}, 30_000);
