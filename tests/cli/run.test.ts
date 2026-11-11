// `yian run` and the rest of `main`'s command table: input parsing and the input gate, non-TTY
// and prompted runs, `--json` documents, setup errors, stops and warnings, `check`, `status`, and
// the run-log checks that run before the flow is read.

import { afterEach, beforeEach, describe, expect, mock, spyOn, test } from "bun:test";
import { EventEmitter, getEventListeners } from "node:events";
import { execFileSync, spawn, spawnSync } from "node:child_process";
import {
  cpSync,
  existsSync,
  mkdirSync,
  mkdtempSync,
  readdirSync,
  readFileSync,
  realpathSync,
  rmSync,
  symlinkSync,
  writeFileSync,
} from "node:fs";
import os, { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { PassThrough } from "node:stream";
import { fileURLToPath } from "node:url";

import { diagnosticJson } from "../../src/cli/render.ts";
import {
  flowBanner,
  jsonValue,
  mergeInputs,
  parseKv,
  readInputsJson,
  skillsCheck,
} from "../../src/cli/run.ts";
import { defaultRunsDir } from "../../src/cli/startup.ts";
import { childCommand, main } from "../../src/cli/index.ts";
import { LICENSES } from "../../src/cli/licenses.ts";
import { userMcpConfig } from "../../src/cli/mcp_config.ts";
import { defaultSkillRoots } from "../../src/compose/skills.ts";
import { AuthStore } from "../../src/llm_clients/auth/index.ts";
import {
  abortOnSignal,
  type CliIo,
  HostStop,
  printJson,
  stopOnSignals,
  UsageError,
  type Writer,
} from "../../src/cli/utils.ts";
import { Diagnostic, DiagnosticFrame, SourceRange } from "../../src/diagnostics.ts";
import { FunctionTool } from "../../src/tools/function_tools.ts";
import { registerTool, TOOL_REGISTRY } from "../../src/tools/registry.ts";
import { inspectRun, readRunIndex } from "../../src/run.ts";
import { JsonlRunLog, NodeStartedRecord, NodeStepRecord, readRunLog, RunStartedRecord } from "../../src/runtime/log.ts";
import { FileRef } from "../../src/typesys/values.ts";
import { compiled, VERSION } from "../../src/platform.ts";
import { NO_ENV } from "../helpers/run_state_fixtures.ts";
import { FakeInput, FakeOutput } from "./terminal_both_runtimes.ts";
import { parseJson } from "../../src/typesys/json.ts";
import { type Schema, validate } from "../helpers/schema.ts";

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
    // Line numbers count `\r\n` and a lone `\r` as one break each.
    expect(() => readInputsJson(inputsFile('{\r\n"a": 1,\r"b" 2}'))).toThrow(
      new UsageError("Invalid value for --inputs: invalid JSON at line 3, column 5: Expecting ':' delimiter"),
    );
    // A leading BOM is kept, so the text is not valid JSON.
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

  // The expected banner boxes at each width.
  const banner = (
    columns: number | undefined,
    flow: { name: string | null; description: string | null; version?: string | null },
    tty = false,
    env: (name: string) => string | undefined = () => undefined,
  ) => {
    const out: Writer = { write: async () => {}, isTTY: tty, columns };
    return flowBanner({ version: null, ...flow }, out, env);
  };
  const lines = (...rows: string[]) => [...rows, ""].join("\n");
  const longDescription =
    "Draft a plan, ask a human to approve or revise it, then finalize accordingly, and then some more words.";
  const approval = "Draft a plan, ask a human to approve or revise it, then finalize accordingly.";
  const sixty = lines(
    "╭──────────────────────────────────────────────────────────╮",
    "│ Running flow: hi                                         │",
    "│ Description: Draft a plan, ask a human to approve or     │",
    "│ revise it, then finalize accordingly, and then some more │",
    "│ words.                                                   │",
    "╰──────────────────────────────────────────────────────────╯",
  );

  test("wraps the text inside a box no wider than the terminal", () => {
    expect(banner(60, { name: "hi", description: longDescription })).toBe(sixty);
    expect(banner(40, { name: "human_approval", description: approval, version: "1.0" })).toBe(
      lines(
        "╭──────────────────────────────────────╮",
        "│ Running flow: human_approval         │",
        "│ (version: 1.0)                       │",
        "│ Description: Draft a plan, ask a     │",
        "│ human to approve or revise it, then  │",
        "│ finalize accordingly.                │",
        "╰──────────────────────────────────────╯",
      ),
    );
  });

  test("wraps wide characters by their cells", () => {
    expect(banner(30, { name: "cjk", description: "中文描述这是一个很长的流程说明文字需要换行显示" })).toBe(
      lines(
        "╭────────────────────────────╮",
        "│ Running flow: cjk          │",
        "│ Description:               │",
        "│ 中文描述这是一个很长的流程 │",
        "│ 说明文字需要换行显示       │",
        "╰────────────────────────────╯",
      ),
    );
  });

  test("folds a word wider than the box", () => {
    expect(banner(20, { name: "x", description: "averyveryverylongwordthatcannotbreak end" })).toBe(
      lines(
        "╭──────────────────╮",
        "│ Running flow: x  │",
        "│ Description:     │",
        "│ averyveryverylon │",
        "│ gwordthatcannotb │",
        "│ reak end         │",
        "╰──────────────────╯",
      ),
    );
  });

  test("is 80 columns wide off a terminal", () => {
    expect(banner(undefined, { name: "human_approval", description: approval })).toBe(
      lines(
        "╭──────────────────────────────────────────────────────────────────────────────╮",
        "│ Running flow: human_approval                                                 │",
        "│ Description: Draft a plan, ask a human to approve or revise it, then         │",
        "│ finalize accordingly.                                                        │",
        "╰──────────────────────────────────────────────────────────────────────────────╯",
      ),
    );
  });

  test("keeps the description's lines, blank ones too, and wraps each", () => {
    expect(banner(40, { name: "multi", description: "line one\n\nline three is long enough to wrap around here ok" })).toBe(
      lines(
        "╭──────────────────────────────────────╮",
        "│ Running flow: multi                  │",
        "│ Description: line one                │",
        "│                                      │",
        "│ line three is long enough to wrap    │",
        "│ around here ok                       │",
        "╰──────────────────────────────────────╯",
      ),
    );
  });

  test("expands tabs to 8-cell stops, sizing the box with a tab as no cells as rich does", () => {
    expect(banner(40, { name: "tab", description: "tab\there\tand\tthere and more words to wrap" })).toBe(
      lines(
        "╭──────────────────────────────────────╮",
        "│ Running flow: tab                    │",
        "│ Description: tab        here    and  │",
        "│ there and more words to wrap         │",
        "╰──────────────────────────────────────╯",
      ),
    );
    expect(banner(80, { name: "t", description: "a\tb" })).toBe(
      lines("╭─────────────────╮", "│ Running flow: t │", "│ Description: a  │", "│ b               │", "╰─────────────────╯"),
    );
  });

  test("paints the wrapped rows on a colour terminal", () => {
    const text = banner(60, { name: "hi", description: longDescription }, true)!;
    expect(text.replace(/\x1b\[[0-9;]*m/g, "")).toBe(sixty);
    expect(text).toContain("\x1b[1mRunning flow: \x1b[0m");
    expect(text).toContain("\x1b[1mDescription: \x1b[0m");
    expect(text).toContain("\x1b[36m╭");
  });
});

describe("the home fallback", () => {
  afterEach(() => {
    mock.restore();
  });

  const env = () => undefined;
  /** Each `~` path of the CLI with `HOME` unset: runs, MCP config, a skill root, logins. */
  const homePaths = [
    () => defaultRunsDir(env),
    () => userMcpConfig(env),
    () => defaultSkillRoots({ cwd: tmp, env })[0],
    () => new AuthStore(null, { env }).path,
  ];

  test("every ~ path falls back to the account's home when HOME is unset", () => {
    spyOn(os, "userInfo").mockReturnValue({ ...os.userInfo(), homedir: "/account" });
    expect(homePaths.map((path) => path())).toEqual([
      join("/account", ".yian", "runs"),
      join("/account", ".yian", "mcp.json"),
      join("/account", ".claude", "skills"),
      join("/account", ".config", "yian", "auth.json"),
    ]);
  });

  test("every ~ path throws the same error when the account has no home either", () => {
    const failure = Object.assign(new Error("no passwd entry"), { code: "ENOENT", syscall: "uv_os_get_passwd" });
    spyOn(os, "userInfo").mockImplementation(() => {
      throw failure;
    });
    for (const path of homePaths) expect(path).toThrow(failure);
  });
});

describe("abortOnSignal", () => {
  test("a forwarded stop then a SIGINT: one stop; only the signal removes listeners", () => {
    const target = new EventEmitter();
    const stop = abortOnSignal("SIGUSR2", new HostStop("SIGINT"), target);
    // As `run` links `io.stop` to its controller beside the signal listeners.
    const controller = new AbortController();
    stopOnSignals(controller, target);
    stop.addEventListener("abort", () => controller.abort(stop.reason), { once: true });
    const names = ["SIGINT", "SIGTERM", "SIGHUP", "SIGUSR2"];
    const counts = () => names.map((name) => target.listenerCount(name));
    target.emit("SIGUSR2");
    const first = controller.signal.reason;
    expect(first).toBeInstanceOf(HostStop);
    expect(first.signal).toBe("SIGINT");
    expect(counts()).toEqual([1, 1, 1, 1]);
    target.emit("SIGINT");
    target.emit("SIGUSR2");
    expect(controller.signal.reason).toBe(first);
    expect(stop.reason).toBe(first);
    expect(counts()).toEqual([0, 0, 0, 1]);
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
    stderrStream: new PassThrough(),
    env: (name) => (Object.hasOwn(env, name) ? env[name] : NO_ENV(name)),
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

  test("the command runs under io.env's YIAN_ defaults: the key prompt's cap", async () => {
    const flow = file(
      "flow.yaml",
      "id: k\nname: k\nnodes:\n  agent: {kind: agent, prompt: hi}\noutput: ${agent.output}\n" +
        "llm_config: {provider: anthropic, model: claude-sonnet-4-5}\n",
    );
    mkdirSync(join(tmp, "home"), { recursive: true });
    const { io, stderr } = testIo({ HOME: join(tmp, "home"), YIAN_MAX_KEY_PROMPTS: "1" });
    const input = new FakeInput();
    // Empty answers for the built-in cap of 3; io.env's cap of 1 takes only the first.
    input.write("\r\r\r");
    const argv = ["run", flow, "-q", "--log", join(tmp, "run.jsonl")];
    expect(await main(argv, { ...io, stdin: input, stderrStream: new FakeOutput() })).toBe(1);
    expect(stderr()).toContain("no API key provided for 'anthropic' ($ANTHROPIC_API_KEY) after 1 attempts");
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
    expect(stderr()).toStartWith("Usage: yian run [OPTIONS] [FLOW]\nTry 'yian run --help' for help.\n");
  });

  test("a missing flow, resume log or --root directory exits 2", async () => {
    const flow = join(tmp, "flow.yaml");
    writeFileSync(flow, "id: f\n");
    const missing = join(tmp, "missing.yaml");
    const log = join(tmp, "missing.jsonl");
    const nodir = join(tmp, "nodir");
    for (const [argv, message] of [
      [["run", missing], `Invalid value for 'FLOW': File '${missing}' does not exist.`],
      [["run", "--resume", log], `Invalid value for '--resume': File '${log}' does not exist.`],
      [["check", "--root", nodir, "x.yaml"], `Invalid value for '--root': Directory '${nodir}' does not exist.`],
    ] as const) {
      const { io, stdout, stderr } = testIo();
      expect(await main(argv, io)).toBe(2);
      expect(stdout()).toBe("");
      expect(stderr()).toEndWith(`Error: ${message}\n`);
    }
  });

  test("the provider commands reach cli/auth.ts", async () => {
    let { io, stdout, stderr } = testIo();
    expect(await main(["providers"], io)).toBe(0);
    expect(stdout().split("\n")[0]).toBe("openai: api-key, subscription (OPENAI_API_KEY)");
    expect(stderr()).toBe("");
    ({ io, stdout, stderr } = testIo());
    expect(await main(["auth", "--json"], io)).toBe(2);
    expect(stdout()).toBe("");
    expect(stderr()).toContain(
      "Error: Invalid value: --json applies to --remove and to openai --method subscription --login device\n",
    );
    // `--timeout` is a float option: Click's message, before the handler runs.
    ({ io, stdout, stderr } = testIo());
    expect(await main(["models", "--provider", "openai", "--timeout", "soon"], io)).toBe(2);
    expect(stderr()).toContain("Error: Invalid value for '--timeout': 'soon' is not a valid float.\n");
  });

  test("--licenses prints LICENSES and exits 0; the root help lists it", async () => {
    let { io, stdout, stderr } = testIo();
    expect(await main(["--licenses"], io)).toBe(0);
    expect(stdout()).toBe(LICENSES);
    expect(stderr()).toBe("");
    ({ io, stdout } = testIo());
    expect(await main(["--help"], io)).toBe(0);
    expect(stdout()).toMatch(/^ {2}--licenses +\S/m);
  });

  test("--version prints VERSION and exits 0, before the defaults check; the root help lists it", async () => {
    let { io, stdout, stderr } = testIo();
    expect(await main(["--version"], io)).toBe(0);
    expect(stdout()).toBe(`${VERSION}\n`);
    expect(stderr()).toBe("");
    ({ io, stdout, stderr } = testIo(BAD_DEFAULTS));
    expect(await main(["--version"], io)).toBe(0);
    expect(stdout()).toBe(`${VERSION}\n`);
    expect(stderr()).toBe("");
    ({ io, stdout } = testIo());
    expect(await main(["--help"], io)).toBe(0);
    expect(stdout()).toMatch(/^ {2}--version +Show the version and exit\.$/m);
    ({ io, stderr } = testIo());
    expect(await main(["status", "--version"], io)).toBe(2);
    expect(stderr()).toContain("No such option: --version");
  });

  test("a command's --help prints its help and exits 0", async () => {
    const { io, stdout } = testIo();
    expect(await main(["skills", "check", "--help"], io)).toBe(0);
    expect(stdout()).toStartWith("Usage: yian skills check [OPTIONS] DIRS...\n");
  });

  test("run's old memory flags are gone: --memory-root is an unknown option and the help lists only --memory", async () => {
    const flow = join(tmp, "flow.yaml");
    writeFileSync(flow, "id: f\n");
    for (const flag of ["--memory-root", "--memory-storage"]) {
      const { io, stdout, stderr } = testIo();
      expect(await main(["run", flow, flag, tmp], io)).toBe(2);
      expect(stdout()).toBe("");
      expect(stderr()).toContain(`No such option: ${flag}`);
    }
    const { io, stdout } = testIo();
    expect(await main(["run", "--help"], io)).toBe(0);
    expect(stdout()).toContain("--memory NAME=PATH");
    expect(stdout()).not.toContain("--memory-");
  });
});

// Bytes that aren't UTF-8: a flow holding them can't be read.
const NOT_UTF8 = new Uint8Array([0x69, 0x64, 0x3a, 0x20, 0xff, 0x0a]);

const FLOW = "id: f\nname: f\ninput:\n  n: !int\nnodes: {}\noutput: ${input.n}\n";

/** Write `content` at `tmp/<name>`, creating its directory, and return its path. */
function file(name: string, content: string | Uint8Array): string {
  const path = join(tmp, name);
  mkdirSync(join(path, ".."), { recursive: true });
  writeFileSync(path, content);
  return path;
}

describe("run", () => {
  test("--resume with --input exits 2 naming the conflict, before the log is read", async () => {
    const old = file("old.jsonl", "");
    const { io, stdout, stderr } = testIo();
    expect(await main(["run", "--resume", old, "--input", "n=1"], io)).toBe(2);
    expect(stdout()).toBe("");
    expect(stderr()).toEndWith("Error: Invalid value: --resume cannot be combined with --input\n");
  });

  test("--resume with a flow file exits 2: the resume runs the recorded flow", async () => {
    const flow = file("flow.yaml", NOT_UTF8);
    const old = file("old.jsonl", "");
    const { io, stdout, stderr } = testIo();
    expect(await main(["run", flow, "--resume", old], io)).toBe(2);
    expect(stdout()).toBe("");
    expect(stderr()).toEndWith("Error: Invalid value: --resume runs the flow recorded in the log; leave out <flow>\n");
  });

  test("a fresh run without a flow file exits 2", async () => {
    const { io, stderr } = testIo();
    expect(await main(["run"], io)).toBe(2);
    expect(stderr()).toContain("Missing argument 'FLOW'.");
  });

  test("an existing --log exits 2, before the flow is read", async () => {
    const flow = file("flow.yaml", NOT_UTF8);
    const log = file("taken.jsonl", "keep");
    const { io, stdout, stderr } = testIo();
    expect(await main(["run", flow, "--log", log], io)).toBe(2);
    expect(stdout()).toBe("");
    expect(stderr()).toEndWith(`Error: Invalid value for --log: --log path already exists: ${log}\n`);
  });

  test("a flow that isn't UTF-8 exits 1, with a compose.load document under --json", async () => {
    const flow = file("flow.yaml", NOT_UTF8);
    const message = `can't read ${flow}: not UTF-8 text`;
    const text = testIo();
    expect(await main(["run", flow], text.io)).toBe(1);
    expect(text.stdout()).toBe("");
    expect(text.stderr()).toBe(`${message}\n`);

    const json = testIo();
    expect(await main(["run", flow, "--json"], json.io)).toBe(1);
    expect(json.stderr()).toBe(`${message}\n`);
    expect(JSON.parse(json.stdout())).toEqual({
      format: "yian.cli.v1",
      error: { code: "compose.load", message },
    });
  });

});

/** A flow with one `topic` input whose CODE node `body` makes the output's `topic`. */
function topicFlow(body: string): string {
  return (
    "id: cli_echo\nname: cli_echo\ninput:\n  topic: !str\nnodes:\n  echo:\n    kind: code\n" +
    `    input:\n      topic: \${input.topic}\n    output: !str\n    code: ${body}\n` +
    "output:\n  topic: ${echo.output}\n"
  );
}

const ECHO_FLOW = topicFlow("return inputs.topic");
const INLINE_FLOW = topicFlow("|\n      return inputs.topic.toUpperCase();");
const FAIL_FLOW = topicFlow('throw new Error("always fails")');

const OMITTABLE_INPUTS_FLOW = `id: cli_omittable_inputs
name: cli_omittable_inputs
input:
  topic: !str
  limit: !int 7
  labels: !list(str) [alpha, beta]
  note: !Optional(str)
nodes:
  show:
    kind: code
    input:
      topic: \${input.topic}
      limit: \${input.limit}
      labels: \${input.labels}
      note: \${input.note}
    output: !object
    code: return inputs
output: \${show.output}
`;

const NULLABLE_STRUCTURED_INPUT_FLOW = `id: cli_nullable_structured_input
name: cli_nullable_structured_input
input:
  items: !Optional(list(int))
nodes:
  inspect:
    kind: code
    input:
      items: \${input.items}
    output: !object
    code: 'return { items: inputs.items, is_none: inputs.items === null }'
output: \${inspect.output}
`;

const REGISTERED_TOOL_FLOW =
  "id: cli_registered_tool\nname: cli_registered_tool\nnodes:\n  call:\n    kind: tool\n" +
  "    tool_id: cli_run_probe\noutput: ${call.output}\n";

/** Run `yian run flow.yaml argv` on a flow holding `body`, its default log under `tmp/home`. */
async function runFlowCli(body: string, argv: readonly string[]) {
  const flow = file("flow.yaml", body);
  mkdirSync(join(tmp, "home"), { recursive: true });
  const { io, stdout, stderr } = testIo({ HOME: join(tmp, "home") });
  const code = await main(["run", flow, ...argv], io);
  return { code, stdout: stdout(), stderr: stderr() };
}

describe("run outcomes", () => {
  test("run succeeds with input flag", async () => {
    const { code, stdout, stderr } = await runFlowCli(ECHO_FLOW, ["--input", "topic=clouds"]);
    expect(code).toBe(0);
    expect(stdout).toBe('{"topic":"clouds"}\n');
    expect(stderr).toContain("✓ echo");
  });

  test("a string output prints as is", async () => {
    const flow = ECHO_FLOW.replace("output:\n  topic: ${echo.output}", "output: ${echo.output}");
    const { code, stdout } = await runFlowCli(flow, ["-q", "--input", "topic=a *b*"]);
    expect(code).toBe(0);
    expect(stdout).toBe("a *b*\n");
  });

  test("run keeps registered tools", async () => {
    const probe = new FunctionTool({
      name: "cli_run_probe",
      description: "Probe.",
      parametersJsonSchema: { type: "object", properties: {} },
      execute: () => "from the registry",
    });
    registerTool("cli_run_probe", probe);
    try {
      const { code, stdout } = await runFlowCli(REGISTERED_TOOL_FLOW, ["-q"]);
      expect(code).toBe(0);
      expect(stdout).toBe("from the registry\n");
    } finally {
      TOOL_REGISTRY.delete("cli_run_probe");
    }
  });

  test("run inline code in process", async () => {
    const { code, stdout } = await runFlowCli(INLINE_FLOW, ["--input", "topic=clouds"]);
    expect(code).toBe(0);
    expect(stdout).toContain("CLOUDS");
  });

  test("run reads inputs json, a flag overriding it", async () => {
    const inputs = inputsFile('{"topic": "rivers"}');
    let { code, stdout } = await runFlowCli(ECHO_FLOW, ["--inputs", inputs]);
    expect(code).toBe(0);
    expect(stdout).toContain("rivers");
    ({ code, stdout } = await runFlowCli(ECHO_FLOW, ["--inputs", inputs, "--input", "topic=mountains"]));
    expect(code).toBe(0);
    expect(stdout).toContain("mountains");
    expect(stdout).not.toContain("rivers");
  });

  test("failed run exits nonzero, the located failure on stderr", async () => {
    const { code, stdout, stderr } = await runFlowCli(FAIL_FLOW, ["-q", "--input", "topic=clouds"]);
    expect(code).toBe(1);
    expect(stdout).toBe("");
    expect(stderr).toContain("flow.yaml:");
    expect(stderr).toContain("always fails");
    expect(stderr).not.toContain("─ traceback ");
  });

  test("failed run under --engine-trace adds the captured traceback", async () => {
    const { code, stderr } = await runFlowCli(FAIL_FLOW, ["-q", "--engine-trace", "--input", "topic=clouds"]);
    expect(code).toBe(1);
    expect(stderr).toContain("always fails");
    expect(stderr).toContain("─ traceback ");
  });

  test("non-TTY run omits optional and defaulted inputs", async () => {
    const log = join(tmp, "run.jsonl");
    const { code, stdout, stderr } = await runFlowCli(OMITTABLE_INPUTS_FLOW, [
      "--input",
      "topic=clouds",
      "--log",
      log,
      "--quiet",
    ]);
    expect(code).toBe(0);
    expect(existsSync(log)).toBe(true);
    expect(JSON.parse(stdout)).toEqual({ topic: "clouds", limit: 7, labels: ["alpha", "beta"], note: null });
    // Quiet: only the run log's path is on stderr.
    expect(stderr).toBe(`Run log: ${log}\n`);
  });

  test("non-TTY inputs json preserves explicit structured null", async () => {
    const inputs = inputsFile('{"items": null}');
    const log = join(tmp, "run.jsonl");
    const { code, stdout } = await runFlowCli(NULLABLE_STRUCTURED_INPUT_FLOW, [
      "--inputs",
      inputs,
      "--log",
      log,
      "--quiet",
    ]);
    expect(code).toBe(0);
    expect(JSON.parse(stdout)).toEqual({ items: null, is_none: true });
  });

  test("the banner precedes progress, and --quiet drops both", async () => {
    const flow = ECHO_FLOW.replace("name: cli_echo\n", "name: cli_echo\ndescription: Echoes.\n");
    const loud = await runFlowCli(flow, ["--input", "topic=x"]);
    expect(loud.stderr).toStartWith("╭");
    expect(loud.stderr).toContain("Running flow: cli_echo");
    expect(loud.stderr.indexOf("Running flow")).toBeLessThan(loud.stderr.indexOf("✓ echo"));
    const quiet = await runFlowCli(flow, ["-q", "--input", "topic=x"]);
    expect(quiet.stderr).toStartWith("Run log: ");
    expect(quiet.stderr).not.toContain("Running flow");
    expect(quiet.stderr).not.toContain("✓");
  });
});

describe("warnings during a run", () => {
  // `warnOnce` shows each text once per process, so each run names its own unknown model.
  for (const [label, argv] of [["through progress", []], ["under --quiet", ["-q"]]] as const) {
    test(`print as warning: lines, ${label}`, async () => {
      const model = `no-such-model-${Date.now()}-${argv.length}`;
      const flow = file(
        "flow.yaml",
        `id: w\nname: w\nnodes:\n  agent: {kind: agent, prompt: hi}\noutput: \${agent.output}\n` +
          `llm_config: {provider: anthropic, model: ${model}}\n`,
      );
      mkdirSync(join(tmp, "home"), { recursive: true });
      const plain = testIo({ HOME: join(tmp, "home"), ANTHROPIC_API_KEY: "sk-ant-test" });
      const stderr = plain.stderr;
      // Through progress, a TTY stderr (stdin stays non-TTY, so nothing prompts) makes the
      // reporter paint the line yellow, which `main`'s plain handler never does.
      const io = argv.length === 0 ? { ...plain.io, stderr: { ...plain.io.stderr, isTTY: true } } : plain.io;
      const send = globalThis.fetch;
      globalThis.fetch = (async () => Response.json({ error: { message: "nope" } }, { status: 401 })) as never;
      const emit = spyOn(process, "emitWarning");
      try {
        expect(await main(["run", flow, ...argv], io)).toBe(1);
        expect(emit).not.toHaveBeenCalled();
      } finally {
        emit.mockRestore();
        globalThis.fetch = send;
      }
      const line = `warning: Model '${model}' is not in the known model list for provider 'anthropic'. Continuing anyway.`;
      if (argv.length === 0) expect(stderr()).toContain(`\x1b[33m${line}\x1b[0m`);
      else expect(stderr().split("\n")).toContain(line);
      expect(stderr()).not.toContain("(node:");
      expect(stderr()).not.toContain("--trace-warnings");
    });
  }
});

describe("a provider's text with escape sequences", () => {
  for (const isTTY of [true, false]) {
    test(`a warning and the provider's error body are ${isTTY ? "stripped on a TTY" : "kept exactly when piped"}`, async () => {
      const flow = file(
        "flow.yaml",
        `id: w\nname: w\nnodes:\n  agent: {kind: agent, prompt: hi}\noutput: \${agent.output}\n` +
          `llm_config: {provider: anthropic, model: "m-${isTTY}-${Date.now()}\\e]52;c;aGk=\\a"}\n`,
      );
      mkdirSync(join(tmp, "home"), { recursive: true });
      const plain = testIo({ HOME: join(tmp, "home"), ANTHROPIC_API_KEY: "sk-ant-test", NO_COLOR: "1" });
      const io = { ...plain.io, stderr: { ...plain.io.stderr, isTTY } };
      const send = globalThis.fetch;
      const body = "bad\x1b]8;;https://e.example\x07 key\x1b[2J";
      globalThis.fetch = (async () => Response.json({ error: { message: body } }, { status: 401 })) as never;
      try {
        expect(await main(["run", flow, "-q"], io)).toBe(1);
      } finally {
        globalThis.fetch = send;
      }
      const stderr = plain.stderr();
      if (isTTY) {
        expect(stderr).toMatch(/warning: Model 'm-true-\d+' is not in the known model list/);
        expect(stderr).toContain("bad key");
        expect(stderr).not.toContain("\x1b");
        expect(stderr).not.toContain("\x07");
      } else {
        expect(stderr).toMatch(/warning: Model 'm-false-\d+\x1b\]52;c;aGk=\x07' is not in the known model list/);
        expect(stderr).toContain(body);
      }
    });
  }
});

const REQUIRED_INPUTS_FLOW = `id: cli_required_inputs
name: cli_required_inputs
input:
  first: !str
  optional: !Optional(str)
  second: !int
nodes:
  show:
    kind: code
    input:
      first: \${input.first}
      optional: \${input.optional}
      second: \${input.second}
    output: !object
    code: return inputs
output: \${show.output}
`;

/** A flow declaring `inputs` (YAML lines under `input:`) whose output is `null`. */
function inputsFlow(inputs: string): string {
  return `id: gate\nname: gate\ninput:\n${inputs}nodes: {}\noutput: null\n`;
}

/** The input names the run at `log` was given, from its `RunStarted` record. */
function suppliedNames(log: string): string[] {
  const record = readRunLog(log)[0]!.record;
  if (!(record instanceof RunStartedRecord)) throw new Error("the log doesn't start with RunStarted");
  return Object.keys(record.startWiringOverride);
}

const INPUT_HINT = "Invalid value for '--input' / '--inputs'";

describe("run's input gate", () => {
  test("undeclared input flag is a bad parameter", async () => {
    const argv = ["--input", "topic=clouds", "--input", "topik=x"];
    const { code, stdout, stderr } = await runFlowCli(ECHO_FLOW, argv);
    expect(code).toBe(2);
    expect(stdout).toBe("");
    expect(stderr).toEndWith(`Error: ${INPUT_HINT}: unknown input "topik"; declared inputs: topic\n`);
  });

  test("undeclared inputs json key is a bad parameter", async () => {
    const inputs = inputsFile('{"topic": "clouds", "topik": "x"}');
    const { code, stderr } = await runFlowCli(ECHO_FLOW, ["--inputs", inputs]);
    expect(code).toBe(2);
    expect(stderr).toContain('unknown input "topik"');
  });

  test("an unknown name fails before a missing required input is reported", async () => {
    const { code, stderr } = await runFlowCli(REQUIRED_INPUTS_FLOW, ["-q", "--input", "nope=1"]);
    expect(code).toBe(2);
    expect(stderr).toContain('unknown input "nope"');
    expect(stderr).not.toContain("missing required");
  });

  test("gather non-TTY leaves optional and defaulted inputs absent", async () => {
    const log = join(tmp, "run.jsonl");
    const flow = inputsFlow("  optional: !Optional(str)\n  defaulted: !int 7\n");
    const { code } = await runFlowCli(flow, ["-q", "--log", log]);
    expect(code).toBe(0);
    expect(suppliedNames(log)).toEqual([]);
  });

  test("gather non-TTY reports all missing required inputs in order", async () => {
    const { code, stdout, stderr } = await runFlowCli(REQUIRED_INPUTS_FLOW, ["-q"]);
    expect(code).toBe(2);
    expect(stdout).toBe("");
    expect(stderr).toEndWith(`Error: ${INPUT_HINT}: missing required flow inputs: first, second\n`);
  });

  test("gather non-TTY filters required inputs by exact key membership", async () => {
    const declared = "  none: !Optional(str)\n  flag: !bool\n  zero: !int\n  blank: !str\n  missing: !int\n";
    const flow = inputsFlow(declared);
    const inputs = inputsFile('{"none": null, "flag": false, "zero": 0, "blank": ""}');
    const { code, stderr } = await runFlowCli(flow, ["-q", "--inputs", inputs]);
    expect(code).toBe(2);
    expect(stderr).toEndWith(`Error: ${INPUT_HINT}: missing required flow input: missing\n`);
  });

  test("non-TTY run reports all missing required inputs before setup", async () => {
    const { code, stderr } = await runFlowCli(REQUIRED_INPUTS_FLOW, ["-q"]);
    expect(code).toBe(2);
    const error = stderr.slice(stderr.indexOf("Error: "));
    expect(error.indexOf("first")).toBeLessThan(error.indexOf("second"));
    expect(error).not.toContain("optional");
    expect(stderr).not.toContain("Run log:");
    // The default log path was never made.
    expect(existsSync(join(tmp, "home", ".yian"))).toBe(false);
  });
});

/** Run `yian run flow.yaml -q argv` on a log at `tmp/run.jsonl`, stdin a fake TTY given `keys`. */
async function runOnTty(body: string, argv: readonly string[], keys: (input: FakeInput) => void) {
  const flow = file("flow.yaml", body);
  const { io, stdout, stderr } = testIo();
  const input = new FakeInput();
  const output = new FakeOutput();
  // Typed ahead: the terminal reads them once it prompts.
  keys(input);
  const code = await main(["run", flow, "-q", "--log", join(tmp, "run.jsonl"), ...argv], {
    ...io,
    stdin: input,
    stderrStream: output,
  });
  return { code, stdout: stdout(), stderr: stderr(), drawn: output.text, input };
}

const PROMPT_DEFAULTS_FLOW = `id: prompt-defaults
name: prompt-defaults
typedefs:
  Mode: Literal(z, a)
input:
  items: !list(str) [a, b]
  mode: !Mode z
  flag: !Optional(bool)
nodes:
  show:
    kind: code
    code: return inputs
    input:
      items: \${input.items}
      mode: \${input.mode}
      flag: \${input.flag}
    output: !object
output: \${show.output}
`;

describe("run's input prompts on a TTY", () => {
  test("accepting each omission leaves START's defaults", async () => {
    const enter = (keys: FakeInput) => keys.write("\r\r\r");
    const { code, stdout, drawn, input } = await runOnTty(PROMPT_DEFAULTS_FLOW, [], enter);
    expect(code).toBe(0);
    for (const label of ["items (list(str)) [default: ", "mode (Literal(a, z))", "flag (Optional(bool))"]) {
      expect(drawn).toContain(label);
    }
    expect(input.rawCalls).toEqual([true, false]);
    expect(JSON.parse(stdout)).toEqual({ flag: null, items: ["a", "b"], mode: "z" });
    expect(suppliedNames(join(tmp, "run.jsonl"))).toEqual([]);
  });

  test("a cancelled prompt is run cancelled, run.cancelled under --json", async () => {
    const stops: [string, (keys: FakeInput) => void][] = [
      ["Ctrl-C", (keys) => keys.write("\x03")],
      ["the end of input", (keys) => keys.end()],
    ];
    for (const [name, stop] of stops) {
      const { code, stdout, stderr } = await runOnTty(REQUIRED_INPUTS_FLOW, ["--json"], stop);
      expect(code, name).toBe(1);
      expect(stderr, name).toBe("run cancelled\n");
      expect(JSON.parse(stdout), name).toEqual({
        format: "yian.cli.v1",
        error: { code: "run.cancelled", message: "run cancelled" },
      });
    }
  });
});

const ASK_FLOW =
  "id: ask\nname: ask\nnodes:\n  ok:\n    kind: tool\n    tool_id: ask\n" +
  '    input:\n      prompt: "Proceed?"\n    output: !str\noutput: ${ok.output}\n';

// A CODE node that runs until the run is stopped (or 30 s pass), after a node whose progress line
// says the run is going.
const PAUSE_FLOW =
  "id: pause\nname: pause\nnodes:\n  first: {kind: code, output: !str, code: return 'x'}\n" +
  "  settle:\n    kind: code\n    depends_on: [first]\n    code: |\n" +
  "      await new Promise((resolve) => setTimeout(resolve, 30_000));\n      return null;\n" +
  "output: ${first.output}\n";

test("an ask off a terminal writes its prompt with no escape codes", async () => {
  const flow = file("flow.yaml", ASK_FLOW);
  const { io, stdout, stderr } = testIo();
  // A piped run: stdin has ended and stderr isn't a TTY.
  const input = new FakeInput({ tty: false });
  input.end();
  const output = new FakeOutput({ tty: false });
  const argv = ["run", flow, "-q", "--json", "--log", join(tmp, "run.jsonl")];
  const code = await main(argv, { ...io, stdin: input, stderrStream: output });
  expect(code).toBe(1);
  expect(stderr()).toEndWith("\nrun cancelled\n");
  expect(JSON.parse(stdout()).error.code).toBe("run.cancelled");
  expect(output.text).toContain("Proceed?");
  expect(output.text).not.toContain("\x1b");
});

test("a value for an inbox input is a usage error, as in chat", async () => {
  const flow =
    "id: f\nname: f\ninput:\n  messages: !Inbox(str)\nnodes:\n" +
    "  heard:\n    kind: tool\n    tool_id: take\n    input:\n      from: ${input.messages}\n" +
    "output: ${heard.output}\n";
  const { code, stdout, stderr } = await runFlowCli(flow, ["-q", "-i", "messages=hello"]);
  expect(code).toBe(2);
  expect(stdout).toBe("");
  expect(stderr).toContain('input "messages" is an inbox; it takes messages through send, not a value');
  expect(stderr).not.toContain("Run log: ");
});

describe("run's stops", () => {
  test("Ctrl-C at a prompt during the run: cancelled by user, exit 130, no document", async () => {
    const ctrlC = (keys: FakeInput) => keys.write("\x03");
    const { code, stdout, stderr, input } = await runOnTty(ASK_FLOW, ["--json"], ctrlC);
    expect(code).toBe(130);
    expect(stdout).toBe("");
    expect(stderr.endsWith("cancelled by user\n")).toBe(true);
    expect(input.rawCalls).toEqual([true, false]);
  });

  test("an already aborted io.stop stops the run before its log exists", async () => {
    const flow = file("flow.yaml", ECHO_FLOW);
    const log = join(tmp, "run.jsonl");
    const { io, stdout, stderr } = testIo();
    const stop = new AbortController();
    stop.abort(new HostStop("SIGTERM"));
    const argv = ["run", flow, "--json", "-q", "--log", log, "--input", "topic=clouds"];
    expect(await main(argv, { ...io, stop: stop.signal })).toBe(130);
    expect(stdout()).toBe("");
    expect(stderr()).toBe("cancelled by user\n");
    expect(existsSync(log)).toBe(false);
  });

  test("io.stop aborting mid-run stops it; its and the signals' listeners are gone", async () => {
    const flow = file("flow.yaml", PAUSE_FLOW);
    const log = join(tmp, "run.jsonl");
    const stop = new AbortController();
    const chunks: string[] = [];
    const stderr: Writer = {
      write: async (text) => {
        chunks.push(text);
        if (text.includes("✓ first")) {
          stop.abort(new HostStop("SIGTERM"));
          sigintWhileDraining = process.listenerCount("SIGINT");
        }
      },
    };
    const signalListeners = () =>
      ["SIGINT", "SIGTERM", "SIGHUP"].map((name) => process.listenerCount(name));
    const before = signalListeners();
    let sigintWhileDraining = -1;
    const io: CliIo = { ...testIo().io, stderr, stop: stop.signal };
    let code: number;
    try {
      code = await main(["run", flow, "--log", log], io);
    } finally {
      // The blocker never outlives the test, even if the progress line never comes.
      stop.abort(new HostStop("SIGTERM"));
    }
    expect(code).toBe(130);
    expect(chunks.at(-1)).toBe("cancelled by user\n");
    // A forwarded stop leaves the signal listeners, so a real signal still stops the drain.
    expect(sigintWhileDraining).toBe(before[0]! + 1);
    expect(existsSync(log)).toBe(true);
    expect(getEventListeners(stop.signal, "abort")).toHaveLength(0);
    expect(signalListeners()).toEqual(before);
  }, 20_000);

  test("io.stop aborting while an input prompt is up closes it: exit 130", async () => {
    const flow = file("flow.yaml", REQUIRED_INPUTS_FLOW);
    const { io, stdout, stderr } = testIo();
    const input = new FakeInput();
    const output = new FakeOutput();
    const stop = new AbortController();
    const argv = ["run", flow, "-q", "--json", "--log", join(tmp, "run.jsonl")];
    const running = main(argv, { ...io, stdin: input, stderrStream: output, stop: stop.signal });
    while (output.text === "") await new Promise((resolve) => setTimeout(resolve, 1));
    stop.abort(new HostStop("SIGTERM"));
    expect(await running).toBe(130);
    expect(stdout()).toBe("");
    expect(stderr()).toBe("cancelled by user\n");
    expect(input.rawCalls).toEqual([true, false]);
  });
});

// Two nodes that start together, so a TTY shows two spinner lines.
const PARALLEL_FLOW =
  "id: par\nname: par\nnodes:\n  a: {kind: code, output: !str, code: return 'a'}\n" +
  "  b: {kind: code, output: !str, code: return 'b'}\noutput:\n  a: ${a.output}\n  b: ${b.output}\n";

describe("run setup errors", () => {
  test("on a TTY the run failed: line comes after the spinners are erased", async () => {
    const flow = file("flow.yaml", PARALLEL_FLOW);
    const chunks: string[] = [];
    let broke = false;
    // A stderr whose spinner draw for `b` throws: the reporter's error rejects the run while
    // spinner lines are on screen.
    const stderr: Writer = {
      isTTY: true,
      write: (text) => {
        if (!broke && text.includes("⠋ b")) {
          broke = true;
          throw new TypeError("stderr broke");
        }
        chunks.push(text);
        return Promise.resolve();
      },
    };
    const io: CliIo = {
      stdout: sink().out,
      stderr,
      stdin: new PassThrough(),
      stderrStream: new PassThrough(),
      env: (name) => (name === "NO_COLOR" ? "1" : NO_ENV(name)),
    };
    expect(await main(["run", flow, "-q", "--log", join(tmp, "run.jsonl")], io)).toBe(0);
    chunks.length = 0;
    broke = false;
    expect(await main(["run", flow, "--log", join(tmp, "run2.jsonl")], io)).toBe(1);
    const last = chunks.at(-1)!;
    expect(last).toBe("run failed: stderr broke\n");
    expect(chunks.slice(0, -1).some((chunk) => chunk.startsWith("\r\x1b[K"))).toBe(true);
  });

  test("--engine-trace prints the stack box after run failed:", async () => {
    // A log whose name is longer than a file system takes: the run can't create it.
    const log = join(tmp, `${"n".repeat(300)}.jsonl`);
    const argv = ["-q", "--engine-trace", "--log", log, "--input", "topic=clouds"];
    const { code, stderr } = await runFlowCli(ECHO_FLOW, argv);
    expect(code).toBe(1);
    const failed = stderr.indexOf("run failed: ");
    expect(failed).toBeGreaterThan(-1);
    expect(stderr.indexOf("─ traceback ")).toBeGreaterThan(failed);
    expect(stderr).toContain("ENAMETOOLONG");
    expect(stderr).toContain("    at ");
  });
});

// A JS string expression holding an OSC 52 clipboard write, an OSC 8 link and a cursor move.
const ESCAPED_JS = String.raw`"x\u001b]52;c;aGk=\u0007y\u001b]8;;https://e.example\u001b\\z\u001b[2J"`;
const ESCAPED = "x\x1b]52;c;aGk=\x07y\x1b]8;;https://e.example\x1b\\z\x1b[2J";

/** Return `text` with each progress line's elapsed time (` [0s]`) taken out. */
function untimed(text: string): string {
  return text.replace(/ \[\d+s\]\n/g, "\n");
}

/** Run `yian run` on `body` with stdout and stderr TTYs or not, colour off. */
async function runAt(body: string, argv: readonly string[], isTTY: boolean) {
  const flow = file("flow.yaml", body);
  mkdirSync(join(tmp, "home"), { recursive: true });
  const out = sink();
  const err = sink();
  const io: CliIo = {
    stdout: { ...out.out, isTTY },
    stderr: { ...err.out, isTTY },
    stdin: new PassThrough(),
    stderrStream: new PassThrough(),
    env: (name) => (name === "NO_COLOR" ? "1" : name === "HOME" ? join(tmp, "home") : NO_ENV(name)),
  };
  const code = await main(["run", flow, ...argv], io);
  return { code, stdout: out.text(), stderr: err.text() };
}

describe("run's model and tool text with escape sequences", () => {
  const output = `id: f\nname: f\nnodes:\n  a: {kind: code, output: !str, code: 'return ${ESCAPED_JS}'}\noutput: \${a.output}\n`;
  const failing = output.replace("return ", "throw new Error(").replace("'}", ")'}");

  test("the output is stripped on a TTY and kept exactly when piped", async () => {
    expect((await runAt(output, ["-q"], true)).stdout).toBe("xyz\n");
    expect((await runAt(output, ["-q"], false)).stdout).toBe(`${ESCAPED}\n`);
  });

  test("a verbose node output is stripped on a TTY and kept exactly when piped", async () => {
    expect(untimed((await runAt(output, ["-v"], true)).stderr)).toContain("✓ a\n    xyz\n");
    expect(untimed((await runAt(output, ["-v"], false)).stderr)).toContain(`✓ a\n    ${ESCAPED}\n`);
  });

  test("a failure's message and traceback are stripped on a TTY and kept exactly when piped", async () => {
    const tty = await runAt(failing, ["--engine-trace"], true);
    expect(tty.code).toBe(1);
    expect(untimed(tty.stderr)).toContain("✗ a\n    Error: xyz\n");
    expect(tty.stderr).toContain("\nError: xyz\n╭─ traceback ");
    expect(tty.stderr).toContain("│ Error: xyz ");
    expect(tty.stderr).not.toContain("\x1b]");
    expect(tty.stderr).not.toContain("\x07");
    const piped = await runAt(failing, [], false);
    expect(piped.code).toBe(1);
    expect(untimed(piped.stderr)).toContain(`✗ a\n    Error: ${ESCAPED}\n`);
    expect(piped.stderr).toContain(`\nError: ${ESCAPED}\n`);
  });

  test("a setup failure's message is stripped on a TTY and kept exactly when piped", async () => {
    // A log whose name is longer than a file system takes can't be created, and the error names
    // its path.
    const folder = join(tmp, "t\x1b]52;c;aGk=\x07");
    mkdirSync(folder);
    const argv = ["-q", "--log", join(folder, `${"n".repeat(300)}.jsonl`)];
    const tty = await runAt(output, argv, true);
    expect(tty.code).toBe(1);
    expect(tty.stderr).toMatch(/run failed: ENAMETOOLONG\b.*\/t\/n+\.jsonl'\n$/);
    const piped = await runAt(output, argv, false);
    expect(piped.stderr).toMatch(/run failed: ENAMETOOLONG\b.*\/t\x1b\]52;c;aGk=\x07\/n+\.jsonl'\n$/);
  });
});

describe("run --json", () => {
  test("succeeded: the output in the codec form, only the document on stdout", async () => {
    const { code, stdout } = await runFlowCli(ECHO_FLOW, ["--json", "--input", "topic=clouds"]);
    expect(code).toBe(0);
    expect(JSON.parse(stdout)).toEqual({ format: "yian.cli.v1", status: "succeeded", output: { topic: "clouds" } });
  });

  test("failed: the diagnostic and the failed node", async () => {
    const { code, stdout } = await runFlowCli(FAIL_FLOW, ["--json", "--input", "topic=clouds"]);
    expect(code).toBe(1);
    const document = JSON.parse(stdout);
    expect(document.status).toBe("failed");
    expect(document.failure.node).toBe("echo");
    expect(document.failure.message).toContain("always fails");
    expect(document.failure.frames[0].label).toBe("flow.yaml");
  });

  test("a load error: the located diagnostic on stderr, compose.load on stdout", async () => {
    const { code, stdout, stderr } = await runFlowCli(ECHO_FLOW.replace("${echo.output}", "${ech.output}"), [
      "--json",
    ]);
    expect(code).toBe(1);
    expect(stderr).toContain("flow.yaml:");
    expect(JSON.parse(stdout).error.code).toBe("compose.load");
  });

  test("an unknown input is a usage error: exit 2, nothing on stdout", async () => {
    const argv = ["--json", "--input", "topic=clouds", "--input", "topik=x"];
    const { code, stdout, stderr } = await runFlowCli(ECHO_FLOW, argv);
    expect(code).toBe(2);
    expect(stdout).toBe("");
    expect(stderr).toContain("unknown input");
  });
});

/** A flow named `main` whose one node calls the `uses:` alias `child`, bound to `ref`. */
function parentFlow(ref: string): string {
  return (
    "id: main\nname: main\ninput:\n  n: !int\nnodes:\n  work:\n    kind: call\n    call: child\n" +
    `    input:\n      n: \${input.n}\noutput: \${work.output}\nuses:\n  child: ${ref}\n`
  );
}

/** A flow taking `n` whose one node `a` runs the CODE reference `ref`. */
function codeFlowText(ref: string): string {
  return `id: c\nname: c\ninput:\n  n: !int\nnodes:\n  a: {kind: code, code: ${ref}, input: {n: "\${input.n}"}}\noutput: \${a.output}\n`;
}

/** Run `yian check --json --root root flow`, which fails, and return its one diagnostic. */
async function rootDiagnostic(root: string, flow: string) {
  const { code, flows } = await checkJson(["--root", root, flow]);
  expect(code).toBe(1);
  expect(flows[0].diagnostics).toHaveLength(1);
  return flows[0].diagnostics[0];
}

/** Run `yian check --json` on `argv` and return the exit code and the document's `flows`. */
async function checkJson(argv: string[]) {
  const { io, stdout, stderr } = testIo();
  const code = await main(["check", "--json", ...argv], io);
  expect(stderr()).toBe("");
  return { code, flows: JSON.parse(stdout()).flows };
}

describe("check", () => {
  test("a missing or non-UTF-8 file is compose.load, without loading", async () => {
    const missing = join(tmp, "missing.yaml");
    const bad = file("bad.yaml", NOT_UTF8);
    expect(await checkJson([missing, bad])).toEqual({
      code: 1,
      flows: [
        {
          path: missing,
          ok: false,
          diagnostics: [
            { code: "compose.load", message: `can't read ${missing}: No such file or directory` },
          ],
        },
        {
          path: bad,
          ok: false,
          diagnostics: [{ code: "compose.load", message: `can't read ${bad}: not UTF-8 text` }],
        },
      ],
    });
  });

  test("text: the path then the message on stderr", async () => {
    const missing = join(tmp, "missing.yaml");
    const { io, stdout, stderr } = testIo();
    expect(await main(["check", missing], io)).toBe(1);
    expect(stdout()).toBe("");
    expect(stderr()).toBe(`${missing}:\ncan't read ${missing}: No such file or directory\n`);
  });

  test("text: a flow that loads is an ok line on stdout", async () => {
    const flow = file("flow.yaml", FLOW);
    const { io, stdout, stderr } = testIo();
    expect(await main(["check", flow], io)).toBe(0);
    expect(stdout()).toBe(`${flow}: ok\n`);
    expect(stderr()).toBe("");
  });

  test("the ok document: canonical type labels, a typedef expanded", async () => {
    const flow = file(
      "profile.yaml",
      "id: profile\nname: profile\ndescription: Greets a person.\n" +
        "typedefs:\n  Person: {name: str, age: int}\n" +
        "input:\n  who: !Person\n  tag: !Optional(str)\n  w: !int 3\n" +
        "nodes: {}\noutput: ${input.who}\n",
    );
    const bare = file("bare.yaml", "id: bare\nname: bare\nnodes: {}\noutput: 1\n");
    expect(await checkJson([flow, bare])).toEqual({
      code: 0,
      flows: [
        {
          path: flow,
          ok: true,
          id: "profile",
          name: "profile",
          description: "Greets a person.",
          inputs: [
            { name: "who", type: '{"age":int,"name":str}', required: true, has_default: false },
            { name: "tag", type: "Optional(str)", required: false, has_default: false },
            { name: "w", type: "int", required: false, has_default: true },
          ],
        },
        { path: bare, ok: true, id: "bare", name: "bare", description: null, inputs: [] },
      ],
    });
  });

  test("--root: a flow outside the root or setting system.paths fails, labelled from the root", async () => {
    const root = join(tmp, "company");
    const outside = file("other/flow.yaml", FLOW);
    const paths = file("company/main.yaml", `system:\n  paths: [lib]\n${FLOW}`);
    expect(await checkJson(["--root", root, outside, paths])).toEqual({
      code: 1,
      flows: [
        {
          path: outside,
          ok: false,
          diagnostics: [{ code: "compose.outside_root", message: "../other/flow.yaml is outside the root" }],
        },
        {
          path: paths,
          ok: false,
          diagnostics: [
            {
              code: "compose.load",
              message: "system.paths is not supported when loading confined to a folder; use a relative uses: path",
            },
          ],
        },
      ],
    });
  });

  test("--root: invalid YAML is located under its path relative to the root", async () => {
    const root = join(tmp, "company");
    const flow = file("company/sub/main.yaml", "id: [f\n");
    const { code, flows } = await checkJson(["--root", root, flow]);
    expect(code).toBe(1);
    const [diagnostic] = flows[0].diagnostics;
    expect(diagnostic.code).toBe("compose.load");
    expect(diagnostic.message).toStartWith("invalid YAML: ");
    expect(diagnostic.frames[0].label).toBe(join("sub", "main.yaml"));
  });

  test("--root: a missing flow can't be read, a symlink loop stays an OS error", async () => {
    const root = join(tmp, "company");
    mkdirSync(root);
    const missing = join(root, "missing.yaml");
    const loop = join(tmp, "loop.yaml");
    symlinkSync(loop, loop);
    const { code, flows } = await checkJson(["--root", root, missing, loop]);
    expect(code).toBe(1);
    expect(flows[0].diagnostics).toEqual([
      { code: "compose.load", message: `can't read ${missing}: No such file or directory` },
    ]);
    expect(flows[1].diagnostics[0].code).toBe("compose.load");
    expect(flows[1].diagnostics[0].message).toStartWith(`can't read ${loop}: Too many symbolic links`);
  });

  test("--root: a flow and its uses: child load from the importing file's directory", async () => {
    const root = join(tmp, "company");
    const flow = file("company/main.yaml", parentFlow("lib/child"));
    file("company/lib/child.yaml", FLOW);
    expect(await checkJson(["--root", root, flow])).toEqual({
      code: 0,
      flows: [
        {
          path: flow,
          ok: true,
          id: "main",
          name: "main",
          description: null,
          inputs: [{ name: "n", type: "int", required: true, has_default: false }],
        },
      ],
    });
  });

  test("--root: a missing child is not found, at its path relative to the root", async () => {
    const root = join(tmp, "company");
    const flow = file("company/main.yaml", parentFlow("lib/missing"));
    const diagnostic = await rootDiagnostic(root, flow);
    expect(diagnostic.code).toBe("compose.load");
    expect(diagnostic.message).toEndWith(`flow 'lib/missing' not found: ${join("lib", "missing.yaml")}`);
    expect(diagnostic.frames.map((f: { note: string }) => f.note)).toEqual(["uses alias 'child'"]);
  });

  test("a ref written with .yaml gets the same hint with and without --root", async () => {
    const root = join(tmp, "company");
    const flow = file("company/main.yaml", parentFlow("./lib/child.yaml"));
    file("company/lib/child.yaml", FLOW);
    const hint = "; write a uses: ref without its extension";
    const { code, flows } = await checkJson([flow]);
    expect(code).toBe(1);
    expect(flows[0].diagnostics[0].message).toEndWith(`(looked for './lib/child.yaml.yaml')${hint}`);
    expect((await rootDiagnostic(root, flow)).message).toEndWith(`not found: lib/child.yaml.yaml${hint}`);
  });

  test("--root: a child under a regular file is not found", async () => {
    const root = join(tmp, "company");
    const flow = file("company/main.yaml", parentFlow("lib/child"));
    file("company/lib", "not a directory\n");
    const diagnostic = await rootDiagnostic(root, flow);
    expect(diagnostic.code).toBe("compose.load");
    expect(diagnostic.message).toEndWith(`flow 'lib/child' not found: ${join("lib", "child.yaml")}`);
  });

  test("--root: a child outside the root keeps its outside_root code", async () => {
    const root = join(tmp, "company");
    const flow = file("company/main.yaml", parentFlow("../lib/child"));
    file("lib/child.yaml", FLOW);
    const diagnostic = await rootDiagnostic(root, flow);
    expect(diagnostic.code).toBe("compose.outside_root");
    expect(diagnostic.message).toBe(`${join("..", "lib", "child.yaml")} is outside the root`);
  });

  test("--root: a CODE reference outside the root fails, written as in the flow, whether or not its file exists", async () => {
    const root = join(tmp, "company");
    file("helpers.ts", "export function fn(inputs) { return 1 }\n");
    for (const ref of ["../helpers.ts:fn", "../missing.ts:fn"]) {
      const flow = file("company/main.yaml", codeFlowText(ref));
      const diagnostic = await rootDiagnostic(root, flow);
      expect(diagnostic.code).toBe("compose.load");
      expect(diagnostic.message).toBe(`node 'a': \`code:\` reference '${ref}' is outside the code root`);
    }
  });

  test("--root: an !include outside the root fails, written as in the flow, whether or not its file exists; inside, it loads", async () => {
    const root = join(tmp, "company");
    file("prompt.md", "outside\n");
    file("company/prompt.md", "inside\n");
    for (const path of ["../prompt.md", "../missing.md"]) {
      const flow = file("company/main.yaml", `id: c\nname: c\nnodes: {}\noutput: !include ${path}\n`);
      const diagnostic = await rootDiagnostic(root, flow);
      expect(diagnostic.code).toBe("compose.load");
      expect(diagnostic.message).toBe(`\`!include\` file '${path}' is outside the code root; include a file inside it`);
    }
    const flow = file("company/main.yaml", "id: c\nname: c\nnodes: {}\noutput: !include prompt.md\n");
    expect((await checkJson(["--root", root, flow])).code).toBe(0);
  });

  test("--root: an !include path holding a NUL is a load diagnostic", async () => {
    const root = join(tmp, "company");
    const flow = file("company/main.yaml", 'id: c\nname: c\nnodes: {}\noutput: !include "in\\0.md"\n');
    const diagnostic = await rootDiagnostic(root, flow);
    expect(diagnostic.code).toBe("compose.load");
    expect(diagnostic.message).toBe("`!include` file 'in\\x00.md' holds a NUL character; write the path without it");
  });

  test("--root: a uses: child's CODE reference outside the root fails; inside, it loads", async () => {
    const root = join(tmp, "company");
    const flow = file("company/main.yaml", parentFlow("lib/child"));
    file("helpers.ts", "export function fn(inputs) { return 1 }\n");
    file("company/lib/helpers.ts", "export function fn(inputs) { return 1 }\n");
    file("company/lib/child.yaml", codeFlowText("../../helpers.ts:fn"));
    const diagnostic = await rootDiagnostic(root, flow);
    expect(diagnostic.message).toEndWith("`code:` reference '../../helpers.ts:fn' is outside the code root");
    file("company/lib/child.yaml", codeFlowText("./helpers.ts:fn"));
    expect((await checkJson(["--root", root, flow])).code).toBe(0);
  });

  test("--root: a reference cycle names the flow it reaches again", async () => {
    const root = join(tmp, "company");
    const a = file("company/a.yaml", parentFlow("b").replace("id: main", "id: a"));
    file("company/b.yaml", parentFlow("a").replace("id: main", "id: b"));
    const diagnostic = await rootDiagnostic(root, a);
    expect(diagnostic.code).toBe("compose.load");
    expect(diagnostic.message).toEndWith("flow-reference cycle at a.yaml");
    expect(diagnostic.frames.map((f: { label: string }) => f.label)).toEqual(["a.yaml", "b.yaml"]);
  });

  test("--root: a pinned version the child doesn't have", async () => {
    const root = join(tmp, "company");
    const pinned = file("company/pinned.yaml", parentFlow("lib/child@2"));
    file("company/lib/child.yaml", FLOW.replace("name: f\n", "name: f\nversion: '1'\n"));
    expect((await rootDiagnostic(root, pinned)).message).toBe(
      'uses: ref pinned lib/child@2 but the resolved file declares version: "1" ' +
        '(file-per-version: name a file that declares "2")',
    );
    const unversioned = file("company/unversioned.yaml", parentFlow("lib/bare@2"));
    file("company/lib/bare.yaml", FLOW);
    expect((await rootDiagnostic(root, unversioned)).message).toBe(
      "uses: ref pinned lib/bare@2 but the resolved file declares version: null " +
        '(file-per-version: name a file that declares "2")',
    );
  });

  test("a CODE reference resolves against its own file's directory, not the cwd", async () => {
    const node = "  a:\n    kind: code\n    code: ./m.ts:f\n    output: !str\n";
    const fn = "export function f() { return 'x'; }\n";
    const alone = file("alone/flow.yaml", `id: alone\nname: alone\nnodes:\n${node}output: \${a.output}\n`);
    file("alone/m.ts", fn);
    const root = join(tmp, "company");
    const flow = file(
      "company/sub/main.yaml",
      `id: main\nname: main\nnodes:\n${node}  work:\n    kind: call\n    call: child\n` +
        "output: ${work.output}\nuses:\n  child: lib/child\n",
    );
    file("company/sub/m.ts", fn);
    // The child's own file is only beside it, so resolving it against its importer's directory fails.
    const childNode = node.replace("./m.ts:f", "./n.ts:f");
    file("company/sub/lib/child.yaml", `id: child\nname: child\nnodes:\n${childNode}output: \${a.output}\n`);
    file("company/sub/lib/n.ts", fn);
    for (const argv of [[alone], ["--root", root, flow]]) {
      const { code, flows } = await checkJson(argv);
      expect([code, flows[0].ok]).toEqual([0, true]);
    }
  });
});

describe("status", () => {
  test("--help names the run log the CLI writes, and the list without one", async () => {
    const { io, stdout } = testIo();
    expect(await main(["status", "--help"], io)).toBe(0);
    expect(stdout()).toMatch(
      /^ {2}\[LOG\] +Path to a run log, such as ~\/\.yian\/runs\/<run_id>\/log\.jsonl; left out, list the runs in ~\/\.yian\/runs$/m,
    );
  });

  test("with no log, lists the runs of ~/.yian/runs from the index each run updates", async () => {
    const runs = join(tmp, "home", ".yian", "runs");
    expect((await runFlowCli(ECHO_FLOW, ["-q", "--input", "topic=clouds"])).code).toBe(0);
    expect((await runFlowCli(FAIL_FLOW, ["-q", "--input", "topic=rain"])).code).toBe(1);
    // `yian run` updated the index.
    const rows = readRunIndex(runs);
    expect(rows.map((row) => row.status).sort()).toEqual(["failed", "succeeded"]);
    const flow = join(realpathSync(tmp), "flow.yaml");
    expect(rows.map((row) => row.flow)).toEqual([flow, flow]);
    // A run folder the index hasn't seen yet is listed too: `status` updates first.
    mkdirSync(join(runs, "held"));
    const held = JsonlRunLog.create(join(runs, "held", "log.jsonl"), "held", {
      clock: () => Date.parse("2999-01-01T00:00:00.000Z"),
    });
    held.append(
      new RunStartedRecord({ flowFingerprint: "a".repeat(64), startWiringOverride: {}, effectiveLlmByStaticPath: {}, numWorkers: 0 }),
    );
    held.append(new NodeStartedRecord("agent"));
    const usage = { input_tokens: 9n, output_tokens: 2n, cache_read_tokens: 3n, cache_creation_tokens: 5n, reasoning_tokens: 1n };
    held.append(new NodeStepRecord("agent", 1, { kind: "model_call", usage }));
    try {
      const text = testIo({ HOME: join(tmp, "home") });
      expect(await main(["status"], text.io)).toBe(0);
      const width = rows[0]!.runId.length;
      const line = (id: string, status: string, started: string, shown: string) =>
        `${id.padEnd(width)}  ${status.padEnd(9)}  ${started.padEnd(24)}  ${shown}\n`;
      expect(text.stdout()).toBe(
        line("run id", "status", "started", "flow") +
          rows.map((row) => line(row.runId, row.status, row.startedAt!, flow)).join("") +
          line("held", "running", "2999-01-01T00:00:00.000Z", "-"),
      );
      expect(text.stderr()).toBe("");
      const json = testIo({ HOME: join(tmp, "home") });
      expect(await main(["status", "--json"], json.io)).toBe(0);
      const document = JSON.parse(json.stdout());
      expect(document.format).toBe("yian.cli.v1");
      const schema = JSON.parse(readFileSync(fileURLToPath(new URL("../conformance/schema/cli.schema.json", import.meta.url)), "utf8")) as Schema;
      const runsDocument = (schema["$defs"] as Record<string, Schema>)["runs_document"]!;
      expect(validate(schema, runsDocument, parseJson(json.stdout(), "plain"))).toEqual([]);
      expect(document.runs.map((run: { run_id: string }) => run.run_id)).toEqual([...rows.map((row) => row.runId), "held"]);
      const succeeded = rows.find((row) => row.status === "succeeded")!;
      expect(document.runs.find((run: { run_id: string }) => run.run_id === succeeded.runId)).toEqual({
        run_id: succeeded.runId,
        flow,
        status: "succeeded",
        started_at: succeeded.startedAt,
        ended_at: succeeded.endedAt,
        parent_run_id: null,
        memories: [],
        last_seq: succeeded.lastSeq,
        result: { topic: "clouds" },
        input_tokens: 0,
        output_tokens: 0,
        cache_read_tokens: 0,
        cache_creation_tokens: 0,
        reasoning_tokens: 0,
        ready: false,
        wake_at: null,
        unreadable: null,
      });
      expect(document.runs[2]).toMatchObject({
        run_id: "held",
        flow: null,
        status: "running",
        result: null,
        input_tokens: 9,
        output_tokens: 2,
        cache_read_tokens: 3,
        cache_creation_tokens: 5,
        reasoning_tokens: 1,
      });
    } finally {
      held.close();
    }
  });

  test("with no log and no runs folder, lists nothing; a run logged elsewhere isn't indexed; a bad log warns once and is listed unreadable", async () => {
    const empty = testIo({ HOME: join(tmp, "home") });
    expect(await main(["status"], empty.io)).toBe(0);
    expect([empty.stdout(), empty.stderr()]).toEqual(["", ""]);
    const json = testIo({ HOME: join(tmp, "home") });
    expect(await main(["status", "--json"], json.io)).toBe(0);
    expect(json.stdout()).toBe('{"format":"yian.cli.v1","runs":[]}\n');
    mkdirSync(join(tmp, "elsewhere", "run-1"), { recursive: true });
    const log = join(tmp, "elsewhere", "run-1", "log.jsonl");
    expect((await runFlowCli(ECHO_FLOW, ["-q", "--input", "topic=clouds", "--log", log])).code).toBe(0);
    expect(existsSync(join(tmp, "elsewhere", "index.sqlite"))).toBe(false);
    // A log the update can't read is a warning, and a row unreadable at its first record.
    const damaged = join(tmp, "home", ".yian", "runs", "damaged");
    mkdirSync(damaged, { recursive: true });
    writeFileSync(join(damaged, "log.jsonl"), "not a run log\n");
    const listed = `run id   status               started  flow\ndamaged  unreadable at seq 0  -        -\n`;
    const warned = testIo({ HOME: join(tmp, "home") });
    expect(await main(["status"], warned.io)).toBe(0);
    expect(warned.stdout()).toBe(listed);
    expect(warned.stderr()).toStartWith(`warning: run index skipped ${join(damaged, "log.jsonl")}: run log `);
    // Until the log changes, the next update doesn't warn again.
    const again = testIo({ HOME: join(tmp, "home") });
    expect(await main(["status", "--json"], again.io)).toBe(0);
    expect(again.stderr()).toBe("");
    const schema = JSON.parse(readFileSync(fileURLToPath(new URL("../conformance/schema/cli.schema.json", import.meta.url)), "utf8")) as Schema;
    const runsDocument = (schema["$defs"] as Record<string, Schema>)["runs_document"]!;
    expect(validate(schema, runsDocument, parseJson(again.stdout(), "plain"))).toEqual([]);
    expect(JSON.parse(again.stdout()).runs).toEqual([
      expect.objectContaining({
        run_id: "damaged",
        status: "crashed",
        started_at: null,
        last_seq: null,
        unreadable: { seq: 0, reason: expect.stringMatching(/^run log .*line 1/) },
      }),
    ]);
  });

  test("a damaged index is rebuilt from the logs: the runs are listed, with one warning", async () => {
    expect((await runFlowCli(ECHO_FLOW, ["-q", "--input", "topic=clouds"])).code).toBe(0);
    const runs = join(tmp, "home", ".yian", "runs");
    const index = join(runs, "index.sqlite");
    writeFileSync(index, "garbage");
    const text = testIo({ HOME: join(tmp, "home") });
    expect(await main(["status"], text.io)).toBe(0);
    expect(text.stdout()).toContain("succeeded");
    expect(text.stderr()).toBe(
      `warning: run index ${index} was damaged (file is not a database); rebuilt it from the logs\n`,
    );
    writeFileSync(index, "garbage");
    const json = testIo({ HOME: join(tmp, "home") });
    expect(await main(["status", "--json"], json.io)).toBe(0);
    expect(JSON.parse(json.stdout()).runs.map((run: { status: string }) => run.status)).toEqual(["succeeded"]);
    const again = testIo({ HOME: join(tmp, "home") });
    expect(await main(["status"], again.io)).toBe(0);
    expect(again.stderr()).toBe("");
  });

  test("a copied run folder: status LOG names the run id its folder lacks, as the list does", async () => {
    expect((await runFlowCli(ECHO_FLOW, ["-q", "--input", "topic=clouds"])).code).toBe(0);
    const runs = join(tmp, "home", ".yian", "runs");
    const [runId] = readdirSync(runs).filter((name) => name !== "index.sqlite");
    cpSync(join(runs, runId!), join(runs, `${runId}-backup`), { recursive: true });
    const listed = testIo({ HOME: join(tmp, "home") });
    expect(await main(["status", "--json"], listed.io)).toBe(0);
    const backup = JSON.parse(listed.stdout()).runs.find((run: { run_id: string }) => run.run_id === `${runId}-backup`);
    const reason = `the log is run '${runId}', not its folder's '${runId}-backup'`;
    expect(backup.unreadable).toEqual({ seq: 0, reason });
    const log = join(runs, `${runId}-backup`, "log.jsonl");
    const json = testIo({ HOME: join(tmp, "home") });
    expect(await main(["status", log, "--json"], json.io)).toBe(1);
    const message = `run log ${log}: ${reason}`;
    expect(json.stderr()).toBe(`${message}\n`);
    expect(JSON.parse(json.stdout())).toEqual({ format: "yian.cli.v1", error: { code: "run.setup", message } });
    // The same log outside the runs folder is a plain run log.
    mkdirSync(join(tmp, "elsewhere-copy"));
    const elsewhere = join(tmp, "elsewhere-copy", "log.jsonl");
    cpSync(log, elsewhere);
    const plain = testIo({ HOME: join(tmp, "home") });
    expect(await main(["status", elsewhere], plain.io)).toBe(0);
    expect(plain.stdout()).toBe("status: succeeded\n");
  });

  /** Run `ECHO_FLOW` to the end with its log at `tmp/<name>`, and return the log's path. */
  async function finishedLog(name = "finished.jsonl"): Promise<string> {
    const log = join(tmp, name);
    const { code } = await runFlowCli(ECHO_FLOW, ["-q", "--input", "topic=clouds", "--log", log]);
    expect(code).toBe(0);
    return log;
  }

  /** The finished log cut after the `echo` node's start: a run whose process ended mid-run. */
  async function crashedLog(): Promise<string> {
    const log = await finishedLog("crashed.jsonl");
    const lines = readFileSync(log, "utf8").split("\n");
    const started = lines.findIndex(
      (line) => line.includes('"kind":"node_started"') && line.includes('"node_id":"echo"'),
    );
    expect(started).toBeGreaterThan(0);
    writeFileSync(log, `${lines.slice(0, started + 1).join("\n")}\n`);
    return log;
  }

  test("--json: a finished run's document", async () => {
    const log = await finishedLog();
    const { io, stdout, stderr } = testIo();
    expect(await main(["status", log, "--json"], io)).toBe(0);
    expect(stdout()).toBe(
      '{"format":"yian.cli.v1","crashed_attempts":{},"in_flight":[],"ready":false,"status":"succeeded","wake_at":null}\n',
    );
    expect(stderr()).toBe("");
  });

  test("--json: a crashed run's nodes in flight and crashed attempts", async () => {
    const log = await crashedLog();
    const before = readFileSync(log, "utf8");
    const { io, stdout } = testIo();
    expect(await main(["status", log, "--json"], io)).toBe(0);
    expect(stdout()).toBe(
      '{"format":"yian.cli.v1","crashed_attempts":{"echo":1},"in_flight":["echo"],"ready":false,"status":"crashed","wake_at":null}\n',
    );
    // It never appends.
    expect(readFileSync(log, "utf8")).toBe(before);
  });

  test("text: the status, then the nodes in flight and crashed attempts when any", async () => {
    const finished = testIo();
    expect(await main(["status", await finishedLog()], finished.io)).toBe(0);
    expect(finished.stdout()).toBe("status: succeeded\n");
    expect(finished.stderr()).toBe("");
    const crashed = testIo();
    expect(await main(["status", await crashedLog()], crashed.io)).toBe(0);
    expect(crashed.stdout()).toBe(
      "status: crashed\nin flight: echo\ncrashed attempts: echo=1\n",
    );
  });

  test("a log a writer holds is running", async () => {
    const path = join(tmp, "held.jsonl");
    const log = JsonlRunLog.create(path, "run-1");
    try {
      const { io, stdout } = testIo();
      expect(await main(["status", path, "--json"], io)).toBe(0);
      expect(JSON.parse(stdout())).toEqual({
        format: "yian.cli.v1",
        status: "running",
        in_flight: [],
        crashed_attempts: {},
        ready: false,
        wake_at: null,
      });
    } finally {
      log.close();
    }
  });

  test("a missing log exits 2", async () => {
    const log = join(tmp, "missing.jsonl");
    const { io, stdout, stderr } = testIo();
    expect(await main(["status", log], io)).toBe(2);
    expect(stdout()).toBe("");
    expect(stderr()).toEndWith(`Error: Invalid value for 'LOG': File '${log}' does not exist.\n`);
  });

  test("a damaged or empty log is run.setup: its message on stderr, exit 1", async () => {
    const damaged = file("damaged.jsonl", "not a run log\n");
    const json = testIo();
    expect(await main(["status", damaged, "--json"], json.io)).toBe(1);
    const message = json.stderr().trimEnd();
    expect(message).toStartWith(`run log ${damaged} line 1 (seq 0): `);
    expect(JSON.parse(json.stdout())).toEqual({
      format: "yian.cli.v1",
      error: { code: "run.setup", message },
    });
    const empty = file("empty.jsonl", "");
    const text = testIo();
    expect(await main(["status", empty], text.io)).toBe(1);
    expect(text.stdout()).toBe("");
    expect(text.stderr()).toBe(`cannot inspect an empty run log: ${empty}\n`);
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

/** Spawn the entry with `args` in `tmp`, collecting its streams. */
function spawnEntry(args: string[]) {
  const runtime = process.versions.bun !== undefined ? ["--no-install", "--no-env-file"] : [];
  const command = [...runtime, CLI_ENTRY, ...args];
  const child = spawn(process.execPath, command, { cwd: tmp, env: { HOME: tmp } });
  const streams = { out: "", err: "" };
  child.stdout.setEncoding("utf8").on("data", (text: string) => (streams.out += text));
  child.stderr.setEncoding("utf8").on("data", (text: string) => (streams.err += text));
  const closed = new Promise<{ code: number | null; signal: NodeJS.Signals | null }>((resolve) => {
    child.on("close", (code, signal) => resolve({ code, signal }));
  });
  return { child, streams, closed };
}

/**
 * Run the entry with `args` in `tmp` (as its HOME) and wait for it, at most 20 seconds, so a
 * command that hangs fails the test instead of stalling the suite.
 */
function runEntrySync(args: string[]) {
  const runtime = process.versions.bun !== undefined ? ["--no-install", "--no-env-file"] : [];
  return spawnSync(process.execPath, [...runtime, CLI_ENTRY, ...args], {
    cwd: tmp,
    env: { HOME: tmp },
    encoding: "utf8",
    timeout: 20_000,
    killSignal: "SIGKILL",
  });
}

describe.skipIf(process.platform === "win32")("a FIFO at a run log's path", () => {
  test("status of it is run.setup, exit 1; the list skips a run folder's FIFO log with a warning", () => {
    const fifo = join(tmp, "fifo.jsonl");
    execFileSync("mkfifo", [fifo]);
    const inspected = runEntrySync(["status", fifo]);
    expect([inspected.status, inspected.signal, inspected.stdout, inspected.stderr]).toEqual([
      1,
      null,
      "",
      `can't read ${fifo}: Not a regular file\n`,
    ]);
    const log = join(tmp, ".yian", "runs", "ffff", "log.jsonl");
    mkdirSync(dirname(log), { recursive: true });
    execFileSync("mkfifo", [log]);
    const listed = runEntrySync(["status"]);
    expect([listed.status, listed.signal, listed.stdout, listed.stderr]).toEqual([
      0,
      null,
      "",
      `warning: run index skipped ${log}: EINVAL: not a regular file, open '${log}'\n`,
    ]);
  }, 45_000);

  test("run --resume of it is a resume failure without a stack", () => {
    const fifo = join(tmp, "fifo.jsonl");
    execFileSync("mkfifo", [fifo]);
    const resumed = runEntrySync(["run", "--resume", fifo, "-q"]);
    expect([resumed.status, resumed.signal]).toEqual([1, null]);
    expect(resumed.stderr).toStartWith(`resume failed: EINVAL: not a regular file, open '${fifo}'\n`);
    expect(resumed.stderr).not.toContain("    at ");
  }, 25_000);
});

describe("childCommand", () => {
  test("a binary re-runs itself with the marker; a script passes its runtime flags and its path", () => {
    const args = ["run", "flow.yaml", "--json"];
    expect(childCommand(args, true)).toEqual(["--yian-json-child", ...args]);
    expect(childCommand(args, false)).toEqual([
      ...process.execArgv,
      process.argv[1]!,
      "--yian-json-child",
      ...args,
    ]);
  });

  test("outside a binary, compiled is false", () => {
    expect(compiled).toBe(false);
  });
});

describe("the entry's run --json", () => {
  test("stdout holds only the large document; the flow's stdout writes go to stderr", async () => {
    const code = "|\n      console.log('noise');\n      return 'x'.repeat(300000);";
    const flow = file("flow.yaml", topicFlow(code));
    const log = join(tmp, "run.jsonl");
    const args = ["run", flow, "-q", "--json", "--log", log, "--input", "topic=t"];
    const { streams, closed } = spawnEntry(args);
    expect(await closed).toEqual({ code: 0, signal: null });
    const output = { topic: "x".repeat(300000) };
    expect(JSON.parse(streams.out)).toEqual({ format: "yian.cli.v1", status: "succeeded", output });
    expect(streams.err).toContain("noise\n");
  }, 30_000);

  test("a grandchild holding the flow's stdout doesn't hold the command", async () => {
    // Detached and unref'd so the run ends; it keeps fd 1 and fd 2, which go to stderr, for 60 s.
    const pidFile = join(tmp, "sleep.pid");
    const spawnSleep = "const sleeper = (await import('node:child_process'))" +
      ".spawn('sleep', ['60'], { stdio: 'inherit', detached: true });\n      sleeper.unref();\n      " +
      `(await import('node:fs')).writeFileSync(${JSON.stringify(pidFile)}, String(sleeper.pid));`;
    const code = `|\n      ${spawnSleep}\n      return 'done';`;
    const flow = file("flow.yaml", topicFlow(code));
    const log = join(tmp, "run.jsonl");
    const args = ["run", flow, "-q", "--json", "--log", log, "--input", "topic=t"];
    const { child, streams } = spawnEntry(args);
    try {
      // Not `close`: the grandchild holds this test's stderr pipe too.
      const ended = new Promise((resolve) => child.stdout.on("end", resolve));
      const exited = await new Promise((resolve) => child.on("exit", (code) => resolve(code)));
      await ended;
      expect(exited).toBe(0);
      expect(JSON.parse(streams.out).output).toEqual({ topic: "done" });
      // The command ended while the grandchild still runs, however slow the machine is.
      expect(() => process.kill(Number(readFileSync(pidFile, "utf8")), 0)).not.toThrow();
    } finally {
      if (existsSync(pidFile)) {
        try {
          process.kill(Number(readFileSync(pidFile, "utf8")), "SIGKILL");
        } catch {
          // Already gone.
        }
      }
    }
  }, 30_000);

  test("a second SIGINT kills a run stuck in a sync body: exit 130, no document", async () => {
    // The first SIGINT ends a CODE body now; only a blocked tool reaches the SIGKILL path.
    const code = "|\n      console.log('ready');\n      while (true) {}";
    const flow = file("flow.yaml", topicFlow(code));
    const log = join(tmp, "run.jsonl");
    const args = ["run", flow, "-q", "--json", "--log", log, "--input", "topic=t"];
    const { child, streams, closed } = spawnEntry(args);
    try {
      for (let i = 0; i < 500 && !streams.err.includes("ready"); i++) {
        await new Promise((resolve) => setTimeout(resolve, 20));
      }
      expect(streams.err).toContain("ready");
      child.kill("SIGINT");
      await new Promise((resolve) => setTimeout(resolve, 100));
      child.kill("SIGINT");
      const timeout = new Promise((resolve) => setTimeout(() => resolve("timed out"), 10_000));
      expect(await Promise.race([closed, timeout])).toEqual({ code: 130, signal: null });
      expect(streams.out).toBe("");
    } finally {
      child.kill("SIGKILL"); // no-op once it has exited
    }
  }, 30_000);
});

describe("the entry's run --json killed without a stop", () => {
  test("SIGKILL of the command stops its run, which never runs on and resumes", async () => {
    const marker = join(tmp, "marker");
    const write = `(await import('node:fs')).writeFileSync(${JSON.stringify(marker)}, 'ran')`;
    const wait = "await new Promise((r) => setTimeout(r, 3000))";
    const code = `|\n      console.log('ready');\n      ${wait};\n      ${write};\n      return 'done';`;
    const flow = file("flow.yaml", topicFlow(code));
    const log = join(tmp, "run.jsonl");
    const args = ["run", flow, "-q", "--json", "--log", log, "--input", "topic=t"];
    const { child, streams, closed } = spawnEntry(args);
    for (let i = 0; i < 500 && !streams.err.includes("ready"); i++) {
      await new Promise((resolve) => setTimeout(resolve, 20));
    }
    expect(streams.err).toContain("ready");
    child.kill("SIGKILL");
    // The run's process shares the command's stderr, so `close` waits for it too.
    const timeout = new Promise((resolve) => setTimeout(() => resolve("timed out"), 10_000));
    expect(await Promise.race([closed, timeout])).toEqual({ code: null, signal: "SIGKILL" });
    expect(streams.out).toBe("");
    expect(inspectRun(log).status).toBe("stopped");
    // Past when the body would have written it.
    await new Promise((resolve) => setTimeout(resolve, 3500));
    expect(existsSync(marker)).toBe(false);
    const resumed = runEntrySync(["run", "--resume", log, "--json"]);
    expect(resumed.status).toBe(0);
    expect(JSON.parse(resumed.stdout)).toMatchObject({ status: "succeeded", output: { topic: "done" } });
  }, 30_000);
});

describe("the entry's run", () => {
  test("one Ctrl-C ends a run stuck in a CODE body (exit 130)", async () => {
    const code = "|\n      console.log('ready');\n      while (true) {}";
    const flow = file("flow.yaml", topicFlow(code));
    const log = join(tmp, "run.jsonl");
    const { child, streams, closed } = spawnEntry(["run", flow, "--log", log, "--input", "topic=t"]);
    try {
      for (let i = 0; i < 500 && !streams.out.includes("ready"); i++) {
        await new Promise((resolve) => setTimeout(resolve, 20));
      }
      expect(streams.out).toContain("ready");
      child.kill("SIGINT");
      const timeout = new Promise((resolve) => setTimeout(() => resolve("timed out"), 10_000));
      expect(await Promise.race([closed, timeout])).toEqual({ code: 130, signal: null });
      expect(streams.err).toContain("cancelled by user");
    } finally {
      child.kill("SIGKILL"); // no-op once it has exited
    }
  }, 30_000);
});
