/**
 * `yian run` starts the MCP servers a flow needs from `~/.yian/mcp.json` and `--mcp-config`, over
 * the fake server in `tests/fixtures/mcp/`, and a resume opens the servers of the flow the log
 * recorded. Each run waits for every recorded server pid to exit.
 */

import { afterAll, afterEach, beforeEach, expect, test } from "bun:test";
import { chmodSync, existsSync, mkdirSync, mkdtempSync, readdirSync, realpathSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { PassThrough } from "node:stream";

import { main } from "../../src/cli/index.ts";
import { loadMcpConfig } from "../../src/cli/mcp_config.ts";
import { openMcpServers } from "../../src/cli/run.ts";
import type { CliIo } from "../../src/cli/utils.ts";
import { loadFlow } from "../../src/compose/loader.ts";
import { loadSkills } from "../../src/compose/skills.ts";
import { runFlow } from "../../src/run.ts";
import { FunctionTool, tool } from "../../src/tools/function_tools.ts";
import { McpSdkMissingError, McpServer, mcpServer } from "../../src/tools/mcp.ts";
import { TOOL_REGISTRY } from "../../src/tools/registry.ts";
import { withAppendHook } from "../helpers/append_hook.ts";
import { startHttpServer } from "../fixtures/mcp/fake_server.ts";
import { childAlive, killMarkers, pids, stdioOptions, strayProcesses, waitFor } from "../helpers/mcp_helpers.ts";
import { crashAfter, NO_ENV } from "../helpers/run_state_fixtures.ts";

// Each row starts server processes; the default 5 s is too tight on a loaded machine.
const SLOW = 60_000;

const AGENT_FLOW = `id: mcp_agent
name: mcp_agent
nodes:
  worker:
    kind: agent
    prompt: go
    tools: [fake]
output: \${worker.output}
`;

const PLAIN_FLOW = `id: plain
name: plain
nodes:
  pick:
    kind: code
    code: "return 'plain'"
    output: str
output: \${pick.output}
`;

let tmp: string;
let home: string;
let runs: number;
const markers: string[] = [];

beforeEach(() => {
  tmp = realpathSync(mkdtempSync(join(tmpdir(), "yian-cli-run-mcp-")));
  home = join(tmp, "home");
  mkdirSync(home);
  runs = 0;
});

afterEach(() => {
  killMarkers(markers.splice(0));
  rmSync(tmp, { recursive: true, force: true });
});

afterAll(() => {
  expect(strayProcesses()).toEqual([]);
});

/** A fresh marker file the fake server appends its pid to; `afterEach` kills what it lists. */
function marker(name = "pid"): string {
  const file = join(tmp, name);
  markers.push(file);
  return file;
}

/** Write a flow whose one TOOL node calls `toolId` with `text: "hi"`. */
function toolFlow(toolId = "echo"): string {
  const flow = join(tmp, "flow.yaml");
  writeFileSync(
    flow,
    `id: mcp_tool\nname: mcp_tool\nnodes:\n  call:\n    kind: tool\n    tool_id: ${toolId}\n` +
      '    input: {text: "hi"}\noutput: ${call.output}\n',
  );
  return flow;
}

/** Write `text` as a flow file; return its path. */
function flowFile(text: string): string {
  const flow = join(tmp, "flow.yaml");
  writeFileSync(flow, text);
  return flow;
}

/** A config entry that starts the fake server over stdio with `flags`. */
function stdio(...flags: string[]): { command: string; args: string[] } {
  return stdioOptions(...flags);
}

/** Write `servers` as an `mcpServers` file at `path`; return the path. */
function config(path: string, servers: Record<string, unknown>): string {
  mkdirSync(dirname(path), { recursive: true });
  writeFileSync(path, JSON.stringify({ mcpServers: servers }));
  return path;
}

/** Assert every server child recorded in `files` is gone. */
async function expectNoChildren(...files: string[]): Promise<void> {
  for (const file of files) {
    for (const pid of pids(file)) {
      expect(await waitFor(() => !childAlive(pid), 5_000)).toBe(true);
    }
  }
}

/** The last line of `text`, trimmed. */
function lastLine(text: string): string {
  return text.trim().split("\n").at(-1)!;
}

/**
 * Run `yian run <args> --quiet` with `HOME` and `vars` as its env, on a fresh `--log` or, with
 * `resume`, recovering that log; return its code and output.
 */
async function run(args: string[], vars: Record<string, string> = {}, resume?: string) {
  const out: string[] = [];
  const err: string[] = [];
  const env: Record<string, string> = { HOME: home, ...vars };
  const io: CliIo = {
    stdout: { write: async (text) => void out.push(text) },
    stderr: { write: async (text) => void err.push(text) },
    stdin: new PassThrough(),
    stderrStream: new PassThrough(),
    env: (name) => (Object.hasOwn(env, name) ? env[name] : undefined),
  };
  runs += 1;
  const logArgs = resume === undefined ? ["--log", join(tmp, `run-${runs}.jsonl`)] : ["--resume", resume];
  const code = await main(["run", ...args, "--quiet", ...logArgs], io);
  const stdout = out.join("");
  const stderr = err.join("");
  return { code, stdout, stderr, output: stdout + stderr };
}

/** The run's output, its whitespace runs collapsed to one space. */
function flat(text: string): string {
  return text.split(/\s+/).join(" ");
}

test(
  "a tool node calls a server from mcp config",
  async () => {
    const pid = marker();
    const cfg = config(join(tmp, "cfg.json"), { fake: stdio("--marker", pid) });

    const result = await run([toolFlow(), "--mcp-config", cfg]);

    expect(result.code, result.output).toBe(0);
    expect(lastLine(result.stdout)).toBe("hi");
    expect(pids(pid)).toHaveLength(1);
    await expectNoChildren(pid);
  },
  SLOW,
);

test(
  "a tool node calls an http server",
  async () => {
    const http = await startHttpServer();
    let result;
    try {
      const cfg = config(join(tmp, "cfg.json"), { web: { url: http.url } });
      result = await run([toolFlow(), "--mcp-config", cfg]);
    } finally {
      await http.stop();
    }

    expect(result.code, result.output).toBe(0);
    expect(lastLine(result.stdout)).toBe("hi");
  },
  SLOW,
);

test(
  "the user config is read and a flag file replaces its server",
  async () => {
    const flow = toolFlow();
    const pid = marker();
    config(join(home, ".yian", "mcp.json"), { fake: stdio("--marker", pid) });

    const fromUser = await run([flow]);

    expect(fromUser.code, fromUser.output).toBe(0);
    expect(lastLine(fromUser.stdout)).toBe("hi");
    expect(pids(pid)).toHaveLength(1);

    config(join(home, ".yian", "mcp.json"), { fake: { command: join(tmp, "missing") } });
    const cfg = config(join(tmp, "cfg.json"), { fake: stdio("--marker", pid) });

    const replaced = await run([flow, "--mcp-config", cfg]);

    expect(replaced.code, replaced.output).toBe(0);
    expect(lastLine(replaced.stdout)).toBe("hi");
    expect(pids(pid)).toHaveLength(2);
    await expectNoChildren(pid);
  },
  SLOW,
);

test(
  "a flow without tools starts no server",
  async () => {
    const pid = marker();
    const cfg = config(join(tmp, "cfg.json"), { fake: stdio("--marker", pid) });

    const result = await run([flowFile(PLAIN_FLOW), "--mcp-config", cfg]);

    expect(result.code, result.output).toBe(0);
    expect(lastLine(result.stdout)).toBe("plain");
    expect(existsSync(pid)).toBe(false);
  },
  SLOW,
);

test(
  "unused servers with colliding tools dont fail the run",
  async () => {
    const files = { a: marker("a.pid"), b: marker("b.pid"), c: marker("c.pid") };
    const cfg = config(join(tmp, "cfg.json"), {
      a: stdio("--only", "add", "--marker", files.a),
      b: stdio("--only", "add", "--marker", files.b),
      c: stdio("--only", "echo", "--marker", files.c),
    });

    const result = await run([toolFlow(), "--mcp-config", cfg]);

    expect(result.code, result.output).toBe(0);
    expect(lastLine(result.stdout)).toBe("hi");
    // No server is named `echo`, so every server opened to find it; a and b then closed.
    expect(Object.values(files).map((file) => pids(file).length)).toEqual([1, 1, 1]);
    await expectNoChildren(...Object.values(files));
  },
  SLOW,
);

test(
  "probing stops at the first server that supplies the tool",
  async () => {
    const files = { none: marker("none.pid"), a: marker("a.pid"), b: marker("b.pid") };
    const cfg = config(join(tmp, "cfg.json"), {
      none: stdio("--only", "add", "--marker", files.none),
      a: stdio("--marker", files.a),
      b: stdio("--marker", files.b),
    });

    const result = await run([toolFlow(), "--mcp-config", cfg]);

    expect(result.code, result.output).toBe(0);
    expect(lastLine(result.stdout)).toBe("hi");
    // `a` supplies `echo`, so `b`, which lists it too, never starts.
    expect(Object.values(files).map((file) => pids(file).length)).toEqual([1, 1, 0]);
    await expectNoChildren(...Object.values(files));
  },
  SLOW,
);

test(
  "a tool id named like a server still probes for the tool",
  async () => {
    const files = { echo: marker("echo.pid"), fake: marker("fake.pid") };
    const cfg = config(join(tmp, "cfg.json"), {
      echo: stdio("--only", "add", "--marker", files.echo),
      fake: stdio("--only", "echo", "--marker", files.fake),
    });

    const result = await run([toolFlow(), "--mcp-config", cfg]);

    expect(result.code, result.output).toBe(0);
    expect(lastLine(result.stdout)).toBe("hi");
    // A TOOL node can't call a group, so server `echo` opened only to show its tools.
    expect(Object.values(files).map((file) => pids(file).length)).toEqual([1, 1]);
    await expectNoChildren(...Object.values(files));
  },
  SLOW,
);

test(
  "a tool node naming a server's tool as {server: tool} starts that server and no other",
  async () => {
    const files = { other: marker("other.pid"), fake: marker("fake.pid") };
    // Listed first and supplying `echo` too, so probing for a bare `echo` would pick it.
    const cfg = config(join(tmp, "cfg.json"), {
      other: stdio("--only", "echo", "--marker", files.other),
      fake: stdio("--marker", files.fake),
    });

    const result = await run([toolFlow("{fake: echo}"), "--mcp-config", cfg]);

    expect(result.code, result.output).toBe(0);
    expect(lastLine(result.stdout)).toBe("hi");
    expect(pids(files.fake)).toHaveLength(1);
    expect(existsSync(files.other)).toBe(false);
    await expectNoChildren(files.fake);
  },
  SLOW,
);

test(
  "a granted server that supplies the missing tool skips probing",
  async () => {
    const fakePid = marker("fake.pid");
    const otherPid = marker("other.pid");
    const cfg = config(join(tmp, "cfg.json"), {
      fake: stdio("--marker", fakePid),
      broken: { command: join(tmp, "missing") },
      other: stdio("--marker", otherPid),
    });
    const servers = loadMcpConfig([cfg], () => undefined);
    const warned: string[] = [];
    const signal = new AbortController().signal;
    const warn = (message: string) => void warned.push(message);

    const names = await openMcpServers(
      servers,
      new Set(["echo"]),
      new Set(["fake"]),
      new Set(),
      { signal, warn },
      async (opened) => Object.keys(opened),
    );

    expect(names).toEqual(["fake"]);
    expect(servers.fake!.isOpen).toBe(false);
    expect(pids(fakePid)).toHaveLength(1);
    expect(existsSync(otherPid)).toBe(false);
    expect(warned).toEqual([]);
    await expectNoChildren(fakePid);
  },
  SLOW,
);

/** A one-word ("done") chat-completions stream. */
const DONE =
  [
    { choices: [{ index: 0, delta: { role: "assistant", content: "done" }, finish_reason: null }] },
    { choices: [{ index: 0, delta: {}, finish_reason: "stop" }] },
  ]
    .map((chunk) => `data: ${JSON.stringify(chunk)}\n\n`)
    .join("") + "data: [DONE]\n\n";

test.each(["fake", "{fake: [echo]}"])(
  "an agent grant starts only the granted server: tools: [%s]",
  async (grant) => {
    const bound: string[][] = [];
    const fakeFetch = async (input: string | URL | Request, init?: RequestInit) => {
      type Body = { tools?: { function: { name: string } }[] };
      const body = (await new Request(input as string, init).json()) as Body;
      bound.push((body.tools ?? []).map((tool) => tool.function.name));
      return new Response(DONE, { headers: { "content-type": "text/event-stream" } });
    };
    const fakePid = marker("fake.pid");
    const otherPid = marker("other.pid");
    const cfg = config(join(tmp, "cfg.json"), {
      fake: stdio("--marker", fakePid),
      other: stdio("--marker", otherPid),
    });
    const vars = { XAI_API_KEY: "test-xai-key", YIAN_DEFAULT_PROVIDER: "xai", YIAN_DEFAULT_MODEL: "grok-4.3" };
    const send = globalThis.fetch;
    globalThis.fetch = fakeFetch as typeof fetch;
    let result;
    try {
      result = await run([flowFile(AGENT_FLOW.replace("tools: [fake]", `tools: [${grant}]`)), "--mcp-config", cfg], vars);
    } finally {
      globalThis.fetch = send;
    }

    expect(result.code, result.output).toBe(0);
    expect(bound[0]).toContain("fake_echo");
    expect(pids(fakePid)).toHaveLength(1);
    expect(existsSync(otherPid)).toBe(false);
    await expectNoChildren(fakePid);
  },
  SLOW,
);

const WHOAMI_FLOW = `id: mcp_resume
name: mcp_resume
nodes:
  call:
    kind: tool
    tool_id: whoami
output: \${call.output}
`;

test(
  "resume re supplies the servers",
  async () => {
    const flow = flowFile(WHOAMI_FLOW);
    const log = join(tmp, "run.jsonl");
    const crash = new Error("simulated crash");
    let frozen = false;
    const whoami = tool({
      name: "whoami",
      description: "Crash the process on the call.",
      parameters: { type: "object", properties: {} },
      execute: () => {
        frozen = true;
        return "never recorded";
      },
    });
    const freeze = () => {
      if (frozen) throw crash;
    };
    const options = { log, tools: { whoami }, runId: "run-5c2e", env: NO_ENV };
    const loaded = await loadFlow(WHOAMI_FLOW);
    await expect(withAppendHook(freeze, () => runFlow(loaded, {}, options))).rejects.toBe(crash);
    const pid = marker();
    const cfg = config(join(tmp, "cfg.json"), { fake: stdio("--marker", pid) });

    const result = await run(["--mcp-config", cfg], {}, log);

    expect(result.code, result.output).toBe(0);
    expect(lastLine(result.stdout)).toStartWith("run-5c2e/call ");
    expect(pids(pid)).toHaveLength(1);
    await expectNoChildren(pid);
  },
  SLOW,
);

test(
  "resuming a finished run starts no granted server",
  async () => {
    const flow = flowFile(AGENT_FLOW);
    const log = join(tmp, "run.jsonl");
    const vars = { XAI_API_KEY: "test-xai-key", YIAN_DEFAULT_PROVIDER: "xai", YIAN_DEFAULT_MODEL: "grok-4.3" };
    const echo = tool({
      name: "echo",
      description: "Return `text`.",
      parameters: { type: "object", properties: { text: { type: "string" } }, required: ["text"] },
      execute: ({ text }) => text,
    });
    const send = globalThis.fetch;
    globalThis.fetch = (async () =>
      new Response(DONE, { headers: { "content-type": "text/event-stream" } })) as unknown as typeof fetch;
    try {
      const env = (name: string) => (Object.hasOwn(vars, name) ? vars[name as keyof typeof vars] : NO_ENV(name));
      const finished = await runFlow(await loadFlow(AGENT_FLOW), {}, { log, tools: { fake: [echo] }, env });
      expect(finished.status, String(finished.failure?.message)).toBe("succeeded");
    } finally {
      globalThis.fetch = send;
    }
    const pid = marker();
    const cfg = config(join(tmp, "cfg.json"), { fake: stdio("--marker", pid) });

    const result = await run(["--mcp-config", cfg], vars, log);

    expect(result.code, result.output).toBe(0);
    expect(lastLine(result.stdout)).toBe("done");
    expect(existsSync(pid)).toBe(false);
  },
  SLOW,
);

/** A flow whose AGENT grants the tool `tool` and the skill `skill`. */
const grantingFlow = (tool: string, skill: string) => `id: mcp_skill_resume
name: mcp_skill_resume
nodes:
  worker:
    kind: agent
    prompt: go
    tools: [${tool}]
    skills: [${skill}]
output: \${worker.output}
`;

test(
  "a resume runs the recorded flow, not the flow file edited since",
  async () => {
    const vars = { XAI_API_KEY: "test-xai-key", YIAN_DEFAULT_PROVIDER: "xai", YIAN_DEFAULT_MODEL: "grok-4.3" };
    const env = (name: string) => (Object.hasOwn(vars, name) ? vars[name as keyof typeof vars] : NO_ENV(name));
    const skillDir = join(home, ".claude", "skills", "brief");
    mkdirSync(skillDir, { recursive: true });
    writeFileSync(join(skillDir, "SKILL.md"), "---\nname: brief\ndescription: Answer briefly.\n---\nBe brief.\n");
    const skills = loadSkills([join(home, ".claude", "skills")]);
    const flow = flowFile(grantingFlow("echo", "brief"));
    const log = join(tmp, "run.jsonl");
    const pid = marker();
    const cfg = config(join(tmp, "cfg.json"), { fake: stdio("--marker", pid) });
    // Cut after START (the run start's five records and START's three): the run records the
    // server's `echo` spec and the skill, and its AGENT never starts.
    const loaded = await loadFlow(grantingFlow("echo", "brief"), { searchPaths: [tmp], sourceLabel: "flow.yaml" });
    const server = mcpServer(stdio("--marker", pid));
    await crashAfter(8, () => runFlow(loaded, {}, { log, tools: { fake: server }, skills, env }));
    // An edit the resume must not read: a tool no server has, a skill no folder holds.
    writeFileSync(flow, grantingFlow("nowhere", "missing-skill"));

    const refused = await run([flow, "--mcp-config", cfg], vars, log);
    const send = globalThis.fetch;
    globalThis.fetch = (async () =>
      new Response(DONE, { headers: { "content-type": "text/event-stream" } })) as unknown as typeof fetch;
    let result;
    try {
      result = await run(["--mcp-config", cfg], vars, log);
    } finally {
      globalThis.fetch = send;
    }

    expect(refused.code).toBe(2);
    expect(refused.stderr).toContain("--resume runs the flow recorded in the log; leave out <flow>");
    expect(result.code, result.output).toBe(0);
    expect(lastLine(result.stdout)).toBe("done");
    // The cut run and the resume each started the recorded flow's server.
    expect(pids(pid)).toHaveLength(2);
    await expectNoChildren(pid);
  },
  SLOW,
);

test.skipIf(process.platform === "win32")(
  "a user config others may write is refused before any server starts",
  async () => {
    const pid = marker();
    const user = config(join(home, ".yian", "mcp.json"), { fake: stdio("--marker", pid) });
    chmodSync(user, 0o664);

    const result = await run([toolFlow()]);

    expect(result.code).toBe(1);
    expect(flat(result.output)).toContain(`${user}: the MCP config can be changed by other users`);
    expect(existsSync(pid)).toBe(false);
  },
  SLOW,
);

test("a bad config exits with its message", async () => {
  const cfg = join(tmp, "cfg.json");
  writeFileSync(cfg, "{not json");

  const result = await run([toolFlow(), "--mcp-config", cfg]);

  expect(result.code).toBe(1);
  expect(result.output).toContain("the MCP config is not valid JSON");
});

test(
  "a needed server that fails to open exits naming it",
  async () => {
    const cfg = config(join(tmp, "cfg.json"), { broken: { command: join(tmp, "missing") } });

    const result = await run([toolFlow(), "--mcp-config", cfg]);

    expect(result.code).toBe(1);
    expect(result.output).toContain("MCP server 'broken'");
    expect(flat(result.output)).toContain('the flow needs ["echo"], which no started server supplies');
  },
  SLOW,
);

test(
  "a fresh run on the default log path that a server keeps from starting leaves no run folder",
  async () => {
    const cfg = config(join(tmp, "cfg.json"), { broken: { command: join(tmp, "missing") } });
    const err: string[] = [];
    const code = await main(["run", toolFlow(), "--mcp-config", cfg, "--quiet"], {
      stdout: { write: async () => {} },
      stderr: { write: async (text) => void err.push(text) },
      stdin: new PassThrough(),
      stderrStream: new PassThrough(),
      env: (name) => (name === "HOME" ? home : undefined),
    });

    expect(code).toBe(1);
    expect(err.join("")).toContain("MCP server 'broken'");
    const runsDir = join(home, ".yian", "runs");
    expect(readdirSync(runsDir).filter((name) => !name.startsWith("index.sqlite"))).toEqual([]);
  },
  SLOW,
);

test(
  "the --json document of a needed server that fails to open keeps the error's notes",
  async () => {
    const cfg = config(join(tmp, "cfg.json"), {
      broken: { command: join(tmp, "missing") },
      gone: { command: join(tmp, "also-missing") },
    });

    const result = await run([toolFlow(), "--mcp-config", cfg, "--json"]);

    expect(result.code).toBe(1);
    const { error } = JSON.parse(result.stdout) as { error: { code: string; message: string; notes: string[] } };
    expect(error.code).toBe("run.setup");
    expect(error.message).toContain("MCP server 'broken'");
    expect(error.notes).toHaveLength(2);
    expect(error.notes[0]).toBe('the flow needs ["echo"], which no started server supplies');
    expect(error.notes[1]).toContain("MCP server 'gone'");
  },
  SLOW,
);

test(
  "a broken server the flow doesnt need is a warning",
  async () => {
    const pid = marker();
    const cfg = config(join(tmp, "cfg.json"), {
      broken: { command: join(tmp, "missing") },
      c: stdio("--only", "echo", "--marker", pid),
    });

    const result = await run([toolFlow(), "--mcp-config", cfg]);

    expect(result.code, result.output).toBe(0);
    expect(lastLine(result.stdout)).toBe("hi");
    const stderr = flat(result.stderr);
    expect(stderr).toContain("warning: MCP server 'broken'");
    expect(stderr).toContain("; skipped, the flow doesn't need it");
    expect(pids(pid)).toHaveLength(1);
    await expectNoChildren(pid);
  },
  SLOW,
);

test("a flow that needs no mcp tool reads no config", async () => {
  const flow = flowFile(PLAIN_FLOW);
  mkdirSync(join(home, ".yian"));
  writeFileSync(join(home, ".yian", "mcp.json"), "{not json");
  const cfg = join(tmp, "cfg.json");
  writeFileSync(cfg, "{not json");

  const fromUser = await run([flow]);
  const withFlag = await run([flow, "--mcp-config", cfg]);

  expect(fromUser.code, fromUser.output).toBe(0);
  expect(lastLine(fromUser.stdout)).toBe("plain");
  expect(withFlag.code, withFlag.output).toBe(0);
});

test(
  "config warnings print as warning lines",
  async () => {
    const cfg = config(join(tmp, "cfg.json"), { fake: { ...stdio(), timeout: 5 } });

    const result = await run([toolFlow(), "--mcp-config", cfg]);

    expect(result.code, result.output).toBe(0);
    const stderr = flat(result.stderr);
    expect(stderr).toContain("warning: ");
    expect(stderr).toContain("server 'fake': ignoring unknown key 'timeout'");
  },
  SLOW,
);

test("a stop while servers start opens no more", async () => {
  const files = { a: marker("a.pid"), b: marker("b.pid") };
  const cfg = config(join(tmp, "cfg.json"), { a: stdio("--marker", files.a), b: stdio("--marker", files.b) });
  const servers = loadMcpConfig([cfg], () => undefined);
  const stop = new AbortController();
  stop.abort();
  const warned: string[] = [];
  const warn = (message: string) => void warned.push(message);

  const opened = await openMcpServers(
    servers,
    new Set(["echo"]),
    new Set(),
    new Set(),
    { signal: stop.signal, warn },
    async (selected) => selected,
  );

  expect(opened).toEqual({});
  expect(Object.values(files).some((file) => existsSync(file))).toBe(false);
  expect(warned).toEqual([]);
});

test("a server named like a registered tool exits", async () => {
  const shadowed = new FunctionTool({
    name: "shadowed",
    description: "A registered tool a server may not shadow.",
    parametersJsonSchema: { type: "object", properties: {} },
    execute: () => "shadowed",
  });
  TOOL_REGISTRY.set("shadowed", shadowed);
  const pid = marker();
  const cfg = config(join(tmp, "cfg.json"), { shadowed: stdio("--marker", pid) });
  let result;
  try {
    result = await run([toolFlow(), "--mcp-config", cfg]);
  } finally {
    TOOL_REGISTRY.delete("shadowed");
  }

  expect(result.code).toBe(1);
  expect(result.output).toContain(
    `${cfg}: server 'shadowed': the name is reserved (a registered tool or a yian built-in: take, sleep, ask, ` +
      "ask_user, load_skill)",
  );
  expect(existsSync(pid)).toBe(false);
});

test("a server named ask user exits", async () => {
  const cfg = config(join(tmp, "cfg.json"), { ask_user: stdio() });

  const result = await run([toolFlow(), "--mcp-config", cfg]);

  expect(result.code).toBe(1);
  expect(result.output).toContain("'ask_user'");
});

test("a server named after a built-in exits", async () => {
  const cfg = config(join(tmp, "cfg.json"), { take: stdio() });

  const result = await run([toolFlow(), "--mcp-config", cfg]);

  expect(result.code).toBe(1);
  expect(result.output).toContain(`${cfg}: server 'take': the name is reserved`);
});

test("a later file disables a user server", async () => {
  const pid = marker();
  config(join(home, ".yian", "mcp.json"), { fake: stdio("--marker", pid) });
  const cfg = config(join(tmp, "cfg.json"), { fake: { disabled: true } });

  const result = await run([toolFlow(), "--mcp-config", cfg]);

  expect(result.code).toBe(1);
  expect(flat(result.output)).toContain("tool 'echo', which isn't in this run's tools");
  expect(existsSync(pid)).toBe(false);
});

test("only the missing mcp sdk import error is a host error", async () => {
  const cfg = config(join(tmp, "cfg.json"), { fake: stdio() });
  const flow = toolFlow();
  const open = McpServer.prototype.open;
  const failWith = (error: Error) => {
    McpServer.prototype.open = async () => {
      throw error;
    };
  };
  const SDK = "MCP servers need the MCP SDK; install @modelcontextprotocol/client 2.3.1";
  let reported;
  let other;
  let worded;
  let json;
  try {
    failWith(new McpSdkMissingError(SDK));
    reported = await run([flow, "--mcp-config", cfg]);
    failWith(new Error("no module named 'x'"));
    other = await run([flow, "--mcp-config", cfg]);
    json = await run([flow, "--mcp-config", cfg, "--json"]);
    // Known by its class, not its message.
    failWith(new Error(SDK));
    worded = await run([flow, "--mcp-config", cfg]);
  } finally {
    McpServer.prototype.open = open;
  }

  expect(reported.code).toBe(1);
  expect(reported.output).toContain("run failed: MCP servers need the MCP SDK");
  // Any other error is a bug: `main` prints its stack, not a `run failed:` line, and exits 70.
  for (const bug of [other, worded, json]) {
    expect(bug.code).toBe(70);
    expect(bug.stderr).toContain("\n    at ");
    expect(bug.stderr).not.toContain("run failed:");
  }
  expect(other.stderr).toContain("Error: no module named 'x'\n    at ");
  // So a host reading --json tells a crash, with no document, from a reported failure.
  expect(json.stdout).toBe("");
});
