/**
 * `yian run` supplies Claude Code and Codex skill folders to the run; the system prompt is read
 * from the request a fake `fetch` model gets.
 */

import { afterEach, beforeEach, expect, test } from "bun:test";
import { mkdirSync, mkdtempSync, realpathSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { PassThrough } from "node:stream";

import { main } from "../../src/cli/index.ts";
import type { CliIo } from "../../src/cli/utils.ts";

const FLOW = `id: cli-skills
name: cli-skills
nodes:
  review:
    kind: agent
    prompt: Review.
    skills: [code-review, release-notes]
    output: !str
output: \${review.output}
`;

let tmp: string;
let home: string;
// Restores the cwd and `fetch` even when a test times out, before `afterEach` removes `tmp`.
let restore: (() => void) | null = null;

beforeEach(() => {
  tmp = realpathSync(mkdtempSync(join(tmpdir(), "yian-cli-skills-")));
  home = join(tmp, "home");
  mkdirSync(home);
});

afterEach(() => {
  restore?.();
  restore = null;
  rmSync(tmp, { recursive: true, force: true });
});

/** A one-word ("done") chat-completions stream. */
const DONE = [
  { choices: [{ index: 0, delta: { role: "assistant", content: "done" }, finish_reason: null }] },
  { choices: [{ index: 0, delta: {}, finish_reason: "stop" }] },
]
  .map((chunk) => `data: ${JSON.stringify(chunk)}\n\n`)
  .join("") + "data: [DONE]\n\n";

/** Each system prompt the fake model was sent. */
let systems: string[];

/** Answer every chat-completions request with `DONE`, recording its system prompt. */
async function fakeFetch(input: string | URL | Request, init?: RequestInit): Promise<Response> {
  const request = new Request(input as string, init);
  const body = (await request.json()) as { messages: { role: string; content: unknown }[] };
  const system = body.messages.find((message) => message.role === "system");
  systems.push(typeof system?.content === "string" ? system.content : JSON.stringify(system?.content));
  return new Response(DONE, { headers: { "content-type": "text/event-stream" } });
}

/** Write a skill `name` under `root` with `description`. */
function skill(root: string, name: string, description: string): void {
  mkdirSync(join(root, name), { recursive: true });
  writeFileSync(join(root, name, "SKILL.md"), `---\ndescription: ${description}\n---\nBody.\n`);
}

/** Run `yian run <flow> --quiet` from `cwd` with a fake model; return its code and stderr. */
async function runIn(cwd: string, flow: string) {
  const chunks: string[] = [];
  const vars: Record<string, string> = {
    HOME: home,
    XAI_API_KEY: "test-xai-key",
    YIAN_DEFAULT_PROVIDER: "xai",
    YIAN_DEFAULT_MODEL: "grok-4.3",
  };
  const io: CliIo = {
    stdout: { write: async () => {} },
    stderr: { write: async (text) => void chunks.push(text) },
    stdin: new PassThrough(),
    stderrStream: new PassThrough(),
    env: (name) => (Object.hasOwn(vars, name) ? vars[name] : undefined),
  };
  const before = process.cwd();
  const send = globalThis.fetch;
  systems = [];
  process.chdir(cwd);
  globalThis.fetch = fakeFetch as typeof fetch;
  restore = () => {
    globalThis.fetch = send;
    process.chdir(before);
  };
  const code = await main(["run", flow, "--quiet", "--log", join(tmp, "run.jsonl")], io);
  return { code, stderr: chunks.join("") };
}

test("run reads user and project skill roots", async () => {
  const project = join(tmp, "project");
  mkdirSync(join(project, ".git"), { recursive: true });
  skill(join(project, ".claude", "skills"), "code-review", "Project review.");
  skill(join(project, ".agents", "skills"), "release-notes", "Project notes.");
  skill(join(home, ".claude", "skills"), "code-review", "Personal review.");
  mkdirSync(join(project, ".claude", "skills", "broken"));
  writeFileSync(join(project, ".claude", "skills", "broken", "SKILL.md"), "no frontmatter\n");
  const flow = join(project, "flow.yaml");
  writeFileSync(flow, FLOW);

  const { code, stderr } = await runIn(project, flow);

  expect(code, stderr).toBe(0);
  expect(stderr).toContain("warning: skipped skill: ");
  expect(stderr).toContain("broken");
  expect(systems[0]).toEndWith("- code-review: Personal review.\n- release-notes: Project notes.");
});

test("run reports a missing granted skill", async () => {
  mkdirSync(join(tmp, ".git"));
  const flow = join(tmp, "flow.yaml");
  writeFileSync(flow, FLOW);

  const { code, stderr } = await runIn(tmp, flow);

  expect(code).toBe(1);
  expect(stderr).toContain("grants skills not supplied to this run");
});

test("run without grants reads no skill roots", async () => {
  mkdirSync(join(tmp, ".git"));
  mkdirSync(join(tmp, ".claude", "skills", "broken"), { recursive: true });
  writeFileSync(join(tmp, ".claude", "skills", "broken", "SKILL.md"), "no frontmatter\n");
  const flow = join(tmp, "flow.yaml");
  writeFileSync(flow, FLOW.replace("    skills: [code-review, release-notes]\n", ""));

  const { code, stderr } = await runIn(tmp, flow);

  expect(code, stderr).toBe(0);
  expect(stderr).not.toContain("skipped skill");
});
