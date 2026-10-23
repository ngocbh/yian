/**
 * TS-only: media as files beside the run. `writeRunFile` stores bytes under the run folder's
 * `media/`, named by their sha256; `inlineToFile` turns a message's inline media into file
 * references; `fileToInline` turns a request's file references back into inline bytes for the
 * provider. The run folder (`dirname` of the log) reaches an AGENT's run context, on a fresh run
 * and on resume.
 */

import { createHash } from "node:crypto";
import { mkdirSync, mkdtempSync, readdirSync, readFileSync, rmSync, statSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { afterAll, afterEach, beforeEach, expect, spyOn, test } from "bun:test";

import { loadFlow } from "../../src/compose/loader.ts";
import { BUILTIN, forget, install } from "../../src/defaults.ts";
import { MediaPart, Message, Request, TextPart, ToolSpec } from "../../src/llm/messages.ts";
import { ScriptedModel } from "../../src/llm/testing.ts";
import { type AgentRunContext, fileToInline, inlineToFile, writeRunFile } from "../../src/nodes/agent/context.ts";
import { AgentNode } from "../../src/nodes/agent/node.ts";
import { resumeFlow, runFlow } from "../../src/run.ts";
import { FileRef } from "../../src/typesys/values.ts";
import { hostEnv } from "./mcp_helpers.ts";
import { crashAfter, serving, textReply } from "./run_state_fixtures.ts";

const root = mkdtempSync(join(tmpdir(), "yian-agent-media-files-"));
afterAll(() => rmSync(root, { recursive: true, force: true }));

beforeEach(() => install({ ...BUILTIN, modelRetries: 0 }));
afterEach(forget);

/** Return a fresh empty folder under the test root. */
function folder(): string {
  return mkdtempSync(join(root, "run-"));
}

/** Return the sha256 of `bytes` in hex. */
function sha256(bytes: Uint8Array): string {
  return createHash("sha256").update(bytes).digest("hex");
}

const PNG = new TextEncoder().encode("not really a png");

// --- writeRunFile --- //

test("writeRunFile writes <dir>/media/<sha256>.<ext> with mode 0600 and returns its FileRef", () => {
  const dir = folder();
  const name = `${sha256(PNG)}.png`;
  const path = join(dir, "media", name);

  const file = writeRunFile(dir, PNG, "image/png");

  expect(file).toEqual(new FileRef({ uri: path, mime: "image/png", name }));
  expect(new Uint8Array(readFileSync(path))).toEqual(PNG);
  expect(statSync(path).mode & 0o777).toBe(0o600);
  expect(statSync(join(dir, "media")).mode & 0o777).toBe(0o700);
});

test("the same bytes twice keep one file", () => {
  const dir = folder();
  const first = writeRunFile(dir, PNG, "image/png");
  expect(writeRunFile(dir, PNG, "image/png")).toEqual(first);
  expect(readdirSync(join(dir, "media"))).toEqual([first.name!]);
});

test("the extension follows the media type; an unknown one is .bin", () => {
  const dir = folder();
  const digest = sha256(PNG);
  expect(writeRunFile(dir, PNG, "image/jpeg").name).toBe(`${digest}.jpg`);
  expect(writeRunFile(dir, PNG, "Image/PNG; foo=bar").name).toBe(`${digest}.png`);
  expect(writeRunFile(dir, PNG, "application/x-mystery").name).toBe(`${digest}.bin`);
});

// --- inlineToFile and fileToInline --- //

const DATA = Buffer.from(PNG).toString("base64");

/** An assistant message holding text, an inline image and an image by URL. */
function reply(): Message {
  return new Message({
    role: "assistant",
    parts: [
      new TextPart("here"),
      new MediaPart({ mime: "image/png", data: DATA, name: "chart.png", providerPayload: { detail: "low" } }),
      new MediaPart({ mime: "image/png", url: "https://example.com/a.png" }),
    ],
    protocol: "openai_responses",
    providerPayload: { id: "resp_1" },
  });
}

test("inlineToFile stores each inline media part and gives its file form; other parts stay", () => {
  const dir = folder();
  const name = `${sha256(PNG)}.png`;

  const filed = inlineToFile(reply(), dir);

  const expected: { [key: string]: any } = reply().toJSON();
  expected["parts"][1] = {
    type: "media",
    mime: "image/png",
    file: { uri: join(dir, "media", name), mime: "image/png", name },
    name: "chart.png",
    provider_payload: { detail: "low" },
  };
  expect(filed.toJSON()).toEqual(expected);
  expect(new Uint8Array(readFileSync(join(dir, "media", name)))).toEqual(PNG);
});

test("inlineToFile returns a message without inline media, or any message without a folder, as is", () => {
  const plain = new Message({ role: "user", parts: [new TextPart("hi")] });
  expect(inlineToFile(plain, folder())).toBe(plain);
  const filed = inlineToFile(reply(), folder());
  expect(inlineToFile(filed, folder())).toBe(filed);
  const inline = reply();
  expect(inlineToFile(inline, null)).toBe(inline);
});

test("inlineToFile refuses inline data that isn't base64, storing nothing", () => {
  const dir = folder();
  for (const data of ["not base64!", "a-b_"]) {
    const message = new Message({ role: "assistant", parts: [new MediaPart({ mime: "image/png", data })] });
    expect(() => inlineToFile(message, dir)).toThrow(new RangeError("image/png media data isn't base64"));
  }
  // Unpadded base64 is still base64.
  const unpadded = new Message({ role: "assistant", parts: [new MediaPart({ mime: "image/png", data: "YWI" })] });
  expect((inlineToFile(unpadded, dir).parts[0] as MediaPart).file).not.toBeNull();
  expect(readdirSync(join(dir, "media"))).toHaveLength(1);
});

test("fileToInline gives each file media part its bytes back as data; other parts stay", async () => {
  const filed = inlineToFile(reply(), folder());
  const tool = new ToolSpec({ name: "noop", description: "Do nothing.", parameters: { type: "object" } });
  const request = new Request({
    messages: [new Message({ role: "user", parts: [new TextPart("draw")] }), filed],
    tools: [tool],
    toolChoice: "auto",
    parallelToolCalls: false,
  });

  const sent = await fileToInline(request);

  expect(sent.messages.map((message) => message.toJSON())).toEqual([
    request.messages[0]!.toJSON(),
    reply().toJSON(),
  ]);
  expect([sent.tools, sent.toolChoice, sent.parallelToolCalls]).toEqual([[tool], "auto", false]);
});

test("fileToInline returns a request without file media as is", async () => {
  const request = new Request({ messages: [reply()] });
  expect(await fileToInline(request)).toBe(request);
});

test("fileToInline rejects naming the path of a missing file", async () => {
  const dir = folder();
  const filed = inlineToFile(reply(), dir);
  const path = join(dir, "media", `${sha256(PNG)}.png`);
  rmSync(path);
  const error = await fileToInline(new Request({ messages: [filed] })).then(
    () => null,
    (reason: unknown) => reason,
  );
  expect(error).toBeInstanceOf(Error);
  expect((error as Error).message).toContain(path);
});

// --- the run folder reaches the AGENT --- //

const FLOW = `\
id: media
name: media
llm_config: {provider: anthropic, model: claude-test}
input:
  message: !str
nodes:
  reply:
    kind: agent
    input: {message: "\${input.message}"}
    prompt: \${message}
    output: str
output: \${reply.output}
`;
// The records before \`reply\`'s success: RunStarted, START's start and success, \`reply\`'s start.
const BEFORE_REPLY_SUCCESS = 4;

/** Run \`fn\` and return the \`runDir\` of each AGENT run context it built, in order. */
async function runDirs(fn: () => Promise<unknown>): Promise<(string | null | undefined)[]> {
  // \`runContext\` is private to the type checker only, so the test can watch what it returns.
  const target = AgentNode.prototype as unknown as { runContext: (...args: unknown[]) => AgentRunContext };
  const spy = spyOn(target, "runContext");
  try {
    await fn();
    return spy.mock.results.map((result) => (result.value as AgentRunContext).runDir);
  } finally {
    spy.mockRestore();
  }
}

/** Run options serving a model that replies each of \`texts\` in turn, with no host credentials. */
function served(...texts: string[]) {
  const model = new ScriptedModel(texts.map(textReply), { provider: "anthropic", model: "claude-test" });
  return { models: serving(model), env: hostEnv(folder()) };
}

test("an AGENT's run folder is the folder of the run's log", async () => {
  const run = join(folder(), "r1");
  mkdirSync(run);
  const log = join(run, "log.jsonl");
  const flow = await loadFlow(FLOW);
  expect(await runDirs(() => runFlow(flow, { message: "hi" }, { log, ...served("ok") }))).toEqual([run]);
});

test("a run without a log gives its AGENT no run folder", async () => {
  const flow = await loadFlow(FLOW);
  expect(await runDirs(() => runFlow(flow, { message: "hi" }, served("ok")))).toEqual([null]);
});

test("a resumed run gives its AGENT the same run folder", async () => {
  const run = join(folder(), "r1");
  mkdirSync(run);
  const log = join(run, "log.jsonl");
  const flow = await loadFlow(FLOW);
  await crashAfter(BEFORE_REPLY_SUCCESS, () => runFlow(flow, { message: "hi" }, { log, ...served("lost") }));
  const dirs = await runDirs(async () => {
    const resumed = await resumeFlow(flow, { log, ...served("ok") });
    expect(resumed.status, resumed.failure?.message).toBe("succeeded");
  });
  expect(dirs).toEqual([run]);
});
