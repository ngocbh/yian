/**
 * TS-only: media as files beside the run. `writeRunFile` stores bytes under the run folder's
 * `media/`, named by their sha256; `inlineToFile` turns a message's inline media into file
 * references; `fileToInline` turns a request's file references back into inline bytes for the
 * provider. The run folder (`dirname` of the log) reaches an AGENT's run context, on a fresh run
 * and on resume.
 */

import { createHash } from "node:crypto";
import { mkdtempSync, readdirSync, readFileSync, rmSync, statSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { afterAll, expect, test } from "bun:test";

import { MediaPart, Message, Request, TextPart, ToolSpec } from "../../src/llm/messages.ts";
import { fileToInline, inlineToFile, writeRunFile } from "../../src/nodes/agent/context.ts";
import { FileRef } from "../../src/typesys/values.ts";

const root = mkdtempSync(join(tmpdir(), "yian-agent-media-files-"));
afterAll(() => rmSync(root, { recursive: true, force: true }));

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
