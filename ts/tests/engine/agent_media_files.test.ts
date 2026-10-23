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

import { writeRunFile } from "../../src/nodes/agent/context.ts";
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
