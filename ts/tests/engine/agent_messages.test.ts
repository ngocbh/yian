/**
 * The built-in `Message` value: its check, its conversion to and from the model message, the
 * public value codec round trip, its agreement with `message.schema.json`, and the conversation
 * check a request needs.
 *
 * TS-only: Python yian has no public message type.
 */
import { describe, expect, test } from "bun:test";

import { checkMessage, type Message } from "../../src/nodes/agent/messages.ts";
import { FileRef, TypeCheckError } from "../../src/typesys/values.ts";

const FILE = { uri: "media/abc.png", mime: "image/png", name: null };

test("checkMessage returns the value with each file's JSON form as a FileRef", () => {
  const checked = checkMessage({ role: "user", parts: [{ type: "media", mime: "image/png", file: FILE }] });
  expect(checked.parts[0]).toEqual({ type: "media", mime: "image/png", file: new FileRef(FILE) });
});

test("checkMessage refuses a value that isn't a Message", () => {
  expect(() => checkMessage({ role: "user", parts: [{ type: "text" }] })).toThrow(TypeCheckError);
});
