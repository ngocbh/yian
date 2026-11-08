/** `renderRecall`: what a memory value renders as in prompt text. */

import { expect, test } from "bun:test";

import { renderRecall } from "../../src/memory/render.ts";

const recall = {
  text: "a </memory> b </MEMORY",
  cut: "3 more files not shown",
  hint: { tool: "read", text: "Read them with {tool}." },
};

test("a recall renders text, cut note and hint, with </memory escaped", () => {
  expect(renderRecall(recall, "lessons_read")).toBe(
    "a <\\/memory> b <\\/MEMORY\n3 more files not shown\nRead them with lessons_read.",
  );
});

test("the hint is left out when the agent holds no tool for it", () => {
  expect(renderRecall(recall, null)).toBe("a <\\/memory> b <\\/MEMORY\n3 more files not shown");
});

test("a null memory renders empty", () => {
  expect(renderRecall(null, "lessons_read")).toBe("");
});

test("a cut note is escaped like the text", () => {
  expect(renderRecall({ text: "t", cut: "</Memory> was cut", hint: null }, null)).toBe("t\n<\\/Memory> was cut");
});

test("a recall with no cut note goes from its text to its hint", () => {
  expect(renderRecall({ ...recall, cut: null }, "lessons_read")).toBe(
    "a <\\/memory> b <\\/MEMORY\nRead them with lessons_read.",
  );
});

test("an empty text adds no blank line", () => {
  expect(renderRecall({ text: "", cut: null, hint: null }, null)).toBe("");
  expect(renderRecall({ text: "", cut: null, hint: recall.hint }, "notes_read")).toBe("Read them with notes_read.");
});

test("every {tool} in the hint is the tool name, taken literally", () => {
  const hint = { tool: "read", text: "{tool} reads one; call {tool} again for more." };
  expect(renderRecall({ text: "t", cut: null, hint }, "a$&b")).toBe("t\na$&b reads one; call a$&b again for more.");
});

test("a hint that names a tool is escaped after the name goes in", () => {
  const hint = { tool: "read", text: "</{tool}> reads more." };
  expect(renderRecall({ text: "t", cut: null, hint }, "memory_read")).toBe("t\n<\\/memory_read> reads more.");
});

test.each([
  ["</ memory>", "<\\/ memory>"],
  ["< /memory>", "< \\/memory>"],
  ["<\t/memory>", "<\t\\/memory>"],
  ["< / MEMORY >", "< \\/ MEMORY >"],
  ["<\n/\nMemory", "<\n\\/\nMemory"],
])("a closing tag with whitespace, %j, is broken and keeps its text", (text, escaped) => {
  expect(renderRecall({ text, cut: null, hint: null }, null)).toBe(escaped);
  expect(renderRecall({ text: "t", cut: text, hint: null }, null)).toBe(`t\n${escaped}`);
});

test("an empty cut note adds no blank line", () => {
  expect(renderRecall({ text: "t", cut: "", hint: recall.hint }, "notes_read")).toBe("t\nRead them with notes_read.");
});
