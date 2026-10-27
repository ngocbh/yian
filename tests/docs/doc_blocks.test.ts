/** The docs block reader and the `yian` line check that `examples.test.ts` uses. */

import { expect, test } from "bun:test";

import { checkOf, cliProblem, readBlocks, readCommands, readMeta, shellWords } from "./doc_blocks.ts";

const PAGE = [
  "Intro.",
  "",
  '<!-- yian-test {"replies": ["Hi"]} -->',
  "",
  '```yaml title="hello.yaml" check=runnable',
  "id: hello",
  "```",
  "",
  "1. A step:",
  "",
  "   ~~~~console",
  "   $ yian run hello.yaml \\",
  "       --input name=Ada",
  "   Hi",
  "   ~~~~",
  "",
  '<!-- yian-test {"replies": []} -->',
  "Text between.",
  "```yaml",
  "a: 1",
  "```",
].join("\n");

test("blocks keep their line, language, options, text and note", () => {
  const blocks = readBlocks(PAGE, "p.md");
  expect(blocks.map((b) => [b.line, b.lang, b.meta, b.text, b.note])).toEqual([
    [5, "yaml", { title: "hello.yaml", check: "runnable" }, "id: hello\n", { replies: ["Hi"] }],
    [11, "console", {}, "$ yian run hello.yaml \\\n    --input name=Ada\nHi\n", null],
    [19, "yaml", {}, "a: 1\n", null],
  ]);
});

test("a note that isn't a one-line JSON object is an error", () => {
  expect(() => readBlocks("<!-- yian-test replies: [] -->\n", "p.md")).toThrow("p.md:1: a yian-test note");
  expect(() => readBlocks("<!-- yian-test {}\n-->\n", "p.md")).toThrow("p.md:1: a yian-test note");
});

test("an unclosed fence is an error", () => {
  expect(() => readBlocks("```yaml\na: 1\n", "p.md")).toThrow("p.md:1: unclosed code fence");
});

test("options take quoted and bare values", () => {
  expect(readMeta(`title="a b.yaml" check=fragment x='y'`)).toEqual({ title: "a b.yaml", check: "fragment", x: "y" });
});

test("the check defaults to loads and refuses unknown names", () => {
  const [block] = readBlocks("```yaml\na: 1\n```\n", "p.md");
  expect(checkOf(block!)).toBe("loads");
  expect(() => checkOf({ ...block!, meta: { check: "run" } })).toThrow("check=run is not one of");
});

test("shell words follow quotes and backslashes", () => {
  expect(shellWords(`yian run "my flow.yaml" -i 'q=a b' --input x=\\"y\\"`)).toEqual([
    "yian", "run", "my flow.yaml", "-i", "q=a b", "--input", 'x="y"',
  ]);
  expect(() => shellWords(`yian run "x`)).toThrow("unclosed quote");
});

test("console commands join continued lines and keep their output", () => {
  const [, console_] = readBlocks(PAGE, "p.md");
  expect(readCommands(console_!)).toEqual([
    { line: 12, argv: ["yian", "run", "hello.yaml", "--input", "name=Ada"], output: "Hi" },
  ]);
});

test("the CLI table takes real command lines", () => {
  for (const line of [
    "yian --version",
    "yian run flow.yaml --input name=Ada -i n=2",
    "yian chat chat.yaml",
    "yian check a.yaml b.yaml",
    "yian skills check ./skills",
    "yian config",
  ]) {
    expect([line, cliProblem(shellWords(line))]).toEqual([line, null]);
  }
});

test("the CLI table refuses unknown commands and options", () => {
  expect(cliProblem(shellWords("yian walk flow.yaml"))).toBe("no command 'yian walk'");
  expect(cliProblem(shellWords("yian run flow.yaml --nope"))).toContain("--nope");
  expect(cliProblem(shellWords("yian skills list"))).toBe("no command 'yian skills list'");
});
