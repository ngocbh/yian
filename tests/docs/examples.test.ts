/**
 * Every example on the docs site is checked against the code: each YAML block loads with the real
 * loader (or fails to, or is only parsed, as its `check=` says), each `check=runnable` block runs on
 * a scripted model and prints the output its page shows, and each `yian` line in a console block
 * is one the CLI's command table takes. `docs/STYLE.md` says how writers mark blocks.
 */

import { expect, test } from "bun:test";
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { fileURLToPath } from "node:url";

import { parseDocument } from "yaml";

import { parseKv } from "../../src/cli/run.ts";
import type { ComposedFlow } from "../../src/compose/build/pipeline.ts";
import { LoadError } from "../../src/compose/helpers/errors.ts";
import { loadFlow } from "../../src/compose/loader.ts";
import { ScriptedModel } from "../../src/llm/testing.ts";
import { runFlow } from "../../src/run.ts";
import { type Value, writeAsText } from "../../src/typesys/values.ts";
import { NO_ENV, serving, textReply } from "../engine/run_state_fixtures.ts";
import { type Block, checkOf, cliProblem, parseYian, readBlocks, readCommands, readDocs } from "./doc_blocks.ts";

const ROOT = fileURLToPath(new URL("../..", import.meta.url));
const DOCS = join(ROOT, "docs", "src", "content", "docs");
const EXAMPLES = join(ROOT, "examples");
// The root README is the npm page, so its examples are checked like a page's.
const BLOCKS = [...readDocs(DOCS), ...readBlocks(readFileSync(join(ROOT, "README.md"), "utf8"), "README.md")];
// A `source=` block shows a file of examples/ whole.
const SOURCE = /^examples\/[\w.-]+\.yaml$/;
const YAML_BLOCKS = BLOCKS.filter((block) => block.lang === "yaml");
const SHELLS = new Set(["sh", "bash", "shell", "zsh"]);
// Any provider and model: the scripted model answers whichever the flow asks for.
const LLM_CONFIG = { provider: "anthropic", model: "claude-test" };

const where = (block: Block) => `${block.file}:${block.line}`;

test("the docs have YAML examples", () => {
  expect(YAML_BLOCKS.length).toBeGreaterThan(0);
});

test("YAML blocks are tagged yaml, not yml", () => {
  expect(BLOCKS.filter((block) => block.lang === "yml").map(where)).toEqual([]);
});

test.each(YAML_BLOCKS.map((block) => [where(block), block] as const))("%s", async (label, block) => {
  const source = block.meta["source"];
  if (source !== undefined) {
    if (!SOURCE.test(source)) throw new Error(`${label}: source=${source} is not examples/<name>.yaml`);
    expect(block.text).toBe(readFileSync(join(ROOT, source), "utf8"));
  }
  const check = checkOf(block);
  if (check === "fragment") {
    expect(parseDocument(block.text).errors.map(String)).toEqual([]);
    return;
  }
  const load = () => loadFlow(block.text, { sourceLabel: label, searchPaths: [EXAMPLES] });
  if (check === "fails") {
    await expect(load()).rejects.toBeInstanceOf(LoadError);
    return;
  }
  const flow = await load();
  if (check === "runnable") await expectShownOutput(block, flow);
});

/**
 * Run `flow` the way the console block after `block` shows (`$ yian run <title> --input k=v ...`),
 * the model answering with the block's `replies` note, and expect the output the page shows.
 */
async function expectShownOutput(block: Block, flow: ComposedFlow): Promise<void> {
  const next = BLOCKS[BLOCKS.indexOf(block) + 1];
  if (next?.file !== block.file || next.lang !== "console") {
    throw new Error(`${where(block)}: a runnable block is followed by a console block`);
  }
  const [shown] = readCommands(next);
  const command = shown && parseYian(shown.argv);
  if (!command || command.key !== "run") throw new Error(`${where(next)}: starts with a $ yian run line`);
  expect(command.parsed.args["flow"]).toBe(block.meta["title"]);
  const replies = (block.note?.["replies"] ?? []) as string[];
  const model = new ScriptedModel(replies.map(textReply));
  const inputs = parseKv(command.parsed.options["input"] as string[]);
  const result = await runFlow(flow, inputs, { env: NO_ENV, numWorkers: 0, models: serving(model), llmConfig: LLM_CONFIG });
  expect(result.failure).toBeNull();
  expect(shown!.output).toBe(writeAsText(result.output as Value));
  expect(model.requests).toHaveLength(replies.length);
}

const YIAN_LINES = BLOCKS.filter((block) => block.lang === "console").flatMap((block) =>
  readCommands(block)
    .filter((command) => command.argv[0] === "yian")
    .map((command) => [`${block.file}:${command.line} ${command.argv.join(" ")}`, command.argv] as const),
);

test.each(YIAN_LINES)("%s", (_label, argv) => {
  expect(cliProblem(argv)).toBeNull();
});

test("yian lines are in console blocks", () => {
  const stray = BLOCKS.filter((block) => SHELLS.has(block.lang)).flatMap((block) =>
    block.text
      .split("\n")
      .filter((line) => /^\s*(\$\s+)?yian\b/.test(line))
      .map((line) => `${where(block)}: ${line}`),
  );
  expect(stray).toEqual([]);
});
