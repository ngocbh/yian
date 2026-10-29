// Check the built site for what the Astro build doesn't fail on: every YAML block is highlighted
// (at least two token colours), every Mermaid block reached the client renderer, and no
// `check=` option of a fence leaked into a code block's frame (its title or attributes). Mermaid
// renders in the browser, so a diagram that doesn't parse only shows as an error on the page:
// parse every Mermaid block of the pages and of ARCHITECTURE.md with Mermaid's own parser.
import { readdirSync, readFileSync } from "node:fs";
import { join, relative } from "node:path";
import { fileURLToPath } from "node:url";

import DOMPurify from "dompurify";

const dist = fileURLToPath(new URL("../dist/", import.meta.url));
const pages = readdirSync(dist, { recursive: true, encoding: "utf8" }).filter((path) => path.endsWith(".html"));
const problems = [];
let yamlBlocks = 0;
for (const page of pages) {
  const html = readFileSync(join(dist, page), "utf8");
  for (const [, code] of html.matchAll(/<pre data-language="yaml"><code>(.*?)<\/code><\/pre>/gs)) {
    yamlBlocks++;
    const colours = new Set([...code.matchAll(/--0:(#[0-9A-Fa-f]{3,8})/g)].map((m) => m[1]));
    if (colours.size < 2) problems.push(`${page}: a YAML block is not highlighted`);
  }
  if (/<pre[^>]*data-language="mermaid"/.test(html)) problems.push(`${page}: a mermaid block was rendered as code`);
  // A code block's frame is everything from its <figure> to its <code>: prose may say `check=`.
  for (const [frame] of html.matchAll(/<figure class="frame[^"]*"[^>]*>.*?<code>/gs)) {
    if (frame.includes("check=")) problems.push(`${page}: a fence's check= option shows on the page`);
  }
}
if (yamlBlocks === 0) problems.push("no YAML block found in the built site");

// Without a DOM, DOMPurify has no hooks and can't sanitize; parsing needs neither, so stub what
// mermaid calls before loading it. package.json declares dompurify in mermaid's range, so the
// install holds one copy: this one is the instance mermaid imports.
Object.assign(DOMPurify, { addHook() {}, removeHook() {}, removeHooks() {}, sanitize: (text) => text });
const { default: mermaid } = await import("mermaid");
const repo = fileURLToPath(new URL("../../", import.meta.url));
const content = fileURLToPath(new URL("../src/content/docs/", import.meta.url));
const sources = [
  ...readdirSync(content, { recursive: true, encoding: "utf8" })
    .filter((path) => /\.mdx?$/.test(path))
    .map((path) => join(content, path)),
  join(repo, "ARCHITECTURE.md"),
];
let diagrams = 0;
for (const source of sources) {
  const text = readFileSync(source, "utf8");
  for (const block of text.matchAll(/^( *)```mermaid[^\n]*\n([\s\S]*?)^ *```\s*$/gm)) {
    diagrams++;
    const line = text.slice(0, block.index).split("\n").length;
    try {
      await mermaid.parse(block[2]);
    } catch (error) {
      problems.push(`${relative(repo, source)}:${line}: a Mermaid block doesn't parse: ${error.message ?? error}`);
    }
  }
}
if (diagrams === 0) problems.push("no Mermaid block found in the docs");
if (problems.length > 0) {
  console.error(problems.join("\n"));
  process.exit(1);
}
console.log(`check-build: ${pages.length} pages, ${yamlBlocks} YAML blocks highlighted, ${diagrams} Mermaid blocks parsed`);
