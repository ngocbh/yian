// Check the built site for what the Astro build doesn't fail on: every YAML block is highlighted
// (at least two token colours), every Mermaid block reached the client renderer, and no
// `check=` option of a fence leaked into a code block's frame (its title or attributes).
import { readdirSync, readFileSync } from "node:fs";
import { join } from "node:path";

const dist = new URL("../dist/", import.meta.url).pathname;
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
if (problems.length > 0) {
  console.error(problems.join("\n"));
  process.exit(1);
}
console.log(`check-build: ${pages.length} pages, ${yamlBlocks} YAML blocks highlighted`);
