// @ts-check
import starlight from "@astrojs/starlight";
import { defineConfig } from "astro/config";
import mermaid from "astro-mermaid";
import starlightLinksValidator from "starlight-links-validator";

// Read the Docs gives each build its canonical URL; a local build uses the published site's.
const canonical = new URL(process.env.READTHEDOCS_CANONICAL_URL ?? "https://yian.readthedocs.io/");
// Pages link to each other by absolute path (`/types/`), so the site must be served at the root:
// the Read the Docs project is single-version. A pull request preview is served under its own
// path while the project is multi-version, so it builds with that base instead.
const preview = process.env.READTHEDOCS_VERSION_TYPE === "external";
if (canonical.pathname !== "/" && !preview) {
  throw new Error(`The docs are served at ${canonical.pathname}, not /: make the Read the Docs project single-version.`);
}

/**
 * An integration that puts `base` in front of each root-absolute Markdown link (`/types/`), so a
 * preview served under a path links within itself. It edits the Markdown tree, before the link
 * check reads the HTML tree.
 */
function prefixBase(base) {
  const prefix = (node, ctx) => {
    if (/^\/(?!\/)/.test(node.url)) ctx.setProperty(node, "url", base + node.url.slice(1));
  };
  return {
    name: "prefix-base",
    hooks: {
      "astro:config:setup": ({ config }) => {
        config.markdown.processor.options.mdastPlugins.push({ name: "prefix-base", link: prefix, definition: prefix });
      },
    },
  };
}

export default defineConfig({
  site: canonical.origin,
  base: canonical.pathname,
  trailingSlash: "always",
  integrations: [
    ...(canonical.pathname === "/" ? [] : [prefixBase(canonical.pathname)]),
    // Before Starlight, so it sees `mermaid` blocks before Expressive Code does.
    mermaid({ autoTheme: true }),
    starlight({
      title: "Yian - The Agent Composer",
      description: "Readable YAML flows of agents, code and tools that people and LLMs write, check and improve.",
      plugins: [starlightLinksValidator({ errorOnLocalLinks: true })],
      sidebar: [
        { label: "Home", slug: "index" },
        { label: "Get started", slug: "get-started" },
        {
          label: "Write flows",
          items: ["flows", "expressions", "types", "agents", "code-and-tools", "models", "control-flow", "inboxes"],
        },
        { label: "Run and embed", items: ["cli", "embedding"] },
        { label: "Understand", items: ["how-it-works"] },
        { label: "Look up", items: ["reference", "examples"] },
      ],
    }),
  ],
});
