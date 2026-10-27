// @ts-check
import starlight from "@astrojs/starlight";
import { defineConfig } from "astro/config";
import mermaid from "astro-mermaid";
import starlightLinksValidator from "starlight-links-validator";

// Read the Docs gives each build its canonical URL; a local build uses the published site's.
const canonical = new URL(process.env.READTHEDOCS_CANONICAL_URL ?? "https://yian.readthedocs.io/");
// Pages link to each other by absolute path (`/types/`), so the site must be served at the root:
// the Read the Docs project is single-version.
if (canonical.pathname !== "/") {
  throw new Error(`The docs are served at ${canonical.pathname}, not /: make the Read the Docs project single-version.`);
}

export default defineConfig({
  site: canonical.origin,
  base: "/",
  trailingSlash: "always",
  integrations: [
    // Before Starlight, so it sees `mermaid` blocks before Expressive Code does.
    mermaid({ autoTheme: true }),
    starlight({
      title: "yian",
      description: "Readable YAML flows of agents, code and tools that people and LLMs write, check and improve.",
      plugins: [starlightLinksValidator({ errorOnLocalLinks: true })],
      sidebar: [
        { label: "Home", slug: "index" },
        { label: "Get started", slug: "get-started" },
        {
          label: "Write flows",
          items: ["flows", "expressions", "types", "agents", "code-and-tools", "models", "control-flow", "inboxes", "learning"],
        },
        { label: "Run and embed", items: ["cli", "embedding"] },
        { label: "Understand", items: ["how-it-works"] },
        { label: "Look up", items: ["reference", "examples"] },
      ],
    }),
  ],
});
