/**
 * The bundled flows `yian chat`'s assistant writes and reviews flows with: `author_flow.yaml`,
 * which builds or reworks a flow with the user, and `review_flow.yaml`, which reviews one and
 * which the first calls on each draft. The chat starts each as a child run, granting it the
 * chat's tools `AUTHOR_TOOLS` or `REVIEW_TOOLS`.
 *
 * Knows about: compose and the bundle.
 */

import type { ComposedFlow } from "../compose/build/pipeline.ts";
import { loadFlow } from "../compose/loader.ts";
import type { Env } from "../defaults.ts";
import { YianRangeError } from "../typesys/errors.ts";
import { FLOWS } from "./bundled.ts";

/** The bundled flows' files, and the alias `author_flow.yaml` uses `review_flow.yaml` by. */
const AUTHOR_FILE = "author_flow.yaml";
const REVIEW_FILE = "review_flow.yaml";
const REVIEW_ALIAS = "review_flow";

/** A tool grant as a flow writes it: a tool's name, or `{group: [tool, ...]}`. */
export type AuthoredGrant = string | Readonly<Record<string, readonly string[]>>;

/**
 * The chat's tools the author flow is granted, as it grants them, each under the same confinement
 * and approvals as the assistant's own; no other child run gets them. Its `flows` group is its own:
 * it starts its tries, and waits on them.
 */
export const AUTHOR_TOOLS: readonly AuthoredGrant[] = [
  "check_flow",
  "read_flow",
  "read_docs",
  "write_flow",
  { flows: ["start", "wait"] },
];

/** The chat's tools the review flow is granted: it only reads. */
export const REVIEW_TOOLS: readonly AuthoredGrant[] = ["read_flow", "read_docs"];

/**
 * Load the bundled review flow.
 *
 * @throws {LoadError} Never for the bundle as committed; its test loads it.
 */
export function loadReviewFlow(env: Env): Promise<ComposedFlow> {
  return loadFlow(FLOWS[REVIEW_FILE]!, { sourceLabel: REVIEW_FILE, env });
}

/**
 * Load the bundled author flow, its `uses:` resolved to the bundled review flow.
 *
 * @throws {LoadError} Never for the bundle as committed; its test loads it.
 */
export async function loadAuthorFlow(env: Env): Promise<ComposedFlow> {
  const review = await loadReviewFlow(env);
  return loadFlow(FLOWS[AUTHOR_FILE]!, {
    sourceLabel: AUTHOR_FILE,
    env,
    childResolver: async (ref) => {
      if (ref !== REVIEW_ALIAS) throw new YianRangeError(`no bundled flow ${ref}`);
      return review;
    },
  });
}
