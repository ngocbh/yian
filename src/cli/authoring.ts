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
import { FLOWS } from "./bundled.ts";

/** The bundled flows' files, and the alias `author_flow.yaml` uses `review_flow.yaml` by. */
const AUTHOR_FILE = "author_flow.yaml";
const REVIEW_FILE = "review_flow.yaml";
const REVIEW_ALIAS = "review_flow";

/**
 * The chat's tools the author flow is granted, each under the same confinement and approvals as
 * the assistant's own; no other child run gets them. Its `start_flow` and `wait_flow` are its own.
 */
export const AUTHOR_TOOLS: readonly string[] = [
  "check_flow",
  "read_flow",
  "read_docs",
  "write_flow",
  "start_flow",
  "wait_flow",
];

/** The chat's tools the review flow is granted: it only reads. */
export const REVIEW_TOOLS: readonly string[] = ["read_flow", "read_docs"];

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
      if (ref !== REVIEW_ALIAS) throw new RangeError(`no bundled flow ${ref}`);
      return review;
    },
  });
}
