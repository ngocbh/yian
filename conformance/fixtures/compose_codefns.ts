/**
 * CODE-node functions harvested cases reference: a TypeScript port of yian's
 * `tests/engine/_compose_codefns.py`. Functions are ported as harvested cases reference them.
 */

// --- CALL/MAP child-flow CODE fns (Ollama-free subflows) ---
/** Return the bound `topic` unchanged (MAP order/identity checks). */
export function echo(inputs: Record<string, any>): unknown {
  return inputs.topic;
}
