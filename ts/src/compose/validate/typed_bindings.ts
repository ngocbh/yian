/**
 * Graph-stage: reject typed bindings whose value can never fit the declared type.
 *
 * A typed binding (`name: !Type value`, on a node input or a flow output) states the type its
 * value must have; the binder enforces it at runtime. A literal right side is already checked
 * against its type when the binding is split, where the authored text is at hand. This load pass
 * rejects the remaining bindings that could never pass the runtime check:
 *
 * - a string source with no `${...}` span (an escaped `$${x}`) is literal text, so it is checked
 *   exactly with the same write-boundary check the binder uses;
 * - a reference or expression is inferred from its raw source over the producer and flow-input
 *   contracts, and rejected only when that type is disjoint from the declared one. A narrower
 *   inferred type (an Optional source, a str into a Literal) loads and is left to the runtime
 *   check, as is an unproven source;
 * - a lone `${item}` or `${item.field}` read on a node that binds per item (MAP) is inferred from
 *   the element type of its `over` source, when that is a proven list;
 * - a value template is inferred from its structure, and each typed binding nested in it is
 *   checked the same way at its own path and origin.
 *
 * An AGENT's reserved `__llm_config` input is skipped: `llm_config_sources.ts` checks its tags at
 * the authored `llm_config` field (`typedValueError`), without CASE narrowing.
 *
 * `inferBindingType` returns the type one binding is proven to produce (a typed binding's
 * declared type, a literal's exact type, or a reference's inferred type), for passes that record
 * binding types rather than reject them. `localInputContracts` returns the proven contracts each
 * node-local name may have, for the passes that field-check what a node reads from its own inputs
 * (prompts).
 *
 * Node inputs are checked as soon as producer contracts exist, before the CALL, MAP, and LOOP
 * checks that compare the declared type with a child's parameters; flow outputs are checked once
 * their contracts are attached. A node input error locates at the binding's registered value
 * origin, else its `input:` entry range, else the node range; a flow output error at its
 * registered origin, else the `output:` range.
 *
 * Ported so far: `LocalInput`; the typed-binding checks are not ported yet (T3-19).
 *
 * Knows about: `compile/validation` (the type relation), `expr` (template scanning), `typesys`,
 * `nodes` (binding carriers, the LOOP node, the AGENT's reserved key), and `compose/helpers`
 * (output-contract inference and load diagnostics).
 * Never imports: the loader, the compose package facade, or the CALL/MAP check (`call_types.ts`,
 * for its literal and `${item}` helpers); they import this back.
 */

import type { OutputContract } from "../../nodes/contracts.ts";

/** The proven contracts a node-local name's value has, and where it is bound from; frozen. */
export class LocalInput {
  /** A read is an error only when every alternative rejects it. */
  readonly alternatives: readonly OutputContract[];
  /**
   * The pool path the name is bound from when the binding's one `${...}` span reads exactly one
   * reference path (`${d.output}`, `${d.output :- null}`), else `null`; a member-only union hint
   * names it instead of the local name.
   */
  readonly source: string | null;

  /** Build the local input over a frozen copy of `alternatives`. */
  constructor(alternatives: readonly OutputContract[], source: string | null = null) {
    this.alternatives = Object.freeze([...alternatives]);
    this.source = source;
    Object.freeze(this);
  }
}
