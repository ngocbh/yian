/**
 * The declared types the run time reads fields through: a pool root's, a node local's and a MAP
 * body's `item`, derived from a compiled flow's contracts.
 *
 * A field a value's declared type declares reads null when it's absent; any other absent field
 * raises. The types come from the same contract functions the loader checks reads with
 * (`output_contracts`), without the loader's narrowing, so every field the loader accepts on a
 * typed value is declared here too. They depend only on the flow's contracts, params and wiring,
 * which its fingerprint covers, so a resume or a replay derives the same types.
 *
 * Knows about: `output_contracts`, `model` (`Flow`), the node kinds whose output has no contract
 * (START, END) and the AGENT's steer inbox key, `expr` (`declaredAt`) and `typesys`.
 * Never imports: compose or runtime.
 */

import { declaredAt, type DeclaredTypes } from "../expr/expressions.ts";
import { STEER_INBOX_KEY } from "../nodes/agent/node.ts";
import { type Node, NodeKind } from "../nodes/base.ts";
import { type OutputContract, outputContractType, ValueContract } from "../nodes/contracts.ts";
import type { EndNode } from "../nodes/end/node.ts";
import { messagesKey, NEW_MESSAGES, poolRootLength } from "../typesys/pool.ts";
import { newMessagesType } from "../typesys/types.ts";
import { isInboxType, joinedType, Type, ValueKind } from "../typesys/values.ts";
import type { Flow } from "./model.ts";
import {
  bindingAlternatives,
  contractType,
  itemContract,
  projectOutputContract,
  resolveOutputBindingContract,
} from "./output_contracts.ts";

/** The record of `params`' types, untyped ones `Any`, the required ones required. */
function paramsRecord(node: Node): Type {
  const params = node.params ?? [];
  const any = Type.scalar(ValueKind.ANY);
  const fields = new Map(params.map((param): [string, Type] => [param.name, param.contract?.type ?? any]));
  const required = new Set(params.filter((param) => param.required).map((param) => param.name));
  return new Type({ kind: ValueKind.OBJECT, fields, required });
}

/**
 * The producer contracts of `flow`, as the loader's inference reads them, looked up on demand in
 * its live nodes: a node's output contract by id (a START's the record of its params), and a
 * message producer's `list(Message)` under its messages key. The inference only calls `get` and
 * `has`, so a lookup costs the same however many nodes an expansion added.
 */
class FlowProducers extends Map<string, OutputContract> {
  readonly #flow: Flow;

  constructor(flow: Flow) {
    super();
    this.#flow = flow;
  }

  override get(name: string): OutputContract | undefined {
    const node = this.#flow.nodes.get(name);
    if (node !== undefined) {
      if (node.outputContract !== null) return node.outputContract;
      return node.kind === NodeKind.START ? new ValueContract(paramsRecord(node)) : undefined;
    }
    const suffix = `.${NEW_MESSAGES}`;
    if (!name.endsWith(suffix)) return undefined;
    const producer = this.#flow.nodes.get(name.slice(0, -suffix.length));
    return producer?.producesMessages === true && messagesKey(producer.id) === name
      ? new ValueContract(newMessagesType())
      : undefined;
  }

  override has(name: string): boolean {
    return this.get(name) !== undefined;
  }
}

/** The producer and flow-input contracts of `flow`: `FlowProducers`, and the root START's typed params by name. */
function flowContracts(flow: Flow): [Map<string, OutputContract>, Map<string, ValueContract>] {
  const inputs = new Map<string, ValueContract>();
  for (const param of flow.nodes.get(flow.startId)?.params ?? []) {
    if (param.contract !== null) inputs.set(param.name, param.contract);
  }
  return [new FlowProducers(flow), inputs];
}

/** The type of a binding of `node`'s wiring, or `null` when unproven. */
function bindingType(
  node: Node,
  binding: unknown,
  contracts: [Map<string, OutputContract>, Map<string, ValueContract>],
): Type | null {
  const typeFn = (name: string) => node.types.get(name) ?? null;
  return contractType(resolveOutputBindingContract(binding, contracts[0], contracts[1], { typeFn }));
}

/** An END's output type: the record (or, in value mode, the value) of its inputs' types. */
function endType(flow: Flow, end: EndNode): Type | null {
  if (end.mode === "list") return null;
  const wiring = flow.wiring.get(end.id) ?? new Map<string, unknown>();
  const contracts = flowContracts(flow);
  const typeOf = (name: string): Type | null => {
    const param = (end.params ?? []).find((candidate) => candidate.name === name);
    if (param?.contract != null) return param.contract.type;
    return wiring.has(name) ? bindingType(end, wiring.get(name), contracts) : null;
  };
  const names = end.backwardInputNames();
  if (end.mode === "value") return names.length === 1 ? typeOf(names[0]!) : null;
  const fields = new Map(names.map((name): [string, Type] => [name, typeOf(name) ?? Type.scalar(ValueKind.ANY)]));
  return new Type({ kind: ValueKind.OBJECT, fields, required: new Set(names) });
}

/**
 * The declared type of node `node`'s output: its output contract's type; a START's the record of
 * its params; an END's the record or value of its inputs' types; else `null` (untyped).
 */
export function nodeOutputType(flow: Flow, node: Node): Type | null {
  if (node.outputContract !== null) {
    // Checks the contract's shape first: a malformed one throws its own TypeError.
    outputContractType(node.outputContract);
    return contractType(node.outputContract);
  }
  if (node.kind === NodeKind.START) return paramsRecord(node);
  if (node.kind === NodeKind.END) return endType(flow, node as EndNode);
  return null;
}

/**
 * The declared type of a pool reference's names in `flow`: `<node>.output` is the node's output
 * type (`nodeOutputType`), `<node>.new_messages` is `list(Message)`, `input.<k>` is the START's
 * param `k`, and `system.<k>` is text; the names after the root project with `declaredAt`. Lookups
 * go through the live `flow.nodes`, so a cloned node types itself by its id.
 */
export function poolDeclaredTypes(flow: Flow): DeclaredTypes {
  return (names) => {
    const used = poolRootLength(names);
    if (used === 1) return null;
    const [head, second] = names as [string, string];
    let root: Type | null = null;
    if (head === "input") {
      const param = (flow.nodes.get(flow.startId)?.params ?? []).find((candidate) => candidate.name === second);
      root = param?.contract?.type ?? null;
    } else if (head === "system") {
      root = Type.scalar(ValueKind.STRING);
    } else {
      const node = flow.nodes.get(head);
      if (node !== undefined && second === NEW_MESSAGES) root = node.producesMessages ? newMessagesType() : null;
      else if (node !== undefined) root = nodeOutputType(flow, node);
    }
    return declaredAt(root, names.slice(used));
  };
}

/** The one type of a local's alternatives: their join when equal up to `Optional`, else a union of records. */
function joinedAlternatives(types: readonly Type[]): Type | null {
  if (types.length <= 1) return types[0] ?? null;
  const joined = joinedType(types);
  if (joined !== null) return joined;
  if (types.every((typ) => typ.fields !== undefined)) return new Type({ kind: ValueKind.OBJECT, variants: types });
  return types[0]!;
}

// Each node's locals' types. An expansion adds nodes but never changes one, so a node's entry stays right.
const LOCAL_TYPES = new WeakMap<Node, ReadonlyMap<string, Type | null>>();

/**
 * Return node `node`'s locals' declared types, memoized per node: each param's contract type, or
 * for an untyped param its wiring binding's types from `bindingAlternatives` (and a LOOP's carried
 * field), without narrowing; `output`, the node's own output type, for a post assert; and
 * `event`, the steer inbox's element type, for `render:`.
 */
export function localTypes(flow: Flow, node: Node): ReadonlyMap<string, Type | null> {
  const memo = LOCAL_TYPES.get(node);
  if (memo !== undefined) return memo;
  const [producers, inputs] = flowContracts(flow);
  const wiring = flow.wiring.get(node.id) ?? new Map<string, unknown>();
  const typeFn = (name: string) => node.types.get(name) ?? null;
  const types = new Map<string, Type | null>();
  for (const param of node.params ?? []) {
    if (param.contract !== null) {
      types.set(param.name, param.contract.type);
      continue;
    }
    const alternatives = wiring.has(param.name)
      ? (bindingAlternatives(wiring.get(param.name), producers, inputs, typeFn) ?? [])
      : [];
    if (node.isLoop && node.outputContract !== null) {
      const carried = projectOutputContract(node.outputContract, [param.name]);
      if (carried !== null) alternatives.push(carried);
    }
    const known = alternatives.map(contractType).filter((typ): typ is Type => typ !== null);
    types.set(param.name, joinedAlternatives(known));
  }
  types.set("output", nodeOutputType(flow, node));
  if (wiring.has(STEER_INBOX_KEY)) {
    const source = wiring.get(STEER_INBOX_KEY);
    const inbox = contractType(resolveOutputBindingContract(source, producers, inputs, { typeFn }));
    types.set("event", inbox !== null && isInboxType(inbox) ? (inbox.element ?? null) : null);
  }
  LOCAL_TYPES.set(node, types);
  return types;
}

/** Return the element type `item` has in MAP node `node`, from `itemContract`; `null` when untyped. */
export function itemDeclaredType(flow: Flow, node: Node): Type | null {
  const [producers, inputs] = flowContracts(flow);
  const wiring = flow.wiring.get(node.id) ?? new Map<string, unknown>();
  const typeFn = (name: string) => node.types.get(name) ?? null;
  return contractType(itemContract(node, wiring, producers, inputs, typeFn));
}
