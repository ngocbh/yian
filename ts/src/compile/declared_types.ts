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
 * (START, END), `expr` (`declaredAt`) and `typesys`.
 * Never imports: compose or runtime.
 */

import { declaredAt, type DeclaredTypes } from "../expr/expressions.ts";
import { type Node, NodeKind } from "../nodes/base.ts";
import { type OutputContract, ValueContract } from "../nodes/contracts.ts";
import type { EndNode } from "../nodes/end/node.ts";
import { messagesKey, NEW_MESSAGES, poolRootLength } from "../typesys/pool.ts";
import { newMessagesType } from "../typesys/types.ts";
import { Type, ValueKind } from "../typesys/values.ts";
import type { Flow } from "./model.ts";
import { contractType, resolveOutputBindingContract } from "./output_contracts.ts";

/** The record of `params`' types, untyped ones `Any`, the required ones required. */
function paramsRecord(node: Node): Type {
  const params = node.params ?? [];
  const any = Type.scalar(ValueKind.ANY);
  const fields = new Map(params.map((param): [string, Type] => [param.name, param.contract?.type ?? any]));
  const required = new Set(params.filter((param) => param.required).map((param) => param.name));
  return new Type({ kind: ValueKind.OBJECT, fields, required });
}

/**
 * The producer and flow-input contracts of `flow`, as the loader's inference reads them: each
 * node's output contract by id (a START's the record of its params), each message producer's
 * `list(Message)` under its messages key, and the root START's typed params by name.
 */
function flowContracts(flow: Flow): [Map<string, OutputContract>, Map<string, ValueContract>] {
  const producers = new Map<string, OutputContract>();
  for (const [id, node] of flow.nodes) {
    if (node.outputContract !== null) producers.set(id, node.outputContract);
    else if (node.kind === NodeKind.START) producers.set(id, new ValueContract(paramsRecord(node)));
    if (node.producesMessages) producers.set(messagesKey(id), new ValueContract(newMessagesType()));
  }
  const inputs = new Map<string, ValueContract>();
  for (const param of flow.nodes.get(flow.startId)?.params ?? []) {
    if (param.contract !== null) inputs.set(param.name, param.contract);
  }
  return [producers, inputs];
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
  if (node.outputContract !== null) return contractType(node.outputContract);
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
