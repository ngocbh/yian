/**
 * The local origin hooks of the `Node` base: adapted rows of
 * `tests/engine/test_origin_traversal.py`.
 *
 * Changed: `test_generated_origin_context_is_not_local_authored_provenance` and
 * `test_invalid_mutable_node_origin_fails_before_iteration_or_mapping` go through the compile
 * layer's `iter_flow_origins` and `rewrite_flow_origins` in Python; here they call the node's own
 * `iterLocalOrigins` and `rewriteLocalOrigins`. Python's bare `Node` is a minimal subclass.
 *
 * TS-only: the local traversal order with shared carriers yielded once; `rewriteLocalOrigins` on
 * a shallow node copy leaves the original unchanged and keeps shared carriers shared; a shallow
 * copy validates its assertions and runs; and `failureTrace`.
 *
 * Later (T3-10): the 25 compile tests.
 */

import { expect, test } from "bun:test";

import { OriginRef } from "../../src/diagnostics.ts";
import { Node, NodeKind, type NodeResult, Output } from "../../src/nodes/base.ts";
import { Assertion, ContractOrigins, Parameter, ValueContract } from "../../src/nodes/contracts.ts";
import { Type, ValueKind } from "../../src/typesys/values.ts";

/** A minimal concrete node; the base hooks are the subject. Its body lets a copy run. */
class BareNode extends Node {
  readonly kind = NodeKind.CODE;

  override forward(): NodeResult {
    return new Output("done");
  }
}

/** Return one opaque source identity with a readable test key. */
function origin(key: string): OriginRef {
  return new OriginRef(key);
}

/** Return a node whose output contract and param share one contract, and whose asserts share one. */
function richNode(): { node: BareNode; contract: ValueContract; shared: Assertion } {
  const contract = new ValueContract(
    Type.scalar(ValueKind.STRING),
    new ContractOrigins({ default: origin("contract"), entries: [[["leaf"], origin("contract/leaf")]] }),
  );
  const node = new BareNode("n", { outputContract: contract, origin: origin("node") });
  const shared = new Assertion("x > 0", "x is positive", { origin: origin("shared") });
  node.params = [
    new Parameter("x", {
      contract,
      origin: origin("param"),
      hasDefault: true,
      default: 1n,
      defaultOrigin: origin("param/default"),
    }),
  ];
  node.preAsserts = [new Assertion("x < 9", "x is small", { origin: origin("pre") }), shared];
  node.postAsserts = [shared, new Assertion("output != ''", "output is set", { origin: origin("post") })];
  return { node, contract, shared };
}

/** Return a shallow copy of `node`: the same prototype, its own fields copied. */
function shallowCopy(node: BareNode): BareNode {
  return Object.assign(Object.create(Object.getPrototypeOf(node)) as BareNode, node);
}

test("generated origin context is not local authored provenance", () => {
  const node = new BareNode("child", { origin: origin("child/node") });
  const ancestry = [origin("outer/call")];
  node.originContext = ancestry;
  const visited: OriginRef[] = [];

  expect([...node.iterLocalOrigins()]).toEqual([origin("child/node")]);

  node.rewriteLocalOrigins((o) => {
    visited.push(o);
    return origin(`mapped/${o.key}`);
  });

  expect(visited).toEqual([origin("child/node")]);
  expect(node.origin).toEqual(origin("mapped/child/node"));
  expect(node.originContext).toBe(ancestry);
  expect(ancestry).toEqual([origin("outer/call")]);
});

test("invalid mutable node origin fails before iteration or mapping", () => {
  const node = new BareNode("invalid");
  (node as { origin: unknown }).origin = "not-an-origin";

  expect(() => [...node.iterLocalOrigins()]).toThrow(TypeError);
  expect(() => [...node.iterLocalOrigins()]).toThrow(/node origin must be an OriginRef/);
  expect(() => node.rewriteLocalOrigins((o) => o)).toThrow(TypeError);
  expect(() => node.rewriteLocalOrigins((o) => o)).toThrow(/node origin must be an OriginRef/);
});

test("local origins come in structural order, each shared carrier once", () => {
  const { node } = richNode();

  expect([...node.iterLocalOrigins()].map((o) => o.key)).toEqual([
    "node",
    "contract",
    "contract/leaf",
    "param",
    "param/default",
    "pre",
    "shared",
    "post",
  ]);
});

test("a graph traversal's seen set skips carriers it already visited", () => {
  const { node, contract, shared } = richNode();

  expect([...node.iterLocalOrigins(new Set([contract, shared]))].map((o) => o.key)).toEqual([
    "node",
    "param",
    "param/default",
    "pre",
    "post",
  ]);
});

test("rewriting a shallow copy leaves the original and keeps shared carriers shared", () => {
  const { node, contract, shared } = richNode();
  const { params, preAsserts, postAsserts } = node;
  const copy = shallowCopy(node);

  copy.rewriteLocalOrigins((o) => origin(`mapped/${o.key}`));

  expect([...node.iterLocalOrigins()].map((o) => o.key)).toEqual([
    "node",
    "contract",
    "contract/leaf",
    "param",
    "param/default",
    "pre",
    "shared",
    "post",
  ]);
  expect(node.outputContract).toBe(contract);
  expect(node.params).toBe(params);
  expect(node.preAsserts).toBe(preAsserts);
  expect(node.postAsserts).toBe(postAsserts);
  expect(params![0]!.contract).toBe(contract);
  expect(contract.origins.default).toEqual(origin("contract"));
  expect(shared.origin).toEqual(origin("shared"));

  expect([...copy.iterLocalOrigins()].map((o) => o.key)).toEqual([
    "mapped/node",
    "mapped/contract",
    "mapped/contract/leaf",
    "mapped/param",
    "mapped/param/default",
    "mapped/pre",
    "mapped/shared",
    "mapped/post",
  ]);
  expect(copy.outputContract).not.toBe(contract);
  expect(copy.params![0]!.contract).toBe(copy.outputContract as ValueContract);
  expect(copy.preAsserts[1]).toBe(copy.postAsserts[0]!);
  expect(copy.preAsserts[1]).not.toBe(shared);
});

test("a shallow copy validates its assertions and runs", async () => {
  const { node } = richNode();
  const copy = shallowCopy(node);

  expect(copy.validateAssertions()).toBeUndefined();
  expect(await copy.run({ x: 5n })).toEqual(new Output("done"));
  copy.preAsserts = [new Assertion("x >", "broken")];
  expect(() => copy.validateAssertions()).toThrow(/^node 'n' pre-assert 'broken' \(`x >`\) is not/);
});

test("failure trace appends the operation origin, else the node origin, to the ancestry", () => {
  const node = new BareNode("n", { origin: origin("node") });
  node.originContext = [origin("outer"), origin("outer")];

  expect(node.failureTrace(origin("op"))).toEqual([origin("outer"), origin("outer"), origin("op")]);
  expect(node.failureTrace()).toEqual([origin("outer"), origin("outer"), origin("node")]);
  node.origin = null;
  expect(node.failureTrace()).toEqual([origin("outer"), origin("outer")]);
});

test("failure trace rejects malformed ancestry", () => {
  const node = new BareNode("n");
  (node as { originContext: unknown }).originContext = origin("outer");
  expect(() => node.failureTrace()).toThrow(TypeError);
  expect(() => node.failureTrace()).toThrow("node origin_context must be an array");

  (node as { originContext: unknown }).originContext = ["outer"];
  expect(() => node.failureTrace()).toThrow("node origin_context item must be an OriginRef");

  const forged = Object.create(OriginRef.prototype) as OriginRef;
  node.originContext = [forged];
  expect(() => node.failureTrace()).toThrow(RangeError);
});
