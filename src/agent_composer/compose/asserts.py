"""`asserts:` parse + compile-validate + boundary/post-terminal classify.

A flow's `asserts:` is a list of boolean checks in the `when:`/`asserts:` grammar
(comparisons + `and`/`or`/`not` + arithmetic + `in`). Each enforces an invariant; a false
one fails the run. This module is the COMPILE-TIME half: it parses each assert, validates
every `${...}` reference it reads (reusing the leaf checker `compile.validation._classify_path`
— a dangling node/input or a bad record field is a loud `LoadError`), and CLASSIFIES each
assert by which side it constrains:

- a `${input.X}` / `${system.X}`-only assert -> **boundary**: it depends only on the run
  arguments + ambients, so it can fire BEFORE any node runs (fail-fast).
- an assert that reads ANY `${<id>.output[.X]}` -> **post-terminal**: it depends on a
  produced value, so it can only fire AFTER the run reaches a terminal.

This module does NOT run the asserts — `run_flow` evaluates each split against the
appropriate pool via `expr.evaluate_when`. Here we just parse, validate, and split.

Imports flow DOWN only: `compile.validation` (the leaf `_classify_path`), `expr`
(the grammar parse-check), `typesys.values` (Type, for typing). Nothing imports this back.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from agent_composer.compile.validation import _classify_path
from agent_composer.expr.expressions import ExpressionError, condition_refs
from agent_composer.typesys.values import Type
from agent_composer.compose.helpers.errors import LoadError


@dataclass(frozen=True)
class AssertSet:
    """The compile-time split of a flow's `asserts:`, for `run_flow` to enforce in two phases.

    `boundary` fires pre-run against the `${input}`/`${system}` pool; `post` fires after the
    run reaches a terminal, against the full pool (it reads `${<id>.output[.X]}`). Each list
    holds the raw assert strings (run via `expr.evaluate_when`), order-preserving.
    """

    boundary: list[str] = field(default_factory=list)
    post: list[str] = field(default_factory=list)


def classify_asserts(
    assert_list: list[str],
    flow_inputs: "set[str]",
    valid_targets: "set[str]",
    producers: "dict[str, Type]",
) -> AssertSet:
    """Parse, validate, and split a flow's `asserts:` into boundary vs post-terminal.

    `flow_inputs` is the declared flow-input names (the `${input.X}` set); `valid_targets`
    is the node ids (the `${<id>.output}` set); `producers` maps node id -> its
    `output_type` (drives the dotted-field walk, e.g. `${synth.output.confidence}`
    against the `View` record). Each assert is parse-checked in the boolean grammar (a
    malformed expression is loud) and each `${...}` ref is name-checked via
    `_classify_path` (a dangling ref / bad field is loud). An assert is
    **post-terminal** iff ANY of its refs has a head OTHER than `input`/`system`;
    otherwise **boundary**.

    Raises `LoadError` on the first malformed assert or bad ref.
    """
    result = AssertSet()
    # Only resolvable producers participate in the dotted walk (an opaque/None producer
    # stays lenient) — mirrors validate.validate_references.
    producer_types = {nid: sh for nid, sh in producers.items() if sh is not None}

    for expr in assert_list:
        # 1. parse-check + collect refs on the ONE unified condition walk. `condition_refs`
        # parses the WHOLE value as an expression (all three brace spellings) and returns its
        # reference leaves — so a whole-span `${a <= 10}` and a bare `a <= 10` extract the same
        # refs a mixed `${a} <= 10` does (a flat `${...}` regex could not).
        try:
            refs = condition_refs(expr)
        except ExpressionError as exc:
            raise LoadError(
                f"assert {expr!r} is not a valid boolean expression: {exc}"
            ) from exc

        # 2. validate each ref + 3. detect any outputs-head ref (-> post-terminal).
        is_post = False
        for path in refs:
            err = _classify_path(path, valid_targets, flow_inputs, (), producer_types)
            if err is not None:
                raise LoadError(f"assert {expr!r}: {err}")
            # post-terminal iff a ref has a head other than input/system.
            # The `item` head is body-local (not valid at flow-level), so a stray
            # `${item.X}` falls into "head not in input/system" → POST and raises later
            # at resolve time. Legacy `inputs`/`outputs` heads are already rejected by
            # _classify_path with the typo hint so they never reach here.
            head = path.split(".")[0]
            if head not in ("input", "system"):
                is_post = True

        (result.post if is_post else result.boundary).append(expr)

    return result
