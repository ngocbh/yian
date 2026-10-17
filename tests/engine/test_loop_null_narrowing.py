"""A LOOP `while:` null check narrows what enters the body on every iteration.

`while:` is checked before each iteration, the first included, so a carried value it proves
not null is never null when the body starts. The seed check and the check that each returned
field fits its body input therefore read such a value, and each record above it, as
possibly null; an inline body's generated input is not null there. `until:` is checked after
the body and `times:` has no condition, so neither narrows anything.
"""

from __future__ import annotations

import json
import re

import pytest

from yian.compose.helpers.errors import LoadError
from yian.compose.loader import load_flow
from yian.compose.nodes.loop import while_non_null_paths
from yian.events import NodeStarted
from yian.run import resume_flow, run_flow


@pytest.mark.parametrize(
    "predicate, paths",
    [
        ("cursor != null", {("cursor",)}),
        ("n < 3 and cursor != null", {("cursor",)}),
        ("not (cursor == null)", {("cursor",)}),
        ("page.next != null", {("page", "next")}),
        ("${cursor != null}", {("cursor",)}),
        ("cursor != null or n > 3", set()),
        ("isinstance(item, Item)", set()),
        ("cursor != null and other != null", {("cursor",), ("other",)}),
        ("(a != null and b != null) or (a != null and n > 1)", {("a",)}),
    ],
)
def test_while_proves_the_null_checked_paths(predicate: str, paths: set) -> None:
    assert while_non_null_paths("while", predicate) == paths


@pytest.mark.parametrize(
    "kind, predicate",
    [("until", "cursor == null"), ("until", "cursor != null"), ("times", None)],
)
def test_until_and_times_prove_nothing(kind: str, predicate: str | None) -> None:
    assert while_non_null_paths(kind, predicate) == set()


# Pagination over a named body that takes `cursor: str`. Its CODE node fails on a null
# cursor, so a run that succeeds shows the null never entered the body.
_PAGE = """id: pages
name: pages
input:
  {inputs}
defs:
  fetch_page:
    input:
      cursor: !str
    nodes:
      get:
        kind: code
        input:
          cursor: ${{input.cursor}}
        output: Optional(str)
        code: |
          assert inputs["cursor"] is not None, "body saw a null cursor"
          return {{"p1": "p2", "p2": None}}.get(inputs["cursor"])
    output:
      cursor: ${{get.output}}
nodes:
  pages:
    kind: loop
    {driver}
    input:
      cursor: {seed}{carried}
    call: fetch_page
output: ${{pages.output}}
"""


def _page(
    seed: str = "${input.start}",
    driver: str = "while: cursor != null",
    inputs: str = "start: !Optional(str)",
    carried: str = "",
) -> str:
    """Return the pagination flow with the given seed, driver, flow inputs and extra seeds."""
    return _PAGE.format(seed=seed, driver=driver, inputs=inputs, carried=carried)


def _body_runs(result) -> int:
    """Count the body's CODE node runs in a finished run."""
    return sum(
        1 for e in result.events if isinstance(e, NodeStarted) and e.node_id.endswith("/get")
    )


def test_optional_seed_fills_a_strict_body_input_under_a_while_null_check() -> None:
    flow = load_flow(_page())
    for start, runs in ((None, 0), ("p1", 2)):
        result = run_flow(flow, {"start": start})
        assert result.status == "succeeded", result.failure
        assert result.output == {"cursor": None}
        assert _body_runs(result) == runs


def test_optional_seed_pagination_resumes_after_its_first_expansion(tmp_path) -> None:
    source = _page()
    path = tmp_path / "pages.jsonl"
    first = run_flow(load_flow(source), {"start": "p1"}, log=path, num_workers=0)
    assert first.status == "succeeded", first.failure
    lines = path.read_text().splitlines(keepends=True)
    expanded = [i for i, line in enumerate(lines) if json.loads(line)["kind"] == "node_expanded"]
    assert expanded
    path.write_text("".join(lines[: expanded[0] + 1]))

    resumed = resume_flow(load_flow(source), log=path)

    assert resumed.status == "succeeded", resumed.failure
    assert resumed.output == {"cursor": None}


@pytest.mark.parametrize(
    "seed, inputs, runs",
    [
        (
            "!Optional(str) ${input.start}",
            "start: !Optional(str)",
            [({"start": None}, 0), ({"start": "p1"}, 2)],
        ),
        (
            "${input.a :- input.b}",
            "a: !Optional(str)\n  b: !Optional(str)",
            [
                ({"a": None, "b": None}, 0),
                ({"a": None, "b": "p1"}, 2),
                ({"a": "p2", "b": None}, 1),
            ],
        ),
        ("null", "start: !Optional(str)", [({"start": None}, 0)]),
    ],
)
def test_every_seed_shape_is_narrowed_by_a_while_null_check(seed, inputs, runs) -> None:
    flow = load_flow(_page(seed=seed, inputs=inputs))
    for flow_inputs, count in runs:
        result = run_flow(flow, flow_inputs)
        assert result.status == "succeeded", result.failure
        assert result.output == {"cursor": None}
        assert _body_runs(result) == count


# A LOOP routed by a `case` on a `Union(Buy, Hold)` producer: the seed `order` is narrowed
# to `Buy` only under `when: buy`, and `cursor` only by the `while:`.
_ROUTED = """id: p
name: p
typedefs:
  Buy: {{kind: 'Literal(buy)', qty: int}}
  Hold: {{kind: 'Literal(hold)', note: str}}
input:
  s: !str
  start: !Optional(str)
defs:
  body:
    input:
      order: !Buy
      cursor: !str
      k: !int
    nodes:
      step:
        kind: code
        input:
          order: ${{input.order}}
          cursor: ${{input.cursor}}
          k: ${{input.k}}
        output:
          cursor: Optional(str)
          k: int
        code: |
          assert inputs["cursor"] is not None
          return dict(cursor=None, k=inputs["k"] + inputs["order"]["qty"])
    output: ${{step.output}}
nodes:
  d:
    kind: code
    input:
      s: ${{input.s}}
    output: Union(Buy, Hold)
    code: |
      return dict(kind="buy", qty=2) if inputs["s"] == "b" else dict(kind="hold", note="n")
  route:
    kind: case
    on: ${{d.output.kind}}
    cases:
      - when: buy
        then: {then}
    else: {other}
  loop:
    kind: loop
    while: cursor != null
    input:
      order: ${{d.output}}
      cursor: ${{input.start}}
      k: 0
    call: body
  e:
    kind: code
    code: return 1
output: ${{loop.output.k}}
"""


def test_while_null_check_keeps_the_case_narrowed_seed() -> None:
    flow = load_flow(_ROUTED.format(then="loop", other="e"))
    for start, k in (("c1", 2), (None, 0)):
        result = run_flow(flow, {"s": "b", "start": start})
        assert result.status == "succeeded", result.failure
        assert result.output == k

    with pytest.raises(LoadError, match="loop node 'loop': binding 'order' — child expects"):
        load_flow(_ROUTED.format(then="e", other="loop"))


_SEED_MESSAGE = (
    "loop node 'pages': binding 'cursor' — child expects 'str', source is 'Optional(str)'"
)
_RETURNED_MESSAGE = (
    "loop node 'pages': the body 'fetch_page' returns 'cursor' as Optional(str), "
    "but takes 'cursor' as str"
)


def test_returned_null_fits_a_strict_body_input_under_a_while_null_check() -> None:
    flow = load_flow(_page(seed="p1"))
    result = run_flow(flow, {"start": None})
    assert result.status == "succeeded", result.failure
    assert result.output == {"cursor": None}
    assert _body_runs(result) == 2


def test_until_null_check_narrows_neither_the_seed_nor_a_returned_field() -> None:
    driver = "until: cursor == null"
    with pytest.raises(LoadError, match=re.escape(_SEED_MESSAGE)):
        load_flow(_page(driver=driver))
    with pytest.raises(LoadError, match=re.escape(_RETURNED_MESSAGE)):
        load_flow(_page(seed="p1", driver=driver))


def test_null_check_under_or_narrows_nothing() -> None:
    driver = "while: cursor != null or n > 3"
    carried = "\n      n: 0"
    with pytest.raises(LoadError, match=re.escape(_SEED_MESSAGE)):
        load_flow(_page(driver=driver, carried=carried))
    with pytest.raises(LoadError, match=re.escape(_RETURNED_MESSAGE)):
        load_flow(_page(seed="p1", driver=driver, carried=carried))


def test_returned_field_still_keeps_its_carried_type_under_a_while_null_check() -> None:
    # With zero iterations the loop outputs the seed, so a body returning `str` can't
    # carry an `Optional(str)` seed even though the `while:` keeps a null out of the body.
    source = """id: so
name: so
input:
  start: !Optional(str)
defs:
  step:
    input:
      cursor: !str
      n: !int
    nodes:
      get:
        kind: code
        input:
          cursor: ${input.cursor}
          n: ${input.n}
        output:
          cursor: str
          n: int
        code: return dict(cursor=inputs["cursor"] + "x", n=inputs["n"] + 1)
    output: ${get.output}
nodes:
  pages:
    kind: loop
    while: cursor != null and n < 2
    input:
      cursor: ${input.start}
      n: 0
    call: step
output: ${pages.output}
"""
    with pytest.raises(LoadError) as caught:
        load_flow(source)
    message = str(caught.value)
    assert "loop node 'pages': carried field 'cursor'" in message
    assert "a returned field must keep its carried type" in message


# A partial loop: the body takes `cursor` but returns only `n`, so `cursor` carries over.
_PARTIAL = """id: p
name: p
input:
  start: !Optional(str)
defs:
  step:
    input:
      cursor: !str
      n: !int
    nodes:
      get:
        kind: code
        input:
          cursor: ${{input.cursor}}
          n: ${{input.n}}
        output:
          n: int
        code: |
          assert inputs["cursor"] is not None
          return dict(n=inputs["n"] + 1)
    output: ${{get.output}}
nodes:
  loop:
    kind: loop
    {driver}
    input:
      cursor: ${{input.start}}
      n: 0
    call: step
output: ${{loop.output}}
"""


def test_partial_loop_seed_check_alone_decides_a_carried_over_field() -> None:
    flow = load_flow(_PARTIAL.format(driver="while: cursor != null and n < 2"))
    for start, n in ((None, 0), ("c", 2)):
        result = run_flow(flow, {"start": start})
        assert result.status == "succeeded", result.failure
        assert result.output == {"cursor": start, "n": n}

    message = (
        "loop node 'loop': binding 'cursor' — child expects 'str', source is 'Optional(str)'"
    )
    with pytest.raises(LoadError, match=re.escape(message)):
        load_flow(_PARTIAL.format(driver="while: n < 2"))


# `page.next != null` proves both `page` and `page.next` not null, so an `Optional(Page)`
# seed fills a body input `Strict` whose `next` is `str`.
_NESTED = """id: nl
name: nl
typedefs:
  Page:
    next: Optional(str)
    n: int
  Strict:
    next: str
    n: int
input:
  start: !Optional(str)
defs:
  step:
    input:
      page: !Strict
    nodes:
      get:
        kind: code
        input:
          page: ${{input.page}}
        output:
          page: {returned}
        code: |
          assert inputs["page"]["next"] is not None
          n = inputs["page"]["n"] + 1
          return dict(page=dict(next=None if n >= 2 else "c", n=n))
    output: ${{get.output}}
  seed:
    input:
      s: !Optional(str)
    nodes:
      mk:
        kind: code
        input:
          s: ${{input.s}}
        output: Optional(Page)
        code: return None if inputs["s"] is None else dict(next=inputs["s"], n=0)
    output: ${{mk.output}}
nodes:
  first:
    kind: call
    call: seed
    input:
      s: ${{input.start}}
  pages:
    kind: loop
    while: page.next != null
    input:
      page: ${{first.output}}
    call: step
output: ${{pages.output}}
"""


def test_nested_null_check_narrows_the_record_above_it() -> None:
    flow = load_flow(_NESTED.format(returned="Optional(Page)"))
    for start, output in ((None, {"page": None}), ("c", {"page": {"next": None, "n": 2}})):
        result = run_flow(flow, {"start": start})
        assert result.status == "succeeded", result.failure
        assert result.output == output


def test_nested_null_check_keeps_the_carried_type_of_the_returned_record() -> None:
    with pytest.raises(LoadError) as caught:
        load_flow(_NESTED.format(returned="Page"))
    message = str(caught.value)
    assert "loop node 'pages': carried field 'page'" in message
    assert "a returned field must keep its carried type" in message


# Pagination over an inline body whose CALL node passes `cursor` to a strict `str` input.
_INLINE = """id: pages
name: pages
input:
  start: !Optional(str)
defs:
  fetch_page:
    input:
      cursor: !str
    nodes:
      get:
        kind: code
        input:
          cursor: ${{input.cursor}}
        output: Optional(str)
        code: |
          assert inputs["cursor"] is not None, "body saw a null cursor"
          return {{"p1": "p2", "p2": None}}.get(inputs["cursor"])
    output: ${{get.output}}
nodes:
  pages:
    kind: loop
    {driver}
    input:
      cursor: {seed}
    nodes:
      fetch:
        kind: call
        call: fetch_page
        input:
          cursor: ${{input.cursor}}
    output:
      cursor: ${{fetch.output}}
output: ${{pages.output}}
"""


def _inline(seed: str = "${input.start}", driver: str = "while: cursor != null") -> str:
    """Return the inline pagination flow with the given seed and driver."""
    return _INLINE.format(seed=seed, driver=driver)


@pytest.mark.parametrize("seed", ["${input.start}", "p1"])
def test_inline_body_reads_a_while_null_checked_seed_as_not_null(seed: str) -> None:
    flow = load_flow(_inline(seed=seed))
    for start in (None, "p1"):
        runs = 2 if seed == "p1" or start == "p1" else 0
        result = run_flow(flow, {"start": start})
        assert result.status == "succeeded", result.failure
        assert result.output == {"cursor": None}
        assert _body_runs(result) == runs


def test_inline_body_under_until_reads_the_seed_as_optional() -> None:
    driver = "until: cursor == null"
    message = "binding 'cursor' — child expects 'str', source is 'Optional(str)'"
    with pytest.raises(LoadError, match=re.escape(message)):
        load_flow(_inline(driver=driver))
    message = (
        "loop node 'pages': the body returns 'cursor' as Optional(str), "
        "but the carried field is typed str from its seed"
    )
    with pytest.raises(LoadError, match=re.escape(message)):
        load_flow(_inline(seed="p1", driver=driver))
