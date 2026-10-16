"""`drive` and `flatten` run nested generator walkers at constant stack depth.

Each test pairs a generator walker with its recursive twin and checks that the
trampoline gives the same results, side-effect order, caught errors, and exception
context, and that depth is not bounded by the recursion limit.
"""

import random

import pytest

from yian.trampoline import Descend, drive, flatten

_DEEP = 100_000


class Boom(Exception):
    """An error a walker raises at a chosen depth."""


class Own(Exception):
    """An error a walker raises after handling a nested one."""


def _random_tree(seed: int, size: int = 300) -> list:
    """Return a nested list of ints with `size` leaves, nesting chosen by `seed`."""
    rng = random.Random(seed)
    root: list = []
    open_lists = [root]
    for leaf in range(size):
        roll = rng.random()
        if roll < 0.3:
            child: list = []
            open_lists[-1].append(child)
            open_lists.append(child)
        elif roll < 0.5 and len(open_lists) > 1:
            open_lists.pop()
        open_lists[-1].append(leaf)
    return root


# drive


def _count(n: int):
    """Return `n` by descending `n` levels, one nested walker per level."""
    if n == 0:
        return 0
    return (yield _count(n - 1)) + 1


def test_drive_returns_a_result_from_deep_nesting():
    assert drive(_count(_DEEP)) == _DEEP


def _visit_recursive(node: list, log: list) -> int:
    log.append(("enter", len(node)))
    total = 0
    for child in node:
        total += _visit_recursive(child, log) if isinstance(child, list) else child
    log.append(("exit", total))
    return total


def _visit(node: list, log: list):
    log.append(("enter", len(node)))
    total = 0
    for child in node:
        total += (yield _visit(child, log)) if isinstance(child, list) else child
    log.append(("exit", total))
    return total


def test_drive_sends_each_nested_result_back_in_order():
    tree = _random_tree(7)
    expected_log: list = []
    expected = _visit_recursive(tree, expected_log)

    log: list = []
    assert drive(_visit(tree, log)) == expected
    assert log == expected_log


def _dive_recursive(n: int, k: int, log: list) -> str:
    try:
        if n == k:
            raise Boom(n)
        try:
            return _dive_recursive(n + 1, k, log)
        except Boom:
            if n == 0:
                return "caught"
            raise
    finally:
        log.append(n)


def _dive(n: int, k: int, log: list):
    try:
        if n == k:
            raise Boom(n)
        try:
            return (yield _dive(n + 1, k, log))
        except Boom:
            if n == 0:
                return "caught"
            raise
    finally:
        log.append(n)


def test_drive_throws_a_nested_error_in_at_each_yield():
    expected_log: list = []
    expected = _dive_recursive(0, 5, expected_log)

    log: list = []
    assert drive(_dive(0, 5, log)) == expected == "caught"
    assert log == expected_log == [5, 4, 3, 2, 1, 0]


def test_drive_raises_an_uncaught_error_after_every_finally():
    log: list = []

    def leaf():
        try:
            raise Boom("leaf")
        finally:
            log.append("leaf")
        yield  # pragma: no cover - makes this a generator

    def middle():
        try:
            yield leaf()
        finally:
            log.append("middle")

    with pytest.raises(Boom, match="leaf"):
        drive(middle())
    assert log == ["leaf", "middle"]


def _handled_then_own_recursive() -> None:
    def child() -> None:
        raise Boom("child")

    try:
        child()
    except Boom:
        pass
    raise Own("parent")


def _handled_then_own():
    def child():
        raise Boom("child")
        yield  # pragma: no cover - makes this a generator

    try:
        yield child()
    except Boom:
        pass
    raise Own("parent")


def test_drive_leaks_no_context_into_an_error_raised_after_handling():
    with pytest.raises(Own) as recursive:
        _handled_then_own_recursive()
    with pytest.raises(Own) as driven:
        drive(_handled_then_own())

    assert recursive.value.__context__ is None
    assert driven.value.__context__ is None


def test_drive_keeps_the_context_of_an_error_raised_while_handling():
    def child():
        raise Boom("child")
        yield  # pragma: no cover - makes this a generator

    def parent():
        try:
            yield child()
        except Boom:
            raise Own("parent")

    with pytest.raises(Own) as driven:
        drive(parent())
    assert isinstance(driven.value.__context__, Boom)


def test_drive_refuses_a_yielded_non_generator():
    def walker():
        yield 5

    with pytest.raises(TypeError, match="int"):
        drive(walker())


# flatten


def _items_recursive(node: list):
    for child in node:
        if isinstance(child, list):
            yield from _items_recursive(child)
        else:
            yield child


def _items(node: list):
    for child in node:
        if isinstance(child, list):
            yield Descend(_items(child))
        else:
            yield child


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_flatten_yields_items_in_yield_from_order(seed):
    tree = _random_tree(seed)
    assert list(flatten(_items(tree))) == list(_items_recursive(tree))


def test_flatten_streams_deep_nesting():
    def chain(n: int):
        yield n
        if n:
            yield Descend(chain(n - 1))

    assert list(flatten(chain(_DEEP))) == list(range(_DEEP, -1, -1))


def test_flatten_throws_a_nested_error_into_the_parent():
    def child_recursive():
        yield 1
        raise Boom("child")

    def parent_recursive():
        try:
            yield from child_recursive()
        except Boom:
            yield "recovered"
        yield "after"

    def child():
        yield 1
        raise Boom("child")

    def parent():
        try:
            yield Descend(child())
        except Boom:
            yield "recovered"
        yield "after"

    assert list(flatten(parent())) == list(parent_recursive()) == [
        1,
        "recovered",
        "after",
    ]


def test_flatten_raises_an_uncaught_nested_error():
    def child():
        yield 1
        raise Boom("child")

    def parent():
        yield Descend(child())

    with pytest.raises(Boom, match="child"):
        list(flatten(parent()))


def test_flatten_closes_open_walkers_inner_to_outer_when_stopped_early():
    def chain_recursive(n: int, log: list):
        try:
            yield n
            yield from chain_recursive(n + 1, log)
        finally:
            log.append(n)

    def chain(n: int, log: list):
        try:
            yield n
            yield Descend(chain(n + 1, log))
        finally:
            log.append(n)

    expected_log: list = []
    recursive = chain_recursive(0, expected_log)
    expected = [next(recursive) for _ in range(3)]
    recursive.close()

    log: list = []
    flat = flatten(chain(0, log))
    assert [next(flat) for _ in range(3)] == expected
    flat.close()
    assert log == expected_log == [2, 1, 0]


def test_flatten_refuses_a_non_generator_stream():
    def walker():
        yield Descend([1, 2])

    with pytest.raises(TypeError, match="list"):
        list(flatten(walker()))
