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


def _context_chain(error: BaseException | None) -> list[tuple[type, tuple]]:
    """Return the type and args of `error` and of each `__context__` link below it."""
    chain = []
    while error is not None:
        chain.append((type(error), error.args))
        error = error.__context__
    return chain


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


def test_drive_lets_a_parent_catch_a_nested_base_exception():
    def child_recursive() -> None:
        raise KeyboardInterrupt

    def parent_recursive() -> str:
        try:
            child_recursive()
        except BaseException as caught:
            return type(caught).__name__
        return "missed"  # pragma: no cover - the child always raises

    def child():
        raise KeyboardInterrupt
        yield  # pragma: no cover - makes this a generator

    def parent():
        try:
            yield child()
        except BaseException as caught:
            return type(caught).__name__
        return "missed"  # pragma: no cover - the child always raises

    assert drive(parent()) == parent_recursive() == "KeyboardInterrupt"


class _Interrupting:
    """A yielded value whose type check raises inside the trampoline's own frame."""

    @property
    def __class__(self):
        raise KeyboardInterrupt


def test_drive_closes_open_walkers_inner_to_outer_when_interrupted():
    log: list = []

    def chain(n: int):
        try:
            if n == 3:
                yield _Interrupting()
            yield chain(n + 1)
        finally:
            log.append(n)

    with pytest.raises(KeyboardInterrupt):
        drive(chain(0))
    assert log == [3, 2, 1, 0]


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


def test_flatten_sends_a_nested_result_back_into_the_parent():
    def child_recursive(n: int):
        yield n
        return n * 10

    def parent_recursive():
        got = yield from child_recursive(1)
        yield got
        got = yield from child_recursive(2)
        yield got

    def child(n: int):
        yield n
        return n * 10

    def parent():
        got = yield Descend(child(1))
        yield got
        got = yield Descend(child(2))
        yield got

    assert list(flatten(parent())) == list(parent_recursive()) == [1, 10, 2, 20]


def _closing_chain_recursive(n: int, log: list, catch_at: int | None):
    try:
        try:
            yield n
            yield from _closing_chain_recursive(n + 1, log, catch_at)
        finally:
            log.append(n)
            if n == 2:
                raise Boom(n)
    except Boom:
        if n != catch_at:
            raise


def _closing_chain(n: int, log: list, catch_at: int | None):
    try:
        try:
            yield n
            yield Descend(_closing_chain(n + 1, log, catch_at))
        finally:
            log.append(n)
            if n == 2:
                raise Boom(n)
    except Boom:
        if n != catch_at:
            raise


@pytest.mark.parametrize("catch_at", [None, 1, 0])
def test_flatten_closes_every_walker_when_a_finally_raises_as_yield_from_does(catch_at):
    def close_after_four(walker) -> tuple[list, BaseException | None]:
        items = [next(walker) for _ in range(4)]
        try:
            walker.close()
        except Boom as raised:
            return items, raised
        return items, None

    expected_log: list = []
    recursive = _closing_chain_recursive(0, expected_log, catch_at)
    expected, expected_error = close_after_four(recursive)

    log: list = []
    items, error = close_after_four(flatten(_closing_chain(0, log, catch_at)))

    assert items == expected == [0, 1, 2, 3]
    assert log == expected_log == [3, 2, 1, 0]
    assert _context_chain(error) == _context_chain(expected_error)
    assert (error is None) == (catch_at is not None)


def _thrown_chain_recursive(n: int, log: list):
    try:
        yield n
        if n < 3:
            yield from _thrown_chain_recursive(n + 1, log)
        else:
            yield "unreached"
    except Boom:
        if n != 1:
            raise
        yield "caught"
    finally:
        log.append(n)


def _thrown_chain(n: int, log: list):
    try:
        yield n
        if n < 3:
            yield Descend(_thrown_chain(n + 1, log))
        else:
            yield "unreached"
    except Boom:
        if n != 1:
            raise
        yield "caught"
    finally:
        log.append(n)


def test_flatten_forwards_a_thrown_error_to_the_innermost_walker():
    def throw_after_four(walker) -> tuple[list, list]:
        items = [next(walker) for _ in range(4)]
        items.append(walker.throw(Boom("thrown")))
        rest = list(walker)
        return items, rest

    expected_log: list = []
    expected = throw_after_four(_thrown_chain_recursive(0, expected_log))

    log: list = []
    assert throw_after_four(flatten(_thrown_chain(0, log))) == expected
    assert expected == ([0, 1, 2, 3, "caught"], [])
    assert log == expected_log == [3, 2, 1, 0]


def test_flatten_lets_a_parent_catch_a_nested_base_exception():
    def child_recursive():
        yield 1
        raise KeyboardInterrupt

    def parent_recursive():
        try:
            yield from child_recursive()
        except BaseException as caught:
            yield type(caught).__name__

    def child():
        yield 1
        raise KeyboardInterrupt

    def parent():
        try:
            yield Descend(child())
        except BaseException as caught:
            yield type(caught).__name__

    expected = [1, "KeyboardInterrupt"]
    assert list(flatten(parent())) == list(parent_recursive()) == expected


def test_flatten_closes_open_walkers_inner_to_outer_when_interrupted():
    log: list = []

    def chain(n: int):
        try:
            yield n
            yield _Interrupting() if n == 3 else Descend(chain(n + 1))
        finally:
            log.append(n)

    flat = flatten(chain(0))
    assert [next(flat) for _ in range(4)] == [0, 1, 2, 3]
    with pytest.raises(KeyboardInterrupt):
        next(flat)
    assert log == [3, 2, 1, 0]


def test_flatten_reports_a_walker_that_yields_while_closing_as_yield_from_does():
    def chain_recursive(n: int, log: list):
        try:
            try:
                yield n
                yield from chain_recursive(n + 1, log)
            finally:
                log.append(n)
                if n == 2:
                    raise Boom(n)
        except Boom:
            if n != 1:
                raise
            yield "late"

    def chain(n: int, log: list):
        try:
            try:
                yield n
                yield Descend(chain(n + 1, log))
            finally:
                log.append(n)
                if n == 2:
                    raise Boom(n)
        except Boom:
            if n != 1:
                raise
            yield "late"

    expected_log: list = []
    recursive = chain_recursive(0, expected_log)
    assert [next(recursive) for _ in range(4)] == [0, 1, 2, 3]
    with pytest.raises(RuntimeError, match="ignored GeneratorExit") as expected:
        recursive.close()

    log: list = []
    flat = flatten(chain(0, log))
    assert [next(flat) for _ in range(4)] == [0, 1, 2, 3]
    with pytest.raises(RuntimeError, match="ignored GeneratorExit") as raised:
        flat.close()
    assert log == expected_log == [3, 2, 1, 0]
    assert _context_chain(raised.value) == _context_chain(expected.value)


def test_flatten_reports_a_walker_that_ignores_generator_exit_as_yield_from_does():
    def chain_recursive(n: int, log: list):
        try:
            yield n
            if n < 2:
                yield from chain_recursive(n + 1, log)
        except GeneratorExit:
            log.append(n)
            if n == 2:
                yield "late"
            raise

    def chain(n: int, log: list):
        try:
            yield n
            if n < 2:
                yield Descend(chain(n + 1, log))
        except GeneratorExit:
            log.append(n)
            if n == 2:
                yield "late"
            raise

    expected_log: list = []
    recursive = chain_recursive(0, expected_log)
    assert [next(recursive) for _ in range(3)] == [0, 1, 2]
    with pytest.raises(RuntimeError, match="ignored GeneratorExit") as expected:
        recursive.close()

    log: list = []
    flat = flatten(chain(0, log))
    assert [next(flat) for _ in range(3)] == [0, 1, 2]
    with pytest.raises(RuntimeError, match="ignored GeneratorExit") as raised:
        flat.close()
    assert log == expected_log == [2]
    assert _context_chain(raised.value) == _context_chain(expected.value)
    # Unlike the `yield from` chain, flatten is finished once closing fails.
    assert list(flat) == []
