"""CODE-node functions whose outputs would make an unbounded error message.

A CODE node references one of these as `tests.engine._bounded_codefns:fn`.
"""

from typing import Any

LONG_FIELD = "f" * 250


def _nest(levels: int, wrap, leaf: Any = 1) -> Any:
    """Return `leaf` wrapped in `levels` levels of `wrap`."""
    value = leaf
    for _ in range(levels):
        value = wrap(value)
    return value


class BrokenRepr:
    """A hashable key whose repr raises."""

    def __repr__(self) -> str:
        raise RuntimeError("no repr")


def big_set(inputs: dict) -> dict:
    return {"s": set(range(200_000))}


def long_str_keys(inputs: dict) -> dict:
    return {"s": _nest(200, lambda value: {"k" * 10_000: value})}


def long_tuple_key(inputs: dict) -> dict:
    return {(tuple(range(100_000)),): _nest(200, lambda value: [value])}


def long_int_key(inputs: dict) -> dict:
    return {"s": {10**5000: _nest(200, lambda value: [value])}}


def broken_repr_key(inputs: dict) -> dict:
    return {"s": {BrokenRepr(): _nest(200, lambda value: [value])}}


def many_unknown_fields(inputs: dict) -> dict:
    value = {f"key{i}": i for i in range(200_000)}
    value["a"] = 1
    return value


def wrong_long_field(inputs: dict) -> dict:
    return {LONG_FIELD: "not an int", "g": 1}


DEEP = 200_000


def deep_list(inputs: dict) -> list:
    return _nest(DEEP, lambda value: [value])


def deep_tuple(inputs: dict) -> tuple:
    return _nest(DEEP, lambda value: (value,))


def dict_of_deep_list(inputs: dict) -> dict:
    return {"s": _nest(DEEP, lambda value: [value])}


def deep_frozenset(inputs: dict) -> frozenset:
    return _nest(DEEP, lambda value: frozenset({value}))


class HidingList(list):
    """A list whose iteration yields nothing, so a walk over it sees no items."""

    def __iter__(self):
        return iter(())


def hidden_deep_list(inputs: dict) -> dict:
    return {"s": HidingList([_nest(DEEP, lambda value: [value])])}


class YieldingList(list):
    """A list that stores nothing while its iteration yields `shown`."""

    def __init__(self, shown: list):
        super().__init__()
        self.shown = shown

    def __iter__(self):
        return iter(self.shown)


def yielded_deep_list(inputs: dict) -> dict:
    return {"s": YieldingList([_nest(DEEP, lambda value: [value])])}
