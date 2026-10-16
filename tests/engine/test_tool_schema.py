"""Tool schemas come from a function's signature and docstring, with an optional ToolContext."""

import dataclasses
import functools
import warnings
from typing import Annotated, Optional, get_args, get_origin

import pytest
from pydantic import BaseModel, Field, ValidationError
from pydantic.errors import PydanticInvalidForJsonSchema
from pydantic.v1 import BaseModel as BaseModelV1

from yian.tools.context import ToolContext
from yian.tools.schema import function_schema


class Db:
    pass


class Foo:
    pass


def place(ctx: ToolContext[Db], symbol: str, qty: int = 1) -> str:
    """Place an order.

    Args:
        symbol: The ticker.
        qty: How many shares.

    Returns:
        The order id.
    """
    return f"{symbol}:{qty}"


def test_tool_context_is_a_frozen_generic_dataclass():
    ctx = ToolContext[dict](
        call_id="r/n", attempt=1, tool_name="t", run_id="r", node="n", context={}
    )

    assert (ctx.call_id, ctx.attempt, ctx.tool_name, ctx.run_id, ctx.node, ctx.context) == (
        "r/n",
        1,
        "t",
        "r",
        "n",
        {},
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        ctx.attempt = 2
    alias = ToolContext[int]
    assert get_origin(alias) is ToolContext
    assert get_args(alias) == (int,)


def test_context_parameter_is_skipped_in_the_schema():
    schema = function_schema(place)

    assert schema.name == "place"
    assert schema.takes_context is True
    assert schema.description == "Place an order."
    props = schema.params_json_schema["properties"]
    assert set(props) == {"symbol", "qty"}
    assert schema.params_json_schema["required"] == ["symbol"]
    assert schema.params_json_schema["title"] == "place"
    assert props["symbol"]["description"] == "The ticker."
    assert props["qty"]["description"] == "How many shares."
    assert props["qty"]["default"] == 1
    assert schema.return_annotation is str


def test_to_call_args_skips_the_context_parameter():
    schema = function_schema(place)

    data = schema.params_model(symbol="AAPL", qty=3)

    assert schema.to_call_args(data) == (["AAPL", 3], {})


def test_unparameterized_context_is_detected():
    def f(ctx: ToolContext, x: int) -> int:
        return x

    assert function_schema(f).takes_context is True


def test_detection_is_by_type_not_name():
    def by_name(ctx: str, x: int) -> int:
        return x

    def by_type(context: ToolContext, x: int) -> int:
        return x

    plain = function_schema(by_name)
    assert plain.takes_context is False
    assert set(plain.params_json_schema["properties"]) == {"ctx", "x"}
    assert function_schema(by_type).takes_context is True


def test_string_annotations_from_a_future_module_are_resolved():
    from tests.engine._future_tools import place as future_place

    schema = function_schema(future_place)

    assert schema.takes_context is True
    assert set(schema.params_json_schema["properties"]) == {"symbol", "qty"}
    assert schema.return_annotation is str


def test_context_in_a_later_position_is_rejected():
    def f(x: int, ctx: ToolContext) -> int:
        return x

    with pytest.raises(TypeError) as excinfo:
        function_schema(f)

    assert str(excinfo.value) == (
        "tool function 'f' declares ToolContext parameter 'ctx' at position 2; "
        "it must be the first parameter"
    )


def test_keyword_only_context_is_rejected():
    def f(*, ctx: ToolContext, x: int) -> int:
        return x

    with pytest.raises(TypeError, match="'f'.*'ctx'"):
        function_schema(f)


def test_annotated_string_describes_a_parameter():
    def f(n: Annotated[int, "How many"]) -> int:
        return n

    def g(n: Annotated[int, "How many"]) -> int:
        """Count.

        Args:
            n: The count.
        """
        return n

    assert function_schema(f).params_json_schema["properties"]["n"]["description"] == "How many"
    assert function_schema(g).params_json_schema["properties"]["n"]["description"] == "The count."


def test_field_constraints_are_kept():
    def with_default(qty: int = Field(default=1, ge=1, description="d")) -> int:
        return qty

    def with_annotated(qty: Annotated[int, Field(ge=1)]) -> int:
        return qty

    default_schema = function_schema(with_default)
    prop = default_schema.params_json_schema["properties"]["qty"]
    assert (prop["default"], prop["minimum"], prop["description"]) == (1, 1, "d")
    assert "required" not in default_schema.params_json_schema
    with pytest.raises(ValidationError):
        default_schema.params_model(qty=0)

    annotated_schema = function_schema(with_annotated)
    assert annotated_schema.params_json_schema["properties"]["qty"]["minimum"] == 1
    assert annotated_schema.params_json_schema["required"] == ["qty"]
    with pytest.raises(ValidationError):
        annotated_schema.params_model(qty=0)


def test_docstring_description_wins_over_a_field_description():
    def f(qty: int = Field(default=1, description="d")) -> int:
        """Do it.

        Args:
            qty: From the docstring.
        """
        return qty

    prop = function_schema(f).params_json_schema["properties"]["qty"]
    assert prop["description"] == "From the docstring."


def test_overrides_replace_name_and_description():
    schema = function_schema(place, name_override="buy", description_override="Buy it.")

    assert schema.name == "buy"
    assert schema.description == "Buy it."
    assert schema.params_json_schema["title"] == "buy"


def test_no_docstring_and_no_override_gives_no_description():
    def f(x: int) -> int:
        return x

    assert function_schema(f).description is None


def test_positional_keyword_only_and_variadic_parameters():
    def f(ctx: ToolContext, a: int, *args: int, b: str, **kwargs: str) -> None:
        return None

    schema = function_schema(f)
    props = schema.params_json_schema["properties"]
    assert props["args"]["type"] == "array"
    assert props["args"]["items"] == {"type": "integer"}
    assert props["kwargs"]["type"] == "object"
    assert props["kwargs"]["additionalProperties"] == {"type": "string"}
    assert schema.params_json_schema["required"] == ["a", "b"]

    data = schema.params_model(a=1, args=[2, 3], b="x", kwargs={"k": "v"})
    assert schema.to_call_args(data) == ([1, 2, 3], {"b": "x", "k": "v"})
    with pytest.raises(ValidationError):
        schema.params_model(a=1, b="x", k="v")


def test_keyword_only_parameters_are_passed_by_name():
    def f(a: int, *, b: int = 2) -> int:
        return a + b

    schema = function_schema(f)

    assert schema.to_call_args(schema.params_model(a=1)) == ([1], {"b": 2})


def test_var_keyword_key_naming_a_parameter_is_rejected():
    def f(a: int, **kw: int) -> int:
        return a

    schema = function_schema(f)
    data = schema.params_model(a=1, kw={"a": 2})

    with pytest.raises(ValueError, match="'a'"):
        schema.to_call_args(data)


def test_bound_method():
    class Session:
        def lookup(self, key: str) -> str:
            """Look up a key.

            Args:
                key: The key.
            """
            return key

    schema = function_schema(Session().lookup)

    assert schema.name == "lookup"
    assert set(schema.params_json_schema["properties"]) == {"key"}
    assert schema.to_call_args(schema.params_model(key="k")) == (["k"], {})


def test_arbitrary_class_annotation_builds_a_model_with_a_lazy_json_schema():
    def f(x: Foo) -> None:
        return None

    schema = function_schema(f)

    foo = Foo()
    assert schema.params_model.model_validate({"x": foo}).x is foo
    with pytest.raises(PydanticInvalidForJsonSchema):
        schema.params_json_schema


def test_unresolvable_parameter_annotation_is_rejected():
    def f(x: "Missing") -> None:  # noqa: F821
        return None

    with pytest.raises(TypeError, match="'f'") as excinfo:
        function_schema(f)

    assert isinstance(excinfo.value.__cause__, NameError)


def test_unresolvable_return_annotation_is_rejected():
    def f(x: int) -> "Missing":  # noqa: F821
        return None

    with pytest.raises(TypeError, match="'f'") as excinfo:
        function_schema(f)

    assert isinstance(excinfo.value.__cause__, NameError)


def test_optional_context_first_is_rejected():
    def f(ctx: Optional[ToolContext], x: int) -> int:
        return x

    def g(ctx: ToolContext | None, x: int) -> int:
        return x

    with pytest.raises(TypeError, match="'f'"):
        function_schema(f)
    with pytest.raises(TypeError, match="'g'"):
        function_schema(g)


@pytest.mark.parametrize("first", [True, False])
def test_context_inside_another_type_is_rejected(first):
    if first:

        def f(items: list[ToolContext], x: int) -> int:
            return x

    else:

        def f(x: int, items: list[ToolContext]) -> int:
            return x

    with pytest.raises(TypeError, match="'f'.*'items'"):
        function_schema(f)


def test_fixed_length_tuple_var_positional_is_rejected():
    def f(*args: tuple[int, str]) -> None:
        return None

    with pytest.raises(TypeError, match="'f'"):
        function_schema(f)


def test_context_subclass_is_detected_first_and_rejected_elsewhere():
    class MyCtx(ToolContext[Db]):
        pass

    def first(ctx: MyCtx, x: int) -> int:
        return x

    def later(x: int, ctx: MyCtx) -> int:
        return x

    assert function_schema(first).takes_context is True
    with pytest.raises(TypeError, match="position 2"):
        function_schema(later)


def test_leading_underscore_parameter_is_rejected():
    def f(_x: int) -> int:
        return _x

    with pytest.raises(TypeError, match="'f'"):
        function_schema(f)


def test_async_functions_are_rejected():
    async def f(x: int) -> int:
        return x

    class Session:
        async def lookup(self, key: str) -> str:
            return key

    with pytest.raises(TypeError, match="synchronously"):
        function_schema(f)
    with pytest.raises(TypeError, match="synchronously"):
        function_schema(Session().lookup)


def test_variadic_items_keep_only_annotated_constraints():
    def f(*args: Annotated[int, Field(ge=1, description="each")]) -> None:
        return None

    def g(**kwargs: Annotated[int, Field(ge=1, description="each")]) -> None:
        return None

    args_prop = function_schema(f).params_json_schema["properties"]["args"]
    assert args_prop["items"] == {"type": "integer", "minimum": 1}
    kwargs_prop = function_schema(g).params_json_schema["properties"]["kwargs"]
    assert kwargs_prop["additionalProperties"] == {"type": "integer", "minimum": 1}


def test_variadic_field_default_is_dropped_without_a_warning():
    def f(*args: Annotated[int, Field(default=5)]) -> None:
        return None

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        schema = function_schema(f)
        prop = schema.params_json_schema["properties"]["args"]

    assert caught == []
    assert prop["items"] == {"type": "integer"}


def test_parameter_pydantic_drops_is_rejected():
    def f(model_config: str) -> None:
        return None

    with pytest.raises(TypeError, match="'f'.*'model_config'"):
        function_schema(f)


class _NoName:
    def __call__(self, x: int) -> int:
        return x


def test_callable_without_a_name_needs_an_override():
    with pytest.raises(TypeError, match="name_override"):
        function_schema(_NoName())


def test_nameless_callable_errors_use_the_override():
    def add(x: int, y: int) -> int:
        return x + y

    with pytest.raises(TypeError, match="'adder'"):
        function_schema(functools.partial(add, y=1), name_override="adder")


class Order(BaseModel):
    symbol: str


def test_model_arguments_arrive_as_model_instances():
    def f(order: Order, orders: list[Order]) -> None:
        return None

    schema = function_schema(f)
    data = schema.params_model.model_validate(
        {"order": {"symbol": "A"}, "orders": [{"symbol": "B"}]}
    )

    args, kwargs = schema.to_call_args(data)
    assert kwargs == {}
    assert args[0] == Order(symbol="A")
    assert args[1] == [Order(symbol="B")]


def test_positional_only_name_may_be_a_var_keyword_key():
    def f(a: int, /, **kw: int) -> None:
        return None

    schema = function_schema(f)

    assert schema.to_call_args(schema.params_model(a=1, kw={"a": 9})) == ([1], {"a": 9})


def test_annotated_with_several_metadata_items():
    def f(n: Annotated[int, "first", Field(ge=1), "second", Field(le=5)]) -> int:
        return n

    prop = function_schema(f).params_json_schema["properties"]["n"]

    assert (prop["description"], prop["minimum"], prop["maximum"]) == ("first", 1, 5)


class _LegacyOrder(BaseModelV1):
    symbol: str


@pytest.mark.parametrize("annotation", [_LegacyOrder, Optional[_LegacyOrder], list[_LegacyOrder]])
def test_a_pydantic_v1_model_parameter_is_rejected_when_the_tool_is_built(annotation):
    def f(order: annotation) -> None:
        return None

    with pytest.raises(TypeError) as excinfo:
        function_schema(f, name_override="legacy")

    message = str(excinfo.value)
    assert "'legacy'" in message
    assert "'order'" in message
    assert "pydantic v2" in message
    assert "BaseTool" in message
    assert "args_schema" in message
