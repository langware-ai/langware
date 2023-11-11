from functools import partial
import asyncio
import inspect
from collections.abc import Callable
from inspect import Parameter
from typing import Optional, Union, Mapping, Any, Iterable, Awaitable, cast

from pydantic import BaseModel, Field, RootModel, create_model, ConfigDict
from pydantic.config import ExtraValues
from pydantic.fields import FieldInfo
from pydantic.json_schema import JsonSchemaValue


Function = Union[Callable[..., Any], type[BaseModel]]
"""
The `Function` type represents either a function or a Pydantic `BaseModel` subclass that can be used in the prompt of the model.

Examples:
    - `function`s are `Function`s:
    
    >>> def simple_function(arg1: int, arg2: int) -> str:
    ...     return f"Arguments received: {arg1}, {arg2}"
    ...
    >>> simple_function(1, 2)
    'Arguments received: 1, 2'
    
    - Any instances of `Callable` are `Function`s:
    
    >>> class CallableTool:
    ...     __name__ = "CallableTool"
    ...     def __call__(self, arg1: int, arg2: int) -> str:
    ...         return f"Arguments received: {arg1}, {arg2}"
    ...
    >>> callable_tool = CallableTool()
    >>> callable_tool(1, 2)
    'Arguments received: 1, 2'
    
    - Pydantic BaseModel are `Function`s:
    
    >>> class MyModel(BaseModel):
    ...     field1: int
    ...     field2: int
    ...
    >>> MyModel(field1=1, field2=2)
    MyModel(field1=1, field2=2)
    
    - A more advanced example of stateful and asynchronous `Function`:
    
    >>> class SearchTool(BaseModel):
    ...     __name__ = "SearchTool"
    ...     api_key: str
    ...     def __call__(self, query: str) -> Awaitable[list[str]]:
    ...         return self.acall(query=query)
    ...     async def acall(self, query: str) -> list[str]:
    ...         await asyncio.sleep(2)
    ...         return ["result1", "result2"]
    ...
    >>> search = SearchTool(api_key="123")
    >>> asyncio.run(search(query="test"))
    ['result1', 'result2']
"""

Functions = dict[str, Function]
"""
The `Functions` dictionary is a mapping of tool names to their respective `Function` implementations. 
"""


class OpenAIChatAPIFunction(BaseModel):
    """
    One single function as defined by OpenAI Chat API.
    The OpenAI API sees the [`Function`](promptchain.function.Function) as a JSON Schema object. You can see what model sees by running
    [`OpenAIChatFunction.from_callable(calculate).model_dump_json(indent=2))`](promptchain.prompt.OpenAIChatFunction.from_tool)

    Examples:
        >>> class MyModel(BaseModel):
        ...     "model_docstring"
        ...     field1: int
        ...     field2: int = Field(2, title="model_field with title", description="model_field with description")
        ...
        >>> def my_function(
        ...         arg1: int,
        ...         arg2: int = Field(2, title="function_field with title", description="function_field with description")
        ... ):
        ...     "function_docstring"
        ...     return f"{arg1} {arg2}"
        ...
        >>> OpenAIChatAPIFunction.from_function(MyModel)
        OpenAIChatFunction(name='MyModel', description='model docstring', parameters={'title': 'model_field with title', 'description': 'model_field with description', 'default': 2, 'type': 'integer'})
        >>> OpenAIChatAPIFunction.from_function(my_function)
        OpenAIChatFunction(name='my_function', description='function_docstring', parameters={'title': 'function_field with title', 'description': 'function_field with description', 'default': 2, 'type': 'integer'})
    """
    name: str = Field(..., pattern=r'^[a-zA-Z0-9_-]{1,64}$')
    description: Optional[str] = Field(None)
    parameters: JsonSchemaValue = Field({})

    @classmethod
    def from_function(cls, tool: Function, *, name: Optional[str] = None, description: Optional[str] = None) -> "OpenAIChatAPIFunction":
        if isinstance(tool, type) and issubclass(tool, BaseModel):
            return cls.from_model(tool, name=name, description=description)
        elif callable(tool):
            return cls.from_callable(tool.__class__.__name__ if isinstance(tool, type) else tool.__name__,
                                     tool,
                                     description=description)
        else:
            raise ValueError(f"Unknown type {type(tool)}")

    @classmethod
    def from_model(cls, model: type[BaseModel], *, name: Optional[str] = None, description: Optional[str] = None, parameters: Optional[JsonSchemaValue] = None) -> "OpenAIChatAPIFunction":
        name = name or model.__name__
        description = description or model.__doc__
        parameters = parameters or model.model_json_schema(mode="serialization")
        return cls(
            name=name,
            description=description,
            parameters=parameters
        )

    @classmethod
    def from_callable(cls, name: str, callable: Callable[..., Any], *, description: Optional[str] = None) -> "OpenAIChatAPIFunction":
        description = description or callable.__doc__

        sig = inspect.signature(callable)

        model_config = ConfigDict()
        model_config["extra"] = cast(ExtraValues, "forbid")
        field_definitions: dict[str, tuple[type, FieldInfo]] = {}
        for param_name, param in sig.parameters.items():
            if param.kind in [Parameter.VAR_POSITIONAL, Parameter.VAR_KEYWORD]:
                if param.kind == Parameter.VAR_KEYWORD:
                    model_config["extra"] = cast(ExtraValues, "allow")
                continue

            if isinstance(param.default, FieldInfo):
                field_definitions[param_name] = (param.annotation, param.default)
                continue

            default_value = param.default if param.default != param.empty else ...
            field_annotation = param.annotation if param.annotation != param.empty else type(default_value) if default_value is not ... else Any
            field_value = Field(default=default_value, description=None)

            field_definitions[param_name] = (field_annotation, field_value)

        model = create_model(
            name,
            __config__=model_config,
            **field_definitions
        )
        parameters = model.model_json_schema(mode="serialization")

        return cls(
            name=name,
            description=description,
            parameters=parameters
        )


class OpenAIChatFunctions(list[OpenAIChatAPIFunction]):
    @classmethod
    def from_iterable(cls, tools: Iterable[Function]):
        return cls([
            OpenAIChatAPIFunction.from_tool(tool)
            for tool in tools
        ])

    @classmethod
    def from_mapping(cls, tools: Mapping[str, Function]):
        return cls([
            OpenAIChatAPIFunction.from_callable(name, tool)
            for name, tool in tools.items()
        ])
