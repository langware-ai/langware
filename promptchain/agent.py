import asyncio
import difflib
import inspect
import json
from dataclasses import dataclass, asdict, is_dataclass
from typing import Optional, Sequence, Any, List, Dict, Callable, Awaitable, TypeVar, Generic, Union, Iterable, \
    MutableSequence

import aiohttp
import sympy
from pydantic import BaseModel, TypeAdapter, ValidationError, Field, ConfigDict

from promptchain.logger import logger
from promptchain.module import OpenAIChatModule
from promptchain.prompt import OpenAIChatMessage
from promptchain.tool import OpenAIChatFunctions, Tools
from promptchain.utilities.common import MAPPING_DEFAULT


class ToolException(Exception):
    pass


class StopAgentException(ToolException):
    pass


class OpenAIChatFunctionsAgent(BaseModel):
    """
    An agent's chat that uses OpenAI Chat Functions API. Agent is an entity that chats with the model.
    """
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    tools: Tools = Field(...)
    prompt: MutableSequence[OpenAIChatMessage] = Field(...)
    module: OpenAIChatModule = Field(default_factory=OpenAIChatModule)
    session: aiohttp.ClientSession = Field(default_factory=aiohttp.ClientSession, exclude=True)

    def __enter__(self):
        raise RuntimeError("Use 'async with' instead of 'with'")

    def __exit__(self, exc_type, exc_val, exc_tb):
        raise RuntimeError("Use 'async with' instead of 'with'")

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    async def close(self):
        await self.session.close()

    @property
    def functions(self):
        return OpenAIChatFunctions.from_mapping(self.tools)

    async def on_prediction(
            self,
            prediction: OpenAIChatMessage,
            *,
            limit: int = 10,
            openai_chat_params: Optional[Dict[str, Any]] = None,
            aiohttp_params: Optional[Dict[str, Any]] = None,
    ) -> OpenAIChatMessage | Any:
        """
        Handles prediction from the model, evaluates tools and calls next prediction.
        """
        if limit <= 0:
            raise RuntimeError(f"Operation exceeded limit")

        if prediction.function_call is None:
            # No function call, perhaps a new message to the user.
            return prediction

        function_call = prediction.function_call

        # Find tool.
        tool_name = function_call.name
        if (tool := self.tools.get(tool_name, MAPPING_DEFAULT)) is MAPPING_DEFAULT:
            suggestions = difflib.get_close_matches(tool_name, self.tools.keys(), n=1, cutoff=0.6)
            tool_result = {
                "type": "ToolNotFoundError",
                "str": f"Tool '{tool_name!r}' not found." + (f" Did you mean '{suggestions[0]}'?" if suggestions else "")
            }
            message = OpenAIChatMessage(role="function", name="error", content=json.dumps(tool_result))
            return await self.predict(
                message,
                limit=limit - 1,
                openai_chat_params=openai_chat_params,
                aiohttp_params=aiohttp_params)

        # Call tool.
        try:
            tool_result = TypeAdapter(tool).validate_json(prediction.function_call.arguments)
            if inspect.isawaitable(tool_result):
                tool_result = await tool_result
        except ValidationError as e:
            result = {
                "type": "ValidationError",
                "errors": [{k: v for k, v in error.items() if k not in {"url", "loc", "ctx"}} for error in e.errors()],
            }
            message = OpenAIChatMessage(role="function", name="error", content=json.dumps(result))
            return await self.predict(
                message,
                limit=limit - 1,
                openai_chat_params=openai_chat_params,
                aiohttp_params=aiohttp_params)

        # Handle tool result.
        if tool_result is None:
            message = None
        elif isinstance(tool_result, str):
            message = OpenAIChatMessage(role="function", name=tool_name, content=tool_result)
        elif callable(tool_result):
            sig = inspect.signature(tool_result)
            params = {
                'agent': self,
                'prediction': prediction,
                'limit': limit,
                'openai_chat_params': openai_chat_params,
                'aiohttp_params': aiohttp_params
            }
            kwargs = {k: v for k, v in params.items() if k in sig.parameters}
            unexpected_args = set(sig.parameters.keys()) - set(params.keys())
            if unexpected_args:
                raise TypeError(f"Tool '{tool}' named '{tool_name}' have returned callable object that has unexpected argument(s): {', '.join(unexpected_args)}")

            result = tool_result(**kwargs)
            if inspect.isawaitable(result):
                result = await result
            return result
        elif isinstance(tool_result, Exception):
            result = {
                "type": type(tool_result).__name__,
                "str": str(tool_result)
            }
            message = OpenAIChatMessage(role="function", name="error", content=json.dumps(result))
        elif isinstance(tool_result, BaseModel):
            message = OpenAIChatMessage(role="function", name=tool_name, content=tool_result.model_dump_json())
        elif is_dataclass(tool_result):
            message = OpenAIChatMessage(role="function", name=tool_name, content=json.dumps(asdict(tool_result)))
        else:
            try:
                message = OpenAIChatMessage(role="function", name=tool_name, content=json.dumps(tool_result))
            except TypeError:
                message = OpenAIChatMessage(role="function", name=tool_name, content=str(tool_result))

        return await self.predict(
            message,
            limit=limit - 1,
            openai_chat_params=openai_chat_params,
            aiohttp_params=aiohttp_params)

    async def predict(
            self,
            add_prompt: Union[OpenAIChatMessage, Iterable[OpenAIChatMessage], None] = None,
            *,
            limit: int = 10,
            openai_chat_params: Optional[Dict[str, Any]] = None,
            aiohttp_params: Optional[Dict[str, Any]] = None,
    ) -> OpenAIChatMessage | Any:
        """
        Runs agent's chat loop. This is the entry point for handling user's messages.
        """
        if isinstance(add_prompt, OpenAIChatMessage):
            self.prompt.append(add_prompt)
        elif isinstance(add_prompt, Iterable):
            self.prompt.extend(add_prompt)

        prediction: OpenAIChatMessage = await self.module(
            session=self.session, messages=self.prompt, functions=self.functions,
            params=openai_chat_params, aiohttp_params=aiohttp_params
        )
        self.prompt.append(prediction)

        result = self.on_prediction(prediction, limit=limit,
                                    openai_chat_params=openai_chat_params,
                                    aiohttp_params=aiohttp_params)
        if inspect.isawaitable(result):
            result = await result
        return result
