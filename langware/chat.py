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

from langware.logger import logger
from langware.model import OpenAIChatAPIModel, OpenAIChatAPIRetryModel
from langware.prompt import OpenAIChatMessage
from langware.function import Functions, OpenAIChatFunctions
from langware.utilities.common import sentinel


class OpenAIFunctionsAPIChat(BaseModel):
    """
    An autonomous chat that uses OpenAI Chat Functions API.
    """
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    functions: Functions = Field(...)
    messages: MutableSequence[OpenAIChatMessage] = Field(...)
    model: OpenAIChatAPIModel = Field(default_factory=OpenAIChatAPIModel)
    session: aiohttp.ClientSession = Field(..., exclude=True)

    @property
    def openai_functions(self):
        return OpenAIChatFunctions.from_mapping(self.functions)

    @staticmethod
    def result_to_message(
            function_name: str,
            function_result: Any,
    ) -> OpenAIChatMessage:
        if isinstance(function_result, str):
            message = OpenAIChatMessage(role="function", name=function_name, content=function_result)
        elif isinstance(function_result, BaseModel):
            message = OpenAIChatMessage(role="function", name=function_name, content=function_result.model_dump_json())
        elif is_dataclass(function_result):
            message = OpenAIChatMessage(role="function", name=function_name, content=json.dumps(asdict(function_result)))
        else:
            try:
                message = OpenAIChatMessage(role="function", name=function_name, content=json.dumps(function_result))
            except TypeError:
                result = {
                    "type": type(function_result).__name__,
                    "str": str(function_result)
                }
                message = OpenAIChatMessage(role="function", name=function_name, content=json.dumps(result))
        return message

    async def on_prediction(
            self,
            prediction: OpenAIChatMessage,
            *,
            limit: int = 10,
            openai_chat_params: Optional[Dict[str, Any]] = None,
            aiohttp_params: Optional[Dict[str, Any]] = None,
    ) -> OpenAIChatMessage | Any:
        """
        Handles prediction from the model, evaluates functions, appends result and calls next prediction.
        """
        logger.debug(f"on_prediction: prediction: {prediction!r}, limit: {limit}, openai_chat_params: {openai_chat_params!r}, aiohttp_params: {aiohttp_params!r}")

        if limit <= 0:
            raise RuntimeError(f"Operation exceeded limit")

        if prediction.function_call is None:
            # No function call, perhaps a new message to the user.
            return prediction

        function_call = prediction.function_call

        # Find function.
        function_name = function_call.name
        if (function := self.functions.get(function_name, sentinel)) is sentinel:
            suggestions = difflib.get_close_matches(function_name, self.functions.keys(), n=1, cutoff=0.6)
            function_result = {
                "type": "ToolNotFoundError",
                "str": f"Tool '{function_name!r}' not found." + (f" Did you mean '{suggestions[0]}'?" if suggestions else "")
            }

            message = self.result_to_message("error", function_result)
            self.messages.append(message)

            return await self.predict(
                limit=limit - 1,
                openai_chat_params=openai_chat_params,
                aiohttp_params=aiohttp_params)

        # Call function.
        try:
            # FIXME: Pydantic does not await coroutines defined in the function arguments, i.e. other tools, so we need to await them manually in the function body. At the function execution time, the function receives awaitable objects.
            #  Possible solutions: make an async validator, set attribute __pydantic_validator__.
            function_result = TypeAdapter(function).validate_json(prediction.function_call.arguments)
            if inspect.isawaitable(function_result):
                # # Get all arguments that are awaitable.
                # sig = inspect.signature(function)
                # awaitable_args = {k: v for k, v in function_result.items() if k in sig.parameters and inspect.isawaitable(v)}
                # # Wait for them.
                # for k, v in awaitable_args.items():
                #     function_result[k] = await v
                # # Call function.
                function_result = await function_result
        except ValidationError as e:
            logger.debug(f"ValidationError: {e}")

            result = {
                "type": "ValidationError",
                # "errors": [{k: v for k, v in error.items() if k not in {"url", "loc", "ctx"}} for error in e.errors()],
                "str": str(e)
            }

            message = OpenAIChatMessage(role="function", name="error", content=json.dumps(result))
            self.messages.append(message)

            return await self.predict(
                limit=limit - 1,
                openai_chat_params=openai_chat_params,
                aiohttp_params=aiohttp_params)

        # Handle function result.
        if callable(function_result):
            sig = inspect.signature(function_result)
            params = {
                'chat': self,
                'prediction': prediction,
                'limit': limit,
                'openai_chat_params': openai_chat_params,
                'aiohttp_params': aiohttp_params
            }
            kwargs = {k: v for k, v in params.items() if k in sig.parameters}
            unexpected_args = set(sig.parameters.keys()) - set(params.keys())
            if unexpected_args:
                raise TypeError(
                    f"Tool '{function}' named '{function_name}' have returned callable object that has unexpected argument(s): {', '.join(unexpected_args)}")

            result = function_result(**kwargs)
            if inspect.isawaitable(result):
                result = await result
            return result

        message = self.result_to_message(function_name, function_result)
        self.messages.append(message)

        return await self.predict(
            limit=limit - 1,
            openai_chat_params=openai_chat_params,
            aiohttp_params=aiohttp_params)

    async def predict(
            self,
            *,
            limit: int = 10,
            openai_chat_params: Optional[Dict[str, Any]] = None,
            aiohttp_params: Optional[Dict[str, Any]] = None,
    ) -> OpenAIChatMessage | Any:
        """
        Runs agent's chat loop. This is the entry point for handling user's messages.
        """
        logger.debug(f"predict: self.prompt: {self.messages!r}, limit: {limit}, openai_chat_params: {openai_chat_params!r}, aiohttp_params: {aiohttp_params!r}")
        prediction: OpenAIChatMessage = await self.model(
            session=self.session, messages=self.messages, openai_functions=self.openai_functions,
            params=openai_chat_params, aiohttp_params=aiohttp_params
        )
        self.messages.append(prediction)

        result = self.on_prediction(prediction, limit=limit,
                                    openai_chat_params=openai_chat_params,
                                    aiohttp_params=aiohttp_params)
        if inspect.isawaitable(result):
            result = await result
        return result
