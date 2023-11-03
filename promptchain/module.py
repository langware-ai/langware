import asyncio
import json
import os
import random
from abc import abstractmethod
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Sequence, Optional, Dict, Union, Iterable, Awaitable

import aiohttp
from pydantic import BaseModel, Field, field_validator, RootModel, SecretStr, ConfigDict

from promptchain.logger import logger
from promptchain.prompt import OpenAIChatMessage
from promptchain.tool import OpenAIChatFunction

from promptchain.utilities.common import collect, parse_sse, clamp


class Module(BaseModel):
    """
    Module is an entity that interacts with the neural networks via their API using aiohttp,
    or subprocess using asyncio (llama.cpp), or Torch model using PyTorch.
    Only OpenAI Chat Completions API is supported for now.
    """
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    # modules: OrderedDict[str, "Module"] = Field(default_factory=OrderedDict, exclude=True)
    #
    # def __setattr__(self, name, value):
    #     # Custom behavior: if value is an instance of Module, register it.
    #     if isinstance(value, Module):
    #         self.modules[name] = value
    #     # Default behavior: set the attribute normally.
    #     super().__setattr__(name, value)

    @abstractmethod
    def __call__(self, *args, **kwargs) -> Any:
        pass


class OpenAIChatModule(Module):
    """
    A module that uses OpenAI Chat Completions API.
    See [OpenAI Chat Completions API](https://platform.openai.com/docs/api-reference/chat) for available parameters.
    """
    model_config = ConfigDict(extra="forbid")

    api_key: str = Field(default_factory=lambda: os.getenv("OPENAI_API_KEY"))
    model: str = "gpt-3.5-turbo"

    @field_validator("api_key")
    @classmethod
    def check_api_key(cls, v: Any) -> Any:
        if not v:
            raise ValueError("API key is not set")
        return v

    def headers(self, stream: bool) -> Dict[str, str]:
        return {
            **({"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}),
            "Content-Type": "application/json",
            **({"Accept": "text/event-stream"} if stream else {}),
            "OpenAI-Debug": "true",
        }

    def __call__(
            self,
            session: aiohttp.ClientSession,
            messages: Iterable[OpenAIChatMessage],
            functions: Optional[Iterable[OpenAIChatFunction]] = None,
            params: Optional[Dict[str, Any]] = None,
            aiohttp_params: Optional[Dict[str, Any]] = None
    ) -> Awaitable[OpenAIChatMessage]:
        return self.acall(session, messages, functions, params, aiohttp_params)

    async def acall(
            self,
            session: aiohttp.ClientSession,
            messages: Iterable[OpenAIChatMessage],
            functions: Optional[Iterable[OpenAIChatFunction]] = None,
            params: Optional[Dict[str, Any]] = None,
            aiohttp_params: Optional[Dict[str, Any]] = None
    ) -> OpenAIChatMessage:
        data_messages = RootModel(messages).model_dump(exclude_defaults=True)
        data_functions = RootModel(functions).model_dump(exclude_defaults=True) if functions else None

        method = "POST"
        url = "https://api.openai.com/v1/chat/completions"
        stream = params.get("stream", False) if params else False
        headers = self.headers(stream)
        data = {
            "model": self.model,
            "messages": data_messages,
            **({"functions": data_functions} if data_functions else {}),
            **(params if params else {}),
        }

        request_id = random.randint(0, 1000000)
        logger.debug(f"Request {request_id:06d}: Fetching: method: {method}, url: {url}, headers: {headers}, data: {data}, aiohttp_kwargs: {aiohttp_params}")

        message = {}
        async with session.request(method, url, headers=headers, json=data, **(aiohttp_params if aiohttp_params else {})) as response:
            logger.debug(f"Request {request_id:06d}: Response: {repr(response)}")

            if response.status != 200:
                raise Exception(f"Request failed with status {response.status}: {response.reason}")

            if stream:
                async for event in parse_sse(response.content.iter_any()):
                    if event[b'data'] == b'[DONE]':
                        break
                    completion = json.loads(event[b'data'])
                    if "error" in completion:
                        raise Exception(f"ChatAPI error: {completion['error']}")
                    choice = completion["choices"][0]
                    message = collect(message, choice["delta"])
            else:
                completions = await response.json()
                choice = completions["choices"][0]
                message = choice["message"]

        prediction = OpenAIChatMessage(**message)
        return prediction


class OpenAIChatRetryModule(OpenAIChatModule):
    """
    Same as [`OpenAIChatModule`](promptchain.module.OpenAIChatModule), but retries the request when it fails.
    With default parameters, wait schema is the following (in seconds):
    1 2 4 8 16 32 64 64 64 64
    """
    max_retries: int = 10
    base: float = 2
    backoff_factor: float = 1
    min_wait: float = 1
    max_wait: float = 64

    def __call__(self, *args, **kwargs):
        return self.acall(*args, **kwargs)

    async def acall(self, *args, **kwargs):
        retries = 0
        while retries < self.max_retries:
            try:
                return await super().acall(*args, **kwargs)
            except Exception as e:
                wait_time = clamp(self.min_wait, self.backoff_factor * (self.base ** retries), self.max_wait)
                logger.warning(f"Request failed with error: {e}, retries: {retries}, max_retries: {self.max_retries}, wait_time: {wait_time}")
                if retries >= self.max_retries:
                    raise e
                await asyncio.sleep(wait_time)
                retries += 1
