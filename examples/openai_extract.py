import asyncio
import logging
import os
from typing import Optional, Union, Sequence, Any, Literal, MutableSequence

import aiohttp
import dotenv
from pydantic import BaseModel, Field

from promptchain.agent import OpenAIChatFunctionsAgent
from promptchain.logger import logger
from promptchain.module import OpenAIChatModule, OpenAIChatRetryModule
from promptchain.prompt import OpenAIChatMessage
from promptchain.tool import Tools, Tool


class Address(BaseModel):
    country: str = Field(...)
    city: Optional[str] = Field(None)
    street: Optional[str] = Field(None)


class Person(BaseModel):
    name: str = Field(...)
    home_address: Address = Field(...)
    work_address: Optional[Address] = Field(None)


async def openai_chat_extract_pipeline(
        tool: Tool,
        prompt: MutableSequence[OpenAIChatMessage],
        *,
        module: Optional[OpenAIChatModule] = None,
        session: Optional[aiohttp.ClientSession] = None,
        openai_chat_params: Optional[dict[str, Any]] = None,
        aiohttp_params: Optional[dict[str, Any]] = None,
        ) -> Tool | None:
    """
    Casts text to selected Tool using OpenAI Chat Functions API.
    """

    # Make a tool that extracts an entity from a text. It does so by providing type hint for an entity that is required to be extracted from the text.
    def extract(entity: tool):
        """Extract an entity from a text."""

        # Returning a callable is a way to control the agent execution flow.
        # By returning a callable object, you can stop the agent from running, and return result of calling this object directly from the agent.
        # Also, callable can accept named arguments like `OpenAIChatFunctionsAgent.on_prediction` method does and call the `agent.predict` itself, if you need to change, for example, sampling parameters for next agent prediction.
        return lambda: entity

    tools = {
        extract.__name__: extract,
    }

    async with OpenAIChatFunctionsAgent(tools=tools,
                                        prompt=prompt,
                                        **({"module": module} if module else {}),
                                        **({"session": session} if session else {})
                                        ) as agent:
        extracted = await agent.predict(
            limit=5,
            openai_chat_params={
                "function_call": {"name": extract.__name__},
                **(openai_chat_params or {}),
            },
            aiohttp_params=aiohttp_params)
    return extracted


async def amain():
    # For demonstration purposes, we will use a logger to show what's happening.
    # logger.setLevel(logging.DEBUG)

    # Load environment variables.
    logger.info('Loading environment variables from .env file')
    dotenv.load_dotenv()

    assert "OPENAI_API_KEY" in os.environ, "Please set OPENAI_API_KEY environment variable."

    prompt = [
        OpenAIChatMessage(role="user", content=f"""Hello, my name is Ivan Stepanov, I live in Istanbul, 123 Sokak."""),
    ]

    # Overriding default OpenAIChatModule with OpenAIChatRetryModule in order to retry failed API calls.
    module = OpenAIChatRetryModule()
    timeout = aiohttp.ClientTimeout(total=600, connect=30, sock_read=10)
    # Depending on how stable your internet connection, you may want to increase `sock_read` timeout.
    # It is used as a timeout to read 1 delta chunk (new predicted tokens) from streamed connection.

    extracted = await openai_chat_extract_pipeline(Union[Person, Address], prompt,
                                                   module=module,
                                                   openai_chat_params={"temperature": 0, "stream": True},
                                                   aiohttp_params={"timeout": timeout})

    print(f"Extracted: {extracted!r}")

    assert extracted == Person(
        name="Ivan Stepanov",
        home_address=Address(
            country="Turkey",
            city="Istanbul",
            street="123 Sokak"
        ),
        work_address=None
    )


def main():
    asyncio.run(amain())


if __name__ == '__main__':
    main()
