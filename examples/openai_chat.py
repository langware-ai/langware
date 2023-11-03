import asyncio
import logging
import os
from datetime import datetime
from functools import wraps
from typing import Optional

import aiohttp
import dotenv
import sympy
from pydantic import Field, BaseModel

from promptchain.agent import OpenAIChatFunctionsAgent
from promptchain.logger import logger
from promptchain.module import OpenAIChatModule, OpenAIChatRetryModule
from promptchain.prompt import OpenAIChatMessage


# This is a tool that is available for model and agent to use. See more about [`Tool`](promptchain.tool.Tool) type.
def calculate(
        expression: str = Field(..., description="Expects SymPy-like expression."),
        rational: bool = Field(True, description="Simplify to rational numbers."),
        evaluate: Optional[bool] = Field(True, description="Whether to evaluate the expression to a number."),
) -> str | Exception:
    """SymPy computer algebra calculator."""
    try:
        # Note: sympify() should be used with trusted input.
        result = sympy.sympify(expression, rational=rational, evaluate=evaluate)
        if evaluate:
            result = sympy.N(result)
    except Exception as e:
        # Returned `Exception` objects by default behavior will be passed back to the agent's chat for auto repair.
        # Similar to how TypeChat by Microsoft works with malformed tool input.
        return e

    # If you need to control the agent's chat execution flow, you can return a callable object.
    # For instance, to stop the agent from running, and return directly from agent, you can do:
    # ```py
    # return lambda: result
    # ```
    # If you need to change parameters of next `predict`, you can return a callable object that takes arguments like `OpenAIChatFunctionsAgent.on_prediction` method do and call the `agent.predict` itself.
    # Example:
    # ```py
    # def result(agent: OpenAIChatFunctionsAgent, limit: int):
    #     agent.predict(OpenAIChatMessage(role="function", name="sympify" content=result), limit=limit - 1)
    # return result
    # ```
    result = str(result)
    return result


async def get_user_content(input_mock):
    print(">>> ", end="")
    while True:
        if input_mock and len(input_mock) > 0:
            user_content = input_mock.pop(0)
            print(user_content)
        else:
            user_content = input()
        if user_content != "":
            break
    return user_content


async def amain():
    # For demonstration purposes, we will use a logger to show what's happening.
    # logger.setLevel(logging.DEBUG)

    # Load environment variables.
    logger.info('Loading environment variables from .env file')
    dotenv.load_dotenv()

    assert "OPENAI_API_KEY" in os.environ, "Please set OPENAI_API_KEY environment variable."

    initial_prompt = [
        OpenAIChatMessage(role="system", content=f"""You are ChatGPT, a large language model trained by OpenAI, based on the GPT-4 architecture.
Knowledge cutoff: 2022-01
Current date: {datetime.now().strftime('%Y-%m-%d')}""")
    ]
    prompt = initial_prompt
    tools = {
        calculate.__name__: calculate,
    }
    input_mock = [
        "Hi!",
        "Calculate sin(1/3) and integral of 2*x + y for x from 1 to 3",
    ]

    # Overriding default OpenAIChatModule with OpenAIChatRetryModule in order to retry failed API calls.
    module = OpenAIChatRetryModule(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-3.5-turbo")
    timeout = aiohttp.ClientTimeout(total=600, connect=30, sock_read=10)
    # Depending on how stable your internet connection, you may want to increase `sock_read` timeout.
    # It is used as a timeout to read 1 delta chunk (new predicted tokens) from streamed connection.

    # For demonstration purposes, last agent instance will be stored here.
    last_agent: Optional[OpenAIChatFunctionsAgent] = None

    while True:
        user_content = await get_user_content(input_mock)

        # Handle user commands.
        if user_content == "help":
            print("Commands:\n"
                  "  help  - show this help\n"
                  "  exit  - exit"
                  "  reset - reset prompt\n"
                  "  dump  - dump prompt\n"
                  "  dump_agent - dump last agent's chat prompt\n"
                  "  +     - continue generation without adding user message to the prompt\n"
                  "  -     - regenerate last message\n"
                  "  <any other text> - add user message to the prompt and continue generation")
            continue
        elif user_content == "exit":
            break
        elif user_content == "reset":
            prompt = initial_prompt
            continue
        elif user_content == "dump":
            for msg in prompt:
                print(f"{msg!r}")
            continue
        elif user_content == "dump_agent":
            for msg in last_agent.prompt or []:
                print(f"{msg!r}")
            continue
        elif user_content == "+":
            # Continue generation without adding user message to the prompt.
            pass
        elif user_content == "-":
            # Regenerate.
            prompt.pop()
            pass
        else:
            # No command, just add user message to the prompt.
            user_message = OpenAIChatMessage(role="user", content=user_content)
            print(user_message.chatml_str(pretty=True))
            prompt.append(user_message)

        # Run agent's chat loop.
        async with OpenAIChatFunctionsAgent(tools=tools, prompt=prompt, module=module) as agent:
            last_agent = agent
            prediction = await agent.predict(limit=10,
                                             openai_chat_params={"temperature": 0, "stream": True},
                                             aiohttp_params={"timeout": timeout})

        # Add last prediction to the prompt. Or use `prompt.extend(agent.prompt)` to add all predictions.
        # Note that `agent.predict` may return result of some tool, so you may want to check if prediction is a `OpenAIChatMessage` instance. In our case, our tools don't have this behavior.
        assert isinstance(prediction, OpenAIChatMessage)
        prompt.append(prediction)
        print(prediction.chatml_str(pretty=True))


def main():
    asyncio.run(amain())


if __name__ == '__main__':
    main()
