import asyncio
import os
from datetime import datetime
from typing import Optional

import aiohttp
import dotenv
import sympy
from pydantic import Field, BaseModel

from langware.logger import logger
from langware.prompt import OpenAIChatMessage
from langware.model import OpenAIChatAPIRetryModel
from langware.chat import OpenAIFunctionsAPIChat


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
    #     agent.predict(OpenAIChatMessage(role="function", name="sympify" content=str(result)), limit=limit - 1)
    # return result
    # ```
    return str(result)


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

    initial_messages = [
        OpenAIChatMessage(role="system", content=f"""You are ChatGPT, a large language model trained by OpenAI, based on the GPT-4 architecture.
Knowledge cutoff: 2022-01
Current date: {datetime.now().strftime('%Y-%m-%d')}""")
    ]
    messages = initial_messages
    functions = {
        calculate.__name__: calculate,
    }
    input_mock = [
        "Hi!",
        "Calculate sin(1/3) and integral of 2*x + y for x from 1 to 3",
        "dump_agent",
        "dump",
        "help",
    ]

    session = aiohttp.ClientSession()

    # Depending on how stable your internet connection, you may want to increase `sock_read` timeout.
    # It is used as a timeout to read 1 delta chunk (new predicted tokens) from streamed connection.
    timeout = aiohttp.ClientTimeout(total=600, connect=30, sock_read=10)

    # Overriding default OpenAIChatModule with OpenAIChatRetryModule in order to retry failed API calls.
    model = OpenAIChatAPIRetryModel(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-3.5-turbo")
    chat: Optional[OpenAIFunctionsAPIChat] = None

    while True:
        user_content = await get_user_content(input_mock)

        # Handle user commands.
        if user_content == "help":
            print("Commands:\n"
                  "  help  - show this help\n"
                  "  exit  - exit\n"
                  "  reset - reset messages\n"
                  "  dump  - dump messages\n"
                  "  dump_agent - dump last agent's chat messages\n"
                  "  +     - continue generation without adding user message to the messages\n"
                  "  -     - regenerate last message\n"
                  "  <any other text> - add user message to the messages and continue generation")
            continue
        elif user_content == "exit":
            break
        elif user_content == "reset":
            messages = initial_messages
            continue
        elif user_content == "dump":
            for msg in messages:
                print(f"{msg!r}")
            continue
        elif user_content == "dump_agent":
            for msg in chat.messages or []:
                print(f"{msg!r}")
            continue
        elif user_content == "+":
            # Continue generation without adding user message to the messages.
            pass
        elif user_content == "-":
            # Regenerate.
            messages.pop()
            pass
        else:
            # No command, just add user message to the messages.
            user_message = OpenAIChatMessage(role="user", content=user_content)
            print(user_message.chatml_str(pretty=True))
            messages.append(user_message)

        # Run agent's chat loop.
        chat = OpenAIFunctionsAPIChat(functions=functions, messages=messages, model=model, session=session)
        prediction = await chat.predict(limit=10,
                                         openai_chat_params={"temperature": 0, "stream": True},
                                         aiohttp_params={"timeout": timeout})

        # Add last prediction to the messages. Or use `messages.extend(agent.messages)` to add all predictions.
        # Note that `agent.predict` may return result of some tool, so you may want to check if prediction is a `OpenAIChatMessage` instance. In our case, our tools don't have this behavior.
        assert isinstance(prediction, OpenAIChatMessage)
        messages.append(prediction)
        print(prediction.chatml_str(pretty=True))

    await session.close()


def main():
    asyncio.run(amain())


if __name__ == '__main__':
    main()

"""
>>> Hi!
<|im_start|>user
Hi!<|im_end|>
<|im_start|>assistant
Hello! How can I assist you today?<|im_end|>
>>> Calculate sin(1/3) and integral of 2*x + y for x from 1 to 3
<|im_start|>user
Calculate sin(1/3) and integral of 2*x + y for x from 1 to 3<|im_end|>
<|im_start|>assistant
The value of sin(1/3) is approximately 0.3272.

The integral of 2*x + y with respect to x, from 1 to 3, is 2.0*y + 8.0.<|im_end|>
>>> dump_agent
OpenAIChatMessage(role='system', name=None, content='You are ChatGPT, a large language model trained by OpenAI, based on the GPT-4 architecture.\nKnowledge cutoff: 2022-01\nCurrent date: 2023-11-04', function_call=None)
OpenAIChatMessage(role='user', name=None, content='Hi!', function_call=None)
OpenAIChatMessage(role='assistant', name=None, content='Hello! How can I assist you today?', function_call=None)
OpenAIChatMessage(role='user', name=None, content='Calculate sin(1/3) and integral of 2*x + y for x from 1 to 3', function_call=None)
OpenAIChatMessage(role='assistant', name=None, content=None, function_call=OpenAIChatFunctionCall(name='calculate', arguments='{\n  "expression": "sin(1/3)"\n}'))
OpenAIChatMessage(role='function', name='calculate', content='0.327194696796152', function_call=None)
OpenAIChatMessage(role='assistant', name=None, content=None, function_call=OpenAIChatFunctionCall(name='calculate', arguments='{\n  "expression": "integrate(2*x + y, (x, 1, 3))"\n}'))
OpenAIChatMessage(role='function', name='calculate', content='2.0*y + 8.0', function_call=None)
OpenAIChatMessage(role='assistant', name=None, content='The value of sin(1/3) is approximately 0.3272.\n\nThe integral of 2*x + y with respect to x, from 1 to 3, is 2.0*y + 8.0.', function_call=None)
>>> dump
OpenAIChatMessage(role='system', name=None, content='You are ChatGPT, a large language model trained by OpenAI, based on the GPT-4 architecture.\nKnowledge cutoff: 2022-01\nCurrent date: 2023-11-04', function_call=None)
OpenAIChatMessage(role='user', name=None, content='Hi!', function_call=None)
OpenAIChatMessage(role='assistant', name=None, content='Hello! How can I assist you today?', function_call=None)
OpenAIChatMessage(role='user', name=None, content='Calculate sin(1/3) and integral of 2*x + y for x from 1 to 3', function_call=None)
OpenAIChatMessage(role='assistant', name=None, content='The value of sin(1/3) is approximately 0.3272.\n\nThe integral of 2*x + y with respect to x, from 1 to 3, is 2.0*y + 8.0.', function_call=None)
>>> help
Commands:
  help  - show this help
  exit  - exit
  reset - reset prompt
  dump  - dump prompt
  dump_agent - dump last agent's chat prompt
  +     - continue generation without adding user message to the prompt
  -     - regenerate last message
  <any other text> - add user message to the prompt and continue generation
>>> 
"""
