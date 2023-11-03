import asyncio
import copy
from collections.abc import Callable
from typing import AsyncGenerator, AsyncIterator, Dict, Iterator, Generator, Union, Any, Optional, Mapping, Iterable, \
    Collection, Container, Sequence


# Used as a default value for dict.get() to distinguish missing key, as an alternative to None.
MAPPING_DEFAULT = object()


def _dict_filter_some(**kwargs: Any) -> dict[Any, Any]:
    """ Make a dict from keyword arguments, removing all None values. """
    return {k: v for k, v in kwargs.items() if v is not None}


def _copy_filter_some_deep(a: Union[dict, list, set, frozenset, tuple, Any]) -> Any:
    """ Shallow copy removing all None values from: dict, list, set, frozenset, tuple. """
    if isinstance(a, dict):
        return {k: _copy_filter_some_deep(v) for k, v in a.items() if v is not None}
    elif isinstance(a, (list, set, frozenset, tuple)):
        return type(a)((_copy_filter_some_deep(v) for v in a if v is not None))
    return a


def collect(collected: dict, delta: dict) -> dict:
    if not delta:
        return collected

    for key, value in delta.items():
        if key in collected and isinstance(collected[key], dict) and isinstance(value, dict):
            collected[key] = collect(collected[key], value)
        elif key in collected and isinstance(collected[key], str) and isinstance(value, str):
            collected[key] += value
        else:
            collected[key] = value

    return collected


async def parse_sse(response: AsyncIterator[bytes]) -> AsyncGenerator[dict[bytes, bytes], None]:
    """
    Parse server-sent events from a response generator.
    Specs: https://html.spec.whatwg.org/multipage/server-sent-events.html#server-sent-events
    """
    buffer = b''
    data: Dict[bytes, bytes] = {}

    async for chunk in response:
        buffer += chunk
        while b'\n\n' in buffer:  # check if a full message is available
            message, buffer = buffer.split(b'\n\n', 1)  # split off the first message

            for line in message.split(b'\n'):
                # Ignore comments and empty lines
                if line.startswith(b':') or not line:
                    continue

                key, value = line.split(b':', 1)
                if value.startswith(b' '):
                    value = value[1:]
                if key == b'data' and key in data:
                    data[key] += b'\n' + value
                else:
                    data[key] = value

            if data:  # if we have any data, yield it and reset for the next message
                yield data
                data = {}


def clamp(min_value, value, max_value):
    if min_value > max_value:
        min_value, max_value = max_value, min_value
    return max(min_value, min(value, max_value))


async def main():
    async def mock_response() -> AsyncGenerator[bytes, None]:
        """
        Mock async generator that simulates the behavior of an SSE stream.
        """
        data = [
            b'data: This is the first message\n\n',
            b'data: This is the second message\n',
            b'id: 12345\n',
            b'data: with two lines an id field\n\n',
            b'event: custom_event\n',
            b'data: Custom event data\n\n'
        ]
        for d in data:
            yield d
            await asyncio.sleep(0.5)

    async for event in parse_sse(mock_response()):
        print(event)


if __name__ == '__main__':
    asyncio.run(main())


# deltas = [
#     {"role": "assistant", "content": "" },
#     {"content": "Hello"},
#     {"content": " how"},
#     {"content": " are"},
#     {"content": " you"},
#     {"content": "?"},
#     {}
# ]
#
# deltas2 = [
#     {
#         "role": "assistant",
#         "content": None,
#         "function_call": {
#             "name": "sympify",
#             "arguments": ""
#         }
#     },
#     {"function_call": {"arguments": "{\n"}},
#     {"function_call": {"arguments": " "}},
#     {"function_call": {"arguments": " \""}},
#     {"function_call": {"arguments": "expression"}},
#     {"function_call": {"arguments": "\":"}},
#     {"function_call": {"arguments": " \""}},
#     {"function_call": {"arguments": "sin"}},
#     {"function_call": {"arguments": "("}},
#     {"function_call": {"arguments": "1"}},
#     {"function_call": {"arguments": "/"}},
#     {"function_call": {"arguments": "3"}},
#     {"function_call": {"arguments": ")\"\n"}},
#     {"function_call": {"arguments": "}"}},
#     {}
# ]
#
# collected = {}
# for delta in deltas:
#     collected = collect(collected, delta)
#
# print(collected)
