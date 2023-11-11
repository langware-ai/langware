from typing import Literal
from typing import Optional

from pydantic import BaseModel, Field, ConfigDict
from pydantic import model_validator


class OpenAIFunctionCall(BaseModel):
    """Received function call from assistant."""
    name: str = Field(...)
    arguments: str = Field(...)


class OpenAIChatMessage(BaseModel):
    """
    Represents a piece of chat message as per OpenAI Chat Completions API.
    """
    model_config = ConfigDict(extra="forbid")

    role: Literal["system", "user", "assistant", "function"] = Field(...)
    name: Optional[str] = Field(None, pattern=r'^[a-zA-Z0-9_-]{1,64}$')
    content: Optional[str] = Field(...)
    function_call: Optional[OpenAIFunctionCall] = Field(None)

    @model_validator(mode='before')
    @classmethod
    def validator(cls, values):
        role, name, content, function_call = values.get('role'), values.get('name'), values.get('content'), values.get('function_call')
        if content is None:
            if role != "assistant":
                raise ValueError('null content allowed only for role "assistant"')
            if function_call is None:
                raise ValueError('content is null and function_call is null')
        if role == "function" and not name:
            raise ValueError('name is required for role "function"')
        if content and function_call:
            raise ValueError('content and function_call cannot both be provided')
        return values

    def chatml_str(self, *, pretty=False) -> str:
        """
        Unsafe string representation of the message in ChatML v0 format.
        """
        if pretty:
            import colorama
            from colorama import Fore, Style
            colorama.init()
            role_color = {
                "system": Fore.CYAN,
                "user": Fore.GREEN,
                "assistant": Fore.BLUE,
                "function": Fore.YELLOW,
            }[self.role]
            return f"{role_color}{Style.BRIGHT}<|im_start|>{Style.RESET_ALL}{role_color}" + self.role + (f" name={self.name}" if self.name else "") + (f" to={self.function_call.name}" if self.function_call else "") + "\n" + (self.content if self.content else "") + (self.function_call.arguments if self.function_call else "") + f"{Style.BRIGHT}<|im_end|>{Style.RESET_ALL}"
        return "<|im_start|>" + self.role + (f" {self.name}" if self.name else "") + (f" {self.function_call.name}" if self.function_call else "") + "\n" + (self.content if self.content else "") + (self.function_call.arguments if self.function_call else "") + "<|im_end|>"
