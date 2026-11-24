from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class LLMSendMessageTypeEnum(StrEnum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class UserPromptSchema(BaseModel):
    role: LLMSendMessageTypeEnum
    content: str

    model_config = ConfigDict(use_enum_values=True)


class ResultMessage(BaseModel):
    done: bool = False
    chunk: str | None = None
    state: str | None = None
    error: Any | None = None
    metadata: Any | None = Field(default=None)
