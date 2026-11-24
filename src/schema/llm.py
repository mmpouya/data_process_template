from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel


class LLMSendMessageTypeEnum(StrEnum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class UserPromptSchema(BaseModel):
    role: LLMSendMessageTypeEnum
    content: str

    class Config:
        use_enum_values = True


class Skill(BaseModel):
    """A markdown skill resource."""

    name: str
    path: Path
    content: str
    description: str | None = None
