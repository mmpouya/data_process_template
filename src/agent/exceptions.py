class AgentError(Exception):
    """Base exception for all agent module errors."""


class AgentPromptError(AgentError):
    """Raised when a prompt file, key, or variable is missing or invalid."""


class AgentSkillError(AgentError):
    """Raised when a skill cannot be found or loaded."""


class AgentHarnessError(AgentError):
    """Raised when an error raised by the framework"""


class AgentToolError(AgentError):
    """Raised when an error raised by a tool. mostly the cause is not the model but the executor"""
