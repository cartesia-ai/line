from abc import ABC, abstractmethod
from typing import Dict

try:
    from google.genai import types as gemini_types
except ImportError:
    gemini_types = None


class ToolDefinition(ABC):
    """Abstract base class for static tool definitions.

    This class should be implemented by all system tools. Each tool should define
    its name, description, and return type as class methods.
    """

    @classmethod
    @abstractmethod
    def name(cls) -> str:
        """Tool name for LLM usage."""
        pass

    @classmethod
    @abstractmethod
    def description(cls) -> str:
        """Tool description for LLM understanding."""
        pass

    @classmethod
    @abstractmethod
    def to_gemini_tool(cls) -> "gemini_types.Tool":
        """Map to Gemini tool format. https://ai.google.dev/gemini-api/docs/function-calling"""
        pass

    @classmethod
    @abstractmethod
    def to_openai_tool(cls) -> Dict[str, object]:
        """Map to OpenAI tool format. https://platform.openai.com/docs/guides/tools?tool-type=function-calling"""
        pass

    @classmethod
    @abstractmethod
    def to_anthropic_tool(cls) -> Dict[str, object]:
        """Map to Anthropic tool format. https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/overview"""
        pass
