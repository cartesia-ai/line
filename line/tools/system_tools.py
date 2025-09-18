"""System tool definitions for Cartesia Voice Agents SDK."""

from typing import AsyncGenerator, Dict, Union

from pydantic import BaseModel, Field

from line.events import AgentResponse, EndCall
from line.tools.tool_types import ToolDefinition

try:
    from google.genai import types as gemini_types
except ImportError:
    gemini_types = None


class EndCallArgs(BaseModel):
    """Arguments for the end_call tool."""

    goodbye_message: str = Field(description="The final message to say before ending the call")


class EndCallTool(ToolDefinition):
    """End call system tool definition.

    Usage example (Gemini):
    ```python
    self.generation_config = GenerateContentConfig(
        ...
        tools=[EndCallTool.to_gemini_tool()],
    )

    async def process_context(
        self, context: ConversationContext
    ) -> AsyncGenerator[Union[AgentResponse, EndCall], None]:
        ...
        function_call = <LLM function call request>
        if function_call.name == EndCallTool.name():
            goodbye_message = function_call.args.get("goodbye_message", "Goodbye!")
            args = EndCallArgs(goodbye_message=goodbye_message)
            async for item in end_call(args):
                yield item
    """

    @classmethod
    def name(cls) -> str:
        return "end_call"

    @classmethod
    def description(cls) -> str:
        return (
            "End the conversation with a goodbye message. "
            "Call this when the user says something 'goodbye' or something similar indicating they are ready "
            "to end the call."
            "Before calling this tool, do not send any text back, just use the goodbye_message field."
        )

    @classmethod
    def to_gemini_tool(cls) -> "gemini_types.Tool":
        """Convert to Gemini tool format"""

        return gemini_types.Tool(
            function_declarations=[
                gemini_types.FunctionDeclaration(
                    name=cls.name(),
                    description=cls.description(),
                    parameters={
                        "type": "object",
                        "properties": {
                            "goodbye_message": {
                                "type": "string",
                                "description": EndCallArgs.model_fields["goodbye_message"].description,
                            }
                        },
                        "required": ["goodbye_message"],
                    },
                )
            ]
        )

    @classmethod
    def to_openai_tool(cls) -> Dict[str, object]:
        """Convert to OpenAI tool format for Responses API.

        Note: This returns the format expected by OpenAI's Responses API,
        not the Chat Completions API format.
        """
        return {
            "type": "function",
            "name": cls.name(),
            "description": cls.description(),
            "parameters": {
                "type": "object",
                "properties": {
                    "goodbye_message": {
                        "type": "string",
                        "description": EndCallArgs.model_fields["goodbye_message"].description,
                    }
                },
                "required": ["goodbye_message"],
                "additionalProperties": False,
            },
            "strict": True,
        }


async def end_call(
    args: EndCallArgs,
) -> AsyncGenerator[Union[AgentResponse, EndCall], None]:
    """
    End the call with a goodbye message.

    Yields:
        AgentResponse: The goodbye message to be spoken to the user
        EndCall: Event to end the call
    """
    # Send the goodbye message
    yield AgentResponse(content=args.goodbye_message)

    # End the call
    yield EndCall()


class DTMFToolCall(ToolDefinition):
    """Arguments for the dtmf_tool_call tool."""

    @classmethod
    def name(cls) -> str:
        return "dtmf_tool_call"

    @classmethod
    def description(cls) -> str:
        return "Send a DTMF tone to the user. Use this when you find the appropriate selection and the voice system asks you to press a button"

    @classmethod
    def parameters_description(cls) -> str:
        return "The DTMF button to send"

    @classmethod
    def to_gemini_tool(cls) -> "gemini_types.Tool":
        """Convert to Gemini tool format"""
        return gemini_types.Tool(
            function_declarations=[
                gemini_types.FunctionDeclaration(
                    name=cls.name(),
                    description=cls.description(),
                    parameters={
                        "type": "object",
                        "properties": {
                            "button": {
                                "type": "string",
                                "description": cls.parameters_description(),
                                "enum": ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "*", "#"],
                            }
                        },
                        "required": ["button"],
                        "additionalProperties": False,
                        "strict": True,
                    },
                )
            ]
        )

    @classmethod
    def to_openai_tool(cls) -> Dict[str, object]:
        """Convert to OpenAI tool format for Responses API.

        Note: This returns the format expected by OpenAI's Responses API,
        not the Chat Completions API format.
        """
        return {
            "type": "function",
            "name": cls.name(),
            "description": cls.description(),
            "parameters": {
                "type": "object",
                "properties": {
                    "button": {
                        "type": "string",
                        "enum": ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "*", "#"],
                        "description": cls.parameters_description(),
                    },
                },
                "required": ["button"],
                "additionalProperties": False,
                "strict": True,
            },
        }


class DTMFToolCallTool(ToolDefinition):
    """DTMF tool call system tool definition."""
