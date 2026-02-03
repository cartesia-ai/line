import json
from typing import List

from line.events import (
    AgentResponse,
    ToolResult,
    UserTranscriptionReceived,
)
from line.tools.system_tools import EndCallArgs

end_call_schema = {
    "type": "function",
    "function": {
        "name": "end_call",
        "strict": True,
        "description": "Ends the call when the user says \
            they need to leave or they want to stop.",
        "parameters": {
            "type": "object",
            "properties": {
                "goodbye_message": {
                    "type": "string",
                    "description": EndCallArgs.model_fields["goodbye_message"].description,
                }
            },
            "required": ["goodbye_message"],
        },
    },
}
interview_schema = {
    "type": "function",
    "function": {
        "name": "start_interview",
        "strict": True,
        "description": "Starts the interview after user confirmation. \
            Call this when the user says they are ready to\
                  start the interview.",
        "parameters": {
            "type": "object",
            "properties": {
                "confirmed": {
                    "type": "boolean",
                    "description": "Set to true if the user confirms they\
                          are ready to begin the interview.",
                }
            },
            "required": ["confirmed"],
        },
    },
}


def make_table(json_str: str, name_str: str) -> str:
    """
    This method can be used to create output report text files.
    Parameters:
    - json_str: The json formatted data
    - name_str: Name of the node
    Returns:
    - A string that includes the extracted values.
    """

    data = json.loads(json_str)
    # header = f"Response Assessment"
    # separator = "-" * len(header)
    rows = "\n".join(f"{key:<12} | {value}" for key, value in data.items())

    # return f"\n{separator}\n{rows}"
    return rows + "\n"


def convert_messages_to_cs(messages: List[dict], sys_message: str) -> List:
    """
    Convert conversation messages to CS format

    Args:
        messages: List of conversation instances compatible with Line SDK

    Returns:
        List of Cerebras-formatted messages
    """

    cs_messages = [{"role": "system", "content": sys_message}]

    for message in messages:
        if isinstance(message, AgentResponse):
            cs_messages.append({"role": "assistant", "content": message.content})
        elif isinstance(message, UserTranscriptionReceived):
            cs_messages.append({"role": "user", "content": message.content})
        elif isinstance(message, ToolResult):
            cs_messages.append(
                {
                    "role": "system",
                    "content": f"The tool {message.tool_name} was called. \
                        Don't share this with the user.",
                }
            )
        else:
            continue

    return cs_messages
