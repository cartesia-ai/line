from config import END_CALL_TOOL_DESCRIPTION
from google.genai import types as gemini_types

END_CALL_TOOL = gemini_types.Tool(
    function_declarations=[
        gemini_types.FunctionDeclaration(
            name="end_call",
            description=END_CALL_TOOL_DESCRIPTION,
            parameters={
                "type": "object",
                "properties": {
                    "goodbye_message": {
                        "type": "string",
                        "description": "Say that is all you need and thank them for their time.",
                    }
                },
                "required": ["goodbye_message"],  # type: ignore
            },
        )
    ]
)
