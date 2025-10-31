from datetime import datetime
from typing import List
import uuid

from line.events import (
    AgentResponse,
    ToolResult,
    UserTranscriptionReceived,
)
from line.tools.system_tools import EndCallArgs

# Tool schemas compatible with OpenAI function calling
end_call_schema = {
    "type": "function",
    "function": {
        "name": "end_call",
        "description": "Ends the call when the conversation is complete or customer requests to end",
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

search_knowledge_base_schema = {
    "type": "function",
    "function": {
        "name": "search_knowledge_base",
        "description": "Search the company knowledge base for answers to customer questions",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "Search query to look up in knowledge base (e.g., 'login', 'billing', 'technical')"
                    ),
                }
            },
            "required": ["query"],
        },
    },
}

create_ticket_schema = {
    "type": "function",
    "function": {
        "name": "create_ticket",
        "description": "Create a support ticket for issues that need follow-up",
        "parameters": {
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "Brief title describing the issue"},
                "description": {
                    "type": "string",
                    "description": "Detailed description of the customer's issue",
                },
                "priority": {
                    "type": "string",
                    "enum": ["low", "medium", "high"],
                    "description": "Priority level for the ticket",
                },
            },
            "required": ["title", "description", "priority"],
        },
    },
}

escalate_to_human_schema = {
    "type": "function",
    "function": {
        "name": "escalate_to_human",
        "description": "Transfer the customer to a human agent",
        "parameters": {
            "type": "object",
            "properties": {
                "reason": {"type": "string", "description": "Reason for escalation to human agent"},
                "urgency": {
                    "type": "string",
                    "enum": ["standard", "urgent"],
                    "description": "Urgency level for human handoff",
                },
            },
            "required": ["reason", "urgency"],
        },
    },
}


def convert_messages_to_openai(messages: List[dict], sys_message: str) -> List:
    """
    Convert conversation messages to OpenAI format

    Args:
        messages: List of conversation instances compatible with Line SDK
        sys_message: System prompt for the conversation

    Returns:
        List of OpenAI-formatted messages
    """

    openai_messages = [{"role": "system", "content": sys_message}]

    for message in messages:
        if isinstance(message, AgentResponse):
            openai_messages.append({"role": "assistant", "content": message.content})
        elif isinstance(message, UserTranscriptionReceived):
            openai_messages.append({"role": "user", "content": message.content})
        elif isinstance(message, ToolResult):
            openai_messages.append(
                {
                    "role": "system",
                    "content": f"Tool {message.tool_name} executed. Result available for context.",
                }
            )
        else:
            continue

    return openai_messages


def search_knowledge_base(query: str) -> str:
    """
    Mock knowledge base search function

    Args:
        query: Search term

    Returns:
        Knowledge base result
    """
    from config import KNOWLEDGE_BASE

    query_lower = query.lower()
    for key, value in KNOWLEDGE_BASE.items():
        if key in query_lower:
            return f"Knowledge Base Result: {value}"

    return (
        "No relevant information found in knowledge base. "
        "Consider creating a support ticket for personalized assistance."
    )


def create_support_ticket(title: str, description: str, priority: str) -> str:
    """
    Mock ticket creation function

    Args:
        title: Ticket title
        description: Issue description
        priority: Priority level

    Returns:
        Ticket creation confirmation
    """
    ticket_id = str(uuid.uuid4())[:8].upper()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    return (
        f"Support ticket #{ticket_id} created successfully at {timestamp}. "
        f"Priority: {priority.upper()}. You will receive updates via email."
    )


def escalate_to_human_agent(reason: str, urgency: str) -> str:
    """
    Mock escalation function

    Args:
        reason: Escalation reason
        urgency: Urgency level

    Returns:
        Escalation confirmation
    """
    wait_time = "2-3 minutes" if urgency == "urgent" else "5-7 minutes"

    return (
        f"Transferring you to a human agent. Reason: {reason}. "
        f"Estimated wait time: {wait_time}. Please stay on the line."
    )


def format_escalation_report(escalation_data: dict) -> str:
    """
    Format escalation analysis for logging

    Args:
        escalation_data: Escalation analysis results

    Returns:
        Formatted report string
    """
    return f"""
Escalation Analysis:
- Level: {escalation_data.get("escalation_level", "Unknown")}
- Reason: {escalation_data.get("reason", "Not specified")}
- Action: {escalation_data.get("recommended_action", "No action specified")}
"""
