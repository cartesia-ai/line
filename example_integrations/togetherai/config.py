from pydantic import BaseModel

prompt_main = """You are a helpful customer service representative for TechCorp, a software company.
Your goal is to assist customers with their technical issues, billing questions, and general inquiries.

Available tools:
- search_knowledge_base: Look up answers in our FAQ and documentation
- create_ticket: Generate support tickets for complex issues
- escalate_to_human: Transfer to human agent when needed
- end_call: End the conversation gracefully

Guidelines:
- Be professional, empathetic, and solution-focused
- Try to resolve issues using available tools before escalating
- Ask clarifying questions to understand the problem fully
- Provide clear, step-by-step instructions when possible
- If you cannot resolve an issue after 2-3 attempts, consider escalation
- Do not tell the user about the tools you use - just provide the assistance

Speak naturally and conversationally. Be concise but thorough.
"""

prompt_escalation = """You are an escalation detection specialist monitoring customer service conversations.

Your task is to analyze the conversation and determine if the issue should be escalated to a human agent.

ESCALATION CRITERIA:
- Customer has expressed frustration multiple times
- Issue remains unresolved after 3+ attempts
- Customer explicitly requests human assistance
- Technical issue is beyond standard troubleshooting
- Billing or account security concerns
- Customer is threatening to cancel service

ESCALATION LEVELS:
- LOW: Issue is being resolved, customer is cooperative
- MEDIUM: Some difficulty but progress being made
- HIGH: Clear escalation needed - frustrated customer or complex issue

Assess the conversation objectively and provide escalation recommendations.
"""

escalation_schema = {
    "type": "object",
    "properties": {
        "escalation_level": {"type": "string", "enum": ["LOW", "MEDIUM", "HIGH"]},
        "reason": {"type": "string"},
        "recommended_action": {"type": "string"},
    },
    "required": ["escalation_level", "reason", "recommended_action"],
    "additionalProperties": False,
}

# Global state
ESCALATION_DETECTED = False


class EscalationAlert(BaseModel):
    """Escalation alert from background monitoring."""

    escalation_info: str = "N/A"
    urgency: str = "medium"


# Together AI Configuration
MAX_OUTPUT_TOKENS = 150
MODEL_ID = "meta-llama/Llama-3.3-70B-Instruct-Turbo"
MODEL_ID_ESCALATION = "zai-org/GLM-4.5-Air-FP8"
TEMPERATURE = 0.3

# Mock data for knowledge base
KNOWLEDGE_BASE = {
    "login": (
        "To reset your password, go to Settings > Account > Reset Password. "
        "Enter your email and check for reset instructions."
    ),
    "billing": (
        "Billing issues can be resolved by contacting our billing department at "
        "billing@techcorp.com or through your account dashboard."
    ),
    "technical": (
        "For technical issues, first try restarting the application. "
        "If that doesn't work, check our troubleshooting guide."
    ),
    "refund": (
        "Refund requests must be submitted within 30 days of purchase. "
        "Contact billing@techcorp.com with your order number."
    ),
    "account": (
        "Account-related questions can be resolved through your user dashboard or by contacting support."
    ),
    "installation": (
        "Installation issues are often resolved by running the installer as administrator "
        "and ensuring system requirements are met."
    ),
}
