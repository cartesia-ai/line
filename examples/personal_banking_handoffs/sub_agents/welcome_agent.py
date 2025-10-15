"""Welcome Agent for Personal Banking Customer Support.

This agent serves as the entry point for customer conversations and routes
them to appropriate specialized agents based on their initial request.
"""

from typing import AsyncGenerator, List, Optional, Union

from config import BANK_NAME
from context import BankContext
from google.genai import Client
from google.genai.types import (
    Content,
    FunctionDeclaration,
    GenerateContentConfig,
    Tool,
)
from loguru import logger
from prompts import make_voice_system_prompt

from line.events import AgentResponse, ToolCall, TransferCall
from line.tools.system_tools import TransferToolCall

from .sub_agent import SubAgent

# Welcome agent LLM settings
WELCOME_MODEL_ID = "gemini-2.5-flash"

# To test this, replace this the number you want to transfer to (e.g. a coworker's or another voice agent, if you're calling from your own phone)
REPRESENTATIVE_PHONE_NUMBER = "+12293047209"

# Welcome agent prompt
WELCOME_PROMPT = f"""
You are a {BANK_NAME} customer service representative routing system. Your role is to analyze customer requests and immediately route them to the appropriate specialist without providing any text response.

### Your Responsibilities:
1. **Analyze the customer request** - Understand what they need
2. **Route immediately** - Call the appropriate handoff function without any text response

### Routing Logic:
- **General questions** about {BANK_NAME} services, website help, or general information → Use `handoff_to_faq`
- **Banking needs** like account access, transactions, balances, transfers, or fraud reports → Use `handoff_to_transaction`

### IMPORTANT:
- Do NOT provide any text response
- Do NOT greet the customer
- Only call the appropriate handoff function
- The specialist you route to will handle the greeting and response

### Example Interactions:
**General Questions:**
Customer: "What are your current mortgage rates?"
You: [Call handoff_to_faq immediately - no text response]

**Banking Needs:**
Customer: "I need to check my account balance"
You: [Call handoff_to_transaction immediately - no text response]

**Unclear Requests:**
Customer: "I need help"
You: [Call handoff_to_faq for general assistance - no text response]

**Transfer**
Customer: "I need to speak with a representative"
You: transfer_tool
"""

# Initial welcome message
WELCOME_INITIAL_MESSAGE = f"Hello! Welcome to {BANK_NAME} customer support. How can I help you today?"


class WelcomeAgent(SubAgent):
    """Welcome agent that greets customers and routes them to appropriate services.

    This agent:
    1. Provides warm, professional greeting
    2. Analyzes customer intent from initial request
    3. Routes to FAQ agent for general questions
    4. Routes to verification/transaction flow for banking needs
    5. Allows the user to perform a cold transfer to a representative if they say 'speak with a representative'
    """

    def __init__(self, context: BankContext):
        """Initialize the welcome agent.

        Args:
            context: The bank context object containing user and bank details
        """
        super().__init__(context)
        self.client = Client()
        self.system_prompt = WELCOME_PROMPT

        self._transfer_tool_call = TransferToolCall(target_phone_numbers=[REPRESENTATIVE_PHONE_NUMBER])

        self.generation_config = GenerateContentConfig(
            system_instruction=make_voice_system_prompt(self.system_prompt),
            tool_config={},
            tools=[
                Tool(
                    function_declarations=[
                        FunctionDeclaration(
                            name="handoff_to_faq",
                            description=f"Route customer to FAQ specialist for general {BANK_NAME} information, website help, or general questions that don't require account access.",
                        ),
                        FunctionDeclaration(
                            name="handoff_to_transaction",
                            description="Route customer to banking services for account access, transactions, balances, transfers, or fraud reports.",
                        ),
                        self._transfer_tool_call.to_gemini_tool().function_declarations[0],
                    ]
                ),
            ],
        )

        logger.info("👋 Welcome agent initialized")

    async def initial_message(
        self, user_message: Optional[str] = None, message_history: Optional[List[Content]] = None
    ) -> AsyncGenerator[AgentResponse, None]:
        """Return the initial welcome message.

        Args:
            user_message: The latest message from the user (optional)
            message_history: Previous conversation messages in Gemini format (optional)

        Yields:
            AgentResponse: Initial welcome message
        """
        yield AgentResponse(content=WELCOME_INITIAL_MESSAGE)

    async def generate(
        self, user_message: str, message_history: list[Content]
    ) -> AsyncGenerator[Union[AgentResponse, ToolCall], None]:
        """Handle customer greeting and routing.

        Args:
            user_message: The latest user message
            message_history: Previous conversation messages

        Yields:
            AgentResponse: Welcome responses and routing confirmations
            ToolCall: Handoff tool calls for routing
        """
        logger.info(f"👋 Welcome agent processing: {user_message}")

        # Generate response using Gemini
        response = await self.client.aio.models.generate_content(
            model=WELCOME_MODEL_ID,
            contents=message_history,
            config=self.generation_config,
        )

        # Yield text response if present
        if response.text:
            yield AgentResponse(content=response.text)
            logger.info(f'👋 Welcome agent response: "{response.text}" ({len(response.text)} chars)')

        # Yield handoff function calls if present
        if response.function_calls:
            for function_call in response.function_calls:
                logger.info(f"👋 Welcome agent routing via: {function_call.name}")

                if function_call.name == self._transfer_tool_call.name():
                    logger.info(f"Doing a cold transfer to: {function_call.args['target_phone_number']}")
                    yield TransferCall(target_phone_number=function_call.args["target_phone_number"])
                else:
                    yield ToolCall(tool_name=function_call.name)
