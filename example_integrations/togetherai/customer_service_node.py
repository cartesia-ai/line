import json
from typing import AsyncGenerator

import config
from openai_utils import (
    convert_messages_to_openai,
    end_call_schema,
    search_knowledge_base_schema,
    create_ticket_schema,
    escalate_to_human_schema,
    search_knowledge_base,
    create_support_ticket,
    escalate_to_human_agent
)
from loguru import logger

from line.events import AgentResponse, ToolResult
from line.nodes.conversation_context import ConversationContext
from line.nodes.reasoning import ReasoningNode
from line.tools.system_tools import EndCallArgs, EndCallTool, end_call


class CustomerServiceNode(ReasoningNode):
    """
    Customer service node using OpenAI API for customer support interactions.
    
    Inherits conversation management from ReasoningNode and adds customer service tools.
    """

    def __init__(
        self,
        system_prompt: str,
        client,
    ):
        self.sys_prompt = system_prompt
        super().__init__(self.sys_prompt)

        self.client = client
        self.tools = [
            end_call_schema,
            search_knowledge_base_schema,
            create_ticket_schema,
            escalate_to_human_schema
        ]

    async def process_context(self, context: ConversationContext) -> AsyncGenerator[AgentResponse, None]:
        """
        Process customer service requests from conversation context.

        Args:
            context: Conversation context with messages.

        Yields:
            AgentResponse: Customer service responses.
        """

        if not context.events:
            logger.info("No conversation messages to process")
            return

        try:
            # Convert messages to OpenAI format
            openai_messages = convert_messages_to_openai(context.events, self.sys_prompt)

            # Call OpenAI API
            response = await self.client.chat.completions.create(
                model=config.MODEL_ID,
                messages=openai_messages,
                max_tokens=config.MAX_OUTPUT_TOKENS,
                temperature=config.TEMPERATURE,
                tools=self.tools,
                tool_choice="auto"
            )

            message = response.choices[0].message
            
            # Handle tool calls
            if message.tool_calls:
                for tool_call in message.tool_calls:
                    function_name = tool_call.function.name
                    arguments = json.loads(tool_call.function.arguments)
                    
                    yield ToolResult(tool_name=function_name, tool_args=arguments)
                    
                    # Execute the appropriate tool
                    if function_name == "end_call":
                        args = EndCallArgs(**arguments)
                        logger.info(f"🤖 End call requested: {args.goodbye_message}")
                        async for item in end_call(args):
                            yield item
                            
                    elif function_name == "search_knowledge_base":
                        query = arguments.get("query", "")
                        result = search_knowledge_base(query)
                        logger.info(f"🔍 Knowledge base search: {query}")
                        
                        # Get follow-up response with search result
                        follow_up_messages = openai_messages + [
                            {"role": "assistant", "content": f"I found this information: {result}"},
                            {"role": "system", "content": f"Knowledge base search result: {result}. Use this to help the customer."}
                        ]
                        
                        follow_up_response = await self.client.chat.completions.create(
                            model=config.MODEL_ID,
                            messages=follow_up_messages,
                            max_tokens=config.MAX_OUTPUT_TOKENS,
                            temperature=config.TEMPERATURE
                        )
                        
                        yield AgentResponse(content=follow_up_response.choices[0].message.content)
                        
                    elif function_name == "create_ticket":
                        title = arguments.get("title", "")
                        description = arguments.get("description", "")
                        priority = arguments.get("priority", "medium")
                        
                        result = create_support_ticket(title, description, priority)
                        logger.info(f"🎫 Ticket created: {title} ({priority})")
                        
                        # Provide confirmation to customer
                        yield AgentResponse(content=result)
                        
                    elif function_name == "escalate_to_human":
                        reason = arguments.get("reason", "")
                        urgency = arguments.get("urgency", "standard")
                        
                        result = escalate_to_human_agent(reason, urgency)
                        logger.info(f"👨‍💼 Escalating to human: {reason} ({urgency})")
                        
                        # Set global escalation flag
                        config.ESCALATION_DETECTED = True
                        
                        yield AgentResponse(content=result)

            # Handle regular response (no tools)
            elif message.content:
                yield AgentResponse(content=message.content)
            
            else:
                logger.warning("No response content from OpenAI")

        except Exception as e:
            logger.exception(f"Error during customer service processing: {e}")
            yield AgentResponse(
                content="I apologize, but I'm experiencing technical difficulties. Let me create a support ticket for you so our team can help directly."
            )