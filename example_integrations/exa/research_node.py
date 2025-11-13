import json
import sys

sys.path.append("../../")  # Add path to line SDK

from typing import AsyncGenerator

import config
from exa_utils import ExaSearchClient, convert_messages_to_openai, end_call_schema, web_search_schema
from loguru import logger
import openai

from line.events import AgentResponse, ToolResult
from line.nodes.conversation_context import ConversationContext
from line.nodes.reasoning import ReasoningNode
from line.tools.system_tools import EndCallArgs, EndCallTool, end_call


class ResearchNode(ReasoningNode):
    """
    Web research node that combines conversation with real-time web search.

    This node can engage in conversation and perform web searches when needed
    to provide accurate, up-to-date information to users.
    """

    def __init__(
        self,
        system_prompt: str,
        openai_client: openai.AsyncOpenAI,
        exa_client: ExaSearchClient,
    ):
        self.system_prompt = system_prompt
        super().__init__(system_prompt)

        self.openai_client = openai_client
        self.exa_client = exa_client
        self.tools = [web_search_schema, end_call_schema]

    async def process_context(self, context: ConversationContext) -> AsyncGenerator[AgentResponse, None]:
        """
        Process conversation context and generate responses with web search capability.

        Args:
            context: Conversation context with messages

        Yields:
            AgentResponse: Text responses to the user
            ToolResult: Results from tool calls
        """

        if not context.events:
            logger.info("No conversation messages to process")
            return

        try:
            # Convert messages to OpenAI format
            openai_messages = convert_messages_to_openai(context.events, self.system_prompt)

            # Call OpenAI with tools
            response = await self.openai_client.chat.completions.create(
                model="gpt-4o-mini",  # Fast and cost-effective
                messages=openai_messages,
                max_tokens=config.MAX_OUTPUT_TOKENS,
                temperature=config.TEMPERATURE,
                tools=self.tools,
                tool_choice="auto",
            )

            message = response.choices[0].message

            # Handle tool calls
            if message.tool_calls:
                for tool_call in message.tool_calls:
                    function_name = tool_call.function.name
                    arguments = json.loads(tool_call.function.arguments)

                    # Yield tool result for observability
                    yield ToolResult(tool_name=function_name, tool_args=arguments, tool_call_id=tool_call.id)

                    if function_name == "web_search":
                        # Perform web search
                        search_query = arguments.get("query", "")

                        logger.info(f"🔍 Performing web search: '{search_query}'")

                        # Get search results
                        search_results = await self.exa_client.search_and_get_content(query=search_query)

                        if "error" in search_results:
                            error_msg = search_results["error"]
                            logger.error(f"Search failed: {error_msg}")

                            # Continue with error message
                            openai_messages.append(
                                {
                                    "role": "system",
                                    "content": (
                                        f"Web search failed: {error_msg}. Please provide an answer "
                                        "based on your existing knowledge and let the user know about "
                                        "the search limitation."
                                    ),
                                }
                            )
                        else:
                            # Add search results to conversation
                            search_content = search_results["formatted_content"]
                            logger.info(
                                f"📊 Search completed: {search_results['source_count']} sources found"
                            )

                            openai_messages.append(
                                {
                                    "role": "system",
                                    "content": (
                                        f"Web search results:\n{search_content}\n\n"
                                        "Please synthesize this information to answer the user's question. "
                                        "Cite sources when relevant."
                                    ),
                                }
                            )

                        # Generate response with search results
                        final_response = await self.openai_client.chat.completions.create(
                            model="gpt-4o-mini",
                            messages=openai_messages,
                            max_tokens=config.MAX_OUTPUT_TOKENS,
                            temperature=config.TEMPERATURE,
                        )

                        final_content = final_response.choices[0].message.content
                        if final_content:
                            yield AgentResponse(content=final_content)

                    elif function_name == EndCallTool.name():
                        # Handle end call
                        args = EndCallArgs(**arguments)
                        logger.info(f"🤖 End call requested: {args.goodbye_message}")

                        async for item in end_call(args):
                            yield item

            else:
                # No tool calls, just return the response
                if message.content:
                    yield AgentResponse(content=message.content)
                else:
                    logger.warning("No content in response and no tool calls")
                    yield AgentResponse(
                        content=(
                            "I'm sorry, I didn't understand that. Could you please rephrase your question?"
                        )
                    )

        except Exception as e:
            logger.exception(f"Error in research node processing: {e}")
            yield AgentResponse(
                content=(
                    "I apologize, but I encountered an error while processing your request. Please try again."
                )
            )
