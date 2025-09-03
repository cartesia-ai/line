import json, sys
from typing import AsyncGenerator

#import types
from pathlib import Path
from loguru import logger
import json

from line.events import AgentResponse
from line.nodes.conversation_context import ConversationContext
from line.nodes.reasoning import ReasoningNode
from line.tools.system_tools import EndCallTool


from cs_utils import *
from config import *



class TalkingNode(ReasoningNode):
    """
    Node that extracts information from conversations using Cerebras API call.
    
    Inherits conversation management from ReasoningNode and adds agent-specific processing.
    """

    def __init__(
        self,
        system_prompt: str,
        client,
        ):
        self.sys_prompt = system_prompt
        super().__init__(self.sys_prompt,)

        self.client = client
        self.model_name = 'llama-3.3-70b'
        self.tools = [end_call_schema, interview_schema] 
    
    
    async def process_context(
        self, context: ConversationContext
        ) -> AsyncGenerator[AgentResponse, None]:
        """
        evaluate response quality from conversation context.

        Args:
            context: Conversation context with messages.

        Yields:
            NodeMessage: evaluation results.
        """
        #logger.info("Starting performance analysis")

        if not context.events:
            logger.info("No conversation messages to analyze performance")
            return
        
        
        try:
            # Convert messages and tools to cs format
            cs_messages = convert_messages_to_cs(context.events, self.sys_prompt)
            #logger.debug(f"CS messages: {cs_messages}")
            # Call Cerebras API
            
            stream = await self.client.chat.completions.create(
                    messages=cs_messages,
                    model=MODEL_ID,
                    max_tokens= MAX_OUTPUT_TOKENS,
                    temperature= TEMPERATURE,
                    stream=False,
                    tools=self.tools,
                    parallel_tool_calls=True,
                    )
            extracted_info = None
            
            if stream:
                #logger.warning(f"choice: {stream.choices}")
                choice = stream.choices[0].message
                

                if choice.tool_calls:
                    function_call = choice.tool_calls[0].function
                    #logger.info("tools are called")
                
                    if function_call.name == "start_interview":
                        # Logging that the model is executing a function named "calculate".
                        # logger.info(f"Model executing function '{function_call.name}' with arguments {function_call.arguments}")

                        # Parse the arguments from JSON format and perform the requested calculation.
                        arguments = json.loads(function_call.arguments)
                        #logger.info(f"start_interview tool called")
                        
                        config.INTERVIEW_STARTED = arguments["confirmed"]
                        logger.warning(f"Interview started: {config.INTERVIEW_STARTED}")

                        # Note: This is the result of executing the model's request (the tool call), not the model's own output.
                        # logger.error(f"Confirmation for interview: {config.interview_started}")
                    
                        # Send the result back to the model to fulfill the request.
                        if config.INTERVIEW_STARTED:
                            cs_messages.append({
                                "role": "system",
                                "content": "Based on the current conversation context, ask the next question. /no_think ",
                                
                            })

                
                        # Request the final response from the model, now that it has the calculation result.
                        final_response = await self.client.chat.completions.create(
                                messages=cs_messages,
                                model=MODEL_ID,
                                stream=False,
                                )
                        
                        extracted_info = final_response.choices[0].message.content
                        
                
                else:    
                    #logger.info("no tool calling")
                    extracted_info=stream.choices[0].message.content

            #Process the extracted information
            if extracted_info:

                    yield AgentResponse(content=f"{extracted_info}")
                    

            else:
                logger.warning("No evaluation extracted from conversation")

            

        except Exception as e:
            logger.exception(f"Error during evaluation: {e}")

        
