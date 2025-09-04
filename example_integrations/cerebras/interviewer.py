import json
from typing import AsyncGenerator
from loguru import logger
from line.events import AgentResponse, ToolResult
from line.nodes.conversation_context import ConversationContext
from line.nodes.reasoning import ReasoningNode
from line.tools.system_tools import EndCallArgs, EndCallTool, end_call
from cs_utils import end_call_schema, interview_schema, convert_messages_to_cs
import config



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
        

        if not context.events:
            logger.info("No conversation messages to analyze performance")
            return
        
        
        try:
            # Convert messages to cs format
            cs_messages = convert_messages_to_cs(context.events, self.sys_prompt)
            
            # Call Cerebras API
            
            stream = await self.client.chat.completions.create(
                    messages=cs_messages,
                    model=config.MODEL_ID,
                    max_tokens= config.MAX_OUTPUT_TOKENS,
                    temperature= config.TEMPERATURE,
                    stream=False,
                    tools=self.tools,
                    parallel_tool_calls=True,
                    )
            extracted_info = None
            
            if stream:
                
                choice = stream.choices[0].message
                

                if choice.tool_calls:
                   
                    function_call = choice.tool_calls[0].function
                    arguments = json.loads(function_call.arguments)
                    yield ToolResult(tool_name=function_call.name, tool_args=arguments)
                    
                    if function_call.name == EndCallTool.name():
                        
                       
                        
                        args = EndCallArgs(**arguments)
                        
                        logger.info(
                            f"🤖 End call tool called. Ending conversation with goodbye message: "
                            f"{args.goodbye_message}"
                        )
                        async for item in end_call(args):
                            yield item
                        
                        
                    if function_call.name == "start_interview":
                                                
                        config.INTERVIEW_STARTED = arguments["confirmed"]
                        logger.info(f"🤖 Interview started: {config.INTERVIEW_STARTED}")

                        # Note: This is the result of executing the model's request (the tool call), not the model's own output.
                        
                    
                        # Send the result back to the model to fulfill the request.
                        if config.INTERVIEW_STARTED:
                            cs_messages.append({
                                "role": "system",
                                "content": "Based on the current conversation context, ask the next question. /no_think ",
                                
                            })

                
                        # Request the final response from the model, now that it has the result.
                        final_response = await self.client.chat.completions.create(
                                messages=cs_messages,
                                model=config.MODEL_ID,
                                stream=False,
                                )
                        
                        extracted_info = final_response.choices[0].message.content
                        
                
                else:    
                    
                    extracted_info=stream.choices[0].message.content

            #Process the extracted information
            if extracted_info:
                    
                    yield AgentResponse(content=f"{extracted_info}")
                    

            else:
                logger.warning("No evaluation extracted from conversation")

            

        except Exception as e:
            logger.exception(f"Error during interviewer node operation: {e}")

        
