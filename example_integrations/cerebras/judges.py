import json
from typing import AsyncGenerator


from loguru import logger
from pydantic import BaseModel, Field

from line.events import AgentResponse
from line.nodes.conversation_context import ConversationContext
from line.nodes.reasoning import ReasoningNode

from report_logger import SimpleLogger

from cs_utils import *
import config


class EvalInfo(BaseModel):
    """Schema for extracted information."""

    competence: str =  Field(...,description="competence")
    strengths: str =  Field(...,description="strengths")
    weaknesses: str =  Field(...,description="weaknesses")



class JudgeNode(ReasoningNode):
    """
    Node that extracts information from conversations using Cerebras API call.
    
    Inherits conversation management from ReasoningNode and adds agent-specific processing.
    """

    def __init__(
        self,
        system_prompt: str,
        client,
        node_schema = None,
        node_name = "background node"
        
        ):
        self.sys_prompt = system_prompt
        super().__init__(self.sys_prompt)

        self.client = client
        self.model_name = config.MODEL_ID
        self.node_name = node_name
        self.text_logger = SimpleLogger(node_name) # <-- optional to save reports from the background agents
        self.schema = node_schema
    
    
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
        

        if config.INTERVIEW_STARTED:
        
            logger.warning("starting the interview analysis process for current agent")
        else:
            logger.warning("background agents not working")
            return
        
        
        
        if isinstance(context.events[-1], AgentResponse):
            logger.warning("last message from agent, not user, skipping analysis")
            return

        latest_response = context.get_latest_user_transcript_message()

        if not context.events:
            logger.warning("No conversation messages to analyze performance")
            return

        try:
            # Convert messages to cs format

            cs_messages = convert_messages_to_cs(context.events[:-1], self.sys_prompt)

            

            if self.schema:
                format_ = {
                            "type": "json_schema", 
                            "json_schema": {
                                "name": "analysis_schema",
                                "strict": True,
                                "schema": self.schema
                            }
                         } 
            else:
                
                format_ = None
            

            
            stream = await self.client.chat.completions.create(
                    messages=cs_messages,
                    model=self.model_name,
                    max_tokens= config.MAX_OUTPUT_TOKENS,
                    temperature= config.TEMPERATURE,
                    stream=False,
                    response_format= format_
                    )
            
            extracted_info = None
            if stream:
                extracted_info=stream.choices[0].message.content

            
            # Process the extracted information
            if extracted_info:
                
                if format_:
                    
                    try:
                        
                        # Parse as JSON to validate structure
                        perf_data = json.loads(extracted_info)
                        
                        perf_info = EvalInfo.model_validate(perf_data)
                        
                        self.text_logger._write('\n[RESPONSE] \n'+latest_response+"\n")
                        self.text_logger._write("-"*len(latest_response)+"\n")
                        self.text_logger._write(make_table(perf_info.model_dump_json(), self.node_name))
                        self.text_logger._write("-"*len(latest_response)+"\n")
                        
                        yield PerfAnalysis(content=f"Evaluation from {self.node_name}:\n{perf_info.model_dump_json()}")
                        
                    except (json.JSONDecodeError, ValueError) as e:
                        
                        logger.warning(f"Failed to parse evaluation as JSON: {e}")
                        yield PerfAnalysis(content=f"Unparsed evaluation from {self.node_name}:\n{extracted_info}")
                        
                        
                else:
                    logger.info(f"No structured report: {extracted_info}")

                    yield PerfAnalysis(content=f"Unstructured feedback from {self.node_name}:\n{extracted_info}")
                    

            else:
                logger.warning("No evaluation extracted from conversation")

            

        except Exception as e:
            logger.exception(f"Error during evaluation: {e}")

        
