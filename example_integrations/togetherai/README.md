# Customer Service Agent with Cartesia and Together AI

A real-time customer service voice agent integrating Together AI models with Cartesia Line SDK. This example demonstrates how to build a comprehensive customer support system with automated issue resolution, knowledge base search, ticket creation, and intelligent escalation to human agents.

## Architecture Overview

```
┌─────────────────┐    ┌─────────────────┐
│   Customer      │    │ Escalation      │
│   Service Node  │    │ Monitor Node    │
│   (Llama-3.3)   │    │ (GLM-4.5)       │
├─────────────────┤    ├─────────────────┤
│ • Knowledge Base│    │ • Frustration   │
│ • Ticket System │    │   Detection     │
│ • Human Handoff │    │ • Complexity    │
│ • Issue Routing │    │   Assessment    │
└─────────────────┘    └─────────────────┘
         │                       │
         └───────────────────────┘
                    │
            ┌───────────────┐
            │ Line SDK      │
            │ Voice System  │
            └───────────────┘
```

## Features

### **Customer Service Capabilities**
- **Knowledge Base Search**: Automated lookup of common solutions
- **Support Ticket Creation**: Generate tickets for complex issues
- **Human Escalation**: Intelligent handoff to human agents
- **Issue Classification**: Automatic routing based on problem type

### **Background Monitoring**
- **Escalation Detection**: Monitors conversation for frustration signals
- **Complexity Assessment**: Identifies issues requiring human intervention
- **Real-time Analysis**: Continuous conversation monitoring

### **Customer Support Tools**
- `search_knowledge_base`: Query FAQ and documentation
- `create_ticket`: Generate support tickets with priorities
- `escalate_to_human`: Transfer to human agents with context
- `end_call`: Graceful conversation termination

## Getting Started

### Prerequisites
- [Together AI API Key](https://api.together.xyz/settings/api-keys)
- [Cartesia Account](https://play.cartesia.ai/agents) and API key

### Setup

1. **Install Dependencies**
   ```bash
   pip install cartesia-line openai python-dotenv loguru aiohttp uvicorn
   ```

2. **Environment Configuration**
   Create a `.env` file or add to your Cartesia account:
   ```
   TOGETHER_API_KEY=your_together_ai_api_key_here
   ```

3. **Configuration**
   - System prompts and model settings in `config.py`
   - Knowledge base entries can be customized
   - Escalation triggers are configurable

## Implementation Details

### **Main Components**

**CustomerServiceNode** (`customer_service_node.py`)
- Primary conversational agent using meta-llama/Llama-3.3-70B-Instruct-Turbo
- Handles customer interactions and tool execution
- Manages knowledge base searches and ticket creation
- Coordinates human escalation when needed

**EscalationNode** (`escalation_node.py`)
- Background monitoring using zai-org/GLM-4.5-Air-FP8 for efficiency
- Analyzes conversation patterns for escalation triggers
- Provides structured escalation recommendations
- Tracks escalation history and patterns

**Utility Functions** (`openai_utils.py`)
- Message format conversion for Together AI API compatibility
- Mock implementations of customer service backends
- Tool schema definitions for Together AI function calling
- Helper functions for ticket creation and knowledge base search

### **Customer Service Workflow**

1. **Initial Greeting**: Welcome customer and identify issue type
2. **Issue Assessment**: Classify the problem and search knowledge base
3. **Resolution Attempt**: Provide solutions using available tools
4. **Escalation Decision**: Monitor for complexity or frustration
5. **Human Handoff**: Transfer with context when escalation needed
6. **Ticket Creation**: Generate support tickets for follow-up

### **Escalation Triggers**
- Customer expresses frustration multiple times
- Issue remains unresolved after several attempts
- Explicit request for human assistance
- Technical complexity beyond automation
- Account security or billing concerns

## Deployment

### **Local Testing**
```bash
python main.py
```

### **Cartesia Platform**
1. Add this directory to your [Agents Dashboard](https://play.cartesia.ai/agents)
2. Configure environment variables in the platform
3. Deploy and test with voice interactions

### **Configuration Files**
- `pyproject.toml`: Dependencies and project metadata
- `config.py`: Prompts, models, and behavioral settings
- `.env`: API keys and sensitive configuration

## Customization

### **Knowledge Base**
Edit `KNOWLEDGE_BASE` in `config.py` to add company-specific information:
```python
KNOWLEDGE_BASE = {
    "login": "Your login instructions...",
    "billing": "Your billing process...",
    "technical": "Your technical support steps...",
}
```

### **Escalation Criteria**
Modify `prompt_escalation` in `config.py` to adjust escalation sensitivity:
- Frustration thresholds
- Complexity indicators
- Keywords for automatic escalation

### **Customer Service Tools**
Extend `openai_utils.py` to add new capabilities:
- CRM system integration
- Live chat transfer
- Email support routing
- Analytics and reporting

## Example Interactions

**Knowledge Base Query:**
- Customer: "I can't log in to my account"
- Agent: Searches knowledge base → Provides password reset instructions

**Ticket Creation:**
- Customer: "My software keeps crashing with error code 503"
- Agent: Creates high-priority technical support ticket

**Human Escalation:**
- Customer: "This is the third time calling about this billing issue"
- System: Detects escalation need → Transfers to human agent

## Integration Patterns

This example demonstrates key patterns for Line SDK integrations:
- **Multi-node Architecture**: Primary + background processing
- **Tool Integration**: OpenAI function calling with custom tools
- **Event-driven Communication**: Bridge system for inter-node messaging
- **Structured Outputs**: JSON schemas for consistent data handling
- **Error Handling**: Graceful degradation and user-friendly error messages

## Support

- **Documentation**: [Line Docs](https://docs.cartesia.ai/line/introduction)
- **Community**: [Discord](https://discord.gg/cartesia)
- **Examples**: Check out other integrations in this repository
