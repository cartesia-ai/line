# Web Research Agent with Exa and Cartesia

A real-time web research voice agent that combines Cartesia's voice capabilities with Exa's powerful web search API to provide accurate, up-to-date information through natural conversation.

## Getting Started

### Prerequisites

- [Cartesia account](https://play.cartesia.ai/agents) and API key
- [OpenAI API key](https://platform.openai.com/api-keys) for conversation synthesis
- [Exa API key](https://dashboard.exa.ai/api-keys) for web search

### Environment Variables

Make sure to add these API keys to your `.env` file or Cartesia dashboard:

| Variable | Description | Required |
|----------|-------------|----------|
| `OPENAI_API_KEY` | OpenAI API key for conversation | Yes |
| `EXA_API_KEY` | Exa API key for web search | Yes |

### Installation

```bash
# Install dependencies
pip install exa-py openai cartesia-line python-dotenv loguru

# Or using uv (recommended)
uv add exa-py openai cartesia-line python-dotenv loguru
```

## Architecture

### Core Components

1. **ResearchNode** (`research_node.py`)
   - Extends ReasoningNode for conversation management
   - Integrates OpenAI for natural language processing
   - Uses Exa for real-time web search
   - Synthesizes search results into conversational responses

2. **ExaSearchClient** (`exa_utils.py`)
   - Wrapper for Exa API with optimized search parameters
   - Formats search results for LLM consumption
   - Handles errors and edge cases gracefully

3. **Configuration** (`config.py`)
   - System prompts optimized for web research
   - Exa search parameters for best results
   - Custom events for search result handling

## Configuration

### Exa Search Parameters

The integration uses optimized search settings:

```python
# Using the exact API call format
result = exa.search_and_contents(
    "query",
    num_results=10,
    type="fast",
    livecrawl="never",
    text={
        "max_characters": 1000
    }
)
```

## Local Development

1. **Set up environment variables**:
   ```bash
   # Copy the example file
   cp .env.example .env

   # Edit .env with your actual API keys
   # OPENAI_API_KEY=your-openai-key-here
   # EXA_API_KEY=your-exa-key-here
   ```

2. **Run locally**:
   ```bash
   python3 main.py
   ```

   The `.env` file will automatically load your API keys!

3. **Test with voice**:
   ```bash
   cartesia chat 8000
   ```

## Deployment

Deploy to Cartesia's platform:

1. **Add API keys** to your Cartesia dashboard
2. **Upload the integration** files
3. **Deploy** and start talking to your research agent

Perfect for building voice-powered research tools, fact-checking assistants, and information discovery applications.
