# Background Browser-Use Agent

A voice agent that extends the Chat/Supervisor pattern with a **browser-use** tool for real-time web browsing. The agent can chat, reason deeply, and autonomously navigate web pages — all while keeping the conversation going.

## Overview

This example creates a voice agent with three tools:

| Tool | Model | Purpose |
|------|-------|---------|
| *(direct chat)* | Claude 4.5 Haiku | Fast conversational responses |
| `ask_supervisor` | Claude 4.5 Opus | Complex reasoning (background) |
| `browse_web` | Claude Sonnet + headless Chrome | Live web browsing (background) |

Both `ask_supervisor` and `browse_web` are **background loopback tools** — they run asynchronously while the chat model continues talking to the user.

## Architecture

```
User Question
     │
     ▼
┌────────────────────────────────────────────────────┐
│           BrowserSupervisorAgent                   │
│  ┌─────────────┐                                   │
│  │    Chat      │  (Claude Haiku — fast, cheap)    │
│  │  LlmAgent    │                                   │
│  └──────┬──────┘                                   │
│         │                                          │
│    Simple question? ──────────────► Respond         │
│         │                                          │
│    Complex reasoning?                              │
│         │ ask_supervisor (background)              │
│         ▼                                          │
│  ┌─────────────┐                                   │
│  │ Supervisor   │  (Claude Opus — deep reasoning)  │
│  │ LlmAgent     │                                   │
│  └─────────────┘                                   │
│         │                                          │
│    Needs live web data?                            │
│         │ browse_web (background)                  │
│         ▼                                          │
│  ┌─────────────┐                                   │
│  │ Browser-Use  │  (Claude Sonnet + headless       │
│  │ Agent        │   Chrome via Playwright)          │
│  └─────────────┘                                   │
│         │                                          │
│    Chat synthesizes results into speech            │
└────────────────────────────────────────────────────┘
     │
     ▼
Spoken Response
```

## Prerequisites

1. **ANTHROPIC_API_KEY** — used by the chat model (Haiku), supervisor (Opus), and the browser-use agent (Sonnet).
2. **Playwright browsers** — browser-use uses Playwright under the hood:
   ```bash
   playwright install chromium
   ```

## Running the Example

```bash
cd examples/background_browser_use_agent

# Install dependencies
pip install -e .

# Install Playwright browsers (first time only)
playwright install chromium

# Run the agent
ANTHROPIC_API_KEY=your-key python main.py
```

## Example Conversations

**Simple question (Haiku responds directly):**
> User: "What's the capital of France?"
> Agent: "The capital of France is Paris!"

**Complex question (escalated to Opus):**
> User: "Prove the square root of 2 is irrational."
> Agent: "Let me think carefully about that..."
> *(supervisor runs in background)*
> Agent: "Here's the classic proof by contradiction..."

**Live web data (browser-use in background):**
> User: "What's NVIDIA's stock price right now?"
> Agent: "Let me look that up for you..."
> *(browser navigates to a finance site in background)*
> Agent: "NVIDIA is currently trading at $..."

**Combined:**
> User: "Find the latest ML Engineer jobs in San Francisco and tell me which ones look most promising."
> Agent: "I'll search for those online. Give me a moment..."
> *(browser searches LinkedIn/job sites)*
> Agent: "I found several positions. The most interesting ones are..."

## How browse_web Works

1. The chat model calls `browse_web(task="Search Google for ...")`.
2. The tool immediately yields a status message ("Opening my browser...").
3. A headless Chrome browser is launched via Playwright.
4. A browser-use `Agent` (powered by Claude Sonnet) autonomously navigates pages, clicks buttons, reads content, and extracts information.
5. The result is yielded back to the chat model, triggering a new completion.
6. The chat model synthesizes the browser output into a conversational spoken response.

The entire browser session runs in the background — the user can keep chatting while it works.

## Cost Considerations

- **Haiku** handles most turns cheaply and with low latency.
- **Opus** is only invoked for genuinely complex reasoning.
- **Browser-use** (Sonnet) is invoked only for live web tasks. Each browser session involves multiple Sonnet API calls as it navigates pages, so costs can add up for complex browsing tasks.
- Concurrent browser sessions are blocked (one at a time) to control costs and resource usage.

## Cartesia Agent Commands

- **Installation** - curl -fsSL https://cartesia.sh | sh 
- source ~/.bashrc
- cd submodules/line/examples/background_browser_use_agent
- cartesia init
- cartesia deploy
- pip install cartesia-line
- python main.py
- cartesia chat 8000

