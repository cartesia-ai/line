"""Quick-service ordering agent with Inception Mercury 2 and Cartesia Line SDK."""

import os
from typing import Annotated, List

from dotenv import load_dotenv
from loguru import logger

from line.llm_agent import LlmAgent, LlmConfig, ToolEnv, end_call, loopback_tool
from line.voice_agent_app import AgentEnv, CallRequest, VoiceAgentApp

load_dotenv()

# Mercury 2 speaks the OpenAI chat-completions protocol, so it runs on the SDK's
# HTTP/LiteLLM backend using the `openai/` prefix with a custom `api_base`.
MODEL_ID = "openai/mercury-2"
INCEPTION_API_BASE = "https://api.inceptionlabs.ai/v1"

# Mercury 2 decoding control: "instant", "low", "medium", or "high".
# "instant" skips extended reasoning for the lowest time-to-first-token,
# which is usually the right trade-off on a live voice call.
REASONING_EFFORT = "instant"

MAX_OUTPUT_TOKENS = 300
TEMPERATURE = 0.3

SYSTEM_PROMPT = """You are the order-taker at Mercury Coffee, a quick-service coffee bar. \
Keep the line moving: short, friendly, instant replies.

MENU
Drinks (small, medium, large): drip coffee, latte, cappuccino, cold brew, chai, hot chocolate.
Milk options: whole, oat, almond. Extras: extra shot, vanilla syrup, caramel syrup.
Food: butter croissant, blueberry muffin, bagel with cream cheese.

HOW TO WORK THE ORDER
- When the customer asks for something on the menu, call add_item with a complete description
  (size, drink, milk, extras), then confirm it back in a few words.
- If they change their mind, call remove_item for the old item and add_item for the new one.
- If they ask for something not on the menu, say so and suggest the closest item.
- When they say that's everything, call confirm_order, read the order back in one sentence,
  and tell them the total number of items.
- After they confirm, thank them and call end_call.

VOICE RULES
This is a phone call. Plain natural sentences only: no markdown, no lists, no special characters.
One question at a time. Never mention the tools or your reasoning."""

INTRODUCTION = "Hi, welcome to Mercury Coffee! What can I get started for you?"


class OrderTools:
    """Holds the in-progress order for a single call."""

    def __init__(self):
        self._items: List[str] = []

    @loopback_tool
    async def add_item(
        self,
        ctx: ToolEnv,
        item: Annotated[str, "Complete item description, e.g. 'medium latte with oat milk'."],
    ) -> str:
        """Add one item to the customer's order. Call once per item."""
        self._items.append(item)
        logger.info(f"Added item: {item} (order size: {len(self._items)})")
        return f"Added: {item}. Current order: {'; '.join(self._items)}."

    @loopback_tool
    async def remove_item(
        self,
        ctx: ToolEnv,
        item: Annotated[str, "The item to remove, matching how it was added."],
    ) -> str:
        """Remove an item from the order when the customer changes their mind."""
        for existing in self._items:
            if item.lower() in existing.lower() or existing.lower() in item.lower():
                self._items.remove(existing)
                logger.info(f"Removed item: {existing} (order size: {len(self._items)})")
                current = "; ".join(self._items) if self._items else "empty"
                return f"Removed: {existing}. Current order: {current}."
        return f"No item matching '{item}' found. Current order: {'; '.join(self._items)}."

    @loopback_tool
    async def confirm_order(self, ctx: ToolEnv) -> str:
        """Finalize the order once the customer says they are done ordering."""
        if not self._items:
            return "The order is empty. Ask the customer what they would like."
        logger.info(f"Order confirmed: {self._items}")
        return f"Order confirmed with {len(self._items)} item(s): {'; '.join(self._items)}."


async def get_agent(env: AgentEnv, call_request: CallRequest):
    """Create a Mercury 2 ordering agent for this call."""
    api_key = os.environ.get("INCEPTION_API_KEY")
    if not api_key:
        raise RuntimeError("INCEPTION_API_KEY is not set.")

    order = OrderTools()
    return LlmAgent(
        model=MODEL_ID,
        api_key=api_key,
        tools=[order.add_item, order.remove_item, order.confirm_order, end_call],
        config=LlmConfig(
            system_prompt=SYSTEM_PROMPT,
            introduction=INTRODUCTION,
            temperature=TEMPERATURE,
            max_tokens=MAX_OUTPUT_TOKENS,
            extra={
                # Point LiteLLM's OpenAI client at the Inception API.
                "api_base": INCEPTION_API_BASE,
                # Mercury-specific parameters go in extra_body so they are
                # forwarded verbatim to the API.
                "extra_body": {"reasoning_effort": REASONING_EFFORT},
            },
        ),
    )


app = VoiceAgentApp(get_agent=get_agent)

if __name__ == "__main__":
    app.run()
