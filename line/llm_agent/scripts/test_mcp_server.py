#!/usr/bin/env python3
"""
A simple local MCP server for testing MCP tool integration.

Exposes a dice-rolling tool via stdio transport.

Usage:
    python line/llm_agent/scripts/test_mcp_server.py
"""

import asyncio
import json
import random

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

server = Server("dice")


@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="roll",
            description="Roll one or more dice. Returns the individual rolls and their total.",
            inputSchema={
                "type": "object",
                "properties": {
                    "sides": {
                        "type": "integer",
                        "description": "Number of sides on each die (e.g. 6 for a standard die)",
                    },
                    "count": {
                        "type": "integer",
                        "description": "How many dice to roll (default 1)",
                    },
                },
                "required": ["sides"],
            },
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    if name == "roll":
        sides = arguments["sides"]
        count = arguments.get("count", 1)
        rolls = [random.randint(1, sides) for _ in range(count)]
        return [
            TextContent(
                type="text",
                text=json.dumps({"rolls": rolls, "total": sum(rolls)}),
            )
        ]
    else:
        return [TextContent(type="text", text=json.dumps({"error": f"Unknown tool: {name}"}))]


async def main():
    async with stdio_server() as (r, w):
        await server.run(r, w, server.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
