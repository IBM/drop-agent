"""Base classes and utilities for droplet tools"""

import asyncio
import json
from typing import AsyncIterator

from gpt_oss.tools.tool import Tool
from openai_harmony import (Author, Message, Role, TextContent,
                            ToolNamespaceConfig)


def convert_tool_config_to_openai(tool_config_dict):
    """
    Convert ToolNamespaceConfig format to OpenAI function calling format.

    ToolNamespaceConfig format:
    {
        "name": "browser",
        "description": "...",
        "tools": [
            {"name": "search", "description": "...", "parameters": {...}},
            {"name": "open", "description": "...", "parameters": {...}}
        ]
    }

    OpenAI format:
    [
        {"type": "function", "function": {"name": "search", "description": "...", "parameters": {...}}},
        {"type": "function", "function": {"name": "open", "description": "...", "parameters": {...}}}
    ]

    Args:
        tool_config_dict: Dictionary from ToolNamespaceConfig.model_dump()

    Returns:
        List of OpenAI-formatted tool definitions
    """
    openai_tools = []

    if "tools" in tool_config_dict:
        for tool_def in tool_config_dict["tools"]:
            openai_tool = {
                "type": "function",
                "function": {
                    "name": tool_def["name"],
                    "description": tool_def["description"],
                    "parameters": tool_def["parameters"]
                }
            }
            openai_tools.append(openai_tool)

    return openai_tools


class SimpleFunctionTool(Tool):
    """
    Base class for simple function-based tools.
    Wraps a function with OpenAI-compatible tool configuration.
    """

    def __init__(self, name, description, parameters, function):
        self._name = name
        self._description = description
        self._parameters = parameters
        self._function = function

    @property
    def name(self):
        return self._name

    @property
    def tool_config(self):
        config = ToolNamespaceConfig(
            name=self._name,
            description=self._description,
            tools=[{
                "name": self._name,
                "description": self._description,
                "parameters": self._parameters
            }]
        )
        return config

    def instruction(self):
        return self._description

    async def _process(self, message: Message) -> AsyncIterator[Message]:
        # Extract function arguments from message
        try:
            if len(message.content) == 1 and isinstance(message.content[0], TextContent):
                contents = message.content[0].text
                if contents:
                    function_args = json.loads(contents)
                else:
                    function_args = {}
            else:
                function_args = {}

            # Execute function (run in thread pool if sync)
            loop = asyncio.get_event_loop()
            if asyncio.iscoroutinefunction(self._function):
                result = await self._function(**function_args)
            else:
                result = await loop.run_in_executor(None, lambda: self._function(**function_args))

            # Return result as Message
            yield Message(
                author=Author(role=Role.TOOL, name=self._name),
                content=[TextContent(text=json.dumps(result))],
            ).with_recipient("assistant")

        except Exception as e:
            error_msg = f"Error executing {self._name}: {str(e)}"
            yield Message(
                author=Author(role=Role.TOOL, name=self._name),
                content=[TextContent(text=json.dumps({"error": error_msg}))],
            ).with_recipient("assistant")
