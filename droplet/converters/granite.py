"""Granite message converter for Granite and other Hugging Face models"""

import json
import re

from openai_harmony import Author, Message, Role, SystemContent, TextContent
from transformers import AutoTokenizer

from droplet.converters.base import MessageConverter
from droplet.tools.base import convert_tool_config_to_openai

# Model-specific context limits
GRANITE_MODELS = {
    "granite-3.0-8b-instruct": 4096,
    "granite-3.0-2b-instruct": 4096,
    "granite-4.0-h-small": 8192,
}


class GraniteMessageConverter(MessageConverter):
    """
    Message converter for Granite and other Hugging Face models.

    Uses transformers.apply_chat_template() for message formatting
    and parses OpenAI-style JSON function calls from responses.
    """

    def __init__(self, model_name):
        """
        Initialize Granite converter with model-specific tokenizer

        Args:
            model_name: Name of the model (used to load tokenizer)
        """
        self.model_name = model_name

        # manual mapping for 8b dense
        if model_name == "ibm-granite/granite-4.0-8b":
            model_name = "ibm-granite/granite-4.0-h-small"

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

        # Determine context limit
        self.max_context_tokens = self._get_context_limit(model_name)

    def _get_context_limit(self, model_name):
        """
        Get context limit for model

        Args:
            model_name: Model name

        Returns:
            int: Context limit in tokens
        """
        # Check if model name contains known granite model
        for granite_model, limit in GRANITE_MODELS.items():
            if granite_model in model_name:
                return limit

        # Default fallback
        return 4096

    def _normalize_message_to_dict(self, msg):
        """
        Convert Message object to simple dict format for apply_chat_template

        Args:
            msg: Message object or dict

        Returns:
            dict with 'role' and 'content' keys
        """
        if isinstance(msg, dict):
            return msg

        # Extract role
        if hasattr(msg.author.role, 'value'):
            role = msg.author.role.value
        else:
            role = str(msg.author.role)

        # Map harmony roles to standard chat roles
        role_map = {
            'system': 'system',
            'user': 'user',
            'assistant': 'assistant',
            'tool': 'tool',
            'developer': 'system',  # Map developer to system
        }
        role = role_map[role]

        # Extract content
        if isinstance(msg.content, list) and len(msg.content) > 0:
            if hasattr(msg.content[0], 'text'):
                content = msg.content[0].text
            else:
                content = str(msg.content[0])
        else:
            content = str(msg.content)

        return {'role': role, 'content': content}

    def _extract_system_message_with_tools(self, messages):
        """
        Extract system message and convert tools to OpenAI format

        Args:
            messages: List of Message objects

        Returns:
            Tuple of (system_text, tools_list, other_messages)
        """
        system_text = None
        tools_list = []
        other_messages = []

        for msg in messages:
            if hasattr(msg, 'author') and msg.author.role == Role.SYSTEM:
                # Extract system content
                if isinstance(msg.content, list) and len(msg.content) > 0:
                    system_content = msg.content[0]

                    if isinstance(system_content, SystemContent):
                        # Extract model identity as system text
                        system_text = system_content.model_identity

                        # Convert tool configurations to OpenAI format
                        # system_content.tools is a dict with namespace names as keys
                        if hasattr(system_content, 'tools') and system_content.tools:
                            for namespace_name, tool_namespace in system_content.tools.items():
                                # Convert ToolNamespaceConfig to dict
                                if hasattr(tool_namespace, 'model_dump'):
                                    tool_config_dict = tool_namespace.model_dump()
                                else:
                                    tool_config_dict = tool_namespace

                                # Convert to OpenAI format using the utility function
                                # This already returns tools in the correct format with "type": "function" wrapper
                                openai_tools = convert_tool_config_to_openai(tool_config_dict)
                                tools_list.extend(openai_tools)
                    else:
                        # Simple text content
                        if hasattr(system_content, 'text'):
                            system_text = system_content.text
                        else:
                            system_text = str(system_content)
            else:
                other_messages.append(msg)

        return system_text, tools_list, other_messages

    def messages_to_prompt_string(self, messages):
        """
        Convert message list to formatted prompt string using apply_chat_template

        Args:
            messages: List of Message objects or dicts

        Returns:
            str: Formatted prompt string
        """
        # Extract system message and tools
        system_text, tools_list, other_messages = self._extract_system_message_with_tools(messages)

        # Normalize messages to dict format
        normalized_messages = []

        # Add system message if present
        if system_text:
            normalized_messages.append({'role': 'system', 'content': system_text})

        # Add other messages
        for msg in other_messages:
            normalized_messages.append(self._normalize_message_to_dict(msg))

        # Use apply_chat_template with tools parameter
        # Pass tools directly to apply_chat_template as per Granite documentation
        if tools_list:
            prompt_string = self.tokenizer.apply_chat_template(
                normalized_messages,
                tools=tools_list,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            prompt_string = self.tokenizer.apply_chat_template(
                normalized_messages,
                tokenize=False,
                add_generation_prompt=True
            )

        return prompt_string


    def response_string_to_messages(self, response_text):
        """
        Parse response text to Message objects, detecting function calls

        Granite uses <tool_call> tags around JSON function calls:
        <tool_call>
        {"name": "function_name", "arguments": {...}}
        </tool_call>

        Args:
            response_text: Raw response from backend

        Returns:
            List of Message objects
        """
        # Try to detect Granite tool call pattern with tags
        # Pattern: <tool_call>...JSON...</tool_call>
        # Extract everything between tags instead of trying to match JSON structure
        tool_call_pattern = r'<tool_call>\s*(.*?)\s*</tool_call>'

        match = re.search(tool_call_pattern, response_text, re.DOTALL)

        if match:
            # Extract JSON content (everything between tags)
            json_content = match.group(1).strip()

            # Parse the JSON
            tool_call = json.loads(json_content)

            # Extract function name
            function_name = tool_call['name']

            # Create tool call message
            message = Message(
                author=Author(role=Role.ASSISTANT),
                content=[TextContent(text=json_content)]
            ).with_recipient(f"functions.{function_name}")

            return [message]

        # Fallback: try to detect function call without tags (for backward compatibility)
        # Pattern: {"name": "function_name", "arguments": {...}}
        function_call_pattern = r'\{["\']name["\']\s*:\s*["\']([^"\']+)["\']\s*,\s*["\']arguments["\']\s*:\s*(\{.*?\})\s*\}'

        match = re.search(function_call_pattern, response_text, re.DOTALL)

        if match:
            # Extract the full matched JSON
            full_json = match.group(0)
            function_name = match.group(1)

            # Create tool call message
            message = Message(
                author=Author(role=Role.ASSISTANT),
                content=[TextContent(text=full_json)]
            ).with_recipient(f"functions.{function_name}")

            return [message]

        # No function call detected - return as plain text message
        message = Message.from_role_and_content(Role.ASSISTANT, response_text)
        return [message]

    def get_stop_tokens(self):
        """
        Get stop tokens for Granite models

        Returns:
            List of stop token IDs (empty for Granite - use EOS token)
        """
        # Granite uses standard EOS token, no special stop tokens needed
        return []

    def get_max_context_tokens(self):
        """Get maximum context window size"""
        return self.max_context_tokens

    def count_tokens(self, text):
        """
        Count tokens using transformers tokenizer

        Args:
            text: String to tokenize

        Returns:
            int: Number of tokens
        """
        return len(self.tokenizer.encode(text))

    def get_default_system_prompt(self, model_name):
        """Get Granite-specific system prompt"""
        return f"You are DROP a Deep Research On Premise agent designed by IBM. You are currently using the Granite language model (model {model_name})."

    def debug_print_prompt(self, prompt_string):
        """Print Granite-formatted prompt with syntax highlighting"""
        import sys

        # Check if output supports colors (is a TTY)
        use_colors = hasattr(sys.stdout, 'isatty') and sys.stdout.isatty()

        # Check if we're inside a <tools> block and collect JSON
        in_tools_block = False
        tools_json_lines = []

        # Split into lines and process
        lines = prompt_string.split('\n')

        for line in lines:
            # Check for <tools> block start
            if '<tools>' in line:
                in_tools_block = True
                # Print the opening tag
                if use_colors:
                    sys.stdout.write(' \033[96m<tools>\033[0m\n')
                else:
                    sys.stdout.write(' <tools>\n')
                sys.stdout.flush()
                continue

            # Check for </tools> block end
            if '</tools>' in line:
                in_tools_block = False
                # Parse and pretty-print the collected JSON
                if tools_json_lines:
                    # Parse each line as JSON (they might be separate objects)
                    for json_line in tools_json_lines:
                        json_line = json_line.strip()
                        if json_line:
                            parsed = json.loads(json_line)
                            # Pretty print with indentation
                            formatted = json.dumps(parsed, indent=2)
                            for formatted_line in formatted.split('\n'):
                                self._print_colored_json_line(formatted_line, use_colors)

                    tools_json_lines = []

                # Print the closing tag
                if use_colors:
                    sys.stdout.write(' \033[96m</tools>\033[0m\n')
                else:
                    sys.stdout.write(' </tools>\n')
                sys.stdout.flush()
                continue

            # If inside tools block, collect the line
            if in_tools_block:
                tools_json_lines.append(line)
                continue

            # Regular line processing (not in tools block)
            highlighted = line

            if use_colors:
                # Highlight role markers in blue
                highlighted = highlighted.replace('<|start_of_role|>', '\033[94m<|start_of_role|>\033[0m')
                highlighted = highlighted.replace('<|end_of_role|>', '\033[94m<|end_of_role|>\033[0m')
                highlighted = highlighted.replace('<|end_of_text|>', '\033[94m<|end_of_text|>\033[0m')

                # Highlight tool call tags in yellow
                highlighted = highlighted.replace('<tool_call>', '\033[93m<tool_call>\033[0m')
                highlighted = highlighted.replace('</tool_call>', '\033[93m</tool_call>\033[0m')

                # Highlight tool response tags in blue
                highlighted = highlighted.replace('<tool_response>', '\033[94m<tool_response>\033[0m')
                highlighted = highlighted.replace('</tool_response>', '\033[94m</tool_response>\033[0m')

                # Highlight JSON-like content (tool definitions)
                if '"type":' in line or '"function":' in line or '"name":' in line or '"arguments":' in line:
                    highlighted = self._colorize_json_line(highlighted, use_colors)

            # Write directly to stdout with flush to ensure colors render
            sys.stdout.write(f" {highlighted}\n")
            sys.stdout.flush()

    def _colorize_json_line(self, line, use_colors=True):
        """Colorize a JSON line with syntax highlighting"""
        if not use_colors:
            return line
        import re

        # First, color brackets in bold (they don't overlap with quotes)
        highlighted = re.sub(r'([{}\[\]])', r'\033[1m\1\033[0m', line)
        # Then color JSON keys (quoted strings followed by colon)
        highlighted = re.sub(r'"([^"]+)"\s*:', r'\033[96m"\1"\033[0m:', highlighted)
        # Finally color string values (quoted strings after colon, but not followed by colon)
        highlighted = re.sub(r':\s*"([^"]*)"(?!\s*:)', r': \033[93m"\1"\033[0m', highlighted)
        # Color numbers in green
        highlighted = re.sub(r':\s*(-?\d+\.?\d*)', r': \033[92m\1\033[0m', highlighted)
        # Color booleans in magenta
        highlighted = re.sub(r'\b(true|false)\b', r'\033[95m\1\033[0m', highlighted)
        return highlighted

    def _print_colored_json_line(self, line, use_colors=True):
        """Print a JSON line with syntax highlighting"""
        import sys
        highlighted = self._colorize_json_line(line, use_colors)
        sys.stdout.write(f" {highlighted}\n")
        sys.stdout.flush()
