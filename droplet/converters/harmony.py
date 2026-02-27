"""Harmony message converter for GPT-OSS models"""

import time

import tiktoken
from openai_harmony import (Conversation, HarmonyEncodingName, Message, Role,
                            StreamableParser, load_harmony_encoding)

from droplet.converters.base import MessageConverter


def load_tiktoken_with_retry(encoding_name, max_retries=3, delay=2):
    """
    Load tiktoken encoding with retry logic for network issues

    Args:
        encoding_name: Name of the tiktoken encoding to load
        max_retries: Maximum number of retry attempts
        delay: Delay in seconds between retries

    Returns:
        tiktoken.Encoding object

    Raises:
        RuntimeError: If encoding cannot be loaded after all retries
    """
    last_error = None

    for attempt in range(max_retries):
        try:
            return tiktoken.get_encoding(encoding_name)
        except Exception as e:
            last_error = e
            error_msg = str(e)

            if "error downloading" in error_msg.lower() or "error decoding" in error_msg.lower():
                if attempt < max_retries - 1:
                    print(f"\n⚠️  Failed to download tiktoken encoding (attempt {attempt + 1}/{max_retries}). Retrying in {delay}s...")
                    time.sleep(delay)
                    continue
            raise RuntimeError(f"Failed to load tiktoken encoding '{encoding_name}': {error_msg}")

    raise RuntimeError(f"Failed to load tiktoken encoding '{encoding_name}' after {max_retries} attempts: {last_error}")


class HarmonyMessageConverter(MessageConverter):
    """
    Harmony-based message converter for GPT-OSS models.

    Uses openai_harmony encoding for message formatting and parsing.
    Supports the full harmony markup with tool calls, reasoning, etc.
    """

    def __init__(self):
        """Initialize harmony encoding and tiktoken for token/text conversion"""
        self.encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
        # Use tiktoken for token-to-text conversion with retry logic
        self.tiktoken_encoding = load_tiktoken_with_retry("o200k_harmony")

    def messages_to_tokens(self, messages):
        """
        Convert message list to token IDs

        Args:
            messages: List of Message objects or dicts

        Returns:
            List of token IDs
        """
        # Convert dicts to Message objects if needed
        message_objects = []
        for msg in messages:
            if isinstance(msg, dict):
                message_objects.append(Message.from_dict(msg))
            else:
                message_objects.append(msg)

        # Create conversation and render to tokens
        conversation = Conversation.from_messages(message_objects)
        tokens = self.encoding.render_conversation_for_completion(conversation, Role.ASSISTANT)
        return tokens

    def messages_to_prompt_string(self, messages):
        """Convert message list to a formatted string prompt using harmony rendering"""
        # Convert to tokens using harmony
        tokens = self.messages_to_tokens(messages)

        # Decode tokens to string
        prompt_string = self.tiktoken_encoding.decode(tokens)

        return prompt_string

    def tokens_to_messages(self, tokens):
        """
        Parse token IDs back to Message objects

        Args:
            tokens: List of token IDs

        Returns:
            List of Message objects
        """
        messages = self.encoding.parse_messages_from_completion_tokens(tokens, Role.ASSISTANT)
        return list(messages)

    def response_string_to_messages(self, response_text):
        """Parse response string back to Message objects using harmony"""
        # Encode string to tokens
        tokens = self.tiktoken_encoding.encode(response_text)

        # Parse tokens back to messages
        messages = self.tokens_to_messages(tokens)

        return messages

    def get_stop_tokens(self):
        """Get harmony stop token IDs for assistant actions"""
        return self.encoding.stop_tokens_for_assistant_actions()

    def get_max_context_tokens(self):
        """Get maximum context window size for GPT-OSS models"""
        return 128000

    def count_tokens(self, text):
        """
        Count tokens in text using tiktoken encoding

        Args:
            text: String to tokenize

        Returns:
            int: Number of tokens
        """
        return len(self.tiktoken_encoding.encode(text, allowed_special='all'))

    def create_parser(self):
        """Create a streamable parser for incremental token processing"""
        return StreamableParser(self.encoding, role=Role.ASSISTANT)

    def get_default_system_prompt(self, model_name):
        """Get GPT-OSS specific system prompt"""
        return f"You are DROP a Deep Research On Premise agent designed by IBM. You are currently using the backend ChatGPT, a large language model trained by OpenAI (model {model_name})."

    def debug_print_prompt(self, prompt_string):
        """Print harmony-formatted prompt with syntax highlighting"""
        from droplet.rich_terminal import \
            debug_print_prompt as harmony_debug_print
        harmony_debug_print(prompt_string)
