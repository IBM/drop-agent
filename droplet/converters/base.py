"""Base abstract class for message converters"""

from abc import ABC, abstractmethod


class MessageConverter(ABC):
    """
    Abstract base class for message format converters.

    Different model families require different message formatting and parsing.
    This interface defines the contract all converters must implement to enable
    model-independent agent operation.
    """

    @abstractmethod
    def messages_to_prompt_string(self, messages):
        """
        Convert message list to a formatted prompt string for the backend.

        Args:
            messages: List of Message objects or dicts with message data

        Returns:
            str: Formatted prompt string ready for backend consumption
        """
        pass

    @abstractmethod
    def response_string_to_messages(self, response_text):
        """
        Parse backend response text back to Message objects.

        Args:
            response_text: Raw text response from backend

        Returns:
            List of Message objects parsed from response
        """
        pass

    @abstractmethod
    def get_stop_tokens(self):
        """
        Get model-specific stop token IDs.

        Returns:
            List of integer token IDs that signal completion
        """
        pass

    @abstractmethod
    def get_max_context_tokens(self):
        """
        Get maximum context window size in tokens.

        Returns:
            int: Maximum number of tokens the model can handle
        """
        pass

    @abstractmethod
    def count_tokens(self, text):
        """
        Count tokens in text using model's tokenizer.

        Args:
            text: String to tokenize

        Returns:
            int: Number of tokens
        """
        pass

    def get_default_system_prompt(self, model_name):
        """
        Get model-specific default system prompt.

        Args:
            model_name: Name of the model

        Returns:
            str: Default system prompt for this model family
        """
        return f"You are a helpful AI assistant using the {model_name} model."

    def debug_print_prompt(self, prompt_string):
        """
        Print prompt string in debug mode with model-specific formatting.

        Args:
            prompt_string: The formatted prompt string to display
        """
        # Default implementation: just print with line numbers
        lines = prompt_string.split('\n')
        for i, line in enumerate(lines, 1):
            print(f" {i:4d} | {line}")
