"""Generation orchestrator for message-level LLM operations"""

import time
from dataclasses import dataclass
from typing import List

import requests
from openai_harmony import Message


@dataclass
class GenerationResult:
    """Result from a generation request."""
    messages: List[Message]
    prompt_token_count: int
    response_token_count: int
    elapsed_time: float


class GenerationError(Exception):
    """Base class for generation errors."""
    pass


class Backend500Error(GenerationError):
    """Backend returned HTTP 500."""
    def __init__(self, message, prompt_token_count):
        super().__init__(message)
        self.prompt_token_count = prompt_token_count


class BackendConnectionError(GenerationError):
    """Backend connection failed."""
    pass


class BackendHTTPError(GenerationError):
    """Backend returned non-500 HTTP error."""
    def __init__(self, message, status_code, response_text):
        super().__init__(message)
        self.status_code = status_code
        self.response_text = response_text


class GenerationOrchestrator:
    """
    Orchestrates message-to-message generation flow.

    Encapsulates: messages → prompt → backend → response → messages
    Handles both token-based (Harmony) and text-based (Granite) parsing.
    Catches and wraps backend errors.
    """

    def __init__(self, backend, converter):
        self.backend = backend
        self.converter = converter

    def generate_messages(self, messages, model, options, timeout=300):
        """
        Generate completion from messages, return parsed messages.

        Args:
            messages: List of Message objects
            model: Model name
            options: Generation options dict
            timeout: Request timeout in seconds

        Returns:
            GenerationResult with parsed messages and token counts

        Raises:
            Backend500Error: When backend returns HTTP 500
            BackendHTTPError: When backend returns other HTTP error
            BackendConnectionError: When backend connection fails
            GenerationError: For unexpected errors
            HarmonyError: When parsing fails (only Harmony models) - NOT caught
        """
        start_time = time.time()

        # 1. Convert messages to prompt string
        prompt_string = self.converter.messages_to_prompt_string(messages)
        prompt_token_count = self.converter.count_tokens(prompt_string)

        # 2. Call backend with error handling
        try:
            result = self.backend.generate(
                prompt=prompt_string,
                model=model,
                options=options,
                timeout=timeout
            )
        except requests.exceptions.HTTPError as e:
            if hasattr(e, 'response') and e.response is not None:
                status_code = e.response.status_code
                if status_code == 500:
                    raise Backend500Error(
                        "Backend server error (500)",
                        prompt_token_count=prompt_token_count
                    )
                else:
                    raise BackendHTTPError(
                        f"Backend returned status {status_code}",
                        status_code=status_code,
                        response_text=e.response.text[:200]
                    )
            raise GenerationError(f"Backend HTTP error: {e}")
        except requests.exceptions.RequestException as e:
            raise BackendConnectionError(f"Backend connection error: {e}")
        except Exception as e:
            raise GenerationError(f"Unexpected error during generation: {e}")

        # 3. Parse response based on converter capability
        if hasattr(self.converter, 'tokens_to_messages') and "context" in result:
            # Token-based parsing (Harmony) - let HarmonyError bubble up
            response_tokens = result["context"][prompt_token_count:]
            parsed_messages = self.converter.tokens_to_messages(response_tokens)
            response_token_count = len(response_tokens)
        else:
            # Text-based parsing (Granite)
            response_text = result["response"]
            parsed_messages = self.converter.response_string_to_messages(response_text)
            response_token_count = self.converter.count_tokens(response_text)

        elapsed_time = time.time() - start_time

        return GenerationResult(
            messages=parsed_messages,
            prompt_token_count=prompt_token_count,
            response_token_count=response_token_count,
            elapsed_time=elapsed_time
        )
