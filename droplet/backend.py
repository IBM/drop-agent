"""Backend server management for Ollama and vLLM"""

import shutil
import subprocess
import time
from abc import ABC, abstractmethod

import requests

from droplet.rich_terminal import blue_print


class BaseBackend(ABC):
    """Abstract base class for LLM backends"""

    def __init__(self, base_url):
        self.base_url = base_url.rstrip('/')
        self.generate_url = None

    @abstractmethod
    def start(self, timeout=30):
        """Start the backend server if needed"""
        pass

    @abstractmethod
    def stop(self):
        """Stop the backend server if we started it"""
        pass

    @abstractmethod
    def ensure_model(self, model_name):
        """Ensure a model is available"""
        pass

    @abstractmethod
    def generate(self, prompt, model, options, timeout=300):
        """
        Generate completion from the backend

        Args:
            prompt: The prompt string
            model: The model name
            options: Generation options dict
            timeout: Request timeout in seconds

        Returns:
            Response dict with generation results
        """
        pass


class OllamaBackend(BaseBackend):
    """Manages the Ollama server backend"""

    def __init__(self, base_url="http://localhost:11434", debug=False):
        super().__init__(base_url)
        self.host = base_url
        self.generate_url = f"{self.base_url}/api/generate"
        self.process = None
        self.debug = debug

    def is_running(self):
        """Check if Ollama server is already running"""
        response = requests.get(f"{self.host}/api/tags")
        return response.status_code == 200

    def _check_server_running(self):
        """
        Check if server is running, return True/False without raising

        Note: Uses try/except for control flow since connection errors
        are expected when server is not running (not a bug to debug)
        """
        try:
            response = requests.get(f"{self.host}/api/tags", timeout=1)
            return response.status_code == 200
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
            return False

    def start(self, timeout=30):
        """
        Start the Ollama server if not already running

        Args:
            timeout: Maximum seconds to wait for server to start
        """
        # 1. Check if ollama binary is available
        if not shutil.which("ollama"):
            raise RuntimeError(
                "Ollama binary not found. Please install Ollama first.\n"
                "See README.md for installation and server activation instructions.\n\n"
                "Quick install (macOS): brew install ollama\n"
                "Or download from: https://ollama.com/download"
            )

        # 2. Check if server is already running
        if self._check_server_running():
            return

        # 3. Server not running, start it
        self.process = subprocess.Popen(
            ["ollama", "serve"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

        # Wait for server to be ready
        start_time = time.time()
        while time.time() - start_time < timeout:
            if self._check_server_running():
                return
            time.sleep(0.5)

        raise RuntimeError(f"Ollama server failed to start within {timeout} seconds")

    def stop(self):
        """Stop the Ollama server if we started it"""
        if self.process:
            blue_print("🤖 Stopping Ollama server...")
            self.process.terminate()
            self.process.wait()
            self.process = None

    def ensure_model(self, model_name):
        """
        Ensure a model is pulled and ready to use

        Args:
            model_name: Name of the model to pull (e.g., 'llama2')
        """
        response = requests.get(f"{self.host}/api/tags")
        models = response.json()["models"]

        if any(model["name"].startswith(model_name) for model in models):
            return

        blue_print(f"🤖 Pulling model '{model_name}'... (this may take a while)")
        result = subprocess.run(
            ["ollama", "pull", model_name],
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            raise RuntimeError(f"Failed to pull model '{model_name}': {result.stderr}")

        blue_print(f"🤖 Model '{model_name}' pulled successfully")

    def generate(self, prompt, model, options, timeout=300):
        """
        Generate completion using Ollama /api/generate endpoint

        Args:
            prompt: The prompt string
            model: The model name
            options: Generation options dict (temperature, max_tokens, etc.)
            timeout: Request timeout in seconds

        Returns:
            Response dict with 'context', 'prompt_eval_count', and optional 'thinking'
        """
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": options,
            # CRITICAL: Override the model's template to get raw completion mode
            # Without this, Ollama wraps our harmony prompts with its own system message
            # and template, breaking token counting and harmony parsing
            # The "{{ .Prompt }}" template passes our prompt through unchanged
            "template": "{{ .Prompt }}"
            # this wont output the tokens
            # "raw": True
        }

        # Retry logic for Ollama server requests
        max_retries = 3
        retry_delay = 0.1
        result = None

        for retry_attempt in range(max_retries):
            response = requests.post(self.generate_url, json=payload, timeout=timeout)

            if response.status_code != 200:
                error_msg = f"Backend returned status {response.status_code}"
                if self.debug:
                    error_msg += f": {response.text[:200]}"

                if retry_attempt < max_retries - 1:
                    print(f"\033[93m⚠️  {error_msg}. Retrying... (attempt {retry_attempt + 1}/{max_retries})\033[0m")
                    time.sleep(retry_delay)
                    continue

                print(f"\033[91m✗ All {max_retries} attempts failed: {error_msg}\033[0m")
                response.raise_for_status()

            response.raise_for_status()
            result = response.json()
            break

        return result


class VLLMBackend(BaseBackend):
    """Backend for vLLM server (OpenAI-compatible API)"""

    def __init__(self, base_url):
        super().__init__(base_url)
        self.host = base_url
        self.generate_url = f"{self.base_url}/v1/completions"

    def start(self, timeout=30):
        """
        vLLM is expected to be running externally, so this is a no-op
        Just check if the server is accessible
        """
        response = requests.get(f"{self.base_url}/health", timeout=5)
        if response.status_code != 200:
            raise RuntimeError(
                f"vLLM server at {self.base_url} is not accessible. "
                f"Please ensure vLLM server is running at the specified URL."
            )

    def stop(self):
        """vLLM is managed externally, so this is a no-op"""
        pass

    def ensure_model(self, model_name):
        """
        vLLM is expected to have the model loaded, so this is a no-op
        Just verify the model is available
        """
        response = requests.get(f"{self.base_url}/v1/models")
        models_data = response.json()
        available_models = [m["id"] for m in models_data["data"]]

        if model_name not in available_models:
            raise RuntimeError(
                f"Model '{model_name}' not found on vLLM server. "
                f"Available models: {', '.join(available_models)}"
            )

    def generate(self, prompt, model, options, timeout=300):
        """
        Generate completion using vLLM /v1/completions endpoint (OpenAI-compatible)

        Args:
            prompt: The prompt string
            model: The model name
            options: Generation options dict (temperature, max_tokens, etc.)
            timeout: Request timeout in seconds

        Returns:
            Response dict converted to Ollama format with 'context', 'prompt_eval_count'
        """
        # vLLM uses OpenAI-compatible API format
        payload = {
            "model": model,
            "prompt": prompt,
            "temperature": options["temperature"],
            "max_tokens": options.get("max_tokens", 32768),
            "stream": False,
            "skip_special_tokens": False,  # Preserve harmony markup tokens
            "return_token_ids": True,  # Return actual token IDs (vLLM v0.10.2+)
        }

        # Add stop tokens if provided
        if "stop_tokens" in options:
            payload["stop_token_ids"] = options["stop_tokens"]

        # Retry logic for vLLM server requests
        max_retries = 3
        retry_delay = 0.1

        for retry_attempt in range(max_retries):
            response = requests.post(self.generate_url, json=payload, timeout=timeout)

            if response.status_code != 200:
                error_msg = f"Backend returned status {response.status_code}: {response.text[:200]}"

                if retry_attempt < max_retries - 1:
                    print(f"\033[93m⚠️  {error_msg}. Retrying... (attempt {retry_attempt + 1}/{max_retries})\033[0m")
                    time.sleep(retry_delay)
                    continue

                print(f"\033[91m✗ All {max_retries} attempts failed: {error_msg}\033[0m")
                response.raise_for_status()

            break

        result = response.json()

        choice = result["choices"][0]

        # Convert vLLM response format to Ollama-compatible format
        # With return_token_ids=True, vLLM returns actual token IDs
        # vLLM returns: {"prompt_token_ids": [...], "choices": [{"text": ..., "token_ids": [...]}]}
        # Ollama expects: {"context": [...], "prompt_eval_count": N, "response": "..."}

        response_text = choice["text"]

        # Get actual token IDs from vLLM (requires vLLM v0.10.2+)
        # Both are in the choice object
        prompt_token_ids = choice["prompt_token_ids"]
        response_token_ids = choice["token_ids"]

        # Build context as concatenation of prompt and response tokens
        context_tokens = prompt_token_ids + response_token_ids

        # Return in Ollama-compatible format
        return {
            "response": response_text,
            "context": context_tokens,
            "prompt_eval_count": len(prompt_token_ids),
        }


class RITSBackend(VLLMBackend):
    """Backend for RITS (vLLM-based API with per-model endpoints)"""

    def __init__(self, base_url, api_key):
        # Don't call super().__init__ yet - we need to fetch the model endpoint first
        self.api_key = api_key
        self.rits_info_url = "https://rits.fmaas.res.ibm.com/ritsapi/inferenceinfo"
        self.model_endpoints = None
        # Placeholder until we fetch the actual endpoint
        self.base_url = base_url
        self.host = base_url

    def _fetch_available_models(self):
        """Fetch list of available models from RITS API"""
        from collections import Counter
        from urllib.parse import urlparse

        headers = {"RITS_API_KEY": self.api_key}
        response = requests.get(self.rits_info_url, headers=headers, timeout=10)

        if response.status_code != 200:
            raise RuntimeError(
                f"Failed to fetch RITS model list (status {response.status_code}). "
                f"Please check your RITS API key.\n{response.text}"
            )

        # Store raw model data for listing purposes
        model_data = response.json()
        self.raw_model_data = model_data

        # Count occurrences of each model name to find duplicates
        name_counts = Counter(m["model_name"] for m in model_data)

        # Build map of model_name -> endpoint, filtering out invalid mappings
        self.model_endpoints = {}
        for model in model_data:
            model_name = model["model_name"]
            endpoint = model["endpoint"]

            # If name is unique, always include it
            if name_counts[model_name] == 1:
                self.model_endpoints[model_name] = f"{endpoint}/v1"
            else:
                # Name appears multiple times - only use if basename matches
                parsed_url = urlparse(endpoint)
                basename = parsed_url.path.strip('/').split('/')[-1]

                # Check if basename matches the model name (or last part after /)
                model_basename = model_name.split('/')[-1]
                if basename == model_basename:
                    self.model_endpoints[model_name] = f"{endpoint}/v1"

        return self.model_endpoints

    def _suggest_similar_models(self, requested_model, available_models):
        """Suggest similar model names in case of typo"""
        from difflib import get_close_matches

        suggestions = get_close_matches(requested_model, available_models, n=3, cutoff=0.6)
        return suggestions

    def start(self, timeout=30):
        """
        Fetch available models from RITS - no server to start

        Note: Uses try/except for control flow since connection errors
        are expected when RITS is unreachable (not a bug to debug)
        """
        try:
            self._fetch_available_models()
        except (requests.exceptions.ConnectTimeout, requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
            raise RuntimeError(
                f"Failed to connect to RITS server.\n"
                f"Connection to rits.fmaas.res.ibm.com timed out.\n"
                f"Please check your network connection or VPN."
            )

    def stop(self):
        """RITS is managed externally, so this is a no-op"""
        pass

    def ensure_model(self, model_name):
        """
        Verify the model is available on RITS and set the correct endpoint
        """
        if self.model_endpoints is None:
            self._fetch_available_models()

        if model_name not in self.model_endpoints:
            # Model not found - suggest similar ones
            suggestions = self._suggest_similar_models(model_name, list(self.model_endpoints.keys()))

            error_msg = f"Model '{model_name}' not found on RITS."

            if suggestions:
                error_msg += "\n\nDid you mean one of these?\n  • " + "\n  • ".join(suggestions)
            else:
                available = list(self.model_endpoints.keys())
                error_msg += f"\n\nAvailable models: {', '.join(available[:5])}"
                if len(available) > 5:
                    error_msg += f", ... ({len(available)} total)"

            raise RuntimeError(error_msg)

        # Set the base_url to the model-specific endpoint
        self.base_url = self.model_endpoints[model_name]
        self.host = self.base_url
        self.generate_url = f"{self.base_url}/completions"

        # Verify the endpoint is accessible
        # Health endpoint is at base without /v1
        base_endpoint = self.base_url.rstrip('/v1')
        headers = {"RITS_API_KEY": self.api_key}
        health_url = f"{base_endpoint}/health"
        response = requests.get(health_url, headers=headers, timeout=2)
        if response.status_code != 200:
            raise RuntimeError(
                f"RITS endpoint for model '{model_name}' is not accessible.\n"
                f"Endpoint: {self.base_url}\n"
                f"Health check returned status {response.status_code}"
            )

    def generate(self, prompt, model, options, timeout=300):
        """
        Generate completion using RITS /v1/completions endpoint (OpenAI-compatible)
        Adds RITS_API_KEY header to requests
        """
        # vLLM uses OpenAI-compatible API format
        payload = {
            "model": model,
            "prompt": prompt,
            "temperature": options["temperature"],
            "max_tokens": options.get("max_tokens", 32768),
            "stream": False,
            "skip_special_tokens": False,
            "return_token_ids": True,
        }

        # Add stop tokens if provided
        if "stop_tokens" in options:
            payload["stop_token_ids"] = options["stop_tokens"]

        # Add RITS API key header
        headers = {"RITS_API_KEY": self.api_key}

        response = requests.post(self.generate_url, json=payload, headers=headers, timeout=timeout)

        if response.status_code != 200:
            error_msg = f"RITS returned status {response.status_code}: {response.text}"
            print(f"\033[91mERROR: {error_msg}\033[0m")
            response.raise_for_status()

        result = response.json()
        choice = result["choices"][0]

        response_text = choice["text"]
        prompt_token_ids = choice["prompt_token_ids"]
        response_token_ids = choice["token_ids"]

        context_tokens = prompt_token_ids + response_token_ids

        return {
            "response": response_text,
            "context": context_tokens,
            "prompt_eval_count": len(prompt_token_ids),
        }
