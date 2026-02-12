"""Agent scaffold using completions API with Harmony message conversion"""

import asyncio
import datetime
import json
import logging
import time

import requests
import tiktoken
from openai_harmony import (Author, Conversation, HarmonyEncodingName, HarmonyError, Message,
                            ReasoningEffort, Role, StreamableParser, SystemContent,
                            TextContent, load_harmony_encoding)

from droplet.backend import OllamaBackend, RITSBackend, VLLMBackend
from droplet.rich_terminal import (blue_print, debug_print_error,
                                   debug_print_prompt)


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

# Configure logging to reduce verbosity from gpt_oss and related libraries
# gpt_oss uses structlog, so we need to configure it
try:
    import structlog

    # Get the current configuration
    processors = structlog.get_config().get("processors", [])

    # Add a filter to drop warning/error logs from gpt_oss components
    def filter_gpt_oss_logs(logger, method_name, event_dict):
        # Drop logs from gpt_oss components
        if "component" in event_dict:
            component = event_dict.get("component", "")
            if component.startswith("gpt_oss"):
                raise structlog.DropEvent
        return event_dict

    # Reconfigure structlog with our filter
    structlog.configure(
        processors=[filter_gpt_oss_logs] + processors,
    )
except ImportError:
    pass

# Also configure standard logging as fallback
for logger_name in [
    'gpt_oss',
    'gpt_oss.tools',
    'gpt_oss.tools.simple_browser',
    'gpt_oss.tools.simple_browser.simple_browser_tool',
]:
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.CRITICAL)
    logger.propagate = False




class HarmonyMessageConverter:
    """Encapsulates harmony encoding/decoding logic for message conversion"""

    def __init__(self):
        self.encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
        # Use tiktoken for token-to-text conversion with retry logic
        self.tiktoken_encoding = load_tiktoken_with_retry("o200k_harmony")

    def messages_to_tokens(self, messages):
        """Convert message list to token IDs"""
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
        """Parse token IDs back to Message objects"""
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

    def create_parser(self):
        """Create a streamable parser for incremental token processing"""
        return StreamableParser(self.encoding, role=Role.ASSISTANT)


class DropletAgent:
    """
    Agent scaffold using pluggable backends (Ollama or vLLM) with Harmony message conversion

    This implementation demonstrates proper harmony encoding/decoding for completions API:
    - Uses HarmonyMessageConverter to convert messages -> tokens -> string (for prompt)
    - Calls backend's generate() method (text-based completions API)
    - Parses response tokens back to messages using harmony (for tool calls)

    Supported Backends:
    - Ollama: Local backend using /api/generate endpoint
    - vLLM: Remote backend using OpenAI-compatible /v1/completions endpoint

    Key Design:
    - Harmony encoding formats the input prompt with proper message structure
    - Tools are included in the system message via SystemContent.with_tools()
    - Response parsing uses backend's 'context' field (full token sequence)
    - New tokens are extracted and parsed back to messages via harmony
    - Tool calls are properly detected via the 'recipient' field in parsed messages

    See HarmonyMessageConverter for the encapsulated conversion logic.
    """

    # Class-level prompt constants
    INITIAL_PROMPT = (
        "Please list the contents of the current directory and provide a summary of what you find. Avoid the use of "
        "tables. Finally ask the user they have any question about this content and propose some possible questions. "
        "Number the options."
    )

    LOOP_TOOL_FAIL = (
        "I couldn't find what you were looking for after several attempts. Can you rephrase your question or provide "
        "more details about what you need?"
    )

    def __init__(
        self,
        model="gpt-oss:20b",
        base_url="http://localhost:11434",
        backend_type="ollama",
        rits_api_key=None,
        debug=False,
        # io
        log_file=None,
        out_messages=None,
        # tool config
        restricted_tools=None,
        tool_names=None,
        # milvus tools
        milvus_db=None,
        milvus_model=None,
        milvus_collection=None,
        # bcp tool
        bcp_server_url=None,
        # semantic scholar
        semantic_scholar_api_key=None,
        # prompts
        no_droplet_sytem_prompt=False,
        system_prompt=None,
        developer_prompt=None,
        initial_prompt=None,
        loop_tool_fail=None,
        gpt_reasoning=None,
        # generation
        temperature=0.0,
        max_tokens=32768,
        max_iterations=10,
    ):
        """
        Initialize the agent

        Args:
            model: Model name (default: gpt-oss:20b)
            base_url: Base URL for backend API (default: http://localhost:11434)
            backend_type: Backend type - "ollama", "vllm", or "rits-vllm" (default: ollama)
            debug: Enable debug mode (default: False)
            restricted_tools: Set of tool class names that require user permission
            tool_names: List of tool class names to load (default: ['FileBrowserTool'])
            milvus_db: Path to Milvus database file (required for RetrieverBrowserTool)
            milvus_model: SentenceTransformer model name or path
            milvus_collection: Milvus collection name
            log_file: Path to JSON file to log conversation history
            out_messages: Path to JSON file to write final message list (for batch evaluation)
            bcp_server_url: BCP search server URL (required for BCPBrowserTool)
            rits_api_key: RITS API key (required for rits-vllm backend)
            semantic_scholar_api_key: Semantic Scholar API key (optional, for higher rate limits)
            no_droplet_sytem_prompt: Disable default Droplet system prompt (default: False)
            system_prompt: Override system prompt (used if no_droplet_sytem_prompt=True)
            developer_prompt: Additional developer instructions added as developer message
            initial_prompt: Override default initial prompt (default: class-level INITIAL_PROMPT)
            loop_tool_fail: Override default loop failure message (default: class-level LOOP_TOOL_FAIL)
            gpt_reasoning: GPT-OSS reasoning effort level: "low", "medium", or "high" (default: None, uses model default)
            temperature: Sampling temperature (default: 0.0)
            max_tokens: Maximum tokens to generate (default: 32768)
            max_iterations: Maximum tool call iterations per user input (default: 10)
        """
        # Initialize backend based on type
        if backend_type == "ollama":
            self.backend = OllamaBackend(base_url=base_url, debug=debug)
        elif backend_type == "vllm":
            self.backend = VLLMBackend(base_url=base_url)
        elif backend_type == "rits-vllm":
            if not rits_api_key:
                raise RuntimeError("RITS API key is required for rits-vllm backend. Use --rits-api-key argument.")
            self.backend = RITSBackend(base_url=base_url, api_key=rits_api_key)
        else:
            raise RuntimeError(f"Unknown backend type: {backend_type}. Must be 'ollama', 'vllm', or 'rits-vllm'")

        self.backend.start()
        self.backend.ensure_model(model)

        self.model = model
        self.debug = debug
        self.base_url = base_url.rstrip('/')

        # Get model info from Ollama and setup tokenizer
        self._setup_tokenizer_and_limits()

        # Initialize harmony converter
        self.harmony = HarmonyMessageConverter()

        # generator options
        self.options = {
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stop_tokens": self.harmony.get_stop_tokens(),  # Harmony stop tokens
        }
        self.max_iterations = max_iterations

        # Initialize state
        self.state = {}

        # Tool permission tracking
        self.restricted_tools = restricted_tools or set()
        self.allowed_tools = set()  # Tools allowed for this session

        # Log file for conversation history
        if log_file and not log_file.endswith('.json'):
            raise RuntimeError(f"Log file must end with .json, got: {log_file}")
        self.log_file = log_file

        # Output messages file for batch evaluation
        if out_messages and not out_messages.endswith('.json'):
            raise RuntimeError(f"Output messages file must end with .json, got: {out_messages}")
        self.out_messages = out_messages

        # Store prompts and reasoning level for later use
        self.developer_prompt = developer_prompt
        self.gpt_reasoning = gpt_reasoning

        # Setup prompts before initializing tools (needed by _setup_tool_messages)
        if not no_droplet_sytem_prompt:
            self.SYSTEM_PROMPT = (
                "You are DROP a Deep Research On Premise agent designed by IBM. You are currently using the backend "
                "ChatGPT, a large language model trained by OpenAI (model {}).".format(self.model)
            )
        else:
            self.SYSTEM_PROMPT = system_prompt

        # Set instance-level prompts (use class defaults if not provided)
        if initial_prompt is not None:
            self.INITIAL_PROMPT = initial_prompt
        else:
            self.INITIAL_PROMPT = DropletAgent.INITIAL_PROMPT

        if loop_tool_fail is not None:
            self.LOOP_TOOL_FAIL = loop_tool_fail
        else:
            self.LOOP_TOOL_FAIL = DropletAgent.LOOP_TOOL_FAIL

        # Initialize tools
        tools = self._initialize_tools(tool_names, milvus_db, milvus_model, milvus_collection, bcp_server_url, semantic_scholar_api_key)

        # Setup initial conversation with system messages and tools
        initial_messages, self.tool_instances = self._setup_tool_messages(tools)
        self.conversation_history = initial_messages

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup backend"""
        self.backend.stop()
        return False

    def _save_conversation_log(self):
        """Save conversation history to JSON file"""
        if not self.log_file:
            return

        # Convert Message objects to dicts
        conversation_dicts = []
        for msg in self.conversation_history:
            if hasattr(msg, 'to_dict'):
                conversation_dicts.append(msg.to_dict())
            else:
                # Fallback if to_dict not available
                conversation_dicts.append(msg)

        # Write to file with indentation
        with open(self.log_file, 'w') as f:
            json.dump(conversation_dicts, f, indent=2)

    def _save_out_messages(self):
        """Save final message list to JSON file for batch evaluation"""
        if not self.out_messages:
            return

        # Convert Message objects to dicts
        message_dicts = []
        for msg in self.conversation_history:
            if hasattr(msg, 'to_dict'):
                message_dicts.append(msg.to_dict())
            else:
                # Fallback if to_dict not available
                message_dicts.append(msg)

        # Write to file with indentation
        with open(self.out_messages, 'w') as f:
            json.dump(message_dicts, f, indent=2)

    def _initialize_tools(self, tool_names, milvus_db, milvus_model, milvus_collection, bcp_server_url, semantic_scholar_api_key):
        """
        Initialize tool instances from tool names

        Args:
            tool_names: List of tool class names to load
            milvus_db: Path to Milvus database file
            milvus_model: SentenceTransformer model name
            milvus_collection: Milvus collection name
            bcp_server_url: BCP search server URL
            semantic_scholar_api_key: Semantic Scholar API key

        Returns:
            List of tool instances
        """
        import inspect

        from droplet import tools as tools_module

        # Use default tools if none specified
        if tool_names is None:
            tool_names = ['FileBrowserTool']

        # Discover all *Tool classes from droplet.tools
        # Exclude base classes
        excluded_tools = {'SimpleFunctionTool'}
        available_tools = {}
        for name, obj in inspect.getmembers(tools_module, inspect.isclass):
            if name.endswith('Tool') and name not in excluded_tools:
                if hasattr(obj, '__module__') and obj.__module__.startswith('droplet'):
                    available_tools[name] = obj

        # Initialize tool instances
        tools = []
        for tool_name in tool_names:
            if tool_name not in available_tools:
                raise RuntimeError(
                    f"Unknown tool '{tool_name}'. "
                    f"Available tools: {', '.join(sorted(available_tools.keys()))}"
                )

            tool_class = available_tools[tool_name]

            # Special handling for RetrieverBrowserTool which requires Milvus configuration
            if tool_name == 'RetrieverBrowserTool':
                if not milvus_db:
                    raise RuntimeError(
                        "RetrieverBrowserTool requires milvus_db argument"
                    )

                tools.append(tool_class(
                    milvus_db=milvus_db,
                    milvus_model=milvus_model,
                    milvus_collection=milvus_collection
                ))
            # BCPBrowserTool requires BCP server URL
            elif tool_name == 'BCPBrowserTool':
                if not bcp_server_url:
                    raise RuntimeError(
                        "BCPBrowserTool requires bcp_server_url argument (use --bcp-server-url)"
                    )
                tools.append(tool_class(base_url=bcp_server_url))
            # SemanticScholarTool accepts optional API key
            elif tool_name == 'SemanticScholarTool':
                tools.append(tool_class(api_key=semantic_scholar_api_key))
            else:
                tools.append(tool_class())

        if not tools:
            raise RuntimeError("No tools were loaded")

        return tools

    def _setup_tokenizer_and_limits(self):
        """
        Setup tokenizer and context limits for supported models
        """
        # Only support gpt-oss models
        if "gpt-oss" not in self.model:
            raise RuntimeError(
                f"Unsupported model '{self.model}'. "
                f"Only gpt-oss models are currently supported."
            )

        # FIXME: Hard-coded values for gpt-oss
        self.max_context_tokens = 128000
        self.encoding = load_tiktoken_with_retry("o200k_harmony")

    def _format_tokens(self, tokens):
        """Format token count as thousands (K)"""
        if tokens >= 1000:
            result = f"{tokens / 1000:.1f}K"
        else:
            result = str(tokens)
        return result

    def _ask_tool_permission(self, tool_class_name, function_name, function_args):
        """Ask user for permission to execute a restricted tool"""
        # Check if already allowed for this session (check by tool class, not function)
        permission_granted = tool_class_name in self.allowed_tools

        if not permission_granted:
            print("\n \033[93m⚠️\033[0m  \033[94mRestricted Tool Request\033[0m")

            # Special handling for PythonTool - pretty print the code
            if tool_class_name == "PythonTool":
                # Extract the script from function_args
                script = function_args.get('script', '')

                print(f" \033[94mTool: {tool_class_name}.{function_name}\033[0m")
                print("\n \033[94mPython code to execute:\033[0m")

                # Pretty print Python code using rich
                from rich.console import Console
                from rich.syntax import Syntax

                console = Console()
                syntax = Syntax(script, "python", theme="monokai", line_numbers=True)
                console.print(syntax)
            else:
                # Format arguments for display (non-Python tools)
                args_str = ", ".join(f"{k}={repr(v)}" for k, v in function_args.items())
                print(f" \033[94mTool: {function_name}({args_str})\033[0m")

            print("\n \033[94mOptions:\033[0m")
            print(" \033[94m  [1] Yes, execute once\033[0m")
            print(f" \033[94m  [2] Yes, allow all calls to {tool_class_name} this session\033[0m")
            print(" \033[94m  [3] No, cancel\033[0m")

            valid_choice = False
            while not valid_choice:
                choice = input(" \033[94mYour choice:\033[0m ").strip()

                if choice == '1':
                    permission_granted = True
                    valid_choice = True
                elif choice == '2':
                    self.allowed_tools.add(tool_class_name)
                    print(f" \033[94m✓ All '{tool_class_name}' calls allowed for this session\033[0m")
                    permission_granted = True
                    valid_choice = True
                elif choice == '3':
                    print(" \033[94m✗ Tool call cancelled\033[0m")
                    permission_granted = False
                    valid_choice = True
                else:
                    print(" \033[94mInvalid choice. Please enter 1, 2, or 3\033[0m")

        return permission_granted

    def _setup_tool_messages(self, tools):
        """Setup initial messages with tool configuration"""
        tool_instances = {}

        if tools:
            # Register each tool with its own namespace (files.search, wiki.search, etc.)
            # This matches gpt-oss SimpleBrowserTool behavior

            tool_configs = []
            for t in tools:
                tool_config = t.tool_config
                tool_configs.append(tool_config)

                # Register tool instance for each function in its namespace
                namespace_name = tool_config.name
                # Convert to dict to access tool definitions
                tool_config_dict = tool_config.model_dump()
                tool_defs = tool_config_dict.get('tools', [])

                if len(tool_defs) == 0:
                    # Direct-content tool (no functions) - register by namespace name only
                    # E.g., "python" for PythonTool
                    tool_instances[namespace_name] = t
                    tool_instances[f"functions.{namespace_name}"] = t
                else:
                    # Function-based tool - register each function
                    for tool_def in tool_defs:
                        # Format: "namespace.function" (e.g., "files.search", "wiki.open")
                        function_name = f"{namespace_name}.{tool_def['name']}"
                        tool_instances[function_name] = t
                        # Also register with "functions." prefix since Harmony may add it
                        tool_instances[f"functions.{function_name}"] = t
                        # FIXME: Also register with "browser." prefix since model may use that for search tools
                        tool_instances[f"browser.{function_name}"] = t

            # Create SystemContent with multiple tool namespaces
            system_content = SystemContent.new()

            # Set reasoning effort if specified
            if self.gpt_reasoning:
                reasoning_map = {
                    "low": ReasoningEffort.LOW,
                    "medium": ReasoningEffort.MEDIUM,
                    "high": ReasoningEffort.HIGH,
                }
                if self.gpt_reasoning.lower() in reasoning_map:
                    system_content = system_content.with_reasoning_effort(reasoning_map[self.gpt_reasoning.lower()])

            # Set conversation start date (important for proper harmony formatting)
            system_content = system_content.with_conversation_start_date(
                datetime.datetime.now().strftime("%Y-%m-%d")
            )

            for config in tool_configs:
                system_content = system_content.with_tools(config)

            # DON'T set model_identity when using tools - the tools configuration handles harmony format
            # If custom prompt needed, use developer_prompt instead
            system_message = Message.from_role_and_content(Role.SYSTEM, system_content)

            messages = [system_message]

            # Add developer prompt as an additional developer message if provided
            if self.developer_prompt:
                developer_message = Message.from_role_and_content(
                    Role.DEVELOPER,
                    TextContent(text=self.developer_prompt)
                )
                messages.append(developer_message)
        else:
            # No tools - create SystemContent for reasoning effort
            system_content = SystemContent.new()

            # Set reasoning effort if specified
            if self.gpt_reasoning:
                reasoning_map = {
                    "low": ReasoningEffort.LOW,
                    "medium": ReasoningEffort.MEDIUM,
                    "high": ReasoningEffort.HIGH,
                }
                if self.gpt_reasoning.lower() in reasoning_map:
                    system_content = system_content.with_reasoning_effort(reasoning_map[self.gpt_reasoning.lower()])

            # Set conversation start date (important for proper harmony formatting)
            system_content = system_content.with_conversation_start_date(
                datetime.datetime.now().strftime("%Y-%m-%d")
            )

            # Only override system prompt if explicitly provided (keeps default ChatGPT identity otherwise)
            if self.SYSTEM_PROMPT is not None:
                system_content.model_identity = self.SYSTEM_PROMPT

            system_message = Message.from_role_and_content(Role.SYSTEM, system_content)
            messages = [system_message]

            # Add developer prompt as an additional developer message if provided
            if self.developer_prompt:
                developer_message = Message.from_role_and_content(
                    Role.DEVELOPER,
                    TextContent(text=self.developer_prompt)
                )
                messages.append(developer_message)

        return messages, tool_instances

    def _execute_tool_call(self, last_message, tool_instances):
        """Execute a single tool call and return result messages"""
        # Extract tool call information from harmony message
        recipient = str(last_message.recipient)

        # Recipient should be "functions.files_search", "functions.wiki_open", etc.
        # Extract just the function name part for display
        if recipient.startswith("functions."):
            function_name = recipient[10:]  # Remove "functions." prefix for display
        else:
            function_name = recipient

        # Extract arguments from message content
        content_text = last_message.content[0].text if last_message.content else ""

        # Check if tool exists first to determine how to parse content
        tool = tool_instances.get(recipient)

        # Determine if this is a direct-content tool (like PythonTool) or function-based tool
        # Direct-content tools have empty tools array in their config
        is_direct_content_tool = False
        if tool:
            tool_config = tool.tool_config
            if hasattr(tool_config, 'tools') and len(tool_config.tools) == 0:
                is_direct_content_tool = True

        # Parse arguments based on tool type
        if is_direct_content_tool:
            # Direct content tool - content is the raw data (not JSON)
            function_args = {"script": content_text}  # Wrap for display purposes
        else:
            # Function-based tool - content is JSON with arguments
            if not content_text or not content_text.strip():
                function_args = {}
            else:
                function_args = json.loads(content_text)

        # Format and display tool call with tool class name
        if is_direct_content_tool:
            # For direct content tools, show a summary instead of full content
            content_preview = content_text[:50] + "..." if len(content_text) > 50 else content_text
            args_str = f"script='{content_preview}'"
        else:
            args_str = ", ".join(f"{k}={repr(v)}" for k, v in function_args.items())

        if recipient in tool_instances:
            tool_class_name = tool_instances[recipient].__class__.__name__
        else:
            tool_class_name = "Unknown"
        blue_print(f"🔧 {tool_class_name}.{function_name}({args_str})")

        # Check if tool exists (use full recipient as key)
        if recipient not in tool_instances:
            # Tool not found - return error to model
            available_tools = ", ".join(sorted(tool_instances.keys()))
            result = {
                "error": "Tool not found",
                "message": f"Tool '{recipient}' is not available. Available tools: {available_tools}"
            }

            result_messages = [Message(
                author=Author(role=Role.TOOL, name=recipient),
                content=[TextContent(text=json.dumps(result))],
            )]
            # Print error in red
            debug_print_error(f"Tool '{recipient}' not found")
            return result_messages

        # Check if tool requires permission and see if user grants it
        requires_permission = tool_class_name in self.restricted_tools

        if (
            requires_permission
            and not self._ask_tool_permission(tool_class_name, recipient, function_args)
        ):
            # Permission denied
            result = {
                "error": "Tool execution cancelled by user",
                "message": f"User declined to execute {recipient}"
            }

            # Return cancelled result message with proper tool metadata
            result_messages = [Message(
                author=Author(role=Role.TOOL, name=recipient),
                content=[TextContent(text=json.dumps(result))],
            )]
            debug_print_error("Cancelled by user")

        else:

            # Execute the tool
            start_function_time = time.time()

            tool = tool_instances[recipient]

            # Create a Message for the tool
            # Strip "functions." prefix so tool receives "files_search" format it expects
            tool_recipient = recipient[10:] if recipient.startswith("functions.") else recipient

            # For direct content tools, pass raw content; for function-based tools, pass JSON args
            if is_direct_content_tool:
                message_content = content_text
            else:
                message_content = json.dumps(function_args)

            tool_message_input = Message(
                author=Author(role=Role.USER, name="user"),
                content=[TextContent(text=message_content)],
            ).with_recipient(tool_recipient)

            # Execute tool asynchronously
            result_messages = []
            try:
                async def run_tool():
                    async for msg in tool._process(tool_message_input):
                        result_messages.append(msg)

                # Run the async tool
                asyncio.run(run_tool())

            except Exception as e:
                # Catch any tool errors and convert to error message
                error_msg = f"{type(e).__name__}: {str(e)}"
                result_messages = [Message(
                    author=Author(role=Role.TOOL, name=recipient),
                    content=[TextContent(text=json.dumps({"error": error_msg}))],
                )]
                debug_print_error(error_msg)
                return result_messages

            elapsed_time = time.time() - start_function_time

            # Count tokens in tool result
            result_text = result_messages[0].content[0].text if result_messages else ""
            result_tokens = len(self.encoding.encode(result_text, allowed_special='all'))

            # Check if result contains an error
            is_error = False
            error_msg = None

            # Try JSON format first
            try:
                result_json = json.loads(result_text)
                if isinstance(result_json, dict) and "error" in result_json:
                    is_error = True
                    error_msg = result_json.get("error", "Unknown error")
            except (json.JSONDecodeError, KeyError):
                # Check for plain text errors (SimpleBrowserTool format)
                if result_text.startswith("Error ") or result_text.startswith("Invalid "):
                    is_error = True
                    error_msg = result_text

            # Print timing and error if present
            blue_print(f"└── {elapsed_time:.1f}s | ~{result_tokens} tokens")
            if is_error:
                debug_print_error(error_msg)

        # Return the actual tool result messages
        return result_messages

    def user_input(self, prompt, max_iterations=None):
        """Run tool calling loop with retry logic using completions API + harmony"""

        if max_iterations is None:
            max_iterations = self.max_iterations

        # Continue from a copy of conversation history
        self.conversation_history.append(Message.from_role_and_content(Role.USER, prompt))

        # grow conversation over one or more necessary tool calls
        for iteration in range(max_iterations):

            start_llm_time = time.time()

            # Operate at string level
            prompt_string = self.harmony.messages_to_prompt_string(self.conversation_history)

            # Count actual tokens from the prompt string that will be sent
            # Allow all special tokens since harmony uses them for markup
            prompt_token_count = len(self.encoding.encode(prompt_string, allowed_special='all'))

            if prompt_token_count > self.max_context_tokens:
                raise RuntimeError(
                    f"Max number of tokens exceeded {self._format_tokens(prompt_token_count)} "
                    f"(max: {self._format_tokens(self.max_context_tokens)}). "
                )

            # If debug mode, show the full prompt with highlighted markup
            if self.debug:
                debug_print_prompt(prompt_string)

            # Print model call info
            blue_print(f"🤖 {prompt_token_count} tokens input to {self.model}")

            # Retry generation if harmony parsing fails (model may generate incorrect format occasionally)
            max_retries = 20
            last_parse_error = None

            for attempt in range(1, max_retries + 1):
                # Use backend to generate completion
                try:
                    result = self.backend.generate(
                        prompt=prompt_string,
                        model=self.model,
                        options=self.options,
                        timeout=300
                    )
                except requests.exceptions.HTTPError as e:
                    error_msg = f"Backend HTTP error: {e}"
                    if hasattr(e, 'response') and e.response is not None:
                        status_code = e.response.status_code
                        if status_code == 500:
                            error_msg = (
                                f"Backend server error (500). This may be due to:\n"
                                f"  • Large context size (current: {self._format_tokens(prompt_token_count)} tokens)\n"
                                f"  • Model out of memory\n"
                                f"  • Backend server issue\n"
                                f"Try reducing tool output size or restarting the backend."
                            )
                        else:
                            error_msg = f"Backend returned status {status_code}: {e.response.text[:200]}"
                    print(f"\n\033[91m✗ Error: {error_msg}\033[0m\n")
                    self._save_conversation_log()
                    self._save_out_messages()
                    return "I encountered a backend error and cannot continue. Please check the backend status."
                except requests.exceptions.RequestException as e:
                    error_msg = f"Backend connection error: {e}"
                    print(f"\n\033[91m✗ Error: {error_msg}\033[0m\n")
                    self._save_conversation_log()
                    self._save_out_messages()
                    return "I cannot connect to the backend. Please check if the backend is running."
                except Exception as e:
                    error_msg = f"Unexpected error during generation: {e}"
                    print(f"\n\033[91m✗ Error: {error_msg}\033[0m\n")
                    self._save_conversation_log()
                    self._save_out_messages()
                    return "I encountered an unexpected error and cannot continue."

                elapsed_time = time.time() - start_llm_time

                # parse string
                if "context" in result:
                    # ollama using dummy template
                    response_tokens = result["context"][prompt_token_count:]
                else:
                    # ollama using raw
                    response_tokens = self.encoding.encode(result["response"], allowed_special='all')

                # Try to parse the response with harmony
                try:
                    parsed_messages = self.harmony.tokens_to_messages(response_tokens)
                    self.conversation_history.extend(parsed_messages)
                    break
                except HarmonyError as he:
                    last_parse_error = he
                    if attempt < max_retries:
                        print(f"\n\033[93m⚠ Harmony parse error on attempt {attempt}/{max_retries}, retrying generation...\033[0m")
                        print(f"   Error: {str(he)[:100]}")
                        continue
                    print(f"\n\033[91m✗ Failed to generate valid harmony format after {max_retries} attempts\033[0m")
                    print(f"   Last error: {str(he)}")
                    self._save_conversation_log()
                    self._save_out_messages()
                    raise

            # inform user
            blue_print(f"└── {elapsed_time:.1f}s | ~{len(response_tokens)} tokens generated")

            # Check if model wants to call a tool (check for recipient field in harmony messages)
            if hasattr(self.conversation_history[-1], 'recipient') and self.conversation_history[-1].recipient:
                # Execute the tool call and get result messages
                tool_result_messages = self._execute_tool_call(self.conversation_history[-1], self.tool_instances)
                self.conversation_history.extend(tool_result_messages)

            else:
                # Extract text content from last message
                response_text = self.conversation_history[-1].content[0].text
                self._save_conversation_log()
                self._save_out_messages()
                return response_text

        # If we reach here, model didn't finish in max_iterations
        # Update conversation history and ask user if they need help
        self._save_conversation_log()
        self._save_out_messages()
        return self.LOOP_TOOL_FAIL
