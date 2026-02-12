"""Python execution tool using current environment

This tool executes Python code in the current droplet environment,
with access to installed packages like pandas and numpy.

Uses stateful Jupyter kernel execution where variables and imports persist between calls.
"""

import asyncio
import contextlib
import logging
from typing import AsyncIterator

from gpt_oss.tools.tool import Tool
from openai_harmony import (Author, Content, Message, Role, TextContent,
                            ToolNamespaceConfig)

# Suppress IPKernelApp warnings
logging.getLogger('ipykernel').setLevel(logging.ERROR)


class LocalJupyterSession:
    """Stateful helper that proxies execution through a local Jupyter kernel."""

    def __init__(self, timeout: float = 120.0) -> None:
        import sys
        from jupyter_client import KernelManager
        from jupyter_client.kernelspec import KernelSpecManager, NoSuchKernel

        self._default_timeout = timeout

        # Ensure python3 kernel spec exists
        ksm = KernelSpecManager()
        kernel_name = 'python3'

        # Check if kernel exists, if not try to find any python kernel
        kernels = ksm.find_kernel_specs()
        if kernel_name not in kernels:
            # Try to find any python kernel
            python_kernels = [k for k in kernels.keys() if 'python' in k.lower()]
            if python_kernels:
                kernel_name = python_kernels[0]
            else:
                # Install python3 kernel spec for current environment
                import subprocess
                subprocess.run([sys.executable, '-m', 'ipykernel', 'install', '--user', '--name', 'python3'],
                             check=True, capture_output=True)

        # Start our own kernel
        km = KernelManager(kernel_name=kernel_name)
        km.start_kernel()
        client = km.blocking_client()
        client.start_channels()
        client.wait_for_ready(timeout=self._default_timeout)

        self._client = client
        self._km = km

    def execute(self, code: str, timeout: float | None = None) -> str:
        """Execute code in the kernel, returning combined stdout/stderr output."""

        # Check for IPython shell command and magic command attempts
        for line in code.split('\n'):
            stripped = line.strip()

            # Block shell commands (!)
            if stripped.startswith('!') and len(stripped) > 1:
                return (
                    "[ERROR] Shell command execution is not allowed in Python tool.\n"
                    f"Attempted command: {stripped}\n\n"
                    "Shell commands (lines starting with '!') cannot be executed through the Python tool. "
                    "If you need to run shell commands, use the appropriate system tool or file operation tool instead."
                )

            # Block IPython magic commands (%)
            if stripped.startswith('%') and len(stripped) > 1:
                return (
                    "[ERROR] IPython magic commands are not allowed in Python tool.\n"
                    f"Attempted command: {stripped}\n\n"
                    "IPython magic commands (lines starting with '%') cannot be executed through the Python tool. "
                    "Use standard Python code instead. If you need system operations, use the appropriate tool."
                )

        client = self._client
        effective_timeout = timeout or self._default_timeout
        msg_id = client.execute(
            code,
            store_history=True,
            allow_stdin=False,
            stop_on_error=False,
        )

        stdout_parts: list[str] = []
        stderr_parts: list[str] = []

        while True:
            msg = client.get_iopub_msg(timeout=effective_timeout)

            if msg.get("parent_header", {}).get("msg_id") != msg_id:
                continue

            msg_type = msg.get("msg_type")
            content = msg.get("content", {})

            if msg_type == "stream":
                text = content.get("text", "")
                if content.get("name") == "stdout":
                    stdout_parts.append(text)
                else:
                    stderr_parts.append(text)
            elif msg_type == "error":
                traceback_data = content.get("traceback")
                if traceback_data:
                    stderr_parts.append("\n".join(traceback_data))
                else:
                    ename = content.get("ename", "")
                    evalue = content.get("evalue", "")
                    stderr_parts.append(f"{ename}: {evalue}".strip())
            elif msg_type in {"execute_result", "display_data"}:
                data = content.get("data", {})
                text = data.get("text/plain")
                if text:
                    stdout_parts.append(text if text.endswith("\n") else f"{text}\n")
            elif msg_type == "status" and content.get("execution_state") == "idle":
                break

        # Drain the shell channel to capture final execution status.
        while True:
            reply = client.get_shell_msg(timeout=effective_timeout)

            if reply.get("parent_header", {}).get("msg_id") != msg_id:
                continue

            reply_content = reply.get("content", {})
            if reply_content.get("status") == "error":
                traceback_data = reply_content.get("traceback")
                if traceback_data:
                    stderr_parts.append("\n".join(traceback_data))
                else:
                    ename = reply_content.get("ename", "")
                    evalue = reply_content.get("evalue", "")
                    stderr_parts.append(f"{ename}: {evalue}".strip())
            break

        stdout = "".join(stdout_parts)
        stderr = "".join(stderr_parts)

        if stderr:
            if stdout:
                stdout = f"{stdout.rstrip()}\n{stderr}"
            else:
                stdout = stderr

        if not stdout.strip():
            stdout = (
                "[WARN] No output available. Use print() to output anything to stdout to "
                "receive the output"
            )

        return stdout

    def close(self) -> None:
        if hasattr(self, '_client'):
            try:
                self._client.stop_channels()
            except Exception:
                pass

        if hasattr(self, '_km'):
            try:
                self._km.shutdown_kernel(now=True)
            except Exception:
                pass

    def __del__(self) -> None:
        if hasattr(self, 'close'):
            self.close()


class PythonTool(Tool):
    """
    Tool for executing Python code in the current environment.

    Features:
    - Runs in droplet's Python environment
    - Access to installed packages (pandas, numpy, etc.)
    - Stateful execution via Jupyter kernel - variables and imports persist between calls
    - 120-second execution timeout

    Security:
    - Shell commands (!) are blocked
    - IPython magic commands (%) are blocked
    - Process isolation via separate kernel
    """

    def __init__(self, name: str = "python", timeout: float = 120.0):
        assert name == "python"

        self._timeout = timeout
        self._execution_lock = asyncio.Lock()
        self._jupyter_session = LocalJupyterSession(timeout=timeout)

    @classmethod
    def get_tool_name(cls) -> str:
        return "python"

    @property
    def name(self) -> str:
        return self.get_tool_name()

    @property
    def instruction(self) -> str:
        return """
Use this tool to execute Python code in your chain of thought. The code will not be shown to the user. This tool should be used for internal reasoning, but not for code that is intended to be visible to the user (e.g. when creating plots, tables, or files).
When you send a message containing Python code to python, it will be executed in a stateful Jupyter notebook environment. python will respond with the output of the execution or time out after 120.0 seconds.
Available packages: Standard library (math, json, datetime, re, statistics, etc.) plus pandas, numpy (data analysis), pypdf, camelot-py (PDF processing and table extraction), requests (HTTP), openpyxl (Excel).
Note: The Python environment IS STATEFUL - variables, imports, and function definitions persist between calls.
IMPORTANT: Shell commands (starting with !) and IPython magic commands (starting with %) are NOT allowed. Use standard Python code only.
        """.strip()

    @property
    def tool_config(self) -> ToolNamespaceConfig:
        return ToolNamespaceConfig(
            name=self.get_tool_name(),
            description=self.instruction,
            tools=[]
        )

    def make_response(
        self,
        content: Content,
        channel: str | None = None,
    ) -> Message:
        """Create a tool response message."""
        tool_name = self.get_tool_name()
        author = Author(role=Role.TOOL, name=tool_name)

        message = Message(
            author=author,
            content=[content],
        ).with_recipient("assistant")

        if channel:
            message = message.with_channel(channel)

        return message

    async def _process(self, message: Message) -> AsyncIterator[Message]:
        """Execute Python code and return the output."""
        script = message.content[0].text
        channel = message.channel

        async with self._execution_lock:
            output = self._jupyter_session.execute(script)

        content = TextContent(text=output)
        yield self.make_response(content=content, channel=channel)

    def close(self) -> None:
        if hasattr(self, '_jupyter_session') and self._jupyter_session is not None:
            try:
                self._jupyter_session.close()
            except Exception:
                pass

    def __del__(self) -> None:
        if hasattr(self, 'close'):
            try:
                self.close()
            except Exception:
                pass
