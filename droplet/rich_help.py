"""Rich-formatted help display for droplet CLI"""

import argparse

from rich.console import Console
from rich.panel import Panel
from rich.table import Table


class RichHelpFormatter(argparse.HelpFormatter):
    """Custom help formatter using rich for colorful output"""

    def __init__(self, prog):
        super().__init__(prog, max_help_position=40, width=100)
        self.console = Console()

    def format_help(self):
        """Format help with rich styling"""
        # Don't use rich formatting, just return standard format
        # We'll handle rich formatting in print_help instead
        return super().format_help()


def print_rich_help(parser, available_tools):
    """Print help message with rich formatting"""
    console = Console()

    # Header
    console.print()
    console.print(Panel.fit(
        "[bold cyan]Droplet[/bold cyan]\n[dim]Deep Research On Premise Agent[/dim]",
        border_style="cyan"
    ))
    console.print()

    # Backend Configuration
    backend_table = Table(title="[bold cyan]Backend Configuration[/bold cyan]", show_header=True, header_style="bold magenta")
    backend_table.add_column("Argument", style="green", width=25)
    backend_table.add_column("Description", width=60)

    backend_table.add_row(
        "-b, --backend-type",
        "[dim]Backend type:[/dim] ollama, vllm, rits-vllm [dim](default: ollama)[/dim]"
    )
    backend_table.add_row(
        "-u, --backend-url",
        "[dim]Backend URL[/dim] [dim](default: http://localhost:11434)[/dim]"
    )
    backend_table.add_row(
        "-m, --model",
        "[dim]Model name[/dim] [dim](default: gpt-oss:20b)[/dim]"
    )
    backend_table.add_row(
        "--rits-api-key",
        "[dim]RITS API key (required for rits-vllm backend)[/dim]"
    )
    backend_table.add_row(
        "--rits-list-models",
        "[dim]List all available RITS models and exit[/dim]"
    )

    console.print(backend_table)
    console.print()

    # Tools Configuration
    tools_table = Table(title="[bold cyan]Tools Configuration[/bold cyan]", show_header=True, header_style="bold magenta")
    tools_table.add_column("Argument", style="green", width=25)
    tools_table.add_column("Description", width=60)

    tools_table.add_row(
        "-t, --tools",
        f"[dim]Tools to use[/dim]\n[dim]Default: FileBrowserTool, SemanticScholarTool, PythonTool[/dim]\n[dim]Available: {', '.join(sorted(available_tools.keys()))}[/dim]"
    )
    tools_table.add_row(
        "--require-approval",
        "[dim]Tools that require user approval[/dim]\n[dim](default: WikipediaBrowserTool, PythonTool)[/dim]"
    )
    tools_table.add_row(
        "--semantic-scholar-api-key",
        "[dim]Semantic Scholar API key (optional, higher rate limits)[/dim]"
    )
    tools_table.add_row(
        "--milvus-db",
        "[dim]Path to Milvus database (for RetrieverBrowserTool)[/dim]"
    )
    tools_table.add_row(
        "--milvus-model",
        "[dim]SentenceTransformer model name or path[/dim]"
    )
    tools_table.add_row(
        "--milvus-collection",
        "[dim]Milvus collection name[/dim]"
    )
    tools_table.add_row(
        "--bcp-server-url",
        "[dim]BCP search server URL (for BCPBrowserTool)[/dim]"
    )

    console.print(tools_table)
    console.print()

    # Generation Parameters
    gen_table = Table(title="[bold cyan]Generation Parameters[/bold cyan]", show_header=True, header_style="bold magenta")
    gen_table.add_column("Argument", style="green", width=25)
    gen_table.add_column("Description", width=60)

    gen_table.add_row(
        "--temperature",
        "[dim]Sampling temperature[/dim] [dim](default: 0.0)[/dim]"
    )
    gen_table.add_row(
        "--max-tokens",
        "[dim]Maximum tokens to generate[/dim] [dim](default: 32768)[/dim]"
    )
    gen_table.add_row(
        "--max-iterations",
        "[dim]Maximum tool call iterations[/dim] [dim](default: 25)[/dim]"
    )
    gen_table.add_row(
        "--gpt-reasoning",
        "[dim]GPT-OSS reasoning effort: low, medium, high[/dim]"
    )

    console.print(gen_table)
    console.print()

    # Prompt Configuration
    prompt_table = Table(title="[bold cyan]Prompt Configuration[/bold cyan]", show_header=True, header_style="bold magenta")
    prompt_table.add_column("Argument", style="green", width=25)
    prompt_table.add_column("Description", width=60)

    prompt_table.add_row(
        "--no-droplet-system-prompt",
        "[dim]Disable default Droplet system prompt[/dim]"
    )
    prompt_table.add_row(
        "--system-prompt",
        "[dim]Custom system prompt[/dim]"
    )
    prompt_table.add_row(
        "--developer-prompt",
        "[dim]Additional developer instructions[/dim]"
    )
    prompt_table.add_row(
        "--initial-prompt",
        "[dim]Override default initial prompt[/dim]"
    )
    prompt_table.add_row(
        "--loop-tool-fail",
        "[dim]Override default loop failure message[/dim]"
    )
    prompt_table.add_row(
        "--input-prefix",
        "[dim]Prefix for user input (e.g., 'Question: ')[/dim]"
    )

    console.print(prompt_table)
    console.print()

    # Other Options
    other_table = Table(title="[bold cyan]Other Options[/bold cyan]", show_header=True, header_style="bold magenta")
    other_table.add_column("Argument", style="green", width=25)
    other_table.add_column("Description", width=60)

    other_table.add_row(
        "-d, --debug",
        "[dim]Enable debug mode[/dim]"
    )
    other_table.add_row(
        "-i, --input",
        "[dim]Initial prompt (replaces default directory summary)[/dim]"
    )
    other_table.add_row(
        "--no-initial-summary",
        "[dim]Skip default initial directory summary[/dim]"
    )
    other_table.add_row(
        "--cwd",
        "[dim]Change working directory before starting[/dim]"
    )
    other_table.add_row(
        "--log",
        "[dim]Path to JSON file for conversation log[/dim]"
    )
    other_table.add_row(
        "--out-messages",
        "[dim]Path to JSON file for final messages[/dim]"
    )
    other_table.add_row(
        "-c, --load-config",
        "[dim]Load a saved configuration[/dim]"
    )
    other_table.add_row(
        "-l, --list-configs",
        "[dim]List all saved configurations and exit[/dim]"
    )
    other_table.add_row(
        "-s, --save-config",
        "[dim]Save current configuration[/dim]"
    )

    console.print(other_table)
    console.print()

    # Examples
    console.print(Panel(
        """[bold yellow]Examples:[/bold yellow]

[cyan]# Basic usage with Ollama (default)[/cyan]
droplet

[cyan]# Use Granite model with vLLM[/cyan]
droplet --backend-type vllm --model ibm-granite/granite-4.0-h-small

[cyan]# Use RITS backend[/cyan]
droplet --backend-type rits-vllm --model ibm-granite/granite-3.0-8b-instruct --rits-api-key YOUR_KEY

[cyan]# Custom tools[/cyan]
droplet --tools FileBrowserTool WikipediaBrowserTool SemanticScholarTool

[cyan]# With Python execution[/cyan]
droplet --tools PythonTool

[cyan]# Custom working directory[/cyan]
droplet --cwd ~/my-project

[cyan]# Save configuration[/cyan]
droplet --backend-type vllm --model my-model --save-config my-setup

[cyan]# Load saved configuration[/cyan]
droplet --load-config my-setup

[cyan]# Debug mode[/cyan]
droplet --debug""",
        title="[bold cyan]Quick Start[/bold cyan]",
        border_style="yellow"
    ))
    console.print()

    # Supported Models
    console.print(Panel(
        """[bold]GPT-OSS Models:[/bold]
  • gpt-oss-20b, gpt-oss:20b, gpt-oss-70b
  • Context: 128,000 tokens
  • Uses Harmony encoding

[bold]Granite Models:[/bold]
  • ibm-granite/granite-4.0-h-small (8K context)
  • ibm-granite/granite-3.0-8b-instruct (4K context)
  • ibm-granite/granite-3.0-2b-instruct (4K context)
  • Uses transformers chat templates

[dim]The agent automatically selects the appropriate converter based on model name.[/dim]""",
        title="[bold cyan]Supported Models[/bold cyan]",
        border_style="green"
    ))
    console.print()


def create_argument_parser_with_rich_help(available_tools):
    """Create argument parser that uses rich for help display"""

    class RichHelpAction(argparse.Action):
        """Custom action to trigger rich help display"""
        def __init__(self, option_strings, dest=argparse.SUPPRESS, default=argparse.SUPPRESS, help=None):
            super().__init__(
                option_strings=option_strings,
                dest=dest,
                default=default,
                nargs=0,
                help=help
            )

        def __call__(self, parser, namespace, values, option_string=None):
            print_rich_help(parser, available_tools)
            parser.exit()

    # Create parser with custom help action
    parser = argparse.ArgumentParser(
        description='Droplet: Deep Research On Premise agent',
        add_help=False,  # Disable default help
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Add custom help argument
    parser.add_argument(
        '-h', '--help',
        action=RichHelpAction,
        help='Show this help message and exit'
    )

    return parser
