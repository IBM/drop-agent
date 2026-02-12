"""Rich terminal output utilities"""
import re
import shutil

from rich.box import ROUNDED
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text

from droplet import __version__

LOGO_SUCCESS = [
    "[cyan]   ▄▄[/cyan]",
    "[cyan]  ▐██▌[/cyan]",
    "[cyan] ▐████▌[/cyan]",
    "[cyan]▐██████▌[/cyan]",
    "[cyan]▝██▛▛██▘[/cyan]",
    "[cyan]  ▀▀▀▀[/cyan]"
]

LOGO_FAILURE = [
    "[cyan]   ▄▄[/cyan]",
    "[cyan]  ▐██▌[/cyan]",
    "[cyan] ▐████▌[/cyan]",
    "[cyan]▐██████▌[/cyan]",
    "[cyan]▝██[/cyan][black on cyan]x x[/black on cyan][cyan]█▘[/cyan]",
    "[cyan]  ▀▀▀▀[/cyan]"
]


def blue_print(text, end="\n"):
    """Print text in blue color"""
    print(f" \033[94m{text}\033[0m", end=end, flush=True)


def crop_to_lines(text, prefix_len, max_lines=3, indent=4):
    """
    Crop text to fit within max_lines given terminal width and prefix.

    Args:
        text: The text to crop
        prefix_len: Length of prefix on first line (e.g., "🤖 model: ")
        max_lines: Maximum number of lines to display (default: 3)
        indent: Left indentation in characters (default: 4)

    Returns:
        Cropped text with "..." if truncated
    """
    terminal_width = shutil.get_terminal_size().columns
    # Available width per line (account for indent and safety margin)
    line_width = terminal_width - indent - 2

    # First line has less space due to prefix
    first_line_width = line_width - prefix_len

    # Split text by newlines to handle existing line breaks
    lines = text.split('\n')

    result_chars = 0
    current_line = 0
    current_line_chars = 0
    max_chars = 0

    for line_idx, line in enumerate(lines):
        if current_line >= max_lines:
            break

        # For lines after the first explicit newline, we start a new display line
        if line_idx > 0:
            current_line += 1
            current_line_chars = 0
            if current_line >= max_lines:
                break

        # Calculate available width for current display line
        available = first_line_width if current_line == 0 else line_width

        for char in line:
            # Check if adding this char would wrap to next line
            if current_line_chars >= available:
                current_line += 1
                current_line_chars = 0
                if current_line >= max_lines:
                    break
                available = line_width

            result_chars += 1
            current_line_chars += 1

        if current_line >= max_lines:
            break

        # Account for the newline character itself (except for last line)
        if line_idx < len(lines) - 1:
            result_chars += 1  # for the \n

    max_chars = result_chars

    # Crop text if needed
    if len(text) > max_chars:
        # Find a good cut point (don't cut mid-word if possible)
        cut_point = max_chars - 3  # Reserve space for "..."
        if cut_point > 0:
            # Try to cut at a space
            space_pos = text.rfind(' ', max(0, cut_point - 20), cut_point)
            if space_pos > cut_point - 20:
                cut_point = space_pos
        return text[:cut_point].rstrip() + "..."

    return text


def print_logo(model=None, backend=None, tools=None, logo=LOGO_SUCCESS):
    """Print the DROP logo with configuration information side-by-side"""
    console = Console()

    info_lines = [""]
    info_lines.append(f"v{__version__}")
    if model:
        info_lines.append(f"LLM: {model}")
    if backend:
        info_lines.append(f"back-end: {backend}")
    if tools:
        tools_str = ', '.join(tools)
        info_lines.append(f"tools: {tools_str}")

    while len(info_lines) < len(logo):
        info_lines.append("")

    # Calculate max display width of logo using Rich's Text for proper Unicode width
    max_logo_width = 0
    for logo_line in logo:
        logo_text = Text.from_markup(logo_line)
        max_logo_width = max(max_logo_width, logo_text.cell_len)

    spacing = 4

    # Calculate maximum width for info lines to prevent overflow
    terminal_width = shutil.get_terminal_size().columns
    # Account for: panel borders (4), panel padding (4), logo width, spacing
    max_info_width = terminal_width - max_logo_width - spacing - 8
    max_info_width = max(max_info_width, 40)

    # Build content using Rich Text objects for proper width handling
    combined_text = Text()
    for i, (logo_line, info_line) in enumerate(zip(logo, info_lines)):
        if i > 0:
            combined_text.append("\n")

        logo_text = Text.from_markup(logo_line)
        padding_needed = max_logo_width - logo_text.cell_len

        combined_text.append_text(logo_text)
        combined_text.append(' ' * padding_needed)
        combined_text.append(' ' * spacing)

        # Truncate info_line if needed before parsing
        target_len = max_info_width - 3
        plain_without_ansi = re.sub(r'\033\[[0-9;]+m', '', info_line)

        if len(plain_without_ansi) > target_len:
            # Need to truncate - extract ANSI codes and text separately
            result_chars = []
            ansi_stack = []
            char_count = 0
            i = 0

            while i < len(info_line) and char_count < target_len:
                if info_line[i:i+2] == '\033[':
                    # Start of ANSI code - find the end
                    end = info_line.find('m', i)
                    if end != -1:
                        ansi_code = info_line[i:end+1]
                        ansi_stack.append(ansi_code)
                        result_chars.append(ansi_code)
                        i = end + 1
                    else:
                        i += 1
                else:
                    result_chars.append(info_line[i])
                    char_count += 1
                    i += 1

            truncated_line = ''.join(result_chars) + "..."
            info_text = Text.from_ansi(truncated_line)
        else:
            # Parse ANSI codes in info_line for proper rendering
            info_text = Text.from_ansi(info_line)

        combined_text.append_text(info_text)

    panel = Panel(
        combined_text,
        title="Deep Research On Premise by IBM",
        border_style="cyan",
        box=ROUNDED,
        padding=(0, 1)
    )

    console.print(panel)


def _format_table(lines):
    """
    Format markdown table with proper padding

    Args:
        lines: List of table lines

    Returns:
        List of formatted table lines
    """
    # Parse all rows
    rows = []
    for line in lines:
        # Strip leading/trailing whitespace and pipes
        line = line.strip()
        if not line.startswith('|') or not line.endswith('|'):
            continue
        # Split by | and clean up cells
        cells = [cell.strip() for cell in line.split('|')[1:-1]]
        rows.append(cells)

    if not rows:
        return lines

    # Find maximum number of columns across all rows
    num_cols = max(len(row) for row in rows)
    col_widths = [0] * num_cols

    # Pad all rows to have the same number of columns
    for row in rows:
        while len(row) < num_cols:
            row.append('')

    for row in rows:
        for i, cell in enumerate(row):
            # Strip ANSI codes for width calculation
            clean_cell = re.sub(r'\033\[[0-9;]+m', '', cell)
            col_widths[i] = max(col_widths[i], len(clean_cell))

    # Format rows with padding
    formatted = []
    for row_idx, row in enumerate(rows):
        padded_cells = []
        for i, cell in enumerate(row):
            # Check if this is a separator row
            if re.match(r'^-+$', cell.strip()):
                padded_cells.append('-' * col_widths[i])
            else:
                # Calculate actual display width (without ANSI codes)
                clean_cell = re.sub(r'\033\[[0-9;]+m', '', cell)
                padding = col_widths[i] - len(clean_cell)
                padded_cells.append(cell + ' ' * padding)

        formatted.append('| ' + ' | '.join(padded_cells) + ' |')

    return formatted


def droplet_print(string):
    """
    Print agent output using Rich Markdown rendering with leading space.
    Adds blank line before and indents the rendered output by 1 space.
    """
    import io

    # Print blank line before
    print()

    # Capture Rich output to add indentation
    console = Console(file=io.StringIO(), force_terminal=True, width=shutil.get_terminal_size().columns - 1)
    md = Markdown(string)
    console.print(md)

    # Get the rendered output and add leading space to each line
    output = console.file.getvalue()
    for line in output.rstrip('\n').split('\n'):
        print(f" {line}")


def debug_print_error(error_text):
    """
    Print error text in red using ANSI color codes.

    Args:
        error_text: The error message to display
    """
    # ANSI code: \033[91m for red, \033[0m to reset
    print(f" \033[91m└── Error: {error_text}\033[0m")


def _colorize_system_line(line, indent_level):
    """
    Colorize a single line of system content with syntax highlighting.

    Args:
        line: The line to colorize
        indent_level: Current indentation level for tracking brace depth

    Returns:
        Tuple of (colored_line, new_indent_level)
    """
    import re

    stripped = line.lstrip()
    original_indent = line[:len(line) - len(stripped)]

    # Color lines starting with # (markdown headers)
    if stripped.startswith('#'):
        return f" \033[96m{line}\033[0m", indent_level

    # Color lines starting with // (full-line comments)
    if stripped.startswith('//'):
        # Add indentation based on current brace depth
        extra_indent = '  ' * indent_level
        return f" {original_indent}{extra_indent}\033[90m{stripped}\033[0m", indent_level

    # Check for inline comments (// with content before it)
    comment_match = re.search(r'(.+?)(//.*)$', line)
    if comment_match:
        code_part = comment_match.group(1)
        comment_part = comment_match.group(2)
        # Process the code part and add colored comment
        colored_code, new_indent = _colorize_code_line(code_part, indent_level)
        # colored_code already has proper indentation, just append the comment
        return f"{colored_code}\033[90m{comment_part}\033[0m", new_indent

    # Process as code line
    return _colorize_code_line(line, indent_level)


def _colorize_code_line(line, indent_level):
    """
    Colorize a code line with bracket highlighting and indentation.

    Args:
        line: The line to colorize
        indent_level: Current indentation level

    Returns:
        Tuple of (colored_line, new_indent_level)
    """
    import re

    stripped = line.lstrip()
    original_indent = line[:len(line) - len(stripped)]

    # Count opening and closing braces to track nesting
    open_braces = stripped.count('{')
    close_braces = stripped.count('}')

    # Calculate effective indent level for this line
    effective_indent = indent_level

    # Decrease indent if line starts with }
    if stripped.startswith('}'):
        effective_indent = max(0, indent_level - 1)

    # Apply extra indentation based on effective level
    extra_indent = '  ' * effective_indent

    # Color brackets and arguments
    # Match: { } ( ) [ ] and type annotations
    colored_line = stripped

    # Color brackets in yellow
    colored_line = re.sub(r'([{}()\[\]])', r'\033[93m\1\033[0m', colored_line)

    # Color type annotations (: type) in green
    colored_line = re.sub(r':\s*([a-zA-Z_][a-zA-Z0-9_|?\[\]]*)', r': \033[92m\1\033[0m', colored_line)

    # Color default values (// default: value) - already handled by comment coloring
    # but we can make the "default" keyword stand out
    colored_line = re.sub(r'(//\s*)(default)(:)', r'\1\033[33m\2\033[0m\3', colored_line)

    # Build final line with indentation
    result = f" {original_indent}{extra_indent}{colored_line}"

    # Calculate new indent level for next line
    new_indent_level = indent_level

    # Adjust indent based on brace balance
    if open_braces > close_braces:
        new_indent_level += (open_braces - close_braces)
    elif close_braces > open_braces:
        new_indent_level = max(0, indent_level - (close_braces - open_braces))

    return result, new_indent_level


def _colorize_json(json_str):
    """
    Colorize JSON string with syntax highlighting.
    If JSON contains an "error" key, display everything in red.

    Args:
        json_str: JSON string to colorize

    Returns:
        Colorized JSON string
    """
    import json
    import re

    # Check if this is an error message
    try:
        parsed = json.loads(json_str)
        if isinstance(parsed, dict) and "error" in parsed:
            # Display error JSON in red
            return f'\033[91m{json_str}\033[0m'
    except json.JSONDecodeError:
        pass

    # Normal JSON coloring
    # Color brackets FIRST (before adding any ANSI codes that contain brackets)
    result = re.sub(r'([{}\[\]])', r'\033[1m\1\033[0m', json_str)

    # Color keys (strings before :) in cyan
    result = re.sub(r'"([^"]+)"\s*:', r'\033[96m"\1"\033[0m:', result)

    # Color string values (strings after : or in arrays) in yellow
    result = re.sub(r':\s*"([^"]*)"', r': \033[93m"\1"\033[0m', result)

    # Color numbers in green
    result = re.sub(r'\b(\d+\.?\d*)\b', r'\033[92m\1\033[0m', result)

    # Color booleans and null in magenta
    result = re.sub(r'\b(true|false|null)\b', r'\033[95m\1\033[0m', result)

    return result


def debug_print_prompt(prompt_string):
    """
    Print debug prompt with colored system content and highlighted harmony tags.

    System content coloring:
    - Lines starting with # (markdown headers) in cyan
    - Lines starting with // (full-line comments) in gray
    - Inline comments (// after code) in gray
    - Brackets {}, (), [] in yellow
    - Type annotations in green
    - Auto-indentation for code between braces
    - JSON content between tags with syntax highlighting

    Args:
        prompt_string: The harmony-encoded prompt string to display
    """
    import re

    # Parse the prompt to identify system sections
    # System section: <|start|>system<|message|>...content...<|end|>
    system_pattern = r'<\|start\|>system<\|message\|>(.*?)<\|end\|>'

    # Find all system sections
    system_matches = list(re.finditer(system_pattern, prompt_string, re.DOTALL))

    if system_matches:
        # Process the prompt in parts
        last_end = 0
        for match in system_matches:
            # Print non-system content before this match with highlighted tags and JSON coloring
            before_content = prompt_string[last_end:match.start()]
            if before_content:
                _print_non_system_content(before_content)

            # Print system section header tags
            print(" \033[94m<|start|>system<|message|>\033[0m")

            # Process system content with custom coloring and indentation tracking
            system_content = match.group(1)
            indent_level = 0
            for line in system_content.split('\n'):
                colored_line, indent_level = _colorize_system_line(line, indent_level)
                print(colored_line)

            # Print closing tag
            print(" \033[94m<|end|>\033[0m")

            last_end = match.end()

        # Print remaining content after last system section
        remaining = prompt_string[last_end:]
        if remaining:
            _print_non_system_content(remaining)
    else:
        # No system sections, just highlight tags and JSON
        _print_non_system_content(prompt_string)


def _print_non_system_content(content):
    """
    Print non-system content with tag highlighting and JSON coloring.
    Also detects and colors plain text error messages in red.

    Args:
        content: The content to print
    """
    import re

    # Pattern to match harmony tags
    tag_pattern = r'<\|[^|]+\|>'

    # Pattern to match JSON between <|message|> and <|call|> or <|end|>
    json_pattern = r'(<\|message\|>)(\{[^<]+\})(<\|call\|>|<\|end\|>)'

    # FIRST: Process JSON blocks (before adding newlines)
    def replace_json(match):
        tag_start = match.group(1)
        json_content = match.group(2)
        tag_end = match.group(3)

        # Colorize JSON (tags will be colored in final pass)
        colored_json = _colorize_json(json_content)

        return f'{tag_start}\n {colored_json}{tag_end}'

    content = re.sub(json_pattern, replace_json, content)

    # THEN: Add newlines after certain patterns for readability
    # Add newline after <|message|> when followed by content (not another tag or already has newline)
    content = re.sub(r'(<\|message\|>)([^\s<\033])', r'\1\n \2', content)

    # Add newline after <|end|> tags
    content = content.replace('<|end|>', '<|end|>\n')

    # Highlight remaining tags in blue AND detect plain text errors
    for line in content.split('\n'):
        highlighted_line = re.sub(
            tag_pattern,
            lambda m: f'\033[94m{m.group(0)}\033[0m',
            line
        )

        # Check if this line contains a plain text error message
        # Look for tool responses (lines with content after tags)
        # Match pattern: "  Error ..." or "  Invalid ..." or "  Error: ..."
        stripped = line.strip()
        if stripped and not stripped.startswith('<|'):
            # Check if it's an error message (with space or colon after keyword)
            if (stripped.startswith('Error ') or stripped.startswith('Error:') or
                stripped.startswith('Invalid ') or stripped.startswith('Invalid:')):
                # Color the entire line content (not the tags) in red
                # Replace non-tag content with red version
                highlighted_line = re.sub(
                    r'(\s+)([^<\033]+)',
                    lambda m: f'{m.group(1)}\033[91m{m.group(2)}\033[0m',
                    highlighted_line,
                    count=1
                )

        print(f" {highlighted_line}")


if __name__ == "__main__":
    print_logo()
