"""Rich command line interface using prompt_toolkit"""

import shutil

from prompt_toolkit import PromptSession
from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.lexers import Lexer
from prompt_toolkit.styles import Style


class CommandLexer(Lexer):
    """
    Custom lexer that highlights system commands (starting with !) in red
    """
    def lex_document(self, document):
        def get_line_tokens(lineno):
            line = document.lines[lineno]
            if line.startswith('!'):
                # Entire line is a command - make it red
                return [('class:command', line)]
            else:
                # Regular text - no special styling
                return [('', line)]

        return get_line_tokens


# Define color scheme with softer red
style = Style.from_dict({
    'command': '#ff6b6b',      # Soft red for system commands
    'prompt.command': '#ff6b6b',  # Soft red for prompt when typing commands
    'prompt.normal': '#5f87ff',  # Blue for normal prompt
    'border': '#00d7ff',       # Cyan for borders (matches logo)
})


# Create session with custom lexer and style
session = PromptSession(
    lexer=CommandLexer(),
    style=style
)


def get_dynamic_prompt():
    """
    Returns a dynamic prompt that changes color based on buffer content
    """
    def _get_prompt():
        # Get current buffer text
        buffer_text = session.default_buffer.text

        if buffer_text.startswith('!'):
            # Red prompt for system commands
            return FormattedText([('class:border', '│ '), ('class:prompt.command', '> ')])
        else:
            # Blue prompt for normal input
            return FormattedText([('class:border', '│ '), ('class:prompt.normal', '> ')])

    return _get_prompt


def get_user_input():
    """
    Get user input with rich command line interface

    Returns:
        tuple: (input_text, is_system_command)
               input_text: The user's input (without ! prefix if command)
               is_system_command: True if input starts with !
    """
    # Get terminal width
    terminal_width = shutil.get_terminal_size().columns

    # Print top border (using same style as logo)
    print(f"\033[96m╭{'─' * (terminal_width - 2)}╮\033[0m")

    # Right border prompt
    def get_right_border():
        return FormattedText([('class:border', ' │')])

    # Get input with dynamic prompt
    try:
        user_input = session.prompt(get_dynamic_prompt(), rprompt=get_right_border).strip()
    except (KeyboardInterrupt, EOFError) as e:
        # Print bottom border even on exception
        print(f"\033[96m╰{'─' * (terminal_width - 2)}╯\033[0m")
        raise e

    # Print bottom border
    print(f"\033[96m╰{'─' * (terminal_width - 2)}╯\033[0m", flush=True)

    # Check if it's a system command
    if user_input.startswith('!'):
        return user_input[1:].strip(), True
    else:
        return user_input, False
