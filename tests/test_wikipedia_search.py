"""Test Wikipedia tools integration with Droplet agent

This test uses the Wikipedia browser tool which provides search, open, and find functionality.
"""
from droplet.agent import DropletAgent
from droplet import dbg_tools
from droplet.main import build_agent_config
from droplet.rich_terminal import droplet_print
# Build config from CLI args and saved configs, then override for test
config, _, _, _, _ = build_agent_config()
if config is None:
    print("Special flag handled, exiting test")
    exit(0)

# Activate debugger if --debug flag was passed
if config.get('debug'):
    dbg_tools.pm_breakpoint()
# Override specific test parameters
config['debug'] = True
config['restricted_tools'] = set()
#  config['tool_names'] = ['WikipediaBrowserTool']  # Use only Wikipedia tool (no PythonTool)
with DropletAgent(**config) as agent:
    # Auto-allow WikipediaBrowserTool for testing (no permission prompts)
    agent.allowed_tools.add("WikipediaBrowserTool")
    response = agent.user_input(
        "Search Wikipedia for 'Python programming language' and give me a brief summary of what you find",
    )
    assert response != DropletAgent.LOOP_TOOL_FAIL, "Agent failed with LOOP_TOOL_FAIL message"
    droplet_print(f"\n{response}\n")
