"""Test Python execution tool integration with Droplet agent

WARNING: This test executes Python code without sandboxing.
"""
from droplet.agent import DropletAgent
from droplet import dbg_tools
from droplet.main import build_agent_config
from droplet.rich_terminal import droplet_print
# Build config from CLI args and saved configs
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
config['tool_names'] = ['PythonTool']  # Use only the Python tool
with DropletAgent(**config) as agent:
    response = agent.user_input(
        "Calculate the sum of squares of numbers from 1 to 10 using Python. Show me the result.",
    )
    assert response != DropletAgent.LOOP_TOOL_FAIL, "Agent failed with LOOP_TOOL_FAIL message"
    droplet_print(f"\n{response}\n")
print("✓ Test passed: Python code executed successfully")
