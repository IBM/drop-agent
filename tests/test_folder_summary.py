"""Test script for Harmony agent"""

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
# Test the agent with config
with DropletAgent(**config) as agent:
    response = agent.user_input(DropletAgent.INITIAL_PROMPT)
    assert response != DropletAgent.LOOP_TOOL_FAIL, "Agent failed with LOOP_TOOL_FAIL message"
    droplet_print(f"\n{response}\n")
