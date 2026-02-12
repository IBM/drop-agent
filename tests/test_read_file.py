"""Unit test for gpt-oss tool calling: read and analyze file using unified browse tool"""

from droplet.agent import DropletAgent
from droplet import dbg_tools
from droplet.main import build_agent_config
from droplet.rich_terminal import droplet_print
def test_read_file():
    # Build config from CLI args and saved configs, then override for test
    config, _, _, _, _ = build_agent_config()
    if config is None:
        print("Special flag handled, exiting test")
        return

# Activate debugger if --debug flag was passed
if config.get('debug'):
    dbg_tools.pm_breakpoint()
    # Override specific test parameters
    config['debug'] = True
    with DropletAgent(**config) as agent:
        # Run tool calling via user_input
        final_answer = agent.user_input(
            prompt="Please read the README.md file using browse.open('README.md') and provide a brief summary of what it contains."
        )
        # Verify we got a meaningful response
        assert final_answer is not None
        assert final_answer != DropletAgent.LOOP_TOOL_FAIL, "Agent failed with LOOP_TOOL_FAIL message"
        droplet_print(f"\n{final_answer}\n")
        assert len(final_answer) > 0
        print("✓ Test passed: Model provided a file summary")
if __name__ == "__main__":
    test_read_file()
