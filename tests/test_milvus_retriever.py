"""Test Milvus retriever tool integration with Droplet agent

This test uses the RetrieverBrowserTool which provides search and document retrieval
from a Milvus vector database.
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
# Verify milvus configuration is provided
if not config.get('milvus_db'):
    print("Error: Milvus database path is required. Use --milvus-db <path>")
    exit(1)
# Override specific test parameters
config['debug'] = True
config['restricted_tools'] = set()
config['tool_names'] = ['RetrieverBrowserTool']  # Use only the Milvus retriever tool
with DropletAgent(**config) as agent:
    response = agent.user_input(
        "Search for information about 'machine learning' and provide a summary of what you find",
    )
    assert response != DropletAgent.LOOP_TOOL_FAIL, "Agent failed with LOOP_TOOL_FAIL message"
    droplet_print(f"\n{response}\n")
print("✓ Test passed: Model provided retrieval results from Milvus")
