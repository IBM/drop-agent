"""Test Python sandbox isolation - verify packages are not accessible"""

from droplet.agent import DropletAgent
from droplet import dbg_tools
from droplet.main import build_agent_config
# Build config from CLI args and saved configs
config, _, _, _, _ = build_agent_config()
if config is None:
    print("Special flag handled, exiting test")
    exit(0)

# Activate debugger if --debug flag was passed
if config.get('debug'):
    dbg_tools.pm_breakpoint()
# Override specific test parameters
config['debug'] = False
config['restricted_tools'] = set()
config['tool_names'] = ['PythonTool']
with DropletAgent(**config) as agent:
    # Test 1: Standard library should work
    response = agent.user_input(
        "Use Python to import json and datetime modules, then print 'stdlib works'",
    )
    assert response != DropletAgent.LOOP_TOOL_FAIL, "Agent failed with LOOP_TOOL_FAIL message"
    assert "stdlib works" in response.lower() or "standard library" in response.lower(), "Stdlib test failed"
    print("✓ Test 1 passed: Standard library accessible")
    # Test 2: Third-party packages should NOT be available
        "Try to import the 'requests' package in Python and tell me what happens",
    # Should mention error or not available
    assert "error" in response.lower() or "not" in response.lower() or "cannot" in response.lower(), "Package isolation test failed"
    print("✓ Test 2 passed: Third-party packages not accessible")
print("\n✓ All sandbox isolation tests passed")
