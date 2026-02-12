"""Test BCP (BrowseComp-Plus) browser tool integration with Droplet agent

This test requires the BCP search service to be running on localhost:8000.
Start the service with: bash scripts/launch_bcp_tool.sh

Usage:
    python tests/test_bcp_browser.py
    python tests/test_bcp_browser.py --bcp_input_file /path/to/bcp_questions.jsonl
"""

import argparse
import json
import random
import sys

from droplet import dbg_tools
from droplet.agent import DropletAgent
from droplet.main import build_agent_config
from droplet.rich_terminal import droplet_print

DEVELOPER_CONTENT = """You are a deep research agent. You need to answer the given question by interacting with a search engine, using the search tool provided. Please perform reasoning and use the tool step by step, in an interleaved manner. You may use the search tool multiple times.
Your response should be in the following format:
Explanation: {{your explanation for your final answer. For this explanation section only, you should cite your evidence documents inline by enclosing their docids in square brackets [] at the end of sentences. For example, [20].}}
Exact Answer: {{your succinct, final answer}}
Confidence: {{your confidence score between 0% and 100% for your answer}}
""".strip()

parser = argparse.ArgumentParser(description="Test BCP browser tool with optional JSONL input")
parser.add_argument(
    "--bcp_input_file",
    type=str,
    default=None,
    help="Optional JSONL file with BCP questions. If provided, picks one question at random."
)

args, unknown_args = parser.parse_known_args()

# Temporarily replace sys.argv with unknown args so build_agent_config can parse them
original_argv = sys.argv
sys.argv = [sys.argv[0]] + unknown_args

# Build config from CLI args and saved configs, then override for test
config, _, _, _, _ = build_agent_config()

# Restore original argv
sys.argv = original_argv

if config is None:
    print("Special flag handled, exiting test")
    exit(0)

# Activate debugger if --debug flag was passed
if config.get('debug'):
    dbg_tools.pm_breakpoint()

# Override specific test parameters
config['debug'] = True
config['restricted_tools'] = set()
config['tool_names'] = ['BCPBrowserTool']
if 'bcp_server_url' not in config or not config['bcp_server_url']:
    config['bcp_server_url'] = 'http://localhost:8000'
if 'developer_prompt' not in config or not config['developer_prompt']:
    config['developer_prompt'] = DEVELOPER_CONTENT

# Load question from input file or use default
if args.bcp_input_file:
    print(f"Loading questions from: {args.bcp_input_file}")
    questions = []
    with open(args.bcp_input_file, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            questions.append(record)

    selected = random.choice(questions)
    question = selected['question']
    qid = selected.get('qid', 'unknown')
    answer = selected.get('answer', 'N/A')

    print(f"\nSelected question (QID {qid}):")
    print(f"  Question: {question}")
    print(f"  Expected Answer: {answer}\n")
else:
    question = "Search for information about 'machine learning algorithms' and give me a summary of the top 3 results"
    print(f"\nUsing default test question: {question}\n")

with DropletAgent(**config) as agent:
    # Auto-allow BCPBrowserTool for testing (no permission prompts)
    agent.allowed_tools.add("BCPBrowserTool")

    response = agent.user_input(question)
    assert response != agent.LOOP_TOOL_FAIL, "Agent failed with LOOP_TOOL_FAIL message"
    droplet_print(f"\n{response}\n")
