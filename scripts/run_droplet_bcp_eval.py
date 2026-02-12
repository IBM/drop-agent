import os
import sys
import json
import time
import argparse
import asyncio
import multiprocessing as mp
from tqdm import tqdm
from droplet.agent import DropletAgent
from droplet.main import build_agent_config
from droplet.rich_terminal import blue_print


DEFAULT_NUM_WORKERS = 4
DEFAULT_CONCURRENCY_PER_WORKER = 4
DEFAULT_MAX_RETRIES = 5

DEVELOPER_CONTENT = """You are a deep research agent. You need to answer the given question by interacting with a search engine, using the search tool provided. Please perform reasoning and use the tool step by step, in an interleaved manner. You may use the search tool multiple times.
Your response should be in the following format:
Explanation: {{your explanation for your final answer. For this explanation section only, you should cite your evidence documents inline by enclosing their docids in square brackets [] at the end of sentences. For example, [20].}}
Exact Answer: {{your succinct, final answer}}
Confidence: {{your confidence score between 0% and 100% for your answer}}
""".strip()


def load_jsonl_data(jsonl_path):
    data = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            record['qid'] = int(record['qid'])
            record['question'] = str(record['question'])
            record['answer'] = str(record['answer'])
            data.append(record)
    return data


def worker_entry(worker_idx, num_workers, eval_args, agent_config):
    os.environ["OMP_NUM_THREADS"] = "1"
    node_rank = int(os.getenv("RANK", 0))
    node_size = int(os.getenv("WORLD_SIZE", 1))
    asyncio.run(_run_worker(worker_idx, num_workers, node_rank, node_size, eval_args, agent_config))


async def _run_worker(worker_idx, num_workers, node_rank, node_size, eval_args, agent_config):
    sem = asyncio.Semaphore(eval_args.max_concurrency_per_worker)
    loop = asyncio.get_event_loop()

    shard_path = os.path.join(eval_args.output_dir, f"node_{node_rank}_shard_{worker_idx}.jsonl")
    os.makedirs(eval_args.output_dir, exist_ok=True)

    processed_qids = set()
    if os.path.exists(shard_path):
        print(f"[Worker {worker_idx}] Resuming from {shard_path}")
        with open(shard_path, "r", encoding="utf-8") as f:
            for line in f:
                record = json.loads(line)
                processed_qids.add(record['qid'])
        print(f"[Worker {worker_idx}] Found {len(processed_qids)} completed.")

    data = load_jsonl_data(eval_args.input_file)

    total_workers = node_size * num_workers
    global_worker_idx = num_workers * node_rank + worker_idx
    my_task_chunk = data[global_worker_idx::total_workers]
    tasks_to_process = [x for x in my_task_chunk if x['qid'] not in processed_qids]

    if not tasks_to_process:
        print(f"[Worker {worker_idx}] Nothing to do.")
        return

    print(f"[Worker {worker_idx}] Processing {len(tasks_to_process)} items")

    async def process_item(item_data):
        async with sem:
            qid = item_data['qid']
            question = item_data['question']
            answer = item_data['answer']

            blue_print(f"\n[Worker {worker_idx}] QID {qid}: {question}\n")

            attempt = 0
            error_msg = None
            messages = []
            t0 = time.time()

            while attempt < eval_args.max_retries:
                attempt += 1

                def run_agent():
                    with DropletAgent(**agent_config) as agent:
                        agent.user_input(question, max_iterations=agent_config['max_iterations'])
                        agent_messages = [msg.to_dict() for msg in agent.conversation_history]
                        return agent_messages

                try:
                    messages = await loop.run_in_executor(None, run_agent)
                    dt = time.time() - t0
                    rec = item_data.copy()
                    rec.update({
                        "messages": messages,
                        "latency_s": dt,
                        "error": None,
                        "attempts": attempt,
                        "status": "success"
                    })
                    return rec
                except Exception as e:
                    import traceback
                    error_msg = traceback.format_exc()
                    print(f"[Worker {worker_idx}] qid {qid} attempt {attempt}/{eval_args.max_retries} failed: {e}")

            dt = time.time() - t0
            rec = item_data.copy()
            rec.update({
                "messages": [],
                "latency_s": dt,
                "error": error_msg,
                "attempts": attempt,
                "status": "fail"
            })
            return rec

    tasks = [asyncio.create_task(process_item(task)) for task in tasks_to_process]

    with open(shard_path, "a", encoding="utf-8") as writer:
        for fut in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc=f"Worker {worker_idx}"):
            rec = await fut
            writer.write(json.dumps(rec, ensure_ascii=False) + "\n")
            writer.flush()

    print(f"[Worker {worker_idx}] Done.")


def main():
    # Check for common mistake: --config instead of --load-config
    if '--config' in sys.argv:
        print("\n\033[91mError: Unknown flag '--config'\033[0m")
        print("Did you mean '--load-config' or '-c'?\n")
        print("Example:")
        print("  python scripts/run_droplet_bcp_eval.py --load-config bcp --input_file data.jsonl --output_dir results\n")
        sys.exit(1)

    parser = argparse.ArgumentParser(
        description="Run DropletAgent on BCP+ data in parallel",
        epilog="""
This script uses droplet's build_agent_config() to load agent configuration.
You can use all standard droplet flags (--model, --backend-type, --tools, etc.)
plus the evaluation-specific flags below.

Examples:
  # Basic usage with saved config
  python scripts/run_droplet_bcp_eval.py \\
      --load-config bcp \\
      --input_file bcp_questions.jsonl \\
      --output_dir results/bcp_eval

  # Override model and backend
  python scripts/run_droplet_bcp_eval.py \\
      --input_file bcp_questions.jsonl \\
      --output_dir results/bcp_eval \\
      --model gpt-oss:20b \\
      --backend-type vllm \\
      --backend-url http://localhost:8000

  # Multi-node distributed evaluation
  RANK=0 WORLD_SIZE=2 python scripts/run_droplet_bcp_eval.py \\
      --input_file bcp_questions.jsonl \\
      --output_dir results/bcp_eval \\
      --num_workers 4
"""
    )

    # Evaluation-specific arguments
    parser.add_argument("--input_file", required=True, help="Input JSONL file with BCP questions")
    parser.add_argument("--output_dir", required=True, help="Output directory for JSONL results")
    parser.add_argument("--num_workers", type=int, default=DEFAULT_NUM_WORKERS, help="Number of worker processes")
    parser.add_argument("--max_concurrency_per_worker", type=int, default=DEFAULT_CONCURRENCY_PER_WORKER, help="Max concurrent tasks per worker")
    parser.add_argument("--max_retries", type=int, default=DEFAULT_MAX_RETRIES, help="Max retries per question")

    # Parse only known args to allow build_agent_config to handle the rest
    args, unknown_args = parser.parse_known_args()

    # Validate input file is JSONL
    if not args.input_file.endswith('.jsonl'):
        parser.error(f"Input file must be a .jsonl file, got: {args.input_file}")

    # Temporarily replace sys.argv with unknown args so build_agent_config can parse them
    original_argv = sys.argv
    sys.argv = [sys.argv[0]] + unknown_args

    # Build agent config using droplet's standard config builder
    agent_config, backend_name, _, _, _ = build_agent_config()

    # Restore original argv
    sys.argv = original_argv

    if agent_config is None:
        print("Special flag handled by build_agent_config, exiting")
        return

    # Add developer prompt for BCP evaluation
    agent_config['developer_prompt'] = DEVELOPER_CONTENT

    # Validate BCPBrowserTool is present
    tool_names = agent_config.get('tool_names', [])
    if 'BCPBrowserTool' not in tool_names:
        print("\n\033[91mError: BCPBrowserTool must be included in tools for BCP evaluation\033[0m")
        print(f"Current tools: {tool_names}")
        print("\nAdd --tools BCPBrowserTool or include it with other tools:")
        print("  --tools BCPBrowserTool FileBrowserTool\n")
        sys.exit(1)

    # Warn if other tools are present (may impact performance)
    if len(tool_names) > 1:
        other_tools = [t for t in tool_names if t != 'BCPBrowserTool']
        print(f"\n\033[93m⚠️  Warning: Additional tools detected: {other_tools}\033[0m")
        print("For optimal BCP evaluation performance, use only BCPBrowserTool:")
        print("  --tools BCPBrowserTool")
        print("Additional tools may slow down evaluation and affect results.\n")

    # Ensure BCP server URL is set
    if not agent_config.get('bcp_server_url'):
        agent_config['bcp_server_url'] = 'http://localhost:8000'
        print(f"Using default BCP server URL: {agent_config['bcp_server_url']}")

    print(f"\nEvaluation Configuration:")
    print(f"  Input file: {args.input_file}")
    print(f"  Output directory: {args.output_dir}")
    print(f"  Model: {agent_config['model']}")
    print(f"  Backend: {agent_config['backend_type']} ({agent_config['base_url']})")
    print(f"  Tools: {agent_config['tool_names']}")
    print(f"  BCP Server: {agent_config['bcp_server_url']}")
    print(f"  Workers: {args.num_workers}")
    print(f"  Concurrency per worker: {args.max_concurrency_per_worker}")
    print(f"  Max iterations: {agent_config['max_iterations']}")
    print(f"  Max retries: {args.max_retries}")
    print()

    os.makedirs(args.output_dir, exist_ok=True)

    procs = []
    for i in range(args.num_workers):
        p = mp.Process(
            target=worker_entry,
            args=(i, args.num_workers, args, agent_config)
        )
        p.start()
        procs.append(p)

    try:
        for p in procs:
            p.join(timeout=None)
            if p.exitcode != 0:
                print(f"Worker process {p.pid} exited with code {p.exitcode}")

        print("All workers finished. Script done.")
    except KeyboardInterrupt:
        print("\n\nReceived Ctrl-C, terminating all workers...")
        for p in procs:
            if p.is_alive():
                print(f"Terminating worker process {p.pid}")
                p.terminate()

        print("Waiting for workers to terminate...")
        for p in procs:
            p.join(timeout=5)
            if p.is_alive():
                print(f"Force killing worker process {p.pid}")
                p.kill()
                p.join()

        print("All workers terminated. Exiting.")
        sys.exit(1)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
