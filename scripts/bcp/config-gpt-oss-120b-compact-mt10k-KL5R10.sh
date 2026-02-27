DEVELOPER_PROMPT="You are a deep research agent. You need to answer the given question by interacting with a search engine, using the search tool provided. Please perform reasoning and use the tool step by step, in an interlea ved manner. You may use the search tool multiple times.  Your response should be in the following format: Explanation: {{your explanation for your final answer. For this explanation section only, you should cite your evidence documents inline by enclosing their docids in square brackets [] at the end of sentences. For example, [20].}} Exact Answer: {{your succinct, final answer}} Confidence: {{your confidence score between 0% and 100% for your answer}}"
LOOP_TOOL_FAIL="You have reached the maximum number of reasoning iterations. Please provide your final answer now."

GEN_MODEL=openai/gpt-oss-120b

RUN_DROPLET_BCP_EVAL_ARGS=(
    --input_file GPT-OSS-BrowseCompPlus-Eval/bcp_questions.jsonl
    --input-prefix "Question: "
    --gpt-reasoning high
    --num_workers 16
    --max_concurrency_per_worker 1
    --backend-type vllm
    --model $GEN_MODEL
    --gpt-reasoning high
    --temperature 1.0
    --tools BCPBrowserTool
    --max-iterations 10000
    --no-droplet-system-prompt
    --developer-prompt "$DEVELOPER_PROMPT"
    --loop-tool-fail "$LOOP_TOOL_FAIL"
    --context-compaction-method keep_last_n
    --context-compaction-threshold 64000
    --max-context-compactions 10
)
