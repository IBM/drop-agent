set -o errexit
set -o pipefail
set -o nounset

mkdir -p results/
config="$1"
RESULTS_TAG=$(basename $config .sh)-$LSB_JOBID
RESULTS_PATH=$(realpath results/$RESULTS_TAG/)

if [ ! -f $RESULTS_PATH/.done ];then

    # load config
    source $config

    # This is a modified version of GPT-OSS-BrowseCompPlus-Eval to run on IBM LSF job scheduler, otherwise equal
    cd GPT-OSS-BrowseCompPlus-Eval/
    source serve_bcp_test.sh $GEN_MODEL $LSB_JOBID
    cd ..
    
    # Hostnames and ports are now available from serve_bcp_test.sh via exported variables:
    # HOSTNAME_EMB, PORT_EMB, HOSTNAME_VLLM, PORT_VLLM, JOBID_EMB, JOBID_VLLM
    
    # Function to cleanup jobs on exit or interrupt
    cleanup_parent_jobs() {
        echo "Parent script cleaning up jobs..."
    
        if [ -n "${JOBID_EMB:-}" ]; then
            echo "Killing embedding job $JOBID_EMB"
            bkill $JOBID_EMB 2>/dev/null || true
        fi
    
        if [ -n "${JOBID_VLLM:-}" ]; then
            echo "Killing vLLM job $JOBID_VLLM"
            bkill $JOBID_VLLM 2>/dev/null || true
        fi
    }
    
    # Set trap to cleanup jobs on exit, interrupt, or termination
    trap cleanup_parent_jobs EXIT INT TERM
    
    source set_environment.sh
    
    python scripts/bcp/run_eval.py \
        "${RUN_DROPLET_BCP_EVAL_ARGS[@]}" \
        --output_dir $RESULTS_PATH/ \
        --backend-url http://$HOSTNAME_VLLM:$PORT_VLLM \
        --bcp-server-url http://$HOSTNAME_EMB:$PORT_EMB

    # mark as done
    touch $RESULTS_PATH/.done
fi    

cd GPT-OSS-BrowseCompPlus-Eval
# initialize conda
source set_environment.sh
python eval.py \
    --input_dir $RESULTS_PATH \
    --model openai/gpt-oss-120b \
    --base-url https://rits.fmaas.res.ibm.com/v1 \
    --qps 5 | tee $RESULTS_PATH/results.txt

