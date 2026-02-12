# Deep Research On Premise (DROP) Agent

<img src="assets/droplet-logo.png" width="100" alt="DROP Agent Logo" align="left" style="margin-right: 20px;"/>

Deep Research agent designed to work fully on-premise. It uses local models so no data leaves your local network. It
can use tools to browse the internet if given permission to do so. 



To use it, see install instruction below, then just go
to a folder you want to work on and type

    droplet

this will activate the command line agent which will summarize the content of the current folder and given you some
options to start doing deep research

<br clear="left"/>

## Install

Install via pip (needs at least Python 3.12):

    pip install git+ssh://git@github.ibm.com/generative-computing/drop-agent.git

you can do a basic check with the script below. You need to install a back-end, see below

    python tests/test_install.py

### RITS Back-End

this is the simplest and it will work inside the IBM intranet (remember to use TUNNELALL if you are using VPN).

    droplet -b rits-vllm -m openai/gpt-oss-20b --rits-api-key <YOUR KEY> --save-config

the last argument will also store this config in `~/.droplet/config.json` so that future calls to droplet can omit 
these arguments (unless you want to override them)

### Ollama Back-End

If you want to use Ollama as your backend (good option for local laptop usage), install the Ollama server:

    brew install ollama

To start Ollama now and restart at login:

    brew services start ollama

this is the default so its just

    droplet

### vLLM Back-End

If you have access to a vLLM server, run droplet with the vLLM backend:

    droplet -b vllm -m $MODEL -u http://${HOSTNAME}:${PORT}

you can do a basic check with

    python tests/test_install.py -b vllm -u http://${HOSTNAME}:${PORT} -m $MODEL

if you have to launch vLLM yourself, you would run on $HOSTNAME

    vllm serve $MODEL --host 0.0.0.0 --port $PORT

if you serve the command above from a network only accessible through a login node to your laptop, on your laptop leave open

    ssh -L ${PORT}:0.0.0.0:${PORT} ${USER}@${HOSTNAME}

and then on another window of your laptop

    droplet -b vllm -m $MODEL -u http://localhost:${PORT}

you can also use e.g. `--save-config remote-vllm` to store this config (with that port) for later use, with

    droplet -c remote-vllm

any further arguments will override the defaults above

## Developer Install

Clone and install in editable mode (here `uv` is used, but pip works too):

    git clone git@github.ibm.com:generative-computing/drop-agent.git
    cd drop-agent
    uv venv -p 3.12
    source .venv/bin/activate
    uv pip install --editable .

## Set up BCP Tool

BrowseComp-Plus search service provides web search capabilities to droplet. Install with:

    bash scripts/install_bcp.sh

Then download BrowseComp-Plus corpus and indexes to `Tevatron/browsecomp-plus-corpus/` and `Tevatron/browsecomp-plus-indexes/`. Launch with:

    bash scripts/launch_bcp_tool.sh

## Unit Tests

to sanity check the code, you can run

    bash tests/all.sh

test will load the default configuration, but droplet flags can be passed here

if you have Milvus available

    python tests/test_milvus_retriever.py \
            --milvus-db /path/to/db \
            --milvus-model <model>  \
            --milvus-collection <collection>

