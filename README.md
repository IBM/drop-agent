# Deep Research On Premise (DROP) Agent

<img src="assets/granite-droplet-logo.png" width="100" alt="DROP Agent Logo" align="left" style="margin-right: 20px;"/>

Deep Research agent designed to work fully on-premise. It uses local models so no data leaves your local network. It
can use tools to browse the internet if given permission to do so. 



To use it, see install instruction below, then just go to a folder you want to work on and type

    droplet

this will activate the command line agent which will summarize the content of the current folder and given you some
options to start doing deep research

<br clear="left"/>

## Benchmarking

Below the BrowseCompPlus [[1](https://arxiv.org/abs/2508.06600)] baseline scores for the DROP agent using gpt-oss-120b and two context compaction strategies: summarization and keep-last-N messages. Numbers are averages over three seeds.

| Experiment Group                              | Accuracy (%) |
|-----------------------------------------------|--------------|
| Drop-agent baseline                           | 65.5 (0.1)   |
| recursive summarization (LLMR10)              | 67.6 (0.4)   |
| Keep-last-5 (KL5R20)                          | 68.5 (1.1)   |

See `scripts/bcp` for evaluation details.

## Install

Install via pip (needs at least Python 3.12), for example

    uv venv -p 3.12
    source .venv/bin/activate
    uv pip install git+ssh://git@github.com/IBM/drop-agent.git

### Ollama Back-End

If you want to use Ollama as your backend (good option for local laptop usage), install the Ollama server, in OSX

    brew install ollama

To start Ollama now and restart at login:

    brew services start ollama

### vLLM Back-End

if you have to launch vLLM yourself, you would run on $HOSTNAME

    vllm serve $MODEL --host 0.0.0.0 --port $PORT

If you have access to a vLLM server, run droplet with the vLLM backend:

    droplet -b vllm -m $MODEL -u http://${HOSTNAME}:${PORT}

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

