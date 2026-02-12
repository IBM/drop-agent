#!/bin/bash

# Load BCP virtual environment
source .bcp_venv/bin/activate

# Set JAVA_HOME if not already set
if [ -z "$JAVA_HOME" ]; then
    if [[ "$OSTYPE" == "darwin"* ]]; then
        if [ -d "/opt/homebrew/opt/openjdk/libexec/openjdk.jdk/Contents/Home" ]; then
            export JAVA_HOME="/opt/homebrew/opt/openjdk/libexec/openjdk.jdk/Contents/Home"
        elif [ -d "/usr/local/opt/openjdk/libexec/openjdk.jdk/Contents/Home" ]; then
            export JAVA_HOME="/usr/local/opt/openjdk/libexec/openjdk.jdk/Contents/Home"
        elif [ -d "/opt/homebrew/opt/openjdk@21/libexec/openjdk.jdk/Contents/Home" ]; then
            export JAVA_HOME="/opt/homebrew/opt/openjdk@21/libexec/openjdk.jdk/Contents/Home"
        elif [ -d "/usr/local/opt/openjdk@21/libexec/openjdk.jdk/Contents/Home" ]; then
            export JAVA_HOME="/usr/local/opt/openjdk@21/libexec/openjdk.jdk/Contents/Home"
        elif command -v /usr/libexec/java_home &> /dev/null; then
            export JAVA_HOME=$(/usr/libexec/java_home 2>/dev/null)
        fi
    elif [ -d "/usr/lib/jvm/java-21-openjdk" ]; then
        export JAVA_HOME="/usr/lib/jvm/java-21-openjdk"
    fi
fi

if [ -z "$JAVA_HOME" ]; then
    echo "ERROR: JAVA_HOME not set and could not auto-detect Java installation"
    echo "Please set JAVA_HOME manually or install Java"
    exit 1
fi

echo "Using JAVA_HOME: $JAVA_HOME"

# Set JVM_PATH for pyjnius
if [ -z "$JVM_PATH" ]; then
    if [ -f "$JAVA_HOME/lib/server/libjvm.dylib" ]; then
        export JVM_PATH="$JAVA_HOME/lib/server/libjvm.dylib"
    elif [ -f "$JAVA_HOME/lib/server/libjvm.so" ]; then
        export JVM_PATH="$JAVA_HOME/lib/server/libjvm.so"
    elif [ -f "$JAVA_HOME/jre/lib/server/libjvm.so" ]; then
        export JVM_PATH="$JAVA_HOME/jre/lib/server/libjvm.so"
    fi
fi

if [ -n "$JVM_PATH" ]; then
    echo "Using JVM_PATH: $JVM_PATH"
fi

# Set up environment variables
export LUCENE_EXTRA_DIR="${LUCENE_EXTRA_DIR:-$HOME/.bcp/lucene}"
export CORPUS_PARQUET_PATH="${CORPUS_PARQUET_PATH:-Tevatron/browsecomp-plus-corpus/data/*.parquet}"
export DENSE_INDEX_PATH="${DENSE_INDEX_PATH:-Tevatron/browsecomp-plus-indexes/main/qwen3-embedding-8b/*.pkl}"
export DENSE_MODEL_NAME="${DENSE_MODEL_NAME:-Qwen/Qwen3-Embedding-8B}"
export SEARCHER_TYPE="${SEARCHER_TYPE:-dense}"

# Launch the BCP server
cd GPT-OSS-BrowseCompPlus-Eval
uvicorn deploy_search_service:app --host 0.0.0.0 --port 8000
