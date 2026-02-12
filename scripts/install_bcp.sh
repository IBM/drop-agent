#!/bin/bash

set -e

echo "Setting up BCP Tool environment..."

# Create virtual environment with uv using Python 3.9 (required for qwen_omni_utils dependencies which need Python <3.10)
uv venv .bcp_venv -p 3.9
source .bcp_venv/bin/activate

# Install Python dependencies
uv pip install fastapi uvicorn pyjnius pyserini duckdb loguru faiss-cpu torch transformers python-box pydantic tqdm numpy datasets qwen_omni_utils torchvision peft

# Clone and install tevatron
if [ ! -d "tevatron/.git" ]; then
    rm -rf tevatron
    git clone https://github.com/texttron/tevatron.git
fi
.bcp_venv/bin/python -m ensurepip
.bcp_venv/bin/python -m pip install -e ./tevatron

# Download Lucene JARs
mkdir -p ~/.bcp/lucene
cd ~/.bcp/lucene
wget -nc https://repo1.maven.org/maven2/org/apache/lucene/lucene-highlighter/9.9.1/lucene-highlighter-9.9.1.jar
wget -nc https://repo1.maven.org/maven2/org/apache/lucene/lucene-queries/9.9.1/lucene-queries-9.9.1.jar
wget -nc https://repo1.maven.org/maven2/org/apache/lucene/lucene-memory/9.9.1/lucene-memory-9.9.1.jar
cd -

# Check for Java
if ! command -v java &> /dev/null; then
    echo ""
    echo "WARNING: Java not found. BCP Tool requires OpenJDK 21."
    echo "Install with: brew install openjdk@21 (macOS) or via SDKMan (https://sdkman.io/)"
    echo ""
fi

echo ""
echo "BCP Tool setup complete!"
echo "Next steps:"
echo "1. Download BrowseComp-Plus corpus and indexes to Tevatron/browsecomp-plus-corpus/ and Tevatron/browsecomp-plus-indexes/"
echo "2. Launch with: bash scripts/launch_bcp_tool.sh"
