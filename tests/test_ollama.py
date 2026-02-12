"""Basic test for Ollama with gpt-oss:20b"""

import subprocess
import time

import ollama

# Pull the model
print("Pulling gpt-oss:20b model...")
subprocess.run(["ollama", "pull", "gpt-oss:20b"], check=True)

# Start server (if not already running)
print("Starting Ollama server...")
server_process = subprocess.Popen(
    ["ollama", "serve"],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE
)

# Wait for server to start
time.sleep(3)

# Query the model
print("Querying model with dummy sentence...")
response = ollama.chat(
    model='gpt-oss:20b',
    messages=[{'role': 'user', 'content': 'Hello, how are you?'}]
)

print(f"\nResponse: {response['message']['content']}")

# Cleanup
server_process.terminate()
server_process.wait()
print("\nServer stopped")
