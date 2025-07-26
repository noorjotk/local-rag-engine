#!/bin/bash

set -e  # Exit immediately if any command fails

# Start Ollama server in the background
/usr/bin/ollama serve &

# Wait until the server is ready
echo "⏳ Waiting for Ollama server to be ready..."
until curl -s http://localhost:11434/api/tags > /dev/null; do
  sleep 2
done
echo "✅ Ollama server is up."

# Pull the model if not already present
if ! ollama list | awk '{print $1}' | grep -Fqx "phi3:3.8b-mini-128k-instruct-q4_0"; then
  echo "🔄 Pulling base model phi3:3.8b-mini-128k-instruct-q4_0..."
  ollama pull phi3:3.8b-mini-128k-instruct-q4_0
else
  echo "✅ Model already present — skipping pull."
fi


# Preload model into memory (warm-up)
echo "🔥 Warming up model phi3:3.8b-mini-128k-instruct-q4_0..."
curl -s -X POST http://localhost:11434/api/chat \
  -H "Content-Type: application/json" \
  -d '{"model": "phi3:3.8b-mini-128k-instruct-q4_0", "messages": [{"role": "user", "content": "hello"}]}' \
  > /dev/null || echo "⚠️ Warm-up failed, but continuing..."

echo "✅ Model loaded into memory."

# Keep the container running
tail -f /dev/null
