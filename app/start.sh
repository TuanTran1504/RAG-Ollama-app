#!/bin/bash
source /opt/conda/etc/profile.d/conda.sh
conda activate rag-env
echo "Waiting for Ollama to be ready..."
until curl -s http://ollama:11434 > /dev/null; do
  sleep 1
done
echo "Ollama is up. Starting app..."
python app.py