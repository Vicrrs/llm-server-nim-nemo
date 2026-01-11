#!/usr/bin/env sh
set -e

export OLLAMA_HOST="${OLLAMA_HOST:-http://ollama:11434}"
MODELS="${OLLAMA_MODELS:-deepseek-r1:8b}"

echo "[ollama-init] waiting for ollama at $OLLAMA_HOST ..."
i=0
until ollama list >/dev/null 2>&1; do
  i=$((i+1))
  if [ "$i" -gt 60 ]; then
    echo "[ollama-init] timeout waiting for ollama"
    exit 1
  fi
  sleep 1
done

echo "[ollama-init] ollama is up"
echo "[ollama-init] pulling: $MODELS"

for m in $MODELS; do
  echo "[ollama-init] ==> ollama pull $m"
  ollama pull "$m"
done

echo "[ollama-init] done"
