#!/usr/bin/env bash
set -euo pipefail

set -a
if [ -f .env ]; then
  source .env
fi
set +a

echo "Checking gateway..."
curl -sf http://127.0.0.1:8080/health >/dev/null
echo "Gateway OK"

echo "Listing models..."
AUTH_HEADER=()
if [ -n "${API_KEY:-}" ]; then
  AUTH_HEADER=(-H "Authorization: Bearer ${API_KEY}")
fi

curl -s http://127.0.0.1:8080/v1/models "${AUTH_HEADER[@]}" | python -m json.tool

echo "Running chat request..."

MODEL="${DEFAULT_MODEL:-deepseek-r1:8b}"

curl -s http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  "${AUTH_HEADER[@]}" \
  -d "{
    \"model\": \"${MODEL}\",
    \"messages\": [{\"role\":\"user\",\"content\":\"Me diga um haicai sobre GPUs.\"}],
    \"max_tokens\": 64
  }" | python -m json.tool
