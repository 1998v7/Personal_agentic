#!/usr/bin/env bash
source "$(dirname "$0")/env.sh"
curl -s --noproxy '*' http://127.0.0.1:8100/v1/completions -H 'Content-Type: application/json' \
  -d '{"model":"Qwen3.5-2B","prompt":[248045, 846, 198],"max_tokens":8,"return_token_ids":true,"logprobs":1}' | head -c 800; echo
