#!/usr/bin/env bash
source "$(dirname "$0")/env.sh"
set -a; . "$PAT_IMACT_ROOT/.env" 2>/dev/null; set +a
for s in google-search fetch yfinance google-maps; do
  code=$(curl -s -o /dev/null -w "%{http_code}" -m 20 -H "Authorization: Bearer $MCP_GATEWAY_TOKEN" "$MCP_GATEWAY_ADDRESS/$s/sse" || echo timeout)
  echo "$s -> $code"
done
