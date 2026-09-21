#!/usr/bin/env bash
# Test smoke bout-en-bout : déclenche le webhook n8n puis vérifie que des
# résultats de matching sont bien écrits côté WordPress.
#
# Variables requises :
#   WEBHOOK_URL, WEBHOOK_SECRET, JOB_ID, WP_BASE_URL
# Variables optionnelles :
#   API_URL, N8N_URL (santé de la stack)
set -euo pipefail

if [[ -z "${WEBHOOK_URL:-}" || -z "${WEBHOOK_SECRET:-}" || -z "${JOB_ID:-}" || -z "${WP_BASE_URL:-}" ]]; then
  echo "Variables requises : WEBHOOK_URL, WEBHOOK_SECRET, JOB_ID, WP_BASE_URL" >&2
  exit 1
fi

if [[ -n "${API_URL:-}" ]]; then
  echo "Vérification santé matching-api..."
  curl -fsS "$API_URL/health" > /dev/null
fi

if [[ -n "${N8N_URL:-}" ]]; then
  echo "Vérification santé n8n..."
  curl -fsS "$N8N_URL/healthz" > /dev/null
fi

started_ms="$(python3 - <<'PY'
import time
print(int(time.time() * 1000))
PY
)"

echo "Déclenchement du webhook n8n pour le job $JOB_ID"
curl -fsS -X POST "$WEBHOOK_URL" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $WEBHOOK_SECRET" \
  -d "{\"job_post_id\":$JOB_ID,\"workflow_started_at_ms\":$started_ms}" > /dev/null

result_ok=0
for i in $(seq 1 20); do
  count="$(curl -fsS "$WP_BASE_URL/wp-json/keoni/v1/matching/$JOB_ID?limit=1" | python3 - <<'PY'
import json
import sys
payload = json.load(sys.stdin)
print(int(payload.get('count', 0)))
PY
)"

  if [[ "$count" -gt 0 ]]; then
    result_ok=1
    break
  fi

  sleep 15
done

if [[ "$result_ok" -ne 1 ]]; then
  echo "Échec E2E : aucun résultat de matching trouvé pour le job $JOB_ID" >&2
  exit 1
fi

echo "Test smoke E2E OK pour le job $JOB_ID"
