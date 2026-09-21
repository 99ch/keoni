#!/usr/bin/env bash
# Contrôle rapide de la santé de la stack IA (conteneurs + endpoints).
set -euo pipefail

DEPLOY_PATH="${DEPLOY_PATH:-/srv/keoni-ai}"
ENV_FILE="${ENV_FILE:-.env.prod}"
COMPOSE_FILE="${COMPOSE_FILE:-docker-compose.prod.yml}"
API_URL="${API_URL:-http://127.0.0.1:8000}"
N8N_URL="${N8N_URL:-http://127.0.0.1:5678}"

cd "$DEPLOY_PATH"

echo "== État des conteneurs =="
docker compose --env-file "$ENV_FILE" -f "$COMPOSE_FILE" ps

echo "== Santé matching-api =="
curl -fsS "$API_URL/health" | cat

echo "== Santé n8n =="
curl -fsS "$N8N_URL/healthz" | cat

echo "Contrôle stack OK"
