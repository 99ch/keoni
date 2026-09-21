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
# Le chargement du modèle d'embeddings + cross-encoder au démarrage prend
# 3-5 min à froid (téléchargement HuggingFace inclus) -- /health ne répond
# pas avant la fin du warmup (event startup FastAPI, bloquant). On retente
# plutôt que d'échouer immédiatement juste après un redéploiement.
API_HEALTH_RETRIES="${API_HEALTH_RETRIES:-40}"
API_HEALTH_DELAY="${API_HEALTH_DELAY:-15}"
api_ok=0
for i in $(seq 1 "$API_HEALTH_RETRIES"); do
  if curl -fsS "$API_URL/health" | cat; then
    api_ok=1
    break
  fi
  echo "matching-api pas encore prêt (tentative $i/$API_HEALTH_RETRIES), nouvelle tentative dans ${API_HEALTH_DELAY}s..."
  sleep "$API_HEALTH_DELAY"
done
if [[ "$api_ok" -ne 1 ]]; then
  echo "matching-api ne répond toujours pas après $((API_HEALTH_RETRIES * API_HEALTH_DELAY))s" >&2
  exit 1
fi

echo "== Santé n8n =="
curl -fsS "$N8N_URL/healthz" | cat

echo "Contrôle stack OK"
