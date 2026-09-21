#!/usr/bin/env bash
# Rollback rapide de la stack IA.
# Usage :
#   ./rollback.sh                          # arrête la stack et laisse WordPress fonctionner seul
#   ./rollback.sh <fichier_backup.sql.gz>  # arrête la stack et restaure la base Postgres
set -euo pipefail

DEPLOY_PATH="${DEPLOY_PATH:-/srv/keoni-ai}"
ENV_FILE="${ENV_FILE:-.env.prod}"
COMPOSE_FILE="${COMPOSE_FILE:-docker-compose.prod.yml}"
BACKUP_FILE="${1:-}"

cd "$DEPLOY_PATH"

if [[ ! -f "$ENV_FILE" ]]; then
  echo "Fichier d'environnement manquant : $DEPLOY_PATH/$ENV_FILE" >&2
  exit 1
fi

echo "Arrêt de la stack IA..."
docker compose --env-file "$ENV_FILE" -f "$COMPOSE_FILE" down

if [[ -n "$BACKUP_FILE" ]]; then
  if [[ ! -f "$BACKUP_FILE" ]]; then
    echo "Fichier de sauvegarde introuvable : $BACKUP_FILE" >&2
    exit 1
  fi

  echo "Restauration de la base Postgres depuis $BACKUP_FILE..."
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a

  docker compose --env-file "$ENV_FILE" -f "$COMPOSE_FILE" up -d db
  sleep 5
  gunzip -c "$BACKUP_FILE" | docker compose --env-file "$ENV_FILE" -f "$COMPOSE_FILE" exec -T db \
    psql -U "$POSTGRES_USER" -d "$POSTGRES_DB"

  echo "Base restaurée. Redémarrage de la stack..."
  docker compose --env-file "$ENV_FILE" -f "$COMPOSE_FILE" up -d
fi

echo "Rollback terminé."
echo "Pour désactiver aussi le plugin WordPress :"
echo "  wp plugin deactivate keoni-bridge --path=/var/www/html"
