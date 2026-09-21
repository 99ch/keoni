#!/usr/bin/env bash
# Génère des secrets forts pour le fichier .env.prod de Keoni en production.
# Usage : ./ops/scripts/generate-secrets.sh
#
# Affiche des lignes KEY=valeur qui peuvent être redirigées vers un fichier :
#   ./ops/scripts/generate-secrets.sh > /tmp/secrets.env
# Puis copier manuellement chaque valeur dans .env.prod.
#
# Dépendance : openssl.

set -euo pipefail

if ! command -v openssl >/dev/null 2>&1; then
    echo "ERREUR : openssl n'est pas installé" >&2
    exit 1
fi

rand_b64() {
    openssl rand -base64 "$1" | tr -d '\n=+/' | cut -c1-"$2"
}

rand_hex() {
    openssl rand -hex "$1"
}

cat <<EOF
# ============================================================
# Secrets générés — $(date -u +'%Y-%m-%dT%H:%M:%SZ')
# Copier chaque valeur dans .env.prod
# NE JAMAIS commiter ces valeurs dans git.
# ============================================================

POSTGRES_PASSWORD=$(rand_b64 48 32)
MATCHING_API_KEY=$(rand_b64 64 64)
WP_API_KEY=$(rand_b64 64 64)
N8N_BASIC_AUTH_PASSWORD=$(rand_b64 32 24)
N8N_ENCRYPTION_KEY=$(rand_hex 32)

# Note : WP_API_KEY doit être copié tel quel dans :
#   wp-admin > Keoni Bridge > Settings > webhook_secret
# (ou bien regénérer une nouvelle clé via l'UI puis la copier dans .env.prod).
EOF
