#!/usr/bin/env bash
# Purge les anciennes images ghcr.io/99ch/keoni-matching-api laissées par
# les déploiements précédents.
#
# Chaque déploiement tire une nouvelle image (~3.4GB) sans jamais retirer
# la précédente : sur un VPS partagé avec d'autres stacks (app1-6,
# airealtime...), une dizaine de déploiements en une journée a suffi à
# faire monter le disque à 93% (2026-09-23). On ne touche qu'aux images
# de ce dépôt -- jamais aux autres images présentes sur le VPS, qui ne
# nous appartiennent pas. Seule la plus récente (celle en cours
# d'exécution) est gardée -- pas de rollback local par image Docker :
# ghcr.io conserve l'historique, un rollback se fait en redéployant un
# commit antérieur via le pipeline.
set -euo pipefail

IMAGE_REPO="ghcr.io/99ch/keoni-matching-api"
KEEP=1

docker image ls "$IMAGE_REPO" --format '{{.ID}} {{.CreatedAt}}' \
    | sort -k2 -r \
    | awk -v keep="$KEEP" 'NR>keep {print $1}' \
    | xargs -r docker rmi -f
