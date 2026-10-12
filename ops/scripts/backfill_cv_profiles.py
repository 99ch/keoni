"""Recalcule, pour chaque CV déjà indexé (cv_embeddings), ses compétences et son
profil (expérience, contrat, formation, langues) avec la taxonomie et les
règles de l'image qui tourne.

À relancer après CHAQUE mise à jour de la taxonomie : depuis que le scoring lit
les compétences enregistrées au lieu de relire le texte de chaque CV
(cv_embeddings.skills_canonical), un CV indexé avec l'ancienne taxonomie serait
sinon jugé avec elle.

Usage (sur le serveur, depuis le dossier du dépôt) -- la redirection depuis un
fichier évite les erreurs de collage d'un bloc "<< EOF" dans le terminal :

    docker exec -i keoni_matching_api_prod python3 - < ops/scripts/backfill_cv_profiles.py

Sûr à relancer (idempotent) : seuls les CV dont une valeur change sont écrits.
`updated_at` est volontairement conservé : le scoring l'utilise comme "filigrane"
de la réindexation incrémentale (cv_embeddings.updated_at le plus récent moins
une marge). Le laisser avancer à l'heure du backfill ferait croire que tous les
CV ajoutés avant sont déjà indexés, et ils ne seraient jamais repris.
"""
import sys

from sqlalchemy import select, update

from app.db import get_engine
from app.document_profile import build_document_profile
from app.models import CvEmbedding
from app.taxonomy import find_skills

BATCH = 200

engine = get_engine()
if engine is None:
    print("DB indisponible")
    sys.exit(1)

seen = 0
changed = 0
last_cv_id = 0

while True:
    with engine.begin() as conn:
        rows = conn.execute(
            select(
                CvEmbedding.cv_id,
                CvEmbedding.text_content,
                CvEmbedding.skills_canonical,
                CvEmbedding.experience_years,
                CvEmbedding.contract_type,
                CvEmbedding.education_text,
                CvEmbedding.language_terms,
            )
            .where(CvEmbedding.cv_id > last_cv_id)
            .where(CvEmbedding.text_content != "")
            .order_by(CvEmbedding.cv_id)
            .limit(BATCH)
        ).all()
        if not rows:
            break
        for row in rows:
            last_cv_id = row.cv_id
            skills = sorted(find_skills(row.text_content))
            profile = build_document_profile(row.text_content)
            new_values = {
                "skills_canonical": skills,
                "experience_years": profile.experience_years,
                "contract_type": profile.contract_type,
                "education_text": profile.education_text,
                "language_terms": list(profile.language_terms),
            }
            old_values = {
                "skills_canonical": list(row.skills_canonical or []),
                "experience_years": row.experience_years,
                "contract_type": row.contract_type,
                "education_text": row.education_text,
                "language_terms": list(row.language_terms or []),
            }
            if new_values == old_values:
                continue
            conn.execute(
                update(CvEmbedding)
                .where(CvEmbedding.cv_id == row.cv_id)
                # updated_at conservé : voir l'en-tête du fichier.
                .values(**new_values, updated_at=CvEmbedding.updated_at)
            )
            changed += 1
        seen += len(rows)
    print(f"...{seen} CV relus, {changed} modifiés", flush=True)

print(f"Terminé. {seen} CV relus, {changed} modifiés (les autres étaient déjà à jour).", flush=True)
