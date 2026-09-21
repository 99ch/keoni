"""Jeu de validation basé sur des scénarios réels déjà audités, portage de
test_validation_dataset_real_cases.py côté AI Real-Time (fa4644b).

Contexte côté AI Real-Time : chaque calibration de leur formule (zones
d'expérience, plancher du plafond skills/mots-clés, crédit sémantique sur
les mots-clés prioritaires, mécanisme mot-clé-cœur) s'appuyait sur une
lecture manuelle ponctuelle de vraies offres/CV, jamais figée en test —
un futur changement de formule pouvait silencieusement défaire un
calibrage passé sans qu'aucun test ne le détecte. Ce fichier fait la même
chose côté Keoni, avec le même esprit (bandes larges, pas de valeurs
exactes — le cross-encoder et le modèle d'embedding de compétences sont
indisponibles en CI, `similarity=0.5`/`semantic_skill_credit_fn=None`
reproduit fidèlement cet état dégradé plutôt que de le contourner).

Portage honnête, pas un copier-coller : le Cas 1 (zones d'expérience) est
une traduction FIDÈLE d'un des trois cas réels d'AI Real-Time — même
domaine (PHP/Laravel/VueJS), la taxonomie de Keoni le couvre aussi bien
que la leur, donc les mêmes textes produisent le même verdict qualitatif.
Les Cas 2 et 3 d'AI Real-Time (Business Intelligence, gouvernance IT) ne
transposent PAS : leur texte ne produit AUCUNE compétence reconnue par la
taxonomie plus restreinte de Keoni (_TECH_SKILLS est orientée
développement web — voir app/taxonomy.py), donc les rejouer tels quels ne
validerait rien de réel ici. Recréés ci-dessous dans le domaine que la
taxonomie de Keoni couvre effectivement (développement web), en gardant
exactement la même leçon de calibration que l'original : une couverture
générale solide ne doit ni s'effondrer sous un mot-clé manquant précis
(Cas 2), ni un cas de référence "tout va bien" perdre son signal positif
mots-clés (Cas 3).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.scoring import (  # noqa: E402
    PreparedCv,
    PreparedJob,
    compute_final_score,
    split_priority_keyword_terms,
)
from app.taxonomy import find_skills  # noqa: E402


def _job(text: str, title: str = "", keywords_raw: str | None = None, min_experience_years=None) -> PreparedJob:
    return PreparedJob(
        text=text,
        semantic_text=text,
        tokens=set(),
        keywords=[],
        keyword_set=set(),
        skills_canonical=set(find_skills(text)),
        location="",
        category="",
        jobtype="",
        min_experience_years=min_experience_years,
        salary_min=None,
        salary_max=None,
        title=title,
        keyword_terms_raw=split_priority_keyword_terms(keywords_raw),
        scoring_profile=None,
    )


def _cv(text: str, experience_years=None) -> PreparedCv:
    return PreparedCv(
        payload=object(),
        text=text,
        text_tokens=set(),
        title_tokens=set(),
        keywords=[],
        keyword_set=set(),
        skills_canonical=set(find_skills(text)),
        location="",
        category="",
        jobtype="",
        experience_years=experience_years,
        salary_expected_min=None,
        salary_expected_max=None,
        qualified=None,
    )


# ---------------------------------------------------------------------------
# Cas 1 (porté à l'identique depuis AI Real-Time, PHP_JOB/PHP_JUNIOR_CV/
# PHP_SENIOR_CV) : offre "5 ans minimum" -- un junior avec le bon stack (3
# ans réels) contre un senior avec le même stack (8 ans réels). Un
# recruteur classerait le senior nettement devant : l'écart d'années par
# rapport à un seuil EXPLICITE prime sur un léger avantage de couverture
# skills. Régression réelle côté AI Real-Time : avant leur recalibrage du
# poids expérience (0.12->0.20), ces deux candidats étaient scorés à moins
# d'un point d'écart.
# ---------------------------------------------------------------------------
PHP_JOB_TEXT = """
Développeur Full Stack PHP, Laravel, VueJS.
Compétences requises : PHP, Laravel, Vue.js, Back-office, SQL Server, SQL,
Jira, Git, Frontend, ORM, API REST, HTML/CSS, Responsive design.
"""
PHP_JUNIOR_TEXT = """
Consultant développeur fullstack spécialisé en PHP, Laravel et Vue.js.
Compétences : PHP (Laravel, Symfony), HTML, CSS, Bootstrap, Vue.js, MySQL,
SQL Server, Git, GitHub.
"""
PHP_SENIOR_TEXT = """
Développeur Sénior PHP / Symfony / Laravel / Vue.js / Angular.
Compétences : PHP, Symfony, Laravel, VueJS, Angular, Git, Jira, API REST,
HTML/CSS, Responsive design, Back-office.
"""


def test_explicit_years_threshold_ranks_the_senior_clearly_above_the_junior():
    job = _job(PHP_JOB_TEXT, title="Développeur Full Stack PHP, Laravel, VueJS", min_experience_years=5)
    junior = _cv(PHP_JUNIOR_TEXT, experience_years=3)
    senior = _cv(PHP_SENIOR_TEXT, experience_years=8)

    r_junior = compute_final_score(job, junior, similarity=0.5, rerank_score=None)
    r_senior = compute_final_score(job, senior, similarity=0.5, rerank_score=None)

    assert r_senior.score > r_junior.score + 5, (
        "une offre avec un seuil d'années EXPLICITE (5 ans minimum) doit "
        "classer nettement devant un candidat qui le dépasse largement (8 "
        f"ans) face à un candidat très en dessous (3 ans) -- obtenu "
        f"junior={r_junior.score} senior={r_senior.score}"
    )
    assert r_senior.breakdown["experience"] == 1.0, (
        "8 ans pour 5 requis (ratio 1.6) est un cas 'légèrement supérieur' "
        "-- crédit plein, pas de pénalité de sur-qualification"
    )
    assert r_junior.breakdown["experience"] < 0.8, (
        "3 ans pour 5 requis doit rester nettement en dessous du plein crédit"
    )


# ---------------------------------------------------------------------------
# Cas 2 (adapté au domaine web -- voir docstring du module) : offre dont la
# liste de mots-clés prioritaires est précise (5 outils nommés) -- un
# candidat solide en couverture générale, qui n'a simplement jamais
# mentionné l'un des 5 outils, ne doit pas s'effondrer sous un score
# médiocre juste par manque d'un item.
# ---------------------------------------------------------------------------
BACKEND_JOB_TEXT = """
Développeur Backend Node.js Expert.
Stack recherchée : Node.js, Express, MongoDB, Docker, Kubernetes,
architecture microservices, API REST, tests automatisés.
"""
BACKEND_JOB_KEYWORDS = "Node.js\nExpress\nMongoDB\nDocker\nKubernetes"
BACKEND_CV_TEXT = """
Développeur backend avec une solide expertise Node.js et Express,
architecture de microservices en production, API REST, bases de données
MongoDB, tests automatisés. Conteneurisation Docker en environnement de
développement et CI/CD.
"""


def test_solid_profile_does_not_collapse_under_one_missing_named_tool():
    job = _job(
        BACKEND_JOB_TEXT,
        title="Développeur Backend Node.js Expert",
        keywords_raw=BACKEND_JOB_KEYWORDS,
    )
    cv = _cv(BACKEND_CV_TEXT)
    result = compute_final_score(job, cv, similarity=0.5, rerank_score=None)

    assert result.breakdown["skills"] >= 0.75, (
        "couverture générale solide pour ce profil (4 compétences "
        "reconnues sur 5), malgré l'absence totale de Kubernetes"
    )
    assert result.score >= 65, (
        "un profil solide manquant un seul outil nommé sur cinq ne doit "
        f"pas s'effondrer sous 65% -- obtenu {result.score}"
    )


# ---------------------------------------------------------------------------
# Cas 3 (adapté au domaine web) : offre avec mots-clés prioritaires bien
# alignés avec le profil -- cas de référence "tout va bien", pour détecter
# une régression qui affaiblirait le signal positif des mots-clés sur un
# cas simple.
# ---------------------------------------------------------------------------
FRONTEND_JOB_TEXT = """
Développeur Frontend React Senior.
Stack recherchée : React, TypeScript, GraphQL, Redux, tests unitaires.
"""
FRONTEND_JOB_KEYWORDS = "React\nTypeScript\nGraphQL\nRedux"
FRONTEND_CV_TEXT = """
Développeur frontend spécialisé React et TypeScript depuis 5 ans, API
GraphQL, gestion d'état Redux, tests unitaires avec Jest. Plusieurs
missions en environnement agile pour des grands comptes.
"""


def test_well_aligned_priority_keywords_still_score_high():
    job = _job(
        FRONTEND_JOB_TEXT,
        title="Développeur Frontend React Senior",
        keywords_raw=FRONTEND_JOB_KEYWORDS,
    )
    cv = _cv(FRONTEND_CV_TEXT)
    result = compute_final_score(job, cv, similarity=0.5, rerank_score=None)

    assert result.breakdown["keywords"] == 1.0, (
        "couverture complète des mots-clés prioritaires attendue pour ce "
        f"profil parfaitement aligné, obtenu {result.breakdown['keywords']}"
    )
    assert result.score >= 80, (
        f"couverture quasi totale (skills + mots-clés) attendue >= 80%, "
        f"obtenu {result.score}"
    )
