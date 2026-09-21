"""Formule de score CV/offre, portée depuis AI Real-Time (backend/app/services/matcher.py).

Keoni utilisait jusqu'ici une formule additive (base sémantique + bonus/malus
en points bruts). AI Real-Time utilise une moyenne pondérée, renormalisée sur
les seules composantes qui ont un vrai signal à comparer (`has_signal`), avec
un plafond de couverture skills/mots-clés — voir `_weights()` et
`match_parsed_documents()` dans le fichier source. Porté ici à l'identique
dans sa structure, avec les composantes adaptées aux données réellement
disponibles côté Keoni (pas d'extraction education/langues séparée par
exemple — ces composantes n'ont simplement jamais `has_signal=True` chez
Keoni et sont donc exclues de la moyenne, elles ne sont pas inventées).

Volontairement sans import lourd (pas de sentence-transformers/faiss/tika) :
la mesure de similarité sémantique et le crédit sémantique de compétences
sont calculés ailleurs (main.py, qui a accès aux modèles) et seulement
injectés ici en paramètre (`rerank_score`, `semantic_skill_credit_fn`), ce
qui permet de tester toute cette formule sans installer la pile ML (voir
tests/test_scoring.py et requirements-dev.txt).

Composantes AI Real-Time désormais toutes portées :
- Profils de scoring (SCORING_PROFILES / job.scoring_profile) : presets de
  poids nommés, comme _SCORING_PROFILES côté AI Real-Time. Keoni n'a pas
  encore d'UI recruteur pour en choisir un par offre -- le champ existe
  côté API (JobPayload.scoring_profile / job.meta["scoring_profile"]),
  prêt pour le câblage frontend.
- Couverture "core keyword" (core_keyword_coverage) : pénalité
  multiplicative quand un mot-clé "appuyé" par le recruteur (répété
  plusieurs fois, ou présent dans le titre de l'offre) est absent du CV.
  Portée sur le champ WP "keywords" existant de Keoni (un texte
  potentiellement multi-termes) plutôt que sur un textarea dédié "Mots
  Clés.docx" qui n'existe pas dans son formulaire -- voir
  split_priority_keyword_terms.
- Inférence de séniorité depuis le titre de l'offre
  (infer_seniority_years), utilisée par experience_component quand
  l'offre ne précise aucune durée explicite.
- Crédit sémantique partiel sur les compétences/mots-clés non matchés
  littéralement : le calcul d'embedding lui-même vit dans main.py (accès
  au modèle), injecté ici via `semantic_skill_credit_fn` dans
  skills_component/priority_keyword_component -- même contrat que
  _semantic_skill_credit côté AI Real-Time, réutilisé pour les deux
  composantes comme chez eux.
"""
from __future__ import annotations

import re
import threading
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable, FrozenSet, List, Optional, Sequence, Tuple, Union

from app.taxonomy import normalize_skill

WORD_PATTERN = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ0-9']+")
DEFAULT_STOPWORDS = {
    "and",
    "the",
    "for",
    "les",
    "des",
    "une",
    "avec",
    "sur",
    "par",
    "un",
    "aux",
    "von",
    "und",
    "pour",
    "entre",
    "dans",
    "from",
    "chez",
    "nos",
    "vos",
    "mon",
    "ton",
    "son",
    "his",
    "her",
    "our",
    "your",
    "their",
}

SemanticCreditFn = Callable[[FrozenSet[str], FrozenSet[str]], float]


def normalize_whitespace(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def fold_text(value: str) -> str:
    """Translittération ASCII minuscule (accents retirés), même principe que
    `_fold()` côté AI Real-Time -- utilisé pour comparer des libellés sans
    être sensible aux accents/casse."""
    nfkd = unicodedata.normalize("NFKD", value)
    return nfkd.encode("ascii", "ignore").decode("ascii").lower()


def tokenize(text: str) -> set[str]:
    tokens: set[str] = set()
    for match in WORD_PATTERN.finditer(text.lower()):
        word = match.group()
        if len(word) <= 2 or word in DEFAULT_STOPWORDS:
            continue
        tokens.add(word)
    return tokens


def overlap_text(a: str, b: str) -> bool:
    if not a or not b:
        return False
    if a in b or b in a:
        return True
    return bool(tokenize(a) & tokenize(b))


@dataclass(slots=True)
class PreparedJob:
    text: str
    semantic_text: str
    tokens: set[str]
    keywords: List[str]
    keyword_set: set[str]
    skills_canonical: set[str]
    location: str
    category: str
    jobtype: str
    min_experience_years: Optional[float]
    salary_min: Optional[float]
    salary_max: Optional[float]
    title: str = ""
    # Termes bruts du champ "keywords", répétitions conservées, puces/entête
    # nettoyées -- alimente à la fois priority_keyword_component et
    # core_keyword_coverage. Voir split_priority_keyword_terms.
    keyword_terms_raw: List[str] = field(default_factory=list)
    # Preset de pondération recruteur (JobPayload.scoring_profile /
    # job.meta["scoring_profile"]), ou None pour le défaut plateforme.
    scoring_profile: Optional[str] = None


@dataclass(slots=True)
class PreparedCv:
    payload: object
    text: str
    text_tokens: set[str]
    title_tokens: set[str]
    keywords: List[str]
    keyword_set: set[str]
    skills_canonical: set[str]
    location: str
    category: str
    jobtype: str
    experience_years: Optional[float]
    salary_expected_min: Optional[float]
    salary_expected_max: Optional[float]
    qualified: Optional[bool]


# ── Poids ──────────────────────────────────────────────────────────────────
#
# Mêmes ordres de grandeur que _DEFAULT_W chez AI Real-Time (semantic=0.10,
# skills=0.40, priority_keywords=0.40, experience=0.20, education=0.08,
# languages=0.05, contract=0.05) : Keoni n'a pas de composantes
# education/languages séparées, remplacées ici par category/location, des
# signaux structurels du même ordre d'importance secondaire dans le domaine
# recrutement de Keoni. `salary` et `qualification` n'existent pas chez AI
# Real-Time (pas de notion de prétention salariale ni de "qualifié déclaré"
# dans leur modèle) ; ajoutés avec un poids modeste, du même ordre que
# jobtype/location, plutôt que de les supprimer silencieusement.
DEFAULT_WEIGHTS: dict[str, float] = {
    "semantic": 0.10,
    "skills": 0.35,
    "keywords": 0.35,
    "experience": 0.20,
    "jobtype": 0.05,
    "category": 0.08,
    "location": 0.05,
    "salary": 0.05,
    "qualification": 0.10,
}

# Presets nommés, portage de _SCORING_PROFILES côté AI Real-Time : un
# ensemble FERMÉ de variantes réfléchies de DEFAULT_WEIGHTS, pas des poids
# libres saisis par un recruteur (même raisonnement que chez eux -- voir
# leur commentaire sur la suppression de /feedback/apply-weights : un
# recruteur n'est pas data scientist, et un poids qui "sonne bien"
# intuitivement peut réintroduire un biais déjà mesuré et corrigé).
# Mêmes deltas que leurs presets, appliqués aux clés équivalentes chez
# Keoni (keywords ~ priority_keywords) :
#   priorite_experience : skills -0.05, keywords -0.10, experience +0.15
#   priorite_mots_cles  : keywords +0.10, experience -0.08
SCORING_PROFILES: dict[str, dict[str, float]] = {
    "equilibre": dict(DEFAULT_WEIGHTS),
    "priorite_experience": {
        **DEFAULT_WEIGHTS,
        "skills": DEFAULT_WEIGHTS["skills"] - 0.05,
        "keywords": DEFAULT_WEIGHTS["keywords"] - 0.10,
        "experience": DEFAULT_WEIGHTS["experience"] + 0.15,
    },
    "priorite_mots_cles": {
        **DEFAULT_WEIGHTS,
        "keywords": DEFAULT_WEIGHTS["keywords"] + 0.10,
        "experience": DEFAULT_WEIGHTS["experience"] - 0.08,
    },
}


def weights_for_profile(
    profile: Optional[str], base: Optional[dict[str, float]] = None
) -> dict[str, float]:
    """Poids pour ce match. `profile` prend le pas quand reconnu (choix
    recruteur explicite pour CETTE offre) ; repli silencieux sur `base` (ou
    DEFAULT_WEIGHTS) pour un nom de profil vide ou obsolète -- même règle
    que _weights(profile) côté AI Real-Time."""
    if profile and profile in SCORING_PROFILES:
        return dict(SCORING_PROFILES[profile])
    return dict(base) if base is not None else dict(DEFAULT_WEIGHTS)


# Plafond de couverture skills/mots-clés : même valeur et même mécanisme que
# _SKILL_CAP_FLOOR chez AI Real-Time (matcher.py), calibrée par eux sur un
# jeu de validation de 16 CV / 1 offre avec cible humaine indépendante par
# paire. Keoni n'a pas encore son propre jeu de validation équivalent, donc
# on part de leur valeur mesurée plutôt que d'un chiffre choisi au hasard.
SKILL_CAP_FLOOR = 0.30

# Réglage live (sans redéploiement) du seuil/plafond du crédit sémantique de
# compétences (utilisé par skills_component ET priority_keyword_component
# via semantic_skill_credit_fn, côté main.py) -- portage de
# _skill_embedding_tuning_override côté AI Real-Time. État en mémoire
# uniquement (réinitialisé au prochain redémarrage), volontairement pas
# auto-appliqué : un changement de paramètre de scoring doit rester une
# action admin délibérée et visible, jamais automatique. Vit ici (pas dans
# main.py) car c'est un simple état threadsafe sans dépendance ML -- garde
# scoring.py testable sans installer sentence-transformers.
_skill_embedding_tuning_override: Optional[dict[str, float]] = None
_skill_embedding_tuning_lock = threading.Lock()


def set_skill_embedding_tuning(threshold: Optional[float], max_credit: Optional[float]) -> None:
    """Passer threshold ET max_credit à None réinitialise l'override et
    revient aux valeurs par défaut (settings)."""
    global _skill_embedding_tuning_override
    with _skill_embedding_tuning_lock:
        if threshold is None and max_credit is None:
            _skill_embedding_tuning_override = None
            return
        current = dict(_skill_embedding_tuning_override or {})
        if threshold is not None:
            current["threshold"] = threshold
        if max_credit is not None:
            current["max_credit"] = max_credit
        _skill_embedding_tuning_override = current


def get_skill_embedding_tuning(
    default_threshold: float, default_max_credit: float
) -> Tuple[float, float, bool]:
    """Retourne (threshold, max_credit, is_overridden) -- l'override actif
    s'il existe, sinon les valeurs par défaut passées par l'appelant (lues
    depuis Settings côté main.py)."""
    with _skill_embedding_tuning_lock:
        override = dict(_skill_embedding_tuning_override) if _skill_embedding_tuning_override else {}
    threshold = override.get("threshold", default_threshold)
    max_credit = override.get("max_credit", default_max_credit)
    return threshold, max_credit, bool(override)


# Zones d'expérience : portées à l'identique depuis AI Real-Time matcher.py
# (calibration du 2026-09-11 sur cas de production réel) — voir le
# commentaire historique dans matcher.py::_experience_zone_score pour le
# raisonnement (un poste "au moins N ans" est un plancher, pas une fenêtre :
# plus d'expérience que le plancher n'est jamais en soi un problème jusqu'à
# un multiple raisonnable, au-delà duquel le risque de surqualification
# dégrade progressivement le score plutôt qu'une chute brutale).
EXPERIENCE_COMFORTABLE_OVER_RATIO = 2.0
EXPERIENCE_SEVERE_OVER_RATIO = 4.0
EXPERIENCE_SEVERE_OVER_FLOOR = 0.85

# Signaux de séniorité dans le titre/texte de l'offre, utilisés quand aucune
# durée explicite n'est extraite -- portage à l'identique de
# _SENIORITY_YEARS_SIGNALS côté AI Real-Time.
SENIORITY_YEARS_SIGNALS: List[Tuple[re.Pattern, int]] = [
    (re.compile(r"\bexpert(?:e)?s?\b", re.IGNORECASE), 8),
    (re.compile(r"\bs[ée]nior(?:e)?s?\b", re.IGNORECASE), 5),
    (re.compile(r"\bconfirm[ée]e?s?\b", re.IGNORECASE), 3),
    (re.compile(r"\bjunior(?:e)?s?\b", re.IGNORECASE), 1),
    (re.compile(r"\bd[ée]butant(?:e)?s?\b", re.IGNORECASE), 1),
]

# Une exigence INFÉRÉE (pas explicitement chiffrée par le recruteur) est un
# signal plus faible et ambigu qu'un "5 ans minimum" écrit noir sur blanc --
# amortie à mi-chemin du neutre (0.75) plutôt que comptée à pleine force.
# Même valeur que côté AI Real-Time.
INFERRED_EXPERIENCE_DAMPENING = 0.5


def experience_zone_score(cv_years: float, job_years: float) -> float:
    """Calcul de zone pur (inferieur / egal / legerement superieur / trop
    superieur), porté à l'identique depuis AI Real-Time. Suppose
    cv_years > 0 et job_years > 0 (voir experience_component)."""
    if cv_years == job_years:
        return 1.0
    if cv_years > job_years:
        ratio = cv_years / job_years
        if ratio <= EXPERIENCE_COMFORTABLE_OVER_RATIO:
            return 1.0
        if ratio >= EXPERIENCE_SEVERE_OVER_RATIO:
            return EXPERIENCE_SEVERE_OVER_FLOOR
        span = EXPERIENCE_SEVERE_OVER_RATIO - EXPERIENCE_COMFORTABLE_OVER_RATIO
        progress = (ratio - EXPERIENCE_COMFORTABLE_OVER_RATIO) / span
        return 1.0 - progress * (1.0 - EXPERIENCE_SEVERE_OVER_FLOOR)
    shortfall = (job_years - cv_years) / job_years
    return max(0.0, 1.0 - shortfall)


def infer_seniority_years(job: PreparedJob) -> int:
    """Plancher d'expérience implicite déduit du vocabulaire de séniorité
    dans le titre (repli sur le début du texte complet) quand aucune durée
    explicite n'a été extraite. Retourne 0 (aucun signal) si rien ne
    correspond. Portage à l'identique de _infer_seniority_years."""
    haystack = job.title or job.text[:500]
    for pattern, years in SENIORITY_YEARS_SIGNALS:
        if pattern.search(haystack):
            return years
    return 0


# ── Mots-clés prioritaires (champ WP "keywords") ─────────────────────────
#
# AI Real-Time alimente ce mécanisme depuis un textarea recruteur dédié,
# une ligne par mot-clé ("Mots Clés.docx" collé tel quel, avec puces).
# Keoni n'a pas ce textarea : son formulaire WP n'a qu'un seul champ texte
# "keywords" (typiquement une liste séparée par virgules). On y applique le
# même nettoyage (puce/numérotation en tête, ligne d'en-tête "Mots Clés :")
# mais on NE découpe PAS sur "/" comme le fait le champ mots-clés général
# de Keoni (parse_keywords) : un couple "MOA / AMOA" saisi par un
# recruteur doit rester une seule ligne pour que
# normalize_priority_keyword() puisse appliquer sa logique de repli sur
# chaque moitié -- le découper ici en amont la rendrait inutile.

_BULLET_PREFIX_RE = re.compile(r"^(?:[•◦‣▪·\-\*]|\(?\d+[.)])\s+")
_KEYWORDS_HEADER_MARKERS = ("mots cle", "mot cle", "keyword")


def split_priority_keyword_terms(raw: Optional[Union[str, Sequence[str]]]) -> List[str]:
    """Termes de mots-clés bruts, répétitions conservées (pas de dédup --
    nécessaire pour core_keyword_coverage), puce/numérotation de tête et
    ligne d'en-tête "Mots Clés :" retirées. Portage de
    split_priority_keywords() côté AI Real-Time, adapté au séparateur du
    champ WP de Keoni (virgule/point-virgule/retour à la ligne, PAS "/" --
    voir le commentaire ci-dessus)."""
    if not raw:
        return []
    if isinstance(raw, str):
        lines: Sequence[str] = re.split(r"[,;\n]+", raw)
    else:
        lines = list(raw)

    terms: List[str] = []
    for line in lines:
        term = _BULLET_PREFIX_RE.sub("", str(line).strip())
        term = term.strip().strip(":,;").strip()
        if not term:
            continue
        folded_term = fold_text(term).replace("-", " ")
        if len(term.split()) <= 6 and any(marker in folded_term for marker in _KEYWORDS_HEADER_MARKERS):
            continue
        terms.append(term)
    return terms


def normalize_priority_keyword(raw_term: str) -> Optional[str]:
    """normalize_skill(), étendu pour l'abréviation recruteur combinant deux
    synonymes sur une ligne avec un "/" ("MOA / AMOA"). Portage à
    l'identique de _normalize_priority_keyword."""
    canonical = normalize_skill(raw_term)
    if canonical:
        return canonical
    if "/" not in raw_term:
        return None
    parts_canonical = {normalize_skill(part) for part in raw_term.split("/") if part.strip()}
    parts_canonical.discard(None)
    if len(parts_canonical) == 1:
        return next(iter(parts_canonical))
    return None


def enrich_cv_skills(job: PreparedJob, cv: PreparedCv) -> FrozenSet[str]:
    """Complète cv.skills_canonical avec les mots-clés prioritaires bruts
    trouvés littéralement dans le texte du CV mais sans entrée taxonomie
    (sigles métier absents du référentiel ROME, ex. réels chez AI
    Real-Time : "LOD2", "DORA", "TRM"). Ne mute jamais cv.skills_canonical
    directement -- un même PreparedCv pourrait en théorie être comparé à
    plusieurs offres. Portage de _apply_priority_keywords, qui mutait
    cv.skill_terms côté AI Real-Time."""
    if not job.keyword_terms_raw:
        return frozenset(cv.skills_canonical)
    extra: set[str] = set()
    cv_text_folded = fold_text(cv.text)
    for raw_term in job.keyword_terms_raw:
        canonical = normalize_priority_keyword(raw_term)
        if canonical or raw_term in cv.skills_canonical:
            continue  # déjà détectable via skills_canonical comme n'importe quelle autre compétence
        if re.search(rf"\b{re.escape(fold_text(raw_term))}\b", cv_text_folded):
            extra.add(raw_term)
    return frozenset(cv.skills_canonical) | extra


def resolve_priority_keywords(
    job: PreparedJob, cv_skills: FrozenSet[str]
) -> Tuple[List[str], List[str]]:
    """(matched, all_terms) : mots-clés prioritaires de l'offre résolus à
    leur canonique taxonomie (ou laissés en terme brut si inconnu),
    DÉDUPLIQUÉS par canonique. Portage de _resolve_priority_keywords."""
    seen: dict[str, None] = {}
    matched_seen: dict[str, None] = {}
    for raw_term in job.keyword_terms_raw:
        canonical = normalize_priority_keyword(raw_term) or raw_term
        seen.setdefault(canonical, None)
        if canonical in cv_skills:
            matched_seen.setdefault(canonical, None)
    return list(matched_seen), list(seen)


# Répétition + présence dans le titre de l'offre = mot-clé "appuyé" par le
# recruteur (le tool/concept central de l'offre, pas juste un item parmi
# d'autres). Portage à l'identique de _CORE_KEYWORD_MIN_REPEATS /
# _CORE_PENALTY_FLOOR / _TITLE_ALTERNATION_RE côté AI Real-Time.
_CORE_KEYWORD_MIN_REPEATS = 3
_CORE_PENALTY_FLOOR = 0.45
_TITLE_ALTERNATION_RE = re.compile(r"/|\bou\b", re.IGNORECASE)


def core_keyword_coverage(job: PreparedJob, cv_skills: FrozenSet[str]) -> float:
    """Couverture (0.0-1.0) des mots-clés "appuyés" de l'offre. Portage à
    l'identique de _core_keyword_coverage : 1.0 (aucune pénalité) quand
    aucun mot-clé n'est appuyé."""
    if not job.keyword_terms_raw:
        return 1.0
    counts = Counter(normalize_priority_keyword(t) or t for t in job.keyword_terms_raw)
    core = {canonical for canonical, n in counts.items() if n >= _CORE_KEYWORD_MIN_REPEATS}

    title_folded = "" if _TITLE_ALTERNATION_RE.search(job.title) else fold_text(job.title)
    matched = {c for c in core if c in cv_skills}
    if title_folded:
        for raw_term in job.keyword_terms_raw:
            canonical = normalize_priority_keyword(raw_term) or raw_term
            if canonical in core:
                continue
            term_folded = fold_text(raw_term)
            if not term_folded or not re.search(rf"\b{re.escape(term_folded)}\b", title_folded):
                continue
            core.add(canonical)
            if canonical in cv_skills or raw_term in cv_skills:
                matched.add(canonical)

    if not core:
        return 1.0
    return len(matched) / len(core)


# ── Composantes structurées ───────────────────────────────────────────────
#
# Chaque fonction retourne (valeur 0-1, has_signal). has_signal=False quand
# le score neutre/pénalité est dû à une INFORMATION MANQUANTE plutôt qu'à
# une vraie comparaison (ex : l'offre ne précise pas de type de contrat) —
# même contrat que les fonctions `_*_score` de matcher.py. Ces composantes
# sont exclues de la moyenne pondérée quand has_signal=False (voir
# compute_final_score), pour qu'une extraction manquante ne fasse pas
# chuter le score à pleine pondération au lieu de ne simplement pas compter.


def skills_component(
    job: PreparedJob,
    cv_skills: FrozenSet[str],
    semantic_credit_fn: Optional[SemanticCreditFn] = None,
) -> Tuple[float, bool, List[str]]:
    """Couverture des compétences de l'offre (taxonomie ROME/technique) par
    le CV. Hybride (comme _skill_score chez AI Real-Time) : les
    correspondances exactes comptent toujours plein pot ; pour chaque
    compétence requise non matchée littéralement, `semantic_credit_fn` (si
    fournie) peut ajouter un crédit partiel par similarité d'embedding --
    ne peut jamais faire baisser le score vs lexical pur, jamais dépasser
    une correspondance exacte."""
    if not job.skills_canonical:
        return 0.5, False, []  # offre sans compétence identifiable à comparer
    if not cv_skills:
        return 0.0, False, []  # rien d'extrait côté CV pour comparer

    matched = job.skills_canonical & cv_skills
    unmatched = job.skills_canonical - matched
    credit = float(len(matched))
    if unmatched and semantic_credit_fn is not None:
        credit += semantic_credit_fn(frozenset(unmatched), frozenset(cv_skills))

    coverage = credit / len(job.skills_canonical)
    return min(1.0, coverage), True, sorted(matched)


def priority_keyword_component(
    job: PreparedJob,
    cv_skills: FrozenSet[str],
    semantic_credit_fn: Optional[SemanticCreditFn] = None,
) -> Tuple[float, bool, List[str]]:
    """Couverture des mots-clés prioritaires du recruteur (champ WP
    "keywords", traité comme la liste curatée d'AI Real-Time -- voir le
    commentaire en tête de section). has_signal uniquement quand l'offre a
    des mots-clés (même règle que chez eux : "only counted when the job
    has priority keywords"). Portage de _priority_keyword_score."""
    if not job.keyword_terms_raw:
        return 0.5, False, []
    matched, all_terms = resolve_priority_keywords(job, cv_skills)
    if not all_terms:
        return 0.5, True, []
    unmatched = set(all_terms) - set(matched)
    credit = float(len(matched))
    if unmatched and semantic_credit_fn is not None:
        credit += semantic_credit_fn(frozenset(unmatched), frozenset(cv_skills))
    coverage = min(1.0, credit / len(all_terms))
    return coverage, True, sorted(matched)


def experience_component(job: PreparedJob, cv: PreparedCv) -> Tuple[float, bool, Optional[float]]:
    """Retourne (valeur 0-1, has_signal, effective_job_years). Le 3e élément
    expose le plancher d'années réellement utilisé (explicite ou déduit du
    titre par infer_seniority_years) — job.min_experience_years reste None
    quand seule l'inférence a produit un chiffre, donc les appelants qui
    veulent afficher/comparer ce plancher doivent lire cette valeur plutôt
    que le champ brut de PreparedJob."""
    job_y = job.min_experience_years or 0.0
    cv_y = cv.experience_years or 0.0
    inferred = False
    if job_y <= 0:
        inferred_years = infer_seniority_years(job)
        if inferred_years > 0:
            job_y = float(inferred_years)
            inferred = True
    if job_y <= 0 and cv_y <= 0:
        return 0.5, False, None  # aucune année extraite ni inférable ni côté offre ni côté CV
    if job_y <= 0:
        return 0.75, False, None  # offre sans exigence d'expérience à comparer
    if cv_y <= 0:
        return 0.2, False, job_y  # expérience du CV non extraite
    raw = experience_zone_score(cv_y, job_y)
    if inferred:
        raw = INFERRED_EXPERIENCE_DAMPENING * raw + (1 - INFERRED_EXPERIENCE_DAMPENING) * 0.75
    return raw, True, job_y


def jobtype_component(job: PreparedJob, cv: PreparedCv) -> Tuple[float, bool]:
    if not job.jobtype or not cv.jobtype:
        return 0.5, False
    return (1.0 if overlap_text(job.jobtype, cv.jobtype) else 0.0), True


def category_component(job: PreparedJob, cv: PreparedCv) -> Tuple[float, bool]:
    if not job.category or not cv.category:
        return 0.5, False
    return (1.0 if overlap_text(job.category, cv.category) else 0.0), True


def location_component(job: PreparedJob, cv: PreparedCv) -> Tuple[float, bool]:
    if not job.location or not cv.location:
        return 0.5, False
    return (1.0 if (job.location in cv.location or cv.location in job.location) else 0.0), True


def salary_component(job: PreparedJob, cv: PreparedCv) -> Tuple[float, bool]:
    if job.salary_max is None or cv.salary_expected_min is None:
        return 0.5, False
    return (1.0 if cv.salary_expected_min <= job.salary_max else 0.0), True


def qualification_component(cv: PreparedCv) -> Tuple[float, bool]:
    if cv.qualified is None:
        return 0.5, False
    return (1.0 if cv.qualified else 0.0), True


@dataclass(slots=True)
class ScoreResult:
    score: float  # 0-100
    semantic: float
    skill_hits: List[str] = field(default_factory=list)
    keyword_hits: List[str] = field(default_factory=list)
    keyword_total: int = 0
    core_keyword_coverage: float = 1.0
    breakdown: dict = field(default_factory=dict)
    low_confidence_components: List[str] = field(default_factory=list)
    weights: dict = field(default_factory=dict)


def compute_final_score(
    job: PreparedJob,
    cv: PreparedCv,
    similarity: float,
    rerank_score: Optional[float] = None,
    weights: Optional[dict[str, float]] = None,
    semantic_skill_credit_fn: Optional[SemanticCreditFn] = None,
) -> ScoreResult:
    """Moyenne pondérée renormalisée sur les composantes ayant un vrai
    signal, plafond de couverture skills/mots-clés, puis pénalité
    multiplicative "core keyword" -- même structure et même ordre que
    `match_parsed_documents()` chez AI Real-Time.

    `semantic` n'a pas de has_signal : le cross-encoder produit toujours
    une comparaison (réelle, ou un repli neutre à 0.5 s'il est
    indisponible), donc il compte toujours — même règle que chez eux.
    """
    w = dict(weights) if weights is not None else dict(DEFAULT_WEIGHTS)

    effective_cv_skills = enrich_cv_skills(job, cv)

    semantic = max(0.0, min(1.0, rerank_score if rerank_score is not None else similarity))

    skills_val, skills_ok, skill_hits = skills_component(
        job, effective_cv_skills, semantic_skill_credit_fn
    )
    keywords_val, keywords_ok, keyword_hits = priority_keyword_component(
        job, effective_cv_skills, semantic_skill_credit_fn
    )
    experience_val, experience_ok, experience_required_years = experience_component(job, cv)
    jobtype_val, jobtype_ok = jobtype_component(job, cv)
    category_val, category_ok = category_component(job, cv)
    location_val, location_ok = location_component(job, cv)
    salary_val, salary_ok = salary_component(job, cv)
    qualification_val, qualification_ok = qualification_component(cv)

    structured = (
        ("skills", skills_val, skills_ok),
        ("keywords", keywords_val, keywords_ok),
        ("experience", experience_val, experience_ok),
        ("jobtype", jobtype_val, jobtype_ok),
        ("category", category_val, category_ok),
        ("location", location_val, location_ok),
        ("salary", salary_val, salary_ok),
        ("qualification", qualification_val, qualification_ok),
    )

    weighted_sum = w["semantic"] * semantic
    weight_total = w["semantic"]
    for name, value, ok in structured:
        if ok:
            weighted_sum += w[name] * value
            weight_total += w[name]

    final = weighted_sum / weight_total if weight_total > 0 else 0.5
    final = max(0.0, min(1.0, final))

    if skills_ok:
        final = min(final, SKILL_CAP_FLOOR + (1 - SKILL_CAP_FLOOR) * skills_val)
    if keywords_ok:
        final = min(final, SKILL_CAP_FLOOR + (1 - SKILL_CAP_FLOOR) * keywords_val)

    # Pénalité multiplicative (pas un plafond) pour un mot-clé "appuyé"
    # manquant -- appliquée APRÈS les plafonds ci-dessus, donc elle réduit
    # proportionnellement un score déjà différencié plutôt que d'écraser
    # tout le monde sur le même plancher. Même ordre que
    # match_parsed_documents côté AI Real-Time.
    core_coverage = core_keyword_coverage(job, effective_cv_skills)
    if core_coverage < 1.0:
        final *= _CORE_PENALTY_FLOOR + (1 - _CORE_PENALTY_FLOOR) * core_coverage

    breakdown = {
        "semantic": round(semantic, 4),
        "skills": round(skills_val, 4) if skills_ok else None,
        "keywords": round(keywords_val, 4) if keywords_ok else None,
        "experience": round(experience_val, 4) if experience_ok else None,
        "jobtype": round(jobtype_val, 4) if jobtype_ok else None,
        "category": round(category_val, 4) if category_ok else None,
        "location": round(location_val, 4) if location_ok else None,
        "salary": round(salary_val, 4) if salary_ok else None,
        "qualification": round(qualification_val, 4) if qualification_ok else None,
        "core_keyword_coverage": round(core_coverage, 4),
        "experience_required_years": experience_required_years,
    }
    low_confidence = [name for name, _value, ok in structured if not ok]
    _matched, all_priority_terms = resolve_priority_keywords(job, effective_cv_skills)

    return ScoreResult(
        score=round(final * 100, 2),
        semantic=semantic,
        skill_hits=skill_hits,
        keyword_hits=keyword_hits,
        keyword_total=len(all_priority_terms),
        core_keyword_coverage=round(core_coverage, 4),
        breakdown=breakdown,
        low_confidence_components=low_confidence,
        weights=w,
    )
