"""API de matching CV / offre d'emploi (Keoni).

Rôle
    Reçoit une offre + une liste de CV, calcule un score de compatibilité 0-100
    pour chaque CV, et renvoie le classement trié.

Pipeline de scoring
    1. Filtres durs (qualification, type de contrat, expérience) : élimine les
       CV incompatibles avant tout calcul coûteux.
    2. Embeddings sémantiques via sentence-transformers (MiniLM-L6-v2) puis
       recherche des top_k plus proches par similarité cosinus (FAISS).
    3. Score final = base sémantique (70 %) + bonus mots-clés / titre /
       localisation / structure (jobtype, catégorie, expérience, salaire),
       borné à [0, 100].

Endpoints
    GET  /health  : sonde de vie (pas d'auth)
    POST /score   : matching (auth via header X-API-Key)

Variables d'environnement principales
    MATCHING_API_KEY, SENTENCE_MODEL, MATCHING_TOP_K, MATCHING_MIN_SIMILARITY,
    MATCHING_*_WEIGHT, MATCHING_HARD_FILTER_*, DATA_DIR, MODEL_CACHE, LOG_LEVEL.
"""

from __future__ import annotations

import logging
import os
import re
import secrets
import time
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Iterable, List, Optional, Sequence, Tuple, Union

import faiss
import numpy as np
import pytesseract
from fastapi import Depends, FastAPI, Header, HTTPException, status
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field
from sentence_transformers import SentenceTransformer
from tika import parser

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))

app = FastAPI(title="Keoni Matching API", version="0.2.0")

# ---------------------------------------------------------------------------
# Constantes de tokenisation
# ---------------------------------------------------------------------------

# Motif de mot : lettres latines (accents inclus), chiffres, apostrophe.
WORD_PATTERN = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ0-9']+")

# Liste minimale de mots vides FR/EN/DE écartés de la comparaison de mots-clés.
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
    "une",
    "aux",
    "von",
    "und",
    "pour",
    "entre",
    "dans",
    "from",
    "avec",
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


# ---------------------------------------------------------------------------
# Configuration runtime (pilotable via variables d'environnement)
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class Settings:
    """Paramètres de matching et pondérations du score final.

    Toutes les valeurs peuvent être surchargées via variables d'environnement.
    Les *_WEIGHT contrôlent la contribution de chaque signal au score global ;
    les HARD_FILTER_* activent les filtres éliminatoires appliqués en amont.
    """

    sentence_model: str = os.getenv("SENTENCE_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    top_k: int = int(os.getenv("MATCHING_TOP_K", "200"))
    min_similarity: float = float(os.getenv("MATCHING_MIN_SIMILARITY", "0.2"))
    keyword_weight: float = float(os.getenv("MATCHING_KEYWORD_WEIGHT", "5"))
    title_weight: float = float(os.getenv("MATCHING_TITLE_WEIGHT", "10"))
    location_weight: float = float(os.getenv("MATCHING_LOCATION_WEIGHT", "5"))
    category_weight: float = float(os.getenv("MATCHING_CATEGORY_WEIGHT", "6"))
    jobtype_weight: float = float(os.getenv("MATCHING_JOBTYPE_WEIGHT", "8"))
    experience_weight: float = float(os.getenv("MATCHING_EXPERIENCE_WEIGHT", "8"))
    salary_weight: float = float(os.getenv("MATCHING_SALARY_WEIGHT", "6"))
    qualification_penalty: float = float(os.getenv("MATCHING_QUALIFICATION_PENALTY", "20"))
    hard_filter_jobtype: bool = os.getenv("MATCHING_HARD_FILTER_JOBTYPE", "1") == "1"
    hard_filter_qualification: bool = os.getenv("MATCHING_HARD_FILTER_QUALIFICATION", "1") == "1"
    embed_batch_size: int = int(os.getenv("EMBED_BATCH_SIZE", "32"))
    preload_model: bool = os.getenv("MATCHING_PRELOAD_MODEL", "1") == "1"
    data_dir: Path = Path(os.getenv("DATA_DIR", "/data/cv_raw"))


settings = Settings()

# Modèle sentence-transformer chargé paresseusement (thread-safe via lock).
_model: Optional[SentenceTransformer] = None
_model_lock = Lock()


@app.on_event("startup")
def warmup_model() -> None:
    """Précharge le modèle au démarrage pour éviter la latence du 1er /score."""
    if not settings.preload_model:
        logging.info("Model warmup disabled (MATCHING_PRELOAD_MODEL=0)")
        return

    started = time.perf_counter()
    try:
        model = get_model()
        # Un encode "à vide" force l'allocation des tenseurs et le JIT éventuel.
        model.encode(["warmup"], batch_size=1, convert_to_numpy=True, normalize_embeddings=True)
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        logging.info("Model preloaded and warmed in %d ms", elapsed_ms)
    except Exception as exc:  # noqa: BLE001
        logging.warning("Model warmup failed: %s", exc)


# ---------------------------------------------------------------------------
# Schémas d'entrée / sortie (Pydantic)
# ---------------------------------------------------------------------------


class JobPayload(BaseModel):
    """Représentation d'une offre d'emploi envoyée par WordPress via n8n."""

    id: int
    title: str
    description: str = ""
    content: str = ""
    excerpt: str = ""
    keywords: Optional[Union[str, List[str]]] = Field(default=None, description="Liste ou chaîne de mots-clés")
    location: Optional[str] = None
    meta: Optional[dict] = None


class CvPayload(BaseModel):
    """Représentation d'un CV candidat. `extra='allow'` : on tolère les champs
    additionnels envoyés par le plugin (variables selon la version WP)."""

    model_config = ConfigDict(extra="allow")

    id: int = Field(..., description="Identifiant interne du CV")
    candidate_email: Optional[str] = None
    application_title: Optional[str] = None
    title: Optional[str] = None
    resume: Optional[str] = None
    text_content: Optional[str] = None
    skills: Optional[str] = None
    keywords: Optional[Union[str, List[str]]] = None
    metadata: Optional[dict] = None
    file_path: Optional[str] = None
    location: Optional[str] = None


class ScoreRequest(BaseModel):
    """Corps de la requête POST /score."""

    job: JobPayload
    cvs: List[CvPayload]


class ScoreItem(BaseModel):
    """Résultat individuel pour un CV : score + explications."""

    cv_id: int
    score: float
    strengths: List[str] = Field(default_factory=list)
    weaknesses: List[str] = Field(default_factory=list)
    keywords: List[str] = Field(default_factory=list)
    extra: dict = Field(default_factory=dict)


class ScoreResponse(BaseModel):
    """Corps de la réponse POST /score : job traité + résultats classés."""

    job_id: int
    count: int
    duration_ms: int
    results: List[ScoreItem]


# ---------------------------------------------------------------------------
# Structures internes préparées (job/CV normalisés en amont du scoring)
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class PreparedJob:
    """Offre prête à scorer : texte concaténé, tokens, mots-clés, méta parsées."""

    text: str
    tokens: set[str]
    keywords: List[str]
    keyword_set: set[str]
    location: str
    category: str
    jobtype: str
    min_experience_years: Optional[float]
    salary_min: Optional[float]
    salary_max: Optional[float]


@dataclass(slots=True)
class PreparedCv:
    """CV prêt à scorer : texte assemblé + signaux structurés pour comparaison."""

    payload: CvPayload
    text: str
    title_tokens: set[str]
    keywords: List[str]
    keyword_set: set[str]
    location: str
    category: str
    jobtype: str
    experience_years: Optional[float]
    salary_expected_min: Optional[float]
    salary_expected_max: Optional[float]
    qualified: Optional[bool]


# ---------------------------------------------------------------------------
# Auth & chargement modèle
# ---------------------------------------------------------------------------


def require_api_key(x_api_key: str = Header(default="")) -> None:
    """Dépendance FastAPI : refuse l'appel si l'en-tête X-API-Key est absent
    ou ne correspond pas à MATCHING_API_KEY (comparaison constant-time)."""
    expected = os.getenv("MATCHING_API_KEY", "")
    if not expected or not secrets.compare_digest(x_api_key, expected):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")


def get_model() -> SentenceTransformer:
    """Retourne le modèle sentence-transformer (lazy-load, thread-safe)."""
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                logging.info("Loading sentence-transformer model %s", settings.sentence_model)
                _model = SentenceTransformer(settings.sentence_model)
    return _model


# ---------------------------------------------------------------------------
# Fonctions utilitaires : normalisation, parsing, tokenisation
# ---------------------------------------------------------------------------


def normalize_whitespace(value: str) -> str:
    """Compresse tout blanc (espaces, tabs, retours ligne) en un seul espace."""
    return re.sub(r"\s+", " ", value).strip()


def tokenize(text: str) -> set[str]:
    """Extrait un set de tokens lowercase, hors stopwords et mots de ≤2 lettres."""
    tokens: set[str] = set()
    for match in WORD_PATTERN.finditer(text.lower()):
        word = match.group()
        if len(word) <= 2 or word in DEFAULT_STOPWORDS:
            continue
        tokens.add(word)
    return tokens


def parse_keywords(raw: Optional[Union[str, Sequence[str]]]) -> List[str]:
    """Convertit une chaîne "a, b; c" ou une liste en liste dédoublonnée
    de mots-clés lowercase, en écartant les entrées ≤2 caractères."""
    if not raw:
        return []

    if isinstance(raw, str):
        candidates: Iterable[str] = re.split(r"[,;/\n]+", raw)
    else:
        candidates = raw

    seen: set[str] = set()
    keywords: List[str] = []
    for candidate in candidates:
        cleaned = normalize_whitespace(str(candidate)).lower()
        if not cleaned or len(cleaned) <= 2:
            continue
        if cleaned in seen:
            continue
        seen.add(cleaned)
        keywords.append(cleaned)
    return keywords


def normalized_text(value: Optional[object]) -> str:
    """Convertit n'importe quelle valeur en texte lowercase compacté."""
    if value is None:
        return ""
    return normalize_whitespace(str(value)).lower()


def parse_float(value: Optional[object]) -> Optional[float]:
    """Extrait le premier nombre d'une chaîne ("5 ans", "3.5k", ...) -> float."""
    if value is None:
        return None
    text = str(value).strip().replace(",", ".")
    if not text:
        return None
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def parse_range(value: Optional[object]) -> Tuple[Optional[float], Optional[float]]:
    """Extrait (min, max) depuis une chaîne comme "45-60k" ou "500 EUR/j"."""
    if value is None:
        return None, None
    text = str(value).strip().replace(",", ".")
    if not text:
        return None, None
    numbers = [float(m) for m in re.findall(r"\d+(?:\.\d+)?", text)]
    if not numbers:
        return None, None
    if len(numbers) == 1:
        return numbers[0], numbers[0]
    return min(numbers), max(numbers)


def parse_boolish(value: Optional[object]) -> Optional[bool]:
    """Interprète oui/non/true/false/1/0/qualifié/ok en bool ; None si inconnu."""
    if value is None:
        return None
    text = normalized_text(value)
    if not text:
        return None
    yes_values = {"yes", "oui", "true", "1", "qualifie", "qualifié", "ok"}
    no_values = {"no", "non", "false", "0", "ko"}
    if text in yes_values:
        return True
    if text in no_values:
        return False
    return None


def find_first(source: dict, keys: Sequence[str]) -> Optional[object]:
    """Retourne la 1re valeur non vide parmi une liste de clés candidates.
    Sert à absorber les variations de noms de champs entre plugins WP."""
    for key in keys:
        if key in source and source.get(key) not in (None, ""):
            return source.get(key)
    return None


def overlap_text(a: str, b: str) -> bool:
    """Vrai si a et b se chevauchent : inclusion directe ou tokens communs."""
    if not a or not b:
        return False
    if a in b or b in a:
        return True
    a_tokens = set(tokenize(a))
    b_tokens = set(tokenize(b))
    return bool(a_tokens.intersection(b_tokens))


# ---------------------------------------------------------------------------
# Extraction de texte depuis les fichiers CV (PDF, DOC, images OCR...)
# ---------------------------------------------------------------------------


def resolve_file_path(raw_path: str) -> Optional[Path]:
    """Résout un chemin CV : tel quel, sinon relatif à DATA_DIR."""
    candidate = Path(raw_path)
    if candidate.is_file():
        return candidate
    nested = settings.data_dir / raw_path
    if nested.is_file():
        return nested
    return None


def text_from_file(path: Path) -> str:
    """Extrait le texte d'un fichier CV.

    Stratégie : Apache Tika en premier (PDF, DOCX, ODT, ...). Si Tika renvoie
    vide et que le fichier est une image, on tombe sur OCR (Tesseract) FR+EN.
    """
    try:
        parsed = parser.from_file(str(path))
        content = parsed.get("content") or ""
        if content.strip():
            return normalize_whitespace(content)
    except Exception as exc:  # noqa: BLE001
        logging.warning("Tika parsing failed for %s: %s", path, exc)

    if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".tiff", ".bmp"}:
        try:
            with Image.open(path) as img:
                content = pytesseract.image_to_string(img, lang="fra+eng")
                if content.strip():
                    return normalize_whitespace(content)
        except Exception as exc:  # noqa: BLE001
            logging.warning("OCR parsing failed for %s: %s", path, exc)

    return ""


def assemble_cv_text(cv: CvPayload) -> str:
    """Construit le texte à vectoriser pour un CV.

    Ordre de priorité :
      1. Champs texte fournis par le plugin (text_content, resume, skills)
         + métadonnées libres (summary, experience, notes).
      2. Sinon, extraction depuis le fichier attaché (Tika / OCR).
    """
    parts: List[str] = []
    for value in [cv.text_content, cv.resume, cv.skills]:
        if value:
            parts.append(value)

    meta = cv.metadata or {}
    for key in ("summary", "experience", "notes", "resume"):
        value = meta.get(key)
        if isinstance(value, str):
            parts.append(value)

    combined = normalize_whitespace(" ".join(parts))
    if combined:
        return combined

    file_candidate = cv.file_path or meta.get("file_path")
    if file_candidate:
        resolved = resolve_file_path(file_candidate)
        if resolved:
            file_text = text_from_file(resolved)
            if file_text:
                return file_text

    return ""


# ---------------------------------------------------------------------------
# Préparation offre / CV : normalisation et extraction des signaux structurés
# ---------------------------------------------------------------------------


def prepare_job(job: JobPayload) -> PreparedJob:
    """Normalise l'offre : concat texte, extraction jobtype/catégorie/salaire..."""
    meta = job.meta or {}
    keywords = parse_keywords(job.keywords) + parse_keywords(meta.get("skills"))
    keywords = list(dict.fromkeys(keywords))
    text_parts = [job.title, job.description, job.content, job.excerpt, " ".join(keywords), meta.get("experience", "")]
    text = normalize_whitespace(" ".join(filter(None, text_parts)))
    location = (job.location or meta.get("location") or "").lower()
    tokens = tokenize(f"{job.title} {job.description}")
    category = normalized_text(find_first(meta, ["jobcategory_text", "category_text", "job_category", "category"]))
    jobtype = normalized_text(find_first(meta, ["jobtype_text", "job_type", "type", "jobtype"]))
    min_experience_years = parse_float(find_first(meta, ["experience", "min_experience", "required_experience"]))

    salary_min = parse_float(find_first(meta, ["salaryfrom", "salary_min", "salary_from"]))
    salary_max = parse_float(find_first(meta, ["salaryto", "salary_max", "salary_to", "tjm", "salary"]))

    return PreparedJob(
        text=text,
        tokens=tokens,
        keywords=keywords,
        keyword_set=set(keywords),
        location=location,
        category=category,
        jobtype=jobtype,
        min_experience_years=min_experience_years,
        salary_min=salary_min,
        salary_max=salary_max,
    )


def prepare_cv(cv: CvPayload) -> PreparedCv:
    """Normalise un CV : texte assemblé + signaux structurés côté candidat."""
    meta = cv.metadata or {}
    raw = cv.model_dump()
    text = assemble_cv_text(cv)
    title_tokens = tokenize(" ".join(filter(None, [cv.application_title, cv.title])))
    keywords = parse_keywords(cv.keywords) + parse_keywords(meta.get("keywords")) + parse_keywords(cv.skills)
    keywords = list(dict.fromkeys(keywords))
    location = (cv.location or meta.get("location") or "").lower()

    # Chaque champ est cherché à la fois au niveau racine du payload et dans meta,
    # avec plusieurs alias possibles selon la version du plugin.
    category = normalized_text(
        find_first(raw, ["category_text", "category", "job_category"]) or find_first(meta, ["category_text", "category", "job_category"])
    )
    jobtype = normalized_text(
        find_first(raw, ["job_type", "type", "jobtype", "jobtype_text"]) or find_first(meta, ["job_type", "type", "jobtype", "jobtype_text"])
    )
    experience_years = parse_float(
        find_first(raw, ["total_experience", "experience", "experience_years"]) or find_first(meta, ["total_experience", "experience", "experience_years"])
    )
    salary_min, salary_max = parse_range(
        find_first(raw, ["salary_requested", "salary", "salary_range", "expected_salary", "tjm"]) or find_first(meta, ["salary_requested", "salary", "salary_range", "expected_salary", "tjm"])
    )
    qualified = parse_boolish(
        find_first(raw, ["qualified_for_job", "qualified", "qualifie", "qualifié"]) or find_first(meta, ["qualified_for_job", "qualified", "qualifie", "qualifié"])
    )

    return PreparedCv(
        payload=cv,
        text=text,
        title_tokens=title_tokens,
        keywords=keywords,
        keyword_set=set(keywords),
        location=location,
        category=category,
        jobtype=jobtype,
        experience_years=experience_years,
        salary_expected_min=salary_min,
        salary_expected_max=salary_max,
        qualified=qualified,
    )


# ---------------------------------------------------------------------------
# Filtres durs et scoring
# ---------------------------------------------------------------------------


def passes_hard_filters(job: PreparedJob, cv: PreparedCv) -> Tuple[bool, List[str]]:
    """Applique les filtres éliminatoires (jamais scorés s'ils échouent).

    Filtres actuels : non qualifié explicite, type de contrat incompatible,
    déficit d'expérience > 0.5 an sous le seuil requis.
    """
    reasons: List[str] = []

    if settings.hard_filter_qualification and cv.qualified is False:
        reasons.append("Candidat non qualifié pour le poste")

    if settings.hard_filter_jobtype and job.jobtype and cv.jobtype and not overlap_text(job.jobtype, cv.jobtype):
        reasons.append("Type de contrat incompatible")

    if job.min_experience_years is not None and cv.experience_years is not None:
        if cv.experience_years + 0.5 < job.min_experience_years:
            reasons.append("Expérience insuffisante")

    return len(reasons) == 0, reasons


def compute_experience_adjustment(min_years: float, cv_years: float, weight: float) -> Tuple[float, float]:
    """Retourne (bonus, malus) d'expérience avec une échelle progressive.

    Au seuil requis : bonus = 50 % du poids ; puis monte jusqu'à 100 % du poids
    quand le profil dépasse largement le seuil. En-dessous : malus progressif
    entre 50 % et 100 % du poids selon l'écart relatif.
    """
    gap = cv_years - min_years
    denominator = max(1.0, min_years)

    if gap >= 0:
        ratio = min(1.0, 0.5 + (gap / denominator) * 0.5)
        return weight * ratio, 0.0

    deficit_ratio = min(1.0, abs(gap) / denominator)
    return 0.0, weight * (0.5 + 0.5 * deficit_ratio)


def encode_texts(texts: Sequence[str]) -> np.ndarray:
    """Vectorise une liste de textes en embeddings normalisés float32."""
    model = get_model()
    embeddings = model.encode(list(texts), batch_size=settings.embed_batch_size, convert_to_numpy=True, normalize_embeddings=True)
    if not isinstance(embeddings, np.ndarray):
        embeddings = np.asarray(embeddings)
    return embeddings.astype("float32")


def search_top_matches(job_embedding: np.ndarray, cv_embeddings: np.ndarray, limit: int) -> Tuple[np.ndarray, np.ndarray]:
    """Retourne (indices, scores) des top-`limit` CV les plus proches de l'offre.
    Utilise FAISS IndexFlatIP : produit scalaire = cosinus car vecteurs normalisés."""
    limit = max(1, min(limit, len(cv_embeddings)))
    index = faiss.IndexFlatIP(cv_embeddings.shape[1])
    index.add(cv_embeddings)
    scores, indices = index.search(np.expand_dims(job_embedding, axis=0), limit)
    return indices[0], scores[0]


def build_score(job: PreparedJob, cv: PreparedCv, similarity: float, rank: int) -> ScoreItem:
    """Construit le ScoreItem final pour un CV : score borné [0, 100] + explications.

    Composition du score :
      - base sémantique   : similarité cosinus x 70 (contribution max ~70 pts)
      - bonus mots-clés   : +N x keyword_weight par mot-clé commun
      - bonus titre       : +title_weight si tokens du titre en commun
      - bonus localisation: +location_weight si compatibilité géographique
      - bonus structure   : jobtype / catégorie / expérience / salaire compatibles
      - malus structure   : non qualifié, contrat incompatible, sous-expérience,
                            prétention salariale au-dessus du budget
    Les listes strengths / weaknesses fournissent les explications lisibles
    utilisées par le workflow n8n pour l'affichage côté WordPress.
    """
    strengths: List[str] = []
    weaknesses: List[str] = []

    keyword_hits = sorted(job.keyword_set.intersection(cv.keyword_set))
    if keyword_hits:
        strengths.append(f"Mots-clés ({', '.join(keyword_hits[:5])})")
    else:
        weaknesses.append("Aucun mot-clé commun identifié")

    title_overlap = job.tokens.intersection(cv.title_tokens)
    if title_overlap:
        strengths.append(f"Titre proche ({', '.join(sorted(title_overlap)[:3])})")
    else:
        weaknesses.append("Titre éloigné du besoin")

    structure_bonus = 0.0
    structure_penalty = 0.0

    if cv.qualified is True:
        strengths.append("Profil déclaré qualifié pour le poste")
    elif cv.qualified is False:
        weaknesses.append("Profil déclaré non qualifié")
        structure_penalty += settings.qualification_penalty

    if job.jobtype and cv.jobtype:
        if overlap_text(job.jobtype, cv.jobtype):
            strengths.append("Type de contrat compatible")
            structure_bonus += settings.jobtype_weight
        else:
            weaknesses.append("Type de contrat différent")
            structure_penalty += settings.jobtype_weight

    if job.category and cv.category:
        if overlap_text(job.category, cv.category):
            strengths.append("Catégorie métier alignée")
            structure_bonus += settings.category_weight
        else:
            weaknesses.append("Catégorie métier différente")
            # Malus catégorie divisé par 2 : signal moins déterminant que le jobtype.
            structure_penalty += settings.category_weight / 2

    if job.min_experience_years is not None and cv.experience_years is not None:
        experience_bonus, experience_penalty = compute_experience_adjustment(
            job.min_experience_years,
            cv.experience_years,
            settings.experience_weight,
        )
        if cv.experience_years >= job.min_experience_years:
            strengths.append(f"Expérience suffisante ({cv.experience_years:g} ans)")
        else:
            weaknesses.append(f"Expérience inférieure ({cv.experience_years:g} ans)")
        structure_bonus += experience_bonus
        structure_penalty += experience_penalty

    if job.salary_max is not None and cv.salary_expected_min is not None:
        if cv.salary_expected_min <= job.salary_max:
            strengths.append("Prétention salariale compatible")
            structure_bonus += settings.salary_weight
        else:
            weaknesses.append("Prétention salariale au-dessus du budget")
            structure_penalty += settings.salary_weight

    location_bonus = 0.0
    if job.location and cv.location:
        if job.location in cv.location or cv.location in job.location:
            strengths.append("Localisation compatible")
            location_bonus = settings.location_weight
        else:
            weaknesses.append("Localisation différente")

    # Score final : base sémantique + bonus/malus, borné à [0, 100].
    base_score = max(0.0, similarity) * 70
    score = base_score + len(keyword_hits) * settings.keyword_weight
    if title_overlap:
        score += settings.title_weight
    score += location_bonus
    score += structure_bonus
    score -= structure_penalty
    score = float(max(0.0, min(100.0, score)))

    extra = {
        "vector_similarity": round(float(similarity), 4),
        "keyword_hits": keyword_hits,
        "rank": rank + 1,
        "cv_category": cv.category,
        "cv_jobtype": cv.jobtype,
        "cv_experience_years": cv.experience_years,
        "cv_salary_min": cv.salary_expected_min,
        "cv_salary_max": cv.salary_expected_max,
        "cv_qualified": cv.qualified,
        "score_breakdown": {
            "semantic": round(base_score, 4),
            "keyword_bonus": round(len(keyword_hits) * settings.keyword_weight, 4),
            "title_bonus": round(settings.title_weight if title_overlap else 0.0, 4),
            "location_bonus": round(location_bonus, 4),
            "structure_bonus": round(structure_bonus, 4),
            "structure_penalty": round(structure_penalty, 4),
        },
    }

    return ScoreItem(
        cv_id=cv.payload.id,
        score=round(score, 2),
        strengths=strengths,
        weaknesses=weaknesses,
        keywords=keyword_hits,
        extra=extra,
    )


# ---------------------------------------------------------------------------
# Endpoints HTTP
# ---------------------------------------------------------------------------


@app.get("/health")
def healthcheck() -> dict:
    """Sonde de vie utilisée par n8n, le healthcheck Docker et check_stack.sh."""
    return {
        "status": "ok",
        "model": settings.sentence_model,
        "model_loaded": _model is not None,
        "model_cache": os.getenv("MODEL_CACHE", "/models"),
        "data_dir": str(settings.data_dir),
        "time": time.time(),
    }


@app.post("/score", response_model=ScoreResponse)
def score(payload: ScoreRequest, _: None = Depends(require_api_key)) -> ScoreResponse:
    """Endpoint principal : reçoit offre + CV, renvoie le classement scoré.

    Enchaînement :
      1. Préparation job + CV, filtrage des CV sans texte exploitable.
      2. Application des filtres durs. Si tous les CV sont éliminés, on
         retombe sur le set brut pour ne jamais renvoyer une réponse vide.
      3. Vectorisation (batch) + recherche FAISS top_k.
      4. Filtrage par similarité minimale + construction des ScoreItem.
         Si aucun CV ne dépasse le seuil, on garde le meilleur candidat
         (fallback : mieux vaut un résultat faible que rien).
      5. Tri déterministe final : score décroissant, puis similarité, puis
         expérience, puis cv_id (ordre stable pour tests / rejeu).
    """
    if not payload.cvs:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Aucun CV fourni")

    started = time.perf_counter()

    job = prepare_job(payload.job)
    prepared_cvs = [prepare_cv(cv) for cv in payload.cvs]
    usable = [cv for cv in prepared_cvs if cv.text]

    if not usable:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Aucun texte exploitable pour les CV")

    filtered_usable: List[PreparedCv] = []
    for cv in usable:
        ok, _ = passes_hard_filters(job, cv)
        if ok:
            filtered_usable.append(cv)

    # Filet de sécurité : si les filtres durs éliminent tout, on scorе quand
    # même sur le pool complet pour retourner un classement dégradé plutôt
    # que vide (le workflow n8n s'attend toujours à au moins un résultat).
    candidates = filtered_usable if filtered_usable else usable

    job_embedding = encode_texts([job.text])[0]
    cv_embeddings = encode_texts([cv.text for cv in candidates])

    indices, similarities = search_top_matches(job_embedding, cv_embeddings, settings.top_k)

    scored_items: List[ScoreItem] = []
    for rank, (idx, sim) in enumerate(zip(indices, similarities)):
        if idx < 0:
            continue
        if sim < settings.min_similarity:
            continue
        scored_items.append(build_score(job, candidates[idx], sim, rank))

    # Fallback : si aucun CV ne passe le seuil de similarité, on garde le
    # meilleur pour éviter une réponse vide côté WordPress.
    if not scored_items and len(indices):
        idx = int(indices[0])
        if idx >= 0:
            scored_items.append(build_score(job, candidates[idx], float(similarities[0]), 0))

    # Le classement final suit le score total (pas le rang brut d'embedding),
    # avec des critères secondaires déterministes pour un ordre stable.
    scored_items.sort(
        key=lambda item: (
            -float(item.score),
            -float(item.extra.get("vector_similarity", 0.0)),
            -float(item.extra.get("cv_experience_years") or -1),
            int(item.cv_id),
        )
    )

    for final_rank, item in enumerate(scored_items, start=1):
        item.extra["rank"] = final_rank

    duration_ms = int((time.perf_counter() - started) * 1000)

    return ScoreResponse(job_id=payload.job.id, count=len(scored_items), duration_ms=duration_ms, results=scored_items)
