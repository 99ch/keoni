"""Lecture structuree du texte reel d'un CV/d'une offre -- portage a
l'identique depuis AI Real-Time (backend/app/services/parser.py), pour que
l'extraction se fasse "exactement comme pour rh console" (exigence 2026-10-06).

Keoni savait deja lire le texte brut d'un fichier (extraction.py, deja porte
depuis AI Real-Time). Ce module ajoute l'etape suivante, qui manquait
entierement : relire ce texte pour en sortir des faits precis -- annees
d'experience, type de contrat, langues, section "formation" -- directement
dans les mots du document, au lieu de s'appuyer sur des champs de formulaire
WordPress remplis par le candidat (voir prepare_cv/prepare_job cote main.py,
qui lisaient jusqu'ici cv.metadata/job.meta pour ces informations).

Verifie avant de porter : ces 4 informations sont produites chez AI
Real-Time par de la detection de sections par mots-cles d'en-tete + des
expressions regulieres -- PAS par le modele de reconnaissance d'entites
(spaCy/NER, qui sert uniquement a person_name/organization_terms/
location_terms/date_terms, jamais utilises par le scoring). Keoni n'a donc
pas besoin d'ajouter spaCy pour ce pilier.

Portage volontairement cible sur le sous-ensemble necessaire a
experience_years/contract_type/language_terms/education_text -- pas le
pipeline complet de parser.py (titre, domaine, assemblage skill_src...),
qui sert a d'autres usages cote AI Real-Time deja couverts autrement chez
Keoni (taxonomy.find_skills).
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime

# Portage verbatim de _fold (parser.py) -- PAS un alias vers
# scoring.fold_text : celle de Keoni ne normalise pas les apostrophes
# typographiques (', ´, `) avant de retirer les accents, ce qui fait
# disparaitre silencieusement l'apostrophe de "d'experience"/"aujourd'hui"
# (U+2019 n'a pas d'equivalent ASCII, NFKD+ascii-ignore la supprime
# purement) -- constate en testant ce module (le cas de reference "25 ans
# d'experience" de rhconsole echouait avec fold_text). La frontiere de mot
# que _extract_years/_detect_contract/_detect_languages attendent a besoin
# de cette normalisation specifique pour matcher exactement comme chez eux.
def _fold(text: str) -> str:
    text = text.replace("’", "'").replace("´", "'").replace("`", "'")
    nfkd = unicodedata.normalize("NFKD", text)
    return nfkd.encode("ascii", "ignore").decode("ascii").lower()


# ── Nettoyage du texte avant decoupage en sections ──────────────────────────
# Portage verbatim de _clean_text (parser.py) : deshyphenation en fin de
# ligne, dedoublonnage des lignes repetees (en-tetes/pieds de page), retrait
# des puces de tete de ligne.

_BOILERPLATE_MIN_LENGTH = 20


def _clean_text(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    text = text.replace("■", "fi")
    text = re.sub(r"(?<=[a-zA-ZÀ-ɏ])\?(?=[a-zA-ZÀ-ɏ])", "ti", text)
    lines: list[str] = []
    seen: dict[str, int] = {}
    prev = ""
    for raw in text.split("\n"):
        line = re.sub(r"­", "", raw)
        line = re.sub(r"\s+", " ", line.strip())
        line = re.sub(r"^[\-*•·◦ü°ð►▪▫●○]+\s*", "", line).strip()
        if not line or len(line) < 2:
            prev = ""
            continue
        key = _fold(line)
        if key == prev:
            continue
        if len(line) > _BOILERPLATE_MIN_LENGTH:
            seen[key] = seen.get(key, 0) + 1
            if seen[key] > 2:
                continue
        lines.append(line)
        prev = key
    return "\n".join(lines).strip()


# ── Detection des en-tetes de section ───────────────────────────────────────
# Portage verbatim de _SECTION_DEFS/_section_lookup/_match_section/
# _is_heading (parser.py) -- la table COMPLETE des 11 sections est gardee
# telle quelle (pas juste education/experience/languages/contract) : le
# classifieur desambigue une ligne en verifiant si ses mots "en trop"
# appartiennent a une AUTRE section connue (ex. "Location de vehicule" doit
# rester non reconnu grace a "location" existant comme section a part
# entiere) -- une table reduite aux 4 sections utiles ici changerait ce
# comportement de desambiguation pour CES sections aussi. build_document_
# profile() ne retourne que education/experience/languages/contract, mais
# le classifieur doit voir le meme univers de sections que rhconsole pour
# etre "exactement" identique (exigence explicite 2026-10-06).

_SECTION_DEFS: list[tuple[str, list[str]]] = [
    ("summary", [
        "profil", "profile", "about", "a propos", "a propos de moi",
        "bio", "biographie", "resume", "summary", "presentation",
        "profil professionnel", "objectif", "objectifs", "who am i",
        "profil du candidat",
    ]),
    ("skills", [
        "competences", "skills", "stack", "stack technique", "technologies",
        "outils", "outillage", "frameworks", "langages", "expertise",
        "hard skills", "soft skills", "aptitudes", "technical skills",
        "savoir faire", "savoir-faire", "connaissances", "maitrise",
    ]),
    ("experience", [
        "experience", "experiences", "work experience", "professional experience",
        "parcours", "missions", "mission", "emploi", "postes occupes",
        "employment history", "career", "parcours professionnel",
        "realisations", "achievements", "portfolio", "postes",
        "experience requise", "experience demandee", "experience souhaitee",
        "experience minimum",
        "references projets", "references significatives",
    ]),
    ("education", [
        "formation", "formations", "education", "etudes", "diplome", "diplomes",
        "diplomes obtenus", "academique", "scolarite", "universite",
        "ecole", "school", "cursus", "parcours academique", "enseignement",
    ]),
    ("certifications", [
        "certification", "certifications", "certificat", "certificats",
        "habilitations", "licences", "certifications professionnelles",
    ]),
    ("languages", [
        "langues", "languages", "language", "langues parlees",
        "language skills", "bilingue", "maitrise des langues",
    ]),
    ("job_required", [
        "competences requises", "exigences", "must have", "prerequis",
        "profil recherche", "profil attendu", "responsabilites",
        "missions principales", "taches", "role", "description du poste",
        "requirements", "qualifications", "ce que nous recherchons",
        "votre profil", "what we are looking for", "profil ideal",
    ]),
    ("job_nice", [
        "nice to have", "atouts", "bonus", "souhaite", "souhaitable",
        "apprecie", "sera apprecie", "un plus", "avantages",
        "ce serait un plus", "optionnel",
    ]),
    ("contract", [
        "contrat", "type de contrat", "statut", "disponibilite",
        "duree", "conditions", "contrat propose",
    ]),
    ("location", [
        "localisation", "location", "ville", "teletravail",
        "remote", "hybride", "mobilite", "lieu de travail",
    ]),
    ("contact", [
        "contact", "coordonnees", "email", "telephone", "adresse",
        "linkedin", "github", "portfolio", "me contacter",
    ]),
    ("hobbies", [
        "interets", "loisirs", "hobbies", "centres d interet",
        "passions", "activites extra",
    ]),
]


def _section_lookup() -> dict[str, str]:
    lookup: dict[str, str] = {}
    for section, aliases in _SECTION_DEFS:
        for alias in aliases:
            lookup[_fold(alias)] = section
    return lookup


_SECTION_LOOKUP = _section_lookup()


def _alias_pattern(alias: str) -> re.Pattern:
    return re.compile(r"\b" + re.escape(alias) + r"\b")


_SECTION_STOPWORDS = {
    "et", "de", "des", "du", "la", "le", "les", "d", "l", "a", "à",
    "en", "pour", "sur", "avec", "au", "aux", "&",
    "domaines", "domaine", "cle", "cles",
    "clef", "clefs", "majeur", "majeure", "majeurs", "majeures",
}

_SECTION_QUALIFIER_SUFFIX_RE = re.compile(
    r".*(?:ique|iques|el|elle|els|elles|al|ale|aux|ales|aire|aires|if|ive|ifs|ives)$"
)

_COMPATIBLE_SECTION_PAIRS: dict[frozenset, str] = {
    frozenset(("education", "certifications")): "education",
}

_SECTION_LIST_INTRO_RE = re.compile(r"^(?:utilise|employe|maitrise|acquis)e?s?$")

_LETTER_SPACED_SUBSTRING_MIN_ALIAS_LEN = 8


def _collapse_letter_spacing(line: str) -> str | None:
    tokens = line.split()
    if len(tokens) < 4:
        return None
    single_letter = sum(1 for t in tokens if len(t) == 1 and t.isalpha())
    if single_letter / len(tokens) < 0.8:
        return None
    return "".join(tokens)


def _match_section(line: str) -> str | None:
    stripped = line.strip()
    lookup = _SECTION_LOOKUP

    collapsed = _collapse_letter_spacing(stripped)
    if collapsed is not None:
        collapsed_folded = _fold(collapsed)
        if collapsed_folded in lookup:
            return lookup[collapsed_folded]
        substring_matches = [
            (alias, section) for alias, section in lookup.items()
            if len(alias) >= _LETTER_SPACED_SUBSTRING_MIN_ALIAS_LEN and alias in collapsed_folded
        ]
        if substring_matches:
            matched_sections = {section for _, section in substring_matches}
            if len(matched_sections) > 1:
                resolved = _COMPATIBLE_SECTION_PAIRS.get(frozenset(matched_sections))
                if resolved is not None:
                    return resolved
            return max(substring_matches, key=lambda m: len(m[0]))[1]

    folded = re.sub(r"[^\w\s]", " ", _fold(stripped))
    words = folded.split()
    if not words or len(words) > 6:
        return None

    if folded in lookup:
        return lookup[folded]
    for alias, section in lookup.items():
        if len(alias) < 4 or not _alias_pattern(alias).search(folded):
            continue
        alias_words = set(alias.split())
        extra = [w for w in words if w not in alias_words and w not in _SECTION_STOPWORDS]
        ok = True
        resolved_section = section
        for w in extra:
            other_section = lookup.get(w)
            if other_section is not None:
                if other_section == section:
                    continue
                pair = frozenset((section, other_section))
                if pair in _COMPATIBLE_SECTION_PAIRS:
                    resolved_section = _COMPATIBLE_SECTION_PAIRS[pair]
                    continue
                ok = False
                break
            if _SECTION_LIST_INTRO_RE.match(w) or not _SECTION_QUALIFIER_SUFFIX_RE.match(w):
                ok = False
                break
        if ok:
            return resolved_section
    return None


def _is_heading(line: str, next_line: str | None) -> bool:
    s = line.strip()
    if not s:
        return False
    if s.endswith(":") and len(s) < 100:
        return True
    if _match_section(s):
        return True
    folded = _fold(s)
    words = folded.split()
    if not words:
        return False
    letters = [c for c in s if c.isalpha()]
    if letters:
        upper = sum(1 for c in letters if c.isupper()) / len(letters)
        if upper > 0.65 and 1 <= len(words) <= 6:
            return True
    if next_line and re.match(r"^\s*[-•*•\d]\s+", next_line):
        if 1 <= len(words) <= 6:
            return True
    return False


def _split_sections(cleaned: str) -> dict[str, str]:
    """Decoupe le texte deja nettoye en sections reconnues. Portage verbatim
    de la boucle de decoupage de parse_document() (parser.py)."""
    lines = [ln for ln in cleaned.split("\n") if ln.strip()]
    sections: dict[str, list[str]] = {}
    current = "other"
    for idx, line in enumerate(lines):
        next_line = lines[idx + 1] if idx + 1 < len(lines) else None

        if ":" in line and len(line) < 100:
            head, _, payload = line.partition(":")
            section = _match_section(head.strip())
            if section:
                if section == "skills" and current == "experience" and payload.strip():
                    sections.setdefault(section, []).append(payload.strip())
                else:
                    current = section
                    if payload.strip():
                        sections.setdefault(current, []).append(payload.strip())
                continue

        section = _match_section(line)
        if section and _is_heading(line, next_line):
            current = section
            continue

        if _is_heading(line, next_line) and not _match_section(line):
            continue

        sections.setdefault(current, []).append(line)

    return {name: "\n".join(content).strip() for name, content in sections.items()}


# ── Annees d'experience ──────────────────────────────────────────────────────
# Portage verbatim de _extract_years/_explicit_years_statement/
# _years_from_date_ranges/_merge_intervals (parser.py).

_YEAR_CTX_RE = re.compile(
    r"(\d{1,2})\s*\+?\s*(?:years?|ans?|ann[eé]e?s?)\s+d[e']\s*(?:exp[eé]rience|exp\b)"
    r"|(?:exp[eé]rience|exp)\s+(?:de\s+)?(\d{1,2})\s*\+?\s*(?:years?|ans?|ann[eé]e?s?)"
    r"|depuis\s+(\d{1,2})\s*\+?\s*(?:years?|ans?|ann[eé]e?s?)"
    r"|\+\s*(\d{1,2})\s*(?:years?|ans?|ann[eé]e?s?)",
    re.IGNORECASE,
)
_YEAR_PLAIN_RE = re.compile(r"(\d{1,2})\s*\+?\s*(?:years?|ans?|ann[eé]e?s?)", re.IGNORECASE)
_AGE_CTX_RE = re.compile(r"\bne\b.{0,20}\d{4}|\bnaissance\b|\bage\s*[:\-]?\s*\d{1,2}\b", re.IGNORECASE)
_EXP_CTX_RE = re.compile(r"exp[eé]rience|exp\b|pratique", re.IGNORECASE)

_DASH = r"[-–—]"
_APOS = r"['’´`]?"
_ONGOING = rf"(?:à ce jour|a ce jour|aujourd{_APOS}hui|pr[eé]sent|actuel(?:le)?|en cours|now)"
_MONTH_NAME = (
    r"(?:jan(?:vier)?|f[eé]v(?:rier)?|mars|avr(?:il)?|mai|juin|juil(?:let)?|"
    r"ao[uû]t|sept?(?:embre)?|oct(?:obre)?|nov(?:embre)?|d[eé]c(?:embre)?|"
    r"january|february|march|april|may|june|july|august|september|october|november|december)"
)
_MONTH_PREFIX = rf"(?:{_MONTH_NAME}\.?\s*|\d{{1,2}}\s*/\s*)?"
_YEAR_RANGE_RE = re.compile(
    rf"\b{_MONTH_PREFIX}(19[7-9]\d|20\d\d)\s*(?:{_DASH}|au|to|\bà\b)\s*"
    rf"{_MONTH_PREFIX}(19[7-9]\d|20\d\d|{_ONGOING})",
    re.IGNORECASE,
)
_SINCE_RE = re.compile(rf"\bdepuis\s+{_MONTH_PREFIX}(19[7-9]\d|20\d\d)\b", re.IGNORECASE)
_ONGOING_RE = re.compile(_ONGOING, re.IGNORECASE)
_SPLIT_RANGE_START_RE = re.compile(
    rf"\b(?:de|du|depuis|d['’´`])\s*{_MONTH_NAME}\.?\s*(19[7-9]\d|20\d\d)\s*(?:à|au)\s*$",
    re.IGNORECASE,
)
_SPLIT_RANGE_END_RE = re.compile(
    rf"^\s*{_MONTH_NAME}\.?\s*(19[7-9]\d|20\d\d)\s*$", re.IGNORECASE
)


def _merge_intervals(intervals: list[tuple[int, int]]) -> int:
    if not intervals:
        return 0
    ordered = sorted(intervals)
    merged: list[list[int]] = [list(ordered[0])]
    for start, end in ordered[1:]:
        last = merged[-1]
        if start <= last[1] + 1:
            last[1] = max(last[1], end)
        else:
            merged.append([start, end])
    return sum(end - start for start, end in merged)


def _years_from_date_ranges(text: str) -> int:
    current_year = datetime.now().year
    intervals: list[tuple[int, int]] = []

    for m in _YEAR_RANGE_RE.finditer(text):
        start = int(m.group(1))
        end_raw = m.group(2).strip()
        if _ONGOING_RE.fullmatch(end_raw):
            end = current_year
        else:
            end = int(end_raw)
        if end < start or start > current_year or end > current_year:
            continue
        intervals.append((start, end))

    for m in _SINCE_RE.finditer(text):
        start = int(m.group(1))
        if start <= current_year:
            intervals.append((start, current_year))

    lines = text.split("\n")
    for i, line in enumerate(lines):
        m_start = _SPLIT_RANGE_START_RE.search(line.strip())
        if not m_start:
            continue
        start = int(m_start.group(1))
        for candidate in lines[i + 1: i + 4]:
            candidate = candidate.strip()
            m_end = _SPLIT_RANGE_END_RE.match(candidate)
            if m_end:
                end = int(m_end.group(1))
                if start <= end <= current_year:
                    intervals.append((start, end))
                break
            if _ONGOING_RE.fullmatch(candidate):
                intervals.append((start, current_year))
                break

    total = _merge_intervals(intervals)
    return total if 1 <= total <= 40 else 0


def _explicit_years_statement(text: str) -> int:
    if not text:
        return 0
    folded = _fold(text)
    for m in _YEAR_CTX_RE.finditer(folded):
        groups = [g for g in m.groups() if g and g.isdigit()]
        if groups:
            v = int(groups[0])
            if 1 <= v <= 40:
                return v
    return 0


def _extract_years(text: str) -> int:
    if not text:
        return 0
    folded = _fold(text)

    explicit = _explicit_years_statement(text)
    if explicit:
        return explicit

    from_dates = _years_from_date_ranges(text)
    if from_dates:
        return from_dates

    safe_lines = []
    for line in folded.split("\n"):
        if re.fullmatch(r"\d{1,2}\s+ans?\.?", line.strip()):
            continue
        if _AGE_CTX_RE.search(line) and not _EXP_CTX_RE.search(line):
            continue
        safe_lines.append(line)
    values = [
        int(m) for m in _YEAR_PLAIN_RE.findall("\n".join(safe_lines))
        if m.isdigit() and 1 <= int(m) <= 40
    ]
    return max(values) if values else 0


# ── Langues ──────────────────────────────────────────────────────────────────
# Portage verbatim de _LANG_PATTERNS/_detect_languages (parser.py).

_LANG_PATTERNS: list[tuple[str, str]] = [
    ("français", r"\bfran[cç]ais\b|\bfrench\b"),
    ("anglais", r"\banglais\b|\benglish\b|\bbilingue anglais\b"),
    ("espagnol", r"\bespagnol\b|\bspanish\b"),
    ("allemand", r"\ballemand\b|\bgerman\b|\bdeutsch\b"),
    ("arabe", r"\barabe\b|\barabic\b"),
    ("portugais", r"\bportugais\b|\bportuguese\b"),
    ("italien", r"\bitalien\b|\bitaliano\b|\bitalian\b"),
    ("mandarin", r"\bchinois\b|\bmandarin\b|\bchinese\b"),
]


def _detect_languages(text: str) -> list[str]:
    folded = _fold(text)
    found: list[str] = []
    seen: set[str] = set()
    for lang, pattern in _LANG_PATTERNS:
        if re.search(pattern, folded) and lang not in seen:
            seen.add(lang)
            found.append(lang)
    return found


# ── Type de contrat ──────────────────────────────────────────────────────────
# Portage verbatim de _CONTRACT_PATTERNS/_detect_contract (parser.py).

_CONTRACT_PATTERNS: list[tuple[str, str]] = [
    ("CDI", r"\bcdi\b"),
    ("CDD", r"\bcdd\b"),
    ("Freelance", r"\bfreelance\b|\bindependant\b|\bconsultant independant\b"),
    ("Stage", r"\bstage\b|\binternship\b|\bstagiaire\b"),
    ("Alternance", r"\balternance\b|\bapprentissage\b|\bcontrat d apprentissage\b"),
    ("Temps plein", r"\btemps plein\b|\bfull.?time\b"),
    ("Temps partiel", r"\btemps partiel\b|\bpart.?time\b"),
    ("Télétravail", r"\bteletravail\b|\bremote\b|\bfull remote\b"),
]


def _detect_contract(text: str) -> str | None:
    folded = _fold(text)
    for label, pattern in _CONTRACT_PATTERNS:
        if re.search(pattern, folded):
            return label
    return None


# ── Point d'entree ───────────────────────────────────────────────────────────

@dataclass(slots=True)
class DocumentProfile:
    education_text: str = ""
    experience_text: str = ""
    languages_text: str = ""
    contract_text: str = ""
    experience_years: int = 0
    language_terms: list[str] = field(default_factory=list)
    contract_type: str | None = None


def build_document_profile(text: str) -> DocumentProfile:
    """Relit le texte reel (deja extrait du fichier) pour en sortir des
    faits precis -- memes regles qu'AI Real-Time (structured.py/parser.py).
    Remplace l'usage des champs de formulaire WordPress pour ces 4
    informations cote CV comme cote offre."""
    if not text:
        return DocumentProfile()

    cleaned = _clean_text(text)
    sections = _split_sections(cleaned)

    education_text = sections.get("education", "")
    experience_text = sections.get("experience", "")
    languages_text = sections.get("languages", "")
    contract_text = sections.get("contract", "")

    lang_src = "\n".join(p for p in [languages_text, cleaned[:2000]] if p)
    language_terms = _detect_languages(lang_src)

    contract_src = "\n".join(p for p in [contract_text, cleaned[:2000]] if p)
    contract_type = _detect_contract(contract_src)

    explicit_years = _explicit_years_statement(cleaned)
    if explicit_years:
        experience_years = explicit_years
    else:
        exp_src = experience_text or cleaned
        experience_years = _extract_years(exp_src)

    return DocumentProfile(
        education_text=education_text,
        experience_text=experience_text,
        languages_text=languages_text,
        contract_text=contract_text,
        experience_years=experience_years,
        language_terms=language_terms,
        contract_type=contract_type,
    )
