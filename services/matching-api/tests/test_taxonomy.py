"""Tests de non-régression pour app/taxonomy.py.

Chaque cas ci-dessous correspond à un bug réel trouvé soit directement
chez Keoni, soit chez AI Real-Time sur le même fichier rome_skills_data.json
partagé entre les deux projets (référencé par hash de commit dans les
commentaires) — pas des cas hypothétiques.
"""

from __future__ import annotations

from app.taxonomy import find_skills, normalize_skill


# ── Faux positif "vue" (AI Real-Time 36673c6/69112d5) ──────────────────────


def test_french_word_vue_is_not_vuejs():
    assert find_skills("point de vue") == []
    assert find_skills("en vue des sprints") == []
    assert find_skills("revue de code") == []


def test_real_vuejs_still_detected():
    assert "Vue.js" in find_skills("developpeur vue.js")
    assert "Vue.js" in find_skills("vuejs et react")


# ── Stopword ROME "son" (AI Real-Time cd8e45e, même fichier ROME) ──────────


def test_son_possessive_does_not_match_rome_alias():
    result = find_skills("pour son équipe, c'était un succès collectif")
    assert "Son" not in result
    assert "C" not in result


# ── Vocabulaire professionnel générique exclu (AI Real-Time a4f576f/61886a9,
#    réduit aux libellés confirmés présents dans notre rome_skills_data.json) ──


def test_generic_professional_vocabulary_is_excluded():
    result = find_skills("bonne écoute active et gestion du temps au quotidien")
    assert "Ecoute active" not in result
    assert "Gestion du temps" not in result


# ── "Grande distribution" exclu (trouvé chez Keoni, pas un portage AI
#    Real-Time -- voir le commentaire sur _GENERIC_SKILL_CANONICALS) :
#    boilerplate ESN "secteurs clients", pas une exigence de poste, présent
#    dans 4/5 offres réelles testées et pesant ~25 points de couverture sur
#    un candidat par ailleurs bien aligné (Stéphane Burgevin, 45.5 -> mesure
#    attendue AI Real-Time ~65+ sur le même profil) ──────────────────────


def test_agency_client_sector_boilerplate_is_excluded():
    result = find_skills(
        "nous accompagnons nos clients de l'industrie, banque & assurance, "
        "grande distribution & e-commerce, et médias & communication"
    )
    assert "Grande distribution" not in result


# ── Auto-indexation du canonique retirée (AI Real-Time 2d670be) : vérifie que
#    le matching de base marche toujours sans l'alias implicite ────────────


def test_common_tech_stack_still_detected():
    result = find_skills("MongoDB, MySQL, NodeJS, JavaScript et Docker")
    for skill in ("MongoDB", "MySQL", "Node.js", "JavaScript", "Docker"):
        assert skill in result


# ── Ponctuation en fin de token ("Docker.") ─────────────────────────────────


def test_trailing_punctuation_does_not_break_matching():
    assert "Docker" in find_skills("On utilise Docker.")
    assert "Node.js" in find_skills("Stack : Node.js, React.")


# ── Alias transposé depuis AI Real-Time (4b99ff9) ───────────────────────────


def test_bitbucket_maps_to_git():
    assert find_skills("utilisation quotidienne de bitbucket") == ["Git"]


# ── Synonymes JS/Dev web (raison d'être de la couche _TECH_SKILLS) ─────────


def test_js_and_javascript_are_the_same_canonical():
    assert normalize_skill("JS") == "JavaScript"
    assert normalize_skill("JavaScript") == "JavaScript"


def test_dev_web_and_developpeur_web_are_the_same_canonical():
    assert normalize_skill("Dev web") == "Développeur Web"
    assert normalize_skill("Développeur Web") == "Développeur Web"


# ── Résilience : fichier ROME absent ne doit jamais faire planter le module ──


def test_empty_or_missing_text_returns_empty_list():
    assert find_skills("") == []
    assert find_skills("   ") == []
