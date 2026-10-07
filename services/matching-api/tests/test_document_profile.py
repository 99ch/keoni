"""Tests de non-regression pour document_profile.py -- portage depuis AI
Real-Time (backend/tests/test_experience_extraction.py et
test_section_heading_detection.py), exigence 2026-10-06 : l'extraction CV/
offre doit se faire exactement comme rh console.

Memes cas de reference que cote AI Real-Time, adaptes a l'API de ce module
(build_document_profile() plutot que parse_document(), _extract_years
expose directement comme cote rhconsole).
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.document_profile import (  # noqa: E402
    _extract_years,
    _match_section,
    build_document_profile,
)

CURRENT_YEAR = datetime.now().year


# ── _match_section : desambiguation des en-tetes ─────────────────────────────


def test_alias_glued_inside_unrelated_word_is_not_a_heading():
    assert _match_section("Controle de gestion et suivi budgetaire mensuel.") is None


def test_long_sentence_containing_an_alias_word_is_not_a_heading():
    assert _match_section(
        "J'ai acquis une solide experience en gestion de projet sur plusieurs annees."
    ) is None


def test_real_short_headings_still_match():
    assert _match_section("Expérience") == "experience"
    assert _match_section("Formation") == "education"
    assert _match_section("Compétences requises") == "job_required"


def test_bullet_label_with_a_skill_alias_word_is_not_a_heading():
    assert _match_section("Technologies et outils utilises") is None
    assert _match_section("Outils utilises") is None


def test_heading_with_a_qualifier_word_still_matches():
    assert _match_section("Experiences professionnelles") == "experience"
    assert _match_section("Competences principales") == "skills"


def test_location_de_vehicule_is_not_misread_as_a_heading():
    assert _match_section("Location de véhicule") is None


def test_parenthesized_qualifier_still_matches():
    assert _match_section("COMPETENCES (PRINCIPALES)") == "skills"


def test_references_projets_maps_to_experience():
    assert _match_section("REFERENCES PROJETS") == "experience"
    assert _match_section("RÉFÉRENCES SIGNIFICATIVES") == "experience"


def test_letter_spaced_headings_are_collapsed_and_recognized():
    assert _match_section("C O M P É T E N C E S") == "skills"
    assert _match_section("F O R M A T I O N") == "education"
    assert _match_section("C E R T I F I C A T I O N") == "certifications"


def test_letter_spaced_short_name_is_not_a_false_positive():
    assert _match_section("C A R O L E") is None
    assert _match_section("C A R O L E   D U P O N T") is None


def test_old_spelling_qualifier_still_matches_experience():
    assert _match_section("Réalisations Clef") == "experience"
    assert _match_section("REALISATIONS MAJEURES") == "experience"


def test_compound_education_certifications_heading_resolves_to_education():
    assert _match_section("Diplômes / Certifications") == "education"
    assert _match_section("Formation et Certifications") == "education"
    assert _match_section("Certifications et Diplômes") == "education"


# ── _extract_years : formulation explicite ───────────────────────────────────


def test_explicit_years_phrasing_still_wins():
    assert _extract_years("9 années d'expérience") == 9
    assert _extract_years("5 ans d'expériences") == 5
    assert _extract_years("Expérience: 3 ans") == 3
    assert _extract_years("Minimum 8 ans en développement") == 8


# ── _extract_years : periodes datees ──────────────────────────────────────────


def test_simple_date_range():
    assert _extract_years("Développeur 2018 - 2023") == 5


def test_ongoing_period_counts_to_current_year():
    assert _extract_years("Chef de projet 2024 – à ce jour") == CURRENT_YEAR - 2024
    assert _extract_years("Data Analyst 2021 – Aujourd'hui") == CURRENT_YEAR - 2021


def test_depuis_year():
    assert _extract_years("Depuis 2018 chez Acme") == CURRENT_YEAR - 2018


def test_multiple_periods_are_summed():
    assert _extract_years("Ingénieur BI 2020 – 2024\nConsultant 2017 – 2020") == 7


def test_overlapping_periods_are_merged_not_double_counted():
    assert _extract_years("Poste A: 2018-2023\nPoste B: 2020-2024") == 6


def test_range_with_prepositions():
    assert _extract_years("Expérience 2015 au 2020") == 5
    assert _extract_years("De 2019 à 2023") == 4


def test_isolated_years_do_not_count():
    assert _extract_years("Diplômé en 2015") == 0
    assert _extract_years("Né en 1990") == 0
    assert _extract_years("Certification obtenue 2020") == 0


def test_result_stays_within_bounds():
    assert _extract_years("1960 - 2024") == 0
    assert 0 <= _extract_years("2000 - 2024") <= 40


def test_month_name_before_year_is_handled():
    assert _extract_years("janvier 2019 - décembre 2023") == 4
    assert _extract_years("jan 2019 - dec 2023") == 4


def test_numeric_month_before_year_is_handled():
    assert _extract_years("01/2019 - 12/2023") == 4


def test_since_with_month_name_is_handled():
    assert _extract_years("depuis janvier 2020") == CURRENT_YEAR - 2020


def test_many_short_consecutive_missions_are_not_undercounted():
    cv_text = (
        "Experience\n"
        "11/2022 – 08/2023\nMission A\n"
        "10/2021 – 06/2022\nMission B\n"
        "10/2020 – 07/2021\nMission C\n"
        "08/2019 – 07/2020\nMission D\n"
        "01/2019 – 06/2019\nMission E\n"
        "08/2018 – 11/2018\nMission F\n"
        "01/2018 – 07/2018\nMission G\n"
        "08/2017 – 11/2017\nMission H\n"
        "01/2017 – 04/2017\nMission I\n"
    )
    years = _extract_years(cv_text)
    assert years >= 6, (
        f"9 missions courtes s'enchainant sans interruption de 2017 a 2023 "
        f"(couverture reelle ~6-7 ans) ne doivent pas s'annuler a ~0, obtenu {years}"
    )


def test_ongoing_with_curly_apostrophe_is_recognized():
    assert _extract_years("Ingenieur BI 2020 – Aujourd’hui") == CURRENT_YEAR - 2020


def test_ongoing_with_acute_accent_apostrophe_is_recognized():
    assert _extract_years("Ingenieur BI 2020 – Aujourd´hui") == CURRENT_YEAR - 2020


def test_month_glued_to_year_is_still_a_valid_range():
    assert _extract_years("Ingenieur BI Decembre2022 – Mai 2023") == 1


# ── build_document_profile : portee section-aware de l'experience ────────────


def test_education_date_ranges_are_not_counted_as_experience():
    cv_text = (
        "Formation\n"
        "Master Informatique - 2015 - 2017\n"
        "Licence Informatique - 2012 - 2015\n"
        "\n"
        "Experience\n"
        "Ingenieur logiciel chez ACME - 2020 - 2023\n"
    )
    profile = build_document_profile(cv_text)
    assert profile.experience_years == 3, "seule la periode 2020-2023 est une vraie experience professionnelle"


def test_formations_plural_heading_is_still_recognized_as_education():
    cv_text = (
        "Experience\n"
        "Ingenieur BI chez ACME - 2020 - 2023\n"
        "Formations\n"
        "2017 - 2020 - Diplome d ingenieur\n"
        "2014 - 2017 - Technicien specialise\n"
    )
    profile = build_document_profile(cv_text)
    assert profile.experience_years == 3, (
        f"les dates de Formations (pluriel) ne doivent pas gonfler l'experience, obtenu {profile.experience_years}"
    )


def test_technologies_bullet_label_does_not_truncate_the_experience_section():
    cv_text = (
        "Experience\n"
        "Ingenieur BI 2024 – Aujourd'hui | SQLI\n"
        "Technologies et outils utilises : Power BI, SQL Server, Jira\n"
        "Ingenieur BI 2020 – 2022 | Aria Group\n"
        "Technologies et outils utilises : SSIS, SSAS, Power BI\n"
    )
    profile = build_document_profile(cv_text)
    assert "Aria Group" in profile.experience_text, (
        "le deuxieme poste ne doit pas etre bascule hors de la section Experience"
    )
    assert profile.experience_years == 2 + (CURRENT_YEAR - 2024)


def test_job_offer_experience_requise_heading_still_maps_to_experience_section():
    job_text = (
        "Competences requises: Python, Django.\n"
        "Experience requise: Minimum 8 ans en developpement backend.\n"
    )
    profile = build_document_profile(job_text)
    assert "8 ans" in profile.experience_text
    assert profile.experience_years == 8


def test_date_range_split_across_a_side_column_is_still_counted():
    cv_text = (
        "Experience\n"
        "Data Engineer; Euro Information\n"
        "De mai 2020 à\n"
        "UTI GROUP Dijon, France\n"
        "déc. 2022\n"
        "Realisations : mission reussie\n"
    )
    profile = build_document_profile(cv_text)
    assert profile.experience_years == 2


def test_explicit_statement_outside_the_experience_section_still_wins():
    cv_text = (
        "Guillaume SAHA\n"
        "25 ans d’expérience\n"
        "Competences\n"
        "Pilotage de projets data, Architecture, Formation\n"
        "References Projets\n"
        "PROJET 1 : Mise en place des evolutions BCE\n"
        "Assistance du chef de Projet sur les taches de pilotage.\n"
        "PROJET 2 : Implementation des schemas XSD\n"
    )
    profile = build_document_profile(cv_text)
    assert profile.experience_years == 25


# ── build_document_profile : langues et contrat ───────────────────────────────


def test_languages_section_is_detected():
    cv_text = "Langues\nFrancais courant, Anglais professionnel, Espagnol notions\n"
    profile = build_document_profile(cv_text)
    assert set(profile.language_terms) == {"français", "anglais", "espagnol"}


def test_contract_type_is_detected_from_real_text_not_a_form_field():
    cv_text = "Profil\nConsultant freelance disponible immediatement.\n"
    profile = build_document_profile(cv_text)
    assert profile.contract_type == "Freelance"


def test_no_contract_mention_returns_none():
    profile = build_document_profile("Profil\nDeveloppeur passionne par le clean code.\n")
    assert profile.contract_type is None


def test_empty_text_returns_an_empty_profile():
    profile = build_document_profile("")
    assert profile.experience_years == 0
    assert profile.contract_type is None
    assert profile.language_terms == []
    assert profile.education_text == ""
