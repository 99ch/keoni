"""Tests de non-régression pour app/extraction.py.

Le premier cas ci-dessous correspond à un bug trouvé chez AI Real-Time
(629a1bb, validation sur ~1500 CV réels) sur le même genre d'extraction
PDF/DOCX que nous portons ici, et observé directement sur le CV utilisé
tout au long de cette fonctionnalité (titre "DÉVELOPPEUR WEB" rendu
"D É V E L O P P E U R W E B" par l'extraction PyMuPDF sur un template au
style décoratif).

Les cas _is_plausible_table/_detect_column_gutter ci-dessous portent deux
bugs réels trouvés par AI Real-Time sur un corpus de validation de 13 715
CV (2026-09-14, fa63df2 et 38d44a8) sur ces mêmes fonctions, portées à
l'identique côté Keoni (point 9).

Le cas test_docx_captures_text_hidden_in_a_textbox porte un bug réel
trouvé par AI Real-Time sur 888 vrais CV DOCX de production (2026-09-18,
50e976a) : le texte placé dans une zone de texte/forme (<w:txbxContent>)
était entièrement invisible à l'extraction -- 165 fichiers (18,6%) avaient
plus de 500 caractères ainsi cachés, certains jusqu'à l'intégralité du CV.
"""

from __future__ import annotations

import docx
from docx.oxml import parse_xml
from docx.oxml.ns import nsdecls

from app.extraction import (
    _collapse_letter_spacing,
    _detect_column_gutter,
    _is_plausible_table,
    clean_text,
    extract_text_from_docx,
)


def test_letter_spaced_heading_is_collapsed():
    assert _collapse_letter_spacing("C O M P É T E N C E S") == "COMPÉTENCES"
    assert _collapse_letter_spacing("D É V E LO P P E U R W E B") == "DÉVELOPPEURWEB"


def test_normal_prose_is_left_alone():
    assert _collapse_letter_spacing("Ceci est une phrase normale") == "Ceci est une phrase normale"
    # Mots d'une lettre légitimes dans une phrase normale (proportion trop faible pour déclencher).
    assert _collapse_letter_spacing("Il a dit à y aller") == "Il a dit à y aller"


def test_short_line_is_not_collapsed():
    # Sous le seuil de 4 tokens : jamais touché, même si tout est en lettres isolées.
    assert _collapse_letter_spacing("R et D") == "R et D"


def test_clean_text_applies_letter_spacing_collapse():
    result = clean_text("D É V E L O P P E U R W E B\nCompétences: Python, Docker")
    assert "DÉVELOPPEURWEB" in result
    assert "Python" in result


def test_whole_page_misdetected_as_a_table_is_rejected():
    """Régression réelle (validation du corpus de 13 715 CV chez AI
    Real-Time, 2026-09-14, CV Rachid Aissaoui) : sur un CV à deux colonnes
    SANS AUCUN cadre de tableau, le détecteur géométrique de PyMuPDF a
    quand même pris la page ENTIÈRE pour un tableau 2 lignes / 2 colonnes
    -- chaque "cellule" étant le texte complet d'une colonne (nom, contact,
    tout un paragraphe...), soit une douzaine de lignes. Le nombre de
    lignes seul ne suffit pas à rejeter ce faux tableau (il en avait bien
    2, comme un petit tableau légitime) -- c'est la taille démesurée de ses
    cellules qui le trahit."""
    rows = [
        [
            "Rachid\nAISSAOUI\nConsultant\nQA\n78280 Guyancourt\n06.24.11.53.98\n"
            "aissaoui.r@gmail.com\nlinkedin.com/in/riln\nPolyvalent et curieux\n"
            "d'adaptation et d'apprentissage\nnouveaux defis\nDIPLOMES / CERTIFICATIONS",
            "EXPERIENCE PROFESSIONNELLE\nCONSULTANT QA\nBig Ben Corp\n"
            "Qualification de sites\nRealisation de tests\nGestion des retours\n"
            "CONSULTANT TEST RECETTE\nGMF Assurances\nProjets Evolutions",
        ],
        ["2013 Licence Assurance\nUniversite Paris II\nISTQB Foundation\nHP Quality Center", None],
    ]
    assert _is_plausible_table(rows) is False


def test_a_real_small_table_is_still_accepted():
    """Garde-fou : un vrai (petit) tableau CV, cellules d'une ou deux
    lignes, ne doit pas être rejeté par le garde-fou de taille."""
    rows = [
        ["Langages", "Python, Java, SAS, SQL, R et Scala"],
        ["Outils de\ndéveloppement", "Toad Siebel SAS6 SAS91 SAS92 SAS94"],
        ["SGBD", "Oracle, PostgreSQL, MySQL, MongoDB"],
    ]
    assert _is_plausible_table(rows) is True


def test_a_single_row_table_is_rejected():
    assert _is_plausible_table([["Langages", "Python, Java"]]) is False


def test_narrow_real_gutter_is_not_rejected_as_pixel_noise():
    """Régression réelle (validation du corpus de 13 715 CV chez AI
    Real-Time, 2026-09-14, CV Aurelien Torres) : le vrai gouffre entre les
    deux colonnes de ce CV ne mesurait que ~20.5pt sur une page de 595pt de
    large (3.4% de la largeur) -- juste sous l'ancien seuil de 4% (23.8pt),
    qui traitait ce gouffre bien réel comme du bruit et retombait sur le
    milieu exact de page (297.5), en plein milieu de la colonne de droite."""
    left = [
        {"bbox": (40.0, 60.0, 113.3, 63.0), "text": "Jean DUPONT"},
        {"bbox": (40.0, 90.0, 103.6, 93.0), "text": "Certifications"},
        {"bbox": (40.0, 120.0, 95.6, 123.0), "text": "Profil perso"},
        {"bbox": (40.0, 150.0, 108.5, 153.0), "text": "Curiosite forte"},
    ]
    right = [
        {"bbox": (134.0, 60.0, 232.4, 63.0), "text": "Chef de Projet MOA"},
        {"bbox": (134.0, 90.0, 472.1, 93.0), "text": "Pilotage de projets SI en environnement multi-pays"},
        {"bbox": (134.0, 120.0, 183.5, 123.0), "text": "Formation"},
        {"bbox": (134.0, 150.0, 258.7, 153.0), "text": "Master informatique 2018"},
    ]
    mid = _detect_column_gutter(left + right, page_width=595.0)
    assert 113.3 < mid < 134.0, (
        f"le vrai gouffre (113.3-134.0) ne doit pas être rejeté au profit "
        f"du milieu de page fixe (297.5), obtenu mid={mid}"
    )


def test_docx_captures_text_hidden_in_a_textbox(tmp_path):
    doc = docx.Document()
    doc.add_paragraph("En-tête visible normalement.")

    # <w:txbxContent> minimal imbriqué dans un run, comme le font les vrais
    # templates de CV pour placer toute une barre latérale/section dans une
    # forme "zone de texte". python-docx n'a pas d'API haut niveau pour les
    # zones de texte, donc ceci est construit directement en XML, de la même
    # façon que python-docx lui-même l'émettrait.
    textbox_xml = (
        f'<w:r {nsdecls("w")} xmlns:v="urn:schemas-microsoft-com:vml">'
        '<w:pict><v:shape>'
        '<v:textbox><w:txbxContent>'
        '<w:p><w:r><w:t>Compétences: Python, Django, PostgreSQL</w:t></w:r></w:p>'
        '<w:p><w:r><w:t>Expérience: 5 ans en développement backend</w:t></w:r></w:p>'
        '</w:txbxContent></v:textbox>'
        '</v:shape></w:pict></w:r>'
    )
    p = doc.add_paragraph()
    p._p.append(parse_xml(textbox_xml))
    doc.add_paragraph("Pied de page visible normalement.")

    path = tmp_path / "cv_with_textbox.docx"
    doc.save(path)

    result = extract_text_from_docx(path)

    assert "En-tête visible normalement." in result
    assert "Pied de page visible normalement." in result
    assert "Compétences: Python, Django, PostgreSQL" in result
    assert "Expérience: 5 ans en développement backend" in result
