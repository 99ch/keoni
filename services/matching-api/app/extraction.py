"""Extraction de texte depuis PDF/DOCX/TXT, avec lecture consciente des colonnes.

Port fidèle de l'équivalent AI Real-Time (backend/app/services/extraction.py),
demandé explicitement pour que Keoni obtienne la même qualité d'extraction —
et donc des scores comparables sur un même échantillon de CV — que l'autre
plateforme, plutôt que de se contenter de l'extraction Tika générique.

Le problème que ce module résout : les extracteurs PDF "naïfs" (Tika/PDFBox
comme PyMuPDF par défaut) lisent le texte dans l'ordre du flux de dessin du
PDF, pas dans l'ordre de lecture visuel. Sur un CV à deux colonnes (barre
latérale nom/contact/compétences à côté d'un corps principal plus haut), une
ligne de la barre latérale et une ligne du corps principal qui se trouvent à
une hauteur (y) proche sont émises l'une après l'autre, mélangeant deux
sections sans rapport en plein milieu d'une phrase.

Les docstrings ci-dessous sont conservées du code source d'origine : elles
documentent des bugs de production réels rencontrés en développant ce
mécanisme (voir historique git d'AI Real-Time), pas des cas hypothétiques —
ce sont les raisons d'être de chaque choix, à ne pas défaire par erreur.

Ce module ne couvre que .pdf/.docx/.txt. Les autres formats (et le cas où
l'extraction ci-dessous échoue/renvoie du vide) retombent sur Tika+OCR déjà
en place dans main.py::text_from_file().
"""

from __future__ import annotations

import logging
import os
import re
import unicodedata
from pathlib import Path

import fitz  # PyMuPDF
import pytesseract
from docx import Document
from docx.oxml.ns import qn
from docx.table import Table
from docx.text.paragraph import Paragraph
from pdf2image import convert_from_path

logger = logging.getLogger(__name__)

# Configuration OCR, indépendante de app.main.Settings pour éviter tout
# import circulaire (ce module est importé depuis main.py).
OCR_MIN_TEXT_LENGTH = int(os.getenv("MATCHING_OCR_MIN_TEXT_LENGTH", "20"))
OCR_DPI = int(os.getenv("MATCHING_OCR_DPI", "200"))
OCR_LANGUAGES = os.getenv("MATCHING_OCR_LANGUAGES", "fra+eng")
OCR_OEM = int(os.getenv("MATCHING_OCR_OEM", "3"))
OCR_MAX_PAGES = int(os.getenv("MATCHING_OCR_MAX_PAGES", "20"))
OCR_PAGE_TIMEOUT_SECONDS = int(os.getenv("MATCHING_OCR_PAGE_TIMEOUT_SECONDS", "25"))


# Les lignes plus longues que ce seuil sont candidates à la déduplication
# d'en-tête/pied de page répété ci-dessous ; les lignes courtes ne le sont
# jamais. Un en-tête/pied de page ("Curriculum Vitae - Jean Dupont",
# "Document confidentiel") est presque toujours une phrase complète, alors
# qu'une puce courte récurrente ("Python", "SQL", "Rigueur") est exactement
# le genre de contenu légitimement répété entre plusieurs expériences d'un
# CV, et ne doit pas être traité pareil.
_BOILERPLATE_MIN_LENGTH = 20


def _collapse_letter_spacing(line: str) -> str:
    """Recolle un titre/mot rendu en espacement de lettres décoratif par le
    template du CV ("C O M P É T E N C E S" -> "COMPÉTENCES").

    Porté d'AI Real-Time (629a1bb), trouvé sur une validation à grande
    échelle (~1500 CV réels) : ~1% des CV perdaient toute leur section
    compétences parce que le titre de section ("COMPÉTENCES") était rendu
    en espacement large par le template PDF/DOCX, et l'extraction le relit
    alors comme un token par lettre. Chez eux ça cassait la détection de
    section ; on ne l'a pas, mais le même rendu pollue directement le texte
    envoyé au cross-encoder (du bruit hors distribution plutôt qu'une vraie
    phrase) et empêcherait le matching par n-gramme de la taxonomie si un
    intitulé de compétence lui-même était rendu ainsi.

    Ne se déclenche que si la quasi-totalité des tokens de la ligne sont des
    lettres isolées, pour ne jamais toucher une prose normale contenant
    quelques mots d'une lettre ("à", "y", "a"...).
    """
    tokens = line.split()
    if len(tokens) < 4:
        return line
    single_letter = sum(1 for t in tokens if len(t) == 1 and t.isalpha())
    if single_letter / len(tokens) < 0.8:
        return line
    return "".join(tokens)


def clean_text(text: str) -> str:
    """
    Normalise le texte extrait :
    - recolle les mots coupés par un tiret en fin de ligne
    - retire les préfixes de puces/tirets
    - déduplique les en-têtes/pieds de page répétés (lignes longues seulement)
    """
    if not text:
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Recolle les mots coupés par un tiret en fin de ligne
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)

    # Corrige les substitutions de ligatures de police que PyMuPDF ne décode pas
    # U+25A0 (■) est utilisé à la place de la ligature "fi" (fiabilité → ■abilité)
    text = text.replace("■", "fi")
    # Un "?" entre deux lettres est un glyphe de ligature "ti" non mappé (conception → concep?on)
    text = re.sub(r"(?<=[a-zA-ZÀ-ɏ])\?(?=[a-zA-ZÀ-ɏ])", "ti", text)

    lines: list[str] = []
    seen: dict[str, int] = {}
    prev = ""

    for raw in text.split("\n"):
        # Retire les tirets doux et caractères de puce, réduit les espaces
        line = re.sub(r"­", "", raw)
        line = re.sub(r"\s+", " ", line.strip())
        # retire aussi les puces Wingdings/Symbol (ü→U+00FC, ð→U+00F0) utilisées comme marqueurs de liste dans certains PDF
        line = re.sub(r"^[\-*•·•◦ü°ð►▪▫●○◦]+\s*", "", line).strip()
        line = _collapse_letter_spacing(line)

        if not line or len(line) < 2:
            prev = ""
            continue

        key = unicodedata.normalize("NFKD", line).encode("ascii", "ignore").decode().lower()
        if key == prev:
            continue
        if len(line) > _BOILERPLATE_MIN_LENGTH:
            seen[key] = seen.get(key, 0) + 1
            if seen[key] > 2:
                continue

        lines.append(line)
        prev = key

    return "\n".join(lines).strip()


def _iter_lines(blocks: list[dict]) -> list[dict]:
    """Aplatit les blocks de get_text('dict') en lignes individuelles {"bbox", "text"}.

    Alimente _order_lines_by_column() au niveau ligne plutôt que block --
    voir le docstring de cette fonction pour pourquoi les bbox au niveau
    block ne sont pas fiables pour détecter les colonnes.
    """
    result: list[dict] = []
    for block in blocks:
        for line in block.get("lines", []):
            text = "".join(span.get("text", "") for span in line.get("spans", []))
            if text.strip():
                result.append({"bbox": line["bbox"], "text": text})
    return result


def _detect_column_gutter(lines: list[dict], page_width: float) -> float:
    """Trouve l'espace vertical entre colonnes de texte au lieu de supposer
    qu'il se situe exactement au milieu horizontal de la page.

    Un découpage fixe à page_width/2 fonctionne pour une barre latérale à
    peu près aussi large que le corps principal, mais beaucoup de vrais
    templates de CV utilisent une barre latérale ÉTROITE (~25-30% de la
    page) avec le corps principal commençant bien avant 50%. Face à un
    milieu fixe, toute ligne courte du corps principal se terminant avant
    ce milieu (un titre de poste, une puce d'une ligne) est classée à tort
    comme contenu de barre latérale et interleavée avec la vraie barre
    latérale par position y -- produisant exactement le genre de mélange de
    fragments en plein milieu de phrase que ce module entier existe pour
    empêcher, juste déplacé de "blocks entiers" (le bug que
    _order_lines_by_column corrige déjà) à "lignes courtes par rapport au
    mauvais point de découpage".

    Projette la portée horizontale de chaque ligne sur la largeur de la
    page, fusionne les portées qui se chevauchent/touchent, et retourne le
    milieu du plus large espace entre elles -- le vrai espace entre deux
    colonnes réelles. Retombe sur page_width/2 quand aucun espace n'est
    assez large pour être un vrai espace entre colonnes (texte mono-colonne,
    ou texte qui chevauche par hasard le point de découpage), donc les CV
    mono-colonne classiques ne sont pas affectés.
    """
    spans = sorted((ln["bbox"][0], ln["bbox"][2]) for ln in lines)
    merged: list[list[float]] = []
    for x0, x1 in spans:
        if merged and x0 <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], x1)
        else:
            merged.append([x0, x1])

    best_gap, best_mid = 0.0, page_width / 2
    for (_, prev_end), (next_start, _) in zip(merged, merged[1:]):
        gap = next_start - prev_end
        if gap > best_gap:
            best_gap, best_mid = gap, (prev_end + next_start) / 2

    # Exige un vrai espace, pas du bruit entre mots proches -- mais pas trop
    # strict au point de rejeter une vraie gouttière de barre latérale
    # étroite. Cas réel côté AI Real-Time (audit corpus 13k CV, 2026-09-14,
    # 38d44a8) : une gouttière réelle mesurant ~3,4% de la largeur de page
    # tombait juste sous l'ancien seuil de 4%, ce qui faisait retomber cette
    # fonction sur le milieu exact de la page -- lequel tombait en plein
    # milieu de la colonne large, réduisant sa part de caractères à ~3% et
    # déclenchant le repli total sur un tri (y, x) qui mélange barre
    # latérale et corps principal phrase par phrase. Le seuil de largeur
    # n'est qu'un garde-fou contre le bruit d'arrondi (un "espace" de 1-2pt),
    # pas la vraie défense contre un faux positif de coupure -- ce rôle
    # revient au filtre de part de caractères dans _order_lines_by_column,
    # qui voit la distribution RÉELLE du texte plutôt qu'une simple largeur
    # en pixels.
    min_gutter = page_width * 0.02
    return best_mid if best_gap >= min_gutter else page_width / 2


def _order_lines_by_column(lines: list[dict], page_width: float) -> list[dict]:
    """Réordonne les lignes de texte pour lire une mise en page à deux
    colonnes colonne par colonne, au lieu de les interleaver par position y.

    get_text(sort=True) de PyMuPDF ordonne les spans par (y, x), ce qui est
    correct pour du texte mono-colonne et pour des colonnes côte à côte de
    hauteur similaire, mais mélange le template de CV courant d'une barre
    latérale courte (nom/contact/compétences) à côté d'un corps principal
    bien plus haut : une ligne de barre latérale et une ligne de corps
    principal qui se trouvent à un y similaire sont toutes les deux émises à
    ce point, interleavant deux sections sans rapport en plein milieu d'une
    phrase.

    Fonctionne ligne par ligne, pas block par block : une version antérieure
    classait et triait des blocks entiers de get_text('dict'), mais la bbox
    d'un block est l'union de toutes les lignes qu'il contient, donc une
    seule ligne longue dépassant le milieu de la page entraîne tout son
    block avec elle de l'autre côté. PyMuPDF fusionne couramment toute la
    colonne de corps principal d'un CV en un seul block, donc une seule
    ligne large fait basculer tout ce block de "colonne de gauche" à
    "chevauche les deux colonnes", désactivant la détection deux-colonnes et
    retombant sur un simple tri (y, x) de blocks entiers -- où le block de
    corps ENTIER, multi-lignes, peut alors dépasser dans le tri un block
    d'en-tête/nom bien plus étroit situé à un y similaire (les égalités se
    tranchent sur x, et un corps commençant à x=40 l'emporte sur un nom à
    x=420), au lieu que les deux s'interleavent ligne par ligne comme le
    faisait get_text(sort=True). Ça a poussé le nom d'un vrai candidat des
    dizaines de lignes plus bas, au-delà de la fenêtre que l'heuristique de
    nom scanne.

    Détecte une séparation verticale nette -- chaque ligne se trouve
    entièrement à gauche ou entièrement à droite du milieu horizontal de la
    page, sauf les lignes pleine largeur comme une bannière nom/en-tête --
    et alors seulement lit les lignes pleine largeur dans leur position
    naturelle de haut en bas, puis toute la colonne de gauche de haut en
    bas, puis toute la colonne de droite. Retombe sur un simple tri (y, x)
    quand aucune séparation nette n'existe, donc les CV mono-colonne
    ordinaires ne sont pas affectés.
    """
    if not lines:
        return []

    mid = _detect_column_gutter(lines, page_width)
    left, right, full = [], [], []
    for ln in lines:
        x0, _, x1, _ = ln["bbox"]
        if x1 <= mid:
            left.append(ln)
        elif x0 >= mid:
            right.append(ln)
        else:
            full.append(ln)

    # Exige que les deux côtés portent une part significative du texte de la
    # page, pas juste un compte de lignes : une vraie colonne de barre
    # latérale peut être une seule ligne courte, tandis qu'un CV mono-colonne
    # peut aussi avoir une petite ligne entièrement à droite du milieu (une
    # date alignée à droite, un numéro de page) sans être une mise en page à
    # deux colonnes du tout. Filtrer sur la part de caractères plutôt que le
    # compte de lignes évite de réordonner (et de mélanger) ce cas mono-
    # colonne courant.
    total_chars = sum(len(ln["text"]) for ln in lines) or 1
    left_chars = sum(len(ln["text"]) for ln in left)
    right_chars = sum(len(ln["text"]) for ln in right)
    min_share = 0.15
    if left_chars < total_chars * min_share or right_chars < total_chars * min_share:
        return sorted(lines, key=lambda ln: (ln["bbox"][1], ln["bbox"][0]))

    left.sort(key=lambda ln: ln["bbox"][1])
    right.sort(key=lambda ln: ln["bbox"][1])
    full.sort(key=lambda ln: ln["bbox"][1])

    first_col_y = min(left[0]["bbox"][1], right[0]["bbox"][1])
    header = [ln for ln in full if ln["bbox"][1] < first_col_y]
    trailer = [ln for ln in full if ln["bbox"][1] >= first_col_y]
    return header + left + right + trailer


def _find_tables(page) -> list:
    """Détecte les tableaux d'une page via le détecteur géométrique intégré
    de PyMuPDF (embarqué depuis PyMuPDF 1.23 -- pas de nouvelle dépendance,
    pas de modèle ML). La stratégie par défaut exige de vraies lignes/
    remplissages de tracé, donc un simple template de CV à deux colonnes
    sans aucun cadrage de tableau (le cas déjà géré par
    _order_lines_by_column) est correctement laissé de côté.

    Rejette un tableau détecté si une de ses cellules contient plus d'une
    poignée de lignes de texte. Cas réel côté AI Real-Time (audit corpus
    13k CV, 2026-09-14, fa63df2) : sur un CV deux colonnes SANS aucun
    cadrage de tableau, le détecteur de PyMuPDF (basé sur les espaces
    blancs) a quand même pris la PAGE ENTIÈRE pour un tableau 2 lignes x 2
    colonnes -- chaque "cellule" était le texte complet d'une colonne (nom,
    coordonnées, une section entière, plus d'une dizaine de lignes).
    _render_table_rows joint alors chaque ligne par tabulation, et
    clean_text() réduit ensuite cette tabulation à un simple espace comme
    n'importe quel autre blanc -- fusionnant le dernier titre de la colonne
    de gauche directement dans le premier titre de la colonne de droite,
    sans aucun séparateur. Le nombre de lignes seul ne suffit pas à
    détecter ce cas : le faux tableau avait 2 lignes, comme un petit
    tableau légitime. Ce qui distingue vraiment les deux, c'est la taille
    des cellules -- une vraie cellule de tableau de CV (un intitulé de
    poste, une courte liste de compétences) tient sur une ligne, parfois
    deux ; une "cellule" contenant plus d'une dizaine de lignes est en
    réalité une section entière qui n'a jamais été tabulaire.
    """
    try:
        tables = list(page.find_tables().tables)
    except Exception as exc:
        logger.warning("Détection de tableau échouée sur une page de %s: %s", getattr(page, "number", "?"), exc)
        return []
    return [t for t in tables if _is_plausible_table(t.extract())]


# Une vraie cellule de tableau de CV (un intitulé de poste, une courte
# liste de compétences) tient sur une ligne, parfois deux -- une "cellule"
# contenant ce nombre de lignes est en réalité une section entière qui
# n'a jamais été tabulaire (voir _find_tables).
_MAX_PLAUSIBLE_TABLE_CELL_LINES = 5


def _is_plausible_table(rows: list[list[str | None]]) -> bool:
    if len(rows) < 2:
        return False
    return not any(
        (cell or "").count("\n") >= _MAX_PLAUSIBLE_TABLE_CELL_LINES
        for row in rows
        for cell in row
    )


def _render_table_rows(rows: list[list[str | None]]) -> str:
    """Rend les lignes de tableau extraites en lignes jointes par tabulation,
    même convention que pour les tableaux DOCX (extract_text_from_docx) --
    garde chaque cellule catégorie/label collée à sa/ses propre(s)
    cellule(s) valeur sur une ligne, plutôt que de les laisser aux
    heuristiques d'ordre de lecture ligne par ligne ci-dessous, qui n'ont
    aucune notion de "ces lignes forment une seule ligne de tableau" et
    peuvent interleaver une cellule wrappée avec une ligne adjacente sans
    rapport.
    """
    lines = []
    for row in rows:
        cells = [str(c).strip() for c in row if c and str(c).strip()]
        if cells:
            lines.append("\t".join(cells))
    return "\n".join(lines)


def _line_center_in_bbox(line_bbox: tuple[float, float, float, float], table_bbox) -> bool:
    """True si le centre d'une ligne de texte tombe dans la bbox d'un
    tableau -- sert à retirer les lignes individuelles déjà prises en compte
    par find_tables(), pour que le contenu d'un tableau ne soit pas émis
    deux fois (une fois mélangé par l'ordre ligne par ligne, une fois
    propre via _render_table_rows).
    """
    cx = (line_bbox[0] + line_bbox[2]) / 2
    cy = (line_bbox[1] + line_bbox[3]) / 2
    return table_bbox[0] <= cx <= table_bbox[2] and table_bbox[1] <= cy <= table_bbox[3]


def extract_text_from_pdf(path: Path) -> str:
    """Extrait le texte d'un PDF via PyMuPDF avec ordre conscient des
    colonnes ; repli OCR pour les pages scannées.

    Les lignes sont lues via get_text('dict') (bounding boxes) et
    réordonnées par _order_lines_by_column() plutôt que le sort=True natif
    de PyMuPDF, qui gère le mono-colonne et les colonnes côte à côte de
    hauteur assortie mais pas un template de barre latérale asymétrique --
    voir le docstring de cette fonction.

    Les tableaux sont détectés séparément via _find_tables() et rendus en
    lignes propres jointes par tabulation, remplaçant les lignes
    individuelles qui tombaient dedans. Bug réel corrigé par ceci : le
    tableau "Compétences Techniques" d'un CV (une colonne catégorie/label
    étroite à côté d'une colonne de listes d'outils wrappées, multi-lignes)
    lu ligne par ligne collait le label de la ligne N+1 en plein milieu de
    la valeur wrappée de la ligne N. Le score sémantique du cross-encoder
    (composant de scoring le plus lourdement pondéré) jugeait alors ce
    français incohérent bien plus bas que ce que la couverture de
    compétences réelle du CV méritait.

    Le seuil OCR est vérifié par page, pas sur le texte combiné du
    document : une poignée de vraies pages passe facilement un seuil
    document-entier à elles seules, ce qui sautait silencieusement l'OCR --
    et donc perdait tout le contenu -- de n'importe quelle page purement
    scannée mêlée à un document par ailleurs textuel (ex. un diplôme/
    certificat scanné ajouté à un CV exporté depuis Word).
    """
    try:
        doc = fitz.open(str(path))
        page_texts: list[str] = []
        weak_pages: list[int] = []  # pages 0-indexées sous le seuil OCR
        for i, page in enumerate(doc):
            raw = page.get_text("dict")
            blocks = [b for b in raw.get("blocks", []) if b.get("type") == 0]
            lines = _iter_lines(blocks)

            table_lines = []
            table_bboxes = []
            for table in _find_tables(page):
                rendered = _render_table_rows(table.extract())
                if rendered:
                    table_bboxes.append(table.bbox)
                    table_lines.append({"bbox": table.bbox, "text": rendered})

            if table_bboxes:
                lines = [
                    ln for ln in lines
                    if not any(_line_center_in_bbox(ln["bbox"], tb) for tb in table_bboxes)
                ]
            lines.extend(table_lines)

            ordered = _order_lines_by_column(lines, raw.get("width") or page.rect.width)
            t = "\n".join(ln["text"] for ln in ordered)
            page_texts.append(t)
            if len(t.strip()) < OCR_MIN_TEXT_LENGTH:
                weak_pages.append(i)
        doc.close()

        if weak_pages:
            if len(weak_pages) > OCR_MAX_PAGES:
                logger.warning(
                    "PDF %s a %d page(s) nécessitant l'OCR, OCR limité aux %d premières (MATCHING_OCR_MAX_PAGES)",
                    path.name,
                    len(weak_pages),
                    OCR_MAX_PAGES,
                )
                weak_pages = weak_pages[:OCR_MAX_PAGES]

            logger.info(
                "PDF %s: %d page(s) sous le seuil OCR (%d car.), OCR lancé sur ces pages seulement",
                path.name,
                len(weak_pages),
                OCR_MIN_TEXT_LENGTH,
            )
            ocr_by_page = _ocr_pdf_pages(path, weak_pages)
            for i in weak_pages:
                ocr_text = ocr_by_page.get(i, "")
                if len(ocr_text.strip()) > len(page_texts[i].strip()):
                    page_texts[i] = ocr_text

        return clean_text("\n".join(page_texts))

    except Exception as exc:
        logger.exception("Extraction PDF échouée pour %s: %s", path.name, exc)
        return ""


def _ocr_pdf_pages(path: Path, page_numbers: list[int]) -> dict[int, str]:
    """OCR Tesseract sur des pages 0-indexées spécifiques uniquement ;
    utilise psm=3 pour les mises en page multi-colonnes.

    Chaque page est rendue et passée à l'OCR une à la fois -- plutôt que de
    convertir tout le PDF en images d'un coup -- donc le coût mémoire/temps
    dépend des pages qui ont réellement besoin d'OCR, pas du nombre total de
    pages du document, et chaque page a son propre timeout dur pour qu'une
    page pathologique dégrade en texte partiel plutôt que de bloquer tout
    le traitement.
    """
    results: dict[int, str] = {}
    config = f"--psm 3 --oem {OCR_OEM}"
    for page_num in page_numbers:
        try:
            images = convert_from_path(
                str(path),
                dpi=OCR_DPI,
                first_page=page_num + 1,
                last_page=page_num + 1,
            )
        except Exception as exc:
            logger.exception("Rendu OCR échoué pour %s page %d: %s", path.name, page_num + 1, exc)
            continue
        if not images:
            continue
        try:
            t = pytesseract.image_to_string(
                images[0],
                lang=OCR_LANGUAGES,
                config=config,
                timeout=OCR_PAGE_TIMEOUT_SECONDS,
            )
        except RuntimeError:
            logger.warning(
                "OCR expiré sur la page %d de %s (> %ds), page ignorée",
                page_num + 1, path.name, OCR_PAGE_TIMEOUT_SECONDS,
            )
            continue
        if t.strip():
            results[page_num] = t
    return results


def _iter_docx_block_items(doc):
    """Produit chaque paragraphe/tableau enfant du corps du document, dans
    l'ordre du document.

    python-docx expose doc.paragraphs et doc.tables comme deux listes plates
    séparées qui ne préservent pas leur position relative -- extraire "tous
    les paragraphes puis tous les tableaux" déplace chaque tableau à la fin
    du texte peu importe où il se trouve réellement dans le document,
    mélangeant un CV qui alterne texte libre et tableau expérience/
    compétences.
    """
    body = doc.element.body
    for child in body.iterchildren():
        if child.tag == qn("w:p"):
            yield Paragraph(child, doc)
        elif child.tag == qn("w:tbl"):
            yield Table(child, doc)


def _docx_textbox_lines(paragraph_element) -> list[str]:
    """Retourne le texte trouvé dans tout <w:txbxContent> (zone de texte)
    imbriqué dans ce paragraphe.

    Trouvé chez AI Real-Time (50e976a, audit sur 888 vrais CV DOCX réels) :
    18,6% des DOCX de leur corpus construisent tout leur mise en page avec
    des zones de texte/formes pour des raisons de design -- Paragraph.text
    de python-docx ne parcourt que les <w:r> directs du paragraphe, donc le
    texte niché dans la structure propre <w:txbxContent><w:p> d'une zone de
    texte est silencieusement invisible, jusqu'à perdre le corps entier du
    CV (un fichier de leur corpus perdait 39 594 caractères ainsi). Pas un
    problème d'OCR : c'est du texte XML réel, juste jamais atteint par le
    parcours normal paragraphe/tableau.
    """
    lines: list[str] = []
    for txbx in paragraph_element.iter(qn("w:txbxContent")):
        for inner_p in txbx.iter(qn("w:p")):
            texts = [t.text for t in inner_p.iter(qn("w:t")) if t.text]
            line = "".join(texts).strip()
            if line:
                lines.append(line)
    return lines


def _docx_header_footer_lines(doc) -> tuple[list[str], list[str]]:
    """Retourne (lignes_entête, lignes_pied_de_page) sur toutes les sections.

    doc.paragraphs ne couvre que le corps du document -- le texte placé
    dans un en-tête ou pied de page (très souvent le nom/les coordonnées du
    candidat dans un template de CV) est sinon silencieusement perdu à
    l'extraction.
    """
    headers: list[str] = []
    footers: list[str] = []
    for section in doc.sections:
        if section.header is not None:
            headers.extend(p.text.strip() for p in section.header.paragraphs if p.text.strip())
        if section.footer is not None:
            footers.extend(p.text.strip() for p in section.footer.paragraphs if p.text.strip())
    return headers, footers


def extract_text_from_docx(path: Path) -> str:
    """Extrait en-têtes/pieds de page, paragraphes et tableaux DOCX (dans l'ordre du document)."""
    try:
        doc = Document(path)
        parts: list[str] = []

        header_lines, footer_lines = _docx_header_footer_lines(doc)
        parts.extend(header_lines)

        for block in _iter_docx_block_items(doc):
            if isinstance(block, Paragraph):
                t = block.text.strip()
                if t:
                    parts.append(t)
                parts.extend(_docx_textbox_lines(block._p))
            elif isinstance(block, Table):
                for row in block.rows:
                    # joint les cellules d'une même ligne par tabulation pour garder les colonnes lisibles
                    cells = [c.text.strip() for c in row.cells if c.text.strip()]
                    if cells:
                        parts.append("\t".join(cells))
                    for cell in row.cells:
                        for cell_p in cell.paragraphs:
                            parts.extend(_docx_textbox_lines(cell_p._p))

        parts.extend(footer_lines)

        return clean_text("\n".join(parts))
    except Exception as exc:
        logger.exception("Extraction DOCX échouée pour %s: %s", path.name, exc)
        return ""


def extract_text_from_txt(path: Path) -> str:
    """Extrait du texte brut, en essayant UTF-8, puis cp1252, puis latin-1.

    latin-1 accepte n'importe quelle valeur d'octet et ne lève jamais
    UnicodeDecodeError, donc il était auparavant essayé juste après UTF-8 --
    mais un .txt sauvegardé sous Windows avec des guillemets/tirets "intelligents"
    (très courant en copiant-collant depuis Word) est encodé en cp1252, et
    le décoder en latin-1 transforme ces caractères en codes de contrôle C1
    (mojibake) au lieu d'échouer bruyamment. cp1252 est essayé en premier :
    c'est un sur-ensemble de latin-1 pour le cas courant d'Europe de l'Ouest
    et lève encore une erreur sur la poignée de valeurs d'octet qu'il laisse
    indéfinies, donc latin-1 reste le tout dernier repli.
    """
    for encoding in ("utf-8", "cp1252", "latin-1"):
        try:
            return clean_text(path.read_text(encoding=encoding))
        except UnicodeDecodeError:
            continue
        except Exception as exc:
            logger.exception("Extraction TXT échouée pour %s: %s", path.name, exc)
            return ""
    return ""


def extract_text(path: Path) -> str:
    """Distribue l'extraction selon l'extension du fichier. Retourne '' en cas d'échec."""
    if not path.exists() or not path.is_file():
        logger.warning("Fichier introuvable: %s", path)
        return ""
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return extract_text_from_pdf(path)
    if suffix == ".docx":
        return extract_text_from_docx(path)
    if suffix == ".txt":
        return extract_text_from_txt(path)
    logger.warning("Format non supporté par ce module: %s pour %s (repli Tika)", suffix, path.name)
    return ""
