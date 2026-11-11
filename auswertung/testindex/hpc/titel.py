"""
Anzeigetitel — aus URL-Pfad und Überschrift einen lesbaren Dokumenttitel bauen.

Wozu
----
`title` stammt bei PDFs aus dem Dateinamen. Gemessen am Korpus vom 14.08.2026
tragen 95.841 von 98.451 PDF-Chunks (97,3 %) und 4.709 von 22.683 HTML-Chunks
(20,8 %) damit keinen aussagekräftigen Titel:

    01-D3_F5-PO-BT-MTA_2021          09-PO_MGB_BT_11_09_2008
    BVM_RiggingPoint_13_20210910_web 20193009_BPO_EWI_verk

Das ist kein kosmetisches Problem. Die erste LLM-Runde des RAG wählt aus einer
Liste von Kandidaten anhand genau dieser Titel aus — sie sieht den Volltext
nicht. Zwischen `Besondere_PO_2009` und `BVM_RiggingPoint_13_20210910_web` kann
kein Modell sinnvoll entscheiden. Beide standen bei der Frage „Wie melde ich
mich zur Prüfung an?" unter den Treffern; dass das zweite ein Handbuch für
Veranstaltungstechnik ist, war aus dem Titel nicht zu erkennen.

Warum nicht im Crawler reparieren
---------------------------------
Naheliegend wäre, im Crawler das PDF-Metadatum `/Title` auszulesen. Das ist in
dieser Pipeline aber der teuerste Weg: `title` geht über `informative_title()`
in den Dedup-Schlüssel ein. Andere Titel bedeuten andere Gruppen, andere
Punktmenge — und damit eine ungültige Anreicherung. Crawl, 1,7 GPU-Stunden
doc2query und Vektorisierung müssten wiederholt werden. Der Anzeigetitel
gehört deshalb auf die Empfängerseite und wird zur Importzeit berechnet.

Woher die Information kommt
---------------------------
Der URL-Pfad trägt sie bereits und steht ohnehin in der Payload:

    …/fileadmin/HsH/Fakultaet_III/Veranstaltungsmanagement/BVM_RiggingPoint_13…
    →  f3 › Fakultaet III › Veranstaltungsmanagement › BVM RiggingPoint 13 …

Ergänzend die `section_heading` des ersten Chunks: bei Ordnungen trägt eher der
Pfad, bei Flyern und Steckbriefen eher die Überschrift.

Benutzt von `local_importer.py` (Qdrant-Payload) und `build_doc_index.py`
(doc_index.db). Beide müssen denselben Titel erzeugen — die Auswahlrunde liest
aus doc_index, Debug-Werkzeuge lesen aus der Payload, und zwei verschiedene
Titel für dasselbe Dokument wären eine Fehlerquelle ohne jeden Nutzen.
"""

import re
from urllib.parse import unquote, urlparse

# Pfadsegmente ohne Aussagewert. Sie beschreiben die Ablage, nicht den Inhalt.
TECHNISCHE_SEGMENTE = {
    "fileadmin", "hsh", "forms", "dokumente", "dokumente_", "bilder_datein",
    "bilder", "wp-content", "uploads", "upload", "content", "media", "files",
    "file", "download", "downloads", "assets", "images", "img", "pdf", "pdfs",
    "data", "sites", "default", "de", "en", "index", "_downloads",
}

# Hostpräfixe, die als Kürzel vorangestellt werden. Alles andere übernimmt
# seinen eigenen Hostnamen (z. B. "blog.bib" → "blog").
MAX_SEGMENTE = 3


def _ist_aussagekraeftig(titel: str) -> bool:
    """Trägt der vorhandene Titel schon Information?

    Kriterium ist bewusst grob: Ein Titel mit Leerzeichen, der keine URL ist und
    nicht wie ein Dateiname aussieht, wird übernommen. Damit bleiben die guten
    HTML-Titel („Bewerbung und Zulassungsvoraussetzungen") unangetastet, und nur
    die Dateinamen werden ersetzt.
    """
    t = (titel or "").strip()
    if not t or t.startswith("http"):
        return False
    if " " not in t:
        return False
    # "26-04-29 Kurzanleitung Pruefungsan- und -abmeldung" ist grenzwertig, aber
    # brauchbar; reine Versionsnummernketten sind es nicht.
    if re.fullmatch(r"[\d\s.\-_]+", t):
        return False
    return True


def _pfadteile(url: str) -> list[str]:
    """Die inhaltstragenden Pfadsegmente einer URL, von hinten begrenzt."""
    pfad = urlparse(url).path
    segmente = [unquote(s) for s in pfad.strip("/").split("/") if s]
    if segmente:
        segmente[-1] = re.sub(r"\.(pdf|html?|docx?|pptx?|xlsx?)$", "",
                              segmente[-1], flags=re.IGNORECASE)
    segmente = [
        s for s in segmente
        if s.lower() not in TECHNISCHE_SEGMENTE and not s.isdigit()
    ]
    return [re.sub(r"[_\-]+", " ", s).strip() for s in segmente[-MAX_SEGMENTE:]]


def _ueberschrift(section_heading: str) -> str:
    """Die unterste Ebene einer section_heading, von Markdown-Auszeichnung befreit.

    `section_heading` kommt als Pfad („Kapitel > Abschnitt"); interessant ist das
    letzte Glied. Verworfen wird, was offensichtlich kein Titel ist: zu kurz, zu
    lang, oder mit Doppelpunkt endend (das sind Tabellenbeschriftungen wie
    „Anerkennungen:", die mitten im Dokument stehen und in die Irre führen).
    """
    h = (section_heading or "").split(" > ")[-1]
    h = h.replace("*", "").replace("#", "").strip()
    if not (4 <= len(h) <= 70) or h.endswith(":"):
        return ""
    return h


def anzeigetitel(source_url: str, section_heading: str = "", roh_titel: str = "") -> str:
    """Baut einen lesbaren Titel für die Anzeige und die LLM-Dokumentauswahl.

    Ist `roh_titel` bereits aussagekräftig, wird er unverändert zurückgegeben —
    die gute Mehrheit der HTML-Seiten braucht keine Reparatur.

    Sonst entsteht der Titel aus Host-Kürzel, den letzten inhaltstragenden
    Pfadsegmenten und, falls sie etwas Neues beiträgt, der Überschrift des
    ersten Abschnitts.
    """
    if _ist_aussagekraeftig(roh_titel):
        return roh_titel.strip()

    host = urlparse(source_url).netloc.split(".")[0] or "hs-hannover"
    teile = _pfadteile(source_url)
    pfad = " › ".join(t for t in teile if t)

    if not pfad:
        # Wurzelseiten einer Subdomain haben keinen Pfad. Dann trägt der
        # Hostname allein, plus was die Überschrift hergibt.
        pfad = urlparse(source_url).netloc

    titel = f"{host} › {pfad}" if not pfad.startswith(host) else pfad

    ueberschrift = _ueberschrift(section_heading)
    if ueberschrift and ueberschrift.lower() not in titel.lower():
        titel = f"{titel} — {ueberschrift}"

    return titel
