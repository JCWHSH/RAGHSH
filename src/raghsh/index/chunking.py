"""
Gemeinsame Chunking-Logik für hpc_vectorizer.py und enricher.py.

Warum eine eigene Datei:
    Beide Skripte müssen aus denselben Markdown-Dateien exakt dieselben Chunks
    mit denselben IDs erzeugen — sonst passen die Anreicherungsdaten (Fragen)
    beim Join nicht mehr zu den Vektoren, und zwar ohne Fehlermeldung. Zwei
    Kopien derselben Logik driften über die Zeit auseinander; eine gemeinsame
    Quelle kann das nicht.

    Der zweite Grund ist die Laufzeitumgebung: hpc_vectorizer.py braucht
    fastembed und ONNX Runtime, enricher.py braucht PyTorch und transformers.
    Diese Stacks werden sinnvollerweise in getrennten venvs gehalten. Dieses
    Modul hat als einzige Abhängigkeit langchain-text-splitters und lässt sich
    deshalb in beiden installieren.

WICHTIG — Stabilität der IDs:
    make_point_id() bildet UUID5 aus source_url + chunk_index. Der chunk_index
    hängt an CHUNK_SIZE, CHUNK_OVERLAP und HEADERS_TO_SPLIT. Wer eine dieser
    Konstanten ändert, verschiebt sämtliche IDs — vorhandene Anreicherungen
    passen dann zu nichts mehr und müssen neu erzeugt werden. Änderungen hier
    sind also nie lokal.

    Die Deduplizierung gehört NICHT dazu: Sie entscheidet nur, welche der
    ohnehin feststehenden IDs behalten werden. Deshalb gibt es zwei
    Fingerabdrücke — config_hash() für die Grenzen, selection_hash() für die
    Auswahl. Siehe die Erläuterung bei config_hash().

Abhängigkeiten:
    pip install langchain-text-splitters
    titel.py aus demselben Verzeichnis (Kontextkopf, nur Standardbibliothek)
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import uuid
from pathlib import Path
from urllib.parse import urlparse

from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

try:
    from raghsh.index.titel import anzeigetitel
except ImportError:  # auf dem HPC-Cluster liegt titel.py daneben
    from titel import anzeigetitel

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Konfiguration — siehe Warnung oben, bevor hier etwas geändert wird
# ---------------------------------------------------------------------------

CHUNK_SIZE    = 1000
CHUNK_OVERLAP = 200

# Abschnitte, die nur aus einer Überschrift bestehen oder kürzer als diese
# Grenze sind, werden mit dem FOLGENDEN Abschnitt verschmolzen statt eigene
# Chunks zu werden.
#
# Grund: Die PDF→Markdown-Konvertierung hat die Hierarchie auf eine Ebene (##)
# abgeflacht. Paragraphennummer, Überschrift und Inhalt sind dadurch drei
# gleichrangige Abschnitte. Ohne Verschmelzung entsteht ein wertloser Chunk
# "## § 22" — und schlimmer: der Chunk mit dem Text des § 22 weiß nicht mehr,
# dass er zu § 22 gehört und ist darüber nicht auffindbar.
#
# Der Wert ist bewusst niedrig: reine Überschriften fängt _is_heading_only
# unabhängig von der Länge ab. Die Längenprüfung braucht es nur für Reste wie
# eine nackte Seitenzahl. Höhere Werte verschlucken echte kurze Aussagen
# ("Es gibt keine unbenoteten Prüfungsleistungen") und hängen sie unter die
# Überschrift des Nachbarn.
MERGE_BELOW_CHARS = 60

# Obergrenze für Abschnitte, die nur aus Überschriftenzeilen bestehen. Darüber
# gelten sie als Inhalt (Linklisten mit '#####  [![Mehr Info](…)](…)' je Zeile).
MAX_HEADING_MERGE_CHARS = 200

HEADERS_TO_SPLIT = [("#", "h1"), ("##", "h2"), ("###", "h3")]

# ── Tabellen (seit 22.09.2026) ─────────────────────────────────────────────
# Markdown-Tabellen aus PDFs (Stundenpläne, Semesterübersichten, Prüfungs-
# pläne) wurden vom RecursiveCharacterTextSplitter wie Fließtext zerschnitten:
# mitten in der Zeile, mitten in der Zelle, und nur das erste Stück kannte die
# Kopfzeile. Ein Chunk „|141|Praxisphase in IID/MTD|IID-141-01|…|9|0|" weiß
# dann weder, dass die 9 Kreditpunkte sind, noch zu welchem Studiengang die
# Zeile gehört (eval_gold g31, g33, g34, g50).
#
# Jetzt: Eine Tabelle wird in Zeilengruppen bis CHUNK_SIZE geteilt, Zeilen
# bleiben ganz (auch mehrzeilige Zellen), und jede Gruppe beginnt mit der
# Kopfzeile und der Trennzeile. Kein Überlappen — Zeilen sind atomar, eine
# doppelte Zeile wäre doppelter Inhalt.
TABELLEN_VERSION = "zeilen-atomar-mit-kopf/v2"
_TABLE_LINE_RE = re.compile(r"^\s*\|")
_TABLE_SEP_RE  = re.compile(r"^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$")

# ── Kontextkopf (seit 22.09.2026) ──────────────────────────────────────────
# Der Index bettete bisher nur `text` ein. Titel und Überschrift lagen in der
# Payload, aber kein Vektor und kein BM25-Term kannte sie: Ein Chunk aus der
# Mitte des MDT-Stundenplans wusste nicht, dass er zum Master Digitale
# Transformation gehört. `kontext` ist der lesbare Dokumenttitel (titel.py,
# derselbe, den die LLM-Auswahl sieht) plus die Abschnittsüberschrift;
# einbettungstext() stellt ihn dem Text voran — nur für Dense und BM25, der
# LLM-Kontext bekommt weiter den reinen `text`.
KONTEXT_VERSION = "anzeigetitel+ueberschrift/v1"

FACULTY_MAP = {
    "f1": "Fakultät I",
    "f2": "Fakultät II",
    "f3": "Fakultät III",
    "f4": "Fakultät IV",
    "f5": "Fakultät V",
}

# Host-Label → Einrichtung. Abgeleitet aus den im Crawl tatsächlich
# vorkommenden Hosts; 'bibliothek.' und 'international.' kamen darin nicht vor.
# '*.wp.hs-hannover.de' sind WordPress-Instanzen einzelner Projekte.
ORGANISATION_MAP = {
    "karriere":          "Karriere",
    "mint":              "MINT",
    "bib":               "Bibliothek",
    "bibliothek":        "Bibliothek",
    "international":     "International",
    "forschungscluster": "Forschungscluster",
    "wp":                "Projektwebseite",
    "it":                "IT-Service",
}

# ---------------------------------------------------------------------------
# Markdown lesen und chunken
# ---------------------------------------------------------------------------


def _parse_yaml_lines(lines) -> dict[str, str]:
    """Minimaler Key-Value-Parser für den Frontmatter-Block."""
    meta: dict[str, str] = {}
    for line in lines:
        key, sep, value = line.partition(":")
        if sep:
            meta[key.strip()] = value.strip().strip('"')
    return meta


def parse_markdown_file(path: Path) -> tuple[dict, str] | None:
    """Liest YAML-Frontmatter und Body aus einer Markdown-Datei.

    Verlangt, dass die Datei mit '---' beginnt. Ohne diese Prüfung wird eine
    headerlose Datei, die im Body eine horizontale Linie enthält, fälschlich
    als Frontmatter-Datei erkannt und der halbe Text als YAML interpretiert.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError) as exc:
        # In var/korpus liegt mindestens eine Datei mit Binärinhalt. Ohne
        # dieses except stirbt der komplette Lauf an einer einzigen kaputten
        # Datei — nach Stunden Rechenzeit und ohne verwertbare Ausgabe.
        logger.warning("%s nicht lesbar (%s) — übersprungen", path.name, exc)
        return None
    if not text.lstrip().startswith("---"):
        logger.warning("Kein YAML-Header in %s — übersprungen", path.name)
        return None
    parts = text.split("---", 2)
    if len(parts) < 3:
        logger.warning("Frontmatter nicht abgeschlossen in %s — übersprungen", path.name)
        return None
    body = parts[2].strip()
    if not body:
        logger.warning("Leerer Body in %s — übersprungen", path.name)
        return None
    return _parse_yaml_lines(parts[1].splitlines()), body


def read_frontmatter(path: Path) -> dict[str, str]:
    """Liest ausschließlich den YAML-Header und bricht danach ab.

    Wird für den --urls-Filter benutzt: dort interessiert nur die source_url,
    und die Dateien müssen nicht komplett in den Speicher gelesen werden.
    """
    with path.open(encoding="utf-8") as fh:
        if fh.readline().strip() != "---":
            return {}
        header: list[str] = []
        for line in fh:
            if line.strip() == "---":
                return _parse_yaml_lines(header)
            header.append(line)
    return {}


def _netloc_labels(url: str) -> list[str]:
    """Zerlegt den Host in seine Punkt-Bestandteile.

    Label-weise statt per startswith(), weil Fakultäts-Subdomains ihrerseits
    Präfixe haben: 'mediathek.f3.hs-hannover.de' gehört zu Fakultät III.
    """
    return urlparse(url.lower()).netloc.split(".")


def extract_faculty(url: str) -> str:
    """Fakultätszugehörigkeit aus Host-Label oder URL-Pfad. Sonst ''.

    Beide Formen kommen vor: 'https://f4.hs-hannover.de/...' (der Regelfall im
    HsH-Crawl) und '.../f4/...' (Pfadsegment). Nur auf den Pfad zu prüfen
    liefert für den Großteil der Seiten stillschweigend einen Leerstring.
    """
    for label in _netloc_labels(url):
        if label in FACULTY_MAP:
            return FACULTY_MAP[label]
    lower = url.lower()
    for code, name in FACULTY_MAP.items():
        if f"/{code}/" in lower:
            return name
    return ""


def extract_organisation(url: str) -> str:
    """Zentrale Einrichtung aus einem Host-Label. Sonst ''.

    Bewusst getrennt von extract_faculty: 'Bibliothek' ist keine Fakultät, und
    ein gemeinsames Feld macht jeden Fakultätsfilter im Retrieval unsauber.
    Ein Treffer schließt eine Fakultät nicht aus — 'mediathek.f3.…' bekommt
    beide Felder.
    """
    for label in _netloc_labels(url):
        if label in ORGANISATION_MAP:
            return ORGANISATION_MAP[label]
    return ""


def chunk_document(meta: dict, body: str,
                   header_splitter: MarkdownHeaderTextSplitter,
                   rec_splitter: RecursiveCharacterTextSplitter) -> list[dict]:
    """Zweistufiges strukturelles Chunking (Überschriften + Größenbegrenzung)."""
    source_url = meta.get("source_url", "")
    if not source_url:
        # Ohne source_url wäre make_point_id("", i) für JEDE solche Datei
        # identisch. In Qdrant überschreiben sich diese Punkte gegenseitig —
        # ohne Fehler, ohne Warnung, mit stillem Datenverlust. Deshalb hier
        # abbrechen statt kollidierende IDs zu erzeugen.
        logger.warning("Frontmatter ohne source_url (title=%r) — Datei übersprungen",
                       meta.get("title", ""))
        return []

    header_sections = header_splitter.split_text(clean_body(body))

    base_payload = {
        "source_url":   source_url,
        "title":        meta.get("title", ""),
        "crawl_date":   meta.get("crawl_date", ""),
        "content_type": meta.get("content_type", "html"),
        "faculty":      extract_faculty(source_url),
        "organisation": extract_organisation(source_url),
    }

    final_chunks: list[dict] = []
    chunk_idx = 0

    # Vorgemerkter Abschnitt, der zu klein war, um allein zu stehen. Er wird
    # dem nächsten Abschnitt vorangestellt, damit die Überschrift bei ihrem
    # Inhalt bleibt.
    pending_text = ""
    pending_heading = ""

    sections = list(header_sections)
    for pos, section in enumerate(sections):
        heading_parts = [
            section.metadata[k]
            for k in ("h1", "h2", "h3")
            if section.metadata.get(k)
        ]
        section_heading = " > ".join(heading_parts)
        text = section.page_content.strip()
        if not text:
            continue

        if pending_text:
            text = pending_text + "\n\n" + text
            section_heading = " > ".join(
                h for h in (pending_heading, section_heading) if h
            )
            pending_text = pending_heading = ""

        if _too_small(text) and len(text) < CHUNK_SIZE:
            pending_text, pending_heading = text, section_heading
            continue

        for sub in _split_section(text, rec_splitter):
            final_chunks.append({
                **base_payload,
                "text":            sub,
                "section_heading": section_heading,
                "chunk_index":     chunk_idx,
            })
            chunk_idx += 1

    # Rest am Dateiende. Er hat keinen Nachfolger mehr, an den er könnte —
    # also an den vorherigen Chunk hängen. Nur wenn es gar keinen gibt (die
    # ganze Datei ist kurz), wird daraus ein eigener Chunk.
    if pending_text:
        if final_chunks:
            final_chunks[-1]["text"] += "\n" + pending_text
        else:
            final_chunks.append({
                **base_payload,
                "text":            pending_text,
                "section_heading": pending_heading,
                "chunk_index":     chunk_idx,
            })
            chunk_idx += 1

    total = len(final_chunks)
    erste = final_chunks[0]["section_heading"] if final_chunks else ""
    titel = anzeigetitel(source_url, erste, meta.get("title", ""))
    for c in final_chunks:
        c["total_chunks"] = total
        c["kontext"] = _kontext(titel, c["section_heading"])
    return final_chunks


def _kontext(titel: str, section_heading: str) -> str:
    """Titel plus Überschrift, ohne Wiederholung — was der Chunk über seine
    Herkunft wissen muss."""
    ueberschrift = _WS_RE.sub(" ", section_heading.replace("*", "").replace("#", "")).strip()
    if ueberschrift and ueberschrift.lower() not in titel.lower():
        return f"{titel} › {ueberschrift}"
    return titel


def einbettungstext(chunk: dict) -> str:
    """Der Text, der in Dense- und BM25-Vektor eingeht: Kontextkopf + Text.

    Zentral hier, damit hpc_vectorizer.py und jede lokale Nachvektorisierung
    denselben Text einbetten. `text` selbst bleibt unverändert — er geht in
    den LLM-Kontext und in die Anzeige.
    """
    kontext = chunk.get("kontext") or ""
    return f"{kontext}\n{chunk['text']}" if kontext else chunk["text"]


# ---------------------------------------------------------------------------
# Tabellen
# ---------------------------------------------------------------------------


# Kein Tabellenstück wird größer als das: Eine 10-KB-„Zeile" (Studienangebot
# als Linktabelle) ließ die Attention des Einbettungsmodells auf 7 GB wachsen.
MAX_TABLE_PIECE = 2 * CHUNK_SIZE


def _table_rows(lines: list[str]) -> list[str]:
    """Zeilen einer Tabelle — mehrzeilige Zellen (PDF-Konvertate) bleiben in
    ihrer Zeile. Eine Zeile beginnt mit „|" und endet, wenn sie auf „|" endet
    ODER so viele Zellentrenner hat wie die Kopfzeile: Linktabellen aus HTML
    enden ihre Zeilen mit „)", nicht mit „|", und wären sonst eine einzige Zeile."""
    rows: list[str] = []
    aktuell: list[str] = []
    n_kopf = lines[0].count("|") if lines else 0
    for ln in lines:
        if aktuell and _TABLE_LINE_RE.match(ln) and "\n".join(aktuell).count("|") >= n_kopf > 0:
            rows.append("\n".join(aktuell))
            aktuell = []
        aktuell.append(ln)
        if ln.rstrip().endswith("|"):
            rows.append("\n".join(aktuell))
            aktuell = []
    if aktuell:
        rows.append("\n".join(aktuell))
    return rows


def _split_blocks(text: str) -> list[tuple[str, str]]:
    """Zerlegt einen Abschnitt in ("text", …) und ("table", …)-Blöcke.

    Ein Tabellenblock beginnt an einer Zeile mit „|" und läuft, solange Zeilen
    mit „|" beginnen oder eine offene Zeile (mehrzeilige Zelle) fortsetzen.
    """
    bloecke: list[tuple[str, str]] = []
    puffer: list[str] = []
    art = "text"
    offen = False                          # Tabellenzeile noch nicht mit „|" beendet
    for ln in text.splitlines():
        ist_tabelle = bool(_TABLE_LINE_RE.match(ln)) or (art == "table" and offen and ln.strip() != "")
        neue_art = "table" if ist_tabelle else "text"
        if neue_art != art and puffer:
            bloecke.append((art, "\n".join(puffer)))
            puffer = []
        art = neue_art
        puffer.append(ln)
        offen = art == "table" and not ln.rstrip().endswith("|")
    if puffer:
        bloecke.append((art, "\n".join(puffer)))
    return bloecke


def _table_pieces(block: str, rec_splitter: RecursiveCharacterTextSplitter) -> list[str]:
    """Zeilengruppen bis CHUNK_SIZE, jede mit Kopf- und Trennzeile.

    Eine einzelne Zeile, die mit Kopf größer als MAX_TABLE_PIECE wäre, geht
    ohne Kopf an den Größensplitter — lieber ein Stück ohne Spaltennamen als
    ein Chunk, der das Einbettungsmodell sprengt."""
    rows = _table_rows(block.splitlines())
    if len(rows) < 3 or not _TABLE_SEP_RE.match(rows[1].splitlines()[0]):
        return rec_splitter.split_text(block) if len(block) > CHUNK_SIZE else [block]
    kopf = rows[0] + "\n" + rows[1]
    if len(kopf) > CHUNK_SIZE // 2:
        kopf = rows[0][:CHUNK_SIZE // 2]           # absurde Kopfzeile: nur den Anfang
    pieces: list[str] = []
    gruppe: list[str] = []
    laenge = len(kopf)
    for row in rows[2:]:
        if len(kopf) + len(row) + 1 > MAX_TABLE_PIECE:
            if gruppe:
                pieces.append(kopf + "\n" + "\n".join(gruppe))
                gruppe, laenge = [], len(kopf)
            pieces.extend(rec_splitter.split_text(row))
            continue
        if gruppe and laenge + len(row) + 1 > CHUNK_SIZE:
            pieces.append(kopf + "\n" + "\n".join(gruppe))
            gruppe, laenge = [], len(kopf)
        gruppe.append(row)
        laenge += len(row) + 1
    if gruppe:
        pieces.append(kopf + "\n" + "\n".join(gruppe))
    return pieces


def _split_section(text: str, rec_splitter: RecursiveCharacterTextSplitter) -> list[str]:
    """Fließtext wie bisher (Größensplit + Kleinteile vorwärts), Tabellen
    zeilenweise mit Kopf. Die Reihenfolge der Blöcke bleibt erhalten."""
    bloecke = _split_blocks(text)
    if all(art == "text" for art, _ in bloecke):
        return (_merge_small_forward(rec_splitter.split_text(text))
                if len(text) > CHUNK_SIZE else [text])
    out: list[str] = []
    for art, block in bloecke:
        block = block.strip("\n")
        if not block.strip():
            continue
        if art == "table":
            out.extend(_table_pieces(block, rec_splitter))
        else:
            out.extend(_merge_small_forward(rec_splitter.split_text(block))
                       if len(block) > CHUNK_SIZE else [block])
    # Ein kurzer Textrest vor oder nach einer Tabelle („Stand: 10.07.2026")
    # soll nicht allein stehen — an den Nachbarn hängen, wie im Fließtext.
    return _merge_small_forward(out)


def make_point_id(source_url: str, chunk_index: int) -> str:
    """Deterministischer UUID5 — identisch mit ingest_to_qdrant.py."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source_url}#{chunk_index}"))


# ---------------------------------------------------------------------------
# Globale Deduplizierung
# ---------------------------------------------------------------------------
#
# Gemessen am Bestand vom 13.08.2026 (196.934 Chunks aus 5.604 Dateien):
# 40,1 % der Chunks sind wortgleiche Wiederholungen. Ursache sind gespiegelte
# PDFs (dieselbe Prüfungsordnung unter mehreren Fakultätspfaden), Navigations-
# reste und Boilerplate in Stellenausschreibungen. Die häufigste Wiederholung
# kommt 234-mal vor.
#
# Behalten wird je Gruppe EIN Vertreter JE ZUORDNUNG (siehe DEDUPE_AXES) —
# dadurch geht kein einziger Text verloren; es verschwinden nur Kopien, die
# auch keine eigene Zuordnung mitbringen. 298 Seiten (5,3 %) behalten danach
# keinen eigenen Chunk mehr und sind als Quelle nicht mehr auffindbar. Das ist
# eine bewusste Abwägung: Ihr Inhalt ist andernorts wortgleich vorhanden.

DEDUPE_CHUNKS = True

# Version des Schlüssels. Ändert sich die Definition unten, muss diese
# Zeichenkette mitwandern — sie geht in selection_hash() ein und ist der
# einzige Schutz davor, dass zwei Läufe mit verschiedenen Regeln
# zusammengejoint werden.
DEDUPE_KEY_VERSION = "text+aussagekraeftiger-titel/v1"

# Zuordnungen, die einen eigenen Vertreter rechtfertigen — siehe
# select_unique_ids(). Beide sind Payload-Felder, über die im Retrieval
# gefiltert wird; eine Zuordnung, die im Bestand vorkommt, aber in keinem
# behaltenen Punkt steht, ist für ihren Filter unauffindbar.
DEDUPE_AXES = ("faculty", "organisation")

# Version der Auswahlregel, getrennt von der des Schlüssels: Die Gruppen sind
# dieselben geblieben (v1), nur die Wahl der Vertreter innerhalb einer Gruppe
# hat sich geändert.
DEDUPE_SELECTION_VERSION = "vertreter-je-fakultaet-und-organisation/v2"

_TITLE_FILE_RE = re.compile(r"[\w.\-]+\.(?:pdf|docx?|xlsx?|html?)", re.IGNORECASE)
_WS_RE = re.compile(r"\s+")


def _norm_for_dedupe(text: str) -> str:
    return _WS_RE.sub(" ", text or "").strip().lower()


def informative_title(title: str) -> str:
    """Titel, sofern er Bedeutung trägt — sonst leer.

    Warum das nötig ist: Der Prompt bekommt den Seitentitel mit. Zwei Chunks
    mit gleichem Text, aber verschiedenem Titel erzeugen deshalb verschiedene
    Fragen, und sie zusammenzufassen wäre falsch. Beispiel aus dem Bestand:
    derselbe Absatz zum Veranstaltungsort unter "Material Monday 2026: Asbest",
    "… Kunststoffe" und "… Pflanzen und Tiere".

    Gemessen weichen bei 85,3 % der Duplikatgruppen die Titel ab — aber 96,7 %
    davon sind URLs oder Dateinamen ohne jede Aussage (bei den Stellenanzeigen
    IST der Titel die URL). Nur 3,3 % tragen einen echten Unterschied. Diese
    Funktion trennt beides, damit die Deduplizierung 39,2 % statt 40,1 %
    einspart und dafür die 892 echten Fälle unangetastet lässt.
    """
    t = (title or "").strip()
    if not t:
        return ""
    if t.lower().startswith(("http://", "https://", "www.")):
        return ""
    if _TITLE_FILE_RE.fullmatch(t):
        return ""
    # Dateiname-Stil: keine Leerzeichen, aber Trennzeichen ("ATPO_2015",
    # "1_aenderung_2008_fuer_alle")
    if " " not in t and ("_" in t or "-" in t) and len(t) > 6:
        return ""
    return _norm_for_dedupe(t)


def dedupe_key(chunk: dict) -> str:
    """Zwei Chunks mit gleichem Schlüssel gelten als austauschbar."""
    material = "\x00".join((_norm_for_dedupe(chunk["text"]),
                            informative_title(chunk.get("title"))))
    return hashlib.blake2b(material.encode("utf-8"), digest_size=16).hexdigest()


def select_unique_ids(md_files) -> set[str]:
    """Punkt-IDs der zu behaltenden Chunks.

    Je Gruppe wortgleicher Chunks wird EIN Vertreter behalten — und zusätzlich
    je Zuordnung aus DEDUPE_AXES einer, die in der Gruppe vorkommt, aber noch
    nicht vertreten ist.

    Warum der Zusatz: Der Vertreter trägt seine eigene faculty und
    organisation ins Payload; die der verworfenen Kopien verschwinden. Ein
    Filter faculty='Fakultät II' findet den Text dann nicht mehr, obwohl er in
    einem Dokument der Fakultät II wortgleich existiert — und zwar besonders
    bei den gespiegelten Prüfungsordnungen, wo genau die Frage 'gilt das für
    mich?' zählt. Gemessen am Bestand vom 14.08.2026: 1.343 solcher Paare,
    908 auf der Fakultäts-, 435 auf der Organisationsachse.

    Die Regel in zwei Zeilen:

        leere Zuordnung   → behalten, wenn der Schlüssel noch nie vorkam
        volle Zuordnung   → behalten, wenn (Schlüssel, Zuordnung) noch nie
                            vorkam

    Die leere Zuordnung erbt also vom Vertreter, statt einen eigenen Punkt zu
    erzwingen. Der Unterschied ist nicht klein: Nähme man faculty stumpf in
    dedupe_key() auf, kämen 7.839 statt 1.343 Punkte hinzu — 6.931 davon ohne
    jede Fakultät, also genau die, die ein Fakultätsfilter nie liefert.

    Kosten: 121.134 statt 119.791 Punkte (+1,12 %).

    ACHTUNG — die wichtigste Eigenschaft dieser Funktion: Sie ist **global**.
    Das Ergebnis hängt davon ab, welche Dateien übergeben werden. Wird sie mit
    einer Teilmenge aufgerufen, fällt die Wahl des Vertreters anders aus, und
    die IDs zweier Läufe passen nicht mehr zusammen.

    Deshalb gilt für BEIDE Skripte: immer mit der vollständigen Dateiliste aus
    list_markdown(var/korpus) aufrufen und ERST DANACH auf Teilmengen
    filtern (--urls, --resume-from, --shard). Nicht umgekehrt.

    Der Vertreter ist der erste Treffer in Dateireihenfolge; list_markdown()
    sortiert, iter_chunks() erhält die Reihenfolge — die Wahl ist damit
    reproduzierbar.
    """
    if not DEDUPE_CHUNKS:
        return None          # None = "keine Filterung", vom Aufrufer geprüft
    keep: set[str] = set()
    seen_key: set[str] = set()
    seen_pair: set[tuple[str, tuple[str, ...]]] = set()
    for _path, chunks in iter_chunks(md_files):
        for c in chunks:
            k = dedupe_key(c)
            zuordnung = tuple(c.get(a) or "" for a in DEDUPE_AXES)
            neu = (k not in seen_key
                   or (any(zuordnung) and (k, zuordnung) not in seen_pair))
            seen_key.add(k)
            seen_pair.add((k, zuordnung))
            if neu:
                keep.add(make_point_id(c["source_url"], c["chunk_index"]))
    return keep


def config_hash() -> str:
    """Fingerabdruck aller Parameter, die die Chunk-GRENZEN bestimmen.

    Zweck: Der Join zwischen enrichment.parquet und hsh_vectors.parquet läuft
    über make_point_id(source_url, chunk_index). Verschiebt sich das Chunking
    zwischen den beiden Läufen, passen die IDs weiterhin syntaktisch, zeigen
    aber auf andere Textstellen — der Join liefert dann Fragen zum falschen
    Abschnitt, und zwar ohne jede Fehlermeldung. Dagegen hilft nur Abbruch.

    Die Bereinigungsmuster gehören dazu: clean_body() läuft VOR dem Splitten,
    ein geändertes Muster verschiebt also ebenfalls sämtliche Grenzen.

    Warum die Deduplizierung hier NICHT mehr eingeht (bis 13.08.2026 tat sie
    es): Sie verschiebt keine Grenze und keine ID, sie entscheidet nur, welche
    der feststehenden IDs behalten werden. Ändert sich die Regel, bleibt jede
    vorhandene Anreicherungszeile gültig — es kommen höchstens Chunks hinzu,
    für die noch keine Fragen vorliegen. Das ist ein anderer, viel harmloserer
    Fall als eine Grenzverschiebung, und ihn mit demselben Abbruch zu
    behandeln zwang zu --ignore-enrichment-hash, das den echten Schutz gleich
    mit abschaltet. Deshalb steht die Auswahl jetzt in selection_hash().
    """
    material = json.dumps(
        {
            "chunk_size":              CHUNK_SIZE,
            "chunk_overlap":           CHUNK_OVERLAP,
            "merge_below_chars":       MERGE_BELOW_CHARS,
            "max_heading_merge_chars": MAX_HEADING_MERGE_CHARS,
            "headers":                 HEADERS_TO_SPLIT,
            "tabellen":                TABELLEN_VERSION,
            "clean_patterns":          {k: p.pattern for k, p in CLEAN_PATTERNS.items()},
            "clean_replacements":      {k: (p.pattern, r)
                                        for k, (p, r) in CLEAN_REPLACEMENTS.items()},
        },
        sort_keys=True, ensure_ascii=False,
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:16]


def selection_hash() -> str:
    """Fingerabdruck der Regel, WELCHE Chunks behalten werden.

    Gegenstück zu config_hash(): Weicht dieser Hash zwischen Anreicherung und
    Vektorisierung ab, zeigen die IDs weiterhin auf denselben Text — die
    Punktmengen decken sich nur nicht mehr. Der Vektorisierer warnt deshalb,
    statt abzubrechen, und die Trefferquote des Joins zeigt das Ausmaß.
    """
    material = json.dumps(
        {
            "dedupe_chunks":    DEDUPE_CHUNKS,
            "dedupe_key":       DEDUPE_KEY_VERSION,
            "dedupe_axes":      list(DEDUPE_AXES),
            "dedupe_selection": DEDUPE_SELECTION_VERSION,
        },
        sort_keys=True, ensure_ascii=False,
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Artefakte der PDF→Markdown-Konvertierung
# ---------------------------------------------------------------------------

# Platzhalter für ausgelassene Bilder: **==> picture [183 x 78] … <==**
_PICTURE_RE = re.compile(r"\*\*\s*==>.*?<==\s*\*\*", re.DOTALL)

# Base64-Blöcke. Der Crawler schreibt eingebettete Binärdaten (Bilder,
# vCards, Data-URIs) als Zeichenkette ins Markdown; auf den
# Personenfinder-Seiten entstehen daraus hunderte Chunks am Stück. Gemessen:
# 16,8 % aller Chunks und 23,8 % aller Zeichen im Bestand.
#
# Warum 200 Zeichen als Schwelle und nicht weniger: Deutscher Fließtext hat
# alle paar Zeichen ein Leerzeichen. 200 zusammenhängende Zeichen ohne jedes
# Trennzeichen kommen in echtem Text nicht vor — auch das längste Kompositum
# nicht. Niedrigere Schwellen träfen lange URLs, Pfade und Hashes.
_BASE64_INLINE_RE = re.compile(r"[A-Za-z0-9+/]{200,}={0,2}")

# Dieselben Daten in der klassischen, auf ~76 Zeichen umgebrochenen Form:
# mehrere aufeinanderfolgende Zeilen aus reinem Base64-Alphabet. Einzeln
# blieben sie unter der Schwelle oben, in Serie sind sie eindeutig.
_BASE64_BLOCK_RE = re.compile(
    r"(?:^[A-Za-z0-9+/]{60,}={0,2}[ \t]*$\n?){3,}", re.MULTILINE)

# Markierungen der OCR-Bildtexte aus PDFs. NUR die Marken, nicht der Inhalt:
# darin stecken teils echte Studienpläne (Semesterwochen, Credits), teils
# Werbeslogans. Bei 0,9 % der Chunks lohnt die Unterscheidung nicht, und
# echten Inhalt zu verwerfen wäre der teurere Fehler.
_PICTURE_TEXT_MARK_RE = re.compile(
    r"\*\*-{2,}\s*(?:Start|End)\s+of\s+picture\s+text\s*-{2,}\*\*",
    re.IGNORECASE)

# Seitenfußzeilen aus PDFs, jeweils als eigene Zeile
_PAGE_FOOTER_RE = re.compile(
    r"^\s*(?:_?\*{0,2})?Seite\s+\d+\s+von\s+\d+(?:\*{0,2}_?)?\s*$",
    re.MULTILINE | re.IGNORECASE,
)

# Der Hochschul-Logo-Link, der auf vielen Seiten als Textblock landet
_LOGO_LINK_RE = re.compile(
    r"\[\s*Hochschule\s+Hannover[^\]]{0,200}?applied\s+sciences[^\]]{0,80}?\]"
    r"\(\s*https?://[^)]*\)",
    re.IGNORECASE | re.DOTALL,
)


# Nackte Seitenzahl als eigene Zeile, ggf. mit Markdown-Auszeichnung.
# Jahreszahlen (1900-2099) sind ausgenommen: die längsten Dokumente im Bestand
# haben rund 165 Seiten, eine "2022" auf eigener Zeile ist also eine Jahres-
# angabe und keine Paginierung.
_PAGE_NUMBER_RE = re.compile(
    r"^[ \t]*[*_]{0,4}[ \t]*(?!(?:19|20)\d{2}[ \t*_]*$)\d{1,4}[ \t]*[*_]{0,4}[ \t]*$",
    re.MULTILINE)

# Inhaltsverzeichnis-Zeilen mit Punktführung: "IBS-411 BWL 1 ......... 9"
# Sechs aufeinanderfolgende Punkte statt einer Gesamtzahl — ein normaler Absatz
# hat viele Punkte, aber nie sechs am Stück.
_TOC_LINE_RE = re.compile(r"^.*\.{6,}.*$", re.MULTILINE)

# Laufende Kopfzeile "Fakultät IV – Wirtschaft und Informatik"
_RUNNING_HEADER_RE = re.compile(
    r"^[ \t]*[*_]{0,4}[ \t]*Fakultät\s+[IVX]+\s*[–—-][^\n]{0,60}$",
    re.MULTILINE)

# Fußzeile "Stand: 01.09.2025"
_STAND_RE = re.compile(
    r"^[ \t]*[*_]{0,4}[ \t]*Stand:\s*\d{1,2}\.\d{1,2}\.\d{2,4}[ \t]*[*_]{0,4}[ \t]*$",
    re.MULTILINE)

# Navigationsleiste des Bewerbungsportals. Steht auf jeder Stellenausschreibung
# zweimal — einmal über und einmal unter dem Anzeigentext — und besteht
# ausschließlich aus zwei Links ohne Fließtext:
#
#     [ Jetzt bewerben](…/stelle-900000413-10057.html?naviID=…)
#     [ Alle Stellenangebote](…/uebersicht.html?gruppe_id=0&…)
#
# Gemessen: Die daraus entstehenden Chunks sind untereinander zu 0,60 (Jaccard)
# identisch und erzeugten wortgleiche Fragen auf sieben verschiedenen Anzeigen
# ("Wo finde ich die Bewerbung für diese Stelle?"). Das senkt das IDF dieser
# Terme und macht die Anzeigen im Retrieval ununterscheidbar.
#
# Jeder Link wird einzeln getroffen, damit auch die Reihenfolge oder ein
# alleinstehender Link erfasst wird.
_BEWERBUNG_NAV_RE = re.compile(
    r"\[\s*(?:Jetzt\s+bewerben|Alle\s+Stellenangebote)\s*\]\(\s*https?://[^)]*\)",
    re.IGNORECASE,
)

# Reihenfolge egal, alle werden nacheinander angewandt. Als Dict, damit
# check_pdf_artefakte.py dieselben Muster benennen und einzeln vorführen kann —
# eine Regel, die versehentlich Inhalt entfernt, soll vorher auffallen.
CLEAN_PATTERNS = {
    "base64-inline":       _BASE64_INLINE_RE,
    "base64-block":        _BASE64_BLOCK_RE,
    "bildtext-marke":      _PICTURE_TEXT_MARK_RE,
    "bild-platzhalter":    _PICTURE_RE,
    "logo-link":           _LOGO_LINK_RE,
    "seite-n-von-m":       _PAGE_FOOTER_RE,
    "seitenzahl":          _PAGE_NUMBER_RE,
    "inhaltsverzeichnis":  _TOC_LINE_RE,
    "kopfzeile-fakultaet": _RUNNING_HEADER_RE,
    "stand-datum":         _STAND_RE,
    "bewerbung-nav":       _BEWERBUNG_NAV_RE,
}


# Ersetzt statt entfernt: <br> ist im PDF-Markdown der Zeilenumbruch. Bliebe
# es stehen, lieferte es dem BM25 zigtausendfach den Term "br" und dem
# Dense-Modell nutzlose Token. Ersatz durch einen echten Umbruch gibt dem
# Splitter zugleich eine Trennstelle, an der er sauber schneiden kann.
CLEAN_REPLACEMENTS = {
    "br-tag": (re.compile(r"<br\s*/?>", re.IGNORECASE), "\n"),
}


def clean_body(body: str) -> str:
    """Entfernt Artefakte, die kein Inhalt sind.

    Diese Muster stammen aus der Konvertierung, nicht von der Webseite. Sie
    stehen teils als eigene Abschnitte (dann entstehen leere Chunks), teils
    mitten im Text (dann verwässern sie das Embedding und liefern BM25
    Begriffe wie "picture" oder "Seite").

    Besonders die Inhaltsverzeichnis-Zeilen sind schädlich: Sie enthalten
    Modulcodes und Modulnamen ohne jeden Inhalt und konkurrieren im BM25
    direkt mit den echten Modulbeschreibungen.
    """
    for pattern in CLEAN_PATTERNS.values():
        body = pattern.sub("", body)
    for pattern, ersatz in CLEAN_REPLACEMENTS.values():
        body = pattern.sub(ersatz, body)
    return body


def _is_heading_only(text: str) -> bool:
    """Wahr, wenn der Abschnitt außer Überschriftenzeilen nichts enthält."""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return bool(lines) and all(ln.startswith("#") for ln in lines)


def _too_small(text: str) -> bool:
    """Soll dieser Abschnitt an seinen Nachbarn gehängt werden?

    Die Längengrenze für Überschriften ist nötig, weil 'nur Überschriften'
    nicht 'kein Inhalt' bedeutet: Auf Linklisten-Seiten ist jede Zeile eine
    Markdown-Überschrift mit einem Link darin. Ohne diese Grenze gilt eine
    ganze solche Seite als verschmelzbar und kettet sich zu einem einzigen
    riesigen Chunk zusammen.
    """
    if len(text) < MERGE_BELOW_CHARS:
        return True
    return _is_heading_only(text) and len(text) < MAX_HEADING_MERGE_CHARS


def _merge_small_forward(pieces: list[str]) -> list[str]:
    """Hängt zu kleine Teilstücke an das jeweils folgende an.

    Nötig, weil der RecursiveCharacterTextSplitter die Verschmelzung sonst
    wieder aufhebt: Abschnitte werden mit einer Leerzeile verbunden, und genau
    daran trennt der Splitter zuerst. Bei einem Paragraphen über 1000 Zeichen
    stünde die Nummer danach erneut allein.

    Ein Rest am Ende geht an das vorherige Stück — Text soll nie verloren
    gehen und auch nicht als Fragment zurückbleiben.
    """
    out: list[str] = []
    pending = ""
    for piece in pieces:
        piece = piece.strip()
        if not piece:
            continue
        if pending:
            piece = pending + "\n" + piece
            pending = ""
        # Deckel: sobald das Angesammelte selbst Chunk-Größe erreicht, wird es
        # ausgegeben, egal wie es aussieht. Ohne das kann eine Kette kleiner
        # Teilstücke unbegrenzt wachsen.
        if _too_small(piece) and len(piece) < CHUNK_SIZE:
            pending = piece
            continue
        out.append(piece)
    if pending:
        if out:
            out[-1] = out[-1] + "\n" + pending
        else:
            out.append(pending)
    return out


# ---------------------------------------------------------------------------
# Bequemer Zugriff für beide Skripte
# ---------------------------------------------------------------------------


def make_splitters() -> tuple[MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter]:
    """Erzeugt beide Splitter. Zustandslos — einmal pro Lauf genügt."""
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=HEADERS_TO_SPLIT,
        strip_headers=False,
    )
    rec_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    return header_splitter, rec_splitter


def iter_chunks(md_files):
    """Liefert (Pfad, Chunk-Liste) je lesbarer Datei.

    Kapselt die Schleife, die sonst in beiden Skripten identisch stünde:
    Frontmatter parsen, Chunken, unlesbare oder leere Dateien überspringen.
    """
    header_splitter, rec_splitter = make_splitters()
    for path in md_files:
        result = parse_markdown_file(path)
        if result is None:
            continue
        meta, body = result
        chunks = chunk_document(meta, body, header_splitter, rec_splitter)
        if chunks:
            yield path, chunks


def list_markdown(directory: Path) -> list[Path]:
    """Sortierte Liste der .md-Dateien — sortiert, damit die Reihenfolge
    zwischen Läufen und zwischen Skripten identisch ist."""
    return sorted(Path(directory).glob("*.md"))
