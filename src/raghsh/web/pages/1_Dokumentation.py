"""
Dokumentationsseite — Wie funktioniert der HsH-Assistent?
"""

import streamlit as st

st.set_page_config(
    page_title="Dokumentation — HsH-Assistent",
    page_icon="📖",
    layout="wide",
)

st.title("📖 Wie funktioniert der HsH-Assistent?")
st.caption(
    "Diese Seite erklärt die technische Funktionsweise des Chatbots — "
    "von der Datenerhebung bis zur fertigen Antwort."
)

# ---------------------------------------------------------------------------
# Überblick
# ---------------------------------------------------------------------------

st.header("Überblick")
st.markdown(
    """
Der HsH-Assistent ist ein **RAG-System** (Retrieval-Augmented Generation):
Er beantwortet Fragen nicht aus dem Gedächtnis eines KI-Modells, sondern sucht
zuerst in einer aktuellen Wissensdatenbank nach relevanten Dokumenten und leitet
die Antwort ausschließlich daraus ab.

Das System besteht aus vier Phasen, die einmal aufgebaut und dann dauerhaft
genutzt werden:
"""
)

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.info("**Phase 1**\n\nWeb-Crawling\n\n🕷️ hs-hannover.de → Markdown-Dateien")
with col2:
    st.info("**Phase 2**\n\nVektorisierung\n\n🔢 Markdown → Vektoren in Qdrant")
with col3:
    st.info("**Phase 3**\n\nHybrid-Suche\n\n🔍 Frage → relevante Chunks")
with col4:
    st.info("**Phase 4**\n\nAntwort-Pipeline\n\n🤖 Chunks → LLM → Antwort")

st.markdown(
    """
```
Nutzeranfrage
     │
     ▼
┌─────────────────────────────────────────────────────────────────────┐
│  RAG-Pipeline (MCP-Server)                                          │
│                                                                     │
│  Schritt 1: Hybrid-Suche (Dense + Sparse → RRF → Reranking)        │
│       ↓ 30 Chunks aus Qdrant                                        │
│  Schritt 2: Kandidaten bestimmen (URL-Deduplizierung → 8 Docs)      │
│       ↓ Dokument-Metadaten (Titel, Struktur)                        │
│  Schritt 3: LLM-Runde 1 — Dokumentauswahl                          │
│       ↓ 4 ausgewählte Dokumente                                     │
│  Schritt 4: Volltexte aus Qdrant laden (Token-Budget: 100 000)      │
│       ↓ Kontext                                                     │
│  Schritt 5: LLM-Runde 2 — Antwort generieren                       │
└─────────────────────────────────────────────────────────────────────┘
     │
     ▼
Antwort mit Quellenangaben
```
"""
)

st.divider()

# ---------------------------------------------------------------------------
# Phase 1: Web-Crawling
# ---------------------------------------------------------------------------

st.header("Phase 1 — Web-Crawling")

with st.expander("Details zum Crawling-Prozess", expanded=False):
    st.markdown(
        """
### Was wird gecrawlt?

Der Crawler startet auf `https://www.hs-hannover.de/` und folgt allen internen
Links per **Breadth-First Search (BFS)** — d. h. zuerst alle Links der Startseite,
dann alle Links der gefundenen Seiten usw. Bis zu **10 000 Seiten** werden so
erfasst, einschließlich aller Subdomains wie `f1.hs-hannover.de` (Fakultät 1),
`f2.hs-hannover.de` (Fakultät 2) usw.

Ausgenommen ist `serwiss.bib.hs-hannover.de` (Publikationsserver), da er
technisch separate Inhalte enthält.

### Wie werden Seiten verarbeitet?

**HTML-Seiten** werden mit [Crawl4AI](https://crawl4ai.com/) geladen.
CSS-Selektoren beschränken die Extraktion auf den Hauptinhalt (`<main>`,
`.content-main` etc.). Navigation, Footer, Cookie-Banner und Sidebars werden
automatisch ausgefiltert. Das Ergebnis wird als **Markdown** gespeichert.

**PDF-Dokumente** werden per HTTP heruntergeladen und mit
[PyMuPDF4LLM](https://pymupdf.readthedocs.io/en/latest/pymupdf4llm/) in
Markdown konvertiert.

### Ausgabeformat

Jede Seite wird als Datei mit YAML-Frontmatter gespeichert:

```yaml
---
source_url: "https://www.hs-hannover.de/studium/..."
title:       "Studiengänge — Hochschule Hannover"
crawl_date:  "2026-03-08"
content_type: "html"
faculty:     "f1"          # leer wenn nicht einer Fakultät zugeordnet
---
```

Der eigentliche Seiteninhalt folgt darunter als Markdown-Text.

### Cache-Strategie

Seiten, die jünger als **7 Tage** sind, werden nicht neu gecrawlt (nur für
Link-Extraktion besucht). PDFs bei frischem Cache werden vollständig übersprungen.
Fehlerhafte Seiten werden in einem Excel-Protokoll festgehalten.

### Verlässlichkeit

Der Crawler fügt zwischen zwei Requests eine **2-Sekunden-Pause** ein, um den
Webserver der HsH nicht zu belasten.
"""
    )

st.divider()

# ---------------------------------------------------------------------------
# Phase 2: Vektorisierung
# ---------------------------------------------------------------------------

st.header("Phase 2 — Vektorisierung")

with st.expander("Details zur Vektorisierung", expanded=False):
    st.markdown(
        """
### Was ist ein Vektor?

Ein Vektor ist eine Liste von Zahlen, die den *Bedeutungsinhalt* eines Texts
codiert. Semantisch ähnliche Texte liegen im Vektorraum nah beieinander —
unabhängig davon, ob sie dieselben Wörter verwenden.

### Chunking (Aufteilen in Abschnitte)

Lange Seiten werden in kleinere **Chunks** (Abschnitte) aufgeteilt, damit die
Suche gezielt einzelne Textabschnitte finden kann. Der Prozess ist zweistufig:

1. **Markdown-Header-Split**: Trennt an Überschriften (`# Titel`, `## Unterabschnitt`).
2. **Rekursiver Zeichensplit**: Teilt zu lange Abschnitte weiter auf.

Jeder Chunk behält Metadaten aus dem YAML-Header (URL, Titel, Datum, Fakultät).

### Zwei Embedding-Modelle

Das System verwendet **zwei verschiedene Repräsentationen** pro Chunk:

| Arm | Modell | Grundlage | Zweck |
|-----|--------|-----------|-------|
| **`text`** (dense) | [Jina Embeddings v3](https://jina.ai/embeddings/) | Chunktext | Semantische Ähnlichkeit |
| **`questions`** (dense) | Jina Embeddings v3 | generierte Fragen zum Chunk | Frage trifft auf Frage |
| **`sparse`** | [BM25 via FastEmbed](https://github.com/qdrant/fastembed) | Chunktext | Exakte Schlüsselwörter |
| **`sparse_expanded`** | BM25 via FastEmbed | Chunktext **+** Fragen | Schlüsselwörter aus der Fragesicht |

Das Dense-Modell versteht Bedeutungen (»Studiengebühren« ≈ »Semesterbeitrag«).
Das Sparse-Modell (BM25) findet exakte Begriffe zuverlässig (Modulnummern, Namen).

Zu jedem Chunk erzeugt ein Sprachmodell vorab Fragen, die er beantwortet
(*doc2query*). Diese Fragen liegen als zwei zusätzliche Arme im Index — sie
sind reines Suchsignal und gehen **nicht** in den Kontext des antwortenden
Modells ein.

### Speicherung in Qdrant

[Qdrant](https://qdrant.tech/) ist eine spezialisierte **Vektordatenbank**.
Die Collection `hsh_knowledge` enthält vier benannte Vektor-Felder: `"text"`
und `"questions"` (dense) sowie `"sparse"` und `"sparse_expanded"` (BM25,
beide mit `modifier=IDF`). Chunk-IDs sind deterministisch (UUID5 aus
`source_url#chunk_index`), sodass Re-Imports idempotent sind (kein
Datenverlust bei erneutem Einlesen).
"""
    )

st.divider()

# ---------------------------------------------------------------------------
# Phase 3: Hybrid-Suche
# ---------------------------------------------------------------------------

st.header("Phase 3 — Hybrid-Suche")

with st.expander("Details zur Suchmethodik", expanded=False):
    st.markdown(
        """
### Warum Hybrid?

- Reine **Vektorsuche** übersieht exakte Schlüsselwörter (Modulnummern, Namen).
- Reine **Keyword-Suche** versteht keine Bedeutungen (Synonyme, Umformulierungen).
- **Hybrid** kombiniert beides für höhere Treffergenauigkeit.

### Suchablauf

**Schritt 1 — Zwei parallele Suchanfragen**

Die Nutzeranfrage wird gleichzeitig in beiden Formen kodiert:
- Als **Dense-Vektor** (Jina Embeddings v3, Task `retrieval.query`)
- Als **Sparse-Vektor** (BM25)

Beide Suchen laufen als ein einziger `query_points()`-Aufruf in Qdrant
(sog. *Prefetch*).

**Schritt 2 — RRF-Fusion (Reciprocal Rank Fusion)**

Qdrant führt die Ergebnisse beider Arme serverseitig mit **RRF** zusammen.
Chunks, die in *beiden* Armen gut ranken, erhalten einen höheren Gesamtscore.
Es wird kein manueller Python-Code für die Fusion benötigt.

**Schritt 3 — URL-Deduplizierung**

Pro Quell-URL werden maximal **2 Chunks** behalten. Dadurch kann eine Seite nicht
alle Top-Plätze belegen, aber ein zweiter relevanter Absatz derselben Quelle bleibt
erhalten.

**Schritt 4 — Semantisches Reranking (Cross-Encoder)**

Die verbleibenden Chunks werden mit dem
[Jina Reranker v2](https://jina.ai/reranker/) neu bewertet.
Im Gegensatz zur Vektorsuche bewertet ein Cross-Encoder jedes `(Frage, Chunk)`-Paar
*gemeinsam* — deutlich präziser, aber rechenintensiver.

**Schritt 5 — Kontext-Augmentierung**

Für die **Top-3-Treffer** werden die unmittelbar benachbarten Chunks
(Chunk n−1 und n+1 derselben URL) nachgeladen. So werden Informationen,
die an Chunk-Grenzen abgeschnitten wurden, nicht verloren.

### Ergebnis

8 relevante Chunks mit Metadaten (URL, Titel, Abschnitt, Crawl-Datum, Fakultät),
bereit für die Antwort-Pipeline.
"""
    )

st.divider()

# ---------------------------------------------------------------------------
# Phase 4: Antwort-Pipeline
# ---------------------------------------------------------------------------

st.header("Phase 4 — Zweistufige Antwort-Pipeline")

with st.expander("Details zur Antwortgenerierung", expanded=False):
    st.markdown(
        """
Die eigentliche Antwortgenerierung läuft in **zwei LLM-Runden**.

---

### LLM-Runde 1 — Dokumentauswahl

**Eingabe:** Liste von bis zu 8 Kandidaten-Dokumenten mit Titel, URL und
Inhaltsstruktur (Heading-Baum).

**Aufgabe:** Das Modell wählt die **bis zu 4 relevantesten Dokumente** aus
und gibt ihre Nummern als JSON zurück (`{"selected": [2, 5, 1, 3]}`).

**Warum dieser Schritt?**
Die Hybrid-Suche findet Chunks — kleine Textausschnitte. Für eine gute Antwort
werden aber oft die *vollständigen Seiteninhalte* benötigt. Durch die Vorauswahl
werden nur wirklich relevante Volltexte geladen — das spart Token-Budget und
verbessert die Antwortqualität.

---

### Volltextladen

Die ausgewählten Dokumente werden mit allen ihren Chunks aus Qdrant geladen
und zu vollständigen Texten zusammengesetzt. Das Token-Budget ist auf
**100 000 Tokens** begrenzt; Dokumente werden in Reihenfolge der Relevanz
so lange aufgenommen, bis das Budget erschöpft ist.

---

### LLM-Runde 2 — Antwortgenerierung

**Eingabe:** Vollständige Dokumententexte + Nutzeranfrage.

**Aufgabe:** Das Modell formuliert eine präzise Antwort auf Deutsch,
**ausschließlich** auf Basis der bereitgestellten Dokumente.

Das System-Prompt verpflichtet das Modell:
- Nur Informationen aus dem Kontext zu verwenden
- Fehlende Informationen explizit auszuweisen: *»Dazu liegen mir keine Informationen aus den offiziellen Dokumenten der HsH vor.«*
- Quellenangaben am Ende der Antwort zu nennen

---

### Kommunikation via MCP

Die Pipeline läuft auf einem **MCP-Server** (Model Context Protocol,
Spec 2025-03-26) und kommuniziert mit diesem Streamlit-Frontend über
das MCP-Protokoll (Streamable HTTP). Zwischenstatus-Meldungen werden
als SSE-Events (Server-Sent Events) live übertragen — daher erscheinen
die Fortschrittsschritte in Echtzeit.
"""
    )

st.divider()

# ---------------------------------------------------------------------------
# Gedanken / Think-Blöcke
# ---------------------------------------------------------------------------

st.header("💭 Gedanken des Modells")

st.markdown(
    """
Manche Modelle (z. B. *DeepSeek R1*) veröffentlichen ihren internen
Denkprozess als sog. **Reasoning / Chain-of-Thought**. Diese Gedanken
stehen im Rohtext zwischen `<think>` und `</think>`.

In dieser Oberfläche werden Gedanken **eingeklappt** unter dem Button
„💭 Gedanken" angezeigt — sie sind auf Wunsch lesbar, stören aber nicht
den Lesefluss der eigentlichen Antwort.

Die Gedanken werden **nicht** in der Wissensdatenbank gespeichert; in der
Antwort-Datenbank (`qa_log.jsonl`) sind sie mit dem Rohtext enthalten.
"""
)

st.divider()

# ---------------------------------------------------------------------------
# Modelle
# ---------------------------------------------------------------------------

st.header("🤖 Verfügbare Modelle")

st.markdown(
    """
Das System verwendet die **GWDG ChatAI API**
(`https://chat-ai.academiccloud.de/v1`) — eine datenschutzkonforme
KI-Infrastruktur des Göttinger Hochschulrechenzentrums für deutsche
Hochschulen. Alle Anfragen laufen mit `temperature=0.0` für reproduzierbare
Antworten. Die Modellliste wird live abgerufen und alle 5 Minuten aktualisiert.

> **MoE** (Mixture of Experts) = Das Modell hat viele Parameter gesamt,
> aktiviert aber pro Token nur einen Bruchteil davon — effizienter bei
> gleicher Ausgabequalität. „A3B" = 3 Milliarden *aktive* Parameter.
"""
)

with st.expander("⚡ Allzweck-Modelle", expanded=True):
    st.markdown(
        """
| Modell-ID | Hersteller | Parameter | Kontext | Stärken | Schwächen |
|-----------|------------|-----------|---------|---------|-----------|
| `llama-3.3-70b-instruct` | Meta | 70 B (dense) | 128 K | Sehr gute Allround-Qualität; solide Deutsch-Kenntnisse; **Empfehlung für den Normalbetrieb** | Langsamer als 8B-Modell |
| `meta-llama-3.1-8b-instruct` | Meta | 8 B (dense) | 128 K | Sehr schnell, ressourcenschonend; gut für kurze Anfragen | Schwächer bei komplexen Zusammenhängen und langen Kontexten |
| `qwen3.5-27b` | Alibaba | 27 B (dense) | 256 K | Starke Mehrsprachigkeit (201 Sprachen); sehr großes Kontextfenster; Thinking-Modus | Etwas langsamer als LLaMA 8B |
| `apertus-70b-instruct-2509` | Swiss AI (EPFL/ETH) | 70 B (dense) | 65 K | Vollständig offene Trainingsdaten; 1 800+ Sprachen; DSGVO-konformer Ursprung | Jüngeres Modell, weniger evaluiert als LLaMA |
"""
    )

with st.expander("🧠 Reasoning-Modelle (mit Denk-Chain)", expanded=False):
    st.markdown(
        """
Diese Modelle veröffentlichen ihren internen Denkprozess als `<think>`-Block
(sichtbar als „💭 Gedanken"-Expander in dieser App).

| Modell-ID | Hersteller | Parameter | Kontext | Stärken | Schwächen |
|-----------|------------|-----------|---------|---------|-----------|
| `deepseek-r1-distill-llama-70b` | DeepSeek → Meta | 70 B (dense) | 128 K | Sehr starkes Reasoning; Chain-of-Thought nachvollziehbar; gut für komplexe Sachfragen | Antworten länger durch Denkprozess; Thinking-Block kann störend wirken |
| `glm-4.7` | Zhipu AI | 355 B / **32 B aktiv** (MoE) | 200 K | Herausragendes Coding & Agentic-Reasoning; SWE-bench 73,8 %; interleaved Thinking | Chinesischer Hersteller; auf Coding optimiert, nicht auf Wissensbasis-RAG |
| `openai-gpt-oss-120b` | OpenAI | 120 B / **5,1 B aktiv** (MoE) | 128 K | Sehr starkes Reasoning (nahe o4-mini); Apache-2.0-Lizenz; effizient auf Single-GPU | Wenig Erfahrungswerte; noch neu (Aug. 2025) |
"""
    )

with st.expander("💻 Code-Modelle", expanded=False):
    st.markdown(
        """
Für den HsH-RAG-Chatbot weniger relevant — hier zur Vollständigkeit aufgeführt.

| Modell-ID | Hersteller | Parameter | Kontext | Stärken | Schwächen |
|-----------|------------|-----------|---------|---------|-----------|
| `devstral-2-123b-instruct-2512` | Mistral AI | 123 B (dense) | 256 K | Bestes Open-Source-Coding-Modell (SWE-bench 72,2 %); Multi-File-Edits; Agentic Coding | Stark auf Code optimiert; für Wissensfragen suboptimal |
| `qwen3-coder-30b-a3b-instruct` | Alibaba | 30 B / **3 B aktiv** (MoE) | 256 K | Schnell, code-spezialisiert; MoE-Effizienz | Für Allgemeinwissen schwächer als größere Modelle |
"""
    )

with st.expander("🌐 Große MoE-Modelle (Flaggschiff-Klasse)", expanded=False):
    st.markdown(
        """
| Modell-ID | Hersteller | Parameter | Kontext | Stärken | Schwächen |
|-----------|------------|-----------|---------|---------|-----------|
| `mistral-large-3-675b-instruct-2512` | Mistral AI | 675 B / **41 B aktiv** (MoE) | 256 K | Flaggschiff-Qualität; multimodal (Text + Bilder); Apache 2.0; sehr großes Kontextfenster | Hoher Rechenaufwand trotz MoE; Bildverarbeitung im HsH-RAG nicht genutzt |
| `qwen3.5-397b-a17b` | Alibaba | 397 B / **17 B aktiv** (MoE) | 256 K | Stärkste Qwen-3.5-Variante; 201 Sprachen; Thinking-Modus | Höchste Latenz in der Reihe |
| `qwen3.5-122b-a10b` | Alibaba | 122 B / **10 B aktiv** (MoE) | 256 K | Gute Balance aus Qualität und Geschwindigkeit; Thinking-Modus | MoE-Overhead bei kurzen Anfragen |
| `qwen3-30b-a3b-instruct-2507` | Alibaba | 30 B / **3 B aktiv** (MoE) | 256 K | Schnell; solide Allround-Qualität; 201 Sprachen | Schwächer als 122B/397B-Varianten |
| `qwen3.5-35b-a3b` | Alibaba | 35 B / **3 B aktiv** (MoE) | 256 K | Effizient; Thinking-Modus; 201 Sprachen | Geringere Qualität als 122B+ |
"""
    )

with st.expander("🖼️ Multimodale Modelle (Text + Bild)", expanded=False):
    st.markdown(
        """
Diese Modelle können Bilder verarbeiten — im HsH-RAG-System werden
Bilder aktuell nicht genutzt, die Modelle funktionieren aber auch rein textbasiert.

| Modell-ID | Hersteller | Parameter | Kontext | Stärken | Schwächen |
|-----------|------------|-----------|---------|---------|-----------|
| `internvl3.5-30b-a3b` | OpenGVLab | 30 B / **3 B aktiv** (MoE) | 128 K | Starkes Open-Source-Multimodal-Modell; effiziente Bildverarbeitung (256 Tokens/Bild) | Im reinen Textmodus kein Vorteil gegenüber Textmodellen |
| `qwen3-omni-30b-a3b-instruct` | Alibaba | 30 B / **3 B aktiv** (MoE) | 256 K | Multimodal (Text + Bild + Audio); 201 Sprachen | Für reine Textfragen überdimensioniert |
| `gemma-3-27b-it` | Google | 27 B (dense) | 128 K | Starke Mehrsprachigkeit; gute Bild-Verstehens-Fähigkeiten; von Google optimiert | Kein Reasoning-Modus |
"""
    )

with st.expander("🎯 Spezialisierte Modelle", expanded=False):
    st.markdown(
        """
| Modell-ID | Hersteller | Parameter | Kontext | Spezialisierung | Stärken | Schwächen |
|-----------|------------|-----------|---------|-----------------|---------|-----------|
| `medgemma-27b-it` | Google | 27 B (dense) | 128 K | **Medizin** | Auf medizinische Texte, Bildgebung (Radiologie, Pathologie) und EHR-Daten trainiert; 87,7 % auf MedQA | Für allgemeine HsH-Fragen unnötig spezialisiert |
| `teuken-7b-instruct-research` | OpenGPT-X (Fraunhofer/Jülich/TU Dresden/DFKI) | 7 B (dense) | 8 K | **EU-Mehrsprachigkeit** | Auf alle 24 EU-Amtssprachen optimiert; 4 T Token Training; deutsches Forschungsmodell; BMWK-gefördert | Kleinstes Modell; kurzes Kontextfenster; für komplexe RAG-Aufgaben zu schwach |
"""
    )

st.info(
    "**Empfehlung für den HsH-Assistenten:** `llama-3.3-70b-instruct` bietet die beste "
    "Balance aus Qualität, Geschwindigkeit und Deutschkenntnissen. Für komplexe "
    "Sachfragen lohnt sich `deepseek-r1-distill-llama-70b` mit sichtbarem Denkprozess."
)

st.divider()

# ---------------------------------------------------------------------------
# Datenschutz & Protokollierung
# ---------------------------------------------------------------------------

st.header("🔒 Datenschutz & Protokollierung")

st.markdown(
    """
### Was wird gespeichert?

Jede Anfrage wird in `qa_log.jsonl` protokolliert:

| Feld | Inhalt |
|------|--------|
| `ts` | Zeitstempel (UTC) |
| `frage` | Die Nutzeranfrage |
| `antwort` | Die generierte Antwort (inkl. etwaiger Gedanken) |
| `modell` | Verwendetes LLM |
| `fakultaet` | Gewählter Fakultätsfilter |
| `quellen` | Liste der verwendeten Dokumente |
| `tokens` | Verbrauchte Kontext-Token |

### Was wird nicht gespeichert?

- Keine Sitzungs-IDs oder Nutzeridentitäten
- Keine Cookies
- Keine Browserinformationen

### Wo werden die Daten verarbeitet?

Alle Anfragen gehen über die **GWDG ChatAI API** (Betreiber: GWDG Göttingen,
datenschutzkonform nach deutschem Recht). Die Wissensdatenbank (Qdrant) läuft
**lokal** auf dem Server `nemo.christianwill.de`.
"""
)

st.divider()

# ---------------------------------------------------------------------------
# Technischer Stack
# ---------------------------------------------------------------------------

st.header("🛠️ Technischer Stack")

col_a, col_b = st.columns(2)

with col_a:
    st.markdown(
        """
**Datenbeschaffung**
- [Crawl4AI](https://crawl4ai.com/) — asynchroner Web-Crawler
- [PyMuPDF4LLM](https://pymupdf.readthedocs.io/en/latest/pymupdf4llm/) — PDF → Markdown

**Vektordatenbank**
- [Qdrant](https://qdrant.tech/) — Vektordatenbank mit Named-Vectors-Support

**Embedding-Modelle**
- [Jina Embeddings v3](https://jina.ai/embeddings/) (1024-dim Dense)
- [BM25 via FastEmbed](https://github.com/qdrant/fastembed) (Sparse)
- [Jina Reranker v2](https://jina.ai/reranker/) (Cross-Encoder)
"""
    )

with col_b:
    st.markdown(
        """
**Server-Backend**
- [FastMCP](https://github.com/jlowin/fastmcp) — MCP-Server (Streamable HTTP)
- [OpenAI Python Client](https://github.com/openai/openai-python) → GWDG ChatAI API

**Frontend**
- [Streamlit](https://streamlit.io/) — dieses Web-Interface

**Datenformate**
- Wissensdatenbank: Markdown-Dateien + YAML-Frontmatter
- Vektorspeicher: Qdrant-Collection `hsh_knowledge`
- Protokoll: JSONL (`qa_log.jsonl`)
"""
    )

st.divider()
st.caption(
    "Dieses System wurde im Rahmen eines Projekts an der Hochschule Hannover entwickelt. "
    "Die Wissensbasis basiert ausschließlich auf öffentlich zugänglichen Seiten von hs-hannover.de."
)
