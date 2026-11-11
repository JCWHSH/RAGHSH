"""
Anreicherung: erzeugt je Chunk Fragen, die der Textabschnitt beantwortet.

Zweck
-----
Retrieval scheitert oft an der Formulierungslücke: Nutzer fragen "Wie melde ich
mich zurück?", auf der Seite steht "Die Rückmeldung erfolgt durch fristgerechte
Überweisung des Semesterbeitrags". Generierte Fragen schließen diese Lücke, weil
sie in Nutzersprache formuliert sind. Die Technik heißt doc2query bzw. document
expansion und hilft vor allem der BM25-Seite, weil sie dem Dokument Vokabular
hinzufügt, das darin nicht vorkommt.

Dieses Skript erzeugt die Fragen und legt sie ab. Es vektorisiert nichts und
fasst hsh_vectors.parquet nicht an. Die Verwendung im Index ist ein späterer,
getrennter Schritt.

Workflow:
    1. python enricher.py --input-dir testdaten --output enrichment_test.parquet --limit 50
       → Fragen durchlesen, Prompt anpassen, wiederholen
    2. python enricher.py                      → enrichment.parquet über alles
    3. hpc_vectorizer.py --enrichment enrichment.parquet

Join-Schlüssel
--------------
Die Zeilen sind über 'id' mit der Ausgabe des Vektorisierers verknüpft —
UUID5 aus source_url + chunk_index, erzeugt von chunking.make_point_id().
Beide Skripte müssen deshalb dieselbe chunking.py benutzen. Ändert sich dort
CHUNK_SIZE, CHUNK_OVERLAP oder HEADERS_TO_SPLIT, verschieben sich alle IDs und
die Anreicherung muss neu erzeugt werden.

Resumierbarkeit
---------------
Der Lauf über den Gesamtbestand dauert Stunden. Geschrieben wird deshalb in
Teildateien <output>.parts/part_00001.parquet — eine je FLUSH_ROWS Zeilen.
Beim Neustart werden bereits vorhandene IDs übersprungen. Am Ende fügt
--merge die Teile zu einer Datei zusammen; PyArrow kann das Verzeichnis aber
auch direkt als Dataset lesen.

Generierung
-----------
Zwei Backends, umschaltbar über --backend:

  vllm          Standard, sofern importierbar. Kontinuierliches Batching und
                Guided Decoding gegen ein JSON-Schema — die Antwort ist damit
                per Konstruktion ein Array aus genau N Strings. Für den
                Gesamtbestand die richtige Wahl.
  transformers  Gebatchte Generierung mit model.generate(). Braucht keine
                zusätzliche Abhängigkeit und reicht für Testläufe über einige
                hundert Chunks.

Beide rendern den Prompt über dasselbe render_chat(), damit ein Wechsel des
Backends die erzeugten Fragen nicht verändert.

Parallelbetrieb
---------------
--shard I --num-shards N bearbeitet nur einen Anteil des Bestands, aufgeteilt
über die Chunk-ID. Damit lässt sich der Lauf als Slurm-Array über mehrere GPUs
verteilen; jeder Shard schreibt eigene Teildateien (part_sII_NNNNN.parquet) und
kollidiert nicht mit den anderen. Zusammengefügt wird danach einmalig mit
--merge, nicht von den Array-Jobs selbst.

Ausgabeformat je Zeile:
    id            string
    source_url    string
    chunk_index   int32
    questions     list<string>   — 0 bis N Fragen, leer wenn nichts brauchbar kam
    model         string         — welches Modell die Fragen erzeugt hat
    ok            bool           — konnte die Antwort geparst werden?
    chunking_config_hash string  — Fingerabdruck der Chunk-GRENZEN; der
                                   Vektorisierer bricht beim Join ab, wenn er
                                   nicht zu seinem eigenen passt
    selection_hash string        — Fingerabdruck der Dedup-Auswahlregel; eine
                                   Abweichung verschiebt keine Grenze, sondern
                                   nur die Punktmenge — der Vektorisierer
                                   warnt dann, statt abzubrechen

Abhängigkeiten:
    pip install torch transformers pyarrow langchain-text-splitters
    pip install vllm            # für --backend vllm
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
import sys
from pathlib import Path

# Der Anreicherer lädt nie herunter — alle Modelle müssen vorher im Cache
# liegen. Explizit offline zu schalten verhindert, dass Bibliotheken bei
# Nebenprüfungen die HuggingFace-API ansprechen und auf einem Compute-Node
# ohne Internet abbrechen.
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import pyarrow as pa
import pyarrow.parquet as pq

from chunking import (config_hash, iter_chunks, list_markdown, make_point_id,
                      read_frontmatter, select_unique_ids, selection_hash)

CHUNKING_HASH  = config_hash()
SELECTION_HASH = selection_hash()

# ---------------------------------------------------------------------------
# Konfiguration
# ---------------------------------------------------------------------------

INGESTED_DIR = Path(__file__).parent / "data" / "ingested"
OUTPUT_FILE  = Path(__file__).parent / "enrichment.parquet"

MODEL_ID   = "Qwen/Qwen3.5-9B"
HF_CACHE   = os.environ.get("HF_HOME") or str(
    Path(__file__).resolve().parent.parent / "hf_cache"
)

QUESTIONS_PER_CHUNK = 3
GEN_BATCH_SIZE      = 16     # Prompts gleichzeitig durch das Modell
MAX_NEW_TOKENS      = 256    # drei kurze Fragen brauchen ~80; Reserve für
                             # den Fall, dass doch ein <think>-Block kommt
MAX_CHUNK_CHARS     = 2000   # Sicherheitsnetz gegen Ausreißer im Prompt
FLUSH_ROWS          = 500    # Zeilen je Teildatei

# Sehr kurze Chunks sind fast immer Navigationsreste, Überschriftenzeilen oder
# Datumsangaben. Fragen darauf sind wertlos und kosten trotzdem Rechenzeit.
MIN_CHUNK_CHARS = 200

SYSTEM_PROMPT = (
    "Du bist ein Assistent, der Suchanfragen für eine Hochschul-Website erzeugt."
)

USER_TEMPLATE = """Unten steht ein Abschnitt von der Website der Hochschule Hannover.

{context}
Formuliere genau {n} Fragen, die dieser Abschnitt beantwortet.

Regeln:
- Nutze den Seitentitel, um den Zusammenhang zu erkennen (Stellenausschreibung,
  Studiengang, Veranstaltung, Verwaltungsinformation, …) und formuliere die
  Fragen aus Sicht der Person, die diese Seite sucht.
- Beginne jede Frage mit einem Fragewort: Wer, Was, Wann, Wo, Wie, Welche,
  Warum, Wozu, Woran, An wen.
- Schreibe aus Sicht der suchenden Person ("Wo finde ich …", "Wie beantrage
  ich …"), niemals in der Sie-Form der Webseite.
- Benenne den Gegenstand konkret. Schreibe "Wie viele Credits hat MAT-201?",
  nicht "Wie viele Credits hat das Modul?". Formulierungen wie "das Modul",
  "die Veranstaltung" oder "der Studiengang" ohne Namen sind unbrauchbar,
  weil sie auf tausende andere Seiten genauso passen. Nimm den Namen aus
  Abschnitt, vorherigem Abschnitt oder Text.
- Verweise nie auf den Abschnitt zurück. "diese Stelle", "dieses Angebot",
  "diese Position", "diese Seite" sind verboten — wer sucht, kennt den
  Abschnitt noch nicht. Schreibe "Wie bewerbe ich mich auf die W2-Professur
  Angewandte Pflegewissenschaft?", nicht "… auf diese Stelle?".
- Der Dokumenttitel ist oft ein Dateiname. Verwende ihn nie als Namen eines
  Moduls oder einer Veranstaltung.
- Gib in jedem Fall {n} Fragen aus. Findest du keinen Namen, greife den
  konkreten Inhalt des Abschnitts auf — eine leere Antwort ist nie richtig.
- Jede Frage muss allein aus diesem Abschnitt beantwortbar sein.
- Formuliere so, wie jemand suchen würde — keine internen Kennziffern, keine
  Fragen nach URLs.
- Deutsch, je Frage höchstens 15 Wörter.
- Antworte ausschließlich mit einem JSON-Array aus {n} Strings.

Beispiel für das Ausgabeformat:
["Wann beginnt die Vorlesungszeit?", "Wo finde ich den Semesterplan?", "Wer berät zum Studienstart?"]

Abschnitt:
\"\"\"
{text}
\"\"\"
"""

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PARQUET_SCHEMA = pa.schema([
    pa.field("id",          pa.string()),
    pa.field("source_url",  pa.string()),
    pa.field("chunk_index", pa.int32()),
    pa.field("questions",   pa.list_(pa.string())),
    pa.field("section_heading", pa.string()),
    pa.field("raw",         pa.string()),
    pa.field("model",       pa.string()),
    pa.field("ok",          pa.bool_()),
    # Fingerabdruck der Chunk-Grenzen. Der Vektorisierer vergleicht ihn mit
    # seinem eigenen und bricht bei Abweichung ab — sonst würden Fragen
    # stillschweigend an falsche Textstellen geheftet.
    pa.field("chunking_config_hash", pa.string()),
    # Fingerabdruck der Auswahlregel (welche Chunks die Deduplizierung
    # behält). Getrennt geführt, weil eine Abweichung hier harmlos ist: Die
    # IDs zeigen weiterhin auf denselben Text, es fehlen höchstens Fragen für
    # neu hinzugekommene Chunks.
    pa.field("selection_hash", pa.string()),
])


# ---------------------------------------------------------------------------
# Antwort auswerten
# ---------------------------------------------------------------------------

# Zwei Varianten: erst gierig (erste '[' bis LETZTE ']'), dann genügsam.
# Die genügsame allein bricht beim ersten ']' ab — enthält eine Frage eine
# eckige Klammer, scheitert json.loads und der Fallback greift unnötig.
# Umgekehrt schluckt die gierige allein nachgestelltes Geschwätz des Modells.
_ARRAY_RE_GREEDY = re.compile(r"\[.*\]", re.DOTALL)
_ARRAY_RE_LAZY   = re.compile(r"\[.*?\]", re.DOTALL)

# Fragen, die mit einem dieser Wörter beginnen, sind Ja/Nein-Fragen. Sie sind
# als Suchanfrage wertlos, weil sie keinen Inhalt transportieren — und das
# Modell erzeugt sie trotz gegenteiliger Anweisung regelmäßig. Deshalb wird
# hier nachgeprüft statt nur im Prompt gebeten.
_YES_NO_START = {
    "ist", "sind", "war", "waren", "kann", "können", "kannst", "muss", "müssen",
    "darf", "dürfen", "gibt", "hat", "haben", "habe", "bietet", "bieten",
    "wird", "werden", "soll", "sollte", "sollen", "braucht", "benötigt",
    "besteht", "existiert", "erfüllt", "gilt",
}

# Rückverweisende Formulierungen ("Wo finde ich die Bewerbung für DIESE
# STELLE?"). Als Suchanfrage wertlos: Das Demonstrativpronomen zeigt auf den
# Chunk zurück, den die suchende Person noch gar nicht kennt — die Frage passt
# damit gleich gut auf jede andere Stellenausschreibung, und die erhöhte
# Dokumentfrequenz senkt zusätzlich das IDF dieser Terme.
#
# Gemessen am Lauf vom 13.08.2026: 5,5 % der Fragen, konzentriert auf
# Stellenausschreibungen. Der Prompt verbietet das bereits; wie bei den
# Ja/Nein-Fragen greift die Regel erst zuverlässig, wenn sie zusätzlich
# nachgeprüft wird.
#
# Bewusst nur Demonstrativpronomen: "das Modul" ist reparierbar, indem ein
# Name folgt ("Wie viele Credits hat das Modul MAT-201?") und darf deshalb
# nicht pauschal verworfen werden. "dieses Modul MAT-201" ist dagegen kein
# Deutsch — bei dies-* kann kein Name folgen, der die Frage rettet.
_VAGUE_REF_RE = re.compile(
    r"\bdies(?:e|er|es|em|en)\s+"
    r"(?:stelle|position|ausschreibung|angebot|modul|veranstaltung|"
    r"studiengang|studiengangs|projekt|abschnitt|seite|dokument|"
    r"einrichtung|kurs|seminar|vorlesung|programm)\b",
    re.IGNORECASE,
)

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def strip_thinking(raw: str) -> tuple[str, bool]:
    """Entfernt einen Reasoning-Block aus der Antwort.

    Qwen3.5 und Gemma 4 können vor der eigentlichen Antwort einen <think>-Block
    erzeugen. Ist er da, steht das JSON dahinter — und ohne Entfernung findet
    der Parser die Anführungszeichen aus dem Gedankengang statt die der Fragen.
    Ein unabgeschlossener Block bedeutet, dass max_new_tokens mitten im Denken
    endete; dann ist für die Antwort nichts übrig.
    """
    cleaned = _THINK_RE.sub("", raw)
    had_think = cleaned != raw
    if "<think>" in cleaned:                 # nicht geschlossen -> abgeschnitten
        cleaned = cleaned.split("<think>")[0]
        had_think = True
    return cleaned.strip(), had_think


_MIN_Q_LEN = 15
_MAX_Q_WORDS = 20


def _clean_question(raw: str) -> str | None:
    """Bereinigt eine einzelne Frage und verwirft sie, wenn sie nichts taugt.

    Die führenden Zeichen umfassen bewusst auch eckige Klammern: Antwortet das
    Modell mit einem nummerierten Array (`[1. "Wann …?", 2. …]`), scheitert
    JSON, der Zeilen-Fallback greift, und ohne diese Bereinigung landet der
    Klammer-Rest im Fragetext.
    """
    q = raw.strip().strip('",').lstrip("[]-*0123456789.) \"'").strip().strip('"[]').strip()
    if len(q) < _MIN_Q_LEN or not q.endswith("?"):
        return None
    words = q.split()
    if len(words) > _MAX_Q_WORDS:
        return None
    if words[0].lower().strip(",") in _YES_NO_START:
        return None
    if _VAGUE_REF_RE.search(q):
        return None
    return q


def build_context(chunk: dict) -> str:
    """Seitentitel und Abschnittsüberschrift für den Prompt.

    Ohne diesen Kontext rät das Modell den Dokumenttyp aus dem Absatz allein.
    Bei einer Stellenausschreibung führt das verlässlich zu Fragen wie "… um
    an der Hochschule Hannover studieren zu können?", weil der Absatz über
    Einstellungsvoraussetzungen für sich genommen auch eine Zugangsordnung
    sein könnte.
    """
    lines = []
    if chunk.get("title"):
        # Als "Dokument" statt "Seitentitel" bezeichnet: bei PDFs ist der Wert
        # ein Dateiname, und das Modell hat ihn als Modulnamen missbraucht
        # ("Qualifikationsziele des Moduls BVK_PO_2022_23").
        lines.append(f"Dokument: {chunk['title']}")
    if chunk.get("section_heading"):
        lines.append(f"Abschnitt: {chunk['section_heading']}")
    # Generische Unterabschnitte ("Literatur", "Angestrebte Lernergebnisse")
    # tragen den Modulnamen nicht. Der steht dann in einem früheren Abschnitt.
    # Nur, wenn er sich von der eigenen unterscheidet: bei mehreren Chunks
    # desselben Abschnitts stünde sonst zweimal dieselbe Zeile im Prompt.
    prev = chunk.get("prev_heading", "")
    if prev and prev != chunk.get("section_heading", "") and not chunk.get("_ohne_prev"):
        lines.append(f"Vorheriger Abschnitt: {prev}")
    return ("\n".join(lines) + "\n") if lines else ""


def parse_questions(raw: str, expected: int) -> list[str]:
    """Holt die Fragenliste aus der Modellantwort.

    Ohne Guided Decoding ist die Ausgabe nicht garantiert wohlgeformt. Deshalb
    zwei Stufen: erst das erste JSON-Array im Text, sonst zeilenweise. Jede
    Frage durchläuft anschließend _clean_question; was übrig bleibt, ist
    verwendbar. Bleibt nichts übrig, wird die Zeile mit ok=False markiert —
    nicht stillschweigend verworfen, damit du die Quote siehst.
    """
    candidates: list[str] = []

    for pattern in (_ARRAY_RE_GREEDY, _ARRAY_RE_LAZY):
        match = pattern.search(raw)
        if not match:
            continue
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            continue
        if isinstance(data, list) and data:
            candidates = [str(q) for q in data]
            break

    if not candidates:
        # Zwischenstufe für kaputtes JSON: die Strings einzeln herausziehen.
        # Deckt genau den beobachteten Fall ab, in dem das Modell die Elemente
        # innerhalb des Arrays nummeriert (`[1. "…", 2. "…"]`) — dann scheitert
        # json.loads, die Anführungszeichen stehen aber korrekt.
        quoted = re.findall(r'"([^"]{10,})"', raw)
        if quoted:
            candidates = quoted

    if not candidates:
        candidates = raw.splitlines()

    out: list[str] = []
    seen: set[str] = set()
    for cand in candidates:
        q = _clean_question(cand)
        if q and q.lower() not in seen:
            seen.add(q.lower())
            out.append(q)
    return out[:expected]


# ---------------------------------------------------------------------------
# Generierung (hier gegen vLLM austauschbar)
# ---------------------------------------------------------------------------


def resolve_local_model(model_id: str, cache_dir: str) -> str:
    """Übersetzt eine HF-Repo-ID in den lokalen Snapshot-Pfad.

    Nötig wegen eines Fehlers in transformers 5.x: Beim Laden des Tokenizers
    prüft _patch_mistral_regex(), ob ein Mistral-Basismodell vorliegt, und ruft
    dafür die HuggingFace-API auf — auch bei local_files_only=True. Auf einem
    Compute-Node ohne Internet endet das mit 'Network is unreachable'.

    Die Prüfung beginnt mit 'is_local or …'. Bekommt from_pretrained() einen
    existierenden Pfad statt einer Repo-ID, ist is_local wahr und der
    Netzwerkaufruf unterbleibt.
    """
    if Path(model_id).exists():
        return model_id
    try:
        from huggingface_hub import snapshot_download
        pfad = snapshot_download(model_id, cache_dir=cache_dir, local_files_only=True)
        logger.info("Lokaler Snapshot: %s", pfad)
        return pfad
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Kein lokaler Snapshot für '%s' auffindbar (%s). Es wird die Repo-ID "
            "verwendet — das kann auf einem Node ohne Internet fehlschlagen.",
            model_id, exc)
        return model_id


_TEMPLATE_WARNED = False


def render_chat(tokenizer, prompt: str, enable_thinking: bool) -> str:
    """Wendet die Chat-Vorlage an — gemeinsam für beide Backends.

    Die Anweisung steht in einer echten 'system'-Rolle statt vorne im
    Nutzertext. Instruct-Modelle gewichten sie dort stärker, und die Vorlage
    setzt sie an die Stelle, für die das Modell trainiert wurde.

    enable_thinking wird als Vorlagen-Parameter durchgereicht. Vorlagen, die
    ihn nicht kennen, ignorieren ihn stillschweigend — deshalb zählt der
    Hauptlauf zusätzlich, ob trotzdem ein <think>-Block ankommt.
    """
    global _TEMPLATE_WARNED
    msgs = [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt}]
    try:
        return tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
    except TypeError:
        if not _TEMPLATE_WARNED:
            logger.warning("Chat-Vorlage kennt 'enable_thinking' nicht — ignoriert.")
            _TEMPLATE_WARNED = True
        return tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True)


# JSON-Schema für Guided Decoding (nur vLLM). Erzwingt genau N Strings —
# damit ist die Ausgabe per Konstruktion wohlgeformt und die Fallback-Kaskade
# in parse_questions() nur noch Absicherung.
def questions_json_schema(n: int) -> dict:
    # Kein "uniqueItems": True. Zweimal geprüft und zweimal verworfen:
    #
    # 1. xgrammar implementiert das Schlüsselwort nicht. Der Lauf bricht ab
    #    mit `Grammar error: Unimplemented keys: ["uniqueItems"]` — und zwar
    #    erst bei der Generierung, nicht beim Bau der SamplingParams. Die
    #    Fallback-Kaskade in _sampling_params() greift dort nicht mehr.
    # 2. Es wäre ohnehin wirkungslos: parse_questions() verwirft wortgleiche
    #    Fragen bereits über das 'seen'-Set.
    return {
        "type": "array",
        "items": {"type": "string"},
        "minItems": n,
        "maxItems": n,
    }


class TransformersGenerator:
    """Gebatchte Generierung mit transformers.

    Batchgröße 1 — wie in einem einfachen Testskript — verschenkt auf einer
    A100/H100 fast die gesamte Leistung. Entscheidend ist Left-Padding: bei
    Decoder-Modellen muss die Auffüllung vorne stehen, sonst generiert das
    Modell hinter dem Padding weiter und die Ausgabe wird Unsinn.
    """

    def __init__(self, model_id: str, cache_dir: str, dtype: str = "bfloat16",
                 enable_thinking: bool = False, temperature: float = 0.0,
                 top_p: float = 0.8, top_k: int = 20, allow_cpu: bool = False):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.model_id = model_id
        self.enable_thinking = enable_thinking
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k

        logger.info("Lade Modell '%s' (Cache: %s) …", model_id, cache_dir)
        quelle = resolve_local_model(model_id, cache_dir)

        self.tokenizer = AutoTokenizer.from_pretrained(
            quelle, cache_dir=cache_dir, local_files_only=True,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        # transformers 5.x hat torch_dtype in dtype umbenannt; für ältere
        # Versionen fällt der Aufruf auf den alten Namen zurück.
        common = dict(cache_dir=cache_dir, local_files_only=True,
                      device_map="auto", trust_remote_code=True)

        def _load(cls):
            try:
                return cls.from_pretrained(quelle, dtype=getattr(torch, dtype), **common)
            except TypeError:
                return cls.from_pretrained(quelle, torch_dtype=getattr(torch, dtype), **common)

        try:
            self.model = _load(AutoModelForCausalLM)
        except (ValueError, KeyError, OSError) as exc:
            # Qwen3.5 und Gemma 4 sind Vision-Language-Modelle. Ihre Konfiguration
            # ist bei AutoModelForCausalLM nicht zwingend registriert; die
            # Text-Generierung funktioniert über die ImageTextToText-Klasse
            # trotzdem, weil wir nur Text hineingeben.
            logger.info("AutoModelForCausalLM nicht nutzbar (%s) — versuche "
                        "AutoModelForImageTextToText.", exc)
            from transformers import AutoModelForImageTextToText
            self.model = _load(AutoModelForImageTextToText)

        self.model.eval()

        # device_map='auto' lagert Layer nach CPU oder Disk aus, wenn der VRAM
        # nicht reicht. Der Job LÄUFT dann — nur um Größenordnungen langsamer,
        # und das Zeitlimit reißt mitten im Bestand. Gegenstück zu --require-gpu
        # im Vektorisierer: lieber sofort abbrechen.
        dmap = getattr(self.model, "hf_device_map", None) or {}
        ausgelagert = sorted({str(d) for d in dmap.values()
                              if str(d) in ("cpu", "disk")})
        if ausgelagert:
            raise RuntimeError(
                f"Modell teilweise auf {', '.join(ausgelagert)} ausgelagert — der "
                f"Lauf wäre um Größenordnungen langsamer. Kleineres Modell, "
                f"Quantisierung oder mehr GPU-Speicher anfordern. Karte: {dmap}"
            )
        # Maßgeblich ist, wo die Gewichte TATSÄCHLICH liegen, nicht was die
        # Geräte-Karte sagt: Ohne CUDA schaltet accelerate sich gar nicht ein,
        # hf_device_map fehlt dann komplett, und eine Prüfung, die nur auf
        # diese Karte schaut, lässt ausgerechnet den Fall 'gar keine GPU'
        # durch. Genau so ist ein Testlauf versehentlich auf dem Login-Node
        # gestartet — mit einem 9B-Modell auf der CPU dauert das Tage.
        device = next(self.model.parameters()).device
        if device.type != "cuda" and not allow_cpu:
            raise RuntimeError(
                f"Modell liegt auf '{device}', nicht auf der GPU. Läuft das hier "
                f"auf dem Login-Node? Erst eine GPU anfordern:\n"
                f"    srun --partition=kisski --gres=gpu:A100:1 --cpus-per-task=8 "
                f"--mem=64G --time=01:00:00 --pty bash\n"
                f"Bewusst auf der CPU rechnen: --allow-cpu (nur für winzige "
                f"Stichproben sinnvoll)."
            )

        logger.info("Modell bereit auf %s. Denkmodus: %s, Sampling: %s",
                    device,
                    "an" if enable_thinking else "aus",
                    "greedy" if temperature <= 0 else
                    f"T={temperature} top_p={top_p} top_k={top_k}")

    def generate_batch(self, prompts: list[str], max_new_tokens: int) -> list[str]:
        texts = [render_chat(self.tokenizer, p, self.enable_thinking) for p in prompts]
        enc = self.tokenizer(texts, return_tensors="pt", padding=True,
                             truncation=True, max_length=8192).to(self.model.device)

        gen_kwargs = dict(max_new_tokens=max_new_tokens,
                          pad_token_id=self.tokenizer.pad_token_id)
        if self.temperature and self.temperature > 0:
            # Qwen empfiehlt für den Nicht-Denk-Modus Sampling; reines Greedy
            # neigt bei dieser Modellfamilie zu Wiederholungsschleifen.
            gen_kwargs.update(do_sample=True, temperature=self.temperature,
                              top_p=self.top_p, top_k=self.top_k)
        else:
            gen_kwargs.update(do_sample=False)   # reproduzierbar

        with self.torch.inference_mode():
            out = self.model.generate(**enc, **gen_kwargs)
        gen = out[:, enc["input_ids"].shape[1]:]
        return self.tokenizer.batch_decode(gen, skip_special_tokens=True)


class VLLMGenerator:
    """Generierung mit vLLM — kontinuierliches Batching und Guided Decoding.

    Warum überhaupt: transformers verarbeitet einen Batch im Gleichschritt. Der
    längste Prompt bestimmt die Padding-Kosten für alle, und fertige Sequenzen
    warten auf die langsamste. vLLM schiebt fertige Anfragen aus dem Batch und
    zieht neue nach; bei kurzen Prompts und kurzen Ausgaben wie hier ist das
    der Unterschied zwischen Stunden und einer Stunde.

    Guided Decoding ist der zweite Grund: Das JSON-Schema wird während des
    Dekodierens erzwungen, die Antwort ist also per Konstruktion ein Array aus
    genau N Strings. Die Ausfallquote 'ohne Fragen' geht damit gegen null.

    ACHTUNG, versionsabhängig: Der Parametername für Guided Decoding hat
    zwischen vLLM-Versionen gewechselt (guided_json → guided_decoding=
    GuidedDecodingParams → structured_outputs=StructuredOutputsParams).
    _sampling_params() probiert die Varianten durch und arbeitet notfalls ohne
    — dann greift wieder die Parse-Kaskade. Was tatsächlich verwendet wird,
    steht im Log.

    Die Prompts werden mit demselben render_chat() aufbereitet wie im
    Transformers-Pfad, damit ein Backend-Wechsel die Fragen nicht verändert.
    """

    def __init__(self, model_id: str, cache_dir: str, dtype: str = "bfloat16",
                 enable_thinking: bool = False, temperature: float = 0.0,
                 top_p: float = 0.8, top_k: int = 20,
                 questions: int = QUESTIONS_PER_CHUNK,
                 max_model_len: int = 4096,
                 gpu_memory_utilization: float = 0.90):
        # MUSS vor dem vllm-Import gesetzt werden: vLLM startet den EngineCore
        # als Unterprozess und wählt dafür standardmäßig fork. Ist im
        # Elternprozess bereits ein CUDA-Kontext angelegt, scheitert das
        # Kind mit "Cannot re-initialize CUDA in forked subprocess" — ein
        # geforkter Prozess erbt den Kontext, kann ihn aber nicht neu
        # initialisieren. spawn startet ein frisches Python und umgeht das.
        # Zulässig nur, weil enricher.py einen __main__-Guard hat; ohne den
        # würde spawn beim Reimport das ganze Skript erneut ausführen.
        os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

        from transformers import AutoTokenizer
        from vllm import LLM

        self.enable_thinking = enable_thinking
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.questions = questions
        self.model_id = model_id
        self._sp_mode: str | None = None

        quelle = resolve_local_model(model_id, cache_dir)
        logger.info("Lade Modell '%s' mit vLLM …", quelle)

        self.tokenizer = AutoTokenizer.from_pretrained(
            quelle, cache_dir=cache_dir, local_files_only=True,
            trust_remote_code=True,
        )
        self.llm = LLM(
            model=quelle,
            dtype=dtype,
            trust_remote_code=True,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
        )
        logger.info("vLLM bereit. Denkmodus: %s, Sampling: %s",
                    "an" if enable_thinking else "aus",
                    "greedy" if temperature <= 0 else
                    f"T={temperature} top_p={top_p} top_k={top_k}")

    def _sampling_params(self, max_new_tokens: int):
        from vllm import SamplingParams

        base = dict(max_tokens=max_new_tokens)
        if self.temperature and self.temperature > 0:
            base.update(temperature=self.temperature, top_p=self.top_p, top_k=self.top_k)
        else:
            base.update(temperature=0.0)

        schema = questions_json_schema(self.questions)

        # Reihenfolge: neueste API zuerst.
        versuche = []
        try:
            from vllm.sampling_params import StructuredOutputsParams
            versuche.append(("structured_outputs",
                             lambda: dict(structured_outputs=StructuredOutputsParams(json=schema))))
        except ImportError:
            pass
        try:
            from vllm.sampling_params import GuidedDecodingParams
            versuche.append(("guided_decoding",
                             lambda: dict(guided_decoding=GuidedDecodingParams(json=schema))))
        except ImportError:
            pass
        versuche.append(("guided_json", lambda: dict(guided_json=schema)))

        for name, bauen in versuche:
            try:
                sp = SamplingParams(**base, **bauen())
            except (TypeError, ValueError) as exc:
                logger.debug("Guided-Decoding-Variante '%s' nicht nutzbar: %s", name, exc)
                continue
            if self._sp_mode != name:
                logger.info("Guided Decoding aktiv über '%s'.", name)
                self._sp_mode = name
            return sp

        if self._sp_mode != "aus":
            logger.warning(
                "Keine Guided-Decoding-Variante dieser vLLM-Version nutzbar — es "
                "wird frei generiert. Die Ausgabe durchläuft weiterhin "
                "parse_questions(); beobachte die Quote 'ohne Fragen'.")
            self._sp_mode = "aus"
        return SamplingParams(**base)

    def generate_batch(self, prompts: list[str], max_new_tokens: int) -> list[str]:
        texts = [render_chat(self.tokenizer, p, self.enable_thinking) for p in prompts]
        sp = self._sampling_params(max_new_tokens)
        outs = self.llm.generate(texts, sp)
        # vLLM gibt die Ergebnisse nicht zwingend in Eingabereihenfolge zurück;
        # request_id ist der laufende Index als String. Ohne diese Sortierung
        # bekämen Chunks die Fragen ihrer Nachbarn — lautlos.
        try:
            outs = sorted(outs, key=lambda o: int(o.request_id))
        except (AttributeError, ValueError):
            logger.debug("request_id nicht sortierbar — Reihenfolge wie geliefert.")
        return [o.outputs[0].text for o in outs]


def make_generator(args):
    """Wählt das Backend. 'auto' nimmt vLLM, wenn es importierbar ist."""
    backend = args.backend
    if backend == "auto":
        try:
            import vllm  # noqa: F401
            backend = "vllm"
        except ImportError:
            backend = "transformers"
            logger.info("vLLM nicht installiert — Backend 'transformers'. "
                        "Für den Gesamtbestand lohnt die Installation deutlich.")

    if backend == "vllm":
        return VLLMGenerator(
            args.model, args.cache_dir,
            enable_thinking=args.thinking, temperature=args.temperature,
            top_p=args.top_p, top_k=args.top_k, questions=args.questions,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_memory_utilization,
        )
    return TransformersGenerator(
        args.model, args.cache_dir,
        enable_thinking=args.thinking, temperature=args.temperature,
        top_p=args.top_p, top_k=args.top_k, allow_cpu=args.allow_cpu,
    )


# ---------------------------------------------------------------------------
# Bereits erledigte IDs einlesen
# ---------------------------------------------------------------------------


def load_done_ids(parts_dir: Path) -> set[str]:
    """IDs aus vorhandenen Teildateien — Grundlage für den Wiederanlauf."""
    if not parts_dir.is_dir():
        return set()
    done: set[str] = set()
    for part in sorted(parts_dir.glob("part_*.parquet")):
        try:
            done.update(pq.read_table(part, columns=["id"])["id"].to_pylist())
        except Exception as exc:  # noqa: BLE001
            logger.warning("Teildatei %s nicht lesbar (%s) — ignoriert", part.name, exc)
    if done:
        logger.info("Wiederanlauf: %d Chunk(s) bereits angereichert.", len(done))
    return done


def part_pattern(shard: int) -> str:
    """Dateimuster der Teildateien EINES Shards.

    Der Shard steckt im Namen, damit parallele Array-Jobs sich nicht
    gegenseitig überschreiben: Der laufende Index wird aus dem Verzeichnis
    gezählt, und ohne Trennung vergäben zwei Jobs dieselbe Nummer.
    """
    return f"part_s{shard:02d}_*.parquet"


def write_part(rows: list[dict], parts_dir: Path, index: int, shard: int = 0) -> Path:
    """Schreibt eine Teildatei atomar (erst .partial, dann umbenennen)."""
    parts_dir.mkdir(parents=True, exist_ok=True)
    target = parts_dir / f"part_s{shard:02d}_{index:05d}.parquet"
    tmp = target.with_suffix(".parquet.partial")
    table = pa.Table.from_pylist(rows, schema=PARQUET_SCHEMA)
    pq.write_table(table, tmp, compression="zstd", compression_level=3)
    os.replace(tmp, target)
    return target


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Erzeugt je Chunk Fragen, die der Abschnitt beantwortet.")
    p.add_argument("--input-dir", type=Path, default=INGESTED_DIR,
                   help=f"Quellverzeichnis (Standard: {INGESTED_DIR.name}).")
    p.add_argument("--output", type=Path, default=OUTPUT_FILE,
                   help=f"Zieldatei (Standard: {OUTPUT_FILE.name}). Teildateien "
                        "landen in <output>.parts/.")
    p.add_argument("--model", default=MODEL_ID, help="HF-Modell-ID.")
    p.add_argument("--cache-dir", default=HF_CACHE, help="HuggingFace-Cache.")
    p.add_argument("--batch-size", type=int, default=GEN_BATCH_SIZE,
                   help="Prompts gleichzeitig durch das Modell. Bei OOM verkleinern.")
    p.add_argument("--questions", type=int, default=QUESTIONS_PER_CHUNK,
                   help="Fragen je Chunk.")
    p.add_argument("--limit", type=int, default=None, metavar="N",
                   help="Nach N Chunks aufhören — in Dateireihenfolge, also meist "
                        "wenige Seiten. Zum Beurteilen der Qualität --sample nehmen.")
    p.add_argument("--sample", type=int, default=None, metavar="N",
                   help="N zufällige Chunks aus dem gesamten Bestand statt der "
                        "ersten N. Für die Qualitätsbeurteilung: --limit zeigt "
                        "sonst immer dieselben paar Seiten.")
    p.add_argument("--seed", type=int, default=42,
                   help="Zufallsstartwert für --sample. Gleicher Wert = gleiche "
                        "Stichprobe, damit Prompt-Änderungen vergleichbar bleiben.")
    p.add_argument("--min-chars", type=int, default=MIN_CHUNK_CHARS,
                   help="Chunks unter dieser Länge überspringen (Navigationsreste).")
    p.add_argument("--no-prev-heading", action="store_true",
                   help="Die Überschrift des vorherigen Abschnitts NICHT in den "
                        "Prompt geben. Für den A/B-Vergleich: sie hilft bei "
                        "generischen Unterabschnitten, kann aber den Namen eines "
                        "fremden Moduls einschleppen.")
    p.add_argument("--thinking", action="store_true",
                   help="Denkmodus des Modells aktivieren (Qwen3.5, Gemma 4). "
                        "Standard: aus — der <think>-Block kostet nur Tokens.")
    p.add_argument("--temperature", type=float, default=0.0,
                   help="0 = greedy und reproduzierbar (Standard). Qwen empfiehlt "
                        "für den Nicht-Denk-Modus 0.7; bei Wiederholungsschleifen "
                        "hochsetzen.")
    p.add_argument("--top-p", type=float, default=0.8, help="Nur bei temperature > 0.")
    p.add_argument("--top-k", type=int, default=20, help="Nur bei temperature > 0.")
    p.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS,
                   help=f"Ausgabelänge je Chunk (Standard: {MAX_NEW_TOKENS}). Mit "
                        "--thinking deutlich erhöhen, sonst endet die Ausgabe im "
                        "Gedankengang.")
    p.add_argument("--backend", choices=("auto", "vllm", "transformers"), default="auto",
                   help="Generierungs-Backend. 'auto' nimmt vLLM, wenn es "
                        "importierbar ist (Standard). vLLM ist für den "
                        "Gesamtbestand deutlich schneller und erzwingt das "
                        "Ausgabeformat per Guided Decoding.")
    p.add_argument("--max-model-len", type=int, default=4096,
                   help="Nur vLLM: Kontextlänge, die reserviert wird. Prompt "
                        "plus Ausgabe bleiben hier weit darunter; kleinere "
                        "Werte lassen mehr Platz für den KV-Cache.")
    p.add_argument("--gpu-memory-utilization", type=float, default=0.90,
                   help="Nur vLLM: Anteil des VRAM für Gewichte und KV-Cache.")
    p.add_argument("--shard", type=int, default=0, metavar="I",
                   help="Diesen Anteil des Bestands bearbeiten (0 … N-1). Für "
                        "Slurm-Array-Jobs über mehrere GPUs.")
    p.add_argument("--num-shards", type=int, default=1, metavar="N",
                   help="Anzahl paralleler Anteile. Bei N>1 wird am Ende NICHT "
                        "automatisch zusammengefügt — das erledigt ein "
                        "abschließender Lauf mit --merge.")
    p.add_argument("--allow-cpu", action="store_true",
                   help="Auch ohne GPU rechnen. Standard ist Abbruch: ein 9B-"
                        "Modell auf der CPU braucht für den Bestand Tage, und "
                        "der häufigste Grund dafür ist ein Start auf dem "
                        "Login-Node statt in einer srun-Sitzung.")
    p.add_argument("--no-length-sort", action="store_true",
                   help="Chunks NICHT nach Länge sortiert batchen. Die "
                        "Sortierung spart Padding (nur beim Transformers-"
                        "Backend relevant) und ändert nichts am Ergebnis, weil "
                        "der Join über 'id' läuft.")
    p.add_argument("--urls", type=Path, metavar="FILE", default=None,
                   help="Delta-Betrieb: nur Chunks dieser source_urls anreichern "
                        "(eine pro Zeile, # = Kommentar). Die Deduplizierung "
                        "läuft trotzdem über das ganze --input-dir. Verlangt "
                        "ein eigenes --output, z.B. enrichment_delta.parquet.")
    p.add_argument("--merge", action="store_true",
                   help="Nur die vorhandenen Teildateien zu --output zusammenfügen "
                        "und beenden. Kein Modell wird geladen.")
    p.add_argument("--dry-run", action="store_true",
                   help="Nur zählen und einen Beispiel-Prompt zeigen. Kein Modell.")
    return p


def merge_parts(parts_dir: Path, out_file: Path) -> None:
    parts = sorted(parts_dir.glob("part_*.parquet"))
    if not parts:
        logger.error("Keine Teildateien in %s", parts_dir)
        sys.exit(1)
    table = pa.concat_tables([pq.read_table(p) for p in parts])
    tmp = out_file.with_name(out_file.name + ".partial")
    pq.write_table(table, tmp, compression="zstd", compression_level=3)
    os.replace(tmp, out_file)
    logger.info("%d Teildatei(en) → %s  (%d Zeilen)", len(parts), out_file, table.num_rows)


def main() -> None:
    args = build_arg_parser().parse_args()
    parts_dir = args.output.with_name(args.output.name + ".parts")

    if args.merge:
        merge_parts(parts_dir, args.output)
        return

    if not 0 <= args.shard < args.num_shards:
        logger.error("--shard %d liegt außerhalb von 0 … %d.",
                     args.shard, args.num_shards - 1)
        sys.exit(1)

    logger.info("Fingerabdruck Grenzen: %s | Auswahl: %s",
                CHUNKING_HASH, SELECTION_HASH)

    md_files = list_markdown(args.input_dir)
    if not md_files:
        logger.error("Keine .md-Dateien in %s", args.input_dir)
        sys.exit(1)
    logger.info("Gefunden: %d .md-Datei(en) in %s", len(md_files), args.input_dir)

    done = load_done_ids(parts_dir)

    # ── Globale Deduplizierung ────────────────────────────────────────────
    # Über die VOLLE Dateiliste, bevor --shard, --sample oder --limit greifen.
    # select_unique_ids() wählt den Vertreter in Dateireihenfolge; eine
    # Teilmenge würde eine andere Wahl treffen und die IDs gegenüber dem
    # Vektorisierer verschieben.
    keep_ids = select_unique_ids(md_files)
    if keep_ids is not None:
        logger.info("Deduplizierung: %d eindeutige Chunk(s) behalten.", len(keep_ids))

    # ── Delta-Betrieb (--urls, seit 11.11.2026) ──────────────────────────
    # Erst NACH select_unique_ids() auf die Liste einschränken, aus demselben
    # Grund wie im Vektorisierer: Über eine Teilmenge berechnet, fiele die Wahl
    # der Gruppenvertreter anders aus, und die Fragen hingen an Punkten, die
    # der Vektorisierer gar nicht behält.
    if args.urls:
        if args.output.resolve() == OUTPUT_FILE.resolve():
            logger.error("--urls zusammen mit der Standard-Ausgabe %s würde die "
                         "Gesamtanreicherung durch das Delta ersetzen. Bitte "
                         "--output setzen, z.B. enrichment_delta.parquet.",
                         OUTPUT_FILE.name)
            sys.exit(1)
        wanted = {z.strip() for z in args.urls.read_text(encoding="utf-8").splitlines()
                  if z.strip() and not z.strip().startswith("#")}
        md_files = [f for f in md_files
                    if read_frontmatter(f).get("source_url", "") in wanted]
        logger.info("Delta-Betrieb: %d von %d URL(s) der Liste im Verzeichnis gefunden.",
                    len(md_files), len(wanted))
        if not md_files:
            logger.error("Keine der gelisteten URLs liegt in %s — abgebrochen.",
                         args.input_dir)
            sys.exit(1)

    # ── Arbeitsvorrat einsammeln ──────────────────────────────────────────
    todo: list[dict] = []
    skipped_short = 0
    skipped_dup = 0
    for _path, chunks in iter_chunks(md_files):
        prev_heading = ""
        for c in chunks:
            eigene = c.get("section_heading", "")
            if len(c["text"]) < args.min_chars:
                skipped_short += 1
                if eigene and eigene != prev_heading:
                    prev_heading = eigene
                continue
            cid = make_point_id(c["source_url"], c["chunk_index"])
            # Nach der Längenprüfung, aber vor dem Wiederanlauf-Filter: Die
            # Fortschreibung von prev_heading läuft in beiden Fällen weiter,
            # sonst verlöre der nächste Chunk seinen Abschnittsbezug.
            if keep_ids is not None and cid not in keep_ids:
                skipped_dup += 1
                if eigene and eigene != prev_heading:
                    prev_heading = eigene
                continue
            if cid in done:
                if eigene and eigene != prev_heading:
                    prev_heading = eigene
                continue
            todo.append({"id": cid, "source_url": c["source_url"],
                         "chunk_index": c["chunk_index"], "text": c["text"],
                         "title": c.get("title", ""),
                         "section_heading": c.get("section_heading", ""),
                         "prev_heading": prev_heading,
                         "_ohne_prev": args.no_prev_heading})
            if eigene and eigene != prev_heading:
                prev_heading = eigene
            if args.limit and not args.sample and len(todo) >= args.limit:
                break
        if args.limit and not args.sample and len(todo) >= args.limit:
            break

    if args.sample and len(todo) > args.sample:
        # Bewusst über den GESAMTEN Bestand ziehen und erst danach kürzen:
        # --limit liefert die ersten N Chunks in Dateireihenfolge, also meist
        # zwei oder drei Seiten desselben Typs. Für eine Qualitätsaussage über
        # 200k heterogene Chunks ist das keine Stichprobe.
        rng = random.Random(args.seed)
        todo = rng.sample(todo, args.sample)
        todo.sort(key=lambda c: (c["source_url"], c["chunk_index"]))
        logger.info("Stichprobe: %d zufällige Chunk(s), Startwert %d",
                    len(todo), args.seed)

    if args.num_shards > 1:
        # Aufteilung über die ID, nicht über die Position: so bleibt die
        # Zuordnung Chunk → Shard stabil, auch wenn beim Wiederanlauf schon
        # erledigte Chunks fehlen. Eine positionsbasierte Aufteilung würde
        # nach jedem Neustart andere Chunks in andere Shards schieben.
        vorher = len(todo)
        todo = [c for c in todo
                if int(c["id"][-8:], 16) % args.num_shards == args.shard]
        logger.info("Shard %d von %d: %d von %d Chunk(s).",
                    args.shard, args.num_shards, len(todo), vorher)

    if not args.no_length_sort:
        # Ähnlich lange Prompts in einen Batch. Beim Transformers-Backend
        # bestimmt der längste Chunk die Padding-Kosten des gesamten Batches;
        # gemischt lang zu batchen verschenkt hier den größten Teil.
        todo.sort(key=lambda c: len(c["text"]))

    logger.info("Zu verarbeiten: %d Chunk(s)  (zu kurz übersprungen: %d, "
                "Duplikate übersprungen: %d)",
                len(todo), skipped_short, skipped_dup)
    if not todo:
        logger.info("Nichts zu tun.")
        return

    if args.dry_run:
        print("\n─── Beispiel-Prompt ───\n")
        print(USER_TEMPLATE.format(n=args.questions,
                                   context=build_context(todo[0]),
                                   text=todo[0]["text"][:MAX_CHUNK_CHARS]))
        return

    generator = make_generator(args)

    rows: list[dict] = []
    part_index = (len(list(parts_dir.glob(part_pattern(args.shard))))
                  if parts_dir.is_dir() else 0)
    failed = 0
    thinking_seen = 0

    for start in range(0, len(todo), args.batch_size):
        batch = todo[start:start + args.batch_size]
        prompts = [
            USER_TEMPLATE.format(n=args.questions, context=build_context(c),
                                 text=c["text"][:MAX_CHUNK_CHARS])
            for c in batch
        ]
        answers = generator.generate_batch(prompts, args.max_new_tokens)

        for chunk, answer in zip(batch, answers):
            answer, had_think = strip_thinking(answer)
            if had_think:
                thinking_seen += 1
            questions = parse_questions(answer, args.questions)
            if not questions:
                failed += 1
            rows.append({
                "id":          chunk["id"],
                "source_url":  chunk["source_url"],
                "chunk_index": chunk["chunk_index"],
                "questions":   questions,
                # Diagnose: Ohne section_heading lässt sich nicht unterscheiden,
                # ob dem Modell der Bezug fehlte oder ob es ihn ignoriert hat.
                "section_heading": chunk.get("section_heading", ""),
                # Rohantwort nur bei Fehlschlag — sonst bläht sie die Datei auf.
                "raw":         "" if questions else answer[:2000],
                "model":       args.model,
                "ok":          bool(questions),
                "chunking_config_hash": CHUNKING_HASH,
                "selection_hash":        SELECTION_HASH,
            })

        processed = start + len(batch)
        logger.info("  [%d/%d  %5.1f%%]  ohne Fragen: %d  mit <think>: %d",
                    processed, len(todo), 100.0 * processed / len(todo),
                    failed, thinking_seen)

        if len(rows) >= FLUSH_ROWS:
            part_index += 1
            write_part(rows, parts_dir, part_index, args.shard)
            rows = []

    if rows:
        part_index += 1
        write_part(rows, parts_dir, part_index, args.shard)

    if args.num_shards > 1:
        # Zusammenfügen darf nur EIN Prozess, und erst wenn alle fertig sind.
        # Ein Array-Job, der hier merged, schriebe eine Datei aus dem
        # Zwischenstand der anderen Shards.
        logger.info("Shard %d fertig. Nach Abschluss aller %d Shards:",
                    args.shard, args.num_shards)
        logger.info("  python enricher.py --merge --output %s", args.output)
    else:
        merge_parts(parts_dir, args.output)

    total = len(todo)
    logger.info("─" * 60)
    logger.info("Angereichert : %d Chunk(s)", total)
    logger.info("Ohne Fragen  : %d  (%.1f%%)", failed, 100.0 * failed / total)
    if thinking_seen and not args.thinking:
        logger.warning(
            "%d Antwort(en) enthielten trotz abgeschaltetem Denkmodus einen "
            "<think>-Block. Die Chat-Vorlage ignoriert 'enable_thinking' also. "
            "Sie wurden entfernt; erhöhe --max-new-tokens, falls die Fragenquote "
            "darunter leidet.", thinking_seen)
    logger.info("Ausgabe      : %s", args.output)
    logger.info("Teildateien  : %s  (bleiben für den Wiederanlauf liegen)", parts_dir)
    logger.info("─" * 60)
    logger.info("Jetzt die Fragen durchlesen, bevor sie in den Index gehen:")
    logger.info("  python -c \"import pyarrow.parquet as pq; t=pq.read_table('%s'); "
                "[print(r) for r in t.to_pylist()[:20]]\"", args.output)


if __name__ == "__main__":
    main()
