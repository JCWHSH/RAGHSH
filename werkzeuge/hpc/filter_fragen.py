"""
Verwirft Fragen, die zu allgemein sind, um eine einzelne Stelle zu finden.

Warum nicht das LLM urteilen lassen
-----------------------------------
"Zu allgemein" heißt: passt auf zu viele andere Chunks. Das ist eine Eigenschaft
des Korpus, nicht des einzelnen Abschnitts — das Modell sieht beim Erzeugen nur
einen Chunk und kann es deshalb prinzipiell nicht beurteilen. Es würde raten.
Hier wird stattdessen gezählt.

Warum als eigener Schritt nach dem Lauf
---------------------------------------
Die Schwellenwerte lassen sich beliebig oft ändern, ohne die Generierung zu
wiederholen. Die Eingabedatei bleibt unangetastet; geschrieben wird eine
gefilterte Kopie. Kalibriere die Schwellen gegen die Auswertungsmenge:
Index mit mehreren Werten bauen, Recall@10 messen, besten nehmen.

Zwei Kriterien
--------------
wiederholung — An wie vielen VERSCHIEDENEN Texten hängt dieselbe
               (normalisierte) Frage? "Wie viele Credits hat das Modul?"
               taucht hundertfach auf und unterscheidet damit per Definition
               nichts.

dokumentfrequenz — Für jedes Inhaltswort: in wie vielen Chunks kommt es vor?
               Das SELTENSTE Wort einer Frage begrenzt nach oben, wie viele
               Chunks überhaupt alle ihre Wörter enthalten können. Liegt dieses
               Minimum bei mehreren Prozent des Korpus, kann die Frage nichts
               eingrenzen — egal wie spezifisch sie klingt.

Am 14.08.2026 korrigiert (Messung: Messung_Fragenfilter_2026-08-14.md)
----------------------------------------------------------------------
Gemessen an 329.506 Fragen trug die Wiederholung 60.057 Verwerfungen, die
Dokumentfrequenz 5.681. Das Kriterium, über das nie jemand nachgedacht hatte,
war also das wirksame. Vier Punkte haben sich dabei als falsch erwiesen:

1. SCHWELLE. Voreinstellung 3 verwarf eine Frage ab dem vierten Chunk — bei
   121.134 Punkten sind das 0,003 % des Index. Dasselbe Skript nennt eine
   Frage im anderen Kriterium noch spezifisch, wenn ihr seltenstes Wort auf
   2 % der Chunks passt: ein Faktor 600 zwischen zwei Maßstäben, die dasselbe
   messen sollen. Die Stichprobe bestätigte es — bei genau vier Vorkommen
   standen dort Fragen wie "Was gehört zum Modul SBW-223?". Neue
   Voreinstellung: 10. Auch das ist noch 250-mal strenger als das
   Frequenzkriterium.

2. ZÄHLWEISE. Gezählt wurde je Chunk. Seit der Vererbung (erbe_fragen.py)
   tragen 1.123 Zeilen die Fragen ihres wortgleichen Gruppenvertreters; sie
   zählten als eigene Chunks und kippten 826 Fragen über die Schwelle —
   darunter die zu § 3 ATPO, ausgerechnet weil dieser Paragraph jetzt unter
   drei Fakultäten auffindbar ist. Gezählt wird deshalb je dedupe_key.

3. WORTREGEL. _WORT verlangte vier Kleinbuchstaben. Studiengangskürzel (IBS,
   MKO, MAB-PS) waren damit keine Inhaltswörter, und die Frage wurde nach
   ihrem allgemeinsten Wort beurteilt — genau falsch herum, denn das Kürzel
   ist ihr trennschärfstes Token. 934 Fragen betroffen. Die Lösung ist nicht,
   kürzere Wörter zuzulassen (dann würden "der", "die", "und" zu
   Inhaltswörtern und das Kriterium wäre zerstört), sondern die
   Großschreibung auszuwerten, BEVOR text.lower() sie wegwirft.

4. KORPUS. dokumentfrequenz() zählte über alle 196.934 Chunks, auch die
   wortgleichen Kopien, die nie in den Index kommen. Eine 15-fach gespiegelte
   Prüfungsordnung zählte 15-mal; betroffen war damit ausgerechnet die
   Dokumentklasse, deren Auffindbarkeit dieselbe Sitzung repariert hat. 1.461
   Fragen. Gezählt wird jetzt über die Behaltemenge.

Was bewusst NICHT repariert wurde: Modulcodes aus Einzelbuchstaben
("M-G-16-01", "E-H-60-02") erkennt auch _ABK nicht, die Frage wird dann nach
ihrem allgemeinsten Wort beurteilt. Nachgezählt sind das 43 von 329.506 Fragen
(0,013 %), dazu 4 Fragen ohne jedes Inhaltswort. Die Regex dafür zu erweitern
verschöbe die Dokumentfrequenzen des gesamten Korpus — ein messbares Risiko
für einen nicht messbaren Gewinn.

Die Schwelle 10 ist begründet, nicht gemessen: Ob mehr Fragen das Retrieval
besser machen, entscheidet ein Vergleich gegen die Testfragen, nicht dieser
Docstring. Für den Vergleich sind mehrere Läufe zu erzeugen — die Datei bleibt
unangetastet, --max-wiederholung ist ein Schalter.

Aufruf:
    python filter_fragen.py enrichment.parquet              # Bericht, schreibt nichts
    python filter_fragen.py enrichment.parquet --schreiben  # erzeugt gefilterte Datei
    python filter_fragen.py enrichment.parquet --max-df 0.01 --max-wiederholung 2 --schreiben

Ein Chunk darf am Ende null Fragen haben. Das ist kein Verlust — er steht dann
wieder da, wo er ohne Anreicherung stünde.
"""

from __future__ import annotations

import argparse
import re
import unicodedata
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from chunking import (dedupe_key, iter_chunks, list_markdown, make_point_id,
                      select_unique_ids)

INGESTED_DIR = Path(__file__).parent / "data" / "ingested"

# Wortformen UND Bezeichner. Ohne den zweiten Zweig ist "lmv-210" unsichtbar,
# und eine Frage mit Modulcode wird anhand ihres unspezifischsten Wortes
# beurteilt ("anforderungen") — also genau falsch herum: Der Code ist das
# trennschärfste Token, das sie hat.
# Ohne Leerzeichen im Trenner: sonst werden auch "von 98" oder "kapitel 3"
# zu Tokens und blähen die Statistik um Hunderttausende Scheinbegriffe auf.
_WORT = re.compile(r"[a-zäöüß]{4,}|[a-zäöüß]{2,5}[-_]?\d{1,4}(?:-\d{1,3})?")

# Abkürzungen — GROSSbuchstaben im Originaltext, deshalb case-sensitiv und vor
# dem Kleinschreiben anzuwenden. Deckt Studiengangskürzel (IBS, MKO, BSW),
# zusammengesetzte (MAB-PS, VEU-ET) und Fachbegriffe (HGÜ, ECTS) ab.
#
# Warum nicht einfach _WORT auf {2,} lockern: Dann wären "der", "die", "und",
# "hat" Inhaltswörter. Weil das Kriterium das SELTENSTE Wort einer Frage nimmt
# und diese Wörter in fast jedem Chunk stehen, läge die Dokumentfrequenz jeder
# Frage danach bei nahezu 100 % — der Filter verwürfe alles. Die Großschreibung
# ist das Signal, das Kürzel von Füllwörtern trennt.
_ABK = re.compile(r"\b[A-ZÄÖÜ]{2,6}(?:-[A-ZÄÖÜ0-9]{1,4})?\b")

# Wörter, die in Fragen fast immer vorkommen und nichts eingrenzen. Ohne diese
# Liste bestimmt "studierende" die Dokumentfrequenz jeder zweiten Frage.
_STOPP = {
    "welche", "welcher", "welches", "wieviel", "viele", "finde", "gibt", "wird",
    "werden", "sind", "haben", "kann", "muss", "eine", "einen", "einem", "eines",
    "dieser", "diesem", "dieses", "nach", "über", "unter", "durch", "beim",
    "wann", "wozu", "warum", "woran", "hochschule", "hannover",
}


def normalisiere(frage: str) -> str:
    """Vereinheitlicht eine Frage für den Wiederholungsvergleich."""
    s = unicodedata.normalize("NFKD", frage.lower())
    s = re.sub(r"[^a-zäöüß ]+", " ", s)
    return " ".join(s.split())


def inhaltswoerter(text: str) -> set[str]:
    """Inhaltswörter einer Frage oder eines Chunks.

    Erwartet den ORIGINALTEXT, nicht die kleingeschriebene Fassung: _ABK
    braucht die Großschreibung als Signal.
    """
    woerter = set(_WORT.findall(text.lower()))
    woerter |= {a.lower() for a in _ABK.findall(text)}
    return woerter - _STOPP


def korpus_statistik(input_dir: Path, nur_index: bool = True
                     ) -> tuple[dict[str, int], int, dict[str, str]]:
    """Dokumentfrequenzen und die Zuordnung Punkt-ID → Duplikatgruppe.

    nur_index=True zählt ausschließlich die Chunks, die die Deduplizierung
    behält. Das ist der Korpus, gegen den später gesucht wird; die verworfenen
    Kopien sind für die Frage 'wie viele Chunks enthalten dieses Wort?'
    schlicht nicht da. Sie mitzuzählen benachteiligt systematisch die
    gespiegelten Ordnungen und Modulhandbücher.

    Die Gruppenzuordnung fällt im selben Durchlauf ab und wird für das
    Wiederholungskriterium gebraucht.
    """
    md_files = list_markdown(input_dir)
    keep = select_unique_ids(md_files) if nur_index else None

    df: Counter = Counter()
    gruppe_von_id: dict[str, str] = {}
    n = 0
    for _pfad, chunks in iter_chunks(md_files):
        for c in chunks:
            cid = make_point_id(c["source_url"], c["chunk_index"])
            gruppe_von_id[cid] = dedupe_key(c)
            if keep is not None and cid not in keep:
                continue
            n += 1
            df.update(inhaltswoerter(c["text"]))
    return dict(df), n, gruppe_von_id


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("eingabe", type=Path)
    ap.add_argument("--ausgabe", type=Path, default=None,
                    help="Standard: <eingabe>_gefiltert.parquet")
    ap.add_argument("--input-dir", type=Path, default=INGESTED_DIR)
    ap.add_argument("--max-df", type=float, default=0.02,
                    help="Verwerfen, wenn das seltenste Inhaltswort in mehr als "
                         "diesem Anteil aller Chunks vorkommt (Standard 0.02).")
    ap.add_argument("--max-wiederholung", type=int, default=10,
                    help="Verwerfen, wenn dieselbe Frage an mehr VERSCHIEDENEN "
                         "Texten hängt (Standard 10; wortgleiche Chunks zählen "
                         "einmal). Bis zum 14.08.2026 waren es 3 je Chunk — "
                         "siehe Docstring.")
    ap.add_argument("--alle-chunks", action="store_true",
                    help="Dokumentfrequenz über den gesamten Bestand statt nur "
                         "über die deduplizierte Behaltemenge. Nur für den "
                         "Vergleich mit der alten Zählweise.")
    ap.add_argument("--schreiben", action="store_true",
                    help="Ohne dieses Flag wird nur berichtet, nichts geschrieben.")
    args = ap.parse_args()

    tabelle = pq.read_table(args.eingabe)

    # Prüfung VOR to_pylist(): Wird versehentlich hsh_vectors.parquet übergeben,
    # würde das Materialisieren von 175.000 Zeilen mit je 1024 Floats mehrere
    # hundert MB belegen, nur um danach am fehlenden Feld zu scheitern.
    fehlend = [s for s in ("questions", "id") if s not in tabelle.schema.names]
    if fehlend:
        raise SystemExit(
            f"{args.eingabe.name} hat keine Spalte {fehlend[0]!r}.\n"
            f"Vorhanden: {', '.join(tabelle.schema.names)}\n\n"
            "Erwartet wird die Ausgabe von enricher.py (enrichment.parquet), "
            "nicht die des Vektorisierers (hsh_vectors.parquet)."
        )

    zeilen = tabelle.to_pylist()
    alle = [(i, q) for i, z in enumerate(zeilen) for q in z["questions"]]
    if not alle:
        raise SystemExit("Keine Fragen in der Eingabedatei.")

    print(f"Eingabe: {args.eingabe.name}  —  {len(zeilen)} Chunks, {len(alle)} Fragen\n")

    # ── Korpus einlesen: Frequenzen und Gruppenzuordnung ──────────────────
    print("Zähle Dokumentfrequenzen über den Chunk-Korpus …")
    df, n_chunks, gruppe_von_id = korpus_statistik(
        args.input_dir, nur_index=not args.alle_chunks)
    print(f"  {n_chunks} Chunks ({'gesamter Bestand' if args.alle_chunks else 'nur Index'}), "
          f"{len(df)} verschiedene Inhaltswörter\n")

    # ── Kriterium 1: Wiederholung, je Duplikatgruppe ──────────────────────
    # Nicht je Zeile: Wortgleiche Chunks sind derselbe Text. Eine Frage, die an
    # drei Kopien desselben Absatzes hängt, unterscheidet sehr wohl — sie hat
    # nur einen Text zu unterscheiden.
    gruppen_je_frage: dict[str, set[str]] = {}
    for i, q in alle:
        ident = zeilen[i]["id"]
        gruppen_je_frage.setdefault(normalisiere(q), set()).add(
            gruppe_von_id.get(ident, ident))
    haeufig = {nq: len(g) for nq, g in gruppen_je_frage.items()}

    def min_df(frage: str) -> float:
        woerter = inhaltswoerter(frage)
        if not woerter:
            return 1.0                     # nur Stoppwörter -> maximal unspezifisch
        return min(df.get(w, 0) for w in woerter) / n_chunks

    verworfen_wdh, verworfen_df = [], []
    behalten: dict[int, list[str]] = {i: [] for i in range(len(zeilen))}

    for i, q in alle:
        if haeufig[normalisiere(q)] > args.max_wiederholung:
            verworfen_wdh.append((q, haeufig[normalisiere(q)]))
        elif min_df(q) > args.max_df:
            verworfen_df.append((q, min_df(q)))
        else:
            behalten[i].append(q)

    n_behalten = sum(len(v) for v in behalten.values())
    ohne = sum(1 for v in behalten.values() if not v)

    print("─" * 70)
    print(f"Behalten            : {n_behalten:6d}  ({100*n_behalten/len(alle):.1f} %)")
    print(f"Verworfen Wiederholung: {len(verworfen_wdh):6d}  "
          f"(an mehr als {args.max_wiederholung} verschiedenen Texten)")
    print(f"Verworfen Frequenz  : {len(verworfen_df):6d}  "
          f"(seltenstes Wort in mehr als {100*args.max_df:.1f} %% der Chunks)")
    print(f"Chunks ohne Frage   : {ohne:6d}  ({100*ohne/len(zeilen):.1f} %)")
    print("─" * 70)

    if verworfen_wdh:
        print("\nHäufigste verworfene Wiederholungen (Zahl = verschiedene Texte):")
        for q, n in Counter(dict(verworfen_wdh)).most_common(10):
            print(f"  {n:5d}x  {q}")
    if verworfen_df:
        print("\nBeispiele verworfen wegen Frequenz:")
        for q, wert in sorted(verworfen_df, key=lambda x: -x[1])[:10]:
            print(f"  {100*wert:5.1f} %  {q}")

    print("\nStichprobe behaltener Fragen:")
    beispiele = [q for v in behalten.values() for q in v][:10]
    for q in beispiele:
        print(f"  {q}")

    if not args.schreiben:
        print("\n(Nur Bericht. Mit --schreiben wird die gefilterte Datei erzeugt.)")
        return

    ziel = args.ausgabe or args.eingabe.with_name(
        args.eingabe.stem + "_gefiltert.parquet")
    neu = []
    for i, z in enumerate(zeilen):
        z = dict(z)
        z["questions"] = behalten[i]
        z["ok"] = bool(behalten[i])
        neu.append(z)
    pq.write_table(pa.Table.from_pylist(neu, schema=tabelle.schema), ziel,
                   compression="zstd", compression_level=3)
    print(f"\nGeschrieben: {ziel}")
    print("Die Eingabedatei bleibt unverändert — Schwellen lassen sich jederzeit "
          "neu setzen, ohne die Generierung zu wiederholen.")


if __name__ == "__main__":
    main()
