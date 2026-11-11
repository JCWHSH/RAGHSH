"""HTML-Bericht für einen Retrieval-Lauf: Ränge und Arme statt Antworten.

Ein Retrieval-Lauf (`lauf.py --nur-retrieval`) endet an der Kandidatenliste,
es gibt keine Antwort und keine Quellen. Die Frage ist nicht „stimmt die
Antwort", sondern „wie weit oben steht das Gold, und welcher Arm hat es
gesehen" — denn Runde 1 sieht nur die Kandidatenliste, und ein Gold auf Rang
8 hat schlechtere Karten als eines auf Rang 1.

Aufruf über bericht.py; der Modus steht im Manifest:
    python auswertung/gold/bericht.py <retrieval-lauf-id>

Gezeigt wird dieselbe Rechnung wie in `auswertung.zeige_retrieval`: Trichter
bis zur Kandidatenliste, Rangverteilung, Arm-Tabelle (Top-10, Top-50, als
einziger Arm), die Fälle ohne Kandidaten und je Fall die Arme einzeln.
"""
from __future__ import annotations

import statistics
import sys
from pathlib import Path

_BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(_BASE))

from auswertung import trichter, wilson  # noqa: E402
from bericht_teile import (_e, _kurz, fuss, herkunft_satz, konfig_band, marken,  # noqa: E402
                           panel_arme, panel_chunks, panel_kandidaten, panel_messwerte,
                           seite, trichter_html, weg, ziel_urls)

ARME = ("text", "questions", "sparse", "sparse_expanded")
ARM_TEXTE = {
    "text": "dichter Vektor über den Chunktext",
    "questions": "dichter Vektor über die zum Chunk erzeugten Fragen (doc2query)",
    "sparse": "BM25 über den Chunktext",
    "sparse_expanded": "BM25 über Chunktext und erzeugte Fragen",
}
DIAGNOSE_TEXTE = {
    "gefunden": "Gold steht in der Kandidatenliste.",
    "kandidatenlimit": "Gold stand unter den Chunks, aber zu viele andere Dokumente lagen davor — "
                       "die Kandidatenliste war voll.",
    "retrieval_fehler": "Kein Chunk der Goldquelle unter den Treffern der Fusion.",
    "korpus_luecke": "Die Goldquelle wurde nie gecrawlt.",
    "lauffehler": "Der Durchlauf ist mit einem Fehler abgebrochen.",
}


def _rangklassen(gefunden: list[dict]) -> list[tuple[str, list[dict]]]:
    klassen = (("Rang 1", lambda r: r == 1), ("Rang 2–3", lambda r: 2 <= r <= 3),
               ("Rang 4–8", lambda r: 4 <= r <= 8), ("Rang > 8", lambda r: r > 8))
    return [(label, [b for b in gefunden if test(b["rang_kandidat"])]) for label, test in klassen]


def _arm_tabelle(moeglich: list[dict], arme: list[str]) -> str:
    n = len(moeglich)
    zeilen = []
    for a in arme:
        in10 = sum(1 for b in moeglich if b["rang_arm"].get(a) and b["rang_arm"][a] <= 10)
        in50 = sum(1 for b in moeglich if b["rang_arm"].get(a))
        exklusiv = [b for b in moeglich if b["rang_arm"].get(a)
                    and not any(b["rang_arm"].get(x) for x in arme if x != a)]
        zeilen.append(
            f'<tr><td><b>{_e(a)}</b><br><span class="pfad-zusatz">{_e(ARM_TEXTE.get(a, ""))}</span></td>'
            f'<td class="z mono">{in10}/{n}</td><td class="z mono">{in50}/{n}</td>'
            f'<td class="z mono">{len(exklusiv)}</td>'
            f'<td><div class="marken">{marken(exklusiv)}</div></td></tr>')
    keiner = [b for b in moeglich if not any(b["rang_arm"].get(a) for a in arme)]
    vertan = [b for b in moeglich if not b["gold_in_kandidaten"]
              and any(b["rang_arm"].get(a) and b["rang_arm"][a] <= 10 for a in arme)]
    return (f'<div class="tabellenhuelle"><table><thead><tr><th>Arm</th>'
            f'<th class="z">Gold in Top-10</th><th class="z">Gold in Top-50</th>'
            f'<th class="z">als einziger Arm</th><th>Fälle, die nur dieser Arm sieht</th></tr></thead>'
            f'<tbody>{"".join(zeilen)}</tbody></table></div>'
            f'<p class="merkmal"><b>Kein Arm</b> sieht das Gold in seinen Top-50 ({len(keiner)}): '
            f'<span class="mono">{_e(" ".join(b["id"] for b in keiner)) or "—"}</span> — hier hilft '
            f'keine Fusion, nur Index oder Vertrag.</p>'
            f'<p class="merkmal"><b>Ein Arm hatte es in den Top-10, die Kandidatenliste nicht</b> '
            f'({len(vertan)}): <span class="mono">{_e(" ".join(b["id"] for b in vertan)) or "—"}</span> '
            f'— Fusion oder Deduplizierung haben es verloren.</p>')


def _falltabelle(moeglich: list[dict], arme: list[str]) -> str:
    kopf = "".join(f'<th class="z">{_e(a)}</th>' for a in arme)
    zeilen = []
    for b in sorted(moeglich, key=lambda b: (b["rang_kandidat"] or 999, b["id"])):
        arm_zellen = "".join(
            (f'<td class="z mono">{b["rang_arm"][a]}</td>' if b["rang_arm"].get(a)
             else '<td class="z mono fehlt">—</td>') for a in arme)
        fak = f' <span class="pfad-zusatz">[{_e(b["fakultaet_erkannt"])}]</span>' if b["fakultaet_erkannt"] else ""
        zeilen.append(
            f'<tr class="{"" if b["gold_in_kandidaten"] else "verloren"}">'
            f'<td class="mono">{_e(b["id"])}</td>'
            f'<td>{_e(_kurz(b["frage"], 80))}{fak}</td>'
            f'<td class="z mono{"" if b["rang_chunk"] else " fehlt"}">{b["rang_chunk"] or "—"}</td>'
            f'<td class="z mono{"" if b["rang_kandidat"] else " fehlt"}">{b["rang_kandidat"] or "—"}</td>'
            f'{arm_zellen}<td>{_e(b["diagnose"].replace("_", " "))}</td></tr>')
    return (f'<div class="tabellenhuelle"><table class="rangtabelle"><thead><tr><th>Fall</th><th>Frage</th>'
            f'<th class="z">Chunk</th><th class="z">Kandidat</th>{kopf}<th>Diagnose</th></tr></thead>'
            f'<tbody>{"".join(zeilen)}</tbody></table></div>')


def _fallkarte(b: dict, d: dict | None, k: dict) -> str:
    titel = {"gefunden": "gefunden", "kandidatenlimit": "Kandidatenlimit",
             "retrieval_fehler": "nicht gefunden", "korpus_luecke": "Korpuslücke",
             "lauffehler": "Lauffehler"}.get(b["diagnose"], b["diagnose"])
    rolle = {"gefunden": "treffer", "kandidatenlimit": "antwort",
             "retrieval_fehler": "retrieval"}.get(b["diagnose"], "luecke")
    kopf = (f'<summary><span class="fall-id">{_e(b["id"])}</span>'
            f'<span class="fall-frage">{_e(b["frage"])}</span>'
            f'<span class="fall-rolle">{_e(titel)}</span></summary>')
    if d is None:
        return (f'<details class="fall zelle--{rolle}">{kopf}<div class="fall-koerper">'
                f'<p class="merkmal">Keine Rohdatei zu diesem Fall gefunden.</p></div></details>')
    fall = d["fall"]
    ziele = ziel_urls(fall)
    diagnose = (f'<div class="diagnose-band">'
                f'<span class="diagnose-name">{_e(b["diagnose"].replace("_", " "))}</span>'
                f'<span class="diagnose-text">{_e(DIAGNOSE_TEXTE.get(b["diagnose"], ""))}</span></div>')
    gold = fall.get("source_url", "")
    vertrag = (f'<p class="merkmal">Goldquelle: <a href="{_e(gold)}">{_e(_kurz(gold, 110))}</a>'
               + (f' · Fakultät <b>{_e(b["fakultaet_erkannt"])}</b> aus der Frage erkannt'
                  if b["fakultaet_erkannt"] else "") + "</p>")
    return (f'<details class="fall zelle--{rolle}">{kopf}<div class="fall-koerper">'
            f'{diagnose}<p class="merkmal">{weg(b)} · {len(d.get("chunks", []))} Chunks, '
            f'{len(d.get("kandidaten", []))} Kandidaten</p>{vertrag}'
            f'{panel_arme(d, ziele, b)}'
            f'{panel_chunks(d, ziele, (k.get("FUSION") or "dbsf").upper())}'
            f'{panel_kandidaten(d, ziele, k.get("TOP_DOCS") or 8)}{panel_messwerte(d)}'
            f'</div></details>')


def baue_retrieval(manifest: dict, bewertet: list[dict], roh: dict[str, dict]) -> str:
    k = manifest["konfiguration"]
    stufen = trichter(bewertet)[:4]          # bis „in der Kandidatenliste"
    moeglich = [b for b in bewertet if b["gold_im_korpus"]]
    gefunden = [b for b in moeglich if b["gold_in_kandidaten"]]
    verloren = [b for b in moeglich if not b["gold_in_kandidaten"]]
    n = len(moeglich)
    raenge = sorted(b["rang_kandidat"] for b in gefunden)
    rang1 = sum(1 for r in raenge if r == 1)
    lo, hi = wilson(len(gefunden), n)
    arme = [a for a in ARME if any(a in b["rang_arm"] for b in moeglich)]

    klassen_html = "".join(
        f'<div class="stufe"><div class="stufe-name">{_e(label)}</div>'
        f'<div class="marken">{marken(faelle)}</div>'
        f'<div class="stufe-wert mono">{len(faelle)}</div></div>'
        for label, faelle in _rangklassen(gefunden))

    verloren_html = ""
    if verloren:
        zeilen = "".join(
            f'<tr><td class="mono">{_e(b["id"])}</td><td>{_e(b["diagnose"].replace("_", " "))}</td>'
            f'<td class="mono">{("Chunk-Rang " + str(b["rang_chunk"])) if b["rang_chunk"] else "nicht unter den Chunks"}</td>'
            f'<td>{_e(_kurz(b["frage"], 90))}</td></tr>'
            for b in sorted(verloren, key=lambda b: (b["diagnose"], b["id"])))
        verloren_html = (f'<h2>Nicht in der Kandidatenliste</h2>'
                         f'<div class="tabellenhuelle"><table><thead><tr><th>Fall</th><th>Diagnose</th>'
                         f'<th>Wo</th><th>Frage</th></tr></thead><tbody>{zeilen}</tbody></table></div>')

    arm_html = ""
    if arme:
        arm_html = (f'<h2>Welcher Arm sieht das Gold</h2>'
                    f'<p class="unterzeile">Jeder Arm wurde zusätzlich allein gefragt (Top-50). '
                    f'Ein Arm, der das Gold als einziger sieht, ist unverzichtbar; ein Gold, das ein '
                    f'Arm oben hat und die Kandidatenliste nicht, ist ein Fusionsproblem.</p>'
                    f'{_arm_tabelle(moeglich, arme)}')

    ordnung = {"retrieval_fehler": 0, "kandidatenlimit": 1, "lauffehler": 2, "gefunden": 3, "korpus_luecke": 4}
    sortiert = sorted(bewertet, key=lambda b: (ordnung.get(b["diagnose"], 9), b["rang_kandidat"] or 999, b["id"]))
    faelle_html = "".join(_fallkarte(b, roh.get(b["id"]), k) for b in sortiert)

    koerper = f"""
<h1>Wo das Gold steht</h1>
<p class="unterzeile">Retrieval-Lauf ohne Sprachmodell über {len(bewertet)} Fälle —
{_e(herkunft_satz(bewertet))}. Gemessen wird, ob und wie weit oben die Goldquelle in
der Kandidatenliste steht, die die erste LLM-Runde zu sehen bekäme.</p>

{konfig_band(manifest)}
<p class="merkmal">Gold in der Kandidatenliste bei <b>{len(gefunden)} von {n}</b> möglichen
Fällen ({len(gefunden) / max(n, 1) * 100:.0f} %, 95-%-Intervall {lo * 100:.0f}–{hi * 100:.0f} %) ·
auf Rang 1 bei {rang1} · Median Rang {statistics.median(raenge) if raenge else "—"}.</p>

<h2>Der Weg bis zur Kandidatenliste</h2>
{trichter_html(stufen)}

<h2>Wo das Gold steht</h2>
<p class="unterzeile">Rang in der Kandidatenliste. Was hier hinter Rang 8 steht, sieht
die LLM-Auswahl nur, weil die Liste in diesem Lauf {k.get("TOP_DOCS") or 8} Plätze hat.</p>
{klassen_html}

{arm_html}
{verloren_html}

<h2>Alle Fälle</h2>
<p class="unterzeile">Ränge je Fall: in der Fusion (Chunk), in der Kandidatenliste und in
jedem Arm allein (Top-50). Grau hinterlegt, was die Kandidatenliste nicht erreicht hat.</p>
{_falltabelle(moeglich, arme)}

<h2>Einzelfälle</h2>
<p class="unterzeile">Erst die verlorenen, dann die knappen, dann die gefundenen. Je Fall
die Arme einzeln, die Trefferliste der Fusion und die Kandidatenliste.</p>
{faelle_html}

{fuss(f"auswertung/laeufe/{manifest['lauf_id']}", [
    'Ein Retrieval-Lauf ist deterministisch: Dieselbe Konfiguration liefert dieselben '
    'Listen. Unterschiede zwischen zwei Retrieval-Läufen sind Wirkung, kein Rauschen — '
    'anders als in Vollläufen, wo die LLM-Auswahl um ein bis zwei Fälle streut.',
])}
"""
    return seite("Wo das Gold steht", koerper)
