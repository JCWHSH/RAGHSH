"""
Erzeugt aus einem Lauf eine eigenständige HTML-Seite.

Warum zusätzlich zum Terminal
-----------------------------
Das Terminal beantwortet „hat sich etwas bewegt?" direkt nach dem Lauf. Der
Bericht ist für das Verstehen und das Weitergeben: Er zeigt den Verlust als
Fläche statt als Zahl, macht die Einzelfälle aufklappbar und lässt sich an
Dritte weiterreichen — etwa an Charghy und El Beidawi, deren Fallbasis hier
läuft.

Drei Berichte, ein Aufruf
-------------------------
    python auswertung/gold/bericht.py --letzter
    python auswertung/gold/bericht.py <lauf-id> [--ziel DATEI]
    python auswertung/gold/bericht.py --vergleich <lauf-a> <lauf-b> [--ziel DATEI]
    python auswertung/gold/bericht.py <lauf-id> --vertrag aktuell     → bericht_vertrag-<kürzel>.html
    python auswertung/gold/bericht.py --po <lauf-id|pfad>             → PO-Testmenge je Fall
    python auswertung/gold/bericht.py --holdout <lauf-id|pfad>        → Holdout je Fall (bleibt im Holdout-Ordner)
    python auswertung/gold/bericht.py --gesamt [--eval L] [--po L] [--holdout L]
                                  → alle Fragen (eval_gold, PO, Holdout) auf einer Seite,
                                    abgelegt unter RAGHSH_holdout/berichte/

PO- und Holdout-Läufe haben eigene Fälle und eine eigene Wertung (Rückfrage,
Fassung); ihre Seite baut bericht_po.py.

Ein Volllauf bekommt diesen Bericht (Antwort, Quellen, Vier-Felder-Diagnose),
ein Retrieval-Lauf den aus bericht_retrieval.py (Ränge, Arme), zwei Läufe
den Vergleich aus bericht_vergleich.py. Der Modus steht im Manifest; der
Aufruf ist derselbe.

Alle drei lesen dieselben Rohdaten — manifest.json und faelle/*.json — und
werten sie mit denselben Funktionen aus auswertung.py aus: bewerte_fall,
trichter, quadranten, kennzahlen. Die Auswertungslogik liegt an einer Stelle;
doppelt ist nur die Darstellung. Andernfalls driften Terminalzahl und
Berichtszahl auseinander, und das wäre schlimmer als gar kein Bericht. Die
gemeinsamen Bausteine (Seitengerüst, Konfigurationsband, Panels) liegen in
bericht_teile.py.

Was der Bericht nicht verschweigt
---------------------------------
Fälle ohne Pflichtbegriffe, deren Gold nicht unter den Quellen steht, sind
„nicht prüfbar" (Diagnose `unbestimmt`). Sie stehen in keinem der vier Felder,
aber unter ihnen — mit der Angabe, wo ihr Gold verlorenging. Vorher fehlten
sie stillschweigend, und die Ansicht zeigte „Retrievalproblem 0" neben einem
Trichter mit elf Retrievalverlusten.

Gestaltung
----------
Das Stylesheet steht in `bericht.css` daneben. Der Trichter bleibt einfarbig:
Seine Reihenfolge steckt in der Position, zusätzliche Farbtöne wären
Dekoration. Die geprüfte Palette und ihre Bedingungen sind im Kopf der
CSS-Datei beschrieben.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(_BASE))

from auswertung import (LAEUFE, REFERENZ_CB, ist_treffer, kennzahlen, lade_lauf,  # noqa: E402
                        loese_lauf_id, quadranten, trichter, unbestimmt_wo)
from bericht_teile import (HERKUNFT, _e, _kurz, _liste, fuss, herkunft_satz,  # noqa: E402
                           ist_gold, konfig_band, lade_roh, marken, panel, panel_chunks,
                           panel_kandidaten, panel_messwerte, panel_quellen, seite,
                           trichter_html, ziel_urls)
from vergleich import anzeige  # noqa: E402

# Rolle → (Titel, Erklärung, Symbol). Die Farben stehen im Stylesheet als
# Token, damit heller und dunkler Grund je eigene, geprüfte Werte bekommen.
# Die ersten vier sind die Felder der Diagnose, die letzten zwei stehen
# daneben: nicht prüfbar und nicht gewinnbar.
QUADRANT_TEXTE = {
    "treffer":    ("Treffer", "Gefunden und richtig beantwortet.", "✓"),
    "antwort":    ("Antwortproblem",
                   "Das Dokument lag vor, die Antwort nutzt es nicht.", "!"),
    "vertrag":    ("Treffer (andere Quelle)",
                   "Antwort richtig, aber über eine andere offizielle Quelle — zählt als Treffer, "
                   "der Trichter zeigt trotzdem einen Retrievalverlust.", "≈"),
    "retrieval":  ("Retrievalproblem",
                   "Die Suche hat das Dokument nicht gefunden, und die Antwort nennt nicht alle "
                   "Pflichtbegriffe. Fälle ohne Pflichtbegriffe stehen unten bei „nicht prüfbar“.", "✕"),
    "unbestimmt": ("Nicht prüfbar",
                   "Keine Pflichtbegriffe im Vertrag und die Goldquelle nicht unter den Quellen — "
                   "ob die Antwort stimmt, weiß niemand. Zählt weder als Treffer noch als Fehler.", "?"),
    "luecke":     ("Korpuslücke",
                   "Die Goldquelle wurde nie gecrawlt — mit diesem Index nicht gewinnbar.", "∅"),
}

# Die Diagnose aus auswertung.bewerte_fall in einem Satz — damit in der
# Einzelfallkarte steht, warum der Fall dort gelandet ist, und nicht nur wo.
DIAGNOSE_TEXTE = {
    "treffer": "Goldquelle gefunden, in den Kontext übernommen, alle Pflichtbegriffe in der Antwort.",
    "treffer_andere_quelle": "Alle Pflichtbegriffe in der Antwort, aber ohne die Goldquelle — richtig "
                             "beantwortet aus einer anderen Seite; der Trichter zählt es als Verlust.",
    "antwort_verweigert": "Die Goldquelle lag im Kontext, das Modell hat trotzdem keine Antwort gegeben.",
    "unbestimmt": "Keine Pflichtbegriffe im Vertrag und die Goldquelle nicht in den Quellen — nicht bewertbar.",
    "retrieval_fehler": "Die Goldquelle stand nicht unter den Kandidaten — die Suche hat sie nicht geliefert.",
    "auswahl_fehler": "Die Goldquelle war Kandidat, die LLM-Auswahl in Runde 1 hat sie verworfen.",
    "antwort_unvollstaendig": "Die Goldquelle lag im Kontext, die Antwort nennt aber nicht alle Pflichtbegriffe.",
    "verbotene_aussage": "Die Antwort enthält einen Begriff, den der Fallvertrag ausschließt.",
    "korpus_luecke": "Die Goldquelle wurde nie gecrawlt — mit diesem Index ist der Fall nicht gewinnbar.",
    "lauffehler": "Der Durchlauf ist mit einem Fehler abgebrochen.",
}

VERHALTEN = {
    "direct_answer": "direkt antworten",
    "clarify_then_answer": "erst zurückfragen, dann antworten",
    "missing_detail_answer": "antworten und die fehlende Angabe benennen",
}
POLITIK = {
    "exact_gold_source": "genau diese eine Quelle zählt",
    "accepted_sources": "Goldquelle oder eine der akzeptierten Alternativen zählt",
}

# Wo die unbestimmten Fälle ihr Gold verloren — in der Sprache des Trichters.
UNBESTIMMT_WO = {"retrieval": "von der Suche nicht gefunden",
                 "kandidaten": "im Kandidatenlimit verloren",
                 "auswahl": "von der LLM-Auswahl verworfen"}


def rollen(bewertet: list[dict], q: dict) -> dict[str, str]:
    """Jeder Fall bekommt genau eine Rolle: eines der vier Felder, „unbestimmt"
    oder „luecke". Vorher hieß alles ohne Feld „Korpuslücke" — auch die
    sechzehn nicht prüfbaren Fälle."""
    zu = {b["id"]: s for s, liste in q.items() for b in liste}
    for b in bewertet:
        if b["id"] not in zu:
            zu[b["id"]] = "unbestimmt" if b["diagnose"] == "unbestimmt" else "luecke"
    return zu


def waehle_beispiel(bewertet: list[dict], q: dict) -> dict | None:
    """Sucht den lehrreichsten Fall für die Lesehilfe.

    Bevorzugt einen Fall, der alle drei Ränge trägt UND am Ende trotzdem
    scheitert: Daran lässt sich zeigen, wozu die drei Stufen überhaupt getrennt
    gemessen werden. Ein reiner Treffer würde die Frage nicht beantworten.
    """
    def vollstaendig(b):
        return b["rang_chunk"] and b["rang_kandidat"] and b["rang_quelle"]

    for gruppe in ("antwort", "treffer"):
        passend = [b for b in q.get(gruppe, []) if vollstaendig(b)]
        if passend:
            # der mit dem weitesten Weg zeigt am meisten
            return max(passend, key=lambda b: b["rang_kandidat"])
    alle = [b for b in bewertet if vollstaendig(b)]
    return alle[0] if alle else None


def baue_lesehilfe(manifest: dict, bewertet: list[dict], q: dict, roh: dict) -> str:
    """Erklärt Chunk, Kandidat, Quelle und die übrigen Begriffe an einem echten Fall."""
    k = manifest["konfiguration"]
    punkte    = f'{k.get("points_count"):,}'.replace(",", ".") if isinstance(k.get("points_count"), int) else "—"
    n_chunks  = k.get("RAG_CANDIDATE_K") or 30
    n_kand    = k.get("TOP_DOCS") or 8
    n_quellen = k.get("SELECT_N") or 4
    faktor    = k.get("FAKULTAET_FAKTOR")
    fak_satz  = (f" Nennt die Frage eine Fakultät, laufen zwei Suchen (Fakultät und zentrale "
                 f"Seiten), und beide Listen werden länger — Chunks ×4, Kandidaten ×{faktor}."
                 if faktor else "")

    stufen_karten = [
        ("Index", f"{punkte} Chunks",
         "Jede gecrawlte Seite und jedes PDF ist in <b>Chunks</b> zerlegt — "
         "Textabschnitte von rund tausend Zeichen. Der Chunk ist die Einheit, "
         "die gesucht wird, nicht das Dokument."),
        ("Suche", f"{n_chunks} Chunks",
         "Die Hybrid-Suche liefert die ähnlichsten Chunks. <b>Rang Chunk</b> ist "
         "die Position des ersten Chunks, der aus der Goldquelle stammt." + fak_satz),
        ("Kandidaten", f"{n_kand} Dokumente",
         "Die Chunks werden zu <b>Dokumenten</b> zusammengefasst — hier wechselt "
         "die Einheit. <b>Rang Kandidat</b> ist die Position des Golddokuments in "
         "dieser Liste. Sie ist alles, was die erste LLM-Runde zu sehen bekommt."),
        ("Quellen", f"bis {n_quellen} Dokumente",
         "Das erste Sprachmodell wählt aus den Kandidaten aus. <b>Rang Quelle</b> "
         "ist die Position des Golddokuments in dieser Auswahl."),
        ("Antwort", "1 Text",
         "Das zweite Sprachmodell formuliert aus den Volltexten der Quellen. "
         "Geprüft wird, ob die <b>Pflichtbegriffe</b> darin vorkommen."),
    ]
    band = "".join(
        f'<li class="stufe-karte"><div class="karte-kopf"><span class="karte-titel">{_e(t)}</span>'
        f'<span class="karte-menge mono">{_e(menge)}</span></div>'
        f'<p class="karte-text">{text}</p></li>'
        for t, menge, text in stufen_karten)

    beispiel = waehle_beispiel(bewertet, q)
    if beispiel:
        d = roh.get(beispiel["id"]) or {}
        # Die Zahlen des Falls selbst, nicht die Konstanten: Ein Fakultätsfall
        # hat 120 Chunks und 24 Kandidaten, und „Platz 13 von 12" wäre Unsinn.
        habe_chunks = len(d.get("chunks", [])) or n_chunks
        habe_kand = len(d.get("kandidaten", [])) or n_kand
        rc, rk, rq = beispiel["rang_chunk"], beispiel["rang_kandidat"], beispiel["rang_quelle"]
        lage_chunk = ("ganz oben" if rc <= 3 else "im vorderen Teil" if rc <= 10
                      else "weit hinten, gefunden, aber nicht prominent")
        lage_kand = ("ganz vorn" if rk == 1 else "im Mittelfeld" if rk < habe_kand * 0.75
                     else "ganz hinten")
        rolle = next((s for s, liste in q.items() for b in liste
                      if b["id"] == beispiel["id"]), "treffer")
        titel_rolle = QUADRANT_TEXTE[rolle][0]
        fehlend = (f'<li><b>Fehlende Pflichtbegriffe:</b> '
                   f'{_e(", ".join(beispiel["begriffe_fehlend"]))} — die Antwort nennt sie nicht.</li>'
                   if beispiel["begriffe_fehlend"] else "")
        fazit = ("Genau dafür werden die drei Ränge getrennt gemessen: Eine einzige Trefferquote "
                 "hätte hier „nicht getroffen“ gesagt — und den Verdacht auf die Suche gelenkt, "
                 "die ihre Arbeit getan hat." if rolle == "antwort" else
                 "Genau dafür werden die drei Ränge getrennt gemessen: Ein Treffer sagt nichts "
                 "darüber, wie knapp es war — hier musste die LLM-Auswahl ein Dokument vom "
                 "Ende der Liste nach vorn holen.")
        beispiel_html = f"""
<div class="beispiel">
  <p class="beispiel-marke">Ein Fall, gelesen</p>
  <p class="beispiel-frage">„{_e(beispiel['frage'])}"</p>
  <div class="weg beispiel-weg">Chunk <span class="rang mono">{rc}</span>
    <span class="pfeil">→</span> Kandidat <span class="rang mono">{rk}</span>
    <span class="pfeil">→</span> Quelle <span class="rang mono">{rq}</span>
    <span class="fall-rolle" style="--rolle:var(--{rolle})">{_e(titel_rolle)}</span></div>
  <ol class="beispiel-schritte">
    <li>Unter den {habe_chunks} gefundenen Chunks steht der erste aus der Goldquelle auf
        Platz <b>{rc}</b> — {lage_chunk}.</li>
    <li>In der Kandidatenliste landet das Dokument auf Platz
        <b>{rk}</b> von {habe_kand}, also {lage_kand}.</li>
    <li>Die erste LLM-Runde erkennt es und stellt es auf Platz
        <b>{rq}</b> der Quellen. Bis hierhin hat alles funktioniert.</li>
    {fehlend}
  </ol>
  <blockquote class="zitat">{_e(beispiel['antwort_auszug'])}</blockquote>
  <p class="beispiel-fazit">{fazit}</p>
</div>"""
    else:
        beispiel_html = ""

    begriffe = [
        ("Goldquelle",
         "Die im Fallvertrag hinterlegte Seite, die die Frage belegt. Bei einigen "
         "Fällen sind zusätzlich <i>akzeptierte Quellen</i> erlaubt, wenn mehrere "
         "offizielle Seiten die Frage gleichwertig beantworten. Steht der Text der "
         "Goldquelle nach der Deduplizierung nur unter einer fremden URL, zählt auch "
         "diese (<i>Beleg-URL</i>)."),
        ("Fallvertrag",
         "Was ein Fall außer der Frage mitbringt: Referenzantwort, Goldquelle, "
         "Pflichtfakten, verbotene Aussagen, Schwierigkeit und Fairness-Risiko. "
         f"Dieser Lauf: {_e(herkunft_satz(bewertet))}."),
        ("Pflichtbegriffe",
         "Kurze Zeichenketten, die in einer richtigen Antwort vorkommen müssen — "
         "Namen, Fristen, Raumnummern. Der Vergleich ist tolerant gegenüber "
         "Schreibweise, Zahlwörtern und Datumsformen, aber nicht gegenüber anderen "
         "Zahlen. Wo sich kein solcher Begriff finden lässt, bleibt die Liste leer "
         "und der Fall gilt als <i>nicht prüfbar</i>, nicht als erfüllt."),
        ("Nicht prüfbar",
         "Keine Pflichtbegriffe und die Goldquelle nicht unter den Quellen. Steht die "
         "Goldquelle unter den Quellen, gilt der Fall als Treffer — die Antwort kam "
         "aus dem richtigen Dokument, mehr lässt sich ohne Begriffe nicht sagen."),
        ("Fragment-Dokument",
         "Ein Dokument, das nur teilweise im Index liegt: Die Deduplizierung behält "
         "von wortgleichen Chunks nur einen Vertreter, wodurch 29 % der Dokumente "
         "Lücken in ihrer Chunkfolge haben."),
        ("Korpuslücke",
         "Die Goldquelle wurde nie gecrawlt. Kein Retrieval-Fehler — kein Ranking "
         "kann finden, was nicht im Index ist. Solche Fälle werden getrennt "
         "ausgewiesen und nicht in die Quote gerechnet."),
        ("Score",
         "Der Fusionswert eines Chunks aus den Armen der Suche. Er ordnet "
         "die Treffer <i>einer</i> Frage; zwischen zwei Fragen sind die Zahlen "
         "nicht vergleichbar, weil die Fusion je Anfrage neu normalisiert."),
        ("Chunk-Nr.",
         "Die Position des Chunks im Dokument. Sprünge in der Folge zeigen, dass "
         "dazwischenliegende Chunks der Deduplizierung zum Opfer fielen — das ist "
         "dasselbe Phänomen wie beim <i>Fragment-Dokument</i>."),
        ("Auswahlplatz",
         "Die Reihenfolge, in der die erste LLM-Runde die Kandidaten zurückgibt. "
         "Sie ist eine eigene Wertung, kein durchgereichter Suchrang — ein Dokument "
         "auf Kandidatenplatz 8 kann Auswahlplatz 1 bekommen."),
        ("Fremde Fundstellen",
         "Andere Dokumente, in denen derselbe Text wortgleich steht. Der Index "
         "speichert ihn nur einmal; die Liste hält fest, wo er sonst noch stünde — "
         "meist Prüfungsordnungen und Verkündungsblätter in mehreren Jahrgängen."),
    ]
    liste = "".join(f"<dt>{_e(t)}</dt><dd>{text}</dd>" for t, text in begriffe)

    return f"""
<h2>So ist der Bericht zu lesen</h2>
<p class="unterzeile">Eine Frage durchläuft fünf Stufen. Auf jeder wird die Menge
kleiner, und auf jeder kann das gesuchte Dokument verlorengehen — aus jeweils
anderem Grund. Die drei Ränge in den Einzelfällen benennen, wo es auf den
mittleren drei Stufen stand.</p>
<ol class="stufenband">{band}</ol>
{beispiel_html}
<dl class="begriffe">{liste}</dl>
"""


# ---------------------------------------------------------------------------
# Einzelfall: alles, was der Lauf über den Fall weiß
# ---------------------------------------------------------------------------
def _pfadband(b: dict, d: dict) -> str:
    """Die fünf Stationen des Falls mit seinen eigenen Zahlen."""
    chunks, kand, quellen = d.get("chunks", []), d.get("kandidaten", []), d.get("quellen", [])
    gewaehlt = [str(x.get("nr")) for x in d.get("ausgewaehlte_docs", [])]
    felder = [
        ("Chunks", len(chunks), b["rang_chunk"], "aus dem Index geholt"),
        ("Kandidaten", len(kand), b["rang_kandidat"], "Dokumente nach Zusammenfassung"),
        ("Ausgewählt", len(gewaehlt), None, "von der LLM-Auswahl behalten"),
        ("Quellen", len(quellen), b["rang_quelle"], "vollständig im Kontext"),
    ]
    kacheln = []
    for name, menge, rang, zusatz in felder:
        leer = rang is None and name != "Ausgewählt"
        wert = (f'Gold auf Rang <span class="mono">{rang}</span>' if rang
                else (f'Nr. <span class="mono">{", ".join(gewaehlt) or "—"}</span>'
                      if name == "Ausgewählt" else "Gold nicht dabei"))
        kacheln.append(
            f'<div class="pfad-feld{" pfad-feld--leer" if leer else ""}">'
            f'<div class="pfad-titel">{_e(name)} · {menge}</div>'
            f'<div class="pfad-wert">{wert}</div>'
            f'<div class="pfad-zusatz">{_e(zusatz)}</div></div>')
    quote = b["begriffe_quote"]
    if quote is None:
        wert, zusatz = "nicht prüfbar", "keine Pflichtbegriffe hinterlegt"
    else:
        wert = f'<span class="mono">{quote * 100:.0f} %</span> der Begriffe'
        zusatz = "in der Antwort gefunden"
    kacheln.append(
        f'<div class="pfad-feld{"" if quote in (None, 1.0) else " pfad-feld--leer"}">'
        f'<div class="pfad-titel">Antwort</div><div class="pfad-wert">{wert}</div>'
        f'<div class="pfad-zusatz">{_e(zusatz)}</div></div>')
    return f'<div class="pfad">{"".join(kacheln)}</div>'


def _panel_vertrag(fall: dict, b: dict) -> str:
    zeilen = []

    def zeile(dt: str, dd: str) -> None:
        zeilen.append(f"<dt>{_e(dt)}</dt><dd>{dd}</dd>")

    zeile("Herkunft", _e(HERKUNFT.get(fall.get("herkunft"), fall.get("herkunft"))))
    if fall.get("question_original"):
        # Eine geänderte Frage macht Läufe von vorher und nachher unvergleichbar.
        # Also steht sie in der Karte, nicht nur in der Fallbasis.
        zeile("Frage präzisiert",
              f'früher „{_e(fall["question_original"])}“<br>'
              f'<span class="pfad-zusatz">{_e(fall.get("anpassung") or "")}</span>')
    zeile("Erwartetes Verhalten",
          _e(VERHALTEN.get(fall.get("expected_behavior"), fall.get("expected_behavior") or "—")))
    zeile("Quellenregel", _e(POLITIK.get(fall.get("source_policy"), fall.get("source_policy") or "—")))
    gold = fall.get("source_url", "")
    hinweis = "" if fall.get("gold_im_korpus", True) else ' <span class="chip chip--nein">nicht im Korpus</span>'
    zeile("Goldquelle", f'<a href="{_e(gold)}">{_e(gold)}</a>{hinweis}')
    if fall.get("accepted_source_urls"):
        zeile("Ebenfalls akzeptiert",
              "<ul>" + "".join(f'<li><a href="{_e(u)}">{_e(u)}</a></li>'
                               for u in fall["accepted_source_urls"]) + "</ul>")
    if fall.get("beleg_urls"):
        zeile("Beleg-URLs (Vertretung)",
              "<ul>" + "".join(f'<li><a href="{_e(u)}">{_e(u)}</a></li>'
                               for u in fall["beleg_urls"]) + "</ul>")
    zeile("Einstufung",
          f'{_e(fall.get("case_difficulty") or "—")} · Fairness-Risiko '
          f'{_e(fall.get("fairness_risk") or "—")}')

    begriffe = [anzeige(x) for x in (fall.get("pflicht_begriffe") or [])]
    if begriffe:
        fehlend = set(b["begriffe_fehlend"])
        chips = "".join(
            f'<span class="chip chip--{"nein" if x in fehlend else "ja"}">'
            f'{"✕" if x in fehlend else "✓"} {_e(x)}</span>' for x in begriffe)
        zeile("Pflichtbegriffe", f'<div class="chipreihe">{chips}</div>')
    else:
        zeile("Pflichtbegriffe",
              '<span class="pfad-zusatz">keine hinterlegt — dieser Fall wird nur '
              'über die Quellenlage bewertet</span>')

    zeile("Muss die Antwort sagen", _liste(fall.get("required_facts")))
    if fall.get("optional_facts"):
        zeile("Darf sie ergänzen", _liste(fall.get("optional_facts")))
    if fall.get("forbidden_claims"):
        zeile("Darf sie nicht behaupten", _liste(fall.get("forbidden_claims")))
    kl = fall.get("clarification") or {}
    if kl:
        zeile("Rückfrage erwartet",
              f'{_e(kl.get("clarification_text") or "")}<br>'
              f'<span class="pfad-zusatz">Gewählte Lesart: '
              f'{_e(kl.get("selected_option") or "—")}</span>')
    if fall.get("answer_variants"):
        zeile("Antwortvarianten", _liste(fall.get("answer_variants")))
    zeile("Referenzantwort", _e(fall.get("reference_answer") or "—"))
    if fall.get("evaluation_notes"):
        zeile("Prüfhinweis", _e(fall["evaluation_notes"]))

    return panel(
        "Der Fallvertrag",
        "Was der Fall verlangt — festgelegt, bevor der Lauf startete. Die "
        "Pflichtbegriffe sind der maschinell prüfbare Teil davon: Sie werden "
        "in der Antwort gesucht, ohne LLM-Urteil.",
        f'<dl class="daten">{"".join(zeilen)}</dl>')


def _panel_antwort(d: dict, b: dict) -> str:
    antwort = d.get("antwort") or ""
    if d.get("fehler"):
        koerper = (f'<p class="antworttext antworttext--leer">Lauffehler: '
                   f'{_e(d["fehler"])}</p>')
    elif antwort.strip():
        koerper = f'<p class="antworttext">{_e(antwort)}</p>'
    else:
        koerper = '<p class="antworttext antworttext--leer">(keine Antwort)</p>'
    if b["verbotene"]:
        koerper += (f'<p class="merkmal"><b>Verbotene Begriffe in der Antwort:</b> '
                    f'{_e(", ".join(b["verbotene"]))}</p>')
    if b["widerspruch"]:
        koerper += ('<p class="merkmal"><b>Widerspruch gemeldet</b> — die Antwort sagt, '
                    'dass die Quellen einander widersprechen.</p>')
    return panel("Die Antwort des Systems",
                 f'Vollständig, so wie {_e(d.get("modell", "das Modell"))} sie ausgegeben hat.',
                 koerper)


def _fallkarte(b: dict, d: dict | None, rolle: str, k: dict) -> str:
    titel = QUADRANT_TEXTE[rolle][0]
    stil = f"zelle--{rolle}"
    kopf = (f'<summary><span class="fall-id">{_e(b["id"])}</span>'
            f'<span class="fall-frage">{_e(b["frage"])}</span>'
            f'<span class="fall-rolle">{_e(titel)}</span></summary>')
    if d is None:  # sollte nicht vorkommen — lieber sichtbar als stillschweigend leer
        return (f'<details class="fall {stil}">{kopf}<div class="fall-koerper">'
                f'<p class="merkmal">Keine Rohdatei zu diesem Fall gefunden.</p>'
                f'</div></details>')

    fall = d["fall"]
    ziele = ziel_urls(fall)
    diagnose = (f'<div class="diagnose-band">'
                f'<span class="diagnose-name">{_e(b["diagnose"].replace("_", " "))}</span>'
                f'<span class="diagnose-text">'
                f'{_e(DIAGNOSE_TEXTE.get(b["diagnose"], ""))}</span></div>')
    hinweise = []
    if b.get("fakultaet_erkannt"):
        hinweise.append(f'<p class="merkmal"><b>Fakultät {_e(b["fakultaet_erkannt"])}</b> aus der '
                        f'Frage erkannt — zwei Suchen, längere Listen.</p>')
    if b["fragment_im_kontext"]:
        hinweise.append('<p class="merkmal"><b>Fragment im Kontext</b> — mindestens ein '
                        'ausgewähltes Dokument liegt unvollständig im Index. Fehlende '
                        'Stellen kann die Antwort nicht enthalten.</p>')
    if b["begriffe_fehlend"]:
        hinweise.append(f'<p class="merkmal"><b>Fehlende Pflichtbegriffe:</b> '
                        f'{_e(", ".join(b["begriffe_fehlend"]))}</p>')
    if b["rang_chunk"] and not b["rang_kandidat"]:
        # Der wichtigste Sonderfall: gefunden, aber beim Zusammenfassen verloren.
        hinweise.append(
            f'<p class="merkmal"><b>Nicht die Suche, das Kandidatenlimit</b> — die '
            f'Goldquelle stand als Chunk auf Rang {b["rang_chunk"]}, aber nur die ersten '
            f'{len(d.get("kandidaten", [])) or k.get("TOP_DOCS") or 8} verschiedenen Quellen '
            f'werden Kandidaten. Davor lagen zu viele andere Dokumente.</p>')
    if rolle == "unbestimmt":
        wo = ("retrieval" if b["rang_chunk"] is None else
              "kandidaten" if b["rang_kandidat"] is None else "auswahl")
        hinweise.append(
            f'<p class="merkmal"><b>Nicht prüfbar, aber einzuordnen:</b> Das Gold wurde '
            f'{UNBESTIMMT_WO[wo]}. Ob die Antwort aus den anderen Quellen trotzdem stimmt, '
            f'muss ein Mensch lesen — die Referenzantwort steht im Vertrag darunter.</p>')

    return (f'<details class="fall {stil}">{kopf}<div class="fall-koerper">'
            f'{diagnose}{_pfadband(b, d)}{"".join(hinweise)}'
            f'{_panel_antwort(d, b)}'
            f'{_panel_vertrag(fall, b)}'
            f'<div class="panel"><p class="panel-titel">Die Rohdaten des Durchlaufs</p>'
            f'<p class="panel-hinweis">Vier Ansichten auf denselben Fall, in der '
            f'Reihenfolge der Pipeline. Aufklappen, wenn die Diagnose oben belegt '
            f'werden soll.</p>'
            f'{panel_chunks(d, ziele, (k.get("FUSION") or "dbsf").upper())}'
            f'{panel_kandidaten(d, ziele, k.get("TOP_DOCS") or 8)}'
            f'{panel_quellen(d, ziele)}{panel_messwerte(d)}</div>'
            f'</div></details>')


# ---------------------------------------------------------------------------
# Die Seite
# ---------------------------------------------------------------------------
def baue_html(manifest: dict, bewertet: list[dict], roh: dict[str, dict]) -> str:
    k = manifest["konfiguration"]
    z = kennzahlen(bewertet)
    stufen = trichter(bewertet)
    q = quadranten(bewertet)
    zu_rolle = rollen(bewertet, q)
    gesamt = stufen[0]["anzahl"]
    luecken = [b for b in bewertet if zu_rolle[b["id"]] == "luecke"]
    unbestimmt = [b for b in bewertet if zu_rolle[b["id"]] == "unbestimmt"]
    wo = unbestimmt_wo(bewertet)

    # ── Vier Felder plus die zwei daneben ─────────────────────────────────
    def zelle(schluessel: str, faelle: list[dict], zusatz: str = "", breit: bool = False) -> str:
        titel, text, symbol = QUADRANT_TEXTE[schluessel]
        return (
            f'<div class="zelle zelle--{schluessel}{" zelle--breit" if breit else ""}">'
            f'  <div class="zelle-kopf"><span class="zelle-symbol">{symbol}</span>'
            f'  <span class="zelle-titel">{_e(titel)}</span>'
            f'  <span class="zelle-zahl mono">{len(faelle)}</span></div>'
            f'  <p class="zelle-text">{_e(text)}</p>{zusatz}'
            f'  <div class="marken">{marken(faelle)}</div>'
            f'</div>')

    wo_html = ""
    if unbestimmt:
        teile = "".join(
            f'<li><b>{len(wo[s])}</b> {_e(UNBESTIMMT_WO[s])}: '
            f'<span class="mono">{_e(" ".join(b["id"] for b in wo[s]))}</span></li>'
            for s in ("retrieval", "kandidaten", "auswahl") if wo[s])
        wo_html = (f'<p class="zelle-text">Wo ihr Gold verlorenging — das ist messbar, '
                   f'auch wenn die Antwort es nicht ist:</p><ul class="zelle-liste">{teile}</ul>')

    in_feldern = sum(len(v) for v in q.values())
    raster = (f'<p class="achse">Zeile oben: Suche fand das Zieldokument &nbsp;·&nbsp; '
              f'Spalte links: Antwort enthält die Pflichtbegriffe &nbsp;·&nbsp; '
              f'{in_feldern} von {gesamt} Fällen in den vier Feldern</p>'
              f'<div class="raster">{zelle("treffer", q["treffer"])}{zelle("antwort", q["antwort"])}'
              f'{zelle("vertrag", q["vertrag"])}{zelle("retrieval", q["retrieval"])}'
              f'{zelle("unbestimmt", unbestimmt, wo_html, breit=True) if unbestimmt else ""}'
              f'{zelle("luecke", luecken, breit=True) if luecken else ""}</div>')

    # ── Gruppen ───────────────────────────────────────────────────────────
    moeglich = [b for b in bewertet if b["gold_im_korpus"]]
    gruppen_zeilen = []
    for feld, titel in (("schwierigkeit", "Schwierigkeit"), ("herkunft", "Herkunft")):
        gruppen: dict[str, list] = {}
        for b in moeglich:
            gruppen.setdefault(b[feld] or "—", []).append(b)
        for i, (name, v) in enumerate(sorted(gruppen.items())):
            n_unb = sum(1 for x in v if x["diagnose"] == "unbestimmt")
            gruppen_zeilen.append(
                f"<tr><td>{_e(titel) if i == 0 else ''}</td><td>{_e(name)}</td>"
                f'<td class="z mono">{len(v)}</td>'
                f'<td class="z mono">{sum(1 for x in v if x["gold_in_kandidaten"])}</td>'
                f'<td class="z mono">{sum(1 for x in v if x["gold_in_quellen"])}</td>'
                f'<td class="z mono">{sum(1 for x in v if ist_treffer(x))}</td>'
                f'<td class="z mono">{n_unb or "—"}</td></tr>')

    # ── Einzelfälle, nach Handlungsbedarf ─────────────────────────────────
    ordnung = {"retrieval": 0, "antwort": 1, "unbestimmt": 2, "vertrag": 3, "treffer": 4, "luecke": 5}
    sortiert = sorted(bewertet, key=lambda b: (ordnung[zu_rolle[b["id"]]], b["id"]))
    faelle_html = [_fallkarte(b, roh.get(b["id"]), zu_rolle[b["id"]], k) for b in sortiert]

    ref = REFERENZ_CB
    m = z["faelle_moeglich"]
    kennzahl_satz = (f'{z["gold_in_kandidaten"]} von {m} möglichen Fällen in der Kandidatenliste, '
                     f'{z["gold_in_quellen"]} in den Quellen, {z["treffer"]} Treffer, '
                     f'{z["unbestimmt"]} nicht prüfbar.')

    koerper = f"""
<h1>Wo die Fälle verlorengehen</h1>
<p class="unterzeile">Diagnostische Auswertung der RAG-Pipeline über
{gesamt} Fälle — {_e(herkunft_satz(bewertet))}. Gemessen wird nicht, <em>ob</em> das
System richtig lag, sondern <em>wo</em> es falsch abbog.</p>

{konfig_band(manifest)}
<p class="merkmal">{_e(kennzahl_satz)}</p>

<h2>Der Weg durch die Pipeline</h2>
{trichter_html(stufen)}

<h2>Woran es liegt</h2>
{raster}

<h2>Nach Gruppen</h2>
<div class="tabellenhuelle"><table>
<thead><tr><th></th><th></th><th class="z">Fälle</th>
<th class="z">Gold in Kandidaten</th><th class="z">Gold in Quellen</th>
<th class="z">Treffer</th><th class="z">nicht prüfbar</th></tr></thead>
<tbody>{"".join(gruppen_zeilen)}</tbody></table></div>
<p class="merkmal">Ohne die {len(luecken)} Korpuslücken. „Treffer" zählt auch Antworten
aus einer anderen offiziellen Quelle; „nicht prüfbar" sind Fälle ohne Pflichtbegriffe,
deren Gold nicht in den Quellen stand.</p>

{baue_lesehilfe(manifest, bewertet, q, roh)}

<h2>Einzelfälle</h2>
<p class="unterzeile">Sortiert nach Handlungsbedarf: erst was die Suche nicht
fand, dann was die Antwort verschenkte, dann die nicht prüfbaren Fälle, dann die
zu eng gefassten. Jede Karte enthält alles, was der Lauf über den Fall
festgehalten hat — Diagnose und Antwort offen, die Rohdaten der vier
Pipeline-Stufen darunter zum Aufklappen.</p>
{"".join(faelle_html)}

{fuss(f"auswertung/laeufe/{manifest['lauf_id']}", [
    f'<b>Vergleichsmarke.</b> {_e(ref["lauf"])}: Goldquelle gefunden '
    f'{ref["gold_gefunden"][0]}/{ref["gold_gefunden"][1]}, auf Rang 1 '
    f'{ref["gold_rang1"][0]}/{ref["gold_rang1"][1]} — gemessen über K={ref["k"]} finale Chunks. '
    f'Unsere Stufen benutzen andere K ({m} mögliche Fälle, Kandidatenliste '
    f'{k.get("TOP_DOCS") or 8}, Quellen bis {k.get("SELECT_N") or 4}). '
    f'Vergleichbar ist die Richtung, nicht die Nachkommastelle.',
    (f'{len(luecken)} {"Fall ist" if len(luecken) == 1 else "Fälle sind"} nicht gewinnbar: '
     f'Die Goldquelle wurde nie gecrawlt ({_e(", ".join(b["id"] for b in luecken))}). '
     f'Sie bleiben in der Fallbasis und werden getrennt ausgewiesen, statt die Quote zu '
     f'schönen.') if luecken else 'Alle Goldquellen liegen im Korpus.',
])}
"""
    return seite("Wo die Fälle verlorengehen", koerper)


def baue(manifest: dict, bewertet: list[dict], roh: dict[str, dict]) -> str:
    """Der passende Bericht für den Modus des Laufs — auch für lauf.py --bericht."""
    if manifest.get("modus") == "retrieval":
        from bericht_retrieval import baue_retrieval
        return baue_retrieval(manifest, bewertet, roh)
    return baue_html(manifest, bewertet, roh)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("lauf", nargs="?", help="Lauf-ID unter laeufe/")
    ap.add_argument("--letzter", action="store_true", help="den jüngsten Lauf nehmen")
    ap.add_argument("--vergleich", nargs=2, metavar=("LAUF_A", "LAUF_B"),
                    help="zwei Läufe gegeneinander (B gegen A)")
    ap.add_argument("--vertrag", choices=("lauf", "aktuell"), default="lauf",
                    help="Vertrag zur Laufzeit (Beleg) oder heutige fallbasis.json")
    ap.add_argument("--po", metavar="LAUF", help="PO-Lauf (auswertung/po/laeufe/…) je Fall")
    ap.add_argument("--holdout", metavar="LAUF", help="Holdout-Lauf (RAGHSH_holdout/laeufe/…) je Fall")
    ap.add_argument("--gesamt", action="store_true",
                    help="alle Fragen (eval_gold, PO, Holdout) auf einer Seite, jüngste Läufe, "
                         "abgelegt unter RAGHSH_holdout/berichte/")
    ap.add_argument("--eval", metavar="LAUF", help="mit --gesamt: eval_gold-Lauf statt des jüngsten")
    ap.add_argument("--ziel", type=Path)
    args = ap.parse_args()

    if args.gesamt:
        from bericht_gesamt import schreibe_gesamt
        ziel = schreibe_gesamt(args.eval, args.po, args.holdout, args.ziel)
        print(f"Gesamtbericht geschrieben: {ziel}")
        return

    if args.po or args.holdout:
        from bericht_po import baue_po_bericht, finde_lauf, ziel_fuer
        art = "po" if args.po else "holdout"
        lauf = finde_lauf(args.po or args.holdout, art)
        ziel = args.ziel or ziel_fuer(lauf, art)
        ziel.write_text(baue_po_bericht(lauf, art), encoding="utf-8")
        print(f"Bericht geschrieben: {ziel}")
        return
    aktuell = args.vertrag == "aktuell"

    # Eine Bewertung nach heutigem Vertrag bekommt einen eigenen Dateinamen mit
    # dem Vertragskürzel — sie darf den Beleg-Bericht des Laufs nicht ersetzen.
    def zusatz() -> str:
        if not aktuell:
            return ""
        from vertrag import hash_fallbasis, kurz, lade_fallbasis
        return f"_vertrag-{kurz(hash_fallbasis(lade_fallbasis()))}"

    if args.vergleich:
        from bericht_vergleich import baue_vergleich
        a, b = args.vergleich
        ziel = args.ziel or (LAEUFE / b / f"vergleich_gegen_{a}{zusatz()}.html")
        ziel.write_text(baue_vergleich(a, b, vertrag_aktuell=aktuell), encoding="utf-8")
        print(f"Vergleich geschrieben: {ziel}")
        return

    lauf_id = loese_lauf_id(args.lauf, args.letzter)
    manifest, bewertet = lade_lauf(lauf_id, vertrag_aktuell=aktuell)
    roh = lade_roh(lauf_id)
    if aktuell:
        # Die Karten zeigen den Vertrag aus der Rohdatei — der wäre hier der alte.
        aktuelle = {b["id"]: b for b in bewertet}
        from vertrag import lade_fallbasis
        neu = {f["id"]: f for f in lade_fallbasis()}
        for fid, d in roh.items():
            if fid in neu:
                d["fall"] = neu[fid]
        roh = {fid: d for fid, d in roh.items() if fid in aktuelle}
    ziel = args.ziel or (LAEUFE / lauf_id / f"bericht{zusatz()}.html")
    ziel.write_text(baue(manifest, bewertet, roh), encoding="utf-8")
    print(f"Bericht geschrieben: {ziel}")


if __name__ == "__main__":
    main()
