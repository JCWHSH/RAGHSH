"""
Wertet einen Lauf aus auswertung/laeufe/ aus — deterministisch, ohne LLM.

Warum ohne Judge
----------------
Chargy und Beidawi bewerten mit zwei LLM-Judges und benennen die Grenze selbst:
„Judge bleibt ein Modell, Bewertungsrauschen bleibt." Der größte Teil ihrer
Diagnose lässt sich aber ohne Modell gewinnen — aus dem Vergleich zweier Listen,
die die Pipeline ohnehin erzeugt. Diese Signale sind reproduzierbar, kostenlos
und rauschfrei. Ein LLM-Judge kann später darauf aufsetzen; er ersetzt sie nicht.

Die fünf Signale je Fall
------------------------
gold_in_kandidaten  + Rang    Hat die SUCHE das Dokument gefunden?
gold_in_quellen     + Rang    Hat die LLM-AUSWAHL es behalten?
politik_erfuellt              exact_gold_source / accepted_sources beachtet?
begriffe_quote                Anteil gefundener pflicht_begriffe (None = nicht prüfbar)
fragment_im_kontext           War ein ausgewähltes Dokument unvollständig?

Daraus die Diagnose — die Antwort zuerst, dann die Herleitung
--------------------------------------------------------------
Ein Treffer ist eine richtige Antwort. Ob sie aus der Goldquelle stammt, ist
eine zweite Frage (`herkunft_antwort`), die dem Retrieval-Trichter gehört, nicht
dem Urteil. Bis 15.09.2026 stand die Reihenfolge andersherum — erst Retrieval,
dann Antwort — und eine richtige Antwort aus einer anderen offiziellen Seite
(n01: alle fünf Fakultäten, aus dem Jahresabschluss) zählte als Fehlschlag.

verbotene_aussage       Ein verbotener Begriff steht in der Antwort.
treffer                 Antwort vollständig, Goldquelle unter den Quellen.
treffer_andere_quelle   Antwort vollständig, aber ohne Goldquelle — richtig
                        beantwortet, der Trichter zeigt trotzdem einen Verlust.
korpus_luecke           Antwort unvollständig und Gold gar nicht im Index —
                        Crawl-Lücke, kein Retrieval-Fehler. Betrifft q10, q20.
retrieval_fehler        Gold nicht unter den Kandidaten.
auswahl_fehler          Gold war Kandidat, fiel aber in Runde 1 heraus.
antwort_verweigert      Gold war Quelle, das Modell antwortet trotzdem
                        „dazu liegen mir keine Informationen vor".
antwort_unvollstaendig  Gold war Quelle, aber Pflichtbegriffe fehlen.

Dazu zwei Merkmale, die keine Diagnose sind: `widerspruch` (die Antwort
meldet einander widersprechende Angaben — meist ein Korpusbefund, zwei
Dokumente verschiedenen Alters) und `herkunft_antwort` (gold / akzeptiert /
andere / keine).

`korpus_luecke` ist ein Zusatz gegenüber ihrem System — und einer, der ihnen
geholfen hätte: Ihre Priorität 1 lautet „q10/q20 im Retrieval debuggen", dabei
fehlen diese Dokumente in beiden Korpora schlicht.

Aufrufe
-------
    python auswertung/gold/auswertung.py                     Übersicht aller Läufe
    python auswertung/gold/auswertung.py --letzter           jüngster Lauf
    python auswertung/gold/auswertung.py <lauf-id> --detail
    python auswertung/gold/auswertung.py --vergleich <lauf-a> <lauf-b>
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

_BASE = Path(__file__).resolve().parent
from raghsh import konfig  # noqa: E402
sys.path.insert(0, str(_BASE))

from raghsh.crawl.crawler_utils import normalize_url  # noqa: E402
from vergleich import anzeige, enthaelt_begriff  # noqa: E402
from vertrag import hash_aus_faellen, hash_fall, hash_fallbasis, kurz, lade_fallbasis  # noqa: E402

LAEUFE = konfig.LAEUFE

# Ihr Llama-Lauf vom 07.05.2026 als Vergleichsmarke. ACHTUNG beim Lesen: Sie
# messen über ihre sechs finalen Chunks, wir über Kandidaten (8) bzw. Quellen
# (bis 4). Andere Grundgesamtheit — vergleichbar ist die Richtung, nicht die
# zweite Nachkommastelle.
REFERENZ_CB = {"lauf": "Chargy/Beidawi, Llama 3.3-70B, 07.05.2026",
               "gold_gefunden": (12, 20), "gold_rang1": (6, 20), "k": 6}


TREFFER = ("treffer", "treffer_andere_quelle")
# „unbestimmt": keine Pflichtbegriffe, Gold nicht in den Quellen — nicht bewertbar.
# Diagnosen eines Retrieval-Laufs (kein LLM): gefunden | kandidatenlimit |
# retrieval_fehler | korpus_luecke | lauffehler

# Formulierungen, mit denen das Modell eine Antwort verweigert. Wird nur
# geprüft, wenn Pflichtbegriffe fehlen — eine vollständige Antwort mit dem Satz
# „zu S2 liegen mir keine Informationen vor" bleibt vollständig.
_VERWEIGERT = ("liegen mir keine informationen", "keine informationen aus den",
               "nicht beschrieben", "keine angaben", "nicht enthalten",
               "kann ich nicht beantworten", "geht aus den dokumenten nicht hervor")
# Formulierungen, mit denen das Modell zwei Quellen gegeneinanderstellt.
_WIDERSPRUCH = ("widersprechen sich", "widerspricht", "widerspruechlich",
                "uneinheitlich", "hingegen", "abweichend", "unterschiedliche angaben",
                "weichen voneinander ab")


def _casefold(text: str) -> str:
    return " ".join((text or "").split()).casefold().replace("ü", "ue").replace("ö", "oe") \
                                                    .replace("ä", "ae").replace("ß", "ss")


def ist_treffer(b: dict) -> bool:
    return b["diagnose"] in TREFFER


def _rang(urls: list[str], ziele: list[str]) -> int | None:
    """1-basierter Rang der ersten Ziel-URL in der Liste, sonst None."""
    ziel_norm = {normalize_url(z) for z in ziele if z}
    for i, u in enumerate(urls, start=1):
        if normalize_url(u) in ziel_norm:
            return i
    return None


def bewerte_fall(datensatz: dict, nur_retrieval: bool = False) -> dict:
    fall = datensatz["fall"]
    antwort = datensatz.get("antwort") or ""

    gold = [fall.get("source_url", "")]
    akzeptiert = gold + list(fall.get("accepted_source_urls") or [])
    # Bei accepted_sources zählt jede erlaubte Quelle, bei exact_gold_source
    # nur die konfigurierte.
    politik_ziele = akzeptiert if fall.get("source_policy") == "accepted_sources" else gold

    kandidaten_urls = [k.get("url", "") for k in datensatz.get("kandidaten", [])]
    quellen_urls    = [q.get("url", "") for q in datensatz.get("quellen", [])]
    chunk_urls      = [c.get("source_url", "") for c in datensatz.get("chunks", [])]
    chunk_ids       = [str(c.get("id", "")) for c in datensatz.get("chunks", [])]

    # Punktgenau statt nur URL (seit 21.09.2026, siehe baue_fallbasis.
    # ergaenze_belegpunkte): Der Text einer Goldquelle liegt nach der
    # Deduplizierung oft unter fremden URLs. Ein Chunk zählt, wenn seine ID
    # eine Goldposition vertritt; eine Kandidaten-URL zählt, wenn sie einen
    # Belegpunkt trägt. Ohne Belegpunkte gilt die URL-Regel allein.
    beleg_ids  = set(fall.get("beleg_punkte") or [])
    gold_ids   = set(fall.get("gold_punkte") or [])
    beleg_urls = list(fall.get("beleg_urls") or [])
    url_ziele  = politik_ziele + beleg_urls

    def _rang_chunk() -> int | None:
        treffer_ids = beleg_ids or gold_ids
        for i, (cid, url) in enumerate(zip(chunk_ids, chunk_urls), start=1):
            if cid in treffer_ids or normalize_url(url) in {normalize_url(z) for z in politik_ziele if z}:
                return i
        return None

    rang_kand   = _rang(kandidaten_urls, url_ziele)
    rang_quelle = _rang(quellen_urls, url_ziele)
    rang_chunk  = _rang_chunk()
    # Goldrang je Arm (nur Retrieval-Läufe seit 21.09.2026 tragen `arme`):
    # welcher Arm hat das Gold in seinen Top-50, und wie weit oben?
    rang_arm = {arm: _rang([t.get("source_url", "") for t in treffer], url_ziele)
                for arm, treffer in (datensatz.get("arme") or {}).items()}

    # Woraus wurde geantwortet? Gold selbst, eine akzeptierte Zweitquelle oder
    # etwas ganz anderes. Bei accepted_sources zählt beides als Politik erfüllt;
    # die Unterscheidung bleibt trotzdem sichtbar.
    if _rang(quellen_urls, gold) is not None:
        herkunft = "gold"
    elif _rang(quellen_urls, akzeptiert + beleg_urls) is not None:
        herkunft = "akzeptiert"
    elif quellen_urls:
        herkunft = "andere"
    else:
        herkunft = "keine"

    # Toleranter Vergleich (auswertung/gold/vergleich.py): Zahlwörter, Datumsformen,
    # Interpunktion, eingeschobene Wörter, Alternativen. Der wörtliche
    # Substring-Vergleich davor hat richtige Antworten als unvollständig gewertet.
    begriffe = fall.get("pflicht_begriffe") or []
    if begriffe:
        fehlend = [anzeige(b) for b in begriffe if not enthaelt_begriff(antwort, b)]
        quote = round((len(begriffe) - len(fehlend)) / len(begriffe), 3)
    else:
        # Leere Liste heißt „nicht prüfbar", nicht „erfüllt".
        quote, fehlend = None, []

    verbotene = [anzeige(b) for b in (fall.get("verbotene_begriffe") or [])
                 if enthaelt_begriff(antwort, b)]

    fragment = any(c.get("text_vollstaendig") is False
                   for c in datensatz.get("chunks", [])
                   if c.get("source_url") in set(quellen_urls))

    antwort_cf = _casefold(antwort)
    widerspruch = any(m in antwort_cf for m in _WIDERSPRUCH)
    verweigert = any(m in antwort_cf for m in _VERWEIGERT)
    if quote is None:
        status = "nicht_pruefbar"
    elif quote == 1.0:
        status = "vollstaendig"
    elif quote == 0.0 and verweigert:
        status = "verweigert"
    elif quote == 0.0:
        status = "keine"
    else:
        status = "teilweise"
    antwort_ok = status in ("vollstaendig", "nicht_pruefbar")

    # ── Diagnose: erst die Antwort, dann woran es lag ─────────────────────
    if nur_retrieval:
        # Kein LLM gelaufen: Antwortachse entfällt, die Diagnose endet an der
        # Kandidatenliste. Ein Treffer ist hier: Gold steht unter den Kandidaten.
        status, herkunft, widerspruch = "entfaellt", "entfaellt", False
        quote, fehlend = None, []
        if datensatz.get("fehler"):
            diagnose = "lauffehler"
        elif not fall.get("gold_im_korpus", True):
            diagnose = "korpus_luecke"
        elif rang_kand is not None:
            diagnose = "gefunden"
        elif rang_chunk is not None:
            diagnose = "kandidatenlimit"
        else:
            diagnose = "retrieval_fehler"
    elif datensatz.get("fehler"):
        diagnose = "lauffehler"
    elif verbotene:
        diagnose = "verbotene_aussage"
    elif antwort_ok:
        if rang_quelle is not None:
            diagnose = "treffer"
        elif status == "nicht_pruefbar":
            # Keine Pflichtbegriffe und das Gold nicht in den Quellen: Ob die
            # Antwort stimmt, weiß niemand. Weder Treffer noch Fehler — sonst
            # zählen 50 Fälle ohne Begriffe (eval_gold_50) automatisch als
            # richtig, und der Volllauf meldet 70/75.
            diagnose = "unbestimmt"
        else:
            diagnose = "treffer_andere_quelle"
    elif not fall.get("gold_im_korpus", True):
        diagnose = "korpus_luecke"
    elif rang_kand is None:
        diagnose = "retrieval_fehler"
    elif rang_quelle is None:
        diagnose = "auswahl_fehler"
    elif status == "verweigert":
        diagnose = "antwort_verweigert"
    else:
        diagnose = "antwort_unvollstaendig"

    return {
        "id":                 fall["id"],
        "herkunft":           fall["herkunft"],
        "antwort_auszug":     " ".join(antwort.split())[:220],
        "gold_url":           fall.get("source_url", ""),
        "schwierigkeit":      fall.get("case_difficulty", ""),
        "fairness_risiko":    fall.get("fairness_risk", ""),
        "gold_im_korpus":     fall.get("gold_im_korpus", True),
        "rang_kandidat":      rang_kand,
        "rang_quelle":        rang_quelle,
        "rang_chunk":         rang_chunk,
        "rang_arm":           rang_arm,
        "gold_in_kandidaten": rang_kand is not None,
        "gold_in_quellen":    rang_quelle is not None,
        "begriffe_quote":     quote,
        "begriffe_fehlend":   fehlend,
        "antwort_status":     status,
        "herkunft_antwort":   herkunft,
        "widerspruch":        widerspruch,
        "verbotene":          verbotene,
        "fragment_im_kontext": fragment,
        "diagnose":           diagnose,
        "vertrag_fall":       hash_fall(fall),
        "fakultaet_erkannt":  datensatz.get("fakultaet_erkannt", ""),
        "dauer_s":            datensatz.get("dauer_s"),
        "frage":              fall["question"],
    }


# Ein Lauf ist ein Verzeichnis mit manifest.json. `archiv/` sammelt überholte
# Läufe und ist selbst keiner — beide Auflistungen unten lassen es aus.
def vorhandene_laeufe() -> list[str]:
    return sorted(p.name for p in LAEUFE.glob("*")
                  if p.is_dir() and p.name != "archiv" and (p / "manifest.json").exists())


def letzter_lauf() -> str:
    """Der jüngste Lauf — spart das Abtippen der ID direkt nach einem Durchlauf."""
    laeufe = vorhandene_laeufe()
    if not laeufe:
        raise SystemExit(f"Keine Läufe unter {LAEUFE}")
    return max(laeufe, key=lambda n: (LAEUFE / n / "manifest.json").stat().st_mtime)


def loese_lauf_id(lauf: str | None, letzter: bool) -> str:
    if letzter:
        return letzter_lauf()
    if lauf:
        return lauf
    raise SystemExit("Lauf-ID angeben oder --letzter benutzen.")


def _git_commit() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                              cwd=konfig.WURZEL, capture_output=True, text=True,
                              timeout=10).stdout.strip() or None
    except Exception:
        return None


def schreibe_zusammenfassung(lauf_id: str, bewertet: list[dict]) -> Path:
    """Maschinenlesbarer Export eines ausgewerteten Laufs.

    Wird von keinem Code hier gelesen — sie existiert für die Weitergabe: Wer
    unsere Zahlen nachrechnen will, braucht dafür nicht unseren Code. `stand`
    hält fest, wann und mit welchem Commit ausgewertet wurde; ändert sich die
    Auswertungslogik, ist eine ältere Datei daran erkennbar.
    """
    ziel = LAEUFE / lauf_id / "zusammenfassung.json"
    ziel.write_text(json.dumps({
        "stand": {"erzeugt": datetime.now().isoformat(timespec="seconds"),
                  "git_commit": _git_commit()},
        "kennzahlen": kennzahlen(bewertet),
        "faelle": bewertet,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return ziel


FALLBASIS = Path(__file__).resolve().parent / "fallbasis.json"


def lade_lauf(lauf_id: str, vertrag_aktuell: bool = False) -> tuple[dict, list[dict]]:
    """Lädt Rohdaten und bewertet sie.

    Standard ist der Vertrag, der zur Laufzeit galt — er liegt in jedem
    faelle/qXX.json unter `fall` und ist Teil des Belegs. `vertrag_aktuell`
    tauscht ihn gegen die heutige fallbasis.json: Rohdaten bleiben, nur die
    Messlatte ist eine andere. Der Kopf der Ausgabe sagt das dann, und der
    Vergleich mit einer Bewertung nach altem Vertrag ist keine Messung, sondern
    zeigt, was der Vertrag selbst an der Zahl ausmacht.
    """
    d = LAEUFE / lauf_id
    if not d.exists():
        raise SystemExit(f"Kein Lauf: {d}")
    manifest = dict(json.loads((d / "manifest.json").read_text(encoding="utf-8")))
    faelle = [json.loads(p.read_text(encoding="utf-8"))
              for p in sorted((d / "faelle").glob("*.json"))]

    # Vertrag des Laufs: aus dem Manifest (seit 15.09.2026) oder aus den
    # Rohdaten rekonstruiert (ältere Läufe) — dann nur über die gelaufenen Fälle.
    fb = manifest.get("fallbasis") or {}
    if not fb.get("hash"):
        fb = {"hash": None, "hash_auswahl": hash_aus_faellen(faelle), "rekonstruiert": True}
    manifest["fallbasis"] = fb

    if vertrag_aktuell:
        liste = lade_fallbasis(FALLBASIS)
        aktuell = {f["id"]: f for f in liste}
        gestrichen = [f["fall"]["id"] for f in faelle if f["fall"]["id"] not in aktuell]
        faelle = [f for f in faelle if f["fall"]["id"] in aktuell]
        for f in faelle:
            f["fall"] = aktuell[f["fall"]["id"]]
        manifest["fallbasis"] = {**fb, "bewertet_mit": hash_fallbasis(liste)}
        manifest["vertrag"] = (f"aktuell {kurz(hash_fallbasis(liste))} statt Lauf-Vertrag "
                               f"{kurz(fb.get('hash') or fb.get('hash_auswahl'))}"
                               + (f" — nicht mehr im Vertrag, weggelassen: {', '.join(gestrichen)}"
                                  if gestrichen else ""))
    nur_retrieval = manifest.get("modus") == "retrieval"
    return manifest, [bewerte_fall(f, nur_retrieval=nur_retrieval) for f in faelle]


# ---------------------------------------------------------------------------
# Die zwei Hauptansichten — von Terminal und HTML-Bericht gemeinsam genutzt
# ---------------------------------------------------------------------------

# Gemeinsame Farb-/Namenszuordnung. Trichterverlust und Quadrant benennen
# dieselben Ursachen — wer beide Ansichten liest, soll die Verbindung nicht
# suchen müssen.
URSACHEN = {
    "korpus_luecke":   ("Korpuslücke",      "nie gecrawlt",                     "#9aa0a6"),
    "retrieval":       ("Retrievalproblem", "Retrieval findet es nicht",        "#c0392b"),
    "kandidaten":      ("Kandidatenlimit",  "in Dedup/Limit verloren",          "#d68910"),
    "auswahl":         ("Auswahlfehler",    "LLM-Auswahl verwirft es",          "#8e44ad"),
    "antwort":         ("Antwortproblem",   "Antwort nutzt das Dokument nicht", "#1f6feb"),
    "treffer":         ("Treffer",          "",                                 "#1e8e3e"),
    "vertrag":         ("Treffer (andere Quelle)", "Antwort richtig, Gold nicht gefunden", "#00838f"),
}


def _antwort_ok(b: dict) -> bool:
    """Enthält die Antwort alle Pflichtbegriffe? Nicht prüfbar zählt als ok."""
    return b["antwort_status"] in ("vollstaendig", "nicht_pruefbar")


def trichter(bewertet: list[dict]) -> list[dict]:
    """Die Stufen von der gestellten Frage bis zur vollständigen Antwort.

    Jede Stufe ist eine echte Teilmenge der vorigen — sonst wäre die Grafik
    eine Lüge. Deshalb werden die Bedingungen geschachtelt geprüft und nicht
    einzeln gezählt: Ein Fall, der über den Keyword-Boost in die Kandidaten
    gerät, ohne unter den Chunks zu stehen, würde sonst eine Stufe „wachsen"
    lassen.
    """
    rest = list(bewertet)
    stufen = [{"label": "gestellt", "faelle": rest, "verlust": None}]

    def stufe(label: str, ursache: str, pruefung) -> None:
        vorher = stufen[-1]["faelle"]
        behalten = [b for b in vorher if pruefung(b)]
        stufen.append({"label": label, "faelle": behalten, "verlust": ursache})

    stufe("Goldquelle im Korpus",   "korpus_luecke", lambda b: b["gold_im_korpus"])
    stufe("von der Suche gefunden", "retrieval",     lambda b: b["rang_chunk"] is not None)
    stufe("in der Kandidatenliste", "kandidaten",    lambda b: b["gold_in_kandidaten"])
    stufe("von Runde 1 behalten",   "auswahl",       lambda b: b["gold_in_quellen"])
    stufe("Antwort vollständig",    "antwort",       _antwort_ok)

    ergebnis = []
    for i, s in enumerate(stufen):
        verloren = (len(stufen[i - 1]["faelle"]) - len(s["faelle"])) if i else 0
        ergebnis.append({
            "label":     s["label"],
            "anzahl":    len(s["faelle"]),
            "verloren":  verloren,
            "ursache":   s["verlust"],
            "ids":       [b["id"] for b in s["faelle"]],
            "verlust_ids": ([b["id"] for b in stufen[i - 1]["faelle"]
                             if b not in s["faelle"]] if i else []),
        })
    return ergebnis


def quadranten(bewertet: list[dict]) -> dict[str, list[dict]]:
    """Zwei Achsen: Fand die Suche das Dokument? Stimmt die Antwort?

    Korpuslücken bleiben außen vor — ein nicht gecrawltes Dokument ist kein
    Retrievalfehler, und es in den Quadranten zu zeigen hieße, dem Retrieval
    etwas anzulasten, das der Crawler verantwortet.

    Die Ansicht fasst die Stufe „Kandidat → Quelle" mit ein: Ein Fall, den
    Runde 1 verwirft, dessen Antwort aber trotzdem stimmt, steht hier unter
    Treffer und im Trichter als Auswahlverlust. Beide Schnitte sind richtig,
    sie beantworten verschiedene Fragen.
    """
    # Eine Korpuslücke mit richtiger Antwort (q09 bis 14.09.2026) gehört
    # trotzdem hinein — sie ist ein Treffer aus anderer Quelle.
    moeglich = [b for b in bewertet if (b["gold_im_korpus"] or ist_treffer(b))
                and b["diagnose"] != "unbestimmt"]
    felder = {"treffer": [], "antwort": [], "vertrag": [], "retrieval": []}
    for b in moeglich:
        gefunden, ok = b["gold_in_kandidaten"], _antwort_ok(b)
        schluessel = ("treffer" if ok else "antwort") if gefunden else \
                     ("vertrag" if ok else "retrieval")
        felder[schluessel].append(b)
    return felder


def unbestimmt_wo(bewertet: list[dict]) -> dict[str, list[dict]]:
    """Wo das Gold der nicht prüfbaren Fälle verlorenging.

    „unbestimmt" heißt: keine Pflichtbegriffe und Gold nicht in den Quellen.
    Ob die Antwort stimmt, weiß niemand — aber *wo* das Gold blieb, steht in
    den Rängen. Ohne diese Aufschlüsselung zeigt die Vier-Felder-Ansicht
    „Retrievalproblem 0", während der Trichter elf Retrievalverluste nennt:
    beides richtig, zusammen irreführend.
    """
    wo = {"retrieval": [], "kandidaten": [], "auswahl": []}
    for b in bewertet:
        if b["diagnose"] != "unbestimmt":
            continue
        if b["rang_chunk"] is None:
            wo["retrieval"].append(b)
        elif b["rang_kandidat"] is None:
            wo["kandidaten"].append(b)
        else:
            wo["auswahl"].append(b)
    return wo


def kennzahlen(bewertet: list[dict]) -> dict:
    moeglich = [b for b in bewertet if b["gold_im_korpus"]]
    r_kand = [b["rang_kandidat"] for b in moeglich if b["rang_kandidat"]]
    return {
        "faelle_gesamt":     len(bewertet),
        "faelle_moeglich":   len(moeglich),
        "gold_in_kandidaten": sum(1 for b in moeglich if b["gold_in_kandidaten"]),
        "gold_in_quellen":   sum(1 for b in moeglich if b["gold_in_quellen"]),
        "gold_kandidat_rang1": sum(1 for b in moeglich if b["rang_kandidat"] == 1),
        "median_rang_kandidat": statistics.median(r_kand) if r_kand else None,
        "diagnosen":         dict(Counter(b["diagnose"] for b in bewertet)),
        "treffer":           sum(1 for b in bewertet if ist_treffer(b)),
        "unbestimmt":        sum(1 for b in bewertet if b["diagnose"] == "unbestimmt"),
        "treffer_gold":      sum(1 for b in bewertet if b["diagnose"] == "treffer"),
        "widersprueche":     sum(1 for b in bewertet if b["widerspruch"]),
        "herkunft":          dict(Counter(b["herkunft_antwort"] for b in bewertet)),
        "fragmente":         sum(1 for b in bewertet if b["fragment_im_kontext"]),
    }


def _balken(anzahl: int, gesamt: int, breite: int = 42) -> str:
    """Volle Zeilenbreite: gefüllt für Behaltenes, Punkte für Verlorenes.

    Der Verlust MUSS sichtbar bleiben. Ein sich verjüngender Trichter — die
    übliche Darstellung — verzerrt die Verhältnisse und lässt gerade das
    verschwinden, worum es geht.
    """
    voll = round(anzahl / max(gesamt, 1) * breite)
    return "█" * voll + "░" * (breite - voll)


def zeige(manifest: dict, bewertet: list[dict], detail: bool) -> None:
    k = manifest["konfiguration"]
    z = kennzahlen(bewertet)

    # ── Kopf: die Konfiguration entscheidet über Vergleichbarkeit ─────────
    print("=" * 78)
    print(f"{manifest['lauf_id']}")
    print(f"  Modell {manifest['modell'] or '— (nur Retrieval)'}   ·   Index {k.get('points_count')} Punkte")
    reranker = k.get("reranker_aktiv")
    if reranker is None:                       # Manifest von vor dem 21.09.2026
        reranker = "unbekannt (Konstante USE_RERANKER war " + str(k.get("USE_RERANKER")) + ", der Server rief keinen auf)"
    print(f"  Arme {k.get('DENSE_ARMS')} / {k.get('SPARSE_ARMS')}   ·   Fusion {k.get('FUSION')}"
          f"   ·   Reranker {reranker}")
    extra = []
    for name in ("CANDIDATE_LIMIT", "DEDUP_BUFFER", "MAX_PER_URL", "TOP_DOCS", "keyword_boost"):
        if name in k:
            extra.append(f"{name} {k[name]}")
    if extra:
        print("  " + " · ".join(extra) + ("   (wirksam, vom Server gemeldet)" if k.get("_wirksam") else "   (Modulkonstanten)"))
    if manifest.get("experiment"):
        print(f"  Experiment: {json.dumps(manifest['experiment'], ensure_ascii=False)}")
    if manifest.get("notiz"):
        print(f"  {manifest['notiz']}")
    fb = manifest.get("fallbasis") or {}
    if fb.get("rekonstruiert"):
        print(f"  Vertrag {kurz(fb.get('hash_auswahl'))} (rekonstruiert aus faelle/, "
              f"Lauf von vor der Versionierung)")
    else:
        print(f"  Vertrag {kurz(fb.get('hash'))} · gelaufene Auswahl {kurz(fb.get('hash_auswahl'))}"
              f" · Stand {str(fb.get('stand') or '')[:10]}")
    if manifest.get("vertrag"):
        print(f"  ⚠ Bewertet mit anderem Vertrag: {manifest['vertrag']}")
    lz = manifest.get("laufzeit") or {}
    if lz:
        print(f"  {lz['wall_clock_s']/60:.1f} min gesamt "
              f"(Arbeit {lz['gemessen_s']/60:.1f} · Pausen {lz['geplante_pausen_s']/60:.1f} "
              f"· ungetrackt {lz['ungetrackt_s']/60:.1f})")
    print("=" * 78)

    nur_retrieval = manifest.get("modus") == "retrieval"

    # ── 1. Trichter ───────────────────────────────────────────────────────
    stufen = trichter(bewertet)
    if nur_retrieval:
        stufen = stufen[:4]                 # bis „in der Kandidatenliste“
    gesamt = stufen[0]["anzahl"]
    print("\nWO DIE FÄLLE VERLORENGEHEN" + ("   (nur Retrieval, kein LLM)" if nur_retrieval else "") + "\n")
    for s in stufen:
        print(f"  {s['label']:<26} {_balken(s['anzahl'], gesamt)} {s['anzahl']:>3}")
        if s["verloren"]:
            _, grund, _ = URSACHEN[s["ursache"]]
            # Die Verlustzeile beginnt dort, wo der Balken aufhört — so steht
            # der Grund optisch in der Lücke, die er erklärt.
            einzug = 28 + round(s["anzahl"] / max(gesamt, 1) * 42)
            print(" " * einzug + f"└ {s['verloren']} {grund}: {', '.join(s['verlust_ids'])}")

    if nur_retrieval:
        zeige_retrieval(bewertet, detail)
        return

    # ── 2. Quadranten ─────────────────────────────────────────────────────
    q = quadranten(bewertet)
    luecken = sum(1 for b in bewertet if b["diagnose"] == "korpus_luecke")
    print("\n\nWORAN ES LIEGT"
          f"   (ohne die {luecken} Korpuslücken)\n")

    def zelle(schluessel: str) -> list[str]:
        name = URSACHEN[schluessel][0]
        faelle = q[schluessel]
        zeilen_ = [f"{name} ({len(faelle)})"]
        ids = [b["id"] for b in faelle]
        for i in range(0, len(ids), 5):
            zeilen_.append("  " + " ".join(ids[i:i + 5]))
        return zeilen_

    kopf = ("Antwort vollständig", "Antwort unvollständig")
    print(f"  {'':<20} {kopf[0]:<30} {kopf[1]:<30}")
    for zeile_label, links, rechts in (("Gold gefunden", "treffer", "antwort"),
                                       ("Gold nicht gefunden", "vertrag", "retrieval")):
        a, b_ = zelle(links), zelle(rechts)
        for i in range(max(len(a), len(b_))):
            vorn = zeile_label if i == 0 else ""
            print(f"  {vorn:<20} {a[i] if i < len(a) else '':<30} "
                  f"{b_[i] if i < len(b_) else '':<30}")
        print()

    print(f"  Treffer gesamt {z['treffer']}  ·  davon aus der Goldquelle {z['treffer_gold']}"
          f"  ·  unbestimmt (keine Begriffe, Gold nicht in Quellen) {z['unbestimmt']}"
          f"  ·  Widerspruch gemeldet {z['widersprueche']}")
    if z["unbestimmt"]:
        wo = unbestimmt_wo(bewertet)
        print(f"  Unbestimmt, Gold verloren bei: Suche {len(wo['retrieval'])} · "
              f"Kandidatenlimit {len(wo['kandidaten'])} · Auswahl {len(wo['auswahl'])}"
              f"   ({' '.join(b['id'] for b in wo['retrieval'])})")
    herk = z["herkunft"]
    print(f"  Herkunft der Antworten: gold {herk.get('gold', 0)} · akzeptiert {herk.get('akzeptiert', 0)}"
          f" · andere {herk.get('andere', 0)} · keine {herk.get('keine', 0)}")

    # ── 3. Aufschlüsselung ────────────────────────────────────────────────
    moeglich = [b for b in bewertet if b["gold_im_korpus"]]
    print("\nNACH GRUPPEN\n")
    print(f"  {'':<16} {'':<18} {'Fälle':>6} {'Gold gef.':>10} {'Treffer':>8}")
    for feld, titel in (("schwierigkeit", "Schwierigkeit"), ("herkunft", "Herkunft")):
        gruppen: dict[str, list] = {}
        for b in moeglich:
            gruppen.setdefault(b[feld] or "—", []).append(b)
        for i, (name, v) in enumerate(sorted(gruppen.items())):
            print(f"  {titel if i == 0 else '':<16} {name:<18} {len(v):>6} "
                  f"{sum(1 for x in v if x['gold_in_kandidaten']):>10} "
                  f"{sum(1 for x in v if ist_treffer(x)):>8}")
        print()

    ref = REFERENZ_CB
    print(f"Vergleichsmarke — {ref['lauf']}:")
    print(f"  Gold gefunden {ref['gold_gefunden'][0]}/{ref['gold_gefunden'][1]}, "
          f"Rang 1 {ref['gold_rang1'][0]}/{ref['gold_rang1'][1]}, gemessen über K={ref['k']}.")
    print("  Unsere Stufen benutzen andere K — vergleichbar ist die Richtung, "
          "nicht die Nachkommastelle.")

    # ── 4. Fall-Karten, nach Handlungsbedarf sortiert ─────────────────────
    if detail:
        zu_quadrant = {b["id"]: s for s, liste in q.items() for b in liste}
        reihenfolge = {"retrieval": 0, "antwort": 1, "vertrag": 2, "treffer": 3}
        sortiert = sorted(bewertet, key=lambda b: (
            reihenfolge.get(zu_quadrant.get(b["id"], ""), 4), b["id"]))
        print("\n" + "=" * 78)
        print("EINZELFÄLLE — nach Handlungsbedarf")
        for b in sortiert:
            kategorie = URSACHEN[zu_quadrant[b["id"]]][0] if b["id"] in zu_quadrant \
                        else URSACHEN["korpus_luecke"][0]
            print("\n" + "-" * 78)
            print(f"{b['id']}  {kategorie.upper():<18} "
                  f"{b['schwierigkeit']} · fairness {b['fairness_risiko']} · {b['herkunft']}")
            print(f"  {b['frage']}")
            weg = (f"Chunk {b['rang_chunk'] or '—'} → Kandidat {b['rang_kandidat'] or '—'} "
                   f"→ Quelle {b['rang_quelle'] or '—'}")
            print(f"  {weg}    Gold: {b['gold_url'][:52]}")
            print(f"  Diagnose {b['diagnose']} · Antwort {b['antwort_status']} · aus {b['herkunft_antwort']}"
                  + (" · ⚠ Widerspruch gemeldet" if b["widerspruch"] else ""))
            if b["begriffe_fehlend"]:
                print(f"  fehlend: {', '.join(b['begriffe_fehlend'])}")
            if b["fragment_im_kontext"]:
                print("  ⚠ Fragment-Dokument im Kontext")
            if b["antwort_auszug"]:
                print(f"  Antwort: {b['antwort_auszug'][:150]}")


def zeige_retrieval(bewertet: list[dict], detail: bool) -> None:
    """Zweite Hälfte der Ausgabe für Retrieval-Läufe: Ränge statt Antworten.

    Die Frage hier ist nicht „stimmt die Antwort", sondern „wie weit oben steht
    das Gold" — denn Runde 1 sieht nur die Kandidatenliste, und ein Gold auf
    Rang 8 hat schlechtere Karten als eines auf Rang 1.
    """
    moeglich = [b for b in bewertet if b["gold_im_korpus"]]
    gefunden = [b for b in moeglich if b["gold_in_kandidaten"]]
    raenge = sorted(b["rang_kandidat"] for b in gefunden)
    print("\n\nWO DAS GOLD STEHT   (Rang in der Kandidatenliste)\n")
    klassen = (("Rang 1", lambda r: r == 1), ("Rang 2–3", lambda r: 2 <= r <= 3),
               ("Rang 4–8", lambda r: 4 <= r <= 8), ("Rang > 8", lambda r: r > 8))
    for label, test in klassen:
        ids = [b["id"] for b in gefunden if test(b["rang_kandidat"])]
        print(f"  {label:<10} {len(ids):>3}   {' '.join(ids)}")
    if raenge:
        print(f"\n  Median Rang {statistics.median(raenge):g} · gefunden {len(gefunden)}/{len(moeglich)}"
              f" · Rang 1 {sum(1 for r in raenge if r == 1)}/{len(moeglich)}")

    # ── Welcher Arm trägt das Gold? ───────────────────────────────────────
    arme = [a for a in ("text", "questions", "sparse", "sparse_expanded")
            if any(a in b["rang_arm"] for b in moeglich)]
    if arme:
        print("\n\nWELCHER ARM SIEHT DAS GOLD   (Rang in den Top-50 des Arms allein, — = nicht)\n")
        kopf = "  " + f"{'Fall':<5}" + "".join(f"{a[:8]:>10}" for a in arme) + "   Kandidat"
        print(kopf)
        for b in sorted(moeglich, key=lambda b: (b["rang_kandidat"] or 999, b["id"])):
            zeile = "  " + f"{b['id']:<5}"
            for a in arme:
                r = b["rang_arm"].get(a)
                zeile += f"{(str(r) if r else '—'):>10}"
            zeile += f"   {b['rang_kandidat'] or '—'}"
            print(zeile)
        print()
        n = len(moeglich)
        for a in arme:
            in10 = sum(1 for b in moeglich if b["rang_arm"].get(a) and b["rang_arm"][a] <= 10)
            in50 = sum(1 for b in moeglich if b["rang_arm"].get(a))
            exklusiv = sum(1 for b in moeglich if b["rang_arm"].get(a)
                           and not any(b["rang_arm"].get(x) for x in arme if x != a))
            print(f"  {a:<16} in Top-10 {in10:>3}/{n}   in Top-50 {in50:>3}/{n}   "
                  f"als einziger Arm {exklusiv:>2}")
        keiner = [b["id"] for b in moeglich if not any(b["rang_arm"].get(a) for a in arme)]
        print(f"  {'kein Arm':<16} {len(keiner):>3}   {' '.join(keiner)}")
        # Gold, das ein Arm sieht, die Fusion aber nicht bis in die Kandidaten trägt
        vertan = [b["id"] for b in moeglich if not b["gold_in_kandidaten"]
                  and any(b["rang_arm"].get(a) and b["rang_arm"][a] <= 10 for a in arme)]
        print(f"  {'Arm ≤10, Kand. —':<16} {len(vertan):>3}   {' '.join(vertan)}"
              "   ← ein Arm hatte es oben, Fusion/Dedup haben es verloren")

    verloren = [b for b in moeglich if not b["gold_in_kandidaten"]]
    if verloren:
        print("\nNICHT IN DER KANDIDATENLISTE\n")
        for b in sorted(verloren, key=lambda b: (b["diagnose"], b["id"])):
            wo = f"Chunk-Rang {b['rang_chunk']}" if b["rang_chunk"] else "nicht unter den Chunks"
            print(f"  {b['id']}  {b['diagnose']:<16} {wo:<24} {b['frage'][:50]}")

    if detail:
        print("\n" + "=" * 78)
        print("EINZELFÄLLE")
        for b in sorted(bewertet, key=lambda b: (b["rang_kandidat"] or 999, b["id"])):
            print(f"  {b['id']}  Chunk {b['rang_chunk'] or '—':<3} → Kandidat {b['rang_kandidat'] or '—':<3}"
                  f"  {b['diagnose']:<16} {('[' + b['fakultaet_erkannt'] + '] ') if b['fakultaet_erkannt'] else ''}"
                  f"{b['frage'][:56]}")


def vergleiche(a: str, b: str) -> None:
    ma, ba = lade_lauf(a)
    mb, bb = lade_lauf(b)
    modus_a, modus_b = ma.get("modus", "voll"), mb.get("modus", "voll")
    if modus_a != modus_b:
        raise SystemExit(f"Verschiedene Modi: {a} = {modus_a}, {b} = {modus_b} — "
                         f"ein Retrieval-Lauf hat keine Antworten, ein Volllauf misst "
                         f"dieselben Kandidaten mit LLM dahinter. Nicht vergleichbar.")
    nur_retrieval = modus_a == "retrieval"
    da = {x["id"]: x for x in ba}
    db = {x["id"]: x for x in bb}
    gemeinsam = [i for i in da if i in db]

    # Gleicher Vertrag? Sonst ist der Vergleich keine Messung — sagen, nicht
    # verschweigen. Fallweise über vertrag_fall, damit auch ein Lauf von vor der
    # Versionierung geprüft werden kann.
    nur_a = [i for i in da if i not in db]
    nur_b = [i for i in db if i not in da]
    anders = [i for i in gemeinsam if da[i]["vertrag_fall"] != db[i]["vertrag_fall"]]
    fa, fbm = ma.get("fallbasis") or {}, mb.get("fallbasis") or {}
    ka = kurz(fa.get("hash") or fa.get("hash_auswahl"))
    kb = kurz(fbm.get("hash") or fbm.get("hash_auswahl"))
    if nur_a or nur_b or anders:
        print(f"⚠ Verschiedene Verträge: {a} = {ka}, {b} = {kb}")
        if nur_a:
            print(f"  nur in {a[:20]}: {', '.join(nur_a)}")
        if nur_b:
            print(f"  nur in {b[:20]}: {', '.join(nur_b)}")
        if anders:
            print(f"  anderer Fallvertrag ({len(anders)}): {', '.join(anders)}")
        print("  Die Zahlen unten vergleichen Läufe gegen verschiedene Messlatten.\n")
    else:
        print(f"Gleicher Vertrag {ka} in beiden Läufen.")

    za, zb = kennzahlen([da[i] for i in gemeinsam]), kennzahlen([db[i] for i in gemeinsam])
    print(f"Verglichene Fälle: {len(gemeinsam)}\n")
    print(f"{'':28} {a[:20]:>22} {b[:20]:>22}")
    print("-" * 74)
    for feld, label in (("gold_in_kandidaten", "Gold in Kandidaten"),
                        ("gold_kandidat_rang1", "davon Rang 1"),
                        ("gold_in_quellen",   "Gold in Quellen"),
                        ("fragmente",         "Fragment im Kontext")):
        print(f"{label:28} {za[feld]:>22} {zb[feld]:>22}")
    print(f"{'Suche':28} {suche_kurz(ma):>22} {suche_kurz(mb):>22}")

    # Im Retrieval-Modus heißt „getroffen": Gold steht in der Kandidatenliste.
    erfolg = (lambda x: x["gold_in_kandidaten"]) if nur_retrieval else ist_treffer
    besser = [i for i in gemeinsam if not erfolg(da[i]) and erfolg(db[i])]
    schlechter = [i for i in gemeinsam if erfolg(da[i]) and not erfolg(db[i])]
    was = "Neu in Kandidaten" if nur_retrieval else "Neu getroffen"
    print(f"\n{was} ({len(besser)}): {', '.join(besser) or '—'}")
    print(f"Verloren ({len(schlechter)}): {', '.join(schlechter) or '—'}")
    if nur_retrieval:
        bewegt = [(i, da[i]["rang_kandidat"], db[i]["rang_kandidat"]) for i in gemeinsam
                  if da[i]["rang_kandidat"] and db[i]["rang_kandidat"]
                  and da[i]["rang_kandidat"] != db[i]["rang_kandidat"]]
        if bewegt:
            print("Rang verändert: " + ", ".join(f"{i} {ra}→{rb}" for i, ra, rb in bewegt))
    for i in schlechter:
        print(f"   {i}: {db[i]['diagnose']}  — {db[i]['frage'][:56]}")


def suche_kurz(manifest: dict) -> str:
    """Arme und Fusion in einem Kürzel: t_se/dbsf, t+q_s/rrf …"""
    k = manifest.get("konfiguration") or {}
    ab = {"text": "t", "questions": "q", "sparse": "s", "sparse_expanded": "se"}
    d = "+".join(ab.get(a, a) for a in (k.get("DENSE_ARMS") or [])) or "-"
    sp = "+".join(ab.get(a, a) for a in (k.get("SPARSE_ARMS") or [])) or "-"
    kurz_ = f"{d}_{sp}/{k.get('FUSION', '?')}"
    if k.get("DEDUP_BUFFER") not in (None, 4):
        kurz_ += f" db{k['DEDUP_BUFFER']}"
    if k.get("TOP_DOCS") not in (None, 8):
        kurz_ += f" td{k['TOP_DOCS']}"
    if k.get("keyword_boost") is False:
        kurz_ += " nb"
    return kurz_


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95-%-Wilson-Intervall für eine Quote k/n — damit neben 46/73 steht, was
    das bei n = 73 überhaupt heißt (± zehn Punkte)."""
    if n == 0:
        return (0.0, 0.0)
    ph = k / n
    mitte = (ph + z * z / (2 * n)) / (1 + z * z / n)
    halb = z * ((ph * (1 - ph) / n + z * z / (4 * n * n)) ** 0.5) / (1 + z * z / n)
    return (max(0.0, mitte - halb), min(1.0, mitte + halb))


def gitter(basis_id: str, vertrag_aktuell: bool = False) -> None:
    """Alle Retrieval-Läufe mit dem Vertrag der Basis, paarweise gegen sie.

    Eine Zeile je Lauf: Suche, Gold in Kandidaten (mit Wilson-Intervall),
    Rang 1, Median, und gegen die Basis: neu gefunden / verloren — gesamt und
    getrennt nach Herkunft, denn eine Änderung, die nur die 25 bekannten Fälle
    rettet, ist auf sie zugeschnitten, nicht besser.
    """
    mb, bb = lade_lauf(basis_id, vertrag_aktuell)
    if mb.get("modus") != "retrieval":
        raise SystemExit("Die Basis muss ein Retrieval-Lauf sein.")
    fb = (mb.get("fallbasis") or {}).get("hash")
    basis = {x["id"]: x for x in bb if x["gold_im_korpus"]}
    gruppen = sorted({x["herkunft"] for x in basis.values()})

    zeilen = []
    for lauf_id in vorhandene_laeufe():
        try:
            m, b = lade_lauf(lauf_id, vertrag_aktuell)
        except Exception:
            continue
        if m.get("modus") != "retrieval":
            continue
        # Bei --vertrag aktuell werden alle mit demselben heutigen Vertrag
        # bewertet; der Hash des Laufs ist dann kein Ausschlusskriterium mehr,
        # die Fallmenge (unten) schon.
        if not vertrag_aktuell and (m.get("fallbasis") or {}).get("hash") != fb:
            continue
        d = {x["id"]: x for x in b if x["gold_im_korpus"] and x["id"] in basis}
        if len(d) != len(basis):
            continue                          # Teil-Lauf (--faelle/--limit)
        n = len(d)
        gef = [x for x in d.values() if x["gold_in_kandidaten"]]
        raenge = sorted(x["rang_kandidat"] for x in gef)
        neu = [i for i in d if d[i]["gold_in_kandidaten"] and not basis[i]["gold_in_kandidaten"]]
        weg = [i for i in d if not d[i]["gold_in_kandidaten"] and basis[i]["gold_in_kandidaten"]]
        je_gruppe = {}
        for g in gruppen:
            ids = [i for i in d if d[i]["herkunft"] == g]
            je_gruppe[g] = (sum(1 for i in ids if d[i]["gold_in_kandidaten"]), len(ids),
                            sum(1 for i in ids if i in neu), sum(1 for i in ids if i in weg))
        lo, hi = wilson(len(gef), n)
        zeilen.append({
            "lauf": lauf_id, "suche": suche_kurz(m), "gef": len(gef), "n": n,
            "ci": (lo, hi), "rang1": sum(1 for r in raenge if r == 1),
            "median": statistics.median(raenge) if raenge else None,
            "neu": neu, "weg": weg, "gruppen": je_gruppe, "basis": lauf_id == basis_id,
        })

    if vertrag_aktuell:
        print(f"GITTER — alle Retrieval-Läufe mit dieser Fallmenge, paarweise gegen {basis_id}\n"
              f"⚠ bewertet mit dem AKTUELLEN Vertrag {kurz((mb.get('fallbasis') or {}).get('bewertet_mit'))}, "
              f"nicht dem der Läufe\n")
    else:
        print(f"GITTER — Retrieval-Läufe mit Vertrag {kurz(fb)}, paarweise gegen {basis_id}\n")
    kopf = f"{'Suche':<20} {'Kandidat':>9} {'95 %':>13} {'Rang1':>6} {'Med':>4} {'+':>3} {'−':>3}"
    for g in gruppen:
        kopf += f"   {g[:12]:>12}"
    print(kopf)
    print("-" * len(kopf))
    for z in sorted(zeilen, key=lambda z: (-z["gef"], z["suche"])):
        lo, hi = z["ci"]
        zeile = (f"{z['suche']:<20} {z['gef']:>5}/{z['n']:<3} {lo*100:>5.0f}–{hi*100:<5.0f}  "
                 f"{z['rang1']:>6} {z['median'] if z['median'] is not None else '—':>4} "
                 f"{len(z['neu']):>3} {len(z['weg']):>3}")
        for g in gruppen:
            gef, n, neu, weg = z["gruppen"][g]
            zeile += f"   {gef:>3}/{n:<3} +{neu}−{weg}"
        print(zeile + ("   ← Basis" if z["basis"] else ""))
    print("\n+ / − : gegen die Basis neu in der Kandidatenliste / aus ihr verloren (paarweise je Fall).")
    print("\nJe Lauf, welche Fälle:")
    for z in sorted(zeilen, key=lambda z: (-z["gef"], z["suche"])):
        if z["basis"]:
            continue
        print(f"  {z['suche']:<20} + {', '.join(z['neu']) or '—'}")
        print(f"  {'':<20} − {', '.join(z['weg']) or '—'}")


def uebersicht() -> None:
    """Alle Läufe mit ihrem Ergebnis in je einer Zeile.

    Ersetzt die frühere nackte Verzeichnisliste: Die Frage nach einem Lauf ist
    fast nie „wie heißt er", sondern „was kam raus" — und dafür musste man
    bisher drei Auswertungen nacheinander starten.
    """
    laeufe = vorhandene_laeufe()
    if not laeufe:
        raise SystemExit(f"Keine Läufe unter {LAEUFE}")
    print(f"{'Lauf':<44} {'Vertrag':<9} {'Modus':<5} {'Suche':<16} {'Fälle':>6} {'Gold':>6} {'Treffer':>8}  Notiz")
    print("-" * 128)
    for lauf_id in laeufe:
        try:
            manifest, bewertet = lade_lauf(lauf_id)
        except Exception as exc:                      # unvollständiger Ordner
            print(f"{lauf_id:<44} {'—':<9} {'—':<5} {'—':<16} {'—':>6} {'—':>6} {'—':>8}  ({type(exc).__name__})")
            continue
        z = kennzahlen(bewertet)
        treffer = sum(1 for b in bewertet if ist_treffer(b))
        fb = manifest.get("fallbasis") or {}
        # Ohne Manifest-Eintrag nur der Auswahl-Hash, mit * gekennzeichnet.
        vertrag = kurz(fb.get("hash")) if fb.get("hash") else kurz(fb.get("hash_auswahl")) + "*"
        modus = "retr" if manifest.get("modus") == "retrieval" else "voll"
        treffer_s = "—" if modus == "retr" else (str(treffer) + (f"+{z['unbestimmt']}?" if z["unbestimmt"] else ""))
        print(f"{lauf_id:<44} {vertrag:<9} {modus:<5} {suche_kurz(manifest):<16} {z['faelle_gesamt']:>6} "
              f"{z['gold_in_kandidaten']:>3}/{z['faelle_moeglich']:<2} {treffer_s:>8}  "
              f"{manifest.get('notiz') or ''}")
    print("\n* Vertrag aus den Rohdaten rekonstruiert (Lauf von vor der Versionierung); "
          "gleiche Kürzel = gleicher Vertrag.")
    print(f"Auswertung: python auswertung/gold/auswertung.py --letzter")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("lauf", nargs="?", help="Lauf-ID unter laeufe/")
    ap.add_argument("--letzter", action="store_true", help="den jüngsten Lauf nehmen")
    ap.add_argument("--detail", action="store_true", help="Tabelle je Fall")
    ap.add_argument("--vergleich", nargs=2, metavar=("LAUF_A", "LAUF_B"))
    ap.add_argument("--gitter", metavar="BASIS", help="alle Retrieval-Läufe gleichen Vertrags gegen BASIS")
    ap.add_argument("--vertrag", choices=("lauf", "aktuell"), default="lauf",
                    help="Vertrag zur Laufzeit (Beleg) oder heutige fallbasis.json")
    args = ap.parse_args()

    if args.vergleich:
        vergleiche(*args.vergleich)
        return
    if args.gitter:
        gitter(args.gitter, vertrag_aktuell=(args.vertrag == "aktuell"))
        return
    if not args.lauf and not args.letzter:
        uebersicht()
        return

    lauf_id = loese_lauf_id(args.lauf, args.letzter)
    manifest, bewertet = lade_lauf(lauf_id, vertrag_aktuell=(args.vertrag == "aktuell"))
    zeige(manifest, bewertet, args.detail)
    if args.vertrag == "lauf":
        # Die Zusammenfassung ist ein Erzeugnis des Laufs; eine Bewertung nach
        # fremdem Vertrag darf sie nicht überschreiben.
        schreibe_zusammenfassung(lauf_id, bewertet)


if __name__ == "__main__":
    main()
