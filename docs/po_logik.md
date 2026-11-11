# Prüfungsordnungen (PO-Logik)

Stand: 11.11.2026 (Code wie Übernahme 8c0d1d0, nur verschoben). Code:
`src/raghsh/po/pfad.py` (Erkennung, Vielfalt, Schranke, Vergleich,
Rückfrage, Kopfzeilen), `po/filter.py` (Fassungen je Studiengang, Qdrant-Filter,
Geltung), `po/werte.py` (Wertetabelle); Aufrufe in `frag_hsh`
(`server/mcp_server.py`). An den Konstanten stehen Anlass (Sitzung S0xx) und
Messung.

## Problem

Viele Studiengänge haben mehrere gleichzeitig geltende Fassungen ihrer
Prüfungsordnung (Studierende bleiben in ihrer Fassung). Dieselbe Frage — „Wie
lang ist die Bachelorarbeit?“ — hat dann je Fassung eine andere Antwort. Das
System soll die Fassung nennen, wenn die Frage sie angibt; nachfragen, wenn
die Antwort davon abhängt; und sonst normal antworten.

## Daten (`stammdaten/po/`)

| Datei | Inhalt |
|---|---|
| `studiengaenge.tsv` | Kürzel, Langform, Abschluss, Fakultät, dual |
| `kurzformen.tsv` | Kurzformen („BWL“) → Kürzel, mehrdeutige markiert |
| `po_zuordnung.tsv` | URL des PO-Dokuments → Studiengang, Fassung |
| `po_geltung.tsv` | Fassung gilt bis … (nur wo im Dokument belegt); `po_geltung_unklar.tsv` Rest |
| `po_punkte.tsv` | Vergleichspunkte (Bachelorarbeit, Praxisphase, …) mit Stichwörtern |
| `po_werte.tsv` | Wert je Studiengang, Fassung, Punkt mit wörtlichem Auszug und Qdrant-Punkt-ID |
| `module.tsv` | Modulkürzel → Modulname je Studiengang und Fassung |

Erzeugt bzw. geprüft von `werkzeuge/po/` (`extrahiere_geltung.py`,
`module_ziehen.py`, `po_werte_pruefen.py`). Nach einem Neuimport mit anderem
Chunking `po_werte_pruefen.py` laufen lassen: Die Punkt-IDs in `po_werte.tsv`
veralten sonst unbemerkt.

## Ablauf in `frag_hsh`

```
(1) Fakultät aus der Frage                       _fakultaet_aus_frage
(2) Studiengang bzw. Namensfamilie               _studiengang_familie_aus_frage
    mehrdeutige Kurzform ohne Studiengang        → Rückfrage nach dem Studiengang
(3) Fassung: Parameter po | „PO 2018“ | Neubeginn („ich fange an“ → jüngste aktuelle)
                                                 _po_aus_frage, _po_plan
(4) Suche: Fakultät des Studiengangs; bekannte Fassung → Filter schließt andere aus
(5) Vielfalt: ≥ 2 geltende Fassungen?            _po_vielfalt, _po_geltende (po_geltung.tsv)
(6) nur ohne jede PO-Angabe:
      Wertetabelle trifft genau einen Punkt      _po_tabellenurteil → verschieden | gleich
      sonst Schranke: kein PO-Dokument auf Rang ≤ PO_SCHRANKE_RANG → normale Antwort
      sonst LLM-Vergleich der Fassungen          _po_entscheiden (Nachsuche je Fassung,
                                                 bis 3 parallele Versuche, Grenze 60 s)
      Urteil „verschieden“                       → Rückfrage mit Fassungsliste (_po_rueckfrage)
(7) Antwort: „Gewählte PO“-Zeile in Runde 1, Kopfzeile je Dokument mit Fassung und Geltung
```

Rückfrage nur bei „verschieden“ und nur ohne PO-Angabe. Den Gesprächszustand
hält der Client: Die Web-App schickt die gewählte Fassung als `po` mit.

## Bekannte Grenzen (Analyse 02.11.2026)

Offline nachgewiesen, noch nicht behoben:

1. **Neubeginn-Muster zu weit:** „Wann *fange* ich mit der Bachelorarbeit *an*?“
   gilt als Studienbeginn und legt still die jüngste Fassung fest; die anderen
   werden herausgefiltert.
2. **Tabellenurteil umgeht die Schranke:** Fragen, die einen Punkt nur
   erwähnen („Wo finde ich einen Platz für die Praxisphase?“), lösen eine
   unnötige Rückfrage aus. Der Punkt `praxisphase` misst Credits; eine Frage
   nach Wochen wird danach beurteilt.
3. **Erkennung:** „BPO 2017“ wird nicht erkannt; eine genannte, aber nicht
   vorhandene Fassung wird still ignoriert; „seit 2019“ wird nicht genutzt;
   ein Studiengang ohne „Bachelor/Master“ („Ich studiere Elektrotechnik“) wird
   meist nicht erkannt.
4. **Geltungsdaten lückenhaft:** 54 von 85 Studiengängen ohne Zeile in
   `po_geltung.tsv`; Fassungen ohne Eintrag gelten unbegrenzt, die Rückfrage
   bietet dann Fassungen ab 2006 an. `AUSGELAUFEN_VOR = 2016` ist fest.
5. **Wertetabelle deckt nur die Studiengänge der Testmenge ab** (10 von 57
   mit mehreren geltenden Fassungen); Masterarbeit fehlt als Punkt.
6. **Web-App:** Die gewählte Fassung gilt nur für die nächste Frage.
7. **Doppelter Code:** Modulmuster in `pfad.py` und `werte.py` (weichen
   voneinander ab); drei verschiedene Zuordnungen URL → Fassung.

Vorschläge dazu: Analyse vom 02.11.2026 (Reihenfolge 1–3, dann 4/3/6, dann
Tabelle automatisch füllen).
