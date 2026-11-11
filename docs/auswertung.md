# Auswertung

Stand: 11.11.2026. Alles unter `auswertung/`.

| Ort | Inhalt |
|---|---|
| `gold/` | Fallbasis (`fallbasis.json`, Vertrag `fallbasis.sha256`), Messlauf `lauf.py`, deterministische Diagnose `auswertung.py`, HTML-Berichte. Ausführlich: [auswertung/gold/README.md](../auswertung/gold/README.md) |
| `laeufe/` | Rohdaten jedes Messlaufs (`manifest.json`, `faelle/*.json`) — der Beleg; Berichte entstehen daraus neu |
| `fragen/` | Eingaben der Fallbasis: `fragen.csv` (kuratierte Fragen mit Goldquelle), `fragen_nutzer.csv` (echte Nutzerfragen — privat) |
| `testindex/` | Testkopie des Index mit neuen Fragen (HPC-Aufträge unter `testindex/hpc/`) |
| `po/` | PO-Testmenge — liegt derzeit nur im Agenten-Repository |

## Messen

```bash
# Server für Messungen auf eigenem Port
HSH_MCP_PORT=8775 python -m raghsh.server &

# nur Retrieval (ohne LLM, Minuten): Gold in den Kandidaten, Ränge, Arme
python auswertung/gold/lauf.py --nur-retrieval --mcp-url http://127.0.0.1:8775/mcp

# Volllauf mit Modell, danach Auswertung und Bericht
python auswertung/gold/lauf.py --modell glm-4.7 --mcp-url http://127.0.0.1:8775/mcp --auswerten --bericht

# Konfiguration ohne Codeänderung variieren
python auswertung/gold/lauf.py --nur-retrieval --fusion rrf --top-docs 8 ...
```

Ein Lauf misst gegen den Vertrag der Fallbasis: Ändert sich `fallbasis.json`,
ändert sich der Fingerabdruck, und alte und neue Läufe sind nicht mehr
vergleichbar. Tests: `pytest` (u. a. `auswertung/gold/test_vergleich.py`).

## Holdout

Für die Entwicklungsagenten gibt es zurückgehaltene Fälle außerhalb des
Repositorys; ihre Bewertung gibt nur Summen aus. Holdout-Fragen gehören in
keine Datei, die die Agenten lesen können — auch nicht in dieses Repository.
