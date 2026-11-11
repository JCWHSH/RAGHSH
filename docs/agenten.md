# Entwicklungsagenten

Stand: 11.11.2026. Seit 28.09.2026 entwickeln Claude-Agenten RAGHSH in
Sitzungen von höchstens zwei Stunden weiter (Planer → Umsetzer → Prüfer).

- **Eigener Unix-Nutzer** `raghsh-agent` mit eigener Arbeitskopie (Branch
  `agent`, kein Remote), eigener Qdrant-Instanz und eigenem MCP-Port. Eine
  Firewall-Regel sperrt ihn für die Produktionsports.
- **Ziel, Regeln, Messmengen** gehören dem Projektverantwortlichen und sind
  für die Agenten schreibgeschützt. Vorschläge der Agenten zu Ziel und Regeln
  stehen in einer eigenen Datei und werden dort freigegeben.
- **Bewertung:** eigene PO-Testmenge, Gold-Fallbasis und ein Holdout
  außerhalb ihrer Reichweite (nur Summen, begrenzte Zahl Läufe je Tag).
- **Protokoll:** Jede Rolle schreibt ein vollständiges Protokoll, das nach
  Abschluss unveränderlich gemacht wird.
- **Übernahme:** Ergebnisse kommen nur als geprüfter Commit in `main`
  („Übernahme Agentenstand …“), zuletzt 8c0d1d0 (11.10.2026). Mit jeder
  Übernahme ist die betroffene Doku in `docs/` nachzuführen.

Die Arbeitskopie der Agenten folgt noch der alten Verzeichnisstruktur
(`hsh_scraper/`); sie wird bei der nächsten Sitzung auf die neue Struktur
gebracht.
