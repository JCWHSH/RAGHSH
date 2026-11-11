"""Erkennung von Studiengang, Fakultät und PO-Fassung aus der Frage (raghsh.po.pfad).

Die mit xfail markierten Fälle sind bekannte Fehler aus der PO-Analyse vom
02.11.2026 (docs/po_logik.md, „Bekannte Grenzen“). strict=True: Wird einer
behoben, schlägt der Test fehl und die Markierung muss weg.
"""
import pytest

from raghsh import konfig

# Die PO-Erkennung liest var/doc_index.db (Laufzeitdaten, nicht im Repository).
if not konfig.DOC_INDEX.exists():
    pytest.skip(f"{konfig.DOC_INDEX} fehlt (erst crawlen und importieren)",
                allow_module_level=True)

from raghsh.server import mcp_server  # noqa: F401  verbindet pfad mit dem Server
from raghsh.po import pfad


def erkenne(frage):
    familie = pfad._studiengang_familie_aus_frage(frage)
    return familie, pfad._po_aus_frage(frage, familie)


@pytest.mark.parametrize("frage, studiengang, fassung", [
    ("Ich studiere Angewandte Mathematik nach der PO 2018. Wie viele Credits hat die Bachelorarbeit?", "MAT", 2018),
    ("Veranstaltungsmanagement, PO 2022: Aus wie vielen Vertiefungsrichtungen kann ich wählen?", "BVM", 2022),
    ("Religionspädagogik und Soziale Arbeit, BT-PO von 2018: Was muss für die Zulassung vorliegen?", "BRS", 2018),
    ("Ich fange jetzt mit dem Bachelor Informatik an, wie lange dauert die Praxisphase?", "BIN", None),
])
def test_genannte_fassung_wird_erkannt(frage, studiengang, fassung):
    familie, po = erkenne(frage)
    assert familie == [studiengang]
    assert po is not None and po["studiengang"] == studiengang
    if fassung is not None:
        assert po["fassung"] == fassung


@pytest.mark.parametrize("frage", [
    "Wo finde ich einen Platz für die Praxisphase im Bachelor Medizinisches Informationsmanagement?",
    "Wie viele Wochen dauert die Praxisphase im Bachelor Medizinisches Informationsmanagement?",
])
def test_ohne_po_angabe_keine_fassung(frage):
    familie, po = erkenne(frage)
    assert familie == ["BMI"]
    assert po is None


def test_fakultaet_aus_frage():
    assert pfad._fakultaet_aus_frage("Wer berät in der Fakultät IV zur Praxisphase?") == "f4"
    assert pfad._fakultaet_aus_frage("Gibt es eine Mensa?") == ""


# --- bekannte Fehler -------------------------------------------------------

@pytest.mark.xfail(strict=True, reason="Neubeginn-Muster: „fange … an“ ohne Studienbeginn legt die jüngste Fassung fest")
@pytest.mark.parametrize("frage", [
    "Wann fange ich im Bachelor Journalistik mit der Bachelorarbeit an?",
    "Ab wann fangen im Bachelor Angewandte Mathematik die Vorlesungen an?",
])
def test_bekannt_neubeginn_zu_weit(frage):
    _, po = erkenne(frage)
    assert po is None


@pytest.mark.xfail(strict=True, reason="„BPO 2017“ wird nicht als PO-Angabe erkannt")
def test_bekannt_bpo():
    _, po = erkenne("Bachelor Journalistik, BPO 2017: Umfang der Bachelorarbeit?")
    assert po is not None


@pytest.mark.xfail(strict=True, reason="Studiengang ohne Bachelor/Master meist nicht erkannt")
def test_bekannt_kernwort_ohne_abschluss():
    familie, _ = erkenne("Ich studiere Elektrotechnik. Wie viele Credits hat die Bachelorarbeit?")
    assert familie
