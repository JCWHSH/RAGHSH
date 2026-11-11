"""Orte aus raghsh.konfig: Standard unter dem Repository, per Umgebung umlenkbar."""
import importlib
import os

import pytest

import raghsh.konfig as konfig


@pytest.mark.skipif(bool(os.getenv("RAGHSH_VAR")), reason="RAGHSH_VAR ist gesetzt")
def test_standardorte_liegen_im_repository():
    assert (konfig.WURZEL / "pyproject.toml").exists()
    assert konfig.VAR == konfig.WURZEL / "var"
    assert konfig.KORPUS.parent == konfig.VAR
    assert konfig.PO_DATEN == konfig.WURZEL / "stammdaten" / "po"
    assert (konfig.PO_DATEN / "studiengaenge.tsv").exists()
    assert konfig.LAEUFE == konfig.WURZEL / "auswertung" / "laeufe"


def test_var_umlenkbar(monkeypatch, tmp_path):
    monkeypatch.setenv("RAGHSH_VAR", str(tmp_path))
    k = importlib.reload(konfig)
    try:
        assert k.VAR == tmp_path
        assert k.DOC_INDEX == tmp_path / "doc_index.db"
        assert k.QA_LOG == tmp_path / "qa_log.jsonl"
    finally:
        monkeypatch.delenv("RAGHSH_VAR")
        importlib.reload(konfig)
