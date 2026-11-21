import argparse
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from ideam_seasonal import cpt, kairos


# --- lectura de valores ----------------------------------------------------------------------

def test_parse_models():
    assert kairos.parse_models(None) == ["ncep", "dwd", "bom"]
    assert kairos.parse_models(["dwd", "NCEP"]) == ["ncep", "dwd"]  # orden fijo, sin mayúsculas
    assert kairos.parse_models(["bom,ncep", "ncep"]) == ["ncep", "bom"]
    with pytest.raises(argparse.ArgumentTypeError):
        kairos.parse_models(["ecmwf"])


@pytest.mark.parametrize("text, expected", [("2026-10", (2026, 10)), ("2025-1", (2025, 1))])
def test_parse_start(text, expected):
    assert kairos.parse_start(text) == expected


@pytest.mark.parametrize("text", ["2026", "2026-13", "octubre", "26-10"])
def test_parse_start_rejects(text):
    with pytest.raises(argparse.ArgumentTypeError):
        kairos.parse_start(text)


def test_parse_leads():
    assert kairos.parse_leads("2-4") == [2, 3, 4]
    assert kairos.parse_leads("3") == [3]
    for bad in ("4-2", "0-2", "2-7", "a-b"):
        with pytest.raises(argparse.ArgumentTypeError):
            kairos.parse_leads(bad)


def test_parse_years():
    assert kairos.parse_years("1995-2010") == (1995, 2010)
    for bad in ("2010-1995", "1995", "x-y"):
        with pytest.raises(argparse.ArgumentTypeError):
            kairos.parse_years(bad)


# --- opciones y valores por defecto ----------------------------------------------------------

@pytest.fixture
def captured(monkeypatch):
    """Reemplaza cpt.build: registra con qué se llamó, sin descargar nada."""
    calls = []

    def fake_build(models, variable, **kw):
        calls.append({"models": models, "variable": variable, **kw})
        return [Path(f"X_{variable}.txt")]

    monkeypatch.setattr(cpt, "build", fake_build)
    return calls


def test_no_options_uses_config_defaults(captured):
    assert kairos.main([]) == 0
    assert [c["variable"] for c in captured] == ["precipitation", "temperature"]
    c = captured[0]
    assert c["models"] == ["ncep", "dwd", "bom"]
    assert c["region"] is None and c["start"] is None and c["leads"] is None
    assert c["years"] is None and c["force"] is False and c["out_dir"] == Path("data") / "cpt"
    assert c["command"] == "bash kairos.sh"


def test_options_combined(captured):
    argv = ["-r", "5,-75,-2,-70", "-m", "ncep,dwd", "-v", "precipitation", "--inicio", "2026-10",
            "-t", "3-5", "-a", "1995-2015", "-o", "salida", "-f"]
    assert kairos.main(argv) == 0
    (c,) = captured
    assert c["models"] == ["ncep", "dwd"] and c["region"] == "5,-75,-2,-70"
    assert c["start"] == (2026, 10) and c["leads"] == [3, 4, 5] and c["years"] == (1995, 2015)
    assert c["out_dir"] == Path("salida") and c["force"] is True
    assert c["command"] == "bash kairos.sh " + " ".join(argv)


@pytest.mark.parametrize("argv", [["--region", "5,-75"], ["--region", "atlantida"],
                                  ["-m", "ecmwf"], ["--inicio", "2026-13"], ["-t", "0-2"],
                                  ["-v", "viento"]])
def test_bad_options_stop_before_running(captured, argv):
    with pytest.raises(SystemExit) as e:
        kairos.main(argv)
    assert e.value.code == 2 and captured == []


def test_run_errors_give_message_and_exit_1(monkeypatch, capsys):
    def boom(*a, **k):
        raise RuntimeError("el pronóstico con inicio 2030-01 no está publicado")

    monkeypatch.setattr(cpt, "build", boom)
    assert kairos.main(["--inicio", "2030-01"]) == 1
    assert "ERROR: el pronóstico con inicio 2030-01" in capsys.readouterr().err


def test_qc_subcommand(monkeypatch):
    seen = []
    monkeypatch.setattr(kairos, "run_qc", lambda args: seen.append(args.force))
    assert kairos.main(["qc", "--force"]) == 0 and seen == [True]


# --- validaciones de cpt.build ---------------------------------------------------------------

CFG = {"centres": {"ncep": {"years": [1993, 2016]}, "dwd": {"years": [1993, 2016]},
                   "bom": {"years": [1995, 2016]}}}


def test_hindcast_years():
    assert cpt.hindcast_years(CFG, ["ncep", "dwd"], None) == (1993, 2016)
    assert cpt.hindcast_years(CFG, ["ncep", "bom"], None) == (1995, 2016)  # periodo común
    assert cpt.hindcast_years(CFG, ["ncep"], (2000, 2010)) == (2000, 2010)
    with pytest.raises(ValueError, match="fuera del periodo"):
        cpt.hindcast_years(CFG, ["ncep", "bom"], (1993, 2016))


def test_write_log_records_command(tmp_path):
    cfg = {"centres": {"ncep": {"system": "2"}}, "cpt": {"regions": {}},
           "area": {"north": 6, "west": -76, "south": 0, "east": -70}}
    grid = xr.DataArray(np.zeros((2, 1, 1)), dims=("year", "lat", "lon"),
                        coords={"year": [1995, 1996], "lat": [0.5], "lon": [-70.5]})
    files = {"ncep": {"X": tmp_path / "X.txt", "Z": tmp_path / "Z.txt"}}
    log = cpt.write_log(tmp_path / "r.md", cfg, "precipitation", 2026, 10, [3, 4, 5],
                        cpt.resolve_region(cfg, "prueba"), ["ncep"], {"ncep": (24, 124)}, files,
                        grid, command="bash kairos.sh -m ncep", start_mode="pedido con --inicio")
    text = log.read_text(encoding="utf-8")
    assert "| Comando | `bash kairos.sh -m ncep` |" in text
    assert "| Elección del inicio | pedido con --inicio |" in text
    assert "1995–1996" in text and "leads 3, 4, 5" in text
