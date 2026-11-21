import sys
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from ideam_seasonal import common, cpt, cpt_run

CFG = {"area": {"north": 6, "west": -76, "south": 0, "east": -70},
       "cpt": {"regions": {"colombia": {"north": 14, "west": -82, "south": -5, "east": -66}}}}


def _field(values, years=(1993, 1994)):
    return xr.DataArray(np.asarray(values, dtype=float), dims=("year", "lat", "lon"),
                        coords={"year": list(years), "lat": [0.5, 1.5], "lon": [-71.5, -70.5]},
                        attrs={"units": "mm/day"})


STATIONS_Y = """xmlns:cpt=http://iri.columbia.edu/CPT/v10/
cpt:field=prcp, cpt:nrow=27, cpt:ncol=2, cpt:col=station, cpt:row=T, cpt:units=mm, cpt:missing=-999
\tA1\tB2
cpt:X\t-75\t-70
cpt:Y\t4\t1
""" + "".join(f"{1993 + k // 12}-{k % 12 + 1:02d}\t{10 * (k + 1)}\t{k + 1}\n" for k in range(27))


def test_batch_input_sequence():
    text = cpt_run.batch_input(CFG["area"], (1, 5))
    lines = text.splitlines()
    assert lines[:9] == ["612", "1", "X.txt", "6", "0", "-76", "-70", "1", "5"]
    assert lines[9:16] == ["2", "Y.txt", "6", "0", "-76", "-70", "3"]
    assert "311" in lines and "451" in lines and lines[-2:] == ["0", "0"]
    for code in cpt_run.OUTPUTS:
        assert lines[lines.index(code) + 1] == cpt_run.OUTPUTS[code]


def test_batch_input_global_domain():
    lines = cpt_run.batch_input(None, (1, 3)).splitlines()
    assert lines[3:7] == ["90", "-90", "-180", "180"]


def test_clean_output_removes_backspaced_text_and_colors():
    raw = b"\x1b[1mTitulo\x1b[0m\r\n C\x08P\x08T\x08 \x08 Northernmost latitude :"
    assert cpt_run.clean_output(raw) == "Titulo\n  Northernmost latitude :"


def test_compare_grids_detects_changes(tmp_path):
    a = cpt.write_cpt(_field(np.arange(8).reshape(2, 2, 2)), tmp_path / "a.txt", "pr", 11, [2, 3, 4])
    b = cpt.write_cpt(_field(np.arange(8).reshape(2, 2, 2)), tmp_path / "b.txt", "pr", 11, [2, 3, 4])
    assert cpt_run.compare_grids(a, b)[0]
    c = cpt.write_cpt(_field(np.arange(8).reshape(2, 2, 2) + 0.5), tmp_path / "c.txt", "pr", 11,
                      [2, 3, 4])
    ok, msg = cpt_run.compare_grids(a, c)
    assert not ok and "5.0e-01" in msg


def test_read_cpt_grid_inherits_tags(tmp_path):
    """Como escribe CPT: el segundo bloque solo trae cpt:T."""
    p = tmp_path / "g.txt"
    p.write_text("xmlns:cpt=http://iri.columbia.edu/CPT/v10/\n"
                 "cpt:field=pr, cpt:T=1993-12/1994-02, cpt:nrow=2, cpt:ncol=1, cpt:missing=-999\n"
                 "\t-70.5\n0.5\t1\n1.5\t2\n"
                 "cpt:T=1994-12/1995-02\n\t-70.5\n1.5\t4\n0.5\t-999\n", encoding="utf-8")
    g = common.read_cpt_grid(p)
    assert list(g) == ["1993-12/1994-02", "1994-12/1995-02"]
    assert list(g["1993-12/1994-02"].lat.values) == [1.5, 0.5]  # norte a sur
    assert np.isnan(g["1994-12/1995-02"].sel(lat=0.5).item())


def test_compare_predictand_is_seasonal_sum(tmp_path):
    y = tmp_path / "Y.txt"
    y.write_text(STATIONS_Y, encoding="utf-8")
    # DJF 1993/94: A1 = 120+130+140, B2 = 12+13+14
    good = tmp_path / "Ycpt.txt"
    good.write_text("\tA1\tB2\n1993-12/1994-02\t390\t39\n", encoding="utf-8")
    assert cpt_run.compare_predictand(y, good, [1993])[0]
    mean = tmp_path / "Ymean.txt"
    mean.write_text("\tA1\tB2\n1993-12/1994-02\t130\t13\n", encoding="utf-8")
    assert not cpt_run.compare_predictand(y, mean, [1993])[0]
    assert not cpt_run.compare_predictand(y, good, [1994])[0]  # años distintos al hindcast


def test_forecast_table(tmp_path):
    (tmp_path / "pronostico.txt").write_text("\tA1\tB2\n2026-12/2027-02\t96.3\t0\n", encoding="utf-8")
    (tmp_path / "probabilidades.txt").write_text(
        "cpt:C=1\n\tA1\tB2\n2026-12/2027-02\t68\t50\ncpt:C=2\n\tA1\tB2\n2026-12/2027-02\t14\t12\n"
        "cpt:C=3\n\tA1\tB2\n2026-12/2027-02\t18\t38\n", encoding="utf-8")
    df = cpt_run.forecast_table(tmp_path)
    assert list(df["estacion"]) == ["A1", "B2"] and df.attrs["temporada"] == "2026-12/2027-02"
    np.testing.assert_allclose(df[["p_bajo", "p_normal", "p_sobre"]].sum(axis=1), 100)


def _fake_cpt(tmp_path, code: str) -> list[str]:
    script = tmp_path / "fake_cpt.py"
    script.write_text(code, encoding="utf-8")
    return [sys.executable, str(script)]


def test_endless_prompt_is_stopped(tmp_path):
    argv = _fake_cpt(tmp_path, "import sys\nwhile True:\n    sys.stdout.write(' Number of modes :  ')\n")
    with pytest.raises(cpt_run.CPTError, match="entrada no prevista.*Number of modes"):
        cpt_run.run_cpt_batch(Path("x"), Path("."), tmp_path, "612\n", 30, argv=argv)
    assert (tmp_path / "cpt_salida.log").exists()


def test_cpt_error_message_is_reported(tmp_path):
    argv = _fake_cpt(tmp_path, "print(' ERROR:  Unable to locate cpt.ini.')\n")
    with pytest.raises(cpt_run.CPTError, match="Unable to locate cpt.ini"):
        cpt_run.run_cpt_batch(Path("x"), Path("."), tmp_path, "0\n", 30, argv=argv)


def test_missing_exe(tmp_path):
    with pytest.raises(cpt_run.CPTError, match="no se encontró CPT"):
        cpt_run.run_cpt_batch(tmp_path / "no_existe.exe", tmp_path, tmp_path, "0\n", 5)


def test_real_cpt_run_if_installed(tmp_path):
    """Corrida real si CPT y los archivos existen en esta máquina."""
    cfg = common.load_config()
    exe = Path(cfg["cpt_batch"]["exe"])
    x = Path("data") / "cpt" / "X_ncep_precipitation_i202611_colombia.txt"
    z = Path("data") / "cpt" / "Z_ncep_precipitation_i202611_colombia.txt"
    y = common.predictand_path(cfg, "precipitation")
    if not (exe.exists() and x.exists() and z.exists() and y and y.exists()):
        pytest.skip("CPT o los archivos de ejemplo no están en esta máquina")
    region = common.resolve_region(cfg, "colombia")
    r = cpt_run.run_one("ncep", x, y, z, tmp_path / "run", cfg["cpt_batch"], region,
                        list(range(1993, 2017)))
    assert r.ok, r.checks
    assert r.goodness is not None and len(r.forecast) == 10
