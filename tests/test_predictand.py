import numpy as np
import pandas as pd
import pytest

from ideam_seasonal import common, verify

CFG = {"area": {"north": 6, "west": -76, "south": 0, "east": -70},
       "cpt": {"regions": {"colombia": {"north": 14, "west": -82, "south": -5, "east": -66}}}}
COL = common.resolve_region(CFG, "colombia")
YEARS = [1993, 1994]


def _write(tmp_path, values=None, lon=(-75.0, -70.0), lat=(4.0, 1.0), start="1993-01",
           n=27, units="mm", missing="-999", header_tags=None, name="Y.txt"):
    """Archivo CPT de estaciones como el del IDEAM (2 estaciones, mensual)."""
    dates = pd.date_range(start, periods=n, freq="MS")
    if values is None:
        values = np.tile([[100.0, 200.0]], (n, 1))
    tags = header_tags or (f"cpt:field=prcp, cpt:nrow={n}, cpt:ncol=2, cpt:col=station, "
                           f"cpt:row=T, cpt:units={units}, cpt:missing={missing}")
    lines = ["xmlns:cpt=http://iri.columbia.edu/CPT/v10/\t\t", "cpt:nfield=1\t\t", tags + "\t",
             "\tA1\tB2", "cpt:X\t" + "\t".join(map(str, lon)), "cpt:Y\t" + "\t".join(map(str, lat))]
    lines += [f"{d:%Y-%m}\t" + "\t".join(f"{v:g}" for v in row) for d, row in zip(dates, values)]
    p = tmp_path / name
    p.write_bytes(("\r\n".join(lines) + "\r\n").encode("utf-8"))  # CRLF como el del IDEAM
    return p


def _check(path, region=COL, start_month=11, leads=(2, 3, 4)):
    da = common.open_cpt_stations(path)
    return verify.check_predictand(da, "precipitation", YEARS, start_month, list(leads), region)


def test_reader(tmp_path):
    da = common.open_cpt_stations(_write(tmp_path))
    assert da.dims == ("time", "station") and list(da["station"].values) == ["A1", "B2"]
    assert da.attrs["units"] == "mm" and list(da["lon"].values) == [-75.0, -70.0]
    assert pd.Timestamp(da["time"].values[0]) == pd.Timestamp(1993, 1, 1)


def test_reader_missing_to_nan(tmp_path):
    vals = np.tile([[100.0, 200.0]], (27, 1))
    vals[3, 1] = -999
    da = common.open_cpt_stations(_write(tmp_path, values=vals))
    assert int(da.isnull().sum()) == 1


@pytest.mark.parametrize("kwargs, match", [
    ({"header_tags": "cpt:field=prcp, cpt:nrow=27, cpt:ncol=2, cpt:col=X, cpt:row=Y"}, "estaciones"),
    ({"header_tags": "cpt:field=prcp, cpt:nrow=30, cpt:ncol=2, cpt:col=station, cpt:row=T"},
     "tamaño"),
])
def test_reader_rejects_bad_format(tmp_path, kwargs, match):
    with pytest.raises(ValueError, match=match):
        common.open_cpt_stations(_write(tmp_path, **kwargs))


def test_good_file_passes(tmp_path):
    res = _check(_write(tmp_path))
    assert all(ok for ok, _ in res.values()), res
    # DJF 1993/94 y 1994/95 -> necesita hasta febrero de 1995
    assert "6 meses" in res["periodo"][1]


def test_period_must_cover_season_crossing_year(tmp_path):
    res = _check(_write(tmp_path, n=25))  # termina en enero de 1995: falta febrero
    assert not res["periodo"][0] and "1995-02" in res["periodo"][1]


def test_region_without_stations_fails(tmp_path):
    lejos = common.resolve_region(CFG, "40,-10,30,0")
    assert not _check(_write(tmp_path), region=lejos)["región"][0]


def test_region_partially_outside_passes_with_note(tmp_path):
    res = _check(_write(tmp_path), region=common.resolve_region(CFG, "prueba"))
    assert res["región"][0]  # 1 de 2 dentro
    res = _check(_write(tmp_path, lon=(-75.0, -60.0)), region=common.resolve_region(CFG, "prueba"))
    assert res["región"][0] and "excluye" in res["región"][1]


def test_longitudes_0_360_fail(tmp_path):
    assert not _check(_write(tmp_path, lon=(285.0, 290.0)), region=common.Region("global", None))["longitudes"][0]


def test_units_checked(tmp_path):
    assert not _check(_write(tmp_path, units="inches"))["unidades"][0]
    huge = np.tile([[1e5, 1e5]], (27, 1))  # 100 000 mm/mes: error de escala
    assert not _check(_write(tmp_path, values=huge))["unidades"][0]


def test_accumulated_from_january_detected(tmp_path):
    n = 36
    month = pd.date_range("1993-01", periods=n, freq="MS").month.to_numpy()
    vals = np.column_stack([month * 100.0, month * 150.0])  # crece todo el año
    assert not _check(_write(tmp_path, values=vals, n=n))["acumulación"][0]


def test_too_many_missing_fails(tmp_path):
    vals = np.tile([[100.0, 200.0]], (27, 1))
    vals[11:14, 0] = -999  # DJF 1993/94 completo de la estación A1: 3 de 6 meses -> 50 %
    res = _check(_write(tmp_path, values=vals))
    assert not res["faltantes"][0]


def test_ideam_file_if_present():
    cfg = common.load_config()
    path = common.predictand_path(cfg, "precipitation")
    if path is None or not path.exists():
        pytest.skip("Y del IDEAM no está en esta copia del repositorio")
    region = common.resolve_region(cfg, "colombia")
    assert verify.verify_predictand(path, "precipitation", range(1993, 2017), 11, [2, 3, 4], region)
