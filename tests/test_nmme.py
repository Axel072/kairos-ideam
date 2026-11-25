from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from ideam_seasonal import nmme
from ideam_seasonal.common import Region

CFG = {"nmme": {"url": "https://iridl.ldeo.columbia.edu/SOURCES/.Models/.NMME",
                "variables": {"precipitation": "prec", "temperature": "tref"},
                "models": {"A": {"hindcast": ".A/.HINDCAST/.MONTHLY", "forecast": ".A/.FORECAST/.MONTHLY"},
                           "B": {"hindcast": ".B/.HINDCAST/.MONTHLY", "forecast": ".B/.FORECAST/.MONTHLY"}}}}
REGION = Region("colombia", {"north": 14, "west": -82, "south": -5, "east": -66})


def _nc(path: Path, years, units="mm/day", value=None, lon=(277.0, 278.0, 279.0)):
    lat = [0.0, 1.0, 2.0]
    s = [(y - 1960) * 12 + 9 for y in years]
    data = np.ones((len(years), 1, len(lat), len(lon))) if value is None else value
    xr.Dataset({"prec": (("S", "L", "Y", "X"), data, {"units": units})},
               coords={"S": s, "L": [2.5], "Y": lat, "X": list(lon)}).to_netcdf(path)
    return path


def test_url_selects_start_leads_and_box():
    u = nmme.url(CFG, "A", "precipitation", 10, (1993, 2016), [2, 3, 4], REGION, False)
    assert u.startswith("https://iridl.ldeo.columbia.edu/SOURCES/.Models/.NMME/.A/.HINDCAST/.MONTHLY/.prec/")
    assert "S/(0000%201%20Oct%201993-2016)/VALUES" in u
    assert "L/1.5/3.5/RANGEEDGES" in u and "%5BM%5D/average" in u
    assert "Y/-6/15/RANGEEDGES/X/-83/-65/RANGEEDGES" in u and u.endswith("data.nc")
    f = nmme.url(CFG, "A", "temperature", 11, (2026, 2026), [2, 3, 4], Region("global", None), True)
    assert ".A/.FORECAST/.MONTHLY/.tref/S/(0000%201%20Nov%202026)/VALUES" in f and "RANGEEDGES/X" not in f


def test_open_season_units_years_and_longitudes(tmp_path):
    da = nmme.open_season(_nc(tmp_path / "t.nc", [1993, 1994], "Kelvin_scale",
                              np.full((2, 1, 3, 3), 300.0)), "temperature")
    assert da.dims == ("year", "lat", "lon")
    assert da["year"].values.tolist() == [1993, 1994]
    assert da["lon"].values.tolist() == [-83.0, -82.0, -81.0]
    assert np.allclose(da.values, 26.85) and da.attrs["units"] == "Celsius"


def test_open_season_rejects_unknown_units(tmp_path):
    with pytest.raises(ValueError, match="unidad"):
        nmme.open_season(_nc(tmp_path / "t.nc", [1993], "kg/m2"), "precipitation")


def test_to_grid_midpoint_is_mean_of_neighbours():
    v = np.arange(9.0).reshape(1, 3, 3)
    da = xr.DataArray(v, dims=("year", "lat", "lon"),
                      coords={"year": [1993], "lat": [0.0, 1.0, 2.0], "lon": [-83.0, -82.0, -81.0]})
    out = nmme.to_grid(da, np.array([0.5]), np.array([-82.5]))
    assert np.isclose(out.values[0, 0, 0], v[0, :2, :2].mean())


def test_to_grid_wraps_global_longitudes():
    lon = np.arange(-180.0, 180.0)
    da = xr.DataArray(np.broadcast_to(lon, (1, 2, lon.size)).copy(), dims=("year", "lat", "lon"),
                      coords={"year": [1993], "lat": [0.0, 1.0], "lon": lon})
    out = nmme.to_grid(da, np.array([0.5]), np.array([179.5]))
    assert np.isclose(out.values[0, 0, 0], (179.0 + -180.0) / 2)


def test_to_grid_rejects_uncovered_region():
    da = xr.DataArray(np.zeros((1, 2, 2)), dims=("year", "lat", "lon"),
                      coords={"year": [1993], "lat": [0.0, 1.0], "lon": [-83.0, -82.0]})
    with pytest.raises(ValueError, match="no cubre"):
        nmme.to_grid(da, np.array([0.5]), np.array([-70.0]))


def test_build_averages_complete_models_and_skips_the_rest(tmp_path, monkeypatch):
    hind = {"A": _nc(tmp_path / "ah.nc", [1993, 1994], value=np.full((2, 1, 3, 3), 2.0)),
            "B": _nc(tmp_path / "bh.nc", [1993])}
    fore = {"A": _nc(tmp_path / "af.nc", [2026], value=np.full((1, 1, 3, 3), 4.0)),
            "B": _nc(tmp_path / "bf.nc", [2026])}

    def fake(cfg, model, variable, month, years, leads, region, forecast, force=False):
        if forecast:
            if model not in fore:
                raise nmme.Unavailable("no publicado en la IRI Data Library")
            return fore[model]
        return hind[model]

    monkeypatch.setattr(nmme, "download", fake)
    x, z, status = nmme.build(CFG, "precipitation", 10, 2026, (1993, 1994), [2, 3, 4], REGION,
                              np.array([0.5, 1.5]), np.array([-82.5, -81.5]))
    assert status["A"] == "entra" and status["B"].startswith("no entra: hindcast incompleto")
    assert x.shape == (2, 2, 2) and np.allclose(x, 2.0) and np.allclose(z, 4.0)


def test_build_fails_without_models(tmp_path, monkeypatch):
    def fake(*a, **k):
        raise nmme.Unavailable("no publicado en la IRI Data Library")

    monkeypatch.setattr(nmme, "download", fake)
    with pytest.raises(RuntimeError, match="ningún modelo NMME"):
        nmme.build(CFG, "precipitation", 10, 2026, (1993, 1994), [2, 3, 4], REGION,
                   np.array([0.5]), np.array([-82.5]))


def test_open_season_drops_height_dimension(tmp_path):
    path = tmp_path / "z.nc"
    xr.Dataset({"tref": (("S", "L", "Z", "Y", "X"), np.full((1, 1, 1, 2, 2), 300.0),
                         {"units": "Kelvin_scale"})},
               coords={"S": [801], "L": [2.5], "Z": [2.0], "Y": [0.0, 1.0],
                       "X": [277.0, 278.0]}).to_netcdf(path)
    da = nmme.open_season(path, "temperature")
    assert da.dims == ("year", "lat", "lon") and da.shape == (1, 2, 2)
