from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from ideam_seasonal import common, homogenize, ingest
from test_homogenize import _lagged_ds

CFG = {"area": {"north": 6, "west": -76, "south": 0, "east": -70},
       "cpt": {"regions": {"colombia": {"north": 14, "west": -82, "south": -5, "east": -66}}}}


def _da(units):
    return xr.DataArray(np.array([1.0, 2.0]), dims="x", attrs={"units": units})


def test_load_config_independent_of_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = common.load_config()
    assert "centres" in cfg and common.CONFIG_PATH.is_absolute()
    assert ingest.load_config is common.load_config


def test_variable_of_by_units():
    assert common.variable_of(_da("kg m**-2 s**-1")).key == "precipitation"
    assert common.variable_of(_da("K")).key == "temperature"
    with pytest.raises(ValueError, match="no reconocida"):
        common.variable_of(_da("furlongs"))
    with pytest.raises(ValueError, match="sin atributo"):
        common.variable_of(xr.DataArray([1.0]))


def test_conversion_to_standard():
    np.testing.assert_allclose(common.to_standard(_da("m s**-1"), np.array([1e-8])), [0.864])
    np.testing.assert_allclose(common.to_standard(_da("K"), np.array([273.15])), [0.0], atol=1e-9)
    with pytest.raises(ValueError, match="acumulada"):
        common.conversion(_da("m"))


def test_resolve_named_regions():
    assert common.resolve_region(CFG, "global") == common.Region("global", None)
    assert common.resolve_region(CFG, "prueba") == common.Region("box", CFG["area"])
    assert common.resolve_region(CFG).name == "box"  # por defecto, la caja de prueba
    assert common.resolve_region(CFG, "colombia").area["north"] == 14
    assert common.resolve_region(CFG, None, default="colombia").name == "colombia"


def test_resolve_custom_box():
    r = common.resolve_region(CFG, "5, -75, -2.5, -70")
    assert r.area == {"north": 5.0, "west": -75.0, "south": -2.5, "east": -70.0}
    assert r.name == "N5_W-75_S-2.5_E-70"
    assert r.bounds() == "N 5.0, W -75.0, S -2.5, E -70.0"


@pytest.mark.parametrize("spec", ["5,-75,-2", "-2,-75,5,-70", "5,285,-2,290", "atlantida"])
def test_resolve_rejects_bad_regions(spec):
    with pytest.raises(ValueError):
        common.resolve_region(CFG, spec)


def test_raw_and_chirps_paths():
    box = common.resolve_region(CFG, "prueba")
    col = common.resolve_region(CFG, "colombia")
    d = Path("data") / "raw"
    assert common.raw_path(d, "ncep", "precipitation", 5, box).name == "ncep_precipitation_m05_box.grib"
    assert (common.raw_path(d, "bom", "temperature", 11, col, forecast_year=2026).name
            == "bom_temperature_f202611_colombia.grib")
    assert common.chirps_path(d, 1993, box).name == "chirps_v3_1993_box.nc"
    assert common.chirps_path(d, "*", box).name == "chirps_v3_*_box.nc"
    # un cuadro propio no reutiliza la descarga de otro
    assert (common.raw_path(d, "ncep", "precipitation", 5, common.resolve_region(CFG, "5,-75,-2,-70"))
            != common.raw_path(d, "ncep", "precipitation", 5, common.resolve_region(CFG, "6,-75,-2,-70")))


def test_request_area_follows_region():
    cfg = ingest.load_config()
    assert ingest.build_request(cfg, "ncep")["area"] == [6, -76, 0, -70]
    glob = ingest.build_request(cfg, "ncep", region=common.resolve_region(cfg, "global"))
    assert "area" not in glob
    box = ingest.build_request(cfg, "ncep", region=common.resolve_region(cfg, "5,-75,-2,-70"))
    assert box["area"] == [5.0, -75.0, -2.0, -70.0]


def test_monthly_mean_is_simple_mean_over_members():
    out = homogenize.reduce_lagged(_lagged_ds(), 5, 6)
    m = homogenize.monthly_mean(out)
    assert m.dims == ("year", "lead", "lat", "lon")
    # miembro 0 vale 1..6 y miembro 1 vale 101..106 por lead -> promedio 51..56
    np.testing.assert_allclose(m.isel(year=0, lat=0, lon=0).values, [51, 52, 53, 54, 55, 56])
    assert m.attrs["units"] == "mm/day"
