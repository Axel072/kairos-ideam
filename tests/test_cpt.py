import numpy as np
import pytest
import xarray as xr

from ideam_seasonal import cpt, homogenize
from test_homogenize import _lagged_ds


def _field(values, years=(1993, 1994)):
    return xr.DataArray(np.asarray(values, dtype=float), dims=("year", "lat", "lon"),
                        coords={"year": list(years), "lat": [0.5, 1.5], "lon": [-71.5, -70.5]},
                        attrs={"units": "mm/day"})


def test_season_label_crosses_year():
    assert cpt.season_label(2026, 10, [2, 3, 4]) == "2026-11/2027-01"
    assert cpt.season_label(2025, 5, [2, 3, 4]) == "2025-06/2025-08"


def test_write_cpt_blocks_and_north_first(tmp_path):
    da = _field(np.arange(8).reshape(2, 2, 2))
    path = cpt.write_cpt(da, tmp_path / "X.txt", "precipitation_ncep", 10, [2, 3, 4])
    lines = path.read_text(encoding="utf-8").splitlines()
    assert lines[0] == cpt.CPT_HEADER and lines[1] == "cpt:nfields=1"
    tags = [l for l in lines if l.startswith("cpt:field=")]
    assert len(tags) == 2
    assert "cpt:T=1993-11/1994-01" in tags[0] and "cpt:S=1993-10-01T00:00" in tags[0]
    assert "cpt:L=1.5 months" in tags[0]
    assert lines[3] == "\t-71.5\t-70.5"
    assert lines[4].split("\t") == ["1.5", "2", "3"]


def test_ensemble_is_equal_weight_mean():
    out = cpt.ensemble({"a": _field(np.zeros((2, 2, 2))), "b": _field(np.full((2, 2, 2), 3.0))})
    np.testing.assert_allclose(out.values, 1.5)


def test_ensemble_rejects_different_grid():
    b = _field(np.zeros((2, 2, 2))).assign_coords(lon=[-72.5, -71.5])
    with pytest.raises(ValueError):
        cpt.ensemble({"a": _field(np.zeros((2, 2, 2))), "b": b})


def test_reduce_lagged_temperature_in_celsius():
    ds = _lagged_ds()
    ds["tprate"] = ds["tprate"].where(ds["tprate"].isnull(), 300.0)
    ds["tprate"].attrs["units"] = "K"
    out = homogenize.reduce_lagged(ds, 5, 6)
    assert "tas" in out and out["tas"].attrs["units"] == "Celsius"
    np.testing.assert_allclose(out["tas"].values, 300.0 - 273.15)


def test_reduce_lagged_single_start_without_time_dim():
    ds = _lagged_ds().isel(number=[0], time=0)
    out = homogenize.reduce_lagged(ds, 5, 6)
    assert out["pr"].sizes["year"] == 1 and not out["pr"].isnull().any()


def test_season_mean_averages_members_and_leads():
    out = homogenize.reduce_lagged(_lagged_ds(), 5, 6)
    m = cpt.season_mean(out, [2, 3, 4])
    np.testing.assert_allclose(m.values, np.mean([2, 3, 4, 102, 103, 104]))


def test_region_args():
    cfg = {"cpt": {"regions": {"colombia": {"north": 14, "west": -82, "south": -5, "east": -66}}}}
    assert cpt.region_args(cfg, "global") == {"use_area": False, "region": "global"}
    assert cpt.region_args(cfg, "prueba") == {"use_area": True, "region": "box"}
    assert cpt.region_args(cfg, "colombia")["area"]["north"] == 14


def test_write_log_reports_members(tmp_path):
    cfg = {"centres": {"ncep": {"system": "2"}}, "cpt": {"regions": {}},
           "area": {"north": 6, "west": -76, "south": 0, "east": -70}}
    files = {"ncep": {"X": tmp_path / "X.txt", "Z": tmp_path / "Z.txt"}}
    log = cpt.write_log(tmp_path / "r.md", cfg, "precipitation", 2026, 10, [2, 3, 4], "prueba",
                        ["ncep"], {"ncep": (24, 120)}, files, _field(np.zeros((2, 2, 2))))
    text = log.read_text(encoding="utf-8")
    assert "| ncep | 2 | 24 | 120 | X.txt | Z.txt |" in text
    assert "2026-11/2027-01" in text and "1993–1994" in text
