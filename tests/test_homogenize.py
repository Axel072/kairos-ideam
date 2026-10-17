import numpy as np
import pandas as pd
import pytest
import xarray as xr

from ideam_seasonal import homogenize

F = 1 / 86_400_000  # mm/día -> m s**-1


def _lagged_ds():
    """2 miembros: el 0 inicia el 1 may, el 1 el 26 abr; producto cartesiano disperso."""
    valid = np.array([[pd.Timestamp(1993, 6 + s % 6, 1) for s in range(12)]] * 2, dtype="datetime64[ns]")
    arr = np.full((2, 2, 12, 2, 1), np.nan)
    for lead in range(6):
        arr[0, 0, lead] = (0 * 100 + lead + 1) * F      
        arr[1, 1, 6 + lead] = (1 * 100 + lead + 1) * F  
    return xr.Dataset(
        {"tprate": (("number", "time", "step", "latitude", "longitude"), arr,
                    {"units": "m s**-1"})},
        coords={"number": [0, 1], "time": np.array(["1993-05-01", "1993-04-26"], dtype="datetime64[ns]"),
                "step": np.arange(12), "latitude": [1.5, 0.5], "longitude": [-70.5],
                "valid_time": (("time", "step"), valid)})


def test_reduce_lagged_maps_member_year_lead():
    out = homogenize.reduce_lagged(_lagged_ds(), start_month=5, n_leads=6)
    assert out["pr"].dims == ("year", "member", "lead", "lat", "lon")
    assert out["pr"].sizes["lead"] == 6 and not out["pr"].isnull().any()
    assert list(out["lat"].values) == [0.5, 1.5]  # latitudes crecientes
    np.testing.assert_allclose(out["pr"].sel(year=1993, member=1, lead=3).values.ravel(), 103)
    assert list(out["target_month"].values) == [5, 6, 7, 8, 9, 10]
    assert pd.Timestamp(out["init_time"].sel(year=1993, member=1).item()) == pd.Timestamp(1993, 4, 26)


def test_reduce_lagged_rejects_unknown_unit():
    ds = _lagged_ds()
    ds["tprate"].attrs["units"] = "m"
    with pytest.raises(ValueError):
        homogenize.reduce_lagged(ds, 5, 6)


def test_reduce_lagged_rejects_partial_fill():
    ds = _lagged_ds()
    ds["tprate"][0, 0, 0] = np.nan  
    with pytest.raises(ValueError):
        homogenize.reduce_lagged(ds, 5, 6)


def test_chirps_lead1_is_start_month(tmp_path):
    """lead 1 debe ser el mes de inicio (mayo), no el anterior."""
    days = (pd.date_range("2000-01-01", periods=12, freq="MS") - pd.Timestamp("1980-01-01")).days
    dim = pd.date_range("2000-01-01", periods=12, freq="MS").days_in_month.to_numpy()
    vals = (np.arange(1, 13) * dim)[:, None, None] * np.ones((12, 2, 2))  # mm/month => mm/día = mes
    ds = xr.Dataset({"precip": (("time", "latitude", "longitude"), vals.astype("float32"),
                                {"units": "mm/month", "_FillValue": np.float32(-9999)})},
                    coords={"time": ("time", days.astype("float64"), {"units": "days since 1980-1-1 0:0:0"}),
                            "latitude": [0.025, 0.075], "longitude": [-75.975, -75.925]})
    p = tmp_path / "chirps_v3_2000_box.nc"
    ds.to_netcdf(p)
    out = homogenize.chirps_by_lead([p], np.array([2000]), start_month=5, n_leads=6)
    np.testing.assert_allclose(out.isel(year=0, lat=0, lon=0).values, [5, 6, 7, 8, 9, 10])
