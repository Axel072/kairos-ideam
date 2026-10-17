import numpy as np
import pytest
import xarray as xr

from ideam_seasonal import ingest, verify


def _da(vals, units, step_type="avg"):
    return xr.DataArray(vals, dims=("forecastMonth",), attrs={"units": units, "GRIB_stepType": step_type})


def test_longitudes_mixed_fails():
    ds = xr.Dataset(coords={"longitude": [-10.0, 10.0, 200.0]})
    assert not verify.check_longitudes(ds)[0]


def test_longitudes_signed_ok():
    ds = xr.Dataset(coords={"longitude": [-75.5, -74.5, -70.5]})
    assert verify.check_longitudes(ds)[0]


def test_units_rate_ok_and_scale_error_caught():
    assert verify.check_units(_da(np.full(6, 3e-5), "kg m**-2 s**-1"))[0]
    assert not verify.check_units(_da(np.full(6, 3e-2), "kg m**-2 s**-1"))[0]
    assert not verify.check_units(_da(np.full(6, 3.0), "furlongs"))[0]


def test_accumulated_detected():
    assert not verify.check_accumulation(_da([1, 2, 3, 4], "m", "accum"))[0]
    assert not verify.check_accumulation(_da([1.0, 2.0, 3.0, 4.0], "m"))[0]
    assert verify.check_accumulation(_da([1e-5, 2e-5, 1e-5, 3e-5], "kg m**-2 s**-1"))[0]


def test_requests_per_centre():
    cfg = ingest.load_config()
    expected = {"ncep": "2", "dwd": "22", "bom": "2"}
    for c, system in expected.items():
        r = ingest.build_request(cfg, c)
        assert r["originating_centre"] == c and r["system"] == system
        assert len(r["year"]) == 24 and r["area"] == [6, -76, 0, -70]
