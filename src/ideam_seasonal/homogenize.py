"""
Uso:  python -m ideam_seasonal.homogenize
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from ideam_seasonal.ingest import load_config
from ideam_seasonal.verify import (CENTRES, PRECIP_UNIT_TO_MM_DAY, _chirps_open, _open_grib,
                                   get_units)


def reduce_lagged(ds: xr.Dataset, start_month: int, n_leads: int) -> xr.Dataset:
    """Reduce el producto cartesiano (number, time, step) a (year, member, lead)."""
    var = next(iter(ds.data_vars))
    da = ds[var]
    unit = get_units(da)
    factor = PRECIP_UNIT_TO_MM_DAY.get(unit)
    if factor is None:
        raise ValueError(f"unidad '{unit}' no reconocida o acumulada; no se homogeneiza")
    arr = da.transpose("number", "time", "step", "latitude", "longitude").values
    init = pd.DatetimeIndex(ds["time"].values)
    valid = pd.DatetimeIndex(ds["valid_time"].transpose("time", "step").values.ravel())
    valid = valid.values.reshape(len(init), -1)

    has = ~np.isnan(arr).all(axis=(3, 4))  # (number, time, step)
    years = np.sort(np.unique(init.year))
    members = ds["number"].values
    out = np.full((len(years), len(members), n_leads, arr.shape[3], arr.shape[4]), np.nan,
                  dtype="float32")
    init_time = np.full((len(years), len(members)), np.datetime64("NaT", "ns"))
    for i, _ in enumerate(members):
        ti, si = np.nonzero(has[i])
        vt = pd.DatetimeIndex(valid[ti, si])
        if not ((vt.day == 1) & (vt.hour == 0)).all():
            raise ValueError("valid_time no cae en el día 1 00 h; mapeo de anticipación dudoso")
        yr = init.year[ti]
        lead = (vt.year - yr) * 12 + vt.month - start_month  # fin de periodo - mes de inicio
        if lead.min() < 1 or lead.max() > n_leads:
            raise ValueError(f"member {members[i]}: anticipación fuera de 1..{n_leads}: "
                             f"{sorted(set(lead))}")
        yi = np.searchsorted(years, yr)
        if len(set(zip(yi, lead))) != len(yi):
            raise ValueError(f"member {members[i]}: (año, anticipación) duplicado")
        out[yi, i, lead - 1] = arr[i, ti, si] * factor
        init_time[yi, i] = init[ti].values
    partial = np.isnan(out).any(axis=(3, 4)) & ~np.isnan(out).all(axis=(3, 4))
    if np.isnan(out).any():
        raise ValueError(f"el arreglo reducido tiene {np.isnan(out).mean():.2%} NaN "
                         f"(celdas parciales: {int(partial.sum())}); no se rellena")
    lat = ds["latitude"].values
    res = xr.Dataset(
        {"pr": (("year", "member", "lead", "lat", "lon"), out,
                {"units": "mm/day", "source_units": unit,
                 "long_name": "precipitación, tasa media mensual"}),
         "init_time": (("year", "member"), init_time)},
        coords={"year": years, "member": members, "lead": np.arange(1, n_leads + 1),
                "lat": lat, "lon": ds["longitude"].values})
    res = res.assign_coords(
        target_month=("lead", (start_month - 1 + np.arange(n_leads)) % 12 + 1))
    return res.sortby("lat")


def chirps_by_lead(paths: list[Path], years: np.ndarray, start_month: int, n_leads: int):
    """CHIRPS (mm/día) por (year, lead): el mes calendario `start_month + lead - 1`."""
    by_year = {}
    for p in paths:
        ds = _chirps_open(p)
        da = ds["precip"]
        if "scale_factor" in da.attrs:
            raise ValueError("CHIRPS trae scale_factor; la lectura sin escalar sería errónea")
        fill = da.attrs.get("_FillValue")
        da = da.astype("float64")
        if fill is not None:
            da = da.where(da != fill)
        da = da / da["time"].dt.days_in_month  
        by_year[int(da["time"].dt.year[0])] = da
    arrs = []
    for y in years:
        m = start_month + np.arange(n_leads)  
        if m.max() > 12:
            raise ValueError("leads cruzan de año; no soportado todavía")
        t = [pd.Timestamp(int(y), int(mm), 1) for mm in m]
        arrs.append(by_year[int(y)].sel(time=t).rename(time="lead").assign_coords(lead=np.arange(1, n_leads + 1)))
    pr = xr.concat(arrs, dim=xr.DataArray(years, dims="year", name="year"))
    pr = pr.rename(latitude="lat", longitude="lon").sortby("lat").drop_vars("time", errors="ignore")
    pr.attrs = {"units": "mm/day", "source_units": "mm/month"}
    return pr


def chirps_to_model_grid(pr: xr.DataArray, lat: np.ndarray, lon: np.ndarray,
                         step: float = 1.0) -> xr.DataArray:
    """Promedio por celda de `step`° centrada en las coordenadas del modelo (20x20 celdas)."""
    cell_lat = xr.DataArray(np.floor(pr["lat"].values / step) * step + step / 2, dims="lat")
    cell_lon = xr.DataArray(np.floor(pr["lon"].values / step) * step + step / 2, dims="lon")
    g = pr.assign_coords(cell_lat=cell_lat, cell_lon=cell_lon)
    n_block = round(step / abs(float(pr["lat"][1] - pr["lat"][0])))
    sizes = g["cell_lat"].to_series().value_counts().unique(), g["cell_lon"].to_series().value_counts().unique()
    if any(list(s) != [n_block] for s in sizes):
        raise ValueError(f"bloques de CHIRPS no uniformes de {n_block} celdas: {sizes}")
    out =g.groupby("cell_lat").mean("lat").groupby("cell_lon").mean("lon")
    out = out.rename(cell_lat="lat", cell_lon="lon").sel(lat=lat, lon=lon, method="nearest")
    if not (np.allclose(out["lat"], lat) and np.allclose(out["lon"], lon)):
        raise ValueError("las celdas de CHIRPS no coinciden con la rejilla de los modelos")
    out.attrs = pr.attrs
    return out


def build(raw: Path = Path("data") / "raw", processed: Path = Path("data") / "processed",
          variable: str = "precipitation") -> Path:
    cfg = load_config()
    month, n_leads = cfg["month"], len(cfg["leadtime_months"])
    out_path = processed / f"homogeneizado_{variable}_m{month:02d}_box.nc"
    processed.mkdir(parents=True, exist_ok=True)
    out_path.unlink(missing_ok=True)

    groups = {}
    for c in CENTRES:
        groups[c] = reduce_lagged(_open_grib(raw / f"{c}_{variable}_m{month:02d}_box.grib"),
                                  month, n_leads)
    years = groups["ncep"]["year"].values
    for c, g in groups.items():
        if not np.array_equal(g["year"].values, years):
            raise ValueError(f"{c}: años distintos a NCEP")
    for c in ("dwd", "bom"):
        if not (np.allclose(groups[c]["lat"], groups["ncep"]["lat"]) and
                np.allclose(groups[c]["lon"], groups["ncep"]["lon"])):
            raise ValueError(f"{c}: rejilla distinta a NCEP")

    ch = chirps_by_lead(sorted(raw.glob("chirps_v3_*_box.nc")), years, month, n_leads)
    groups["chirps"] = ch.to_dataset(name="pr")
    groups["chirps_1deg"] = chirps_to_model_grid(
        ch, groups["ncep"]["lat"].values, groups["ncep"]["lon"].values).to_dataset(name="pr")

    for i, (name, ds) in enumerate(groups.items()):
        ds.attrs = {"source": name, "start_month": month, "nota": "no se corrigió ni normalizó "
                    "nada; solo reducción de dimensiones y conversión a mm/día"}
        ds.to_netcdf(out_path, group=name, mode="w" if i == 0 else "a", engine="netcdf4")
    return out_path


if __name__ == "__main__":
    print(build())
