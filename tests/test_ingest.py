import datetime as dt
from pathlib import Path

import pytest
import requests

from ideam_seasonal import ingest

# respuesta real del CDS (2026-11-21), abreviada
MARS_NOT_PUBLISHED = (
    "400 Client Error: Bad Request for url: https://cds.climate.copernicus.eu/api/retrieve/v1/"
    "jobs/x/results\nThe job has failed\nMARS has returned an error, please check your selection.\n"
    "mars - ERROR  - 20261121.160742 - Mars server task finished in error\n"
    "mars - ERROR  - 20261121.160742 - AccessError: Restricted access to current C3S seasonal "
    "forecast non-ECMWF contribution. Please check that your date selection is valid.\n")


def _http_error(status: int, text: str) -> requests.HTTPError:
    resp = requests.Response()
    resp.status_code = status
    return requests.HTTPError(text, response=resp)


def _fake_download(available: set[tuple[int, int]], error: Exception | None = None):
    calls = []

    def fake(cfg, centre, variable, region=None, force=False, month=None, forecast_year=None):
        calls.append((centre, forecast_year, month))
        if (forecast_year, month) in available:
            return Path(f"{centre}_f{forecast_year}{month:02d}.grib")
        raise error or _http_error(400, MARS_NOT_PUBLISHED)

    return fake, calls


CFG = {}


def test_unavailable_detection():
    assert ingest.forecast_unavailable(_http_error(400, MARS_NOT_PUBLISHED))
    assert not ingest.forecast_unavailable(_http_error(401, "401 Unauthorized: invalid key"))
    assert not ingest.forecast_unavailable(_http_error(403, "required licences not accepted"))
    assert not ingest.forecast_unavailable(_http_error(400, "400 Bad Request: invalid area"))
    assert not ingest.forecast_unavailable(_http_error(500, "MARS internal error"))
    assert not ingest.forecast_unavailable(requests.ConnectionError("no network"))
    assert not ingest.forecast_unavailable(RuntimeError("otro"))


def test_uses_current_month_when_published(monkeypatch):
    fake, calls = _fake_download({(2026, 11)})
    monkeypatch.setattr(ingest, "download", fake)
    y, m, paths = ingest.latest_forecast(CFG, ["ncep", "dwd"], today=dt.date(2026, 11, 21))
    assert (y, m) == (2026, 11) and set(paths) == {"ncep", "dwd"}


def test_falls_back_when_not_published(monkeypatch, capsys):
    fake, calls = _fake_download({(2026, 11)})
    monkeypatch.setattr(ingest, "download", fake)
    y, m, _ = ingest.latest_forecast(CFG, ["ncep"], today=dt.date(2026, 12, 3))
    assert (y, m) == (2026, 11)
    assert "AccessError: Restricted access" in capsys.readouterr().out


def test_falls_back_across_year(monkeypatch):
    fake, _ = _fake_download({(2025, 12)})
    monkeypatch.setattr(ingest, "download", fake)
    assert ingest.latest_forecast(CFG, ["bom"], today=dt.date(2026, 1, 2))[:2] == (2025, 12)


def test_all_centres_must_share_the_month(monkeypatch):
    """Si un centro no tiene el mes, se retrocede para todos."""
    calls = []

    def fake(cfg, centre, variable, region=None, force=False, month=None, forecast_year=None):
        calls.append((centre, month))
        if centre == "dwd" and month == 11:
            raise _http_error(400, MARS_NOT_PUBLISHED)
        return Path(f"{centre}_{month}.grib")

    monkeypatch.setattr(ingest, "download", fake)
    y, m, paths = ingest.latest_forecast(CFG, ["ncep", "dwd"], today=dt.date(2026, 11, 21))
    assert m == 10 and paths["ncep"].name == "ncep_10.grib"


@pytest.mark.parametrize("error", [
    _http_error(401, "401 Unauthorized: invalid key"),
    _http_error(403, "403 Forbidden: required licences not accepted"),
    _http_error(503, "503 Service Unavailable"),
    requests.ConnectionError("no network"),
])
def test_other_errors_are_not_hidden(monkeypatch, error):
    fake, calls = _fake_download({(2026, 10)}, error=error)
    monkeypatch.setattr(ingest, "download", fake)
    with pytest.raises(type(error)):
        ingest.latest_forecast(CFG, ["ncep"], today=dt.date(2026, 11, 21))
    assert calls == [("ncep", 2026, 11)]  # no siguió con octubre


def test_gives_up_after_months_back(monkeypatch):
    fake, calls = _fake_download(set())
    monkeypatch.setattr(ingest, "download", fake)
    with pytest.raises(RuntimeError, match="ningún pronóstico publicado"):
        ingest.latest_forecast(CFG, ["ncep"], today=dt.date(2026, 11, 21), months_back=3)
    assert [c[2] for c in calls] == [11, 10, 9]
