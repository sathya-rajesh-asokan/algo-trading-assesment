"""
Fetch daily historical price data from Tiingo and persist it as CSV files.

Designed for programmatic use (no CLI). The core `fetch_stock_data` function can
be imported and called from Streamlit pages or other modules. The Tiingo API
token is hard-coded in this module—update `TIINGO_API_TOKEN` before use.

Requirements:
  * `pandas`
  * `requests`
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
import requests


TIINGO_BASE_URL = "https://api.tiingo.com/tiingo/daily/{ticker}/prices"
TIINGO_API_TOKEN = "8d8cd7a6a268ebdbd9ad1d8bcaba82e1318d8ce1"
SCRIPT_DIR = Path(__file__).resolve().parent
SRC_ROOT = SCRIPT_DIR.parent
DEFAULT_OUTPUT_DIR = SRC_ROOT / "data" / "prices"


@dataclass(frozen=True)
class TiingoConfig:
    api_token: str
    base_url: str = TIINGO_BASE_URL


def _resolve_output_dir(path_value: Optional[Path]) -> Path:
    if path_value is None:
        return DEFAULT_OUTPUT_DIR
    if path_value.is_absolute():
        return path_value
    return (SCRIPT_DIR / path_value).resolve()


def _parse_iso_date(value: str) -> datetime:
    try:
        return datetime.strptime(value, "%Y-%m-%d")
    except ValueError as exc:
        raise ValueError(f"Invalid date '{value}'. Expected YYYY-MM-DD.") from exc


def _segment_by_year(start_dt: datetime, end_dt: datetime) -> List[Tuple[datetime, datetime]]:
    segments: List[Tuple[datetime, datetime]] = []
    current_start = start_dt
    while current_start <= end_dt:
        year_end = datetime(current_start.year, 12, 31, 23, 59, 59)
        segment_end = min(end_dt, year_end)
        segments.append((current_start, segment_end))
        current_start = (segment_end + timedelta(days=1)).replace(hour=0, minute=0, second=0)
    return segments


def _fetch_segment_from_tiingo(
    config: TiingoConfig,
    ticker: str,
    segment_start: datetime,
    segment_end: datetime,
    resample_freq: str,
) -> pd.DataFrame:
    url = config.base_url.format(ticker=ticker)
    params = {
        "startDate": segment_start.strftime("%Y-%m-%d"),
        "endDate": segment_end.strftime("%Y-%m-%d"),
        "resampleFreq": resample_freq,
        "format": "json",
    }
    headers = {"Authorization": f"Token {config.api_token}"}

    response = requests.get(url, params=params, headers=headers, timeout=30)
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        raise RuntimeError(
            f"Tiingo request failed for {ticker} "
            f"[{params['startDate']} -> {params['endDate']}]: {response.text}"
        ) from exc

    payload = response.json()
    if not payload:
        return pd.DataFrame()

    return pd.DataFrame(payload)


def fetch_stock_data(
    start_date: str,
    end_date: str,
    ticker: str,
    category: str,
    *,
    api_token: Optional[str] = None,
    output_dir: Optional[Path] = None,
    resample_freq: str = "daily",
) -> Path:
    """
    Fetch 1-day interval historical prices for the given ticker from Tiingo and persist to CSV.

    Parameters
    ----------
    start_date : str
        Start date in YYYY-MM-DD format.
    end_date : str
        End date in YYYY-MM-DD format.
    ticker : str
        Ticker symbol recognised by Tiingo.
    category : str
        Logical grouping used to organise the CSV output.
    api_token : str, optional
        Tiingo API token. If omitted, the TIINGO_API_TOKEN environment variable is used.
    output_dir : Path
        Directory where the CSV will be stored.
    resample_freq : str
        Tiingo resample frequency (default daily).
    """
    start_dt = _parse_iso_date(start_date)
    end_dt = _parse_iso_date(end_date)
    if end_dt < start_dt:
        raise ValueError("End date must be on or after the start date.")

    token = api_token or TIINGO_API_TOKEN
    if not token or token == "REPLACE_WITH_YOUR_TIINGO_TOKEN":
        raise RuntimeError("Tiingo API token is not configured. Update TIINGO_API_TOKEN in this module.")

    config = TiingoConfig(api_token=token)

    output_dir = _resolve_output_dir(output_dir) / category
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{ticker}_{start_date}_{end_date}.csv"
    if csv_path.exists():
        return csv_path

    frames: List[pd.DataFrame] = []
    for seg_start, seg_end in _segment_by_year(start_dt, end_dt):
        segment_df = _fetch_segment_from_tiingo(
            config,
            ticker,
            seg_start,
            seg_end,
            resample_freq=resample_freq,
        )
        if not segment_df.empty:
            frames.append(segment_df)

    if not frames:
        raise RuntimeError(f"No historical data returned for {ticker}.")

    df = pd.concat(frames, ignore_index=True)

    if "date" not in df.columns:
        raise RuntimeError("Tiingo response missing 'date' field; check API response.")

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").drop_duplicates(subset=["date"], keep="last")
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")

    df["category"] = category
    df["ticker"] = ticker
    df["source"] = "Tiingo"
    df["resampleFreq"] = resample_freq

    df.to_csv(csv_path, index=False)
    return csv_path
