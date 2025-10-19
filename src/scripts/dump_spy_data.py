"""
Download SPY benchmark data from Tiingo and store it under data/prices/benchmarks.

The script mirrors the Streamlit data-fetching workflow and honours the date
window configured in `data/date_ranges.json` unless explicit overrides are
passed on the command line.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Tuple

from fetch_tiingo_price_data import fetch_stock_data

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_ROOT = SCRIPT_DIR.parent
DATA_DIR = SRC_ROOT / "data"
DATE_RANGES_FILE = DATA_DIR / "date_ranges.json"
DEFAULT_CATEGORY = "benchmarks"
DEFAULT_TICKER = "SPY"


def _load_date_ranges() -> dict:
    if not DATE_RANGES_FILE.exists():
        raise FileNotFoundError(
            f"date_ranges.json not found at {DATE_RANGES_FILE}. "
            "Run the Streamlit configuration or create the file manually."
        )
    with DATE_RANGES_FILE.open("r", encoding="utf-8") as handle:
        try:
            payload = json.load(handle)
        except json.JSONDecodeError as exc:
            raise ValueError("date_ranges.json is not valid JSON.") from exc
    if not isinstance(payload, dict):
        raise ValueError("Expected date_ranges.json to contain a JSON object.")
    return payload


def _resolve_default_window() -> Tuple[str, str]:
    payload = _load_date_ranges()
    start = payload.get("historical_start")
    trade_window = payload.get("trade_simulation_window", {})
    backtest_window = payload.get("backtesting_window", {})

    end = None
    if isinstance(trade_window, dict):
        end = trade_window.get("end")
    if end is None and isinstance(backtest_window, dict):
        end = backtest_window.get("end")

    if not start or not end:
        raise ValueError(
            "Could not determine default start/end from date_ranges.json. "
            "Provide --start/--end explicitly."
        )
    return str(start), str(end)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch SPY benchmark data from Tiingo into data/prices/benchmarks."
    )
    parser.add_argument(
        "--start",
        help="Override start date (YYYY-MM-DD). Defaults to historical_start in date_ranges.json.",
    )
    parser.add_argument(
        "--end",
        help="Override end date (YYYY-MM-DD). Defaults to trade_simulation_window end.",
    )
    parser.add_argument(
        "--token",
        help="Override Tiingo API token (falls back to value in fetch_tiingo_price_data.py).",
    )
    parser.add_argument(
        "--resample",
        default="daily",
        help="Tiingo resample frequency (default: daily).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.start and args.end:
        start, end = args.start, args.end
    else:
        start, end = _resolve_default_window()
        if args.start:
            start = args.start
        if args.end:
            end = args.end

    csv_path = fetch_stock_data(
        start_date=start,
        end_date=end,
        ticker=DEFAULT_TICKER,
        category=DEFAULT_CATEGORY,
        api_token=args.token,
        output_dir=DATA_DIR / "prices",
        resample_freq=args.resample,
    )

    print(f"Stored SPY data at: {csv_path}")


if __name__ == "__main__":
    main()
