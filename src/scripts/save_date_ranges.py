"""
Helper utilities for persisting user-selected date ranges.
"""

import json
from datetime import date
from pathlib import Path
from typing import Iterable, Optional, Tuple

OUTPUT_FILE = Path(__file__).resolve().parent.parent / "data" / "date_ranges.json"


def _coerce_date(value: Optional[date]) -> Optional[str]:
    return value.isoformat() if value else None


def _normalize_range(range_values: Iterable[Optional[date]]) -> Tuple[Optional[str], Optional[str]]:
    start, end = range_values
    return _coerce_date(start), _coerce_date(end)


def save_date_ranges(
    historical_start: Optional[date],
    backtest_range: Iterable[Optional[date]],
    simulation_range: Iterable[Optional[date]],
    output_file: Path = OUTPUT_FILE,
) -> Path:
    output_file.parent.mkdir(parents=True, exist_ok=True)

    backtest_start, backtest_end = _normalize_range(backtest_range)
    simulation_start, simulation_end = _normalize_range(simulation_range)

    payload = {
        "historical_start": _coerce_date(historical_start),
        "backtesting_window": {
            "start": backtest_start,
            "end": backtest_end,
        },
        "trade_simulation_window": {
            "start": simulation_start,
            "end": simulation_end,
        },
    }

    with output_file.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    return output_file


if __name__ == "__main__":
    raise SystemExit(
        "This module is intended to be imported. Use `save_date_ranges` from Streamlit."
    )
