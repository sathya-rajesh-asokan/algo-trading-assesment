"""
Utility script to list NYSE trading days from 2008 through today.

Requires `pandas` and `pandas_market_calendars` to be installed in the
environment (pip install pandas pandas_market_calendars).
"""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from typing import Iterable, List

import pandas as pd
import pandas_market_calendars as mcal

OUTPUT_FILE = Path(__file__).resolve().parent.parent / "data" / "nyse_trading_days.csv"


def validate_date(date_str: str) -> datetime:
    try:
        return datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError as exc:
        raise ValueError(f"Invalid date '{date_str}'. Expected YYYY-MM-DD.") from exc


def get_trading_days(start: str, end: str, calendar_code: str) -> List[date]:
    start_dt = validate_date(start)
    end_dt = validate_date(end)
    if start_dt > end_dt:
        raise ValueError("Start date must be earlier than or equal to end date.")

    calendar = mcal.get_calendar(calendar_code)
    schedule = calendar.schedule(start_date=start_dt, end_date=end_dt)

    index = schedule.index
    if getattr(index, "tz", None) is not None:
        index = index.tz_localize(None)

    trading_dates = index.date
    return list(trading_dates)


def build_dataframe(dates: Iterable[date]) -> pd.DataFrame:
    dates_list = list(dates)
    if not dates_list:
        return pd.DataFrame(
            columns=[
                "trade_date",
                "year",
                "quarter",
                "trading_day_of_month",
                "is_month_last",
                "is_quarter_last",
            ]
        )

    trading_index = pd.to_datetime(dates_list)
    frame = pd.DataFrame({"trade_date": trading_index.date})
    frame["year"] = trading_index.year
    frame["quarter"] = ((trading_index.month - 1) // 3) + 1
    frame["month_key"] = trading_index.to_period("M")
    frame["quarter_key"] = trading_index.to_period("Q")
    frame["trading_day_of_month"] = frame.groupby("month_key").cumcount() + 1
    month_last = frame.groupby("month_key")["trade_date"].transform("last")
    frame["is_month_last"] = frame["trade_date"] == month_last
    quarter_last = frame.groupby("quarter_key")["trade_date"].transform("last")
    frame["is_quarter_last"] = frame["trade_date"] == quarter_last
    frame["trade_date"] = frame["trade_date"].astype(str)
    return frame.drop(columns=["month_key", "quarter_key"])


def write_csv(dates: Iterable[date], output_path: Path) -> None:
    dataframe = build_dataframe(dates)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(output_path, index=False)


def main() -> None:
    trading_days = get_trading_days("2008-01-01", date.today().strftime("%Y-%m-%d"), "XNYS")
    write_csv(trading_days, OUTPUT_FILE)
    print(f"Wrote {len(trading_days)} trading days to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
