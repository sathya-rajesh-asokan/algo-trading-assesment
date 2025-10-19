from datetime import date

import streamlit as st

from scripts.save_date_ranges import save_date_ranges
import scripts.fetch_trading_days as fetch_trading_days
import scripts.load_file_stocks as load_file_stocks

st.set_page_config(page_title="Termbo Algo Trading Corp", layout="wide")

st.title("Termbo Algo Trading Corp")
st.caption("Grow your capital with us")
st.caption("We are in emulation mode. No live orders will be placed.")

date_cols = st.columns(3)

with date_cols[0]:
    historical_start = st.date_input(
        "Historical Data Start",
        value=date(2008, 1, 1),
        key="historical_data_start",
    )

with date_cols[1]:
    backtest_range = st.date_input(
        "Backtesting Window",
        value=(date(2012, 1, 1), date(2017, 12, 31)),
        key="backtesting_window",
    )

with date_cols[2]:
    simulation_range = st.date_input(
        "Trade Simulation Window",
        value=(date(2018, 1, 1), date(2024, 12, 31)),
        key="trade_simulation_window",
    )


def _ensure_range(range_value):
    if isinstance(range_value, tuple):
        return range_value
    return (range_value, range_value)


if st.button("Save Date Selections"):
    saved_path = save_date_ranges(
        historical_start,
        _ensure_range(backtest_range),
        _ensure_range(simulation_range),
    )
    fetch_trading_days.main()
    load_file_stocks.main()
    st.success("Date selections saved")
