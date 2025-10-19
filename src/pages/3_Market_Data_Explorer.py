from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from scripts.fetch_tiingo_price_data import fetch_stock_data
from scripts.load_file_stocks import load_stock_dataframe

try:
    import talib
except ImportError:  # pragma: no cover
    talib = None

CATEGORY_LABELS = {
    "hreturn": "High Return / Growth",
    "hdividend": "High Dividend / Income",
    "hgold": "Gold Hedge ETF",
}

TICKER_LABELS = {
    "AAPL": "Apple Inc.",
    "MSFT": "Microsoft Corp.",
    "NVDA": "NVIDIA Corp.",
    "AMZN": "Amazon.com Inc.",
    "GOOGL": "Alphabet Inc.",
    "LLY": "Eli Lilly & Co.",
    "COST": "Costco Wholesale Corp.",
    "TMO": "Thermo Fisher Scientific",
    "ADBE": "Adobe Inc.",
    "INTC": "Intel Corp.",
    "JNJ": "Johnson & Johnson",
    "PG": "Procter & Gamble",
    "KO": "Coca-Cola Co.",
    "PEP": "PepsiCo Inc.",
    "CVX": "Chevron Corp.",
    "XOM": "Exxon Mobil Corp.",
    "PFE": "Pfizer Inc.",
    "IBM": "International Business Machines",
    "MCD": "McDonald's Corp.",
    "GLD": "SPDR Gold Shares ETF",
}

st.set_page_config(page_title="Termbo Algo Trading Corp", layout="wide")

st.title("Market Data Explorer")
st.caption("Review stored price history, key metrics, and performance charts.")


@st.cache_data(show_spinner=False)
def _load_stock_index() -> pd.DataFrame:
    return load_stock_dataframe(fetch_prices=False)


def _ensure_price_file(row: pd.Series) -> Optional[Path]:
    price_path = row.get("price_data_path")
    if price_path:
        path = Path(price_path)
        if path.exists():
            return path
    # Attempt to infer default location based on naming convention.
    start = row.get("historical_data_start")
    end = row.get("trade_simulation_end")
    if start and end:
        inferred = (
            Path(__file__).resolve().parent.parent
            / "data"
            / "prices"
            / str(row["category"])
            / f"{str(row['ticker'])}_{start}_{end}.csv"
        )
        if inferred.exists():
            return inferred
    return None


def _compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    working = df.copy()
    close = working["close"].to_numpy(dtype=float)
    high = working["high"].to_numpy(dtype=float) if "high" in working else close
    low = working["low"].to_numpy(dtype=float) if "low" in working else close

    if talib is not None:
        working["SMA20"] = talib.SMA(close, timeperiod=20)
        working["SMA50"] = talib.SMA(close, timeperiod=50)
        working["RSI14"] = talib.RSI(close, timeperiod=14)
        macd, macd_signal, macd_hist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
        working["MACD"] = macd
        working["MACD_signal"] = macd_signal
        working["MACD_hist"] = macd_hist
        working["ATR14"] = talib.ATR(high, low, close, timeperiod=14)
    else:  # pragma: no cover
        window20 = working["close"].rolling(window=20, min_periods=1)
        window50 = working["close"].rolling(window=50, min_periods=1)
        working["SMA20"] = window20.mean()
        working["SMA50"] = window50.mean()
        delta = working["close"].diff()
        gain = np.where(delta > 0, delta, 0.0)
        loss = np.where(delta < 0, -delta, 0.0)
        avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean()
        avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean()
        rs = avg_gain / (avg_loss.replace(0, np.nan))
        working["RSI14"] = 100 - (100 / (1 + rs))
        ema12 = working["close"].ewm(span=12, adjust=False).mean()
        ema26 = working["close"].ewm(span=26, adjust=False).mean()
        working["MACD"] = ema12 - ema26
        working["MACD_signal"] = working["MACD"].ewm(span=9, adjust=False).mean()
        working["MACD_hist"] = working["MACD"] - working["MACD_signal"]
        tr1 = working["high"] - working["low"]
        tr2 = (working["high"] - working["close"].shift()).abs()
        tr3 = (working["low"] - working["close"].shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        working["ATR14"] = tr.rolling(window=14, min_periods=14).mean()
    return working


with st.sidebar:
    st.header("Ticker Selection")
    index_df = _load_stock_index()

    if index_df.empty:
        st.error("No stocks available. Populate data via the Time Ranges page.")
        st.stop()

    category = st.selectbox(
        "Category",
        sorted(index_df["category"].unique()),
        format_func=lambda code: CATEGORY_LABELS.get(code, code),
    )

    category_label = CATEGORY_LABELS.get(category, category)
    category_rows = index_df[index_df["category"] == category]
    ticker_list = category_rows["ticker"].tolist()
    ticker = st.selectbox(
        "Ticker",
        ticker_list,
        format_func=lambda symbol: f"{symbol} – {TICKER_LABELS.get(symbol, 'Unknown')}",
    )

    ticker_label = TICKER_LABELS.get(ticker, ticker)
    selected_row = category_rows[category_rows["ticker"] == ticker].iloc[0].copy()
    start_date = selected_row.get("historical_data_start")
    end_date = selected_row.get("trade_simulation_end")
    if selected_row.get("fetch_error"):
        st.warning(f"Previous fetch error: {selected_row['fetch_error']}")


price_file = _ensure_price_file(selected_row)
if price_file is None or not price_file.exists():
    st.warning("No price file available for the selected ticker.")
    st.stop()

price_df = pd.read_csv(price_file, parse_dates=["date"])
if price_df.empty:
    st.warning("Price file is empty.")
    st.stop()

price_df = price_df.sort_values("date")

numeric_cols = ["open", "high", "low", "close", "adjClose", "volume"]
for col in numeric_cols:
    if col in price_df.columns:
        price_df[col] = pd.to_numeric(price_df[col], errors="coerce")

# Date filtering controls
min_date = price_df["date"].min().date()
max_date = price_df["date"].max().date()

def _parse_iso_date(value: Optional[str]) -> Optional[pd.Timestamp]:
    if not value:
        return None
    return pd.to_datetime(value, errors="coerce")

preferred_start = pd.Timestamp("2012-01-01")
preferred_end = pd.Timestamp("2017-12-31")

default_start_ts = preferred_start if min_date <= preferred_start.date() <= max_date else pd.Timestamp(min_date)
default_end_ts = preferred_end if min_date <= preferred_end.date() <= max_date else pd.Timestamp(max_date)

default_start_date = max(default_start_ts.date(), min_date)
default_end_date = min(default_end_ts.date(), max_date)

with st.sidebar:
    st.header("Date Range")
    start_filter = st.date_input("From", value=default_start_date, min_value=min_date, max_value=max_date)
    end_filter = st.date_input("To", value=default_end_date, min_value=min_date, max_value=max_date)

if start_filter > end_filter:
    st.error("Start date must be on or before end date.")
    st.stop()

filtered_df = price_df[
    (price_df["date"] >= pd.to_datetime(start_filter))
    & (price_df["date"] <= pd.to_datetime(end_filter))
].copy()
if filtered_df.empty:
    st.warning("No data available for the selected date range.")
    st.stop()

analysis_df = _compute_indicators(filtered_df)

if talib is None:  # pragma: no cover
    st.info(
        "python-ta-lib is not installed. Falling back to pandas-based indicator calculations. "
        "Install ta-lib for native implementations."
    )

latest_row = analysis_df.iloc[-1]
prev_row = analysis_df.iloc[-2] if len(analysis_df) > 1 else latest_row

st.subheader(f"{ticker_label} ({ticker})")

latest_close = latest_row["close"]

first_row = analysis_df.iloc[0]
first_close = first_row["close"]
period_change = ((latest_close - first_close) / first_close * 100) if first_close else 0.0

metric_cols = st.columns(3)
metric_cols[0].metric("First Close", f"${first_close:,.2f}", help=f"{first_row['date'].strftime('%Y-%m-%d')}")
metric_cols[1].metric("Last Close", f"${latest_close:,.2f}", help=f"{latest_row['date'].strftime('%Y-%m-%d')}")

delta_str = f"{period_change:,.2f}%"
if period_change > 0:
    metric_cols[2].markdown(
        f"<div style='color:#059669;font-weight:bold;'>▲ Period Change<br>{delta_str}</div>",
        unsafe_allow_html=True,
    )
elif period_change < 0:
    metric_cols[2].markdown(
        f"<div style='color:#DC2626;font-weight:bold;'>▼ Period Change<br>{delta_str}</div>",
        unsafe_allow_html=True,
    )
else:
    metric_cols[2].markdown(
        f"<div style='color:#4B5563;font-weight:bold;'>Period Change<br>{delta_str}</div>",
        unsafe_allow_html=True,
    )

rolling_window = st.slider("Custom SMA window", min_value=5, max_value=120, value=30, step=5)
analysis_df["SMA_custom"] = analysis_df["close"].rolling(window=rolling_window, min_periods=1).mean()

price_series = ["close", "SMA20", "SMA50"]
if rolling_window not in (20, 50):
    price_series.append("SMA_custom")

rename_map = {
    "close": "Close",
    "SMA20": "SMA 20",
    "SMA50": "SMA 50",
    "SMA_custom": f"SMA {rolling_window}",
}
plot_df = analysis_df[["date"] + price_series].melt(id_vars="date", var_name="Series", value_name="Price")
plot_df["Series"] = plot_df["Series"].map(rename_map)

chart = px.line(
    plot_df,
    x="date",
    y="Price",
    color="Series",
    labels={"date": "Date"},
    title=f"{ticker_label} ({ticker}) Price History – {category_label}",
)
chart.update_layout(legend_title_text="")
st.plotly_chart(chart, use_container_width=True)

st.subheader("Technical Indicators")
indicator_cols = st.columns(4)
indicator_cols[0].metric("SMA 20", f"{latest_row.get('SMA20', float('nan')):,.2f}")
indicator_cols[1].metric("SMA 50", f"{latest_row.get('SMA50', float('nan')):,.2f}")
indicator_cols[2].metric("RSI 14", f"{latest_row.get('RSI14', float('nan')):,.2f}")
indicator_cols[3].metric("ATR 14", f"{latest_row.get('ATR14', float('nan')):,.2f}")

macd_df = analysis_df[["date", "MACD", "MACD_signal"]].copy()
macd_plot = px.line(
    macd_df.melt(id_vars="date", var_name="Series", value_name="Value"),
    x="date",
    y="Value",
    color="Series",
    labels={"date": "Date"},
    title="MACD vs Signal",
)
macd_hist = analysis_df[["date", "MACD_hist"]].copy()
macd_hist_chart = px.bar(macd_hist, x="date", y="MACD_hist", title="MACD Histogram")

rsi_chart = px.line(
    analysis_df,
    x="date",
    y="RSI14",
    labels={"date": "Date", "RSI14": "RSI"},
    title="RSI (14)",
)
rsi_chart.add_hline(y=70, line_dash="dash", line_color="red")
rsi_chart.add_hline(y=30, line_dash="dash", line_color="green")

indicator_tabs = st.tabs(["MACD", "MACD Histogram", "RSI"])
with indicator_tabs[0]:
    st.plotly_chart(macd_plot, use_container_width=True)
with indicator_tabs[1]:
    st.plotly_chart(macd_hist_chart, use_container_width=True)
with indicator_tabs[2]:
    st.plotly_chart(rsi_chart, use_container_width=True)
