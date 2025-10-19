import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from scripts.load_file_stocks import load_stock_dataframe

try:
    import talib
except ImportError:  # pragma: no cover
    talib = None

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
STRATEGY_FILE = DATA_DIR / "strategies.json"
PRICE_DIR = DATA_DIR / "prices"
DATE_RANGES_FILE = DATA_DIR / "date_ranges.json"

INITIAL_CAPITAL = 1_000_000.0
ALLOCATION_MAP = {
    "hreturn": 0.60,
    "hdividend": 0.25,
    "hgold": 0.15,
}
BASE_GOLD_TARGET = 0.10  # Portion invested in GLD
GOLD_REBALANCE_FREQUENCY = "Q"  # Quarterly
GOLD_TICKER = "GLD"

CATEGORY_LABELS = {
    "hreturn": "High Return / Growth",
    "hdividend": "High Dividend / Income",
    "hgold": "Gold Hedge ETF",
}

st.set_page_config(page_title="Strategy Backtesting", layout="wide")

st.title("Strategy Backtesting")
st.caption(
    "Simulate rule-based strategies against stored price history with an automated gold hedge allocation."
)


@dataclass
class BacktestResult:
    ticker: str
    category: str
    trades: int
    starting_capital: float
    ending_value: float
    return_pct: float
    equity_curve: pd.DataFrame
    buy_rules: Iterable[Dict]
    sell_rules: Iterable[Dict]

        
def _calculate_sharpe_ratio(equity: pd.Series, risk_free_rate: float = 0.0) -> float:
    returns = equity.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    if returns.empty:
        return float("nan")
    excess = returns - risk_free_rate / 252
    std = excess.std()
    if std == 0 or pd.isna(std):
        return float("nan")
    return (excess.mean() * 252) / std


def _calculate_max_drawdown(equity: pd.Series) -> float:
    running_max = equity.cummax()
    drawdown = equity / running_max - 1.0
    return drawdown.min() if not drawdown.empty else float("nan")


def _calculate_volatility(equity: pd.Series) -> float:
    returns = equity.pct_change().dropna()
    if returns.empty:
        return float("nan")
    return returns.std() * np.sqrt(252)

def _load_strategies() -> Dict[str, Dict]:
    if STRATEGY_FILE.exists():
        try:
            return json.loads(STRATEGY_FILE.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            st.error("Failed to read strategies.json. Fix the JSON format and retry.")
            st.stop()
    return {}


def _load_backtesting_window() -> Tuple[pd.Timestamp, pd.Timestamp]:
    if not DATE_RANGES_FILE.exists():
        st.error("date_ranges.json not found. Configure date ranges before running the backtest.")
        st.stop()
    try:
        payload = json.loads(DATE_RANGES_FILE.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        st.error("Could not parse date_ranges.json. Ensure it is valid JSON.")
        st.stop()
    window = payload.get("backtesting_window")
    if not window:
        st.error("backtesting_window is missing in date_ranges.json.")
        st.stop()
    try:
        start = pd.to_datetime(window["start"])
        end = pd.to_datetime(window["end"])
    except (KeyError, TypeError, ValueError):
        st.error("Invalid backtesting_window dates in date_ranges.json.")
        st.stop()
    if start > end:
        st.error("backtesting_window start date must be before end date.")
        st.stop()
    return start, end


def _compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    working = df.copy()
    for column in ("open", "high", "low", "close", "adjClose", "volume"):
        if column in working:
            working[column] = pd.to_numeric(working[column], errors="coerce")

    close = working["close"].to_numpy(dtype=float)
    high = working.get("high", working["close"]).to_numpy(dtype=float)
    low = working.get("low", working["close"]).to_numpy(dtype=float)

    if talib is not None:
        working["SMA20"] = talib.SMA(close, timeperiod=20)
        working["SMA50"] = talib.SMA(close, timeperiod=50)
        working["SMA_custom"] = talib.SMA(close, timeperiod=30)
        working["RSI14"] = talib.RSI(close, timeperiod=14)
        macd, macd_signal, macd_hist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
        working["MACD"] = macd
        working["MACD_signal"] = macd_signal
        working["MACD_hist"] = macd_hist
        working["ATR14"] = talib.ATR(high, low, close, timeperiod=14)
    else:  # pragma: no cover
        working["SMA20"] = working["close"].rolling(window=20, min_periods=1).mean()
        working["SMA50"] = working["close"].rolling(window=50, min_periods=1).mean()
        working["SMA_custom"] = working["close"].rolling(window=30, min_periods=1).mean()
        delta = working["close"].diff()
        gain = np.where(delta > 0, delta, 0.0)
        loss = np.where(delta < 0, -delta, 0.0)
        avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean()
        avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
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


def _load_price_data(category: str, ticker: str, start_date: pd.Timestamp, end_date: pd.Timestamp) -> Optional[pd.DataFrame]:
    inferred = PRICE_DIR / category / f"{ticker}_{start_date.strftime('%Y-%m-%d')}_{end_date.strftime('%Y-%m-%d')}.csv"
    if inferred.exists():
        df = pd.read_csv(inferred, parse_dates=["date"])
        return df

    # Search for longest matching file if default naming not found
    category_dir = PRICE_DIR / category
    if not category_dir.exists():
        return None
    candidates = list(category_dir.glob(f"{ticker}_*.csv"))
    if not candidates:
        return None

    # Pick the candidate with the latest end date
    selected_file = max(candidates, key=lambda path: path.stat().st_mtime)
    df = pd.read_csv(selected_file, parse_dates=["date"])
    return df


def _resolve_comparison(rule: Dict, df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    indicator_series = df[rule["indicator"]]
    compare = rule.get("compare", {})
    if compare.get("type") == "indicator":
        target_series = df[compare.get("value")]
    else:
        target_value = float(compare.get("value", 0))
        target_series = pd.Series(target_value, index=df.index)
    return indicator_series, target_series


def _evaluate_rule(rule: Dict, df: pd.DataFrame) -> pd.Series:
    lhs, rhs = _resolve_comparison(rule, df)
    operator = rule["operator"]

    if operator == ">":
        return lhs > rhs
    if operator == ">=":
        return lhs >= rhs
    if operator == "<":
        return lhs < rhs
    if operator == "<=":
        return lhs <= rhs
    if operator == "equal to":
        return lhs == rhs
    if operator == "crosses above":
        prev_diff = lhs.shift(1) - rhs.shift(1)
        current_diff = lhs - rhs
        return (current_diff > 0) & (prev_diff <= 0)
    if operator == "crosses below":
        prev_diff = lhs.shift(1) - rhs.shift(1)
        current_diff = lhs - rhs
        return (current_diff < 0) & (prev_diff >= 0)

    raise ValueError(f"Unsupported operator: {operator}")


def _build_signal(df: pd.DataFrame, rules: Iterable[Dict]) -> pd.Series:
    if not rules:
        return pd.Series(False, index=df.index)
    signals = [_evaluate_rule(rule, df).fillna(False) for rule in rules]
    combined = pd.Series(True, index=df.index)
    for signal in signals:
        combined &= signal
    return combined


def _run_gold_allocation(start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> Optional[BacktestResult]:
    invest_capital = INITIAL_CAPITAL * BASE_GOLD_TARGET
    if invest_capital <= 0:
        return None

    gold_df = _load_price_data("hgold", GOLD_TICKER, start_ts, end_ts)
    if gold_df is None or gold_df.empty:
        st.warning("Gold price data unavailable; skipping gold allocation.")
        return None

    gold_df = gold_df[(gold_df["date"] >= start_ts) & (gold_df["date"] <= end_ts)].sort_values("date")
    if gold_df.empty:
        st.warning("Gold price data missing in the backtesting window; skipping gold allocation.")
        return None

    first_price = gold_df["close"].iloc[0]
    if pd.isna(first_price) or first_price <= 0:
        st.warning("Invalid first price for GLD; skipping gold allocation.")
        return None

    shares = np.floor(invest_capital / first_price)
    invested_amount = shares * first_price
    residual_cash = invest_capital - invested_amount
    cash_reserve = INITIAL_CAPITAL * (ALLOCATION_MAP["hgold"] - BASE_GOLD_TARGET)
    total_cash = residual_cash + cash_reserve

    equity_values = shares * gold_df["close"] + total_cash
    equity_curve = pd.DataFrame({"date": gold_df["date"], "equity": equity_values})
    ending_value = equity_curve["equity"].iloc[-1]
    start_value = invest_capital + cash_reserve
    return_pct = (ending_value - start_value) / start_value * 100 if start_value else 0.0

    return BacktestResult(
        ticker=GOLD_TICKER,
        category="hgold",
        trades=0,
        starting_capital=start_value,
        ending_value=ending_value,
        return_pct=return_pct,
        equity_curve=equity_curve,
        buy_rules=[],
        sell_rules=[],
    )


def _run_ticker_backtest(
    df: pd.DataFrame,
    buy_rules: Iterable[Dict],
    sell_rules: Iterable[Dict],
    starting_capital: float,
    ticker: str,
    category: str,
) -> BacktestResult:
    df = df.sort_values("date").reset_index(drop=True)
    df = _compute_indicators(df)
    df = df.dropna(subset=["close"])

    df["buy_signal"] = _build_signal(df, buy_rules)
    df["sell_signal"] = _build_signal(df, sell_rules)

    cash = starting_capital
    shares = 0.0
    equity = []
    trades = 0

    for idx, row in df.iterrows():
        price = row["close"]
        if pd.isna(price) or price <= 0:
            equity.append(cash + shares * price if not pd.isna(price) else cash)
            continue

        if shares == 0 and row["buy_signal"]:
            purchasable_shares = np.floor(cash / price)
            if purchasable_shares > 0:
                shares = purchasable_shares
                cash -= shares * price
                trades += 1
        elif shares > 0 and row["sell_signal"]:
            cash += shares * price
            shares = 0
            trades += 1

        equity.append(cash + shares * price)

    equity_curve = pd.DataFrame({"date": df["date"], "equity": equity})
    ending_value = equity_curve["equity"].iloc[-1]
    return_pct = (ending_value - starting_capital) / starting_capital * 100 if starting_capital else 0.0

    return BacktestResult(
        ticker=ticker,
        category=category,
        trades=trades,
        starting_capital=starting_capital,
        ending_value=ending_value,
        return_pct=return_pct,
        equity_curve=equity_curve,
        buy_rules=buy_rules,
        sell_rules=sell_rules,
    )


strategies = _load_strategies()
if not strategies:
    st.warning("No strategies available. Define and save a strategy first.")
    st.stop()

strategy_name = st.selectbox("Select strategy", sorted(strategies.keys()))
strategy_payload = strategies[strategy_name]
buy_rules = strategy_payload.get("buy_rules", [])
sell_rules = strategy_payload.get("sell_rules", [])

if not buy_rules and not sell_rules:
    st.warning("The selected strategy has no rules defined.")
    st.stop()

stock_index = load_stock_dataframe(fetch_prices=False)
categories_available = [
    cat
    for cat in sorted(stock_index["category"].unique())
    if cat != "hgold"  # Exclude gold from trading allocation
]

start_ts, end_ts = _load_backtesting_window()

st.sidebar.header("Simulation Settings")
st.sidebar.write(
    f"**Backtesting Window:** {start_ts.strftime('%Y-%m-%d')} → {end_ts.strftime('%Y-%m-%d')}"
)
run_button = st.sidebar.button("Run Backtest", type="primary")

if not run_button:
    st.info("Select a strategy and press **Run Backtest** to execute.")
    st.stop()

results: Dict[str, Dict[str, BacktestResult]] = {}
aggregate_equity = []

for category in categories_available:
    category_label = CATEGORY_LABELS.get(category, category)
    allocation = INITIAL_CAPITAL * ALLOCATION_MAP.get(category, 0.0)
    category_tickers = sorted(stock_index[stock_index["category"] == category]["ticker"].unique())
    if not category_tickers:
        st.warning(f"No tickers available for {category_label}. Skipping.")
        continue

    per_ticker_capital = allocation / len(category_tickers)
    results[category] = {}

    for ticker in category_tickers:
        price_df = _load_price_data(category, ticker, start_ts, end_ts)
        if price_df is None or price_df.empty:
            st.warning(f"No price data for {ticker} in {category_label}. Skipping ticker.")
            continue

        price_df = price_df[(price_df["date"] >= start_ts) & (price_df["date"] <= end_ts)]
        if price_df.empty:
            st.warning(f"No price data for {ticker} within the selected range. Skipping ticker.")
            continue

        result = _run_ticker_backtest(
            price_df,
            buy_rules=buy_rules,
            sell_rules=sell_rules,
            starting_capital=per_ticker_capital,
            ticker=ticker,
            category=category,
        )

        results[category][ticker] = result
        equity_curve = result.equity_curve.copy()
        equity_curve["category"] = category_label
        aggregate_equity.append(equity_curve)

gold_result = _run_gold_allocation(start_ts, end_ts)
if gold_result:
    results.setdefault("hgold", {})[GOLD_TICKER] = gold_result
    gold_curve = gold_result.equity_curve.copy()
    gold_curve["category"] = CATEGORY_LABELS.get("hgold", "Gold Hedge ETF")
    aggregate_equity.append(gold_curve)

if not results or not aggregate_equity:
    st.error("Backtest produced no results. Ensure price data exists for the selected period.")
    st.stop()

combined_equity = pd.concat(aggregate_equity, ignore_index=True)
category_equity = (
    combined_equity.groupby(["date", "category"])["equity"].sum().reset_index()
)
total_equity = (
    combined_equity.groupby("date")["equity"].sum().reset_index().rename(columns={"equity": "total_equity"})
)

col_summary, col_chart = st.columns([0.35, 0.65])

with col_summary:
    st.subheader("Portfolio Summary")
    total_end_value = total_equity["total_equity"].iloc[-1]
    total_return = (total_end_value - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
    sharpe_ratio = _calculate_sharpe_ratio(total_equity["total_equity"])
    max_drawdown = _calculate_max_drawdown(total_equity["total_equity"])
    volatility = _calculate_volatility(total_equity["total_equity"])
    st.metric("Ending Portfolio Value", f"${total_end_value:,.2f}")
    st.metric("Total Return", f"{total_return:,.2f}%")
    st.metric("Sharpe Ratio", "—" if pd.isna(sharpe_ratio) else f"{sharpe_ratio:,.2f}")
    st.metric("Max Drawdown", f"{max_drawdown:.2%}" if not pd.isna(max_drawdown) else "—")
    st.metric("Annualized Volatility", f"{volatility:,.2%}" if not pd.isna(volatility) else "—")

    for category, category_results in results.items():
        category_label = CATEGORY_LABELS.get(category, category)
        category_allocation = INITIAL_CAPITAL * ALLOCATION_MAP.get(category, 0.0)
        category_end = sum(res.ending_value for res in category_results.values())
        category_return = (category_end - category_allocation) / category_allocation * 100 if category_allocation else 0.0
        st.markdown(f"- **{category_label}** — Ending Value: `${category_end:,.2f}` ({category_return:,.2f}%)")

with col_chart:
    st.subheader("Equity Curve by Bucket")
    equity_chart = px.line(
        category_equity,
        x="date",
        y="equity",
        color="category",
        labels={"date": "Date", "equity": "Equity", "category": "Bucket"},
        title=f"Equity Curves — Strategy: {strategy_name}",
    )
    st.plotly_chart(equity_chart, use_container_width=True)

st.subheader("Per-Ticker Results")
rows = []
for category, category_results in results.items():
    for ticker, result in category_results.items():
        rows.append(
            {
                "Category": CATEGORY_LABELS.get(category, category),
                "Ticker": ticker,
                "Trades": result.trades,
                "Start Capital": result.starting_capital,
                "Ending Value": result.ending_value,
                "Return %": result.return_pct,
            }
        )

summary_df = pd.DataFrame(rows)
summary_df["Start Capital"] = summary_df["Start Capital"].map(lambda x: f"${x:,.2f}")
summary_df["Ending Value"] = summary_df["Ending Value"].map(lambda x: f"${x:,.2f}")
summary_df["Return %"] = summary_df["Return %"].map(lambda x: f"{x:,.2f}%")

st.dataframe(summary_df, use_container_width=True)

with st.expander("Strategy Rules"):
    st.markdown("**Buy Rules**")
    if buy_rules:
        for idx, rule in enumerate(buy_rules, start=1):
            st.write(f"{idx}. {rule}")
    else:
        st.write("None")

    st.markdown("**Sell Rules**")
    if sell_rules:
        for idx, rule in enumerate(sell_rules, start=1):
            st.write(f"{idx}. {rule}")
    else:
        st.write("None")
