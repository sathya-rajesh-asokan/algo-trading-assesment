"""
Trade simulation page.

Simulates live trading over the configured trade simulation window,
applying management fees, periodic rebalancing, and strategy rules.
"""

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
BASE_GOLD_TARGET = 0.10
GOLD_TICKER = "GLD"
MANAGEMENT_FEE_ANNUAL = 0.01

CATEGORY_LABELS = {
    "hreturn": "High Return / Growth",
    "hdividend": "High Dividend / Income",
    "hgold": "Gold Hedge ETF",
}


@dataclass
class SimulationResult:
    ticker: str
    category: str
    trades: int
    starting_capital: float
    ending_value: float
    return_pct: float
    equity_curve: pd.DataFrame


st.set_page_config(page_title="Trade Simulation", layout="wide")

st.title("Trade Simulation")
st.caption(
    "Simulate live trading over the trade simulation window with fees and early-month rebalancing."
)


def _load_strategies() -> Dict[str, Dict]:
    if STRATEGY_FILE.exists():
        try:
            return json.loads(STRATEGY_FILE.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            st.error("Failed to read strategies.json. Fix the JSON format and retry.")
            st.stop()
    return {}


def _load_trade_window() -> Tuple[pd.Timestamp, pd.Timestamp]:
    if not DATE_RANGES_FILE.exists():
        st.error("date_ranges.json not found. Configure date ranges before running the simulation.")
        st.stop()
    try:
        payload = json.loads(DATE_RANGES_FILE.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        st.error("Could not parse date_ranges.json. Ensure it is valid JSON.")
        st.stop()
    window = payload.get("trade_simulation_window")
    if not window:
        st.error("trade_simulation_window is missing in date_ranges.json.")
        st.stop()
    try:
        start = pd.to_datetime(window["start"])
        end = pd.to_datetime(window["end"])
    except (KeyError, TypeError, ValueError):
        st.error("Invalid trade_simulation_window dates in date_ranges.json.")
        st.stop()
    if start > end:
        st.error("trade_simulation_window start date must be before end date.")
        st.stop()
    return start, end


def _load_price_data(category: str, ticker: str) -> Optional[pd.DataFrame]:
    category_dir = PRICE_DIR / category
    if not category_dir.exists():
        return None
    candidates = list(category_dir.glob(f"{ticker}_*.csv"))
    if not candidates:
        return None
    selected = max(candidates, key=lambda path: path.stat().st_mtime)
    df = pd.read_csv(selected, parse_dates=["date"])
    return df


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
        working["RSI14"] = talib.RSI(close, timeperiod=14)
        macd, macd_signal, macd_hist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
        working["MACD"] = macd
        working["MACD_signal"] = macd_signal
        working["MACD_hist"] = macd_hist
    else:  # pragma: no cover
        working["SMA20"] = working["close"].rolling(window=20, min_periods=1).mean()
        working["SMA50"] = working["close"].rolling(window=50, min_periods=1).mean()
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

    return working


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


def _resolve_rule(rule: Dict, df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    lhs = df[rule["indicator"]]
    compare = rule.get("compare", {})
    if compare.get("type") == "indicator":
        rhs = df[compare.get("value")]
    else:
        rhs = pd.Series(float(compare.get("value", 0.0)), index=df.index)
    return lhs, rhs


def _evaluate_rule(rule: Dict, df: pd.DataFrame) -> pd.Series:
    lhs, rhs = _resolve_rule(rule, df)
    op = rule["operator"]
    if op == ">":
        return lhs > rhs
    if op == ">=":
        return lhs >= rhs
    if op == "<":
        return lhs < rhs
    if op == "<=":
        return lhs <= rhs
    if op == "equal to":
        return lhs == rhs
    if op == "crosses above":
        prev_diff = lhs.shift(1) - rhs.shift(1)
        diff = lhs - rhs
        return (diff > 0) & (prev_diff <= 0)
    if op == "crosses below":
        prev_diff = lhs.shift(1) - rhs.shift(1)
        diff = lhs - rhs
        return (diff < 0) & (prev_diff >= 0)
    raise ValueError(f"Unsupported operator: {op}")


def _build_signal(df: pd.DataFrame, rules: Iterable[Dict]) -> pd.Series:
    if not rules:
        return pd.Series(False, index=df.index)
    signal = pd.Series(True, index=df.index)
    for rule in rules:
        signal &= _evaluate_rule(rule, df).fillna(False)
    return signal


def _apply_management_fee(equity_curve: pd.Series) -> pd.Series:
    annual_fee = MANAGEMENT_FEE_ANNUAL
    daily_rate = annual_fee / 252
    factor = (1 - daily_rate) ** np.arange(len(equity_curve))
    return equity_curve * factor


def _build_rebalance_signal(dates: pd.Series) -> pd.Series:
    month_group = dates.dt.to_period("M")
    return dates.groupby(month_group).apply(lambda grp: (grp.index - grp.index.min()) < 3)


def _simulate_equity_bucket(
    df: pd.DataFrame,
    buy_rules: Iterable[Dict],
    sell_rules: Iterable[Dict],
    capital: float,
    rebalance_signal: pd.Series,
) -> Tuple[pd.DataFrame, int]:
    df = df.sort_values("date").reset_index(drop=True)
    df = _compute_indicators(df)
    df = df.dropna(subset=["close"])

    df["buy_signal"] = _build_signal(df, buy_rules)
    df["sell_signal"] = _build_signal(df, sell_rules)
    df["rebalance"] = rebalance_signal.reindex(df.index, fill_value=False)

    cash = capital
    shares = 0.0
    trades = 0
    equity = []

    for idx, row in df.iterrows():
        price = row["close"]
        if pd.isna(price) or price <= 0:
            equity.append(cash + shares * price if not pd.isna(price) else cash)
            continue

        if row["rebalance"] and shares > 0:
            cash += shares * price
            shares = 0.0

        if shares == 0 and row["buy_signal"]:
            purchasable = np.floor(cash / price)
            if purchasable > 0:
                shares = purchasable
                cash -= shares * price
                trades += 1
        elif shares > 0 and row["sell_signal"]:
            cash += shares * price
            shares = 0.0
            trades += 1

        equity.append(cash + shares * price)

    curve = pd.DataFrame({"date": df["date"], "equity": equity})
    return curve, trades


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

start_ts, end_ts = _load_trade_window()
st.sidebar.header("Simulation Window")
st.sidebar.write(f"{start_ts.strftime('%Y-%m-%d')} → {end_ts.strftime('%Y-%m-%d')}")

if not st.sidebar.button("Run Simulation", type="primary"):
    st.info("Press **Run Simulation** to begin.")
    st.stop()

stock_index = load_stock_dataframe(fetch_prices=False)
results: Dict[str, Dict[str, SimulationResult]] = {}
aggregate_curves = []

for category in ("hreturn", "hdividend", "hgold"):
    allocation = INITIAL_CAPITAL * ALLOCATION_MAP.get(category, 0.0)
    if allocation <= 0:
        continue

    category_tickers = sorted(stock_index[stock_index["category"] == category]["ticker"].unique())
    if not category_tickers:
        continue

    if category == "hgold":
        invest_capital = INITIAL_CAPITAL * BASE_GOLD_TARGET
        cash_reserve = INITIAL_CAPITAL * (ALLOCATION_MAP["hgold"] - BASE_GOLD_TARGET)
        per_ticker_capital = invest_capital / len(category_tickers)
    else:
        per_ticker_capital = allocation / len(category_tickers)
        cash_reserve = 0.0

    results.setdefault(category, {})

    for ticker in category_tickers:
        price_df = _load_price_data(category, ticker)
        if price_df is None or price_df.empty:
            st.warning(f"No price data for {ticker} ({CATEGORY_LABELS[category]}). Skipping.")
            continue

        price_df = price_df[(price_df["date"] >= start_ts) & (price_df["date"] <= end_ts)].sort_values("date")
        if price_df.empty:
            st.warning(f"No price data for {ticker} within the simulation window. Skipping.")
            continue

        rebalance_signal = _build_rebalance_signal(price_df["date"]).reindex(price_df.index, fill_value=False)
        equity_curve, trades = _simulate_equity_bucket(
            price_df,
            buy_rules=buy_rules,
            sell_rules=sell_rules,
            capital=per_ticker_capital,
            rebalance_signal=rebalance_signal,
        )

        starting_capital = per_ticker_capital + cash_reserve
        if cash_reserve:
            equity_curve["equity"] += cash_reserve

        equity_curve["equity"] = _apply_management_fee(equity_curve["equity"])

        ending_value = equity_curve["equity"].iloc[-1]
        return_pct = (ending_value - starting_capital) / starting_capital * 100 if starting_capital else 0.0

        results[category][ticker] = SimulationResult(
            ticker=ticker,
            category=category,
            trades=trades,
            starting_capital=starting_capital,
            ending_value=ending_value,
            return_pct=return_pct,
            equity_curve=equity_curve,
        )

        curve = equity_curve.copy()
        curve["category"] = CATEGORY_LABELS[category]
        aggregate_curves.append(curve)

if not aggregate_curves:
    st.error("Simulation produced no results. Ensure price data exists for the trade window.")
    st.stop()

combined_equity = pd.concat(aggregate_curves, ignore_index=True)
bucket_equity = combined_equity.groupby(["date", "category"])["equity"].sum().reset_index()
portfolio_equity = combined_equity.groupby("date")["equity"].sum().reset_index().rename(columns={"equity": "total_equity"})

baseline_curves = []
baseline_summaries = []

for category in ("hreturn", "hdividend", "hgold"):
    allocation = INITIAL_CAPITAL * ALLOCATION_MAP.get(category, 0.0)
    if allocation <= 0:
        continue

    tickers = sorted(stock_index[stock_index["category"] == category]["ticker"].unique())
    if not tickers:
        continue

    per_ticker_alloc = allocation / len(tickers)
    ticker_curves = []
    for ticker in tickers:
        price_df = _load_price_data(category, ticker)
        if price_df is None or price_df.empty:
            continue
        price_df = price_df[(price_df["date"] >= start_ts) & (price_df["date"] <= end_ts)].sort_values("date")
        if price_df.empty:
            continue
        first_price = price_df["close"].iloc[0]
        if pd.isna(first_price) or first_price <= 0:
            continue
        shares = per_ticker_alloc / first_price
        values = shares * price_df["close"]
        ticker_curves.append(pd.DataFrame({"date": price_df["date"], "value": values}))

    if not ticker_curves:
        continue

    bucket_curve = pd.concat(ticker_curves).groupby("date")["value"].sum().reset_index()
    bucket_curve = bucket_curve.sort_values("date").reset_index(drop=True)
    bucket_curve["category"] = CATEGORY_LABELS.get(category, category)
    baseline_curves.append(bucket_curve)

    end_value = bucket_curve["value"].iloc[-1]
    baseline_summaries.append(
        {
            "Category": CATEGORY_LABELS.get(category, category),
            "Start Capital": allocation,
            "Ending Value": end_value,
            "Return %": (end_value - allocation) / allocation * 100 if allocation else 0.0,
        }
    )

if baseline_curves:
    baseline_total = pd.concat(baseline_curves).groupby("date")["value"].sum().reset_index().rename(columns={"value": "baseline_total"})
else:
    baseline_total = pd.DataFrame()

col_summary, col_chart = st.columns([0.35, 0.65])

with col_summary:
    st.subheader("Portfolio Summary")
    end_value = portfolio_equity["total_equity"].iloc[-1]
    total_return = (end_value - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
    sharpe_ratio = _calculate_sharpe_ratio(portfolio_equity["total_equity"])
    max_drawdown = _calculate_max_drawdown(portfolio_equity["total_equity"])
    volatility = _calculate_volatility(portfolio_equity["total_equity"])
    st.metric("Ending Portfolio Value", f"${end_value:,.2f}")
    st.metric("Total Return (Net of Fees)", f"{total_return:,.2f}%")
    st.metric("Sharpe Ratio", "—" if pd.isna(sharpe_ratio) else f"{sharpe_ratio:,.2f}")
    st.metric("Max Drawdown", f"{max_drawdown:.2%}" if not pd.isna(max_drawdown) else "—")
    st.metric("Annualized Volatility", f"{volatility:,.2%}" if not pd.isna(volatility) else "—")

    for category, category_results in results.items():
        category_label = CATEGORY_LABELS.get(category, category)
        allocation = INITIAL_CAPITAL * ALLOCATION_MAP.get(category, 0.0)
        ending = sum(res.ending_value for res in category_results.values())
        category_return = (ending - allocation) / allocation * 100 if allocation else 0.0
        st.markdown(f"- **{category_label}** — Ending Value: `${ending:,.2f}` ({category_return:,.2f}%)")

with col_chart:
    st.subheader("Equity Curve by Bucket")
    equity_chart = px.line(
        bucket_equity,
        x="date",
        y="equity",
        color="category",
        labels={"date": "Date", "equity": "Equity", "category": "Bucket"},
        title=f"Simulated Equity Curves — Strategy: {strategy_name}",
    )
    st.plotly_chart(equity_chart, use_container_width=True)

st.subheader("Per-Ticker Performance")
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

if not baseline_total.empty:
    st.subheader("Buy-and-Hold Baseline (Day-1 Allocation)")
    comparison = portfolio_equity.merge(baseline_total, on="date", how="inner")
    comparison = comparison.rename(
        columns={
            "total_equity": "Active (Net Fees)",
            "baseline_total": "Buy & Hold",
        }
    )

    active_final = comparison["Active (Net Fees)"].iloc[-1]
    baseline_final = comparison["Buy & Hold"].iloc[-1]
    relative_diff = active_final - baseline_final
    relative_pct = (relative_diff / baseline_final * 100) if baseline_final else 0.0

    m1, m2, m3 = st.columns(3)
    m1.metric("Active Final Value", f"${active_final:,.2f}")
    m2.metric("Buy & Hold Final Value", f"${baseline_final:,.2f}")
    m3.metric("Active vs Baseline", f"${relative_diff:,.2f}", f"{relative_pct:,.2f}%")

    comparison_chart = px.line(
        comparison,
        x="date",
        y=["Active (Net Fees)", "Buy & Hold"],
        labels={"date": "Date", "value": "Equity"},
        title="Active Strategy vs Buy-and-Hold Baseline",
    )
    st.plotly_chart(comparison_chart, use_container_width=True)

    baseline_df = pd.DataFrame(baseline_summaries)
    baseline_df["Start Capital"] = baseline_df["Start Capital"].map(lambda x: f"${x:,.2f}")
    baseline_df["Ending Value"] = baseline_df["Ending Value"].map(lambda x: f"${x:,.2f}")
    baseline_df["Return %"] = baseline_df["Return %"].map(lambda x: f"{x:,.2f}%")
    st.dataframe(baseline_df, use_container_width=True)

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
