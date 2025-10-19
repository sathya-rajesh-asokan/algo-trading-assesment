"""
Shared utilities for offline strategy playbooks.

These helpers mirror the core trading logic from the Streamlit pages so the
same strategies can be evaluated from scripts or notebooks without launching
the web app.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, MutableMapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from scripts.load_file_stocks import load_stock_dataframe

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
PRICE_DIR = DATA_DIR / "prices"
STRATEGY_FILE = DATA_DIR / "strategies.json"
DATE_RANGES_FILE = DATA_DIR / "date_ranges.json"

INITIAL_CAPITAL = 1_000_000.0
ALLOCATION_MAP = {
    "hreturn": 0.60,
    "hdividend": 0.25,
    "hgold": 0.15,
}
BASE_GOLD_TARGET = 0.10
GOLD_TICKER = "GLD"

try:
    import talib
except ImportError:  # pragma: no cover
    talib = None


@dataclass
class TickerResult:
    ticker: str
    category: str
    trades: int
    starting_capital: float
    ending_value: float
    return_pct: float
    equity_curve: pd.DataFrame
    buy_and_hold_curve: pd.DataFrame
    sharpe: float
    max_drawdown: float
    volatility: float


@dataclass
class PortfolioResult:
    strategy: str
    start: pd.Timestamp
    end: pd.Timestamp
    initial_capital: float
    ticker_results: List[TickerResult]
    portfolio_equity: pd.DataFrame
    category_equity: pd.DataFrame
    metrics: Dict[str, float]


def _load_json(path: Path) -> MutableMapping[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _resolve_date_window(
    start: Optional[pd.Timestamp] = None,
    end: Optional[pd.Timestamp] = None,
) -> Tuple[pd.Timestamp, pd.Timestamp]:
    if start is not None and end is not None:
        return pd.to_datetime(start), pd.to_datetime(end)
    if not DATE_RANGES_FILE.exists():
        raise FileNotFoundError("date_ranges.json not found; supply start/end dates.")
    payload = _load_json(DATE_RANGES_FILE)
    window = payload.get("backtesting_window") or payload.get("trade_simulation_window")
    if not isinstance(window, dict):
        raise ValueError("No usable window found in date_ranges.json.")
    try:
        resolved_start = pd.to_datetime(window["start"])
        resolved_end = pd.to_datetime(window["end"])
    except (KeyError, TypeError, ValueError) as exc:  # pragma: no cover - invalid config
        raise ValueError("Invalid date window inside date_ranges.json.") from exc
    if resolved_start > resolved_end:
        raise ValueError("Start date must be on or before end date.")
    return resolved_start, resolved_end


def load_strategies() -> Dict[str, Dict]:
    if not STRATEGY_FILE.exists():
        raise FileNotFoundError("strategies.json not found in data directory.")
    try:
        payload = json.loads(STRATEGY_FILE.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError("strategies.json is not valid JSON.") from exc
    if not isinstance(payload, dict):
        raise ValueError("Expected strategies.json to contain a JSON object.")
    return payload


def _latest_price_file(category: str, ticker: str) -> Optional[Path]:
    category_dir = PRICE_DIR / category
    if not category_dir.exists():
        return None
    matches = list(category_dir.glob(f"{ticker}_*.csv"))
    if not matches:
        return None
    return max(matches, key=lambda item: item.stat().st_mtime)


def load_price_data(category: str, ticker: str) -> Optional[pd.DataFrame]:
    csv_path = _latest_price_file(category, ticker)
    if csv_path is None:
        return None
    df = pd.read_csv(csv_path, parse_dates=["date"])
    return df.sort_values("date").reset_index(drop=True)


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
        macd, macd_signal, macd_hist = talib.MACD(
            close,
            fastperiod=12,
            slowperiod=26,
            signalperiod=9,
        )
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
        previous = lhs.shift(1) - rhs.shift(1)
        current = lhs - rhs
        return (current > 0) & (previous <= 0)
    if operator == "crosses below":
        previous = lhs.shift(1) - rhs.shift(1)
        current = lhs - rhs
        return (current < 0) & (previous >= 0)
    raise ValueError(f"Unsupported operator: {operator}")


def build_signal(df: pd.DataFrame, rules: Iterable[Dict]) -> pd.Series:
    if not rules:
        return pd.Series(False, index=df.index)
    signal = pd.Series(True, index=df.index)
    for rule in rules:
        signal &= _evaluate_rule(rule, df).fillna(False)
    return signal


def calculate_sharpe_ratio(equity: pd.Series, risk_free_rate: float = 0.0) -> float:
    returns = equity.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    if returns.empty:
        return float("nan")
    excess = returns - risk_free_rate / 252
    std = excess.std()
    if std == 0 or pd.isna(std):
        return float("nan")
    return (excess.mean() * 252) / std


def calculate_max_drawdown(equity: pd.Series) -> float:
    running_max = equity.cummax()
    drawdown = equity / running_max - 1.0
    return drawdown.min() if not drawdown.empty else float("nan")


def calculate_volatility(equity: pd.Series) -> float:
    returns = equity.pct_change().dropna()
    if returns.empty:
        return float("nan")
    return returns.std() * np.sqrt(252)


def _run_single_ticker_backtest(
    df: pd.DataFrame,
    buy_rules: Iterable[Dict],
    sell_rules: Iterable[Dict],
    capital: float,
    ticker: str,
    category: str,
) -> Optional[TickerResult]:
    df = df.sort_values("date").reset_index(drop=True)
    if df.empty:
        return None

    reference = df[["date", "close"]].copy()
    first_price = reference["close"].iloc[0]
    if pd.isna(first_price) or first_price <= 0:
        return None
    shares_buy_hold = capital / first_price
    buy_hold_equity = reference.copy()
    buy_hold_equity["equity"] = shares_buy_hold * buy_hold_equity["close"]

    df = _compute_indicators(df).dropna(subset=["close"])
    if df.empty:
        return None

    df["buy_signal"] = build_signal(df, buy_rules)
    df["sell_signal"] = build_signal(df, sell_rules)

    cash = capital
    shares = 0.0
    equity = []
    trades = 0

    for _, row in df.iterrows():
        price = row["close"]
        if pd.isna(price) or price <= 0:
            equity.append(cash)
            continue

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

    equity_curve = pd.DataFrame({"date": df["date"], "equity": equity})
    buy_hold_equity = buy_hold_equity.merge(
        equity_curve[["date"]], on="date", how="inner"
    )
    ending_value = equity_curve["equity"].iloc[-1]
    return_pct = (
        (ending_value - capital) / capital * 100 if capital else float("nan")
    )

    sharpe = calculate_sharpe_ratio(equity_curve["equity"])
    max_dd = calculate_max_drawdown(equity_curve["equity"])
    volatility = calculate_volatility(equity_curve["equity"])

    return TickerResult(
        ticker=ticker,
        category=category,
        trades=trades,
        starting_capital=capital,
        ending_value=ending_value,
        return_pct=return_pct,
        equity_curve=equity_curve,
        buy_and_hold_curve=buy_hold_equity[["date", "equity"]],
        sharpe=sharpe,
        max_drawdown=max_dd,
        volatility=volatility,
    )


def _run_gold_allocation(start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> Optional[TickerResult]:
    invest_capital = INITIAL_CAPITAL * BASE_GOLD_TARGET
    if invest_capital <= 0:
        return None

    gold_df = load_price_data("hgold", GOLD_TICKER)
    if gold_df is None or gold_df.empty:
        return None

    gold_df = gold_df[(gold_df["date"] >= start_ts) & (gold_df["date"] <= end_ts)]
    if gold_df.empty:
        return None

    gold_df = gold_df.sort_values("date").reset_index(drop=True)
    first_price = gold_df["close"].iloc[0]
    if pd.isna(first_price) or first_price <= 0:
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
    return_pct = (
        (ending_value - start_value) / start_value * 100 if start_value else float("nan")
    )

    sharpe = calculate_sharpe_ratio(equity_curve["equity"])
    max_dd = calculate_max_drawdown(equity_curve["equity"])
    volatility = calculate_volatility(equity_curve["equity"])

    return TickerResult(
        ticker=GOLD_TICKER,
        category="hgold",
        trades=0,
        starting_capital=start_value,
        ending_value=ending_value,
        return_pct=return_pct,
        equity_curve=equity_curve,
        buy_and_hold_curve=equity_curve.copy(),
        sharpe=sharpe,
        max_drawdown=max_dd,
        volatility=volatility,
    )


def run_strategy_portfolio(
    strategy_name: str,
    *,
    start: Optional[pd.Timestamp] = None,
    end: Optional[pd.Timestamp] = None,
    tickers_by_category: Optional[Dict[str, Sequence[str]]] = None,
) -> PortfolioResult:
    strategies = load_strategies()
    if strategy_name not in strategies:
        raise KeyError(f"Strategy '{strategy_name}' not found in strategies.json.")
    payload = strategies[strategy_name]
    buy_rules = payload.get("buy_rules", [])
    sell_rules = payload.get("sell_rules", [])

    start_ts, end_ts = _resolve_date_window(start, end)

    stock_index = load_stock_dataframe(fetch_prices=False)
    if stock_index.empty:
        raise ValueError("No stock metadata found; run scripts/load_file_stocks.py first.")

    ticker_results: List[TickerResult] = []
    aggregate_curves: List[pd.DataFrame] = []

    for category in ("hreturn", "hdividend"):
        allocation = INITIAL_CAPITAL * ALLOCATION_MAP.get(category, 0.0)
        if allocation <= 0:
            continue

        category_mask = stock_index["category"] == category
        tickers = sorted(stock_index[category_mask]["ticker"].unique())
        if tickers_by_category and category in tickers_by_category:
            allow = {ticker.upper() for ticker in tickers_by_category[category]}
            tickers = [ticker for ticker in tickers if ticker.upper() in allow]
        if not tickers:
            continue

        per_ticker_capital = allocation / len(tickers)

        for ticker in tickers:
            price_df = load_price_data(category, ticker)
            if price_df is None or price_df.empty:
                continue

            windowed = price_df[
                (price_df["date"] >= start_ts) & (price_df["date"] <= end_ts)
            ]
            if windowed.empty:
                continue

            result = _run_single_ticker_backtest(
                windowed,
                buy_rules=buy_rules,
                sell_rules=sell_rules,
                capital=per_ticker_capital,
                ticker=ticker,
                category=category,
            )
            if result is None:
                continue

            ticker_results.append(result)
            curve = result.equity_curve.copy()
            curve["category"] = category
            aggregate_curves.append(curve)

    gold_result = _run_gold_allocation(start_ts, end_ts)
    if gold_result is not None:
        ticker_results.append(gold_result)
        curve = gold_result.equity_curve.copy()
        curve["category"] = gold_result.category
        aggregate_curves.append(curve)

    if not aggregate_curves:
        raise ValueError("No equity curves generated; ensure price data exists for the window.")

    combined_equity = pd.concat(aggregate_curves, ignore_index=True)
    combined_equity = combined_equity.sort_values("date")

    category_equity = (
        combined_equity.groupby(["date", "category"])["equity"].sum().reset_index()
    )
    portfolio_equity = (
        combined_equity.groupby("date")["equity"].sum().reset_index(name="equity")
    )

    portfolio_metrics = {
        "total_return_pct": (
            (portfolio_equity["equity"].iloc[-1] - INITIAL_CAPITAL)
            / INITIAL_CAPITAL
            * 100
            if not portfolio_equity.empty
            else float("nan")
        ),
        "sharpe_ratio": calculate_sharpe_ratio(portfolio_equity["equity"]),
        "max_drawdown": calculate_max_drawdown(portfolio_equity["equity"]),
        "volatility": calculate_volatility(portfolio_equity["equity"]),
    }

    return PortfolioResult(
        strategy=strategy_name,
        start=start_ts,
        end=end_ts,
        initial_capital=INITIAL_CAPITAL,
        ticker_results=ticker_results,
        portfolio_equity=portfolio_equity,
        category_equity=category_equity,
        metrics=portfolio_metrics,
    )


def build_buy_and_hold_baseline(
    tickers: Iterable[TickerResult],
) -> pd.DataFrame:
    """Construct an equal-weight buy-and-hold curve using each ticker's equity curve."""
    if not tickers:
        raise ValueError("Ticker list cannot be empty for buy-and-hold baseline.")
    per_curve = []
    for result in tickers:
        curve = result.buy_and_hold_curve.copy()
        curve["category"] = result.category
        per_curve.append(curve)
    combined = pd.concat(per_curve, ignore_index=True)
    combined = combined.sort_values("date")
    baseline = (
        combined.groupby("date")["equity"].sum().reset_index(name="equity")
    )
    baseline["label"] = "buy_and_hold"
    return baseline


def load_benchmark_series(
    ticker: str = "SPY",
    *,
    start: Optional[pd.Timestamp] = None,
    end: Optional[pd.Timestamp] = None,
    initial_capital: float = INITIAL_CAPITAL,
) -> pd.DataFrame:
    """
    Load a benchmark price series (e.g., SPY) and convert it to an equity curve.

    The function searches for CSV files named ``{ticker}_*.csv`` inside ``data/prices``.
    """
    candidates = list(PRICE_DIR.rglob(f"{ticker}_*.csv"))
    if not candidates:
        raise FileNotFoundError(
            f"No price file found for benchmark ticker '{ticker}'. "
            "Place a CSV named '{ticker}_<dates>.csv' under data/prices/benchmarks or "
            "any subdirectory."
        )
    csv_path = max(candidates, key=lambda item: item.stat().st_mtime)
    df = pd.read_csv(csv_path, parse_dates=["date"]).sort_values("date")
    if start is not None:
        df = df[df["date"] >= pd.to_datetime(start)]
    if end is not None:
        df = df[df["date"] <= pd.to_datetime(end)]
    df = df.reset_index(drop=True)
    if df.empty:
        raise ValueError(f"Benchmark data for {ticker} is empty over the requested window.")

    first_price = df["close"].iloc[0]
    if pd.isna(first_price) or first_price <= 0:
        raise ValueError(f"Invalid first benchmark price for {ticker}.")
    shares = np.floor(initial_capital / first_price)
    cash = initial_capital - shares * first_price
    df["equity"] = shares * df["close"] + cash
    df = df[["date", "equity"]]
    df["label"] = ticker
    return df


def summarize_equity(equity: pd.Series) -> Dict[str, float]:
    return {
        "ending_value": equity.iloc[-1] if not equity.empty else float("nan"),
        "total_return_pct": (
            (equity.iloc[-1] - equity.iloc[0]) / equity.iloc[0] * 100 if equity.size >= 2 else float("nan")
        ),
        "sharpe_ratio": calculate_sharpe_ratio(equity),
        "max_drawdown": calculate_max_drawdown(equity),
        "volatility": calculate_volatility(equity),
    }
