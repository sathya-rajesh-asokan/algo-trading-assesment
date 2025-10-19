"""
Performance benchmarking playbook.

Compares a set of strategies against buy-and-hold and an external benchmark
such as the S&P 500 (via SPY) and renders equity charts for the trio.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import pandas as pd
import plotly.express as px

if __package__ in (None, ""):
    import sys

    SRC_ROOT = Path(__file__).resolve().parents[1]
    if str(SRC_ROOT) not in sys.path:
        sys.path.insert(0, str(SRC_ROOT))
    import playbooks.utils as utils  # type: ignore
else:  # pragma: no cover
    from . import utils

STRATEGIES_TO_BENCHMARK = (
    "Ideal Strategy",
    "Momentum Confirmation",
    "RSI Pullback",
)
DEFAULT_BENCHMARK = "SPY"
SELECTED_TICKERS = {
    "hreturn": ("AAPL", "MSFT", "NVDA"),
    "hdividend": ("JNJ", "PG", "KO"),
}


def run_benchmark(
    strategy_name: str,
    *,
    benchmark_ticker: str = DEFAULT_BENCHMARK,
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> pd.DataFrame:
    portfolio = utils.run_strategy_portfolio(
        strategy_name,
        start=pd.to_datetime(start) if start else None,
        end=pd.to_datetime(end) if end else None,
        tickers_by_category=SELECTED_TICKERS,
    )

    strategy_curve = portfolio.portfolio_equity.copy()
    strategy_curve["label"] = strategy_name

    buy_and_hold_curve = utils.build_buy_and_hold_baseline(portfolio.ticker_results)
    benchmark_curve = utils.load_benchmark_series(
        ticker=benchmark_ticker,
        start=portfolio.start,
        end=portfolio.end,
        initial_capital=portfolio.initial_capital,
    )

    combined = pd.concat(
        [strategy_curve, buy_and_hold_curve, benchmark_curve],
        ignore_index=True,
    )
    pivot = combined.pivot(index="date", columns="label", values="equity").dropna()

    summaries = {
        label: utils.summarize_equity(pivot[label]) for label in pivot.columns
    }
    summary_df = (
        pd.DataFrame(summaries)
        .T.reset_index()
        .rename(columns={"index": "series"})
        .sort_values("series")
    )

    print(f"\n=== {portfolio.strategy} vs Benchmarks ===")
    print(f"Window: {portfolio.start.date()} → {portfolio.end.date()}")
    print("\nBenchmark comparison (aligned dates):")
    print(summary_df.to_string(index=False))

    curve_fig = px.line(
        combined,
        x="date",
        y="equity",
        color="label",
        title=f"{portfolio.strategy} vs Buy-and-Hold vs {benchmark_ticker}",
    )
    curve_fig.show()

    return pivot


def run_batch(strategies: Iterable[str]) -> None:
    for name in strategies:
        try:
            run_benchmark(name)
        except Exception as exc:  # pragma: no cover
            print(f"[WARN] Failed to benchmark '{name}': {exc}")


if __name__ == "__main__":
    run_batch(STRATEGIES_TO_BENCHMARK)
