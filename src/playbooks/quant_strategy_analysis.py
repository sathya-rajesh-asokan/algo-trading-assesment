"""
Quantitative investment strategy analysis playbook.

Runs a set of pre-selected strategies, prints their performance metrics,
and renders equity charts for quick visual inspection.
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

STRATEGIES_TO_EVALUATE = (
    "Ideal Strategy",
    "Momentum Confirmation",
    "RSI Pullback",
)
SELECTED_TICKERS = {
    "hreturn": ("AAPL", "MSFT", "NVDA"),
    "hdividend": ("JNJ", "PG", "KO"),
}


def build_report(
    strategy_name: str,
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> pd.DataFrame:
    portfolio = utils.run_strategy_portfolio(
        strategy_name,
        start=pd.to_datetime(start) if start else None,
        end=pd.to_datetime(end) if end else None,
        tickers_by_category=SELECTED_TICKERS,
    )

    rows = []
    for item in portfolio.ticker_results:
        rows.append(
            {
                "category": item.category,
                "ticker": item.ticker,
                "trades": item.trades,
                "start_capital": round(item.starting_capital, 2),
                "ending_value": round(item.ending_value, 2),
                "return_pct": round(item.return_pct, 2),
                "sharpe": round(item.sharpe, 3) if pd.notna(item.sharpe) else float("nan"),
                "max_drawdown": round(item.max_drawdown, 4) if pd.notna(item.max_drawdown) else float("nan"),
                "volatility": round(item.volatility, 4) if pd.notna(item.volatility) else float("nan"),
            }
        )

    detail_df = pd.DataFrame(rows).sort_values(["category", "ticker"])

    summary = pd.DataFrame(
        [
            {
                "metric": "total_return_pct",
                "value": round(portfolio.metrics["total_return_pct"], 2),
            },
            {
                "metric": "sharpe_ratio",
                "value": round(portfolio.metrics["sharpe_ratio"], 3)
                if pd.notna(portfolio.metrics["sharpe_ratio"])
                else float("nan"),
            },
            {
                "metric": "max_drawdown",
                "value": round(portfolio.metrics["max_drawdown"], 4)
                if pd.notna(portfolio.metrics["max_drawdown"])
                else float("nan"),
            },
            {
                "metric": "volatility",
                "value": round(portfolio.metrics["volatility"], 4)
                if pd.notna(portfolio.metrics["volatility"])
                else float("nan"),
            },
        ]
    )

    print(f"\n=== {portfolio.strategy} ===")
    print(f"Window: {portfolio.start.date()} → {portfolio.end.date()}")
    print("\nPortfolio summary:")
    print(summary.to_string(index=False))
    print("\nPer-ticker stats:")
    print(detail_df.to_string(index=False))

    equity_fig = px.line(
        portfolio.portfolio_equity,
        x="date",
        y="equity",
        title=f"{portfolio.strategy} – Portfolio Equity",
    )
    equity_fig.show()

    category_fig = px.area(
        portfolio.category_equity,
        x="date",
        y="equity",
        color="category",
        title=f"{portfolio.strategy} – Category Contribution",
    )
    category_fig.show()

    return detail_df


def run_batch(strategies: Iterable[str]) -> None:
    for name in strategies:
        try:
            build_report(name)
        except Exception as exc:  # pragma: no cover - diagnostic output
            print(f"[WARN] Failed to analyse '{name}': {exc}")


if __name__ == "__main__":
    run_batch(STRATEGIES_TO_EVALUATE)
