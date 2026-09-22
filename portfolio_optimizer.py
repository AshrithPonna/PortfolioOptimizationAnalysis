"""Modern Portfolio Theory portfolio optimizer.

Usage example:
    python portfolio_optimizer.py AAPL MSFT TSLA GOOGL

The program downloads one year of daily prices, estimates annualized returns
and covariance, optimizes a long-only portfolio for maximum Sharpe ratio, and
plots the resulting efficient frontier.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import minimize


TRADING_DAYS_PER_YEAR = 252
DEFAULT_RISK_FREE_RATE = 0.02


@dataclass
class PortfolioResult:
    """Weights and annualized metrics for one portfolio."""

    weights: np.ndarray
    expected_return: float
    volatility: float
    sharpe_ratio: float


def download_prices(tickers: Sequence[str]) -> pd.DataFrame:
    """Download one year of adjusted daily closing prices for ``tickers``."""
    downloaded = yf.download(
        list(tickers),
        period="1y",
        auto_adjust=True,
        group_by="column",
        progress=False,
        threads=False,
    )

    if downloaded.empty:
        raise ValueError("No market data was returned. Check the ticker symbols and internet connection.")

    # yfinance returns a MultiIndex for multiple tickers and a regular index
    # for one ticker. In both cases, normalize the result to ticker columns.
    if isinstance(downloaded.columns, pd.MultiIndex):
        if "Close" in downloaded.columns.get_level_values(0):
            prices = downloaded["Close"]
        elif "Close" in downloaded.columns.get_level_values(1):
            prices = downloaded.xs("Close", level=1, axis=1)
        else:
            raise ValueError("The downloaded data did not contain closing prices.")
    else:
        if "Close" not in downloaded.columns:
            raise ValueError("The downloaded data did not contain closing prices.")
        prices = downloaded[["Close"]].rename(columns={"Close": tickers[0]})

    prices = prices.reindex(columns=list(tickers))
    missing = [ticker for ticker in tickers if ticker not in prices.columns or prices[ticker].dropna().empty]
    if missing:
        raise ValueError("No usable price history was found for: " + ", ".join(missing))

    # Forward-fill isolated missing observations, then remove dates that still
    # cannot be used. Requiring a reasonable history avoids unstable estimates.
    prices = prices.ffill().dropna()
    if len(prices) < 60:
        raise ValueError("Fewer than 60 trading days of complete data were available.")
    return prices


def estimate_inputs(prices: pd.DataFrame) -> Tuple[pd.Series, pd.DataFrame]:
    """Estimate annualized expected returns and covariance from daily prices."""
    daily_returns = prices.pct_change().dropna()
    expected_returns = daily_returns.mean() * TRADING_DAYS_PER_YEAR
    covariance = daily_returns.cov() * TRADING_DAYS_PER_YEAR
    return expected_returns, covariance


def portfolio_metrics(
    weights: np.ndarray,
    expected_returns: pd.Series,
    covariance: pd.DataFrame,
    risk_free_rate: float,
) -> Tuple[float, float, float]:
    """Return annualized expected return, volatility, and Sharpe ratio."""
    expected_return = float(weights @ expected_returns.to_numpy())
    variance = float(weights @ covariance.to_numpy() @ weights)
    volatility = float(np.sqrt(max(variance, 0.0)))
    sharpe_ratio = (expected_return - risk_free_rate) / volatility if volatility else 0.0
    return expected_return, volatility, sharpe_ratio


def make_result(
    weights: np.ndarray,
    expected_returns: pd.Series,
    covariance: pd.DataFrame,
    risk_free_rate: float,
) -> PortfolioResult:
    """Build a result object from weights and estimated portfolio inputs."""
    expected_return, volatility, sharpe_ratio = portfolio_metrics(
        weights, expected_returns, covariance, risk_free_rate
    )
    return PortfolioResult(weights, expected_return, volatility, sharpe_ratio)


def optimize_portfolio(
    expected_returns: pd.Series,
    covariance: pd.DataFrame,
    risk_free_rate: float,
    objective: str,
    target_return: Optional[float] = None,
) -> PortfolioResult:
    """Optimize a long-only portfolio for Sharpe ratio or minimum variance."""
    asset_count = len(expected_returns)
    starting_weights = np.repeat(1.0 / asset_count, asset_count)
    bounds = [(0.0, 1.0)] * asset_count

    def objective_function(weights: np.ndarray) -> float:
        expected_return, volatility, sharpe_ratio = portfolio_metrics(
            weights, expected_returns, covariance, risk_free_rate
        )
        if objective == "sharpe":
            return -sharpe_ratio
        return volatility**2

    constraints = [{"type": "eq", "fun": lambda weights: np.sum(weights) - 1.0}]
    if target_return is not None:
        constraints.append(
            {
                "type": "eq",
                "fun": lambda weights: weights @ expected_returns.to_numpy() - target_return,
            }
        )

    solution = minimize(
        objective_function,
        starting_weights,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 1000, "ftol": 1e-10},
    )
    if not solution.success:
        raise RuntimeError(f"Optimization failed: {solution.message}")

    weights = np.clip(solution.x, 0.0, 1.0)
    weights /= weights.sum()
    return make_result(weights, expected_returns, covariance, risk_free_rate)


def efficient_frontier(
    expected_returns: pd.Series,
    covariance: pd.DataFrame,
    risk_free_rate: float,
    points: int = 60,
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate minimum-volatility portfolios across a range of returns."""
    minimum_variance = optimize_portfolio(
        expected_returns, covariance, risk_free_rate, objective="variance"
    )
    lowest_return = minimum_variance.expected_return
    highest_return = float(expected_returns.max())
    target_returns = np.linspace(lowest_return, highest_return, points)

    frontier_risks: List[float] = []
    frontier_returns: List[float] = []
    for target_return in target_returns:
        try:
            result = optimize_portfolio(
                expected_returns,
                covariance,
                risk_free_rate,
                objective="variance",
                target_return=float(target_return),
            )
        except RuntimeError:
            # A target near a numerical endpoint can be infeasible by a tiny
            # tolerance even though the surrounding frontier is valid.
            continue
        frontier_risks.append(result.volatility)
        frontier_returns.append(result.expected_return)

    return np.array(frontier_risks), np.array(frontier_returns)


def random_portfolios(
    expected_returns: pd.Series,
    covariance: pd.DataFrame,
    risk_free_rate: float,
    count: int,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate random long-only portfolios for the background scatter plot."""
    generator = np.random.default_rng(seed)
    weights = generator.dirichlet(np.ones(len(expected_returns)), size=count)
    metrics = np.array(
        [portfolio_metrics(row, expected_returns, covariance, risk_free_rate) for row in weights]
    )
    return metrics[:, 1], metrics[:, 0], metrics[:, 2]


def print_summary(
    tickers: Sequence[str],
    expected_returns: pd.Series,
    covariance: pd.DataFrame,
    optimal: PortfolioResult,
    minimum_variance: PortfolioResult,
    equal_weight: PortfolioResult,
) -> None:
    """Print allocations and a compact comparison table."""
    print("\nPORTFOLIO OPTIMIZATION SUMMARY")
    print("=" * 35)
    print("Annualized expected returns:")
    for ticker in tickers:
        print(f"  {ticker:<6} {expected_returns[ticker]:>8.2%}")

    print("\nMaximum Sharpe ratio allocation:")
    for ticker, weight in zip(tickers, optimal.weights):
        print(f"  {ticker}: {weight:.2%}")
    print(f"  Expected return: {optimal.expected_return:.2%}")
    print(f"  Risk (volatility): {optimal.volatility:.2%}")
    print(f"  Sharpe ratio: {optimal.sharpe_ratio:.3f}")

    print("\nPortfolio comparison:")
    print(f"{'Portfolio':<24}{'Return':>12}{'Risk':>12}{'Sharpe':>12}")
    print("-" * 60)
    rows = [
        ("Maximum Sharpe", optimal),
        ("Minimum variance", minimum_variance),
        ("Equal weight", equal_weight),
    ]
    for label, result in rows:
        print(f"{label:<24}{result.expected_return:>11.2%}{result.volatility:>11.2%}{result.sharpe_ratio:>12.3f}")

    print("\nAnnualized covariance matrix:")
    print(covariance.to_string(float_format=lambda value: f"{value: .4f}"))


def plot_results(
    tickers: Sequence[str],
    expected_returns: pd.Series,
    covariance: pd.DataFrame,
    risk_free_rate: float,
    optimal: PortfolioResult,
    minimum_variance: PortfolioResult,
    equal_weight: PortfolioResult,
) -> None:
    """Plot random portfolios, the efficient frontier, and key portfolios."""
    frontier_risks, frontier_returns = efficient_frontier(
        expected_returns, covariance, risk_free_rate
    )
    random_risks, random_returns, random_sharpes = random_portfolios(
        expected_returns, covariance, risk_free_rate, count=5000
    )

    figure, axis = plt.subplots(figsize=(10, 7))
    scatter = axis.scatter(
        random_risks,
        random_returns,
        c=random_sharpes,
        cmap="viridis",
        alpha=0.35,
        s=10,
        label="Random portfolios",
    )
    figure.colorbar(scatter, ax=axis, label="Sharpe ratio")
    axis.plot(frontier_risks, frontier_returns, color="black", linewidth=2.5, label="Efficient frontier")
    axis.scatter(
        optimal.volatility,
        optimal.expected_return,
        color="crimson",
        marker="*",
        s=220,
        label="Maximum Sharpe",
        zorder=3,
    )
    axis.scatter(
        minimum_variance.volatility,
        minimum_variance.expected_return,
        color="darkorange",
        marker="D",
        s=70,
        label="Minimum variance",
        zorder=3,
    )
    axis.scatter(
        equal_weight.volatility,
        equal_weight.expected_return,
        color="royalblue",
        marker="o",
        s=70,
        label="Equal weight",
        zorder=3,
    )

    axis.set_title(f"Efficient Frontier: {', '.join(tickers)}")
    axis.set_xlabel("Annualized risk (volatility)")
    axis.set_ylabel("Annualized expected return")
    axis.xaxis.set_major_formatter(plt.FuncFormatter(lambda value, _: f"{value:.0%}"))
    axis.yaxis.set_major_formatter(plt.FuncFormatter(lambda value, _: f"{value:.0%}"))
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    plt.show()


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Optimize a long-only stock portfolio using Modern Portfolio Theory."
    )
    parser.add_argument("tickers", nargs="+", help="One or more stock tickers, such as AAPL MSFT TSLA GOOGL")
    parser.add_argument(
        "--risk-free-rate",
        type=float,
        default=DEFAULT_RISK_FREE_RATE,
        help="Annual risk-free rate as a decimal (default: 0.02 for 2%%)",
    )
    return parser.parse_args()


def main() -> None:
    """Run the complete data, optimization, reporting, and plotting workflow."""
    arguments = parse_arguments()
    tickers = list(dict.fromkeys(ticker.upper() for ticker in arguments.tickers))

    if arguments.risk_free_rate < 0:
        raise ValueError("The risk-free rate cannot be negative.")

    prices = download_prices(tickers)
    expected_returns, covariance = estimate_inputs(prices)
    minimum_variance = optimize_portfolio(
        expected_returns, covariance, arguments.risk_free_rate, objective="variance"
    )
    optimal = optimize_portfolio(
        expected_returns, covariance, arguments.risk_free_rate, objective="sharpe"
    )
    equal_weight = make_result(
        np.repeat(1.0 / len(tickers), len(tickers)),
        expected_returns,
        covariance,
        arguments.risk_free_rate,
    )

    print_summary(tickers, expected_returns, covariance, optimal, minimum_variance, equal_weight)
    plot_results(
        tickers,
        expected_returns,
        covariance,
        arguments.risk_free_rate,
        optimal,
        minimum_variance,
        equal_weight,
    )


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, ValueError) as error:
        raise SystemExit(f"Error: {error}") from error