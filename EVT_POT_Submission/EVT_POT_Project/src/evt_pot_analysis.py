"""EVT/POT tail-risk analysis for NFLX, META, INTC and the S&P 500.

The module:
1. Screens ten liquid US stocks for tail heaviness.
2. Constructs loss series as negative logarithmic returns.
3. Builds mean excess and parameter-stability diagnostics.
4. Fits the Generalized Pareto Distribution by maximum likelihood.
5. Performs a parametric-bootstrap Cramer-von Mises goodness-of-fit test.
6. Estimates EVT VaR and ES at 99% and 99.5%.
7. Performs 600-day rolling EVT-VaR backtests using Kupiec and
   Christoffersen tests over 2024-03-04 to 2026-03-03.
8. Studies threshold sensitivity at the 90th, 92nd and 95th percentiles.

All outputs are written to ../results relative to the project root.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import chi2, genpareto

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "raw"
FIG_DIR = PROJECT_ROOT / "results" / "figures"
TABLE_DIR = PROJECT_ROOT / "results" / "tables"

CANDIDATE_TICKERS = [
    "AAPL", "AMD", "AMZN", "GOOGL", "INTC",
    "META", "MSFT", "NFLX", "NVDA", "TSLA",
]
SELECTED_ASSETS = ["NFLX", "META", "INTC", "SP500"]
SELECTED_THRESHOLDS = {"NFLX": 0.92, "META": 0.95, "INTC": 0.92, "SP500": 0.92}
COMMON_START = pd.Timestamp("2021-07-28")
COMMON_END = pd.Timestamp("2026-03-03")
BACKTEST_START = pd.Timestamp("2024-03-04")
BACKTEST_WINDOW = 600
RISK_LEVELS = (0.99, 0.995)
SENSITIVITY_THRESHOLDS = (0.90, 0.92, 0.95)


@dataclass(frozen=True)
class GPDFit:
    threshold_quantile: float
    threshold: float
    n_total: int
    n_exceedances: int
    xi: float
    beta: float
    exceedances: np.ndarray


def ensure_directories() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)


def load_price_csv(path: Path) -> pd.DataFrame:
    """Load a Yahoo/yfinance-style CSV and return Date and Close columns."""
    if not path.exists():
        raise FileNotFoundError(f"Missing input file: {path}")
    df = pd.read_csv(path)
    required = {"Date", "Close"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{path.name} lacks required columns: {sorted(missing)}")
    df = df[["Date", "Close"]].copy()
    df["Date"] = pd.to_datetime(df["Date"], utc=True, errors="coerce")
    df["Date"] = df["Date"].dt.tz_convert(None).dt.normalize()
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df = df.dropna().drop_duplicates("Date").sort_values("Date")
    if (df["Close"] <= 0).any():
        raise ValueError(f"{path.name} contains non-positive closing prices")
    return df.reset_index(drop=True)


def make_loss_series(price_df: pd.DataFrame, start: pd.Timestamp | None = None,
                     end: pd.Timestamp | None = None) -> pd.DataFrame:
    df = price_df.copy()
    if start is not None:
        df = df[df["Date"] >= start]
    if end is not None:
        df = df[df["Date"] <= end]
    df = df.sort_values("Date").reset_index(drop=True)
    df["log_return"] = np.log(df["Close"] / df["Close"].shift(1))
    df["loss"] = -df["log_return"]
    return df.dropna().reset_index(drop=True)


def load_all_candidate_losses() -> dict[str, pd.DataFrame]:
    return {
        ticker: make_loss_series(load_price_csv(DATA_DIR / f"{ticker}.csv"))
        for ticker in CANDIDATE_TICKERS
    }


def load_selected_common_losses() -> dict[str, pd.DataFrame]:
    result: dict[str, pd.DataFrame] = {}
    for asset in SELECTED_ASSETS:
        file_name = "SP500.csv" if asset == "SP500" else f"{asset}.csv"
        result[asset] = make_loss_series(
            load_price_csv(DATA_DIR / file_name), COMMON_START, COMMON_END
        )
    lengths = {asset: len(df) for asset, df in result.items()}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"Selected series are not aligned: {lengths}")
    return result


def hill_estimator(losses: np.ndarray, tail_fraction: float = 0.05) -> float:
    positive = np.sort(np.asarray(losses)[np.asarray(losses) > 0])
    if len(positive) < 25:
        return float("nan")
    k = max(20, int(math.floor(tail_fraction * len(positive))))
    if len(positive) <= k:
        return float("nan")
    threshold_order_stat = positive[-k - 1]
    top = positive[-k:]
    return float(np.mean(np.log(top / threshold_order_stat)))


def fit_gpd(losses: Iterable[float], threshold_quantile: float) -> GPDFit:
    x = np.asarray(list(losses), dtype=float)
    x = x[np.isfinite(x)]
    if not (0 < threshold_quantile < 1):
        raise ValueError("threshold_quantile must be between 0 and 1")
    threshold = float(np.quantile(x, threshold_quantile))
    exceedances = x[x > threshold] - threshold
    if len(exceedances) < 20:
        raise ValueError("Too few exceedances for a reliable GPD fit")
    xi, _, beta = genpareto.fit(exceedances, floc=0)
    if beta <= 0 or not np.isfinite(beta) or not np.isfinite(xi):
        raise RuntimeError("Invalid GPD maximum-likelihood estimate")
    return GPDFit(
        threshold_quantile=float(threshold_quantile),
        threshold=threshold,
        n_total=len(x),
        n_exceedances=len(exceedances),
        xi=float(xi),
        beta=float(beta),
        exceedances=np.asarray(exceedances),
    )


def evt_var_es(fit: GPDFit, confidence: float) -> tuple[float, float]:
    """Return EVT VaR and ES for the loss distribution."""
    if not (0 < confidence < 1):
        raise ValueError("confidence must be between 0 and 1")
    exceedance_probability = fit.n_exceedances / fit.n_total
    if 1 - confidence >= exceedance_probability:
        raise ValueError("Requested quantile is below the fitted threshold region")
    ratio = exceedance_probability / (1 - confidence)
    if abs(fit.xi) < 1e-8:
        var = fit.threshold + fit.beta * math.log(ratio)
    else:
        var = fit.threshold + fit.beta / fit.xi * (ratio ** fit.xi - 1)
    if fit.xi >= 1:
        es = float("inf")
    else:
        es = (var + fit.beta - fit.xi * fit.threshold) / (1 - fit.xi)
    return float(var), float(es)


def cvm_statistic(exceedances: np.ndarray, xi: float, beta: float) -> float:
    sample = np.sort(np.asarray(exceedances, dtype=float))
    n = len(sample)
    fitted_cdf = genpareto.cdf(sample, c=xi, loc=0, scale=beta)
    order = np.arange(1, n + 1)
    return float(1 / (12 * n) + np.sum((fitted_cdf - (2 * order - 1) / (2 * n)) ** 2))


def cvm_parametric_bootstrap(fit: GPDFit, simulations: int = 500,
                             seed: int = 2026) -> tuple[float, float]:
    """CvM p-value with parameter re-estimation in every bootstrap sample."""
    rng = np.random.default_rng(seed)
    observed = cvm_statistic(fit.exceedances, fit.xi, fit.beta)
    simulated_stats: list[float] = []
    for _ in range(simulations):
        simulated = genpareto.rvs(
            c=fit.xi, loc=0, scale=fit.beta,
            size=fit.n_exceedances, random_state=rng,
        )
        try:
            xi_b, _, beta_b = genpareto.fit(simulated, floc=0)
            simulated_stats.append(cvm_statistic(simulated, float(xi_b), float(beta_b)))
        except Exception:
            continue
    if not simulated_stats:
        return observed, float("nan")
    simulated_array = np.asarray(simulated_stats)
    p_value = (1 + np.sum(simulated_array >= observed)) / (1 + len(simulated_array))
    return observed, float(p_value)


def bootstrap_fit_and_risk(fit: GPDFit, confidences: Iterable[float],
                           simulations: int = 1000, seed: int = 12345) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, float]] = []
    for _ in range(simulations):
        simulated = genpareto.rvs(
            c=fit.xi, loc=0, scale=fit.beta,
            size=fit.n_exceedances, random_state=rng,
        )
        try:
            xi_b, _, beta_b = genpareto.fit(simulated, floc=0)
            fit_b = GPDFit(
                threshold_quantile=fit.threshold_quantile,
                threshold=fit.threshold,
                n_total=fit.n_total,
                n_exceedances=fit.n_exceedances,
                xi=float(xi_b),
                beta=float(beta_b),
                exceedances=np.asarray(simulated),
            )
            row: dict[str, float] = {"xi": float(xi_b), "beta": float(beta_b)}
            for confidence in confidences:
                var, es = evt_var_es(fit_b, confidence)
                row[f"var_{confidence}"] = var
                row[f"es_{confidence}"] = es
            rows.append(row)
        except Exception:
            continue
    return pd.DataFrame(rows)


def mean_excess(losses: np.ndarray, quantiles: np.ndarray) -> pd.DataFrame:
    rows = []
    for q in quantiles:
        threshold = float(np.quantile(losses, q))
        exceedances = losses[losses > threshold] - threshold
        if len(exceedances) < 10:
            continue
        mean_value = float(np.mean(exceedances))
        standard_error = float(np.std(exceedances, ddof=1) / math.sqrt(len(exceedances)))
        rows.append({
            "quantile": float(q), "threshold": threshold,
            "mean_excess": mean_value, "se": standard_error,
            "n_exceedances": len(exceedances),
        })
    return pd.DataFrame(rows)


def screen_heavy_tails(candidate_losses: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for ticker, df in candidate_losses.items():
        losses = df["loss"].to_numpy()
        gpd_xis = [fit_gpd(losses, q).xi for q in SENSITIVITY_THRESHOLDS]
        rows.append({
            "ticker": ticker,
            "observations": len(losses),
            "volatility": float(np.std(losses, ddof=1)),
            "loss_skewness": float(stats.skew(losses, bias=False)),
            "excess_kurtosis": float(stats.kurtosis(losses, fisher=True, bias=False)),
            "q95_loss": float(np.quantile(losses, 0.95)),
            "q99_loss": float(np.quantile(losses, 0.99)),
            "maximum_loss": float(np.max(losses)),
            "q99_q95_ratio": float(np.quantile(losses, 0.99) / np.quantile(losses, 0.95)),
            "hill_xi": hill_estimator(losses),
            "gpd_xi_90": gpd_xis[0],
            "gpd_xi_92": gpd_xis[1],
            "gpd_xi_95": gpd_xis[2],
            "mean_gpd_xi": float(np.mean(gpd_xis)),
        })
    result = pd.DataFrame(rows)
    ranking_metrics = ["excess_kurtosis", "hill_xi", "mean_gpd_xi", "maximum_loss", "q99_q95_ratio"]
    for metric in ranking_metrics:
        result[f"rank_{metric}"] = result[metric].rank(ascending=False, method="min")
    result["average_tail_rank"] = result[[f"rank_{m}" for m in ranking_metrics]].mean(axis=1)
    result = result.sort_values(["average_tail_rank", "ticker"]).reset_index(drop=True)
    result.insert(0, "overall_rank", np.arange(1, len(result) + 1))
    return result


def kupiec_test(violations: Iterable[int], alpha: float) -> tuple[float, float]:
    v = np.asarray(list(violations), dtype=int)
    n = len(v)
    x = int(v.sum())
    p_hat = x / n

    def log_component(count: int, probability: float) -> float:
        if count == 0:
            return 0.0
        if probability <= 0:
            return float("-inf")
        return count * math.log(probability)

    null_ll = log_component(x, alpha) + log_component(n - x, 1 - alpha)
    alt_ll = log_component(x, p_hat) + log_component(n - x, 1 - p_hat)
    statistic = -2 * (null_ll - alt_ll)
    return float(statistic), float(1 - chi2.cdf(statistic, 1))


def christoffersen_tests(violations: Iterable[int], alpha: float) -> dict[str, float | int]:
    v = np.asarray(list(violations), dtype=int)
    previous, current = v[:-1], v[1:]
    n00 = int(np.sum((previous == 0) & (current == 0)))
    n01 = int(np.sum((previous == 0) & (current == 1)))
    n10 = int(np.sum((previous == 1) & (current == 0)))
    n11 = int(np.sum((previous == 1) & (current == 1)))

    def probability(numerator: int, denominator: int) -> float:
        return numerator / denominator if denominator else 0.0

    pi = probability(n01 + n11, n00 + n01 + n10 + n11)
    pi01 = probability(n01, n00 + n01)
    pi11 = probability(n11, n10 + n11)

    def log_component(count: int, p: float) -> float:
        if count == 0:
            return 0.0
        if p <= 0:
            return float("-inf")
        return count * math.log(p)

    independent_ll = log_component(n00 + n10, 1 - pi) + log_component(n01 + n11, pi)
    markov_ll = (
        log_component(n00, 1 - pi01) + log_component(n01, pi01)
        + log_component(n10, 1 - pi11) + log_component(n11, pi11)
    )
    lr_ind = -2 * (independent_ll - markov_ll)
    lr_uc, p_uc = kupiec_test(v, alpha)
    lr_cc = lr_uc + lr_ind
    return {
        "n00": n00, "n01": n01, "n10": n10, "n11": n11,
        "lr_uc": float(lr_uc), "p_uc": float(p_uc),
        "lr_ind": float(lr_ind), "p_ind": float(1 - chi2.cdf(lr_ind, 1)),
        "lr_cc": float(lr_cc), "p_cc": float(1 - chi2.cdf(lr_cc, 2)),
    }


def rolling_evt_backtest(df: pd.DataFrame, threshold_quantile: float,
                         confidences: Iterable[float] = RISK_LEVELS,
                         window: int = BACKTEST_WINDOW,
                         start_date: pd.Timestamp = BACKTEST_START) -> pd.DataFrame:
    data = df.sort_values("Date").reset_index(drop=True)
    candidate_indices = data.index[data["Date"] >= start_date]
    if len(candidate_indices) == 0:
        raise ValueError("Backtest start is outside the sample")
    start_index = int(candidate_indices[0])
    rows = []
    for i in range(start_index, len(data)):
        training = data.loc[max(0, i - window): i - 1, "loss"].to_numpy()
        if len(training) < window:
            continue
        fit = fit_gpd(training, threshold_quantile)
        actual_loss = float(data.loc[i, "loss"])
        for confidence in confidences:
            var, es = evt_var_es(fit, confidence)
            rows.append({
                "Date": data.loc[i, "Date"],
                "confidence": confidence,
                "actual_loss": actual_loss,
                "evt_var": var,
                "evt_es": es,
                "violation": int(actual_loss > var),
                "threshold": fit.threshold,
                "xi": fit.xi,
                "beta": fit.beta,
                "n_exceedances": fit.n_exceedances,
            })
    return pd.DataFrame(rows)


def plot_loss_series(selected_losses: dict[str, pd.DataFrame]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5), sharex=True)
    for ax, (asset, df) in zip(axes.flat, selected_losses.items()):
        ax.plot(df["Date"], 100 * df["loss"], linewidth=0.8)
        ax.axhline(0, linewidth=0.7)
        ax.set_title(asset)
        ax.set_ylabel("Daily loss, %")
        ax.grid(alpha=0.25)
    fig.suptitle("Daily loss series: negative logarithmic returns")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "01_loss_series.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_screening(screening: pd.DataFrame) -> None:
    top = screening.sort_values("average_tail_rank")
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.bar(top["ticker"], top["average_tail_rank"])
    ax.invert_yaxis()
    ax.set_ylabel("Average rank (lower means heavier tail)")
    ax.set_title("Heavy-tail screening of ten liquid US stocks")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "02_heavy_tail_screening.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_mean_excess(selected_losses: dict[str, pd.DataFrame]) -> pd.DataFrame:
    quantiles = np.arange(0.80, 0.981, 0.005)
    all_rows = []
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, (asset, df) in zip(axes.flat, selected_losses.items()):
        me = mean_excess(df["loss"].to_numpy(), quantiles)
        me.insert(0, "asset", asset)
        all_rows.append(me)
        ax.plot(100 * me["threshold"], 100 * me["mean_excess"], marker="o", markersize=2.2, linewidth=1)
        ax.fill_between(
            100 * me["threshold"],
            100 * (me["mean_excess"] - 1.96 * me["se"]),
            100 * (me["mean_excess"] + 1.96 * me["se"]),
            alpha=0.15,
        )
        chosen = SELECTED_THRESHOLDS[asset]
        chosen_u = np.quantile(df["loss"], chosen)
        ax.axvline(100 * chosen_u, linestyle="--", linewidth=1.1, label=f"Chosen u ({chosen:.0%})")
        ax.set_title(asset)
        ax.set_xlabel("Threshold u, % loss")
        ax.set_ylabel("Mean excess, %")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    fig.suptitle("Mean excess plots with 95% pointwise confidence bands")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "03_mean_excess_plots.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    result = pd.concat(all_rows, ignore_index=True)
    result.to_csv(TABLE_DIR / "mean_excess_values.csv", index=False)
    return result


def plot_parameter_stability(selected_losses: dict[str, pd.DataFrame]) -> pd.DataFrame:
    quantiles = np.arange(0.85, 0.971, 0.01)
    rows = []
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, (asset, df) in zip(axes.flat, selected_losses.items()):
        asset_rows = []
        for q in quantiles:
            fit = fit_gpd(df["loss"], q)
            asset_rows.append({
                "asset": asset, "threshold_quantile": q,
                "threshold": fit.threshold, "xi": fit.xi,
                "beta": fit.beta, "n_exceedances": fit.n_exceedances,
            })
        stability = pd.DataFrame(asset_rows)
        rows.append(stability)
        ax.plot(100 * stability["threshold_quantile"], stability["xi"], marker="o", markersize=3)
        ax.axvline(100 * SELECTED_THRESHOLDS[asset], linestyle="--", linewidth=1.1)
        ax.axhline(0, linewidth=0.7)
        ax.set_title(asset)
        ax.set_xlabel("Threshold percentile")
        ax.set_ylabel("GPD shape xi")
        ax.grid(alpha=0.25)
    fig.suptitle("Stability of the GPD shape parameter across thresholds")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "04_shape_parameter_stability.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    result = pd.concat(rows, ignore_index=True)
    result.to_csv(TABLE_DIR / "parameter_stability.csv", index=False)
    return result


def plot_qq(selected_fits: dict[str, GPDFit]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, (asset, fit) in zip(axes.flat, selected_fits.items()):
        empirical = np.sort(fit.exceedances)
        probabilities = (np.arange(1, len(empirical) + 1) - 0.5) / len(empirical)
        theoretical = genpareto.ppf(probabilities, c=fit.xi, loc=0, scale=fit.beta)
        ax.scatter(100 * theoretical, 100 * empirical, s=16)
        limit = max(float(np.max(theoretical)), float(np.max(empirical))) * 100
        ax.plot([0, limit], [0, limit], linestyle="--", linewidth=1)
        ax.set_title(f"{asset}: q={fit.threshold_quantile:.0%}, xi={fit.xi:.3f}")
        ax.set_xlabel("Theoretical GPD quantile, %")
        ax.set_ylabel("Empirical excess quantile, %")
        ax.grid(alpha=0.25)
    fig.suptitle("GPD Q-Q diagnostics")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "05_gpd_qq_plots.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_sensitivity(sensitivity: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, asset in zip(axes.flat, SELECTED_ASSETS):
        subset = sensitivity[(sensitivity["asset"] == asset) & (sensitivity["confidence"] == 0.99)]
        ax.plot(100 * subset["threshold_quantile"], 100 * subset["evt_var"], marker="o", label="99% VaR")
        subset2 = sensitivity[(sensitivity["asset"] == asset) & (sensitivity["confidence"] == 0.995)]
        ax.plot(100 * subset2["threshold_quantile"], 100 * subset2["evt_var"], marker="s", label="99.5% VaR")
        ax.axvline(100 * SELECTED_THRESHOLDS[asset], linestyle="--", linewidth=1)
        ax.set_title(asset)
        ax.set_xlabel("Threshold percentile")
        ax.set_ylabel("EVT VaR, %")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    fig.suptitle("Sensitivity of EVT-VaR to threshold selection")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "07_var_threshold_sensitivity.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_backtests(backtests: dict[str, pd.DataFrame]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
    for ax, (asset, backtest) in zip(axes.flat, backtests.items()):
        pivot = backtest.pivot(index="Date", columns="confidence", values="evt_var")
        actual = backtest.drop_duplicates("Date").set_index("Date")["actual_loss"]
        ax.plot(actual.index, 100 * actual, linewidth=0.75, label="Realized loss")
        ax.plot(pivot.index, 100 * pivot[0.99], linewidth=1.1, label="99% EVT-VaR")
        ax.plot(pivot.index, 100 * pivot[0.995], linewidth=1.1, label="99.5% EVT-VaR")
        violations = backtest[(backtest["confidence"] == 0.99) & (backtest["violation"] == 1)]
        ax.scatter(violations["Date"], 100 * violations["actual_loss"], marker="x", s=35, label="99% breach")
        ax.set_title(asset)
        ax.set_ylabel("Loss / VaR, %")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7, ncol=2)
    fig.suptitle("Rolling 600-day EVT-VaR backtest, 2024-03-04 to 2026-03-03")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "08_evt_var_backtests.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def run_analysis(cvm_simulations: int = 500, risk_simulations: int = 1000, run_backtest: bool = True) -> None:
    ensure_directories()

    # 1. Selection of stocks with the heaviest tails.
    candidate_losses = load_all_candidate_losses()
    screening = screen_heavy_tails(candidate_losses)
    screening.to_csv(TABLE_DIR / "heavy_tail_screening.csv", index=False)
    plot_screening(screening)

    # 2. Common sample and descriptive statistics.
    selected_losses = load_selected_common_losses()
    descriptive_rows = []
    for asset, df in selected_losses.items():
        losses = df["loss"].to_numpy()
        descriptive_rows.append({
            "asset": asset,
            "start_date": df["Date"].min().date().isoformat(),
            "end_date": df["Date"].max().date().isoformat(),
            "observations": len(losses),
            "mean_loss": float(np.mean(losses)),
            "standard_deviation": float(np.std(losses, ddof=1)),
            "loss_skewness": float(stats.skew(losses, bias=False)),
            "excess_kurtosis": float(stats.kurtosis(losses, fisher=True, bias=False)),
            "maximum_loss": float(np.max(losses)),
        })
    pd.DataFrame(descriptive_rows).to_csv(TABLE_DIR / "descriptive_statistics.csv", index=False)
    plot_loss_series(selected_losses)
    plot_mean_excess(selected_losses)
    plot_parameter_stability(selected_losses)

    # 3. Main fits, diagnostic tests and bootstrap intervals.
    selected_fits: dict[str, GPDFit] = {}
    fit_rows = []
    risk_rows = []
    for asset_index, (asset, df) in enumerate(selected_losses.items()):
        losses = df["loss"].to_numpy()
        fit = fit_gpd(losses, SELECTED_THRESHOLDS[asset])
        selected_fits[asset] = fit
        cvm, cvm_p = cvm_parametric_bootstrap(fit, simulations=cvm_simulations, seed=3100 + asset_index)
        bootstrap = bootstrap_fit_and_risk(fit, RISK_LEVELS, simulations=risk_simulations, seed=5100 + asset_index)
        xi_ci = bootstrap["xi"].quantile([0.025, 0.975])
        beta_ci = bootstrap["beta"].quantile([0.025, 0.975])
        undefined_es_share = float(np.mean(bootstrap["xi"] >= 1))
        fit_rows.append({
            "asset": asset,
            "threshold_quantile": fit.threshold_quantile,
            "threshold": fit.threshold,
            "n_total": fit.n_total,
            "n_exceedances": fit.n_exceedances,
            "exceedance_probability": fit.n_exceedances / fit.n_total,
            "xi": fit.xi,
            "xi_ci_lower": float(xi_ci.loc[0.025]),
            "xi_ci_upper": float(xi_ci.loc[0.975]),
            "beta": fit.beta,
            "beta_ci_lower": float(beta_ci.loc[0.025]),
            "beta_ci_upper": float(beta_ci.loc[0.975]),
            "cvm_statistic": cvm,
            "cvm_bootstrap_p": cvm_p,
            "bootstrap_share_xi_ge_1": undefined_es_share,
        })
        for confidence in RISK_LEVELS:
            evt_var, evt_es = evt_var_es(fit, confidence)
            var_ci = bootstrap[f"var_{confidence}"].replace([np.inf, -np.inf], np.nan).dropna().quantile([0.025, 0.975])
            es_values = bootstrap[f"es_{confidence}"].replace([np.inf, -np.inf], np.nan).dropna()
            es_ci = es_values.quantile([0.025, 0.975]) if len(es_values) else pd.Series({0.025: np.nan, 0.975: np.nan})
            risk_rows.append({
                "asset": asset, "confidence": confidence,
                "evt_var": evt_var, "evt_var_ci_lower": float(var_ci.loc[0.025]),
                "evt_var_ci_upper": float(var_ci.loc[0.975]),
                "evt_es": evt_es, "evt_es_ci_lower_finite": float(es_ci.loc[0.025]),
                "evt_es_ci_upper_finite": float(es_ci.loc[0.975]),
                "bootstrap_share_undefined_es": undefined_es_share,
            })

    fit_table = pd.DataFrame(fit_rows)
    risk_table = pd.DataFrame(risk_rows)
    fit_table.to_csv(TABLE_DIR / "gpd_fit_and_diagnostics.csv", index=False)
    risk_table.to_csv(TABLE_DIR / "evt_risk_estimates.csv", index=False)
    plot_qq(selected_fits)

    # 4. Sensitivity to threshold selection.
    sensitivity_rows = []
    for asset_index, (asset, df) in enumerate(selected_losses.items()):
        losses = df["loss"].to_numpy()
        for threshold_index, threshold_quantile in enumerate(SENSITIVITY_THRESHOLDS):
            fit = fit_gpd(losses, threshold_quantile)
            cvm, cvm_p = cvm_parametric_bootstrap(
                fit, simulations=cvm_simulations,
                seed=7100 + 100 * asset_index + threshold_index,
            )
            for confidence in RISK_LEVELS:
                var, es = evt_var_es(fit, confidence)
                sensitivity_rows.append({
                    "asset": asset,
                    "threshold_quantile": threshold_quantile,
                    "threshold": fit.threshold,
                    "n_exceedances": fit.n_exceedances,
                    "xi": fit.xi,
                    "beta": fit.beta,
                    "cvm_statistic": cvm,
                    "cvm_bootstrap_p": cvm_p,
                    "confidence": confidence,
                    "evt_var": var,
                    "evt_es": es,
                })
    sensitivity = pd.DataFrame(sensitivity_rows)
    sensitivity.to_csv(TABLE_DIR / "threshold_sensitivity.csv", index=False)
    plot_sensitivity(sensitivity)

    # 5. Rolling backtests.
    if run_backtest:
        backtests: dict[str, pd.DataFrame] = {}
        test_rows = []
        for asset, df in selected_losses.items():
            backtest = rolling_evt_backtest(df, SELECTED_THRESHOLDS[asset])
            backtests[asset] = backtest
            backtest.to_csv(TABLE_DIR / f"backtest_{asset}.csv", index=False)
            for confidence, group in backtest.groupby("confidence"):
                alpha = 1 - confidence
                test = christoffersen_tests(group["violation"], alpha)
                test_rows.append({
                    "asset": asset,
                    "confidence": confidence,
                    "observations": len(group),
                    "expected_violations": len(group) * alpha,
                    "actual_violations": int(group["violation"].sum()),
                    "violation_rate": float(group["violation"].mean()),
                    **test,
                    "kupiec_decision_5pct": "Do not reject" if test["p_uc"] >= 0.05 else "Reject",
                    "independence_decision_5pct": "Do not reject" if test["p_ind"] >= 0.05 else "Reject",
                    "conditional_coverage_decision_5pct": "Do not reject" if test["p_cc"] >= 0.05 else "Reject",
                })
        backtest_tests = pd.DataFrame(test_rows)
        backtest_tests.to_csv(TABLE_DIR / "backtest_tests.csv", index=False)
        plot_backtests(backtests)

    print("Analysis completed.")
    print(f"Tables: {TABLE_DIR}")
    print(f"Figures: {FIG_DIR}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the EVT/POT tail-risk study")
    parser.add_argument("--cvm-reps", type=int, default=500, help="Bootstrap replications for each CvM test")
    parser.add_argument("--risk-reps", type=int, default=1000, help="Bootstrap replications for parameter and risk intervals")
    parser.add_argument("--skip-backtest", action="store_true", help="Skip the rolling two-year VaR backtest")
    args = parser.parse_args()
    run_analysis(args.cvm_reps, args.risk_reps, not args.skip_backtest)
