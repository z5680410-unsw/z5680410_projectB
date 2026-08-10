"""Station 3 - your funds: optimal portfolios + out-of-sample backtest.

Build at least a combined equity-plus-crypto fund with two optimisation methods.
Backtest rules: walk-forward, no look-ahead, weights from past data only, annualise
with 252 (equity) or 365 (crypto). See the brief, Part B.
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize, linprog

METHODS = ("equal_weight", "min_variance", "max_sharpe", "risk_parity", "mean_cvar")


# ---------------------------------------------------------------------------
# Optimisers - each takes a WINDOW of past returns (assets in columns, no
# 'date' column) and returns a 1-D weight vector: long-only, sums to 1.
#
# All three non-trivial optimisers annualise the covariance (multiply by
# periods_per_year) before solving. This does NOT change the optimal
# weights - minimising w'Sigma*w and minimising w'(c*Sigma)*w for any
# constant c>0 have the same argmin - but it rescales the tiny daily-return
# covariance entries (~1e-4) into a numerically friendlier range, which is
# exactly the "solver scaling" stall the brief warns about.
# ---------------------------------------------------------------------------

def _equal_weight(window: pd.DataFrame, **kwargs) -> np.ndarray:
    n = window.shape[1]
    return np.full(n, 1.0 / n)


def _min_variance(window: pd.DataFrame, **kwargs) -> np.ndarray:
    n = window.shape[1]
    cov = window.cov().to_numpy() * kwargs.get("periods_per_year", 252)

    def objective(w):
        return w @ cov @ w

    return _solve(objective, n, "min_variance")


def _max_sharpe(window: pd.DataFrame, **kwargs) -> np.ndarray:
    n = window.shape[1]
    periods_per_year = kwargs.get("periods_per_year", 252)
    mean_ret = window.mean().to_numpy() * periods_per_year
    cov = window.cov().to_numpy() * periods_per_year
    rf = kwargs.get("rf", 0.0)

    def objective(w):
        port_ret = w @ mean_ret
        port_vol = np.sqrt(w @ cov @ w)
        if port_vol < 1e-12:
            return 0.0
        return -(port_ret - rf) / port_vol  # minimise negative Sharpe

    return _solve(objective, n, "max_sharpe")


def _risk_parity(window: pd.DataFrame, **kwargs) -> np.ndarray:
    n = window.shape[1]
    cov = window.cov().to_numpy() * kwargs.get("periods_per_year", 252)
    target = np.full(n, 1.0 / n)

    def objective(w):
        port_var = w @ cov @ w
        if port_var < 1e-12:
            return 0.0
        risk_contrib = w * (cov @ w) / port_var
        return np.sum((risk_contrib - target) ** 2)

    return _solve(objective, n, "risk_parity")


def _min_cvar(window: pd.DataFrame, beta: float = 0.95, **kwargs) -> np.ndarray:
    """Innovation (structured side): minimise CVaR ("mean-CVaR"), the
    expected loss in the worst (1-beta) share of scenarios, via the
    Rockafellar & Uryasev (2000) linear-programming reformulation - the
    standard, exact way to optimise CVaR without needing it to be smooth.

    Unlike the covariance-based methods above, this works directly on the
    RAW historical daily-return scenarios in the window (not an annualised
    covariance matrix), because CVaR is a property of the empirical
    distribution's tail, not of pairwise covariances. This is a
    "tail-aware objective": variance penalises upside and downside
    surprises equally, CVaR targets only the worst-outcome tail directly.

    Decision vector is [w (N assets), zeta (1, the VaR threshold),
    u (T scenarios, the shortfall beyond zeta in each scenario)]:
        minimise      zeta + (1/(T*(1-beta))) * sum(u)
        subject to    u_t >= -(w . r_t) - zeta   for every scenario t
                       u_t >= 0
                       sum(w) = 1,  0 <= w <= 1  (long-only, matches every other method)
    """
    r = window.to_numpy()
    T, N = r.shape
    n_vars = N + 1 + T

    c = np.concatenate([np.zeros(N), [1.0], np.full(T, 1.0 / (T * (1 - beta)))])

    A_ub = np.zeros((T, n_vars))
    A_ub[:, :N] = -r
    A_ub[:, N] = -1.0
    A_ub[np.arange(T), N + 1 + np.arange(T)] = -1.0
    b_ub = np.zeros(T)

    A_eq = np.zeros((1, n_vars))
    A_eq[0, :N] = 1.0
    b_eq = [1.0]

    bounds = [(0.0, 1.0)] * N + [(None, None)] + [(0.0, None)] * T

    result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                      bounds=bounds, method="highs")
    if not result.success:
        print(f"[warn] mean_cvar optimiser did not converge "
              f"({result.message}) - falling back to equal-weight for this window")
        return np.full(N, 1.0 / N)
    w = np.clip(result.x[:N], 0.0, None)
    return w / w.sum()


def _solve(objective, n_assets: int, method_name: str) -> np.ndarray:
    """Shared SLSQP solve: long-only (0<=w<=1), fully invested (sum(w)=1).

    Falls back to equal-weight (and prints a warning, so a silent stall
    never hides in the output) if the solver does not converge - the brief
    explicitly flags that optimisers on tiny daily-return covariances can
    stall, so this fallback is a safety net, not the expected path.
    """
    x0 = np.full(n_assets, 1.0 / n_assets)
    bounds = [(0.0, 1.0)] * n_assets
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    result = minimize(objective, x0, method="SLSQP", bounds=bounds,
                       constraints=constraints,
                       options={"maxiter": 1000, "ftol": 1e-12})
    if not result.success:
        print(f"[warn] {method_name} optimiser did not converge "
              f"({result.message}) - falling back to equal-weight for this window")
        return x0
    w = np.clip(result.x, 0.0, None)
    return w / w.sum()


_OPTIMIZERS = {
    "equal_weight": _equal_weight,
    "min_variance": _min_variance,
    "max_sharpe": _max_sharpe,
    "risk_parity": _risk_parity,
    "mean_cvar": _min_cvar,
}


# ---------------------------------------------------------------------------

def oos_backtest(returns: pd.DataFrame, method: str = "min_variance",
                  first_live_date: str = "2021-01-01",
                  periods_per_year: int = 252, rf: float = 0.0) -> dict:
    """Walk-forward out-of-sample backtest for ONE fund (one asset universe,
    one optimisation method), using an EXPANDING estimation window with
    MONTHLY re-training: estimate on the first year of returns, trade the
    first live month at those weights, then re-estimate on ALL data seen so
    far (the window only ever grows, it never rolls off), and repeat once a
    calendar month through the end of the sample.

    `returns` is a WIDE panel: a 'date' column plus one return column per
    asset (as built by features.daily_returns / build_combined_returns_panel).

    No look-ahead: at the start of each new calendar month, weights are
    estimated from every trading day strictly BEFORE that month's first
    date - never from the current or a future date. Those weights are then
    held fixed and applied to REALISED returns for the rest of the month.

    Returns a dict with:
        daily_returns     - pd.Series (date-indexed) of the fund's realised
                             daily return, from first_live_date onward
        weights           - pd.DataFrame, one row per MONTHLY rebalance
                             date, one column per asset (the fund's
                             fact-sheet "holdings" history)
        growth_of_dollar  - pd.Series, cumulative growth of $1 invested at
                             the first out-of-sample date
        first_oos_date    - the first live out-of-sample date
        method, periods_per_year - echoed back for downstream bookkeeping
    """
    if method not in _OPTIMIZERS:
        raise ValueError(f"unknown method '{method}', choose from {list(_OPTIMIZERS)}")

    df = returns.sort_values("date").reset_index(drop=True)
    asset_cols = [c for c in df.columns if c != "date"]

    # Drop LEADING rows until every asset column has a value. A single-
    # universe panel (equity-only, crypto-only) has exactly one such row -
    # the panel's first date, with no prior price to compute a return from
    # (see features.daily_returns). The COMBINED panel is trickier: equity's
    # leading-NaN row and crypto's leading-NaN row can fall on DIFFERENT
    # calendar dates (crypto's calendar starts a day earlier), so a plain
    # "all columns NaN" check would miss it. Walking forward to the first
    # fully-populated row handles all three universes the same way.
    first_valid_idx = df[asset_cols].notna().all(axis=1).idxmax()
    df = df.loc[first_valid_idx:].reset_index(drop=True)
    df[asset_cols] = df[asset_cols].fillna(0.0)

    first_live_date = pd.Timestamp(first_live_date)
    live_rows = df.index[df["date"] >= first_live_date]
    if len(live_rows) == 0:
        raise ValueError(
            f"no rows on/after first_live_date={first_live_date.date()} - "
            f"check the returns panel covers this period"
        )
    if live_rows[0] == 0:
        raise ValueError(
            "first_live_date falls on/before the panel's first valid row - "
            "there is no prior data to estimate the first live month's weights from"
        )

    year_month = df["date"].dt.to_period("M")
    optimizer = _OPTIMIZERS[method]
    weight_rows, daily_port_returns = [], []
    current_weights, current_month = None, None

    for i in live_rows:
        row_month = year_month.iloc[i]
        if row_month != current_month:
            # New calendar month: re-estimate on EVERY row strictly before
            # this one - an expanding window that only ever grows, never
            # rolls off.
            window = df.loc[:i - 1, asset_cols]
            current_weights = optimizer(window, periods_per_year=periods_per_year, rf=rf)
            weight_rows.append({"date": df.loc[i, "date"],
                                 **dict(zip(asset_cols, current_weights))})
            current_month = row_month

        today_returns = df.loc[i, asset_cols].to_numpy()
        daily_port_returns.append({"date": df.loc[i, "date"],
                                    "return": float(current_weights @ today_returns)})

    daily_returns = pd.DataFrame(daily_port_returns).set_index("date")["return"]
    weights = pd.DataFrame(weight_rows).set_index("date")
    growth_of_dollar = (1.0 + daily_returns).cumprod()

    return {
        "daily_returns": daily_returns,
        "weights": weights,
        "growth_of_dollar": growth_of_dollar,
        "first_oos_date": daily_returns.index[0],
        "method": method,
        "periods_per_year": periods_per_year,
    }


def performance_metrics(daily_returns: pd.Series, periods_per_year: int = 252,
                         rf: float = 0.0) -> dict:
    """Annualised return (CAGR), annualised volatility, Sharpe ratio, and
    max drawdown from a fund's daily-return series.

    rf=0 is assumed for the Sharpe ratio (stated assumption per the brief,
    a reasonable simplification - see the report for discussion).
    """
    r = daily_returns.dropna()
    if len(r) == 0:
        raise ValueError(
            "performance_metrics() received zero non-NaN daily returns - "
            "the fund/date-window has no data (e.g. a discovery/holdout "
            "window that falls entirely outside the fund's OOS period). "
            "Check the date range against the fund's own first_oos_date."
        )
    n_days = len(r)
    growth = (1.0 + r).cumprod()
    total_return = growth.iloc[-1] - 1.0
    n_years = n_days / periods_per_year
    cagr = (1.0 + total_return) ** (1.0 / n_years) - 1.0 if n_years > 0 else np.nan
    ann_vol = r.std(ddof=0) * np.sqrt(periods_per_year)
    sharpe = (cagr - rf) / ann_vol if ann_vol > 0 else np.nan
    running_max = growth.cummax()
    drawdown = growth / running_max - 1.0
    max_drawdown = drawdown.min()

    return {
        "annualised_return": cagr,
        "annualised_volatility": ann_vol,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_drawdown,
        "n_days": n_days,
    }
