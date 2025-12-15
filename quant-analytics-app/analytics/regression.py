from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class OLSResult:
    hedge_ratio: float
    intercept: float
    r2: float


@dataclass(frozen=True)
class RobustResult:
    hedge_ratio: float
    intercept: float
    r2: float


def hedge_ratio_ols(
    price_a: pd.Series,
    price_b: pd.Series,
    *,
    intercept: bool = True,
) -> OLSResult:
    """Compute hedge ratio via OLS: A ~ intercept + beta * B.

    Returns beta as hedge_ratio.

    Notes:
    - Inputs are aligned on index and NaNs dropped.
    - This keeps dependencies light (closed-form OLS) while matching statsmodels semantics.
    """

    df = pd.concat([price_a.rename("a"), price_b.rename("b")], axis=1).dropna()
    if len(df) < 3:
        raise ValueError("not enough data for regression")

    y = df["a"].to_numpy(dtype=float)
    x = df["b"].to_numpy(dtype=float)

    if intercept:
        X = np.column_stack([np.ones_like(x), x])
        coeffs, residuals, rank, s = np.linalg.lstsq(X, y, rcond=None)
        alpha = float(coeffs[0])
        beta = float(coeffs[1])
        y_hat = alpha + beta * x
    else:
        # no-intercept regression
        beta = float(np.dot(x, y) / np.dot(x, x)) if float(np.dot(x, x)) != 0.0 else float("nan")
        alpha = 0.0
        y_hat = beta * x

    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot != 0.0 else float("nan")

    return OLSResult(hedge_ratio=beta, intercept=alpha, r2=r2)


def hedge_ratio_robust_huber(
    price_a: pd.Series,
    price_b: pd.Series,
    *,
    intercept: bool = True,
    t: float = 1.345,
) -> RobustResult:
    """Robust hedge ratio using Huber loss via statsmodels RLM.

    This is more stable under outliers than OLS.
    """

    df = pd.concat([price_a.rename("a"), price_b.rename("b")], axis=1).dropna()
    if len(df) < 5:
        raise ValueError("not enough data for robust regression")

    try:
        import statsmodels.api as sm
    except Exception as e:  # noqa: BLE001
        raise RuntimeError("statsmodels is required for robust regression") from e

    y = df["a"].to_numpy(dtype=float)
    x = df["b"].to_numpy(dtype=float)

    if intercept:
        X = sm.add_constant(x, has_constant="add")
    else:
        X = x.reshape(-1, 1)

    model = sm.RLM(y, X, M=sm.robust.norms.HuberT(t=t))
    res = model.fit()

    if intercept:
        alpha = float(res.params[0])
        beta = float(res.params[1])
        y_hat = alpha + beta * x
    else:
        alpha = 0.0
        beta = float(res.params[0])
        y_hat = beta * x

    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot != 0.0 else float("nan")

    return RobustResult(hedge_ratio=beta, intercept=alpha, r2=r2)


@dataclass(frozen=True)
class ADFResult:
    adf_stat: float
    pvalue: float
    used_lag: int
    nobs: int
    critical_values: dict[str, float]


def adf_test(
    series: pd.Series,
    *,
    maxlag: Optional[int] = None,
    regression: Literal["c", "ct", "ctt", "n"] = "c",
    autolag: Literal["AIC", "BIC", "t-stat", None] = "AIC",
) -> ADFResult:
    """Run Augmented Dickey-Fuller test via statsmodels (preferred for correctness).

    This is typically a manual trigger in the UI (can be expensive).
    """
    s = pd.Series(series).dropna()
    if len(s) < 20:
        raise ValueError("not enough data for ADF test")

    try:
        from statsmodels.tsa.stattools import adfuller
    except Exception as e:  # noqa: BLE001
        raise RuntimeError("statsmodels is required for ADF test") from e

    adf_stat, pvalue, used_lag, nobs, crit, _icbest = adfuller(
        s.to_numpy(dtype=float),
        maxlag=maxlag,
        regression=regression,
        autolag=autolag,
    )

    return ADFResult(
        adf_stat=float(adf_stat),
        pvalue=float(pvalue),
        used_lag=int(used_lag),
        nobs=int(nobs),
        critical_values={str(k): float(v) for k, v in crit.items()},
    )
