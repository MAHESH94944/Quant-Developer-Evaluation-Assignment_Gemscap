from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class KalmanResult:
    alpha: pd.Series
    beta: pd.Series
    y_hat: pd.Series
    spread: pd.Series


def kalman_filter_regression(
    y: pd.Series,
    x: pd.Series,
    *,
    intercept: bool = True,
    delta: float = 1e-4,
    R: float = 1e-3,
) -> KalmanResult:
    """Estimate time-varying regression y_t ≈ alpha_t + beta_t x_t via a Kalman Filter.

    State model (random walk): theta_t = theta_{t-1} + w_t
    Observation: y_t = H_t theta_t + v_t

    Where:
      - theta_t = [alpha_t, beta_t] if intercept else [beta_t]
      - H_t = [1, x_t] or [x_t]

    Parameters:
      - delta: process noise scale (larger => faster parameter drift)
      - R: observation noise variance

    Returns alpha/beta series aligned to the input index.
    """

    df = pd.concat([pd.Series(y).rename("y"), pd.Series(x).rename("x")], axis=1).dropna()
    if df.empty:
        empty = pd.Series(dtype="float64")
        return KalmanResult(alpha=empty, beta=empty, y_hat=empty, spread=empty)

    yv = df["y"].to_numpy(dtype=float)
    xv = df["x"].to_numpy(dtype=float)

    n = len(df)
    state_dim = 2 if intercept else 1

    # Random-walk transition
    F = np.eye(state_dim)

    # Process noise (Q) – scaled identity
    Q = (delta / (1.0 - delta)) * np.eye(state_dim)

    # Observation noise
    Rm = float(R)

    theta = np.zeros((state_dim,))
    P = np.eye(state_dim) * 1.0

    alpha_out = np.zeros(n)
    beta_out = np.zeros(n)

    for t in range(n):
        # Predict
        theta_pred = F @ theta
        P_pred = F @ P @ F.T + Q

        # Observation matrix
        if intercept:
            H = np.array([[1.0, float(xv[t])]])
        else:
            H = np.array([[float(xv[t])]])

        # Innovation
        y_hat_t = float(H @ theta_pred)
        e = float(yv[t] - y_hat_t)

        S = float(H @ P_pred @ H.T + Rm)
        if S <= 0:
            S = 1e-9

        K = (P_pred @ H.T) / S  # (state_dim x 1)

        # Update
        theta = theta_pred + (K.flatten() * e)
        P = P_pred - K @ H @ P_pred

        if intercept:
            alpha_out[t] = float(theta[0])
            beta_out[t] = float(theta[1])
        else:
            alpha_out[t] = 0.0
            beta_out[t] = float(theta[0])

    idx = df.index
    alpha_s = pd.Series(alpha_out, index=idx, name="alpha")
    beta_s = pd.Series(beta_out, index=idx, name="beta")
    y_hat_s = alpha_s + beta_s * df["x"].astype(float)
    spread_s = df["y"].astype(float) - y_hat_s

    return KalmanResult(alpha=alpha_s, beta=beta_s, y_hat=y_hat_s.rename("y_hat"), spread=spread_s.rename("spread"))
