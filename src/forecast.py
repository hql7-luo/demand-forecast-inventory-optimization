"""Daily demand forecasts and a rolling-origin backtest.

Every model is refit at every origin on the last FIT_WINDOW days only, so a forecast
made on day t never sees data after day t.
"""
from multiprocessing import Pool
import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing, SimpleExpSmoothing

import assumptions as A

MODELS = ["naive", "seasonal_naive", "moving_avg_7", "weekday_avg_4w", "ses", "holt_winters"]
LABELS = {
    "naive": "Naive (yesterday)",
    "seasonal_naive": "Seasonal naive (same day last week)",
    "moving_avg_7": "7-day moving average",
    "weekday_avg_4w": "Same-weekday average, 4 weeks",
    "ses": "Simple exponential smoothing",
    "holt_winters": "Holt-Winters (damped trend, weekly season)",
}


def forecast_one(history: np.ndarray, h: int) -> dict[str, np.ndarray]:
    """Forecast the next h days from one history. Returns model -> array of length h."""
    y = np.asarray(history, dtype=float)
    m = A.SEASON
    steps = np.arange(1, h + 1)
    out = {
        "naive": np.repeat(y[-1], h),
        "seasonal_naive": y[-m:][(steps - 1) % m],
        "moving_avg_7": np.repeat(y[-7:].mean(), h),
        # Average of the same weekday over the last 4 weeks.
        "weekday_avg_4w": y[-4 * m:].reshape(4, m).mean(axis=0)[(steps - 1) % m],
    }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out["ses"] = SimpleExpSmoothing(y, initialization_method="estimated").fit().forecast(h)
        hw = ExponentialSmoothing(
            y, trend="add", damped_trend=True, seasonal="add",
            seasonal_periods=m, initialization_method="estimated",
        ).fit()
        out["holt_winters"] = hw.forecast(h)
    return {k: np.clip(np.asarray(v, dtype=float), 0, None) for k, v in out.items()}


def _backtest_item(args):
    y, origins, h = args
    res = np.full((len(MODELS), len(origins), h), np.nan)
    for j, o in enumerate(origins):
        f = forecast_one(y[o - A.FIT_WINDOW + 1: o + 1], h)
        for k, name in enumerate(MODELS):
            res[k, j] = f[name]
    return res


def backtest(daily: pd.DataFrame, origins: np.ndarray, h: int = A.HORIZON,
             processes: int = 4) -> np.ndarray:
    """Forecasts from each origin (index of the last observed day).

    Returns an array shaped (model, item, origin, horizon)."""
    jobs = [(daily[c].to_numpy(float), origins, h) for c in daily.columns]
    if processes > 1:
        with Pool(processes) as pool:
            per_item = pool.map(_backtest_item, jobs)
    else:
        per_item = [_backtest_item(j) for j in jobs]
    return np.stack(per_item, axis=1)


def actuals_for(daily: pd.DataFrame, origins: np.ndarray, h: int = A.HORIZON) -> np.ndarray:
    """Actual sales after each origin, shaped (item, origin, horizon). NaN past the data."""
    y = daily.to_numpy(float).T
    out = np.full((y.shape[0], len(origins), h), np.nan)
    for j, o in enumerate(origins):
        stop = min(o + 1 + h, y.shape[1])
        out[:, j, : stop - o - 1] = y[:, o + 1: stop]
    return out


def wape(actual: np.ndarray, forecast: np.ndarray) -> float:
    """Weighted absolute percentage error: sum |error| / sum actual."""
    mask = ~np.isnan(actual)
    return float(np.abs(actual[mask] - forecast[mask]).sum() / actual[mask].sum())


def bias(actual: np.ndarray, forecast: np.ndarray) -> float:
    """Sum of (forecast - actual) / sum actual. Positive means over-forecasting."""
    mask = ~np.isnan(actual)
    return float((forecast[mask] - actual[mask]).sum() / actual[mask].sum())


def accuracy_table(fc: np.ndarray, act: np.ndarray, lead: int = A.LEAD_TIME) -> pd.DataFrame:
    """Daily WAPE and bias over horizons 1..H, plus WAPE of the lead-time total,
    which is what the order decision depends on."""
    rows = []
    for k, name in enumerate(MODELS):
        f = fc[k]
        lead_a, lead_f = act[:, :, :lead].sum(axis=2), f[:, :, :lead].sum(axis=2)
        rows.append({
            "model": name,
            "label": LABELS[name],
            "wape_daily": wape(act, f),
            "bias_daily": bias(act, f),
            "wape_lead_time_total": wape(lead_a, lead_f),
        })
    return pd.DataFrame(rows).sort_values("wape_daily").reset_index(drop=True)
