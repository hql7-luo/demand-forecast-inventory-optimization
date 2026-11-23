import numpy as np
import pandas as pd

import forecast as F


def weekly_series(n=400, level=20.0, noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    pattern = np.array([0.9, 0.8, 0.8, 0.85, 1.0, 1.3, 1.35])
    t = np.arange(n)
    return level * pattern[t % 7] + rng.normal(0, noise, n)


def test_simple_models_use_only_the_history():
    y = np.arange(1, 29, dtype=float)              # 28 days: 1..28
    f = F.forecast_one(y, 7)
    assert np.allclose(f["naive"], 28)
    assert np.allclose(f["seasonal_naive"], y[-7:])
    assert np.allclose(f["moving_avg_7"], y[-7:].mean())
    assert np.allclose(f["weekday_avg_4w"], y[-28:].reshape(4, 7).mean(axis=0))


def test_holt_winters_learns_a_weekly_pattern():
    y = weekly_series(noise=0.5)
    hist, future = y[:364], y[364:371]
    f = F.forecast_one(hist, 7)
    assert F.wape(future, f["holt_winters"]) < 0.05
    # A flat forecast cannot follow the weekend peak.
    assert F.wape(future, f["holt_winters"]) < F.wape(future, f["moving_avg_7"])


def test_forecasts_are_never_negative():
    y = np.r_[np.full(357, 5.0), np.zeros(7)]
    for v in F.forecast_one(y, 7).values():
        assert (v >= 0).all()


def test_actuals_line_up_with_origins():
    daily = pd.DataFrame({"a": np.arange(10.0), "b": np.arange(10.0) * 2})
    act = F.actuals_for(daily, np.array([2, 8]), h=3)
    assert act.shape == (2, 2, 3)
    assert np.allclose(act[0, 0], [3, 4, 5])          # days after origin 2
    assert act[0, 1, 0] == 9 and np.isnan(act[0, 1, 1:]).all()


def test_wape_and_bias():
    a, f = np.array([10.0, 10.0]), np.array([12.0, 6.0])
    assert np.isclose(F.wape(a, f), 6 / 20)
    assert np.isclose(F.bias(a, f), -2 / 20)
