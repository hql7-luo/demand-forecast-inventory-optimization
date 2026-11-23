import numpy as np
import pandas as pd
import pytest

import assumptions as A
import forecast as F
import simulate as S


@pytest.fixture(scope="module")
def ctx():
    """Three items, 200 days with a weekly pattern and noise, and a simple forecast
    (same weekday last week) stored in the slot of every model."""
    rng = np.random.default_rng(1)
    n = A.TEST_DAYS + A.WARMUP_DAYS + A.ERROR_WINDOW + A.HORIZON + 40
    pattern = np.array([0.9, 0.8, 0.8, 0.85, 1.0, 1.3, 1.35])
    base = np.array([20.0, 10.0, 5.0])
    t = np.arange(n)
    y = np.maximum(rng.poisson(base[:, None] * pattern[t % 7]), 0).T
    daily = pd.DataFrame(y, columns=["a", "b", "c"],
                         index=pd.date_range("2020-01-01", periods=n, name="date"))
    info = pd.DataFrame({
        "lost_margin": [0.5, 0.8, 1.2], "holding_per_day": [0.001] * 3,
        "avg_daily_28d": daily.iloc[-A.TEST_DAYS - 28: -A.TEST_DAYS].mean().values,
    }, index=daily.columns)
    origins = np.arange(A.HORIZON + 7, n - 1)
    one = np.stack([[y[o - 6: o + 1, i][(np.arange(A.HORIZON)) % 7] for o in origins]
                    for i in range(3)])
    fc = np.repeat(one[None], len(F.MODELS), axis=0).astype(float)
    return S.build_context(daily, info, origins, fc, delivery_mult=1.1, storage_days=3)


@pytest.mark.parametrize("policy", [S.OrderUpTo("holt_winters", 0.95), S.LPPlanner("holt_winters"),
                                    S.ScenarioLP(n_scenarios=5)])
def test_stock_balances_and_capacities_hold(ctx, policy):
    sim = S.simulate(policy, ctx)
    assert len(sim) == 3 * (A.TEST_DAYS + A.WARMUP_DAYS)
    assert (sim.sales <= sim.start_stock + 1e-9).all()
    assert np.allclose(sim.lost, sim.demand - sim.sales)
    assert np.allclose(sim.end_stock, sim.start_stock - sim.sales)
    # Yesterday's end stock + today's delivery = today's start stock.
    for _, g in sim.groupby("item_id"):
        assert np.allclose(g.start_stock.values[1:], g.end_stock.values[:-1] + g.received.values[1:])
    by_day = sim.groupby("date")
    assert (by_day.ordered.sum() <= ctx.delivery_cap + 1e-6).all()
    assert (by_day.start_stock.sum() <= ctx.storage_cap + 1e-6).all()


def test_summary_counts_only_test_days(ctx):
    sim = S.simulate(S.OrderUpTo("holt_winters", 0.95), ctx)
    row = S.summarize(sim, ctx.info)
    assert np.isclose(row["lost_units"], sim[sim.test].lost.sum())
    assert 0 < row["fill_rate"] <= 1
