import numpy as np

from inventory_lp import normal_loss, plan_orders, plan_orders_scenarios, shortage_segments

H, L = 7, 2


def solve(forecast, on_hand=None, pipeline=None, sigma=None, margin=None,
          delivery=1e6, storage=1e6):
    forecast = np.asarray(forecast, float)
    n_items = forecast.shape[0]
    on_hand = np.zeros(n_items) if on_hand is None else np.asarray(on_hand, float)
    pipeline = np.zeros((n_items, L - 1)) if pipeline is None else np.asarray(pipeline, float)
    sigma = np.full(n_items, 1e-6) if sigma is None else np.asarray(sigma, float)
    margin = np.ones(n_items) if margin is None else np.asarray(margin, float)
    return plan_orders(on_hand, pipeline, forecast, sigma, margin, np.full(n_items, 0.001),
                       delivery, storage, lead=L)


def test_shortage_curve_is_convex():
    width, slope = shortage_segments()
    assert (width > 0).all() and (slope < 0).all()
    assert (np.diff(slope) > 0).all()          # each extra unit of buffer helps less
    assert np.isclose(normal_loss(0.0), 0.3989, atol=1e-4)


def test_no_uncertainty_and_enough_capacity_orders_exactly_the_forecast():
    res = solve(np.full((1, H), 10.0), on_hand=[0], pipeline=[[10]])
    assert res["unmet"].sum() < 1e-6
    assert np.allclose(res["orders"][0], 10, atol=1e-6)


def test_buffer_grows_with_forecast_error():
    res = solve(np.full((2, H), 10.0), pipeline=[[10], [10]], sigma=[1, 5])
    buffer = res["stock"][:, L - 1: -1].mean(axis=1)
    assert buffer[1] > buffer[0] + 5


def test_delivery_capacity_is_respected_and_stock_is_built_before_a_peak():
    # Weekend peak on days 6-7 above what one delivery can bring.
    forecast = np.array([[10, 10, 10, 10, 10, 25, 25]], float)
    res = solve(forecast, on_hand=[10], pipeline=[[10]], delivery=15)
    assert (res["orders"].sum(axis=0) <= 15 + 1e-6).all()
    assert res["orders"][0, :3].max() > 10      # orders above demand mid-week
    assert res["unmet"].sum() < 1e-6


def test_tight_capacity_goes_to_the_item_that_loses_more_margin():
    res = solve(np.full((2, H), 10.0), margin=[1.0, 0.2], delivery=10)
    unmet = res["unmet"][:, L - 1:].sum(axis=1)
    assert unmet[0] < unmet[1]


def test_storage_capacity_limits_planned_stock():
    res = solve(np.full((2, H), 5.0), pipeline=[[5], [5]], sigma=[3, 3], storage=12)
    assert (res["stock"].sum(axis=0) <= 12 + 1e-6).all()
    assert (res["orders"].sum(axis=0) <= 12 + 1e-6).all()


def test_scenario_variant_covers_the_high_path_when_lost_sales_are_expensive():
    paths = np.array([[[5.0] * H], [[15.0] * H]])    # demand 5 or 15 a day
    res = plan_orders_scenarios(np.zeros(1), np.array([[15.0]]), paths, np.ones(1),
                                np.full(1, 0.001), 1e6, 1e6, lead=L)
    assert res["orders"][0, 0] > 10
