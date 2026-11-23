"""Day-by-day simulation of replenishment policies on real sales.

Timeline for each day d:
  morning   deliveries arrive (orders placed on the evening of d - LEAD_TIME);
            units that do not fit in storage are refused
  day       customers buy min(stock, demand); the rest is lost
  evening   the policy sees today's sales and places orders that arrive on d + LEAD_TIME
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import norm

import assumptions as A
import forecast as F
from inventory_lp import plan_orders, plan_orders_scenarios


@dataclass
class Context:
    """Everything a policy needs, prepared once."""
    daily: pd.DataFrame          # date x item
    info: pd.DataFrame           # per-item prices and costs
    origins: np.ndarray          # day index of each forecast origin
    fc: np.ndarray               # (model, item, origin, horizon)
    act: np.ndarray              # (item, origin, horizon)
    delivery_cap: float          # units per day
    storage_cap: float           # units

    @property
    def n_days(self):
        return len(self.daily)

    def forecast(self, model: str, t: int) -> np.ndarray:
        j = np.searchsorted(self.origins, t)
        return self.fc[F.MODELS.index(model), :, j, :]

    def recent_errors(self, model: str, t: int) -> np.ndarray:
        """Realized errors (actual - forecast), shaped (item, origin, horizon), for the
        ERROR_WINDOW most recent origins whose whole horizon is already observed on day t."""
        last = t - A.HORIZON
        sel = (self.origins > last - A.ERROR_WINDOW) & (self.origins <= last)
        k = F.MODELS.index(model)
        return self.act[:, sel, :] - self.fc[k][:, sel, :]


def build_context(daily, info, origins, fc, delivery_mult=A.DELIVERY_CAP,
                  storage_days=A.STORAGE_CAP) -> Context:
    avg_total = info["avg_daily_28d"].sum()
    return Context(daily, info, origins, fc, F.actuals_for(daily, origins),
                   delivery_mult * avg_total if delivery_mult else np.inf,
                   storage_days * avg_total if storage_days else np.inf)


# ---------------------------------------------------------------- policies

SHORT = {"moving_avg_7": "Moving average", "holt_winters": "Holt-Winters"}


def lead_time_sigma(ctx: Context, model: str, t: int) -> np.ndarray:
    """Std of the recent errors of the forecast total over the lead time, per item."""
    err = ctx.recent_errors(model, t)[:, :, : A.LEAD_TIME].sum(axis=2)
    return err.std(axis=1, ddof=1)


class OrderUpTo:
    """Textbook periodic-review rule: order up to (forecast over the lead time +
    z x error std). When the truck or storage is full, cut all orders by the same share."""

    def __init__(self, model: str, service_level: float):
        self.model, self.service_level = model, service_level
        self.z = norm.ppf(service_level)
        self.name = f"{model} + order-up-to ({service_level:.2%})"
        self.family = f"{SHORT[model]} + rule"

    def decide(self, ctx: Context, t, on_hand, pipeline):
        L = A.LEAD_TIME
        f = ctx.forecast(self.model, t)
        position = on_hand + pipeline.sum(axis=1)
        target = f[:, :L].sum(axis=1) + self.z * lead_time_sigma(ctx, self.model, t)
        q = np.clip(target - position, 0, None)
        if q.sum() > ctx.delivery_cap:
            q *= ctx.delivery_cap / q.sum()
        # Expected stock when this order arrives, before that day's sales.
        at_arrival = np.clip(position - f[:, : L - 1].sum(axis=1), 0, None)
        room = max(ctx.storage_cap - at_arrival.sum(), 0)
        if q.sum() > room:
            q *= room / q.sum()
        return q


class LPPlanner:
    """Forecast + the LP in inventory_lp.plan_orders, which looks 7 days ahead, shares
    capacity across items and sets each item's buffer from its margin and forecast error."""

    def __init__(self, model: str = "holt_winters"):
        self.model, self.service_level = model, float("nan")
        self.name = f"{model} + LP"
        self.family = f"{SHORT[model]} + LP"

    def plan(self, ctx: Context, t, on_hand, pipeline):
        info = ctx.info
        return plan_orders(on_hand, pipeline, ctx.forecast(self.model, t),
                           lead_time_sigma(ctx, self.model, t),
                           info["lost_margin"].to_numpy(), info["holding_per_day"].to_numpy(),
                           ctx.delivery_cap, ctx.storage_cap)

    def decide(self, ctx: Context, t, on_hand, pipeline):
        return self.plan(ctx, t, on_hand, pipeline)["orders"][:, 0]


class ScenarioLP:
    """Forecast + LP over sampled demand paths, no safety stock (the variant that lost)."""

    def __init__(self, model: str = "holt_winters", n_scenarios: int = A.N_SCENARIOS,
                 seed: int = A.SEED):
        self.model, self.n_scenarios, self.seed = model, n_scenarios, seed
        self.service_level = float("nan")
        self.name = f"{model} + scenario LP ({n_scenarios} paths)"
        self.family = f"{SHORT[model]} + scenario LP"

    def demand_paths(self, ctx: Context, t):
        f = ctx.forecast(self.model, t)
        err = ctx.recent_errors(self.model, t)              # (item, origin, horizon)
        rng = np.random.default_rng(self.seed + t)
        pick = rng.choice(err.shape[1], size=min(self.n_scenarios, err.shape[1]), replace=False)
        # The same past day is used for every item, so items keep moving together.
        paths = f[None, :, :] + np.transpose(err[:, pick, :], (1, 0, 2))
        return np.clip(paths, 0, None)

    def decide(self, ctx: Context, t, on_hand, pipeline):
        info = ctx.info
        res = plan_orders_scenarios(on_hand, pipeline, self.demand_paths(ctx, t),
                                    info["lost_margin"].to_numpy(),
                                    info["holding_per_day"].to_numpy(),
                                    ctx.delivery_cap, ctx.storage_cap)
        return res["orders"][:, 0]


# ---------------------------------------------------------------- simulation

def simulate(policy, ctx: Context) -> pd.DataFrame:
    """Run one policy over the warm-up and test days. Returns one row per item and day."""
    L, n = A.LEAD_TIME, ctx.n_days
    y = ctx.daily.to_numpy(float).T                       # (item, day)
    n_items = y.shape[0]
    first_day = n - A.TEST_DAYS - A.WARMUP_DAYS
    avg = ctx.info["avg_daily_28d"].to_numpy(float)

    arrivals = np.zeros((n_items, n + L + 1))
    arrivals[:, first_day: first_day + L - 1] = avg[:, None]   # same starting pipeline for all
    on_hand = 1.5 * avg                                        # same starting stock for all
    rows = []
    for t in range(first_day - 1, n - 1):
        q = policy.decide(ctx, t, on_hand.copy(), arrivals[:, t + 1: t + L].copy())
        arrivals[:, t + L] = q

        d = t + 1
        incoming = arrivals[:, d].copy()
        room = max(ctx.storage_cap - on_hand.sum(), 0)
        refused = np.zeros(n_items)
        if incoming.sum() > room:
            keep = incoming * room / incoming.sum()
            refused, incoming = incoming - keep, keep
        on_hand = on_hand + incoming
        start = on_hand.copy()
        demand = y[:, d]
        sales = np.minimum(on_hand, demand)
        on_hand = on_hand - sales
        rows.append(pd.DataFrame({
            "date": ctx.daily.index[d], "item_id": ctx.daily.columns, "day": d,
            "received": incoming, "refused": refused, "start_stock": start,
            "demand": demand, "sales": sales, "lost": demand - sales, "end_stock": on_hand,
        }))
    out = pd.concat(rows, ignore_index=True)
    item_idx = np.tile(np.arange(n_items), len(rows))
    out["ordered"] = arrivals[item_idx, out["day"].to_numpy() + L]   # placed that evening
    out = out.drop(columns="day")
    out["policy"] = policy.name
    out["test"] = out["date"] >= ctx.daily.index[n - A.TEST_DAYS]
    return out


def summarize(sim: pd.DataFrame, info: pd.DataFrame) -> dict:
    """Totals over the test days."""
    s = sim[sim.test].merge(info[["lost_margin", "holding_per_day", "avg_daily_28d"]],
                            left_on="item_id", right_index=True)
    lost_cost = (s.lost * s.lost_margin).sum()
    hold_cost = (s.end_stock * s.holding_per_day).sum()
    return_cost = s.refused.sum() * A.RETURN_COST
    days = s.date.nunique()
    by_day = s.groupby("date")
    return {
        "policy": sim.policy.iloc[0],
        "fill_rate": s.sales.sum() / s.demand.sum(),
        "in_stock_item_days": (s.lost <= 1e-9).mean(),
        "lost_units": s.lost.sum(),
        "lost_margin_usd": lost_cost,
        "holding_usd": hold_cost,
        "return_usd": return_cost,
        "total_cost_usd": lost_cost + hold_cost + return_cost,
        "avg_stock_units": by_day.end_stock.sum().mean(),
        "days_of_supply": by_day.end_stock.sum().mean() / (s.demand.sum() / days),
        "refused_units": s.refused.sum(),
        "peak_received_units": by_day.received.sum().max(),
        "peak_start_stock_units": by_day.start_stock.sum().max(),
    }
