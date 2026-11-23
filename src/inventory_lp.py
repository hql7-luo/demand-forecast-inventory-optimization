"""Linear programs for daily replenishment of many items that share one delivery
capacity and one storage capacity.

Each evening the planner decides how many units of each item to order for the next
days. Only tonight's orders are placed; the plan is solved again the next evening
with new sales (a rolling horizon).

plan_orders            the main model: forecast + an expected-shortage curve per item
plan_orders_scenarios  a variant that plans against sampled demand paths instead
                       (kept for comparison; see the README)
"""
import numpy as np
import pulp
from scipy.stats import norm

import assumptions as A


def _solve(m):
    m.solve(pulp.HiGHS(msg=False))
    if pulp.LpStatus[m.status] != "Optimal":
        raise RuntimeError(f"LP status: {pulp.LpStatus[m.status]}")


# Breakpoints (in error std units) for the piecewise-linear expected-shortage curve.
Z_BREAKS = np.array([0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 3.5])


def normal_loss(z):
    """Expected shortage of a standard normal above z: E[(Z - z)+]."""
    return norm.pdf(z) - z * (1 - norm.cdf(z))


def shortage_segments(z_breaks=Z_BREAKS):
    """Width and slope of each linear piece of the expected-shortage curve.
    Slopes are negative and get flatter: the first units of buffer help the most."""
    width = np.diff(z_breaks)
    slope = np.diff(normal_loss(z_breaks)) / width
    return width, slope


def plan_orders(on_hand, pipeline, forecast, sigma, lost_margin, holding_cost,
                delivery_cap, storage_cap, lead=A.LEAD_TIME):
    """Solve one evening's plan.

    on_hand       (I,)       units on hand tonight
    pipeline      (I, L-1)   units already ordered, arriving on days 1..L-1
    forecast      (I, H)     forecast demand for days 1..H
    sigma         (I,)       std of recent lead-time forecast errors
    lost_margin   (I,)       $ lost per unit of demand not met
    holding_cost  (I,)       $ per unit held overnight
    delivery_cap  units that can arrive per day, all items together
    storage_cap   units that fit on hand after a delivery, all items together;
                  units that do not fit go back at A.RETURN_COST each

    Cost of a plan = lost margin on forecast demand it cannot meet
                   + lost margin on the shortage expected from forecast error, which falls
                     as end-of-day stock rises (piecewise-linear normal loss, in sigma units)
                   + holding + returns.
    There is no service-level target: how much buffer each item gets comes from its
    margin, its forecast error and the shared capacity.

    Orders placed on evening j (0 = tonight) arrive on day j + lead.
    """
    n_items, H = forecast.shape
    J = H - lead + 1
    items, days, order_days = range(n_items), range(1, H + 1), range(J)
    width, slope = shortage_segments()
    segs = range(len(width))
    sigma = np.maximum(np.asarray(sigma, float), 1e-6)

    m = pulp.LpProblem("replenishment", pulp.LpMinimize)
    q = pulp.LpVariable.dicts("order", (items, order_days), lowBound=0)
    x = pulp.LpVariable.dicts("stock", (items, days), lowBound=0)        # end of day
    u = pulp.LpVariable.dicts("unmet", (items, days), lowBound=0)        # forecast demand not met
    r = pulp.LpVariable.dicts("returned", (items, days), lowBound=0)     # did not fit
    buf = {(i, k, g): pulp.LpVariable(f"buffer_{i}_{k}_{g}", 0, float(width[g] * sigma[i]))
           for i in items for k in range(lead, H + 1) for g in segs}

    def arrival(i, k):
        return pipeline[i, k - 1] if k < lead else q[i][k - lead]

    def start_stock(i, k):
        prev = on_hand[i] if k == 1 else x[i][k - 1]
        return prev + arrival(i, k) - r[i][k]

    m += (
        pulp.lpSum(lost_margin[i] * u[i][k] + holding_cost[i] * x[i][k] + A.RETURN_COST * r[i][k]
                   for i in items for k in days)
        + pulp.lpSum(lost_margin[i] * slope[g] * buf[i, k, g] for (i, k, g) in buf)
    )
    for k in days:
        for i in items:
            m += x[i][k] == start_stock(i, k) - float(forecast[i, k - 1]) + u[i][k]
            m += u[i][k] <= float(forecast[i, k - 1])
            m += r[i][k] <= arrival(i, k)
            if k >= lead:   # tonight's order can only change stock from day `lead` on
                m += pulp.lpSum(buf[i, k, g] for g in segs) <= x[i][k]
        m += pulp.lpSum(start_stock(i, k) for i in items) <= storage_cap
    for j in order_days:
        m += pulp.lpSum(q[i][j] for i in items) <= delivery_cap

    _solve(m)
    plan = np.array([[q[i][j].value() or 0.0 for j in order_days] for i in items])
    stock = np.array([[x[i][k].value() or 0.0 for k in days] for i in items])
    unmet = np.array([[u[i][k].value() or 0.0 for k in days] for i in items])
    return {"orders": np.clip(plan, 0, None), "stock": stock, "unmet": unmet,
            "objective": pulp.value(m.objective)}


def plan_orders_scenarios(on_hand, pipeline, demand_paths, lost_margin, holding_cost,
                          delivery_cap, storage_cap, lead=A.LEAD_TIME):
    """Same decisions, but planned against S demand paths (S, I, H) with no safety stock:
    minimize the average cost across paths. Orders are the same in every path."""
    S, n_items, H = demand_paths.shape
    J = H - lead + 1
    items, days, scen, order_days = range(n_items), range(1, H + 1), range(S), range(J)

    m = pulp.LpProblem("replenishment_scenarios", pulp.LpMinimize)
    q = pulp.LpVariable.dicts("order", (items, order_days), lowBound=0)
    x = pulp.LpVariable.dicts("stock", (scen, items, days), lowBound=0)
    u = pulp.LpVariable.dicts("lost", (scen, items, days), lowBound=0)
    r = pulp.LpVariable.dicts("returned", (scen, items, days), lowBound=0)

    def arrival(i, k):
        return pipeline[i, k - 1] if k < lead else q[i][k - lead]

    def start_stock(s, i, k):
        prev = on_hand[i] if k == 1 else x[s][i][k - 1]
        return prev + arrival(i, k) - r[s][i][k]

    m += (1.0 / S) * pulp.lpSum(
        lost_margin[i] * u[s][i][k] + holding_cost[i] * x[s][i][k] + A.RETURN_COST * r[s][i][k]
        for s in scen for i in items for k in days
    )
    for s in scen:
        for k in days:
            for i in items:
                m += x[s][i][k] == start_stock(s, i, k) - float(demand_paths[s, i, k - 1]) + u[s][i][k]
                m += u[s][i][k] <= float(demand_paths[s, i, k - 1])
                m += r[s][i][k] <= arrival(i, k)
            m += pulp.lpSum(start_stock(s, i, k) for i in items) <= storage_cap
    for j in order_days:
        m += pulp.lpSum(q[i][j] for i in items) <= delivery_cap

    _solve(m)
    plan = np.array([[q[i][j].value() or 0.0 for j in order_days] for i in items])
    lost = np.array([[sum((u[s][i][k].value() or 0.0) for s in scen) / S for k in days]
                     for i in items])
    return {"orders": np.clip(plan, 0, None), "expected_lost": lost,
            "objective": pulp.value(m.objective)}
