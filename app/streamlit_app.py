"""Dashboard: forecasts, replenishment results, capacity what-ifs and a live LP plan.

    streamlit run app/streamlit_app.py
"""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import assumptions as A  # noqa: E402
import forecast as F  # noqa: E402
import simulate as S  # noqa: E402

st.set_page_config(page_title="Demand Forecast + Replenishment LP", layout="wide")

RULE_MA, RULE_HW, LP = "Moving average + rule", "Holt-Winters + rule", "Holt-Winters + LP"
COLORS = {RULE_MA: "#8c96a0", RULE_HW: "#2f6fb0", LP: "#d9682b"}
WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


@st.cache_data
def load():
    res = ROOT / "results"
    daily = pd.read_csv(ROOT / "data/processed/daily_sales.csv", index_col=0, parse_dates=True)
    info = pd.read_csv(ROOT / "data/processed/items.csv", index_col=0)
    z = np.load(res / "forecasts.npz", allow_pickle=False)
    return dict(
        daily=daily, info=info, origins=z["origins"], fc=z["fc"].astype(float),
        acc=pd.read_csv(res / "forecast_accuracy.csv"),
        best=pd.read_csv(res / "best_per_family.csv"),
        head=pd.read_csv(res / "headline_policies.csv"),
        sims=pd.read_csv(res / "simulation_daily.csv.gz", parse_dates=["date"]),
        other_acc=pd.read_csv(res / "other_stores/forecast_accuracy.csv"),
        other_best=pd.read_csv(res / "other_stores/best_per_family.csv"),
    )


D = load()
daily, info, origins, fc = D["daily"], D["info"], D["origins"], D["fc"]
head = D["head"].set_index("family")
family_of = dict(zip(D["head"].policy, D["head"].family))
sims = D["sims"].assign(family=lambda d: d.policy.map(family_of))
test = sims[sims.test]
avg_total = info.avg_daily_28d.sum()

st.title("Daily replenishment for 50 grocery items: forecast, then optimize")
st.caption(
    f"Walmart store {A.STORE}, department {A.DEPT} (M5 data). Test window "
    f"{daily.index[-A.TEST_DAYS].date()} to {daily.index[-1].date()} ({A.TEST_DAYS} days). "
    "Recorded sales are used as demand. Prices are real; costs and capacities are assumptions."
)

tabs = st.tabs(["Summary", "Forecasts", "Replenishment", "Capacity what-if", "Plan one evening"])

# ------------------------------------------------------------------ summary
with tabs[0]:
    acc = D["acc"].set_index("model")
    ma, hw, lp = head.loc[RULE_MA], head.loc[RULE_HW], head.loc[LP]
    c = st.columns(4)
    c[0].metric("Holt-Winters daily error (WAPE)", f"{acc.loc['holt_winters', 'wape_daily']:.1%}",
                f"{(acc.loc['holt_winters', 'wape_daily'] - acc.loc['moving_avg_7', 'wape_daily']) * 100:+.1f} "
                "pts vs 7-day average", delta_color="inverse")
    c[1].metric("Total cost, Holt-Winters + LP", f"${lp.total_cost_usd:,.0f}",
                f"{lp.total_cost_usd / hw.total_cost_usd - 1:+.1%} vs best Holt-Winters rule",
                delta_color="inverse")
    c[2].metric("Lost margin, Holt-Winters + LP", f"${lp.lost_margin_usd:,.0f}",
                f"{lp.lost_margin_usd / hw.lost_margin_usd - 1:+.1%} vs best Holt-Winters rule",
                delta_color="inverse")
    c[3].metric("Fill rate, Holt-Winters + LP", f"{lp.fill_rate:.1%}",
                f"rule: {hw.fill_rate:.1%}, same stock ({lp.days_of_supply:.2f} vs "
                f"{hw.days_of_supply:.2f} days)", delta_color="off", delta_arrow="off")

    show = D["head"][["family", "service_level", "fill_rate", "lost_units", "lost_margin_usd",
                      "holding_usd", "return_usd", "total_cost_usd", "days_of_supply"]].copy()
    show["service_level"] = show.service_level.map(lambda v: "no target" if pd.isna(v) else f"{v:.2%}")
    st.dataframe(show.rename(columns={
        "family": "Policy", "service_level": "Rule target", "fill_rate": "Fill rate",
        "lost_units": "Lost units", "lost_margin_usd": "Lost margin $", "holding_usd": "Holding $",
        "return_usd": "Returns $", "total_cost_usd": "Total cost $", "days_of_supply": "Avg stock (days)"
    }).style.format({"Fill rate": "{:.2%}", "Lost units": "{:,.0f}", "Lost margin $": "{:,.0f}",
                     "Holding $": "{:,.0f}", "Returns $": "{:,.0f}", "Total cost $": "{:,.0f}",
                     "Avg stock (days)": "{:.2f}"}), hide_index=True, width="stretch")
    st.markdown(
        f"All three share the same limits: **{A.DELIVERY_CAP:.2f}x** average daily demand can "
        f"arrive per day and **{A.STORAGE_CAP:g} days** of average demand fit on hand. Each rule "
        "uses the service target with the lowest total cost in the test window (its best case). "
        "The LP has no target to tune."
    )
    wd = (test.assign(weekday=test.date.dt.day_name()).groupby(["family", "weekday"])
          .agg(lost=("lost", "sum"), demand=("demand", "sum")).reset_index())
    wd["lost share"] = wd.lost / wd.demand
    fig = px.bar(wd, x="weekday", y="lost share", color="family", barmode="group",
                 category_orders={"weekday": WEEKDAYS, "family": [RULE_MA, RULE_HW, LP]},
                 color_discrete_map=COLORS, title="Share of demand lost, by weekday")
    fig.update_layout(yaxis_tickformat=".1%", legend_title=None, xaxis_title=None)
    st.plotly_chart(fig, width="stretch")

    st.subheader("Same steps on two more stores (base capacity)")
    ob = D["other_best"]
    ob = ob[ob.family.isin([RULE_MA, RULE_HW, LP])].pivot(index="store", columns="family",
                                                           values="total_cost_usd")[[RULE_MA, RULE_HW, LP]]
    oa = (D["other_acc"].pivot(index="store", columns="model", values="wape_daily")
          [["moving_avg_7", "holt_winters"]]
          .rename(columns={"moving_avg_7": "WAPE 7-day avg", "holt_winters": "WAPE Holt-Winters"}))
    other = pd.concat([oa, ob.add_prefix("Total cost $: ")], axis=1)
    st.dataframe(other.style.format({c: "{:.1%}" if c.startswith("WAPE") else "{:,.0f}"
                                     for c in other.columns}), width="stretch")

# ------------------------------------------------------------------ forecasts
with tabs[1]:
    a = D["acc"].sort_values("wape_daily", ascending=False)
    fig = px.bar(a, x="wape_daily", y="label", orientation="h", text=a.wape_daily.map("{:.1%}".format),
                 title="Daily forecast error (WAPE), 1 to 7 days ahead, test window")
    fig.update_traces(marker_color=["#2f6fb0" if m == "holt_winters" else "#c3c9cf" for m in a.model])
    fig.update_layout(xaxis_tickformat=".0%", yaxis_title=None, xaxis_title="WAPE (lower is better)")
    st.plotly_chart(fig, width="stretch")
    st.dataframe(D["acc"][["label", "wape_daily", "bias_daily", "wape_lead_time_total"]].rename(columns={
        "label": "Model", "wape_daily": "Daily WAPE", "bias_daily": "Bias",
        "wape_lead_time_total": f"WAPE of {A.LEAD_TIME}-day total"}).style.format(
        {"Daily WAPE": "{:.1%}", "Bias": "{:+.1%}", f"WAPE of {A.LEAD_TIME}-day total": "{:.1%}"}),
        hide_index=True, width="stretch")

    item = st.selectbox("Item", daily.columns, key="fc_item")
    i = list(daily.columns).index(item)
    sel = origins >= len(daily) - A.TEST_DAYS - 1
    dates = daily.index[origins[sel] + 1]
    fig = go.Figure()
    fig.add_bar(x=dates, y=daily[item].loc[dates].values, name="Actual sales", marker_color="#dfe3e8")
    fig.add_scatter(x=dates, y=fc[F.MODELS.index("holt_winters"), i, sel, 0], name="Holt-Winters",
                    line_color="#2f6fb0")
    fig.add_scatter(x=dates, y=fc[F.MODELS.index("moving_avg_7"), i, sel, 0], name="7-day moving average",
                    line=dict(color="#8c96a0", dash="dot"))
    fig.update_layout(title=f"{item}: next-day forecasts in the test window", yaxis_title="units")
    st.plotly_chart(fig, width="stretch")

# ------------------------------------------------------------------ replenishment
with tabs[2]:
    item = st.selectbox("Item", daily.columns, key="rep_item")
    d = test[test.item_id == item]
    fig = go.Figure()
    fig.add_bar(x=d[d.family == LP].date, y=d[d.family == LP].demand, name="Demand",
                marker_color="#dfe3e8")
    for fam in [RULE_HW, LP]:
        g = d[d.family == fam]
        fig.add_scatter(x=g.date, y=g.start_stock, name=f"{fam}: stock after delivery",
                        line_color=COLORS[fam])
    p = info.loc[item]
    fig.update_layout(title=f"{item} (${p.price:.2f}, lost margin ${p.lost_margin:.2f}/unit): "
                            "stock each morning vs demand", yaxis_title="units")
    st.plotly_chart(fig, width="stretch")
    st.dataframe(d.groupby("family").agg(demand=("demand", "sum"), lost=("lost", "sum"),
                                         avg_stock=("end_stock", "mean"))
                 .assign(fill_rate=lambda x: 1 - x.lost / x.demand).style.format(
                     {"demand": "{:,.0f}", "lost": "{:,.1f}", "avg_stock": "{:.1f}", "fill_rate": "{:.1%}"}),
                 width="stretch")

    tot = test.groupby(["family", "date"]).agg(received=("received", "sum"), start=("start_stock", "sum"),
                                               demand=("demand", "sum")).reset_index()
    c = st.columns(2)
    fig = px.line(tot, x="date", y="received", color="family", color_discrete_map=COLORS,
                  title="Units delivered per day, all items")
    fig.add_hline(y=A.DELIVERY_CAP * avg_total, line_dash="dash", annotation_text="delivery limit")
    fig.update_layout(legend_title=None, yaxis_title="units", legend_orientation="h")
    c[0].plotly_chart(fig, width="stretch")
    fig = px.line(tot, x="date", y="start", color="family", color_discrete_map=COLORS,
                  title="Units on hand after delivery, all items")
    fig.add_hline(y=A.STORAGE_CAP * avg_total, line_dash="dash", annotation_text="storage limit")
    fig.update_layout(legend_title=None, yaxis_title="units", legend_orientation="h")
    c[1].plotly_chart(fig, width="stretch")

# ------------------------------------------------------------------ capacity
with tabs[3]:
    b = D["best"][D["best"].family.isin([RULE_MA, RULE_HW, LP])]
    c = st.columns(2)
    for col, key, other, other_val, label in [
        (c[0], "delivery_cap_mult", "storage_cap_days", A.STORAGE_CAP, "delivery limit (x average daily demand)"),
        (c[1], "storage_cap_days", "delivery_cap_mult", A.DELIVERY_CAP, "storage limit (days of demand)"),
    ]:
        r = b[b[other] == other_val].sort_values(key)
        fig = px.line(r, x=key, y="total_cost_usd", color="family", markers=True, color_discrete_map=COLORS,
                      hover_data={"service_level": ":.2%", "fill_rate": ":.2%", "days_of_supply": ":.2f"},
                      title=f"Total cost vs {label.split(' (')[0]}")
        fig.update_layout(xaxis_title=label, yaxis_title="$ over 84 days", legend_title=None,
                          legend_orientation="h", yaxis_rangemode="tozero")
        col.plotly_chart(fig, width="stretch")
    st.markdown(
        "The LP gains the most when deliveries are tight. When the truck or the space runs short, "
        "the rule cuts every item by the same share; the LP keeps the biggest sellers in stock and "
        "lets the lowest-margin items run short. With plenty of storage, "
        "holding stock is almost free under these costs, and a rule with a very high target "
        "(99.99%) does better than the LP, whose buffers assume normally distributed errors."
    )

# ------------------------------------------------------------------ live plan
with tabs[4]:
    st.markdown("Re-solve the LP for one evening in the test window. Stock on hand and orders "
                "in transit come from the Holt-Winters + LP run.")
    lp_sim = sims[sims.family == LP]
    evenings = daily.index[len(daily) - A.TEST_DAYS: len(daily) - 1]
    c = st.columns(3)
    evening = c[0].selectbox("Evening", evenings, index=4, format_func=lambda d: d.strftime("%a %Y-%m-%d"))
    mult = c[1].slider("Delivery limit (x average daily demand)", 0.8, 1.6, float(A.DELIVERY_CAP), 0.05)
    sdays = c[2].slider("Storage limit (days of demand)", 2.0, 5.0, float(A.STORAGE_CAP), 0.25)

    t = daily.index.get_loc(evening)
    tonight = lp_sim[lp_sim.date == evening].set_index("item_id").reindex(daily.columns)
    on_hand = tonight.end_stock.to_numpy()
    pipeline = (lp_sim[lp_sim.date == daily.index[t - 1]].set_index("item_id")
                .reindex(daily.columns).ordered.to_numpy()[:, None])
    ctx = S.build_context(daily, info, origins, fc, mult, sdays)
    with st.spinner("Solving the LP..."):
        res = S.LPPlanner("holt_winters").plan(ctx, t, on_hand, pipeline)
    rule_z = head.loc[RULE_HW].service_level
    rule = S.OrderUpTo("holt_winters", rule_z).decide(ctx, t, on_hand.copy(), pipeline.copy())

    order_days = [(evening + pd.Timedelta(days=j)).strftime("%a %m-%d") for j in range(res["orders"].shape[1])]
    f = ctx.forecast("holt_winters", t)
    arrive = pd.DataFrame({
        "evening ordered": order_days,
        "LP plan (units)": res["orders"].sum(axis=0),
        "forecast demand on arrival day": f[:, A.LEAD_TIME - 1:].sum(axis=0),
    })
    fig = go.Figure()
    fig.add_bar(x=arrive["evening ordered"], y=arrive["LP plan (units)"], name="LP order plan",
                marker_color=COLORS[LP])
    fig.add_scatter(x=arrive["evening ordered"], y=arrive["forecast demand on arrival day"],
                    name="forecast demand on the arrival day", line_color="#333")
    fig.add_hline(y=ctx.delivery_cap, line_dash="dash", annotation_text="delivery limit")
    fig.update_layout(title="Orders planned for each evening, all items (only tonight's are placed)",
                      yaxis_title="units", legend_orientation="h")
    st.plotly_chart(fig, width="stretch")

    table = pd.DataFrame({
        "on hand tonight": on_hand, "arriving tomorrow": pipeline[:, 0],
        "forecast next 2 days": f[:, : A.LEAD_TIME].sum(axis=1),
        "LP order tonight": res["orders"][:, 0], f"rule order tonight ({rule_z:.2%})": rule,
        "lost margin $/unit": info.lost_margin.to_numpy(),
    }, index=daily.columns)
    st.dataframe(table.round(1), width="stretch")
