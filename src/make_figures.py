"""Figures for the README, from the saved results.

  python src/make_figures.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import assumptions as A
import forecast as F

ROOT = Path(__file__).resolve().parents[1]
RESULTS, FIGURES, DATA = ROOT / "results", ROOT / "figures", ROOT / "data" / "processed"

COLORS = {
    "Moving average + rule": "#8c96a0",
    "Holt-Winters + rule": "#2f6fb0",
    "Holt-Winters + LP": "#d9682b",
    "Moving average + LP": "#c9b8a8",
    "Holt-Winters + scenario LP": "#7a5aa6",
}
plt.rcParams.update({
    "figure.dpi": 150, "savefig.bbox": "tight", "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.alpha": 0.25, "axes.titleweight": "bold", "axes.titlesize": 11,
})


def save(fig, name):
    FIGURES.mkdir(exist_ok=True)
    fig.savefig(FIGURES / name)
    plt.close(fig)


def forecast_accuracy():
    acc = pd.read_csv(RESULTS / "forecast_accuracy.csv").sort_values("wape_daily", ascending=False)
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    colors = [COLORS["Holt-Winters + rule"] if m == "holt_winters" else "#c3c9cf" for m in acc.model]
    ax.barh(acc.label, acc.wape_daily, color=colors)
    for y, v in enumerate(acc.wape_daily):
        ax.text(v + 0.005, y, f"{v:.1%}", va="center")
    ax.set_xlim(0, acc.wape_daily.max() * 1.15)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_xlabel("WAPE, lower is better")
    ax.set_title(f"Daily forecast error, 50 items, 1 to 7 days ahead, last {A.TEST_DAYS} days")
    ax.grid(axis="y", visible=False)
    save(fig, "1_forecast_accuracy.png")


def forecast_example():
    daily = pd.read_csv(DATA / "daily_sales.csv", index_col=0, parse_dates=True)
    z = np.load(RESULTS / "forecasts.npz")
    origins, fc = z["origins"], z["fc"]
    item = daily.columns[0]
    sel = origins >= len(daily) - A.TEST_DAYS - 1
    dates = daily.index[origins[sel] + 1]
    fig, ax = plt.subplots(figsize=(9, 3.2))
    ax.bar(dates, daily[item].loc[dates], color="#dfe3e8", label="Actual sales", width=0.9)
    ax.plot(dates, fc[F.MODELS.index("moving_avg_7"), 0, sel, 0], color="#8c96a0", ls=":",
            lw=1.6, label="7-day moving average")
    ax.plot(dates, fc[F.MODELS.index("holt_winters"), 0, sel, 0],
            color=COLORS["Holt-Winters + rule"], lw=1.6, label="Holt-Winters")
    ax.set_ylabel("units")
    ax.set_title(f"{item}, the top seller: next-day forecasts in the test window")
    ax.legend(ncol=3, frameon=False, loc="upper left")
    save(fig, "2_forecast_example.png")


def cost_vs_stock():
    runs = pd.read_csv(RESULTS / "policy_runs.csv")
    base = runs[(runs.delivery_cap_mult == A.DELIVERY_CAP) & (runs.storage_cap_days == A.STORAGE_CAP)]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, col, label in [(axes[0], "lost_margin_usd", "lost margin, $ over 84 days"),
                           (axes[1], "total_cost_usd", "total cost, $ over 84 days")]:
        for fam in ("Moving average + rule", "Holt-Winters + rule"):
            r = base[base.family == fam].sort_values("service_level")
            ax.plot(r.days_of_supply, r[col], "o-", color=COLORS[fam], label=f"{fam} (targets 90% to 99.99%)")
            for _, row in r.iloc[[0, -1]].iterrows():
                ax.annotate(f"{row.service_level:.4g}".replace("0.", "."), (row.days_of_supply, row[col]),
                            textcoords="offset points", xytext=(4, 4), fontsize=8, color=COLORS[fam])
        for fam, marker in [("Holt-Winters + LP", "*"), ("Moving average + LP", "P"),
                            ("Holt-Winters + scenario LP", "X")]:
            r = base[base.family == fam]
            ax.scatter(r.days_of_supply, r[col], s=220 if marker == "*" else 90, marker=marker,
                       color=COLORS[fam], label=fam, zorder=5, edgecolor="white")
        ax.set_xlabel("average stock, days of demand")
        ax.set_ylabel(label)
        ax.set_ylim(0, None)
    axes[0].set_title("Same stock, less margin lost")
    axes[1].set_title("Total cost = lost margin + holding + returns")
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=8, ncol=3, loc="lower center",
               bbox_to_anchor=(0.5, -0.17))
    fig.suptitle(f"Base capacity: deliveries {A.DELIVERY_CAP:.2f}x average demand per day, "
                 f"storage {A.STORAGE_CAP:g} days", y=1.02)
    save(fig, "3_cost_vs_stock.png")


def capacity():
    best = pd.read_csv(RESULTS / "best_per_family.csv")
    fams = ["Moving average + rule", "Holt-Winters + rule", "Holt-Winters + LP"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    d = best[best.storage_cap_days == A.STORAGE_CAP]
    s = best[best.delivery_cap_mult == A.DELIVERY_CAP]
    for fam in fams:
        r = d[d.family == fam].sort_values("delivery_cap_mult")
        axes[0].plot(r.delivery_cap_mult, r.total_cost_usd, "o-", color=COLORS[fam], label=fam)
        r = s[s.family == fam].sort_values("storage_cap_days")
        axes[1].plot(r.storage_cap_days, r.total_cost_usd, "o-", color=COLORS[fam], label=fam)
    axes[0].set_xlabel("delivery limit, x average daily demand (storage 3 days)")
    axes[1].set_xlabel(f"storage limit, days of demand (deliveries {A.DELIVERY_CAP:.2f}x)")
    axes[0].set_ylabel("total cost, $ over 84 days")
    axes[0].set_title("Tighter deliveries: the LP gains the most")
    axes[1].set_title("More storage: high-target rules catch up")
    axes[1].legend(frameon=False)
    for ax in axes:
        ax.set_ylim(0, None)
    fig.text(0.5, -0.06, "Rules use their lowest-cost service target in each setting, chosen after seeing the test window.",
             ha="center", fontsize=8, color="#555")
    save(fig, "4_capacity.png")


def where_lp_saves():
    """Lost margin per item, Holt-Winters + LP minus the best Holt-Winters rule."""
    sims = pd.read_csv(RESULTS / "simulation_daily.csv.gz", parse_dates=["date"])
    head = pd.read_csv(RESULTS / "headline_policies.csv").set_index("family")
    info = pd.read_csv(DATA / "items.csv", index_col=0)
    t = sims[sims.test].merge(info[["lost_margin"]], left_on="item_id", right_index=True)
    t["lost_usd"] = t.lost * t.lost_margin
    by = t.pivot_table(index="item_id", columns="policy", values="lost_usd", aggfunc="sum")
    diff = (by[head.loc["Holt-Winters + LP", "policy"]] - by[head.loc["Holt-Winters + rule", "policy"]])
    total = diff.sum()
    diff = diff[diff.abs() > 1].sort_values()
    rank = info.avg_daily_28d.rank(ascending=False, method="first").astype(int)
    labels = [f"{i}  (#{rank[i]} seller, \\${info.lost_margin[i]:.2f}/unit)" for i in diff.index]
    fig, ax = plt.subplots(figsize=(8, 0.28 * len(diff) + 1.2))
    ax.barh(labels, diff, color=[COLORS["Holt-Winters + LP"] if v < 0 else "#9aa5b1" for v in diff])
    ax.axvline(0, color="#333", lw=0.8)
    ax.set_xlabel("change in lost margin, \\$ over 84 days (negative = LP lost less)")
    sign = "-" if total < 0 else "+"
    ax.set_title(f"Lost margin, LP minus best Holt-Winters rule: {sign}\\${abs(total):.0f} in all\n"
                 "Biggest savings on the top sellers; it gives up some sales of \\$0.24 items",
                 fontsize=10)
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", labelsize=8)
    save(fig, "5_where_lp_saves.png")


def main():
    forecast_accuracy()
    forecast_example()
    cost_vs_stock()
    capacity()
    where_lp_saves()
    print("Figures written to", FIGURES)


if __name__ == "__main__":
    main()
