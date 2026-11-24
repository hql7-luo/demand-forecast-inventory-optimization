"""Build the browser demo: data.json, manifest.json and the static files.

    python scripts/build_web_demo.py                       # writes web_demo/
    python scripts/build_web_demo.py --out ../site/demos/replenishment

The demo re-solves the same LP as src/inventory_lp.py in the browser (HiGHS compiled
to WebAssembly) and replays the next days on real sales. Reference values computed
here in Python are stored in manifest.json so the JavaScript port can be checked.
"""
import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import assumptions as A  # noqa: E402
import forecast as F  # noqa: E402
import run_all as R  # noqa: E402
import simulate as S  # noqa: E402
from inventory_lp import Z_BREAKS, plan_orders, shortage_segments  # noqa: E402

SRC = ROOT / "web_demo"
STATIC = ["index.html", "demo.css", "demo.js", "model.js", "vendor/highs.js", "vendor/highs.wasm", "vendor/LICENSE-highs.txt"]
DIGITS = 3
CHECK_EVENINGS = 6          # evenings with stored reference solutions
REPLAY_DAYS = 7


def rnd(a):
    return np.round(np.asarray(a, float), DIGITS)


def lst(a):
    return [float(v) for v in np.ravel(a)] if np.ndim(a) == 1 else [lst(r) for r in a]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def replay(policy, ev, start, days, info, delivery_cap, storage_cap, items, plan_fn):
    """Same day-by-day rules as simulate.simulate, starting from one evening's LP-run state.
    plan_fn(evening, on_hand, pipeline) -> tonight's orders."""
    n_items = len(items)
    on_hand = np.array(ev[start]["on_hand"], float)
    arriving = np.array(ev[start]["pipeline"], float)          # arrives the next morning
    lost_margin = info["lost_margin"].to_numpy()
    holding = info["holding_per_day"].to_numpy()
    totals = dict(demand=0.0, lost_units=0.0, lost_usd=0.0, holding_usd=0.0, returned_units=0.0)
    for d in range(days):
        e = ev[start + d]
        q = np.asarray(plan_fn(e, on_hand.copy(), arriving[:, None].copy()), float)
        incoming = arriving
        room = max(storage_cap - on_hand.sum(), 0.0)
        refused = np.zeros(n_items)
        if incoming.sum() > room:
            keep = incoming * room / incoming.sum()
            refused, incoming = incoming - keep, keep
        on_hand = on_hand + incoming
        demand = np.array([row[0] for row in e["actual"]], float)
        sales = np.minimum(on_hand, demand)
        on_hand = on_hand - sales
        lost = demand - sales
        totals["demand"] += demand.sum()
        totals["lost_units"] += lost.sum()
        totals["lost_usd"] += float(lost @ lost_margin)
        totals["holding_usd"] += float(on_hand @ holding)
        totals["returned_units"] += refused.sum()
        arriving = q
    totals["returns_usd"] = totals["returned_units"] * A.RETURN_COST
    totals["total_usd"] = totals["lost_usd"] + totals["holding_usd"] + totals["returns_usd"]
    totals["fill_rate"] = 1 - totals["lost_units"] / totals["demand"]
    return {k: float(v) for k, v in totals.items()}


def rule_orders(e, on_hand, pipeline, z, delivery_cap, storage_cap):
    """OrderUpTo.decide from simulate.py, on the exported (rounded) inputs."""
    L = A.LEAD_TIME
    f = np.array(e["forecast"], float)
    sigma = np.array(e["sigma"], float)
    position = on_hand + pipeline.sum(axis=1)
    q = np.clip(f[:, :L].sum(axis=1) + z * sigma - position, 0, None)
    if q.sum() > delivery_cap:
        q *= delivery_cap / q.sum()
    at_arrival = np.clip(position - f[:, : L - 1].sum(axis=1), 0, None)
    room = max(storage_cap - at_arrival.sum(), 0)
    if q.sum() > room:
        q *= room / q.sum()
    return q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=SRC)
    args = ap.parse_args()

    daily, info = R.load()
    origins, fc = R.get_forecasts(daily)
    ctx = S.build_context(daily, info, origins, fc)
    n = len(daily)
    head = pd.read_csv(R.RESULTS / "headline_policies.csv")
    best = pd.read_csv(R.RESULTS / "best_per_family.csv")
    acc = pd.read_csv(R.RESULTS / "forecast_accuracy.csv")
    sims = pd.read_csv(R.RESULTS / "simulation_daily.csv.gz", parse_dates=["date"])
    lp_name = head.set_index("family").loc["Holt-Winters + LP", "policy"]
    rule_z = float(head.set_index("family").loc["Holt-Winters + rule", "service_level"])
    lp_sim = sims[sims.policy == lp_name]
    items = list(daily.columns)

    def state(day):
        return lp_sim[lp_sim.date == daily.index[day]].set_index("item_id").reindex(items)

    hw = F.MODELS.index("holt_winters")
    evenings = []
    for t in range(n - A.TEST_DAYS, n - 1):            # evenings of the test window
        actual = daily.to_numpy(float).T[:, t + 1: t + 1 + A.HORIZON]
        if actual.shape[1] < A.HORIZON:
            actual = np.pad(actual, ((0, 0), (0, A.HORIZON - actual.shape[1])), constant_values=np.nan)
        evenings.append({
            "date": daily.index[t].strftime("%Y-%m-%d"),
            "on_hand": lst(rnd(state(t).end_stock)),
            "pipeline": lst(rnd(state(t - 1).ordered)),
            "forecast": lst(rnd(fc[hw][:, np.searchsorted(origins, t), :])),
            "sigma": lst(rnd(S.lead_time_sigma(ctx, "holt_winters", t))),
            "actual": [[None if np.isnan(v) else float(v) for v in row] for row in actual],
        })

    width, slope = shortage_segments()
    avg_total = float(info.avg_daily_28d.sum())
    data = {
        "meta": {
            "store": A.STORE, "dept": A.DEPT, "items": len(items),
            "test_start": daily.index[n - A.TEST_DAYS].strftime("%Y-%m-%d"),
            "test_end": daily.index[-1].strftime("%Y-%m-%d"),
            "lead_time": A.LEAD_TIME, "horizon": A.HORIZON, "avg_daily_total": avg_total,
            "base_delivery_mult": A.DELIVERY_CAP, "base_storage_days": A.STORAGE_CAP,
            "return_cost": A.RETURN_COST, "gross_margin": A.GROSS_MARGIN, "holding_rate": A.HOLDING_RATE,
            "segment_width": lst(width), "segment_slope": lst(slope), "z_breaks": lst(Z_BREAKS),
            "rule_service_level": rule_z, "rule_z": float(S.norm.ppf(rule_z)),
        },
        "items": [{"id": i, "price": float(info.loc[i, "price"]), "lost_margin": float(info.loc[i, "lost_margin"]),
                   "holding": float(info.loc[i, "holding_per_day"]), "avg": float(info.loc[i, "avg_daily_28d"])}
                  for i in items],
        "evenings": evenings,
        "results": {
            "headline": head[["family", "service_level", "fill_rate", "lost_units", "lost_margin_usd", "holding_usd",
                              "return_usd", "total_cost_usd", "days_of_supply"]].round(4).to_dict("records"),
            "capacity": best[best.family.isin(head.family)][["family", "delivery_cap_mult", "storage_cap_days",
                                                              "service_level", "fill_rate", "total_cost_usd"]].round(4).to_dict("records"),
            "forecast": acc[["model", "label", "wape_daily", "wape_lead_time_total"]].round(4).to_dict("records"),
        },
    }
    data["results"]["headline"] = [{k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in r.items()}
                                   for r in data["results"]["headline"]]
    data["results"]["capacity"] = [{k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in r.items()}
                                   for r in data["results"]["capacity"]]

    # ---- reference solutions on the exported inputs (what the browser sees)
    lm = info["lost_margin"].to_numpy()
    hold = info["holding_per_day"].to_numpy()
    K, C = A.DELIVERY_CAP * avg_total, A.STORAGE_CAP * avg_total

    def lp_plan(e, on_hand, pipeline, K=K, C=C):
        return plan_orders(on_hand, pipeline, np.array(e["forecast"]), np.array(e["sigma"]), lm, hold, K, C)

    picks = np.linspace(0, len(evenings) - REPLAY_DAYS - 1, CHECK_EVENINGS).round().astype(int)
    checks = []
    for s in picks:
        e = evenings[s]
        on_hand, pipeline = np.array(e["on_hand"]), np.array(e["pipeline"])[:, None]
        res = lp_plan(e, on_hand, pipeline)
        tight = lp_plan(e, on_hand, pipeline, K=0.95 * avg_total, C=2.5 * avg_total)
        checks.append({
            "evening": int(s), "date": e["date"],
            "lp_objective": float(res["objective"]), "lp_objective_tight": float(tight["objective"]),
            "lp_tonight_total": float(res["orders"][:, 0].sum()),
            "rule_tonight": lst(rnd(rule_orders(e, on_hand, pipeline, data["meta"]["rule_z"], K, C))),
            "rule_replay": replay("rule", evenings, s, REPLAY_DAYS, info, K, C, items,
                                  lambda e, oh, pl: rule_orders(e, oh, pl, data["meta"]["rule_z"], K, C)),
            "lp_replay": replay("lp", evenings, s, REPLAY_DAYS, info, K, C, items,
                                lambda e, oh, pl: lp_plan(e, oh, pl)["orders"][:, 0]),
        })
        print(f"{e['date']}: LP objective {res['objective']:.4f}, rule replay "
              f"${checks[-1]['rule_replay']['total_usd']:.2f}, LP replay ${checks[-1]['lp_replay']['total_usd']:.2f}")

    # Every 7-day window in the test period, so a single replayed week can be put in context.
    weekly = []
    for setting, dm, sd in [("base", A.DELIVERY_CAP, A.STORAGE_CAP), ("tight_delivery", 0.95, A.STORAGE_CAP)]:
        Kw, Cw = dm * avg_total, sd * avg_total
        diffs, rule_total = [], 0.0
        for s in range(len(evenings) - REPLAY_DAYS + 1):
            rr = replay("rule", evenings, s, REPLAY_DAYS, info, Kw, Cw, items,
                        lambda e, oh, pl: rule_orders(e, oh, pl, data["meta"]["rule_z"], Kw, Cw))
            lr = replay("lp", evenings, s, REPLAY_DAYS, info, Kw, Cw, items,
                        lambda e, oh, pl: lp_plan(e, oh, pl, K=Kw, C=Cw)["orders"][:, 0])
            diffs.append(lr["total_usd"] - rr["total_usd"])
            rule_total += rr["total_usd"]
        weekly.append({"setting": setting, "delivery_mult": dm, "storage_days": sd, "weeks": len(diffs),
                       "lp_cheaper_weeks": int(sum(d < 0 for d in diffs)),
                       "avg_weekly_diff_usd": round(float(np.mean(diffs)), 2),
                       "diff_share_of_rule": round(float(sum(diffs) / rule_total), 4)})
        print(setting, weekly[-1])
    data["results"]["weekly"] = weekly

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "vendor").mkdir(exist_ok=True)
    text = json.dumps(data, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    (out / "data.json").write_text(text)
    if out.resolve() != SRC.resolve():
        for f in STATIC:
            shutil.copyfile(SRC / f, out / f)
    manifest = {
        "source": "https://github.com/hql7-luo/demand-forecast-inventory-optimization",
        "solver": "HiGHS 1.16.0 (highs-js 1.16.0, MIT) in the browser; HiGHS 1.16.0 via PuLP in Python",
        "data_sha256": sha(out / "data.json"),
        "files_sha256": {f: sha(out / f) for f in STATIC},
        "evenings": len(evenings), "items": len(items), "decimals": DIGITS,
        "checks": checks,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"Wrote {out} ({len(text) / 1e3:.0f} KB data, {len(evenings)} evenings)")


if __name__ == "__main__":
    main()
