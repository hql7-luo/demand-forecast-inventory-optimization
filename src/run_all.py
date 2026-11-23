"""Run the whole analysis.

  python src/run_all.py            # reuse saved forecasts if they match the settings
  python src/run_all.py --refit    # refit every forecast (about 3 minutes on 4 cores)
  python src/run_all.py --quick    # base capacity only, prints results, writes nothing (CI)
"""
import os

# One thread per process: work is already spread over one process per core.
for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(var, "1")

import argparse
from multiprocessing import Pool
from pathlib import Path
import time

import numpy as np
import pandas as pd

import assumptions as A
import forecast as F
import simulate as S

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
FORECASTS = RESULTS / "forecasts.npz"

DELIVERY_GRID = [0.95, 1.05, 1.15, 1.30]    # x average daily demand, storage at base
STORAGE_GRID = [2.5, 3.0, 4.0]              # days of average demand, delivery at base
HEADLINE = ["Moving average + rule", "Holt-Winters + rule", "Holt-Winters + LP"]


def load(data_dir: Path = DATA):
    daily = pd.read_csv(data_dir / "daily_sales.csv", index_col=0, parse_dates=True)
    info = pd.read_csv(data_dir / "items.csv", index_col=0)
    return daily, info


def origins_for(n_days: int) -> np.ndarray:
    """Every evening from the start of the first error window to the second-to-last day."""
    first_decision = n_days - A.TEST_DAYS - A.WARMUP_DAYS - 1
    return np.arange(first_decision - A.HORIZON - A.ERROR_WINDOW + 1, n_days - 1)


def get_forecasts(daily, refit: bool = False, path: Path = FORECASTS):
    origins = origins_for(len(daily))
    if path.exists() and not refit:
        z = np.load(path, allow_pickle=False)
        if (list(z["models"]) == F.MODELS and np.array_equal(z["origins"], origins)
                and list(z["items"]) == list(daily.columns)):
            return origins, z["fc"].astype(float)
    t = time.time()
    fc = F.backtest(daily, origins, processes=os.cpu_count() or 1)
    print(f"Backtest: {fc.shape[0]} models x {fc.shape[1]} items x {fc.shape[2]} origins "
          f"in {time.time() - t:.0f}s")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, fc=fc.astype(np.float32), origins=origins,
                        models=np.array(F.MODELS, dtype=str),
                        items=np.array(daily.columns, dtype=str))
    return origins, fc


def evaluation_mask(origins, n_days):
    """Origins whose full 7-day horizon falls inside the test window."""
    return (origins >= n_days - A.TEST_DAYS - 1) & (origins <= n_days - 1 - A.HORIZON)


def _run(job):
    policy, delivery_mult, storage_days, keep_daily, *where = job
    data_dir, fc_path = where if where else (DATA, FORECASTS)
    daily, info = load(data_dir)
    origins, fc = get_forecasts(daily, path=fc_path)
    ctx = S.build_context(daily, info, origins, fc, delivery_mult, storage_days)
    t = time.time()
    sim = S.simulate(policy, ctx)
    row = S.summarize(sim, info)
    row.update(family=policy.family, model=policy.model, service_level=policy.service_level,
               delivery_cap_mult=delivery_mult, storage_cap_days=storage_days,
               seconds=time.time() - t)
    return row, (sim if keep_daily else None)


def policies():
    levels = A.SERVICE_LEVELS
    out = [S.OrderUpTo(m, z) for m in ("moving_avg_7", "holt_winters") for z in levels]
    out += [S.LPPlanner("holt_winters"), S.LPPlanner("moving_avg_7")]
    return out


def best_per_family(runs: pd.DataFrame) -> pd.DataFrame:
    """Lowest-cost row per policy family and capacity setting. For the rules this picks
    the service target after seeing the test window, which is their best case."""
    keys = ["family", "delivery_cap_mult", "storage_cap_days"]
    return runs.loc[runs.groupby(keys).total_cost_usd.idxmin()].reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refit", action="store_true")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    daily, info = load()
    origins, fc = get_forecasts(daily, args.refit)
    act = F.actuals_for(daily, origins)
    ev = evaluation_mask(origins, len(daily))
    acc = F.accuracy_table(fc[:, :, ev], act[:, ev])
    print(acc.round(4).to_string(index=False))

    base = (A.DELIVERY_CAP, A.STORAGE_CAP)
    settings = [base] if args.quick else sorted(
        {(d, A.STORAGE_CAP) for d in DELIVERY_GRID} | {(A.DELIVERY_CAP, s) for s in STORAGE_GRID})
    jobs = [(p, d, s, (d, s) == base) for d, s in settings for p in policies()]
    if not args.quick:
        jobs.insert(0, (S.ScenarioLP(), *base, False))   # slowest job first

    t = time.time()
    with Pool(os.cpu_count() or 1) as pool:
        out = pool.map(_run, jobs, chunksize=1)
    print(f"Simulations: {len(out)} runs in {time.time() - t:.0f}s")

    runs = pd.DataFrame([r for r, _ in out])
    best = best_per_family(runs)
    head = best[(best.delivery_cap_mult == base[0]) & (best.storage_cap_days == base[1])
                & best.family.isin(HEADLINE)].set_index("family").loc[HEADLINE].reset_index()
    cols = ["family", "policy", "fill_rate", "lost_units", "lost_margin_usd", "holding_usd",
            "return_usd", "total_cost_usd", "days_of_supply"]
    print(best[(best.delivery_cap_mult == base[0]) & (best.storage_cap_days == base[1])][cols]
          .round(3).to_string(index=False))
    if args.quick:
        return

    acc.to_csv(RESULTS / "forecast_accuracy.csv", index=False)
    runs.to_csv(RESULTS / "policy_runs.csv", index=False)
    best.to_csv(RESULTS / "best_per_family.csv", index=False)
    head.to_csv(RESULTS / "headline_policies.csv", index=False)
    sims = pd.concat([s for _, s in out if s is not None], ignore_index=True)
    sims = sims[sims.policy.isin(head.policy)]
    num = sims.select_dtypes("number").columns
    sims[num] = sims[num].round(3)
    sims.to_csv(RESULTS / "simulation_daily.csv.gz", index=False)


if __name__ == "__main__":
    main()
