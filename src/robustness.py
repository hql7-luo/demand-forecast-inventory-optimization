"""Same forecasts and policies on two more stores (base capacity only).

  python src/robustness.py
"""
import os

for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(var, "1")

from multiprocessing import Pool

import pandas as pd

import assumptions as A
import forecast as F
import run_all as R


def main():
    acc_rows, best_rows = [], []
    for store in A.CHECK_STORES:
        data_dir = R.DATA / "other_stores" / store
        fc_path = R.RESULTS / "other_stores" / f"forecasts_{store}.npz"
        daily, info = R.load(data_dir)
        origins, fc = R.get_forecasts(daily, path=fc_path)
        ev = R.evaluation_mask(origins, len(daily))
        acc = F.accuracy_table(fc[:, :, ev], F.actuals_for(daily, origins)[:, ev])
        acc_rows.append(acc.assign(store=store, n_items=daily.shape[1]))

        jobs = [(p, A.DELIVERY_CAP, A.STORAGE_CAP, False, data_dir, fc_path) for p in R.policies()]
        with Pool(os.cpu_count() or 1) as pool:
            runs = pd.DataFrame([r for r, _ in pool.map(R._run, jobs, chunksize=1)])
        best_rows.append(R.best_per_family(runs).assign(store=store, n_items=daily.shape[1]))
        print(store, "done")

    acc = pd.concat(acc_rows, ignore_index=True)
    best = pd.concat(best_rows, ignore_index=True)
    acc.to_csv(R.RESULTS / "other_stores" / "forecast_accuracy.csv", index=False)
    best.to_csv(R.RESULTS / "other_stores" / "best_per_family.csv", index=False)
    print(acc[acc.model.isin(["holt_winters", "weekday_avg_4w", "moving_avg_7"])]
          [["store", "model", "wape_daily", "wape_lead_time_total"]].round(4).to_string(index=False))
    print(best[["store", "family", "service_level", "fill_rate", "lost_margin_usd",
                "total_cost_usd", "days_of_supply"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
