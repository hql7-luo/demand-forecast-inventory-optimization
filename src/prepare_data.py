"""Download the M5 data and keep one store, one department, top items.

Writes:
  data/processed/daily_sales.csv  date x item units sold (store-closed days filled)
  data/processed/calendar.csv     weekday, events, SNAP days for the kept dates
  data/processed/items.csv        price and cost inputs per item
"""
from pathlib import Path
import urllib.request
import zipfile

import numpy as np
import pandas as pd

import assumptions as A

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"


def download(zip_path: Path) -> None:
    if zip_path.exists():
        return
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading M5 data (about 50 MB) from {A.M5_URL}")
    urllib.request.urlretrieve(A.M5_URL, zip_path)


def select_items(sales: pd.DataFrame, day_cols: list[str]) -> pd.Index:
    """Top items by last-365-day sales, skipping items with many zero days."""
    last_year = sales[day_cols[-365:]].sum(axis=1)
    zero_share = (sales[day_cols[-A.HISTORY_DAYS:]] == 0).mean(axis=1)
    keep = last_year[zero_share <= A.MAX_ZERO_SHARE]
    return keep.sort_values(ascending=False).index[: A.N_ITEMS]


def fill_closed_days(daily: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """The store is closed on Christmas. Replace those zeros with the average of the
    same weekday one week before and one week after, so models do not learn a fake drop."""
    closed = daily.sum(axis=1) == 0
    filled = daily.copy()
    for day in daily.index[closed]:
        before, after = day - pd.Timedelta(days=7), day + pd.Timedelta(days=7)
        neighbors = [d for d in (before, after) if d in daily.index]
        filled.loc[day] = daily.loc[neighbors].mean().round()
    return filled, closed


def build(zip_path: Path, store: str = A.STORE, dept: str = A.DEPT):
    """Daily sales, calendar and item inputs for one store and department."""
    with zipfile.ZipFile(zip_path) as z:
        sales = pd.read_csv(z.open("sales_train_evaluation.csv"))
        calendar = pd.read_csv(z.open("calendar.csv"), parse_dates=["date"])
        prices = pd.concat(
            chunk[chunk.store_id == store]
            for chunk in pd.read_csv(z.open("sell_prices.csv"), chunksize=1_000_000)
        )

    sales = sales[(sales.store_id == store) & (sales.dept_id == dept)].set_index("item_id")
    day_cols = [c for c in sales.columns if c.startswith("d_")]
    items = select_items(sales, day_cols)

    kept_days = day_cols[-A.HISTORY_DAYS:]
    # Calendar row k is day d_{k+1} (this copy of the calendar has no "d" column).
    calendar["d"] = [f"d_{k + 1}" for k in range(len(calendar))]
    cal = calendar.set_index("d").loc[kept_days].reset_index()
    daily = sales.loc[items, kept_days].T
    daily.index = pd.DatetimeIndex(cal["date"].values, name="date")
    daily, closed = fill_closed_days(daily)

    # Price: median weekly shelf price over the 13 weeks before the test window.
    test_start = daily.index[-A.TEST_DAYS]
    weeks = cal.loc[cal.date < test_start, "wm_yr_wk"].drop_duplicates().iloc[-13:]
    p = prices[prices.item_id.isin(items) & prices.wm_yr_wk.isin(weeks)]
    price = p.groupby("item_id").sell_price.median().reindex(items)

    pre_test = daily.loc[daily.index < test_start]
    info = pd.DataFrame(index=items)
    info.index.name = "item_id"
    info["price"] = price.round(2)
    info["unit_cost"] = (price * (1 - A.GROSS_MARGIN)).round(4)
    info["lost_margin"] = (price * A.GROSS_MARGIN).round(4)
    info["holding_per_day"] = (info.unit_cost * A.HOLDING_RATE / 365).round(6)
    info["avg_daily_28d"] = pre_test.iloc[-28:].mean().round(2)
    info["zero_share_2y"] = (sales.loc[items, kept_days] == 0).mean(axis=1).round(3)

    state = store.split("_")[0]
    cal_out = cal[["date", "weekday", "event_name_1", "event_type_1", f"snap_{state}"]].copy()
    cal_out["store_closed"] = closed.values
    return daily.astype(int), cal_out, info, int(closed.sum())


def write(out_dir: Path, store: str = A.STORE) -> None:
    zip_path = RAW / "m5.zip"
    download(zip_path)
    daily, cal, info, n_closed = build(zip_path, store)
    out_dir.mkdir(parents=True, exist_ok=True)
    daily.to_csv(out_dir / "daily_sales.csv")
    cal.to_csv(out_dir / "calendar.csv", index=False)
    info.to_csv(out_dir / "items.csv")
    print(f"{store}: kept {daily.shape[1]} items, {len(daily)} days "
          f"({daily.index[0].date()} to {daily.index[-1].date()}), "
          f"{n_closed} store-closed days filled.")


def main() -> None:
    write(OUT)
    for store in A.CHECK_STORES:
        write(OUT / "other_stores" / store, store)


if __name__ == "__main__":
    main()
