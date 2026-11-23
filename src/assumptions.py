"""All inputs in one place. Values marked "assumption" are not in the data."""

# ---- Data ----
M5_URL = "https://raw.githubusercontent.com/Nixtla/m5-forecasts/main/datasets/m5.zip"
STORE = "CA_1"            # one Walmart store in California
DEPT = "FOODS_3"          # largest food department in M5
N_ITEMS = 50              # top items by sales in the last 365 days
MAX_ZERO_SHARE = 0.10     # skip items with more than 10% zero-sales days in the last 2 years
HISTORY_DAYS = 730        # days kept before the end of the data
CHECK_STORES = ["TX_1", "WI_1"]   # same steps on two more stores, base capacity only

# ---- Backtest ----
TEST_DAYS = 84            # last 12 weeks: 2016-02-29 to 2016-05-22
ERROR_WINDOW = 56         # recent forecast errors used for safety stock and scenarios
WARMUP_DAYS = 14          # policies run 2 weeks before the test window so starting stock does not matter
FIT_WINDOW = 364          # each forecast is fit on the last 52 weeks before the origin
HORIZON = 7               # days ahead the planner looks
SEASON = 7                # weekly pattern in daily sales

# ---- Replenishment ----
LEAD_TIME = 2             # assumption: an order placed after close on day t can be sold from day t+2
GROSS_MARGIN = 0.30       # assumption: unit cost = 70% of shelf price (grocery-like margin)
HOLDING_RATE = 0.25       # assumption: 25% of unit cost per year (capital, space, handling)
DELIVERY_CAP = 1.05       # assumption: units received per day, as a multiple of average daily demand
STORAGE_CAP = 3.0         # assumption: units on hand after a delivery, as days of average demand
RETURN_COST = 0.10        # assumption: $ per delivered unit that does not fit and goes back on the truck
SERVICE_LEVELS = [0.90, 0.95, 0.975, 0.99, 0.995, 0.999, 0.9999]   # targets tried for the rules
N_SCENARIOS = 20          # demand paths in the scenario LP variant
SEED = 7
