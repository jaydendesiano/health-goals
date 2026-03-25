import numpy as np
import pandas as pd


rng = np.random.default_rng(42)

n_rows = 420
base_dates = pd.date_range("2025-01-01", "2025-03-15", freq="D")

transaction_ids = np.arange(10001, 10001 + n_rows)
store_ids = rng.choice(["S01", "S02", "S03", "S04"], size=n_rows, p=[0.25, 0.3, 0.25, 0.2])
region_clean = rng.choice(["North", "South", "East", "West"], size=n_rows)
category_clean = rng.choice(
    ["coffee", "tea", "pastry", "sandwich", "smoothie"],
    size=n_rows,
    p=[0.34, 0.2, 0.2, 0.16, 0.1],
)
payment_clean = rng.choice(["card", "cash", "mobile"], size=n_rows, p=[0.52, 0.3, 0.18])

units_sold = rng.poisson(lam=8, size=n_rows) + 1
unit_price = rng.normal(loc=6.25, scale=1.8, size=n_rows).round(2)
unit_price = np.clip(unit_price, 1.5, 14.0)
discount_pct = rng.choice([0, 5, 10, 15, 20], size=n_rows, p=[0.52, 0.2, 0.15, 0.1, 0.03])

dates = rng.choice(base_dates, size=n_rows)

df = pd.DataFrame(
    {
        "transaction_id": transaction_ids,
        "sale_date": pd.to_datetime(dates),
        "store_id": store_ids,
        "region": region_clean,
        "product_category": category_clean,
        "units_sold": units_sold,
        "unit_price": unit_price,
        "discount_pct": discount_pct,
        "payment_type": payment_clean,
    }
)

# Create revenue with mild noise from rounding effects.
clean_revenue = (
    df["units_sold"] * df["unit_price"] * (1 - df["discount_pct"] / 100)
).round(2)
df["reported_revenue"] = (clean_revenue + rng.normal(0, 0.6, size=n_rows)).round(2)

# Introduce messy category spelling/casing variants.
category_variants = {
    "coffee": ["coffee", "Coffee", "COFFEE", "cofee", " coffee "],
    "tea": ["tea", "Tea", "TEA", "te", " tea"],
    "pastry": ["pastry", "Pastry", "PASTRY", "pastry ", "pastrie"],
    "sandwich": ["sandwich", "Sandwich", "SANDWICH", "sandwhich", " sandwich"],
    "smoothie": ["smoothie", "Smoothie", "SMOOTHIE", "smothie", " smoothie "],
}
df["product_category"] = [rng.choice(category_variants[c]) for c in df["product_category"]]

# Introduce region variants.
region_variants = {
    "North": ["North", "NORTH", "north", "North ", "Nrth"],
    "South": ["South", "SOUTH", "south", " South", "Sth"],
    "East": ["East", "EAST", "east", "East ", "Est"],
    "West": ["West", "WEST", "west", " West", "Wst"],
}
df["region"] = [rng.choice(region_variants[r]) for r in df["region"]]

# Introduce payment variants.
payment_variants = {
    "card": ["card", "Card", "CARD", "debit card", "credit"],
    "cash": ["cash", "Cash", "CASH", "cash ", " bills"],
    "mobile": ["mobile", "Mobile", "MOBILE", "tap", "app pay"],
}
df["payment_type"] = [rng.choice(payment_variants[p]) for p in df["payment_type"]]

# Mix date formats and inject invalid date strings.
formats = ["%Y-%m-%d", "%m/%d/%Y", "%d-%b-%Y"]
formatted_dates = []
for d in df["sale_date"]:
    fmt = rng.choice(formats, p=[0.55, 0.3, 0.15])
    formatted_dates.append(d.strftime(fmt))
df["sale_date"] = formatted_dates

invalid_date_idx = rng.choice(df.index, size=8, replace=False)
invalid_dates = ["2025-02-30", "13/40/2025", "not_a_date", "2025/99/01"]
for i, idx in enumerate(invalid_date_idx):
    df.at[idx, "sale_date"] = invalid_dates[i % len(invalid_dates)]

# Inject missing values.
for col, count in [("units_sold", 18), ("unit_price", 16), ("payment_type", 14), ("region", 10)]:
    miss_idx = rng.choice(df.index, size=count, replace=False)
    df.loc[miss_idx, col] = np.nan

# Inject extreme outliers and impossible values.
outlier_idx = rng.choice(df.index, size=8, replace=False)
df.loc[outlier_idx[:3], "units_sold"] = [120, 140, 160]
df.loc[outlier_idx[3:6], "unit_price"] = [0.5, 45.0, 60.0]
df.loc[outlier_idx[6:], "discount_pct"] = [120, -10]

# Revenue mismatches and negative revenue rows.
rev_noise_idx = rng.choice(df.index, size=22, replace=False)
df.loc[rev_noise_idx, "reported_revenue"] = (
    df.loc[rev_noise_idx, "reported_revenue"] + rng.normal(12, 6, size=len(rev_noise_idx))
).round(2)
neg_rev_idx = rng.choice(df.index, size=4, replace=False)
df.loc[neg_rev_idx, "reported_revenue"] = -rng.uniform(5, 30, size=4).round(2)

# Add duplicate rows to simulate duplicate imports.
dup_rows = df.sample(20, random_state=7)
df_messy = pd.concat([df, dup_rows], ignore_index=True)

# Shuffle rows.
df_messy = df_messy.sample(frac=1, random_state=99).reset_index(drop=True)

df_messy.to_csv("messy_cafe_sales.csv", index=False)
print("Saved messy_cafe_sales.csv")
print(df_messy.head())
