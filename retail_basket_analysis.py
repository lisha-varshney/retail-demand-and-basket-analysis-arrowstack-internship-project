"""
Retail Demand & Basket Analysis  (without mlxtend)
---------------------------------------------------
Part A : Demand analysis  (pandas + matplotlib)
Part B : Market Basket Analysis - Apriori + Association Rules
         implemented from scratch using only itertools / collections.

Dataset columns:
order_id, user_id, order_date, time, order_hour_of_day,
product_name, quantity, price, category, product_id
"""

import os
from itertools import combinations
from collections import defaultdict

import pandas as pd
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# 0. SETTINGS
# ----------------------------------------------------------------------
CSV_PATH = "retail_dataset_csv.csv"      # change path if needed
OUT_DIR = "output"                       # charts + csv results are saved here
# NOTE: this dataset has 68 products and ~5 items per basket, so any single pair
# appears in <1% of baskets. Thresholds are therefore kept low. For a bigger
# dataset, increase MIN_SUPPORT (e.g. 0.01 - 0.02) and MIN_CONFIDENCE (e.g. 0.3).
MIN_SUPPORT = 0.006                      # 0.6% of baskets (~12 orders)
MIN_CONFIDENCE = 0.10                    # 10%
MIN_LIFT = 1.2                           # keep only positively associated rules
MAX_ITEMSET_SIZE = 3                     # 2 = pairs, 3 = up to triples

os.makedirs(OUT_DIR, exist_ok=True)

# ----------------------------------------------------------------------
# 1. LOAD & CLEAN
# ----------------------------------------------------------------------
df = pd.read_csv(r"C:\Users\PC\Downloads\retail_dataset.csv.csv")

df["order_date"] = pd.to_datetime(df["order_date"])
df["product_name"] = df["product_name"].str.strip()
df["category"] = df["category"].str.strip()

# revenue per line item
df["revenue"] = df["quantity"] * df["price"]

# time features
df["month"] = df["order_date"].dt.to_period("M").astype(str)
df["month_num"] = df["order_date"].dt.month
df["day_name"] = df["order_date"].dt.day_name()
df["weekday_num"] = df["order_date"].dt.weekday

print("=" * 60)
print("DATASET OVERVIEW")
print("=" * 60)
print("Rows            :", len(df))
print("Orders          :", df["order_id"].nunique())
print("Customers       :", df["user_id"].nunique())
print("Products        :", df["product_name"].nunique())
print("Categories      :", df["category"].nunique())
print("Date range      :", df["order_date"].min().date(), "to", df["order_date"].max().date())
print("Total revenue   : {:,.2f}".format(df["revenue"].sum()))
print("Missing values  :", int(df.isna().sum().sum()))

# ----------------------------------------------------------------------
# 2. PART A - DEMAND ANALYSIS
# ----------------------------------------------------------------------
print("\n" + "=" * 60)
print("PART A: DEMAND ANALYSIS")
print("=" * 60)

# --- 2.1 Monthly demand (units, revenue, orders)
monthly = (
    df.groupby("month")
    .agg(units=("quantity", "sum"),
         revenue=("revenue", "sum"),
         orders=("order_id", "nunique"))
    .reset_index()
)
print("\nMonthly demand:\n", monthly.to_string(index=False))

# --- 2.2 Day-of-week demand
dow = (
    df.groupby(["weekday_num", "day_name"])
    .agg(units=("quantity", "sum"),
         revenue=("revenue", "sum"),
         orders=("order_id", "nunique"))
    .reset_index()
    .sort_values("weekday_num")
)
print("\nDay-of-week demand:\n", dow.drop(columns="weekday_num").to_string(index=False))

# --- 2.3 Hourly demand (peak hours)
hourly = (
    df.groupby("order_hour_of_day")
    .agg(units=("quantity", "sum"),
         revenue=("revenue", "sum"),
         orders=("order_id", "nunique"))
    .reset_index()
)
peak_hour = hourly.loc[hourly["orders"].idxmax(), "order_hour_of_day"]
print("\nPeak hour (most orders):", int(peak_hour), ":00")

# --- 2.4 Category performance
cat = (
    df.groupby("category")
    .agg(units=("quantity", "sum"),
         revenue=("revenue", "sum"),
         orders=("order_id", "nunique"))
    .sort_values("revenue", ascending=False)
    .reset_index()
)
cat["revenue_share_%"] = (cat["revenue"] / cat["revenue"].sum() * 100).round(2)
print("\nCategory performance:\n", cat.to_string(index=False))

# --- 2.5 Top products by units & revenue
prod = (
    df.groupby("product_name")
    .agg(units=("quantity", "sum"),
         revenue=("revenue", "sum"),
         orders=("order_id", "nunique"),
         avg_price=("price", "mean"))
    .reset_index()
)
top_units = prod.sort_values("units", ascending=False).head(10)
top_rev = prod.sort_values("revenue", ascending=False).head(10)
print("\nTop 10 products by units:\n", top_units.to_string(index=False))
print("\nTop 10 products by revenue:\n", top_rev.to_string(index=False))

# --- 2.4 ABC classification (Pareto on revenue)
abc = prod.sort_values("revenue", ascending=False).reset_index(drop=True)
abc["cum_rev_%"] = abc["revenue"].cumsum() / abc["revenue"].sum() * 100
abc["ABC_class"] = abc["cum_rev_%"].apply(
    lambda x: "A" if x <= 70 else ("B" if x <= 90 else "C"))
print("\nABC classification counts:\n", abc["ABC_class"].value_counts().sort_index())

# --- 2.6 Order-level metrics
orders = (
    df.groupby("order_id")
    .agg(user_id=("user_id", "first"),
         order_value=("revenue", "sum"),
         items=("product_name", "nunique"),
         units=("quantity", "sum"))
    .reset_index()
)
print("\nAverage order value (AOV): {:,.2f}".format(orders["order_value"].mean()))
print("Average distinct items / basket: {:.2f}".format(orders["items"].mean()))

# --- 2.7 Customer behaviour (repeat customers, top customers)
cust = (
    orders.groupby("user_id")
    .agg(n_orders=("order_id", "nunique"),
         total_spent=("order_value", "sum"))
    .reset_index()
)
repeat_rate = (cust["n_orders"] > 1).mean() * 100
print("Repeat customer rate: {:.1f}%".format(repeat_rate))
print("\nTop 5 customers by spend:\n",
      cust.sort_values("total_spent", ascending=False).head(5).to_string(index=False))

# ----------------------------------------------------------------------
# 3. DEMAND CHARTS
# ----------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

axes[0, 0].plot(monthly["month"], monthly["revenue"], marker="o")
axes[0, 0].set_title("Monthly Revenue")
axes[0, 0].tick_params(axis="x", rotation=45)

axes[0, 1].bar(dow["day_name"], dow["orders"])
axes[0, 1].set_title("Orders by Day of Week")
axes[0, 1].tick_params(axis="x", rotation=45)

axes[1, 0].bar(hourly["order_hour_of_day"], hourly["orders"])
axes[1, 0].set_title("Orders by Hour of Day")
axes[1, 0].set_xlabel("Hour")

axes[1, 1].barh(cat["category"][::-1], cat["revenue"][::-1])
axes[1, 1].set_title("Revenue by Category")

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "demand_overview.png"), dpi=150)
plt.close()

plt.figure(figsize=(9, 6))
plt.barh(top_units["product_name"][::-1], top_units["units"][::-1])
plt.title("Top 10 Products by Units Sold")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "top_products.png"), dpi=150)
plt.close()

# Hour x Day heatmap of orders
heat = (
    df.drop_duplicates("order_id")
    .pivot_table(index="day_name", columns="order_hour_of_day",
                 values="order_id", aggfunc="count", fill_value=0)
    .reindex(["Monday", "Tuesday", "Wednesday", "Thursday",
              "Friday", "Saturday", "Sunday"])
)
plt.figure(figsize=(12, 4.5))
plt.imshow(heat.values, aspect="auto", cmap="YlOrRd")
plt.colorbar(label="Orders")
plt.xticks(range(len(heat.columns)), heat.columns)
plt.yticks(range(len(heat.index)), heat.index)
plt.title("Order Heatmap: Day vs Hour")
plt.xlabel("Hour of day")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "heatmap_day_hour.png"), dpi=150)
plt.close()

# ----------------------------------------------------------------------
# 4. PART B - MARKET BASKET ANALYSIS (Apriori from scratch)
# ----------------------------------------------------------------------
print("\n" + "=" * 60)
print("PART B: MARKET BASKET ANALYSIS (Apriori, no mlxtend)")
print("=" * 60)

# 4.1 Build baskets: one set of unique products per order
#     (the data has repeated product rows inside some orders -> set removes them)
baskets = df.groupby("order_id")["product_name"].apply(frozenset).tolist()
N = len(baskets)
min_count = MIN_SUPPORT * N
print("Total baskets:", N, "| min support count:", round(min_count, 1))


def apriori(baskets, min_count, max_len):
    """Return dict {frozenset(itemset): support_count} for all frequent itemsets."""
    # --- L1 : frequent single items
    counts = defaultdict(int)
    for b in baskets:
        for item in b:
            counts[frozenset([item])] += 1
    current = {k: v for k, v in counts.items() if v >= min_count}
    frequent = dict(current)

    k = 2
    while current and k <= max_len:
        prev_itemsets = list(current.keys())
        prev_set = set(prev_itemsets)

        # --- candidate generation: join step
        candidates = set()
        for i in range(len(prev_itemsets)):
            for j in range(i + 1, len(prev_itemsets)):
                union = prev_itemsets[i] | prev_itemsets[j]
                if len(union) == k:
                    # --- prune step: every (k-1) subset must be frequent
                    if all(frozenset(sub) in prev_set
                           for sub in combinations(union, k - 1)):
                        candidates.add(union)

        # --- count candidates
        cand_counts = defaultdict(int)
        for b in baskets:
            if len(b) < k:
                continue
            for c in candidates:
                if c <= b:
                    cand_counts[c] += 1

        current = {c: n for c, n in cand_counts.items() if n >= min_count}
        frequent.update(current)
        k += 1
    return frequent


frequent_itemsets = apriori(baskets, min_count, MAX_ITEMSET_SIZE)

fi_df = pd.DataFrame(
    [(", ".join(sorted(k)), len(k), v, v / N) for k, v in frequent_itemsets.items()],
    columns=["itemset", "size", "count", "support"],
).sort_values(["size", "support"], ascending=[True, False])

print("\nFrequent itemsets found:", len(fi_df))
print(fi_df.groupby("size").size().rename("n_itemsets").to_string())
print("\nTop frequent pairs:\n",
      fi_df[fi_df["size"] == 2].head(10).to_string(index=False))

# 4.2 Generate association rules
rules = []
for itemset, count_ab in frequent_itemsets.items():
    if len(itemset) < 2:
        continue
    support_ab = count_ab / N
    for r in range(1, len(itemset)):
        for antecedent in combinations(itemset, r):
            A = frozenset(antecedent)
            B = itemset - A
            support_a = frequent_itemsets[A] / N
            support_b = frequent_itemsets[B] / N
            confidence = support_ab / support_a
            lift = confidence / support_b
            leverage = support_ab - support_a * support_b
            conviction = ((1 - support_b) / (1 - confidence)
                          if confidence < 1 else float("inf"))
            if confidence >= MIN_CONFIDENCE and lift >= MIN_LIFT:
                rules.append({
                    "antecedent": ", ".join(sorted(A)),
                    "consequent": ", ".join(sorted(B)),
                    "support": support_ab,
                    "confidence": confidence,
                    "lift": lift,
                    "leverage": leverage,
                    "conviction": conviction,
                })

rule_cols = ["antecedent", "consequent", "support", "confidence",
             "lift", "leverage", "conviction"]
rules_df = pd.DataFrame(rules, columns=rule_cols)
rules_df = (rules_df.sort_values(["lift", "confidence"], ascending=False)
            .reset_index(drop=True))

print("\nAssociation rules generated:", len(rules_df))
if not rules_df.empty:
    print("\nTop 15 rules by lift:\n",
          rules_df.head(15).round(4).to_string(index=False))

# 4.3 Category-level basket analysis (denser, more reliable with this data)
cat_baskets = df.groupby("order_id")["category"].apply(frozenset).tolist()
cat_frequent = apriori(cat_baskets, 0.03 * len(cat_baskets), 2)
cat_rules = []
for itemset, c_ab in cat_frequent.items():
    if len(itemset) != 2:
        continue
    for a in itemset:
        A = frozenset([a])
        B = itemset - A
        s_ab = c_ab / len(cat_baskets)
        s_a = cat_frequent[A] / len(cat_baskets)
        s_b = cat_frequent[B] / len(cat_baskets)
        cat_rules.append({"antecedent": a, "consequent": ", ".join(B),
                          "support": s_ab, "confidence": s_ab / s_a,
                          "lift": s_ab / (s_a * s_b)})
cat_rules_df = (pd.DataFrame(cat_rules)
                .sort_values("lift", ascending=False)
                .reset_index(drop=True))
print("\nTop category-level rules by lift:\n",
      cat_rules_df.head(10).round(4).to_string(index=False))

# ----------------------------------------------------------------------
# 5. RULE CHARTS
# ----------------------------------------------------------------------
if not rules_df.empty:
    plt.figure(figsize=(8, 6))
    sc = plt.scatter(rules_df["support"], rules_df["confidence"],
                     c=rules_df["lift"], s=60, cmap="viridis")
    plt.colorbar(sc, label="Lift")
    plt.xlabel("Support")
    plt.ylabel("Confidence")
    plt.title("Association Rules: Support vs Confidence (colour = lift)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "rules_scatter.png"), dpi=150)
    plt.close()

    top = rules_df.head(10).copy()
    labels = top["antecedent"] + "  ->  " + top["consequent"]
    plt.figure(figsize=(10, 6))
    plt.barh(labels[::-1], top["lift"][::-1])
    plt.axvline(1, color="red", linestyle="--", label="lift = 1")
    plt.xlabel("Lift")
    plt.title("Top 10 Rules by Lift")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "top_rules_lift.png"), dpi=150)
    plt.close()

# ----------------------------------------------------------------------
# 6. SAVE RESULTS
# ----------------------------------------------------------------------
monthly.to_csv(os.path.join(OUT_DIR, "monthly_demand.csv"), index=False)
cat.to_csv(os.path.join(OUT_DIR, "category_performance.csv"), index=False)
abc.to_csv(os.path.join(OUT_DIR, "abc_classification.csv"), index=False)
fi_df.to_csv(os.path.join(OUT_DIR, "frequent_itemsets.csv"), index=False)
rules_df.to_csv(os.path.join(OUT_DIR, "association_rules.csv"), index=False)
cat_rules_df.to_csv(os.path.join(OUT_DIR, "category_rules.csv"), index=False)

print("\nAll charts and CSV files saved in:", os.path.abspath(OUT_DIR))