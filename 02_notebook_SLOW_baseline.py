# Databricks notebook source
# MAGIC %md
# MAGIC # 02 — Baseline Pipeline (SLOW — on purpose)
# MAGIC
# MAGIC This is a "day-1 engineer wrote this and it works but it's slow" notebook.
# MAGIC It is functionally correct but full of realistic anti-patterns. Run it, watch the
# MAGIC Spark UI, and use `03_troubleshooting_guide.md` to find and name every problem
# MAGIC before opening `04_notebook_OPTIMIZED.py`.
# MAGIC
# MAGIC Business ask: *"Build a daily summary: revenue by region/category/channel for
# MAGIC COMPLETED orders, joined with customer segment and product category, for VIP
# MAGIC customer analysis and a top-customers report."*

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.types import *

BASE_PATH = "/tmp/spark_opt_demo"
ORDERS_PATH    = f"{BASE_PATH}/raw/orders_csv"
CUSTOMERS_PATH = f"{BASE_PATH}/raw/customers_csv"
PRODUCTS_PATH  = f"{BASE_PATH}/raw/products_csv"
OUT_PATH       = f"{BASE_PATH}/gold/daily_summary"
TOP_CUST_PATH  = f"{BASE_PATH}/gold/top_customers"

# ---------------------------------------------------------------------
# PROBLEM 1: inferSchema=True on a 25M-row CSV -> forces a full extra
# read pass over the data just to guess types, every single run.
# ---------------------------------------------------------------------
orders = spark.read.option("header", True).option("inferSchema", True).csv(ORDERS_PATH)
customers = spark.read.option("header", True).option("inferSchema", True).csv(CUSTOMERS_PATH)
products = spark.read.option("header", True).option("inferSchema", True).csv(PRODUCTS_PATH)

# ---------------------------------------------------------------------
# PROBLEM 2: debug/sanity count() calls left in -> each one is a full
# job over the whole dataset, and they add up across a long notebook.
# ---------------------------------------------------------------------
print("orders count:", orders.count())
print("customers count:", customers.count())
print("products count:", products.count())

# COMMAND ----------

# ---------------------------------------------------------------------
# PROBLEM 3: cleaning the "$1234.56" amount string with a Python UDF.
# UDFs are a black box to Catalyst: no predicate pushdown, no codegen,
# and (for a plain Python UDF) a serialize/deserialize round trip to a
# Python worker process per row.
# ---------------------------------------------------------------------
def clean_amount(raw):
    if raw is None:
        return None
    return float(raw.replace("$", ""))

clean_amount_udf = F.udf(clean_amount, DoubleType())

orders_clean = orders.withColumn("order_amount", clean_amount_udf(F.col("order_amount_raw")))

# ---------------------------------------------------------------------
# PROBLEM 4: another Python UDF for something a built-in expression
# handles natively (upper-casing / string building).
# ---------------------------------------------------------------------
def make_order_key(order_id, region):
    return f"{region}-{order_id}"

make_order_key_udf = F.udf(make_order_key, StringType())

orders_clean = orders_clean.withColumn("order_key", make_order_key_udf(F.col("order_id"), F.col("region")))

# COMMAND ----------

# ---------------------------------------------------------------------
# PROBLEM 5: filtering AFTER the join instead of before. Every row of
# the huge fact table gets shuffled/joined even though ~75% of it will
# be thrown away by the status filter a few lines later.
# ---------------------------------------------------------------------
joined = (
    orders_clean
    .join(customers, on="customer_id", how="inner")     # PROBLEM 6: plain shuffle join,
    .join(products, on="product_id", how="inner")        #   both dims are small enough to broadcast
)

joined = joined.filter(F.col("status") == "COMPLETED")

# ---------------------------------------------------------------------
# PROBLEM 7: the join above is also skewed. `customer_id` has 1,000
# "hot" values carrying ~40% of all rows (see 01_generate_dataset.py).
# A standard shuffle join sends every hot key's rows to ONE reducer
# task, so a handful of tasks run far longer than the rest ("stragglers").
# ---------------------------------------------------------------------

# COMMAND ----------

# ---------------------------------------------------------------------
# PROBLEM 8: no caching before this DataFrame is used THREE separate
# times below (aggregation, top-customers report, and a preview show).
# Spark's lazy evaluation means the entire read -> UDF -> join chain
# gets recomputed from scratch for each downstream action.
# ---------------------------------------------------------------------

daily_summary = (
    joined
    .groupBy("order_date", "region", "category", "channel")
    .agg(
        F.sum("order_amount").alias("total_revenue"),
        F.count("order_id").alias("order_count"),
        F.countDistinct("customer_id").alias("distinct_customers"),
    )
)

# PROBLEM 9: collect_list on a wide, unfiltered groupBy -> can build huge
# per-key arrays in memory on a single executor for hot keys.
top_customers = (
    joined
    .groupBy("customer_id", "segment")
    .agg(
        F.sum("order_amount").alias("customer_revenue"),
        F.collect_list("order_id").alias("all_order_ids"),   # unbounded list per customer
    )
    .orderBy(F.desc("customer_revenue"))
)

print("Preview of daily_summary:")
daily_summary.show(5)          # action #1 over the whole chain

print("Preview of top_customers:")
top_customers.show(5)          # action #2 over the whole chain, recomputed from scratch

# COMMAND ----------

# ---------------------------------------------------------------------
# PROBLEM 10: writing the final output with coalesce(1). Forcing all
# data through a single task to produce one file both kills write
# parallelism and can blow out executor memory on large outputs.
# ---------------------------------------------------------------------
daily_summary.coalesce(1).write.mode("overwrite").parquet(OUT_PATH)   # action #3

# PROBLEM 11: no partitioning of the output at all -- every future
# consumer that filters by region or order_date has to scan everything.

top_customers.coalesce(1).write.mode("overwrite").parquet(TOP_CUST_PATH)   # action #4

print("Pipeline complete (slowly).")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary of problems planted in this notebook
# MAGIC 1. `inferSchema=True` on large CSVs -> extra full read pass
# MAGIC 2. Stray `.count()` debug calls -> extra full jobs
# MAGIC 3–4. Python UDFs for logic that has a native Spark-SQL equivalent
# MAGIC 5. Filter applied after the join, not before (or pushed into the read)
# MAGIC 6. Shuffle (sort-merge) join used where a broadcast join fits
# MAGIC 7. Skewed join key (`customer_id`) causing straggler tasks
# MAGIC 8. Same DataFrame recomputed 4 separate times (no cache/persist)
# MAGIC 9. `collect_list` with no cap -> large per-key in-memory arrays
# MAGIC 10. `coalesce(1)` on write -> no write parallelism, single huge file
# MAGIC 11. Output not partitioned -> full scans for every downstream reader
# MAGIC
# MAGIC Go to `03_troubleshooting_guide.md` and use the Spark UI to find each
# MAGIC of these yourself *before* looking at the fixed notebook.