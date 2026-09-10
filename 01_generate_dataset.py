# Databricks notebook source
# MAGIC %md
# MAGIC # 01 — Generate Large Synthetic Dataset for Spark Optimization Practice
# MAGIC
# MAGIC Run this once to create the raw data. It generates:
# MAGIC - **orders_raw**  : ~25,000,000 rows  (transaction fact table)
# MAGIC - **customers_raw**: ~500,000 rows    (dimension)
# MAGIC - **products_raw** : ~50,000 rows     (dimension)
# MAGIC
# MAGIC The data is **deliberately skewed and deliberately written as many small files**
# MAGIC so that later notebooks reproduce real performance problems:
# MAGIC data skew, small-file problem, shuffle-heavy joins, and non-partitioned storage.
# MAGIC
# MAGIC Adjust `SCALE` down (e.g. 0.1) first if you want a quick smoke-test run before
# MAGIC committing to the full 25M-row build.

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.types import *

# ---- Config -----------------------------------------------------------
SCALE = 1.0                       # multiply row counts by this (0.1 for a quick test)
N_ORDERS      = int(25_000_000 * SCALE)
N_CUSTOMERS   = int(500_000 * SCALE)
N_PRODUCTS    = int(50_000 * SCALE)

BASE_PATH = "/tmp/spark_opt_demo"          # change to a DBFS/ADLS path in a real workspace
ORDERS_PATH    = f"{BASE_PATH}/raw/orders_csv"
CUSTOMERS_PATH = f"{BASE_PATH}/raw/customers_csv"
PRODUCTS_PATH  = f"{BASE_PATH}/raw/products_csv"

REGIONS   = ["NA", "EU", "APAC", "LATAM", "MEA"]
CHANNELS  = ["web", "mobile_app", "marketplace", "store"]
PAYMENTS  = ["credit_card", "debit_card", "paypal", "upi", "cod"]
STATUSES  = ["COMPLETED", "CANCELLED", "RETURNED", "PENDING"]
CATEGORIES = ["Electronics", "Apparel", "Home", "Grocery", "Sports",
              "Books", "Toys", "Beauty", "Automotive", "Garden"]

# COMMAND ----------

# MAGIC %md ## Dimension: customers

# COMMAND ----------

customers = (
    spark.range(1, N_CUSTOMERS + 1)
    .withColumnRenamed("id", "customer_id")
    .withColumn("full_name", F.concat(F.lit("Customer_"), F.col("customer_id")))
    .withColumn("email", F.concat(F.lit("user"), F.col("customer_id"), F.lit("@example.com")))
    .withColumn("country", F.element_at(F.array(*[F.lit(r) for r in REGIONS]),
                                         (F.abs(F.hash("customer_id")) % len(REGIONS) + 1)))
    .withColumn("signup_date", F.date_add(F.lit("2018-01-01"),
                                           (F.abs(F.hash("customer_id")) % 2500).cast("int")))
    .withColumn("segment", F.when(F.col("customer_id") % 100 == 0, "VIP")
                             .when(F.col("customer_id") % 10 == 0, "Loyal")
                             .otherwise("Regular"))
)

customers.write.mode("overwrite").option("header", True).csv(CUSTOMERS_PATH)
print(f"customers written: {N_CUSTOMERS:,} rows -> {CUSTOMERS_PATH}")

# COMMAND ----------

# MAGIC %md ## Dimension: products

# COMMAND ----------

products = (
    spark.range(1, N_PRODUCTS + 1)
    .withColumnRenamed("id", "product_id")
    .withColumn("product_name", F.concat(F.lit("Product_"), F.col("product_id")))
    .withColumn("category", F.element_at(F.array(*[F.lit(c) for c in CATEGORIES]),
                                          (F.abs(F.hash("product_id")) % len(CATEGORIES) + 1)))
    .withColumn("brand", F.concat(F.lit("Brand_"), (F.col("product_id") % 500)))
    .withColumn("cost", F.round(F.rand(seed=1) * 200 + 5, 2))
    .withColumn("list_price", F.round(F.col("cost") * (1.3 + F.rand(seed=2) * 0.9), 2))
)

products.write.mode("overwrite").option("header", True).csv(PRODUCTS_PATH)
print(f"products written: {N_PRODUCTS:,} rows -> {PRODUCTS_PATH}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Fact table: orders (deliberately skewed)
# MAGIC
# MAGIC Skew is injected on purpose so later notebooks have a real skewed-join problem
# MAGIC to diagnose and fix:
# MAGIC - **customer_id**: 40% of all orders belong to just 1,000 "hot" customers
# MAGIC   (simulates bots / bulk B2B accounts / test accounts that got mixed into prod data)
# MAGIC - **product_id**: a handful of viral SKUs dominate order volume
# MAGIC - **region**: `NA` is overrepresented, `MEA` is sparse

# COMMAND ----------

HOT_CUSTOMERS = 1000     # number of skewed "hot" customer ids
HOT_PRODUCTS  = 25       # number of viral SKUs

orders = spark.range(1, N_ORDERS + 1).withColumnRenamed("id", "order_id")

orders = (
    orders
    # 40% of rows forced onto a tiny pool of hot customer ids -> classic join skew
    .withColumn(
        "customer_id",
        F.when(F.rand(seed=10) < 0.40, (F.abs(F.hash("order_id")) % HOT_CUSTOMERS) + 1)
         .otherwise((F.abs(F.hash("order_id")) % N_CUSTOMERS) + 1)
    )
    # 30% of rows forced onto a tiny pool of viral products
    .withColumn(
        "product_id",
        F.when(F.rand(seed=11) < 0.30, (F.abs(F.hash("order_id", "x")) % HOT_PRODUCTS) + 1)
         .otherwise((F.abs(F.hash("order_id", "x")) % N_PRODUCTS) + 1)
    )
    .withColumn(
        "region",
        F.when(F.rand(seed=12) < 0.55, "NA")
         .otherwise(F.element_at(F.array(*[F.lit(r) for r in REGIONS]),
                                  (F.abs(F.hash("order_id", "r")) % len(REGIONS) + 1)))
    )
    .withColumn("channel", F.element_at(F.array(*[F.lit(c) for c in CHANNELS]),
                                         (F.abs(F.hash("order_id", "c")) % len(CHANNELS) + 1)))
    .withColumn("payment_method", F.element_at(F.array(*[F.lit(p) for p in PAYMENTS]),
                                                (F.abs(F.hash("order_id", "p")) % len(PAYMENTS) + 1)))
    .withColumn("status", F.element_at(F.array(*[F.lit(s) for s in STATUSES]),
                                        (F.abs(F.hash("order_id", "s")) % len(STATUSES) + 1)))
    .withColumn("quantity", (F.abs(F.hash("order_id", "q")) % 5 + 1).cast("int"))
    .withColumn("unit_price", F.round(F.rand(seed=13) * 500 + 5, 2))
    .withColumn("discount_pct", F.round(F.rand(seed=14) * 0.3, 2))
    .withColumn(
        "order_date",
        F.date_add(F.lit("2021-01-01"), (F.abs(F.hash("order_id", "d")) % 1700).cast("int"))
    )
    .withColumn(
        "order_ts",
        F.to_timestamp(F.col("order_date")) + F.expr("INTERVAL 1 HOUR") * (F.abs(F.hash("order_id", "h")) % 24)
    )
    # a raw string amount column on purpose -- forces a cast/clean step downstream
    .withColumn("order_amount_raw", F.concat(F.lit("$"),
                F.round(F.col("quantity") * F.col("unit_price") * (1 - F.col("discount_pct")), 2)))
)

orders = orders.select(
    "order_id", "customer_id", "product_id", "order_date", "order_ts",
    "region", "channel", "payment_method", "status",
    "quantity", "unit_price", "discount_pct", "order_amount_raw"
)

# Deliberately write with a high number of partitions relative to data size
# so the output lands as thousands of small files -> reproduces the "small file problem"
(
    orders
    .repartition(2000)
    .write.mode("overwrite")
    .option("header", True)
    .csv(ORDERS_PATH)
)

print(f"orders written: {N_ORDERS:,} rows -> {ORDERS_PATH}")
print("NOTE: written with high partition count on purpose -> many small CSV files.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Sanity check row counts

# COMMAND ----------

# If running on Databricks, dbutils is available automatically.
# If running locally (plain PySpark, no Databricks), list the output dir with Python's os module instead:
try:
    print("orders files:   ", len(dbutils.fs.ls(ORDERS_PATH)))
    print("customers files:", len(dbutils.fs.ls(CUSTOMERS_PATH)))
    print("products files: ", len(dbutils.fs.ls(PRODUCTS_PATH)))
except NameError:
    import os
    for name, path in [("orders", ORDERS_PATH), ("customers", CUSTOMERS_PATH), ("products", PRODUCTS_PATH)]:
        n = len([f for f in os.listdir(path) if f.endswith(".csv")])
        print(f"{name} files: {n}")
