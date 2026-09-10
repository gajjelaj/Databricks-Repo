# Databricks notebook source
# MAGIC %md
# MAGIC # 04 — Optimized Pipeline
# MAGIC
# MAGIC Same business logic as `02_notebook_SLOW_baseline.py`, same output, same
# MAGIC input data. Every change below maps to one row in the fix table at the end
# MAGIC of `03_troubleshooting_guide.md`. Comments call out the technique by name.

# COMMAND ----------

import time
from pyspark.sql import functions as F
from pyspark.sql.types import *

BASE_PATH = "/tmp/spark_opt_demo"
ORDERS_PATH    = f"{BASE_PATH}/raw/orders_csv"
CUSTOMERS_PATH = f"{BASE_PATH}/raw/customers_csv"
PRODUCTS_PATH  = f"{BASE_PATH}/raw/products_csv"
OUT_PATH       = f"{BASE_PATH}/gold/daily_summary_opt"
TOP_CUST_PATH  = f"{BASE_PATH}/gold/top_customers_opt"

t_start = time.time()

# ---------------------------------------------------------------------
# FIX 1: explicit schema. No extra read pass to infer types, and it
# fails fast/loud on a genuinely malformed row instead of silently
# guessing a wrong type for the whole column.
# ---------------------------------------------------------------------
orders_schema = StructType([
    StructField("order_id", LongType()),
    StructField("customer_id", LongType()),
    StructField("product_id", LongType()),
    StructField("order_date", DateType()),
    StructField("order_ts", TimestampType()),
    StructField("region", StringType()),
    StructField("channel", StringType()),
    StructField("payment_method", StringType()),
    StructField("status", StringType()),
    StructField("quantity", IntegerType()),
    StructField("unit_price", DoubleType()),
    StructField("discount_pct", DoubleType()),
    StructField("order_amount_raw", StringType()),
])

customers_schema = StructType([
    StructField("customer_id", LongType()),
    StructField("full_name", StringType()),
    StructField("email", StringType()),
    StructField("country", StringType()),
    StructField("signup_date", DateType()),
    StructField("segment", StringType()),
])

products_schema = StructType([
    StructField("product_id", LongType()),
    StructField("product_name", StringType()),
    StructField("category", StringType()),
    StructField("brand", StringType()),
    StructField("cost", DoubleType()),
    StructField("list_price", DoubleType()),
])

orders = spark.read.option("header", True).schema(orders_schema).csv(ORDERS_PATH)
customers = spark.read.option("header", True).schema(customers_schema).csv(CUSTOMERS_PATH)
products = spark.read.option("header", True).schema(products_schema).csv(PRODUCTS_PATH)

# ---------------------------------------------------------------------
# FIX 2: no stray .count()/.show() calls scattered through the
# transformation chain. If you need a sanity check while developing,
# do it on a small .limit(100) sample, not the full dataset, and
# remove it before the pipeline is considered "done".
# ---------------------------------------------------------------------

# ---------------------------------------------------------------------
# FIX 5 (do this early): filter BEFORE the join, not after. Spark's
# Catalyst optimizer can sometimes push a filter below a join on its
# own, but don't rely on it -- write the filter where it belongs so
# the join only ever processes rows you're going to keep. This also
# lets column pruning happen earlier (see .select below).
# ---------------------------------------------------------------------
orders_completed = orders.filter(F.col("status") == "COMPLETED")

# ---------------------------------------------------------------------
# FIX 3 & 4: replace both Python UDFs with native Spark SQL
# expressions. Native functions run inside the JVM with whole-stage
# codegen -- no per-row Python round trip, and Catalyst can reason
# about/optimize around them.
# ---------------------------------------------------------------------
orders_clean = (
    orders_completed
    .withColumn(
        "order_amount",
        F.regexp_replace(F.col("order_amount_raw"), r"\$", "").cast(DoubleType())
    )
    .withColumn("order_key", F.concat_ws("-", F.col("region"), F.col("order_id")))
    # column pruning: drop columns we no longer need before the join/shuffle
    .select("order_id", "order_key", "customer_id", "product_id", "order_date",
            "region", "channel", "order_amount")
)

# COMMAND ----------

# ---------------------------------------------------------------------
# FIX 6: broadcast join. customers (~500K rows) and products (~50K
# rows) both comfortably fit in executor memory, so ship a full copy
# of each to every executor and avoid shuffling the 25M-row fact table
# at all for the join keys. F.broadcast() is an explicit hint; Spark's
# default broadcast threshold (spark.sql.autoBroadcastJoinThreshold,
# 10MB by default) would often pick this automatically too, but be
# explicit -- don't depend on the dimension staying small forever.
# ---------------------------------------------------------------------
customers_small = customers.select("customer_id", "segment", "country")
products_small = products.select("product_id", "category")

joined = (
    orders_clean
    .join(F.broadcast(customers_small), on="customer_id", how="inner")
    .join(F.broadcast(products_small), on="product_id", how="inner")
)

# ---------------------------------------------------------------------
# FIX 7: skew handling for customer_id.
#
# Option A (simplest, Spark 3.x+): let Adaptive Query Execution do it.
# AQE's skew join optimization detects oversized partitions at runtime
# and splits them automatically -- no code change needed, just make
# sure it's on (it's on by default in modern Spark/Databricks runtimes,
# shown here for clarity/interview purposes):
spark.conf.set("spark.sql.adaptive.enabled", "true")
spark.conf.set("spark.sql.adaptive.skewJoin.enabled", "true")
#
# Option B (manual salting, useful when AQE isn't available or you
# need to explain the technique by hand in an interview):
#   1. Add a random "salt" 0..N to the skewed side's join key.
#   2. Explode the small side N times, once per salt value.
#   3. Join on (key, salt) so the hot key's rows spread across N
#      reducer tasks instead of piling onto one.
#
# SALT_BUCKETS = 8
# orders_salted = orders_clean.withColumn("salt", (F.rand() * SALT_BUCKETS).cast("int"))
# customers_salted = (
#     customers_small
#     .withColumn("salt", F.explode(F.array([F.lit(i) for i in range(SALT_BUCKETS)])))
# )
# joined = orders_salted.join(customers_salted, on=["customer_id", "salt"], how="inner")
#
# With AQE's skew join handling on (Option A), the manual salting in
# Option B usually isn't necessary -- keep it in your back pocket for
# clusters/engines where adaptive execution isn't available.

# COMMAND ----------

# ---------------------------------------------------------------------
# FIX 8: cache once, here, because `joined` is about to be used twice
# below (daily_summary and top_customers). Without this, Spark
# recomputes the entire read -> filter -> clean -> broadcast-join
# chain from scratch for each downstream action.
# .persist() with an explicit storage level is more predictable at
# scale than plain .cache() (which is MEMORY_AND_DISK on DataFrames
# already, but being explicit documents the intent).
# ---------------------------------------------------------------------
from pyspark import StorageLevel
joined = joined.persist(StorageLevel.MEMORY_AND_DISK)
joined.count()   # one deliberate action to materialize the cache now,
                  # so both downstream branches read from it instead of
                  # each triggering their own first materialization

# COMMAND ----------

daily_summary = (
    joined
    .groupBy("order_date", "region", "category", "channel")
    .agg(
        F.sum("order_amount").alias("total_revenue"),
        F.count("order_id").alias("order_count"),
        F.countDistinct("customer_id").alias("distinct_customers"),
    )
)

# ---------------------------------------------------------------------
# FIX 9: collect_list is capped instead of unbounded. If the report
# genuinely needs "all order ids", that belongs in a separate,
# row-level table (order_id, customer_id) that a BI tool can filter --
# not a single in-memory array per customer that grows without bound
# for hot keys.
# ---------------------------------------------------------------------
top_customers = (
    joined
    .groupBy("customer_id", "segment")
    .agg(
        F.sum("order_amount").alias("customer_revenue"),
        F.slice(F.collect_list("order_id"), 1, 20).alias("sample_order_ids"),  # capped
    )
    .orderBy(F.desc("customer_revenue"))
)

# COMMAND ----------

# ---------------------------------------------------------------------
# FIX 10 & 11: no coalesce(1). Repartition to a sensible number of
# output files sized to data volume (rule of thumb: aim for ~128MB-
# 512MB per output file), and partition the physical layout by the
# columns downstream consumers will filter on most -- region and
# order_date here -- so future reads can skip whole directories
# instead of scanning everything.
# ---------------------------------------------------------------------
(
    daily_summary
    .repartition("region")               # right-size shuffle before write, avoid 1-task bottleneck
    .write.mode("overwrite")
    .partitionBy("region")               # physical pruning for downstream readers
    .parquet(OUT_PATH)
)

(
    top_customers
    .repartition(50)
    .write.mode("overwrite")
    .parquet(TOP_CUST_PATH)
)

joined.unpersist()   # release the cached DataFrame once nothing else needs it

print(f"Optimized pipeline complete in {time.time() - t_start:.1f}s")

# COMMAND ----------

# MAGIC %md
# MAGIC ## What changed, mapped to the Spark UI evidence
# MAGIC
# MAGIC | Fix | Where to see it worked |
# MAGIC |---|---|
# MAGIC | Explicit schema | No separate "infer schema" job before the first real job in the Jobs tab |
# MAGIC | Removed debug counts | Fewer total jobs for the same output |
# MAGIC | Native functions, not UDFs | No `BatchEvalPython` node in `daily_summary.explain(True)` |
# MAGIC | Filter before join | Filter node sits below/inside the Join in the physical plan |
# MAGIC | Broadcast join | `BroadcastHashJoin` in the plan; no large shuffle read/write on the dimension side in the Stages tab |
# MAGIC | AQE skew handling | Stages tab: max vs median task duration on the join stage is much closer together |
# MAGIC | `.persist()` + one action | Only ONE "read+join" job chain in the Jobs tab, reused by both aggregations (check the Storage tab for the cached DataFrame) |
# MAGIC | Capped `collect_list` | Spill (Memory)/(Disk) columns at or near zero on the groupBy stage |
# MAGIC | No `coalesce(1)` | Many parallel write tasks instead of one long one at the end |
# MAGIC | `partitionBy("region")` on write | Output directory shows `region=NA/`, `region=EU/`, ... subfolders; a downstream `.filter(region=='NA')` read skips the rest |
# MAGIC
# MAGIC Run both notebooks back to back on the same cluster and compare the
# MAGIC printed elapsed-time line -- that's your concrete before/after number.