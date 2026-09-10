# Spark Troubleshooting Guide — Diagnosing `02_notebook_SLOW_baseline.py`

Work through this **in order**. Don't jump to the fixed notebook until you've
found the evidence yourself in the Spark UI — that's the actual skill you're
building.

---

## Step 0 — Where is the Spark UI?

- **Databricks**: open the job/notebook run, click the cluster, then the
  **Spark UI** tab. Or from a running command, click **"View"** next to the
  progress bar while it's executing.
- **Local `spark-submit` / `pyspark` shell**: `http://localhost:4040` while
  the job is running (4041, 4042... if multiple sessions are open).

Keep the UI open in a second window the whole time you run the notebook.

---

## Step 1 — Jobs tab: how many jobs, and how long?

Run the notebook cell-by-cell (not all at once) and watch the **Jobs** tab.

**What to look for:**
- A job fires for `orders.count()`, another for `customers.count()`, another
  for `products.count()` — three full jobs before you've even started
  transforming anything.
- A job with a name like `csv at NativeMethodAccessorImpl` that takes a long
  time *before* the count even starts — that's schema inference doing a full
  pre-read pass.
- Later, `show(5)` on `daily_summary` and `show(5)` on `top_customers` each
  trigger their own full job, and the **stage list inside each job repeats
  the same read → UDF → join stages** you already paid for.

**Diagnosis:** count the number of jobs. If you see 4+ jobs that each redo
"read CSV → clean → join" from scratch, you've found the missing-cache
problem (Problem 8) and the schema-inference problem (Problem 1) at once.

---

## Step 2 — Stages tab: find the slowest stage

Open the slowest job, click into its **Stages** tab, sort by **Duration**.

**What to look for in the join stage specifically:**
- **Shuffle Read / Shuffle Write** columns with large byte counts — this
  tells you a wide/shuffle join happened (Problem 6). Compare the size of
  `customers`/`products` (should be small) against `orders` (huge) — if both
  sides are being shuffled, nobody told Spark it could broadcast the small
  side.
- Click into the stage's **task list**. Sort by **Duration** descending.
  If you see most tasks finish in seconds but a handful finish in minutes,
  that's **skew** (Problem 7) — those tasks are the reducers that landed the
  "hot" customer_id or product_id keys.
- Check the **Summary Metrics** for that stage: look at **Max vs Median**
  task duration and **Max vs Median Shuffle Read Size**. A max that's 10–50x
  the median is a skew fingerprint.

**Diagnosis:** wide gap between median and max task duration/shuffle size
inside one stage = data skew on the join key.

---

## Step 3 — SQL / DataFrame tab: read the query plan

Open the **SQL** tab (or run `daily_summary.explain(True)` in a cell) and
look at the physical plan.

**What to look for:**
- `SortMergeJoin` instead of `BroadcastHashJoin` for the customers/products
  joins — confirms Problem 6 directly instead of inferring it from shuffle
  bytes.
- `BatchEvalPython` / `PythonUDF` nodes in the plan — these are your two UDFs
  (Problems 3–4). Everything below/around them can't be fused into Spark's
  whole-stage codegen, and rows physically cross the JVM↔Python boundary.
- The **Filter** node for `status == 'COMPLETED'` sitting *above* the Join
  nodes in the plan tree, not pushed below them — confirms Problem 5: Spark
  joined everything first, then threw ~75% of it away.

---

## Step 4 — Executors tab: memory and spill

Click **Executors**, then look at **Storage Memory** and check the
**Stages** page again for a **Spill (Memory)** / **Spill (Disk)** column.

**What to look for:**
- Any non-zero spill on the `groupBy(...).agg(collect_list(...))` stage
  (Problem 9) — building large arrays per hot key can exceed the task's
  memory allotment and spill to disk, which is very slow.
- A single, very long-running task at write time for `coalesce(1)`
  (Problem 10) with high **Shuffle Read** into it — that's every partition's
  data funneling into one task before the final write.

---

## Step 5 — The filesystem itself: the small-file problem

Outside the Spark UI, list the output directories:

```python
# Databricks
display(dbutils.fs.ls("/tmp/spark_opt_demo/raw/orders_csv"))

# local
import os
print(len(os.listdir("/tmp/spark_opt_demo/raw/orders_csv")))
```

**What to look for:** thousands of small CSV part-files from the
`repartition(2000)` write in `01_generate_dataset.py`. Every downstream read
has to open, schedule, and close a task for each one — massive overhead
relative to the tiny amount of data in each file. This is the classic
**small-file problem**, distinct from (but related to) the `coalesce(1)`
**large-file** problem on the way out.

---

## Step 6 — Build your fix list

Before opening `04_notebook_OPTIMIZED.py`, write down, in your own words, one
fix per problem you found. You should have something like:

| Symptom you saw | Root cause | Fix technique |
|---|---|---|
| Extra job before count | `inferSchema=True` | Explicit `StructType` schema |
| Repeated identical stages across 4 jobs | No caching | `.cache()`/`.persist()` at the reuse point |
| `BatchEvalPython` in plan | Python UDFs | Native `F.regexp_replace`, `F.concat` |
| `SortMergeJoin` on small dim table | No broadcast hint | `F.broadcast(customers)` |
| Few tasks much slower than the rest | Skewed join key | Salting, or AQE skew join optimization |
| Filter above Join in plan | Late filtering | Filter before/during join, or predicate pushdown via Parquet |
| Spill on groupBy+collect_list | Unbounded array per key | Cap with `slice()`, or restructure the aggregation |
| One huge write task | `coalesce(1)` | `repartition(N)` sized to data volume, or just remove the coalesce |
| Thousands of tiny input files | Over-partitioned write | Right-sized `repartition()` on write in the generator, or `OPTIMIZE`/compaction |
| Full table scans downstream | No output partitioning | `.write.partitionBy("region", "order_date")` |

Once you've filled this in yourself, open `04_notebook_OPTIMIZED.py` and
compare — it applies exactly this fix list, plus a short explanation inline
at each change of *why* that technique works and what metric in the Spark UI
should move as a result.

---

## Extra credit: measure the improvement

Wrap each notebook's final action in a timer so you have a real before/after
number to talk about in interviews:

```python
import time
t0 = time.time()
daily_summary.write.mode("overwrite").parquet(OUT_PATH)
print(f"Elapsed: {time.time() - t0:.1f}s")
```

Run the SLOW notebook and the OPTIMIZED notebook on the same cluster size and
record both numbers — that's your "reduced job runtime by X%" talking point.
