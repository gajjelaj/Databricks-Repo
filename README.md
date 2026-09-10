# Spark Optimization Practice Kit

Four files, run in this order:

1. **`01_generate_dataset.py`** — generates ~25M-row `orders` fact table plus
   `customers` (500K) and `products` (50K) dimension tables as CSV, with
   *deliberate* skew (a handful of "hot" customer/product ids carry a
   disproportionate share of rows) and a *deliberate* small-file layout.
   Set `SCALE = 0.1` first for a fast smoke test, then rerun at `SCALE = 1.0`
   for the full dataset.
2. **`02_notebook_SLOW_baseline.py`** — a realistic, functionally-correct but
   slow pipeline with 11 planted anti-patterns (listed at the bottom of the
   file). Run it and watch it struggle.
3. **`03_troubleshooting_guide.md`** — walks you through the Spark UI
   (Jobs → Stages → SQL plan → Executors → filesystem) to find and name each
   problem yourself, with a fix-mapping table to fill in.
4. **`04_notebook_OPTIMIZED.py`** — the same business output, with every
   anti-pattern fixed and an inline comment at each fix explaining the
   technique and why it works.

## Where to run this

Everything is written as **Databricks notebook source** files (the
`# Databricks notebook source` / `# COMMAND ----------` markers), because
that matches the medallion-architecture project you're already building. You
can:

- **Import directly into Databricks**: Workspace → Import → upload the
  `.py` file, choose "Source" format — Databricks renders the `# COMMAND`
  markers as cells automatically.
- **Run as plain PySpark locally**: the `# MAGIC %md` and `# COMMAND` lines
  are just comments to a non-Databricks interpreter, so `python
  01_generate_dataset.py` (with `pyspark` installed and a `spark` session
  created) works too — except for the `dbutils.fs.ls` sanity check at the
  end of file 1, which has a local fallback built in.

`BASE_PATH = "/tmp/spark_opt_demo"` — change this to a DBFS or ADLS/S3 path
if you're on a real workspace and want the data to persist beyond a cluster
restart.

## How to actually use this for learning

1. Run file 1 once.
2. Run file 2 cell by cell, Spark UI open in a second tab, and genuinely try
   to spot each slowdown as it happens — don't skip to the answers.
3. Work through file 3 top to bottom, filling in the fix table with your own
   words before reading the fixed notebook.
4. Run file 4, compare the printed elapsed time against file 2's, and use
   the mapping table at the bottom to confirm each fix shows up in the Spark
   UI the way the guide predicted.
5. Optional next step: convert the raw CSV in `01_generate_dataset.py` to
   Delta Lake tables and repeat the exercise with `OPTIMIZE` + `ZORDER BY`
   on the fact table — a natural extension given the Delta Lake medallion
   project you're already building.
