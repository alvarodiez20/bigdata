# Lab 10: PySpark — Instructions

In this lab you will use PySpark in local mode to understand the distributed
programming model at the core of big-data systems: lazy evaluation, the DAG
execution engine, shuffles, and partition-level parallelism — all without
needing a cluster.

**Spark UI is central to this lab.** Open <http://localhost:4040> in a browser
*before* running your first job and keep it open throughout. Every time you
trigger an action you will see the stages, tasks, and shuffles appear there.

[Tips & Reference Guide](lab10_guide.md)

---

## Pre-flight Checklist

- [ ] Install PySpark: `uv sync --group lab10`
- [ ] Verify: `python -c "import pyspark; print(pyspark.__version__)"`
- [ ] Open `src/lab10.py` — fill in the 8 TODOs
- [ ] Run the tests: `uv run pytest tests/test_lab10.py -v`
- [ ] Run the demo: `uv run python src/lab10.py` (Spark UI at localhost:4040)

---

## Exercise 1.1 — WordCount with RDDs (TODO 1)

!!! objective
    Implement `wordcount_rdd(sc, corpus, num_partitions)` using the low-level
    RDD API. This is the MapReduce pattern made explicit: map, then reduce.

**Function:** `wordcount_rdd` in `src/lab10.py`

**Steps:**

1. Call `sc.parallelize(corpus, numSlices=num_partitions)` to distribute the
   list across partitions. No computation yet — this just registers the intent.
2. `flatMap(lambda line: [(w.lower(), 1) for w in line.split()])` — for each
   line, emit one `(word, 1)` tuple per word. `flatMap` flattens the list of
   lists into a flat stream of pairs.
3. `reduceByKey(lambda a, b: a + b)` — groups by the word key and sums the 1s.
   This is a **wide transformation**: it requires a shuffle.
4. `collect()` — this is the first **action** in the chain. Only now does Spark
   execute the entire plan. Watch it appear in the Spark UI.
5. Sort the Python list by count descending before returning.

**Verify:** `uv run pytest tests/test_lab10.py -k "TestWordcountRdd" -v`

**Reflect:**

1. At which line does Spark actually start computing? Why do all the
   transformations before it do nothing?
2. Check `counts.getNumPartitions()` before the collect. Where did the shuffle
   happen in the DAG?
3. Open the Spark UI → Jobs tab. How many stages does this job have?
   Why that number? (Hint: each shuffle creates a stage boundary.)

---

## Exercise 1.2 — WordCount with DataFrames (TODO 2)

!!! objective
    Re-implement the same word count using the DataFrame API. Compare the
    readability and the execution plan with the RDD version.

**Function:** `wordcount_dataframe` in `src/lab10.py`

**Steps:**

1. `spark.createDataFrame([(line,) for line in corpus], ["text"])` — wrap each
   string in a one-element tuple so Spark can infer a single column schema.
2. `.select(F.explode(F.split(F.lower(F.col("text")), " ")).alias("word"))` —
   chain `lower` → `split` → `explode` to produce one row per word.
3. `.groupBy("word").count()` — aggregate.
4. `.orderBy(F.desc("count"))` — sort descending.

After implementing, call `.explain()` on the result:

```python
df_result = wordcount_dataframe(spark, corpus)
df_result.show(10)
df_result.explain()           # default logical plan
df_result.explain(mode="extended")  # full physical plan
```

**Verify:** `uv run pytest tests/test_lab10.py -k "TestWordcountDataframe" -v`

**Reflect:**

1. Compare the RDD and DataFrame implementations. Which is more readable?
   Which gives Spark more room to optimise?
2. In the physical plan from `.explain()`, find the `Exchange` node. That is
   where the shuffle happens. Is it in the same logical position as in the
   RDD version?
3. Spark UI → SQL/DataFrame tab: click the query link. Can you match the plan
   nodes to the code you wrote?

---

## Exercise 2.1 — Synthetic Sales Data (TODOs 3 & 4)

!!! objective
    Generate two reproducible synthetic DataFrames — `ventas` (500 K rows) and
    `clientes` (10 K rows) — that will feed the analytics pipeline.

**Functions:** `create_ventas` and `create_clientes` in `src/lab10.py`

**`create_ventas` schema** (6 columns):

| Column | Type | Notes |
|---|---|---|
| `venta_id` | int | Sequential (0-based) |
| `pais` | str | From `PAISES` (ES-weighted: 40 out of 100) |
| `producto` | str | From `PRODUCTOS` |
| `importe` | float | `round(random.uniform(10, 2000), 2)` |
| `fecha` | str | `"2025-MM-DD"` |
| `cliente_id` | int | `random.randint(1, 10000)` |

**`create_clientes` schema** (3 columns):

| Column | Type | Notes |
|---|---|---|
| `cliente_id` | int | 1-based sequential |
| `nombre` | str | `"Cliente_<id>"` |
| `tipo` | str | `"premium"` or `"standard"` |

Use `random.seed(seed)` before generating data so results are reproducible.

After implementing, inspect what you built:

```python
ventas = create_ventas(spark, n_rows=500_000)
clientes = create_clientes(spark, n_clientes=10_000)
print(f"Ventas:   {ventas.count():,} rows")
print(f"Clientes: {clientes.count():,} rows")
ventas.printSchema()
ventas.show(5)
```

**Verify:** `uv run pytest tests/test_lab10.py -k "TestCreateVentas or TestCreateClientes" -v`

---

## Exercise 2.2 — Analytics Pipeline (TODO 5)

!!! objective
    Implement a multi-stage pipeline that joins the two DataFrames, filters,
    aggregates, and sorts. Identify every shuffle in the execution plan.

**Function:** `analytics_pipeline` in `src/lab10.py`

**Pipeline (in this order):**

1. `ventas.join(clientes, on="cliente_id", how="inner")` — wide transformation, shuffle
2. `.filter(F.col("pais") == pais)` — narrow transformation, no shuffle
3. `.filter(F.col("tipo") == tipo)` — narrow transformation, no shuffle
4. `.groupBy("producto")` — wide transformation, shuffle
5. `.agg(F.count("*").alias("num_ventas"), F.round(F.sum("importe"), 2).alias("total"), F.round(F.avg("importe"), 2).alias("media"))`
6. `.orderBy(F.desc("total"))` — wide transformation, shuffle

Time the full execution and inspect the plan:

```python
import time
t0 = time.perf_counter()
resultado = analytics_pipeline(ventas, clientes)
resultado.show()
print(f"Time: {time.perf_counter() - t0:.2f}s")
resultado.explain()
```

**Verify:** `uv run pytest tests/test_lab10.py -k "TestAnalyticsPipeline" -v`

**Reflect:**

1. Count the shuffles in `.explain()` — each `Exchange` node is one shuffle.
   How many are there? Does that match your prediction from the code above?
2. Spark UI → Stages tab: how many stages does this job produce? Why?
   (Rule of thumb: stages = shuffles + 1.)
3. Look at the physical plan carefully. Spark applies *predicate pushdown*:
   it moves the `filter` operations as early as possible. Can you see it?
   Does Spark push the `pais` filter before or after the join?

---

## Exercise 3.1 — Partition Distribution (TODO 6)

!!! objective
    Implement `partition_distribution(df)` to count how many rows live in each
    partition. Use this to observe the hot-spot problem from Chapter 6.

**Function:** `partition_distribution` in `src/lab10.py`

**Steps:**

1. `df.rdd.mapPartitionsWithIndex(lambda idx, it: [(idx, sum(1 for _ in it))])` —
   the lambda receives `(partition_index, row_iterator)`. Consuming the iterator
   with `sum(1 for _ in it)` counts rows without materialising them.
2. `.collect()` to bring results to the driver.
3. Sort by `partition_id` ascending before returning.

Use it to inspect the partition layout before and after repartitioning:

```python
print(f"Default partitions: {ventas.rdd.getNumPartitions()}")

ventas_por_pais = ventas.repartition(5, "pais")
print(f"After repartition(5, 'pais'): {ventas_por_pais.rdd.getNumPartitions()}")

for part_id, count in partition_distribution(ventas_por_pais):
    bar = "█" * (count // 5000)
    print(f"  Partition {part_id}: {count:>7} rows  {bar}")
```

**Verify:** `uv run pytest tests/test_lab10.py -k "TestPartitionDistribution" -v`

**Reflect:**

1. Are the 500 K rows evenly distributed after `repartition(5, "pais")`?
   Why or why not? Which country creates the largest partition?
   Connect this to the *hot spot* concept from Chapter 6.
2. How many distinct `pais` values are there? Can you have more balanced
   partitions than the number of distinct values? Why not?

---

## Exercise 3.2 — Repartition and Timing (TODOs 7 & 8)

!!! objective
    Implement `repartition_by_column` and `measure_groupby_time`, then measure
    whether pre-partitioning by `pais` speeds up a `groupBy("pais")` query.

**Functions:** `repartition_by_column` and `measure_groupby_time` in `src/lab10.py`

**`repartition_by_column`:** simply call `df.repartition(n_partitions, col)`.

**`measure_groupby_time`:** use `time.perf_counter()` around the full chain
`df.groupBy(group_col).sum(agg_col).collect()`.

Run the comparison:

```python
ventas_por_pais = repartition_by_column(ventas, "pais", 5)

t_sin = measure_groupby_time(ventas, "pais", "importe")
t_con = measure_groupby_time(ventas_por_pais, "pais", "importe")
print(f"Without repartition: {t_sin:.3f}s")
print(f"With repartition:    {t_con:.3f}s")
```

Open the Spark UI → SQL/DataFrame tab and compare the DAGs for the two queries.

**Verify:** `uv run pytest tests/test_lab10.py -k "TestRepartitionByColumn or TestMeasureGroupbyTime" -v`

**Reflect:**

1. Does `repartition(5, "pais")` eliminate the shuffle in the subsequent
   `groupBy("pais")`? Check the Spark UI DAGs for both queries.
2. What happens if you change the number of partitions to 1? To 200?
   Measure and explain the trade-off.
3. `repartition` always shuffles. `coalesce` only merges partitions (no shuffle).
   When would you use `coalesce` instead? See the [guide](lab10_guide.md#partitioning)
   for details.

---

## What to Submit

1. `src/lab10.py` with all 8 TODOs implemented and your name filled in at the top.
2. All 8 reflection answers written as comments or a markdown cell below each
   exercise section.
3. `uv run pytest tests/test_lab10.py -v` must pass completely.

**Key takeaways to be able to explain:**

- The difference between a *transformation* and an *action* in Spark.
- Why each shuffle creates a stage boundary.
- Why `repartition` by a column does not necessarily eliminate a subsequent
  shuffle on the same column in PySpark (unlike Hive bucketing).
- What the hot-spot problem is and how it relates to partition skew.
