# Lab 10: Tips & Reference Guide

This guide covers the theory behind the lab exercises. Read the relevant
section when you get stuck or want a deeper explanation. The
[instructions](lab10_instructions.md) tell you *what* to do; this file
explains *why* it works.

---

## 1. Spark Architecture

### Driver and Executors

A Spark application has two roles:

- **Driver** — the JVM process that runs your Python script, builds the DAG,
  and schedules tasks. In local mode, this is your laptop.
- **Executor** — the worker processes that run tasks. In `local[*]` mode,
  the driver *is* the executor — it spawns one thread per CPU core.

```
local[*]  →  one process, N threads  (N = number of CPU cores)
local[4]  →  one process, 4 threads
local[1]  →  single-threaded (deterministic, good for tests)
```

!!! note
    `local[*]` is fine for development. On a real cluster you would set
    `.master("spark://host:7077")` or use YARN/Kubernetes.

### The Spark UI

Every SparkSession exposes a web UI at `http://localhost:4040`. Key tabs:

| Tab | What it shows |
|---|---|
| **Jobs** | One row per action (`collect`, `show`, `count`, ...) |
| **Stages** | Sub-units of a job; stage boundaries = shuffles |
| **SQL/DataFrame** | Queryplan DAG for DataFrame/SQL jobs (most useful) |
| **Storage** | Cached DataFrames and RDDs |

Always open the SQL/DataFrame tab after an action — it shows the physical
plan as a visual graph and lets you see exactly where shuffles and filters land.

---

## 2. RDD vs DataFrame API

Spark has two main APIs for data manipulation:

| | RDD | DataFrame |
|---|---|---|
| **Abstraction** | Low-level: distributed collection of Java objects | High-level: distributed table with named columns and types |
| **Optimisation** | None — you control every step | Catalyst optimizer rewrites your plan |
| **Performance** | Slower (Python ↔ JVM serialisation on each operation) | Faster (Catalyst generates JVM bytecode; avoids Python for most work) |
| **Readability** | Verbose, MapReduce style | Concise, SQL-like |
| **When to use** | Fine-grained control, non-tabular data, custom serialisation | Almost always — default choice |

**Rule of thumb:** use DataFrames. Use RDDs only when you need low-level
partition manipulation (like `mapPartitionsWithIndex` in Exercise 3.1) or
when working with non-tabular data.

### Catalyst Optimizer

When you write a DataFrame pipeline, Spark does not execute it literally.
Catalyst applies rule-based rewrites:

- **Predicate pushdown** — moves filters as early as possible (before joins,
  before scans). This is why in Exercise 2.2 you may see filters appear
  before the join in `.explain()` even though you wrote them after.
- **Column pruning** — drops columns that are never used downstream.
- **Join reordering** — in some cases, reorders joins for efficiency.

None of this happens with RDDs — you get exactly what you write.

---

## 3. Lazy Evaluation and the DAG

### Transformations vs Actions

Every Spark operation is either a **transformation** or an **action**:

| Category | Examples | Effect |
|---|---|---|
| **Transformation** | `map`, `filter`, `flatMap`, `reduceByKey`, `join`, `groupBy`, `select`, `withColumn` | Adds a step to the DAG. No execution. |
| **Action** | `collect`, `count`, `show`, `first`, `take`, `save` | Triggers execution of the entire plan. |

```python
rdd = sc.parallelize(data)          # transformation
mapped = rdd.flatMap(...)           # transformation
reduced = mapped.reduceByKey(...)   # transformation — nothing computed yet
result = reduced.collect()          # ACTION — Spark executes all of the above
```

!!! warning "collect() brings all data to the driver"
    Only use `collect()` when the result fits in driver memory. For large
    DataFrames, use `show(n)`, `take(n)`, or write to disk instead.

### How to Read `.explain()`

```
== Physical Plan ==
AdaptiveSparkPlan isFinalPlan=false
+- Sort [count#12 DESC NULLS LAST], true, 0
   +- Exchange rangepartitioning(count#12 DESC NULLS LAST, 200), ENSURE_REQUIREMENTS
      +- HashAggregate(keys=[word#8], functions=[count(1)])
         +- Exchange hashpartitioning(word#8, 200), ENSURE_REQUIREMENTS
            +- HashAggregate(keys=[word#8], functions=[partial_count(1)])
               +- Generate explode(...)
                  +- Project [lower(text#0) AS text#5]
                     +- Scan ExistingRDD[text#0]
```

Read bottom-to-top. Each `Exchange` node is a **shuffle** (data moves across
the network/threads). In this plan there are two shuffles: one for `groupBy`
and one for `orderBy`.

---

## 4. Shuffles and Stages

### What Causes a Shuffle

A shuffle happens when Spark needs to re-group data by a new key, which
requires rows to move between partitions (potentially across nodes):

| Operation | Shuffle? | Why |
|---|---|---|
| `filter`, `select`, `withColumn` | No | Row-local; each partition processes independently |
| `map`, `flatMap` | No | Row-local |
| `reduceByKey`, `groupBy` | Yes | All rows with the same key must land together |
| `join` | Yes | Rows from both sides with the same key must be co-located |
| `orderBy`, `sortBy` | Yes | Global sort requires a global view of all data |
| `repartition` | Yes | Explicitly moves data |
| `coalesce` | No | Merges adjacent partitions; no data movement |

### Stages

Spark splits a job into **stages** at every shuffle boundary. Within a stage,
all tasks run independently (no data exchange needed). Across stages, all
tasks from one stage must complete before the next stage can begin (the
shuffle write must finish before the shuffle read can start).

```
Stage 1: read + flatMap + partial reduceByKey
             ↓  [shuffle: Exchange hashpartitioning]
Stage 2: final reduceByKey + sort
             ↓  [shuffle: Exchange rangepartitioning]
Stage 3: final output
```

**Rule:** number of stages ≈ number of shuffles + 1.

---

## 5. Partitioning

### What a Partition Is

A partition is the unit of parallelism in Spark — one task processes one
partition. More partitions = more tasks = more parallelism (up to the number
of cores).

Default number of partitions:
- `sc.parallelize(data)` → `sc.defaultParallelism` (= number of cores in `local[*]`)
- After a shuffle → `spark.sql.shuffle.partitions` (default: 200)

### `repartition` vs `coalesce`

| | `repartition(n)` | `repartition(n, col)` | `coalesce(n)` |
|---|---|---|---|
| **Shuffle** | Yes | Yes | No |
| **Result** | n balanced partitions | n partitions, same-key rows co-located | n merged partitions |
| **Use case** | Balance after skew | Pre-partition before repeated groupBy/join | Reduce small files before write |

!!! warning "repartition(n, col) does not eliminate shuffles on col"
    Unlike Hive bucketing, PySpark's `repartition(n, col)` does not tell the
    query planner to skip future shuffles on `col`. A subsequent `groupBy(col)`
    will still add an `Exchange` node in the physical plan — Spark does not
    track that the data is already co-located. This is why the timing
    improvement in Exercise 3.2 may be modest or even negative (the
    repartition itself costs a shuffle).

### Hot Spots and Partition Skew

When partitioning by a column with an uneven value distribution, some
partitions will be much larger than others. The task processing the large
partition becomes the bottleneck — all other tasks finish and sit idle while
waiting for it.

In this lab: `PAISES` has 40 "ES" entries out of 100. After
`repartition(5, "pais")`, all ES rows end up in one partition (~200 K rows)
while other partitions may hold only ~60–75 K rows each. This is the
**hot spot** problem described in Chapter 6.

Mitigations (outside the scope of this lab):
- Add a salt (random suffix) to the key to spread hot keys across partitions.
- Use Spark's Adaptive Query Execution (AQE) to split skewed partitions
  automatically (`spark.sql.adaptive.skewJoin.enabled = true`).

---

## 6. When to Use Spark vs Single-Node Tools

Spark adds JVM startup cost, serialisation overhead, and scheduling latency.
For data that fits comfortably in RAM on one machine, tools like
**Polars** or **pandas** are typically 5–20× faster.

| Data size | Fits in RAM? | Recommended tool |
|---|---|---|
| < a few GB | Yes | Polars, pandas |
| RAM-limited, single machine | Borderline | Polars with `LazyFrame`, DuckDB |
| Multi-machine or > RAM | No | Spark, Dask |
| Petabyte-scale | No | Spark on a cluster (EMR, Databricks, GKE) |

**Design question for reflection:** your company has a 50 GB sales dataset
growing at 1 GB/day. At what point does Spark become the right choice?
What if it were 5 TB?

---

## 7. Fault Tolerance: the DAG as a Recovery Mechanism

Spark achieves fault tolerance without replicating intermediate data by
recording the **lineage** of every RDD/DataFrame — the full sequence of
transformations that produced it.

If an executor crashes and loses a partition, Spark re-runs only the tasks
for that partition, using the lineage to recompute from the nearest available
ancestor (the input data or the last checkpoint/cache).

This is fundamentally different from a replicated database (Chapter 5), which
keeps copies of data to recover. Spark keeps copies of *instructions* instead.

### Caching

Recomputing from scratch is fine for short pipelines, but expensive if a
DataFrame is used multiple times. `df.cache()` tells Spark to materialise and
store the result in executor memory after it is first computed, so subsequent
actions reuse it.

```python
df2 = df1.withColumn("iva", F.col("importe") * 0.21)
df2.cache()
df2.count()   # forces materialisation — now stored in memory

df3 = df2.groupBy("producto").agg(F.sum("iva"))   # reuses cache
df4 = df2.filter(F.col("iva") > 100)              # reuses cache

df2.unpersist()   # release memory when no longer needed
```

Trade-off: caching consumes executor memory. Only cache DataFrames that are
reused multiple times in the same application.

---

## 8. Spark UI Quick Reference

### Jobs tab

One row per action. Columns: job ID, description (function name), submitted
time, duration, number of stages, number of tasks. Click a job to drill into
its stages.

### Stages tab

One row per stage. A stage is a set of tasks that can run without a shuffle.
The "Input" and "Shuffle Write/Read" columns show how much data moved across
partitions — this is your shuffle cost.

### SQL/DataFrame tab

Only populated for DataFrame/SQL queries (not raw RDD operations). Shows a
visual DAG of the physical plan. Nodes are colour-coded:

- **WholeStageCodegen** (blue) — optimised path, JVM bytecode generated
- **Exchange** (yellow/orange) — shuffle
- **Scan** (green) — reading from memory or disk

Click a node to see its metrics (rows processed, time, spill to disk).

### Storage tab

Lists cached DataFrames and RDDs. Shows memory and disk usage per partition,
and the fraction of data that is actually cached (may be < 100% if memory is
limited and Spark evicted some partitions).
