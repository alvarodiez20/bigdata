# Lab 10: Tips & Reference Guide

This guide covers the theory behind the lab exercises. Read the relevant
section when you get stuck or want a deeper explanation. The
[instructions](lab10_instructions.md) tell you *what* to do; this file
explains *why* it works.

---

## 1. Spark Architecture

### Java Requirement

PySpark runs Python code but delegates heavy lifting to the JVM. You need
**Java 17 or later** installed and visible on your `PATH`:

```bash
# Check
java -version

# macOS (Homebrew)
brew install openjdk@17
export JAVA_HOME=/opt/homebrew/opt/openjdk@17/libexec/openjdk.jdk/Contents/Home

# Ubuntu / Debian
sudo apt install openjdk-17-jdk

# Add JAVA_HOME to your shell profile (~/.zshrc or ~/.bashrc) to make it permanent.
```

#### Windows setup — recommended: WSL2

The easiest way to run Spark on Windows is **WSL2** (Windows Subsystem for
Linux). It gives you a real Ubuntu shell inside Windows, so you follow the
exact same Linux steps above — no winutils, no PATH gymnastics.

**Step 1 — Install WSL2 with Ubuntu**

Open PowerShell as Administrator and run:

```powershell
wsl --install
```

This installs WSL2 and Ubuntu in one go. Reboot when prompted. On first launch,
Ubuntu will ask you to create a Unix username and password (can be anything,
does not need to match your Windows account).

If you already have WSL but an older version, upgrade it:

```powershell
wsl --update
wsl --set-default-version 2
```

**Step 2 — Install Java 17 inside WSL**

Open the **Ubuntu** terminal (search "Ubuntu" in the Start menu) and run the
Linux commands from the section above:

```bash
sudo apt update && sudo apt install openjdk-17-jdk -y
java -version   # must show 17.x
```

Then add `JAVA_HOME` to your shell profile:

```bash
echo 'export JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64' >> ~/.bashrc
source ~/.bashrc
```

**Step 3 — Run the lab inside WSL**

Open the Ubuntu terminal, navigate to your project folder, and use `uv` / `python`
exactly as you would on Linux or macOS. VS Code users can open the WSL folder
directly with the **Remote - WSL** extension (`code .` from the Ubuntu terminal).

That's it — no winutils needed.

---

#### Windows setup — native (no WSL)

If you cannot use WSL2 (e.g. on a managed corporate machine where it is
disabled), you can run Spark natively on Windows with two extra steps.

**Step 1 — Install Java 17**

```powershell
# Run PowerShell as Administrator
winget install Microsoft.OpenJDK.17
```

If `winget` is unavailable, install [Eclipse Temurin 17](https://adoptium.net/temurin/releases/?version=17) manually.

Verify (open a **new** terminal after installing):

```powershell
java -version   # must show 17.x
```

**Step 2 — Set `JAVA_HOME`**

In PowerShell, first find the exact JDK path:

```powershell
(Get-Command java).Source
# e.g. C:\Program Files\Microsoft\jdk-17.0.15.6-hotspot\bin\java.exe
# JAVA_HOME is everything before \bin
```

Then set it as a system variable (run as Administrator):

```powershell
$jdkPath = "C:\Program Files\Microsoft\jdk-17.0.15.6-hotspot"  # adjust to your path
[System.Environment]::SetEnvironmentVariable("JAVA_HOME", $jdkPath, "Machine")
```

Restart any open terminals after this.

**Step 3 — Install winutils**

Spark needs Hadoop's native tools to manage temp directories on Windows.

1. Download `winutils.exe` and `hadoop.dll` for Hadoop 3.x from
   [`https://github.com/cdarlint/winutils`](https://github.com/cdarlint/winutils)
   (navigate to `hadoop-3.3.6/bin/`).
2. Create `C:\hadoop\bin` and place both files there.
3. Set `HADOOP_HOME` and add `C:\hadoop\bin` to `Path` (run as Administrator):

```powershell
[System.Environment]::SetEnvironmentVariable("HADOOP_HOME", "C:\hadoop", "Machine")
$current = [System.Environment]::GetEnvironmentVariable("Path", "Machine")
[System.Environment]::SetEnvironmentVariable("Path", "$current;C:\hadoop\bin", "Machine")
```

Restart any open terminals — environment variable changes are not picked up by
sessions that were already open.

**Step 4 — Verify**

```powershell
java -version           # 17.x
$env:JAVA_HOME          # JDK path
$env:HADOOP_HOME        # C:\hadoop
winutils ls C:\         # lists files without error
```

Then confirm PySpark loads:

```python
from pyspark.sql import SparkSession
spark = SparkSession.builder.master("local[*]").appName("test").getOrCreate()
spark.range(5).show()
spark.stop()
```

!!! warning "Common native-Windows pitfalls"
    - **`where java` does nothing** — `where` is a PowerShell alias for
      `Where-Object`. Use `where.exe java` or `(Get-Command java).Source`.
    - **Paths with spaces** — if your Windows username contains a space,
      Spark may fail writing temp files. Fix with:
      ```powershell
      [System.Environment]::SetEnvironmentVariable("SPARK_LOCAL_DIRS", "C:\tmp\spark", "Machine")
      New-Item -ItemType Directory -Force -Path "C:\tmp\spark"
      ```
    - **Missing winutils** — errors like `Could not locate executable null\bin\winutils.exe`
      mean `HADOOP_HOME` is wrong or not set. Check that it points to `C:\hadoop`,
      not `C:\hadoop\bin`.
    - **Antivirus quarantine** — some AV tools delete `winutils.exe` on download.
      Add an exception for `C:\hadoop\bin` if it disappears.

If Java is missing, PySpark fails at startup with:
`Java gateway process exited before sending its port number.`

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

### Common Gotcha: Tuple Syntax for Single-Column DataFrames

When creating a DataFrame from a list of strings, each element must be a
**tuple** — even if there is only one column:

```python
# CORRECT — trailing comma makes it a tuple
spark.createDataFrame([(line,) for line in corpus], ["text"])

# WRONG — (line) is just parentheses, NOT a tuple
spark.createDataFrame([(line) for line in corpus], ["text"])
```

Without the comma, Python treats `(line)` as a plain string. Spark then
iterates over each **character** of the string as a separate row, producing
nonsense output. This is a classic Python gotcha — always use `(value,)` for
single-element tuples.

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

### Common Gotcha: `mapPartitionsWithIndex` Return Type

When using `mapPartitionsWithIndex` to count rows per partition, the lambda
must return a **list** containing the result tuple — not a bare tuple:

```python
# CORRECT — returns a list with one tuple per partition
df.rdd.mapPartitionsWithIndex(
    lambda idx, it: [(idx, sum(1 for _ in it))]
)

# WRONG — returns a bare tuple, which Spark iterates as two integers
df.rdd.mapPartitionsWithIndex(
    lambda idx, it: (idx, sum(1 for _ in it))
)
```

`mapPartitionsWithIndex` calls `iter()` on whatever you return. A bare tuple
`(0, 42)` becomes the sequence `0, 42` — two separate elements instead of
one `(partition_id, count)` pair. Wrapping in `[...]` fixes this.

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
`repartition(5, "pais")`, the 5 country codes are hashed into 5 buckets —
but `hash(value) % 5` does not guarantee a 1-to-1 mapping. In practice,
some partitions receive multiple countries (e.g. ~200 K rows) while others
remain **completely empty**. The partition holding all ES rows (~200 K)
becomes the bottleneck. This is the **hot spot** problem described in
Chapter 6, made worse by hash collisions.

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

---

## 9. TODO Reference

Near-solutions for all 8 functions. Read the explanation first — the code
makes more sense once you understand *why* each step is there.

### TODO 1 — `wordcount_rdd`

```python
def wordcount_rdd(sc, corpus, num_partitions=3):
    rdd = sc.parallelize(corpus, numSlices=num_partitions)
    mapped = rdd.flatMap(lambda line: [(word.lower(), 1) for word in line.split()])
    counts = mapped.reduceByKey(lambda a, b: a + b)
    result = counts.collect()                  # first action — triggers execution
    return sorted(result, key=lambda x: -x[1])
```

`parallelize` → `flatMap` → `reduceByKey` are all transformations: they build
the DAG but run nothing. `collect()` is the action that fires the plan.
The final sort happens in Python on the driver — no extra Spark stage needed
for a small result.

### TODO 2 — `wordcount_dataframe`

```python
def wordcount_dataframe(spark, corpus):
    df = spark.createDataFrame([(line,) for line in corpus], ["text"])
    return (
        df
        .select(F.explode(F.split(F.lower(F.col("text")), " ")).alias("word"))
        .groupBy("word")
        .count()
        .orderBy(F.desc("count"))
    )
```

Each string must be wrapped in a one-element tuple `(line,)` — without the
trailing comma Python treats `(line)` as plain parentheses, not a tuple, and
Spark iterates over the characters of the string instead of the strings
themselves. `explode(split(...))` is the DataFrame equivalent of `flatMap`.

### TODO 3 — `create_ventas`

```python
def create_ventas(spark, n_rows=500_000, seed=42):
    random.seed(seed)
    data = []
    for i in range(n_rows):
        data.append((
            i,
            random.choice(PAISES),
            random.choice(PRODUCTOS),
            round(random.uniform(10, 2000), 2),
            f"2025-{random.randint(1, 12):02d}-{random.randint(1, 28):02d}",
            random.randint(1, 10000),
        ))
    return spark.createDataFrame(
        data,
        ["venta_id", "pais", "producto", "importe", "fecha", "cliente_id"],
    )
```

`random.seed(seed)` before any `random.*` call makes the dataset reproducible —
every run with the same seed produces identical rows, which is essential for
deterministic tests. Day range 1–28 avoids invalid dates (e.g. Feb 30).

### TODO 4 — `create_clientes`

```python
def create_clientes(spark, n_clientes=10_000, seed=42):
    random.seed(seed)
    data = [
        (i, f"Cliente_{i}", random.choice(TIPOS))
        for i in range(1, n_clientes + 1)
    ]
    return spark.createDataFrame(data, ["cliente_id", "nombre", "tipo"])
```

`cliente_id` starts at 1 (matching the `random.randint(1, 10000)` range in
`create_ventas`) so the join on `cliente_id` produces matches. Starting at 0
would leave `cliente_id=0` unmatched in an inner join.

### TODO 5 — `analytics_pipeline`

```python
def analytics_pipeline(ventas, clientes, pais="ES", tipo="premium"):
    return (
        ventas
        .join(clientes, on="cliente_id", how="inner")
        .filter(F.col("pais") == pais)
        .filter(F.col("tipo") == tipo)
        .groupBy("producto")
        .agg(
            F.count("*").alias("num_ventas"),
            F.round(F.sum("importe"), 2).alias("total"),
            F.round(F.avg("importe"), 2).alias("media"),
        )
        .orderBy(F.desc("total"))
    )
```

Three shuffles in this plan: one for the join (both sides hash-partitioned by
`cliente_id`), one for `groupBy("producto")`, one for `orderBy`. Catalyst's
predicate pushdown moves the `filter` calls before the join in the physical
plan even though they appear after it in your code — verify with `.explain()`.

### TODO 6 — `partition_distribution`

```python
def partition_distribution(df):
    raw = df.rdd.mapPartitionsWithIndex(
        lambda idx, it: [(idx, sum(1 for _ in it))]
    ).collect()
    return sorted(raw, key=lambda x: x[0])
```

The lambda **must** return a list `[...]`, not a bare tuple. If you return
`(idx, count)` without the outer list, `mapPartitionsWithIndex` calls `iter()`
on the tuple and produces two separate integer rows instead of one pair.
`sum(1 for _ in it)` counts without loading all rows into memory.

### TODO 7 — `repartition_by_column`

```python
def repartition_by_column(df, col, n_partitions):
    return df.repartition(n_partitions, col)
```

One line — but the *effect* is worth understanding: all rows with the same
value of `col` land in the same partition via `hash(col) % n_partitions`.
This always causes a full shuffle. See the gotcha in section 5 about why this
does **not** eliminate a subsequent `groupBy(col)` shuffle.

### TODO 8 — `measure_groupby_time`

```python
def measure_groupby_time(df, group_col, agg_col):
    t0 = time.perf_counter()
    df.groupBy(group_col).sum(agg_col).collect()
    return time.perf_counter() - t0
```

`.collect()` is critical — without an action, Spark does nothing (lazy
evaluation) and the measured time would be near zero. `time.perf_counter()`
gives sub-millisecond resolution and is unaffected by system clock adjustments,
making it more reliable than `time.time()` for short intervals.
