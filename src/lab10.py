"""
Lab 10: PySpark — First Contact

STUDENT NAME: [Your Name Here]

STUDENT REFLECTION:
(Please write a short paragraph here explaining what you learned in this lab)
"""

import random
import time
from typing import Any

import pyspark.sql.functions as F
from pyspark.sql import DataFrame, SparkSession


# ---------------------------------------------------------------------------
# Helper: create a local SparkSession
# ---------------------------------------------------------------------------

def get_spark(app_name: str = "Lab10-BigData") -> SparkSession:
    """
    Creates (or retrieves) a local SparkSession.

    Uses local[*] to run on all available cores. Sets driver memory to 2g.
    The Spark UI is available at http://localhost:4040 while the session is open.

    Args:
        app_name (str): The application name shown in the Spark UI.

    Returns:
        SparkSession: A configured Spark session in local mode.
    """
    return (
        SparkSession.builder
        .appName(app_name)
        .master("local[*]")
        .config("spark.driver.memory", "2g")
        .getOrCreate()
    )


# ---------------------------------------------------------------------------
# Exercise 1: From MapReduce to Spark — WordCount
# ---------------------------------------------------------------------------

def wordcount_rdd(sc: Any, corpus: list[str], num_partitions: int = 3) -> list[tuple[str, int]]:
    """
    Counts word frequencies in a text corpus using the RDD API (MapReduce pattern).

    Steps:
    1. Parallelize the corpus list into an RDD with num_partitions partitions.
    2. flatMap: split each line into lowercase words → emit (word, 1) pairs.
    3. reduceByKey: sum the 1s for each word.
    4. collect() the results and sort them by count descending.

    Note: no computation happens until collect() is called (lazy evaluation).

    Args:
        sc: A SparkContext (e.g. spark.sparkContext).
        corpus (list[str]): List of text lines to count words in.
        num_partitions (int): Number of RDD partitions to create.

    Returns:
        list[tuple[str, int]]: Word-count pairs sorted by count descending.

    Examples:
        >>> spark = get_spark()
        >>> wordcount_rdd(spark.sparkContext, ["the cat and the dog"])
        [('the', 2), ('cat', 1), ('and', 1), ('dog', 1)]
    """
    # TODO 1 — Follow these steps:
    #
    # 1. Parallelize: sc.parallelize(corpus, numSlices=num_partitions)
    #    This creates an RDD but does NOT start any computation.
    #
    # 2. flatMap: for each line, split into words, lowercase them,
    #    and emit (word, 1) pairs.  Hint: line.split() + list comprehension.
    #
    # 3. reduceByKey: sum the 1s for each word key.
    #    This is a wide transformation — it causes a shuffle.
    #
    # 4. collect(): the first ACTION — triggers execution of the full DAG.
    #
    # 5. Sort the collected Python list by count descending, then return it.
    # Step 1: distribute the corpus across num_partitions partitions
    raise NotImplementedError("TODO 1: implement the RDD word count pipeline")


def wordcount_dataframe(spark: SparkSession, corpus: list[str]) -> DataFrame:
    """
    Counts word frequencies in a text corpus using the DataFrame API.

    Steps:
    1. Create a DataFrame from the corpus with a single column "text".
    2. select: explode(split(lower(col("text")), " ")) aliased as "word".
    3. groupBy("word").count().
    4. orderBy count descending.

    The resulting DataFrame has columns: word (string), count (long).
    Spark builds an optimised logical plan — run .explain() to see it.

    Args:
        spark (SparkSession): The active SparkSession.
        corpus (list[str]): List of text lines to count words in.

    Returns:
        DataFrame: DataFrame with columns (word, count), ordered by count desc.

    Examples:
        >>> spark = get_spark()
        >>> df = wordcount_dataframe(spark, ["the cat and the dog"])
        >>> df.show(3)
        +----+-----+
        |word|count|
        +----+-----+
        | the|    2|
        | cat|    1|
        | and|    1|
        +----+-----+
    """
    # TODO 2 — Follow these steps:
    #
    # 1. Create a one-column DataFrame from the corpus:
    #        spark.createDataFrame([(line,) for line in corpus], ["text"])
    #    IMPORTANT: note the trailing comma in (line,) — without it Python
    #    treats (line) as plain parentheses, not a tuple, and Spark will
    #    iterate over each *character* instead of each line.
    #
    # 2. select: chain lower → split → explode to get one row per word.
    #    Use F.lower, F.split, F.explode and .alias("word").
    #
    # 3. groupBy("word").count()
    #
    # 4. orderBy count descending: F.desc("count")
    #
    # Return the resulting DataFrame (do NOT call .collect()).
    raise NotImplementedError("TODO 2: implement the DataFrame word count pipeline")


# ---------------------------------------------------------------------------
# Exercise 2: Multi-Stage Analytics Pipeline
# ---------------------------------------------------------------------------

PAISES = ["ES"] * 40 + ["FR"] * 20 + ["DE"] * 15 + ["IT"] * 15 + ["PT"] * 10
PRODUCTOS = ["laptop", "tablet", "phone", "monitor", "keyboard", "mouse", "headphones"]
TIPOS = ["premium", "standard"]


def create_ventas(spark: SparkSession, n_rows: int = 500_000, seed: int = 42) -> DataFrame:
    """
    Generates a synthetic sales DataFrame.

    Schema:
        venta_id  (int)    — sequential sale identifier
        pais      (str)    — country code, sampled from PAISES (ES-heavy)
        producto  (str)    — product name, sampled from PRODUCTOS
        importe   (float)  — sale amount, uniform in [10.0, 2000.0], 2 decimal places
        fecha     (str)    — date string "2025-MM-DD", month 1-12, day 1-28
                             (day capped at 28 to avoid invalid dates like Feb 30)
        cliente_id (int)   — client ID, uniform in [1, 10000]

    Use random.seed(seed) before generating data for reproducibility.

    Args:
        spark (SparkSession): The active SparkSession.
        n_rows (int): Number of rows to generate.
        seed (int): Random seed for reproducibility.

    Returns:
        DataFrame: Synthetic sales data with 6 columns.

    Examples:
        >>> spark = get_spark()
        >>> ventas = create_ventas(spark, n_rows=100)
        >>> ventas.count()
        100
        >>> ventas.columns
        ['venta_id', 'pais', 'producto', 'importe', 'fecha', 'cliente_id']
    """
    # TODO 3 — Follow these steps:
    #
    # 1. Set the seed: random.seed(seed)
    #
    # 2. Build a list of n_rows tuples.  Each tuple has 6 fields:
    #      - venta_id:   the loop index (0-based)
    #      - pais:       random.choice(PAISES)
    #      - producto:   random.choice(PRODUCTOS)
    #      - importe:    round(random.uniform(10, 2000), 2)
    #      - fecha:      formatted as "2025-MM-DD" (month 1-12, day 1-28)
    #      - cliente_id: random.randint(1, 10000)
    #
    # 3. Return spark.createDataFrame(data, [column_names...])
    raise NotImplementedError("TODO 3: generate the synthetic ventas DataFrame")


def create_clientes(spark: SparkSession, n_clientes: int = 10_000, seed: int = 42) -> DataFrame:
    """
    Generates a synthetic clients DataFrame.

    Schema:
        cliente_id (int)  — unique client identifier (1-based)
        nombre     (str)  — "Cliente_<id>"
        tipo       (str)  — "premium" or "standard", sampled from TIPOS

    Use random.seed(seed) before generating data.

    Args:
        spark (SparkSession): The active SparkSession.
        n_clientes (int): Number of client rows to generate.
        seed (int): Random seed for reproducibility.

    Returns:
        DataFrame: Synthetic client data with 3 columns.

    Examples:
        >>> spark = get_spark()
        >>> clientes = create_clientes(spark, n_clientes=100)
        >>> clientes.count()
        100
        >>> set(clientes.select('tipo').distinct().toPandas()['tipo'])
        {'premium', 'standard'}
    """
    # TODO 4 — Follow these steps:
    #
    # 1. Set the seed: random.seed(seed)
    #
    # 2. Build a list of n_clientes tuples.  Each tuple has 3 fields:
    #      - cliente_id: 1-based sequential (starts at 1, NOT 0)
    #      - nombre:     f"Cliente_{id}"
    #      - tipo:       random.choice(TIPOS)
    #
    # 3. Return spark.createDataFrame(data, [column_names...])
    raise NotImplementedError("TODO 4: generate the synthetic clientes DataFrame")


def analytics_pipeline(
    ventas: DataFrame,
    clientes: DataFrame,
    pais: str = "ES",
    tipo: str = "premium",
) -> DataFrame:
    """
    Runs a multi-stage analytics pipeline on sales data.

    Pipeline (in order):
    1. Join ventas with clientes on "cliente_id" (inner join).
       → This is a wide transformation that causes a shuffle.
    2. Filter rows where pais == pais argument.
    3. Filter rows where tipo == tipo argument.
    4. groupBy("producto").
    5. agg:
       - count("*").alias("num_ventas")
       - round(sum("importe"), 2).alias("total")
       - round(avg("importe"), 2).alias("media")
    6. orderBy("total" desc).

    Run .explain() on the result to see shuffles in the physical plan.
    Open the Spark UI (localhost:4040) after calling .show() to see the stages.

    Args:
        ventas (DataFrame): Sales DataFrame (from create_ventas).
        clientes (DataFrame): Clients DataFrame (from create_clientes).
        pais (str): Country code to filter on.
        tipo (str): Client type to filter on ("premium" or "standard").

    Returns:
        DataFrame: Aggregated result with columns
                   (producto, num_ventas, total, media), ordered by total desc.
    """
    # TODO 5 — Build the pipeline by chaining these operations:
    #
    # 1. Join ventas with clientes on "cliente_id" (inner join).
    #    → Wide transformation: causes a shuffle.
    #
    # 2. Filter: keep only rows where pais equals the pais argument.
    #    → Narrow transformation: no shuffle.
    #
    # 3. Filter: keep only rows where tipo equals the tipo argument.
    #
    # 4. groupBy("producto")
    #    → Wide transformation: causes a shuffle.
    #
    # 5. agg: compute three metrics per product group:
    #      - F.count("*").alias("num_ventas")
    #      - F.round(F.sum("importe"), 2).alias("total")
    #      - F.round(F.avg("importe"), 2).alias("media")
    #
    # 6. orderBy total descending.
    #    → Wide transformation: causes a shuffle.
    #
    # Return the resulting DataFrame.
    raise NotImplementedError("TODO 5: implement the analytics pipeline")


# ---------------------------------------------------------------------------
# Exercise 3: Partitions and Parallelism
# ---------------------------------------------------------------------------

def partition_distribution(df: DataFrame) -> list[tuple[int, int]]:
    """
    Returns the number of rows in each partition of a DataFrame.

    Use df.rdd.mapPartitionsWithIndex to iterate partitions and count rows.
    The lambda receives (partition_index, iterator_of_rows) and should emit
    one (partition_index, row_count) tuple per partition.

    Args:
        df (DataFrame): The DataFrame to inspect.

    Returns:
        list[tuple[int, int]]: List of (partition_id, row_count) pairs,
                               sorted by partition_id ascending.

    Examples:
        >>> spark = get_spark()
        >>> df = spark.range(100).repartition(4)
        >>> partition_distribution(df)
        [(0, 25), (1, 25), (2, 25), (3, 25)]   # approximate
    """
    # TODO 6 — Follow these steps:
    #
    # 1. Use df.rdd.mapPartitionsWithIndex(func) where func is a lambda
    #    that receives (partition_index, row_iterator) and returns an
    #    iterable with ONE element: a (index, count) tuple.
    #
    #    IMPORTANT: the lambda must return a LIST containing the tuple,
    #    e.g. [(idx, count)], not a bare tuple (idx, count).
    #    mapPartitionsWithIndex iterates over whatever you return — a bare
    #    tuple would be iterated as two separate integers.
    #
    #    To count rows without materialising them: sum(1 for _ in iterator)
    #
    # 2. .collect() to bring results to the driver.
    #
    # 3. Sort by partition_id ascending, then return.
    raise NotImplementedError("TODO 6: count rows per partition")


def repartition_by_column(df: DataFrame, col: str, n_partitions: int) -> DataFrame:
    """
    Repartitions a DataFrame by a specific column into n_partitions partitions.

    Uses hash partitioning: all rows with the same value of col are guaranteed
    to end up in the same partition. This causes a shuffle (wide transformation).

    After calling this, you can observe the partition distribution with
    partition_distribution() — expect uneven sizes if the column has a skewed
    value distribution (hot spot problem).

    Args:
        df (DataFrame): The DataFrame to repartition.
        col (str): Column to partition by.
        n_partitions (int): Target number of partitions.

    Returns:
        DataFrame: Repartitioned DataFrame with exactly n_partitions partitions.
    """
    raise NotImplementedError("TODO 7: repartition the DataFrame by column")


def measure_groupby_time(df: DataFrame, group_col: str, agg_col: str) -> float:
    """
    Measures wall-clock time to run a groupBy → sum aggregation.

    Uses time.perf_counter() for high-resolution timing. The aggregation
    .collect() is called to force Spark to execute the full plan.

    Args:
        df (DataFrame): The DataFrame to aggregate.
        group_col (str): Column to group by.
        agg_col (str): Column to sum.

    Returns:
        float: Elapsed time in seconds.

    Examples:
        >>> spark = get_spark()
        >>> df = spark.range(10000).withColumn("v", F.rand())
        >>> t = measure_groupby_time(df, "id", "v")
        >>> t > 0
        True
    """
    # TODO 8 — Follow these steps:
    #
    # 1. Record start time with time.perf_counter()
    # 2. Run df.groupBy(group_col).sum(agg_col).collect()
    #    The .collect() is critical — without an action, Spark does nothing.
    # 3. Return elapsed time: time.perf_counter() - start
    raise NotImplementedError("TODO 8: measure groupBy timing")


# ---------------------------------------------------------------------------
# Demo — run this file directly to see all exercises in action
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    spark = get_spark()
    sc = spark.sparkContext
    sc.setLogLevel("WARN")

    print("=" * 60)
    print("Lab 10: PySpark — First Contact")
    print(f"Spark version: {spark.version}")
    print("Spark UI: http://localhost:4040")
    print("=" * 60)
    input("\nSpark UI is live at http://localhost:4040 — open it now, then press Enter to start...")

    # --- Exercise 1.1: WordCount with RDDs ---
    corpus = [
        "the quick brown fox jumps over the lazy dog",
        "the fox and the dog are good friends",
        "big data is about processing large datasets efficiently",
        "consistency availability and partition tolerance cannot all be guaranteed",
        "spark is the evolution of mapreduce for big data processing",
        "distributed systems require understanding replication and partitioning",
    ]

    print("\n--- Exercise 1.1: WordCount with RDDs ---")
    rdd_result = wordcount_rdd(sc, corpus, num_partitions=3)
    print("Top 10 words:")
    for word, count in rdd_result[:10]:
        print(f"  {word}: {count}")
    input("\nExercise 1.1 done. Check the Spark UI (Jobs tab) then press Enter to continue...")

    # --- Exercise 1.2: WordCount with DataFrames ---
    print("\n--- Exercise 1.2: WordCount with DataFrames ---")
    df_result = wordcount_dataframe(spark, corpus)
    df_result.show(10)
    print("\n=== Execution Plan ===")
    df_result.explain()
    input("\nExercise 1.2 done. Check the SQL/DataFrame tab in the Spark UI then press Enter to continue...")

    # --- Exercise 2.1: Synthetic Data ---
    print("\n--- Exercise 2.1: Generating Synthetic Data ---")
    ventas = create_ventas(spark, n_rows=500_000)
    clientes = create_clientes(spark, n_clientes=10_000)
    print(f"Ventas:   {ventas.count():,} rows")
    print(f"Clientes: {clientes.count():,} rows")
    ventas.printSchema()
    input("\nExercise 2.1 done. Press Enter to continue...")

    # --- Exercise 2.2: Analytics Pipeline ---
    print("\n--- Exercise 2.2: Analytics Pipeline ---")
    t0 = time.perf_counter()
    resultado = analytics_pipeline(ventas, clientes)
    resultado.show()
    print(f"Pipeline time: {time.perf_counter() - t0:.2f}s")
    print("\n=== Pipeline Execution Plan ===")
    resultado.explain()
    input("\nExercise 2.2 done. Count the Exchange nodes in the plan above, then check the Stages tab in the Spark UI. Press Enter to continue...")

    # --- Exercise 3.1: Partition Distribution ---
    print("\n--- Exercise 3.1: Partition Distribution ---")
    print(f"Default partitions: {ventas.rdd.getNumPartitions()}")
    ventas_es = repartition_by_column(ventas, "pais", 5)
    print(f"After repartition(5, 'pais'): {ventas_es.rdd.getNumPartitions()}")
    print("\nRow distribution per partition:")
    for part_id, count in partition_distribution(ventas_es):
        bar = "█" * (count // 5000)
        print(f"  Partition {part_id}: {count:>7} rows  {bar}")
    input("\nExercise 3.1 done. Notice the uneven distribution (hot spot). Press Enter to continue...")

    # --- Exercise 3.2: Partition Timing ---
    print("\n--- Exercise 3.2: Partition Timing ---")
    t_sin = measure_groupby_time(ventas, "pais", "importe")
    t_con = measure_groupby_time(ventas_es, "pais", "importe")
    print(f"Without repartition: {t_sin:.3f}s")
    print(f"With repartition:    {t_con:.3f}s")
    input("\nExercise 3.2 done. Compare the two DAGs in the SQL/DataFrame tab. Press Enter to stop Spark...")

    spark.stop()
    print("\nDone. SparkSession stopped.")
