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
    raise NotImplementedError(
        "TODO 1: parallelize corpus (numSlices=num_partitions) → "
        "flatMap(lambda line: [(w.lower(), 1) for w in line.split()]) → "
        "reduceByKey(lambda a, b: a + b) → collect() → sort by count desc"
    )


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
    raise NotImplementedError(
        "TODO 2: createDataFrame([(line,) for line in corpus], ['text']) → "
        "select(F.explode(F.split(F.lower(F.col('text')), ' ')).alias('word')) → "
        "groupBy('word').count() → orderBy(F.desc('count'))"
    )


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
        fecha     (str)    — date string "YYYY-MM-DD", year 2025
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
    raise NotImplementedError(
        "TODO 3: random.seed(seed), build list of tuples "
        "(i, random.choice(PAISES), random.choice(PRODUCTOS), "
        "round(random.uniform(10, 2000), 2), "
        "f'2025-{random.randint(1,12):02d}-{random.randint(1,28):02d}', "
        "random.randint(1, 10000)) for i in range(n_rows), "
        "then spark.createDataFrame(data, ['venta_id','pais','producto','importe','fecha','cliente_id'])"
    )


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
    raise NotImplementedError(
        "TODO 4: random.seed(seed), build list of tuples "
        "(i, f'Cliente_{i}', random.choice(TIPOS)) for i in range(1, n_clientes+1), "
        "then spark.createDataFrame(data, ['cliente_id','nombre','tipo'])"
    )


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
    raise NotImplementedError(
        "TODO 5: ventas.join(clientes, on='cliente_id', how='inner') → "
        ".filter(F.col('pais') == pais) → "
        ".filter(F.col('tipo') == tipo) → "
        ".groupBy('producto') → "
        ".agg(F.count('*').alias('num_ventas'), "
        "     F.round(F.sum('importe'), 2).alias('total'), "
        "     F.round(F.avg('importe'), 2).alias('media')) → "
        ".orderBy(F.desc('total'))"
    )


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
    raise NotImplementedError(
        "TODO 6: df.rdd.mapPartitionsWithIndex("
        "    lambda idx, it: [(idx, sum(1 for _ in it))]) "
        ".collect(), then sort by partition_id and return"
    )


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
    raise NotImplementedError(
        "TODO 7: return df.repartition(n_partitions, col)"
    )


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
    raise NotImplementedError(
        "TODO 8: t0 = time.perf_counter() → "
        "df.groupBy(group_col).sum(agg_col).collect() → "
        "return time.perf_counter() - t0"
    )


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

    # --- Exercise 1: WordCount ---
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

    print("\n--- Exercise 1.2: WordCount with DataFrames ---")
    df_result = wordcount_dataframe(spark, corpus)
    df_result.show(10)
    print("\n=== Execution Plan ===")
    df_result.explain()

    # --- Exercise 2: Sales Pipeline ---
    print("\n--- Exercise 2: Sales Analytics Pipeline ---")
    ventas = create_ventas(spark, n_rows=500_000)
    clientes = create_clientes(spark, n_clientes=10_000)
    print(f"Ventas:   {ventas.count():,} rows")
    print(f"Clientes: {clientes.count():,} rows")
    ventas.printSchema()

    t0 = time.perf_counter()
    resultado = analytics_pipeline(ventas, clientes)
    resultado.show()
    print(f"Pipeline time: {time.perf_counter() - t0:.2f}s")
    print("\n=== Pipeline Execution Plan ===")
    resultado.explain()

    # --- Exercise 3: Partitions ---
    print("\n--- Exercise 3.1: Partition Distribution ---")
    print(f"Default partitions: {ventas.rdd.getNumPartitions()}")
    ventas_es = repartition_by_column(ventas, "pais", 5)
    print(f"After repartition(5, 'pais'): {ventas_es.rdd.getNumPartitions()}")
    print("\nRow distribution per partition:")
    for part_id, count in partition_distribution(ventas_es):
        bar = "█" * (count // 5000)
        print(f"  Partition {part_id}: {count:>7} rows  {bar}")

    print("\n--- Exercise 3.2: Partition Timing ---")
    t_sin = measure_groupby_time(ventas, "pais", "importe")
    t_con = measure_groupby_time(ventas_es, "pais", "importe")
    print(f"Without repartition: {t_sin:.3f}s")
    print(f"With repartition:    {t_con:.3f}s")

    spark.stop()
    print("\nDone. SparkSession stopped.")
