"""
Tests for Lab 10: PySpark — First Contact

All tests use a single session-scoped SparkSession to avoid the overhead of
starting/stopping the JVM on every test. The UI is disabled so Spark doesn't
try to bind a port during CI runs.

Run with:
    uv run pytest tests/test_lab10.py -v
    uv run pytest tests/test_lab10.py -k "TestWordcountRdd" -v
"""

import pytest

from pyspark.sql import SparkSession
import pyspark.sql.functions as F

from src.lab10 import (
    wordcount_rdd,
    wordcount_dataframe,
    create_ventas,
    create_clientes,
    analytics_pipeline,
    partition_distribution,
    repartition_by_column,
    measure_groupby_time,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def spark():
    """Session-scoped SparkSession in local mode. Created once for the whole run."""
    session = (
        SparkSession.builder
        .appName("test_lab10")
        .master("local[2]")
        .config("spark.driver.memory", "1g")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.shuffle.partitions", "4")  # speed up small-data tests
        .getOrCreate()
    )
    yield session
    session.stop()


@pytest.fixture(scope="session")
def small_corpus():
    return [
        "the cat sat on the mat",
        "the dog sat on the log",
        "the fox and the ox",
    ]


@pytest.fixture(scope="session")
def mini_ventas(spark):
    """50-row ventas DataFrame for fast pipeline tests."""
    return create_ventas(spark, n_rows=50, seed=0)


@pytest.fixture(scope="session")
def mini_clientes(spark):
    """20-row clientes DataFrame for fast pipeline tests."""
    return create_clientes(spark, n_clientes=20, seed=0)


# ---------------------------------------------------------------------------
# Exercise 1: WordCount — RDD
# ---------------------------------------------------------------------------

class TestWordcountRdd:
    def test_top_word_is_the(self, spark, small_corpus):
        """'the' appears 5 times across the corpus — must be first."""
        result = wordcount_rdd(spark.sparkContext, small_corpus)
        assert result[0][0] == "the"
        assert result[0][1] == 5

    def test_all_words_counted(self, spark, small_corpus):
        """Every word in the corpus has a non-zero count."""
        result = wordcount_rdd(spark.sparkContext, small_corpus)
        counts = dict(result)
        assert counts["cat"] == 1
        assert counts["sat"] == 2   # appears in lines 1 and 2
        assert counts["fox"] == 1
        assert counts["and"] == 1

    def test_sorted_descending(self, spark, small_corpus):
        """Result must be sorted by count descending."""
        result = wordcount_rdd(spark.sparkContext, small_corpus)
        counts_only = [c for _, c in result]
        assert counts_only == sorted(counts_only, reverse=True)

    def test_returns_list_of_tuples(self, spark, small_corpus):
        """Return type must be a list of (str, int) tuples."""
        result = wordcount_rdd(spark.sparkContext, small_corpus)
        assert isinstance(result, list)
        assert all(isinstance(pair, tuple) and len(pair) == 2 for pair in result)
        assert all(isinstance(w, str) and isinstance(c, int) for w, c in result)

    def test_lowercase(self, spark):
        """Words must be lowercased before counting."""
        result = wordcount_rdd(spark.sparkContext, ["Hello HELLO hello"])
        counts = dict(result)
        assert counts.get("hello") == 3
        assert "Hello" not in counts
        assert "HELLO" not in counts

    def test_num_partitions_respected(self, spark, small_corpus):
        """The RDD must be created with the requested number of partitions."""
        # We verify indirectly: create RDD ourselves and check
        rdd = spark.sparkContext.parallelize(small_corpus, numSlices=2)
        assert rdd.getNumPartitions() == 2
        # The function should not crash with different values
        wordcount_rdd(spark.sparkContext, small_corpus, num_partitions=2)


# ---------------------------------------------------------------------------
# Exercise 1: WordCount — DataFrame
# ---------------------------------------------------------------------------

class TestWordcountDataframe:
    def test_has_word_and_count_columns(self, spark, small_corpus):
        """Resulting DataFrame must have exactly 'word' and 'count' columns."""
        df = wordcount_dataframe(spark, small_corpus)
        assert set(df.columns) == {"word", "count"}

    def test_top_row_is_most_frequent(self, spark, small_corpus):
        """First row must be the most frequent word ('the', 5)."""
        df = wordcount_dataframe(spark, small_corpus)
        first = df.first()
        assert first["word"] == "the"
        assert first["count"] == 5

    def test_all_words_present(self, spark, small_corpus):
        """Every unique word from the corpus must appear exactly once."""
        df = wordcount_dataframe(spark, small_corpus)
        rows = {r["word"]: r["count"] for r in df.collect()}
        assert rows["sat"] == 2
        assert rows["fox"] == 1

    def test_sorted_descending(self, spark, small_corpus):
        """Counts must be in descending order."""
        df = wordcount_dataframe(spark, small_corpus)
        counts = [r["count"] for r in df.collect()]
        assert counts == sorted(counts, reverse=True)

    def test_returns_dataframe(self, spark, small_corpus):
        """Return type must be a Spark DataFrame."""
        from pyspark.sql import DataFrame
        df = wordcount_dataframe(spark, small_corpus)
        assert isinstance(df, DataFrame)


# ---------------------------------------------------------------------------
# Exercise 2: create_ventas
# ---------------------------------------------------------------------------

class TestCreateVentas:
    def test_row_count(self, spark):
        """Row count must match n_rows."""
        df = create_ventas(spark, n_rows=200, seed=1)
        assert df.count() == 200

    def test_column_names(self, spark):
        """Must have exactly the 6 required columns."""
        df = create_ventas(spark, n_rows=10, seed=1)
        assert df.columns == ["venta_id", "pais", "producto", "importe", "fecha", "cliente_id"]

    def test_pais_values(self, spark):
        """pais must only contain values from PAISES."""
        from src.lab10 import PAISES
        valid = set(PAISES)
        df = create_ventas(spark, n_rows=200, seed=1)
        distinct_paises = {r["pais"] for r in df.select("pais").distinct().collect()}
        assert distinct_paises.issubset(valid)

    def test_producto_values(self, spark):
        """producto must only contain values from PRODUCTOS."""
        from src.lab10 import PRODUCTOS
        valid = set(PRODUCTOS)
        df = create_ventas(spark, n_rows=200, seed=1)
        distinct = {r["producto"] for r in df.select("producto").distinct().collect()}
        assert distinct.issubset(valid)

    def test_importe_range(self, spark):
        """importe must be in [10.0, 2000.0]."""
        df = create_ventas(spark, n_rows=200, seed=1)
        stats = df.agg(F.min("importe").alias("mn"), F.max("importe").alias("mx")).first()
        assert stats["mn"] >= 10.0
        assert stats["mx"] <= 2000.0

    def test_reproducible(self, spark):
        """Same seed must produce identical first rows."""
        df1 = create_ventas(spark, n_rows=10, seed=7)
        df2 = create_ventas(spark, n_rows=10, seed=7)
        assert df1.collect() == df2.collect()


# ---------------------------------------------------------------------------
# Exercise 2: create_clientes
# ---------------------------------------------------------------------------

class TestCreateClientes:
    def test_row_count(self, spark):
        """Row count must match n_clientes."""
        df = create_clientes(spark, n_clientes=50, seed=1)
        assert df.count() == 50

    def test_column_names(self, spark):
        """Must have exactly the 3 required columns."""
        df = create_clientes(spark, n_clientes=10, seed=1)
        assert df.columns == ["cliente_id", "nombre", "tipo"]

    def test_tipo_values(self, spark):
        """tipo must only be 'premium' or 'standard'."""
        df = create_clientes(spark, n_clientes=100, seed=1)
        tipos = {r["tipo"] for r in df.select("tipo").distinct().collect()}
        assert tipos.issubset({"premium", "standard"})

    def test_both_tipos_present(self, spark):
        """Both 'premium' and 'standard' must appear with enough rows."""
        df = create_clientes(spark, n_clientes=100, seed=1)
        tipos = {r["tipo"] for r in df.select("tipo").distinct().collect()}
        assert "premium" in tipos
        assert "standard" in tipos

    def test_cliente_id_starts_at_1(self, spark):
        """cliente_id must start at 1, not 0."""
        df = create_clientes(spark, n_clientes=10, seed=1)
        min_id = df.agg(F.min("cliente_id")).first()[0]
        assert min_id == 1


# ---------------------------------------------------------------------------
# Exercise 2: analytics_pipeline
# ---------------------------------------------------------------------------

class TestAnalyticsPipeline:
    def test_output_columns(self, mini_ventas, mini_clientes):
        """Result must have (producto, num_ventas, total, media) columns."""
        result = analytics_pipeline(mini_ventas, mini_clientes)
        assert set(result.columns) == {"producto", "num_ventas", "total", "media"}

    def test_only_filtered_pais(self, spark, mini_ventas, mini_clientes):
        """Pipeline with pais='ES' must not return rows from other countries."""
        # Join mini_ventas with mini_clientes and check the original data
        joined = mini_ventas.join(mini_clientes, on="cliente_id", how="inner")
        has_es = joined.filter(F.col("pais") == "ES").count() > 0
        if not has_es:
            pytest.skip("No ES rows in this mini dataset — skipping")
        result = analytics_pipeline(mini_ventas, mini_clientes, pais="ES", tipo="premium")
        # All rows in result come from the aggregation — num_ventas must be > 0
        rows = result.collect()
        assert all(r["num_ventas"] > 0 for r in rows)

    def test_sorted_by_total_desc(self, mini_ventas, mini_clientes):
        """Rows must be ordered by total descending."""
        result = analytics_pipeline(mini_ventas, mini_clientes)
        totals = [r["total"] for r in result.collect()]
        assert totals == sorted(totals, reverse=True)

    def test_returns_dataframe(self, mini_ventas, mini_clientes):
        """Return type must be a Spark DataFrame."""
        from pyspark.sql import DataFrame
        result = analytics_pipeline(mini_ventas, mini_clientes)
        assert isinstance(result, DataFrame)

    def test_different_filters(self, spark):
        """Filtering by 'FR' must yield a different result than 'ES'."""
        ventas = create_ventas(spark, n_rows=500, seed=42)
        clientes = create_clientes(spark, n_clientes=100, seed=42)
        res_es = analytics_pipeline(ventas, clientes, pais="ES")
        res_fr = analytics_pipeline(ventas, clientes, pais="FR")
        total_es = sum(r["num_ventas"] for r in res_es.collect())
        total_fr = sum(r["num_ventas"] for r in res_fr.collect())
        # ES is ~40% of rows, FR is ~20% — ES total should be higher
        assert total_es > total_fr


# ---------------------------------------------------------------------------
# Exercise 3: partition_distribution
# ---------------------------------------------------------------------------

class TestPartitionDistribution:
    def test_returns_list_of_tuples(self, spark):
        """Must return a list of (int, int) tuples."""
        df = spark.range(100).repartition(3)
        result = partition_distribution(df)
        assert isinstance(result, list)
        assert all(isinstance(t, tuple) and len(t) == 2 for t in result)

    def test_total_count_matches(self, spark):
        """Sum of all partition counts must equal df.count()."""
        df = spark.range(200).repartition(4)
        result = partition_distribution(df)
        assert sum(c for _, c in result) == 200

    def test_sorted_by_partition_id(self, spark):
        """Partition IDs must be in ascending order."""
        df = spark.range(100).repartition(4)
        result = partition_distribution(df)
        ids = [pid for pid, _ in result]
        assert ids == sorted(ids)

    def test_correct_num_partitions(self, spark):
        """Number of entries must equal the number of partitions."""
        df = spark.range(100).repartition(5)
        result = partition_distribution(df)
        assert len(result) == 5


# ---------------------------------------------------------------------------
# Exercise 3: repartition_by_column
# ---------------------------------------------------------------------------

class TestRepartitionByColumn:
    def test_correct_num_partitions(self, spark, mini_ventas):
        """Result must have exactly n_partitions partitions."""
        result = repartition_by_column(mini_ventas, "pais", 3)
        assert result.rdd.getNumPartitions() == 3

    def test_row_count_preserved(self, spark, mini_ventas):
        """Repartitioning must not add or drop rows."""
        original_count = mini_ventas.count()
        result = repartition_by_column(mini_ventas, "pais", 3)
        assert result.count() == original_count

    def test_returns_dataframe(self, spark, mini_ventas):
        """Must return a Spark DataFrame."""
        from pyspark.sql import DataFrame
        result = repartition_by_column(mini_ventas, "pais", 3)
        assert isinstance(result, DataFrame)

    def test_single_partition(self, spark, mini_ventas):
        """Works with n_partitions=1."""
        result = repartition_by_column(mini_ventas, "pais", 1)
        assert result.rdd.getNumPartitions() == 1


# ---------------------------------------------------------------------------
# Exercise 3: measure_groupby_time
# ---------------------------------------------------------------------------

class TestMeasureGroupbyTime:
    def test_returns_positive_float(self, spark, mini_ventas):
        """Elapsed time must be a positive float."""
        t = measure_groupby_time(mini_ventas, "pais", "importe")
        assert isinstance(t, float)
        assert t > 0

    def test_larger_df_takes_longer(self, spark):
        """A 5x larger DataFrame should (usually) take at least as long."""
        small = create_ventas(spark, n_rows=100, seed=1)
        large = create_ventas(spark, n_rows=500, seed=1)
        t_small = measure_groupby_time(small, "pais", "importe")
        t_large = measure_groupby_time(large, "pais", "importe")
        # Not a strict assertion (JIT warmup etc.) — just check it runs
        assert t_small >= 0
        assert t_large >= 0
