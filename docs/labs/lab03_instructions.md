# Lab 03: Data Types and Efficient Formats

Welcome to the third Big Data laboratory session! In this lab, you'll learn how to dramatically reduce memory usage and improve performance through smart data type choices and efficient storage formats.

## 📚 Additional Resources

- **[Tips & Reference Guide](lab03_guide.md)** - detailed tips, code examples, and cheatsheets for every exercise.
- **[Lab 02 Instructions](lab02_instructions.md)** - if you need to review complexity and profiling.

## 🎯 What You Will Learn

- **Data Type Optimization**: How choosing the right dtypes can reduce memory 5-10x
- **Storage Formats**: Understanding row vs column-oriented storage (CSV vs Parquet vs Feather)
- **Compression**: Trade-offs between different compression algorithms
- **Partitioning**: How to structure data for fast queries on large datasets

## ✅ Pre-flight Checklist

Before starting, ensure you have:

1.  **Completed Lab 02**: You understand profiling and complexity.
2.  **Updated your repo**: Run `git pull` to get the latest changes.
3.  **Installed dependencies**: Run `uv sync` to ensure you have `pyarrow`.
4.  **Verify PyArrow**:
    ```bash
    uv run python -c "import pyarrow; print(f'PyArrow {pyarrow.__version__}')"
    ```

---

## 📝 Lab Steps

Follow along in the notebook `notebooks/lab03_data_types_formats.ipynb`.

### A. Generate the Dataset

We'll generate a synthetic e-commerce dataset with 5 million rows containing:
- `order_id`: Unique order identifier
- `product_id`: Product ID (1-50,000)
- `category`: Product category (15 unique values)
- `price`: Product price (0.01 - 999.99)
- `quantity`: Quantity ordered (1-100)
- `country`: Customer country (30 unique values)
- `timestamp`: Order timestamp

**Goal**: Create `data/raw/ecommerce_5m.csv` (~500MB).

---

### B. Exercise 1: Data Type Optimization (25 min)

You will measure the impact of choosing optimal data types.

**Part 1A: Baseline Measurement**

1.  Read the CSV with default dtypes.
2.  Measure total memory usage with `df.memory_usage(deep=True)`.
3.  Analyze each column's dtype and memory.

**Part 1B: Type Analysis**

For each column, determine the optimal type:
- `product_id`: Range 1-50,000 → Which int type?
- `category`: 15 unique values → `object` or `category`?
- `price`: 0.01-999.99 → `float32` or `float64`?
- `quantity`: 1-100 → Which int type?
- `country`: 30 unique values → `object` or `category`?

**Part 1C: Optimized Loading**

1.  Re-read the CSV with optimal dtypes specified.
2.  Measure the memory reduction.
3.  Implement `optimize_dtypes()` function for automatic optimization.

**Part 1D: Speed Impact**

1.  Benchmark groupby operations on baseline vs optimized.
2.  Benchmark filter operations.
3.  Document the speedup.

**Goal**: Achieve >5x memory reduction.

---

### C. Exercise 2: Format Comparison (30 min)

You will compare different storage formats and compression algorithms.

**Part 2A: Convert to Multiple Formats**

Convert your optimized DataFrame to:
- CSV (uncompressed)
- CSV.gz (gzip compressed)
- Parquet with Snappy compression
- Parquet with Gzip compression
- Parquet with Zstd compression
- Parquet without compression
- Feather format

**Part 2B: Benchmark Writing**

For each format, measure:
- Write time
- File size on disk

**Part 2C: Benchmark Reading**

For each format, measure:
- Full read time
- Partial read time (3 columns only) - where supported

**Part 2D: Create Comparison Table**

| Format | Size (MB) | Write (s) | Read Full (s) | Read 3 cols (s) |
|--------|-----------|-----------|---------------|-----------------|
| CSV | ? | ? | ? | N/A |
| ... | ... | ... | ... | ... |

**Goal**: Identify the best format for different use cases.

---

### D. Exercise 3: Parquet Deep Dive (20 min)

You will explore Parquet's internal structure and configuration options.

**Part 3A: Inspect Metadata**

1.  Use `pyarrow.parquet.ParquetFile` to inspect the file.
2.  List the number of row groups.
3.  Print the schema.
4.  View statistics (min/max) for each column chunk.

**Part 3B: Row Group Size Experiment**

1.  Write Parquet files with different `row_group_size` values (10K, 100K, 1M).
2.  Compare file sizes and read performance.

**Part 3C: Predicate Pushdown**

1.  Read Parquet without filters and measure time.
2.  Read Parquet with filters (e.g., `price > 100`) and measure time.
3.  Explain why filtering is faster with Parquet.

**Goal**: Understand Parquet internals for optimal configuration.

---

### E. Exercise 4: Partitioning Strategies (35 min)

You will implement and benchmark different partitioning approaches.

**Part 4A: Add Partition Columns**

Extract date components from timestamp:
- `year`
- `month`
- `day`

**Part 4B: Implement Partitioning Strategies**

1.  **No partitioning**: Single Parquet file
2.  **By year/month**: Two-level partitioning
3.  **By year/month/day**: Three-level partitioning
4.  **By category**: Partition by product category

**Part 4C: Benchmark Queries**

Test each strategy with:
- **Query 1**: Orders from a specific day
- **Query 2**: All orders in a specific category for one month
- **Query 3**: Full dataset aggregation

**Part 4D: Analysis**

- Count files generated by each strategy
- Measure disk usage
- Document query performance

**Goal**: Understand when and how to use partitioning effectively.

---

## 📦 What to Submit

Submit **exactly these two files**:

1.  **`notebooks/lab03_data_types_formats.ipynb`** — Your completed notebook.
2.  **`results/lab03_metrics.json`** — The JSON file generated by the notebook.

**Do NOT submit:**
-   The generated data files (CSV, Parquet, etc.)
-   The `__pycache__` directories

---

## 🚀 Next Steps

After completing this lab:

1.  Check your `results/lab03_metrics.json`.
2.  Write your reflection in the notebook.
3.  Submit your work!

**Questions?** Check the [Tips & Reference Guide](lab03_guide.md) or ask your instructor.
