# Lab 04: Efficient Formats and Vectorization

Welcome to the fourth Big Data laboratory session! In this lab, you'll compare storage formats (CSV, Parquet, Feather) and master vectorization to build fast data pipelines.

## Additional Resources

- **[Tips & Reference Guide](lab04_guide.md)** - detailed tips, code examples, and cheatsheets.

## What You Will Learn

- **Format Comparison**: Measure size and speed trade-offs between CSV, Parquet (Snappy, Zstd), and Feather
- **Column Pruning & Predicate Pushdown**: Read only what you need from Parquet files
- **Vectorization**: Replace slow Python loops with fast NumPy/Pandas operations (100-200x speedup)
- **Broadcasting**: Apply operations across arrays without explicit loops
- **Pipeline Optimization**: Combine efficient formats + vectorization for maximum performance

## Pre-flight Checklist

Before starting, ensure you have:

1. **Completed Lab 03**: You understand data types and storage formats.
2. **Updated your repo**: Run `git pull` to get the latest changes.
3.  **Checkout main**: Run `git checkout main`.
4.  **Create a local branch**: Run `git checkout -b <your_branch_name>`
5. **Installed dependencies**: Run `uv sync`.
   ```bash
   uv run python -c "import pyarrow; print(f'PyArrow {pyarrow.__version__}')"
   ```

---

## Lab Steps

Follow along in the notebook `notebooks/lab04_formats_vectorization.ipynb`.

### A. Generate Sales Dataset

Create a realistic sales dataset with 5 million rows:

- `id`, `fecha`, `categoria`, `producto`, `precio`, `cantidad`, `ciudad`

---

### B. Exercise 1: CSV vs Parquet vs Feather (25 min)

You will save the dataset in multiple formats and benchmark them.

**Part 1A: Save in All Formats**

Save the DataFrame as CSV, Parquet (Snappy, Zstd, None compression), and Feather. Measure write time and file size.

**Part 1B: Read Benchmarks**

Benchmark three scenarios:

1. **Full read**: Read the entire file
2. **Selective read**: Read only 2 columns (`precio`, `cantidad`)
3. **Filtered read**: Read only rows where `categoria == 'Electronica'` (Parquet predicate pushdown)

**Part 1C: Summary Table**

Create a summary table with size, write time, and read time for each format.

**Goal**: Understand when to use each format and the impact of compression.

---

### C. Exercise 2: Rewrite Loops to Vectorized (25 min)

You will rewrite slow loop-based code using vectorized operations.

**Part 2A: Distance Calculation**

Convert a loop-based Euclidean distance calculation to NumPy broadcasting.

```python
# Original (slow)
def calculate_distances_slow(points_a, points_b):
    distances = []
    for i in range(len(points_a)):
        dist = math.sqrt(
            (points_a[i][0] - points_b[i][0])**2 +
            (points_a[i][1] - points_b[i][1])**2
        )
        distances.append(dist)
    return distances
```

**Part 2B: Age Classification**

Replace conditional loop with `np.select()` or `pd.cut()`.

**Part 2C: Column Normalization**

Replace nested loop with broadcasting: `(data - mean) / std`.

**Part 2D: Score Calculation with Clipping**

Replace loop with vectorized operations and `np.clip()`.

**Goal**: Achieve >50x speedup on each function.

---

### D. Exercise 3: Comprehensive Benchmark (20 min)

You will quantify the performance impact of vectorization across 5 scenarios.

**Benchmarks**:

1. Sum: loop vs `.sum()`
2. Element-wise multiply: loop vs operator `*`
3. Filter + transform: loop vs `.loc[]`
4. `.apply()` with lambda vs vectorized
5. `.apply()` with complex function vs NumPy equivalent

**Goal**: Document speedups for different operation types.

---

### E. Exercise 4: Integrated Pipeline (20 min)

You will build and compare two data pipelines:

**Naive pipeline**: Read CSV + process with Python loops

```python
df = pd.read_csv('ventas.csv')
totals = []
for i in range(len(df)):
    if df.iloc[i]['categoria'] == 'Electronica':
        totals.append(df.iloc[i]['precio'] * df.iloc[i]['cantidad'])
```

**Optimized pipeline**: Read Parquet (selective + filtered) + vectorized operations

```python
df = pd.read_parquet('ventas_snappy.parquet',
                     columns=['categoria', 'precio', 'cantidad'],
                     filters=[('categoria', '==', 'Electronica')])
df['total'] = df['precio'] * df['cantidad']
```

**Goal**: Measure total speedup from combining format + vectorization.

---

## What to Submit

Submit **exactly these two files**:

1. **`notebooks/lab04_formats_vectorization.ipynb`** — Your completed notebook.
2. **`results/lab04_metrics.json`** — Generated metrics file.

**Do NOT submit:**
- Large generated data files
- `__pycache__` directories

---

## Key Takeaways

After completing this lab, remember:

1. **Parquet is 3-10x smaller** and faster than CSV for analytical workloads
2. **Compression trade-offs**: Snappy = fast, Zstd = best ratio, None = fastest write
3. **Feather** is best for intermediate data between pipeline steps
4. **Vectorization is 100-200x faster** than Python loops
5. **Never use `.apply()` with simple lambdas** — always find vectorized alternative
6. **Combining optimizations** (format + vectorization) yields massive speedups

---

**Questions?** Check the [Tips & Reference Guide](lab04_guide.md) or ask your instructor.
