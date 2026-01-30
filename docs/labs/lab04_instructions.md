# Lab 04: Vectorization and Out-of-Core Computing

Welcome to the fourth Big Data laboratory session! In this lab, you'll learn to process data efficiently using vectorization and handle datasets larger than your RAM using out-of-core computing techniques.

## 📚 Additional Resources

- **[Tips & Reference Guide](lab04_guide.md)** - detailed tips, code examples, and cheatsheets.

## 🎯 What You Will Learn

- **Vectorization**: Replace slow Python loops with fast NumPy/Pandas operations (100-200x speedup)
- **Broadcasting**: Apply operations across arrays of different shapes without explicit loops
- **Chunking**: Process datasets larger than RAM by loading data in manageable pieces
- **Online Algorithms**: Calculate statistics in a single pass with O(1) memory
- **Dask Introduction**: Scale Pandas operations to larger-than-memory datasets

## ✅ Pre-flight Checklist

Before starting, ensure you have:

1. **Completed Lab 03**: You understand data types and storage formats.
2. **Updated your repo**: Run `git pull` to get the latest changes.
3. **Installed dependencies**: Run `uv sync`.
4. **Install additional tools**:
   ```bash
   pip install dask[complete]
   ```
5. **Verify installation**:
   ```bash
   uv run python -c "import dask; print(f'Dask {dask.__version__}')"
   ```

---

## 📝 Lab Steps

Follow along in the notebook `notebooks/lab04_vectorization_out_of_core.ipynb`.

### A. Generate Test Datasets

We'll create two datasets:
- **Medium dataset** (10M rows) for vectorization benchmarks
- **Large dataset** (50M+ rows) for out-of-core exercises

---

### B. Exercise 1: Loop to Vectorized Conversion (25 min)

You will rewrite slow loop-based code using vectorized operations.

**Part 1A: Distance Calculation**

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

**Part 1B: Age Classification**

Replace conditional loop with `np.select()` or `pd.cut()`.

**Part 1C: Column Normalization**

Replace nested loop with broadcasting: `(data - mean) / std`.

**Part 1D: Score Calculation with Clipping**

Replace loop with vectorized operations and `np.clip()`.

**Goal**: Achieve >50x speedup on each function.

---

### C. Exercise 2: Vectorization Benchmarks (20 min)

You will quantify the performance impact of vectorization.

**Tasks**:

1. Benchmark simple operations (sum, element-wise multiply)
2. Benchmark complex operations (filter + transform)
3. Benchmark `.apply()` vs vectorized alternatives
4. Create visualization of results

**Goal**: Document speedups for different operation types.

---

### D. Exercise 3: Out-of-Core Processing (30 min)

You will process a dataset larger than RAM using chunking.

**Part 3A: Demonstrate the Problem**

Try loading a large CSV and observe MemoryError.

**Part 3B: Chunked Statistics**

Calculate mean and count using chunking:

```python
total_sum = 0
total_count = 0

for chunk in pd.read_csv('large.csv', chunksize=500_000):
    total_sum += chunk['value'].sum()
    total_count += len(chunk)

mean = total_sum / total_count
```

**Part 3C: Chunked Filtering**

Filter and save a subset without loading the full dataset.

**Part 3D: Chunked Aggregation**

Perform groupby aggregation across chunks.

**Part 3E: Memory Monitoring**

Track memory usage during chunk processing.

**Goal**: Process 20GB+ of data with <1GB memory.

---

### E. Exercise 4: Online Statistics (15 min)

You will implement streaming algorithms for statistics calculation.

**Part 4A: Welford's Algorithm**

Implement `OnlineStats` class with:
- Running mean
- Running variance (Welford's method)
- Min/max tracking

**Part 4B: Validation**

Verify your implementation against NumPy.

**Part 4C: Apply to Large Dataset**

Calculate statistics on a large dataset using chunking + OnlineStats.

**Goal**: Calculate exact statistics with O(1) memory.

---

### F. Exercise 5 (Bonus): Introduction to Dask (20 min)

You will compare Dask with manual chunking.

**Part 5A: Basic Operations**

```python
import dask.dataframe as dd

ddf = dd.read_csv('large.csv')
result = ddf.groupby('category')['price'].mean().compute()
```

**Part 5B: Benchmark Comparison**

Compare Dask vs manual chunking for aggregations.

**Part 5C: Task Graph Visualization**

Visualize Dask's lazy execution plan.

**Goal**: Understand when Dask simplifies out-of-core processing.

---

## 📦 What to Submit

Submit **exactly these two files**:

1. **`notebooks/lab04_vectorization_out_of_core.ipynb`** — Your completed notebook.
2. **`results/lab04_metrics.json`** — Generated metrics file.

**Do NOT submit:**
- Large generated data files
- `__pycache__` directories

---

## 🚀 Key Takeaways

After completing this lab, remember:

1. **Vectorization is 100-200x faster** than Python loops
2. **Never use `.apply()` with simple lambdas** — always find vectorized alternative
3. **Broadcasting eliminates nested loops** for array operations
4. **Chunking enables processing files larger than RAM**
5. **Online algorithms** provide exact statistics with constant memory
6. **Dask** automates out-of-core + parallelism with Pandas API

---

**Questions?** Check the [Tips & Reference Guide](lab04_guide.md) or ask your instructor.
