# Lab 06: Out-of-Core, Streaming & Parallel Processing

Welcome to the sixth Big Data laboratory session! In this lab, you'll learn to process datasets larger than RAM using chunking, implement streaming statistics, and leverage parallelization.

## Additional Resources

- **[Tips & Reference Guide](lab06_guide.md)** - detailed tips, code examples, and cheatsheets.

## What You Will Learn

- **PyArrow Direct**: When to use PyArrow vs Pandas for I/O and why Arrow is faster
- **Projection Pushdown**: Read only the columns you need from Parquet files
- **Out-of-Core Processing**: Handle datasets that don't fit in RAM using `pd.read_csv(chunksize=)`
- **Online Statistics**: Compute mean, variance, and std in a single pass (Welford's algorithm)
- **Threading vs Multiprocessing**: When to use each for I/O-bound vs CPU-bound tasks
- **Pipeline Design**: Combine chunking + parallelization for scalable data processing

## Pre-flight Checklist

Before starting, ensure you have:

1. **Completed Lab 04**: You understand Parquet, vectorization, and efficient formats.
2. **Updated your repo**: Run `git pull` to get the latest changes.
3. **Checkout main**: Run `git checkout main`.
4. **Create a local branch**: Run `git checkout -b <your_branch_name>`
5. **Installed dependencies**: Run `uv sync`.
   ```bash
   uv run python -c "import pyarrow; import psutil; print(f'PyArrow {pyarrow.__version__}, psutil {psutil.__version__}')"
   ```

---

## Lab Steps

Follow along in the notebook `notebooks/lab06_outofcore_parallel.ipynb`.

### A. Exercise 0: PyArrow Benchmark & Warm-up (15 min)

Familiarize yourself with PyArrow by comparing its performance against Pandas.

**TODO 1: Generate Warmup Data**

Create a 5M-row dataset and save as Parquet.

**TODO 2: Benchmark Read Methods**

Compare three ways to read a Parquet file:

1. `pd.read_parquet()` — Pandas (internally uses Arrow, but with conversion overhead)
2. `pq.read_table()` — Arrow Table directly (no conversion)
3. `table.to_pandas()` — Measure the conversion cost separately

**TODO 3: Projection Pushdown**

Compare reading all columns vs only 2 columns. Calculate revenue using `pyarrow.compute`.

**TODO 4: `iter_batches()`**

Process the Parquet file in streaming fashion using Arrow's batch Iterator.

**TODO 5: Schema Inspection** (optional)

Read file metadata (schema, row groups, statistics) without loading data.

**Goal**: Understand when to use each read method and the cost of Arrow → Pandas conversion.

**Questions to answer**:
- Why is `pq.read_table()` faster than `pd.read_parquet()` if Pandas uses Arrow internally?
- In what real-world cases would you use projection pushdown?

---

### B. Exercise 1: Out-of-Core Processing with Chunking (25 min)

Process a 20M-row dataset without loading it entirely into RAM.

**TODO 6: Generate Large Dataset**

Create a 20M-row dataset, save as CSV and partitioned Parquet.

**TODO 7: Chunked Statistics**

Calculate average price using `pd.read_csv(chunksize=500_000)` — memory stays constant!

**TODO 8: Chunked Filter & Save**

Filter only "Electronics" sales chunk by chunk, then save to Parquet.

**TODO 9: Memory Monitoring**

Monitor RSS memory during chunked processing using `psutil`. Generate a memory usage plot.

**Goal**: Prove that chunking keeps memory constant regardless of dataset size.

---

### C. Exercise 2: Online Statistics — Welford's Algorithm (20 min)

Implement streaming statistics that work on infinite data streams.

**TODO 10: `OnlineStats` Class**

Implement Welford's algorithm for numerically stable online mean, variance, and std:

```python
class OnlineStats:
    def update(self, x):
        """Welford's algorithm: update running mean and M2."""
        pass
    def variance(self):
        """Population variance = M2 / count."""
        pass
    def std(self):
        """Standard deviation = sqrt(variance)."""
        pass
```

**TODO 11: Validate Against NumPy**

Compare OnlineStats results with `np.mean()` and `np.std(ddof=0)`.

**TODO 12: Full Streaming Statistics**

Compute mean and std over the full 20M-row dataset using OnlineStats + chunking.

**Goal**: Validated OnlineStats implementation that matches NumPy results.

---

### D. Exercise 3: Practical Parallelization (25 min)

Compare sequential, threaded, and multiprocess execution.

**TODO 13: Create Partitions**

Split the dataset into 16 Parquet partition files.

**TODO 14: Threading Benchmark**

Compare sequential vs `ThreadPoolExecutor` for **reading** 16 partition files. (I/O-bound → threads help)

**TODO 15: Multiprocessing Benchmark**

Compare sequential vs `ProcessPoolExecutor` for **heavy computation** on each partition. (CPU-bound → processes help)

**TODO 16: Worker Scaling Experiment**

Vary number of workers (1, 2, 4, 8) and plot actual speedup vs ideal linear speedup.

**Goal**: Understand when threading vs multiprocessing provides benefits and Amdahl's Law.

---

### E. Exercise 4: Complete Pipeline — Out-of-Core + Parallel (20 min)

Combine everything into a real-world pipeline.

**TODO 17: `process_partition()`**

Process a single partition: calculate revenue, bin prices, group by category.

**TODO 18: `run_parallel_pipeline()`**

Run the pipeline sequentially vs with `ProcessPoolExecutor`, compare total speedup.

```python
with ProcessPoolExecutor(max_workers=4) as executor:
    partial_results = list(executor.map(process_partition, files))
final = pd.concat(partial_results).groupby(level=[0, 1]).sum()
```

**Goal**: Measure total speedup from combining parallelization + efficient formats.

---

## What to Submit

Submit **exactly these two files**:

1. **`notebooks/lab06_outofcore_parallel.ipynb`** — Your completed notebook.
2. **`results/lab06_metrics.json`** — Generated metrics file.

**Do NOT submit:**
- Large generated data files (CSV, Parquet)
- `__pycache__` directories
- Generated image files

---

## Key Takeaways

After completing this lab, remember:

1. **PyArrow is faster than Pandas for reading** — avoid the conversion overhead when possible
2. **Projection pushdown** saves time and memory by reading only needed columns
3. **Chunking keeps memory constant** — process any file size with `chunksize=`
4. **Welford's algorithm** computes statistics in one pass with numerical stability
5. **Threading for I/O**, **multiprocessing for CPU** — choose the right tool
6. **Amdahl's Law** — speedup is limited by the sequential portion of work
7. **Combine optimizations** for maximum throughput in real pipelines

---

**Questions?** Check the [Tips & Reference Guide](lab06_guide.md) or ask your instructor.
