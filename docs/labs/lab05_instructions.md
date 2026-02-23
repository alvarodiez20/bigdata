# Lab 05: Out-of-Core, Streaming & Parallel Processing

Welcome to the fifth Big Data laboratory session! In this lab, you'll learn to process datasets larger than RAM using chunking, implement streaming statistics, and leverage parallelization.

## Additional Resources

- **[Tips & Reference Guide](lab05_guide.md)** - detailed tips, code examples, and cheatsheets.

## What You Will Learn

- **PyArrow Direct**: When to use PyArrow vs Pandas for I/O and why Arrow is faster
- **Projection Pushdown**: Read only the columns you need from Parquet files
- **Out-of-Core Processing**: Handle datasets that don't fit in RAM using chunking
- **Online Statistics**: Compute mean, variance, and std in a single pass (Welford's algorithm)
- **Threading vs Multiprocessing**: When to use each for I/O-bound vs CPU-bound tasks
- **Pipeline Design**: Combine chunking + parallelization for scalable data processing

## Pre-flight Checklist

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

Follow along in the notebook `notebooks/lab05_outofcore_parallel.ipynb`.

### A. Exercise 0: PyArrow Benchmark & Warm-up (15 min)

**TODO 1**: Generate a 5M-row warmup dataset and save as Parquet.

**TODO 2**: Benchmark three read approaches — `pd.read_parquet()` vs `pq.read_table()` vs Arrow→Pandas conversion.

**TODO 3**: Projection pushdown — compare reading all columns vs only 2 columns, compute revenue with `pyarrow.compute`.

**Pre-filled**: `iter_batches()` streaming and schema inspection — study the provided code.

**Questions to answer**:
- Why is `pq.read_table()` faster than `pd.read_parquet()`?
- When would you use projection pushdown?

---

### B. Exercise 1: Out-of-Core Processing with Chunking (25 min)

**TODO 4**: Generate a 20M-row dataset, save as CSV and partitioned Parquet.

**TODO 5**: Calculate average price using `pd.read_csv(chunksize=500_000)`.

**TODO 6**: Filter only "Electronics" sales chunk by chunk, save to Parquet.

**Pre-filled**: Memory monitoring with `psutil` — observe constant memory during chunking.

---

### C. Exercise 2: Online Statistics — Welford's Algorithm (20 min)

**TODO 7**: Implement the `OnlineStats` class with `update()`, `variance()`, and `std()`.

**Pre-filled**: Validation against NumPy and full streaming statistics.

---

### D. Exercise 3: Practical Parallelization (25 min)

**Pre-filled**: `create_partitions()` — splits dataset into 16 Parquet files.

**TODO 8**: Benchmark sequential vs `ThreadPoolExecutor` for reading partitions.

**TODO 9**: Benchmark sequential vs `ProcessPoolExecutor` for heavy computation.

**Pre-filled**: Worker scaling experiment (1, 2, 4, 8 workers).

---

### E. Exercise 4: Complete Pipeline — Out-of-Core + Parallel (20 min)

**Pre-filled**: `process_partition()` — processes a single partition file.

**TODO 10**: Run the full pipeline sequentially vs with `ProcessPoolExecutor`, measure speedup.

---

## What to Submit

Submit **exactly these two files**:

1. **`notebooks/lab05_outofcore_parallel.ipynb`** — Your completed notebook.
2. **`results/lab05_metrics.json`** — Generated metrics file.

**Do NOT submit:** Large generated data files, `__pycache__` directories, generated images.

---

## Key Takeaways

1. **PyArrow is faster than Pandas for reading** — avoid the conversion overhead
2. **Projection pushdown** saves time and memory by reading only needed columns
3. **Chunking keeps memory constant** — process any file size
4. **Welford's algorithm** computes statistics in one pass
5. **Threading for I/O, multiprocessing for CPU** — choose the right tool
6. **Amdahl's Law** — speedup is limited by the sequential portion

---

**Questions?** Check the [Tips & Reference Guide](lab05_guide.md) or ask your instructor.
