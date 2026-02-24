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

### A. Exercise 0: PyArrow Benchmark & Warm-up

!!! objective
    Understand *why* PyArrow is faster than Pandas for reading Parquet files. Internally, `pd.read_parquet()` uses PyArrow under the hood but adds an extra conversion step (Arrow Table → Pandas DataFrame). By benchmarking all three approaches, you will quantify that overhead and learn when it's worth skipping the conversion entirely.

**TODO 1** — `generate_warmup_data()`: Generate a 5M-row warmup dataset and save as Parquet. This gives you a realistic-sized file to benchmark against — small enough to fit in RAM, but large enough that performance differences become measurable.

**TODO 2** — `benchmark_read_methods()`: Benchmark three read approaches — `pd.read_parquet()` vs `pq.read_table()` vs Arrow→Pandas conversion. You are isolating each stage of the pipeline so you can see exactly where time is spent: in the I/O itself, or in the Arrow-to-Pandas conversion.

**TODO 3** — `benchmark_projection_pushdown()`: Projection pushdown — compare reading all columns vs only 2 columns, then compute revenue with `pyarrow.compute`. This demonstrates a key Parquet optimization: since Parquet is columnar, you can skip reading columns you don't need, saving both time and memory.

**Pre-filled**: `iter_batches()` streaming and schema inspection — study the provided code to see how PyArrow can process a Parquet file in small batches without loading it entirely.

**Questions to answer**:

- Why is `pq.read_table()` faster than `pd.read_parquet()`?
- When would you use projection pushdown?

---

### B. Exercise 1: Out-of-Core Processing with Chunking

!!! objective
    Learn to process files that are too large for your available RAM. Instead of loading everything at once, you read and process the file in fixed-size chunks. This "out-of-core" approach keeps memory usage constant regardless of file size — a fundamental technique in Big Data.

**TODO 4** — `generate_large_dataset()`: Generate a 20M-row dataset, save as CSV and partitioned Parquet. This creates the large input file (~1.2 GB CSV) that you'll practice chunking on. The partitioned Parquet version serves as a comparison for later exercises.

**TODO 5** — `chunked_statistics()`: Calculate average price using `pd.read_csv(chunksize=500_000)`. Instead of loading all 20M rows, you accumulate a running sum and count across chunks, then compute the average at the end. This is the "reduce" pattern — the most common chunking strategy.

**TODO 6** — `chunked_filter_save()`: Filter only "Electronics" sales chunk by chunk, then save the result to Parquet. This is the "filter and collect" pattern — you apply a condition to each chunk independently and gather the matching rows. It demonstrates that you can process and transform data without ever holding the full dataset in memory.

**Pre-filled**: Memory monitoring with `psutil` — observe how memory stays constant during chunking, proving that only one chunk lives in RAM at a time.

---

### C. Exercise 2: Online Statistics — Welford's Algorithm

!!! objective
    Learn an algorithm that computes mean, variance, and standard deviation in a **single pass** using **O(1) memory**. The naive approach requires storing all values (to compute the mean first, then iterate again for variance). Welford's algorithm updates all statistics incrementally as each new value arrives — making it ideal for streaming or chunked data.

**TODO 7** — `OnlineStats` class: Implement the `update()`, `variance()`, and `std()` methods using Welford's algorithm. The key insight is that each new value updates the running mean and a running sum of squared differences *simultaneously*, using the difference between the old and new mean. After implementation, your results will be validated against NumPy to confirm correctness.

**Pre-filled**: Validation against NumPy and full streaming statistics — verifies that your implementation matches NumPy's results within floating-point precision.

---

### D. Exercise 3: Practical Parallelization

!!! objective
    Understand **when and why** to use threading vs multiprocessing. Python's GIL (Global Interpreter Lock) prevents true parallelism for CPU-bound code in threads, but the GIL is released during I/O operations (file reads, network calls). This means: use `ThreadPoolExecutor` for I/O-bound tasks (reading files in parallel) and `ProcessPoolExecutor` for CPU-bound tasks (each process has its own GIL).

**Pre-filled**: `create_partitions()` — splits the dataset into 16 partition files (both Parquet and CSV formats), giving you multiple independent files that can be processed in parallel.

**TODO 8** — `benchmark_threading()`: Benchmark sequential vs `ThreadPoolExecutor` for reading CSV partitions. CSV reading is truly I/O-bound (unlike Parquet, which involves CPU-heavy decompression), so threads can overlap the disk reads — you should see a meaningful speedup.

**TODO 9** — `benchmark_multiprocessing()`: Benchmark sequential vs `ProcessPoolExecutor` for heavy computation. The `heavy_process()` function performs CPU-intensive operations (sqrt, log, groupby). Since each process runs in its own interpreter with its own GIL, true CPU parallelism is achieved.

**Pre-filled**: Worker scaling experiment (1, 2, 4, 8 workers) — illustrates Amdahl's Law: speedup is **sub-linear** because there is always a sequential fraction (process startup, result gathering) that limits the maximum possible speedup.

---

### E. Exercise 4: Complete Pipeline — Out-of-Core + Parallel

!!! objective
    Combine everything from the previous exercises into a real-world processing pipeline. This is the pattern used in production Big Data systems: partition the data, process each partition independently (enabling parallelism), then merge the partial results. You'll measure the end-to-end speedup to see how chunking + parallelization work together.

**Pre-filled**: `process_partition()` — processes a single partition file (reads, transforms, aggregates). This is the "map" step in a MapReduce-style pipeline.

**TODO 10** — `run_parallel_pipeline()`: Run the full pipeline sequentially vs with `ProcessPoolExecutor`, then merge partial results and measure speedup. The merging step (`pd.concat().groupby().sum()`) combines aggregations from all partitions into a single final result — this is the "reduce" step. Compare the speedup against the theoretical predictions from Amdahl's Law.

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
