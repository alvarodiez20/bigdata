# Lab 05: Tips & Quick Reference

This guide provides detailed tips, code examples, and cheatsheets to help you complete [Lab 05](lab05_instructions.md).

---

## General Tips

- **Think streaming**: If you're loading the entire file, ask yourself if chunking would work.
- **Profile memory**: Use `psutil` to verify that your approach keeps memory constant.
- **Choose the right parallelism**: Threads for I/O, processes for CPU.

---

## Section A: PyArrow vs Pandas

### Why PyArrow is Faster

```python
# Pandas: reads Parquet → Arrow Table → converts to Pandas DataFrame
df = pd.read_parquet('data.parquet')  # Two steps internally

# PyArrow: reads Parquet → Arrow Table (stops here)
table = pq.read_table('data.parquet')  # One step, no conversion
```

The conversion from Arrow Table to Pandas involves allocating new memory, converting Arrow arrays to NumPy arrays, and creating the DataFrame object.

### Projection Pushdown

```python
# Read ALL columns (slow, wasteful)
table = pq.read_table('data.parquet')

# Read ONLY needed columns (fast, lean)
table = pq.read_table('data.parquet', columns=['price', 'quantity'])
```

### Arrow Compute Functions

```python
import pyarrow.compute as pc

revenue = pc.multiply(table.column('price'), table.column('quantity'))
total = pc.sum(revenue).as_py()  # .as_py() converts to Python scalar
```

### iter_batches() for Streaming

```python
pf = pq.ParquetFile('data.parquet')
for batch in pf.iter_batches(batch_size=500_000, columns=['price', 'quantity']):
    # batch is a RecordBatch, not a DataFrame!
    revenue = pc.sum(pc.multiply(batch.column('price'), batch.column('quantity')))
```

---

## Section B: Chunking (Out-of-Core Processing)

### CSV Chunking

```python
reader = pd.read_csv('big_file.csv', chunksize=500_000)  # Returns iterator
for chunk in reader:
    # chunk is a regular DataFrame with 500K rows
    # Previous chunk is garbage-collected automatically
```

### Common Chunking Patterns

**Pattern 1: Reduce (aggregate)**
```python
total_sum = 0
total_count = 0
for chunk in pd.read_csv('data.csv', chunksize=500_000):
    total_sum += chunk['price'].sum()
    total_count += len(chunk)
avg = total_sum / total_count
```

**Pattern 2: Filter and collect**
```python
results = []
for chunk in pd.read_csv('data.csv', chunksize=500_000):
    filtered = chunk[chunk['category'] == 'Electronics']
    results.append(filtered)
electronics = pd.concat(results)
```

---

## Section C: Welford's Online Algorithm

### The Problem

```python
# Naive: requires storing ALL values
mean = sum(values) / len(values)
variance = sum((x - mean)**2 for x in values) / len(values)
# Problem: N values in memory + two passes over data!
```

### Welford's Algorithm (Single Pass, O(1) Memory)

```python
def update(self, x):
    self.count += 1
    delta = x - self.mean           # Difference from OLD mean
    self.mean += delta / self.count  # Update mean
    delta2 = x - self.mean          # Difference from NEW mean
    self.M2 += delta * delta2       # Update sum of squared diffs
```

Key: `delta * delta2` uses both old and new mean — this telescoping product gives the correct incremental update.

### Validation

```python
# Compare with NumPy (population statistics)
assert np.isclose(stats.mean, data.mean())
assert np.isclose(stats.std(), data.std(ddof=0))  # ddof=0 for population
```

---

## Section D: Threading vs Multiprocessing

| Pattern | GIL Released? | Best For | Module |
|---------|---------------|----------|--------|
| Threading | Only during I/O | File reading, network | `ThreadPoolExecutor` |
| Multiprocessing | Separate processes | CPU computation | `ProcessPoolExecutor` |

### Amdahl's Law

```
Speedup = 1 / (s + (1 - s) / N)
  s = sequential fraction
  N = number of workers

If 10% is sequential:
  4 workers → 3.1x
  8 workers → 4.7x
  ∞ workers → 10x (maximum!)
```

---

## Section E: TODO Function Guide

Step-by-step help for each TODO function. **Try on your own first!**

---

### TODO 1: `generate_warmup_data()`

**Goal**: Create a 5M-row DataFrame and save as Parquet.

**Step-by-step**:

1. Set the random seed:
   ```python
   np.random.seed(seed)
   ```
2. Build the DataFrame:
   ```python
   df = pd.DataFrame({
       'product_id': np.random.randint(1, 10000, n),
       'category': np.random.choice(['Electronics','Clothing','Home','Sports','Food'], n),
       'price': np.random.uniform(1, 1000, n).round(2),
       'quantity': np.random.randint(1, 50, n),
       'customer_id': np.random.randint(1, 100000, n),
   })
   ```
3. Save and return:
   ```python
   df.to_parquet(WARMUP_PARQUET, index=False)
   print(f"File generated: {df.memory_usage(deep=True).sum() / 1e6:.1f} MB in RAM")
   return df
   ```

**Expected output**:
```
File generated: 438.0 MB in RAM
Shape: (5000000, 5)
```

**Key functions**: `np.random.randint()`, `np.random.choice()`, `np.random.uniform()`, `df.to_parquet()`

---

### TODO 2: `benchmark_read_methods()`

**Goal**: Time three approaches to reading Parquet and compute the speedup.

!!! note
    Similarly to the **Time Method A** step, in the next steps store the results in the `results` dict and print the output in the same format.

**Step-by-step**:

0. Create dict to store results
   ```python
   results = {}
   ```
   
1. Time Method A (Pandas):
   ```python
   start = time.perf_counter()
   df_pandas = pd.read_parquet(WARMUP_PARQUET)
   t_pandas = time.perf_counter() - start
   ram_pandas = df_pandas.memory_usage(deep=True).sum() / 1e6
   results['pandas'] = {
        'time_sec': round(t_pandas, 3),
        'ram_mb': round(df_pandas.memory_usage(deep=True).sum() / 1e6, 1)
    }
    print(f"pd.read_parquet:      {t_pandas:.3f}s  |  RAM: {results['pandas']['ram_mb']:.1f} MB")
   ```

2. Time Method B (Arrow direct):
   ```python
   start = time.perf_counter()
   table = pq.read_table(WARMUP_PARQUET)
   t_arrow = time.perf_counter() - start
   ram_arrow = table.nbytes / 1e6
   ```
3. Time Method C (Arrow → Pandas conversion):
   ```python
   start = time.perf_counter()
   df_from_arrow = table.to_pandas()
   t_convert = time.perf_counter() - start
   ```
4. Print results and compute speedup:
   ```python
   print(f"Speedup reading Arrow: {t_pandas / t_arrow:.1f}x faster than Pandas")
   ```
5. Store `speedup` in the `results` dict:
   ```python
   results['speedup'] = round(t_pandas / t_arrow, 1)
   ```

6. Return dict with all timings.

**Expected output format**:
```
pd.read_parquet:      0.823s  |  RAM: 148.5 MB
pq.read_table:        0.412s  |  RAM Arrow: 80.2 MB
Arrow -> Pandas:      0.398s

Speedup reading Arrow: 2.0x faster than Pandas
```

**Return format**:
```python
{
    'pandas': {'time_sec': 0.823, 'ram_mb': 148.5},
    'arrow':  {'time_sec': 0.412, 'ram_mb': 80.2},
    'arrow_to_pandas': {'time_sec': 0.398},
    'speedup': 2.0
}
```

**Key functions**: `time.perf_counter()`, `pd.read_parquet()`, `pq.read_table()`, `table.to_pandas()`, `table.nbytes`, `df.memory_usage(deep=True).sum()`

---

### TODO 3: `benchmark_projection_pushdown()`

**Goal**: Compare reading all columns vs only 2 columns, then compute revenue in Arrow.

**Step-by-step**:

1. Read ALL columns and time it:
   ```python
   start = time.perf_counter()
   table_all = pq.read_table(WARMUP_PARQUET)
   t_all = time.perf_counter() - start
   ```
2. Read ONLY `price` and `quantity`:
   ```python
   start = time.perf_counter()
   table_cols = pq.read_table(WARMUP_PARQUET, columns=['price', 'quantity'])
   t_cols = time.perf_counter() - start
   ```
3. Compute revenue using Arrow compute:
   ```python
   revenue_col = pc.multiply(table_cols.column('price'), table_cols.column('quantity'))
   total_revenue = pc.sum(revenue_col).as_py()
   ```
4. Print speedup and data reduction:
   ```python
   print(f"Speedup: {t_all / t_cols:.1f}x  |  Data reduction: {table_all.nbytes / table_cols.nbytes:.1f}x")
   ```

**Expected output format**:
```
Read all columns: 0.412s  |  80.2 MB
Read 2 columns:   0.135s  |  40.0 MB
Speedup: 3.1x  |  Data reduction: 2.0x
Total revenue: 12,345,678,901
```

**Return format**:
```python
{
    'all_columns': {'time_sec': 0.412, 'size_mb': 80.2},
    'two_columns': {'time_sec': 0.135, 'size_mb': 40.0},
    'speedup': 3.1,
    'data_reduction': 2.0,
    'total_revenue': 12345678901
}
```

---

### TODO 4: `generate_large_dataset()`

**Goal**: Create a 20M-row dataset and save as CSV + partitioned Parquet.

**Step-by-step**:

1. Set seed and create DataFrame:
   ```python
   np.random.seed(seed)
   df = pd.DataFrame({
       'date': pd.date_range('2020-01-01', periods=n, freq='s'),
       'product_id': np.random.randint(1, 10000, n),
       'category': np.random.choice(['Electronics', 'Clothing', 'Home', 'Sports', 'Food'], n),
       'price': np.random.uniform(1, 1000, n).round(2),
       'quantity': np.random.randint(1, 50, n),
       'customer_id': np.random.randint(1, 100000, n),
   })
   ```
2. Save as CSV:
   ```python
   df.to_csv(SALES_CSV, index=False)
   ```
3. Save as partitioned Parquet:
   ```python
   df.to_parquet(SALES_PARTITIONED, partition_cols=['category'], index=False)
   ```

**Expected output**: A CSV file (~1.2 GB) and a `sales_partitioned/` directory with one subfolder per category.

---

### TODO 5: `chunked_statistics()`

**Goal**: Calculate average price without loading the full file in RAM.

**Step-by-step**:

1. Initialize accumulators:
   ```python
   total_sum = 0
   total_count = 0
   ```
2. Iterate over the CSV in chunks of 500,000 rows:
   ```python
   for chunk in pd.read_csv(SALES_CSV, chunksize=500_000):
   ```
3. Inside the loop, update the accumulators with the current chunk's data:
   ```python
       total_sum += chunk['price'].sum()
       total_count += len(chunk)
   ```
4. Calculate the final average price:
   ```python
   avg_price = total_sum / total_count
   ```
5. Return the results in a dictionary.

**Return format**:
```python
{'total_sum': 9999123456.78, 'total_count': 20000000, 'avg_price': 499.9562}
```

**Key insight**: `pd.read_csv(chunksize=N)` returns an **iterator**, not a DataFrame. Each chunk is an independent 500K-row DataFrame.

---

### TODO 6: `chunked_filter_save()`

**Goal**: Filter only Electronics sales chunk by chunk, then save.

**Step-by-step**:

1. Initialize an empty list to store the filtered chunks:
   ```python
   results = []
   ```
2. Iterate over the CSV in chunks:
   ```python
   for chunk in pd.read_csv(SALES_CSV, chunksize=500_000):
   ```
3. Inside the loop, filter for "Electronics" and append to the list:
   ```python
       filtered = chunk[chunk['category'] == 'Electronics']
       results.append(filtered)
   ```
4. Concatenate all filtered DataFrames into one:
   ```python
   electronics = pd.concat(results)
   ```
5. Save the combined DataFrame as a Parquet file:
   ```python
   electronics.to_parquet(ELECTRONICS_PARQUET, index=False)
   ```
6. Return the total number of filtered rows:
   ```python
   return len(electronics)
   ```

**Expected output**: ~4,000,000 Electronics rows (1/5 of 20M).

**Key insight**: We collect filtered chunks in a list, then `pd.concat()` at the end. This uses much less memory than loading the full file.

---

### TODO 7: `OnlineStats` class

**Goal**: Implement Welford's algorithm for streaming mean, variance, and std.

**Step-by-step**:

**`update(self, x)`**:

```python
self.count += 1
delta = x - self.mean           # Step 1: diff from OLD mean
self.mean += delta / self.count  # Step 2: update mean
delta2 = x - self.mean          # Step 3: diff from NEW mean (important!)
self.M2 += delta * delta2       # Step 4: accumulate squared diffs
self.min_val = min(self.min_val, x)
self.max_val = max(self.max_val, x)
```

**`variance(self)`**:
```python
if self.count < 2:
    return 0.0
return self.M2 / self.count  # Population variance
```

**`std(self)`**:
```python
return self.variance() ** 0.5
```

**Validation**: After processing 100K values, `np.isclose(stats.mean, data.mean())` and `np.isclose(stats.std(), data.std(ddof=0))` should both be `True`.

**Common mistake**: Using `ddof=1` (sample std) instead of `ddof=0` (population std). Welford's gives **population** variance.

---

### TODO 8: `benchmark_threading()`

**Goal**: Show that threads speed up I/O-bound file reading.

!!! note
    We use **CSV files** for the threading benchmark because CSV reading is truly I/O-bound (simple text parsing). Parquet reading involves CPU-heavy Snappy decompression, which holds the GIL and prevents threading speedup.

**Step-by-step**:

1. Get the list of CSV partition files:
   ```python
   files = sorted(glob.glob(str(PARTITIONS_DIR / '*.csv')))
   ```
2. Measure the time to read all files **sequentially**:
   ```python
   start = time.time()
   dfs = [pd.read_csv(f) for f in files]
   seq_time = time.time() - start
   ```
3. Measure the time to read all files **in parallel using threads**:
   ```python
   start = time.time()
   with ThreadPoolExecutor(max_workers=n_workers) as executor:
       dfs = list(executor.map(pd.read_csv, files))
   thread_time = time.time() - start
   ```
4. Calculate the speedup:
   ```python
   speedup = seq_time / thread_time
   ```
5. Print and return the timing results.

**Return format**:
```python
{'sequential_sec': 12.5, 'threaded_sec': 4.2, 'speedup': 3.0}
```

**Why threads work here**: CSV reading is I/O-bound — the GIL is released during disk reads and text parsing, so threads can overlap the I/O effectively.

---

### TODO 9: `benchmark_multiprocessing()`

**Goal**: Show that processes speed up CPU-bound computation.

**Step-by-step**:

1. Import the worker function and get the list of partition files:
   ```python
   from lab05_workers import heavy_process
   files = sorted(glob.glob(str(PARTITIONS_DIR / '*.parquet')))
   ```
2. Measure the time to process all files **sequentially**:
   ```python
   start = time.time()
   results_seq = [heavy_process(f) for f in files]
   seq_time = time.time() - start
   ```
3. Measure the time to process all files **in parallel using processes**:
   ```python
   start = time.time()
   with ProcessPoolExecutor(max_workers=n_workers) as executor:
       results_par = list(executor.map(heavy_process, files))
   proc_time = time.time() - start
   ```
4. Calculate the speedup:
   ```python
   speedup = seq_time / proc_time
   ```
5. Print and return the timing results.

**Return format**:
```python
{'sequential_sec': 8.34, 'multiprocessing_sec': 2.85, 'speedup': 2.9}
```

**Why processes work here**: `heavy_process()` does CPU-intensive math (sqrt, log1p, groupby). Each process has its own GIL.

---

### TODO 10: `run_parallel_pipeline()`

**Goal**: Run the complete pipeline sequentially vs in parallel, combining results.

**Step-by-step**:

1. Import the worker function and get the list of partition files:
   ```python
   from lab05_workers import process_partition
   files = sorted(glob.glob(str(PARTITIONS_DIR / '*.parquet')))
   ```
2. Run the **sequential pipeline** (process partitions then concatenate and group):
   ```python
   start = time.time()
   results_seq = [process_partition(f) for f in files]
   final_seq = pd.concat(results_seq).groupby(level=[0, 1]).sum()
   seq_time = time.time() - start
   ```
3. Run the **parallel pipeline** using processes:
   ```python
   start = time.time()
   with ProcessPoolExecutor(max_workers=n_workers) as executor:
       partial_results = list(executor.map(process_partition, files))
   final_par = pd.concat(partial_results).groupby(level=[0, 1]).sum()
   par_time = time.time() - start
   ```
4. Calculate the speedup:
   ```python
   speedup = seq_time / par_time
   ```
5. Print the comparison and the final results:
   ```python
   print(f"Sequential: {seq_time:.2f}s")
   print(f"Parallel:   {par_time:.2f}s")
   print(f"Speedup:    {speedup:.1f}x")
   print(f"\nFinal results:")
   print(final_par)
   ```
6. Return a dictionary with the timing results.

**Return format**:
```python
{'sequential_sec': 12.5, 'parallel_sec': 4.2, 'speedup': 3.0}
```

**Key insight**: `pd.concat(results).groupby(level=[0, 1]).sum()` combines partial aggregations from multiple partitions into a single final result.

---

## Expected Results

```
Exercise 0: PyArrow Benchmark
  pq.read_table() ~2x faster than pd.read_parquet()
  Projection pushdown: 2-3x speedup reading 2 of 5 columns

Exercise 1: Chunking
  Memory stays constant (~50 MB variation) across 40 chunks
  ~4M Electronics rows filtered from 20M total

Exercise 2: Online Statistics
  OnlineStats matches NumPy within float precision
  Full streaming computation in ~60-90 seconds

Exercise 3: Parallelization
  Threading: 2-4x speedup for file reading
  Multiprocessing: 2-4x speedup for CPU-bound work
  Sub-linear scaling (Amdahl's Law)

Exercise 4: Pipeline
  Parallel pipeline: 2-4x faster than sequential
```

---

## Common Pitfalls

| Mistake | Fix |
|---------|-----|
| Using threads for CPU work | GIL prevents parallelism — use `ProcessPoolExecutor` |
| Collecting all chunks in memory | Reduce/aggregate as you go |
| Too many workers | Process creation has overhead — 4-8 is usually optimal |
| Forgetting `ddof=0` | Welford gives population variance; use `std(ddof=0)` in NumPy |
| Not using `observed=True` | `pd.cut()` + `groupby` may include empty categories |

---

## Useful Links

- [PyArrow Parquet Documentation](https://arrow.apache.org/docs/python/parquet.html)
- [Pandas Chunked Reading](https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html)
- [Welford's Algorithm (Wikipedia)](https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm)
- [concurrent.futures Documentation](https://docs.python.org/3/library/concurrent.futures.html)

---

## Files to Submit

1. `notebooks/lab05_outofcore_parallel.ipynb` (with all cells executed)
2. `results/lab05_metrics.json` (generated by the notebook)

---

Good luck!
