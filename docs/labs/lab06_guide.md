# Lab 06: Tips & Quick Reference

This guide provides detailed tips, code examples, and cheatsheets to help you complete [Lab 06](lab06_instructions.md).

---

## General Tips

- **Think streaming**: If you're loading the entire file, ask yourself if chunking would work.
- **Profile memory**: Use `psutil` to verify that your approach keeps memory constant.
- **Choose the right parallelism**: Threads for I/O, processes for CPU.

---

## Section A: PyArrow vs Pandas

### Why PyArrow is Faster

```python
import pyarrow.parquet as pq
import pandas as pd

# Pandas: reads Parquet → Arrow Table → converts to Pandas DataFrame
df = pd.read_parquet('data.parquet')  # Two steps internally

# PyArrow: reads Parquet → Arrow Table (stops here)
table = pq.read_table('data.parquet')  # One step, no conversion
```

The conversion from Arrow Table to Pandas DataFrame involves:
- Allocating new memory for Pandas structures
- Converting Arrow arrays to NumPy arrays
- Creating the DataFrame object with index

### Projection Pushdown

```python
# Read ALL columns (slow, wasteful)
table = pq.read_table('data.parquet')  # Reads everything

# Read ONLY needed columns (fast, lean)
table = pq.read_table('data.parquet', columns=['price', 'quantity'])
# Only reads 2 columns from disk — others are never touched
```

### Arrow Compute Functions

```python
import pyarrow.compute as pc

# Arithmetic on Arrow arrays (no Pandas needed)
revenue = pc.multiply(table.column('price'), table.column('quantity'))
total = pc.sum(revenue).as_py()  # .as_py() converts to Python scalar

# Other useful functions
mean = pc.mean(table.column('price')).as_py()
filtered = pc.filter(table.column('price'), pc.greater(table.column('price'), 100))
```

### iter_batches() for Streaming

```python
pf = pq.ParquetFile('data.parquet')

# Read metadata WITHOUT loading data
print(pf.metadata.num_rows)       # Total rows
print(pf.metadata.num_row_groups) # Number of row groups
print(pf.schema_arrow)            # Column types

# Stream in batches
for batch in pf.iter_batches(batch_size=500_000, columns=['price', 'quantity']):
    # batch is a RecordBatch, not a DataFrame!
    # Process with Arrow compute or convert: chunk_df = batch.to_pandas()
    revenue = pc.sum(pc.multiply(batch.column('price'), batch.column('quantity')))
```

---

## Section B: Chunking (Out-of-Core Processing)

### CSV Chunking

```python
import pandas as pd

# Returns an iterator, NOT a DataFrame
reader = pd.read_csv('big_file.csv', chunksize=500_000)

# Process one chunk at a time
total = 0
for chunk in reader:
    total += chunk['column'].sum()
    # chunk is a regular DataFrame with 500K rows
    # Previous chunk is garbage-collected when the loop advances
```

### Key Insight: Memory Stays Constant

```
Without chunking:
  Load 20M rows → 2 GB RAM → MemoryError!

With chunking (500K rows each):
  Load chunk 1  → 50 MB → process → discard
  Load chunk 2  → 50 MB → process → discard
  ...
  Load chunk 40 → 50 MB → process → discard
  Peak RAM: ~50 MB regardless of file size!
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

**Pattern 3: Transform and save**
```python
for i, chunk in enumerate(pd.read_csv('data.csv', chunksize=500_000)):
    chunk['revenue'] = chunk['price'] * chunk['quantity']
    mode = 'w' if i == 0 else 'a'
    header = i == 0
    chunk.to_csv('output.csv', mode=mode, header=header, index=False)
```

### Memory Monitoring

```python
import psutil, os

process = psutil.Process(os.getpid())
mem_mb = process.memory_info().rss / 1024**2  # RSS in MB
```

---

## Section C: Welford's Online Algorithm

### The Problem with Naive Approaches

```python
# Naive: requires storing ALL values
values = [x1, x2, x3, ..., xN]  # N values in memory
mean = sum(values) / len(values)
variance = sum((x - mean)**2 for x in values) / len(values)

# Also naive: two-pass approach
# Pass 1: compute mean
# Pass 2: compute variance
# Problem: must read data twice!
```

### Welford's Algorithm (Single Pass)

```python
class OnlineStats:
    def __init__(self):
        self.count = 0
        self.mean = 0.0
        self.M2 = 0.0       # Sum of squared differences from mean

    def update(self, x):
        self.count += 1
        delta = x - self.mean           # Difference from old mean
        self.mean += delta / self.count  # Update mean incrementally
        delta2 = x - self.mean          # Difference from NEW mean
        self.M2 += delta * delta2       # Update sum of squared diffs

    def variance(self):
        if self.count < 2:
            return 0.0
        return self.M2 / self.count     # Population variance

    def std(self):
        return self.variance() ** 0.5
```

### Why It Works

The key insight is that `delta * delta2` uses both the old mean (via `delta`) and the new mean (via `delta2`). This telescoping product gives the correct incremental update to the sum of squared differences.

### Validation

```python
import numpy as np

# Compare with NumPy (population statistics)
assert np.isclose(stats.mean, data.mean())
assert np.isclose(stats.std(), data.std(ddof=0))
# Note: ddof=0 for population std, ddof=1 for sample std
```

---

## Section D: Threading vs Multiprocessing

### When to Use Each

| Pattern | GIL Released? | Best For | Python Module |
|---------|---------------|----------|---------------|
| Threading | Only during I/O | File reading, network | `ThreadPoolExecutor` |
| Multiprocessing | Separate processes | CPU computation | `ProcessPoolExecutor` |

### Python's GIL (Global Interpreter Lock)

```
Threading:
  Thread 1: [read file---]          [read file---]
  Thread 2:     [read file---]          [read file---]
  CPU:      [############################]
  → Overlaps I/O waits. Good for I/O-bound tasks.

  Thread 1: [compute-------]
  Thread 2:                 [compute-------]  (must wait!)
  CPU:      [##############][##############]
  → No parallelism for CPU work. GIL blocks.

Multiprocessing:
  Process 1: [compute-------]
  Process 2: [compute-------]  (runs simultaneously!)
  CPU Core1: [##############]
  CPU Core2: [##############]
  → True parallel execution. Each process has its own GIL.
```

### ThreadPoolExecutor

```python
from concurrent.futures import ThreadPoolExecutor

files = ['file1.parquet', 'file2.parquet', ...]

# Sequential
dfs = [pd.read_parquet(f) for f in files]

# Threaded (faster for I/O)
with ThreadPoolExecutor(max_workers=8) as executor:
    dfs = list(executor.map(pd.read_parquet, files))
```

### ProcessPoolExecutor

```python
from concurrent.futures import ProcessPoolExecutor

def heavy_computation(filepath):
    df = pd.read_parquet(filepath)
    # CPU-intensive work
    df['score'] = np.sqrt(df['price']) * np.log1p(df['quantity'])
    return df.groupby('category')['score'].sum()

# Sequential
results = [heavy_computation(f) for f in files]

# Parallel (faster for CPU-bound)
with ProcessPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(heavy_computation, files))
```

### Amdahl's Law

```
Speedup = 1 / (s + (1 - s) / N)

Where:
  s = fraction of work that is sequential (can't be parallelized)
  N = number of workers

Example: if 10% of work is sequential (s = 0.1):
  2 workers:  1 / (0.1 + 0.9/2) = 1.82x
  4 workers:  1 / (0.1 + 0.9/4) = 3.08x
  8 workers:  1 / (0.1 + 0.9/8) = 4.71x
  ∞ workers:  1 / 0.1 = 10x (maximum!)
```

This explains why the speedup curve flattens — you can never be faster than the sequential part.

---

## Section E: TODO Function Guide

Step-by-step help for each TODO function. Try on your own first!

### TODO 1: `generate_warmup_data()`

**Goal**: Create a 5M-row DataFrame and save as Parquet.

```python
np.random.seed(seed)
df = pd.DataFrame({
    'product_id': np.random.randint(1, 10000, n),
    'category': np.random.choice(['Electronics','Clothing','Home','Sports','Food'], n),
    'price': np.random.uniform(1, 1000, n).round(2),
    'quantity': np.random.randint(1, 50, n),
    'customer_id': np.random.randint(1, 100000, n),
})
df.to_parquet(WARMUP_PARQUET, index=False)
return df
```

---

### TODO 2: `benchmark_read_methods()`

**Goal**: Time three approaches to reading Parquet.

```python
# Method A: Pandas
start = time.perf_counter()
df = pd.read_parquet(WARMUP_PARQUET)
t_pandas = time.perf_counter() - start

# Method B: Arrow direct
start = time.perf_counter()
table = pq.read_table(WARMUP_PARQUET)
t_arrow = time.perf_counter() - start

# Method C: Arrow → Pandas
start = time.perf_counter()
df = table.to_pandas()
t_convert = time.perf_counter() - start
```

---

### TODO 3: `benchmark_projection_pushdown()`

**Goal**: Compare all-column vs 2-column read.

```python
table_all = pq.read_table(WARMUP_PARQUET)
table_cols = pq.read_table(WARMUP_PARQUET, columns=['price', 'quantity'])

# Arrow-native computation
revenue = pc.multiply(table_cols.column('price'), table_cols.column('quantity'))
total = pc.sum(revenue).as_py()
```

---

### TODO 4: `process_with_iter_batches()`

**Goal**: Stream through the file in batches.

```python
pf = pq.ParquetFile(WARMUP_PARQUET)
total_revenue = 0
for batch in pf.iter_batches(batch_size=500_000, columns=['price', 'quantity']):
    revenue = pc.multiply(batch.column('price'), batch.column('quantity'))
    total_revenue += pc.sum(revenue).as_py()
```

---

### TODO 7: `chunked_statistics()`

**Goal**: Calculate average price without loading full file.

```python
total_sum = 0
total_count = 0
for chunk in pd.read_csv(SALES_CSV, chunksize=500_000):
    total_sum += chunk['price'].sum()
    total_count += len(chunk)
avg_price = total_sum / total_count
```

---

### TODO 10: `OnlineStats.update()`

**Goal**: Implement Welford's algorithm.

```python
def update(self, x):
    self.count += 1
    delta = x - self.mean
    self.mean += delta / self.count
    delta2 = x - self.mean          # Uses UPDATED mean
    self.M2 += delta * delta2
    self.min_val = min(self.min_val, x)
    self.max_val = max(self.max_val, x)
```

---

### TODO 14: `benchmark_threading()`

**Goal**: Show that threads speed up I/O-bound work.

```python
files = sorted(glob.glob(str(PARTITIONS_DIR / '*.parquet')))

# Sequential
dfs = [pd.read_parquet(f) for f in files]

# Threaded
with ThreadPoolExecutor(max_workers=8) as executor:
    dfs = list(executor.map(pd.read_parquet, files))
```

---

### TODO 17: `process_partition()`

**Goal**: Transform a single partition file.

```python
df = pd.read_parquet(filepath)
df['revenue'] = df['price'] * df['quantity']
df['price_bin'] = pd.cut(df['price'], bins=[0, 50, 200, 500, 1000],
                         labels=['low', 'mid', 'high', 'premium'])
result = df.groupby(['category', 'price_bin'], observed=True).agg({
    'revenue': ['sum', 'mean', 'count'],
    'quantity': 'sum'
})
return result
```

---

### TODO 18: `run_parallel_pipeline()`

**Goal**: Compare sequential vs parallel processing of all partitions.

```python
# Sequential
results = [process_partition(f) for f in files]
final_seq = pd.concat(results).groupby(level=[0, 1]).sum()

# Parallel
with ProcessPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(process_partition, files))
final_par = pd.concat(results).groupby(level=[0, 1]).sum()
```

---

## Quick Reference: Summary Table

| Technique | Memory | Parallelism | Best For |
|-----------|--------|-------------|----------|
| `pd.read_csv(chunksize=)` | Constant | Sequential | CSV files > RAM |
| `pf.iter_batches()` | Constant | Sequential | Parquet streaming |
| `ThreadPoolExecutor` | Proportional | I/O-bound | Reading many files |
| `ProcessPoolExecutor` | Proportional | CPU-bound | Heavy computation |
| Welford's algorithm | O(1) | Sequential | Streaming statistics |
| Chunking + Multiprocessing | Controlled | Both | Full pipeline |

---

## Common Pitfalls

1. **Using threads for CPU work** — GIL prevents true parallelism. Use `ProcessPoolExecutor` instead.
2. **Collecting all chunks in memory** — Defeats the purpose of chunking. Reduce/aggregate as you go.
3. **Too many workers** — Process creation has overhead. Usually 4-8 workers is optimal.
4. **Forgetting `ddof=0`** — NumPy defaults to `ddof=0` (population), but Pandas defaults to `ddof=1` (sample). Welford's gives population variance.
5. **Not using `observed=True`** — With `pd.cut()` categorical bins, `groupby` may include empty categories unless `observed=True`.
