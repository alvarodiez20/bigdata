# Lab 04: Tips & Quick Reference

This guide provides detailed tips, code examples, and cheatsheets to help you complete [Lab 04](lab04_instructions.md).

---

## General Tips

- **Think in arrays, not loops**: If you're iterating over a DataFrame, you're probably doing it wrong.
- **Profile first**: Use `%timeit` to measure before and after optimization.
- **Choose the right format**: CSV for interoperability, Parquet for analytics, Feather for speed.

---

## Section A: Storage Formats

### CSV (Comma-Separated Values)

```python
# Write
df.to_csv('data.csv', index=False)

# Read (full)
df = pd.read_csv('data.csv')

# Read (selective columns)
df = pd.read_csv('data.csv', usecols=['col1', 'col2'])
```

**Pros**: Universal, human-readable, any tool can open it
**Cons**: Slow (text parsing), large (no compression), reads all data

### Parquet

```python
# Write with different compression
df.to_parquet('data_snappy.parquet', compression='snappy')  # Fast compression
df.to_parquet('data_zstd.parquet', compression='zstd')      # Best compression
df.to_parquet('data_none.parquet', compression=None)        # No compression

# Read (full)
df = pd.read_parquet('data.parquet')

# Read (column selection - only reads these columns from disk)
df = pd.read_parquet('data.parquet', columns=['precio', 'cantidad'])

# Read (predicate pushdown - skips row groups that don't match)
df = pd.read_parquet('data.parquet',
                     filters=[('categoria', '==', 'Electronica')])

# Combined: column selection + predicate pushdown
df = pd.read_parquet('data.parquet',
                     columns=['precio', 'cantidad'],
                     filters=[('categoria', '==', 'Electronica')])
```

**Pros**: Columnar (reads only needed columns), compressed, predicate pushdown, type-safe
**Cons**: Not human-readable, requires pyarrow/fastparquet

### Feather (Arrow IPC)

```python
# Write
df.to_feather('data.feather')

# Read
df = pd.read_feather('data.feather')

# Read (column selection)
df = pd.read_feather('data.feather', columns=['col1', 'col2'])
```

**Pros**: Fastest read/write, zero-copy when possible, good for intermediate data
**Cons**: Less compression than Parquet, no predicate pushdown, less widespread

### Compression Comparison

| Compression | Speed | Ratio | Best For |
|-------------|-------|-------|----------|
| Snappy | Fast | Moderate | General use, streaming |
| Zstd | Medium | Best | Archival, cold storage |
| None | Fastest | None | When I/O is not bottleneck |
| LZ4 (Feather) | Fast | Moderate | Intermediate pipeline steps |

### When to Use Each Format

| Scenario | Best Format |
|----------|-------------|
| Share with non-technical users | CSV |
| Long-term storage / data lake | Parquet (Zstd) |
| Analytical queries (column subset) | Parquet (Snappy) |
| Intermediate pipeline steps | Feather |
| Streaming / real-time | Parquet (Snappy) |
| Data exchange between tools | CSV or Parquet |

---

## Section B: Predicate Pushdown

### What Is Predicate Pushdown?

Parquet files contain **row groups** with min/max statistics per column. When you filter, PyArrow can skip entire row groups without reading them:

```
Row Group 1: categoria min='Deportes', max='Hogar'
  → Filter: categoria == 'Electronica'
  → Skip! (Electronica not in range)

Row Group 2: categoria min='Electronica', max='Ropa'
  → Filter: categoria == 'Electronica'
  → Read (might contain matches)
```

### Supported Filter Operators

```python
# Equality
filters=[('col', '==', 'value')]

# Comparison
filters=[('col', '>', 100)]
filters=[('col', '<=', 50)]

# Multiple conditions (AND)
filters=[('col1', '==', 'A'), ('col2', '>', 100)]

# IN operator
filters=[('col', 'in', ['A', 'B', 'C'])]
```

---

## Section C: Vectorization Patterns

### Pattern 1: Simple Arithmetic

```python
# SLOW: Loop
result = []
for i in range(len(df)):
    result.append(df['a'].iloc[i] * 2 + df['b'].iloc[i])

# FAST: Vectorized
result = df['a'] * 2 + df['b']
```

### Pattern 2: Conditional Assignment

```python
# SLOW: Loop with if/else
categories = []
for val in df['age']:
    if val < 18:
        categories.append('child')
    elif val < 65:
        categories.append('adult')
    else:
        categories.append('senior')

# FAST: np.select
conditions = [
    df['age'] < 18,
    df['age'] < 65,
    df['age'] >= 65
]
choices = ['child', 'adult', 'senior']
df['category'] = np.select(conditions, choices, default='unknown')

# Alternative: pd.cut (for numeric bins)
df['category'] = pd.cut(df['age'],
                        bins=[0, 18, 65, 120],
                        labels=['child', 'adult', 'senior'])
```

### Pattern 3: Two-Condition If/Else

```python
# SLOW: Loop
result = []
for val in df['x']:
    if val > 0:
        result.append(val * 2)
    else:
        result.append(0)

# FAST: np.where
result = np.where(df['x'] > 0, df['x'] * 2, 0)
```

### Pattern 4: Clipping Values

```python
# SLOW: Loop
result = []
for val in df['score']:
    if val > 100:
        result.append(100)
    elif val < 0:
        result.append(0)
    else:
        result.append(val)

# FAST: np.clip
result = np.clip(df['score'], 0, 100)
```

### Pattern 5: Distance Calculation (Broadcasting)

```python
# SLOW: Loop
distances = []
for i in range(len(points_a)):
    dist = math.sqrt(
        (points_a[i, 0] - points_b[i, 0])**2 +
        (points_a[i, 1] - points_b[i, 1])**2
    )
    distances.append(dist)

# FAST: NumPy broadcasting
diff = points_a - points_b  # Broadcasting handles alignment
distances = np.sqrt(np.sum(diff**2, axis=1))

# Or using np.linalg.norm
distances = np.linalg.norm(points_a - points_b, axis=1)
```

### Pattern 6: Normalization (Broadcasting)

```python
# SLOW: Nested loops
for col in columns:
    mean = df[col].mean()
    std = df[col].std()
    for i in range(len(df)):
        df.loc[i, col] = (df.loc[i, col] - mean) / std

# FAST: Broadcasting
mean = df[columns].mean()
std = df[columns].std()
df[columns] = (df[columns] - mean) / std
```

---

## Section D: The `.apply()` Problem

### Why `.apply()` is NOT Vectorization

```python
# This looks clean but is SLOW:
df['result'] = df['x'].apply(lambda x: x * 2)

# Internally, it's doing:
results = []
for value in df['x']:
    results.append(lambda_function(value))  # Python loop!
return pd.Series(results)
```

### Benchmark Comparison (10M elements)

| Method | Time | Speedup |
|--------|------|---------|
| Python loop | 12.5s | 1x |
| List comprehension | 8.2s | 1.5x |
| `.apply()` with lambda | 5.8s | 2.2x |
| `.apply()` with builtin | 2.1s | 6x |
| Vectorized (NumPy) | 0.062s | **200x** |

### When `.apply()` is OK

- Complex logic with no vectorized equivalent
- Small DataFrames (< 10K rows)
- One-time operations (not in hot path)

### Alternatives to `.apply()`

| Instead of | Use |
|------------|-----|
| `.apply(lambda x: x * 2)` | `df['x'] * 2` |
| `.apply(lambda x: x.upper())` | `.str.upper()` |
| `.apply(lambda x: 'a' if x > 0 else 'b')` | `np.where(...)` |
| `.apply(complex_function)` | `@numba.jit` decorator |
| `.apply(dict.get)` | `.map(dict)` |

---

## Section E: Broadcasting

### How Broadcasting Works

NumPy automatically expands arrays to match shapes:

```python
# Scalar + Array
arr = np.array([1, 2, 3])
result = arr + 10  # [11, 12, 13]
# 10 is "broadcast" to [10, 10, 10]

# 1D + 2D
matrix = np.array([[1, 2, 3],
                   [4, 5, 6]])  # shape (2, 3)
vector = np.array([10, 20, 30])  # shape (3,)
result = matrix + vector
# [[11, 22, 33],
#  [14, 25, 36]]
# vector is broadcast to each row
```

### Broadcasting Rules

1. If arrays have different number of dimensions, prepend 1s to smaller shape
2. Dimensions are compatible if equal or one is 1
3. Dimensions of size 1 are stretched to match

```python
# Example: (3, 4) + (4,) -> (3, 4) + (1, 4) -> (3, 4)
# Example: (3, 1) + (1, 4) -> (3, 4)
```

### Common Broadcasting Use Cases

```python
# Normalize columns (subtract mean, divide by std)
data = np.random.randn(1000, 50)  # 1000 samples, 50 features
mean = data.mean(axis=0)  # shape (50,)
std = data.std(axis=0)    # shape (50,)
normalized = (data - mean) / std  # Broadcasting!
```

---

## Section F: TODO Function Guide

Step-by-step help for each TODO function in the notebook. Try on your own first!

### TODO 1: `generate_ventas()`

**Goal**: Create a 5M-row sales DataFrame.

**Step-by-step**:

1. Set the random seed for reproducibility:
   ```python
   np.random.seed(seed)
   ```
2. Build each column using the right NumPy generator:
   ```python
   'id':        range(n)                              # sequential integers
   'fecha':     pd.date_range('2020-01-01', periods=n, freq='s')  # one row per second
   'categoria': np.random.choice([...], n)             # pick from 4 categories
   'producto':  np.random.choice([f'prod_{i}' for i in range(1000)], n)
   'precio':    np.random.uniform(1, 1000, n).round(2) # float with 2 decimals
   'cantidad':  np.random.randint(1, 50, n)            # integer 1-49
   'ciudad':    np.random.choice([...], n)              # pick from 5 cities
   ```
3. Wrap in `pd.DataFrame({...})` and return it.

**Key functions**: `np.random.choice()`, `np.random.uniform()`, `np.random.randint()`, `pd.date_range()`

---

### TODO 2: `save_all_formats()`

**Goal**: Save the DataFrame in 5 formats and measure write time + file size.

**Step-by-step**:

1. Create an empty `results = {}` dictionary.
2. For each format, follow this pattern:
   ```python
   start = time.perf_counter()
   df.to_csv(VENTAS_CSV, index=False)        # or .to_parquet() / .to_feather()
   elapsed = time.perf_counter() - start
   size_mb = os.path.getsize(VENTAS_CSV) / 1024**2
   results['csv'] = {'write_sec': round(elapsed, 2), 'size_mb': round(size_mb, 1)}
   ```
3. Repeat for all 5 formats:
   - `df.to_csv(VENTAS_CSV, index=False)`
   - `df.to_parquet(VENTAS_SNAPPY, compression='snappy')`
   - `df.to_parquet(VENTAS_ZSTD, compression='zstd')`
   - `df.to_parquet(VENTAS_NONE, compression=None)`
   - `df.to_feather(VENTAS_FEATHER)`
4. Return the `results` dictionary.

**Key functions**: `time.perf_counter()`, `os.path.getsize()`, `df.to_csv()`, `df.to_parquet()`, `df.to_feather()`

---

### TODO 3: `benchmark_reads()`

**Goal**: Measure read speed for full, selective, and filtered reads.

**Structure of the returned dictionary**:

```python
{
    'full_read': {
        'csv': 5.123,               # median seconds
        'parquet_snappy': 1.234,
        'feather': 0.987
    },
    'selective_read': {
        'csv': 3.456,
        'parquet_snappy': 0.234
    },
    'filtered_read': {
        'parquet_snappy_filtered': 0.567
    }
}
```

**Step-by-step**:

1. Create the main results dictionary and a helper function to time reads:
   ```python
   results = {}
   ```

2. **Test 1 — Full read** (read all columns, all rows). For each format, run `n_runs` times and take the median:
   ```python
   full_read = {}

   # CSV
   times = []
   for _ in range(n_runs):
       start = time.perf_counter()
       pd.read_csv(VENTAS_CSV)
       times.append(time.perf_counter() - start)
   full_read['csv'] = round(np.median(times), 3)

   # Parquet Snappy
   times = []
   for _ in range(n_runs):
       start = time.perf_counter()
       pd.read_parquet(VENTAS_SNAPPY)
       times.append(time.perf_counter() - start)
   full_read['parquet_snappy'] = round(np.median(times), 3)

   # Feather
   times = []
   for _ in range(n_runs):
       start = time.perf_counter()
       pd.read_feather(VENTAS_FEATHER)
       times.append(time.perf_counter() - start)
   full_read['feather'] = round(np.median(times), 3)

   results['full_read'] = full_read
   ```

3. **Test 2 — Selective read** (only 2 columns: `precio`, `cantidad`). Same timing pattern:
   ```python
   selective_read = {}

   # CSV — note: usecols= still parses the whole file, just drops columns after
   times = []
   for _ in range(n_runs):
       start = time.perf_counter()
       pd.read_csv(VENTAS_CSV, usecols=['precio', 'cantidad'])
       times.append(time.perf_counter() - start)
   selective_read['csv'] = round(np.median(times), 3)

   # Parquet — columns= only reads these columns from disk (much faster!)
   times = []
   for _ in range(n_runs):
       start = time.perf_counter()
       pd.read_parquet(VENTAS_SNAPPY, columns=['precio', 'cantidad'])
       times.append(time.perf_counter() - start)
   selective_read['parquet_snappy'] = round(np.median(times), 3)

   results['selective_read'] = selective_read
   ```

4. **Test 3 — Filtered read** (Parquet predicate pushdown):
   ```python
   filtered_read = {}

   times = []
   for _ in range(n_runs):
       start = time.perf_counter()
       pd.read_parquet(VENTAS_SNAPPY,
                       filters=[('categoria', '==', 'Electronica')])
       times.append(time.perf_counter() - start)
   filtered_read['parquet_snappy_filtered'] = round(np.median(times), 3)

   results['filtered_read'] = filtered_read
   ```

5. Return the results dictionary:
   ```python
   return results
   ```

**Key difference**: CSV `usecols=` still parses the whole file; Parquet `columns=` only reads those columns from disk. That's why the selective read speedup is much bigger for Parquet.

---

### TODO 4: `calculate_distances_fast()`

**Goal**: Replace a Python loop with NumPy broadcasting for Euclidean distance.

**The math**: `d = sqrt((ax - bx)^2 + (ay - by)^2)`

**Step-by-step**:

1. Subtract the arrays element-wise (broadcasting):
   ```python
   diff = points_a - points_b  # shape (N, 2)
   ```
2. Square each element:
   ```python
   diff_sq = diff ** 2  # shape (N, 2)
   ```
3. Sum along axis 1 (x and y components):
   ```python
   sum_sq = np.sum(diff_sq, axis=1)  # shape (N,)
   ```
4. Square root:
   ```python
   return np.sqrt(sum_sq)
   ```

**One-liner alternative**: `np.linalg.norm(points_a - points_b, axis=1)`

---

### TODO 5: `classify_ages_fast()`

**Goal**: Replace an if/elif/else loop with `np.select()`.

**Step-by-step**:

1. Define conditions as boolean arrays:
   ```python
   conditions = [
       df['age'] < 18,    # True where age < 18
       df['age'] < 65,    # True where age < 65 (and >= 18 because np.select picks first match)
       df['age'] >= 65    # True where age >= 65
   ]
   ```
2. Define the corresponding labels:
   ```python
   choices = ['child', 'adult', 'senior']
   ```
3. Apply (include `default` to avoid type errors in NumPy 2.x):
   ```python
   return np.select(conditions, choices, default='unknown')
   ```

**How `np.select` works**: It evaluates conditions in order and returns the first matching choice. This is why `age < 65` works for "adult" — ages < 18 already matched the first condition. The `default` parameter sets the value when no condition matches — it must be the same type as the choices (string), otherwise NumPy raises a `TypeError`.

**Alternative**: `pd.cut(df['age'], bins=[0, 18, 65, 120], labels=['child', 'adult', 'senior'])`

---

### TODO 6: `normalize_columns_fast()`

**Goal**: Replace nested loops with broadcasting for z-score normalization.

**The math**: `z = (x - mean) / std`

**Step-by-step**:

1. Copy the DataFrame to avoid modifying the original:
   ```python
   df = df.copy()
   ```
2. Calculate mean and std for all columns at once (returns a Series):
   ```python
   mean = df[columns].mean()  # shape: (n_columns,)
   std = df[columns].std()    # shape: (n_columns,)
   ```
3. Apply broadcasting — pandas aligns by column name automatically:
   ```python
   df[columns] = (df[columns] - mean) / std
   ```
4. Return the normalized DataFrame.

**Why it's fast**: The slow version has two nested loops (columns x rows). Broadcasting does it in a single operation on the whole matrix.

---

### TODO 7: `calculate_scores_fast()`

**Goal**: Replace a row-by-row loop with vectorized arithmetic + `np.clip()`.

**The formula**: `score = (a * 2 + b) / (c + 1)`, capped at 10.

**Step-by-step**:

1. Apply the formula to entire columns at once:
   ```python
   scores = (df['a'] * 2 + df['b']) / (df['c'] + 1)
   ```
2. Clip values to a maximum of 10:
   ```python
   scores = np.clip(scores, None, 10)  # None means no lower bound
   ```
3. Return as array:
   ```python
   return scores.values
   ```

**Why `df.iloc[i]` is slow**: Each `.iloc[i]` call creates a new Series object — that's Python object creation overhead on every iteration.

---

### TODO 8: `run_vectorization_benchmarks()`

**Goal**: Run 5 benchmarks comparing loops vs vectorized code on a 10M-row DataFrame.

**Structure of the returned dictionary**:

```python
{
    'sum':                    {'loop_sec': 1.23, 'vec_sec': 0.01, 'speedup': 123.0},
    'element_multiply':       {'loop_sec': ...,  'vec_sec': ...,  'speedup': ...},
    'filter_transform':       {'loop_sec': ...,  'vec_sec': ...,  'speedup': ...},
    'apply_vs_vectorized':    {'loop_sec': ...,  'vec_sec': ...,  'speedup': ...},
    'apply_complex_vs_numpy': {'loop_sec': ...,  'vec_sec': ...,  'speedup': ...},
}
```

**Strategy**: Loop versions are too slow on 10M rows, so use a 100K-row subset and scale the time:

```python
results = {}
n = len(df)
n_subset = 100_000
df_sub = df.head(n_subset)
scale = n / n_subset  # = 100
```

**Benchmark 1 — Sum** (loop on subset, vectorized on full):

```python
# Loop version (on subset, then scale)
start = time.perf_counter()
total = 0
for x in df_sub['a']:
    total += x
loop_time = (time.perf_counter() - start) * scale

# Vectorized version (on full DataFrame)
start = time.perf_counter()
_ = df['a'].sum()
vec_time = time.perf_counter() - start

results['sum'] = {
    'loop_sec': round(loop_time, 4),
    'vec_sec': round(vec_time, 6),
    'speedup': round(loop_time / vec_time, 1)
}
```

**Benchmark 2 — Element-wise multiply**:

```python
# Loop version (on subset, then scale)
start = time.perf_counter()
result = []
for i in range(len(df_sub)):
    result.append(df_sub.iloc[i]['a'] * df_sub.iloc[i]['b'])
loop_time = (time.perf_counter() - start) * scale

# Vectorized version (on full DataFrame)
start = time.perf_counter()
_ = df['a'] * df['b']
vec_time = time.perf_counter() - start

results['element_multiply'] = {
    'loop_sec': round(loop_time, 4),
    'vec_sec': round(vec_time, 6),
    'speedup': round(loop_time / vec_time, 1)
}
```

**Benchmark 3 — Filter + transform**:

```python
# Loop version (on subset, then scale)
start = time.perf_counter()
result = []
for i in range(len(df_sub)):
    if df_sub.iloc[i]['c'] > 50:
        result.append(df_sub.iloc[i]['a'] * 2)
loop_time = (time.perf_counter() - start) * scale

# Vectorized version (on full DataFrame)
start = time.perf_counter()
_ = df.loc[df['c'] > 50, 'a'] * 2
vec_time = time.perf_counter() - start

results['filter_transform'] = {
    'loop_sec': round(loop_time, 4),
    'vec_sec': round(vec_time, 6),
    'speedup': round(loop_time / vec_time, 1)
}
```

**Benchmark 4 — `.apply()` with lambda vs vectorized** (both on full DataFrame):

```python
# .apply() version (on full DataFrame — slow but not as slow as raw loop)
start = time.perf_counter()
_ = df['a'].apply(lambda x: x * 2 + 1)
apply_time = time.perf_counter() - start

# Vectorized version
start = time.perf_counter()
_ = df['a'] * 2 + 1
vec_time = time.perf_counter() - start

results['apply_vs_vectorized'] = {
    'loop_sec': round(apply_time, 4),
    'vec_sec': round(vec_time, 6),
    'speedup': round(apply_time / vec_time, 1)
}
```

**Benchmark 5 — `.apply()` with complex row function vs NumPy**:

```python
# .apply() with axis=1 (on subset, then scale — very slow!)
start = time.perf_counter()
_ = df_sub.apply(lambda row: (row['a'] * 2 + row['b']) / (row['c'] + 1), axis=1)
apply_time = (time.perf_counter() - start) * scale

# Vectorized version (on full DataFrame)
start = time.perf_counter()
_ = (df['a'] * 2 + df['b']) / (df['c'] + 1)
vec_time = time.perf_counter() - start

results['apply_complex_vs_numpy'] = {
    'loop_sec': round(apply_time, 4),
    'vec_sec': round(vec_time, 6),
    'speedup': round(apply_time / vec_time, 1)
}
```

Finally, return the results:

```python
return results
```

**Note**: Benchmarks 4-5 use `.apply()` (not a raw loop), but `.apply()` is still slow because it's a hidden Python loop. Benchmark 5 uses `axis=1` which is especially slow — it creates a Series for every single row.

---

### TODO 9: `pipeline_naive()`

**Goal**: Implement the slow pipeline (CSV + loops) to establish a baseline.

**Step-by-step**:

```python
start = time.perf_counter()

# 1. Read full CSV (slow: text parsing, all columns)
df = pd.read_csv(VENTAS_CSV)

# 2. Loop through rows to filter and calculate
totals = []
for i in range(len(df)):
    if df.iloc[i]['categoria'] == 'Electronica':
        totals.append(df.iloc[i]['precio'] * df.iloc[i]['cantidad'])

elapsed = time.perf_counter() - start

# 3. Return results
return {'total': sum(totals), 'count': len(totals), 'time_sec': round(elapsed, 2)}
```

**Why it's slow**: CSV parsing + reading all 7 columns + Python loop with `.iloc[]` on every row.

---

### TODO 10: `pipeline_optimized()`

**Goal**: Implement the fast pipeline (Parquet + vectorized) to show the combined speedup.

**Step-by-step**:

```python
start = time.perf_counter()

# 1. Read only needed columns + filter at read time (predicate pushdown)
df = pd.read_parquet(VENTAS_SNAPPY,
                     columns=['categoria', 'precio', 'cantidad'],
                     filters=[('categoria', '==', 'Electronica')])

# 2. Vectorized calculation (no loop!)
df['total'] = df['precio'] * df['cantidad']

elapsed = time.perf_counter() - start

# 3. Return results
return {'total': df['total'].sum(), 'count': len(df), 'time_sec': round(elapsed, 2)}
```

**Where the speedup comes from**:

- Parquet binary format (no text parsing) — ~3-5x
- Column pruning (3 columns instead of 7) — ~2x
- Predicate pushdown (skip non-matching row groups) — ~2-4x
- Vectorized multiply instead of loop — ~100-200x

---

## Expected Results

```
Exercise 1: Format Comparison
  CSV size: ~300 MB (baseline)
  Parquet Snappy: ~60 MB (~5x smaller)
  Parquet Zstd: ~45 MB (~7x smaller)
  Parquet None: ~100 MB (~3x smaller)
  Feather: ~80 MB (~4x smaller)

  Full read: Parquet ~3-5x faster than CSV
  Selective read (2 cols): Parquet ~10x faster than CSV
  Filtered read: Parquet pushdown ~2-4x faster than full read

Exercise 2: Vectorization
  Distance calculation: 150x speedup
  Age classification: 80x speedup
  Normalization: 120x speedup
  Score calculation: 100x speedup

Exercise 3: Benchmarks
  Simple sum: 200x faster vectorized
  Element-wise multiply: 180x faster
  Conditional filter: 50x faster
  .apply() vs vectorized: 100x faster

Exercise 4: Pipeline
  Naive (CSV + loops): ~30-60 seconds
  Optimized (Parquet + vectorized): ~0.3-1 second
  Total speedup: ~50-200x
```

---

## Common Pitfalls

| Mistake | Fix |
|---------|-----|
| Using `.apply()` for simple ops | Use vectorized operations |
| Loop over DataFrame rows | Use broadcasting or vectorized methods |
| Reading full CSV when you need 2 columns | Use Parquet with `columns=` |
| Not using predicate pushdown | Add `filters=` to `read_parquet()` |
| Storing intermediate files as CSV | Use Feather for temp data |

---

## Useful Links

- [Parquet Format Specification](https://parquet.apache.org/)
- [NumPy Broadcasting](https://numpy.org/doc/stable/user/basics.broadcasting.html)
- [Pandas I/O Comparison](https://pandas.pydata.org/docs/user_guide/io.html)
- [Arrow/Feather Format](https://arrow.apache.org/docs/python/feather.html)
- [From Python to NumPy](https://www.labri.fr/perso/nrougier/from-python-to-numpy/)

---

## Files to Submit

1. `notebooks/lab04_formats_vectorization.ipynb` (with all cells executed)
2. `results/lab04_metrics.json` (generated by the notebook)

---

Good luck!
