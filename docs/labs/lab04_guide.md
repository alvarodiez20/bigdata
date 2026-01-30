# Lab 04: Tips & Quick Reference

This guide provides detailed tips, code examples, and cheatsheets to help you complete [Lab 04](lab04_instructions.md).

---

## 📚 General Tips

- **Think in arrays, not loops**: If you're iterating over a DataFrame, you're probably doing it wrong.
- **Profile first**: Use `%timeit` to measure before and after optimization.
- **Memory matters**: Use `psutil` to monitor RAM during out-of-core operations.

---

## 🔑 Why Python Loops Are Slow

Each iteration of a Python loop involves:

1. Interpret bytecode (Python VM)
2. Look up variable in hash table
3. Check type (dynamic typing)
4. Find method (`__mul__`, `__add__`, etc.)
5. Create stack frame for function call
6. Perform operation
7. Check result type
8. Assign result

**Total**: ~200 CPU instructions per operation

**NumPy**: 1-2 CPU instructions (compiled C code)

**Factor**: 100-200x overhead for Python loops!

---

## Section A: Vectorization Patterns

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
df['category'] = np.select(conditions, choices)

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

### Pattern 5: String Operations

```python
# SLOW: Loop
result = []
for s in df['name']:
    result.append(s.lower())

# FAST: Vectorized string method
result = df['name'].str.lower()

# Other useful string operations:
df['name'].str.upper()
df['name'].str.contains('pattern')
df['name'].str.replace('old', 'new')
df['name'].str.split('_')
```

### Pattern 6: Distance Calculation (Broadcasting)

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

### Pattern 7: Normalization (Broadcasting)

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

## Section B: The `.apply()` Problem

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

## Section C: Broadcasting

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
# Example: (3, 4) + (4,) → (3, 4) + (1, 4) → (3, 4)
# Example: (3, 1) + (1, 4) → (3, 4)
```

### Common Broadcasting Use Cases

```python
# Normalize columns (subtract mean, divide by std)
data = np.random.randn(1000, 50)  # 1000 samples, 50 features
mean = data.mean(axis=0)  # shape (50,)
std = data.std(axis=0)    # shape (50,)
normalized = (data - mean) / std  # Broadcasting!

# Outer product
a = np.array([1, 2, 3])  # shape (3,)
b = np.array([10, 20])   # shape (2,)
outer = a[:, np.newaxis] * b  # (3, 1) * (2,) → (3, 2)
# [[10, 20],
#  [20, 40],
#  [30, 60]]
```

---

## Section D: Out-of-Core Computing

### Chunking Pattern

```python
# Basic chunking template
result = initial_value

for chunk in pd.read_csv('large.csv', chunksize=500_000):
    # Process chunk
    partial = process(chunk)

    # Combine with result
    result = combine(result, partial)

# Finalize
final = finalize(result)
```

### Choosing Chunk Size

```
chunksize × row_size × overhead_factor < available_RAM

Example:
- Available RAM: 8 GB (leave 4 GB for OS/other)
- Row size: 1 KB
- Overhead factor: 3 (pandas copies, intermediate results)

chunksize < 4 GB / (1 KB × 3) ≈ 1.3 million rows

Conservative choice: 500,000 rows
```

### Operations Compatible with Chunking

| Operation | Chunking Strategy |
|-----------|-------------------|
| Sum | Accumulate partial sums |
| Count | Accumulate partial counts |
| Mean | Sum + Count, then divide |
| Min/Max | Keep running min/max |
| Groupby + Agg | Accumulate per-group stats |
| Filter | Write matching rows to output |
| Transform | Apply to each chunk, write output |

### Operations NOT Compatible with Simple Chunking

| Operation | Why | Solution |
|-----------|-----|----------|
| Sort | Needs all data | External merge sort |
| Median | Needs all data | Approximate algorithms |
| Complex Join | May span chunks | Hash partition join |
| Window functions | Spans rows | Overlap chunks |

### Memory Monitoring

```python
import psutil
import os

def get_memory_mb():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

# Monitor during processing
for i, chunk in enumerate(pd.read_csv('large.csv', chunksize=500_000)):
    mem = get_memory_mb()
    print(f"Chunk {i}: {mem:.0f} MB")
    process(chunk)
```

---

## Section E: Welford's Online Algorithm

### Why Standard Formulas Fail

```python
# Naive variance (two-pass, all data in memory):
mean = sum(data) / len(data)
variance = sum((x - mean)**2 for x in data) / len(data)
# Problem: Requires two passes and all data in memory!
```

### Welford's Algorithm (One-Pass)

```python
class OnlineStats:
    def __init__(self):
        self.count = 0
        self.mean = 0.0
        self.M2 = 0.0
        self.min_val = float('inf')
        self.max_val = float('-inf')

    def update(self, x):
        self.count += 1
        delta = x - self.mean
        self.mean += delta / self.count
        delta2 = x - self.mean
        self.M2 += delta * delta2
        self.min_val = min(self.min_val, x)
        self.max_val = max(self.max_val, x)

    def variance(self):
        if self.count < 2:
            return 0.0
        return self.M2 / self.count

    def std(self):
        return np.sqrt(self.variance())
```

### Properties

- **Memory**: O(1) - only 5 variables
- **Passes**: 1 - single scan
- **Numerically stable**: Avoids catastrophic cancellation
- **Incremental**: Can update with new data

---

## Section F: Dask Introduction

### Basic Usage

```python
import dask.dataframe as dd

# Read (lazy - no data loaded yet)
ddf = dd.read_csv('large.csv')

# Operations are lazy
result = ddf.groupby('category')['price'].mean()
print(result)  # Shows Dask object, not result

# Trigger computation
result_df = result.compute()  # Now it runs!
```

### When to Use Dask

**Use Dask when:**
- Data is 1-100 GB
- Operations are Pandas-compatible
- You want automatic parallelism
- You don't want to manage chunks manually

**Use manual chunking when:**
- Simple aggregations only
- Maximum control needed
- Dask overhead not worth it

**Use Spark when:**
- Data is > 100 GB
- Multi-node cluster available
- Complex distributed operations

---

## 📊 Expected Results

```
Exercise 1: Vectorization
  Distance calculation: 150x speedup
  Age classification: 80x speedup
  Normalization: 120x speedup
  Score calculation: 100x speedup

Exercise 2: Benchmarks
  Simple sum: 200x faster vectorized
  Element-wise multiply: 180x faster
  Conditional filter: 50x faster
  .apply() vs vectorized: 100x faster

Exercise 3: Out-of-Core
  Processed 20GB with <1GB memory
  Chunked mean matches full-load mean
  Chunked filter produces correct subset

Exercise 4: Online Stats
  Mean error: < 1e-10 (numerical precision)
  Std error: < 1e-10
  Memory: constant throughout processing

Exercise 5: Dask (Bonus)
  Dask vs manual: similar performance
  Dask code: much simpler
  Task graph: shows parallel execution plan
```

---

## ⚠️ Common Pitfalls

| Mistake | Fix |
|---------|-----|
| Using `.apply()` for simple ops | Use vectorized operations |
| Loop over DataFrame rows | Use broadcasting or vectorized methods |
| Loading all data then chunking | Use `chunksize` parameter in `read_csv` |
| Accumulating results in list | Pre-allocate or use online algorithms |
| Forgetting to call `.compute()` in Dask | Dask is lazy - must trigger execution |

---

## 🔗 Useful Links

- [NumPy Broadcasting](https://numpy.org/doc/stable/user/basics.broadcasting.html)
- [Pandas Scaling Guide](https://pandas.pydata.org/docs/user_guide/scale.html)
- [Welford's Algorithm](https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm)
- [Dask Best Practices](https://docs.dask.org/en/latest/best-practices.html)
- [From Python to NumPy](https://www.labri.fr/perso/nrougier/from-python-to-numpy/)

---

## 📦 Files to Submit

1. `notebooks/lab04_vectorization_out_of_core.ipynb` (with all cells executed)
2. `results/lab04_metrics.json` (generated by the notebook)

---

Good luck! 🎉
