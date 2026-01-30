# Lab 03: Tips & Quick Reference

This guide provides detailed tips, code examples, and cheatsheets to help you complete [Lab 03](lab03_instructions.md).

---

## 📚 General Tips

- **Start with analysis**: Before optimizing, understand your data's actual value ranges.
- **Use `memory_usage(deep=True)`**: The `deep=True` flag is essential for accurate object dtype measurement.
- **Profile before and after**: Always measure the impact of your changes.

---

## 🔑 Essential Functions Cheat Sheet

### Memory Measurement
```python
# Total memory usage
df.memory_usage(deep=True).sum() / 1e6  # MB

# Per-column breakdown
df.memory_usage(deep=True)

# Memory of a specific column
df['column'].memory_usage(deep=True) / 1e6  # MB
```

### Data Type Information
```python
# View all dtypes
df.dtypes

# Check unique values (for category decisions)
df['column'].nunique()

# Check value ranges (for int sizing)
df['column'].min(), df['column'].max()
```

### Timing Code
```python
import time

start = time.perf_counter()
# ... operation ...
elapsed = time.perf_counter() - start
print(f"Time: {elapsed:.3f} seconds")
```

---

## Section A: Dataset Generation

### What you need to do
Generate a 5 million row e-commerce dataset.

### Key columns and generators

| Column | Type | Generator |
|--------|------|-----------|
| order_id | int | `np.arange(n)` |
| product_id | int | `np.random.randint(1, 50001, size=n)` |
| category | str | `np.random.choice([...], size=n)` |
| price | float | `np.random.uniform(0.01, 999.99, size=n)` |
| quantity | int | `np.random.randint(1, 101, size=n)` |
| country | str | `np.random.choice([...], size=n)` |
| timestamp | datetime | `pd.date_range(...)` |

### Sample categories
```python
categories = [
    'Electronics', 'Clothing', 'Home', 'Books', 'Toys',
    'Sports', 'Beauty', 'Food', 'Garden', 'Automotive',
    'Health', 'Office', 'Pet', 'Music', 'Games'
]
```

### Sample countries
```python
countries = [
    'Spain', 'France', 'Germany', 'Italy', 'UK',
    'Portugal', 'Netherlands', 'Belgium', 'Poland', 'Sweden',
    'Norway', 'Denmark', 'Finland', 'Austria', 'Switzerland',
    'Ireland', 'Greece', 'Czech', 'Romania', 'Hungary',
    'USA', 'Canada', 'Mexico', 'Brazil', 'Argentina',
    'Japan', 'China', 'Australia', 'India', 'Korea'
]
```

---

## Section B: Exercise 1 - Data Type Optimization

### Type Selection Guide

```
Is it numeric?
├─ NO → Is it a string?
│   ├─ YES → Are <50% values unique?
│   │   ├─ YES → category
│   │   └─ NO → object (or string dtype)
│   └─ Is it a date? → datetime64
└─ YES → Is it an integer?
    ├─ YES → What's the range?
    │   ├─ 0 to 255 → uint8
    │   ├─ -128 to 127 → int8
    │   ├─ 0 to 65,535 → uint16
    │   ├─ -32,768 to 32,767 → int16
    │   ├─ 0 to 4B → uint32
    │   └─ Larger → int64
    └─ NO (decimal) → Precision needed?
        ├─ Low (2-3 decimals) → float32
        └─ High (scientific) → float64
```

### Integer Types Reference

| Type | Min | Max | Bytes |
|------|-----|-----|-------|
| int8 | -128 | 127 | 1 |
| uint8 | 0 | 255 | 1 |
| int16 | -32,768 | 32,767 | 2 |
| uint16 | 0 | 65,535 | 2 |
| int32 | -2.1B | 2.1B | 4 |
| uint32 | 0 | 4.3B | 4 |
| int64 | -9.2Q | 9.2Q | 8 |

### Specifying dtypes when reading
```python
df = pd.read_csv('data.csv', dtype={
    'product_id': 'uint16',      # max 50000 < 65535
    'category': 'category',       # 15 unique values
    'price': 'float32',           # 2 decimal precision enough
    'quantity': 'uint8',          # max 100 < 255
    'country': 'category',        # 30 unique values
})
```

### Automatic downcast function
```python
def optimize_dtypes(df):
    """Automatically reduce numeric types and convert low-cardinality strings."""
    for col in df.columns:
        col_type = df[col].dtype

        if col_type == 'object':
            # Convert strings with few unique values to category
            if df[col].nunique() / len(df) < 0.5:
                df[col] = df[col].astype('category')
        elif 'int' in str(col_type):
            df[col] = pd.to_numeric(df[col], downcast='integer')
        elif 'float' in str(col_type):
            df[col] = pd.to_numeric(df[col], downcast='float')

    return df
```

### Expected memory reduction

| Column | Before | After | Reduction |
|--------|--------|-------|-----------|
| product_id | int64 (8B) | uint16 (2B) | 4x |
| category | object (~50B) | category (~2B) | 25x |
| price | float64 (8B) | float32 (4B) | 2x |
| quantity | int64 (8B) | uint8 (1B) | 8x |
| country | object (~20B) | category (~2B) | 10x |

---

## Section C: Exercise 2 - Format Comparison

### Writing to different formats

```python
# CSV
df.to_csv('data.csv', index=False)

# Compressed CSV
df.to_csv('data.csv.gz', index=False, compression='gzip')

# Parquet with different compressions
df.to_parquet('data_snappy.parquet', compression='snappy')
df.to_parquet('data_gzip.parquet', compression='gzip')
df.to_parquet('data_zstd.parquet', compression='zstd')
df.to_parquet('data_none.parquet', compression=None)

# Feather
df.to_feather('data.feather', compression='zstd')
```

### Reading with column selection (Parquet/Feather only)
```python
# Only read specific columns - much faster!
df = pd.read_parquet('data.parquet', columns=['product_id', 'price', 'quantity'])
```

### Compression comparison

| Codec | Speed | Ratio | Best For |
|-------|-------|-------|----------|
| None | Fastest | 1x | Development |
| Snappy | Fast | 2-3x | Default, balanced |
| LZ4 | Very Fast | 2x | Speed priority |
| Zstd | Medium | 5-8x | Best balance |
| Gzip | Slow | 4-6x | Compatibility |
| Brotli | Very Slow | 8-12x | Maximum compression |

### Format characteristics

| Format | Type | Compression | Column Select | Schema |
|--------|------|-------------|---------------|--------|
| CSV | Row | No | No | No |
| CSV.gz | Row | gzip | No | No |
| Parquet | Column | Various | Yes | Yes |
| Feather | Column | LZ4/Zstd | Yes | Yes |

---

## Section D: Exercise 3 - Parquet Deep Dive

### Inspecting Parquet metadata
```python
import pyarrow.parquet as pq

# Open file for inspection
parquet_file = pq.ParquetFile('data.parquet')

# Basic info
print(f"Row groups: {parquet_file.num_row_groups}")
print(f"Schema: {parquet_file.schema}")

# Row group details
for i in range(parquet_file.num_row_groups):
    rg = parquet_file.metadata.row_group(i)
    print(f"Row group {i}: {rg.num_rows} rows")

    # Column statistics
    for j in range(rg.num_columns):
        col = rg.column(j)
        stats = col.statistics
        if stats:
            print(f"  {col.path_in_schema}: min={stats.min}, max={stats.max}")
```

### Row group size configuration
```python
# Smaller row groups = more parallelism, more overhead
df.to_parquet('small_rg.parquet', row_group_size=10_000)

# Larger row groups = better compression, less overhead
df.to_parquet('large_rg.parquet', row_group_size=1_000_000)

# Default is usually 64MB worth of data
```

### Predicate pushdown
```python
# Without filter - reads everything
df = pd.read_parquet('data.parquet')

# With filter - uses statistics to skip row groups
df = pd.read_parquet('data.parquet',
                     filters=[('price', '>', 100)])

# Multiple conditions
df = pd.read_parquet('data.parquet',
                     filters=[
                         ('price', '>', 100),
                         ('category', '=', 'Electronics')
                     ])
```

---

## Section E: Exercise 4 - Partitioning

### Adding partition columns
```python
df['date'] = pd.to_datetime(df['timestamp'])
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day
```

### Writing partitioned data
```python
# Partition by year and month
df.to_parquet(
    'data_partitioned/',
    partition_cols=['year', 'month'],
    engine='pyarrow'
)

# Results in:
# data_partitioned/
# ├─ year=2024/
# │  ├─ month=1/
# │  │  └─ data-0.parquet
# │  ├─ month=2/
# │  │  └─ data-0.parquet
# │  └─ ...
```

### Reading with partition filters
```python
# Reads only relevant partitions
df = pd.read_parquet(
    'data_partitioned/',
    filters=[
        ('year', '=', 2024),
        ('month', '=', 1)
    ]
)
```

### When to partition

**DO partition when:**
- Dataset > 1 GB
- Queries frequently filter by specific columns
- Data has natural time/category dimensions
- Read >> Write (analytics workloads)

**DON'T partition when:**
- Dataset < 100 MB
- You always read the entire dataset
- High cardinality (>10,000 unique values)
- Very uneven distribution

### Ideal partition sizing
- **Minimum files**: 10-100
- **Maximum files**: 10,000
- **File size**: 100 MB - 1 GB each

---

## 📊 Expected Results

When you complete the lab, you should see something like:

```
Exercise 1: Data Type Optimization
  Baseline memory: ~2,500 MB
  Optimized memory: ~400 MB
  Reduction: 6.3x

Exercise 2: Format Comparison (5M rows)
  CSV write: 45s, 520 MB
  Parquet (zstd) write: 8s, 85 MB
  CSV read: 35s
  Parquet read (full): 3s
  Parquet read (3 cols): 0.8s

Exercise 3: Parquet Config
  Row groups: varies with row_group_size
  Predicate pushdown speedup: 2-5x

Exercise 4: Partitioning
  Query specific day: partitioned 10-50x faster
  Full aggregation: unpartitioned slightly faster
```

---

## ⚠️ Common Pitfalls

| Mistake | Fix |
|---------|-----|
| Forgetting `deep=True` in memory_usage | Always use `df.memory_usage(deep=True)` |
| Using int64 for small ranges | Check `.min()` and `.max()`, use smallest int |
| Not using category for repeated strings | If `nunique() / len(df) < 0.5`, use category |
| Over-partitioning | Keep partitions 100MB-1GB, max 10K files |
| Expecting CSV to support column selection | Only Parquet/Feather support this |

---

## 🔗 Useful Links

- [Pandas dtype reference](https://pandas.pydata.org/docs/reference/arrays.html)
- [Pandas Scaling Guide](https://pandas.pydata.org/docs/user_guide/scale.html)
- [PyArrow Parquet Guide](https://arrow.apache.org/docs/python/parquet.html)
- [Parquet Format Specification](https://parquet.apache.org/docs/)
- [Apache Arrow](https://arrow.apache.org/)

---

## 📦 Files to Submit

1. `notebooks/lab03_data_types_formats.ipynb` (with all cells executed)
2. `results/lab03_metrics.json` (generated by the notebook)

**Do NOT submit:**
- Generated data files
- Partitioned directories

---

Good luck! 🎉
