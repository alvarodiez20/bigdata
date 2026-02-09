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

## Section B: Baseline Measurement

### Measuring memory accurately

**Key steps:**

1. **Get memory usage**: Use `df.memory_usage(deep=True)` to get a Series with memory per column
   - Returns a Series where index = column names, values = bytes
   - Don't forget `deep=True`!

2. **Calculate total memory**: Sum all values and convert bytes to MB
   - Bytes to MB: divide by `1e6`

3. **Build the result dictionary**: Create a dict with two keys:
   - `'total_mb'`: the total memory in MB
   - `'columns'`: a nested dict with info for each column

4. **For each column**, store:
   - `'dtype'`: convert dtype to string with `str(df[col].dtype)`
   - `'memory_mb'`: get memory from the Series, convert to MB
   - `'nunique'`: use `df[col].nunique()`

**Hint**: You can iterate over columns with `for col in df.columns:`

### Expected output structure

```python
# Example return value:
{
    'total_mb': 2456.78,
    'columns': {
        'order_id': {
            'dtype': 'int64',
            'memory_mb': 38.15,
            'nunique': 5000000
        },
        'product_id': {
            'dtype': 'int64', 
            'memory_mb': 38.15,
            'nunique': 50000
        },
        'category': {
            'dtype': 'object',
            'memory_mb': 856.42,
            'nunique': 15
        },
        # ... more columns
    }
}
```

### Why `deep=True` matters

Without `deep=True`, pandas only counts the pointer memory for object columns, not the actual string data!

```python
# Wrong (underestimates strings)
df.memory_usage()

# Correct (includes string content)
df.memory_usage(deep=True)
```

---

## Section C: Type Analysis & Optimization

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
| **uint8** | 0 | **255** | **1** |
| int16 | -32,768 | 32,767 | 2 |
| **uint16** | 0 | **65,535** | **2** |
| int32 | -2.1B | 2.1B | 4 |
| **uint32** | 0 | **4.3B** | **4** |
| int64 | -9.2Q | 9.2Q | 8 |

### Analyzing column ranges

**Key steps for `analyze_column_ranges()`:**

1. **Create an empty result dictionary** to store analysis for each column

2. **Loop through each column** in the DataFrame

3. **Check the column type**:
   - Use `np.issubdtype(col_type, np.number)` to check if numeric
   - Check if `col_type == 'object'` for strings
   - Otherwise it's likely datetime or other special type

4. **For numeric columns**, store:
   - `'min'`: use `df[col].min()`
   - `'max'`: use `df[col].max()`
   - `'nunique'`: use `int(df[col].nunique())`

5. **For string (object) columns**, store:
   - `'nunique'`: number of unique values
   - `'max_len'`: use `int(df[col].str.len().max())`
   - `'sample'`: first 5 unique values as a list

6. **For other types** (datetime, etc.), store:
   - `'dtype'`: the string representation of the dtype
   - `'nunique'`: number of unique values

### Expected output structure

```python
# Example return value:
{
    'order_id': {
        'min': 0,
        'max': 4999999,
        'nunique': 5000000
    },
    'product_id': {
        'min': 1,
        'max': 50000,
        'nunique': 49987
    },
    'category': {
        'nunique': 15,
        'max_len': 11,
        'sample': ['Electronics', 'Clothing', 'Home', 'Books', 'Toys']
    },
    'price': {
        'min': 0.01,
        'max': 999.99,
        'nunique': 99989
    },
    # ... more columns
}
```

### Loading with optimized dtypes

**How to specify dtypes when reading CSV:**

Use the `dtype` parameter in `pd.read_csv()` to specify types for each column:

```python
df = pd.read_csv('file.csv', dtype={
    'column_name': 'dtype_string',
    'another_column': 'dtype_string',
    # ... more columns
})
```

---

## Section D: Performance Impact

### Benchmarking operations

**Key steps for `benchmark_operation()`:**

1. **Use an if-elif structure** to handle different operation types:
   - `'groupby_sum'`: Group by category and sum prices
   - `'filter'`: Filter rows where country equals a specific value
   - `'sort'`: Sort by price column

2. **For each operation type:**
   - Time the baseline DataFrame:
     - Start timer with `start = time.perf_counter()`
     - Execute the operation (store result in `_` to discard it)
     - Calculate elapsed time: `time.perf_counter() - start`
   - Time the optimized DataFrame (same process)

3. **Return a dictionary** with:
   - `'baseline_sec'`: time for baseline (rounded to 4 decimals)
   - `'optimized_sec'`: time for optimized (rounded to 4 decimals)
   - `'speedup'`: baseline_sec / optimized_sec (rounded to 2 decimals)

**Detailed steps for each operation:**

**For `'groupby_sum'`:**

1. Start timer: `start = time.perf_counter()`
2. Execute on baseline: `_ = df_baseline.groupby('category')['price'].sum()`
3. Calculate baseline time: `baseline_sec = time.perf_counter() - start`
4. Start new timer: `start = time.perf_counter()`
5. Execute on optimized: `_ = df_optimized.groupby('category')['price'].sum()`
6. Calculate optimized time: `optimized_sec = time.perf_counter() - start`

**For `'filter'`:**

1. Start timer
2. Execute on baseline: `_ = df_baseline[df_baseline['country'] == 'Spain']`
3. Calculate baseline time
4. Start new timer
5. Execute on optimized: `_ = df_optimized[df_optimized['country'] == 'Spain']`
6. Calculate optimized time

**For `'sort'`:**

1. Start timer
2. Execute on baseline: `_ = df_baseline.sort_values('price')`
3. Calculate baseline time
4. Start new timer
5. Execute on optimized: `_ = df_optimized.sort_values('price')`
6. Calculate optimized time

**Then for all operations**, calculate the speedup and return the dictionary.

### Expected output structure

```python
# Example return value:
{
    'baseline_sec': 0.0234,
    'optimized_sec': 0.0089,
    'speedup': 2.63
}
```

### Why category is faster

The `category` dtype stores:
- A dictionary of unique values: `{0: 'Electronics', 1: 'Clothing', ...}`
- Integer codes for each row: `[0, 1, 0, 2, 1, ...]`

Operations like groupby and filter use integer comparisons instead of string comparisons!

### Calculating total savings

**Key steps for `calculate_savings()`:**

1. **Calculate memory saved:**
   - Subtract optimized memory from baseline memory
   - Access total memory with: `baseline_memory['total_mb']` and `optimized_memory['total_mb']`

2. **Calculate memory reduction factor:**
   - Divide baseline memory by optimized memory
   - This tells you "how many times smaller" the optimized version is

3. **Calculate average speedup:**
   - Extract the `'speedup'` value from each benchmark result in the list
   - Use a list comprehension: `[r['speedup'] for r in benchmark_results]`
   - Calculate the average: `sum(speedups) / len(speedups)`

4. **Return a dictionary** with:
   - `'memory_saved_mb'`: how many MB were saved (rounded to 2 decimals)
   - `'memory_reduction_factor'`: memory reduction factor (rounded to 2 decimals)
   - `'avg_speedup'`: average speedup across all operations (rounded to 2 decimals)

### Expected output structure

```python
# Example return value:
{
    'memory_saved_mb': 2056.34,
    'memory_reduction_factor': 6.25,
    'avg_speedup': 2.42
}
```

---

## 📊 Expected Results

When you complete the lab, you should see something like:

```
Memory Analysis:
  Baseline memory: ~2,500 MB
  Optimized memory: ~400 MB
  Reduction: 6.3x

Performance:
  Groupby speedup: 2-3x
  Filter speedup: 2-4x
  Sort speedup: 1.5-2x
```

---

## ⚠️ Common Pitfalls

| Mistake | Fix |
|---------|-----|
| Forgetting `deep=True` in memory_usage | Always use `df.memory_usage(deep=True)` |
| Using int64 for small ranges | Check `.min()` and `.max()`, use smallest int |
| Not using category for repeated strings | If `nunique() / len(df) < 0.5`, use category |
| Specifying dtype for timestamp | Use `parse_dates=['timestamp']` separately |
| Using signed int when values are ≥ 0 | Use unsigned types (uint8, uint16, etc.) |

---

## 🔗 Useful Links

- [Pandas dtype reference](https://pandas.pydata.org/docs/reference/arrays.html)
- [Pandas Scaling Guide](https://pandas.pydata.org/docs/user_guide/scale.html)
- [NumPy data types](https://numpy.org/doc/stable/reference/arrays.dtypes.html)

---

## 📦 Files to Submit

1. `notebooks/lab03_data_types.ipynb` (with all cells executed)
2. `results/lab03_metrics.json` (generated by the notebook)

**Do NOT submit:**
- Generated data files
- `__pycache__` directories

---

Good luck! 🎉
