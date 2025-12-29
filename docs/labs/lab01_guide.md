# Lab 01: Tips & Quick Reference

Complete guide with detailed tips, code examples, and quick reference for all TODO functions.

---

## 📚 General Tips

Before you start:

- Read the docstring carefully - it tells you exactly what the function should do
- Look at the test cell below each function - it shows you how the function will be used
- Start simple - get something working, then refine it
- Use the Python documentation if you're stuck on a specific function

---

## 🔑 Essential Functions Cheat Sheet

### Path Operations
```python
from pathlib import Path

# Create a Path object (instantiate)
path = Path("data/raw")                    # Relative path
path = Path("/absolute/path/to/file")      # Absolute path
path = Path("data") / "raw" / "file.csv"   # Build path with / operator
path = Path.cwd()                          # Current working directory
path = Path.home()                         # User's home directory

# Check if exists
path.exists()

# Get file size
path.stat().st_size

# Create directory
path.mkdir(parents=True, exist_ok=True)
```

### NumPy Random Data
```python
import numpy as np

# Set seed for reproducibility
np.random.seed(42)

# Random integers (1 to 10000)
np.random.randint(1, 10001, size=n)

# Random floats (0 to 100)
np.random.uniform(0, 100, size=n)

# Random choice from list
np.random.choice(["A", "B", "C"], size=n)

# Median of a list
np.median([1, 2, 3, 4, 5])  # Returns 3
```

### Pandas DataFrames
```python
import pandas as pd

# Create date range
pd.date_range("2024-01-01", periods=100, freq="h")

# Create DataFrame from dict
df = pd.DataFrame({
    "col1": [1, 2, 3],
    "col2": ["a", "b", "c"]
})

# Get shape (rows, cols)
df.shape  # Returns tuple: (n_rows, n_cols)

# Read CSV
df = pd.read_csv("file.csv")

# Write CSV
df.to_csv("file.csv", index=False)

# Read Parquet
df = pd.read_parquet("file.parquet")

# Write Parquet
df.to_parquet("file.parquet", index=False)
```

### Timing Code
```python
import time

# High-precision timer
start = time.perf_counter()
# ... code to time ...
end = time.perf_counter()
elapsed = end - start  # In seconds
```

### JSON Operations
```python
import json

# Write JSON (pretty-printed)
with open("file.json", "w") as f:
    json.dump(my_dict, f, indent=2)

# Read JSON
with open("file.json") as f:
    data = json.load(f)
```

---

## TODO 1: `ensure_dir()`

### What you need to do
Create a directory (and any parent directories) if it doesn't already exist.

### Key concepts
- The `Path` object has a `.mkdir()` method
- You need to handle two cases:

  1. The directory doesn't exist → create it
  2. The directory already exists → don't raise an error

### Detailed hints

**Step 1:** Use the `mkdir()` method on the `path` object
```python
path.mkdir(...)
```

**Step 2:** Add the `parents` parameter

- `parents=True` means "create parent directories if needed"
- Example: If you're creating `data/raw/`, it will also create `data/` if it doesn't exist

**Step 3:** Add the `exist_ok` parameter

- `exist_ok=True` means "don't raise an error if the directory already exists"
- Without this, calling the function twice would cause an error

### Common mistakes
- ❌ Forgetting `parents=True` → fails if parent directory doesn't exist
- ❌ Forgetting `exist_ok=True` → fails if directory already exists
- ❌ Using `os.makedirs()` instead of `Path.mkdir()` → works but not the modern way

---

## TODO 2: `write_synthetic_csv()`

### What you need to do
Generate fake data with 4 columns and save it as a CSV file.

### Key concepts
- Use `numpy` to generate random data
- Use `pandas` to organize data into a DataFrame
- Save the DataFrame as CSV

### Detailed hints

**Step 1:** Set the random seed
```python
np.random.seed(seed)
```
This ensures the same random data is generated every time with the same seed.

**Step 2:** Generate the `timestamp` column
```python
timestamps = pd.date_range("2024-01-01", periods=n_rows, freq="h")
```

- `periods=n_rows` → create exactly n_rows timestamps
- `freq="h"` → one timestamp per hour

**Step 3:** Generate the `user_id` column (random integers from 1 to 10000)
```python
user_ids = np.random.randint(1, 10001, size=n_rows)
```

- Note: `randint(1, 10001)` gives you 1 to 10000 (upper bound is exclusive)

**Step 4:** Generate the `value` column (random floats from 0 to 100)
```python
values = np.random.uniform(0, 100, size=n_rows)
```

**Step 5:** Generate the `category` column (random choice from A, B, C, D, E)
```python
categories = np.random.choice(["A", "B", "C", "D", "E"], size=n_rows)
```

**Step 6:** Create a DataFrame
```python
df = pd.DataFrame({
    "timestamp": timestamps,
    "user_id": user_ids,
    "value": values,
    "category": categories
})
```

**Step 7:** Save to CSV
```python
df.to_csv(csv_path, index=False)
```

- `index=False` → don't write row numbers as a column

**Step 8:** Get file size
```python
file_size = csv_path.stat().st_size
```

**Step 9:** Return metadata
```python
return {
    "rows": df.shape[0],
    "cols": df.shape[1],
    "size_bytes": file_size
}
```

### Common mistakes
- ❌ Using `randint(1, 10000)` → gives you 1 to 9999 (upper bound is exclusive!)
- ❌ Forgetting `index=False` → CSV will have an extra column with row numbers
- ❌ Wrong column names → tests will fail
- ❌ Not setting the random seed → results won't be reproducible

---

## TODO 3: `time_it()`

### What you need to do
Run a function multiple times and measure how long each run takes.

### Key concepts
- Use `time.perf_counter()` for high-precision timing
- Store all run times in a list
- Calculate the median (middle value)

### Detailed hints

**Step 1:** Create an empty list to store times
```python
times = []
```

**Step 2:** Loop `repeats` times
```python
for _ in range(repeats):
    # timing code here
```

**Step 3:** Inside the loop, measure the time
```python
start = time.perf_counter()  # Record start time
fn()                          # Run the function
end = time.perf_counter()    # Record end time
elapsed = end - start        # Calculate elapsed time
times.append(elapsed)        # Add to list
```

**Step 4:** Calculate the median
```python
median_time = np.median(times)
```

**Step 5:** Return the results
```python
return {
    "runs_sec": times,
    "median_sec": median_time
}
```

### Why use median instead of mean?
- Median is less affected by outliers
- If one run is slow (e.g., due to background processes), it won't skew the result
- Median gives you the "typical" performance

### Common mistakes
- ❌ Using `time.time()` instead of `time.perf_counter()` → less precise
- ❌ Calculating mean instead of median → more affected by outliers
- ❌ Forgetting to call `fn()` → you're timing nothing!
- ❌ Timing the wrong thing (e.g., including the append operation)

---

## TODO 4: `read_csv_once()`

### What you need to do
Read a CSV file and return how many rows and columns it has.

### Key concepts
- Use `pd.read_csv()` to load the file
- Use `.shape` to get dimensions

### Detailed hints

**Step 1:** Read the CSV
```python
df = pd.read_csv(csv_path)
```

**Step 2:** Get the shape
```python
return df.shape
```
- `df.shape` returns a tuple: `(n_rows, n_cols)`
- Example: `(200000, 4)` means 200,000 rows and 4 columns

### Common mistakes
- ❌ Returning `df.shape[0]` and `df.shape[1]` separately → return the tuple directly
- ❌ Returning `len(df)` → only gives rows, not columns
- ❌ Forgetting to return anything

---

## TODO 5: `write_parquet()`

### What you need to do
Read a CSV file and save it in Parquet format.

### Key concepts
- Parquet is a binary, columnar storage format
- It's more efficient than CSV (smaller and faster)

### Detailed hints

**Step 1:** Read the CSV
```python
df = pd.read_csv(csv_path)
```

**Step 2:** Write to Parquet
```python
df.to_parquet(parquet_path, index=False)
```

**Step 3:** Get the Parquet file size
```python
parquet_size = parquet_path.stat().st_size
```

**Step 4:** Return metadata
```python
return {
    "parquet_size_bytes": parquet_size,
    "rows": df.shape[0],
    "cols": df.shape[1]
}
```

### What is Parquet?
- **Columnar storage**: Data is stored by column, not by row
- **Compressed**: Uses efficient compression algorithms
- **Typed**: Stores data type information (no need to parse strings)
- **Fast**: Faster to read/write than CSV for large datasets

### Common mistakes
- ❌ Forgetting `index=False` → Parquet will include row numbers
- ❌ Wrong dictionary keys → tests expect exact key names

---

## TODO 6: `read_parquet_once()`

### What you need to do
Read a Parquet file and return its shape.

### Key concepts
- Same as `read_csv_once()`, but for Parquet files

### Detailed hints

**Step 1:** Read the Parquet file
```python
df = pd.read_parquet(parquet_path)
```

**Step 2:** Return the shape
```python
return df.shape
```

---

## TODO 7: `save_json()`

### What you need to do
Save a Python dictionary as a JSON file.

### Key concepts
- Use the `json` module
- Use `indent=2` for pretty-printing (human-readable)

### Detailed hints

**Step 1:** Open the file in write mode
```python
with open(path, "w") as f:
    # write code here
```

- The `with` statement ensures the file is properly closed

**Step 2:** Write the JSON
```python
json.dump(obj, f, indent=2)
```

- `obj` is the dictionary to save
- `f` is the file object
- `indent=2` makes it pretty (2-space indentation)

### What does `indent=2` do?
Without it:
```json
{"name":"Alice","age":30}
```

With it:
```json
{
  "name": "Alice",
  "age": 30
}
```

### Common mistakes
- ❌ Using `json.dumps()` instead of `json.dump()` → `dumps` returns a string, `dump` writes to a file
- ❌ Forgetting to open the file
- ❌ Not using `with` statement → file might not be properly closed

---

## ⚠️ Common Pitfalls

| Mistake | Fix |
|---------|-----|
| `randint(1, 10000)` gives 1-9999 | Use `randint(1, 10001)` for 1-10000 |
| Forgot `index=False` in `to_csv()` | Always use `index=False` |
| Used `time.time()` instead of `perf_counter()` | Use `perf_counter()` for precision |
| Used `json.dumps()` instead of `dump()` | `dumps` → string, `dump` → file |
| Forgot `parents=True` in `mkdir()` | Add `parents=True, exist_ok=True` |
| Calculated mean instead of median | Use `np.median()` for timing |

---

## 🎯 Testing Strategy

After implementing each function:

1. **Run the test cell** - Does it pass?
2. **Read the error message** - What went wrong?
3. **Check the assertion** - What was expected vs. what you got?
4. **Debug** - Add print statements to see intermediate values
5. **Iterate** - Fix and try again

### Example debugging
```python
def write_synthetic_csv(csv_path, n_rows=200_000, seed=0):
    np.random.seed(seed)
    # ... your code ...
    
    # Add debug prints:
    print(f"DataFrame shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"First row: {df.iloc[0]}")
    
    df.to_csv(csv_path, index=False)
    # ...
```

---

## 📊 Expected Results

When you complete the lab, you should see something like:

```
✓ All imports successful!
✓ ensure_dir() works correctly!
✓ write_synthetic_csv() works correctly!
✓ time_it() works correctly!
✓ read_csv_once() works correctly!
✓ write_parquet() works correctly!
✓ read_parquet_once() works correctly!
✓ save_json() works correctly!

==================================================
RESULTS SUMMARY
==================================================
CSV file size:     15-20 MB
Parquet file size: 2-4 MB
Size ratio:        5-8x (CSV is larger)

CSV median read time:     0.1-0.3 sec
Parquet median read time: 0.02-0.08 sec
Speedup: 2-5x (Parquet is faster)
==================================================

✓ Results saved to: ../results/lab01_metrics.json
```

Exact numbers depend on your system, but Parquet should always be:

- ✅ Smaller (5-10x)
- ✅ Faster to read (2-5x)

---

## 📦 Files to Submit

1. `notebooks/lab01_setup_io.ipynb` (with all cells executed)
2. `results/lab01_metrics.json` (generated by the notebook)

**Do NOT submit:**

- CSV or Parquet files
- The entire repository
- Screenshots

---

## 🆘 Getting Help

If you're stuck:

1. **Read the error message carefully** - It often tells you exactly what's wrong
2. **Check the docstring** - Does your function return the right type?
3. **Look at the test** - What does it expect?
4. **Use print statements** - See what your code is actually doing
5. **Ask your instructor or TA** - That's what they're here for!

---

## 🔗 Useful Links

- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [NumPy Random Documentation](https://numpy.org/doc/stable/reference/random/index.html)
- [Pathlib Documentation](https://docs.python.org/3/library/pathlib.html)
- [Parquet Format](https://parquet.apache.org/)

---

## 🚀 Next Steps

After completing all TODOs:

1. Run all cells from top to bottom
2. Check that `results/lab01_metrics.json` exists
3. Read your JSON file - does it look correct?
4. Write your reflection
5. Submit the notebook and JSON file

Good luck! 🎉
