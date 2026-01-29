# Lab 02: Tips & Quick Reference

This guide provides detailed tips, code examples, and cheatsheets to help you complete [Lab 02](lab02_instructions.md).

---

## 📚 General Tips

- **Start Small**: Always test your code with `n=1000` before running it on `n=1_000_000`.
- **Use `%%time`**: In Jupyter, put `%%time` at the top of a cell to quickly measure execution time.
- **Restart Kernel**: If memory usage gets too high, restart the kernel (Circular Arrow icon).

---

## 🔑 Cheatsheet: Big O Complexity

| Complexity | Name | Example Operation | Suitability for 1M Rows |
|------------|------|-------------------|-------------------------|
| **O(1)** | Constant | `set.add()`, `dict[key]` | ✅ Perfect |
| **O(log N)** | Logarithmic | Binary search | ✅ Great |
| **O(N)** | Linear | Loop over list | ✅ Good (1M ops) |
| **O(N log N)** | Linearithmic | `sorted()`, `list.sort()` | ⚠️ OK (20M ops) |
| **O(N²)** | Quadratic | Nested loops | ❌ FATAL (1T ops) |

---

## 🔑 Cheatsheet: Essential Functions

### Timing Code
```python
import time

# High-precision timer
start = time.perf_counter()
# ... code to time ...
end = time.perf_counter()
elapsed = end - start  # In seconds
```

### Memory Measurement
```python
import psutil

# Get current process memory in MB
memory_mb = psutil.Process().memory_info().rss / 1_000_000
```

### Hash-Based Structures
```python
# Set: O(1) membership test
my_set = set(range(1_000_000))
exists = 42 in my_set  # Instant!

# Counter: O(N) frequency counting
from collections import Counter
counts = Counter([1, 2, 2, 3, 3, 3])
# {1: 1, 2: 2, 3: 3}
```

### Profiling with cProfile
```python
import cProfile
import pstats

pr = cProfile.Profile()
pr.enable()
# ... code to profile ...
pr.disable()

stats = pstats.Stats(pr)
stats.sort_stats('cumulative')
stats.print_stats(10)  # Top 10 functions
```

### Pandas Chunking
```python
# Load in chunks
for chunk in pd.read_csv("file.csv", chunksize=50000):
    # Process each chunk
    process(chunk)
```

---

## Section A: Dataset Generation (`generate_user_logs`)

### What you need to do
Generate a 1 million row synthetic dataset representing user activity logs.

### Key columns
| Column | Type | Generator |
|--------|------|-----------|
| user_id | int | `np.random.randint(1, 50001, size=n)` |
| session_id | int | `np.arange(n)` |
| action | str | `np.random.choice([...], size=n)` |
| timestamp | datetime | `pd.date_range(...)` |
| value | float | `np.random.uniform(0, 1000, size=n)` |

### Common mistakes
- ❌ Forgetting `np.random.seed(seed)` — results won't be reproducible
- ❌ Using `index=True` — adds an unwanted column

---

## Section B: Search Efficiency (`benchmark_search`)

### What you need to do
Compare O(N) list search vs O(1) set search.

### Key steps
1. Create `list(range(n))` and `set(range(n))`
2. Generate random keys to search
3. Time each search individually
4. Calculate median times

### Why use median?
- Individual searches can vary due to CPU caching
- Median gives you the "typical" performance

### Expected result
- List: ~10-50 ms per search
- Set: ~0.001 ms per search
- **Speedup: 1000x+**

---

## Section C: Data Flow (`load_full`, `load_chunked`, `load_iterator`)

### What you need to do
Compare three data loading strategies.

### Strategy comparison
| Method | Code | Memory |
|--------|------|--------|
| Full | `pd.read_csv(path)` | High |
| Chunked | `pd.read_csv(path, chunksize=N)` | Medium |
| Iterator | `for line in open(path):` | Minimal |

### Memory measurement tip
```python
start_mem = get_memory_mb()
# ... load data ...
end_mem = get_memory_mb()
memory_used = end_mem - start_mem
```

---

## Section D: Identifying Bottlenecks (`profile_function`)

### What you need to do
Wrap a function with cProfile to identify bottlenecks.

### Template
```python
pr = cProfile.Profile()
pr.enable()
result = fn(*args, **kwargs)
pr.disable()

stats = pstats.Stats(pr)
stats.sort_stats('cumulative')

# Capture to string
import io
string_io = io.StringIO()
stats.stream = string_io
stats.print_stats(10)
stats_string = string_io.getvalue()
```

### Reading profile output
- **ncalls**: Number of times called
- **tottime**: Time in function (excluding sub-calls)
- **cumtime**: Total time including sub-calls

---

## Section E: The 10x Challenge (`find_duplicates_fast`)

### What you need to do
Replace the O(N²) nested loop with an O(N) hash-based solution.

### Solution using Counter
```python
from collections import Counter

def find_duplicates_fast(data):
    counts = Counter(data)
    return [item for item, count in counts.items() if count > 1]
```

### Why is this O(N)?
1. Counter iterates list once: O(N)
2. List comprehension iterates counter once: O(unique items) ≤ O(N)
3. Total: O(N)

### Alternative: Sorting (O(N log N))
```python
def find_duplicates_sorted(data):
    sorted_data = sorted(data)  # O(N log N)
    duplicates = []
    prev = None
    for item in sorted_data:  # O(N)
        if item == prev and item not in duplicates:
            duplicates.append(item)
        prev = item
    return duplicates
```

---

## ⚠️ Common Pitfalls

| Mistake | Fix |
|---------|-----|
| Running O(N²) on 1M items | Test with small samples first! |
| `in list` inside a loop | Convert list to set first |
| Loading full file when memory-limited | Use chunking or iterators |
| Optimizing without profiling | Profile first, then optimize |
| Using mean instead of median for timing | Median is more robust |

---

## 📊 Expected Results

When you complete the lab, you should see something like:

```
Exercise 1: Search Efficiency
  List: ~15 ms per search
  Set: ~0.001 ms per search
  Speedup: 1000x+

Exercise 2: Data Flow
  Full load: ~2 sec, ~300 MB memory
  Chunked: ~3 sec, ~50 MB peak memory
  Iterator: ~5 sec, ~0 MB memory

Exercise 3: Profiling
  Identified nested loop as bottleneck

Exercise 4: Optimization
  Slow: ~15 seconds (10K items)
  Fast: ~0.001 seconds (10K items)
  Speedup: 10000x+
```

---

## 📦 Files to Submit

1. `notebooks/lab02_complexity_dataflow.ipynb` (with all cells executed)
2. `results/lab02_metrics.json` (generated by the notebook)

**Do NOT submit:**
- The 1M row CSV file (too large!)
- Screenshot of results

---

## 🆘 Getting Help

If you're stuck:

1. **Run with small data first** — Use 1,000 or 10,000 rows to test
2. **Check memory usage** — Are you running out of RAM?
3. **Profile your code** — Is the bottleneck where you expect?
4. **Read the error message** — Python errors are usually informative

---

## 🔗 Useful Links

- [collections.Counter Documentation](https://docs.python.org/3/library/collections.html#collections.Counter)
- [cProfile Documentation](https://docs.python.org/3/library/profile.html)
- [psutil Documentation](https://psutil.readthedocs.io/)
- [Pandas read_csv with chunking](https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html)

---

## 🚀 Big O Cheat Sheet

| Complexity | Name | 1K items | 1M items |
|------------|------|----------|----------|
| O(1) | Constant | Instant | Instant |
| O(log N) | Logarithmic | 10 ops | 20 ops |
| O(N) | Linear | 1K ops | 1M ops |
| O(N log N) | Linearithmic | 10K ops | 20M ops |
| O(N²) | Quadratic | 1M ops | 1T ops ❌ |
| O(2^N) | Exponential | ∞ | ∞ |

**Rule**: At N=1,000,000, anything above O(N log N) becomes impractical!

Good luck! 🎉
