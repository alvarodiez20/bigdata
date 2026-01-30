# Lab 02: Tips & Quick Reference

This guide provides detailed tips, code examples, and cheatsheets to help you complete [Lab 02](lab02_instructions.md).

---

## 📚 General Tips

- **Start Small**: Always test your code with `n=1000` before running it on `n=1_000_000`.
- **Use `%%time`**: In Jupyter, put `%%time` at the top of a cell to quickly measure execution time.
- **Restart Kernel**: If memory usage gets too high, restart the kernel (Circular Arrow icon).

---

## 🔑 Essential Functions Cheat Sheet

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

## Section B (Part 2): Sorting Comparison (`bubble_sort`)

### What you need to do
Compare O(N²) bubble sort vs O(N log N) Python sort.

### Bubble sort implementation
```python
def bubble_sort(arr):
    """O(N²) - classic inefficient algorithm"""
    arr = arr.copy()
    n = len(arr)
    for i in range(n):
        for j in range(0, n-i-1):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
    return arr
```

### Why bubble sort is O(N²)
- Two nested loops over N elements
- N × N = N² comparisons in worst case
- Each comparison is O(1), but there are N² of them

### Expected results

| N | Bubble Sort | Python Sort | Speedup |
|---|-------------|-------------|---------|
| 100 | ~0.001s | ~0.00001s | ~100x |
| 1,000 | ~0.1s | ~0.0001s | ~1000x |
| 5,000 | ~2.5s | ~0.001s | ~2500x |
| 10,000 | ~10s | ~0.002s | ~5000x |

### Key insight
The ratio grows with N because:
- Bubble: O(N²) → 4x more work when N doubles
- Python: O(N log N) → ~2x more work when N doubles

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

## Section D (Part 2): Flamegraph with py-spy

### What you need to do
Generate a visual flamegraph to identify bottlenecks.

### Installation
```bash
pip install py-spy
```

### Generating a flamegraph
```python
# Option 1: From within Python (saves to file)
import subprocess
subprocess.run([
    "py-spy", "record",
    "-o", "flamegraph.svg",
    "--", "python", "-c",
    "from your_script import slow_function; slow_function(data)"
])
```

```bash
# Option 2: From command line
py-spy record -o flamegraph.svg -- python your_script.py

# Option 3: Attach to running process
py-spy record -o flamegraph.svg --pid 12345
```

### Reading a flamegraph
```
        [───────────── main: 100% ─────────────]
                    |
    [─ load: 20% ─][────── process: 80% ──────]
                            |
                    [─ iterrows: 95% ─]
```

- **Width** = time spent (wider = slower)
- **Height** = call stack depth
- **Parent** is above, **children** below
- Click to zoom into a section

### Why use flamegraphs?
- Visual representation is easier to understand than text
- Immediately see the "hot path"
- Interactive: click to explore
- Low overhead (~1-5%)

---

## Section D (Part 3): Line Profiler

### What you need to do
Profile a function line by line to find exact slow lines.

### Installation
```bash
pip install line_profiler
```

### Using line_profiler in code
```python
from line_profiler import LineProfiler

def profile_line_by_line(fn, *args, **kwargs):
    """Profile a function line by line."""
    lp = LineProfiler()
    lp.add_function(fn)

    # Run the function
    lp.enable()
    result = fn(*args, **kwargs)
    lp.disable()

    # Print stats
    lp.print_stats()
    return result

# Usage
profile_line_by_line(find_duplicates_slow, sample_data)
```

### Using from command line
```bash
# Add @profile decorator to functions you want to profile
# Then run with kernprof
kernprof -l -v your_script.py
```

### Reading line_profiler output
```
Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
     5                                           def find_duplicates_slow(data):
     6         1          2.0      2.0      0.0      duplicates = []
     7     10001      15234.0      1.5      0.1      for i in range(len(data)):
     8  50005000   45623000.0      0.9     45.2          for j in range(i + 1, len(data)):
     9  50005000   55123000.0      1.1     54.7              if data[i] == data[j]...
```

- **Hits**: How many times the line executed
- **Time**: Total time on this line (microseconds)
- **Per Hit**: Average time per execution
- **% Time**: Percentage of total function time

### Key insight
Line profiler shows that the inner loop (line 8-9) runs 50 million times for just 10K elements! This is why O(N²) is catastrophic.

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
Exercise 1A: Search Efficiency
  List: ~15 ms per search
  Set: ~0.001 ms per search
  Speedup: 1000x+

Exercise 1B: Sorting
  Bubble sort (N=5000): ~2.5 seconds
  Python sort (N=5000): ~0.001 seconds
  Speedup: 2500x

Exercise 2: Data Flow
  Full load: ~2 sec, ~300 MB memory
  Chunked: ~3 sec, ~50 MB peak memory
  Iterator: ~5 sec, ~0 MB memory

Exercise 3A: cProfile
  Identified find_duplicates_slow as bottleneck
  Nested loops consume 99% of time

Exercise 3B: Flamegraph
  Generated flamegraph.svg
  Widest bar: comparison operations in nested loop

Exercise 3C: Line Profiler
  Line 8 (inner loop): 50M hits, 45% of time
  Line 9 (comparison): 50M hits, 55% of time

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
- [py-spy GitHub](https://github.com/benfred/py-spy) - Sampling profiler for Python
- [line_profiler Documentation](https://github.com/pyutils/line_profiler) - Line-by-line profiling
- [psutil Documentation](https://psutil.readthedocs.io/)
- [Pandas read_csv with chunking](https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html)
- [Latency Numbers Every Programmer Should Know](https://colin-scott.github.io/personal_website/research/interactive_latency.html)

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
