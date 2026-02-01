# Lab 02: Tips & Quick Reference

Complete guide with detailed tips, code examples, and quick reference for all TODO functions.

---

## 📚 General Tips

Before you start:

- **Start Small**: Always test your code with `n=1000` before running it on `n=1_000_000`
- **Use `%%time`**: In Jupyter, put `%%time` at the top of a cell to quickly measure execution time
- **Restart Kernel**: If memory usage gets too high, restart the kernel (Circular Arrow icon)
- Read the docstring carefully - it tells you exactly what the function should do
- Look at the test cell below each function - it shows you how the function will be used

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
import tracemalloc

tracemalloc.start()
# ... code to measure ...
current, peak = tracemalloc.get_traced_memory()
tracemalloc.stop()
peak_mb = peak / 1024 / 1024
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
import io

pr = cProfile.Profile()
pr.enable()
# ... code to profile ...
pr.disable()

# Print stats
stats = pstats.Stats(pr)
stats.sort_stats('cumulative')
stats.print_stats(10)  # Top 10 functions

# Capture to string
string_io = io.StringIO()
stats.stream = string_io
stats.print_stats(10)
stats_string = string_io.getvalue()
```

---

## TODO 1: `generate_user_logs()`

### What you need to do
Generate a synthetic dataset with 1 million rows representing user activity logs.

### Key concepts
- Use `numpy` to generate random data efficiently
- Use `pandas` to organize data into a DataFrame
- Save as CSV and return metadata

### Detailed hints

**Step 1:** Set the random seed for reproducibility
```python
np.random.seed(seed)
```

**Step 2:** Generate the `user_id` column (1 to 50,000, allows duplicates!)
```python
user_ids = np.random.randint(1, 50001, size=n_rows)
```
- This creates duplicates on purpose - you'll need them later!

**Step 3:** Generate the `session_id` column (unique session IDs)
```python
session_ids = np.arange(n_rows)
```

**Step 4:** Generate the `action` column (random choice from list)
```python
actions = np.random.choice(
    ["click", "view", "purchase", "scroll", "search"], 
    size=n_rows
)
```

**Step 5:** Generate the `timestamp` column
```python
timestamps = pd.date_range("2024-01-01", periods=n_rows, freq="s")
```

**Step 6:** Generate the `value` column (random floats 0 to 1000)
```python
values = np.random.uniform(0, 1000, size=n_rows)
```

**Step 7:** Create the DataFrame
```python
df = pd.DataFrame({
    "user_id": user_ids,
    "session_id": session_ids,
    "action": actions,
    "timestamp": timestamps,
    "value": values
})
```

**Step 8:** Save to CSV
```python
df.to_csv(path, index=False)
```

**Step 9:** Get file size and return metadata
```python
file_size = path.stat().st_size
return {
    "rows": df.shape[0],
    "cols": df.shape[1],
    "size_mb": file_size / 1_000_000
}
```

### Common mistakes
- ❌ Forgetting `np.random.seed(seed)` — results won't be reproducible
- ❌ Using `index=True` in `to_csv()` — adds an unwanted column
- ❌ Wrong column names — tests expect exact names

---

## TODO 2: `benchmark_search()`

### What you need to do
Compare search performance in a List (O(N)) vs a Set (O(1)).

### Key concepts
- Lists require scanning every element to find an item
- Sets use hash tables for instant lookup
- Measure individual search times and calculate median

### Detailed hints

**Step 1:** Set the random seed
```python
np.random.seed(seed)
```

**Step 2:** Create both data structures
```python
data_list = list(range(n))
data_set = set(range(n))
```

**Step 3:** Generate random keys to search for
```python
keys = np.random.randint(0, n, size=n_searches)
```

**Step 4:** Time each search in the list
```python
list_times = []
for key in keys:
    start = time.perf_counter()
    _ = key in data_list
    end = time.perf_counter()
    list_times.append((end - start) * 1000)  # Convert to ms
```

**Step 5:** Time each search in the set
```python
set_times = []
for key in keys:
    start = time.perf_counter()
    _ = key in data_set
    end = time.perf_counter()
    set_times.append((end - start) * 1000)  # Convert to ms
```

**Step 6:** Calculate medians and speedup
```python
list_median = np.median(list_times)
set_median = np.median(set_times)
speedup = list_median / set_median

return {
    "list_median_ms": list_median,
    "set_median_ms": set_median,
    "speedup": speedup
}
```

### Why use median?
- Individual searches can vary due to CPU caching
- Median is less affected by outliers than mean
- Gives you the "typical" performance

### Common mistakes
- ❌ Not converting to milliseconds — tests expect ms
- ❌ Timing the loop instead of individual searches
- ❌ Using mean instead of median

---

## TODO 3: `bubble_sort()`

### What you need to do
Implement the classic bubble sort algorithm that compares adjacent elements.

### Key concepts
- Two nested loops = O(N²) time complexity
- Compare adjacent elements, swap if out of order
- After each pass, the largest unsorted element "bubbles" to its position

### Detailed hints

**Step 1:** Copy the array (don't modify original)
```python
arr = arr.copy()
n = len(arr)
```

**Step 2:** Outer loop (number of passes)
```python
for i in range(n):
    # Inner loop here
```

**Step 3:** Inner loop (compare adjacent elements)
```python
for j in range(0, n - i - 1):
    if arr[j] > arr[j + 1]:
        # Swap
        arr[j], arr[j + 1] = arr[j + 1], arr[j]
```

**Step 4:** Return sorted array
```python
return arr
```

### Why bubble sort is O(N²)
- Two nested loops over N elements
- N × N = N² comparisons in worst case
- For N=5000: 25,000,000 operations!

### Expected results

| N | Bubble Sort | Python Sort | Speedup |
|---|-------------|-------------|---------|
| 100 | ~0.001s | ~0.00001s | ~100x |
| 1,000 | ~0.1s | ~0.0001s | ~1000x |
| 5,000 | ~2.5s | ~0.001s | ~2500x |

---

## TODO 4-5: Space Complexity Functions

### What you need to do
Implement two different approaches to finding duplicates, demonstrating the time-space trade-off.

### Key concepts
- **Set-based approach**: Uses O(N) extra memory for O(N) time
- **In-place approach**: Uses O(1) extra memory but O(N log N) time
- Neither is "better" — the right choice depends on constraints

### TODO 4: `find_duplicates_set()`

**Step 1:** Create two sets
```python
seen = set()       # Items we've encountered
duplicates = set() # Items we've seen twice
```

**Step 2:** Single pass through data
```python
for item in data:
    if item in seen:
        duplicates.add(item)
    else:
        seen.add(item)
```

**Step 3:** Return as list
```python
return list(duplicates)
```

### TODO 5: `find_duplicates_inplace()`

**Step 1:** Sort in-place
```python
data.sort()  # Modifies the input!
```

**Step 2:** Find adjacent duplicates
```python
duplicates = set()
for i in range(1, len(data)):
    if data[i] == data[i - 1]:
        duplicates.add(data[i])
```

**Step 3:** Return as list
```python
return list(duplicates)
```

### Comparison table

| Method | Time | Space | Pros | Cons |
|--------|------|-------|------|------|
| Set-based | O(N) | O(N) | Fastest, preserves order | Uses extra memory |
| In-place | O(N log N) | O(1) | Memory efficient | Modifies input, slower |

### Common mistakes
- ❌ Forgetting that `sort()` modifies the original list
- ❌ Using a list instead of a set (would be O(N²) for lookups!)
- ❌ Not handling the edge case of empty input

---

## TODO 6: `profile_function()`

### What you need to do
Use Python's `cProfile` to profile a function and identify bottlenecks.

### Key concepts
- cProfile measures where time is spent
- Sort by 'cumulative' time to find the biggest bottlenecks
- Capture stats to a string for analysis

### Detailed hints

**Step 1:** Create profiler and enable it
```python
import cProfile
import pstats
import io

profiler = cProfile.Profile()
profiler.enable()
```

**Step 2:** Run the function
```python
result = fn(*args, **kwargs)
```

**Step 3:** Disable profiler
```python
profiler.disable()
```

**Step 4:** Create stats object and sort
```python
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
```

**Step 5:** Capture stats to string
```python
string_io = io.StringIO()
stats.stream = string_io
stats.print_stats(10)  # Top 10 functions
stats_string = string_io.getvalue()
```

**Step 6:** Return results
```python
return result, stats_string
```

### Reading profile output
- **ncalls**: Number of times called
- **tottime**: Time in function (excluding sub-calls)
- **cumtime**: Total time including sub-calls
- Look for functions with high cumtime - those are your bottlenecks!

### Common mistakes
- ❌ Not disabling the profiler before creating stats
- ❌ Forgetting to call `fn(*args, **kwargs)` — you're profiling nothing!
- ❌ Not sorting by 'cumulative' time

---

## Flamegraph with py-spy (Demonstration)

### What it does
Generate a visual flamegraph to identify bottlenecks.

### Generating a flamegraph from command line
```bash
# Generate flamegraph (run from terminal, not notebook)
py-spy record -o flamegraph.svg -- python flamegraph_profile.py

# Open in browser
open flamegraph.svg
```

### Reading a flamegraph
```
        [───────────── main: 100% ─────────────]
                    |
    [─ load: 20% ─][────── process: 80% ──────]
                            |
                    [─ inner_loop: 95% ─]
```

- **Width** = time spent (wider = slower)
- **Height** = call stack depth
- **Parent** is above, **children** below
- Click to zoom into a section

---

## Line-by-Line Profiling (Demonstration)

### What it does
Use `line_profiler` to identify exactly which lines are slow.

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
- **% Time**: Percentage of total function time

Lines 8-9 run 50 million times for just 10K elements — this is why O(N²) is catastrophic!

---

## TODO 7: `find_duplicates_fast()`

### What you need to do
Replace the O(N²) nested loop with an O(N) hash-based solution using `Counter`.

### Key concepts
- Counter counts occurrences in one pass: O(N)
- Filter for items with count > 1
- This is 1000x+ faster than nested loops!

### Detailed hints

**Step 1:** Import Counter
```python
from collections import Counter
```

**Step 2:** Count all occurrences
```python
counts = Counter(data)
```
- This creates a dictionary: `{value: count}`
- Runs in O(N) time!

**Step 3:** Filter for duplicates (count > 1)
```python
duplicates = [item for item, count in counts.items() if count > 1]
```

**Step 4:** Return the list
```python
return duplicates
```

### Complete solution
```python
def find_duplicates_fast(data):
    counts = Counter(data)
    return [item for item, count in counts.items() if count > 1]
```

### Why is this O(N)?
1. `Counter(data)` iterates list once: **O(N)**
2. List comprehension iterates counter once: **O(unique items) ≤ O(N)**
3. Total: **O(N)**

### Expected results
- Slow (O(N²)): ~15 seconds for 10K items
- Fast (O(N)): ~0.001 seconds for 10K items
- **Speedup: 10,000x+**
- Fast version can handle 1M items in under 1 second!

### Common mistakes
- ❌ Using nested loops — that's what we're trying to avoid!
- ❌ Forgetting to filter for `count > 1` — returns all items
- ❌ Using a set instead of Counter — loses the count information

---

## ⚠️ Common Pitfalls

| Mistake | Fix |
|---------|-----|
| Running O(N²) on 1M items | Test with small samples first! |
| `in list` inside a loop | Convert list to set first |
| Optimizing without profiling | Profile first, then optimize |
| Using mean instead of median for timing | Median is more robust |
| Forgetting `np.random.seed()` | Results won't be reproducible |

---

## 🚀 Big O Cheat Sheet

| Complexity | Name | 1K items | 1M items | Example |
|------------|------|----------|----------|---------|
| O(1) | Constant | Instant | Instant | Set lookup |
| O(log N) | Logarithmic | 10 ops | 20 ops | Binary search |
| O(N) | Linear | 1K ops | 1M ops | Counter |
| O(N log N) | Linearithmic | 10K ops | 20M ops | Python sort |
| O(N²) | Quadratic | 1M ops | 1T ops ❌ | Bubble sort |
| O(2^N) | Exponential | ∞ | ∞ | Brute force |

**Rule**: At N=1,000,000, anything above O(N log N) becomes impractical!

---

## 🆘 Getting Help

If you're stuck:

1. **Run with small data first** — Use 1,000 or 10,000 rows to test
2. **Check memory usage** — Are you running out of RAM?
3. **Profile your code** — Is the bottleneck where you expect?
4. **Read the error message** — Python errors are usually informative
5. **Check the docstring** — Does your function return the right type?
6. **Look at the test** — What does it expect?

---

## 🔗 Useful Links

- [collections.Counter Documentation](https://docs.python.org/3/library/collections.html#collections.Counter)
- [cProfile Documentation](https://docs.python.org/3/library/profile.html)
- [py-spy GitHub](https://github.com/benfred/py-spy)
- [line_profiler Documentation](https://github.com/pyutils/line_profiler)
- [Latency Numbers Every Programmer Should Know](https://colin-scott.github.io/personal_website/research/interactive_latency.html)

Good luck! 🎉
