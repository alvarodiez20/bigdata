# Lab 02: Complexity

Welcome to the second Big Data laboratory session! In this lab, we will leave the safety of small datasets and enter the world where algorithmic complexity matters.

## 📚 Additional Resources

- **[Tips & Reference Guide](lab02_guide.md)** - detailed tips, code examples, and cheatsheets for every TODO.
- **[Lab 01 Instructions](lab01_setup_io.md)** - if you need to review setup or I/O basics.

## 🎯 What You Will Learn

- **Time Complexity**: Why O(N²) fails at scale while O(N log N) succeeds.
- **Space Complexity**: Understanding memory trade-offs in algorithm design.
- **Profiling Tools**: How to use `cProfile`, `py-spy` flamegraphs, and `line_profiler` to find bottlenecks.
- **Optimization**: Converting O(N²) algorithms to O(N) for massive speedups.

## 📝 Implementation Overview

**You will implement these 7 functions:**

| TODO | Function | Purpose |
|------|----------|---------|
| 1 | `generate_user_logs()` | Generate 1M row dataset |
| 2 | `benchmark_search()` | Compare List vs Set |
| 3 | `bubble_sort()` | Classic O(N²) algorithm |
| 4 | `find_duplicates_set()` | Hash-based O(N) approach |
| 5 | `find_duplicates_inplace()` | Sort-based O(N log N) approach |
| 6 | `profile_function()` | Using cProfile |
| 7 | `find_duplicates_fast()` | The 10x optimization challenge |

**Provided for you:**

- ✅ `find_duplicates_slow()` — The deliberately terrible O(N²) version to profile
- ✅ Flamegraph and line_profiler demonstrations

## ✅ Pre-flight Checklist

Before starting, ensure you have:

1.  **Completed Lab 01**: You understand basic I/O and have your environment set up.
2.  **Updated your repo**: Run `git pull` to get the latest changes.
3.  **Create a branch**: `git checkout -b lab02-complexity`
4.  **Installed dependencies**: Run `uv sync --group lab02` to install profiling tools.

---

## 🗺️ Learning Path

```
Dataset → Time Complexity → Space Complexity → Profiling → Optimization
(TODO 1)    (TODO 2-3)        (TODO 4-5)        (TODO 6)     (TODO 7)
   ↓            ↓                  ↓               ↓            ↓
1M rows    O(N) vs O(1)      O(N) space vs    Find the     10x speedup
  CSV     O(N²) vs O(N log N)  O(1) space    bottleneck    with O(N)
```

---

## 📝 Lab Exercises

Follow along in the notebook `notebooks/lab02_complexity_dataflow.ipynb`.

### Exercise 1: Dataset Generation

**TODO 1: `generate_user_logs()`**

We need a dataset large enough to expose inefficient code. Implement this function to create:

-   **Rows**: 1,000,000
-   **Columns**: `user_id`, `session_id`, `action`, `timestamp`, `value`
-   **Features**: Duplicate user IDs (essential for later exercises)

**Goal**: Save this as `data/raw/user_logs_1m.csv`.

---

### Exercise 2: Time Complexity — Search & Sort

Compare the performance impact of different data structures and algorithms.

**TODO 2: `benchmark_search()`** — O(N) vs O(1)

Compare finding an item in a Python **List** versus a **Set**:

1.  Create a list and set of 1M numbers.
2.  Search for 1,000 random keys in both.
3.  Calculate the speedup.

**What to expect**: The Set should be ~1000x faster.

**TODO 3: `bubble_sort()`** — O(N²) vs O(N log N)

Implement the classic bubble sort algorithm:

1.  Compare adjacent elements, swap if out of order.
2.  Repeat until sorted.
3.  Compare against Python's built-in `sorted()` at N = 100, 1000, 5000.

**What to expect**: Python sort should be 100x+ faster at N=5000.

---

### Exercise 3: Space Complexity — Memory Trade-offs

Not just time — sometimes **memory** is the bottleneck.

**TODO 4: `find_duplicates_set()`**

- O(N) time, O(N) space — uses a set to track seen items
- Fast but uses memory proportional to input size

**TODO 5: `find_duplicates_inplace()`**

- O(N log N) time, O(1) extra space — sorts in-place
- Memory efficient but modifies input and is slower

**Goal**: Understand the time-space trade-off.

| Approach | Time | Space | Best When... |
|----------|------|-------|--------------|
| Set-based | O(N) | O(N) | Memory is plentiful, speed is critical |
| Sort in-place | O(N log N) | O(1) | Memory is limited, data can be modified |

---

### Exercise 4: Profiling Bottlenecks

We provide a deliberately terrible O(N²) function: `find_duplicates_slow()`.

**TODO 6: `profile_function()`**

1.  Implement using Python's `cProfile`.
2.  Run on `find_duplicates_slow()` with a small sample (2-5k rows).
3.  Analyze the output to identify the slowest functions.

**Flamegraph Visualization** (Demonstration)

1.  Run the notebook cell to generate a standalone script.
2.  Run from terminal: `py-spy record -o flamegraph.svg -- python src/flamegraph_profile.py`
3.  Open the SVG in browser and identify the widest bar (= bottleneck).

**Line-by-Line Profiling** (Demonstration)

1.  Use the provided `line_profiler` cell.
2.  Identify the exact lines that consume the most time.

**Goal**: Learn that you can't optimize what you can't measure.

---

### Exercise 5: The 10x Challenge 🏆

**TODO 7: `find_duplicates_fast()`**

Your task is to refactor the slow function to be at least **10x faster**.

1.  **Strategy**: Use a Hash Map (Dictionary or `collections.Counter`) to count items in O(N).
2.  **Validation**: Compare your results against the slow version.
3.  **Bonus**: Can you make it 1000x faster?

**Goal**: Prove that Algorithms > Hardware. A better algorithm on a laptop beats a bad algorithm on a supercomputer.

---

## 📊 Expected Results

When you complete the lab successfully:

| Exercise | Metric | Expected Value |
|----------|--------|----------------|
| TODO 2 | Set vs List speedup | ~1000x |
| TODO 3 | Python sort vs Bubble (N=5000) | ~100-2500x |
| TODO 4-5 | Set-based faster than in-place | Yes |
| TODO 7 | Fast vs Slow speedup | **≥10x** (often 10,000x!) |

---

## 📦 What to Submit

Submit **exactly these two files**:

1.  **`notebooks/lab02_complexity.ipynb`** — Your completed notebook.
2.  **`results/lab02_metrics.json`** — The JSON file generated by the notebook.

**Do NOT submit:**
-   The 1M row CSV file (it's ~60-80MB).
-   The `__pycache__` directories.
-   Flamegraph SVG files.

---

## 🚀 Next Steps

After completing this lab:

1.  Check your `results/lab02_metrics.json`.
2.  Write your reflection in the notebook.
3.  Submit your work!

**Questions?** Check the [Tips & Reference Guide](lab02_guide.md) or ask your instructor.
