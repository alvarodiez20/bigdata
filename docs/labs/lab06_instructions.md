# Lab 06: Streaming Algorithms - Instructions

Welcome to Lab 06! In this lab, we are leaving Jupyter Notebooks behind and writing our algorithms directly in standard Python files (`.py`), ensuring code quality by validating them with `pytest`.

We will implement fundamental algorithms that process data streams using limited (`O(1)` or logarithmic) memory. 

## Additional Resources
- **[Tips & Reference Guide](lab06_guide.md)** - theoretical explanations and mathematical details for the algorithms.

## Pre-flight Checklist

1. Checkout the `main` branch: `git checkout main`
2. Pull the latest changes from the repository: `git pull`
3. Create a local branch for your work: `git checkout -b <your_branch_name>`
4. Ensure you have the required dependencies updated. Run:
   ```bash
   uv sync
   ```
5. Familiarize yourself with running `pytest`. You can run tests for this lab by executing:
   ```bash
   uv run pytest tests/test_lab06.py -v
   ```
   (Wait until you implement some functions, otherwise all tests will fail!)

---

## Lab Steps

Unlike previous labs, you will be editing a Python file: `src/lab06.py`.
Your mission is to replace the `NotImplementedError("TODO: ...")` in the classes with working implementations.

### Exercise 1: Running Min/Max and Welford

!!! objective
    Implement basic streaming statistics that maintain properties of a stream using $O(1)$ memory. Welford's algorithm is a robust way to avoid numerical instability when computing variance in a single pass.

1. **`RunningMinMax`**: Open `src/lab06.py` and implement the `update(value)`, `min()`, and `max()` methods. (TODOs 1–3)
2. **`WelfordStats`**: Implement Welford's algorithm for online mean and variance. The `update(value)` method must do the $O(1)$ update step. Ensure `mean()`, `variance()`, and `std()` return the correct values, and handle the edge cases for empty or $n=1$ sequences. (TODOs 4–7)

    For `variance()`, structure your guard clauses in this order:
    1. If `count == 0`, return `0.0`.
    2. If `count == 1` **and** `sample is True`, return `0.0` (N-1 = 0 is undefined).
    3. Otherwise divide `M2` by `(count - 1)` or `count`.

**Validation**: Run `uv run pytest tests/test_lab06.py -k "test_welford or test_min_max"`

### Exercise 2: Reservoir Sampling

!!! objective
    Extract a fair random sample of size $k$ from a stream of unknown length without loading the stream into memory.

Implement the `ReservoirSampling` class in `src/lab06.py`. (TODO 8)

- **`update(value)`**: Process exactly one item from the stream.
- Ensure that the first $k$ items just fill the reservoir. After $k$ items, randomly decide whether to swap the incoming item with an existing member of the reservoir based on the formulas seen in the guide.
- **Tip**: You can use the `random` module (e.g., `random.randint()`) to pick probabilities or indices.

**Validation**: Run `uv run pytest tests/test_lab06.py -k "test_reservoir"`

### Exercise 3: Bloom Filter

!!! objective
    Develop a fast, probabilistic data structure that can check for set membership using minimal memory space, at the cost of a small false positive rate.

Implement the `BloomFilter` class. (TODOs 9–10)

- The constructor provides `size` (number of bits) and `num_hash_functions`.
- A simple `list` of booleans (or integers `0`/`1`) is sufficient for the bit array representation in Python.
- Implement a method to hash an item multiple times. A great trick is using `hash()` combined with the hash index. For example, `(hash(str(item) + str(i))) % self.size`.
- **`add(item)`**: Hash the item $h$ times and set the respective bits to `True`.
- **`contains(item)`**: Hash the item $h$ times. If *all* bits are `True`, return `True`. If *any* bit is `False`, return `False`.

!!! warning "Python's `hash()` is randomized"
    Python randomizes `hash()` at startup. Tests here check structural properties
    (membership bounds, saturation), not specific bit positions — so this is fine for
    the lab. See the [guide](lab06_guide.md) for why production code needs a stable hash.

**Validation**: Run `uv run pytest tests/test_lab06.py -k "test_bloom"`

### Exercise 4: Count-Min Sketch

!!! objective
    Count the frequency of items in a stream without storing every unique key in a hash map, preventing memory from unbounded growth.

Implement the `CountMinSketch` class. (TODOs 11–12)

- The constructor takes `width` (columns) and `depth` (rows/hash functions).
- Initialize a 2D array of counters (lists of lists, initialized to 0).
- **`add(item)`**: Generate `depth` hash values (similar to the Bloom Filter, but modulo `width`). Increment the counter in each corresponding row.
- **`estimate(item)`**: Hash the item similarly and find the **minimum** value among the corresponding counters across the rows.

**Validation**: Run `uv run pytest tests/test_lab06.py -k "test_count_min"`

### Exercise 5: Sliding Window Mean

!!! objective
    Implement a streaming mean that only considers the most recent $w$ elements. Older data expires automatically.

Implement the `SlidingWindowMean` class in `src/lab06.py`. (TODOs 13–15)

- The constructor takes `window_size` and initialises a `deque` with `maxlen=window_size` and a running `_sum`.
- **`update(value)`**: Maintain the running sum efficiently:
    - If the window is **full** (`len(self._window) == self.window_size`), subtract `self._window[0]` from `_sum` **before** appending — that element is about to be evicted.
    - Append the new value (the `deque` evicts automatically), then add it to `_sum`.
- **`mean()`**: Return `_sum / len(_window)`, or `0.0` if the window is empty.
- **`__len__()`**: Return `len(self._window)`.

**Validation**: Run `uv run pytest tests/test_lab06.py -k "test_sliding_window"`

---

## What to Submit

When you are finished and `uv run pytest tests/test_lab06.py` shows **100% passing tests**, you are done!

**Before submitting**, make sure to write a short paragraph in the `STUDENT REFLECTION` section at the top of your `src/lab06.py` file, explaining what you learned in this lab.

Submit **exactly**:
1. **`src/lab06.py`** — Your completed Python file (including your reflection comments).

**Do NOT submit:** Notebooks or the `__pycache__` directories.

---

**Questions?** Check the [Tips & Reference Guide](lab06_guide.md) or ask your instructor.
