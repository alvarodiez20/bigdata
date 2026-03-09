# Lab 07: Probabilistic Data Structures & Polars — Tips & Reference

This guide provides the theoretical background and implementation details needed to complete Lab 07.

Building on the streaming algorithms from Lab 06, we now explore **probabilistic data structures** that trade a small amount of accuracy for dramatic savings in memory and speed. We also introduce **Polars**, a modern DataFrame library built for performance.

---

## 1. Hash Function Quality

Before implementing more complex structures, it is worth understanding **why hash quality matters**. All the structures from Lab 06 (Bloom Filter, Count-Min Sketch) and this lab (HyperLogLog) depend on hashes behaving like random functions.

### What makes a good hash?

A good hash function for probabilistic data structures should have:

1. **Uniform distribution** — each output bit is equally likely to be 0 or 1.
2. **Avalanche effect** — flipping one input bit flips approximately half the output bits.
3. **Determinism** — the same input always produces the same output (within a single run, at minimum).

### Bad hashes cause problems

If a hash function clusters outputs, the probabilistic guarantees break:

- In a **Bloom Filter**, clustering increases the false positive rate because fewer distinct bits get set.
- In a **Count-Min Sketch**, poor distribution leads to more collisions and larger overestimates.
- In **HyperLogLog**, biased bit patterns lead to systematic cardinality errors.

### Example

| Hash Function | Quality | Why |
|---|---|---|
| `len(str(item)) % m` | ❌ Terrible | Only a few distinct outputs (string lengths are small integers) |
| `hash(str(item) + str(i)) % m` | ✅ Acceptable | Python's built-in `hash()` has good distribution (but unstable across runs) |
| `hashlib.sha256(...)` | ✅✅ Excellent | Cryptographic — uniform, deterministic, stable across runs |

---

## 2. HyperLogLog

**HyperLogLog (HLL)** estimates the **cardinality** (number of distinct elements) of a dataset using only $O(\log \log n)$ memory. It was introduced by Flajolet et al. (2007) and is used in production systems like Redis (`PFCOUNT`), BigQuery, and Presto.

### Intuition

Imagine flipping a fair coin repeatedly. On average, you need to flip $2^k$ coins before seeing $k$ consecutive heads. If I tell you "the longest run of leading zeros I saw was 5," you'd estimate I looked at roughly $2^5 = 32$ distinct items.

HyperLogLog formalizes this intuition with multiple buckets and a harmonic-mean correction.

### Algorithm

1. **Hash each element** to get a uniformly distributed integer.
2. **Split the hash** into two parts:
    - The first $p$ bits determine a **bucket** index $j$ (there are $m = 2^p$ buckets).
    - The remaining bits are used to count **leading zeros** (call it $\rho$, the position of the first `1` bit).
3. **Update the register**: `registers[j] = max(registers[j], ρ)`.
4. **Estimate cardinality** using the **harmonic mean** across all registers:

$$
E = \alpha_m \cdot m^2 \cdot \left( \sum_{j=1}^{m} 2^{-\text{registers}[j]} \right)^{-1}
$$

Where $\alpha_m$ is a bias correction constant:

$$
\alpha_m = \frac{0.7213}{1 + \frac{1.079}{m}}
$$

### Small and large range corrections

- **Small range correction**: If $E \le \frac{5}{2} m$ and any register is still 0, use **linear counting** instead:

$$
E^* = m \cdot \ln\left(\frac{m}{V}\right)
$$

where $V$ is the number of registers equal to zero.

- **Large range correction**: If $E > \frac{1}{30} \cdot 2^{32}$, apply:

$$
E^* = -2^{32} \cdot \ln\left(1 - \frac{E}{2^{32}}\right)
$$

### Complexity

- **Memory**: $m$ registers of ~5 bits each = $O(m)$ bits. With $m = 2048$, that's ~1.3 KB for ~2% error.
- **Time**: $O(1)$ per `add`, $O(m)$ for `estimate`.

---

## 3. T-Digest

The **t-digest** (Dunning, 2019) is a data structure for estimating **quantiles** (median, p99, etc.) from a stream of values using bounded memory. It is used in Elasticsearch, Apache Spark, and monitoring systems.

### Why not just sort?

Computing exact quantiles requires storing all data and sorting it — $O(n)$ memory. For a billion-element stream, that is not feasible. T-digest maintains a **compressed summary** of the distribution.

### Core idea: centroids

A t-digest maintains a sorted list of **centroids**, each with a `mean` and a `weight` (how many original values it represents). The centroids approximate the cumulative distribution:

- Centroids near the **tails** (near quantiles 0.0 and 1.0) are kept **small** (high resolution) because extreme quantiles require precision.
- Centroids near the **center** (around the median) can be **larger** (lower resolution) because errors there matter less.

### The scale function

The maximum weight a centroid is allowed to have depends on its position in the distribution. The **scale function** $k(q)$ maps a quantile $q \in [0, 1]$ to a scale value:

$$
k(q) = \frac{\delta}{2\pi} \cdot \arcsin(2q - 1)
$$

Where $\delta$ is the **compression parameter** (higher = more centroids = more accuracy). Two adjacent centroids at quantile $q$ can only be merged if their combined weight satisfies:

$$
w_1 + w_2 \le \delta \cdot k'(q)
$$

In practice, the simplified weight limit used in this lab is:

$$
\text{max\_weight}(q) = 4 \cdot \frac{n}{\delta} \cdot q \cdot (1 - q)
$$

This is smallest near $q = 0$ and $q = 1$ (high precision in the tails) and largest near $q = 0.5$ (allowing more merging in the center).

### Algorithm

1. **Add a value**: Insert it as a new centroid `(mean=value, weight=1)` into a buffer.
2. **Compress** (when the buffer fills): Sort all centroids by mean, then merge adjacent centroids greedily, respecting the weight limit from the scale function.
3. **Query a quantile** $q$: Walk through the sorted centroids, accumulating weight. When the accumulated weight crosses $q \cdot n$, interpolate between the surrounding centroids.

### Complexity

- **Memory**: $O(\delta)$ centroids. Typical $\delta = 100$ gives excellent accuracy.
- **Time**: $O(1)$ amortized per `add` (periodic $O(\delta \log \delta)$ compression).

---

## 4. Polars: A Modern DataFrame Library

### What is Polars?

[Polars](https://pola.rs/) is a DataFrame library written in **Rust** and designed from scratch for performance. It is not a pandas wrapper — it is a completely independent implementation.

### Why Polars?

| Feature | pandas | Polars |
|---|---|---|
| Language | Python (C extensions) | Rust (Python bindings) |
| Memory layout | Row-oriented (internally) | Columnar (Apache Arrow) |
| Execution | Eager only | Lazy + eager |
| Multi-threading | Limited (GIL) | Native multi-threading |
| Type safety | Weak | Strong |
| Streaming | No | Yes (`scan_csv`, `sink_*`) |

### Key concepts

1. **Eager vs Lazy**:
    - `pl.read_csv(...)` loads the file immediately into memory (eager, like pandas).
    - `pl.scan_csv(...)` creates a **lazy query plan**. No data is read until you call `.collect()`. This allows Polars to optimize (predicate pushdown, projection pushdown, etc.).

2. **Expressions**: Polars operations are built from **expressions** (`pl.col("age").mean()`), which can be composed and optimized.

3. **Apache Arrow**: Polars uses Arrow as its in-memory format, enabling zero-copy interop with other tools.

### Quick comparison

```python
# pandas
import pandas as pd
df = pd.read_csv("data.csv")
result = df[df["age"] > 30].groupby("city")["salary"].mean()

# polars (eager)
import polars as pl
df = pl.read_csv("data.csv")
result = df.filter(pl.col("age") > 30).group_by("city").agg(pl.col("salary").mean())

# polars (lazy — optimized)
lf = pl.scan_csv("data.csv")
result = (
    lf.filter(pl.col("age") > 30)
    .group_by("city")
    .agg(pl.col("salary").mean())
    .collect()
)
```

### Relation to Big Data

Polars' lazy evaluation model is the same idea behind Spark's query planning. Understanding `scan → transform → collect` prepares you for distributed systems where this pattern is fundamental.

---

## 5. PySuricata

[PySuricata](https://alvarodiez20.github.io/pysuricata/) is a data profiling library that generates self-contained HTML reports from DataFrames. It works with **pandas**, **polars DataFrames**, and **polars LazyFrames**.

### Under the hood

PySuricata uses the same streaming algorithms you implemented in Lab 06:

- **Welford's algorithm** for numerically stable mean, variance, skewness, and kurtosis.
- **Reservoir sampling** for representative data samples.
- **KMV (K Minimum Values) sketches** for estimating distinct counts (similar in spirit to HyperLogLog).
- **Misra-Gries** for heavy hitters (frequent items).

It processes data in **configurable chunks**, so the entire dataset never needs to fit in memory at once — exactly the streaming model we have been studying.

### Using with Polars

```python
import polars as pl
from pysuricata import profile

lf = pl.scan_parquet("yellow_tripdata_2024-01.parquet")
report = profile(lf)
report.save_html("taxi_report.html")
```

With ~3 million rows, you will notice PySuricata processing the data in multiple chunks — this is streaming in action!

!!! tip "Connecting the dots"
    When you open the generated report, look for: per-column statistics (Welford!), missing-value counts, distinct counts (KMV sketches!), and frequency bars for categorical columns (Misra-Gries!). These are the algorithms you implemented in Labs 06 and 07.

---

## 6. Implementation Hints

This section provides step-by-step guidance for each TODO in the lab. Try to solve each function on your own first — use these hints only when you are stuck.

---

### TODO 1 — `bad_hash(item, table_size)`

This is intentionally trivial. Convert the item to a string, take its length, and return the modulus:

```python
return len(str(item)) % table_size
```

Think about why this is terrible: `"apple"`, `"grape"`, and `"lemon"` all have length 5, so they all map to the same bucket.

---

### TODO 2 — `good_hash(item, table_size)`

Use `hashlib.sha256` to produce a uniformly distributed hash:

1. Convert the item to bytes: `str(item).encode()`
2. Hash it: `hashlib.sha256(...).hexdigest()`
3. Convert the hex string to an integer: `int(digest, 16)`
4. Take modulo: `% table_size`

```python
digest = hashlib.sha256(str(item).encode()).hexdigest()
return int(digest, 16) % table_size
```

---

### TODO 3 — `HyperLogLog._hash(item)`

Similar to `good_hash`, but we only need 32 bits (8 hex characters):

```python
digest = hashlib.sha256(str(item).encode()).hexdigest()
return int(digest[:8], 16)
```

Why 8 hex characters? Each hex digit is 4 bits, so 8 hex digits = 32 bits, which is the size we use for our register indexing and leading-zero counting.

---

### TODO 4 — `HyperLogLog._leading_zeros(hash_val, max_bits)`

Walk through the bits from the most significant to the least significant. Count how many are zero before you hit the first `1`:

```python
count = 0
for i in range(max_bits - 1, -1, -1):   # from MSB to LSB
    if hash_val & (1 << i):              # found a '1' bit
        break
    count += 1
return count + 1                         # minimum return is 1
```

!!! note "Why + 1?"
    We return `count + 1` because the rank $\rho$ in the HyperLogLog paper is defined as the position of the first `1` bit (1-indexed). If the first bit is already `1`, the rank is 1(zero leading zeros), not 0.

---

### TODO 5 — `HyperLogLog.add(item)`

This is the core of HyperLogLog. Break it into clear steps:

1. **Hash** the item to 32 bits.
2. **Extract the bucket** from the first `p` bits by right-shifting:
   ```python
   bucket = h >> (32 - self.p)
   ```
3. **Extract the remaining bits** by masking off the top `p` bits:
   ```python
   remaining = h & ((1 << (32 - self.p)) - 1)
   ```
4. **Count leading zeros** in the remaining bits.
5. **Update the register** with the maximum:
   ```python
   self.registers[bucket] = max(self.registers[bucket], lz)
   ```

---

### TODO 6 — `HyperLogLog.estimate()`

Follow the three-step formula:

```python
# 1. Bias correction
alpha_m = 0.7213 / (1.0 + 1.079 / self.m)

# 2. Harmonic mean (raw estimate)
z = sum(2.0 ** (-r) for r in self.registers)
e = alpha_m * self.m * self.m / z

# 3. Small range correction
if e <= 2.5 * self.m:
    v = self.registers.count(0)    # number of empty registers
    if v > 0:
        e = self.m * math.log(self.m / v)

return e
```

!!! note "Why the small range correction?"
    When most registers are still zero, the harmonic-mean formula underestimates. Linear counting (`m * ln(m/V)`) is more accurate for small cardinalities.

---

### TODO 7 — `TDigest.add(value)`

The simplest TODO in the lab — just three lines:

1. Append a new centroid: `self.centroids.append(Centroid(mean=value, weight=1))`
2. Increment the total weight: `self.total_weight += 1.0`
3. If the list is too long, compress: `if len(self.centroids) > self.max_unmerged: self._compress()`

---

### TODO 8 — `TDigest.quantile(q)`

Handle edge cases first, then walk the centroids:

```python
# Edge cases
if not self.centroids:
    return 0.0
if len(self.centroids) == 1:
    return self.centroids[0].mean

# Ensure sorted
self.centroids.sort(key=lambda c: c.mean)

# Walk and accumulate
target = q * self.total_weight
cumulative = 0.0
for centroid in self.centroids:
    cumulative += centroid.weight
    if cumulative >= target:
        return centroid.mean

# If we get here, return the last centroid's mean
return self.centroids[-1].mean
```

!!! tip "Think of it visually"
    Imagine lining up all centroids on a number line. Each centroid occupies a "width" equal to its weight. You walk along from left to right, adding up weights. When you've walked past $q \times$ total\_weight, you've found the quantile.

---

### TODO 9 — `TDigest.merge(other)`

Merging is the key feature that makes t-digest useful for distributed systems (each node builds its own digest, then they merge):

```python
self.centroids.extend(other.centroids)
self.total_weight += other.total_weight
self._compress()
```

The `_compress()` step re-sorts and re-merges all centroids, keeping memory bounded.

---

### TODO 10 — `load_taxi_eager(path)`

One-liner using Polars' eager parquet reader:

```python
return pl.read_parquet(path)
```

This reads and parses the entire parquet file (~3 million rows) into memory immediately — similar to `pandas.read_parquet()`. With a ~45 MB file, you should notice a brief pause as all the data loads.

---

### TODO 11 — `load_taxi_lazy(path)`

One-liner using Polars' lazy parquet scanner:

```python
return pl.scan_parquet(path)
```

This creates a **query plan** without reading any data. Polars will only fetch and process the data when you call `.collect()`. Notice how `scan_parquet` returns instantly — compare that to the eager `read_parquet` which takes noticeably longer.

---

### TODO 12 — `filter_and_group(df)`

Chain three Polars operations using expressions:

```python
return (
    df.filter(pl.col("trip_distance") > 2.0)
    .group_by("PULocationID")
    .agg(pl.col("fare_amount").mean())
)
```

Compare this to the pandas equivalent:
```python
df[df["trip_distance"] > 2.0].groupby("PULocationID")["fare_amount"].mean()
```

Notice how Polars uses explicit expressions (`pl.col(...)`) instead of bracket indexing. This is more verbose but allows Polars to optimize the query.

---

### TODO 13 — `add_computed_column(df)`

Use `with_columns` to create a new column from existing ones:

```python
return df.with_columns(
    (pl.col("tip_amount") / pl.col("total_amount") * 100)
    .fill_nan(0.0)
    .fill_null(0.0)
    .alias("tip_percentage")
)
```

!!! note "Handling division by zero"
    When `total_amount` is 0, the division produces `NaN` or `null`. We chain `.fill_nan(0.0).fill_null(0.0)` to handle both cases cleanly.

!!! note "`with_columns` vs `assign`"
    In pandas you would use `df.assign(tip_percentage=df["tip_amount"] / df["total_amount"] * 100)`. Polars' `with_columns` is the equivalent, but uses expressions and `.alias()` to name the result.

---

### TODO 14 — `generate_report(lf)`

Two lines — profile the data and save:

```python
report = profile(lf)
report.save_html("taxi_report.html")
```

With ~3 million rows, PySuricata will take a few seconds — watch it process the data in chunks using the same streaming algorithms from Lab 06!

---

### TODO 15 — Reflection (no code)

Open `taxi_report.html` in your browser and write your answers in the `STUDENT REFLECTION` section at the top of `src/lab07_polars.py`. Look for:

- **Welford's algorithm** → per-column mean, variance, std, skewness, kurtosis are all computed in a single streaming pass.
- **Reservoir sampling** → the "sample" section showing representative rows.
- **KMV sketches** → the "distinct count" statistic for each column.
- **Streaming chunk processing** → PySuricata never loads the full dataset at once; it processes configurable chunks. With ~3 million rows, you can actually see this happening.
- **LazyFrame advantage** → Polars can push down filters and projections, meaning PySuricata can avoid reading columns or rows it does not need.
