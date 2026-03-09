# Lab 07: Probabilistic Data Structures & Polars — Instructions

Welcome to Lab 07! This lab has two parts:

- **Part A**: Implement probabilistic data structures (HyperLogLog, t-digest) and explore hash function quality, validated with `pytest`.
- **Part B**: Learn Polars and use PySuricata to see streaming algorithms in action.

## Additional Resources
- **[Tips & Reference Guide](lab07_guide.md)** — theoretical explanations, formulas, and Polars overview.

## Pre-flight Checklist

1. Checkout the `main` branch: `git checkout main`
2. Pull the latest changes from the repository: `git pull`
3. Create a local branch for your work: `git checkout -b <your_branch_name>`
4. Ensure you have the required dependencies updated. Run:
   ```bash
   uv sync
   ```
5. Run the test suite (all tests will fail until you implement the functions):
   ```bash
   uv run pytest tests/test_lab07.py -v
   ```

---

## Part A — Probabilistic Data Structures

You will edit **`src/lab07.py`**. Replace each `NotImplementedError("TODO: ...")` with your implementation.

### Exercise 1: Hash Function Quality (TODOs 1–2)

!!! objective
    Understand why hash function quality is critical for probabilistic data structures by comparing a deliberately bad hash against a proper one.

1. **`bad_hash(item, table_size)`** (TODO 1): Implement a deliberately poor hash function. Use `len(str(item)) % table_size`. This produces very few distinct outputs.
2. **`good_hash(item, table_size)`** (TODO 2): Implement a proper hash using `hashlib.sha256`. Convert the item to bytes, hash it, convert the digest to an integer, and take modulo `table_size`.

**Validation**: Run `uv run pytest tests/test_lab07.py -k "test_hash"`

### Exercise 2: HyperLogLog (TODOs 3–6)

!!! objective
    Estimate the number of distinct elements in a stream using $O(\log \log n)$ memory, implementing the HyperLogLog algorithm.

Implement the `HyperLogLog` class:

1. **`_hash(item)`** (TODO 3): Hash an item using `hashlib.sha256` and return an integer. Convert the hex digest to an integer with `int(digest, 16)`.
2. **`_leading_zeros(hash_val, max_bits)`** (TODO 4): Count the number of leading zeros in the binary representation of the remaining bits (after removing the bucket index bits). Start from `max_bits - 1` and count down while each bit is 0. Return at least 1.
3. **`add(item)`** (TODO 5): Hash the item, extract the bucket index from the first `p` bits, count leading zeros from the remaining bits, and update the register with the maximum value.
4. **`estimate()`** (TODO 6): Compute the cardinality estimate using the harmonic mean formula and bias correction. Apply small range correction when appropriate (see guide for formulas).

    The formulas you need:
    - Bias correction: $\alpha_m = \frac{0.7213}{1 + 1.079/m}$
    - Raw estimate: $E = \alpha_m \cdot m^2 \cdot \left(\sum_{j=0}^{m-1} 2^{-\text{registers}[j]}\right)^{-1}$
    - Small range: if $E \le 2.5 \cdot m$ and any register is 0, use $E^* = m \cdot \ln(m / V)$ where $V$ = number of zero registers.

**Validation**: Run `uv run pytest tests/test_lab07.py -k "test_hyperloglog"`

### Exercise 3: T-Digest (TODOs 7–9)

!!! objective
    Approximate streaming quantiles (median, p99, etc.) without storing all values, using the t-digest algorithm.

Implement the `TDigest` class. The `Centroid` dataclass and `_compress()` method are provided for you.

1. **`add(value)`** (TODO 7): Create a new `Centroid(mean=value, weight=1)` and append it to the internal list. If the number of centroids exceeds `max_unmerged`, call `_compress()`.
2. **`quantile(q)`** (TODO 8): Walk through the sorted centroids, accumulating weight. When the accumulated weight crosses `q * total_weight`, return the mean of the current centroid. Return the last centroid's mean if `q` is very close to 1.0.    Handle edge cases first: return 0.0 if no centroids, return the single centroid's mean if only one.
3. **`merge(other)`** (TODO 9): Merge another TDigest into this one. Extend this digest's centroids with the other's centroids, then call `_compress()`.

!!! tip "Understanding _compress()"
    The provided `_compress()` method sorts centroids by mean, then greedily merges adjacent centroids as long as their combined weight stays within the scale function limit: $\text{max\_weight}(q) = 4 \cdot \frac{n}{\delta} \cdot q \cdot (1 - q)$. This keeps tail centroids small (high precision) and center centroids large (lower precision).

**Validation**: Run `uv run pytest tests/test_lab07.py -k "test_tdigest"`

---

## Part B — Polars & PySuricata

You will edit **`src/lab07_polars.py`**. This part does **not** use pytest — you run the script directly.

### Exercise 4: Introduction to Polars (TODOs 10–13)

!!! objective
    Get hands-on experience with Polars as a modern alternative to pandas. Understand lazy vs eager evaluation using a real-world dataset with ~3 million rows.

The script automatically downloads the **NYC Yellow Taxi Trip Records** dataset (~45 MB parquet, ~3 million rows) on first run.

1. **`load_taxi_eager(path)`** (TODO 10): Use `pl.read_parquet()` to load the dataset eagerly. Notice the brief pause as all ~3M rows load.
2. **`load_taxi_lazy(path)`** (TODO 11): Use `pl.scan_parquet()` to create a LazyFrame. Notice how it returns **instantly** — no data is loaded yet.
3. **`filter_and_group(df)`** (TODO 12): Filter trips where `trip_distance > 2.0`, group by `PULocationID`, and compute the mean `fare_amount`. Use `pl.col()` expressions.
4. **`add_computed_column(df)`** (TODO 13): Add a new column `"tip_percentage"` computed as `(tip_amount / total_amount) * 100` using `with_columns`. Handle division by zero with `.fill_nan(0.0).fill_null(0.0)`.

Run the script to see the results:
```bash
uv run python src/lab07_polars.py
```

### Exercise 5: PySuricata with Polars (TODOs 14–15)

!!! objective
    Use PySuricata to generate a streaming data profile via Polars, and connect the report to the algorithms from Labs 06 and 07. With ~3 million rows, you will see PySuricata's streaming architecture in action.

1. **`generate_report(lf)`** (TODO 14): Call `pysuricata.profile(lf)` to profile the LazyFrame, then save the report as `"taxi_report.html"`. Watch PySuricata process the data in chunks — this is exactly the streaming model we studied in class.
2. **Reflection** (TODO 15): In the `STUDENT REFLECTION` section at the top of the file, answer:
    - Which streaming algorithms from Lab 06 can you identify in the PySuricata report?
    - How does PySuricata handle datasets larger than memory?
    - What advantage does using a `LazyFrame` give PySuricata compared to an eager `DataFrame`?

Run the script and open the generated report:
```bash
uv run python src/lab07_polars.py
open taxi_report.html
```

---

## What to Submit

When you are finished and `uv run pytest tests/test_lab07.py` shows **100% passing tests**, you are done with Part A!

**Before submitting**, make sure to write a short paragraph in the `STUDENT REFLECTION` section at the top of both files.

Submit **exactly**:
1. **`src/lab07.py`** — Your completed probabilistic data structures.
2. **`src/lab07_polars.py`** — Your Polars and PySuricata exercises.
3. **`taxi_report.html`** — The PySuricata report you generated.

**Do NOT submit:** Notebooks, the `__pycache__` directories, or the downloaded parquet file.

---

**Questions?** Check the [Tips & Reference Guide](lab07_guide.md) or ask your instructor.
