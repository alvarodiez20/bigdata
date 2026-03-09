import pytest
import random
from collections import Counter

from src.lab07 import bad_hash, good_hash, HyperLogLog, TDigest


# ---------------------------------------------------------------------------
# Exercise 1: Hash Function Quality
# ---------------------------------------------------------------------------

def test_hash_bad_clusters():
    """
    Tests that bad_hash produces very few distinct outputs.

    Why this is important:
    A hash that returns only a handful of distinct values makes probabilistic
    data structures useless — Bloom Filters saturate instantly and Count-Min
    Sketches degrade to a single counter.
    """
    outputs = set()
    for i in range(1000):
        outputs.add(bad_hash(f"item_{i}", 100))

    # len("item_0") to len("item_999") ranges from 6 to 8 → only ~3 distinct values
    assert len(outputs) < 10, (
        f"bad_hash should produce very few distinct outputs, got {len(outputs)}"
    )


def test_hash_good_distributes():
    """
    Tests that good_hash distributes outputs roughly uniformly.

    Why this is important:
    A well-distributed hash is the foundation of all probabilistic data structures.
    We check that 1000 items across 100 buckets fill at least 90 buckets, confirming
    near-uniform spread (birthday paradox guarantees this for a good hash).
    """
    outputs = set()
    for i in range(1000):
        outputs.add(good_hash(f"item_{i}", 100))

    assert len(outputs) >= 90, (
        f"good_hash should distribute well across buckets, got only {len(outputs)}/100"
    )


def test_hash_good_deterministic():
    """
    Tests that good_hash is deterministic — same input always yields same output.
    """
    val1 = good_hash("hello", 1000)
    val2 = good_hash("hello", 1000)
    assert val1 == val2


def test_hash_good_range():
    """
    Tests that good_hash outputs are within [0, table_size).
    """
    for i in range(500):
        h = good_hash(f"item_{i}", 37)
        assert 0 <= h < 37


# ---------------------------------------------------------------------------
# Exercise 2: HyperLogLog
# ---------------------------------------------------------------------------

def test_hyperloglog_basic():
    """
    Tests HyperLogLog cardinality estimation with 1000 distinct items.

    With p=10 (1024 registers), the standard error is ~3.25%.
    We allow a generous ±30% tolerance to account for variance.
    """
    hll = HyperLogLog(p=10)
    n = 1000
    for i in range(n):
        hll.add(f"user_{i}")

    estimate = hll.estimate()
    assert 700 < estimate < 1300, (
        f"HLL estimate for {n} distinct items was {estimate:.0f}, expected ~{n}"
    )


def test_hyperloglog_duplicates():
    """
    Tests that duplicates do not inflate the cardinality estimate.

    We add 100 distinct items, each 10 times (1000 total adds).
    The estimate should still be close to 100, not 1000.
    """
    hll = HyperLogLog(p=10)
    for i in range(100):
        for _ in range(10):
            hll.add(f"item_{i}")

    estimate = hll.estimate()
    assert 60 < estimate < 150, (
        f"HLL with 100 distinct items (1000 adds) estimated {estimate:.0f}, expected ~100"
    )


def test_hyperloglog_empty():
    """
    Tests that an empty HyperLogLog returns a small estimate (close to zero).
    """
    hll = HyperLogLog(p=8)
    estimate = hll.estimate()
    # All registers are 0, linear counting: m * ln(m/m) = 0
    # But the formula gives m * ln(m/V) where V=m, so ln(1) = 0
    assert estimate == 0.0 or estimate < 5, (
        f"Empty HLL should estimate ~0, got {estimate}"
    )


def test_hyperloglog_large():
    """
    Tests HyperLogLog with a larger cardinality (10,000 distinct items).

    With p=12 (4096 registers), the standard error is ~1.6%.
    We use ±25% tolerance.
    """
    hll = HyperLogLog(p=12)
    n = 10000
    for i in range(n):
        hll.add(f"element_{i}")

    estimate = hll.estimate()
    assert 7500 < estimate < 12500, (
        f"HLL estimate for {n} distinct items was {estimate:.0f}, expected ~{n}"
    )


def test_hyperloglog_leading_zeros():
    """
    Tests the _leading_zeros helper method.
    """
    hll = HyperLogLog()
    assert hll._leading_zeros(0b10000000, 8) == 1  # no leading zeros → returns 1
    assert hll._leading_zeros(0b00100000, 8) == 3  # 2 leading zeros → returns 3
    assert hll._leading_zeros(0b00000001, 8) == 8  # 7 leading zeros → returns 8
    assert hll._leading_zeros(0b00000000, 8) == 9  # all zeros → returns 9


# ---------------------------------------------------------------------------
# Exercise 3: T-Digest
# ---------------------------------------------------------------------------

def test_tdigest_median():
    """
    Tests that t-digest gives a reasonable median estimate for uniform data.

    We feed 10,000 values from [0, 1000) and check that the median
    estimate is within ±5% of the true median (500).
    """
    random.seed(42)
    td = TDigest(compression=100)
    n = 10000
    values = [random.uniform(0, 1000) for _ in range(n)]
    for v in values:
        td.add(v)

    median = td.quantile(0.5)
    true_median = sorted(values)[n // 2]
    assert abs(median - true_median) < 50, (
        f"T-Digest median was {median:.1f}, true median is {true_median:.1f}"
    )


def test_tdigest_extremes():
    """
    Tests that t-digest handles extreme quantiles (p1 and p99) well.

    The t-digest should be more accurate at the tails than in the center.
    """
    random.seed(123)
    td = TDigest(compression=100)
    values = [random.gauss(50, 10) for _ in range(5000)]
    for v in values:
        td.add(v)

    sorted_vals = sorted(values)
    true_p1 = sorted_vals[int(0.01 * len(sorted_vals))]
    true_p99 = sorted_vals[int(0.99 * len(sorted_vals))]

    est_p1 = td.quantile(0.01)
    est_p99 = td.quantile(0.99)

    assert abs(est_p1 - true_p1) < 5, (
        f"T-Digest p1 was {est_p1:.1f}, true p1 is {true_p1:.1f}"
    )
    assert abs(est_p99 - true_p99) < 5, (
        f"T-Digest p99 was {est_p99:.1f}, true p99 is {true_p99:.1f}"
    )


def test_tdigest_empty():
    """
    Tests edge case: quantile of an empty t-digest returns 0.0.
    """
    td = TDigest()
    assert td.quantile(0.5) == 0.0


def test_tdigest_single_value():
    """
    Tests edge case: a single value always returns that value for any quantile.
    """
    td = TDigest()
    td.add(42.0)
    assert td.quantile(0.0) == 42.0
    assert td.quantile(0.5) == 42.0
    assert td.quantile(1.0) == 42.0


def test_tdigest_merge():
    """
    Tests that merging two t-digests produces correct quantile estimates.

    We split 10,000 values across two digests, merge them,
    and check that the merged result is still accurate.
    """
    random.seed(456)
    td1 = TDigest(compression=100)
    td2 = TDigest(compression=100)

    all_values = []
    for i in range(5000):
        v = random.uniform(0, 100)
        td1.add(v)
        all_values.append(v)

    for i in range(5000):
        v = random.uniform(0, 100)
        td2.add(v)
        all_values.append(v)

    td1.merge(td2)

    median = td1.quantile(0.5)
    true_median = sorted(all_values)[len(all_values) // 2]
    assert abs(median - true_median) < 5, (
        f"Merged t-digest median was {median:.1f}, true median is {true_median:.1f}"
    )


def test_tdigest_compression_bounds():
    """
    Tests that the number of centroids stays bounded by compression.

    After adding many values, the number of centroids should not
    exceed a reasonable multiple of the compression parameter.
    """
    td = TDigest(compression=50)
    for i in range(10000):
        td.add(float(i))

    td._compress()
    assert len(td.centroids) < 50 * 10, (
        f"Expected fewer than 500 centroids with compression=50, got {len(td.centroids)}"
    )
