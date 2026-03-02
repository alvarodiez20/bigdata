import pytest
import numpy as np
from src.lab06 import RunningMinMax, WelfordStats, ReservoirSampling, BloomFilter, CountMinSketch, SlidingWindowMean

def test_min_max():
    """
    Tests the basic numerical capabilities of RunningMinMax.
    """
    rmm = RunningMinMax()
    values = [5.0, 2.0, 9.0, 1.0, 7.0]
    for v in values:
        rmm.update(v)
    assert rmm.min() == 1.0
    assert rmm.max() == 9.0

def test_min_max_edge_cases():
    """
    Tests edge cases for RunningMinMax.
    
    Why this is important:
    1. If the stream is empty, calling min/max should gracefully return float('inf') / float('-inf'),
       or handle it without crashing, establishing the state before any stream values.
    2. If all values are the same or there is only one element, min and max must equal each other.
    3. The structure must properly handle negative numbers and updates crossing zero boundaries.
    """
    # Empty case
    rmm = RunningMinMax()
    assert rmm.min() == float('inf')
    assert rmm.max() == float('-inf')
    
    # Single element
    rmm.update(-42.0)
    assert rmm.min() == -42.0
    assert rmm.max() == -42.0
    
    # Adding larger negative numbers
    rmm.update(-100.0)
    assert rmm.min() == -100.0
    assert rmm.max() == -42.0


def test_welford_stats():
    """
    Tests standard variance and mean correctness against numpy implementations.
    """
    ws = WelfordStats()
    values = [10.0, 20.0, 30.0, 40.0, 50.0]
    for v in values:
        ws.update(v)
    
    expected_mean = np.mean(values)
    expected_var_sample = np.var(values, ddof=1)
    expected_var_pop = np.var(values, ddof=0)
    expected_std_sample = np.std(values, ddof=1)
    
    assert pytest.approx(ws.mean()) == expected_mean
    assert pytest.approx(ws.variance(sample=True)) == expected_var_sample
    assert pytest.approx(ws.variance(sample=False)) == expected_var_pop
    assert pytest.approx(ws.std()) == expected_std_sample

def test_welford_stats_edge_cases():
    """
    Tests edge cases for Welford's algorithm variance implementation.
    
    Why this is important:
    1. Sample variance divides by (N-1). If N=0 or N=1, the variance calculation will 
       trigger a ZeroDivisionError if not handled properly. This test ensures the function 
       handles `count < 2` for sample variations gracefully by defaulting to 0.0.
    2. Testing with large offset values verifies Welford's advantage over naive algorithms: 
       avoiding catastrophic cancellation of floating-point arithmetic. Naive formulas
       would lose precision here.
    """
    ws = WelfordStats()
    
    # Empty state
    assert ws.mean() == 0.0
    assert ws.variance(sample=True) == 0.0
    assert ws.std(sample=True) == 0.0
    
    # N=1 state (guard against dividing by N-1=0 for sample variance)
    ws.update(5.0)
    assert ws.mean() == 5.0
    assert ws.variance(sample=True) == 0.0   # N-1 = 0, so return 0.0 by convention
    assert ws.variance(sample=False) == 0.0  # M2 = 0 after one element; 0/N = 0
    
    # Numerical stability check (Huge offset with small variance)
    ws_large = WelfordStats()
    large_values = [1e9 + 1, 1e9 + 2, 1e9 + 3]
    for v in large_values:
        ws_large.update(v)
    
    # Welford should remain highly accurate, naive approach would fail completely.
    assert pytest.approx(ws_large.variance(sample=True)) == np.var(large_values, ddof=1)


def test_reservoir():
    """
    Tests basic capacity constraints of Reservoir Sampling.
    """
    rs = ReservoirSampling(k=5)
    for i in range(100):
        rs.update(i)
    
    sample = rs.get_sample()
    assert len(sample) == 5
    for item in sample:
        assert 0 <= item < 100

def test_reservoir_edge_cases():
    """
    Tests edge cases for Reservoir sampling sizes and insertions.
    
    Why this is important:
    1. If a stream has fewer elements than k (the reservoir capacity), the reservoir 
       should simply hold all seen elements without throwing index errors or missing items.
    2. If a stream is perfectly equal to k, it should act identically to the smaller stream.
    """
    # Stream < Capacity
    rs_small = ReservoirSampling(k=10)
    for i in range(3):
        rs_small.update(i)
        
    assert len(rs_small.get_sample()) == 3
    assert rs_small.get_sample() == [0, 1, 2]
    
    # Capacity exactly equal to max size
    rs_exact = ReservoirSampling(k=3)
    for i in range(3):
        rs_exact.update(i)
        
    assert len(rs_exact.get_sample()) == 3
    assert rs_exact.get_sample() == [0, 1, 2]


def test_bloom():
    """
    Tests membership guarantees: no false negatives, and bounded false positive rate.

    Why this is important:
    1. A Bloom filter guarantees zero false negatives — if an item was added, contains()
       MUST return True. This is an absolute guarantee, not probabilistic.
    2. False positives are bounded by theory. With size=1000, k=3, n=5 items,
       the theoretical FPR is ~0.00014%, so we expect 0 false positives across 200 trials.
       Testing a single unseen string like 'cherry' is fragile (it IS probabilistic);
       testing many unseen strings gives a statistically robust bound.
    """
    bf = BloomFilter(size=1000, num_hash_functions=3)
    added_items = ["apple", "banana", "cherry", "date", "elderberry"]
    for item in added_items:
        bf.add(item)

    # No false negatives — every added item MUST be found (hard guarantee).
    for item in added_items:
        assert bf.contains(item) is True

    # False positive rate check across many unseen items.
    # Theoretical FPR ≈ 0.00014% → expect 0 FPs out of 200 trials.
    unseen = [f"unseen_item_{i}" for i in range(200)]
    fp_count = sum(1 for w in unseen if bf.contains(w))
    assert fp_count < 5, f"False positive rate too high: {fp_count}/200"

def test_bloom_edge_cases():
    """
    Tests collision mechanics and false positive bounds.
    
    Why this is important:
    1. The Bloom filter must definitively return False for items never seen (if space allows).
    2. Saturated Bloom filters degrade to returning True for everything. A tiny filter (size 2) 
       will eventually set all bits to True after very few insertions, demonstrating the 100% false-positive rate.
    """
    bf = BloomFilter(size=10, num_hash_functions=1)
    
    # Verify empty state
    assert bf.contains("never_seen") is False
    
    # Overload the filter to cause guaranteed collision (Saturation)
    for i in range(100):
        bf.add(f"item_{i}")
        
    # Since we mapped 100 items into 10 bits, bits will be completely saturated (all True)
    # Therefore, testing an unseen item will give a false positive = True.
    assert bf.contains("completely_new_item") is True


def test_count_min():
    """
    Tests the basic frequency estimation properties.
    """
    cms = CountMinSketch(width=50, depth=5)
    cms.add("apple")
    cms.add("apple")
    cms.add("banana")
    
    estimate_apple = cms.estimate("apple")
    estimate_banana = cms.estimate("banana")
    estimate_cherry = cms.estimate("cherry")
    
    assert estimate_apple >= 2
    assert estimate_banana >= 1
    assert estimate_cherry >= 0

def test_sliding_window_mean():
    """
    Tests that the mean tracks only the most recent window_size elements.
    """
    swm = SlidingWindowMean(window_size=3)
    swm.update(10.0)
    swm.update(20.0)
    swm.update(30.0)
    assert pytest.approx(swm.mean()) == 20.0  # (10+20+30)/3

    swm.update(40.0)  # 10.0 expires
    assert pytest.approx(swm.mean()) == 30.0  # (20+30+40)/3

    swm.update(50.0)  # 20.0 expires
    assert pytest.approx(swm.mean()) == 40.0  # (30+40+50)/3

    assert len(swm) == 3


def test_sliding_window_mean_edge_cases():
    """
    Tests edge cases: empty window, stream shorter than window, window of size 1.

    Why this is important:
    1. Before any updates, mean() must not crash and should return a sensible default (0.0).
    2. If fewer items arrive than the window size, the window holds all of them — it
       should NOT divide by window_size, but by the actual number of elements seen.
    3. A window of size 1 should always reflect only the latest value.
    """
    # Empty window
    swm = SlidingWindowMean(window_size=5)
    assert swm.mean() == 0.0
    assert len(swm) == 0

    # Stream shorter than window
    swm.update(10.0)
    swm.update(20.0)
    assert len(swm) == 2
    assert pytest.approx(swm.mean()) == 15.0  # (10+20)/2, NOT (10+20)/5

    # Window of size 1
    swm1 = SlidingWindowMean(window_size=1)
    swm1.update(100.0)
    assert pytest.approx(swm1.mean()) == 100.0
    swm1.update(999.0)
    assert pytest.approx(swm1.mean()) == 999.0  # previous value fully expired


def test_count_min_edge_cases():
    """
    Tests that elements never added remain 0, and that collisions correctly bound frequencies.
    
    Why this is important:
    1. Ensures empty sketch behaves correctly (returning 0 for unseen items).
    2. Demonstrates the over-counting property. If we shrink the sketch significantly, 
       we force collisions, causing the estimate to be higher than reality, 
       but *never lower*.
    """
    cms = CountMinSketch(width=50, depth=5)
    
    # Verify empty state
    assert cms.estimate("unseen") == 0
    
    # Force heavy collisions by using a tiny sketch (width=2, depth=2)
    cms_tiny = CountMinSketch(width=2, depth=2)
    for i in range(100):
        cms_tiny.add(f"item_{i}")
        
    cms_tiny.add("target_item")
    
    # "target_item" was only added once, but because of collisions, 
    # the minimum value across all hash fields will surely be > 1.
    assert cms_tiny.estimate("target_item") > 1
