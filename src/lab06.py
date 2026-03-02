"""
Lab 06: Streaming Algorithms

STUDENT NAME: [Alvaro Diez]

STUDENT REFLECTION:
(Please write a short paragraph here explaining what you learned in this lab)
"""

import math
import random
from collections import deque
from typing import List, Any

class RunningMinMax:
    """
    Maintains the running minimum and maximum of a stream of values.
    
    This class is designed for the Cash Register model where we only need 
    to track the minimum and maximum seen across the entire history.
    
    Examples:
        >>> rmm = RunningMinMax()
        >>> rmm.update(5.0)
        >>> rmm.update(2.0)
        >>> rmm.update(9.0)
        >>> rmm.min()
        2.0
        >>> rmm.max()
        9.0
    """
    def __init__(self):
        self._min = float('inf')
        self._max = float('-inf')

    def update(self, value: float) -> None:
        """
        Updates the running minimum and maximum with a new value.
        
        Args:
            value (float): The new numerical value from the stream.
        """
        raise NotImplementedError("TODO 1: Implement update logic for RunningMinMax")

    def min(self) -> float:
        """
        Returns the current running minimum.
        
        Returns:
            float: The smallest value seen so far.
        """
        raise NotImplementedError("TODO 2: Implement returning the current minimum")

    def max(self) -> float:
        """
        Returns the current running maximum.
        
        Returns:
            float: The largest value seen so far.
        """
        raise NotImplementedError("TODO 3: Implement returning the current maximum")


class WelfordStats:
    """
    Computes the running mean and variance using Welford's online algorithm.
    
    Computing variance naively in a single pass can lead to catastrophic cancellation.
    Welford's algorithm is numerically stable and computes variance in O(1) memory.
    
    Examples:
        >>> ws = WelfordStats()
        >>> ws.update(10.0)
        >>> ws.update(20.0)
        >>> ws.mean()
        15.0
        >>> ws.variance(sample=True)
        50.0
    """
    def __init__(self):
        self.count = 0
        self._mean = 0.0
        self._m2 = 0.0

    def update(self, value: float) -> None:
        """
        Updates the running mean and M2 statistic for variance calculation.
        
        Args:
            value (float): The new numerical value from the stream.
        """
        raise NotImplementedError("TODO 4: Implement Welford's update step (update mean and M2)")

    def mean(self) -> float:
        """
        Returns the current running mean.
        
        Returns:
            float: The calculated mean of all values seen so far. 0.0 if empty.
        """
        raise NotImplementedError("TODO 5: Implement returning the mean")

    def variance(self, sample: bool = True) -> float:
        """
        Returns the current running variance.
        
        Args:
            sample (bool): If True, returns sample variance (N-1). If False, returns 
                           population variance (N). Defaults to True.
                           
        Returns:
            float: The calculated variance. Returns 0.0 if count == 0.
                   Returns 0.0 if count == 1 and sample is True (N-1 = 0).
        """
        raise NotImplementedError("TODO 6: Implement variance (check count==0 first, then count==1 and sample, then divide M2)")

    def std(self, sample: bool = True) -> float:
        """
        Returns the current running standard deviation.

        Args:
            sample (bool): If True, uses sample variance. False for population.

        Returns:
            float: The calculated standard deviation.
        """
        raise NotImplementedError("TODO 7: Implement returning the standard deviation")


class ReservoirSampling:
    """
    Maintains a random sample of size `k` from a stream of unknown size.
    
    Guarantees that each element seen so far has an equal probability 
    of k / n of being in the reservoir, where n is the total elements seen.
    
    Examples:
        >>> rs = ReservoirSampling(k=3)
        >>> for i in range(10):  # Stream 10 items
        ...     rs.update(i)
        ...
        >>> len(rs.get_sample())
        3
    """
    def __init__(self, k: int):
        self.k = k
        self.reservoir = []
        self.count = 0

    def update(self, value: Any) -> None:
        """
        Processes a single item from the stream using Algorithm R.
        
        If we have seen <= k items, add to the reservoir.
        If we have seen > k items, pick a random integer j from 1 to count.
        If j <= k, replace the (j-1)-th item in the reservoir.
        
        Args:
            value (Any): The new item from the stream.
        """
        raise NotImplementedError("TODO 8: Implement Algorithm R to conditionally add the element")

    def get_sample(self) -> List[Any]:
        """
        Returns the current reservoir sample.
        
        Returns:
            List[Any]: A list of up to `k` sampled items.
        """
        return self.reservoir


class BloomFilter:
    """
    Probabilistic data structure to test whether an element is a member of a set.
    
    False positive matches are possible, but false negatives are not.
    
    Examples:
        >>> bf = BloomFilter(size=100, num_hash_functions=3)
        >>> bf.add("apple")
        >>> bf.contains("apple")
        True
        >>> bf.contains("banana")
        False
    """
    def __init__(self, size: int, num_hash_functions: int):
        self.size = size
        self.num_hash_functions = num_hash_functions
        self.bit_array = [False] * size

    def _hash(self, item: Any, i: int) -> int:
        """
        Helper method to get the `i`-th hash for an item.
        
        Args:
            item (Any): The item being hashed.
            i (int): The index of the hash function (0 to num_hash_functions - 1).
            
        Returns:
            int: An array index between 0 and size - 1.
        """
        return hash(str(item) + str(i)) % self.size

    def add(self, item: Any) -> None:
        """
        Adds an item to the Bloom Filter.
        
        Hashes the item `num_hash_functions` times and sets the corresponding bits to True.
        
        Args:
            item (Any): The item to add.
        """
        raise NotImplementedError("TODO 9: Hash the item and set corresponding bits to True")

    def contains(self, item: Any) -> bool:
        """
        Checks if the item might be in the set.
        
        Hashes the item `num_hash_functions` times. If *all* bits are True, 
        returns True. If *any* bit is False, returns False.
        
        Args:
            item (Any): The item to check.
            
        Returns:
            bool: True if the item is probably in the set, False if it is definitely not.
        """
        raise NotImplementedError("TODO 10: Hash the item and check if all corresponding bits are True")


class CountMinSketch:
    """
    Probabilistic data structure that serves as a frequency table of events in a stream.
    
    Uses hash functions to map events to frequencies, but unlike a hash table, 
    uses sub-linear space, at the expense of overcounting some events due to collisions.
    
    Examples:
        >>> cms = CountMinSketch(width=50, depth=3)
        >>> cms.add("apple")
        >>> cms.add("apple")
        >>> cms.add("banana")
        >>> cms.estimate("apple")
        2
        >>> cms.estimate("banana")
        1
    """
    def __init__(self, width: int, depth: int):
        self.width = width
        self.depth = depth
        self.table = [[0] * width for _ in range(depth)]

    def _hash(self, item: Any, i: int) -> int:
        """
        Helper method to get the hash for the `i`-th row for an item.
        
        Args:
            item (Any): The item being hashed.
            i (int): The current row/depth index (0 to depth - 1).
            
        Returns:
            int: An array index between 0 and width - 1.
        """
        return hash(str(item) + str(i)) % self.width

    def add(self, item: Any) -> None:
        """
        Records an occurrence of the item in the sketch.
        
        For each row (depth), hashes the item to find the column (width), 
        and increments the counter at that position.
        
        Args:
            item (Any): The item observed in the stream.
        """
        raise NotImplementedError("TODO 11: Increment the counter in each row based on the hash")

    def estimate(self, item: Any) -> int:
        """
        Estimates the frequency of the given item.
        
        For each row, hashes the item to find the counter value.
        The estimated frequency is the minimum of all these counter values.
        
        Args:
            item (Any): The item to query.
            
        Returns:
            int: The estimated minimum frequency.
        """
        raise NotImplementedError("TODO 12: Return the minimum count across all hash functions")


class SlidingWindowMean:
    """
    Computes the running mean over the most recent `window_size` elements.

    Unlike WelfordStats (which processes the entire history), this class only
    cares about the last `window_size` values — older elements expire and no
    longer affect the result. This is the **Sliding Window model** of streaming.

    Memory usage is O(window_size), not O(n). The window itself is bounded;
    the stream can be infinitely long.

    Examples:
        >>> swm = SlidingWindowMean(window_size=3)
        >>> swm.update(10.0)
        >>> swm.update(20.0)
        >>> swm.update(30.0)
        >>> swm.mean()
        20.0
        >>> swm.update(40.0)  # 10.0 expires
        >>> swm.mean()
        30.0
        >>> len(swm)
        3
    """
    def __init__(self, window_size: int):
        self.window_size = window_size
        self._window = deque(maxlen=window_size)
        self._sum = 0.0

    def update(self, value: float) -> None:
        """
        Adds a new value, evicting the oldest element if the window is full.

        Maintaining a running `_sum` avoids recomputing the sum from scratch
        each time. The key insight: when the window is at capacity, the leftmost
        element (`self._window[0]`) is about to be evicted by the next `append`.
        Subtract it from `_sum` first, then append and add the new value.

        Args:
            value (float): The new element from the stream.
        """
        raise NotImplementedError(
            "TODO 13: If window is full, subtract self._window[0] from _sum. "
            "Then append value to _window and add value to _sum."
        )

    def mean(self) -> float:
        """
        Returns the mean of the elements currently in the window.

        Returns:
            float: The mean. Returns 0.0 if the window is empty.
        """
        raise NotImplementedError("TODO 14: Return _sum / len(_window), or 0.0 if empty")

    def __len__(self) -> int:
        """
        Returns the number of elements currently in the window.

        Returns:
            int: Current fill level, between 0 and window_size.
        """
        raise NotImplementedError("TODO 15: Return the current number of elements in the window")
