"""
Lab 07: Probabilistic Data Structures

STUDENT NAME: [Your Name Here]

STUDENT REFLECTION:
(Please write a short paragraph here explaining what you learned in this lab)
"""

import hashlib
import math
from dataclasses import dataclass
from typing import Any, List


# ---------------------------------------------------------------------------
# Exercise 1: Hash Function Quality
# ---------------------------------------------------------------------------

def bad_hash(item: Any, table_size: int) -> int:
    """
    A deliberately poor hash function that produces very few distinct outputs.

    Uses the length of the string representation, which clusters items into
    a tiny number of buckets.

    Args:
        item (Any): The item to hash.
        table_size (int): The size of the hash table (modulus).

    Returns:
        int: A hash value between 0 and table_size - 1.

    Examples:
        >>> bad_hash("apple", 100)  # len("apple") = 5
        5
        >>> bad_hash("grape", 100)  # len("grape") = 5  — same bucket!
        5
    """
    raise NotImplementedError("TODO 1: Return len(str(item)) % table_size")


def good_hash(item: Any, table_size: int) -> int:
    """
    A proper hash function using SHA-256 for uniform distribution.

    Converts the item to its string representation, hashes it with SHA-256,
    converts the hex digest to an integer, and takes modulo table_size.

    Args:
        item (Any): The item to hash.
        table_size (int): The size of the hash table (modulus).

    Returns:
        int: A hash value between 0 and table_size - 1.

    Examples:
        >>> 0 <= good_hash("apple", 100) < 100
        True
    """
    raise NotImplementedError(
        "TODO 2: Use hashlib.sha256 to hash str(item).encode(), "
        "convert hexdigest to int, return modulo table_size"
    )


# ---------------------------------------------------------------------------
# Exercise 2: HyperLogLog
# ---------------------------------------------------------------------------

class HyperLogLog:
    """
    Estimates the cardinality (distinct count) of a stream using O(log log n) memory.

    Uses m = 2^p registers. Each register tracks the maximum number of leading
    zeros seen in the hash values assigned to that bucket.

    Args:
        p (int): Precision parameter. Number of bits used for bucket indexing.
                 Creates m = 2^p registers. Typical values: 4-16.

    Examples:
        >>> hll = HyperLogLog(p=8)
        >>> for i in range(1000):
        ...     hll.add(f"user_{i}")
        >>> estimate = hll.estimate()
        >>> 800 < estimate < 1200  # ~20% error with 256 registers
        True
    """

    def __init__(self, p: int = 8):
        self.p = p
        self.m = 1 << p          # number of registers = 2^p
        self.registers = [0] * self.m
        self._max_bits = 32 - p  # bits used for leading-zero counting

    def _hash(self, item: Any) -> int:
        """
        Hashes an item to a 32-bit unsigned integer using SHA-256.

        Takes the first 8 hex characters (32 bits) of the SHA-256 digest
        and converts them to an integer.

        Args:
            item (Any): The item to hash.

        Returns:
            int: A 32-bit unsigned integer.
        """
        raise NotImplementedError(
            "TODO 3: Hash str(item).encode() with sha256, take first 8 hex chars, "
            "convert to int with base 16"
        )

    def _leading_zeros(self, hash_val: int, max_bits: int) -> int:
        """
        Counts leading zeros in the binary representation of hash_val.

        Examines bits from position (max_bits - 1) down to 0.
        Returns the count of leading zeros + 1 (minimum return value is 1).

        Args:
            hash_val (int): The hash value (after removing bucket index bits).
            max_bits (int): Number of bits to examine.

        Returns:
            int: Number of leading zeros + 1.

        Examples:
            >>> hll = HyperLogLog()
            >>> hll._leading_zeros(0b00010000, 8)  # three leading zeros + 1 = 4
            4
            >>> hll._leading_zeros(0b10000000, 8)  # zero leading zeros + 1 = 1
            1
        """
        raise NotImplementedError(
            "TODO 4: Count leading zeros by checking bits from (max_bits-1) down to 0. "
            "Return count + 1."
        )

    def add(self, item: Any) -> None:
        """
        Adds an item to the HyperLogLog sketch.

        1. Hash the item to get a 32-bit integer.
        2. Extract the bucket index from the first p bits (right-shift by 32-p).
        3. Extract the remaining bits (mask off the top p bits).
        4. Count leading zeros in the remaining bits.
        5. Update the register: registers[bucket] = max(registers[bucket], leading_zeros).

        Args:
            item (Any): The item to add.
        """
        raise NotImplementedError(
            "TODO 5: Hash → extract bucket (first p bits) → count leading zeros "
            "in remaining bits → update register with max"
        )

    def estimate(self) -> float:
        """
        Returns the estimated cardinality.

        Steps:
        1. Compute alpha_m = 0.7213 / (1 + 1.079 / m)
        2. Compute raw estimate E = alpha_m * m^2 * (sum of 2^(-register[j]))^(-1)
        3. Small range correction: if E <= 2.5 * m and any register == 0,
           use linear counting: E* = m * ln(m / V) where V = number of zero registers.
        4. Return the corrected estimate.

        Returns:
            float: The estimated number of distinct elements.
        """
        raise NotImplementedError(
            "TODO 6: Compute alpha_m, raw estimate with harmonic mean, "
            "apply small range correction if needed"
        )


# ---------------------------------------------------------------------------
# Exercise 3: T-Digest
# ---------------------------------------------------------------------------

@dataclass
class Centroid:
    """A centroid in the t-digest, representing a cluster of values."""
    mean: float
    weight: float


class TDigest:
    """
    Estimates streaming quantiles using the t-digest algorithm.

    Maintains a compressed list of weighted centroids that approximate the
    cumulative distribution function. Centroids near the tails (q ≈ 0 or q ≈ 1)
    are kept small for high precision, while centroids near the median can be
    larger.

    Args:
        compression (float): Controls accuracy vs memory. Higher = more accurate.
                             Typical values: 50-200. Default: 100.

    Examples:
        >>> td = TDigest(compression=100)
        >>> import random
        >>> random.seed(42)
        >>> for _ in range(10000):
        ...     td.add(random.gauss(0, 1))
        >>> abs(td.quantile(0.5)) < 0.1  # median should be near 0
        True
    """

    def __init__(self, compression: float = 100.0):
        self.compression = compression
        self.centroids: List[Centroid] = []
        self.total_weight = 0.0
        self.max_unmerged = int(compression * 2)

    def add(self, value: float) -> None:
        """
        Adds a single value to the t-digest.

        Creates a new Centroid with mean=value and weight=1, appends it to
        the centroid list, and increments total_weight. If the number of
        centroids exceeds max_unmerged, triggers compression.

        Args:
            value (float): The value to add.
        """
        raise NotImplementedError(
            "TODO 7: Append a new Centroid(mean=value, weight=1), "
            "increment total_weight, compress if len > max_unmerged"
        )

    def _compress(self) -> None:
        """
        Compresses the centroid list by merging adjacent centroids.

        Sorts centroids by mean, then greedily merges adjacent centroids
        as long as their combined weight respects the scale function limit.

        This method is PROVIDED — you do not need to implement it.
        """
        if not self.centroids:
            return

        self.centroids.sort(key=lambda c: c.mean)

        merged = [self.centroids[0]]
        cumulative = self.centroids[0].weight

        for i in range(1, len(self.centroids)):
            c = self.centroids[i]
            q = cumulative / self.total_weight

            # Scale function: max weight allowed at quantile q
            max_weight = 4.0 * (self.total_weight / self.compression) * q * (1.0 - q)
            max_weight = max(max_weight, 1.0)

            if merged[-1].weight + c.weight <= max_weight:
                # Merge: weighted average of means
                total_w = merged[-1].weight + c.weight
                merged[-1].mean = (
                    merged[-1].mean * merged[-1].weight + c.mean * c.weight
                ) / total_w
                merged[-1].weight = total_w
            else:
                merged.append(Centroid(mean=c.mean, weight=c.weight))

            cumulative += c.weight

        self.centroids = merged

    def quantile(self, q: float) -> float:
        """
        Estimates the value at quantile q.

        Walks through the sorted centroids, accumulating weight. When the
        accumulated weight crosses q * total_weight, returns the mean of
        the current centroid.

        Args:
            q (float): The quantile to estimate, between 0.0 and 1.0.

        Returns:
            float: The estimated value at quantile q.

        Edge cases:
            - No centroids: return 0.0
            - Single centroid: return its mean
            - q very close to 0.0: return first centroid's mean
            - q very close to 1.0: return last centroid's mean
        """
        raise NotImplementedError(
            "TODO 8: Handle edge cases, then walk through sorted centroids "
            "accumulating weight until crossing q * total_weight"
        )

    def merge(self, other: 'TDigest') -> None:
        """
        Merges another TDigest into this one.

        Extends this digest's centroid list with the other's centroids,
        adds the other's total_weight to this one, then compresses.

        Args:
            other (TDigest): The other t-digest to merge.
        """
        raise NotImplementedError(
            "TODO 9: Extend self.centroids with other.centroids, "
            "add other.total_weight, then call _compress()"
        )
