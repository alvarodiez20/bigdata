# Lab 06: Streaming Algorithms - Tips & Reference

This guide provides the theoretical background and implementation details needed to complete Lab 06.

In this lab, we focus on **streaming algorithms**—algorithms designed to process massive datasets that cannot fit in memory and arrive as a continuous stream of data. 

Because the data might only be seen once or is too large to store, these algorithms must be highly memory-efficient (often mapping to $O(1)$ or $O(\log N)$ memory) and fast.

## Streaming Models

Before diving into the algorithms, we need to understand how data streams are typically modeled. We classify data streams into three main categories based on how elements are updated over time:

1.  **Cash Register Model**: Data arrives as a sequence of positive increments. Think of a cash register recording transactions: you only ever add to the total, you never subtract. Wait times, network packets sent, or total likes on a post follow this model.
2.  **Turnstile Model**: Data arrives as a sequence of increments or decrements (positive or negative updates). Think of a subway turnstile: people enter (+1) and people leave (-1). The goal is to track the current state, but the net state is usually assumed to be non-negative.
3.  **Sliding Window Model**: We only care about the most recent $N$ elements or the elements that arrived in the last $T$ time units. Older elements are discarded or "expire." This is useful for analyzing recent trends, like "trending hashtags in the last hour" or "average latency over the last 1000 requests."

---

## 1. Running Min / Max

The simplest streaming algorithms track the minimum and maximum values seen so far.

**How it works:**

*   Initialize `min_val` to $+\infty$ and `max_val` to $-\infty$.
*   For each new value `x`:
    *   If `x < min_val`, update `min_val = x`.
    *   If `x > max_val`, update `max_val = x`.

**Complexity:** $O(1)$ memory, $O(1)$ time per element.
**Model:** Cash Register (insertions only).

---

## 2. Welford's Online Algorithm

Computing the mean and variance of a dataset trivially requires two passes: one to compute the mean, and another to sum the squared differences from the mean. However, in a stream, we can only see each item once.

**Welford's Algorithm** allows us to compute the mean and variance in a single pass with $O(1)$ memory, while being numerically stable (avoiding catastrophic cancellation that plagues naive one-pass variance formulas).

**How it works:**
Maintain three variables:

*   `count` ($n$): Number of elements seen so far.
*   `mean` ($\mu$): The running average.
*   `M2`: The running sum of squares of differences from the current mean.

**Formulas for updating given a new value $x$:**

1.  Increase count: $n_{new} = n + 1$
2.  Compute difference from old mean: $\delta = x - \mu$
3.  Update mean: $\mu_{new} = \mu + \frac{\delta}{n_{new}}$
4.  Compute difference from new mean: $\delta_2 = x - \mu_{new}$
5.  Update M2: $M2_{new} = M2 + \delta \times \delta_2$

**Final Statistics:**

*   Mean: $\mu$
*   Sample Variance: $\frac{M2}{n - 1}$ (for $n > 1$)
*   Population Variance: $\frac{M2}{n}$

---

## 3. Reservoir Sampling

Suppose you have a stream of unknown or extremely large length, and you want to select exactly $k$ random samples such that every element has an equal probability of being selected.

**Reservoir Sampling** solves this elegantly.

**How it works (Algorithm R):**

1.  Store the first $k$ elements in an array (the "reservoir").
2.  For the $i$-th element (where $i > k$):
    *   Generate a random integer $j$ between $1$ and $i$ (inclusive).
    *   If $j \le k$, replace the $j$-th element in the reservoir with the new $i$-th element.
    *   Otherwise, discard the $i$-th element.

**Why it works:**
By induction, at step $i$, each of the $i$ elements seen so far has exactly a $\frac{k}{i}$ probability of being in the reservoir.

---

## 4. Bloom Filter

A **Bloom Filter** is a probabilistic data structure used to test whether an element is a member of a set. It is highly space-efficient.

**Properties:**

*   **False positives are possible**: It might say an element is in the set when it isn't.
*   **False negatives are impossible**: If it says an element is *not* in the set, it is definitely not in the set.

**How it works:**

1.  **Initialization**: An array of $m$ bits, all set to 0. Additionally, $h$ independent hash functions are chosen.
2.  **Add an element**: Pass the element through all $h$ hash functions to get $h$ array indices (modulo $m$). Set the bits at all these indices to 1.
3.  **Check membership**: Pass the queried element through all $h$ hash functions.
    *   If *any* of the bits at these indices is 0, the element is **definitely not** in the set.
    *   If *all* of the bits are 1, the element is **probably** in the set (it could be a collision from other elements, hence the false positive).

*Tip for Implementation*: Since true independent hash functions are hard to come by, a common trick is to use one or two hash functions (like `hash()` and `mmh3`) and combine them linearly: `hash_i(x) = (hash1(x) + i * hash2(x)) % m`. For this lab, simply using Python's built-in `hash()` seeded differently or concatenated with the index will suffice.

---

## 5. Count-Min Sketch

A **Count-Min Sketch (CMS)** is a probabilistic data structure that serves as a frequency table of events in a stream of data. It is the streaming equivalent of a hash map / dictionary for counting, but uses fraction of the memory.

**How it works:**

1.  **Initialization**: A 2D array of counters (width $w$, depth $d$), all initialized to 0. You need $d$ independent hash functions.
2.  **Add an element**: Pass the element through the $d$ hash functions to get $d$ column indices. For each row $i$, increment the counter at `(row=i, col=hash_i(x))`.
3.  **Check frequency**: Pass the element through the $d$ hash functions. Look at the counts in the corresponding cells. The estimated frequency is the **minimum** of these counts.

**Why the minimum?**
Because collisions only cause counts to *increase*. The true count will be recorded in every cell the element hashes to. Intrusions from other elements will push the counts higher. The minimum value across the $d$ cells is the one least affected by collisions and therefore provides the tightest upper bound on the true frequency.

**Complexity:** $O(d)$ time per operation, $O(w \times d)$ space.
