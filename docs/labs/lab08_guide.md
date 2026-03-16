# Lab 08: Kernel Approximation Methods — Tips & Reference Guide

## Background: The Kernel Scalability Problem

A kernel function $K: \mathcal{X} \times \mathcal{X} \to \mathbb{R}$ measures similarity between two points. The most popular is the **Gaussian (RBF) kernel**:

$$K(x, y) = \exp\!\left(-\frac{\|x - y\|^2}{2\sigma^2}\right)$$

For a dataset of $n$ points in $\mathbb{R}^d$:

| Resource | Cost |
|---|---|
| Memory (Gram matrix) | $O(n^2)$ floats |
| Kernel matrix construction | $O(n^2 d)$ |
| Kernel SVM training | $O(n^2)$ to $O(n^3)$ |

At $n = 100{,}000$ the Gram matrix alone requires **~80 GB** of RAM. This is the "kernel wall".

---

## Exercise 1: Exact RBF Kernel

### Vectorised distance computation

Avoid a Python double loop. Use the identity:

$$\|x - y\|^2 = \|x\|^2 + \|y\|^2 - 2\, x^\top y$$

In numpy:
```python
sq_norms = np.einsum("ij,ij->i", X, X)     # shape (n,)
D2 = sq_norms[:, None] + sq_norms[None, :] - 2.0 * (X @ X.T)
D2 = np.maximum(D2, 0.0)                    # clamp floating-point negatives
K = np.exp(-D2 / (2 * sigma**2))
```

This is $O(n^2 d)$ but avoids Python overhead — still quadratic, just fast constant.

---

## Exercise 2: Random Fourier Features

### Theory (Rahimi & Recht, NeurIPS 2007)

By **Bochner's theorem**, any shift-invariant kernel $K(x, y) = k(x - y)$ is the Fourier transform of a non-negative measure $p(\omega)$:

$$K(x, y) = \int p(\omega)\, e^{i\omega^\top(x-y)}\, d\omega$$

For the RBF kernel, $p(\omega) = \mathcal{N}(0,\, I/\sigma^2)$.

Drawing $D$ frequencies $\omega_j \sim p(\omega)$ and phases $b_j \sim \text{Uniform}[0, 2\pi]$, the feature map:

$$z(x) = \sqrt{\frac{2}{D}} \begin{bmatrix} \cos(\omega_1^\top x + b_1) \\ \vdots \\ \cos(\omega_D^\top x + b_D) \end{bmatrix} \in \mathbb{R}^D$$

satisfies $\mathbb{E}[z(x)^\top z(y)] = K(x, y)$. The approximation error is $O(1/\sqrt{D})$.

### Memory comparison

| Method | Matrix shape | Memory |
|---|---|---|
| Exact | $n \times n$ | $O(n^2)$ |
| RFF feature map | $n \times D$ | $O(nD)$ |
| RFF kernel approx | computed on-the-fly | — |

For $n = 10{,}000$, $D = 1{,}000$: exact = 800 MB, RFF features = 80 MB.

### Implementation hint (TODO 3)

```python
omega = rng.standard_normal((D, d)) / sigma   # N(0, I/σ²) each row
b = rng.uniform(0, 2 * np.pi, size=D)
```

### Implementation hint (TODO 4)

```python
D = omega.shape[0]
Z = np.sqrt(2.0 / D) * np.cos(X @ omega.T + b)  # (n, D)
```

Note: `X @ omega.T` has shape `(n, D)`, `b` broadcasts over rows.

---

## Exercise 3: Orthogonal Random Features

### Theory (Yu et al., ICML 2016)

With independent RFF samples, two frequency vectors $\omega_i, \omega_j$ may nearly align, wasting representational capacity. ORF fixes this by **forcing orthogonality**.

Construction:
1. Draw $G \in \mathbb{R}^{d \times d}$ with $G_{ij} \sim \mathcal{N}(0,1)$.
2. Compute QR: $G = QR$ — columns of $Q$ are orthonormal.
3. The rows of $Q^\top$ are orthonormal but have unit norm. We need rows with the same norm distribution as $\mathcal{N}(0, I_d)$ samples, i.e. $\chi_d$-distributed norms.
4. Rescale: draw $s_j = \|g_j\|$ for independent $g_j \sim \mathcal{N}(0, I_d)$, set $\tilde{\omega}_j = s_j \cdot Q^\top_j$.
5. Divide by $\sigma$.

Repeat for $\lceil D/d \rceil$ blocks and trim to $D$ rows.

**Result**: ORF has the same mean (unbiased) as RFF but strictly lower variance for the same $D$.

### Implementation hint (TODO 6)

```python
import math
n_blocks = math.ceil(D / d)
rows = []
for _ in range(n_blocks):
    G = rng.standard_normal((d, d))
    Q, _ = np.linalg.qr(G)          # Q: (d, d) with orthonormal columns
    Q = Q.T                           # rows are orthonormal
    # chi_d norms: draw d-dim Gaussian, take row norms
    norms = np.linalg.norm(rng.standard_normal((d, d)), axis=1)
    rows.append(Q * norms[:, None])
omega = np.vstack(rows)[:D] / sigma
```

---

## Exercise 4: Nyström Approximation

### Theory

Given $m \ll n$ landmark points $\{u_1, \ldots, u_m\} \subset \mathcal{X}$, define:

- $K_{mm} \in \mathbb{R}^{m \times m}$: kernel between landmarks
- $K_{nm} \in \mathbb{R}^{n \times m}$: kernel between all points and landmarks

The Nyström approximation:

$$\tilde{K}_{nn} = K_{nm}\, K_{mm}^{-1}\, K_{mn} = Z Z^\top$$

where $Z = K_{nm}\, K_{mm}^{-1/2}$ is the Nyström feature map.

**Why it works**: If the kernel operator has rapidly decaying eigenvalues (common in practice), $m$ landmarks can capture most of the variance.

### Stable computation of $K_{mm}^{-1/2}$

Use eigendecomposition (more numerically stable than Cholesky for near-singular matrices):

```python
from scipy.linalg import eigh
lam, V = eigh(K_mm)                            # ascending eigenvalues
lam = np.maximum(lam, 1e-10)                   # clip near-zero values
K_mm_inv_sqrt = V @ np.diag(1.0 / np.sqrt(lam)) @ V.T
Z = K_nm @ K_mm_inv_sqrt                       # (n, m)
```

`scipy.linalg.eigh` exploits symmetry and is more stable than `np.linalg.eig`.

### Landmark selection

- **Random**: cheap, works well in practice.
- **k-means centroids**: better coverage but O(n·m·iter) cost (not required here).
- **Leverage score sampling**: theoretically optimal but complex.

---

## Exercise 5: Kernel Ridge Regression

### The dual formulation

Instead of solving in the primal $\mathbb{R}^d$, kernel ridge regression works in the dual:

$$\min_\alpha \|K\alpha - y\|^2 + \lambda \alpha^\top K \alpha$$

Solution: $\alpha^* = (K + \lambda I)^{-1} y$

Predictions: $\hat{y}(x^*) = \sum_i \alpha_i K(x^*, x_i) = k_{*}^\top \alpha^*$

### Why not invert the matrix?

`np.linalg.solve(A, b)` is:
- Faster (LU decomposition, $O(n^3/3)$)
- More numerically stable than explicit inversion
- The standard way to solve $Ax = b$

### Using approximate kernels

The feature map $Z \in \mathbb{R}^{n \times D}$ gives an approximate kernel $\tilde{K} = ZZ^\top$. You can plug this directly into kernel ridge regression. For prediction:

$$\hat{y}(x^*) = z(x^*)^\top Z^\top \alpha^*$$

This reduces cost from $O(n^2)$ to $O(nD)$ at training time.

---

## Exercise 6: Benchmarking

### Memory measurement

```python
import psutil
mem_before = psutil.Process().memory_info().rss / 1024**2  # MB
# ... build matrix ...
mem_after = psutil.Process().memory_info().rss / 1024**2
delta_mb = mem_after - mem_before
```

Note: RSS (Resident Set Size) can be noisy. Taking the max of the delta and the matrix's `nbytes` gives a conservative lower bound.

### Expected observations

| n | Exact time | RFF/ORF time | Nyström time |
|---|---|---|---|
| 500 | < 0.1s | < 0.01s | ~0.01s |
| 2000 | ~0.5s | < 0.05s | ~0.1s |
| 5000 | fails / slow | < 0.1s | ~0.5s |
| 20000 | OOM | < 0.3s | ~5s |

The crossover point (where approximate methods become necessary) is typically around $n = 5{,}000$–$10{,}000$ on a laptop.

---

## Choosing σ: The Median Heuristic

A common practical choice is the **median heuristic**: set $\sigma = \sqrt{\text{median}(\|x_i - x_j\|^2) / 2}$.

For $n$ points drawn from $\mathcal{N}(0, I_d)$, the expected squared distance is $\mathbb{E}[\|x-y\|^2] = 2d$, so a reasonable default is $\sigma \approx \sqrt{d}$.

With $\sigma = 1$ and $d = 10$, most off-diagonal kernel values are $\approx e^{-10} \approx 0$ — the kernel is trivially diagonal and any "approximation" is vacuously good. Always match $\sigma$ to your data's scale.

---

## Common Pitfalls

| Problem | Cause | Fix |
|---|---|---|
| Kernel matrix has negative eigenvalues | Floating point error in D² | `np.maximum(D2, 0)` before `exp` |
| Nyström features are NaN | Zero eigenvalue in $K_{mm}$ | Clip eigenvalues to `max(λ, 1e-10)` |
| RFF approximation has high variance | D too small | Increase D; try ORF |
| `np.linalg.solve` raises `LinAlgError` | $K + \lambda I$ near-singular | Increase λ |
| Broadcasting error in feature map | Shape mismatch `(n, D)` vs `(D,)` | Check `omega.T` vs `omega` |

---

## Complexity Summary

| Method | Feature map | Kernel approx | Memory |
|---|---|---|---|
| Exact | — | $O(n^2 d)$ | $O(n^2)$ |
| RFF | $O(nDd)$ | $O(nD)$ | $O(nD)$ |
| ORF | $O(nDd + D^2)$ | $O(nD)$ | $O(nD)$ |
| Nyström | $O(nm d + m^3)$ | $O(nm)$ | $O(nm)$ |

All approximation methods break the $O(n^2)$ wall. The choice between them depends on:

- **RFF**: simple, well-understood, good for large $d$
- **ORF**: same cost as RFF, lower variance, preferred when $D \approx d$
- **Nyström**: adapts to data geometry, excellent when kernel has fast eigenvalue decay; requires $O(m^3)$ one-time factorisation

---

## Key References

1. Rahimi, A. & Recht, B. (2007). *Random Features for Large-Scale Kernel Machines*. NeurIPS.
2. Yu, F. X., Suresh, A. T., Choromanski, K., Holtmann-Rice, D., & Kumar, S. (2016). *Orthogonal Random Features*. NeurIPS.
3. Williams, C. & Seeger, M. (2001). *Using the Nyström Method to Speed Up Kernel Machines*. NeurIPS.
