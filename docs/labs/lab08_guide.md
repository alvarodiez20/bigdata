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

The naive way to build the Gram matrix is a Python double loop — fine for n=10, catastrophic for n=10,000. The key insight is to expand the squared distance:

$$\|x - y\|^2 = \|x\|^2 + \|y\|^2 - 2\, x^\top y$$

This lets you compute all $n^2$ pairwise distances at once with three numpy operations: row norms, row norms transposed, and a matrix product.

### Implementation hints

**TODO 1 — `rbf_kernel(x, y, sigma)`**

Compute the difference vector `diff = x - y`, then get its squared length as `np.dot(diff, diff)` — this is $\|x-y\|^2$. Plug that directly into the formula:

$$K = \exp\!\left(-\frac{\|x-y\|^2}{2\sigma^2}\right)$$

The result is a Python `float`. Note that `rbf_kernel(x, x, sigma)` must return exactly `1.0` since the distance is zero.

**TODO 2 — `rbf_kernel_matrix(X, sigma)`**

A Python double loop (`for i ... for j ...`) works but is far too slow for large $n$. Use the squared-distance identity instead:

$$\|x_i - x_j\|^2 = \|x_i\|^2 + \|x_j\|^2 - 2\, x_i^\top x_j$$

In numpy this translates to three steps:

1. **Row norms**: compute a shape `(n,)` vector where entry $i$ is $\|x_i\|^2$.

    ```python
    sq_norms = (X ** 2).sum(axis=1)
    ```
    This squares every element of `X` and then sums along `axis=1` (across columns), leaving one value per row — exactly $\sum_j X_{ij}^2 = \|x_i\|^2$.

2. **Distance matrix**: combine with broadcasting and a matrix product:
   ```
   D2 = sq_norms[:, None] + sq_norms[None, :] - 2.0 * (X @ X.T)
   ```
   `sq_norms[:, None]` is shape `(n, 1)` and `sq_norms[None, :]` is `(1, n)` — numpy broadcasts them into an `(n, n)` matrix. `X @ X.T` is the Gram matrix of inner products, also `(n, n)`.

3. **Clamp and exponentiate**: floating point arithmetic can produce small negative values (e.g. `-1e-14`) on the diagonal. Clamp first: `D2 = np.maximum(D2, 0.0)`, then `K = np.exp(-D2 / (2 * sigma**2))`.

For the memory warning, use `warnings.warn("...", ResourceWarning, stacklevel=2)` so the warning points at the caller's line, not inside the function.

---

## Exercise 2: Random Fourier Features

### Theory (Rahimi & Recht, NeurIPS 2007)

By **Bochner's theorem**, any shift-invariant kernel $K(x, y) = k(x - y)$ is the Fourier transform of a non-negative spectral measure $p(\omega)$:

$$K(x, y) = \int p(\omega)\, e^{i\omega^\top(x-y)}\, d\omega$$

For the RBF kernel, the spectral density is $p(\omega) = \mathcal{N}(0,\, I/\sigma^2)$.

Drawing $D$ frequencies $\omega_j \sim p(\omega)$ and phases $b_j \sim \text{Uniform}[0, 2\pi]$, the **feature map** is:

$$z(x) = \sqrt{\frac{2}{D}} \begin{bmatrix} \cos(\omega_1^\top x + b_1) \\ \vdots \\ \cos(\omega_D^\top x + b_D) \end{bmatrix} \in \mathbb{R}^D$$

This satisfies $\mathbb{E}[z(x)^\top z(y)] = K(x, y)$ — a dot product replaces an expensive kernel evaluation. The approximation error converges as $O(1/\sqrt{D})$.

### Memory comparison

| Method | Matrix shape | Memory |
|---|---|---|
| Exact | $n \times n$ | $O(n^2)$ |
| RFF feature map | $n \times D$ | $O(nD)$ |

For $n = 10{,}000$ and $D = 1{,}000$: exact = 800 MB, RFF = 80 MB.

### Implementation hints

**TODO 3 — `sample_rff_weights(D, d, sigma, rng)`**

You need to return two arrays:

- `omega`, shape `(D, d)`: each of the $D$ rows is a frequency vector drawn from $\mathcal{N}(0, I/\sigma^2)$. Start with `rng.standard_normal((D, d))` which gives you $\mathcal{N}(0, 1)$ entries, then divide the whole matrix by `sigma` to rescale the variance. Dividing by `sigma` (not `sigma**2`) is correct because the standard deviation of $\mathcal{N}(0, 1/\sigma^2)$ is $1/\sigma$.
- `b`, shape `(D,)`: each phase shift is drawn uniformly from $[0, 2\pi]$. Use `rng.uniform(0, 2 * np.pi, size=D)`.

Return them as a tuple `(omega, b)`.

**TODO 4 — `rff_features(X, omega, b)`**

Build the feature matrix $Z$ of shape `(n, D)` in three sub-steps:

1. Compute the projection `X @ omega.T` — `X` is `(n, d)` and `omega.T` is `(d, D)`, so the result is `(n, D)`. Entry `[i, j]` is $\omega_j^\top x_i$.
2. Add `b` (shape `(D,)`). NumPy broadcasts it across all `n` rows, adding $b_j$ to every entry in column `j`. The result is still `(n, D)`.
3. Apply `np.cos` element-wise, then multiply by $\sqrt{2/D}$. Get `D` from `omega.shape[0]`.

The $\sqrt{2/D}$ factor is essential — without it the dot product $Z_X Z_Y^\top$ would overestimate the kernel by a factor of $D/2$.

**TODO 5 — `rff_kernel_approx(X, Y, omega, b)`**

Call `rff_features` twice — once for `X` and once for `Y` — using the **same** `omega` and `b` both times. This is important: both datasets must be projected into the same random feature space for the dot product to approximate $K(X, Y)$.

- `Z_X = rff_features(X, omega, b)` → shape `(n, D)`
- `Z_Y = rff_features(Y, omega, b)` → shape `(m, D)`
- Return `Z_X @ Z_Y.T` → shape `(n, m)`, where entry `[i, j]` approximates $K(x_i, y_j)$.

---

## Exercise 3: Orthogonal Random Features

### Theory (Yu et al., ICML 2016)

With independent RFF samples, two frequency vectors $\omega_i, \omega_j$ may nearly align, wasting representational capacity. ORF forces the weight rows to be **mutually orthogonal**, reducing approximation variance without changing the mean.

The construction works in blocks of size $d$:

1. Draw $G \in \mathbb{R}^{d \times d}$ with i.i.d. $\mathcal{N}(0,1)$ entries.
2. QR-decompose $G = QR$. The rows of $Q^\top$ are orthonormal.
3. Orthonormal rows have unit norm, but we need the correct $\chi_d$-distributed norms. Fix this by multiplying each row by an independent $\|g_j\|$ where $g_j \sim \mathcal{N}(0, I_d)$.
4. Divide by $\sigma$.
5. Stack $\lceil D/d \rceil$ such blocks and trim to $D$ rows.

**Result**: ORF has the same expected value as RFF but strictly lower variance.

### Implementation hints

**TODO 6 — `sample_orf_weights(D, d, sigma, rng)`**

The goal is to build `omega` of shape `(D, d)` where the rows are orthogonal to each other (unlike RFF where they are independent). Because you can only have at most `d` mutually orthogonal vectors in $\mathbb{R}^d$, you build the matrix in **blocks** of `d` rows, each block independently orthogonal within itself.

**Step 1 — decide how many blocks you need.**

```python
import math
n_blocks = math.ceil(D / d)
blocks = []
```

If `D=10` and `d=4`, you need 3 blocks (3×4=12 rows, trimmed to 10 later).

**Step 2 — build one block (repeat this in a loop `n_blocks` times).**

*2a. Draw a square Gaussian matrix:*
```python
G = rng.standard_normal((d, d))   # shape (d, d)
```

*2b. QR-decompose it:*
```python
Q, _ = np.linalg.qr(G)            # Q shape (d, d), orthonormal columns
Q = Q.T                            # transpose → orthonormal rows
```
After the transpose, each **row** of `Q` is a unit vector and all rows are mutually perpendicular. This is our set of orthogonal directions.

*2c. Rescale rows to have the right length distribution.*
Orthonormal rows all have norm 1, but in RFF each frequency row has a random norm drawn from the $\chi_d$ distribution (the distribution of $\|g\|$ where $g \sim \mathcal{N}(0, I_d)$). To match this, draw a fresh Gaussian matrix and take its row norms:
```python
S     = rng.standard_normal((d, d))        # independent fresh draw
norms = np.linalg.norm(S, axis=1)          # shape (d,) — one chi_d sample per row
Q     = Q * norms[:, None]                 # scale each row; [:, None] makes norms a column vector
```
`norms[:, None]` has shape `(d, 1)`, so multiplying by `Q` (shape `(d, d)`) scales row `i` of `Q` by `norms[i]`.

*2d. Append to the list:*
```python
blocks.append(Q)
```

**Step 3 — assemble, trim, and scale.**

```python
omega = np.vstack(blocks)   # shape (n_blocks * d, d)
omega = omega[:D]            # trim to exactly (D, d)
omega = omega / sigma        # rescale to N(0, I/σ²)
```

**Step 4 — sample the phase shifts** (identical to RFF):
```python
b = rng.uniform(0, 2 * np.pi, size=D)
```

Return `(omega, b)`.

**TODO 7 — `orf_features(X, omega, b)`**

The feature formula is exactly the same as RFF. Call `rff_features(X, omega, b)` and return the result. The lower variance comes from the orthogonal `omega`, not from a different formula.

---

## Exercise 4: Nyström Approximation

### Theory

Given $m \ll n$ landmark points $\{u_1, \ldots, u_m\}$, define:

- $K_{mm} \in \mathbb{R}^{m \times m}$: kernel matrix between landmarks
- $K_{nm} \in \mathbb{R}^{n \times m}$: kernel between all $n$ points and the $m$ landmarks

The Nyström approximation:

$$\tilde{K}_{nn} = K_{nm}\, K_{mm}^{-1}\, K_{mn} = Z Z^\top \qquad \text{where } Z = K_{nm}\, K_{mm}^{-1/2}$$

**Why it works**: If the kernel's eigenvalues decay quickly (common with smooth kernels), $m$ landmarks capture most of the variance. Unlike RFF/ORF, Nyström adapts to the actual data geometry.

### Stable computation of $K_{mm}^{-1/2}$

Direct inversion is numerically fragile. Instead, eigendecompose $K_{mm} = V \Lambda V^\top$ and compute:

$$K_{mm}^{-1/2} = V \Lambda^{-1/2} V^\top$$

Use `scipy.linalg.eigh` (not `np.linalg.eig`) — it exploits symmetry and returns real eigenvalues in ascending order.

### Implementation hints

**TODO 8 — `select_landmarks(X, m, strategy, rng)`**

Get `n = X.shape[0]` first, then branch on `strategy`:

- `"random"`: pick `m` unique row indices with `rng.choice(n, m, replace=False)`, then return `X[idx]`. The `replace=False` ensures no row is selected twice.
- `"stride"`: pick `m` evenly spaced indices with `np.linspace(0, n-1, m, dtype=int)`, then return `X[idx]`. This spreads landmarks uniformly regardless of how the data is ordered.
- Anything else: `raise ValueError(f"Unknown strategy '{strategy}'...")`.

**TODO 9 — `nystrom_features(X, landmarks, sigma)`**

The goal is to return `Z = K_nm @ K_mm^{-1/2}` with shape `(n, m)`. Here is each step with the exact numpy calls:

*Step 1 — landmark kernel matrix, shape (m, m):*
```python
K_mm = rbf_kernel_matrix(landmarks, sigma)
```

*Step 2 — cross kernel matrix between X and landmarks, shape (n, m):*

Use the same distance identity as TODO 2, but now between two *different* matrices:
```python
sq_X  = (X ** 2).sum(axis=1)           # shape (n,)
sq_L  = (landmarks ** 2).sum(axis=1)   # shape (m,)
D2    = sq_X[:, None] + sq_L[None, :] - 2 * (X @ landmarks.T)  # shape (n, m)
K_nm  = np.exp(-np.maximum(D2, 0.0) / (2 * sigma ** 2))
```
Note: `X @ landmarks.T` is `(n, d) @ (d, m) = (n, m)` — all inner products at once.

*Step 3 — eigendecompose K_mm:*
```python
lam, V = eigh(K_mm)    # lam shape (m,), V shape (m, m)
```
`eigh` is for symmetric matrices and always returns real eigenvalues in ascending order.

*Step 4 — clip eigenvalues and form K_mm^{-1/2}:*
```python
lam           = np.maximum(lam, 1e-10)                      # avoid division by zero
K_mm_inv_sqrt = V @ np.diag(1.0 / np.sqrt(lam)) @ V.T      # shape (m, m)
```
`np.diag(1.0 / np.sqrt(lam))` builds a diagonal matrix from the reciprocal square roots of the eigenvalues.

*Step 5 — return the feature matrix:*
```python
return K_nm @ K_mm_inv_sqrt    # shape (n, m)
```

**TODO 10 — `nystrom_kernel_approx(X, Y, landmarks, sigma)`**

Identical pattern to `rff_kernel_approx` (TODO 5): compute Nyström features for both `X` and `Y` using the **same** `landmarks`, then return their dot product:

- `Z_X = nystrom_features(X, landmarks, sigma)` → shape `(n, m)`
- `Z_Y = nystrom_features(Y, landmarks, sigma)` → shape `(m_pts, m)`
- Return `Z_X @ Z_Y.T` → shape `(n, m_pts)`

Using the same landmarks for both is essential — it puts `X` and `Y` into the same feature space so the dot product approximates $K(X, Y)$.

---

## Exercise 5: Kernel Ridge Regression

### The dual formulation

Kernel ridge regression solves in the *dual* space of coefficients $\alpha \in \mathbb{R}^n$:

$$\min_\alpha \|K\alpha - y\|^2 + \lambda \alpha^\top K \alpha \quad \Longrightarrow \quad \alpha^* = (K + \lambda I)^{-1} y$$

Predictions for new points: $\hat{y}(x^*) = k_*^\top \alpha^*$ where $k_* = [K(x^*, x_i)]_i$.

The same solver works whether `K` is exact or approximate — that's the power of the dual formulation.

### Why not invert the matrix?

`np.linalg.solve(A, b)` solves $Ax = b$ without computing $A^{-1}$ explicitly. It is faster ($O(n^3/3)$ vs $O(n^3)$) and more numerically stable. Always prefer `solve` over `inv @ b`.

### Implementation hints

**TODO 11 — `kernel_ridge_fit(K_train, y, lam)`**

You need to find `alpha` such that `(K_train + lam * I) @ alpha = y`. Two steps:

1. Build the regularised matrix — get `n = K_train.shape[0]`, then add `lam` to every diagonal entry:
   ```python
   A = K_train + lam * np.eye(n)    # shape (n, n)
   ```
2. Solve the linear system with `np.linalg.solve(A, y)`. This finds `alpha` without explicitly inverting `A`, which is both faster and numerically more stable than `np.linalg.inv(A) @ y`.

Return `alpha`, shape `(n,)`.

**TODO 12 — `kernel_ridge_predict(K_test_train, alpha)`**

Each test prediction is a weighted sum over the training kernel values:

$$\hat{y}_i = \sum_j \alpha_j \cdot K(x_i^{\text{test}}, x_j^{\text{train}})$$

Written as a matrix-vector product this is `K_test_train @ alpha`. `K_test_train` has shape `(n_test, n_train)` and `alpha` has shape `(n_train,)`, so the result has shape `(n_test,)` — one prediction per test point.

**TODO 13 — `approximation_error(K_exact, K_approx)`**

The Frobenius norm $\|M\|_F = \sqrt{\sum_{i,j} M_{ij}^2}$ — use `np.linalg.norm(M, "fro")`. Three lines:

1. `numerator   = np.linalg.norm(K_exact - K_approx, "fro")` — how far apart the matrices are.
2. `denominator = np.linalg.norm(K_exact, "fro")` — the scale of the exact matrix.
3. `return float(numerator / denominator)` — the ratio, as a Python float.

A result of `0.0` means perfect approximation; `1.0` means the approximation is as bad as predicting all zeros; anything below `0.1` is generally considered good.

---

## Exercise 6: Benchmarking

### What to measure

For each method and each `n`, you need:

- **Time**: record `time.perf_counter()` before and after building the kernel/feature matrix.
- **Memory**: record `psutil.Process().memory_info().rss` before and after, in MB. RSS can be noisy — take the max of the delta and the matrix's `.nbytes / 1024**2`.

```python
import psutil
mem_before = psutil.Process().memory_info().rss / 1024**2
# ... build matrix ...
mem_after  = psutil.Process().memory_info().rss / 1024**2
```

### Implementation hints

**TODO 14 — `benchmark_methods(n_values, D_values, d, sigma)`**

The outer loop pairs each dataset size with its feature budget:

```python
rng = np.random.default_rng(0)
results = []

for n, D in zip(n_values, D_values):
    X = rng.standard_normal((n, d))
    ...
```

For each method, the timing and memory pattern is always the same three lines:

```python
mem_before = psutil.Process().memory_info().rss / 1024**2
t0 = time.perf_counter()
# ... build kernel matrix or feature matrix Z here ...
time_s = time.perf_counter() - t0
mem_after = psutil.Process().memory_info().rss / 1024**2
```

Then append a dict to `results`:

```python
results.append({
    "method": "RFF",          # or "Exact", "ORF", "Nyström"
    "n": n,
    "D": D,
    "time_s": time_s,
    "memory_mb": max(mem_after - mem_before, Z.nbytes / 1024**2),
    "approx_error": approximation_error(K_exact, Z @ Z.T) if K_exact is not None else None,
})
```

The `max(...)` for memory handles cases where RSS doesn't increase (already-allocated memory reuse); `.nbytes / 1024**2` gives the theoretical minimum.

Run the four methods in this order for each `n`:

1. **Exact** (only if `n <= 3_000`): call `rbf_kernel_matrix(X, sigma)`. Set `approx_error = 0.0` and keep `K_exact` for the others. If `n > 3_000`, set `K_exact = None`.
2. **RFF**: `sample_rff_weights(D, d, sigma, rng)` → `rff_features(X, omega, b)`.
3. **ORF**: `sample_orf_weights(D, d, sigma, rng)` → `orf_features(X, omega, b)`.
4. **Nyström**: `select_landmarks(X, m=D, strategy="random", rng=rng)` → `nystrom_features(X, landmarks, sigma)`.

Return `results` after the loop.

**`plot_results(results)`** — already implemented for you. Call it with the output of `benchmark_methods` to produce the 3-panel figure.

---

## Choosing σ: The Median Heuristic

A common practical choice is the **median heuristic**: set $\sigma = \sqrt{\text{median}(\|x_i - x_j\|^2) / 2}$.

For $n$ points drawn from $\mathcal{N}(0, I_d)$, the expected squared distance is $\mathbb{E}[\|x-y\|^2] = 2d$, so a reasonable default is $\sigma \approx \sqrt{d}$.

With $\sigma = 1$ and $d = 10$, most off-diagonal kernel values are $\approx e^{-10} \approx 0$ — the kernel is trivially diagonal and no approximation can do anything interesting. Always match $\sigma$ to your data's scale.

---

## Common Pitfalls

| Problem | Cause | Fix |
|---|---|---|
| Kernel matrix has negative eigenvalues | Floating point error in $\|x-y\|^2$ | `np.maximum(D2, 0)` before `exp` |
| Nyström features are NaN | Near-zero eigenvalue in $K_{mm}$ | Clip eigenvalues to `max(λ, 1e-10)` |
| RFF approximation has high variance | D too small | Increase D; try ORF |
| `np.linalg.solve` raises `LinAlgError` | $K + \lambda I$ near-singular | Increase λ |
| Broadcasting error in feature map | Wrong transpose | Verify `X @ omega.T` gives `(n, D)` |
| Approximation worse than expected | σ too small for data dimensionality | Use median heuristic: $\sigma \approx \sqrt{d}$ |

---

## Complexity Summary

| Method | Feature map cost | Memory |
|---|---|---|
| Exact kernel | $O(n^2 d)$ | $O(n^2)$ |
| RFF / ORF | $O(n D d)$ | $O(nD)$ |
| Nyström | $O(nm d + m^3)$ | $O(nm)$ |

All three approximate methods break the $O(n^2)$ wall. The choice between them:

- **RFF**: simplest, well-understood, scales to very large $D$.
- **ORF**: same cost as RFF, lower variance — preferred when $D \lesssim d$.
- **Nyström**: adapts to data geometry; great when the kernel has fast eigenvalue decay; requires an $O(m^3)$ factorisation of $K_{mm}$.

---

## Key References

1. Rahimi, A. & Recht, B. (2007). *Random Features for Large-Scale Kernel Machines*. NeurIPS.
2. Yu, F. X., Suresh, A. T., Choromanski, K., Holtmann-Rice, D., & Kumar, S. (2016). *Orthogonal Random Features*. NeurIPS.
3. Williams, C. & Seeger, M. (2001). *Using the Nyström Method to Speed Up Kernel Machines*. NeurIPS.
