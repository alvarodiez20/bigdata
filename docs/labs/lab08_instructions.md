# Lab 08: Kernel Approximation Methods for Big Data — Instructions

Welcome to Lab 08! You will implement and compare four kernel computation strategies:

- **Exact RBF kernel** — the gold standard, and why it breaks at scale
- **Random Fourier Features (RFF)** — the Rahimi & Recht 2007 landmark
- **Orthogonal Random Features (ORF)** — lower-variance improvement over RFF
- **Nyström approximation** — landmark-based low-rank decomposition

## Additional Resources
- **[Tips & Reference Guide](lab08_guide.md)** — theoretical explanations, formulas, and implementation hints.

## Pre-flight Checklist

1. Checkout the `main` branch: `git checkout main`
2. Pull the latest changes: `git pull`
3. Create a local branch: `git checkout -b <your_branch_name>`
4. Ensure dependencies are up to date (scipy was added for this lab):
   ```bash
   uv sync
   ```
5. Run the tests (all will fail until you implement the functions):
   ```bash
   uv run pytest tests/test_lab08.py -v
   ```

---

## Exercise 1: Exact RBF Kernel — Understanding the Problem (TODOs 1–2)

!!! objective
    Implement the Gaussian kernel and its Gram matrix. Observe how memory requirements grow as O(n²).

You will edit **`src/lab08.py`**.

1. **`rbf_kernel(x, y, sigma)`** (TODO 1): Compute the scalar RBF kernel value:

    $$K(x, y) = \exp\!\left(-\frac{\|x - y\|^2}{2\sigma^2}\right)$$

2. **`rbf_kernel_matrix(X, sigma)`** (TODO 2): Compute the full n×n Gram matrix using vectorised numpy (avoid a Python double loop). Emit a `ResourceWarning` when `n > 5_000`.

    **Hint**: use the identity $\|x-y\|^2 = \|x\|^2 + \|y\|^2 - 2x^\top y$ and broadcasting.

**Validation**:
```bash
uv run pytest tests/test_lab08.py -k "TestRbfKernel or TestRbfKernelMatrix" -v
```

Try running the script to see memory grow:
```bash
uv run python src/lab08.py
```

---

## Exercise 2: Random Fourier Features (TODOs 3–5)

!!! objective
    Implement the Rahimi & Recht kernel approximation. Experience how an O(n·D) feature map approximates an O(n²) kernel matrix.

1. **`sample_rff_weights(D, d, sigma, rng)`** (TODO 3): Sample frequency vectors and phase shifts:

    $$\omega_j \sim \mathcal{N}(0,\, I/\sigma^2), \quad b_j \sim \text{Uniform}[0,\, 2\pi]$$

2. **`rff_features(X, omega, b)`** (TODO 4): Compute the explicit feature map:

    $$Z = \sqrt{2/D}\cdot\cos(X\omega^\top + b), \quad Z \in \mathbb{R}^{n \times D}$$

3. **`rff_kernel_approx(X, Y, omega, b)`** (TODO 5): Approximate $K(X, Y) \approx Z_X Z_Y^\top$.

**Validation**:
```bash
uv run pytest tests/test_lab08.py -k "TestRff" -v
```

---

## Exercise 3: Orthogonal Random Features (TODOs 6–7)

!!! objective
    Improve on standard RFF by ensuring frequency vectors are mutually orthogonal, reducing approximation variance for the same budget D.

1. **`sample_orf_weights(D, d, sigma, rng)`** (TODO 6): Build orthogonal frequency blocks:
    - Draw $G \sim \mathcal{N}(0, I)$ of shape $(d, d)$, compute QR decomposition.
    - Rescale rows of $Q^\top$ by independent $\chi_d$-distributed norms.
    - Repeat for $\lceil D/d \rceil$ blocks, stack, trim to $D$ rows, divide by $\sigma$.

2. **`orf_features(X, omega, b)`** (TODO 7): Apply the same feature formula as RFF — the improvement comes entirely from how `omega` was constructed.

**Validation**:
```bash
uv run pytest tests/test_lab08.py -k "TestOrf" -v
```

---

## Exercise 4: Nyström Approximation (TODOs 8–10)

!!! objective
    Implement the landmark-based Nyström method. Unlike RFF/ORF (which use random projections), Nyström leverages actual data points to build the approximation.

1. **`select_landmarks(X, m, strategy, rng)`** (TODO 8): Select $m$ landmark points from $X$. Support `"random"` (uniform without replacement) and `"stride"` (evenly spaced indices).

2. **`nystrom_features(X, landmarks, sigma)`** (TODO 9): Compute the Nyström feature map:

    $$Z = K_{nm}\, K_{mm}^{-1/2}$$

    where $K_{mm}^{-1/2} = V \Lambda^{-1/2} V^\top$ (use `scipy.linalg.eigh` for stability, clip eigenvalues to $\ge 10^{-10}$).

3. **`nystrom_kernel_approx(X, Y, landmarks, sigma)`** (TODO 10): Return $Z_X Z_Y^\top$.

**Validation**:
```bash
uv run pytest tests/test_lab08.py -k "TestNystrom or TestSelect" -v
```

---

## Exercise 5: Kernel Ridge Regression (TODOs 11–13)

!!! objective
    Use exact and approximate kernel matrices in the same downstream task to fairly compare approximation quality vs. prediction performance.

1. **`kernel_ridge_fit(K_train, y, lam)`** (TODO 11): Solve the dual system:

    $$\alpha^* = (K + \lambda I)^{-1} y$$

    Use `np.linalg.solve` (do **not** explicitly invert the matrix).

2. **`kernel_ridge_predict(K_test_train, alpha)`** (TODO 12): Compute predictions:

    $$\hat{y} = K_{\text{test,train}}\, \alpha^*$$

3. **`approximation_error(K_exact, K_approx)`** (TODO 13): Compute the relative Frobenius-norm error:

    $$\text{error} = \frac{\|K - \tilde{K}\|_F}{\|K\|_F}$$

**Validation**:
```bash
uv run pytest tests/test_lab08.py -k "TestKernelRidge or TestApprox" -v
```

---

## Exercise 6: Benchmarking (TODOs 14–15)

!!! objective
    Quantify the time, memory, and accuracy tradeoffs between all four methods across increasing dataset sizes.

1. **`benchmark_methods(n_values, D_values, d, sigma)`** (TODO 14): For each `(n, D)` pair, generate synthetic data, then time and measure peak memory for:
    - Exact kernel (skip for `n > 3_000`)
    - RFF, ORF, Nyström (always)

    Return a list of dicts with keys: `method`, `n`, `D`, `time_s`, `memory_mb`, `approx_error`.

2. **`plot_results(results)`** (TODO 15): Produce a 3-panel figure:
    - Panel 1: Wall-clock time vs n (log-log)
    - Panel 2: Peak memory vs n (log-log)
    - Panel 3: Approximation error vs D

**Validation**:
```bash
uv run pytest tests/test_lab08.py -k "TestBenchmark" -v
```

Run the full solution and observe the scaling:
```bash
uv run python src/lab08_solutions.py
```

---

## What to Submit

When all tests pass:

```bash
uv run pytest tests/test_lab08.py -v
```

Submit **exactly**:

1. **`src/lab08.py`** — your completed implementation with the `STUDENT REFLECTION` filled in.

**Do NOT submit:** `__pycache__` directories, `lab08_benchmark.png`, or the solutions file.

---

**Questions?** Check the [Tips & Reference Guide](lab08_guide.md) or ask your instructor.
