"""
Lab 08: Kernel Approximation Methods for Big Data

STUDENT NAME: [Your Name Here]

STUDENT REFLECTION:
(Please write a short paragraph here explaining what you learned in this lab,
focusing on the tradeoffs between exact kernel computation and the three
approximation strategies you implemented.)
"""

import time
import math
import warnings
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import psutil
from scipy.linalg import eigh


# ---------------------------------------------------------------------------
# Exercise 1: Exact RBF Kernel — Understanding the Problem
# ---------------------------------------------------------------------------

def rbf_kernel(x: np.ndarray, y: np.ndarray, sigma: float) -> float:
    """
    Compute the Gaussian (RBF) kernel between two vectors.

    The RBF kernel is defined as:

        K(x, y) = exp( -||x - y||² / (2σ²) )

    It measures similarity between two points: K = 1 when x = y,
    and K → 0 as the distance grows. σ (sigma) controls the width:
    large σ makes the kernel smoother / longer-range.

    Args:
        x (np.ndarray): First vector, shape (d,).
        y (np.ndarray): Second vector, shape (d,).
        sigma (float): Bandwidth parameter σ > 0.

    Returns:
        float: Kernel value in (0, 1].

    Examples:
        >>> x = np.array([0.0, 0.0])
        >>> rbf_kernel(x, x, sigma=1.0)
        1.0
        >>> 0.0 < rbf_kernel(np.array([0.0]), np.array([1.0]), sigma=1.0) < 1.0
        True
    """
    raise NotImplementedError("TODO 1: Compute exp(-||x-y||² / (2*sigma²))")


def rbf_kernel_matrix(X: np.ndarray, sigma: float) -> np.ndarray:
    """
    Compute the full n×n Gram (kernel) matrix for dataset X.

    K[i, j] = rbf_kernel(X[i], X[j], sigma)

    The Gram matrix is symmetric and positive semi-definite.
    For n samples in ℝ^d this requires O(n²) memory and O(n²·d) time —
    the core scalability problem for kernel methods.

    A warning is printed when n > 5_000, since the resulting matrix
    would exceed ~200 MB of float64 memory.

    Implementation hint: avoid an explicit Python double loop.
    Use the identity ||x-y||² = ||x||² + ||y||² - 2·xᵀy and numpy
    broadcasting to compute the full pairwise distance matrix at once.

    Args:
        X (np.ndarray): Data matrix, shape (n, d).
        sigma (float): Bandwidth parameter σ > 0.

    Returns:
        np.ndarray: Gram matrix, shape (n, n), dtype float64.

    Examples:
        >>> X = np.eye(3)
        >>> K = rbf_kernel_matrix(X, sigma=1.0)
        >>> K.shape
        (3, 3)
        >>> np.allclose(np.diag(K), 1.0)
        True
        >>> np.allclose(K, K.T)
        True
    """
    # TODO 2: Build the n×n kernel matrix WITHOUT a Python double loop.
    #
    # Step 1 — warn if n is large:
    #   n = X.shape[0]
    #   if n > 5_000: warnings.warn("...", ResourceWarning, stacklevel=2)
    #
    # Step 2 — compute squared row norms, shape (n,):
    #   Each entry i should be ||X[i]||² = sum of squares of row i.
    #   Hint: (X ** 2).sum(axis=1)
    #
    # Step 3 — compute the (n, n) pairwise squared-distance matrix D2:
    #   Use the identity ||x_i - x_j||² = ||x_i||² + ||x_j||² - 2·x_iᵀx_j
    #   The first two terms come from broadcasting sq_norms as a column and
    #   a row vector. The third term is 2 * (X @ X.T).
    #   Then clamp: D2 = np.maximum(D2, 0.0)   ← avoids tiny negatives from float arithmetic
    #
    # Step 4 — apply the RBF formula element-wise:
    #   return np.exp(-D2 / (2 * sigma**2))
    raise NotImplementedError(
        "TODO 2: Compute pairwise squared distances with broadcasting, "
        "then apply the RBF formula. Warn if n > 5_000."
    )


# ---------------------------------------------------------------------------
# Exercise 2: Random Fourier Features  (Rahimi & Recht, NeurIPS 2007)
# ---------------------------------------------------------------------------

def sample_rff_weights(
    D: int, d: int, sigma: float, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """
    Sample random frequency vectors and phase shifts for Random Fourier Features.

    By Bochner's theorem, the RBF kernel is the Fourier transform of a Gaussian
    spectral density. This means we can approximate:

        K(x, y) ≈ z(x)ᵀ z(y)

    by drawing D random frequencies ω from the spectral distribution p(ω):

        ω_j ~ N(0, I/σ²)   for j = 1, …, D
        b_j ~ Uniform[0, 2π]

    Args:
        D (int): Number of random features (more → better approximation).
        d (int): Input dimensionality.
        sigma (float): Bandwidth parameter σ of the target RBF kernel.
        rng (np.random.Generator): Seeded random number generator.

    Returns:
        tuple[np.ndarray, np.ndarray]:
            omega: shape (D, d)  — frequency matrix
            b:     shape (D,)    — phase shifts
    """
    # TODO 3: Sample the frequency matrix omega and phase shifts b.
    #
    # omega — shape (D, d):
    #   Each row ω_j must be drawn from N(0, I/σ²), i.e. each element is
    #   an independent normal with mean 0 and std 1/σ.
    #   Use rng.standard_normal((D, d)) to get N(0,1) samples, then divide
    #   by sigma to scale to N(0, I/σ²).
    #
    # b — shape (D,):
    #   Each phase shift b_j is drawn uniformly from [0, 2π].
    #   Use rng.uniform(low, high, size=D) with the right bounds.
    #   Note: 2π is available as 2 * np.pi.
    #
    # Return both as a tuple: (omega, b)
    raise NotImplementedError(
        "TODO 3: Sample omega from N(0, I/sigma²) with shape (D, d), "
        "and b from Uniform[0, 2*pi] with shape (D,)."
    )


def rff_features(
    X: np.ndarray, omega: np.ndarray, b: np.ndarray
) -> np.ndarray:
    """
    Compute the Random Fourier Feature map Z for dataset X.

    The feature map is:

        z(x) = √(2/D) · cos(ω·x + b)

    where ω ∈ ℝ^(D×d) and b ∈ ℝ^D are drawn via `sample_rff_weights`.
    This gives a D-dimensional explicit feature vector such that:

        z(x)ᵀ z(y) ≈ K(x, y)    in expectation.

    The scaling factor √(2/D) ensures the approximation is unbiased.

    Args:
        X (np.ndarray): Data matrix, shape (n, d).
        omega (np.ndarray): Frequency matrix, shape (D, d).
        b (np.ndarray): Phase shift vector, shape (D,).

    Returns:
        np.ndarray: Feature matrix Z, shape (n, D).

    Examples:
        >>> rng = np.random.default_rng(0)
        >>> omega, b = sample_rff_weights(D=100, d=4, sigma=1.0, rng=rng)
        >>> X = rng.standard_normal((10, 4))
        >>> Z = rff_features(X, omega, b)
        >>> Z.shape
        (10, 100)
    """
    # TODO 4: Compute the feature matrix Z with shape (n, D).
    #
    # The formula is:  Z = sqrt(2/D) * cos(X @ omega.T + b)
    #
    # Step by step:
    #   1. X has shape (n, d) and omega has shape (D, d).
    #      X @ omega.T gives shape (n, D) — each entry (i, j) is x_i · ω_j.
    #
    #   2. Adding b (shape (D,)) broadcasts across the n rows automatically,
    #      so (X @ omega.T + b) is still shape (n, D).
    #
    #   3. Apply np.cos element-wise, then multiply by sqrt(2/D).
    #      Get D from omega.shape[0].
    #
    # The scaling factor sqrt(2/D) makes the approximation unbiased:
    #   E[Z[i] · Z[j]] = K(x_i, x_j)
    raise NotImplementedError(
        "TODO 4: Compute Z = sqrt(2/D) * cos(X @ omega.T + b). "
        "D = omega.shape[0]."
    )


def rff_kernel_approx(
    X: np.ndarray, Y: np.ndarray, omega: np.ndarray, b: np.ndarray
) -> np.ndarray:
    """
    Approximate the kernel matrix K(X, Y) using Random Fourier Features.

    K(X, Y) ≈ Z_X · Z_Yᵀ

    where Z_X = rff_features(X, omega, b) and Z_Y = rff_features(Y, omega, b).

    Args:
        X (np.ndarray): First data matrix, shape (n, d).
        Y (np.ndarray): Second data matrix, shape (m, d).
        omega (np.ndarray): Frequency matrix, shape (D, d).
        b (np.ndarray): Phase shift vector, shape (D,).

    Returns:
        np.ndarray: Approximate kernel matrix, shape (n, m).
    """
    # TODO 5: Approximate K(X, Y) ≈ Z_X @ Z_Y.T
    #
    # Both X and Y live in the same d-dimensional space, so you can apply
    # the same (omega, b) to both — they share the same random projection.
    #
    #   Z_X = rff_features(X, omega, b)   shape (n, D)
    #   Z_Y = rff_features(Y, omega, b)   shape (m, D)
    #
    # Their dot product Z_X @ Z_Y.T has shape (n, m), which is exactly the
    # shape of the kernel matrix K(X, Y) where K[i,j] ≈ K(x_i, y_j).
    raise NotImplementedError(
        "TODO 5: Compute rff_features for X and Y, then return Z_X @ Z_Y.T."
    )


# ---------------------------------------------------------------------------
# Exercise 3: Orthogonal Random Features  (Yu et al., ICML 2016)
# ---------------------------------------------------------------------------

def sample_orf_weights(
    D: int, d: int, sigma: float, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """
    Sample orthogonal random frequency vectors for Orthogonal Random Features.

    Standard RFF draws each ω_j independently from N(0, I/σ²). This causes
    redundancy: sampled vectors may nearly align, wasting capacity. ORF fixes
    this by forcing the weight matrix to have orthogonal rows.

    Construction (block-orthogonal):
    1. Draw G ~ N(0, I) of shape (D, d).
    2. Compute the QR decomposition: G = Q · R, giving Q with orthonormal rows.
    3. To preserve the correct marginal distribution (each row should have the
       same length distribution as an N(0, I) draw), rescale each row of Q by
       an independent χ_d-distributed norm:
           s_j ~ χ_d  (i.e., s_j = ||g_j||  where g_j ~ N(0, I_d))
    4. Divide by sigma to match the N(0, I/σ²) spectral density.

    This yields ORF weights with orthogonal directions but the same marginal
    distribution as standard RFF — resulting in lower approximation variance.

    When D > d, tile multiple independent orthogonal blocks.

    Args:
        D (int): Number of random features.
        d (int): Input dimensionality.
        sigma (float): Bandwidth parameter σ of the target RBF kernel.
        rng (np.random.Generator): Seeded random number generator.

    Returns:
        tuple[np.ndarray, np.ndarray]:
            omega: shape (D, d)  — orthogonal frequency matrix
            b:     shape (D,)    — phase shifts (same as RFF)

    Implementation hint:
        Build ceil(D/d) orthogonal blocks of shape (d, d), stack them,
        and trim to (D, d). Use np.linalg.qr on a (d, d) Gaussian matrix,
        then rescale rows by norms drawn from chi_d via ||rng.standard_normal((d, d))||.
    """
    # TODO 6: Build the orthogonal frequency matrix omega, shape (D, d).
    #
    # The idea: instead of drawing D independent rows from N(0, I/σ²) as in RFF,
    # we force the rows to be mutually orthogonal. We do this in blocks of d rows.
    #
    # For each block:
    #   1. Draw a (d, d) Gaussian matrix:  G = rng.standard_normal((d, d))
    #
    #   2. QR-decompose it:  Q, _ = np.linalg.qr(G)
    #      Q has shape (d, d) with orthonormal *columns*.
    #      Transpose it so the rows are orthonormal:  Q = Q.T   → shape (d, d)
    #      Now each row of Q is a unit vector, and all rows are orthogonal to each other.
    #
    #   3. Rescale each row by a chi_d-distributed norm.
    #      A chi_d sample is the L2 norm of a d-dimensional standard normal vector.
    #      Draw a fresh (d, d) Gaussian:  S = rng.standard_normal((d, d))
    #      Compute the norm of each row:  norms = np.linalg.norm(S, axis=1)  → shape (d,)
    #      Multiply each row of Q by its norm:  Q = Q * norms[:, None]
    #      This gives rows with the correct length distribution (same as N(0, I_d) draws).
    #
    #   4. Append this block to a list.
    #
    # Repeat for math.ceil(D / d) blocks, then:
    #   - Stack all blocks:  omega = np.vstack(blocks)   → shape (n_blocks*d, d)
    #   - Trim to D rows:    omega = omega[:D]            → shape (D, d)
    #   - Divide by sigma:   omega = omega / sigma        → rescales to N(0, I/σ²)
    #
    # Sample b the same way as in RFF:
    #   b = rng.uniform(0, 2 * np.pi, size=D)
    #
    # Return (omega, b).
    raise NotImplementedError(
        "TODO 6: Build orthogonal blocks via QR, rescale rows by chi_d norms, "
        "divide by sigma, trim to D rows, and sample b ~ Uniform[0, 2*pi]."
    )



def orf_features(
    X: np.ndarray, omega: np.ndarray, b: np.ndarray
) -> np.ndarray:
    """
    Compute the Orthogonal Random Feature map Z for dataset X.

    The formula is identical to RFF:

        z(x) = √(2/D) · cos(ω·x + b)

    but omega has been constructed with orthogonal rows (from `sample_orf_weights`),
    giving lower variance for the same D compared to standard RFF.

    Args:
        X (np.ndarray): Data matrix, shape (n, d).
        omega (np.ndarray): Orthogonal frequency matrix, shape (D, d).
        b (np.ndarray): Phase shift vector, shape (D,).

    Returns:
        np.ndarray: Feature matrix Z, shape (n, D).
    """
    # TODO 7: Apply the same feature map formula as rff_features.
    #
    # The formula is identical:  Z = sqrt(2/D) * cos(X @ omega.T + b)
    #
    # The improvement over RFF comes entirely from how omega was constructed
    # in TODO 6 (orthogonal rows). The computation here is the same.
    # Simply call rff_features(X, omega, b) and return the result.
    raise NotImplementedError(
        "TODO 7: Same formula as rff_features — reuse it."
    )


# ---------------------------------------------------------------------------
# Exercise 4: Nyström Approximation
# ---------------------------------------------------------------------------

def select_landmarks(
    X: np.ndarray,
    m: int,
    strategy: str = "random",
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Select m landmark (inducing) points from dataset X.

    Landmarks define the low-rank subspace used by the Nyström approximation.
    More landmarks → better approximation but higher cost.

    Two strategies:
    - "random": choose m rows of X uniformly at random (without replacement).
    - "stride": choose m equally-spaced rows using np.linspace indices.

    Args:
        X (np.ndarray): Data matrix, shape (n, d).
        m (int): Number of landmarks (m ≤ n).
        strategy (str): "random" (default) or "stride".
        rng (np.random.Generator | None): Required when strategy="random".

    Returns:
        np.ndarray: Landmark matrix, shape (m, d).

    Raises:
        ValueError: If strategy is not "random" or "stride".
    """
    # TODO 8: Select m rows from X to use as landmarks.
    #
    # Get n from X.shape[0].
    #
    # Strategy "random":
    #   Use rng.choice(n, m, replace=False) to pick m unique row indices.
    #   replace=False ensures no row is selected twice.
    #   Then index into X: return X[idx]
    #
    # Strategy "stride":
    #   Use np.linspace(0, n-1, m, dtype=int) to get m evenly spaced indices
    #   across the full range of rows. Then return X[idx].
    #
    # Any other strategy string: raise ValueError(f"Unknown strategy '{strategy}'...")
    raise NotImplementedError(
        "TODO 8: For 'random', use rng.choice(n, m, replace=False). "
        "For 'stride', use np.linspace(0, n-1, m, dtype=int)."
    )


def nystrom_features(
    X: np.ndarray, landmarks: np.ndarray, sigma: float
) -> np.ndarray:
    """
    Compute the Nyström feature map for dataset X given landmark points.

    The Nyström method approximates the full kernel matrix as:

        K(X, X) ≈ Z · Zᵀ    where Z = K_nm · K_mm^{-1/2}

    Notation:
        K_mm = rbf_kernel_matrix(landmarks, sigma)   shape (m, m)
        K_nm = kernel between X and landmarks         shape (n, m)
        K_mm^{-1/2} = V · Λ^{-1/2} · Vᵀ            via eigendecomposition

    Implementation steps:
    1. Compute K_mm and K_nm.
    2. Eigendecompose K_mm: eigenvalues λ, eigenvectors V (use scipy.linalg.eigh
       for numerical stability with symmetric matrices).
    3. Clip eigenvalues to a small positive floor (e.g. 1e-10) to avoid
       division by zero for near-zero eigenvalues.
    4. Compute K_mm^{-1/2} = V · diag(λ^{-1/2}) · Vᵀ.
    5. Return Z = K_nm @ K_mm^{-1/2}, shape (n, m).

    Args:
        X (np.ndarray): Data matrix, shape (n, d).
        landmarks (np.ndarray): Landmark matrix, shape (m, d).
        sigma (float): Bandwidth parameter σ.

    Returns:
        np.ndarray: Nyström feature matrix Z, shape (n, m).
    """
    # TODO 9: Compute the Nyström feature matrix Z, shape (n, m).
    #
    # Step 1 — kernel matrix between landmarks and themselves, shape (m, m):
    #   K_mm = rbf_kernel_matrix(landmarks, sigma)
    #
    # Step 2 — kernel matrix between all X points and the landmarks, shape (n, m):
    #   Use the same vectorised approach as rbf_kernel_matrix but between
    #   two different matrices. The pairwise squared distances between X (n, d)
    #   and landmarks (m, d) are:
    #     sq_X  = (X ** 2).sum(axis=1)          → shape (n,)
    #     sq_L  = (landmarks ** 2).sum(axis=1)   → shape (m,)
    #     D2    = sq_X[:, None] + sq_L[None, :] - 2 * (X @ landmarks.T)
    #                                             → shape (n, m)
    #     K_nm  = np.exp(-np.maximum(D2, 0) / (2 * sigma ** 2))
    #
    # Step 3 — eigendecompose K_mm to get its inverse square root:
    #   lam, V = eigh(K_mm)      # eigh is for symmetric matrices; returns real eigenvalues
    #   lam is shape (m,), ascending. V is shape (m, m), columns are eigenvectors.
    #
    # Step 4 — clip eigenvalues to avoid division by zero:
    #   lam = np.maximum(lam, 1e-10)
    #
    # Step 5 — compute K_mm^{-1/2} = V @ diag(1/sqrt(lam)) @ V.T:
    #   K_mm_inv_sqrt = V @ np.diag(1.0 / np.sqrt(lam)) @ V.T   → shape (m, m)
    #
    # Step 6 — return the feature matrix:
    #   return K_nm @ K_mm_inv_sqrt    → shape (n, m)
    raise NotImplementedError(
        "TODO 9: Compute K_mm and K_nm, eigendecompose K_mm, "
        "clip eigenvalues, form K_mm^{-1/2}, return K_nm @ K_mm^{-1/2}."
    )
    


def nystrom_kernel_approx(
    X: np.ndarray, Y: np.ndarray, landmarks: np.ndarray, sigma: float
) -> np.ndarray:
    """
    Approximate K(X, Y) using the Nyström method.

    K(X, Y) ≈ Z_X · Z_Yᵀ

    where Z_X = nystrom_features(X, landmarks, sigma)
      and Z_Y = nystrom_features(Y, landmarks, sigma).

    Args:
        X (np.ndarray): First data matrix, shape (n, d).
        Y (np.ndarray): Second data matrix, shape (m_pts, d).
        landmarks (np.ndarray): Landmark matrix, shape (m, d).
        sigma (float): Bandwidth parameter σ.

    Returns:
        np.ndarray: Approximate kernel matrix, shape (n, m_pts).
    """
    # TODO 10: Approximate K(X, Y) ≈ Z_X @ Z_Y.T
    #
    # Same pattern as rff_kernel_approx (TODO 5), but using nystrom_features
    # instead of rff_features. Both X and Y must use the same landmarks so
    # their feature spaces are aligned.
    #
    #   Z_X = nystrom_features(X, landmarks, sigma)   → shape (n, m)
    #   Z_Y = nystrom_features(Y, landmarks, sigma)   → shape (m_pts, m)
    #
    # Return Z_X @ Z_Y.T  → shape (n, m_pts)
    raise NotImplementedError(
        "TODO 10: Compute nystrom_features for X and Y, return Z_X @ Z_Y.T."
    )


# ---------------------------------------------------------------------------
# Exercise 5: Kernel Ridge Regression
# ---------------------------------------------------------------------------

def kernel_ridge_fit(
    K_train: np.ndarray, y: np.ndarray, lam: float
) -> np.ndarray:
    """
    Fit kernel ridge regression given a pre-computed training kernel matrix.

    Kernel ridge regression solves the dual problem:

        α* = (K + λ·I)^{-1} · y

    where:
        K  is the n×n kernel matrix on training points
        λ  is the regularisation strength (prevents overfitting)
        α* is the vector of dual coefficients (one per training point)

    Prediction for new point x* is then:
        ŷ = Σ_i α_i · K(x*, x_i) = k_*ᵀ · α*

    Hint: use np.linalg.solve(A, b) instead of computing an explicit inverse.

    Args:
        K_train (np.ndarray): Training kernel matrix, shape (n, n).
        y (np.ndarray): Training targets, shape (n,).
        lam (float): Regularisation parameter λ > 0.

    Returns:
        np.ndarray: Dual coefficients α*, shape (n,).
    """
    # TODO 11: Compute the dual coefficients alpha*, shape (n,).
    #
    # You need to solve the linear system:
    #   (K_train + lam * I) @ alpha = y
    #
    # Step 1 — build the regularised matrix:
    #   n = K_train.shape[0]
    #   A = K_train + lam * np.eye(n)    # adds lam to every diagonal entry
    #
    # Step 2 — solve for alpha using np.linalg.solve:
    #   alpha = np.linalg.solve(A, y)
    #
    # np.linalg.solve(A, y) finds x such that A @ x = y without explicitly
    # inverting A. It is faster and more numerically stable than np.linalg.inv(A) @ y.
    #
    # Return alpha, shape (n,).
    raise NotImplementedError(
        "TODO 11: Solve (K_train + lam * I) @ alpha = y using np.linalg.solve."
    )


def kernel_ridge_predict(
    K_test_train: np.ndarray, alpha: np.ndarray
) -> np.ndarray:
    """
    Predict with kernel ridge regression using pre-computed test kernel values.

    ŷ = K_test_train · α

    Args:
        K_test_train (np.ndarray): Kernel matrix between test and train,
            shape (n_test, n_train).
        alpha (np.ndarray): Dual coefficients from `kernel_ridge_fit`,
            shape (n_train,).

    Returns:
        np.ndarray: Predicted values, shape (n_test,).
    """
    # TODO 12: Predict targets for the test points, shape (n_test,).
    #
    # Each test prediction is a weighted sum of the training kernel values:
    #   y_hat[i] = sum_j alpha[j] * K(x_test_i, x_train_j)
    #
    # Written in matrix form this is just:
    #   y_hat = K_test_train @ alpha
    #
    # K_test_train has shape (n_test, n_train) and alpha has shape (n_train,),
    # so the result has shape (n_test,) — one prediction per test point.
    raise NotImplementedError(
        "TODO 12: Return K_test_train @ alpha."
    )


def approximation_error(K_exact: np.ndarray, K_approx: np.ndarray) -> float:
    """
    Compute the relative Frobenius-norm error between two kernel matrices.

        error = ||K_exact - K_approx||_F / ||K_exact||_F

    A value of 0 means perfect approximation; 1 means the approximation
    explains nothing; values > 1 indicate a very poor approximation.

    Args:
        K_exact (np.ndarray): Ground-truth kernel matrix.
        K_approx (np.ndarray): Approximated kernel matrix.

    Returns:
        float: Relative Frobenius-norm error ≥ 0.
    """
    # TODO 13: Compute the relative Frobenius-norm error, returned as a float.
    #
    # The Frobenius norm of a matrix M is:
    #   ||M||_F = sqrt(sum of all squared elements)
    # Use np.linalg.norm(M, "fro") to compute it.
    #
    # Step 1 — numerator: norm of the difference matrix:
    #   numerator = np.linalg.norm(K_exact - K_approx, "fro")
    #
    # Step 2 — denominator: norm of the exact matrix:
    #   denominator = np.linalg.norm(K_exact, "fro")
    #
    # Step 3 — return the ratio as a Python float:
    #   return float(numerator / denominator)
    #
    # Interpretation:
    #   0.0  → perfect approximation (K_approx == K_exact)
    #   1.0  → approximation is as far from K_exact as a zero matrix would be
    #   <0.1 → generally considered a good approximation
    raise NotImplementedError(
        "TODO 13: Return np.linalg.norm(K_exact - K_approx, 'fro') "
        "/ np.linalg.norm(K_exact, 'fro')."
    )


# ---------------------------------------------------------------------------
# Exercise 6: Benchmarking
# ---------------------------------------------------------------------------

def benchmark_methods(
    n_values: list[int],
    D_values: list[int],
    d: int = 20,
    sigma: float = 1.0,
) -> list[dict]:
    """
    Benchmark exact kernel, RFF, ORF, and Nyström across dataset sizes.

    For each n in n_values and corresponding D (= m = D_values[i]):
    - Generate synthetic data: X ~ N(0, I), y = sin(Xw) + 0.1·ε
    - Measure wall-clock time and peak memory for building the kernel/feature
      matrix on the full dataset.
    - Compute the relative approximation error against the exact kernel
      (only feasible for small n; skip for n > 3_000).

    Returns a list of dicts, one per (method, n) combination, with keys:
        method (str), n (int), D (int), time_s (float), memory_mb (float),
        approx_error (float | None)

    Memory measurement: record RSS before and after matrix construction using
    `psutil.Process().memory_info().rss / 1024**2`.

    Args:
        n_values (list[int]): Dataset sizes to test, e.g. [500, 1000, 5000].
        D_values (list[int]): Random features / landmarks for each n, e.g. [100, 200, 500].
        d (int): Input dimensionality (default 20).
        sigma (float): RBF bandwidth (default 1.0).

    Returns:
        list[dict]: One entry per (method, n) pair.
    """
    # TODO 14: Loop over n_values and measure time + memory for each method.
    #
    # You will need: import psutil, import time (both already imported at top).
    #
    # Setup — create a fixed rng and an empty results list:
    #   rng = np.random.default_rng(0)
    #   results = []
    #
    # Outer loop — iterate over dataset sizes and feature budgets together:
    #   for n, D in zip(n_values, D_values):
    #
    #     Generate data:
    #       X = rng.standard_normal((n, d))
    #
    #     --- Exact kernel (only when n <= 3_000, otherwise too slow/large) ---
    #     if n <= 3_000:
    #       mem_before = psutil.Process().memory_info().rss / 1024**2
    #       t0 = time.perf_counter()
    #       K_exact = rbf_kernel_matrix(X, sigma)
    #       time_s = time.perf_counter() - t0
    #       mem_after = psutil.Process().memory_info().rss / 1024**2
    #       results.append({
    #           "method": "Exact", "n": n, "D": D,
    #           "time_s": time_s,
    #           "memory_mb": max(mem_after - mem_before, K_exact.nbytes / 1024**2),
    #           "approx_error": 0.0,
    #       })
    #     else:
    #       K_exact = None   # can't compute it; used to skip error calculation below
    #
    #     --- For each approximate method (RFF, ORF, Nyström) do the same pattern ---
    #     Measure mem_before, record t0, build the feature matrix Z, record elapsed
    #     time and mem_after. Then:
    #       approx_error = approximation_error(K_exact, Z @ Z.T) if K_exact is not None else None
    #     Append a dict with keys: "method", "n", "D", "time_s", "memory_mb", "approx_error".
    #
    #     For RFF:
    #       omega, b = sample_rff_weights(D, d, sigma, rng)
    #       Z = rff_features(X, omega, b)
    #
    #     For ORF:
    #       omega, b = sample_orf_weights(D, d, sigma, rng)
    #       Z = orf_features(X, omega, b)
    #
    #     For Nyström (use D as the number of landmarks m):
    #       landmarks = select_landmarks(X, m=D, strategy="random", rng=rng)
    #       Z = nystrom_features(X, landmarks, sigma)
    #
    # return results
    raise NotImplementedError(
        "TODO 14: Loop over n_values; for each n generate data, "
        "then time+measure memory for exact (if n<=3000), RFF, ORF, Nyström. "
        "Collect results into a list of dicts."
    )


def plot_results(results: list[dict]) -> plt.Figure:
    """
    Produce a four-panel figure summarising the benchmark results.

    Panel layout:
        (1) Top-left:  Wall-clock time (s) vs n — log-log scale, one line per method
        (2) Top-right: Peak memory (MB) vs n   — log-log scale, one line per method
        (3) Bottom-left:  Approximation error vs D/m — linear scale
        (4) Bottom-right: (blank or optional) — e.g. a note on the tradeoff

    Each panel should have:
        - Labelled axes with units
        - A legend identifying each method
        - A descriptive title

    Args:
        results (list[dict]): Output of `benchmark_methods`.

    Returns:
        plt.Figure: The matplotlib figure (also calls plt.tight_layout()).
    """

    by_method: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        by_method[r["method"]].append(r)

    colors = {"Exact": "black", "RFF": "tab:blue", "ORF": "tab:orange", "Nyström": "tab:green"}
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Kernel Approximation Benchmark", fontsize=14, fontweight="bold")

    # --- Panel 1: Wall-clock time vs n (log-log) ---
    ax = axes[0, 0]
    for method, rows in by_method.items():
        ns = [r["n"] for r in rows]
        ts = [r["time_s"] for r in rows]
        ax.loglog(ns, ts, "o-", label=method, color=colors.get(method))
    # O(n²) and O(n) reference lines anchored at the first Exact point
    exact_rows = by_method.get("Exact", [])
    if exact_rows:
        n0, t0_ref = exact_rows[0]["n"], exact_rows[0]["time_s"]
        ns_ref = np.array([r["n"] for r in exact_rows] + [r["n"] for r in by_method.get("RFF", [])])
        ns_ref = np.unique(ns_ref)
        ax.loglog(ns_ref, t0_ref * (ns_ref / n0) ** 2, "k--", alpha=0.35, label="O(n²) ref")
        ax.loglog(ns_ref, t0_ref * (ns_ref / n0) ** 1, "k:", alpha=0.35, label="O(n) ref")
    ax.set_xlabel("n (dataset size)")
    ax.set_ylabel("Time (s)")
    ax.set_title("Wall-clock time vs. n\n(log-log — slope = complexity class)")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", ls="--", alpha=0.4)

    # --- Panel 2: Peak memory vs n (log-log) ---
    ax = axes[0, 1]
    for method, rows in by_method.items():
        ns = [r["n"] for r in rows]
        ms = [r["memory_mb"] for r in rows]
        ax.loglog(ns, ms, "o-", label=method, color=colors.get(method))
    if exact_rows:
        n0, m0 = exact_rows[0]["n"], exact_rows[0]["memory_mb"]
        ax.loglog(ns_ref, m0 * (ns_ref / n0) ** 2, "k--", alpha=0.35, label="O(n²) ref")
        ax.loglog(ns_ref, m0 * (ns_ref / n0) ** 1, "k:", alpha=0.35, label="O(n) ref")
    ax.set_xlabel("n (dataset size)")
    ax.set_ylabel("Memory (MB)")
    ax.set_title("Peak memory vs. n\n(log-log — Exact grows as n², approx. as n·D)")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", ls="--", alpha=0.4)

    # --- Panel 3: Approximation error vs D (linear) ---
    ax = axes[1, 0]
    for method, rows in by_method.items():
        valid = [(r["D"], r["approx_error"]) for r in rows if r["approx_error"] is not None]
        if valid:
            Ds, errs = zip(*sorted(valid))
            ax.plot(Ds, errs, "o-", label=method, color=colors.get(method))
    ax.axhline(0.1, color="gray", ls="--", alpha=0.6, label="10% error threshold")
    ax.set_xlabel("D / m (random features / landmarks)")
    ax.set_ylabel("Relative Frobenius error")
    ax.set_title("Approximation quality vs. feature budget\n(lower = closer to exact kernel)")
    ax.legend(fontsize=8)
    ax.grid(True, ls="--", alpha=0.4)

    # --- Panel 4: Method tradeoff summary ---
    ax = axes[1, 1]
    ax.axis("off")
    summary = (
        "Method Tradeoff Summary\n"
        "─────────────────────────────────────\n\n"
        "Exact RBF\n"
        "  Time/Memory:  O(n²·d) — quadratic\n"
        "  Error:        0 (ground truth)\n"
        "  Limit:        infeasible for n > ~5k\n\n"
        "Random Fourier Features (RFF)\n"
        "  Time/Memory:  O(n·D·d) — linear in n\n"
        "  Error:        decreases as 1/√D\n"
        "  Tradeoff:     fast but high variance\n\n"
        "Orthogonal RF (ORF)\n"
        "  Time/Memory:  O(n·D·d) — same as RFF\n"
        "  Error:        lower variance than RFF\n"
        "  Tradeoff:     better quality, same cost\n\n"
        "Nyström\n"
        "  Time/Memory:  O(n·m + m³) — m = landmarks\n"
        "  Error:        often best for fixed budget\n"
        "  Tradeoff:     landmark quality matters\n\n"
        "Rule of thumb: D ≥ 300 gives < 10% error."
    )
    ax.text(
        0.04, 0.97, summary,
        transform=ax.transAxes,
        fontsize=9, verticalalignment="top", fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.6", facecolor="#f5f5f5", edgecolor="#aaaaaa"),
    )

    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Demo (run this file directly to see the O(n²) problem)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Lab 08 — Kernel Approximation for Big Data")
    print("=" * 50)

    rng = np.random.default_rng(42)
    d = 10
    sigma = 1.0

    # --- Exercise 1: exact kernel scaling ---
    print("\nExercise 1: Exact RBF kernel")
    for n in [100, 500, 1_000]:
        X = rng.standard_normal((n, d))
        t0 = time.perf_counter()
        K = rbf_kernel_matrix(X, sigma)
        elapsed = time.perf_counter() - t0
        mem_mb = K.nbytes / 1024**2
        print(f"  n={n:5d}: time={elapsed:.3f}s  memory={mem_mb:.1f} MB")

    # --- Exercise 2: RFF approximation quality ---
    print("\nExercise 2: Random Fourier Features")
    n, D = 300, 500
    X = rng.standard_normal((n, d))
    K_exact = rbf_kernel_matrix(X, sigma)
    omega, b = sample_rff_weights(D, d, sigma, rng)
    K_rff = rff_kernel_approx(X, X, omega, b)
    err = approximation_error(K_exact, K_rff)
    print(f"  n={n}, D={D}: approximation error = {err:.4f}")

    # --- Exercise 3: ORF vs RFF variance ---
    print("\nExercise 3: ORF vs RFF variance (50 seeds)")
    errors_rff, errors_orf = [], []
    for seed in range(50):
        g = np.random.default_rng(seed)
        om_rff, b_rff = sample_rff_weights(D, d, sigma, g)
        om_orf, b_orf = sample_orf_weights(D, d, sigma, g)
        errors_rff.append(approximation_error(K_exact, rff_kernel_approx(X, X, om_rff, b_rff)))
        errors_orf.append(approximation_error(K_exact, rff_kernel_approx(X, X, om_orf, b_orf)))
    print(f"  RFF mean±std error: {np.mean(errors_rff):.4f} ± {np.std(errors_rff):.4f}")
    print(f"  ORF mean±std error: {np.mean(errors_orf):.4f} ± {np.std(errors_orf):.4f}")

    # --- Exercise 4: Nyström ---
    print("\nExercise 4: Nyström approximation")
    m = 100
    landmarks = select_landmarks(X, m=m, strategy="random", rng=rng)
    K_nys = nystrom_kernel_approx(X, X, landmarks, sigma)
    err_nys = approximation_error(K_exact, K_nys)
    print(f"  n={n}, m={m}: approximation error = {err_nys:.4f}")

    # --- Exercise 5: Kernel Ridge Regression ---
    print("\nExercise 5: Kernel Ridge Regression")
    w_true = rng.standard_normal(d)
    y = np.sin(X @ w_true) + 0.1 * rng.standard_normal(n)
    split = int(0.8 * n)
    X_tr, X_te = X[:split], X[split:]
    y_tr, y_te = y[:split], y[split:]

    K_tr = rbf_kernel_matrix(X_tr, sigma)
    K_te_tr = np.array([[rbf_kernel(x, z, sigma) for z in X_tr] for x in X_te])
    alpha = kernel_ridge_fit(K_tr, y_tr, lam=0.01)
    y_pred = kernel_ridge_predict(K_te_tr, alpha)
    mse = np.mean((y_te - y_pred) ** 2)
    print(f"  Exact kernel MSE: {mse:.4f}")

    # --- Exercise 6: Benchmark ---
    print("\nExercise 6: Benchmarking methods")
    results = benchmark_methods(
        n_values=[200, 500, 1_000, 2_000, 3_000, 5_000, 10_000, 20_000],
        D_values=[50,  100,  200,  250,   300,   400,    500,    500],
        d=d,
        sigma=sigma,
    )
    fig = plot_results(results)
    plt.show()
