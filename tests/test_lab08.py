"""
Tests for Lab 08: Kernel Approximation Methods for Big Data.

Tests use the solutions module so they can be run against both the template
(expected to fail) and the completed solution (expected to pass).
"""

import math
import warnings

import numpy as np
import pytest

from src.lab08 import (
    approximation_error,
    benchmark_methods,
    kernel_ridge_fit,
    kernel_ridge_predict,
    nystrom_features,
    nystrom_kernel_approx,
    orf_features,
    plot_results,
    rbf_kernel,
    rbf_kernel_matrix,
    rff_features,
    rff_kernel_approx,
    sample_orf_weights,
    sample_rff_weights,
    select_landmarks,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def small_dataset(rng):
    n, d = 50, 8
    X = rng.standard_normal((n, d))
    return X, d


@pytest.fixture
def medium_dataset(rng):
    n, d = 200, 10
    X = rng.standard_normal((n, d))
    return X, d


# ---------------------------------------------------------------------------
# Exercise 1: Exact RBF Kernel
# ---------------------------------------------------------------------------

class TestRbfKernel:
    def test_self_similarity(self):
        """K(x, x) must equal 1 for any x."""
        x = np.array([1.0, 2.0, -3.0])
        assert rbf_kernel(x, x, sigma=1.0) == pytest.approx(1.0)

    def test_symmetry(self, rng):
        x = rng.standard_normal(5)
        y = rng.standard_normal(5)
        assert rbf_kernel(x, y, 1.0) == pytest.approx(rbf_kernel(y, x, 1.0))

    def test_range(self, rng):
        """Kernel values must be in (0, 1]."""
        for _ in range(20):
            x, y = rng.standard_normal(4), rng.standard_normal(4)
            k = rbf_kernel(x, y, sigma=1.0)
            assert 0.0 < k <= 1.0 + 1e-12

    def test_monotone_in_distance(self):
        """Larger distance → smaller kernel value."""
        x = np.zeros(2)
        y_close = np.array([0.1, 0.0])
        y_far = np.array([2.0, 0.0])
        assert rbf_kernel(x, y_close, 1.0) > rbf_kernel(x, y_far, 1.0)

    def test_sigma_effect(self):
        """Larger sigma → kernel decays more slowly."""
        x, y = np.zeros(2), np.ones(2)
        assert rbf_kernel(x, y, sigma=0.5) < rbf_kernel(x, y, sigma=2.0)


class TestRbfKernelMatrix:
    def test_shape(self, small_dataset):
        X, _ = small_dataset
        K = rbf_kernel_matrix(X, sigma=1.0)
        assert K.shape == (len(X), len(X))

    def test_diagonal_ones(self, small_dataset):
        X, _ = small_dataset
        K = rbf_kernel_matrix(X, sigma=1.0)
        assert np.allclose(np.diag(K), 1.0)

    def test_symmetry(self, small_dataset):
        X, _ = small_dataset
        K = rbf_kernel_matrix(X, sigma=1.0)
        assert np.allclose(K, K.T, atol=1e-10)

    def test_positive_semidefinite(self, small_dataset):
        """All eigenvalues must be >= 0 (PSD property)."""
        X, _ = small_dataset
        K = rbf_kernel_matrix(X, sigma=1.0)
        eigvals = np.linalg.eigvalsh(K)
        assert np.all(eigvals >= -1e-8)

    def test_agrees_with_scalar_kernel(self, rng):
        """Matrix entries must match the scalar rbf_kernel."""
        X = rng.standard_normal((10, 3))
        K = rbf_kernel_matrix(X, sigma=1.5)
        for i in range(10):
            for j in range(10):
                expected = rbf_kernel(X[i], X[j], 1.5)
                assert K[i, j] == pytest.approx(expected, abs=1e-8)

    def test_large_n_emits_resource_warning(self, rng):
        """rbf_kernel_matrix must emit a ResourceWarning when n > 5_000."""
        X = rng.standard_normal((5_001, 2))
        with pytest.warns(ResourceWarning):
            rbf_kernel_matrix(X, sigma=1.0)

    def test_small_n_no_warning(self, small_dataset):
        """No ResourceWarning should be emitted for n <= 5_000."""
        X, _ = small_dataset
        with warnings.catch_warnings():
            warnings.simplefilter("error", ResourceWarning)
            rbf_kernel_matrix(X, sigma=1.0)  # must not raise


# ---------------------------------------------------------------------------
# Exercise 2: Random Fourier Features
# ---------------------------------------------------------------------------

class TestSampleRffWeights:
    def test_shapes(self, rng):
        D, d = 200, 8
        omega, b = sample_rff_weights(D, d, sigma=1.0, rng=rng)
        assert omega.shape == (D, d)
        assert b.shape == (D,)

    def test_b_range(self, rng):
        """Phase shifts b must lie in [0, 2π]."""
        _, b = sample_rff_weights(500, 5, sigma=1.0, rng=rng)
        assert np.all(b >= 0.0)
        assert np.all(b <= 2 * math.pi + 1e-9)

    def test_sigma_scales_omega(self, rng):
        """Doubling sigma should roughly halve ||omega||."""
        om1, _ = sample_rff_weights(1000, 10, sigma=1.0, rng=np.random.default_rng(0))
        om2, _ = sample_rff_weights(1000, 10, sigma=2.0, rng=np.random.default_rng(0))
        # Std of each column should scale inversely with sigma
        assert np.std(om1) == pytest.approx(np.std(om2) * 2.0, rel=0.1)


class TestRffFeatures:
    def test_output_shape(self, small_dataset, rng):
        X, d = small_dataset
        D = 100
        omega, b = sample_rff_weights(D, d, sigma=1.0, rng=rng)
        Z = rff_features(X, omega, b)
        assert Z.shape == (len(X), D)

    def test_unbiased_approximation(self, rng):
        """E[z(x)ᵀz(y)] ≈ K(x,y) over many random draws."""
        d = 5
        x = rng.standard_normal(d)
        y = rng.standard_normal(d)
        k_exact = rbf_kernel(x, y, sigma=1.0)

        approxes = []
        for seed in range(200):
            g = np.random.default_rng(seed)
            om, b = sample_rff_weights(D=1000, d=d, sigma=1.0, rng=g)
            z_x = rff_features(x[None], om, b)[0]
            z_y = rff_features(y[None], om, b)[0]
            approxes.append(np.dot(z_x, z_y))

        assert np.mean(approxes) == pytest.approx(k_exact, abs=0.02)


class TestRffKernelApprox:
    def test_shape(self, small_dataset, rng):
        X, d = small_dataset
        omega, b = sample_rff_weights(200, d, 1.0, rng)
        K_approx = rff_kernel_approx(X, X, omega, b)
        assert K_approx.shape == (len(X), len(X))

    def test_approximation_quality(self, rng):
        """With D=2000, relative Frobenius error should be < 0.10.

        We use sigma=3.0 (≈ median-heuristic for d=10 N(0,1) data) so the
        kernel has meaningful off-diagonal structure to approximate.
        """
        n, d, sigma = 200, 10, 3.0
        X = rng.standard_normal((n, d))
        K_exact = rbf_kernel_matrix(X, sigma=sigma)
        omega, b = sample_rff_weights(D=2000, d=d, sigma=sigma, rng=rng)
        K_approx = rff_kernel_approx(X, X, omega, b)
        err = approximation_error(K_exact, K_approx)
        assert err < 0.10, f"RFF error {err:.4f} exceeds 0.10"


# ---------------------------------------------------------------------------
# Exercise 3: Orthogonal Random Features
# ---------------------------------------------------------------------------

class TestSampleOrfWeights:
    def test_shapes(self, rng):
        D, d = 150, 10
        omega, b = sample_orf_weights(D, d, sigma=1.0, rng=rng)
        assert omega.shape == (D, d)
        assert b.shape == (D,)

    def test_lower_variance_than_rff(self, rng):
        """ORF should have strictly lower std of approximation error than RFF."""
        n, d, sigma = 200, 10, 3.0
        X = rng.standard_normal((n, d))
        K_exact = rbf_kernel_matrix(X, sigma=sigma)
        D = 300
        errs_rff, errs_orf = [], []
        for seed in range(40):
            g = np.random.default_rng(seed)
            om, b = sample_rff_weights(D, d, sigma=sigma, rng=g)
            errs_rff.append(approximation_error(K_exact, rff_kernel_approx(X, X, om, b)))
            g = np.random.default_rng(seed)
            om, b = sample_orf_weights(D, d, sigma=sigma, rng=g)
            errs_orf.append(approximation_error(K_exact, rff_kernel_approx(X, X, om, b)))
        assert np.std(errs_orf) < np.std(errs_rff), (
            f"ORF std ({np.std(errs_orf):.4f}) should be < RFF std ({np.std(errs_rff):.4f})"
        )


class TestOrfFeatures:
    def test_shape(self, small_dataset, rng):
        X, d = small_dataset
        D = 80
        omega, b = sample_orf_weights(D, d, sigma=1.0, rng=rng)
        Z = orf_features(X, omega, b)
        assert Z.shape == (len(X), D)

    def test_approximation_quality(self, rng):
        """With D=2000, ORF relative error < 0.10 (same sigma as RFF test)."""
        n, d, sigma = 200, 10, 3.0
        X = rng.standard_normal((n, d))
        K_exact = rbf_kernel_matrix(X, sigma=sigma)
        omega, b = sample_orf_weights(D=2000, d=d, sigma=sigma, rng=rng)
        Z = orf_features(X, omega, b)
        err = approximation_error(K_exact, Z @ Z.T)
        assert err < 0.10, f"ORF error {err:.4f} exceeds 0.10"


# ---------------------------------------------------------------------------
# Exercise 4: Nyström Approximation
# ---------------------------------------------------------------------------

class TestSelectLandmarks:
    def test_shape_random(self, small_dataset, rng):
        X, _ = small_dataset
        lm = select_landmarks(X, m=15, strategy="random", rng=rng)
        assert lm.shape == (15, X.shape[1])

    def test_shape_stride(self, small_dataset):
        X, _ = small_dataset
        lm = select_landmarks(X, m=10, strategy="stride")
        assert lm.shape == (10, X.shape[1])

    def test_landmarks_are_rows_of_X(self, small_dataset, rng):
        """Every landmark must be an exact row of X."""
        X, _ = small_dataset
        lm = select_landmarks(X, m=20, strategy="random", rng=rng)
        for l in lm:
            assert any(np.allclose(l, x) for x in X)

    def test_invalid_strategy(self, small_dataset):
        X, _ = small_dataset
        with pytest.raises(ValueError):
            select_landmarks(X, m=5, strategy="kmeans")


class TestNystromFeatures:
    def test_shape(self, small_dataset, rng):
        X, _ = small_dataset
        m = 15
        lm = select_landmarks(X, m=m, strategy="random", rng=rng)
        Z = nystrom_features(X, lm, sigma=1.0)
        assert Z.shape == (len(X), m)

    def test_approximation_quality(self, rng):
        """With m=100 on n=200, relative error < 0.15 (using median-heuristic sigma)."""
        n, d, sigma = 200, 10, 3.0
        X = rng.standard_normal((n, d))
        K_exact = rbf_kernel_matrix(X, sigma=sigma)
        lm = select_landmarks(X, m=100, strategy="random", rng=rng)
        Z = nystrom_features(X, lm, sigma=sigma)
        err = approximation_error(K_exact, Z @ Z.T)
        assert err < 0.15, f"Nyström error {err:.4f} exceeds 0.15"


class TestNystromKernelApprox:
    def test_shape(self, small_dataset, rng):
        X, _ = small_dataset
        lm = select_landmarks(X, m=10, strategy="random", rng=rng)
        K_approx = nystrom_kernel_approx(X, X, lm, sigma=1.0)
        assert K_approx.shape == (len(X), len(X))

    def test_cross_matrix_shape(self, small_dataset, rng):
        X, _ = small_dataset
        X2 = rng.standard_normal((30, X.shape[1]))
        lm = select_landmarks(X, m=10, strategy="random", rng=rng)
        K_approx = nystrom_kernel_approx(X, X2, lm, sigma=1.0)
        assert K_approx.shape == (len(X), len(X2))


# ---------------------------------------------------------------------------
# Exercise 5: Kernel Ridge Regression
# ---------------------------------------------------------------------------

class TestKernelRidgeFit:
    def test_returns_vector(self, small_dataset):
        X, _ = small_dataset
        n = len(X)
        K = rbf_kernel_matrix(X, sigma=1.0)
        y = np.random.default_rng(0).standard_normal(n)
        alpha = kernel_ridge_fit(K, y, lam=0.1)
        assert alpha.shape == (n,)

    def test_residual_small_for_zero_lambda(self, small_dataset):
        """With lam→0, (K+λI)α ≈ y so K@α ≈ y."""
        X, _ = small_dataset
        K = rbf_kernel_matrix(X, sigma=1.0)
        y = np.random.default_rng(1).standard_normal(len(X))
        alpha = kernel_ridge_fit(K, y, lam=1e-6)
        residual = np.linalg.norm(K @ alpha - y) / np.linalg.norm(y)
        assert residual < 0.01, f"Residual {residual:.4f} too large"


class TestKernelRidgePredict:
    def test_shape(self, small_dataset, rng):
        X, _ = small_dataset
        n = len(X)
        K_train = rbf_kernel_matrix(X, sigma=1.0)
        y = rng.standard_normal(n)
        alpha = kernel_ridge_fit(K_train, y, lam=0.1)
        X_test = rng.standard_normal((15, X.shape[1]))
        K_test = np.exp(
            -np.sum((X_test[:, None] - X[None]) ** 2, axis=2) / 2.0
        )
        preds = kernel_ridge_predict(K_test, alpha)
        assert preds.shape == (15,)


class TestApproximationError:
    def test_identical_matrices(self, small_dataset):
        X, _ = small_dataset
        K = rbf_kernel_matrix(X, sigma=1.0)
        assert approximation_error(K, K) == pytest.approx(0.0, abs=1e-10)

    def test_zero_matrix(self, small_dataset):
        X, _ = small_dataset
        K = rbf_kernel_matrix(X, sigma=1.0)
        err = approximation_error(K, np.zeros_like(K))
        assert err == pytest.approx(1.0, abs=1e-8)

    def test_nonnegative(self, small_dataset, rng):
        X, _ = small_dataset
        K_exact = rbf_kernel_matrix(X, sigma=1.0)
        omega, b = sample_rff_weights(100, X.shape[1], 1.0, rng)
        K_approx = rff_kernel_approx(X, X, omega, b)
        assert approximation_error(K_exact, K_approx) >= 0.0


# ---------------------------------------------------------------------------
# Exercise 6: Benchmarking
# ---------------------------------------------------------------------------

class TestBenchmarkMethods:
    def test_returns_list_of_dicts(self):
        results = benchmark_methods(
            n_values=[100, 200], D_values=[20, 40], d=5, sigma=1.0
        )
        assert isinstance(results, list)
        assert len(results) > 0
        for r in results:
            assert isinstance(r, dict)
            assert "method" in r
            assert "n" in r
            assert "time_s" in r
            assert "memory_mb" in r
            assert "approx_error" in r

    def test_methods_present(self):
        results = benchmark_methods(
            n_values=[100], D_values=[20], d=5, sigma=1.0
        )
        methods = {r["method"] for r in results}
        assert "RFF" in methods
        assert "ORF" in methods
        assert "Nyström" in methods

    def test_time_positive(self):
        results = benchmark_methods(
            n_values=[100], D_values=[20], d=5, sigma=1.0
        )
        for r in results:
            assert r["time_s"] >= 0.0

    def test_plot_returns_figure(self):
        import matplotlib.pyplot as plt
        results = benchmark_methods(
            n_values=[100, 200], D_values=[20, 40], d=5, sigma=1.0
        )
        fig = plot_results(results)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
