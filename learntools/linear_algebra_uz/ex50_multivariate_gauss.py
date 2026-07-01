"""Hints and solutions — Dars 12.3: Ko'p O'zgaruvchili Gauss va Og'irlikli Kichik Kvadratlar."""
import numpy as np
from learntools.core import EqualityCheckProblem, ThoughtExperiment


class Q1(EqualityCheckProblem):
    """N(mu, Sigma) taqsimotidan namunalar oling."""
    _hints = ["np.random.multivariate_normal(mu, Sigma, size=n)."]
    _solution = "X = np.random.multivariate_normal(mu, Sigma, size=n)"

    def _do_check(self, X, mu, Sigma, n):
        if X.shape != (n, len(mu)):
            return f"X shakli ({n}, {len(mu)}) bo'lishi kerak."
        if not np.allclose(np.mean(X, axis=0), mu, atol=0.5):
            return f"Namunalar o'rtachasi mudan farq qilmoqda: {np.mean(X, axis=0)}"
        return True


class Q2(EqualityCheckProblem):
    """Ko'p o'zgaruvchili Gauss zichligi: p(x) = exp(-0.5*(x-mu)ᵀΣ⁻¹(x-mu)) / sqrt(...)."""
    _hints = [
        "scipy.stats.multivariate_normal(mean=mu, cov=Sigma).pdf(x).",
    ]
    _solution = (
        "from scipy.stats import multivariate_normal\n"
        "density = multivariate_normal(mean=mu, cov=Sigma).pdf(x)"
    )

    def _do_check(self, density, x, mu, Sigma):
        from scipy.stats import multivariate_normal
        expected = multivariate_normal(mean=mu, cov=Sigma).pdf(x)
        if not np.isclose(density, expected, rtol=1e-6):
            return f"p(x) = {expected:.6e}, siz {density:.6e} berdingiz."
        return True


class Q3(EqualityCheckProblem):
    """Marginal taqsimot: N(mu_1, Sigma_11) — birinchi komponent."""
    _hints = [
        "Ko'p o'zgaruvchili Gauss marginal taqsimoti ham normal: N(mu[0], Sigma[0,0]).",
    ]
    _solution = "mu_1 = mu[0]; sigma_1_sq = Sigma[0, 0]"

    def _do_check(self, mu_1, sigma_1_sq, mu, Sigma):
        if not np.isclose(mu_1, mu[0], rtol=1e-8):
            return f"Marginal o'rtacha: {mu[0]}, siz {mu_1} berdingiz."
        if not np.isclose(sigma_1_sq, Sigma[0, 0], rtol=1e-8):
            return f"Marginal dispersiya: {Sigma[0,0]}, siz {sigma_1_sq} berdingiz."
        return True


class Q4(EqualityCheckProblem):
    """Shartli taqsimot: p(x2|x1) — Gauss shartli formula."""
    _hints = [
        "mu_2|1 = mu2 + Sigma_21 @ inv(Sigma_11) @ (x1 - mu1).",
        "Sigma_2|1 = Sigma_22 - Sigma_21 @ inv(Sigma_11) @ Sigma_12.",
    ]
    _solution = (
        "mu_cond = mu[1] + Sigma[1,0]/Sigma[0,0] * (x1 - mu[0])\n"
        "sigma_cond = Sigma[1,1] - Sigma[1,0]**2 / Sigma[0,0]"
    )

    def _do_check(self, mu_cond, sigma_cond, x1, mu, Sigma):
        exp_mu = mu[1] + Sigma[1, 0] / Sigma[0, 0] * (x1 - mu[0])
        exp_sig = Sigma[1, 1] - Sigma[1, 0] ** 2 / Sigma[0, 0]
        if not np.isclose(mu_cond, exp_mu, rtol=1e-6):
            return f"Shartli mu: {exp_mu:.6f}, siz {mu_cond:.6f} berdingiz."
        if not np.isclose(sigma_cond, exp_sig, rtol=1e-6):
            return f"Shartli sigma^2: {exp_sig:.6f}, siz {sigma_cond:.6f} berdingiz."
        return True


class Q5(EqualityCheckProblem):
    """Og'irlikli kichik kvadratlar (WLS): min sum(w_i*(y_i - x_iᵀb)^2)."""
    _hints = [
        "W = diag(w). Normal tenglama: XᵀWX b = XᵀWy.",
        "b_wls = inv(X.T @ W @ X) @ X.T @ W @ y.",
    ]
    _solution = (
        "W = np.diag(w)\n"
        "b_wls = np.linalg.solve(X.T @ W @ X, X.T @ W @ y)"
    )

    def _do_check(self, b_wls, X, y, w):
        W = np.diag(w)
        expected = np.linalg.solve(X.T @ W @ X, X.T @ W @ y)
        if not np.allclose(b_wls, expected, atol=1e-6):
            return "b_wls = (XᵀWX)⁻¹XᵀWy formulasini tekshiring."
        return True


class Q6(EqualityCheckProblem):
    """MLE uchun kovariatsiya matritsasini baholang: Sigma_hat = (1/n) Xc.T @ Xc."""
    _hints = ["Sigma_hat = Xc.T @ Xc / n. OLS + Gauss shovqin uchun MLE."]
    _solution = "Xc = X - X.mean(axis=0); Sigma_hat = Xc.T @ Xc / len(X)"

    def _do_check(self, Sigma_hat, X):
        Xc = X - X.mean(axis=0)
        expected = Xc.T @ Xc / len(X)
        if not np.allclose(Sigma_hat, expected, atol=1e-8):
            return "Sigma_hat = Xc.T @ Xc / n."
        return True


class C1_Q1(EqualityCheckProblem):
    """Kalman filtri: predict va update qadamlari."""
    _hints = [
        "Predict: x_pred = F @ x, P_pred = F @ P @ F.T + Q.",
        "Update: K = P_pred @ H.T @ inv(H @ P_pred @ H.T + R); x = x_pred + K@(z - H@x_pred).",
    ]
    _solution = (
        "# Predict\n"
        "x_pred = F @ x; P_pred = F @ P @ F.T + Q\n"
        "# Update\n"
        "K = P_pred @ H.T @ np.linalg.inv(H @ P_pred @ H.T + R)\n"
        "x_new = x_pred + K @ (z - H @ x_pred)\n"
        "P_new = (np.eye(len(x)) - K @ H) @ P_pred"
    )

    def _do_check(self, x_new, F, H, Q, R, x, P, z):
        x_pred = F @ x
        P_pred = F @ P @ F.T + Q
        K = P_pred @ H.T @ np.linalg.inv(H @ P_pred @ H.T + R)
        expected = x_pred + K @ (z - H @ x_pred)
        if not np.allclose(x_new, expected, atol=1e-6):
            return "Kalman update: x_new = x_pred + K(z - Hx_pred)."
        return True


class C2_Q1(ThoughtExperiment):
    """Ko'p o'zgaruvchili Gauss nima uchun MLda markaziy o'rinda?"""
    _hints = [
        "Gauss: MaxEnt (maksimal entropiya), CLT (markaziy limit teoremasi).",
    ]
    _solution = (
        "Ko'p o'zgaruvchili Gauss ahamiyati:\n\n"
        "1) Markaziy limit teoremasi: ko'p mustaqil o'zgaruvchilar yig'indisi → Gauss.\n"
        "2) Maksimal entropiya: berilgan mu, Sigma uchun eng 'tartibsiz' taqsimot.\n"
        "3) MLE = OLS: Gauss shovqin uchun log-likelihood → normal tenglama.\n"
        "4) Bayesian inference: prior Gauss, likelihood Gauss → posterior Gauss (conjugate).\n"
        "5) Gaussian Processes: funksiyalar ustidagi taqsimot — ML regressor/klassifikator.\n"
        "6) Kalman filtri: chiziqli Gauss dinamik sistema — optimal filtrlash.\n"
        "Kovariatsiya matritsasi Σ = Gauss taqsimotining 'shakli'."
    )
