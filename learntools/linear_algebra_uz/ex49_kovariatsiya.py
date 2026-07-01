"""Hints and solutions — Dars 12.2: Kovariatsiya Matritsalari va Birgalikdagi Ehtimollik."""
import numpy as np
from learntools.core import EqualityCheckProblem, ThoughtExperiment


class Q1(EqualityCheckProblem):
    """Kovariatsiya matritsasini hisoblang: Σ = (1/n) Xc.T @ Xc."""
    _hints = ["X ni markazlashtiring: Xc = X - mean. Keyin Sigma = Xc.T @ Xc / n."]
    _solution = "Xc = X - X.mean(axis=0); Sigma = Xc.T @ Xc / len(X)"

    def _do_check(self, Sigma, X):
        Xc = X - X.mean(axis=0)
        expected = Xc.T @ Xc / len(X)
        if not np.allclose(Sigma, expected, atol=1e-8):
            return "Sigma = Xc.T @ Xc / n. Avval markazlashtiring."
        if not np.allclose(Sigma, Sigma.T, atol=1e-10):
            return "Kovariatsiya matritsasi simmetrik bo'lishi kerak."
        return True


class Q2(EqualityCheckProblem):
    """np.cov dan foydalanib kovariatsiya matritsasini hisoblang."""
    _hints = ["np.cov(X.T) — X ustunlari o'zgaruvchilar bo'lsa."]
    _solution = "Sigma = np.cov(X.T)"

    def _do_check(self, Sigma, X):
        expected = np.cov(X.T)
        if not np.allclose(Sigma, expected, atol=1e-8):
            return "np.cov(X.T) dan foydalaning (X: n_samples × n_features)."
        return True


class Q3(EqualityCheckProblem):
    """Korrelyatsiya koeffitsiyentini hisoblang: r = Cov(X,Y) / (std_X * std_Y)."""
    _hints = ["np.corrcoef(x, y)[0,1] yoki Cov(x,y)/(std(x)*std(y))."]
    _solution = "r = np.corrcoef(x, y)[0, 1]"

    def _do_check(self, r, x, y):
        expected = np.corrcoef(x, y)[0, 1]
        if not np.isclose(r, expected, rtol=1e-6):
            return f"r = {expected:.6f}, siz {r:.6f} berdingiz."
        return True


class Q4(EqualityCheckProblem):
    """Kovariatsiya matritsasi PSD ekanini tekshiring (barcha eig >= 0)."""
    _hints = ["np.linalg.eigvalsh(Sigma) — barcha qiymatlar >= 0 bo'lsa PSD."]
    _solution = "is_psd = np.all(np.linalg.eigvalsh(Sigma) >= -1e-10)"

    def _do_check(self, is_psd, Sigma):
        vals = np.linalg.eigvalsh(Sigma)
        expected = np.all(vals >= -1e-10)
        if is_psd != expected:
            return f"Xususiy qiymatlar: {vals}. PSD? {expected}"
        return True


class Q5(EqualityCheckProblem):
    """PCA: kovariatsiya matritsasining xususiy vektorlari — asosiy yo'nalishlar."""
    _hints = [
        "vals, vecs = np.linalg.eigh(Sigma). Eng katta xususiy qiymatga mos vektor — 1-komponent.",
    ]
    _solution = "vals, vecs = np.linalg.eigh(Sigma); pc1 = vecs[:, -1]  # eng katta lambda"

    def _do_check(self, pc1, Sigma):
        vals, vecs = np.linalg.eigh(Sigma)
        exp_pc1 = vecs[:, -1]
        if not (np.allclose(pc1, exp_pc1, atol=1e-6) or np.allclose(pc1, -exp_pc1, atol=1e-6)):
            return "Birinchi asosiy komponent — eng katta xususiy qiymatga mos vektor."
        return True


class Q6(EqualityCheckProblem):
    """Mahalanobis masofasi: d² = (x-mu)ᵀ Σ⁻¹ (x-mu)."""
    _hints = [
        "Sigma_inv = np.linalg.inv(Sigma). d2 = delta.T @ Sigma_inv @ delta, delta = x - mu.",
    ]
    _solution = "delta = x - mu; Sigma_inv = np.linalg.inv(Sigma); d2 = delta @ Sigma_inv @ delta"

    def _do_check(self, d2, x, mu, Sigma):
        delta = x - mu
        Sigma_inv = np.linalg.inv(Sigma)
        expected = delta @ Sigma_inv @ delta
        if not np.isclose(d2, expected, rtol=1e-6):
            return f"d² = {expected:.6f}, siz {d2:.6f} berdingiz."
        return True


class C1_Q1(EqualityCheckProblem):
    """Cholesky bilan korrelyatsiyalangan ma'lumot generatsiyasi."""
    _hints = [
        "Sigma = [[1, rho],[rho, 1]]. L = cholesky(Sigma). X = (L @ Z.T).T, Z ~ N(0,I).",
    ]
    _solution = (
        "np.random.seed(42)\n"
        "L = np.linalg.cholesky(Sigma)\n"
        "Z = np.random.randn(n, Sigma.shape[0])\n"
        "X = (L @ Z.T).T"
    )

    def _do_check(self, X, Sigma, n):
        if X.shape != (n, Sigma.shape[0]):
            return f"X shakli ({n}, {Sigma.shape[0]}) bo'lishi kerak."
        emp_corr = np.corrcoef(X.T)
        if not np.allclose(np.sign(emp_corr), np.sign(Sigma), atol=0.5):
            return "Korrelyatsiya tuzilishi Sigma ga mos kelmayapti."
        return True


class C2_Q1(ThoughtExperiment):
    """Kovariatsiya matritsasi statistika va MLda nima uchun markaziy o'rinda?"""
    _hints = [
        "Kovariatsiya: PCA, LDA, Gauss taqsimot, Kalman filtri.",
    ]
    _solution = (
        "Kovariatsiya matritsasi Σ markaziy ahamiyati:\n\n"
        "1) PCA: Σ xususiy vektorlari — asosiy yo'nalishlar (o'lcham kamaytirish).\n"
        "2) LDA: sinf ichidagi va sinf orasidagi kovariatsiya — diskriminant tahlil.\n"
        "3) Gauss taqsimot: N(μ, Σ) to'liq Σ bilan aniqlanadi.\n"
        "4) Kalman filtri: Px — xato kovariatsiyasi, yangilanish qoidasi.\n"
        "5) Portfolio optimallashtirish (Markowitz): risk = wᵀΣw.\n"
        "6) Bayes chiziqli regressiya: prior va posterior normal — Σ orqali.\n"
        "Σ simmetrik musbat yarim aniq → Cholesky, SVD, spektral teorema qo'llanadi."
    )
