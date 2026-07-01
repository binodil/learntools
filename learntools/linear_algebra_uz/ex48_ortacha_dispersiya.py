"""Hints and solutions — Dars 12.1: O'rtacha, Dispersiya va Ehtimollik."""
import numpy as np
from learntools.linear_algebra_uz.base import UzCheckProblem, ThoughtExperiment


class Q1(UzCheckProblem):
    """O'rtacha (mean) va dispersiyani (variance) hisoblang."""
    _hints = ["np.mean(x), np.var(x) yoki np.std(x)**2."]
    _solution = "mu = np.mean(x); sigma2 = np.var(x)"

    def _do_check(self, mu, sigma2, x):
        if not np.isclose(mu, np.mean(x), rtol=1e-8):
            return f"O'rtacha: {np.mean(x):.6f}, siz {mu:.6f} berdingiz."
        if not np.isclose(sigma2, np.var(x), rtol=1e-8):
            return f"Dispersiya: {np.var(x):.6f}, siz {sigma2:.6f} berdingiz."
        return True


class Q2(UzCheckProblem):
    """Standart og'ish (std) va normallashtirish: z = (x - mu)/sigma."""
    _hints = ["sigma = np.std(x). z = (x - mu) / sigma."]
    _solution = "sigma = np.std(x); z = (x - np.mean(x)) / sigma"

    def _do_check(self, z, x):
        mu = np.mean(x); sigma = np.std(x)
        expected = (x - mu) / sigma
        if not np.allclose(z, expected, atol=1e-8):
            return "z = (x - mean(x)) / std(x) formulasini tekshiring."
        if not np.isclose(np.mean(z), 0, atol=1e-8):
            return f"Normallashtirilgan z o'rtachasi 0 bo'lishi kerak. Got: {np.mean(z):.4e}"
        if not np.isclose(np.std(z), 1, rtol=1e-6):
            return f"Normallashtirilgan z STD 1 bo'lishi kerak. Got: {np.std(z):.6f}"
        return True


class Q3(UzCheckProblem):
    """Diskret taqsimot uchun kutilgan qiymat E[X] = sum(x_i * p_i)."""
    _hints = ["E[X] = np.dot(values, probs)."]
    _solution = "E_X = np.dot(values, probs)"

    def _do_check(self, E_X, values, probs):
        expected = np.dot(values, probs)
        if not np.isclose(E_X, expected, rtol=1e-8):
            return f"E[X] = {expected:.6f}, siz {E_X:.6f} berdingiz."
        return True


class Q4(UzCheckProblem):
    """Bernoulli taqsimoti: P(X=1) = p. E[X] = p, Var(X) = p(1-p)."""
    _hints = ["E[X] = p. Var(X) = p*(1-p)."]
    _solution = "E = p; Var = p * (1 - p)"

    def _do_check(self, E, Var, p):
        if not np.isclose(E, p, rtol=1e-8):
            return f"E[X] = p = {p}, siz {E} berdingiz."
        if not np.isclose(Var, p * (1 - p), rtol=1e-8):
            return f"Var = p(1-p) = {p*(1-p):.6f}, siz {Var:.6f} berdingiz."
        return True


class Q5(UzCheckProblem):
    """Bayes teoremasi: P(A|B) = P(B|A)*P(A) / P(B)."""
    _hints = ["P(A|B) = P(B|A)*P(A) / (P(B|A)*P(A) + P(B|notA)*P(notA))."]
    _solution = "P_A_given_B = (P_B_given_A * P_A) / (P_B_given_A*P_A + P_B_given_notA*(1-P_A))"

    def _do_check(self, P_A_given_B, P_B_given_A, P_A, P_B_given_notA):
        P_B = P_B_given_A * P_A + P_B_given_notA * (1 - P_A)
        expected = P_B_given_A * P_A / P_B
        if not np.isclose(P_A_given_B, expected, rtol=1e-6):
            return f"P(A|B) = {expected:.6f}, siz {P_A_given_B:.6f} berdingiz."
        return True


class Q6(UzCheckProblem):
    """Markaziy limit teoremasi: n ta namunaning o'rtachasi normal taqsimotga yaqinlashadi."""
    _hints = [
        "n=1000 ta namunaning o'rtachasi: mu_bar = np.mean(samples, axis=1). STD ≈ sigma/sqrt(n).",
    ]
    _solution = (
        "np.random.seed(42)\n"
        "samples = np.random.exponential(2, size=(1000, n))\n"
        "means = np.mean(samples, axis=1)\n"
        "std_means = np.std(means)"
    )

    def _do_check(self, std_means, sigma, n):
        expected = sigma / np.sqrt(n)
        if not np.isclose(std_means, expected, rtol=0.1):
            return f"STD(o'rtachalar) ≈ sigma/sqrt(n) = {expected:.4f}, siz {std_means:.4f} berdingiz."
        return True


class C1_Q1(UzCheckProblem):
    """Chiziqli regressiyada MLE = OLS: nima uchun?"""
    _hints = [
        "y = Xb + e, e ~ N(0, sigma^2 I). Log-likelihood ni maksimizatsiya → ||y - Xb||^2 minimizatsiya.",
    ]
    _solution = (
        "# OLS: b_hat = (X.T @ X)^{-1} X.T @ y\n"
        "b_hat = np.linalg.lstsq(X, y, rcond=None)[0]\n"
        "y_pred = X @ b_hat\n"
        "mse = np.mean((y - y_pred)**2)"
    )

    def _do_check(self, b_hat, X, y):
        expected = np.linalg.lstsq(X, y, rcond=None)[0]
        if not np.allclose(b_hat, expected, atol=1e-6):
            return "b_hat = (XᵀX)⁻¹Xᵀy — OLS / MLE ekvivalentligi."
        return True


class C2_Q1(ThoughtExperiment):
    """Ehtimollik va chiziqli algebra qanday bog'liq?"""
    _hints = [
        "Kutilgan qiymat = proyeksiya, dispersiya = norma kvadrat, Bayes = matritsa inversiya.",
    ]
    _solution = (
        "Ehtimollik nazariyasida chiziqli algebra:\n\n"
        "1) Kutilgan qiymat: E[f(X)] = ∫ f(x)p(x)dx — chiziqli funksional.\n"
        "2) Dispersiya: Var(X) = E[(X-mu)²] — L²-norma kvadrati.\n"
        "3) Bayes yangilash: posterior ∝ likelihood × prior — matritsaviy inversiya.\n"
        "4) MLE = OLS (Gauss shovqin uchun): normal tenglamalarga tushadi.\n"
        "5) Kovariatsiya matritsasi simmetrik musbat yarim aniq — spektral teorema qo'llanadi."
    )
