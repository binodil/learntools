"""Hints and solutions — Dars 6.5: Musbat Aniq Matritsalar."""
import numpy as np
from learntools.core import EqualityCheckProblem, ThoughtExperiment


class Q1(EqualityCheckProblem):
    """Musbat aniqlikni xususiy qiymatlar orqali tekshiring."""
    _hints = [
        "Musbat aniq (PD): barcha xususiy qiymatlar > 0.",
        "np.all(np.linalg.eigh(A)[0] > 0)",
    ]
    _solution = "is_pd = np.all(np.linalg.eigh(A)[0] > 0)"

    def _do_check(self, is_pd, A):
        if not np.allclose(A, A.T):
            return "A simmetrik bo'lishi kerak."
        expected = np.all(np.linalg.eigh(A)[0] > 1e-10)
        if is_pd != expected:
            return f"Xususiy qiymatlar: {np.linalg.eigh(A)[0]}. Barcha > 0? {expected}"
        return True


class Q2(EqualityCheckProblem):
    """xᵀAx > 0 ni tekshiring (musbat aniqlikning ta'rifi)."""
    _hints = [
        "xᵀAx = x @ A @ x. Har qanday nol bo'lmagan x uchun bu musbat bo'lishi kerak.",
    ]
    _solution = "quad_form = x @ A @ x; is_positive = quad_form > 0"

    def _do_check(self, quad_form, A, x):
        expected = x @ A @ x
        if not np.isclose(quad_form, expected, atol=1e-8):
            return f"xᵀAx = {expected:.6f}, siz {quad_form:.6f} berdingiz."
        return True


class Q3(EqualityCheckProblem):
    """Cholesky yoyilmasi: A = LLᵀ."""
    _hints = [
        "np.linalg.cholesky(A) — pastki uchburchak L ni beradi, A = L @ L.T.",
        "Cholesky faqat musbat aniq matritsalar uchun mavjud.",
    ]
    _solution = "L = np.linalg.cholesky(A)"

    def _do_check(self, L, A):
        try:
            expected = np.linalg.cholesky(A)
        except np.linalg.LinAlgError:
            return "A musbat aniq emas — Cholesky mavjud emas."
        if not np.allclose(L, expected, atol=1e-8):
            return "L = cholesky(A) formulasini tekshiring."
        if not np.allclose(L @ L.T, A, atol=1e-8):
            return "L @ Lᵀ = A bo'lishi kerak."
        return True


class Q4(EqualityCheckProblem):
    """Cholesky bilan Ax=b yechish."""
    _hints = [
        "A = LLᵀ, shuning uchun LLᵀx = b. Avval Ly = b, keyin Lᵀx = y.",
        "scipy.linalg.cho_solve bilan ham ishlatsa bo'ladi.",
    ]
    _solution = (
        "L = np.linalg.cholesky(A)\n"
        "y = np.linalg.solve(L, b)\n"
        "x = np.linalg.solve(L.T, y)"
    )

    def _do_check(self, x, A, b):
        expected = np.linalg.solve(A, b)
        if not np.allclose(x, expected, atol=1e-6):
            return f"Kutilgan yechim: {expected}, siz {x} berdingiz."
        return True


class Q5(EqualityCheckProblem):
    """Musbat aniq bo'lish shartlari: barcha yetakchi minorlar > 0."""
    _hints = [
        "Sylvester mezoni: A_{1×1}, A_{2×2}, A_{3×3} determinantlari > 0.",
        "[np.linalg.det(A[:k+1,:k+1]) > 0 for k in range(n)] — barcha > 0 bo'lsa PD.",
    ]
    _solution = "is_pd = all(np.linalg.det(A[:k+1, :k+1]) > 0 for k in range(A.shape[0]))"

    def _do_check(self, is_pd, A):
        expected = all(np.linalg.det(A[:k + 1, :k + 1]) > 0 for k in range(A.shape[0]))
        if is_pd != expected:
            minors = [np.linalg.det(A[:k + 1, :k + 1]) for k in range(A.shape[0])]
            return f"Yetakchi minorlar: {[round(m,4) for m in minors]}. Barcha > 0? {expected}"
        return True


class Q6(EqualityCheckProblem):
    """AᵀA musbat aniq ekanini tekshiring (A to'liq rangda bo'lsa)."""
    _hints = [
        "A to'liq ustun rangida bo'lsa (rank = n), AᵀA musbat aniq.",
        "Tekshirish: np.linalg.matrix_rank(A) == A.shape[1] va np.all(eigh(A.T@A)[0] > 0).",
    ]
    _solution = "S = A.T @ A; is_pd = np.linalg.matrix_rank(A) == A.shape[1]"

    def _do_check(self, is_pd, A):
        full_col_rank = np.linalg.matrix_rank(A) == A.shape[1]
        if is_pd != full_col_rank:
            return f"A ning rangi {np.linalg.matrix_rank(A)}, ustunlar soni {A.shape[1]}. AᵀA PD: {full_col_rank}"
        return True


class C1_Q1(EqualityCheckProblem):
    """Cholesky bilan kovariatsiya matritsasidan namunaviy vektor hosil qiling."""
    _hints = [
        "Σ = LLᵀ bo'lsa, x = L @ z (z ~ N(0,I)) → x ~ N(0,Σ).",
        "L = np.linalg.cholesky(Sigma); samples = L @ np.random.randn(n, N_samples)",
    ]
    _solution = (
        "L = np.linalg.cholesky(Sigma)\n"
        "np.random.seed(42)\n"
        "z = np.random.randn(Sigma.shape[0], N)\n"
        "samples = L @ z  # shape: (n, N)"
    )

    def _do_check(self, samples, Sigma, N):
        n = Sigma.shape[0]
        if samples.shape != (n, N):
            return f"samples shakli ({n}, {N}) bo'lishi kerak, siz {samples.shape} berdingiz."
        emp_cov = np.cov(samples)
        # Rough check — correlation structure should match
        if not np.allclose(np.sign(emp_cov), np.sign(Sigma), atol=0.5):
            return "Namunalar kovariatsiya matritsasi Sigma ga mos kelmayapti."
        return True


class C2_Q1(ThoughtExperiment):
    """Nima uchun Cholesky yoyilmasi musbat aniqlik uchun muhim?"""
    _hints = [
        "Cholesky LU dan 2x tezroq, chunki simmetriyadan foydalanadi.",
        "Moliya, statistika, ML da kovariatsiya matritsalari PD bo'ladi.",
    ]
    _solution = (
        "Cholesky yoyilmasi A = LLᵀ musbat aniq matritsalar uchun:\n"
        "1) Samaradorlik: LU dan ~2x tezroq (simmetriya tufayli faqat n³/6 amal).\n"
        "2) Namunaviy olish: Gauss taqsimoti uchun samples = L @ randn().\n"
        "3) Barqarorlik: Gauss eliminatsiyasidan aniqroq.\n"
        "4) Musbat aniqlik testi: agar Cholesky muvaffaqiyatli bo'lsa — PD.\n"
        "5) Kalman filtri, Bayes metodi, kovariatsiya matritsalari — barchasi Cholesky ishlatadi."
    )
