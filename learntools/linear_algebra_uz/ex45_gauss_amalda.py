"""Hints and solutions — Dars 11.1: Gauss Eliminatsiyasi Amalda."""
import numpy as np
from learntools.linear_algebra_uz.base import UzCheckProblem, ThoughtExperiment


class Q1(UzCheckProblem):
    """To'liq pivoting bilan Gauss eliminatsiyasi."""
    _hints = [
        "scipy.linalg.lu(A) — P, L, U yoyilmasi (qisman pivot bilan).",
        "P @ A = L @ U.",
    ]
    _solution = "from scipy.linalg import lu; P, L, U = lu(A)"

    def _do_check(self, P, L, U, A):
        from scipy.linalg import lu
        exp_P, exp_L, exp_U = lu(A)
        if not np.allclose(P @ A, L @ U, atol=1e-8):
            return "PA = LU bo'lishi kerak."
        if not np.allclose(P @ A, exp_P @ A, atol=1e-8):
            return "P matritsa noto'g'ri."
        return True


class Q2(UzCheckProblem):
    """Pivot tanlash sababini ko'rsating: kichik pivot → katta xato."""
    _hints = [
        "A = [[1e-20, 1],[1, 1]]. Pivot = 1e-20. Xato: np.linalg.solve vs to'g'ri yechim.",
    ]
    _solution = (
        "A = np.array([[1e-20, 1.0],[1.0, 1.0]])\n"
        "b = np.array([1.0, 2.0])\n"
        "x = np.linalg.solve(A, b)\n"
        "relative_error = np.linalg.norm(A @ x - b) / np.linalg.norm(b)"
    )

    def _do_check(self, relative_error, A, b):
        x = np.linalg.solve(A, b)
        expected = np.linalg.norm(A @ x - b) / np.linalg.norm(b)
        if not np.isclose(relative_error, expected, rtol=0.1):
            return f"Nisbiy xato: {expected:.2e}, siz {relative_error:.2e} berdingiz."
        return True


class Q3(UzCheckProblem):
    """Orqaga substitusiya (back substitution) ni qo'lda bajaring."""
    _hints = [
        "U x = b: x[n-1] = b[n-1]/U[n-1,n-1]; x[i] = (b[i] - sum U[i,j]*x[j]) / U[i,i].",
    ]
    _solution = (
        "n = len(b)\n"
        "x = np.zeros(n)\n"
        "for i in range(n-1, -1, -1):\n"
        "    x[i] = (b[i] - np.dot(U[i, i+1:], x[i+1:])) / U[i, i]"
    )

    def _do_check(self, x, U, b):
        expected = np.linalg.solve(U, b)
        if not np.allclose(x, expected, atol=1e-8):
            return "Orqaga substitusiya formulasini tekshiring."
        return True


class Q4(UzCheckProblem):
    """LU bilan bir nechta o'ng tomonlarni yeching."""
    _hints = [
        "LU yoyilmasi bir marta, keyin har B ustuni uchun L y = b va U x = y.",
        "scipy.linalg.lu_factor va lu_solve dan foydalaning.",
    ]
    _solution = (
        "from scipy.linalg import lu_factor, lu_solve\n"
        "lu, piv = lu_factor(A)\n"
        "X = np.column_stack([lu_solve((lu, piv), B[:,j]) for j in range(B.shape[1])])"
    )

    def _do_check(self, X, A, B):
        expected = np.linalg.solve(A, B)
        if not np.allclose(X, expected, atol=1e-6):
            return "X ni tekshiring: A @ X = B bo'lishi kerak."
        return True


class Q5(UzCheckProblem):
    """Sanoat darajasidagi tizim uchun siyrak matritsa (sparse) yeching."""
    _hints = [
        "scipy.sparse.linalg.spsolve — katta siyrak sistemalar uchun.",
        "scipy.sparse.lil_matrix yoki csr_matrix bilan yarating.",
    ]
    _solution = (
        "from scipy.sparse import diags\n"
        "from scipy.sparse.linalg import spsolve\n"
        "A_sp = diags([-1, 2, -1], [-1, 0, 1], shape=(n, n), format='csr')\n"
        "x = spsolve(A_sp, b)"
    )

    def _do_check(self, x, A, b):
        expected = np.linalg.solve(A.toarray() if hasattr(A, 'toarray') else A, b)
        if not np.allclose(x, expected, atol=1e-6):
            return "Yechimni tekshiring: A @ x ≈ b bo'lishi kerak."
        return True


class Q6(UzCheckProblem):
    """Floating-point arifmetikasida yaxlit xato (roundoff)."""
    _hints = [
        "eps = np.finfo(float).eps — mashin aniqligi (~2.2e-16).",
        "Xato ≤ cond(A) * eps * ||b|| / ||b|| — konditsion son xatoni kuchaytiradi.",
    ]
    _solution = "eps = np.finfo(float).eps; error_bound = np.linalg.cond(A) * eps"

    def _do_check(self, error_bound, A):
        eps = np.finfo(float).eps
        expected = np.linalg.cond(A) * eps
        if not np.isclose(error_bound, expected, rtol=0.01):
            return f"Xato chegarasi: cond(A)*eps = {expected:.4e}, siz {error_bound:.4e} berdingiz."
        return True


class C1_Q1(UzCheckProblem):
    """LAPACK DGESV algoritmini scipy bilan solishtiring."""
    _hints = [
        "np.linalg.solve A @ x = b uchun LAPACK DGESV ni chaqiradi.",
        "scipy.linalg.solve ham xuddi shunday, lekin qo'shimcha parametrlar bilan.",
    ]
    _solution = (
        "x_np = np.linalg.solve(A, b)\n"
        "from scipy.linalg import solve\n"
        "x_sp = solve(A, b)\n"
        "same = np.allclose(x_np, x_sp)"
    )

    def _do_check(self, same, A, b):
        x1 = np.linalg.solve(A, b)
        from scipy.linalg import solve
        x2 = solve(A, b)
        expected = np.allclose(x1, x2)
        if same != expected:
            return f"np.linalg.solve va scipy.linalg.solve bir xil? {expected}"
        return True


class C2_Q1(ThoughtExperiment):
    """Nima uchun pivot tanlash (pivoting) sonli barqarorlik uchun zarur?"""
    _hints = [
        "Kichik pivot → katta ko'paytuvchi → katta yaxlit xato.",
        "Qisman pivot: har qadamda eng katta elementni tanlash.",
    ]
    _solution = (
        "Pivot tanlash zarurligi:\n\n"
        "Kichik pivot (masalan, ε ≈ 10⁻²⁰) bo'lganda:\n"
        "ko'paytuvchi l = a_{ij}/a_{kk} ≈ 10²⁰ — juda katta.\n"
        "Keyingi satr elementlari bu katta songa ko'paytiriladi,\n"
        "natijada float aniqligi yo'qoladi (catastrophic cancellation).\n\n"
        "Qisman pivot (partial pivoting): har qadamda ustundagi eng katta elementni\n"
        "diagonal pozitsiyaga almashtiramiz (permutatsiya P).\n"
        "Bu l ≤ 1 kafolatlaydi — barqarorlik ta'minlanadi.\n\n"
        "LAPACK DGESV: qisman pivoting + LU = sanoat standarti (40+ yil).\n"
        "To'liq pivot (full pivoting) yanada barqaror, lekin sekinroq."
    )
