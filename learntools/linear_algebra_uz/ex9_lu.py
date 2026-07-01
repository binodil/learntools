"""Hints and solutions — Dars 2.6: Eliminatsiya = Parchalash: A = LU."""
import numpy as np
from learntools.core import EqualityCheckProblem, ThoughtExperiment


class Q1(EqualityCheckProblem):
    """scipy bilan LU parchalash."""
    _hints = [
        "from scipy.linalg import lu",
        "P, L, U = lu(A) → PA = LU",
    ]
    _solution = (
        "from scipy.linalg import lu\n"
        "P, L, U = lu(A)\n"
        "# Tekshirish: P @ A ≈ L @ U"
    )

    def _do_check(self, P, L, U, A):
        if not np.allclose(P @ A, L @ U, atol=1e-8):
            return "PA ≠ LU."
        if not np.allclose(np.tril(L), L):
            return "L pastki uchburchak emas."
        if not np.allclose(np.triu(U), U):
            return "U yuqori uchburchak emas."
        return True


class Q2(EqualityCheckProblem):
    """LU dan foydalanib Ax=b yechimini toping."""
    _hints = [
        "PA = LU → PAx = Pb → L(Ux) = Pb",
        "1: Ly = Pb → y topamiz (forward substitution)",
        "2: Ux = y → x topamiz (back substitution)",
    ]
    _solution = (
        "from scipy.linalg import lu, solve_triangular\n"
        "P, L, U = lu(A)\n"
        "Pb = P @ b\n"
        "y = solve_triangular(L, Pb, lower=True)\n"
        "x = solve_triangular(U, y)"
    )

    def _do_check(self, x, A, b):
        if not np.allclose(A @ x, b, atol=1e-8):
            return f"A@x ≠ b. A@x = {A@x}"
        return True


class Q3(ThoughtExperiment):
    """L matritsa nima saqlab qoladi?"""
    _hints = [
        "L — pastki uchburchak, diagonal 1 lar.",
        "L ning pastki uchburchak qismi — eliminatsiya multiplikatorlari (mij).",
    ]
    _solution = (
        "L matritsasida:\n"
        "• Diagonal: hammasi 1\n"
        "• Pastki uchburchak: L[i,j] = mij (i-satrni j-satrdan tozalash multiplikatori)\n"
        "• Yuqori uchburchak: hammasi 0\n"
        "Bu eliminatsiya jarayonining 'xotirasi' — har bir qadam qayta tiklash mumkin."
    )


class Q4(EqualityCheckProblem):
    """Oddiy 3x3 uchun qo'lda L va U topish."""
    _hints = [
        "A = [[2,1,1],[4,3,3],[8,7,9]] uchun hisoblang.",
        "m21=2, m31=4, m32=2 → L = [[1,0,0],[2,1,0],[4,2,1]]",
    ]
    _solution = (
        "L = np.array([[1,0,0],[2,1,0],[4,2,1]], dtype=float)\n"
        "U = np.array([[2,1,1],[0,1,1],[0,0,1]], dtype=float)"
    )

    def _do_check(self, L, U, A):
        if not np.allclose(L @ U, A, atol=1e-8):
            return f"L@U ≠ A.\nL@U =\n{L@U}\nA =\n{A}"
        return True


# Challenge 1
class C1_Q1(EqualityCheckProblem):
    """Ko'p b lar uchun LU samaradorligi."""
    _hints = [
        "LU bir marta hisoblang, ko'p b lar uchun ishlatavering.",
        "scipy.linalg.lu_factor va lu_solve funksiyalarini ko'ring.",
    ]
    _solution = (
        "from scipy.linalg import lu_factor, lu_solve\n"
        "lu, piv = lu_factor(A)\n"
        "for b in bs:\n"
        "    x = lu_solve((lu, piv), b)"
    )

    def _do_check(self, solutions, A, bs):
        for x, b in zip(solutions, bs):
            if not np.allclose(A @ x, b, atol=1e-7):
                return f"Yechim noto'g'ri: A@x ≠ b"
        return True


# Challenge 2
class C2_Q1(ThoughtExperiment):
    """LDU parchalash."""
    _hints = [
        "U = DU' bu erda D — diagonal (pivotlar), U' — birlikli yuqori uchburchak.",
        "A = LDU' — simmetrik A uchun L = U'^T.",
    ]
    _solution = (
        "LDU parchalash: A = L D U'\n"
        "D = diag(pivotlar)\n"
        "L = pastki uchburchak (diagonal 1)\n"
        "U' = yuqori uchburchak (diagonal 1)\n\n"
        "Simmetrik A uchun: U' = L^T → A = L D L^T\n"
        "Bu Cholesky parchalashning asosi!"
    )
