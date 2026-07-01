"""Hints and solutions — Dars 2.4: Matritsa Amallari."""
import numpy as np
from learntools.core import EqualityCheckProblem, ThoughtExperiment


class Q1(EqualityCheckProblem):
    """Matritsa qo'shish."""
    _hints = ["A + B — mos elementlarni qo'shing. O'lchamlar teng bo'lishi kerak."]
    _solution = "C = A + B"

    def _do_check(self, C, A, B):
        if not np.allclose(C, A + B):
            return f"Kutilgan:\n{A+B}\nSiz berdingiz:\n{C}"
        return True


class Q2(EqualityCheckProblem):
    """Kommutativlik: AB ≠ BA."""
    _hints = [
        "Matritsa ko'paytmasi odatda kommutativ emas: AB ≠ BA.",
        "Ikkalasini hisoblang va solishtiring.",
    ]
    _solution = (
        "AB = A @ B\n"
        "BA = B @ A\n"
        "# np.allclose(AB, BA) → odatda False"
    )

    def _do_check(self, AB, BA, A, B):
        if not np.allclose(AB, A @ B):
            return "AB noto'g'ri."
        if not np.allclose(BA, B @ A):
            return "BA noto'g'ri."
        return True


class Q3(EqualityCheckProblem):
    """Assotsiativlik: (AB)C = A(BC)."""
    _hints = [
        "Matritsa ko'paytmasi assotsiativ: (AB)C = A(BC).",
        "Ikkala yo'l bilan hisoblang.",
    ]
    _solution = "(A @ B) @ C  # == A @ (B @ C)"

    def _do_check(self, res1, res2, A, B, C):
        exp1 = (A @ B) @ C
        exp2 = A @ (B @ C)
        if not np.allclose(res1, exp1) or not np.allclose(res2, exp2):
            return "Natijalar noto'g'ri."
        if not np.allclose(res1, res2):
            return "Assotsiativlik: ikkala natija teng bo'lishi kerak!"
        return True


class Q4(EqualityCheckProblem):
    """(AB)^T = B^T A^T."""
    _hints = [
        "(AB)^T = B^T A^T — transpoze qoidasi.",
        "Tartibga e'tibor bering: B^T avval, A^T keyin.",
    ]
    _solution = "result = B.T @ A.T  # == (A @ B).T"

    def _do_check(self, result, A, B):
        expected = (A @ B).T
        if not np.allclose(result, expected):
            return f"Kutilgan:\n{expected}\nSiz berdingiz:\n{result}"
        return True


class Q5(EqualityCheckProblem):
    """Matritsa kuchi."""
    _hints = ["np.linalg.matrix_power(A, n) — A^n ni hisoblaydi."]
    _solution = "A4 = np.linalg.matrix_power(A, 4)"

    def _do_check(self, A4, A):
        expected = np.linalg.matrix_power(A, 4)
        if not np.allclose(A4, expected):
            return "A^4 noto'g'ri."
        return True


# Challenge 1
class C1_Q1(EqualityCheckProblem):
    """Blok matritsa ko'paytmasi."""
    _hints = [
        "np.block([[A, B], [C, D]]) blok matritsani yaratadi.",
        "Blok ko'paytmasi odatdagi ko'paytma kabi ishlaydi.",
    ]
    _solution = (
        "M = np.block([[A, B], [C, D]])\n"
        "N = np.block([[E, F], [G, H]])\n"
        "result = M @ N"
    )

    def _do_check(self, result, M, N):
        if not np.allclose(result, M @ N):
            return "Blok ko'paytmasi noto'g'ri."
        return True


# Challenge 2
class C2_Q1(ThoughtExperiment):
    """AB = 0 lekin A ≠ 0, B ≠ 0 bo'lishi mumkinmi?"""
    _hints = [
        "Ha! Masalan: A = [[1,0],[0,0]], B = [[0,0],[1,0]]",
        "Bunday matritsalar 'zero divisors' deyiladi.",
    ]
    _solution = (
        "Ha, bu mumkin:\n"
        "A = np.array([[1,0],[0,0]])\n"
        "B = np.array([[0,0],[1,0]])\n"
        "A @ B = [[0,0],[0,0]]\n"
        "Bu sonlar olamidan farq — real sonlarda ab=0 → a=0 yoki b=0.\n"
        "Matritsalarda bu qoida ishlamaydi!"
    )
