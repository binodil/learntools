"""Hints and solutions — Dars 1.3: Matritsalar."""
import numpy as np
from learntools.core import EqualityCheckProblem, ThoughtExperiment


class Q1(EqualityCheckProblem):
    """Matritsa yarating."""
    _hints = [
        "np.array([[satır1], [satır2], ...]) bilan matritsa yarating.",
        "2x3 matritsa: np.array([[1,2,3],[4,5,6]])",
    ]
    _solution = "A = np.array([[1, 2], [3, 4], [5, 6]])  # 3x2 matritsa"

    def _do_check(self, A):
        if not isinstance(A, np.ndarray):
            return "A NumPy array bo'lishi kerak."
        if A.shape != (3, 2):
            return f"O'lchami (3,2) bo'lishi kerak, siz berdingiz: {A.shape}"
        return True


class Q2(EqualityCheckProblem):
    """Matritsa-vektor ko'paytmasi."""
    _hints = [
        "A @ v yoki np.dot(A, v) — matritsa-vektor ko'paytmasi.",
        "A ning o'lchami (m, n), v ning o'lchami (n,) bo'lishi kerak.",
    ]
    _solution = "result = A @ v"

    def _do_check(self, result, A, v):
        expected = A @ v
        if not np.allclose(result, expected):
            return f"Kutilgan: {expected}, siz berdingiz: {result}"
        return True


class Q3(EqualityCheckProblem):
    """Matritsa-matritsa ko'paytmasi."""
    _hints = [
        "A @ B — matritsa ko'paytmasi. A ning ustunlari soni B ning satrlari soniga teng bo'lishi kerak.",
        "(m×n) @ (n×p) = (m×p)",
    ]
    _solution = "C = A @ B"

    def _do_check(self, C, A, B):
        expected = A @ B
        if not np.allclose(C, expected):
            return f"Kutilgan:\n{expected}\nSiz berdingiz:\n{C}"
        return True


class Q4(EqualityCheckProblem):
    """Transponent matritsa."""
    _hints = [
        "A.T — NumPy da transponent.",
        "Transponentda satrlar va ustunlar almashinadi.",
    ]
    _solution = "A_T = A.T"

    def _do_check(self, A_T, A):
        if not np.allclose(A_T, A.T):
            return f"Kutilgan:\n{A.T}\nSiz berdingiz:\n{A_T}"
        return True


class Q5(EqualityCheckProblem):
    """Matritsa izini (trace) hisoblang."""
    _hints = [
        "Iz (trace) — bosh diagonal elementlari yig'indisi.",
        "np.trace(A) ishlatishingiz mumkin.",
    ]
    _solution = "t = np.trace(A)"

    def _do_check(self, t, A):
        expected = np.trace(A)
        if not np.isclose(t, expected):
            return f"Kutilgan: {expected}, siz berdingiz: {t}"
        return True


class Q6(EqualityCheckProblem):
    """Birlik matritsa (Identity matrix)."""
    _hints = [
        "np.eye(n) — n×n birlik matritsa.",
        "A @ I = A va I @ A = A bo'lishi kerak.",
    ]
    _solution = "I = np.eye(3)"

    def _do_check(self, I):
        n = I.shape[0]
        if not np.allclose(I, np.eye(n)):
            return "Bu birlik matritsa emas!"
        return True


# Challenge 1
class C1_Q1(EqualityCheckProblem):
    """Matritsa kuchini hisoblang: A^3."""
    _hints = [
        "A @ A @ A yoki np.linalg.matrix_power(A, 3)",
        "Faqat kvadrat matritsalar uchun ishlaydi.",
    ]
    _solution = "A3 = np.linalg.matrix_power(A, 3)  # yoki A @ A @ A"

    def _do_check(self, A3, A):
        expected = np.linalg.matrix_power(A, 3)
        if not np.allclose(A3, expected):
            return f"Kutilgan:\n{expected}\nSiz berdingiz:\n{A3}"
        return True


# Challenge 2
class C2_Q1(ThoughtExperiment):
    """Matritsaning geometrik ma'nosi."""
    _hints = [
        "Matritsa vektorni transformatsiya (aylantirish, cho'zish, aks ettirish) qiladi.",
        "[[0,-1],[1,0]] matritsasi 90° aylantiradi.",
        "[[2,0],[0,2]] matritsasi 2 marta kattalashtiradi.",
    ]
    _solution = (
        "Matritsa — bu chiziqli transformatsiya. Masalan:\n"
        "• Aylantirish: [[cosθ, -sinθ], [sinθ, cosθ]]\n"
        "• Ko'paytirish (scaling): [[sx, 0], [0, sy]]\n"
        "• Aks ettirish (x o'qi bo'ylab): [[1, 0], [0, -1]]\n"
        "Har bir ustun — standart bazis vektori transformatsiya qilingan holati."
    )


class C2_Q2(EqualityCheckProblem):
    """90 daraja aylantirish matritsasini yarating va [1,0] ga qo'llang."""
    _hints = [
        "90° aylantirish: [[cos90, -sin90], [sin90, cos90]] = [[0,-1],[1,0]]",
        "Natija: [1,0] → [0,1] bo'lishi kerak.",
    ]
    _solution = (
        "theta = np.pi / 2\n"
        "R = np.array([[np.cos(theta), -np.sin(theta)],\n"
        "              [np.sin(theta),  np.cos(theta)]])\n"
        "result = R @ np.array([1, 0])  # = [0, 1]"
    )

    def _do_check(self, result):
        expected = np.array([0.0, 1.0])
        if not np.allclose(result, expected, atol=1e-9):
            return f"Kutilgan: {expected}, siz berdingiz: {result}"
        return True
