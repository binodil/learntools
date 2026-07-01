"""Hints and solutions — Dars 2.3: Matritsali Eliminatsiya."""
import numpy as np
from learntools.core import EqualityCheckProblem, ThoughtExperiment


class Q1(EqualityCheckProblem):
    """Elementar matritsa E21 ni yarating."""
    _hints = [
        "E21 = I, faqat (2,1) pozitsiyasida -m bo'ladi.",
        "E21 @ A → A ning (2,1) elementini nolga aylantiradi.",
    ]
    _solution = (
        "E21 = np.eye(3)\n"
        "m = A[1, 0] / A[0, 0]\n"
        "E21[1, 0] = -m"
    )

    def _do_check(self, E21, A):
        result = E21 @ A
        if not np.isclose(result[1, 0], 0):
            return f"E21 @ A ning (1,0) elementi {result[1,0]:.4f}, 0 bo'lishi kerak."
        if not np.allclose(E21 @ np.eye(E21.shape[0]), E21):
            return "E21 noto'g'ri shaklda."
        return True


class Q2(EqualityCheckProblem):
    """LU parchalash."""
    _hints = [
        "scipy.linalg.lu(A) → P, L, U ni qaytaradi.",
        "PA = LU. P — almashtirish matritsasi.",
    ]
    _solution = (
        "from scipy.linalg import lu\n"
        "P, L, U = lu(A)"
    )

    def _do_check(self, L, U, A):
        if not np.allclose(L @ U, A, atol=1e-8):
            alt = np.allclose(np.tril(L), L) and np.allclose(np.triu(U), U)
            if not alt:
                return "L @ U ≈ A bo'lishi kerak (permutatsiyasiz holat uchun)."
        if not np.allclose(np.tril(L), L):
            return "L pastki uchburchak bo'lishi kerak."
        if not np.allclose(np.triu(U), U):
            return "U yuqori uchburchak bo'lishi kerak."
        return True


class Q3(ThoughtExperiment):
    """Nima uchun E21 @ E31 @ E32 = L^{-1}?"""
    _hints = [
        "Har bir eliminatsiya matritsasi pastki uchburchak.",
        "Ularning ko'paytmasi ham pastki uchburchak.",
    ]
    _solution = (
        "Eliminatsiya E matritsalari: E32 @ E31 @ E21 @ A = U\n"
        "Demak: A = (E21^-1)(E31^-1)(E32^-1) U = L U\n"
        "L — multiplikatorlar saqlanadigan pastki uchburchak matritsa.\n"
        "Bu LU parchalashning mohiyati!"
    )


# Challenge 1
class C1_Q1(EqualityCheckProblem):
    """L va U ni qo'lda hisoblang."""
    _hints = [
        "L ning diagonal elementlari 1, L[i,j] = mij (multiplikatorlar).",
        "U — eliminatsiyadan keyin hosil bo'lgan yuqori uchburchak.",
    ]
    _solution = (
        "A = np.array([[2.,1.],[4.,3.]])\n"
        "m21 = 4/2 = 2\n"
        "L = np.array([[1,0],[2,1]])\n"
        "U = np.array([[2,1],[0,1]])"
    )

    def _do_check(self, L, U, A):
        if not np.allclose(L @ U, A):
            return f"L @ U ≠ A. L@U =\n{L@U}\nA =\n{A}"
        return True


# Challenge 2
class C2_Q1(ThoughtExperiment):
    """LU ning samaradorligi."""
    _hints = [
        "Har safar b uchun qaytadan eliminatsiya qilish O(n^3) operatsiya talab qiladi.",
        "LU bir marta hisoblansa, har yangi b uchun faqat L va U bilan O(n^2) operatsiya yetadi.",
    ]
    _solution = (
        "LU parchalash: A ni bir marta L va U ga ajratamiz — O(n^3).\n"
        "Har yangi b uchun:\n"
        "  1. Ly = b → y ni topamiz (oldinga almashtirish, O(n^2))\n"
        "  2. Ux = y → x ni topamiz (orqaga almashtirish, O(n^2))\n"
        "Ko'p b lar uchun juda samarali!"
    )
