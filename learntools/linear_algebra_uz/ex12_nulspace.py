"""Hints and solutions — Dars 3.2: Nol Fazo (Nullspace)."""
import numpy as np
from learntools.core import EqualityCheckProblem, ThoughtExperiment


class Q1(EqualityCheckProblem):
    """Nol fazo o'lchamini topish: dim N(A) = n - r."""
    _hints = [
        "dim N(A) = ustunlar soni - rank.",
        "n = A.shape[1], r = np.linalg.matrix_rank(A).",
    ]
    _solution = "dim_null = A.shape[1] - np.linalg.matrix_rank(A)"

    def _do_check(self, dim_null, A):
        expected = A.shape[1] - np.linalg.matrix_rank(A)
        if int(dim_null) != int(expected):
            return f"Kutilgan: {expected}, siz berdingiz: {dim_null}"
        return True


class Q2(EqualityCheckProblem):
    """Vektor nol fazoda yotadimi? Ax=0 tekshiruvi."""
    _hints = [
        "x in N(A) <=> A @ x = 0.",
        "np.allclose(A @ x, 0) ni ishlating.",
    ]
    _solution = "in_null = np.allclose(A @ x, 0)"

    def _do_check(self, in_null, A, x):
        expected = np.allclose(A @ x, 0)
        if bool(in_null) != bool(expected):
            return f"Kutilgan: {expected}, siz berdingiz: {in_null}"
        return True


class Q3(EqualityCheckProblem):
    """Maxsus yechimni tekshirish."""
    _hints = [
        "Maxsus yechim A @ s = 0 ni qanoatlantirishi kerak.",
        "Berilgan s ni A ga ko'paytiring.",
    ]
    _solution = "result = A @ s  # nolga teng bo'lishi kerak"

    def _do_check(self, result, A, s):
        expected = A @ s
        if not np.allclose(result, np.zeros_like(expected)):
            return f"Maxsus yechim uchun A@s=0 bo'lishi kerak, siz: {result}"
        return True


class Q4(EqualityCheckProblem):
    """Pivot va erkin o'zgaruvchilar soni."""
    _hints = [
        "Pivot ustunlar soni = rank. Erkin = n - rank.",
        "(rank, n-rank) juftligini bering.",
    ]
    _solution = "pivots_free = (np.linalg.matrix_rank(A), A.shape[1] - np.linalg.matrix_rank(A))"

    def _do_check(self, pivots_free, A):
        r = np.linalg.matrix_rank(A)
        expected = (r, A.shape[1] - r)
        if tuple(int(x) for x in pivots_free) != expected:
            return f"Kutilgan (pivot, erkin): {expected}, siz: {tuple(pivots_free)}"
        return True


class Q5(EqualityCheckProblem):
    """Faqat nol yechim bormi? N(A)={0} <=> rank = n."""
    _hints = [
        "N(A) = {0} faqat ustunlar mustaqil bo'lganda, ya'ni rank = n.",
        "Bool qiymat qaytaring.",
    ]
    _solution = "only_zero = (np.linalg.matrix_rank(A) == A.shape[1])"

    def _do_check(self, only_zero, A):
        expected = np.linalg.matrix_rank(A) == A.shape[1]
        if bool(only_zero) != bool(expected):
            return f"Kutilgan: {expected}, siz berdingiz: {only_zero}"
        return True


class Q6(EqualityCheckProblem):
    """Harder: maxsus yechimni qo'lda yasash."""
    _hints = [
        "R = [[1,0,2,0],[0,1,0,2]]. Erkin: x3, x4.",
        "x3=1, x4=0 uchun: x1=-2, x2=0. Yechim (-2,0,1,0).",
    ]
    _solution = "s1 = np.array([-2, 0, 1, 0])"

    def _do_check(self, s1):
        A = np.array([[1, 2, 2, 4], [3, 8, 6, 16]], dtype=float)
        if not np.allclose(A @ np.asarray(s1, dtype=float), 0):
            return "A @ s1 = 0 bo'lishi kerak. To'g'ri maxsus yechim: (-2,0,1,0)."
        if np.allclose(s1, 0):
            return "Trivial (nol) yechim qabul qilinmaydi — notrivial maxsus yechim bering."
        return True


# Challenge 1
class C1_Q1(EqualityCheckProblem):
    """Nol fazo bazasini tuzish va o'lchamni tekshirish."""
    _hints = [
        "scipy.linalg.null_space(A) nol fazo bazasini beradi.",
        "Bazadagi ustunlar soni = dim N(A) = n - r.",
    ]
    _solution = (
        "from scipy.linalg import null_space\n"
        "N = null_space(A)\n"
        "dim_null = N.shape[1]"
    )

    def _do_check(self, dim_null, A):
        expected = A.shape[1] - np.linalg.matrix_rank(A)
        if int(dim_null) != int(expected):
            return f"dim N(A) = n - r = {expected}, siz: {dim_null}"
        return True


# Challenge 2
class C2_Q1(ThoughtExperiment):
    """Nol fazo va satr operatsiyalari — nima uchun o'zgarmaydi?"""
    _hints = [
        "Satr operatsiyalari Ax=0 yechimlar to'plamini saqlaydi.",
        "Lekin ustun fazosini o'zgartiradi!",
    ]
    _solution = (
        "Satr operatsiyalari (Gauss eliminatsiyasi) nol fazoni O'ZGARTIRMAYDI,\n"
        "chunki ular Ax=0 tenglamalar tizimini ekvivalent tizimga aylantiradi —\n"
        "yechimlar to'plami bir xil qoladi. Shuning uchun N(A) = N(R).\n\n"
        "Lekin satr operatsiyalari USTUN fazosini O'ZGARTIRADI: C(A) != C(R)\n"
        "umuman olganda, chunki satrlarni aralashtirish ustunlarning\n"
        "qiymatlarini o'zgartiradi. Masalan R ning pivotli ustunlari standart\n"
        "bazis ko'rinishida bo'ladi, A niki esa yo'q."
    )
