"""Hints and solutions — Dars 3.5: To'rt Kichik Fazoning O'lchamlari."""
import numpy as np
from learntools.linear_algebra_uz.base import UzCheckProblem, ThoughtExperiment


class Q1(UzCheckProblem):
    """Ustun fazosi o'lchami = rank."""
    _hints = [
        "dim C(A) = rank(A).",
        "np.linalg.matrix_rank.",
    ]
    _solution = "dim_col = np.linalg.matrix_rank(A)"

    def _do_check(self, dim_col, A):
        expected = np.linalg.matrix_rank(A)
        if int(dim_col) != int(expected):
            return f"Kutilgan: {expected}, siz: {dim_col}"
        return True


class Q2(UzCheckProblem):
    """Satr fazosi o'lchami = rank (= ustun fazosi o'lchami)."""
    _hints = [
        "dim C(A^T) = rank(A) = rank(A^T).",
        "Satr rang = ustun rang.",
    ]
    _solution = "dim_row = np.linalg.matrix_rank(A.T)"

    def _do_check(self, dim_row, A):
        expected = np.linalg.matrix_rank(A)
        if int(dim_row) != int(expected):
            return f"Kutilgan: {expected}, siz: {dim_row}"
        return True


class Q3(UzCheckProblem):
    """Nol fazo o'lchami = n - r."""
    _hints = [
        "dim N(A) = n - r, n = ustunlar soni.",
        "A.shape[1] - rank.",
    ]
    _solution = "dim_null = A.shape[1] - np.linalg.matrix_rank(A)"

    def _do_check(self, dim_null, A):
        expected = A.shape[1] - np.linalg.matrix_rank(A)
        if int(dim_null) != int(expected):
            return f"Kutilgan: {expected}, siz: {dim_null}"
        return True


class Q4(UzCheckProblem):
    """Chap nol fazo o'lchami = m - r."""
    _hints = [
        "dim N(A^T) = m - r, m = satrlar soni.",
        "A.shape[0] - rank.",
    ]
    _solution = "dim_left_null = A.shape[0] - np.linalg.matrix_rank(A)"

    def _do_check(self, dim_left_null, A):
        expected = A.shape[0] - np.linalg.matrix_rank(A)
        if int(dim_left_null) != int(expected):
            return f"Kutilgan: {expected}, siz: {dim_left_null}"
        return True


class Q5(UzCheckProblem):
    """O'lchamlar yig'indisi tekshiruvi: r + (n-r) = n."""
    _hints = [
        "R^n da: satr fazosi (r) + nol fazo (n-r) = n.",
        "Yig'indini hisoblang va n ga teng ekanini tekshiring.",
    ]
    _solution = "total = np.linalg.matrix_rank(A) + (A.shape[1] - np.linalg.matrix_rank(A))"

    def _do_check(self, total, A):
        if int(total) != int(A.shape[1]):
            return f"r + (n-r) = n = {A.shape[1]} bo'lishi kerak, siz: {total}"
        return True


class Q6(UzCheckProblem):
    """Harder: chap nol fazo vektorini topish (A^T x = 0)."""
    _hints = [
        "Chap nol fazo: A^T x = 0, ya'ni x^T A = 0.",
        "Berilgan x uchun A.T @ x = 0 bo'lishini tekshiring.",
    ]
    _solution = "result = A.T @ x  # nolga teng bo'lishi kerak"

    def _do_check(self, result, A, x):
        if not np.allclose(result, 0, atol=1e-8):
            return f"Chap nol fazo uchun A^T@x=0 kerak. Olingan: {result}"
        if np.allclose(x, 0):
            return "Notrivial vektor bering (nol emas)."
        return True


# Challenge 1
class C1_Q1(UzCheckProblem):
    """To'rt o'lchamni bitta to'plamga yig'ish va tekshirish."""
    _hints = [
        "(dim C(A), dim N(A), dim C(A^T), dim N(A^T)) = (r, n-r, r, m-r).",
        "Tartibga e'tibor bering.",
    ]
    _solution = (
        "r = np.linalg.matrix_rank(A); m, n = A.shape\n"
        "dims = (r, n - r, r, m - r)"
    )

    def _do_check(self, dims, A):
        r = np.linalg.matrix_rank(A)
        m, n = A.shape
        expected = (r, n - r, r, m - r)
        if tuple(int(x) for x in dims) != expected:
            return f"Kutilgan (C, N, C^T, N^T): {expected}, siz: {tuple(dims)}"
        return True


# Challenge 2
class C2_Q1(ThoughtExperiment):
    """Asosiy teorema: nima uchun satr rang = ustun rang?"""
    _hints = [
        "Reduksiya pivotlar sonini saqlaydi.",
        "Pivotlar soni ham mustaqil satrlar, ham mustaqil ustunlar sonini beradi.",
    ]
    _solution = (
        "Chiziqli algebraning asosiy teoremasi: dim C(A) = dim C(A^T) = r.\n\n"
        "Sabab: A ni R (reduksiya) ga keltirganda PIVOT lar soni r ni beradi.\n"
        "  - Mustaqil USTUNLAR soni = pivot ustunlar soni = r.\n"
        "  - Mustaqil SATRLAR soni = nolga teng bo'lmagan satrlar soni = r.\n"
        "Bu ikki son bir xil pivotlardan kelib chiqadi, shuning uchun teng.\n\n"
        "Bu hayratlanarli, chunki C(A) R^m da, C(A^T) esa R^n da yashaydi —\n"
        "butunlay boshqa fazolarda — lekin o'lchamlari aynan bir xil.\n"
        "Bu rangning chuqur simmetriyasi: A va A^T bir xil rangga ega."
    )
