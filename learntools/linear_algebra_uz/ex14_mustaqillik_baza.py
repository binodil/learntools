"""Hints and solutions — Dars 3.4: Mustaqillik, Baza va O'lcham."""
import numpy as np
from learntools.core import EqualityCheckProblem, ThoughtExperiment


class Q1(EqualityCheckProblem):
    """Vektorlar chiziqli mustaqilmi? rank == ustunlar soni."""
    _hints = [
        "Ustunlar mustaqil <=> rank(A) == ustunlar soni.",
        "A = np.column_stack([v1, v2, v3]); rank ni n bilan solishtiring.",
    ]
    _solution = "independent = (np.linalg.matrix_rank(A) == A.shape[1])"

    def _do_check(self, independent, A):
        expected = np.linalg.matrix_rank(A) == A.shape[1]
        if bool(independent) != bool(expected):
            return f"Kutilgan: {expected}, siz: {independent}"
        return True


class Q2(EqualityCheckProblem):
    """Span o'lchami = rank."""
    _hints = [
        "Vektorlar spanining o'lchami = rank(A).",
        "np.linalg.matrix_rank.",
    ]
    _solution = "dim_span = np.linalg.matrix_rank(A)"

    def _do_check(self, dim_span, A):
        expected = np.linalg.matrix_rank(A)
        if int(dim_span) != int(expected):
            return f"Kutilgan: {expected}, siz: {dim_span}"
        return True


class Q3(EqualityCheckProblem):
    """Vektorlar to'plami baza bo'la oladimi? (kvadrat, det!=0)."""
    _hints = [
        "n ta vektor R^n ning bazasi <=> ular mustaqil <=> det != 0.",
        "np.linalg.det(A) ni tekshiring.",
    ]
    _solution = "is_basis = not np.isclose(np.linalg.det(A), 0)"

    def _do_check(self, is_basis, A):
        expected = not np.isclose(np.linalg.det(A), 0)
        if bool(is_basis) != bool(expected):
            return f"Kutilgan: {expected}, siz: {is_basis}"
        return True


class Q4(EqualityCheckProblem):
    """Bog'liqlik munosabatini topish (nol fazo vektori)."""
    _hints = [
        "Bog'liq ustunlar uchun A x = 0 ning notrivial yechimi bor.",
        "scipy.linalg.null_space(A) yo'nalishini bering.",
    ]
    _solution = "result = A @ c  # c bog'liqlik vektori bo'lsa, natija ~ 0"

    def _do_check(self, result, A, c):
        if not np.allclose(result, 0, atol=1e-8):
            return f"Bog'liqlik vektori uchun A@c=0 bo'lishi kerak. Olingan: {result}"
        return True


class Q5(EqualityCheckProblem):
    """O'lcham: dim R^(m x n) = m*n."""
    _hints = [
        "Matritsalar fazosi R^(m x n) o'lchami = m * n.",
        "2x3 matritsalar uchun dim = 6.",
    ]
    _solution = "dim = m * n"

    def _do_check(self, dim, m, n):
        if int(dim) != int(m * n):
            return f"dim R^({m}x{n}) = {m*n}, siz: {dim}"
        return True


class Q6(EqualityCheckProblem):
    """Harder: ustun fazosi bazasini (pivot ustunlar) topish."""
    _hints = [
        "Baza = A ning mustaqil ustunlari (pivot ustunlar).",
        "Bazadagi vektorlar soni = rank(A).",
    ]
    _solution = "num_basis_vectors = np.linalg.matrix_rank(A)"

    def _do_check(self, num_basis_vectors, A):
        expected = np.linalg.matrix_rank(A)
        if int(num_basis_vectors) != int(expected):
            return f"Ustun fazosi bazasidagi vektorlar soni = rank = {expected}, siz: {num_basis_vectors}"
        return True


# Challenge 1
class C1_Q1(EqualityCheckProblem):
    """n ta mustaqil vektor R^n ni yoyadimi?"""
    _hints = [
        "R^n da n ta mustaqil vektor avtomatik bazadir (yoyadi).",
        "rank == n bo'lsa, ular butun R^n ni yoyadi.",
    ]
    _solution = "spans_Rn = (np.linalg.matrix_rank(A) == A.shape[0] == A.shape[1])"

    def _do_check(self, spans_Rn, A):
        expected = (np.linalg.matrix_rank(A) == A.shape[0]) and (A.shape[0] == A.shape[1])
        if bool(spans_Rn) != bool(expected):
            return f"Kutilgan: {expected}, siz: {spans_Rn}"
        return True


# Challenge 2
class C2_Q1(ThoughtExperiment):
    """Nima uchun har bir bazada bir xil sonli vektor bor?"""
    _hints = [
        "Almashtirish teoremasi (Steinitz exchange lemma).",
        "k ta yoyuvchi vektor bo'lsa, har qanday mustaqil to'plamda <= k vektor.",
    ]
    _solution = (
        "Bir fazoning har bir bazasida bir xil sonli vektor borligini ALMASHTIRISH\n"
        "TEOREMASI isbotlaydi: agar v1..vm fazoni yoysa va w1..wn mustaqil bo'lsa,\n"
        "u holda n <= m. Endi ikkita baza B1 (m ta) va B2 (n ta) olaylik.\n"
        "B1 yoyadi, B2 mustaqil -> n <= m. B2 yoyadi, B1 mustaqil -> m <= n.\n"
        "Demak m = n. Bu yagona son o'lcham (dimension) deb ataladi.\n\n"
        "Bu juda muhim: o'lcham bazani tanlashga BOG'LIQ EMAS — fazoning\n"
        "ichki xususiyati."
    )
