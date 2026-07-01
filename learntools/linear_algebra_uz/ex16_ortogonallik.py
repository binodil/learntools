"""Hints and solutions — Dars 4.1: To'rt Fazoning Ortogonalligi."""
import numpy as np
from learntools.linear_algebra_uz.base import UzCheckProblem, ThoughtExperiment


class Q1(UzCheckProblem):
    """Ikki vektor ortogonalmi tekshiring."""
    _hints = [
        "Ikkita vektor ortogonal bo'lishi uchun ularning skalyar ko'paytmasi nolga teng bo'lishi kerak.",
        "np.dot(u, v) == 0 bo'lsa ortogonal.",
    ]
    _solution = "result = np.isclose(np.dot(u, v), 0)"

    def _do_check(self, result, u, v):
        expected = np.isclose(np.dot(u, v), 0)
        if result != expected:
            return f"Kutilgan: {expected}. u·v = {np.dot(u, v):.4f}"
        return True


class Q2(UzCheckProblem):
    """Nol fazoni (nullspace) toping: Ax=0."""
    _hints = [
        "scipy.linalg.null_space(A) yoki np.linalg.svd orqali topiladi.",
        "Nullspace — A ga ko'paytirilganda nol beradigan vektorlar to'plami.",
    ]
    _solution = "from scipy.linalg import null_space; N = null_space(A)"

    def _do_check(self, N, A):
        if N.ndim == 1:
            N = N.reshape(-1, 1)
        residual = np.linalg.norm(A @ N)
        if residual > 1e-8:
            return f"A @ N nolga teng bo'lishi kerak, lekin norma = {residual:.6f}"
        return True


class Q3(UzCheckProblem):
    """Qator fazosini toping (row space)."""
    _hints = [
        "Qator fazosi = A^T ning ustun fazosi = A ning transponi ustunlarining spani.",
        "np.linalg.svd(A) orqali qator fazosi bazisini topish mumkin.",
    ]
    _solution = "U, s, Vt = np.linalg.svd(A, full_matrices=False); row_space = Vt[s > 1e-10].T"

    def _do_check(self, row_space, A):
        rank = np.linalg.matrix_rank(A)
        if row_space.shape[1] != rank:
            return f"Qator fazosi {rank} ta bazis vektoriga ega bo'lishi kerak, siz {row_space.shape[1]} berdingiz."
        # Check they span the same space
        combined = np.hstack([row_space, A.T])
        if np.linalg.matrix_rank(combined) != rank:
            return "Berilgan vektorlar haqiqiy qator fazosini to'g'ri ifodalamayapti."
        return True


class Q4(UzCheckProblem):
    """Ortogonal to'ldiruvchini (orthogonal complement) tekshiring."""
    _hints = [
        "Agar v nullspace(A) da bo'lsa va w row_space(A) da bo'lsa, u·w = 0.",
        "np.dot(v_null, v_row) ≈ 0 bo'lishi kerak.",
    ]
    _solution = "ortogonal = np.isclose(np.dot(v_null, v_row), 0)"

    def _do_check(self, ortogonal, v_null, v_row):
        expected = np.isclose(np.dot(v_null, v_row), 0)
        if ortogonal != expected:
            return f"v_null · v_row = {np.dot(v_null, v_row):.6f} — ortogonalmi? {expected}"
        return True


class Q5(UzCheckProblem):
    """Matritsaning rangini (rank) toping."""
    _hints = [
        "np.linalg.matrix_rank(A) — rang hisoblovchi funksiya.",
        "Rang = chiziqli mustaqil qatorlar (yoki ustunlar) soni.",
    ]
    _solution = "r = np.linalg.matrix_rank(A)"

    def _do_check(self, r, A):
        expected = np.linalg.matrix_rank(A)
        if r != expected:
            return f"Kutilgan rang: {expected}, siz {r} berdingiz."
        return True


class Q6(UzCheckProblem):
    """Rank-Nullity teoremasi: rang + nullity = n."""
    _hints = [
        "Rang-Nullity: rank(A) + nullity(A) = A ning ustunlar soni.",
        "nullity = n - rank, bu erda n = A.shape[1].",
    ]
    _solution = "nullity = A.shape[1] - np.linalg.matrix_rank(A)"

    def _do_check(self, nullity, A):
        expected = A.shape[1] - np.linalg.matrix_rank(A)
        if nullity != expected:
            return f"Kutilgan nullity: {expected}, siz {nullity} berdingiz."
        return True


class C1_Q1(UzCheckProblem):
    """To'rt kichik fazoning o'lchamlarini toping."""
    _hints = [
        "m×n matritsa uchun: rank(A)=r bo'lsa, to'rtta o'lcham: r, n-r, r, m-r.",
        "dims = (r, n-r, r, m-r) — (row space, nullspace, col space, left nullspace) o'lchamlari.",
    ]
    _solution = (
        "m, n = A.shape; r = np.linalg.matrix_rank(A)\n"
        "dims = (r, n-r, r, m-r)  # row, null, col, left-null"
    )

    def _do_check(self, dims, A):
        m, n = A.shape
        r = np.linalg.matrix_rank(A)
        expected = (r, n - r, r, m - r)
        if tuple(dims) != expected:
            return f"Kutilgan (row, null, col, left-null): {expected}, siz {tuple(dims)} berdingiz."
        return True


class C2_Q1(ThoughtExperiment):
    """Nima uchun nullspace va row space har doim ortogonal?"""
    _hints = [
        "Ax=0 tenglamasini A ning qatorlari bilan skalyar ko'paytma sifatida ko'ring.",
        "Agar Ax=0 bo'lsa, A ning har bir qatori x ga perpendikulyar.",
    ]
    _solution = (
        "Agar x ∈ N(A) bo'lsa (ya'ni Ax=0), u holda A ning har bir qatori rᵢ uchun "
        "rᵢ · x = 0. Bu x ning A ning barcha qatorlariga perpendikulyar ekanini "
        "bildiradi. Qator fazosi — bu qatorlarning linear kombinatsiyalari to'plami, "
        "shuning uchun har qanday v = c₁r₁ + ... + cₘrₘ ∈ Row(A) uchun ham "
        "v · x = c₁(r₁·x) + ... + cₘ(rₘ·x) = 0. Demak N(A) ⊥ Row(A)."
    )
