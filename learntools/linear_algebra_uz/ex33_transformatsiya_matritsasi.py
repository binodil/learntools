"""Hints and solutions — Dars 8.2: Chiziqli Transformatsiya Matritsasi."""
import numpy as np
from learntools.core import EqualityCheckProblem, ThoughtExperiment


class Q1(EqualityCheckProblem):
    """T(v) = Av matritsasini standart bazada yozing."""
    _hints = [
        "A ning j-ustuni = T(eⱼ), bu erda eⱼ — j-o'q vektori.",
        "A = np.column_stack([T(e1), T(e2), ...])",
    ]
    _solution = "A = np.column_stack([T_e1, T_e2, T_e3])  # ustunlar = T(eⱼ)"

    def _do_check(self, A, T_e1, T_e2, T_e3):
        expected = np.column_stack([T_e1, T_e2, T_e3])
        if not np.allclose(A, expected, atol=1e-8):
            return "A = column_stack([T(e1), T(e2), T(e3)]) formulasini tekshiring."
        return True


class Q2(EqualityCheckProblem):
    """Baza o'zgartirish matritsasini toping: P_{B←S} (standartdan B bazaga)."""
    _hints = [
        "P_{B←S} matritsasining ustunlari — standart baza vektorlarining B bazasidagi koordinatalari.",
        "Agar B = [b1, b2, ...] bo'lsa, P = np.column_stack(b_vecs) ning teskarisi.",
    ]
    _solution = (
        "B = np.column_stack(b_vecs)  # baza vektorlari — ustunlar\n"
        "P = np.linalg.inv(B)  # standart → B bazaga o'tish"
    )

    def _do_check(self, P, b_vecs):
        B = np.column_stack(b_vecs)
        expected = np.linalg.inv(B)
        if not np.allclose(P, expected, atol=1e-6):
            return "P = inv(B), bu erda B — baza vektorlar ustun sifatida."
        return True


class Q3(EqualityCheckProblem):
    """T matritsasini B bazasida toping: A_B = P⁻¹ A P."""
    _hints = [
        "A_B = P⁻¹ A P, bu erda P — baza o'tish matritsasi.",
        "np.linalg.inv(P) @ A @ P",
    ]
    _solution = "A_B = np.linalg.inv(P) @ A @ P"

    def _do_check(self, A_B, A, P):
        expected = np.linalg.inv(P) @ A @ P
        if not np.allclose(A_B, expected, atol=1e-6):
            return "A_B = P⁻¹ A P formulasini tekshiring."
        return True


class Q4(EqualityCheckProblem):
    """O'xshash matritsalar bir xil xususiy qiymatlarga ega ekanini tekshiring."""
    _hints = [
        "A_B = P⁻¹ A P bo'lsa, A va A_B xususiy qiymatlari teng.",
        "np.sort(np.real(np.linalg.eig(A)[0])) va np.sort(np.real(np.linalg.eig(A_B)[0]))",
    ]
    _solution = (
        "vals_A = np.sort(np.real(np.linalg.eig(A)[0]))\n"
        "vals_AB = np.sort(np.real(np.linalg.eig(A_B)[0]))\n"
        "same_eigs = np.allclose(vals_A, vals_AB)"
    )

    def _do_check(self, same_eigs, A, A_B):
        vals_A = np.sort(np.real(np.linalg.eig(A)[0]))
        vals_AB = np.sort(np.real(np.linalg.eig(A_B)[0]))
        expected = np.allclose(vals_A, vals_AB, atol=1e-6)
        if same_eigs != expected:
            return f"A xususiy qiymatlari: {vals_A}, A_B: {vals_AB}. Tengmi? {expected}"
        return True


class Q5(EqualityCheckProblem):
    """T ning ker(T) (yadrosi) ni toping."""
    _hints = [
        "ker(T) = null space of A. np.linalg.null_space(A) yoki scipy.linalg.null_space.",
        "scipy.linalg.null_space(A) — ker(T) uchun ortonormal baza.",
    ]
    _solution = "from scipy.linalg import null_space; ker = null_space(A)"

    def _do_check(self, ker, A):
        from scipy.linalg import null_space
        expected_null = null_space(A)
        expected_dim = expected_null.shape[1] if expected_null.ndim > 1 else 0
        actual_dim = ker.shape[1] if (ker.ndim > 1 and ker.size > 0) else 0
        if actual_dim != expected_dim:
            return f"ker(T) o'lchami {expected_dim}, siz {actual_dim} berdingiz."
        if actual_dim > 0 and not np.allclose(A @ ker, 0, atol=1e-8):
            return "ker(T) ning har bir vektori uchun A @ v = 0 bo'lishi kerak."
        return True


class Q6(EqualityCheckProblem):
    """T ning Im(T) (obrazi) rangini toping."""
    _hints = [
        "Im(T) = column space of A. Rangi = np.linalg.matrix_rank(A).",
        "Bazani: np.linalg.svd(A) ning U ustunlari yoki QR bilan topish mumkin.",
    ]
    _solution = "rank = np.linalg.matrix_rank(A)  # dim(Im(T))"

    def _do_check(self, rank, A):
        expected = np.linalg.matrix_rank(A)
        if rank != expected:
            return f"rang(Im(T)) = rang(A) = {expected}, siz {rank} berdingiz."
        return True


class C1_Q1(EqualityCheckProblem):
    """Xususiy vektorlar bazasida diagonallashtirish: A = PΛP⁻¹."""
    _hints = [
        "vals, P = np.linalg.eig(A). Λ = diag(vals), P — xususiy vektorlar ustun sifatida.",
        "Tekshirish: P @ diag(vals) @ inv(P) ≈ A.",
    ]
    _solution = (
        "vals, P = np.linalg.eig(A)\n"
        "Lambda = np.diag(vals)\n"
        "A_check = P @ Lambda @ np.linalg.inv(P)"
    )

    def _do_check(self, A_check, A):
        vals, P = np.linalg.eig(A)
        expected = P @ np.diag(vals) @ np.linalg.inv(P)
        if not np.allclose(np.real(A_check), np.real(expected), atol=1e-6):
            return "A = P Λ P⁻¹ formulasini tekshiring."
        if not np.allclose(np.real(A_check), A, atol=1e-6):
            return "Natija A ga teng bo'lishi kerak."
        return True


class C2_Q1(ThoughtExperiment):
    """Nima uchun bazani o'zgartirish transformatsiyani soddalashtiradi?"""
    _hints = [
        "Xususiy vektorlar bazasida diagonal matritsa hosil bo'ladi.",
        "A^k, e^A hisoblash: diagonal matritsa uchun trivial.",
    ]
    _solution = (
        "Baza o'zgartirish A_B = P⁻¹AP asosiy g'oyasi:\n\n"
        "1) Xususiy vektor bazasi: agar P = [x₁,...,xₙ] (xususiy vektorlar),\n"
        "   A_B = P⁻¹AP = diag(λ₁,...,λₙ) — diagonal!\n\n"
        "2) Diagonal matritsa bilan hisob sodda:\n"
        "   A^k → diag(λᵢ^k), e^A → diag(e^λᵢ).\n\n"
        "3) Misol: Google PageRank, PCA, kvant holatlari — barchasi\n"
        "   'to'g'ri baza' tanloviga asoslanadi.\n\n"
        "4) Geometrik ma'no: P — koordinata o'zgarishi, A_B — yangi sistemada soddа ko'rinish."
    )
