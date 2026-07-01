"""Hints and solutions — Dars 7.2: SVD da Bazalar va Matritsalar."""
import numpy as np
from learntools.linear_algebra_uz.base import UzCheckProblem, ThoughtExperiment


class Q1(UzCheckProblem):
    """A v_k = sigma_k u_k tekshiring."""
    _hints = [
        "U, s, Vt = np.linalg.svd(A). v_k = Vt[k], u_k = U[:,k].",
        "A @ Vt[k] sigma_k * U[:,k] ga teng bo'lishi kerak.",
    ]
    _solution = "lhs = A @ Vt[k]\nrhs = s[k] * U[:, k]"

    def _do_check(self, lhs, rhs):
        if not np.allclose(lhs, rhs, atol=1e-6):
            return f"A v_k != sigma_k u_k. lhs={lhs}, rhs={rhs}"
        return True


class Q2(UzCheckProblem):
    """A^T A ning xususiy qiymatlari = sigma^2."""
    _hints = [
        "np.linalg.eigvalsh(A.T @ A) xususiy qiymatlarni beradi.",
        "Ular s**2 ga teng (tartibi farq qilishi mumkin).",
    ]
    _solution = "eigs = np.linalg.eigvalsh(A.T @ A)"

    def _do_check(self, eigs, s):
        if not np.allclose(np.sort(eigs), np.sort(s**2), atol=1e-6):
            return f"eig(A^T A) sigma^2 ga teng emas. Kutilgan: {np.sort(s**2)}"
        return True


class Q3(UzCheckProblem):
    """U ortogonalligini tekshiring."""
    _hints = [
        "U^T U birlik matritsaga teng bo'lishi kerak.",
        "np.eye(U.shape[1]) bilan solishtiring.",
    ]
    _solution = "result = U.T @ U  # ~ I"

    def _do_check(self, result):
        n = result.shape[0]
        if not np.allclose(result, np.eye(n), atol=1e-6):
            return "U ortogonal emas: U^T U != I."
        return True


class Q4(UzCheckProblem):
    """Singulyar qiymat = sqrt(xususiy qiymat)."""
    _hints = [
        "sigma_k = sqrt(lambda_k), lambda_k = A^T A xususiy qiymati.",
        "Eng katta xususiy qiymatdan eng katta sigma chiqadi.",
    ]
    _solution = "sigma = np.sqrt(np.sort(np.linalg.eigvalsh(A.T @ A))[::-1])"

    def _do_check(self, sigma, A):
        expected = np.linalg.svd(A, compute_uv=False)
        if not np.allclose(np.sort(sigma)[::-1], expected, atol=1e-6):
            return f"Kutilgan: {expected}"
        return True


class Q5(UzCheckProblem):
    """Rekonstruksiya: U @ diag(s) @ Vt = A."""
    _hints = [
        "A = U @ np.diag(s) @ Vt (full_matrices=False bilan).",
        "Bu SVD ning ta'rifi.",
    ]
    _solution = "A_rec = U @ np.diag(s) @ Vt"

    def _do_check(self, A_rec, A):
        if not np.allclose(A_rec, A, atol=1e-6):
            return "Rekonstruksiya A ga teng emas."
        return True


class Q6(UzCheckProblem):
    """Simmetrik musbat aniq A uchun sigma = lambda."""
    _hints = [
        "Musbat aniq simmetrik matritsa uchun singulyar qiymatlar = xususiy qiymatlar.",
        "eigh va svd bir xil natija beradi.",
    ]
    _solution = "same = np.allclose(np.sort(np.linalg.eigvalsh(A)), np.sort(s))"

    def _do_check(self, same):
        if same is not True:
            return "Musbat aniq simmetrik uchun sigma = lambda bo'lishi kerak (True)."
        return True


# Challenge 1
class C1_Q1(UzCheckProblem):
    """v_k -> u_k aylanishini U ustun fazosida tekshiring."""
    _hints = [
        "Chap singulyar vektorlar u_k C(A) ustun fazosi bazasi.",
        "A @ v_k ustun fazosida yotadi va u_k yo'nalishida.",
    ]
    _solution = (
        "# Birinchi r ta u_k C(A) ni qoplaydi:\n"
        "# proj = U[:, :r] @ U[:, :r].T; A @ x har doim proj @ (A @ x) ga teng"
    )

    def _do_check(self, U, s, A):
        r = int(np.sum(s > 1e-10))
        proj = U[:, :r] @ U[:, :r].T
        x = np.ones(A.shape[1])
        if not np.allclose(proj @ (A @ x), A @ x, atol=1e-6):
            return "U[:,:r] ustun fazosini qoplamayapti."
        return True


# Challenge 2
class C2_Q1(ThoughtExperiment):
    """SVD va spektral yoyilma qachon bir xil?"""
    _hints = [
        "Simmetrik musbat yarim-aniq matritsalar uchun.",
        "Manfiy xususiy qiymat bo'lsa, sigma = |lambda| va u_k ishorasi o'zgaradi.",
    ]
    _solution = (
        "A = A^T va A musbat yarim-aniq bo'lsa, SVD = spektral yoyilma:\n"
        "U = V = Q (xususiy vektorlar), Sigma = Lambda.\n"
        "Agar lambda_k < 0 bo'lsa: sigma_k = |lambda_k|, u_k = -v_k.\n"
        "Simmetrik bo'lmagan A uchun U != V umuman."
    )
