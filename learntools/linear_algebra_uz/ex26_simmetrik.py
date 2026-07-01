"""Hints and solutions — Dars 6.4: Simmetrik Matritsalar."""
import numpy as np
from learntools.linear_algebra_uz.base import UzCheckProblem, ThoughtExperiment


class Q1(UzCheckProblem):
    """A simmetrikmi tekshiring."""
    _hints = [
        "A simmetrik bo'lsa A = Aᵀ. np.allclose(A, A.T) dan foydalaning.",
    ]
    _solution = "is_sym = np.allclose(A, A.T)"

    def _do_check(self, is_sym, A):
        expected = np.allclose(A, A.T)
        if is_sym != expected:
            return f"A simmetrikmi? {expected}. ||A - Aᵀ|| = {np.linalg.norm(A - A.T):.4f}"
        return True


class Q2(UzCheckProblem):
    """Simmetrik matritsa uchun xususiy qiymatlar haqiqiy ekanini tasdiqlang."""
    _hints = [
        "A = Aᵀ bo'lsa, barcha xususiy qiymatlar haqiqiy son. np.linalg.eigh(A) — simmetrik uchun.",
        "np.linalg.eigh — xususiy qiymatlarni o'sish tartibida, haqiqiy son sifatida beradi.",
    ]
    _solution = "vals, vecs = np.linalg.eigh(A)  # haqiqiy qiymatlar kafolatlanadi"

    def _do_check(self, vals, A):
        exp_vals = np.linalg.eigh(A)[0]
        if not np.allclose(np.sort(vals), np.sort(exp_vals), atol=1e-6):
            return f"Kutilgan (o'sish tartibida): {exp_vals}"
        if not np.allclose(vals, np.real(vals), atol=1e-10):
            return "Xususiy qiymatlar haqiqiy son bo'lishi kerak."
        return True


class Q3(UzCheckProblem):
    """Spektral teorema: A = QΛQᵀ (simmetrik uchun)."""
    _hints = [
        "np.linalg.eigh(A) → (vals, Q). Keyin A_reconstructed = Q @ np.diag(vals) @ Q.T",
    ]
    _solution = "vals, Q = np.linalg.eigh(A); A_rec = Q @ np.diag(vals) @ Q.T"

    def _do_check(self, A_rec, A):
        vals, Q = np.linalg.eigh(A)
        expected = Q @ np.diag(vals) @ Q.T
        if not np.allclose(A_rec, expected, atol=1e-8):
            return "A = QΛQᵀ formulasini tekshiring."
        if not np.allclose(A_rec, A, atol=1e-8):
            return "Qayta qurilgan matritsa aslga teng bo'lishi kerak."
        return True


class Q4(UzCheckProblem):
    """Simmetrik uchun xususiy vektorlar ortogonal ekanini tekshiring."""
    _hints = [
        "np.linalg.eigh(A) → (vals, Q). Q ortonormal: Q.T @ Q = I.",
    ]
    _solution = "vals, Q = np.linalg.eigh(A); ortogonal = np.allclose(Q.T @ Q, np.eye(Q.shape[0]))"

    def _do_check(self, ortogonal, A):
        _, Q = np.linalg.eigh(A)
        expected = np.allclose(Q.T @ Q, np.eye(Q.shape[0]))
        if ortogonal != expected:
            return f"Qᵀ Q = I bo'lishi kerak. np.allclose(Q.T@Q, I) = {expected}"
        return True


class Q5(UzCheckProblem):
    """AᵀA doim simmetrik va musbat yarim aniq ekanini tekshiring."""
    _hints = [
        "(AᵀA)ᵀ = AᵀA — simmetrik. Xususiy qiymatlar ≥ 0 — musbat yarim aniq.",
        "np.linalg.eigh(A.T @ A)[0] — barcha qiymatlar ≥ 0 bo'lishi kerak.",
    ]
    _solution = "S = A.T @ A; vals = np.linalg.eigh(S)[0]; is_psd = np.all(vals >= -1e-10)"

    def _do_check(self, is_psd, A):
        S = A.T @ A
        vals = np.linalg.eigh(S)[0]
        expected = np.all(vals >= -1e-10)
        if is_psd != expected:
            return f"AᵀA xususiy qiymatlari: {vals}. Barcha ≥ 0? {expected}"
        return True


class Q6(UzCheckProblem):
    """Rayleigh kotirovkasi: R(x) = xᵀAx / xᵀx."""
    _hints = [
        "R(x) = (x.T @ A @ x) / (x.T @ x). Bu λ_min va λ_max orasida.",
        "R ning eng katta qiymati λ_max, eng kichigi λ_min.",
    ]
    _solution = "R = (x @ A @ x) / (x @ x)"

    def _do_check(self, R, A, x):
        expected = (x @ A @ x) / (x @ x)
        if not np.isclose(R, expected, atol=1e-8):
            return f"R(x) = xᵀAx/xᵀx = {expected:.6f}, siz {R:.6f} berdingiz."
        vals = np.linalg.eigh(A)[0]
        if R < vals[0] - 1e-8 or R > vals[-1] + 1e-8:
            return f"R = {R:.4f} λ_min={vals[0]:.4f} va λ_max={vals[-1]:.4f} orasida bo'lishi kerak."
        return True


class C1_Q1(UzCheckProblem):
    """Simmetrik matritsa uchun spektral yoyilma: A = Σᵢ λᵢ qᵢ qᵢᵀ."""
    _hints = [
        "vals, Q = np.linalg.eigh(A). Keyin A = sum(vals[i] * outer(Q[:,i], Q[:,i]) for i in range(n)).",
    ]
    _solution = (
        "vals, Q = np.linalg.eigh(A)\n"
        "A_spectral = sum(vals[i] * np.outer(Q[:, i], Q[:, i]) for i in range(len(vals)))"
    )

    def _do_check(self, A_spectral, A):
        if not np.allclose(A_spectral, A, atol=1e-8):
            return "Spektral yoyilma: A = Σᵢ λᵢ qᵢqᵢᵀ formulasini tekshiring."
        return True


class C2_Q1(ThoughtExperiment):
    """Spektral teoremaning amaliy ahamiyati nima?"""
    _hints = [
        "A = QΛQᵀ bo'lsa, A² = QΛ²Qᵀ, A⁻¹ = QΛ⁻¹Qᵀ — tez hisoblash mumkin.",
        "PCA, kvant mexanikasi, tebranish tahlili — barchasi spektral teoremaga asoslanadi.",
    ]
    _solution = (
        "Spektral teorema: simmetrik A = QΛQᵀ bo'lganda:\n"
        "1) A^k = QΛ^kQᵀ — kuchlar tez hisoblanadi.\n"
        "2) A⁻¹ = QΛ⁻¹Qᵀ (agar λᵢ ≠ 0).\n"
        "3) f(A) = Qf(Λ)Qᵀ — ixtiyoriy skalyar funksiya.\n"
        "4) PCA: kovariatsiya matritsasi simmetrik → xususiy vektorlar asosiy yo'nalishlar.\n"
        "5) Kvant mexanikasi: Hermitian operatorlar kuzatiladigan kattaliklar."
    )
