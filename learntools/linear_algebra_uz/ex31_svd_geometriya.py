"""Hints and solutions — Dars 7.4: SVD Geometriyasi."""
import numpy as np
from learntools.linear_algebra_uz.base import UzCheckProblem, ThoughtExperiment


class Q1(UzCheckProblem):
    """A = UΣVᵀ yoyilmasini topib, shakllarini tekshiring."""
    _hints = [
        "U, s, Vt = np.linalg.svd(A). s — singular qiymatlar vektori.",
        "full_matrices=True bo'lsa U: (m,m), Vt: (n,n); False bo'lsa U: (m,k), Vt: (k,n).",
    ]
    _solution = "U, s, Vt = np.linalg.svd(A, full_matrices=False)"

    def _do_check(self, U, s, Vt, A):
        eu, es, evt = np.linalg.svd(A, full_matrices=False)
        if not np.allclose(np.sort(s)[::-1], np.sort(es)[::-1], atol=1e-6):
            return f"Singular qiymatlar noto'g'ri. Kutilgan: {es}"
        if not np.allclose(np.abs(U), np.abs(eu), atol=1e-5):
            return "U matritsasi noto'g'ri. svd(A, full_matrices=False) dan foydalaning."
        return True


class Q2(UzCheckProblem):
    """A ni UΣVᵀ dan qayta tiklang."""
    _hints = [
        "A = U @ np.diag(s) @ Vt. np.diag(s) — diagonal matritsa.",
    ]
    _solution = "U, s, Vt = np.linalg.svd(A, full_matrices=False); A_rec = U @ np.diag(s) @ Vt"

    def _do_check(self, A_rec, A):
        if not np.allclose(A_rec, A, atol=1e-8):
            return "A_rec = U @ diag(s) @ Vt formulasini tekshiring."
        return True


class Q3(UzCheckProblem):
    """U va V ortonormal ekanini tekshiring."""
    _hints = [
        "U ortogonal bo'lsa UᵀU = I. np.allclose(U.T @ U, np.eye(k)) ni tekshiring.",
    ]
    _solution = (
        "U, s, Vt = np.linalg.svd(A, full_matrices=False)\n"
        "V = Vt.T\n"
        "u_ortho = np.allclose(U.T @ U, np.eye(U.shape[1]))\n"
        "v_ortho = np.allclose(V.T @ V, np.eye(V.shape[1]))"
    )

    def _do_check(self, u_ortho, v_ortho, A):
        U, s, Vt = np.linalg.svd(A, full_matrices=False)
        V = Vt.T
        exp_u = np.allclose(U.T @ U, np.eye(U.shape[1]), atol=1e-8)
        exp_v = np.allclose(V.T @ V, np.eye(V.shape[1]), atol=1e-8)
        if u_ortho != exp_u:
            return f"UᵀU = I? {exp_u}. Tekshiring: U.T @ U"
        if v_ortho != exp_v:
            return f"VᵀV = I? {exp_v}. Tekshiring: V.T @ V"
        return True


class Q4(UzCheckProblem):
    """A ning konditsion soni (condition number) ni toping."""
    _hints = [
        "Konditsion son = σ_max / σ_min. Singular qiymatlar: svd dan s.",
        "np.linalg.cond(A) yoki s.max() / s.min()",
    ]
    _solution = "U, s, Vt = np.linalg.svd(A); cond = s.max() / s.min()"

    def _do_check(self, cond, A):
        expected = np.linalg.cond(A)
        if not np.isclose(cond, expected, rtol=1e-4):
            return f"Konditsion son: {expected:.4f}, siz {cond:.4f} berdingiz."
        return True


class Q5(UzCheckProblem):
    """A ning psevdoteskari (Moore-Penrose) ni SVD bilan hisoblang."""
    _hints = [
        "A⁺ = V Σ⁺ Uᵀ. Σ⁺ da nol bo'lmagan singular qiymatlar 1/σᵢ bilan almashtiriladi.",
        "np.linalg.pinv(A) yoki U @ np.diag(1/s) @ Vt",
    ]
    _solution = (
        "U, s, Vt = np.linalg.svd(A, full_matrices=False)\n"
        "A_pinv = Vt.T @ np.diag(1.0 / s) @ U.T"
    )

    def _do_check(self, A_pinv, A):
        expected = np.linalg.pinv(A)
        if not np.allclose(A_pinv, expected, atol=1e-6):
            return "A⁺ = Vt.T @ diag(1/s) @ U.T formulasini tekshiring."
        return True


class Q6(UzCheckProblem):
    """Rank-1 taqribdan foydalanib A ni eng yaxshi k-rangdagi taqqoslash."""
    _hints = [
        "A_k = U[:,:k] @ diag(s[:k]) @ Vt[:k,:]. Eckart-Young teoremasi: bu eng yaxshi rang-k taqrib.",
        "Frobenius norma xatosi: ||A - A_k||_F = sqrt(Σ_{i>k} σᵢ²).",
    ]
    _solution = (
        "U, s, Vt = np.linalg.svd(A, full_matrices=False)\n"
        "k = 2\n"
        "A_k = U[:, :k] @ np.diag(s[:k]) @ Vt[:k, :]\n"
        "error = np.sqrt(np.sum(s[k:]**2))"
    )

    def _do_check(self, A_k, k, A):
        U, s, Vt = np.linalg.svd(A, full_matrices=False)
        expected = U[:, :k] @ np.diag(s[:k]) @ Vt[:k, :]
        if A_k.shape != expected.shape:
            return f"A_k shakli {expected.shape} bo'lishi kerak."
        if not np.allclose(A_k, expected, atol=1e-8):
            return "A_k = U[:,:k] @ diag(s[:k]) @ Vt[:k,:] formulasini tekshiring."
        return True


class C1_Q1(UzCheckProblem):
    """SVD bilan chiziqli tenglamalar sistemasini tekshiring (over/under-determined)."""
    _hints = [
        "Agar A to'liq rangda bo'lsa: x = A_pinv @ b. Aks holda eng kichik normali yechim.",
        "np.linalg.lstsq(A, b, rcond=None)[0] — ham over-, ham under-determined uchun.",
    ]
    _solution = "x, res, rank, s = np.linalg.lstsq(A, b, rcond=None)"

    def _do_check(self, x, A, b):
        expected = np.linalg.lstsq(A, b, rcond=None)[0]
        if not np.allclose(x, expected, atol=1e-6):
            return "np.linalg.lstsq(A, b, rcond=None)[0] dan foydalaning."
        return True


class C2_Q1(ThoughtExperiment):
    """SVD geometrik ma'nosini tushuntiring."""
    _hints = [
        "Har qanday chiziqli aks ettirish: rotate → scale → rotate.",
        "U — chiqish fazosidagi burilish, Σ — cho'zish, V — kirish fazosidagi burilish.",
    ]
    _solution = (
        "SVD geometrik talqin: A = UΣVᵀ degani:\n"
        "1) Vᵀ — kirish fazosida burilish/aks ettirish (ortonormal baza tanloviga o'tish).\n"
        "2) Σ — koordinata o'qlar bo'ylab cho'zish (σᵢ ko'paytuvchilari bilan).\n"
        "3) U — chiqish fazosida burilish/aks ettirish.\n\n"
        "Har qanday m×n matritsa = 'ortonormal baza + cho'zish + ortonormal baza' kombinatsiyasi.\n"
        "Singular qiymatlar σᵢ — cho'zish darajalarini bildiradi.\n"
        "σ₁ ≥ σ₂ ≥ ... ≥ 0, va rang = nol bo'lmagan σᵢ soni."
    )
