"""Hints and solutions — Dars 11.2: Normalar va Konditsion Sonlar."""
import numpy as np
from learntools.linear_algebra_uz.base import UzCheckProblem, ThoughtExperiment


class Q1(UzCheckProblem):
    """Vektor normalarini hisoblang: 1-norma, 2-norma, ∞-norma."""
    _hints = [
        "np.linalg.norm(v, 1), np.linalg.norm(v, 2), np.linalg.norm(v, np.inf).",
    ]
    _solution = "n1 = np.linalg.norm(v,1); n2 = np.linalg.norm(v,2); ninf = np.linalg.norm(v,np.inf)"

    def _do_check(self, n1, n2, ninf, v):
        if not np.isclose(n1, np.linalg.norm(v, 1), rtol=1e-8):
            return f"1-norma: {np.linalg.norm(v,1):.6f}, siz {n1:.6f} berdingiz."
        if not np.isclose(n2, np.linalg.norm(v, 2), rtol=1e-8):
            return f"2-norma: {np.linalg.norm(v,2):.6f}, siz {n2:.6f} berdingiz."
        if not np.isclose(ninf, np.linalg.norm(v, np.inf), rtol=1e-8):
            return f"inf-norma: {np.linalg.norm(v,np.inf):.6f}, siz {ninf:.6f} berdingiz."
        return True


class Q2(UzCheckProblem):
    """Matritsa normalarini hisoblang: Frobenius, operator (2-norma), 1-norma."""
    _hints = [
        "np.linalg.norm(A, 'fro'), np.linalg.norm(A, 2) = sigma_max, np.linalg.norm(A, 1) = max col sum.",
    ]
    _solution = "nF = np.linalg.norm(A,'fro'); n2 = np.linalg.norm(A,2); n1 = np.linalg.norm(A,1)"

    def _do_check(self, nF, n2, n1, A):
        if not np.isclose(nF, np.linalg.norm(A, 'fro'), rtol=1e-8):
            return f"Frobenius norma: {np.linalg.norm(A,'fro'):.6f}, siz {nF:.6f}."
        if not np.isclose(n2, np.linalg.norm(A, 2), rtol=1e-6):
            return f"2-norma (sigma_max): {np.linalg.norm(A,2):.6f}, siz {n2:.6f}."
        if not np.isclose(n1, np.linalg.norm(A, 1), rtol=1e-8):
            return f"1-norma: {np.linalg.norm(A,1):.6f}, siz {n1:.6f}."
        return True


class Q3(UzCheckProblem):
    """Konditsion sonni hisoblang: κ(A) = ||A|| · ||A⁻¹||."""
    _hints = [
        "np.linalg.cond(A) — standart usul. Yoki σ_max / σ_min.",
    ]
    _solution = "cond = np.linalg.cond(A)"

    def _do_check(self, cond, A):
        expected = np.linalg.cond(A)
        if not np.isclose(cond, expected, rtol=1e-4):
            return f"κ(A) = {expected:.4e}, siz {cond:.4e} berdingiz."
        return True


class Q4(UzCheckProblem):
    """Yomon konditsionlangan sistema uchun xatoni kuzating."""
    _hints = [
        "A = [[1, 1],[1, 1+ε]] — kichik ε uchun juda yomon konditsion.",
        "np.linalg.cond(A) → katta son (≈ 1/ε).",
    ]
    _solution = "eps = 1e-10; A = np.array([[1.,1.],[1.,1.+eps]]); cond = np.linalg.cond(A)"

    def _do_check(self, cond, A):
        expected = np.linalg.cond(A)
        if not np.isclose(cond, expected, rtol=0.01):
            return f"κ(A) = {expected:.4e}, siz {cond:.4e} berdingiz."
        if cond < 1e8:
            return f"Bu matritsa yomon konditsionlangan bo'lishi kerak (κ >> 1). κ = {cond:.2e}"
        return True


class Q5(UzCheckProblem):
    """Perturbatsiya tahlili: ||δx||/||x|| ≤ κ(A) · ||δb||/||b||."""
    _hints = [
        "b ni δb ga o'zgartiring, yechim o'zgarishini kuzating.",
        "Amplifikatsiya: (||δx||/||x||) / (||δb||/||b||) ≤ κ(A).",
    ]
    _solution = (
        "x = np.linalg.solve(A, b)\n"
        "x_pert = np.linalg.solve(A, b + db)\n"
        "amplification = (np.linalg.norm(x_pert - x)/np.linalg.norm(x)) / (np.linalg.norm(db)/np.linalg.norm(b))"
    )

    def _do_check(self, amplification, A, b, db):
        x = np.linalg.solve(A, b)
        x_pert = np.linalg.solve(A, b + db)
        expected = ((np.linalg.norm(x_pert - x) / np.linalg.norm(x)) /
                    (np.linalg.norm(db) / np.linalg.norm(b)))
        if not np.isclose(amplification, expected, rtol=0.01):
            return f"Amplifikatsiya: {expected:.4f}, siz {amplification:.4f} berdingiz."
        if amplification > np.linalg.cond(A) + 0.1:
            return f"Amplifikatsiya ({amplification:.2f}) > κ(A) ({np.linalg.cond(A):.2f}) bo'lmasligi kerak."
        return True


class Q6(UzCheckProblem):
    """Matritsa normasi va xususiy qiymatlar orasidagi bog'liqlik."""
    _hints = [
        "||A||₂ = σ_max(A). Simmetrik A uchun ||A||₂ = max|λᵢ|.",
    ]
    _solution = "norm2 = np.linalg.norm(A, 2); sigma_max = np.linalg.svd(A, compute_uv=False)[0]"

    def _do_check(self, norm2, sigma_max, A):
        exp_norm2 = np.linalg.norm(A, 2)
        exp_sigma = np.linalg.svd(A, compute_uv=False)[0]
        if not np.isclose(norm2, exp_norm2, rtol=1e-6):
            return f"||A||₂ = {exp_norm2:.6f}, siz {norm2:.6f} berdingiz."
        if not np.isclose(sigma_max, exp_sigma, rtol=1e-6):
            return f"σ_max = {exp_sigma:.6f}, siz {sigma_max:.6f} berdingiz."
        if not np.isclose(norm2, sigma_max, rtol=1e-6):
            return "||A||₂ = σ_max bo'lishi kerak."
        return True


class C1_Q1(UzCheckProblem):
    """Turli normalar uchun yaxshi konditsionlangan tizim yarating."""
    _hints = [
        "A = Q @ diag(s) @ Q.T: singular qiymatlarni tekshiring. κ = s_max/s_min.",
        "Agar s_min > 0.1 va s_max < 10 bo'lsa — yaxshi konditsionlangan.",
    ]
    _solution = (
        "np.random.seed(42)\n"
        "Q, _ = np.linalg.qr(np.random.randn(5,5))\n"
        "s = np.array([5., 4., 3., 2., 1.])\n"
        "A_good = Q @ np.diag(s) @ Q.T\n"
        "cond_good = np.linalg.cond(A_good)"
    )

    def _do_check(self, cond_good, A_good):
        expected = np.linalg.cond(A_good)
        if not np.isclose(cond_good, expected, rtol=0.01):
            return f"κ = {expected:.4f}, siz {cond_good:.4f} berdingiz."
        if cond_good > 100:
            return f"Yaxshi konditsionlangan matritsa uchun κ < 100 bo'lishi kerak. Siz {cond_good:.2f} berdingiz."
        return True


class C2_Q1(ThoughtExperiment):
    """Konditsion son nima va u nima uchun ahamiyatli?"""
    _hints = [
        "κ(A) kichik = tizim barqaror; κ(A) katta = kichik xato katta o'zgarish beradi.",
        "κ(A) ≈ 10^d: d ta raqamli aniqlik yo'qoladi.",
    ]
    _solution = (
        "Konditsion son κ(A) = ||A|| · ||A⁻¹|| = σ_max/σ_min:\n\n"
        "1) Xato kuchaytirish: δb ning ta'siri δx = A⁻¹δb ga κ marta kuchayadi.\n"
        "2) Raqamli aniqlik: 64-bit float da ~16 raqam bor.\n"
        "   κ(A) = 10⁶ → 6 raqam yo'qoladi → faqat 10 raqam qoladi.\n"
        "3) Amalda:\n"
        "   κ < 10³: yaxshi — to'g'ridan-to'g'ri yechish xavfsiz.\n"
        "   κ ≈ 10⁸: ogohlantiruv — nisbiy xato ~10⁻⁸.\n"
        "   κ > 10¹²: yomon — prekonditsionirovanie zarur.\n"
        "4) SVD yordamida: kichik singular qiymatlarni nolga qo'yish (regularizatsiya)."
    )
