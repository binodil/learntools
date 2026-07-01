"""Hints and solutions — Dars 6.2: Matritsani Diagonallashtirish."""
import numpy as np
from learntools.linear_algebra_uz.base import UzCheckProblem, ThoughtExperiment


class Q1(UzCheckProblem):
    """S matritsani xususiy vektorlardan quring."""
    _hints = [
        "S ustunlari — chiziqli mustaqil xususiy vektorlar.",
        "np.linalg.eig(A) ikkinchi qiymati S ni beradi.",
    ]
    _solution = "lams, S = np.linalg.eig(A)"

    def _do_check(self, S, A):
        # S ustunlari xususiy vektor bo'lishini tekshiramiz
        for j in range(S.shape[1]):
            x = S[:, j]
            Ax = A @ x
            ratios = Ax[np.abs(x) > 1e-9] / x[np.abs(x) > 1e-9]
            if not np.allclose(ratios, ratios[0]):
                return f"{j}-ustun xususiy vektor emas."
        return True


class Q2(UzCheckProblem):
    """Lambda diagonal matritsasini quring."""
    _hints = [
        "Lambda = diag(lambda_1, ..., lambda_n).",
        "np.diag(lams) ishlating.",
    ]
    _solution = "Lam = np.diag(lams)"

    def _do_check(self, Lam, lams):
        expected = np.diag(lams)
        if not np.allclose(Lam, expected):
            return f"Kutilgan:\n{np.round(expected,3)}\nsiz berdingiz:\n{np.round(Lam,3)}"
        return True


class Q3(UzCheckProblem):
    """A = S Lambda S^{-1} ni qayta tiklang."""
    _hints = [
        "A = S @ Lambda @ inv(S).",
        "np.linalg.inv(S) dan foydalaning.",
    ]
    _solution = "A_rec = S @ Lam @ np.linalg.inv(S)"

    def _do_check(self, A_rec, A):
        if not np.allclose(A_rec, A):
            return f"Qayta tiklangan A asl A ga teng emas."
        return True


class Q4(UzCheckProblem):
    """A^k = S Lambda^k S^{-1}."""
    _hints = [
        "Lambda^k — diagonal elementlarni k-darajaga ko'tarish.",
        "S @ diag(lams**k) @ inv(S).",
    ]
    _solution = "Ak = S @ np.diag(lams**k) @ np.linalg.inv(S)"

    def _do_check(self, Ak, A, k):
        expected = np.linalg.matrix_power(A, k)
        if not np.allclose(Ak, expected):
            return f"Kutilgan A^{k}:\n{np.round(expected,3)}\nsiz berdingiz:\n{np.round(Ak,3)}"
        return True


class Q5(UzCheckProblem):
    """Fibonachchi: F_10 ni matritsa darajasi bilan."""
    _hints = [
        "F = [[1,1],[1,0]], F^n[0,1] = F_n.",
        "np.linalg.matrix_power(F, 10)[0,1].",
    ]
    _solution = "F10 = np.linalg.matrix_power(np.array([[1,1],[1,0]]), 10)[0,1]  # 55"

    def _do_check(self, F10):
        if not np.isclose(F10, 55):
            return f"Kutilgan: 55, siz berdingiz: {F10}"
        return True


class Q6(UzCheckProblem):
    """Defektli matritsa: yetarli xususiy vektor yo'q."""
    _hints = [
        "[[2,1],[0,2]] takroriy lambda=2 ga ega, lekin bitta xususiy vektor.",
        "Mustaqil xususiy vektorlar soni < n => diagonallashmaydi. Javob: False.",
    ]
    _solution = "can_diagonalize = False  # [[2,1],[0,2]] defektli"

    def _do_check(self, can_diagonalize):
        if can_diagonalize is not False:
            return "[[2,1],[0,2]] takroriy lambda uchun bitta xususiy vektorga ega — diagonallashmaydi."
        return True


class C1_Q1(UzCheckProblem):
    """Diagonallashtirish orqali A^k ning katta k uchun o'sishi."""
    _hints = [
        "Eng katta |lambda| ustun keladi.",
        "lambda_max = max(abs(eigvals)).",
    ]
    _solution = "lam_max = np.max(np.abs(np.linalg.eigvals(A)))"

    def _do_check(self, lam_max, A):
        expected = np.max(np.abs(np.linalg.eigvals(A)))
        if not np.isclose(lam_max, expected):
            return f"Kutilgan: {expected:.4g}, siz berdingiz: {lam_max:.4g}"
        return True


class C2_Q1(ThoughtExperiment):
    """Qaysi matritsalar diagonallashadi, qaysilari yo'q?"""
    _hints = [
        "n ta chiziqli mustaqil xususiy vektor kerak.",
        "Turli xususiy qiymatlar => doimo diagonallashadi.",
        "Takroriy lambda => geometrik karralilik tekshirilishi kerak.",
    ]
    _solution = (
        "Matritsa diagonallashadi <=> n ta chiziqli mustaqil xususiy vektori bor.\n"
        "- Barcha lambda turli bo'lsa: kafolat bilan diagonallashadi.\n"
        "- Simmetrik matritsalar: doimo diagonallashadi (spektral teorema).\n"
        "- Takroriy lambda da: agar geometrik karralilik (xususiy fazo o'lchami)\n"
        "  algebraik karralilikdan kichik bo'lsa, defektli — diagonallashmaydi.\n"
        "Masalan [[2,1],[0,2]]: lambda=2 ikki marta, lekin xususiy fazo 1 o'lchamli."
    )
