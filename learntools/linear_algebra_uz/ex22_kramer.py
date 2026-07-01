"""Hints and solutions — Dars 5.3: Kramer Qoidasi, Teskari va Hajmlar."""
import numpy as np
from learntools.linear_algebra_uz.base import UzCheckProblem, ThoughtExperiment


class Q1(UzCheckProblem):
    """Kramer bilan x ni toping (B_1 ustun almashtirilgan)."""
    _hints = [
        "B1 — A ning 1-ustuni b bilan almashtirilgan matritsa.",
        "x1 = det(B1) / det(A).",
    ]
    _solution = "B1 = A.copy(); B1[:,0] = b; x1 = np.linalg.det(B1)/np.linalg.det(A)"

    def _do_check(self, x1, A, b):
        B1 = A.copy().astype(float); B1[:, 0] = b
        expected = np.linalg.det(B1) / np.linalg.det(A)
        if not np.isclose(x1, expected):
            return f"Kutilgan: {expected:.4g}, siz berdingiz: {x1:.4g}"
        return True


class Q2(UzCheckProblem):
    """Kramer bilan y (x2) ni toping."""
    _hints = [
        "B2 — A ning 2-ustuni b bilan almashtirilgan.",
        "x2 = det(B2)/det(A).",
    ]
    _solution = "B2 = A.copy(); B2[:,1] = b; x2 = np.linalg.det(B2)/np.linalg.det(A)"

    def _do_check(self, x2, A, b):
        B2 = A.copy().astype(float); B2[:, 1] = b
        expected = np.linalg.det(B2) / np.linalg.det(A)
        if not np.isclose(x2, expected):
            return f"Kutilgan: {expected:.4g}, siz berdingiz: {x2:.4g}"
        return True


class Q3(UzCheckProblem):
    """2x2 teskari matritsani formula bilan toping."""
    _hints = [
        "A^{-1} = 1/det * [[d,-b],[-c,a]].",
        "det = ad - bc.",
    ]
    _solution = "Ainv = (1/np.linalg.det(A)) * np.array([[A[1,1],-A[0,1]],[-A[1,0],A[0,0]]])"

    def _do_check(self, Ainv, A):
        expected = np.linalg.inv(A)
        if not np.allclose(Ainv, expected):
            return f"Kutilgan:\n{np.round(expected,3)}\nsiz berdingiz:\n{np.round(Ainv,3)}"
        return True


class Q4(UzCheckProblem):
    """Parallelogramm yuzasi = |det|."""
    _hints = [
        "Ustunlari a1, a2 bo'lgan matritsa determinanti modulini oling.",
        "Yuza = abs(det([a1 a2])).",
    ]
    _solution = "yuza = abs(np.linalg.det(np.column_stack([a1, a2])))"

    def _do_check(self, yuza, a1, a2):
        expected = abs(np.linalg.det(np.column_stack([a1, a2])))
        if not np.isclose(yuza, expected):
            return f"Kutilgan: {expected:.4g}, siz berdingiz: {yuza:.4g}"
        return True


class Q5(UzCheckProblem):
    """Uchburchak yuzasi (uchlari berilgan)."""
    _hints = [
        "Yuza = 0.5 * |det([P2-P1, P3-P1])|.",
        "Farq vektorlarini ustun qilib joylashtiring.",
    ]
    _solution = "yuza = 0.5*abs(np.linalg.det(np.column_stack([P2-P1, P3-P1])))"

    def _do_check(self, yuza, P1, P2, P3):
        expected = 0.5 * abs(np.linalg.det(np.column_stack([P2 - P1, P3 - P1])))
        if not np.isclose(yuza, expected):
            return f"Kutilgan: {expected:.4g}, siz berdingiz: {yuza:.4g}"
        return True


class Q6(UzCheckProblem):
    """det = xususiy qiymatlar ko'paytmasi."""
    _hints = [
        "det(A) = lambda_1 * lambda_2 * ... * lambda_n.",
        "np.linalg.eigvals(A) ni np.prod bilan ko'paytiring.",
    ]
    _solution = "result = np.prod(np.linalg.eigvals(A))"

    def _do_check(self, result, A):
        expected = np.linalg.det(A)
        if not np.isclose(np.real(result), expected):
            return f"Kutilgan: {expected:.4g}, siz berdingiz: {np.real(result):.4g}"
        return True


class C1_Q1(UzCheckProblem):
    """3x3 parallelepiped hajmi."""
    _hints = [
        "Hajm = |det([a1 a2 a3])|.",
        "Uchta ustun vektorni column_stack qiling.",
    ]
    _solution = "hajm = abs(np.linalg.det(np.column_stack([a1, a2, a3])))"

    def _do_check(self, hajm, a1, a2, a3):
        expected = abs(np.linalg.det(np.column_stack([a1, a2, a3])))
        if not np.isclose(hajm, expected):
            return f"Kutilgan: {expected:.4g}, siz berdingiz: {hajm:.4g}"
        return True


class C2_Q1(ThoughtExperiment):
    """Nima uchun Kramer qoidasi katta sistemalar uchun yaroqsiz?"""
    _hints = [
        "Har bir x_j uchun bitta determinant kerak — jami n+1 ta determinant.",
        "Determinantni big formula bilan hisoblash n! amal talab qiladi.",
    ]
    _solution = (
        "Kramer qoidasi n+1 ta n x n determinant talab qiladi. Determinantni to'g'ridan-to'g'ri\n"
        "(big formula) hisoblash ~n! amal. Bu eksponensialdan ham yomon.\n"
        "Gauss eliminatsiyasi esa atigi ~n^3/3 amal bilan sistemani yechadi.\n"
        "Shuning uchun amalda doimo eliminatsiya (np.linalg.solve) ishlatiladi.\n"
        "Kramer qoidasi nazariy formula sifatida va kichik (2x2, 3x3) holatlarda qadrli."
    )
