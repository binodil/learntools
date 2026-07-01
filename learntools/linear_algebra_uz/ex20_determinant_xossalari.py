"""Hints and solutions — Dars 5.1: Determinant Xossalari."""
import numpy as np
from learntools.core import EqualityCheckProblem, ThoughtExperiment


class Q1(EqualityCheckProblem):
    """2x2 determinantni qo'lda hisoblang."""
    _hints = [
        "2x2 uchun det = ad - bc.",
        "A = [[3,1],[2,4]] => 3*4 - 1*2.",
    ]
    _solution = "det = 3*4 - 1*2  # = 10"

    def _do_check(self, det):
        if not np.isclose(det, 10):
            return f"Kutilgan: 10, siz berdingiz: {det}"
        return True


class Q2(EqualityCheckProblem):
    """Singular matritsa determinanti."""
    _hints = [
        "Ustunlar parallel bo'lsa det = 0.",
        "[[1,2],[2,4]]: 1*4 - 2*2 = 0.",
    ]
    _solution = "det = 0  # ustunlar parallel"

    def _do_check(self, det):
        if not np.isclose(det, 0):
            return f"Kutilgan: 0, siz berdingiz: {det}"
        return True


class Q3(EqualityCheckProblem):
    """P9: det(AB) = det(A)det(B)."""
    _hints = [
        "Avval det(A) va det(B) ni alohida toping.",
        "Ularning ko'paytmasi det(AB) ga teng.",
    ]
    _solution = "result = np.linalg.det(A) * np.linalg.det(B)"

    def _do_check(self, result, A, B):
        expected = np.linalg.det(A @ B)
        if not np.isclose(result, expected):
            return f"Kutilgan: {expected:.4g}, siz berdingiz: {result:.4g}"
        return True


class Q4(EqualityCheckProblem):
    """P7: uchburchak matritsa determinanti = diagonal ko'paytmasi."""
    _hints = [
        "Uchburchak matritsa uchun det = diagonal elementlar ko'paytmasi.",
        "np.diag(U) elementlarini ko'paytiring (np.prod).",
    ]
    _solution = "det = np.prod(np.diag(U))"

    def _do_check(self, det, U):
        expected = np.prod(np.diag(U))
        if not np.isclose(det, expected):
            return f"Kutilgan: {expected:.4g}, siz berdingiz: {det:.4g}"
        return True


class Q5(EqualityCheckProblem):
    """det(cA) = c^n det(A)."""
    _hints = [
        "n x n matritsa uchun det(cA) = c^n * det(A).",
        "2x2 uchun c^2 ga ko'paytiring.",
    ]
    _solution = "result = (c**2) * np.linalg.det(A)  # 2x2 uchun"

    def _do_check(self, result, c, A):
        n = A.shape[0]
        expected = (c**n) * np.linalg.det(A)
        if not np.isclose(result, expected):
            return f"Kutilgan: {expected:.4g}, siz berdingiz: {result:.4g}"
        return True


class Q6(EqualityCheckProblem):
    """3x3 determinant — eliminatsiya/numpy bilan."""
    _hints = [
        "np.linalg.det(A) dan foydalaning.",
        "Natija butun songa yaqin bo'lishi mumkin — round qiling.",
    ]
    _solution = "det = np.linalg.det(A)  # -6"

    def _do_check(self, det):
        A = np.array([[2., 1, 1], [4, 3, 1], [6, 2, 5]])
        expected = np.linalg.det(A)
        if not np.isclose(det, expected):
            return f"Kutilgan: {expected:.4g}, siz berdingiz: {det:.4g}"
        return True


class C1_Q1(EqualityCheckProblem):
    """det(A^{-1}) = 1/det(A) ni tekshiring."""
    _hints = [
        "Teskari matritsa determinanti = 1 / det(A).",
        "np.linalg.det(np.linalg.inv(A)) ni 1/det(A) bilan solishtiring.",
    ]
    _solution = "result = 1 / np.linalg.det(A)"

    def _do_check(self, result, A):
        expected = np.linalg.det(np.linalg.inv(A))
        if not np.isclose(result, expected):
            return f"Kutilgan: {expected:.4g}, siz berdingiz: {result:.4g}"
        return True


class C2_Q1(ThoughtExperiment):
    """Eliminatsiya determinantni qanday o'zgartiradi?"""
    _hints = [
        "Satr almashinishi ishorani o'zgartiradi (P2).",
        "Satrdan boshqasining karralisini ayirish determinantni saqlaydi (P5).",
        "Satrni songa ko'paytirish determinantni o'sha songa ko'paytiradi (P3).",
    ]
    _solution = (
        "Gauss eliminatsiyasida:\n"
        "1) R_i <-> R_j (almashtirish): det -> -det.\n"
        "2) R_i -> R_i - c*R_j: det o'zgarmaydi.\n"
        "3) R_i -> c*R_i: det -> c*det.\n"
        "Faqat 1 va 3 determinantga ta'sir qiladi. Pivotlardan oldin almashtirishlar\n"
        "sonini hisoblab, oxirgi det = (-1)^(almashtirishlar) * pivotlar ko'paytmasi."
    )
