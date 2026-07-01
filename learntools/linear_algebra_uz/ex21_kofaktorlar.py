"""Hints and solutions — Dars 5.2: Almashtirishlar va Kofaktorlar."""
import numpy as np
from learntools.core import EqualityCheckProblem, ThoughtExperiment


def _cofactor(A, i, j):
    minor = np.delete(np.delete(A, i, axis=0), j, axis=1)
    return (-1) ** (i + j) * np.linalg.det(minor)


class Q1(EqualityCheckProblem):
    """Minorni toping (1-satr, 1-ustun o'chirilgan)."""
    _hints = [
        "M_00 — 0-satr va 0-ustunni o'chirgach qolgan matritsa determinanti.",
        "np.delete bilan satr va ustunni o'chiring.",
    ]
    _solution = "M = np.linalg.det(np.delete(np.delete(A, 0, 0), 0, 1))"

    def _do_check(self, M, A):
        expected = np.linalg.det(np.delete(np.delete(A, 0, 0), 0, 1))
        if not np.isclose(M, expected):
            return f"Kutilgan: {expected:.4g}, siz berdingiz: {M:.4g}"
        return True


class Q2(EqualityCheckProblem):
    """Kofaktor C_01 ni toping (ishora bilan)."""
    _hints = [
        "C_ij = (-1)^(i+j) * M_ij.",
        "i=0, j=1 => (-1)^1 = -1, ya'ni ishora manfiy.",
    ]
    _solution = "C01 = (-1)**(0+1) * np.linalg.det(np.delete(np.delete(A,0,0),1,1))"

    def _do_check(self, C01, A):
        expected = _cofactor(A, 0, 1)
        if not np.isclose(C01, expected):
            return f"Kutilgan: {expected:.4g}, siz berdingiz: {C01:.4g}"
        return True


class Q3(EqualityCheckProblem):
    """Birinchi satr bo'yicha yoyib determinant toping."""
    _hints = [
        "det = sum_j A[0,j] * C_0j.",
        "Har bir kofaktorni ishora bilan hisoblang.",
    ]
    _solution = "det = sum(A[0,j]*_cofactor(A,0,j) for j in range(3))"

    def _do_check(self, det, A):
        expected = np.linalg.det(A)
        if not np.isclose(det, expected):
            return f"Kutilgan: {expected:.4g}, siz berdingiz: {det:.4g}"
        return True


class Q4(EqualityCheckProblem):
    """n! — necha had bo'ladi (big formula)."""
    _hints = [
        "Leibniz formulasida n! ta had bor.",
        "4x4 uchun 4! = 24.",
    ]
    _solution = "from math import factorial; n_terms = factorial(4)  # 24"

    def _do_check(self, n_terms):
        if n_terms != 24:
            return f"Kutilgan: 24 (=4!), siz berdingiz: {n_terms}"
        return True


class Q5(EqualityCheckProblem):
    """Ustun bo'yicha yoyish ham bir xil natija."""
    _hints = [
        "Istalgan ustun bo'yicha yoyish bir xil det beradi.",
        "det = sum_i A[i,0]*C_i0.",
    ]
    _solution = "det = sum(A[i,0]*_cofactor(A,i,0) for i in range(3))"

    def _do_check(self, det, A):
        expected = np.linalg.det(A)
        if not np.isclose(det, expected):
            return f"Kutilgan: {expected:.4g}, siz berdingiz: {det:.4g}"
        return True


class Q6(EqualityCheckProblem):
    """Eng ko'p nolli ustunni tanlash — samarali yoyish."""
    _hints = [
        "Nollari ko'p satr/ustun bo'yicha yoysangiz, hisob kamayadi.",
        "A = [[3,0,0],[1,2,0],[5,4,6]] — 1-satrda ikkita nol bor.",
    ]
    _solution = "det = 3*2*6  # uchburchak => diagonal ko'paytmasi = 36"

    def _do_check(self, det):
        A = np.array([[3., 0, 0], [1, 2, 0], [5, 4, 6]])
        expected = np.linalg.det(A)
        if not np.isclose(det, expected):
            return f"Kutilgan: {expected:.4g}, siz berdingiz: {det:.4g}"
        return True


class C1_Q1(EqualityCheckProblem):
    """To'liq kofaktor matritsasini quring."""
    _hints = [
        "C[i,j] = _cofactor(A,i,j) har bir i,j uchun.",
        "Natija 3x3 matritsa bo'ladi.",
    ]
    _solution = "C = np.array([[_cofactor(A,i,j) for j in range(3)] for i in range(3)])"

    def _do_check(self, C, A):
        expected = np.array([[_cofactor(A, i, j) for j in range(3)] for i in range(3)])
        if not np.allclose(C, expected):
            return f"Kutilgan:\n{np.round(expected,3)}\nsiz berdingiz:\n{np.round(C,3)}"
        return True


class C2_Q1(ThoughtExperiment):
    """Almashtirish ishorasi (sgn) qanday aniqlanadi?"""
    _hints = [
        "sgn(sigma) = (-1)^(transpozitsiyalar soni).",
        "Inversiyalar (tartibsizliklar) sonini sanang.",
    ]
    _solution = (
        "Almashtirish sigma ni standart tartibga keltirish uchun kerakli juftliklar\n"
        "almashinishi (transpozitsiya) sonini hisoblang. Juft bo'lsa sgn=+1, toq bo'lsa -1.\n"
        "Masalan (2,1,3): bitta almashinish (2<->1) => toq => sgn=-1.\n"
        "Big formula har bir hadga shu ishorani qo'yadi."
    )
