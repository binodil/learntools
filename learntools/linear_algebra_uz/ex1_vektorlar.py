"""Hints and solutions — Dars 1.1: Vektorlar va Chiziqli Kombinatsiyalar."""
import numpy as np
from learntools.linear_algebra_uz.base import UzCheckProblem, ThoughtExperiment


class Q1(UzCheckProblem):
    """Vektor yarating."""
    _hints = [
        "np.array([...]) yordamida vektor yasashingiz mumkin.",
        "Masalan: v = np.array([1, 2, 3]) — bu uch o'lchamli vektor.",
    ]
    _solution = "v = np.array([3, -1, 4])"

    def _do_check(self, v):
        expected = np.array([3, -1, 4])
        if not isinstance(v, np.ndarray):
            return "v NumPy array bo'lishi kerak."
        if not np.allclose(v, expected):
            return f"Kutilgan: {expected}, siz berdingiz: {v}"
        return True


class Q2(UzCheckProblem):
    """Ikkita vektorni qo'shing."""
    _hints = [
        "NumPy da vektorlarni + operatori bilan qo'shish mumkin.",
        "u + v — komponentlarni alohida-alohida qo'shadi.",
    ]
    _solution = "result = u + v  # np.array([5, 7, 9])"

    def _do_check(self, result, u, v):
        expected = u + v
        if not np.allclose(result, expected):
            return f"Kutilgan: {expected}, siz berdingiz: {result}"
        return True


class Q3(UzCheckProblem):
    """Skalyarga ko'paytiring."""
    _hints = [
        "c * v — vektorni skalyarga ko'paytirish.",
        "Har bir komponent c ga ko'paytiriladi.",
    ]
    _solution = "result = 3 * v"

    def _do_check(self, result, c, v):
        expected = c * v
        if not np.allclose(result, expected):
            return f"Kutilgan: {expected}, siz berdingiz: {result}"
        return True


class Q4(UzCheckProblem):
    """Chiziqli kombinatsiya."""
    _hints = [
        "Chiziqli kombinatsiya: c1*v1 + c2*v2 + ... + cn*vn",
        "Avval har bir vektorni o'z koeffitsiyentiga ko'paytiring, keyin qo'shing.",
    ]
    _solution = "result = 2*v1 + (-1)*v2 + 3*v3"

    def _do_check(self, result, v1, v2, v3):
        expected = 2*v1 + (-1)*v2 + 3*v3
        if not np.allclose(result, expected):
            return f"Kutilgan: {expected}, siz berdingiz: {result}"
        return True


class Q5(ThoughtExperiment):
    """Geometrik tushuntirish."""
    _hints = [
        "2D vektorni o'qda nuqta sifatida, yoki o'q boshidan boshlanadigan o'q sifatida tasavvur qiling.",
        "Ikkita vektori yig'indisi — uchburchak qoidasi yoki parallelogramm qoidasi bilan topiladi.",
    ]
    _solution = (
        "v = [1, 2] vektori x o'qi bo'ylab 1, y o'qi bo'ylab 2 birlik ko'chiradi.\n"
        "u + v = [1,0] + [0,1] = [1,1] — parallelogramm diagonal.\n"
        "2*v = [2, 4] — v yo'nalishida, lekin 2 marta uzun."
    )


class Q6(UzCheckProblem):
    """Chiziqli kombinatsiya bilan [1,0] hosil qiling."""
    _hints = [
        "v1 = [1, 1], v2 = [1, -1] vektorlari berilgan. [1, 0] ni hosil qiling.",
        "0.5*v1 + 0.5*v2 ni hisoblang.",
    ]
    _solution = "result = 0.5*v1 + 0.5*v2  # = [1, 0]"

    def _do_check(self, result):
        expected = np.array([1.0, 0.0])
        if not np.allclose(result, expected, atol=1e-9):
            return f"Kutilgan: {expected}, siz berdingiz: {result}"
        return True


# Challenge 1
class C1_Q1(UzCheckProblem):
    """Ixtiyoriy uchta vektor bilan tekislikni qoplang."""
    _hints = [
        "Agar ikkita vektor chiziqli mustaqil bo'lsa, ularning barcha chiziqli kombinatsiyalari R^2 ni qoplaydi.",
        "v1=[1,0], v2=[0,1] standart asos. c1*[1,0]+c2*[0,1]=[c1,c2] — ixtiyoriy nuqta.",
    ]
    _solution = (
        "# c1 va c2 ixtiyoriy son bo'lsa:\n"
        "# result = c1 * np.array([1, 0]) + c2 * np.array([0, 1])\n"
        "# Bu barcha R^2 ni qoplaydi."
    )

    def _do_check(self, v1, v2):
        if np.allclose(np.cross(v1, v2), 0):
            return "v1 va v2 parallel — ular R^2 ni qoplay olmaydi!"
        return True


# Challenge 2
class C2_Q1(ThoughtExperiment):
    """R^3 da uchta vektor bilan butun fazoni qoplash mumkinmi?"""
    _hints = [
        "Agar uchta vektor chiziqli mustaqil bo'lsa, ha — ular R^3 ni qoplaydi.",
        "Determinant = 0 bo'lsa, vektorlar bir tekislikda yotadi.",
    ]
    _solution = (
        "Ha, agar uchta vektor chiziqli mustaqil bo'lsa (det ≠ 0), ularning chiziqli kombinatsiyasi\n"
        "butun R^3 ni qoplaydi. Masalan: e1=[1,0,0], e2=[0,1,0], e3=[0,0,1].\n"
        "Lekin [1,1,0],[2,2,0],[0,0,1] kabi holatda emas — birinchi ikkitasi parallel."
    )
