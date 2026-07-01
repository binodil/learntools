"""Hints and solutions — Dars 3.3: A x = b ning To'liq Yechimi."""
import numpy as np
from learntools.linear_algebra_uz.base import UzCheckProblem, ThoughtExperiment


class Q1(UzCheckProblem):
    """Xususiy yechimni tekshirish: A x_p = b."""
    _hints = [
        "Xususiy yechim A @ x_p = b ni qanoatlantirishi kerak.",
        "Berilgan x_p ni A ga ko'paytirib b bilan solishtiring.",
    ]
    _solution = "result = A @ x_p  # b ga teng bo'lishi kerak"

    def _do_check(self, result, A, b):
        if not np.allclose(result, b):
            return f"Kutilgan b: {b}, A@x_p: {result}"
        return True


class Q2(UzCheckProblem):
    """To'liq yechim = x_p + x_n hali ham yechimmi?"""
    _hints = [
        "Agar A x_p = b va A x_n = 0 bo'lsa, A(x_p+x_n) = b.",
        "x_full = x_p + x_n ni hisoblang.",
    ]
    _solution = "x_full = x_p + x_n"

    def _do_check(self, x_full, x_p, x_n):
        expected = x_p + x_n
        if not np.allclose(x_full, expected):
            return f"Kutilgan: {expected}, siz: {x_full}"
        return True


class Q3(UzCheckProblem):
    """Tizim moslashganmi? rank(A) == rank([A|b])."""
    _hints = [
        "Ax=b yechimga ega <=> rank(A) == rank([A|b]).",
        "np.column_stack([A, b]) bilan kengaytiring.",
    ]
    _solution = "consistent = np.linalg.matrix_rank(A) == np.linalg.matrix_rank(np.column_stack([A, b]))"

    def _do_check(self, consistent, A, b):
        expected = np.linalg.matrix_rank(A) == np.linalg.matrix_rank(np.column_stack([A, b]))
        if bool(consistent) != bool(expected):
            return f"Kutilgan: {expected}, siz: {consistent}"
        return True


class Q4(UzCheckProblem):
    """Yechimlar soni: 0, 1 yoki cheksiz."""
    _hints = [
        "Mos emas -> 0. Mos va r=n -> 1. Mos va r<n -> cheksiz.",
        "Javobni 'bitta', 'cheksiz' yoki 'yoq' deb bering.",
    ]
    _solution = "answer = 'cheksiz'  # r < n va mos bo'lganda"

    def _do_check(self, answer, A, b):
        rA = np.linalg.matrix_rank(A)
        rAb = np.linalg.matrix_rank(np.column_stack([A, b]))
        if rA != rAb:
            expected = "yoq"
        elif rA == A.shape[1]:
            expected = "bitta"
        else:
            expected = "cheksiz"
        if str(answer).strip().lower() != expected:
            return f"Kutilgan: '{expected}', siz: '{answer}'"
        return True


class Q5(UzCheckProblem):
    """lstsq bilan xususiy yechim topish."""
    _hints = [
        "np.linalg.lstsq(A, b, rcond=None)[0] eng kichik normali yechim beradi.",
        "Natijani A ga ko'paytirib tekshiring.",
    ]
    _solution = "x_p = np.linalg.lstsq(A, b, rcond=None)[0]"

    def _do_check(self, x_p, A, b):
        if not np.allclose(A @ x_p, b, atol=1e-6):
            return f"A@x_p, b ga teng bo'lishi kerak. Olingan: {A @ x_p}"
        return True


class Q6(UzCheckProblem):
    """Harder: to'liq yechimni qo'lda yozish."""
    _hints = [
        "x_p=(1,0,6,0). Maxsus yechimlar s1=(-3,1,0,0), s2=(-2,0,-4,1).",
        "c1=c2=0 da x_full = x_p bo'ladi; umumiy holatda A@x_full=b.",
    ]
    _solution = "x_full = x_p + c1*s1 + c2*s2"

    def _do_check(self, x_full):
        A = np.array([[1, 3, 0, 2], [0, 0, 1, 4], [1, 3, 1, 6]], dtype=float)
        b = np.array([1, 6, 7], dtype=float)
        if not np.allclose(A @ np.asarray(x_full, dtype=float), b):
            return f"A@x_full = b bo'lishi kerak ({b}). Olingan: {A @ np.asarray(x_full, dtype=float)}"
        return True


# Challenge 1
class C1_Q1(UzCheckProblem):
    """Mos bo'lmagan tizimni aniqlash."""
    _hints = [
        "Agar reduksiyada [0 ... 0 | c] (c!=0) satr chiqsa, yechim yo'q.",
        "rank(A) < rank([A|b]) -> mos emas.",
    ]
    _solution = (
        "rA = np.linalg.matrix_rank(A)\n"
        "rAb = np.linalg.matrix_rank(np.column_stack([A, b]))\n"
        "has_solution = (rA == rAb)"
    )

    def _do_check(self, has_solution, A, b):
        expected = np.linalg.matrix_rank(A) == np.linalg.matrix_rank(np.column_stack([A, b]))
        if bool(has_solution) != bool(expected):
            return f"Kutilgan: {expected}, siz: {has_solution}"
        return True


# Challenge 2
class C2_Q1(ThoughtExperiment):
    """To'liq yechim geometriyasi: nima uchun parallel siljish?"""
    _hints = [
        "Barcha yechimlar to'plami = x_p + N(A).",
        "Bu nol fazoning siljitilgan (affin) nusxasi.",
    ]
    _solution = (
        "Ax=b ning barcha yechimlari x_p + N(A) to'plamini tashkil qiladi.\n"
        "Geometrik jihatdan bu nol fazo N(A) ning x_p vektoriga siljitilgan\n"
        "nusxasi (affin kichik to'plam). U boshlanish nuqtasidan O'TMAYDI\n"
        "(agar b != 0), shuning uchun KICHIK FAZO EMAS.\n\n"
        "Masalan R^3 da N(A) chiziq bo'lsa, yechimlar to'plami shu chiziqqa\n"
        "parallel, lekin x_p dan o'tuvchi boshqa chiziq. dim(N(A)) = n - r\n"
        "yechimning 'erkinlik darajalari' sonini beradi."
    )
