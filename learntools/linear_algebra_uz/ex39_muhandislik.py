"""Hints and solutions — Dars 10.2: Muhandislikda Matritsalar."""
import numpy as np
from learntools.linear_algebra_uz.base import UzCheckProblem, ThoughtExperiment


def second_diff(n):
    return 2*np.eye(n) - np.eye(n, k=1) - np.eye(n, k=-1)


class Q1(UzCheckProblem):
    """4x4 ikkinchi farq matritsasini quring."""
    _hints = [
        "Diagonalda 2, qo'shni diagonallarda -1.",
        "np.eye va k argumentidan foydalaning.",
    ]
    _solution = "K = 2*np.eye(4) - np.eye(4,k=1) - np.eye(4,k=-1)"

    def _do_check(self, K):
        expected = second_diff(4)
        if not np.allclose(K, expected):
            return f"Kutilgan:\n{expected}\nsiz berdingiz:\n{K}"
        return True


class Q2(UzCheckProblem):
    """K xos qiymatlari musbatligini tekshiring (musbat aniqlik)."""
    _hints = [
        "np.linalg.eigvalsh xos qiymatlarni qaytaradi.",
        "Eng kichik xos qiymat > 0 bo'lsa, musbat aniq.",
    ]
    _solution = "is_pd = np.all(np.linalg.eigvalsh(K) > 0)  # True"

    def _do_check(self, is_pd):
        if bool(is_pd) is not True:
            return f"Kutilgan: True (musbat aniq), siz berdingiz: {is_pd}"
        return True


class Q3(UzCheckProblem):
    """K u = f tizimini yeching."""
    _hints = [
        "np.linalg.solve(K, f).",
        "K — 4x4, f — 4 o'lchamli vektor.",
    ]
    _solution = "u = np.linalg.solve(K, f)"

    def _do_check(self, u, f):
        K = second_diff(4)
        expected = np.linalg.solve(K, f)
        if not np.allclose(u, expected):
            return f"Kutilgan: {expected}, siz berdingiz: {u}"
        return True


class Q4(UzCheckProblem):
    """Potensial energiyani hisoblang."""
    _hints = [
        "P = 0.5 * u^T K u - u^T f.",
        "u @ K @ u va u @ f dan foydalaning.",
    ]
    _solution = "P = 0.5 * u @ K @ u - u @ f"

    def _do_check(self, P, u, f):
        K = second_diff(4)
        expected = 0.5 * u @ K @ u - u @ f
        if not np.isclose(P, expected):
            return f"Kutilgan: {expected}, siz berdingiz: {P}"
        return True


class Q5(UzCheckProblem):
    """K simmetrikligini tekshiring."""
    _hints = [
        "Simmetrik: K == K.T.",
        "np.allclose(K, K.T).",
    ]
    _solution = "sym = np.allclose(K, K.T)  # True"

    def _do_check(self, sym):
        if bool(sym) is not True:
            return f"Kutilgan: True, siz berdingiz: {sym}"
        return True


class Q6(UzCheckProblem):
    """K = A^T A: farq matritsasi A dan K ni quring."""
    _hints = [
        "A — 5x4 oldinga farq matritsasi (mahkamlangan chegaralar bilan).",
        "Berilgan A uchun A.T @ A ni hisoblang.",
    ]
    _solution = "K = A.T @ A"

    def _do_check(self, K, A):
        expected = A.T @ A
        if not np.allclose(K, expected):
            return f"Kutilgan:\n{expected}\nsiz berdingiz:\n{K}"
        return True


class C1_Q1(UzCheckProblem):
    """Erkin chegara: singular K ni quring va rangini toping."""
    _hints = [
        "Erkin-erkin chegarada K ning birinchi va oxirgi diagonali 1.",
        "Bu matritsa singular — rangi n-1.",
    ]
    _solution = (
        "Kfree = second_diff(4)\n"
        "Kfree[0,0] = 1; Kfree[-1,-1] = 1\n"
        "rank = np.linalg.matrix_rank(Kfree)  # 3"
    )

    def _do_check(self, rank):
        Kfree = second_diff(4)
        Kfree[0, 0] = 1
        Kfree[-1, -1] = 1
        expected = np.linalg.matrix_rank(Kfree)
        if int(rank) != int(expected):
            return f"Kutilgan: {expected}, siz berdingiz: {rank}"
        return True


class C2_Q1(ThoughtExperiment):
    """A^T C A tuzilishi muhandislikda."""
    _hints = [
        "A — geometriya (farqlar), C — material xususiyatlari (diagonal).",
        "Muvozanat tenglamasi: A^T C A u = f.",
    ]
    _solution = (
        "Muhandislik tizimlari (prujinalar, to'rlar, elektr zanjirlari) doimo\n"
        "A^T C A u = f shaklida bo'ladi:\n"
        " - A: tugun qiymatlarini qirra farqlariga bog'laydi (e = A u),\n"
        " - C: material/qattiqlik (w = C e),\n"
        " - A^T: muvozanat (A^T w = f).\n"
        "C musbat diagonal va A to'liq ustun rangli bo'lsa, A^T C A musbat aniq."
    )
