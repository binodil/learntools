"""Hints and solutions — Dars 10.1: Graflar va Tarmoqlar."""
import numpy as np
from learntools.linear_algebra_uz.base import UzCheckProblem, ThoughtExperiment


# Graf: 4 tugun, 5 qirra
A_INC = np.array([
    [-1,  1,  0,  0],
    [-1,  0,  1,  0],
    [ 0, -1,  1,  0],
    [ 0, -1,  0,  1],
    [ 0,  0, -1,  1],
], dtype=float)


class Q1(UzCheckProblem):
    """Insidentlik matritsasi rangini toping."""
    _hints = [
        "np.linalg.matrix_rank dan foydalaning.",
        "Bog'langan grafda rang = tugunlar soni - 1.",
    ]
    _solution = "rank = np.linalg.matrix_rank(A)  # 3"

    def _do_check(self, rank):
        if int(rank) != 3:
            return f"Kutilgan: 3, siz berdingiz: {rank}"
        return True


class Q2(UzCheckProblem):
    """A @ x ni doimiy vektor uchun hisoblang."""
    _hints = [
        "Doimiy potensial [1,1,1,1] uchun har bir qirradagi farq 0 bo'ladi.",
        "A @ np.ones(4) ni hisoblang.",
    ]
    _solution = "result = A @ np.ones(4)  # nol vektor"

    def _do_check(self, result):
        expected = A_INC @ np.ones(4)
        if not np.allclose(result, expected):
            return f"Kutilgan: {expected}, siz berdingiz: {result}"
        return True


class Q3(UzCheckProblem):
    """Graf Laplasiani L = A^T A."""
    _hints = [
        "L = A.T @ A.",
        "Natija 4x4 simmetrik matritsa.",
    ]
    _solution = "L = A.T @ A"

    def _do_check(self, L):
        expected = A_INC.T @ A_INC
        if not np.allclose(L, expected):
            return f"Kutilgan:\n{expected}\nsiz berdingiz:\n{L}"
        return True


class Q4(UzCheckProblem):
    """Laplasian qatorlari yig'indisi."""
    _hints = [
        "Laplasian har bir qatori yig'indisi nolga teng.",
        "L.sum(axis=1) ni hisoblang.",
    ]
    _solution = "row_sums = (A.T @ A).sum(axis=1)  # [0,0,0,0]"

    def _do_check(self, row_sums):
        if not np.allclose(row_sums, np.zeros(4)):
            return f"Kutilgan: [0,0,0,0], siz berdingiz: {row_sums}"
        return True


class Q5(UzCheckProblem):
    """Mustaqil konturlar soni m - n + 1."""
    _hints = [
        "Eyler formulasi: konturlar = qirralar - tugunlar + 1.",
        "5 - 4 + 1.",
    ]
    _solution = "loops = 5 - 4 + 1  # 2"

    def _do_check(self, loops):
        if int(loops) != 2:
            return f"Kutilgan: 2, siz berdingiz: {loops}"
        return True


class Q6(UzCheckProblem):
    """A^T y = 0 ni tekshiring (Kirxgof tok qonuni)."""
    _hints = [
        "y null fazoda bo'lsa A.T @ y = 0.",
        "Berilgan y uchun A.T @ y ni hisoblang.",
    ]
    _solution = "result = A.T @ y"

    def _do_check(self, result, y):
        expected = A_INC.T @ y
        if not np.allclose(result, expected):
            return f"Kutilgan: {expected}, siz berdingiz: {result}"
        return True


class C1_Q1(UzCheckProblem):
    """Yangi graf uchun insidentlik matritsasini quring."""
    _hints = [
        "Har bir qirra uchun bitta qator: boshlanishda -1, tugashda +1.",
        "Uchburchak graf (1->2, 2->3, 3->1) uchun 3x3 matritsa.",
    ]
    _solution = (
        "A = np.array([[-1,1,0],[0,-1,1],[1,0,-1]], dtype=float)"
    )

    def _do_check(self, A):
        A = np.asarray(A, dtype=float)
        if A.shape != (3, 3):
            return f"3x3 matritsa kutilgan, sizniki {A.shape}"
        # har qatorda bitta -1, bitta +1
        for r in A:
            if not (np.isclose(r.sum(), 0) and np.isclose(np.abs(r).sum(), 2)):
                return "Har qatorda aniq bitta -1 va bitta +1 bo'lishi kerak."
        return True


class C2_Q1(ThoughtExperiment):
    """Graf bog'langanligi va null fazo o'lchami."""
    _hints = [
        "Null fazo o'lchami = bog'langan komponentlar soni.",
        "Bog'langan grafda faqat doimiy vektor null fazoda.",
    ]
    _solution = (
        "Insidentlik matritsasi A ning null fazo o'lchami grafdagi bog'langan\n"
        "komponentlar soniga teng. Bitta bog'langan graf uchun bu 1 (doimiy\n"
        "potensial vektori). Agar graf 2 ta alohida qismdan iborat bo'lsa,\n"
        "null fazo o'lchami 2 ga teng — har bir qism uchun mustaqil doimiy."
    )
