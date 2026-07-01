"""Hints and solutions — Dars 3.1: Vektor Fazosi."""
import numpy as np
from learntools.linear_algebra_uz.base import UzCheckProblem, ThoughtExperiment


class Q1(UzCheckProblem):
    """Kichik fazo tekshiruvi: nol vektor bormi?"""
    _hints = [
        "Har bir kichik fazo nol vektorni o'z ichiga olishi shart.",
        "Berilgan to'plam {(x, y): x + y = 1} nol vektorni o'z ichiga oladimi?",
    ]
    _solution = "is_subspace = False  # (0,0) ni x+y=1 qanoatlantirmaydi"

    def _do_check(self, is_subspace):
        if is_subspace is not False:
            return "(0,0) uchun 0+0=0 != 1, demak bu kichik fazo EMAS. Javob: False."
        return True


class Q2(UzCheckProblem):
    """Yopiqlikni tekshirish: ikki vektor yig'indisi."""
    _hints = [
        "Kichik fazo qo'shishga yopiq: u, v fazoda bo'lsa, u+v ham fazoda.",
        "(1,2,3) yo'naltirgan chiziqdagi ikki vektorni qo'shing.",
    ]
    _solution = "result = u + v  # yana (1,2,3) yo'nalishida"

    def _do_check(self, result, u, v):
        expected = u + v
        if not np.allclose(result, expected):
            return f"Kutilgan: {expected}, siz berdingiz: {result}"
        return True


class Q3(UzCheckProblem):
    """Ustun fazosining rangi."""
    _hints = [
        "C(A) o'lchami = rank(A).",
        "np.linalg.matrix_rank(A) ni ishlating.",
    ]
    _solution = "r = np.linalg.matrix_rank(A)"

    def _do_check(self, r, A):
        expected = np.linalg.matrix_rank(A)
        if int(r) != int(expected):
            return f"Kutilgan rank: {expected}, siz berdingiz: {r}"
        return True


class Q4(UzCheckProblem):
    """b vektor C(A) da yotadimi?"""
    _hints = [
        "b in C(A) <=> Ax=b yechimga ega <=> rank(A) == rank([A|b]).",
        "np.column_stack bilan kengaytirilgan matritsa yasang.",
    ]
    _solution = "in_col_space = np.linalg.matrix_rank(A) == np.linalg.matrix_rank(np.column_stack([A, b]))"

    def _do_check(self, in_col_space, A, b):
        expected = np.linalg.matrix_rank(A) == np.linalg.matrix_rank(np.column_stack([A, b]))
        if bool(in_col_space) != bool(expected):
            return f"Kutilgan: {expected}, siz berdingiz: {in_col_space}"
        return True


class Q5(UzCheckProblem):
    """Ustun fazosi bazasini skalyar ko'paytmaga yopiqligi."""
    _hints = [
        "Kichik fazo skalyar ko'paytirishga yopiq: c*v ham fazoda.",
        "c * v ni hisoblang.",
    ]
    _solution = "result = c * v"

    def _do_check(self, result, c, v):
        expected = c * v
        if not np.allclose(result, expected):
            return f"Kutilgan: {expected}, siz berdingiz: {result}"
        return True


class Q6(UzCheckProblem):
    """Harder: kichik fazo o'lchamini topish (matritsalar fazosi)."""
    _hints = [
        "2x2 simmetrik matritsalar fazosi: A = A^T sharti.",
        "Erkin elementlar: a11, a12 (=a21), a22 -> 3 ta. dim = 3.",
    ]
    _solution = "dim_sym = 3  # 2x2 simmetrik matritsalar"

    def _do_check(self, dim_sym):
        if int(dim_sym) != 3:
            return "2x2 simmetrik matritsalar fazosi o'lchami 3 (a11, a12, a22)."
        return True


# Challenge 1
class C1_Q1(UzCheckProblem):
    """Berilgan to'plam kichik fazomi? Uchta shartni tekshiring."""
    _hints = [
        "Uch shart: 0 bor; qo'shishga yopiq; ko'paytirishga yopiq.",
        "{(x, y, z): x = 2y} tekisligi nol vektorni o'z ichiga oladi va yopiq.",
    ]
    _solution = (
        "# {(x,y,z): x = 2y} kichik fazo:\n"
        "# (0,0,0): 0 = 2*0 -> True\n"
        "# yopiqlik: agar x1=2y1, x2=2y2 bo'lsa, (x1+x2)=2(y1+y2)\n"
        "is_subspace = True"
    )

    def _do_check(self, is_subspace):
        if is_subspace is not True:
            return "x=2y boshlanish nuqtasidan o'tadi va chiziqli, demak kichik fazo (True)."
        return True


# Challenge 2
class C2_Q1(ThoughtExperiment):
    """Funksiyalar fazosi cheksiz o'lchovli — nima uchun?"""
    _hints = [
        "1, x, x^2, x^3, ... ko'phadlari chiziqli mustaqil.",
        "Cheksiz mustaqil vektorlar bo'lsa, o'lcham cheksiz.",
    ]
    _solution = (
        "Barcha ko'phadlar fazosi P cheksiz o'lchovli, chunki\n"
        "1, x, x^2, x^3, ... cheksiz ko'p chiziqli mustaqil vektor.\n"
        "Hech qanday chekli to'plam butun P ni yoya olmaydi: agar eng katta\n"
        "daraja n bo'lsa, x^(n+1) ni hosil qilib bo'lmaydi.\n"
        "Shuning uchun R^n dan farqli, funksiyalar/ko'phadlar fazosi cheksiz o'lchovli."
    )
