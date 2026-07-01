"""Hints and solutions — Dars 10.3: Markov Matritsalari."""
import numpy as np
from learntools.linear_algebra_uz.base import UzCheckProblem, ThoughtExperiment


P_MARKOV = np.array([[0.9, 0.2],
                     [0.1, 0.8]])


class Q1(UzCheckProblem):
    """Markov matritsasi ustun yig'indilarini tekshiring."""
    _hints = [
        "Har bir ustun yig'indisi 1 ga teng bo'lishi kerak.",
        "P.sum(axis=0).",
    ]
    _solution = "col_sums = P.sum(axis=0)  # [1, 1]"

    def _do_check(self, col_sums):
        if not np.allclose(col_sums, np.ones(2)):
            return f"Kutilgan: [1,1], siz berdingiz: {col_sums}"
        return True


class Q2(UzCheckProblem):
    """Turg'un holatni xos vektordan toping."""
    _hints = [
        "lambda=1 ga mos xos vektorni toping va yig'indisi 1 ga normalang.",
        "Bu masala uchun javob [2/3, 1/3].",
    ]
    _solution = (
        "w, V = np.linalg.eig(P)\n"
        "i = np.argmin(np.abs(w-1))\n"
        "x = np.real(V[:,i]); x = x/x.sum()  # [2/3, 1/3]"
    )

    def _do_check(self, x):
        expected = np.array([2/3, 1/3])
        if not np.allclose(x, expected, atol=1e-4):
            return f"Kutilgan: {expected}, siz berdingiz: {x}"
        return True


class Q3(UzCheckProblem):
    """P x* = x* ekanini tekshiring."""
    _hints = [
        "Turg'un holat uchun P @ x_star = x_star.",
        "P @ x ni hisoblang.",
    ]
    _solution = "result = P @ x_star  # x_star ga teng"

    def _do_check(self, result, x_star):
        expected = P_MARKOV @ x_star
        if not np.allclose(result, expected):
            return f"Kutilgan: {expected}, siz berdingiz: {result}"
        return True


class Q4(UzCheckProblem):
    """50 iteratsiyadan keyingi taqsimot."""
    _hints = [
        "x_0 = [1,0] dan boshlab P ni 50 marta qo'llang.",
        "Natija turg'un holatga yaqinlashadi.",
    ]
    _solution = (
        "x = np.array([1.,0.])\n"
        "for _ in range(50): x = P @ x"
    )

    def _do_check(self, x):
        expected = np.array([2/3, 1/3])
        if not np.allclose(x, expected, atol=1e-3):
            return f"Kutilgan ~{expected}, siz berdingiz: {x}"
        return True


class Q5(UzCheckProblem):
    """Markov matritsasining eng katta xos qiymati 1 ga tengligini tekshiring."""
    _hints = [
        "np.linalg.eigvals dan foydalaning.",
        "Eng katta modulli xos qiymat 1 ga teng.",
    ]
    _solution = "lam_max = np.max(np.abs(np.linalg.eigvals(P)))  # 1.0"

    def _do_check(self, lam_max):
        if not np.isclose(lam_max, 1.0, atol=1e-6):
            return f"Kutilgan: 1.0, siz berdingiz: {lam_max}"
        return True


class Q6(UzCheckProblem):
    """P^k ni hisoblang (matritsa darajasi)."""
    _hints = [
        "np.linalg.matrix_power(P, k) dan foydalaning.",
        "Katta k uchun ustunlar turg'un holatga yaqinlashadi.",
    ]
    _solution = "Pk = np.linalg.matrix_power(P, k)"

    def _do_check(self, Pk, k):
        expected = np.linalg.matrix_power(P_MARKOV, k)
        if not np.allclose(Pk, expected):
            return f"Kutilgan:\n{expected}\nsiz berdingiz:\n{Pk}"
        return True


class C1_Q1(UzCheckProblem):
    """Uchta holatli Markov matritsasi turg'un holati."""
    _hints = [
        "Ustunlari 1 ga yig'iladigan 3x3 matritsa quring.",
        "lambda=1 ga mos normallashtirilgan xos vektor — turg'un holat.",
    ]
    _solution = (
        "w, V = np.linalg.eig(P3)\n"
        "i = np.argmin(np.abs(w-1))\n"
        "x = np.real(V[:,i]); x = x/x.sum()"
    )

    def _do_check(self, x, P3):
        P3 = np.asarray(P3, dtype=float)
        if not np.allclose(P3.sum(axis=0), np.ones(P3.shape[1])):
            return "P3 ustun yig'indilari 1 ga teng emas (Markov emas)."
        if not np.allclose(P3 @ x, x, atol=1e-5):
            return "x turg'un holat emas: P3 @ x != x."
        if not np.isclose(np.sum(x), 1.0):
            return "Turg'un holat yig'indisi 1 ga teng bo'lishi kerak."
        return True


class C2_Q1(ThoughtExperiment):
    """Leontyev iqtisodiy modeli."""
    _hints = [
        "Leontyev: (I - A) x = d, bu yerda A — input-output matritsa.",
        "Yechim x = (I-A)^(-1) d, agar (I-A) teskarilanuvchi bo'lsa.",
    ]
    _solution = (
        "Leontyev input-output modelida iqtisodiyot ishlab chiqarishi x va\n"
        "talab d quyidagicha bog'langan: x = A x + d, ya'ni (I - A) x = d.\n"
        "Agar A ning spektral radiusi < 1 bo'lsa, (I-A)^(-1) = I + A + A^2 + ...\n"
        "manfiy bo'lmagan bo'ladi, demak iqtisodiyot barqaror talabni qondiradi.\n"
        "Bu Markov g'oyalariga o'xshash: ko'paytma darajalari yaqinlashadi."
    )
