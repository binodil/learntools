"""Hints and solutions — Dars 2.7: Transponent va Almashtirishlar."""
import numpy as np
from learntools.linear_algebra_uz.base import UzCheckProblem, ThoughtExperiment


class Q1(UzCheckProblem):
    """Transponent hisoblang."""
    _hints = ["A.T — NumPy da transponent."]
    _solution = "A_T = A.T"

    def _do_check(self, A_T, A):
        if not np.allclose(A_T, A.T):
            return "Transponent noto'g'ri."
        return True


class Q2(UzCheckProblem):
    """Simmetrik matritsa: A = A^T."""
    _hints = [
        "Simmetrik matritsa: A[i,j] = A[j,i] barcha i,j uchun.",
        "np.allclose(A, A.T) bilan tekshiring.",
    ]
    _solution = "is_sym = np.allclose(A, A.T)"

    def _do_check(self, is_sym, A):
        expected = np.allclose(A, A.T)
        if is_sym != expected:
            return f"Kutilgan: {expected}, siz berdingiz: {is_sym}"
        return True


class Q3(UzCheckProblem):
    """Simmetrik matritsa yarating: S = A^T A."""
    _hints = [
        "Har qanday A uchun A^T A — simmetrik va musbat yarim aniq.",
        "S = A.T @ A",
    ]
    _solution = "S = A.T @ A"

    def _do_check(self, S, A):
        expected = A.T @ A
        if not np.allclose(S, expected):
            return "S = A^T A emas."
        if not np.allclose(S, S.T):
            return "S simmetrik emas!"
        return True


class Q4(UzCheckProblem):
    """Almashtirish (permutation) matritsasi."""
    _hints = [
        "P — birlik matritsasining satrlarini qayta tartiblashtirish.",
        "P @ A — A ning satrlarini qayta tartiblashtiradi.",
    ]
    _solution = (
        "P = np.array([[0,1,0],[1,0,0],[0,0,1]])  # 1 va 2 satrlarni almashtiradi\n"
        "PA = P @ A"
    )

    def _do_check(self, PA, P, A):
        if not np.allclose(PA, P @ A):
            return "P @ A noto'g'ri."
        if not np.allclose(P @ P.T, np.eye(P.shape[0])):
            return "P ortogonal emas (P @ P^T ≠ I)."
        return True


class Q5(UzCheckProblem):
    """P^{-1} = P^T."""
    _hints = [
        "Almashtirish matritsalari uchun: P^{-1} = P^T",
        "np.allclose(P.T @ P, np.eye(n)) bilan tekshiring.",
    ]
    _solution = "P_inv = P.T  # almashtirish matritsasi uchun"

    def _do_check(self, P_inv, P):
        n = P.shape[0]
        if not np.allclose(P_inv @ P, np.eye(n), atol=1e-9):
            return "P_inv @ P ≠ I"
        if not np.allclose(P_inv, P.T):
            return "P^{-1} = P^T bo'lishi kerak."
        return True


class Q6(ThoughtExperiment):
    """PA = LU da P nima rol o'ynaydi?"""
    _hints = [
        "P satrlarni almashtiradi — pivot nol bo'lganda kerak.",
        "Partial pivoting: eng katta absolut qiymatli elementni pivot qilish.",
    ]
    _solution = (
        "PA = LU parchalashida P matritsasi 'partial pivoting' uchun kerak.\n"
        "Sabab: eliminatsiya davomida pivot (bo'linuvchi) nol bo'lib qolishi mumkin.\n"
        "P satrlarni oldindan tartiblab, eng katta (absolut qiymat bo'yicha) pivotni\n"
        "bosh pozitsiyaga qo'yadi — bu sonli barqarorlikni ta'minlaydi."
    )


# Challenge 1
class C1_Q1(UzCheckProblem):
    """Barcha 3x3 almashtirish matritsalarini sanang."""
    _hints = [
        "3 elementning barcha permutatsiyalari: 3! = 6 ta.",
        "itertools.permutations ishlatishingiz mumkin.",
    ]
    _solution = (
        "from itertools import permutations\n"
        "import numpy as np\n"
        "perms = list(permutations(range(3)))\n"
        "matrices = [np.eye(3)[list(p)] for p in perms]\n"
        "# 6 ta almashtirish matritsasi"
    )

    def _do_check(self, count):
        if count != 6:
            return f"3x3 uchun 3! = 6 ta almashtirish bor. Siz berdingiz: {count}"
        return True


# Challenge 2
class C2_Q1(ThoughtExperiment):
    """Antisimmetrik matritsa."""
    _hints = [
        "Antisimmetrik: A^T = -A → A[i,j] = -A[j,i].",
        "Diagonal elementlari: A[i,i] = -A[i,i] → A[i,i] = 0.",
    ]
    _solution = (
        "Antisimmetrik matritsa: A^T = -A\n"
        "Xususiyatlar:\n"
        "• Diagonal: hammasi 0\n"
        "• A[i,j] = -A[j,i]\n"
        "Misol: np.array([[0,1,-2],[-1,0,3],[2,-3,0]])\n\n"
        "Ixtiyoriy B uchun B - B^T antisimmetrik.\n"
        "Har qanday kvadrat matritsa = simmetrik + antisimmetrik:\n"
        "A = (A + A^T)/2 + (A - A^T)/2"
    )
