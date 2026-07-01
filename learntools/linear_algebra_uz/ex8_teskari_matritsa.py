"""Hints and solutions — Dars 2.5: Teskari Matritsalar."""
import numpy as np
from learntools.core import EqualityCheckProblem, ThoughtExperiment


class Q1(EqualityCheckProblem):
    """Teskari matritsani hisoblang."""
    _hints = [
        "np.linalg.inv(A) — A^{-1} ni hisoblaydi.",
        "Faqat kvadrat va teskari bo'luvchi (det ≠ 0) matritsalar uchun ishlaydi.",
    ]
    _solution = "A_inv = np.linalg.inv(A)"

    def _do_check(self, A_inv, A):
        n = A.shape[0]
        if not np.allclose(A @ A_inv, np.eye(n), atol=1e-8):
            return "A @ A_inv ≠ I"
        return True


class Q2(EqualityCheckProblem):
    """Tekshiring: A @ A^{-1} = I."""
    _hints = [
        "A @ np.linalg.inv(A) ≈ np.eye(n) bo'lishi kerak.",
        "np.allclose(...) bilan solishtirishingiz mumkin.",
    ]
    _solution = "check = np.allclose(A @ np.linalg.inv(A), np.eye(A.shape[0]))"

    def _do_check(self, check, A):
        expected = np.allclose(A @ np.linalg.inv(A), np.eye(A.shape[0]))
        if check != expected:
            return f"Kutilgan: {expected}, siz berdingiz: {check}"
        return True


class Q3(EqualityCheckProblem):
    """(AB)^{-1} = B^{-1} A^{-1}."""
    _hints = [
        "(AB)^{-1} = B^{-1} A^{-1} — tartibga e'tibor!",
        "np.linalg.inv(B) @ np.linalg.inv(A)",
    ]
    _solution = "inv_AB = np.linalg.inv(B) @ np.linalg.inv(A)"

    def _do_check(self, inv_AB, A, B):
        expected = np.linalg.inv(A @ B)
        if not np.allclose(inv_AB, expected, atol=1e-8):
            return "Noto'g'ri. (AB)^-1 = B^-1 A^-1 ekanini tekshiring."
        return True


class Q4(EqualityCheckProblem):
    """Gauss-Jordan usuli bilan teskari matritsa."""
    _hints = [
        "[A | I] ni kengaytirilgan matritsaga qo'ying.",
        "Eliminatsiya qiling: [A|I] → [I|A^{-1}]",
    ]
    _solution = (
        "aug = np.hstack([A.astype(float), np.eye(A.shape[0])])\n"
        "# Gauss-Jordan eliminatsiyasini bajaring\n"
        "# Natijada o'ng qism A^-1 bo'ladi"
    )

    def _do_check(self, A_inv, A):
        expected = np.linalg.inv(A)
        if not np.allclose(A_inv, expected, atol=1e-6):
            return "Gauss-Jordan natijasi noto'g'ri."
        return True


class Q5(ThoughtExperiment):
    """Qachon teskari matritsa mavjud emas?"""
    _hints = [
        "det(A) = 0 bo'lsa, teskari matritsa yo'q.",
        "Bunday matritsa 'singular' deyiladi.",
    ]
    _solution = (
        "Teskari matritsa mavjud bo'lmaydi (singular matrix) agar:\n"
        "• det(A) = 0\n"
        "• Satrlar chiziqli bog'liq (bir satr boshqasining kombinatsiyasi)\n"
        "• Ustunlar chiziqli bog'liq\n"
        "• Matritsa to'g'ri to'rtburchak (kvadrat emas)\n"
        "Misol: A = [[1,2],[2,4]] → det = 4-4 = 0 → teskari yo'q."
    )


# Challenge 1
class C1_Q1(EqualityCheckProblem):
    """2x2 matritsa uchun analitik formula."""
    _hints = [
        "A = [[a,b],[c,d]] uchun: A^-1 = 1/det * [[d,-b],[-c,a]]",
        "det = ad - bc",
    ]
    _solution = (
        "a, b, c, d = A[0,0], A[0,1], A[1,0], A[1,1]\n"
        "det = a*d - b*c\n"
        "A_inv = (1/det) * np.array([[d, -b], [-c, a]])"
    )

    def _do_check(self, A_inv, A):
        if not np.allclose(A_inv, np.linalg.inv(A), atol=1e-8):
            return "Analitik formula noto'g'ri."
        return True


# Challenge 2
class C2_Q1(ThoughtExperiment):
    """Teskari matritsa va chiziqli tenglamalar."""
    _hints = [
        "Ax = b → x = A^{-1}b",
        "Lekin np.linalg.solve samaraliroq — u teskari matritsani hisoblamaydi.",
    ]
    _solution = (
        "x = np.linalg.inv(A) @ b  # ishlaydi, lekin sekin\n"
        "x = np.linalg.solve(A, b)  # tezroq va aniqroq\n\n"
        "Sabab: inv hisoblash O(n^3) va rounding xatolarini ko'paytiradi.\n"
        "solve LU parchalashdan foydalanadi — samaraliroq."
    )
