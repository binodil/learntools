"""Hints and solutions — Dars 2.1: Vektorlar va Chiziqli Tenglamalar."""
import numpy as np
from learntools.linear_algebra_uz.base import UzCheckProblem, ThoughtExperiment


class Q1(UzCheckProblem):
    """Chiziqli tenglamalar sistemasini matritsali ko'rinishga o'tkazing."""
    _hints = [
        "Ax = b ko'rinishida yozing. A — koeffitsiyentlar matritsasi, x — noma'lumlar, b — o'ng tomon.",
        "2x + 3y = 7 va x - y = 1 → A=[[2,3],[1,-1]], b=[7,1]",
    ]
    _solution = (
        "A = np.array([[2, 3], [1, -1]])\n"
        "b = np.array([7, 1])"
    )

    def _do_check(self, A, b):
        expected_A = np.array([[2., 3.], [1., -1.]])
        expected_b = np.array([7., 1.])
        if not np.allclose(A, expected_A) or not np.allclose(b, expected_b):
            return "A yoki b noto'g'ri."
        return True


class Q2(UzCheckProblem):
    """np.linalg.solve bilan sistemani yeching."""
    _hints = [
        "np.linalg.solve(A, b) Ax=b sistemasini yechadi.",
        "Natija x vektor bo'ladi.",
    ]
    _solution = "x = np.linalg.solve(A, b)"

    def _do_check(self, x, A, b):
        if not np.allclose(A @ x, b):
            return f"Tekshirish muvaffaqiyatsiz: A@x = {A@x}, b = {b}"
        return True


class Q3(ThoughtExperiment):
    """Geometrik tushuntirish: 2D tenglamalar."""
    _hints = [
        "Har bir chiziqli tenglama 2D da to'g'ri chiziq.",
        "Ikkita to'g'ri chiziqning kesishish nuqtasi — yechim.",
    ]
    _solution = (
        "2x + 3y = 7  →  to'g'ri chiziq\n"
        "x - y = 1   →  to'g'ri chiziq\n"
        "Ular kesishgan nuqta (2, 1) — sistemaning yechimi.\n"
        "Matplotlib bilan chizib ko'ring!"
    )


class Q4(UzCheckProblem):
    """3 ta tenglama, 3 ta noma'lum."""
    _hints = [
        "np.linalg.solve(A, b) 3x3 uchun ham ishlaydi.",
        "Avval A (3×3) va b (3,) ni aniqlang.",
    ]
    _solution = (
        "A = np.array([[1,1,1],[2,1,3],[1,2,1]])\n"
        "b = np.array([6, 11, 8])\n"
        "x = np.linalg.solve(A, b)  # [1, 2, 3]"
    )

    def _do_check(self, x, A, b):
        if not np.allclose(A @ x, b, atol=1e-8):
            return f"A@x ≠ b. A@x = {A@x}"
        return True


# Challenge 1
class C1_Q1(UzCheckProblem):
    """Nechta yechim bor? (determinant)"""
    _hints = [
        "det(A) = 0 → yechim yo'q yoki cheksiz ko'p.",
        "np.linalg.det(A) ishlatishingiz mumkin.",
    ]
    _solution = (
        "det = np.linalg.det(A)\n"
        "# det ≠ 0 → bitta yechim\n"
        "# det = 0 → yo'q yoki cheksiz"
    )

    def _do_check(self, det, A):
        expected = np.linalg.det(A)
        if not np.isclose(det, expected, rtol=1e-4):
            return f"Kutilgan: {expected:.4f}, siz berdingiz: {det:.4f}"
        return True


# Challenge 2
class C2_Q1(ThoughtExperiment):
    """Real hayotda chiziqli sistemalar."""
    _hints = [
        "Iqtisodiyot, fizika, elektr sxemalari — barchasi chiziqli sistemalar.",
    ]
    _solution = (
        "Misol: 3 ta tovar, 3 ta narx tengligi.\n"
        "x1 + x2 = 100  (umumiy miqdor)\n"
        "2x1 + 5x2 = 350  (umumiy narx)\n"
        "Yechim: np.linalg.solve(A, b)\n"
        "Bu kabi sistemalar moliya, muhandislik, ML da keng qo'llaniladi."
    )
