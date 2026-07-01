"""Hints and solutions — Dars 4.2: Proyeksiyalar."""
import numpy as np
from learntools.linear_algebra_uz.base import UzCheckProblem, ThoughtExperiment


class Q1(UzCheckProblem):
    """v vektorini a yo'nalishiga proyeksiyalang."""
    _hints = [
        "proj_a(v) = (v·a / a·a) * a formulasidan foydalaning.",
        "proj = (np.dot(v, a) / np.dot(a, a)) * a",
    ]
    _solution = "proj = (np.dot(v, a) / np.dot(a, a)) * a"

    def _do_check(self, proj, v, a):
        expected = (np.dot(v, a) / np.dot(a, a)) * a
        if not np.allclose(proj, expected):
            return f"Kutilgan: {expected}, siz berdingiz: {proj}"
        return True


class Q2(UzCheckProblem):
    """Proyeksiya matritsasini hisoblang: P = a aᵀ / (aᵀa)."""
    _hints = [
        "P = np.outer(a, a) / np.dot(a, a)",
        "P @ v proyeksiya vektorini beradi.",
    ]
    _solution = "P = np.outer(a, a) / np.dot(a, a)"

    def _do_check(self, P, a):
        expected = np.outer(a, a) / np.dot(a, a)
        if not np.allclose(P, expected):
            return f"Proyeksiya matritsasi noto'g'ri. P = aaᵀ/(aᵀa) formulasini tekshiring."
        return True


class Q3(UzCheckProblem):
    """P² = P (idempotentlik) ni tekshiring."""
    _hints = [
        "Proyeksiya matritsasi P uchun P @ P == P bo'lishi kerak.",
        "np.allclose(P @ P, P) dan foydalaning.",
    ]
    _solution = "idempotent = np.allclose(P @ P, P)"

    def _do_check(self, idempotent, P):
        expected = np.allclose(P @ P, P)
        if idempotent != expected:
            return f"P² = P bo'lishi kerak. np.allclose(P @ P, P) = {expected}"
        return True


class Q4(UzCheckProblem):
    """b vektorini A ustun fazosiga proyeksiyalang: p = A(AᵀA)⁻¹Aᵀb."""
    _hints = [
        "p = A @ np.linalg.inv(A.T @ A) @ A.T @ b",
        "Yoki: xhat = np.linalg.lstsq(A, b, rcond=None)[0]; p = A @ xhat",
    ]
    _solution = "x_hat = np.linalg.lstsq(A, b, rcond=None)[0]; p = A @ x_hat"

    def _do_check(self, p, A, b):
        x_hat = np.linalg.lstsq(A, b, rcond=None)[0]
        expected = A @ x_hat
        if not np.allclose(p, expected, atol=1e-8):
            return f"Kutilgan proyeksiya: {expected}, siz berdingiz: {p}"
        return True


class Q5(UzCheckProblem):
    """Qoldiqni (error e = b - p) hisoblang va p ga perpendikulyar ekanini tekshiring."""
    _hints = [
        "e = b - p, keyin np.dot(e, p) ≈ 0 bo'lishi kerak.",
        "Proyeksiya va xato vektori har doim perpendikulyar.",
    ]
    _solution = "e = b - p; ortogonal = np.isclose(np.dot(e, p), 0)"

    def _do_check(self, e, b, p):
        expected_e = b - p
        if not np.allclose(e, expected_e, atol=1e-8):
            return f"e = b - p bo'lishi kerak. Kutilgan: {expected_e}"
        if not np.isclose(np.dot(e, p), 0, atol=1e-8):
            return f"e va p ortogonal bo'lishi kerak, lekin e·p = {np.dot(e, p):.6f}"
        return True


class Q6(UzCheckProblem):
    """To'liq proyeksiya matritsasini hisoblang: P = A(AᵀA)⁻¹Aᵀ."""
    _hints = [
        "P = A @ np.linalg.inv(A.T @ A) @ A.T",
        "P simmetrik va idempotent bo'lishi kerak: P = Pᵀ va P² = P.",
    ]
    _solution = "P = A @ np.linalg.inv(A.T @ A) @ A.T"

    def _do_check(self, P, A):
        AtA = A.T @ A
        expected = A @ np.linalg.inv(AtA) @ A.T
        if not np.allclose(P, expected, atol=1e-8):
            return "P = A(AᵀA)⁻¹Aᵀ formulasini tekshiring."
        if not np.allclose(P, P.T, atol=1e-8):
            return "P simmetrik bo'lishi kerak: P = Pᵀ."
        if not np.allclose(P @ P, P, atol=1e-8):
            return "P idempotent bo'lishi kerak: P² = P."
        return True


class C1_Q1(UzCheckProblem):
    """Uch nuqta orqali eng yaxshi to'g'ri chiziqni toping (least squares)."""
    _hints = [
        "y = c + dx shaklida: A = [[1,x1],[1,x2],[1,x3]], b = [y1,y2,y3].",
        "x_hat = np.linalg.lstsq(A, b, rcond=None)[0] — [c, d] koeffitsiyentlarini beradi.",
    ]
    _solution = (
        "# x_pts = [0,1,2], y_pts = [6,0,0] uchun:\n"
        "A = np.column_stack([np.ones(3), x_pts])\n"
        "c, d = np.linalg.lstsq(A, y_pts, rcond=None)[0]"
    )

    def _do_check(self, coeffs, x_pts, y_pts):
        A = np.column_stack([np.ones(len(x_pts)), x_pts])
        expected = np.linalg.lstsq(A, y_pts, rcond=None)[0]
        if not np.allclose(coeffs, expected, atol=1e-6):
            return f"Kutilgan koeffitsiyentlar [c, d] = {expected}, siz {coeffs} berdingiz."
        return True


class C2_Q1(ThoughtExperiment):
    """Proyeksiya matritsasi P ning xususiy qiymatlari faqat 0 va 1 bo'lishi sababini tushuntiring."""
    _hints = [
        "P² = P bo'lsa, P ning har bir xususiy vektori v uchun P²v = Pv ni ko'ring.",
        "λ²v = λv → λ(λ-1)v = 0, demak λ=0 yoki λ=1.",
    ]
    _solution = (
        "P idempotent: P² = P. Agar Pv = λv bo'lsa, P²v = P(Pv) = P(λv) = λPv = λ²v. "
        "Lekin P²v = Pv = λv, shuning uchun λ²v = λv, ya'ni λ²=λ, demak λ(λ-1)=0. "
        "Bu esa λ=0 yoki λ=1 ekanini bildiradi. "
        "λ=1 ga mos keluvchi xususiy vektorlar — ustun fazosida, "
        "λ=0 ga mos keluvchi vektorlar — ortogonal to'ldiruvchida."
    )
