"""Hints and solutions — Dars 6.1: Xususiy Qiymatlarga Kirish."""
import numpy as np
from learntools.core import EqualityCheckProblem, ThoughtExperiment


class Q1(EqualityCheckProblem):
    """Diagonal matritsa xususiy qiymatlari."""
    _hints = [
        "Diagonal matritsa uchun xususiy qiymatlar — diagonal elementlar.",
        "diag(3,2) => lambda = 3 va 2.",
    ]
    _solution = "lams = np.array([3, 2])  # diagonal elementlar"

    def _do_check(self, lams):
        if not np.allclose(np.sort(lams), [2, 3]):
            return f"Kutilgan: [2, 3], siz berdingiz: {np.sort(lams)}"
        return True


class Q2(EqualityCheckProblem):
    """Xarakteristik tenglamadan lambda toping (2x2)."""
    _hints = [
        "det(A - lambda I) = 0 ni yeching.",
        "A=[[2,1],[1,2]]: lambda^2 - 4 lambda + 3 = 0 => 3 va 1.",
    ]
    _solution = "lams = np.linalg.eigvals(A)  # [3, 1]"

    def _do_check(self, lams, A):
        expected = np.sort(np.linalg.eigvals(A))
        if not np.allclose(np.sort(lams), expected):
            return f"Kutilgan: {expected}, siz berdingiz: {np.sort(lams)}"
        return True


class Q3(EqualityCheckProblem):
    """A x = lambda x ni tekshiring."""
    _hints = [
        "x — xususiy vektor, lambda — mos qiymat.",
        "A @ x natijasi lambda * x ga teng bo'lishi kerak.",
    ]
    _solution = "result = A @ x  # == lam * x"

    def _do_check(self, result, A, x, lam):
        expected = lam * x
        if not np.allclose(result, expected):
            return f"Kutilgan: {expected}, siz berdingiz: {result}"
        return True


class Q4(EqualityCheckProblem):
    """Iz = xususiy qiymatlar yig'indisi."""
    _hints = [
        "trace(A) = lambda_1 + lambda_2 + ... .",
        "np.trace(A) ni hisoblang.",
    ]
    _solution = "result = np.trace(A)"

    def _do_check(self, result, A):
        expected = np.sum(np.linalg.eigvals(A))
        if not np.isclose(np.real(result), np.real(expected)):
            return f"Kutilgan: {np.real(expected):.4g}, siz berdingiz: {np.real(result):.4g}"
        return True


class Q5(EqualityCheckProblem):
    """det = xususiy qiymatlar ko'paytmasi."""
    _hints = [
        "det(A) = lambda_1 * lambda_2.",
        "np.linalg.det(A) ni eigvals ko'paytmasi bilan solishtiring.",
    ]
    _solution = "result = np.prod(np.linalg.eigvals(A))"

    def _do_check(self, result, A):
        expected = np.linalg.det(A)
        if not np.isclose(np.real(result), expected):
            return f"Kutilgan: {expected:.4g}, siz berdingiz: {np.real(result):.4g}"
        return True


class Q6(EqualityCheckProblem):
    """(A - lambda I) singular ekanini tekshiring (det = 0)."""
    _hints = [
        "Agar lambda haqiqiy xususiy qiymat bo'lsa, det(A - lambda I) = 0.",
        "np.linalg.det(A - lam*np.eye(n)) ni hisoblang.",
    ]
    _solution = "result = np.linalg.det(A - lam*np.eye(A.shape[0]))  # ~ 0"

    def _do_check(self, result, A, lam):
        if not np.isclose(result, 0, atol=1e-8):
            return f"Kutilgan: ~0, siz berdingiz: {result:.4g} (lambda xususiy qiymatmi?)"
        return True


class C1_Q1(EqualityCheckProblem):
    """Xususiy vektorni normallashtiring va A x = lambda x tekshiring."""
    _hints = [
        "(A - lambda I) x = 0 yechimini toping.",
        "lambda=3 uchun A=[[2,1],[1,2]]: x = [1,1] yo'nalish.",
    ]
    _solution = "x = np.array([1.,1.]); # A@x == 3*x"

    def _do_check(self, x, A, lam):
        Ax = A @ x
        if np.allclose(x, 0):
            return "Xususiy vektor nolmas bo'lishi kerak."
        if not np.allclose(Ax, lam * x):
            return f"A@x = {Ax}, lekin lam*x = {lam*x}. Bu xususiy vektor emas."
        return True


class C2_Q1(ThoughtExperiment):
    """Aylantirish matritsasining xususiy qiymatlari nima uchun kompleks?"""
    _hints = [
        "Aylantirish hech bir haqiqiy vektorni o'z yo'nalishida qoldirmaydi.",
        "Xarakteristik tenglama lambda^2 + 1 = 0 (90 daraja uchun).",
    ]
    _solution = (
        "R = [[cos t, -sin t],[sin t, cos t]] aylantirish matritsasi har bir haqiqiy\n"
        "vektorni burab yuboradi, shuning uchun A x = lambda x ni qanoatlantiruvchi\n"
        "haqiqiy vektor yo'q. Xarakteristik tenglama lambda^2 - 2cos(t) lambda + 1 = 0\n"
        "bo'lib, ildizlar e^{+-i t} — kompleks. |lambda| = 1, ya'ni uzunlik saqlanadi,\n"
        "faqat faza (burchak) o'zgaradi."
    )
