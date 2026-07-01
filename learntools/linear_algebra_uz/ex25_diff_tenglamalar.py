"""Hints and solutions — Dars 6.3: Differentsial Tenglamalar Sistemasi."""
import numpy as np
from learntools.core import EqualityCheckProblem, ThoughtExperiment


class Q1(EqualityCheckProblem):
    """du/dt = Au ning yechimi u(t) = e^{λt}x."""
    _hints = [
        "A ning xususiy qiymatlari λ va xususiy vektorlari x uchun u(t) = e^{λt}x yechimdir.",
        "np.linalg.eig(A) — (eigenvalues, eigenvectors) ni qaytaradi.",
    ]
    _solution = "vals, vecs = np.linalg.eig(A)"

    def _do_check(self, vals, vecs, A):
        exp_vals, exp_vecs = np.linalg.eig(A)
        # Check eigenvalues (order may differ)
        if not np.allclose(np.sort(np.real(vals)), np.sort(np.real(exp_vals)), atol=1e-6):
            return f"Xususiy qiymatlar noto'g'ri. Kutilgan: {np.sort(np.real(exp_vals))}"
        return True


class Q2(EqualityCheckProblem):
    """Matritsa eksponentsiali e^{At} ni hisoblang (scipy)."""
    _hints = [
        "scipy.linalg.expm(A * t) — matritsa eksponentsiasini beradi.",
        "e^{At} = I + At + (At)²/2! + ... = S e^{Λt} S⁻¹",
    ]
    _solution = "from scipy.linalg import expm; eAt = expm(A * t)"

    def _do_check(self, eAt, A, t):
        from scipy.linalg import expm
        expected = expm(A * t)
        if not np.allclose(eAt, expected, atol=1e-6):
            return "scipy.linalg.expm(A * t) formulasini tekshiring."
        return True


class Q3(EqualityCheckProblem):
    """u(t) = e^{At} u(0) — boshlang'ich shart bilan yechim."""
    _hints = [
        "u(t) = expm(A*t) @ u0, bu erda u0 = u(0) boshlang'ich shart.",
    ]
    _solution = "from scipy.linalg import expm; u_t = expm(A * t) @ u0"

    def _do_check(self, u_t, A, t, u0):
        from scipy.linalg import expm
        expected = expm(A * t) @ u0
        if not np.allclose(u_t, expected, atol=1e-6):
            return f"u(t) = e^{{At}} u(0). Kutilgan: {expected}"
        return True


class Q4(EqualityCheckProblem):
    """Barqarorlik: barcha λ < 0 bo'lsa u(t)→0."""
    _hints = [
        "Barcha xususiy qiymatlarning haqiqiy qismi manfiy bo'lsa — barqaror.",
        "np.all(np.real(np.linalg.eig(A)[0]) < 0)",
    ]
    _solution = "stable = np.all(np.real(np.linalg.eig(A)[0]) < 0)"

    def _do_check(self, stable, A):
        expected = np.all(np.real(np.linalg.eig(A)[0]) < 0)
        if stable != expected:
            vals = np.real(np.linalg.eig(A)[0])
            return f"Haqiqiy qismlar: {vals}. Barqarormi? {expected}"
        return True


class Q5(EqualityCheckProblem):
    """2x2 tizimni Fibonacci-ga o'xshash iteratsiya bilan yeching."""
    _hints = [
        "Fibonacci: F_{n+1} = F_n + F_{n-1} → [[F_{n+1}],[F_n]] = A^n [[1],[0]]",
        "A = [[1,1],[1,0]], keyin A^n @ [1,0]^T birinchi komponent F_{n+1}.",
    ]
    _solution = "A = np.array([[1,1],[1,0]]); Fn = int(round((np.linalg.matrix_power(A, n) @ np.array([1,0]))[1]))"

    def _do_check(self, Fn, n):
        A = np.array([[1, 1], [1, 0]])
        expected = int(round((np.linalg.matrix_power(A, n) @ np.array([1, 0]))[1]))
        if Fn != expected:
            return f"F_{n} = {expected}, siz {Fn} berdingiz."
        return True


class Q6(EqualityCheckProblem):
    """2-tartibli ODE ni 1-tartibli sistemaga keltiring."""
    _hints = [
        "y'' + py' + qy = 0 → u = [y, y']^T, du/dt = [[0,1],[-q,-p]] u.",
        "Hamiltonian shakl: A = np.array([[0, 1], [-q, -p]])",
    ]
    _solution = "A = np.array([[0, 1], [-q, -p]])"

    def _do_check(self, A, p, q):
        expected = np.array([[0, 1], [-q, -p]])
        if not np.allclose(A, expected):
            return f"A = [[0,1],[-q,-p]] = [[0,1],[{-q},{-p}]] bo'lishi kerak."
        return True


class C1_Q1(EqualityCheckProblem):
    """Diagonallashtirish orqali e^{At} = S e^{Λt} S⁻¹ ni hisoblang."""
    _hints = [
        "vals, S = np.linalg.eig(A). Keyin Λt = diag(vals*t), e^{At} = S @ diag(exp(vals*t)) @ inv(S).",
    ]
    _solution = (
        "vals, S = np.linalg.eig(A)\n"
        "eAt = S @ np.diag(np.exp(vals * t)) @ np.linalg.inv(S)"
    )

    def _do_check(self, eAt, A, t):
        from scipy.linalg import expm
        expected = expm(A * t)
        if not np.allclose(np.real(eAt), np.real(expected), atol=1e-5):
            return "e^{At} = S diag(e^{λt}) S⁻¹ formulasini tekshiring."
        return True


class C2_Q1(ThoughtExperiment):
    """Nima uchun xususiy qiymatlar ODE barqarorligini belgilaydi?"""
    _hints = [
        "Yechim u(t) = c₁e^{λ₁t}x₁ + c₂e^{λ₂t}x₂ ko'rinishida.",
        "e^{λt} → 0 faqat Re(λ) < 0 bo'lganda.",
    ]
    _solution = (
        "du/dt = Au tenglamasining umumiy yechimi: u(t) = Σᵢ cᵢ e^{λᵢt} xᵢ. "
        "Bu yerda e^{λt} = e^{(a+bi)t} = e^{at}(cos(bt) + i·sin(bt)). "
        "Demak |e^{λt}| = e^{at} → 0 faqat a = Re(λ) < 0 bo'lganda. "
        "Agar barcha xususiy qiymatlarning haqiqiy qismi manfiy bo'lsa — tizim barqaror, "
        "u(t) → 0 deb t → ∞."
    )
