"""Hints and solutions — Dars 8.1: Chiziqli Transformatsiya g'oyasi."""
import numpy as np
from learntools.core import EqualityCheckProblem, ThoughtExperiment


class Q1(EqualityCheckProblem):
    """T(v) = Av chiziqli ekanini tekshiring: T(u+v) = T(u) + T(v)."""
    _hints = [
        "T chiziqli bo'lsa: T(u+v) = T(u) + T(v) va T(cu) = cT(u).",
        "np.allclose(A @ (u + v), A @ u + A @ v)",
    ]
    _solution = "additive = np.allclose(A @ (u + v), A @ u + A @ v)"

    def _do_check(self, additive, A, u, v):
        expected = np.allclose(A @ (u + v), A @ u + A @ v)
        if additive != expected:
            return f"T(u+v) = T(u)+T(v)? {expected}. Matritsa ko'paytmasini tekshiring."
        return True


class Q2(EqualityCheckProblem):
    """T(cv) = cT(v) homogenlik shartini tekshiring."""
    _hints = [
        "np.allclose(A @ (c * v), c * (A @ v))",
    ]
    _solution = "homogeneous = np.allclose(A @ (c * v), c * (A @ v))"

    def _do_check(self, homogeneous, A, c, v):
        expected = np.allclose(A @ (c * v), c * (A @ v))
        if homogeneous != expected:
            return f"T(cv) = cT(v)? {expected}. np.allclose(A @ (c*v), c*(A@v))"
        return True


class Q3(EqualityCheckProblem):
    """Proyeksiya transformatsiyasining matritsasini toping."""
    _hints = [
        "a yo'nalishiga proyeksiya: P = a aᵀ / aᵀa. Har bir vektor uchun T(v) = Pv.",
    ]
    _solution = "P = np.outer(a, a) / (a @ a)"

    def _do_check(self, P, a):
        expected = np.outer(a, a) / (a @ a)
        if not np.allclose(P, expected, atol=1e-8):
            return "P = a aᵀ / (aᵀa) formulasini tekshiring."
        if not np.allclose(P @ P, P, atol=1e-8):
            return "Proyeksiya idempotent bo'lishi kerak: P² = P."
        return True


class Q4(EqualityCheckProblem):
    """O'q atrofida aks ettirish (reflection) matritsasini hisoblang."""
    _hints = [
        "x-o'q atrofida aks ettirish: [[1,0],[0,-1]]. Umumiy: R = 2P - I.",
        "P = a aᵀ / aᵀa, R = 2P - I.",
    ]
    _solution = "P = np.outer(a, a) / (a @ a); R = 2 * P - np.eye(len(a))"

    def _do_check(self, R, a):
        P = np.outer(a, a) / (a @ a)
        expected = 2 * P - np.eye(len(a))
        if not np.allclose(R, expected, atol=1e-8):
            return "R = 2P - I formulasini tekshiring, bu erda P — a ga proyeksiya."
        if not np.allclose(R @ R, np.eye(len(a)), atol=1e-8):
            return "Aks ettirish: R² = I bo'lishi kerak."
        return True


class Q5(EqualityCheckProblem):
    """2D da burchak θ bo'yicha aylantirish matritsasini toping."""
    _hints = [
        "Rotatsiya matritsasi: [[cos θ, -sin θ], [sin θ, cos θ]].",
        "np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])",
    ]
    _solution = "R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])"

    def _do_check(self, R, theta):
        expected = np.array([[np.cos(theta), -np.sin(theta)],
                             [np.sin(theta), np.cos(theta)]])
        if not np.allclose(R, expected, atol=1e-8):
            return "Rotatsiya: [[cos θ, -sin θ], [sin θ, cos θ]] formulasini tekshiring."
        if not np.allclose(R @ R.T, np.eye(2), atol=1e-8):
            return "Rotatsiya matritsasi ortogonal: RRᵀ = I bo'lishi kerak."
        return True


class Q6(EqualityCheckProblem):
    """Ikki transformatsiyaning kompozitsiyasini hisoblang: T₂ ∘ T₁."""
    _hints = [
        "T₂(T₁(v)) = A₂ (A₁ v) = (A₂ A₁) v. Kompozitsiya: C = A₂ @ A₁.",
    ]
    _solution = "C = A2 @ A1  # T2 ∘ T1"

    def _do_check(self, C, A1, A2):
        expected = A2 @ A1
        if not np.allclose(C, expected, atol=1e-8):
            return "C = A2 @ A1. Matritsa ko'paytmasi tartibiga e'tibor bering."
        return True


class C1_Q1(EqualityCheckProblem):
    """3D da X, Y, Z o'qlar atrofida rotatsiya kompozitsiyasi."""
    _hints = [
        "Rx, Ry, Rz — mos o'qlar atrofida rotatsiya matritsalari.",
        "Rz(γ) @ Ry(β) @ Rx(α) — Euler burchaklari kompozitsiyasi.",
    ]
    _solution = (
        "def Rx(a): return np.array([[1,0,0],[0,np.cos(a),-np.sin(a)],[0,np.sin(a),np.cos(a)]])\n"
        "def Ry(b): return np.array([[np.cos(b),0,np.sin(b)],[0,1,0],[-np.sin(b),0,np.cos(b)]])\n"
        "def Rz(g): return np.array([[np.cos(g),-np.sin(g),0],[np.sin(g),np.cos(g),0],[0,0,1]])\n"
        "R_total = Rz(gamma) @ Ry(beta) @ Rx(alpha)"
    )

    def _do_check(self, R_total, alpha, beta, gamma):
        def Rx(a):
            return np.array([[1,0,0],[0,np.cos(a),-np.sin(a)],[0,np.sin(a),np.cos(a)]])
        def Ry(b):
            return np.array([[np.cos(b),0,np.sin(b)],[0,1,0],[-np.sin(b),0,np.cos(b)]])
        def Rz(g):
            return np.array([[np.cos(g),-np.sin(g),0],[np.sin(g),np.cos(g),0],[0,0,1]])
        expected = Rz(gamma) @ Ry(beta) @ Rx(alpha)
        if not np.allclose(R_total, expected, atol=1e-6):
            return "R_total = Rz(gamma) @ Ry(beta) @ Rx(alpha) formulasini tekshiring."
        return True


class C2_Q1(ThoughtExperiment):
    """Nima uchun chiziqli transformatsiyalar matritsa ko'paytmasi bilan ifodalanadi?"""
    _hints = [
        "Baza vektorlar T(e₁), T(e₂), ... transformatsiyani to'liq belgilaydi.",
        "Har qanday v = Σ cᵢeᵢ uchun T(v) = Σ cᵢ T(eᵢ) — chiziqlilikdan.",
    ]
    _solution = (
        "Chiziqli transformatsiya T: Rⁿ → Rᵐ to'liq ravishda uning baza vektorlardagi "
        "qiymatlari bilan belgilanadi:\n"
        "v = c₁e₁ + c₂e₂ + ... + cₙeₙ bo'lsa,\n"
        "T(v) = c₁T(e₁) + c₂T(e₂) + ... + cₙT(eₙ).\n\n"
        "A matritsasining j-ustuni = T(eⱼ). Shuning uchun T(v) = Av.\n\n"
        "Bu degani: barcha chiziqli transformatsiyalar ↔ barcha m×n matritsalar.\n"
        "Matritsa ko'paytmasi AB = 'avval B, keyin A' transformatsiyasi."
    )
