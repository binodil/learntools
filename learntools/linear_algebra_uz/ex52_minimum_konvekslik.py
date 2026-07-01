"""Hints and solutions — Dars 13.1: Minimum Masalalari: Konvekslik va Newton Usuli."""
import numpy as np
from learntools.linear_algebra_uz.base import UzCheckProblem, ThoughtExperiment


class Q1(UzCheckProblem):
    """f(x) = xᵀAx + bᵀx konveks ekanini tekshiring: A musbat yarim aniq bo'lsa."""
    _hints = ["A konveks uchun PSD bo'lishi kerak: barcha xususiy qiymatlar >= 0."]
    _solution = "is_convex = np.all(np.linalg.eigvalsh(A) >= -1e-10)"

    def _do_check(self, is_convex, A):
        expected = np.all(np.linalg.eigvalsh(A) >= -1e-10)
        if is_convex != expected:
            return f"Konveks? {expected}. Xususiy qiymatlar: {np.linalg.eigvalsh(A)}"
        return True


class Q2(UzCheckProblem):
    """f(x) = xᵀAx + bᵀx minimumini toping: grad f = 2Ax + b = 0."""
    _hints = ["Grad f = 2Ax + b = 0 → x* = -0.5 A^{-1} b."]
    _solution = "x_star = -0.5 * np.linalg.solve(A, b)"

    def _do_check(self, x_star, A, b):
        expected = -0.5 * np.linalg.solve(A, b)
        if not np.allclose(x_star, expected, atol=1e-6):
            return "x* = -0.5 A^{-1} b (grad = 0 shartidan)."
        grad = 2 * A @ x_star + b
        if not np.allclose(grad, 0, atol=1e-6):
            return f"Gradient nolga teng emas: {grad}."
        return True


class Q3(UzCheckProblem):
    """Hessian matritsasini hisoblang: H = ∂²f/∂x²."""
    _hints = ["f(x) = xᵀAx → H = 2A. f(x) = sum(xi^4) → H = diag(12*x^2)."]
    _solution = "H = 2 * A  # kvadratik funksiya uchun"

    def _do_check(self, H, A):
        expected = 2 * A
        if not np.allclose(H, expected, atol=1e-8):
            return "Kvadratik f = xᵀAx uchun H = 2A."
        return True


class Q4(UzCheckProblem):
    """Newton qadam: x_{k+1} = x_k - H^{-1} grad f."""
    _hints = ["p = np.linalg.solve(H, -grad). x_new = x + p (Newton qadam)."]
    _solution = "p = np.linalg.solve(H, -grad); x_new = x + p"

    def _do_check(self, x_new, x, H, grad):
        p = np.linalg.solve(H, -grad)
        expected = x + p
        if not np.allclose(x_new, expected, atol=1e-8):
            return "x_new = x - H^{-1} grad."
        return True


class Q5(UzCheckProblem):
    """Newton usuli bilan kvadratik funksiyaning minimumini toping (1 qadamda)."""
    _hints = [
        "f(x) = xᵀAx + bᵀx uchun Newton 1 qadamda yetadi: x* = -A^{-1}b/2.",
    ]
    _solution = (
        "x = np.zeros(len(b))\n"
        "grad = 2*A@x + b\n"
        "H = 2*A\n"
        "x = x - np.linalg.solve(H, grad)"
    )

    def _do_check(self, x, A, b):
        expected = -0.5 * np.linalg.solve(A, b)
        if not np.allclose(x, expected, atol=1e-6):
            return f"Newton minimumi: {expected}, siz {x} berdingiz."
        return True


class Q6(UzCheckProblem):
    """Ikkinchi darajali optimizatsiya: f(x) = (x-3)^4 uchun Newton iteratsiyalari."""
    _hints = [
        "f'(x) = 4(x-3)^3, f''(x) = 12(x-3)^2. x_new = x - f'/f''.",
    ]
    _solution = (
        "x = 5.0  # boshlang'ich\n"
        "for _ in range(50):\n"
        "    grad = 4*(x-3)**3\n"
        "    hess = 12*(x-3)**2\n"
        "    if abs(hess) < 1e-12: break\n"
        "    x = x - grad/hess"
    )

    def _do_check(self, x, target):
        if not np.isclose(x, target, atol=1e-4):
            return f"Minimum: x={target}, siz {x:.6f} berdingiz."
        return True


class C1_Q1(UzCheckProblem):
    """Logistik regressiya Hessian musbat yarim aniq ekanini tekshiring."""
    _hints = [
        "Logistik regressiya H = XᵀDX, D = diag(p*(1-p)). Bu PSD.",
    ]
    _solution = (
        "p = 1/(1+np.exp(-X @ w))\n"
        "D = np.diag(p*(1-p))\n"
        "H = X.T @ D @ X\n"
        "is_psd = np.all(np.linalg.eigvalsh(H) >= -1e-10)"
    )

    def _do_check(self, is_psd, X, w):
        p = 1 / (1 + np.exp(-X @ w))
        D = np.diag(p * (1 - p))
        H = X.T @ D @ X
        expected = np.all(np.linalg.eigvalsh(H) >= -1e-10)
        if is_psd != expected:
            return f"Hessian PSD? {expected}. XᵀDX, D=diag(p(1-p))."
        return True


class C2_Q1(ThoughtExperiment):
    """Konvekslik va Newton usuli optimallashtirish uchun nima uchun muhim?"""
    _hints = [
        "Konveks → lokal minimum = global minimum. Newton → kvadratik konvergentsiya.",
    ]
    _solution = (
        "Konvekslik optimallashtirish uchun ahamiyati:\n\n"
        "1) Konveks muammo: har qanday lokal minimum = global minimum.\n"
        "   DL: loss funksiyalari ko'pincha konveks (MSE, cross-entropy).\n\n"
        "2) Newton usuli afzalliklari:\n"
        "   - Kvadratik konvergentsiya: xato^{k+1} ≈ C * xato^k².\n"
        "   - Gradient tushishdan ancha tez (ayniqsa katta konditsion son uchun).\n"
        "   - Kvazi-Newton (L-BFGS, BFGS): H ni taqribiy hisoblash.\n\n"
        "3) Cheklov: H = n×n matritsa → O(n³) — katta n uchun qimmat.\n"
        "   SGD/Adam: H ni hisoblamaydi, lekin sekin konvergentsiya."
    )
