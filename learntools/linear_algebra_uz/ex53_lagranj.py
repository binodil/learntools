"""Hints and solutions — Dars 13.2: Lagranj Ko'paytuvchilari."""
import numpy as np
from learntools.linear_algebra_uz.base import UzCheckProblem, ThoughtExperiment


class Q1(UzCheckProblem):
    """Lagranj funksiyasi: L(x,lambda) = f(x) + lambda * g(x)."""
    _hints = ["L = f(x) + lambda_val * g(x). Grad L = grad f + lambda * grad g = 0."]
    _solution = "L = f_val + lambda_val * g_val"

    def _do_check(self, L, f_val, lambda_val, g_val):
        expected = f_val + lambda_val * g_val
        if not np.isclose(L, expected, rtol=1e-8):
            return f"L = f + lambda*g = {expected:.6f}, siz {L:.6f} berdingiz."
        return True


class Q2(UzCheckProblem):
    """KKT shartlari: cheklov g(x*) = 0 va grad f + lambda*grad g = 0."""
    _hints = [
        "KKT: g(x*) = 0, grad_f + lambda*grad_g = 0.",
        "scipy.optimize.minimize bilan equality constraint ham qo'shiladi.",
    ]
    _solution = (
        "from scipy.optimize import minimize\n"
        "res = minimize(f, x0, method='SLSQP',\n"
        "               constraints={'type':'eq','fun': g})\n"
        "x_opt = res.x"
    )

    def _do_check(self, x_opt, f, g, x0):
        from scipy.optimize import minimize
        res = minimize(f, x0, method='SLSQP',
                       constraints={'type': 'eq', 'fun': g})
        if not np.allclose(x_opt, res.x, atol=1e-4):
            return f"Optimal: {res.x}, siz {x_opt} berdingiz."
        if not np.isclose(g(x_opt), 0, atol=1e-4):
            return f"Cheklov g(x*) = 0 bo'lishi kerak: g = {g(x_opt):.4e}"
        return True


class Q3(UzCheckProblem):
    """||x||² minimizatsiya s.t. aᵀx = b (minimal norma yechim)."""
    _hints = [
        "x* = a * b / (aᵀa). Lagranj: L = xᵀx + lambda*(aᵀx - b). grad = 2x + lambda*a = 0.",
    ]
    _solution = "x_star = a * b / (a @ a)"

    def _do_check(self, x_star, a, b):
        expected = a * b / (a @ a)
        if not np.allclose(x_star, expected, atol=1e-8):
            return "x* = a*b/(aᵀa) — minimal norma yechim."
        if not np.isclose(a @ x_star, b, rtol=1e-6):
            return f"Cheklov aᵀx = b bo'lishi kerak: {a@x_star:.6f} != {b}."
        return True


class Q4(UzCheckProblem):
    """SVM hard margin: margin maksimizatsiya = ||w||² minimizatsiya."""
    _hints = [
        "SVM: min ||w||²/2 s.t. y_i(wᵀx_i + b) >= 1. Ekvivalent: wᵀw/2.",
    ]
    _solution = (
        "from scipy.optimize import minimize\n"
        "def objective(params):\n"
        "    w = params[:-1]; b = params[-1]\n"
        "    return 0.5 * np.dot(w, w)\n"
        "def margin_constraint(params, i):\n"
        "    w = params[:-1]; b = params[-1]\n"
        "    return y[i]*(np.dot(w, X[i]) + b) - 1"
    )

    def _do_check(self, margin, w, X, y):
        margins = y * (X @ w)
        expected = 2 / np.linalg.norm(w)
        if not np.isclose(margin, expected, rtol=0.01):
            return f"Margin = 2/||w|| = {expected:.6f}, siz {margin:.6f} berdingiz."
        return True


class Q5(UzCheckProblem):
    """Rayleigh kotirovkasi optimizatsiyasi: max xᵀAx / xᵀx s.t. ||x||=1."""
    _hints = [
        "xᵀAx/xᵀx maksimumi = lambda_max. Lagranj shartida xᵀx=1 → Ax = lambda*x.",
    ]
    _solution = "lambda_max = np.linalg.eigvalsh(A)[-1]; x_opt = np.linalg.eigh(A)[1][:,-1]"

    def _do_check(self, lambda_max, x_opt, A):
        vals, vecs = np.linalg.eigh(A)
        if not np.isclose(lambda_max, vals[-1], rtol=1e-6):
            return f"lambda_max = {vals[-1]:.6f}, siz {lambda_max:.6f} berdingiz."
        if not (np.allclose(x_opt, vecs[:,-1], atol=1e-6) or
                np.allclose(x_opt, -vecs[:,-1], atol=1e-6)):
            return "x_opt — eng katta xususiy qiymatga mos vektor."
        return True


class Q6(UzCheckProblem):
    """Umumlashtirilgan xususiy qiymat: Ax = lambda*Bx (generalized eigenvalue)."""
    _hints = [
        "scipy.linalg.eigh(A, B) — umumlashtirilgan xususiy qiymat masalasi.",
    ]
    _solution = "from scipy.linalg import eigh; vals, vecs = eigh(A, B)"

    def _do_check(self, vals, A, B):
        from scipy.linalg import eigh
        expected_vals, _ = eigh(A, B)
        if not np.allclose(np.sort(vals), np.sort(expected_vals), atol=1e-6):
            return f"Umumlashtirilgan xususiy qiymatlar: {expected_vals}"
        return True


class C1_Q1(UzCheckProblem):
    """Portfolio optimizatsiya: min wᵀΣw s.t. rᵀw=mu_target, 1ᵀw=1."""
    _hints = [
        "Lagranj: L = wᵀΣw + lambda1*(rᵀw-mu) + lambda2*(1ᵀw-1). KKT → chiziqli sistema.",
    ]
    _solution = (
        "from scipy.optimize import minimize\n"
        "def risk(w): return w @ Sigma @ w\n"
        "constraints = [{'type':'eq','fun': lambda w: r@w - mu_target},\n"
        "               {'type':'eq','fun': lambda w: w.sum()-1}]\n"
        "bounds = [(0,1)]*n\n"
        "res = minimize(risk, np.ones(n)/n, constraints=constraints, bounds=bounds)\n"
        "w_opt = res.x"
    )

    def _do_check(self, w_opt, Sigma, r, mu_target):
        if not np.isclose(w_opt.sum(), 1, atol=1e-3):
            return f"Vazn yig'indisi 1 bo'lishi kerak: {w_opt.sum():.4f}"
        if not np.isclose(r @ w_opt, mu_target, atol=1e-3):
            return f"Kutilgan qaytim: {r@w_opt:.4f}, target: {mu_target}"
        return True


class C2_Q1(ThoughtExperiment):
    """Lagranj ko'paytuvchilari va KKT shartlari optimallashtirishda nima uchun muhim?"""
    _hints = [
        "Lagranj = cheklangan optimallashtirish standarti. lambda = shadow price.",
    ]
    _solution = (
        "Lagranj ko'paytuvchilari ahamiyati:\n\n"
        "1) Cheklangan optimallashtirish uchun zaruriy shart: KKT.\n"
        "   Unconstrained uchun: grad f = 0. Constrained: grad f + lambda*grad g = 0.\n\n"
        "2) lambda — 'shadow price': cheklov qiymatining o'zgarishi optimal qiymatga ta'siri.\n\n"
        "3) ML qo'llanishlari:\n"
        "   - SVM: margin maks. → duali LP (alpha Lagranj multiplierlari).\n"
        "   - Portfolio optimallashtirish (Markowitz).\n"
        "   - Regularizatsiya: L2 regularizer = norm cheklovi.\n\n"
        "4) KKT shartlari DL da: batch normalization, weight constraint."
    )
