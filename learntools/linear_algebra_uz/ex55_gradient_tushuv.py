"""Hints and solutions — Dars 13.4: Gradient Tushuv."""
import numpy as np
from learntools.core import EqualityCheckProblem, ThoughtExperiment


class Q1(EqualityCheckProblem):
    """Gradientni hisoblang: f(x) = xᵀAx + bᵀx → grad = 2Ax + b."""
    _hints = ["f(x) = xᵀAx + bᵀx → grad = 2Ax + b."]
    _solution = "grad = 2 * A @ x + b"

    def _do_check(self, grad, x, A, b):
        expected = 2 * A @ x + b
        if not np.allclose(grad, expected, atol=1e-8):
            return "grad f = 2Ax + b."
        return True


class Q2(EqualityCheckProblem):
    """Bir qadam gradient tushuv: x_new = x - lr * grad."""
    _hints = ["x_new = x - learning_rate * gradient."]
    _solution = "x_new = x - lr * grad"

    def _do_check(self, x_new, x, lr, grad):
        expected = x - lr * grad
        if not np.allclose(x_new, expected, atol=1e-8):
            return "x_new = x - lr * grad."
        return True


class Q3(EqualityCheckProblem):
    """Gradient tushuv orqali f(x) = xᵀAx + bᵀx minimumini toping."""
    _hints = [
        "Iteratsiya: x = x - lr * (2*A@x + b). lr kichik bo'lsin.",
    ]
    _solution = (
        "x = np.zeros(len(b))\n"
        "for _ in range(n_iters):\n"
        "    grad = 2*A@x + b\n"
        "    x = x - lr * grad"
    )

    def _do_check(self, x, A, b, tol=1e-4):
        expected = -0.5 * np.linalg.solve(A, b)
        grad_norm = np.linalg.norm(2 * A @ x + b)
        if grad_norm > tol:
            return f"Gradient norma {grad_norm:.2e} — hali yaqinlashmagan. Iteratsiya ko'paytiring."
        if not np.allclose(x, expected, atol=1e-2):
            return f"Optimal: {expected}, siz {x} berdingiz."
        return True


class Q4(EqualityCheckProblem):
    """Momentum bilan gradient tushuv: v = beta*v + grad; x = x - lr*v."""
    _hints = [
        "Heavy ball: v = beta*v + grad_f(x); x = x - lr*v.",
    ]
    _solution = (
        "v = np.zeros_like(x)\n"
        "for _ in range(n_iters):\n"
        "    grad = gradient_fn(x)\n"
        "    v = beta * v + grad\n"
        "    x = x - lr * v"
    )

    def _do_check(self, x, gradient_fn, x_true, tol=1e-3):
        if not np.allclose(x, x_true, atol=tol):
            return f"Momentum bilan gradient tushuv minimumi: {x_true}"
        return True


class Q5(EqualityCheckProblem):
    """Adaptiv o'rganish tezligi: Adagrad — grad^2 yig'indisi bo'yicha normallashtirish."""
    _hints = [
        "Adagrad: G += grad^2; x = x - lr/sqrt(G+eps) * grad.",
    ]
    _solution = (
        "G = np.zeros_like(x); eps = 1e-8\n"
        "for _ in range(n_iters):\n"
        "    grad = gradient_fn(x)\n"
        "    G += grad**2\n"
        "    x = x - lr / np.sqrt(G + eps) * grad"
    )

    def _do_check(self, x, gradient_fn, x_true, tol=1e-2):
        if not np.allclose(x, x_true, atol=tol):
            return f"Adagrad minimumi: {x_true}"
        return True


class Q6(EqualityCheckProblem):
    """O'rganish tezligini tanlash: konditsion son ta'siri."""
    _hints = [
        "Optimal lr = 2/(lambda_min + lambda_max). Katta konditsion son → sekin konvergentsiya.",
    ]
    _solution = (
        "vals = np.linalg.eigvalsh(A)\n"
        "lambda_min, lambda_max = vals[0], vals[-1]\n"
        "optimal_lr = 2 / (lambda_min + lambda_max)\n"
        "cond_number = lambda_max / lambda_min"
    )

    def _do_check(self, optimal_lr, A):
        vals = np.linalg.eigvalsh(A)
        lmin, lmax = vals[0], vals[-1]
        expected = 2 / (lmin + lmax)
        if not np.isclose(optimal_lr, expected, rtol=1e-4):
            return f"Optimal lr = 2/(lmin+lmax) = {expected:.6f}, siz {optimal_lr:.6f} berdingiz."
        return True


class C1_Q1(EqualityCheckProblem):
    """Chiziqli regressiya uchun gradient tushuv: grad L = (2/n) Xᵀ(Xw-y)."""
    _hints = [
        "L = ||Xw-y||² / n. grad = 2*X.T@(X@w - y) / n.",
    ]
    _solution = (
        "w = np.zeros(X.shape[1])\n"
        "for _ in range(n_iters):\n"
        "    residual = X @ w - y\n"
        "    grad = 2 * X.T @ residual / len(y)\n"
        "    w = w - lr * grad"
    )

    def _do_check(self, w, X, y, tol=1e-3):
        w_exact = np.linalg.lstsq(X, y, rcond=None)[0]
        if not np.allclose(w, w_exact, atol=tol):
            return f"Normal tenglama yechimi: {w_exact}"
        return True


class C2_Q1(ThoughtExperiment):
    """Gradient tushuvning chiziqli algebra bilan bog'liqligi va ML ahamiyati."""
    _hints = [
        "Gradient = yo'nalish. Hessian = egrilik. SGD = mini-batch gradient.",
    ]
    _solution = (
        "Gradient tushuv va chiziqli algebra:\n\n"
        "1) Gradientning geometrik ma'nosi:\n"
        "   - grad f(x) — f eng tez o'sgan yo'nalish.\n"
        "   - -grad f — eng tez kamayish yo'nalishi.\n\n"
        "2) Konvergentsiya tezligi:\n"
        "   - Konditsion son kappa = lambda_max/lambda_min.\n"
        "   - Har iteratsiyada xato * (1 - 2/(kappa+1)) ga kamayadi.\n"
        "   - Katta kappa → sekin konvergentsiya (ill-conditioned).\n\n"
        "3) ML da gradient tushuv:\n"
        "   - Chiziqli regressiya: exact solution bor, lekin GD umumiyroq.\n"
        "   - DL: million parametr → Newton imkonsiz → GD/SGD/Adam.\n"
        "   - Batch size: katta batch → past variance, lekin sekin.\n\n"
        "4) Preconditioning: P^{-1}*A → konditsion son kamaytirish.\n"
        "   Adam: o'z-o'zicha preconditioning qiladi."
    )
