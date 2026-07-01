"""Hints and solutions — Dars 13.5: SGD va Adam Optimallashtiruvchilari."""
import numpy as np
from learntools.core import EqualityCheckProblem, ThoughtExperiment


class Q1(EqualityCheckProblem):
    """Mini-batch SGD: tasodifiy namuna olib gradient hisoblash."""
    _hints = [
        "idx = np.random.choice(n, batch_size, replace=False).",
        "grad_batch = (2/batch_size) * X[idx].T @ (X[idx]@w - y[idx]).",
    ]
    _solution = (
        "idx = np.random.choice(len(y), batch_size, replace=False)\n"
        "X_b, y_b = X[idx], y[idx]\n"
        "grad = 2 * X_b.T @ (X_b @ w - y_b) / batch_size\n"
        "w = w - lr * grad"
    )

    def _do_check(self, w, X, y, tol=0.1):
        w_exact = np.linalg.lstsq(X, y, rcond=None)[0]
        if not np.allclose(w, w_exact, atol=tol):
            return f"SGD yechimi normal tenglama yechimiga yaqin bo'lishi kerak: {w_exact}"
        return True


class Q2(EqualityCheckProblem):
    """Adam optimallashtiruvchi: m, v moment yangilash va parametr qadam."""
    _hints = [
        "m = beta1*m + (1-beta1)*grad. v = beta2*v + (1-beta2)*grad**2.",
        "m_hat = m/(1-beta1**t). v_hat = v/(1-beta2**t). x -= lr*m_hat/(sqrt(v_hat)+eps).",
    ]
    _solution = (
        "m = np.zeros_like(x); v = np.zeros_like(x)\n"
        "beta1, beta2, eps = 0.9, 0.999, 1e-8\n"
        "for t in range(1, n_iters+1):\n"
        "    grad = gradient_fn(x)\n"
        "    m = beta1*m + (1-beta1)*grad\n"
        "    v = beta2*v + (1-beta2)*grad**2\n"
        "    m_hat = m / (1-beta1**t)\n"
        "    v_hat = v / (1-beta2**t)\n"
        "    x = x - lr * m_hat / (np.sqrt(v_hat) + eps)"
    )

    def _do_check(self, x, gradient_fn, x_true, tol=1e-3):
        if not np.allclose(x, x_true, atol=tol):
            return f"Adam minimumi: {x_true}, siz {x} berdingiz."
        return True


class Q3(EqualityCheckProblem):
    """RMSprop: eksponentsial o'rtacha kvadrat gradient."""
    _hints = [
        "v = gamma*v + (1-gamma)*grad**2. x = x - lr/sqrt(v+eps)*grad.",
    ]
    _solution = (
        "v = np.zeros_like(x); gamma, eps = 0.9, 1e-8\n"
        "for _ in range(n_iters):\n"
        "    grad = gradient_fn(x)\n"
        "    v = gamma*v + (1-gamma)*grad**2\n"
        "    x = x - lr / np.sqrt(v + eps) * grad"
    )

    def _do_check(self, x, gradient_fn, x_true, tol=1e-2):
        if not np.allclose(x, x_true, atol=tol):
            return f"RMSprop minimumi: {x_true}"
        return True


class Q4(EqualityCheckProblem):
    """O'rganish tezligi jadval: cosine annealing."""
    _hints = [
        "lr_t = lr_min + 0.5*(lr_max - lr_min)*(1 + cos(pi*t/T)).",
    ]
    _solution = (
        "lr_schedule = [lr_min + 0.5*(lr_max-lr_min)*(1+np.cos(np.pi*t/T))\n"
        "               for t in range(T)]"
    )

    def _do_check(self, lr_schedule, lr_min, lr_max, T):
        expected = [lr_min + 0.5 * (lr_max - lr_min) * (1 + np.cos(np.pi * t / T))
                    for t in range(T)]
        if not np.allclose(lr_schedule, expected, rtol=1e-6):
            return "lr_t = lr_min + 0.5*(lr_max-lr_min)*(1+cos(pi*t/T))."
        return True


class Q5(EqualityCheckProblem):
    """SGD bilan neural network parametrlarini yangilash."""
    _hints = [
        "Har bir parametr uchun: param -= lr * grad. Gradient autograd dan keladi.",
    ]
    _solution = (
        "for epoch in range(n_epochs):\n"
        "    for X_b, y_b in mini_batches(X, y, batch_size):\n"
        "        # Forward pass\n"
        "        y_pred = X_b @ W + b\n"
        "        loss = np.mean((y_pred - y_b)**2)\n"
        "        # Backward pass (manual)\n"
        "        dL_dy = 2*(y_pred - y_b)/len(y_b)\n"
        "        W -= lr * X_b.T @ dL_dy\n"
        "        b -= lr * dL_dy.sum()"
    )

    def _do_check(self, W, b, X, y, tol=0.1):
        W_exact = np.linalg.lstsq(np.column_stack([X, np.ones(len(X))]),
                                   y, rcond=None)[0]
        if not np.allclose(W, W_exact[:-1], atol=tol):
            return "W normal tenglama yechimiga yaqin bo'lishi kerak."
        return True


class Q6(EqualityCheckProblem):
    """SGD va GD konvergentsiyasini solishtiring: loss egri chiziqlari."""
    _hints = [
        "SGD: har mini-batch uchun gradient. GD: to'liq dataset gradient.",
        "SGD — shovqinli lekin tez; GD — silliq lekin sekin.",
    ]
    _solution = (
        "# GD\n"
        "losses_gd = []\n"
        "w_gd = np.zeros(X.shape[1])\n"
        "for _ in range(n_iters):\n"
        "    grad = 2*X.T@(X@w_gd-y)/len(y)\n"
        "    w_gd -= lr*grad\n"
        "    losses_gd.append(np.mean((X@w_gd-y)**2))\n"
        "# SGD\n"
        "losses_sgd = []\n"
        "w_sgd = np.zeros(X.shape[1])\n"
        "for _ in range(n_iters):\n"
        "    idx = np.random.choice(len(y), batch_size)\n"
        "    grad = 2*X[idx].T@(X[idx]@w_sgd-y[idx])/batch_size\n"
        "    w_sgd -= lr*grad\n"
        "    losses_sgd.append(np.mean((X@w_sgd-y)**2))"
    )

    def _do_check(self, losses_gd, losses_sgd):
        if len(losses_gd) < 2 or len(losses_sgd) < 2:
            return "Kamida 2 iteratsiya kerak."
        if losses_gd[-1] > losses_gd[0]:
            return "GD loss kamayishi kerak."
        return True


class C1_Q1(EqualityCheckProblem):
    """Adam vs SGD vs RMSprop: Rosenbrock funksiyasida solishtirish."""
    _hints = [
        "Rosenbrock: f(x,y) = (1-x)^2 + 100*(y-x^2)^2. Minimum: (1,1).",
    ]
    _solution = (
        "def rosenbrock_grad(xy):\n"
        "    x, y = xy\n"
        "    gx = -2*(1-x) - 400*x*(y-x**2)\n"
        "    gy = 200*(y-x**2)\n"
        "    return np.array([gx, gy])\n\n"
        "# Adam\n"
        "xy = np.array([-1.0, 1.0]); m = v = np.zeros(2)\n"
        "for t in range(1, 1001):\n"
        "    g = rosenbrock_grad(xy)\n"
        "    m = 0.9*m + 0.1*g; v = 0.999*v + 0.001*g**2\n"
        "    xy -= 0.01 * (m/(1-0.9**t)) / (np.sqrt(v/(1-0.999**t)) + 1e-8)"
    )

    def _do_check(self, xy_final, tol=0.1):
        target = np.array([1.0, 1.0])
        if not np.allclose(xy_final, target, atol=tol):
            return f"Rosenbrock minimumi (1,1), siz {xy_final} berdingiz."
        return True


class C2_Q1(ThoughtExperiment):
    """SGD, Adam va moment usullari nima uchun DL da asosiy vosita?"""
    _hints = [
        "Adam = momentum + adaptiv lr. Mini-batch = stochastic = umumlashtirish.",
    ]
    _solution = (
        "SGD va Adam ahamiyati DL da:\n\n"
        "1) Nima uchun GD emas, SGD?\n"
        "   - n=10^8 namuna: to'liq gradient O(n) → juda sekin.\n"
        "   - Mini-batch: batch_size=256 → tez, GPU ga mos.\n"
        "   - Stochastik shovqin: local minimum dan chiqishga yordam.\n\n"
        "2) Moment afzalliklari:\n"
        "   - Tez yo'nalishlarda tezlashadi, sekin yo'nalishlarda sekinlashadi.\n"
        "   - Zichlik matritsasi Hessian simulyatsiyasi.\n\n"
        "3) Adam = Momentum + RMSprop:\n"
        "   - m: gradient 1-moment (yo'nalish).\n"
        "   - v: gradient 2-moment (o'lcham).\n"
        "   - Bias correction: t kichik bo'lganida zaruriy.\n\n"
        "4) Chiziqli algebra bilan bog'liqligi:\n"
        "   - Adam ~ diagonal Hessian preconditioning.\n"
        "   - L-BFGS: to'liq Hessian, katta MLda ishlatilmaydi.\n"
        "   - Gradient = Jacobian transpozitsiyasi (backprop orqali)."
    )
