"""Hints and solutions — Dars 4.3: Kichik Kvadratlar Yaqinlashuvi."""
import numpy as np
from learntools.core import EqualityCheckProblem, ThoughtExperiment


class Q1(EqualityCheckProblem):
    """Normal tenglamani (AᵀAx̂ = Aᵀb) yeching."""
    _hints = [
        "Normal tenglama: AᵀA x̂ = Aᵀb. np.linalg.solve(A.T@A, A.T@b) dan foydalaning.",
        "Yoki np.linalg.lstsq(A, b, rcond=None)[0] bevosita x̂ ni beradi.",
    ]
    _solution = "x_hat = np.linalg.solve(A.T @ A, A.T @ b)"

    def _do_check(self, x_hat, A, b):
        expected = np.linalg.lstsq(A, b, rcond=None)[0]
        if not np.allclose(x_hat, expected, atol=1e-6):
            return f"Kutilgan x̂ = {expected}, siz {x_hat} berdingiz."
        return True


class Q2(EqualityCheckProblem):
    """Qoldiq normasi ||b - Ax̂||² ni hisoblang."""
    _hints = [
        "Avval x̂ ni toping, keyin e = b - A @ x_hat, so'ng np.linalg.norm(e)**2.",
    ]
    _solution = "x_hat = np.linalg.lstsq(A, b, rcond=None)[0]; residual_sq = np.linalg.norm(b - A @ x_hat)**2"

    def _do_check(self, residual_sq, A, b):
        x_hat = np.linalg.lstsq(A, b, rcond=None)[0]
        expected = np.linalg.norm(b - A @ x_hat) ** 2
        if not np.isclose(residual_sq, expected, atol=1e-6):
            return f"Kutilgan ||e||² = {expected:.6f}, siz {residual_sq:.6f} berdingiz."
        return True


class Q3(EqualityCheckProblem):
    """Chiziqli regressiya uchun koeffitsiyentlarni toping: y = a + bx."""
    _hints = [
        "A = np.column_stack([np.ones(n), x]) dizayn matritsasi.",
        "x_hat = np.linalg.lstsq(A, y, rcond=None)[0] — [a, b] ni beradi.",
    ]
    _solution = "A = np.column_stack([np.ones_like(x), x]); a, b = np.linalg.lstsq(A, y, rcond=None)[0]"

    def _do_check(self, coeffs, x, y):
        A = np.column_stack([np.ones_like(x), x])
        expected = np.linalg.lstsq(A, y, rcond=None)[0]
        if not np.allclose(coeffs, expected, atol=1e-6):
            return f"Kutilgan [a, b] = {expected}, siz {coeffs} berdingiz."
        return True


class Q4(EqualityCheckProblem):
    """Kvadratik regressiya: y = a + bx + cx² uchun koeffitsiyentlarni toping."""
    _hints = [
        "A = np.column_stack([np.ones(n), x, x**2]) — 3 ustunli dizayn matritsasi.",
        "np.linalg.lstsq(A, y, rcond=None)[0] [a, b, c] ni beradi.",
    ]
    _solution = "A = np.column_stack([np.ones_like(x), x, x**2]); a, b, c = np.linalg.lstsq(A, y, rcond=None)[0]"

    def _do_check(self, coeffs, x, y):
        A = np.column_stack([np.ones_like(x), x, x ** 2])
        expected = np.linalg.lstsq(A, y, rcond=None)[0]
        if not np.allclose(coeffs, expected, atol=1e-6):
            return f"Kutilgan [a, b, c] = {expected}, siz {coeffs} berdingiz."
        return True


class Q5(EqualityCheckProblem):
    """R² (determinatsiya koeffitsiyenti) ni hisoblang."""
    _hints = [
        "R² = 1 - SS_res/SS_tot, bu erda SS_res = ||b - Ax̂||², SS_tot = ||b - mean(b)||².",
        "R² = 1 qadar yaqin bo'lsa yaxshiroq moslik.",
    ]
    _solution = (
        "x_hat = np.linalg.lstsq(A, b, rcond=None)[0]\n"
        "SS_res = np.sum((b - A @ x_hat)**2)\n"
        "SS_tot = np.sum((b - np.mean(b))**2)\n"
        "R2 = 1 - SS_res / SS_tot"
    )

    def _do_check(self, R2, A, b):
        x_hat = np.linalg.lstsq(A, b, rcond=None)[0]
        SS_res = np.sum((b - A @ x_hat) ** 2)
        SS_tot = np.sum((b - np.mean(b)) ** 2)
        expected = 1 - SS_res / SS_tot
        if not np.isclose(R2, expected, atol=1e-6):
            return f"Kutilgan R² = {expected:.6f}, siz {R2:.6f} berdingiz."
        return True


class Q6(EqualityCheckProblem):
    """Og'irlikli kichik kvadratlar: W og'irlik matritsasi bilan WA x̂ = Wb."""
    _hints = [
        "Og'irlikli normal tenglama: AᵀWAx̂ = AᵀWb.",
        "W diagonal matritsa: W = np.diag(weights). Keyin np.linalg.solve(A.T@W@A, A.T@W@b).",
    ]
    _solution = "W = np.diag(weights); x_hat = np.linalg.solve(A.T @ W @ A, A.T @ W @ b)"

    def _do_check(self, x_hat, A, b, weights):
        W = np.diag(weights)
        expected = np.linalg.solve(A.T @ W @ A, A.T @ W @ b)
        if not np.allclose(x_hat, expected, atol=1e-6):
            return f"Kutilgan og'irlikli x̂ = {expected}, siz {x_hat} berdingiz."
        return True


class C1_Q1(EqualityCheckProblem):
    """Polynomial regressiya va eng yaxshi daraja tanlash."""
    _hints = [
        "Har bir daraja uchun R² ni hisoblang va eng yuqori R² li darajani tanlang.",
        "for deg in range(1, 6): A = np.vstack([x**i for i in range(deg+1)]).T",
    ]
    _solution = (
        "best_deg, best_r2 = 1, -np.inf\n"
        "for deg in range(1, 6):\n"
        "    A = np.column_stack([x**i for i in range(deg+1)])\n"
        "    xh = np.linalg.lstsq(A, y, rcond=None)[0]\n"
        "    ss_res = np.sum((y - A@xh)**2)\n"
        "    ss_tot = np.sum((y - y.mean())**2)\n"
        "    r2 = 1 - ss_res/ss_tot\n"
        "    if r2 > best_r2: best_r2, best_deg = r2, deg"
    )

    def _do_check(self, best_deg, x, y):
        best, br2 = 1, -np.inf
        for deg in range(1, 6):
            A = np.column_stack([x ** i for i in range(deg + 1)])
            xh = np.linalg.lstsq(A, y, rcond=None)[0]
            ss_res = np.sum((y - A @ xh) ** 2)
            ss_tot = np.sum((y - y.mean()) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 1
            if r2 > br2:
                br2, best = r2, deg
        if best_deg != best:
            return f"Eng yaxshi daraja: {best}, siz {best_deg} berdingiz."
        return True


class C2_Q1(ThoughtExperiment):
    """Nima uchun kichik kvadratlar yechimi normal tenglama orqali topiladi?"""
    _hints = [
        "||b - Ax||² minimumga kelishi uchun gradientni nolga teng qiling.",
        "d/dx ||b-Ax||² = -2Aᵀ(b-Ax) = 0 → AᵀAx = Aᵀb.",
    ]
    _solution = (
        "f(x) = ||b - Ax||² = (b-Ax)ᵀ(b-Ax) ni minimizatsiya qilamiz. "
        "Gradientni hisoblaymiz: ∇f = -2Aᵀ(b - Ax). "
        "Minimumda ∇f = 0, ya'ni AᵀAx̂ = Aᵀb — normal tenglama. "
        "Geometrik ma'no: b - Ax̂ = e xato vektori A ning ustun fazosiga perpendikulyar, "
        "shuning uchun Aᵀe = 0, ya'ni Aᵀ(b - Ax̂) = 0."
    )
