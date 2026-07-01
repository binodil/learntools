"""Hints and solutions — Dars 11.3: Iterativ Usullar va Prekonditsionirovanie."""
import numpy as np
from learntools.linear_algebra_uz.base import UzCheckProblem, ThoughtExperiment


class Q1(UzCheckProblem):
    """Yakobi iteratsiyasi: x^{k+1} = D⁻¹(b - (L+U)x^k)."""
    _hints = [
        "D = diag(A). x_new = (b - (A - D) @ x) / np.diag(A).",
    ]
    _solution = (
        "D = np.diag(np.diag(A))\n"
        "x = np.zeros(len(b))\n"
        "for _ in range(n_iter):\n"
        "    x = (b - (A - D) @ x) / np.diag(A)"
    )

    def _do_check(self, x, A, b, n_iter):
        expected = np.linalg.solve(A, b)
        res = np.linalg.norm(A @ x - b) / np.linalg.norm(b)
        if res > 0.1:
            return f"Nisbiy qoldiq: {res:.4e}. Ko'proq iteratsiya kerak bo'lishi mumkin. Yechim: {expected}"
        return True


class Q2(UzCheckProblem):
    """Gauss-Zeidel iteratsiyasi (oldinga va orqaga substitusiya birlashtirib)."""
    _hints = [
        "x[i] = (b[i] - sum_{j<i} A[i,j]*x[j] - sum_{j>i} A[i,j]*x_old[j]) / A[i,i].",
    ]
    _solution = (
        "x = np.zeros(len(b))\n"
        "for _ in range(n_iter):\n"
        "    for i in range(len(b)):\n"
        "        x[i] = (b[i] - A[i,:i]@x[:i] - A[i,i+1:]@x[i+1:]) / A[i,i]"
    )

    def _do_check(self, x, A, b, n_iter):
        expected = np.linalg.solve(A, b)
        res = np.linalg.norm(A @ x - b) / np.linalg.norm(b)
        if res > 0.1:
            return f"Gauss-Zeidel qoldig'i: {res:.4e}. Kutilgan: {expected}"
        return True


class Q3(UzCheckProblem):
    """Gradiyent tushish (gradient descent) Ax=b uchun."""
    _hints = [
        "r = b - A@x (qoldiq). x = x + alpha*r. alpha = rᵀr/(rᵀAr).",
        "Optimal qadam: alpha_opt = (r.T @ r) / (r.T @ A @ r).",
    ]
    _solution = (
        "x = np.zeros(len(b))\n"
        "for _ in range(n_iter):\n"
        "    r = b - A @ x\n"
        "    alpha = r @ r / (r @ A @ r)\n"
        "    x = x + alpha * r"
    )

    def _do_check(self, x, A, b, n_iter):
        expected = np.linalg.solve(A, b)
        res = np.linalg.norm(A @ x - b) / np.linalg.norm(b)
        if res > 0.5:
            return f"Qoldiq: {res:.4e}. Optimal qadam alpha = rᵀr/(rᵀAr) ni ishlating."
        return True


class Q4(UzCheckProblem):
    """Konjugat gradiyent (CG) usuli — A simmetrik musbat aniq uchun."""
    _hints = [
        "scipy.sparse.linalg.cg(A, b) — CG yechuvchisi.",
    ]
    _solution = "from scipy.sparse.linalg import cg; x, info = cg(A, b)"

    def _do_check(self, x, A, b):
        from scipy.sparse.linalg import cg
        exp_x, info = cg(A, b)
        if not np.allclose(x, exp_x, rtol=1e-4):
            return "scipy.sparse.linalg.cg(A, b) dan foydalaning."
        return True


class Q5(UzCheckProblem):
    """GMRES — nosimmetrik sistemalar uchun."""
    _hints = [
        "scipy.sparse.linalg.gmres(A, b) — umumiy sistemalar uchun Krylov usuli.",
    ]
    _solution = "from scipy.sparse.linalg import gmres; x, info = gmres(A, b)"

    def _do_check(self, x, A, b):
        from scipy.sparse.linalg import gmres
        exp_x, info = gmres(A, b)
        res = np.linalg.norm(A @ x - b) / np.linalg.norm(b)
        if res > 1e-4:
            return f"GMRES qoldig'i: {res:.4e}. scipy.sparse.linalg.gmres(A, b) ni ishlating."
        return True


class Q6(UzCheckProblem):
    """Diagonal prekonditsionirovanie: M⁻¹Ax = M⁻¹b, M = diag(A)."""
    _hints = [
        "M = diag(A). M_inv = 1/diag(A). A_prec = M_inv * A (satr bo'ylab).",
        "scipy.sparse.linalg.cg(A, b, M=M_inv_op).",
    ]
    _solution = (
        "from scipy.sparse.linalg import cg, LinearOperator\n"
        "M_inv = 1.0 / np.diag(A)\n"
        "M_op = LinearOperator((len(b),len(b)), matvec=lambda x: M_inv*x)\n"
        "x, info = cg(A, b, M=M_op)"
    )

    def _do_check(self, x, A, b):
        expected = np.linalg.solve(A, b)
        res = np.linalg.norm(A @ x - b) / np.linalg.norm(b)
        if res > 1e-4:
            return f"Qoldiq: {res:.4e}. Prekonditsionirovanie bilan CG ni ishlating."
        return True


class C1_Q1(UzCheckProblem):
    """Konvergentsiya tezligini solishtiring: Yakobi vs Gauss-Zeidel vs CG."""
    _hints = [
        "Har bir iteratsiyadan keyin qoldiq ||Ax-b|| ni saqlang va solishtiring.",
    ]
    _solution = (
        "residuals_j, residuals_gs, residuals_cg = [], [], []\n"
        "# Yakobi\n"
        "x = np.zeros(n); D = np.diag(np.diag(A))\n"
        "for _ in range(50):\n"
        "    x = (b - (A-D)@x)/np.diag(A)\n"
        "    residuals_j.append(np.linalg.norm(A@x-b))\n"
        "# va h.k."
    )

    def _do_check(self, n_iter_cg, n_iter_jacobi, tol):
        if n_iter_cg >= n_iter_jacobi:
            return f"CG ({n_iter_cg} iter) Yakobi ({n_iter_jacobi} iter) dan tez bo'lishi kerak."
        return True


class C2_Q1(ThoughtExperiment):
    """Iterativ usullar qachon to'g'ridan-to'g'ri usullardan afzal?"""
    _hints = [
        "Katta siyrak sistemalar: n=10⁶, LU O(n³) — imkonsiz.",
        "Iterativ: O(n * k), k — iteratsiyalar soni.",
    ]
    _solution = (
        "Iterativ usullarning afzalliklari:\n\n"
        "1) Katta siyrak sistemalar (n > 10⁴):\n"
        "   LU: O(n³) vaqt, O(n²) xotira — imkonsiz.\n"
        "   CG/GMRES: O(n * k), k — iteratsiyalar soni (odatda k << n).\n\n"
        "2) Matritsa-vektor ko'paytmasi etarli:\n"
        "   Matritsani saqlash shart emas — faqat A@v amal kerak.\n\n"
        "3) Prekonditsionirovanie: M⁻¹A konditsion sonini kamaytiradi.\n"
        "   Yaxshi M → tez konvergentsiya.\n\n"
        "To'g'ridan-to'g'ri afzal:\n"
        "- Kichik/o'rta o'lcham (n < 10³).\n"
        "- Bir xil A bilan ko'p b — LU bir marta, keyin tez yechish.\n"
        "- Zich matritsa (sparse emas)."
    )
