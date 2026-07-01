"""Hints and solutions — Dars 13.3: Chiziqli Dasturlash, O'yin Nazariyasi va Duallik."""
import numpy as np
from learntools.linear_algebra_uz.base import UzCheckProblem, ThoughtExperiment


class Q1(UzCheckProblem):
    """LP: max cᵀx s.t. Ax <= b, x >= 0 (scipy.optimize.linprog)."""
    _hints = [
        "linprog minimizatsiya qiladi: min -cᵀx. A_ub=A, b_ub=b, bounds=(0,None).",
    ]
    _solution = (
        "from scipy.optimize import linprog\n"
        "res = linprog(-c, A_ub=A, b_ub=b, bounds=[(0,None)]*len(c))\n"
        "x_opt = res.x; obj_val = -res.fun"
    )

    def _do_check(self, x_opt, c, A, b):
        from scipy.optimize import linprog
        res = linprog(-c, A_ub=A, b_ub=b, bounds=[(0, None)] * len(c))
        if not np.allclose(x_opt, res.x, atol=1e-4):
            return f"Optimal x: {res.x}, siz {x_opt} berdingiz."
        if not np.all(A @ x_opt <= b + 1e-4):
            return "Cheklovlar buzilgan: Ax <= b bo'lishi kerak."
        return True


class Q2(UzCheckProblem):
    """LP duallik: primal max cᵀx → dual min bᵀy s.t. Aᵀy >= c, y >= 0."""
    _hints = [
        "Dual: min bᵀy s.t. Aᵀy >= c, y >= 0. linprog(-b, A_ub=-A.T, b_ub=-c).",
    ]
    _solution = (
        "from scipy.optimize import linprog\n"
        "res_dual = linprog(b, A_ub=-A.T, b_ub=-c, bounds=[(0,None)]*len(b))\n"
        "y_opt = res_dual.x; dual_val = res_dual.fun"
    )

    def _do_check(self, dual_val, primal_val):
        if not np.isclose(dual_val, primal_val, rtol=1e-3):
            return (f"Kuchli duallik: primal = dual = {primal_val:.4f}, "
                    f"siz dual={dual_val:.4f} berdingiz.")
        return True


class Q3(UzCheckProblem):
    """Nol-summa o'yini: Nash muvozanati (minimax strategiya)."""
    _hints = [
        "Minimax: A o'yin matritsasi. LP: max v s.t. Aᵀp >= v*1, p >= 0, sum(p)=1.",
    ]
    _solution = (
        "from scipy.optimize import linprog\n"
        "m, n = A.shape\n"
        "# Aralash strategiya: min -v; Aᵀp + v >= 0; sum p = 1; p >= 0\n"
        "c_lp = np.zeros(m+1); c_lp[-1] = -1\n"
        "A_ub = np.hstack([-A.T, np.ones((n,1))])\n"
        "b_ub = np.zeros(n)\n"
        "A_eq = np.ones((1,m+1)); A_eq[0,-1] = 0; b_eq = [1]\n"
        "res = linprog(c_lp, A_ub, b_ub, A_eq, b_eq,\n"
        "              bounds=[(0,None)]*m + [(None,None)])\n"
        "p = res.x[:-1]; v = res.x[-1]"
    )

    def _do_check(self, p, A):
        if not np.isclose(p.sum(), 1, atol=1e-3):
            return f"Strategiya ehtimolliklari yig'indisi 1 bo'lishi kerak: {p.sum():.4f}"
        if not np.all(p >= -1e-4):
            return "Strategiya ehtimolliklari >= 0 bo'lishi kerak."
        return True


class Q4(UzCheckProblem):
    """Simpleks usuli: burchak nuqtalarda optimal yechim."""
    _hints = [
        "LP optimumi politopning burchak nuqtasida (vertex). Vertex: faol cheklovlar.",
        "Brute force: barcha vertex kombinatsiyalarini sinab ko'ring.",
    ]
    _solution = (
        "from itertools import combinations\n"
        "m, n = A.shape  # m cheklov, n o'zgaruvchi\n"
        "best_val = -np.inf; best_x = None\n"
        "for cols in combinations(range(m), n):\n"
        "    A_sq = A[list(cols)]; b_sq = b[list(cols)]\n"
        "    try:\n"
        "        x = np.linalg.solve(A_sq, b_sq)\n"
        "        if np.all(x >= -1e-8) and np.all(A@x <= b+1e-8):\n"
        "            if c@x > best_val: best_val = c@x; best_x = x\n"
        "    except: pass"
    )

    def _do_check(self, x_opt, c, A, b):
        from scipy.optimize import linprog
        res = linprog(-c, A_ub=A, b_ub=b, bounds=[(0, None)] * len(c))
        if not np.isclose(c @ x_opt, -res.fun, atol=1e-3):
            return f"Optimal maqsad: {-res.fun:.4f}, siz {c@x_opt:.4f} berdingiz."
        return True


class Q5(UzCheckProblem):
    """Transportatsiya masalasi: LP sifatida formulatsiya."""
    _hints = [
        "Xij: i manbadan j talabgorga miqdor. min sum cij*xij, s.t. supply/demand.",
    ]
    _solution = (
        "from scipy.optimize import linprog\n"
        "m, n = len(supply), len(demand)\n"
        "c_flat = cost.flatten()\n"
        "# Supply: sum_j x_ij <= supply[i] (qator yig'indilari)\n"
        "# Demand: sum_i x_ij >= demand[j] (ustun yig'indilari)\n"
        "A_supply = np.kron(np.eye(m), np.ones((1,n)))\n"
        "A_demand = -np.kron(np.ones((1,m)), np.eye(n))\n"
        "A_ub = np.vstack([A_supply, A_demand])\n"
        "b_ub = np.concatenate([supply, -demand])\n"
        "res = linprog(c_flat, A_ub, b_ub, bounds=[(0,None)]*(m*n))\n"
        "X = res.x.reshape(m, n)"
    )

    def _do_check(self, total_cost, supply, demand, cost):
        from scipy.optimize import linprog
        m, n = len(supply), len(demand)
        c_flat = cost.flatten()
        A_supply = np.kron(np.eye(m), np.ones((1, n)))
        A_demand = -np.kron(np.ones((1, m)), np.eye(n))
        A_ub = np.vstack([A_supply, A_demand])
        b_ub = np.concatenate([supply, -demand])
        res = linprog(c_flat, A_ub, b_ub, bounds=[(0, None)] * (m * n))
        if not np.isclose(total_cost, res.fun, rtol=1e-3):
            return f"Minimal transport xarajati: {res.fun:.4f}, siz {total_cost:.4f} berdingiz."
        return True


class Q6(UzCheckProblem):
    """Kuchli duallik teoremasi: primal optimal = dual optimal."""
    _hints = [
        "Kuchli duallik (Strong Duality): LP uchun har doim primal = dual optimal.",
    ]
    _solution = (
        "# Primal: max cᵀx s.t. Ax<=b, x>=0\n"
        "# Dual: min bᵀy s.t. Aᵀy>=c, y>=0\n"
        "# Kuchli duallik: cᵀx* = bᵀy*\n"
        "gap = abs(primal_val - dual_val)\n"
        "strong_duality = gap < 1e-4"
    )

    def _do_check(self, gap, primal_val, dual_val):
        expected = abs(primal_val - dual_val)
        if not np.isclose(gap, expected, atol=1e-6):
            return f"Gap = |primal - dual| = {expected:.2e}"
        if gap > 1e-3:
            return f"Kuchli duallik: gap ≈ 0, lekin {gap:.2e}. Primal/dual sozlash."
        return True


class C1_Q1(UzCheckProblem):
    """Resurs taqsimlash masalasi: ishlab chiqarish LP."""
    _hints = [
        "max profit s.t. resource constraints. linprog(-profit, A_resources, b_capacity).",
    ]
    _solution = (
        "from scipy.optimize import linprog\n"
        "res = linprog(-profit_per_unit,\n"
        "              A_ub=resource_matrix,\n"
        "              b_ub=capacity,\n"
        "              bounds=[(0,None)]*len(profit_per_unit))\n"
        "x_opt = res.x"
    )

    def _do_check(self, x_opt, profit_per_unit, resource_matrix, capacity):
        from scipy.optimize import linprog
        res = linprog(-profit_per_unit, A_ub=resource_matrix, b_ub=capacity,
                      bounds=[(0, None)] * len(profit_per_unit))
        if not np.isclose(profit_per_unit @ x_opt, -res.fun, rtol=1e-3):
            return f"Maksimal foyda: {-res.fun:.2f}, siz {profit_per_unit@x_opt:.2f} berdingiz."
        return True


class C2_Q1(ThoughtExperiment):
    """LP, duallik va o'yin nazariyasi chiziqli algebrada nima uchun muhim?"""
    _hints = [
        "Duallik = shadow prices. Nash = minimax. LP solvers = industrial optimization.",
    ]
    _solution = (
        "Chiziqli dasturlash va duallik ahamiyati:\n\n"
        "1) LP hamma joyda:\n"
        "   - Ishlab chiqarish rejalashtirish, logistika, portfolio.\n"
        "   - ML: SVM dual LP, L1 regularizatsiya = LP.\n\n"
        "2) Duallik teoremasi:\n"
        "   - Primal optimal = Dual optimal (kuchli duallik).\n"
        "   - Shadow price: b_i birlik oshishi optimal qiymatga ta'siri = y_i*.\n"
        "   - Komplementarlik: x_i* > 0 → i-cheklov faol.\n\n"
        "3) O'yin nazariyasi:\n"
        "   - Nash muvozanati = minimax strategiya.\n"
        "   - von Neumann minimax teoremasi: LP dualligiga ekvivalent.\n\n"
        "4) SVM bilan bog'liqligi:\n"
        "   - SVM primal: min ||w||². Dual: max alpha lagrange masalasi.\n"
        "   - Kernel trick faqat dual formulatsiyada ishlaydi."
    )
