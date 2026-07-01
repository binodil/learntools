"""Hints and solutions — Dars 10.4: Chiziqli Dasturlash."""
import numpy as np
from learntools.linear_algebra_uz.base import UzCheckProblem, ThoughtExperiment


class Q1(UzCheckProblem):
    """LP standart shaklga keltiring: min cᵀx, Ax=b, x≥0."""
    _hints = [
        "max cᵀx → min -cᵀx. Tengsizlik Ax≤b → Ax+s=b, s≥0 (slack o'zgaruvchi).",
    ]
    _solution = "c_std = -c  # max → min; A_std = A; b_std = b; slack added"

    def _do_check(self, c_std, c):
        if not np.allclose(c_std, -c):
            return "Maksimizatsiya → minimizatsiya: c_std = -c."
        return True


class Q2(UzCheckProblem):
    """scipy.optimize.linprog bilan LP yeching."""
    _hints = [
        "from scipy.optimize import linprog; res = linprog(c, A_ub=A, b_ub=b, bounds=bounds).",
    ]
    _solution = (
        "from scipy.optimize import linprog\n"
        "res = linprog(c, A_ub=A, b_ub=b, bounds=[(0,None)]*len(c))\n"
        "opt_val = res.fun; opt_x = res.x"
    )

    def _do_check(self, opt_val, c, A, b):
        from scipy.optimize import linprog
        res = linprog(c, A_ub=A, b_ub=b, bounds=[(0, None)] * len(c))
        if not np.isclose(opt_val, res.fun, rtol=1e-4):
            return f"Optimal qiymat: {res.fun:.4f}, siz {opt_val:.4f} berdingiz."
        return True


class Q3(UzCheckProblem):
    """Ikkita o'zgaruvchili LP ni grafik usulda yeching."""
    _hints = [
        "Burchak nuqtalarini toping: chiziqlar kesishishi va o'q bilan kesishish.",
        "Maqsad funksiyani burchak nuqtalarda hisoblang va minimumni tanlang.",
    ]
    _solution = (
        "# min 2x + 3y s.t. x+y>=4, x>=1, y>=1\n"
        "vertices = [(1,3),(3,1),(1,1) ... ]\n"
        "opt_vertex = min(vertices, key=lambda v: 2*v[0]+3*v[1])"
    )

    def _do_check(self, opt_x, opt_y, c1, c2):
        from scipy.optimize import linprog
        res = linprog([c1, c2], A_ub=[[-1,-1],[-1,0],[0,-1]], b_ub=[-4,-1,-1],
                      bounds=[(0,None),(0,None)])
        exp_x, exp_y = res.x
        if not (np.isclose(opt_x, exp_x, rtol=0.01) and np.isclose(opt_y, exp_y, rtol=0.01)):
            return f"Optimal: x={exp_x:.4f}, y={exp_y:.4f}. Siz ({opt_x:.4f},{opt_y:.4f}) berdingiz."
        return True


class Q4(UzCheckProblem):
    """Transport masalasi (minimal narx bilan yuk tashish)."""
    _hints = [
        "Transport masalasi: manba s, maqsad d, narx C. linprog bilan yechish.",
        "min sum(C[i,j]*x[i,j]) s.t. qatorlar yig'indisi=s, ustunlar yig'indisi=d.",
    ]
    _solution = (
        "from scipy.optimize import linprog\n"
        "# C[i,j] — narx, s — taklif, d — talab\n"
        "c_flat = C.flatten()\n"
        "# cheklovlar: linprog bilan yechiladi"
    )

    def _do_check(self, total_cost, C, supply, demand):
        from scipy.optimize import linprog
        m, n = C.shape
        c = C.flatten()
        A_eq_rows = []
        b_eq = []
        for i in range(m):
            row = np.zeros(m * n)
            row[i * n:(i + 1) * n] = 1
            A_eq_rows.append(row)
            b_eq.append(supply[i])
        for j in range(n):
            row = np.zeros(m * n)
            for i in range(m):
                row[i * n + j] = 1
            A_eq_rows.append(row)
            b_eq.append(demand[j])
        res = linprog(c, A_eq=A_eq_rows, b_eq=b_eq, bounds=[(0, None)] * (m * n))
        if not np.isclose(total_cost, res.fun, rtol=1e-3):
            return f"Minimal narx: {res.fun:.4f}, siz {total_cost:.4f} berdingiz."
        return True


class Q5(UzCheckProblem):
    """LP dual masalasini yozing."""
    _hints = [
        "Primal: min cᵀx, Ax≥b, x≥0. Dual: max bᵀy, Aᵀy≤c, y≥0.",
    ]
    _solution = "# Dual: max b.T @ y, A.T @ y <= c, y >= 0"

    def _do_check(self, dual_val, c, A, b):
        from scipy.optimize import linprog
        res_primal = linprog(c, A_ub=-A, b_ub=-b, bounds=[(0,None)]*len(c))
        res_dual = linprog(-b, A_ub=A.T, b_ub=c, bounds=[(0,None)]*len(b))
        if not np.isclose(dual_val, -res_dual.fun, rtol=1e-3):
            return f"Dual optimal: {-res_dual.fun:.4f}, siz {dual_val:.4f} berdingiz. Kuchli duallik: primal = dual."
        return True


class Q6(UzCheckProblem):
    """Kuchli duallik teoremasi: primal = dual optimal qiymat."""
    _hints = [
        "Kuchli duallik: agar primal va dual feasible bo'lsa, optimal qiymatlar teng.",
    ]
    _solution = (
        "from scipy.optimize import linprog\n"
        "res_p = linprog(c, A_ub=-A, b_ub=-b, bounds=[(0,None)]*len(c))\n"
        "res_d = linprog(-b, A_ub=A.T, b_ub=c, bounds=[(0,None)]*len(b))\n"
        "strong_duality = np.isclose(res_p.fun, -res_d.fun, rtol=1e-4)"
    )

    def _do_check(self, strong_duality, c, A, b):
        from scipy.optimize import linprog
        rp = linprog(c, A_ub=-A, b_ub=-b, bounds=[(0,None)]*len(c))
        rd = linprog(-b, A_ub=A.T, b_ub=c, bounds=[(0,None)]*len(b))
        expected = np.isclose(rp.fun, -rd.fun, rtol=1e-3)
        if strong_duality != expected:
            return f"Primal = {rp.fun:.4f}, Dual = {-rd.fun:.4f}. Teng? {expected}"
        return True


class C1_Q1(UzCheckProblem):
    """Dieta masalasi: minimal xarajat bilan ozuqaviy talablarni qondiring."""
    _hints = [
        "min cᵀx (narxlar) s.t. Ax >= b (ozuqa talablari), x >= 0.",
    ]
    _solution = (
        "from scipy.optimize import linprog\n"
        "# c: narxlar, A: ozuqa matritsa, b: minimal talablar\n"
        "res = linprog(c, A_ub=-A, b_ub=-b, bounds=[(0,None)]*len(c))\n"
        "min_cost = res.fun; amounts = res.x"
    )

    def _do_check(self, min_cost, c, A, b):
        from scipy.optimize import linprog
        res = linprog(c, A_ub=-A, b_ub=-b, bounds=[(0, None)] * len(c))
        if not np.isclose(min_cost, res.fun, rtol=1e-3):
            return f"Minimal xarajat: {res.fun:.4f}, siz {min_cost:.4f} berdingiz."
        return True


class C2_Q1(ThoughtExperiment):
    """Simpleks usuli va ichki nuqta usuli: farqlari va afzalliklari."""
    _hints = [
        "Simpleks: burchak nuqtalar bo'ylab yuradi. Ichki nuqta: feasible to'plam ichida.",
    ]
    _solution = (
        "Chiziqli dasturlash algoritmlari:\n\n"
        "1) Simpleks usuli (Dantzig, 1947):\n"
        "   - Feasible to'plamning burchak nuqtalarini ko'rib chiqadi.\n"
        "   - O'rtacha holat: polinomial; eng yomon holat: eksponential.\n"
        "   - Amalda juda tez, ko'p muammolarda optimal.\n\n"
        "2) Ichki nuqta usuli (Karmarkar, 1984):\n"
        "   - Feasible to'plam ichida yo'nalish bilan harakatlanadi.\n"
        "   - Polinomial vaqt kafolatlangan: O(n^3.5 L).\n"
        "   - Katta masalalar uchun afzal.\n\n"
        "Chiziqli algebra roli: LP = bir nechta yarim-fazo kesishmasi (qirra nuqtalar = bazaviy yechimlar)."
    )
