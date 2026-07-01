"""Hints and solutions — Dars 10.6: Kompyuter Grafika."""
import numpy as np
from learntools.core import EqualityCheckProblem, ThoughtExperiment


class Q1(EqualityCheckProblem):
    """2D aylantirish matritsasi bilan nuqtalar to'plamini aylantiring."""
    _hints = [
        "R = [[cos θ, -sin θ], [sin θ, cos θ]]. P_new = R @ P.",
        "Har bir ustun — bir nuqta: P shaklida (2, N).",
    ]
    _solution = "R = np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]]); P_rot = R @ P"

    def _do_check(self, P_rot, P, theta):
        R = np.array([[np.cos(theta), -np.sin(theta)],
                      [np.sin(theta), np.cos(theta)]])
        expected = R @ P
        if not np.allclose(P_rot, expected, atol=1e-8):
            return "P_rot = R @ P formulasini tekshiring."
        return True


class Q2(EqualityCheckProblem):
    """2D masshtablash matritsasi bilan kengaytiring."""
    _hints = ["S = diag(sx, sy). P_scaled = S @ P."]
    _solution = "S = np.diag([sx, sy]); P_scaled = S @ P"

    def _do_check(self, P_scaled, P, sx, sy):
        S = np.diag([sx, sy])
        expected = S @ P
        if not np.allclose(P_scaled, expected, atol=1e-8):
            return "S = diag([sx, sy]); P_scaled = S @ P."
        return True


class Q3(EqualityCheckProblem):
    """Gomogen koordinatalarda ko'chirish (translation) matritsasi."""
    _hints = [
        "Gomogen: [x, y, 1]. T = [[1,0,tx],[0,1,ty],[0,0,1]]. P_hom = T @ [x,y,1].",
    ]
    _solution = (
        "T = np.array([[1,0,tx],[0,1,ty],[0,0,1]])\n"
        "P_hom = np.vstack([P, np.ones(P.shape[1])])\n"
        "P_new = (T @ P_hom)[:2]"
    )

    def _do_check(self, P_new, P, tx, ty):
        T = np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]])
        P_hom = np.vstack([P, np.ones(P.shape[1])])
        expected = (T @ P_hom)[:2]
        if not np.allclose(P_new, expected, atol=1e-8):
            return "T = [[1,0,tx],[0,1,ty],[0,0,1]]; P_hom = [P; 1]; P_new = (T@P_hom)[:2]."
        return True


class Q4(EqualityCheckProblem):
    """Uch transformatsiyani birlashtirib umumiy matritsani toping."""
    _hints = [
        "Kompozitsiya: M = T @ R @ S (avval masshtab, keyin aylantir, keyin ko'chir).",
    ]
    _solution = "M = T @ R @ S  # kompozitsiya"

    def _do_check(self, M, T, R, S):
        expected = T @ R @ S
        if not np.allclose(M, expected, atol=1e-8):
            return "M = T @ R @ S (tartibi muhim)."
        return True


class Q5(EqualityCheckProblem):
    """3D proyeksiya (perspective): P_proj = P @ M_proj."""
    _hints = [
        "Oddiy perspektiv proyeksiya: x' = x/z, y' = y/z.",
        "z_far, z_near bilan: M_proj formulasini qo'llang.",
    ]
    _solution = "x_proj = points[0] / points[2]; y_proj = points[1] / points[2]"

    def _do_check(self, x_proj, y_proj, points):
        exp_x = points[0] / points[2]
        exp_y = points[1] / points[2]
        if not np.allclose(x_proj, exp_x, atol=1e-6):
            return f"x' = x/z. Kutilgan: {exp_x}"
        if not np.allclose(y_proj, exp_y, atol=1e-6):
            return f"y' = y/z. Kutilgan: {exp_y}"
        return True


class Q6(EqualityCheckProblem):
    """SVD bilan rasm siqish va qayta tiklash."""
    _hints = [
        "U, s, Vt = svd(img). A_k = U[:,:k] @ diag(s[:k]) @ Vt[:k,:] — k-rang taqrib.",
    ]
    _solution = (
        "U, s, Vt = np.linalg.svd(img, full_matrices=False)\n"
        "img_k = U[:,:k] @ np.diag(s[:k]) @ Vt[:k,:]"
    )

    def _do_check(self, img_k, img, k):
        U, s, Vt = np.linalg.svd(img, full_matrices=False)
        expected = U[:, :k] @ np.diag(s[:k]) @ Vt[:k, :]
        if not np.allclose(img_k, expected, atol=1e-8):
            return f"k={k} bilan SVD taqribini tekshiring."
        return True


class C1_Q1(EqualityCheckProblem):
    """B-spline egri chiziq uchun kontrol nuqtalardan koeffitsiyentlarni hisoblang."""
    _hints = [
        "B-spline: P(t) = sum B_{i,k}(t) P_i. Matrits ko'rinish: C = B_mat @ control_pts.",
    ]
    _solution = (
        "# Cubic bezier (n=4 nuqta)\n"
        "def bezier(t, P):\n"
        "    n = len(P) - 1\n"
        "    from math import comb\n"
        "    return sum(comb(n,i)*(1-t)**(n-i)*t**i*P[i] for i in range(n+1))\n"
        "curve = np.array([bezier(t, control_pts) for t in np.linspace(0,1,100)])"
    )

    def _do_check(self, curve, control_pts):
        from math import comb
        n = len(control_pts) - 1
        def bezier(t, P):
            return sum(comb(n, i) * (1-t)**(n-i) * t**i * P[i] for i in range(n+1))
        expected = np.array([bezier(t, control_pts) for t in np.linspace(0, 1, 100)])
        if curve.shape != expected.shape:
            return f"Egri chiziq shakli {expected.shape} bo'lishi kerak."
        if not np.allclose(curve, expected, atol=1e-6):
            return "Bezier egri chizig'i formulasini tekshiring."
        return True


class C2_Q1(ThoughtExperiment):
    """Kompyuter grafikasida chiziqli algebra qanday rol o'ynaydi?"""
    _hints = [
        "OpenGL/DirectX: barcha transformatsiyalar 4x4 matritsalar.",
        "GPU = massiv parallel matritsa ko'paytmasi.",
    ]
    _solution = (
        "Kompyuter grafikasida chiziqli algebra:\n\n"
        "1) Transformatsiyalar: aylantirish, masshtab, perspektiv — 4x4 matritsalar.\n"
        "   GPU'da bir vaqtda millionlab nuqtalar: P_screen = M_proj @ M_view @ M_model @ P.\n"
        "2) Yorug'lik: normal vektorlar, dot product, Phong modeli.\n"
        "3) Animatsiya: interpolyatsiya (SLERP — quaternion bilan aylantirish).\n"
        "4) Rasm siqish: JPEG (DCT = kosinus Furye), SVD asosli kompressiya.\n"
        "5) Soya va ray-tracing: chiziqli tenglamalar — kesishish hisoblash.\n"
        "GPU arxitekturasi = massiv parallel SIMD — matritsa/vektor amallariga optimallashgan."
    )
