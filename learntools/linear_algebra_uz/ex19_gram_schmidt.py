"""Hints and solutions — Dars 4.4: Ortonormal Bazalar va Gram-Schmidt."""
import numpy as np
from learntools.core import EqualityCheckProblem, ThoughtExperiment


class Q1(EqualityCheckProblem):
    """Vektorni normallashtiring (birlik vektori)."""
    _hints = [
        "Birlik vektor: u = v / ||v|| = v / np.linalg.norm(v).",
    ]
    _solution = "u = v / np.linalg.norm(v)"

    def _do_check(self, u, v):
        expected = v / np.linalg.norm(v)
        if not np.allclose(u, expected, atol=1e-8):
            return f"Kutilgan: {expected}, siz {u} berdingiz."
        if not np.isclose(np.linalg.norm(u), 1.0, atol=1e-8):
            return f"Birlik vektor norma 1 bo'lishi kerak, lekin ||u|| = {np.linalg.norm(u):.6f}"
        return True


class Q2(EqualityCheckProblem):
    """Gram-Schmidt: a1, a2 dan q1, q2 ortonormal bazis hosil qiling."""
    _hints = [
        "q1 = a1 / ||a1||, so'ng A2 = a2 - (a2·q1)q1, q2 = A2 / ||A2||.",
        "np.linalg.qr(A) funksiyasi ham QR yoyilmasini beradi.",
    ]
    _solution = (
        "q1 = a1 / np.linalg.norm(a1)\n"
        "A2 = a2 - np.dot(a2, q1) * q1\n"
        "q2 = A2 / np.linalg.norm(A2)"
    )

    def _do_check(self, q1, q2, a1, a2):
        q1_exp = a1 / np.linalg.norm(a1)
        A2 = a2 - np.dot(a2, q1_exp) * q1_exp
        q2_exp = A2 / np.linalg.norm(A2)
        if not np.allclose(np.abs(q1), np.abs(q1_exp), atol=1e-6):
            return "q1 noto'g'ri. q1 = a1 / ||a1|| bo'lishi kerak."
        if not np.allclose(np.abs(q2), np.abs(q2_exp), atol=1e-6):
            return "q2 noto'g'ri. a1 ning q1 yo'nalishidagi qismini ayirish kerak."
        if not np.isclose(np.dot(q1, q2), 0, atol=1e-8):
            return f"q1 va q2 ortogonal bo'lishi kerak. q1·q2 = {np.dot(q1, q2):.6f}"
        return True


class Q3(EqualityCheckProblem):
    """QR yoyilmasini hisoblang."""
    _hints = [
        "np.linalg.qr(A) Q va R ni qaytaradi.",
        "A = Q @ R bo'lishini tekshiring: Q ortonormal, R yuqori uchburchak.",
    ]
    _solution = "Q, R = np.linalg.qr(A)"

    def _do_check(self, Q, R, A):
        if not np.allclose(Q @ R, A, atol=1e-8):
            return "A = Q @ R bo'lishi kerak."
        if not np.allclose(Q.T @ Q, np.eye(Q.shape[1]), atol=1e-8):
            return "Q ortonormal bo'lishi kerak: Qᵀ Q = I."
        return True


class Q4(EqualityCheckProblem):
    """Ortonormal ustunlar matritsasi Q uchun ||Qx|| = ||x|| ekanini tekshiring."""
    _hints = [
        "Q izometrik (uzunlikni saqlovchi): ||Qx||² = xᵀQᵀQx = xᵀIx = ||x||².",
        "np.isclose(np.linalg.norm(Q @ x), np.linalg.norm(x)) dan foydalaning.",
    ]
    _solution = "preserved = np.isclose(np.linalg.norm(Q @ x), np.linalg.norm(x))"

    def _do_check(self, preserved, Q, x):
        expected = np.isclose(np.linalg.norm(Q @ x), np.linalg.norm(x))
        if preserved != expected:
            norm_Qx = np.linalg.norm(Q @ x)
            norm_x = np.linalg.norm(x)
            return f"||Qx|| = {norm_Qx:.6f}, ||x|| = {norm_x:.6f}. Teng: {expected}"
        return True


class Q5(EqualityCheckProblem):
    """Ax = b ni QR yordamida yeching: Rx = Qᵀb."""
    _hints = [
        "A = QR, shuning uchun QRx = b → Rx = Qᵀb (chunki Qᵀ Q = I).",
        "Q, R = np.linalg.qr(A), keyin x = np.linalg.solve(R, Q.T @ b).",
    ]
    _solution = "Q, R = np.linalg.qr(A); x = np.linalg.solve(R, Q.T @ b)"

    def _do_check(self, x, A, b):
        expected = np.linalg.lstsq(A, b, rcond=None)[0]
        if not np.allclose(x, expected, atol=1e-6):
            return f"Kutilgan yechim: {expected}, siz {x} berdingiz."
        return True


class Q6(EqualityCheckProblem):
    """Uch vektor uchun to'liq Gram-Schmidt jarayoni."""
    _hints = [
        "Har bir vektordagi avvalgi bazis vektorlariga tushuvchi qismlarni ayiring.",
        "np.linalg.qr(A)[0] — bu Gram-Schmidt natijasi (Q matritsasi).",
    ]
    _solution = (
        "# a1, a2, a3 ustunlardan iborat A matritsa uchun:\n"
        "A_cols = np.column_stack([a1, a2, a3])\n"
        "Q, _ = np.linalg.qr(A_cols)\n"
        "q1, q2, q3 = Q[:, 0], Q[:, 1], Q[:, 2]"
    )

    def _do_check(self, Q, A):
        Q_exp, _ = np.linalg.qr(A)
        if Q.shape != Q_exp.shape:
            return f"Q shakli {Q_exp.shape} bo'lishi kerak, siz {Q.shape} berdingiz."
        # Check orthonormality
        if not np.allclose(Q.T @ Q, np.eye(Q.shape[1]), atol=1e-6):
            return "Q ustunlari ortonormal bo'lishi kerak (Qᵀ Q = I)."
        # Check same column space
        rank_check = np.linalg.matrix_rank(np.hstack([Q, A]))
        if rank_check != np.linalg.matrix_rank(A):
            return "Q ning ustun fazosi A ning ustun fazosiga teng bo'lishi kerak."
        return True


class C1_Q1(EqualityCheckProblem):
    """QR yordamida kichik kvadratlar yechimi."""
    _hints = [
        "A = QR, normal tenglama: Rx̂ = Qᵀb.",
        "Q, R = np.linalg.qr(A, mode='reduced'); x_hat = np.linalg.solve(R, Q.T @ b).",
    ]
    _solution = "Q, R = np.linalg.qr(A, mode='reduced'); x_hat = np.linalg.solve(R, Q.T @ b)"

    def _do_check(self, x_hat, A, b):
        expected = np.linalg.lstsq(A, b, rcond=None)[0]
        if not np.allclose(x_hat, expected, atol=1e-6):
            return f"Kutilgan x̂ = {expected}, siz {x_hat} berdingiz."
        return True


class C2_Q1(ThoughtExperiment):
    """Nima uchun QR yoyilmasi kichik kvadratlar uchun numerik jihatdan afzalroq?"""
    _hints = [
        "Normal tenglama AᵀAx = Aᵀb da AᵀA ning shart soni (condition number) kattaroq.",
        "QR da R ning shart soni A ga teng, AᵀA gacha ko'tarilmaydi.",
    ]
    _solution = (
        "Normal tenglama AᵀAx̂ = Aᵀb da muammo: κ(AᵀA) = κ(A)² — shart soni kvadrat "
        "ko'tariladi. A yomon shartlangan bo'lsa (kichik singular qiymatlar bor bo'lsa), "
        "AᵀA yanada yomon shartlangan bo'ladi va yechim numerik jihatdan beqaror bo'ladi. "
        "QR yoyilmasida esa Rx̂ = Qᵀb tenglamasini yechamiz: κ(R) = κ(A). "
        "Shuning uchun QR usuli ikki baravar aniqroq."
    )
