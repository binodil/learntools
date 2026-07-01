"""Hints and solutions — Dars 14.3: RNN (Recurrent Neural Networks)."""
import numpy as np
from learntools.linear_algebra_uz.base import UzCheckProblem, ThoughtExperiment


class Q1(UzCheckProblem):
    """Bitta RNN qadami: h_t = tanh(Wh h + Wx x)."""
    _hints = [
        "h_yangi = tanh(Wh @ h + Wx @ x).",
        "np.tanh butun vektorga elementma-element qo'llanadi.",
    ]
    _solution = "h_new = np.tanh(Wh @ h + Wx @ x)"

    def _do_check(self, h_new, Wh, Wx, h, x):
        expected = np.tanh(Wh @ h + Wx @ x)
        if not np.allclose(h_new, expected, atol=1e-6):
            return f"Kutilgan: {expected}, siz berdingiz: {h_new}"
        return True


class Q2(UzCheckProblem):
    """W_h ning spektral radiusi (max |lambda|)."""
    _hints = [
        "Spektral radius = max(|xususiy qiymatlar|).",
        "np.max(np.abs(np.linalg.eigvals(Wh))).",
    ]
    _solution = "rho = np.max(np.abs(np.linalg.eigvals(Wh)))"

    def _do_check(self, rho, Wh):
        expected = np.max(np.abs(np.linalg.eigvals(Wh)))
        if not np.allclose(rho, expected, atol=1e-5):
            return f"Kutilgan: {expected}, siz berdingiz: {rho}"
        return True


class Q3(UzCheckProblem):
    """Chiziqli holatning yoyilishi: h_2 = Wh(Wx x1) + Wx x2."""
    _hints = [
        "h_0 = 0, h_1 = Wx x1, h_2 = Wh h_1 + Wx x2.",
        "Demak h_2 = Wh @ (Wx @ x1) + Wx @ x2.",
    ]
    _solution = "h2 = Wh @ (Wx @ x1) + Wx @ x2"

    def _do_check(self, h2, Wh, Wx, x1, x2):
        expected = Wh @ (Wx @ x1) + Wx @ x2
        if not np.allclose(h2, expected, atol=1e-6):
            return f"Kutilgan: {expected}, siz berdingiz: {h2}"
        return True


class Q4(UzCheckProblem):
    """W_h darajasi: W_h^t."""
    _hints = [
        "Matritsa darajasi: np.linalg.matrix_power(Wh, t).",
        "Bu Wh @ Wh @ ... (t marta) ga teng.",
    ]
    _solution = "Wt = np.linalg.matrix_power(Wh, t)"

    def _do_check(self, Wt, Wh, t):
        expected = np.linalg.matrix_power(Wh, t)
        if not np.allclose(Wt, expected, atol=1e-6):
            return f"Kutilgan: Wh^{t}, siz berdingiz: {Wt}"
        return True


class Q5(UzCheckProblem):
    """Ortogonal matritsaning singulyar qiymatlari hammasi 1."""
    _hints = [
        "Ortogonal Q uchun Q^T Q = I.",
        "Barcha singulyar qiymatlar 1 ga teng: np.linalg.svd(Q)[1].",
    ]
    _solution = "s = np.linalg.svd(Q, compute_uv=False)  # hammasi ~1"

    def _do_check(self, s, Q):
        expected = np.linalg.svd(Q, compute_uv=False)
        if not np.allclose(s, expected, atol=1e-6) or not np.allclose(expected, 1.0, atol=1e-6):
            return f"Kutilgan: barcha singulyar qiymat 1, siz berdingiz: {s}"
        return True


class Q6(UzCheckProblem):
    """Spektral radiusni belgilangan qiymatga masshtablash."""
    _hints = [
        "rho(c*A) = c*rho(A).",
        "Wh = target * A / rho(A) bersa, rho(Wh) = target.",
    ]
    _solution = "rho = np.max(np.abs(np.linalg.eigvals(A))); Wh = target * A / rho"

    def _do_check(self, Wh, A, target):
        rho = np.max(np.abs(np.linalg.eigvals(Wh)))
        if not np.allclose(rho, target, atol=1e-5):
            return f"Kutilgan spektral radius: {target}, siz berdingiz: {rho}"
        return True


# Challenge 1
class C1_Q1(UzCheckProblem):
    """To'liq RNN yurishini amalga oshiring va oxirgi holatni qaytaring."""
    _hints = [
        "h ni ketma-ket yangilang: h = tanh(Wh @ h + Wx @ x).",
        "xs har bir satrini (yoki ustunini) navbat bilan qayta ishlang.",
    ]
    _solution = (
        "h = h0\n"
        "for x in xs:\n"
        "    h = np.tanh(Wh @ h + Wx @ x)\n"
        "h_final = h"
    )

    def _do_check(self, h_final, Wh, Wx, xs, h0):
        h = h0
        for x in xs:
            h = np.tanh(Wh @ h + Wx @ x)
        if not np.allclose(h_final, h, atol=1e-6):
            return f"Kutilgan: {h}, siz berdingiz: {h_final}"
        return True


# Challenge 2
class C2_Q1(ThoughtExperiment):
    """Nima uchun W_h xususiy qiymatlari RNN barqarorligini belgilaydi?"""
    _hints = [
        "Yoyilgan RNN da gradient ~ Wh^T ko'paytmasini o'z ichiga oladi.",
        "Wh = V Lambda V^{-1} bo'lsa, Wh^t = V Lambda^t V^{-1}, ya'ni lambda^t.",
    ]
    _solution = (
        "Yoyilgan RNN da h_t = Wh^t h_0 + ... va orqaga tarqalishda gradient\n"
        "taxminan Wh^T zanjirini o'z ichiga oladi. Wh = V Lambda V^{-1} bo'lsa,\n"
        "Wh^t = V Lambda^t V^{-1}, demak har bir mod lambda_i^t bo'yicha o'sadi.\n"
        "Spektral radius rho = max|lambda_i| ni belgilaymiz:\n"
        "  rho < 1  -> Wh^t -> 0  (gradient o'chadi, uzoq xotira yo'qoladi)\n"
        "  rho > 1  -> Wh^t -> inf (gradient portlaydi, o'qitish beqaror)\n"
        "  rho ~ 1  -> barqaror, uzoq muddatli bog'liqlik saqlanadi.\n"
        "Shu sababli Wh ko'pincha ortogonal (barcha singulyar qiymat = 1)\n"
        "boshlanadi yoki LSTM/GRU kabi mexanizmlar ishlatiladi."
    )
