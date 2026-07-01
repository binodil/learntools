"""Hints and solutions — Dars 14.1: MLP (Multi-Layer Perceptron)."""
import numpy as np
from learntools.core import EqualityCheckProblem, ThoughtExperiment


class Q1(EqualityCheckProblem):
    """Bitta chiziqli qatlam: y = W x."""
    _hints = [
        "Chiziqli qatlam: y = W @ x.",
        "W (m x n), x (n,) bo'lsa, natija (m,) vektor.",
    ]
    _solution = "y = W @ x"

    def _do_check(self, y, W, x):
        expected = W @ x
        if not np.allclose(y, expected, atol=1e-6):
            return f"Kutilgan: {expected}, siz berdingiz: {y}"
        return True


class Q2(EqualityCheckProblem):
    """ReLU aktivatsiyasini elementma-element hisoblang."""
    _hints = [
        "ReLU(z) = max(0, z), har bir komponent uchun.",
        "np.maximum(0, z) butun vektorga qo'llanadi.",
    ]
    _solution = "a = np.maximum(0, z)"

    def _do_check(self, a, z):
        expected = np.maximum(0, z)
        if not np.allclose(a, expected, atol=1e-6):
            return f"Kutilgan: {expected}, siz berdingiz: {a}"
        return True


class Q3(EqualityCheckProblem):
    """Ikki qatlamli tarmoq chiqishi."""
    _hints = [
        "Avval h = ReLU(W1 @ x + b1).",
        "Keyin yhat = W2 @ h + b2 (chiziqli chiqish).",
    ]
    _solution = "h = np.maximum(0, W1 @ x + b1)\nyhat = W2 @ h + b2"

    def _do_check(self, yhat, W1, b1, W2, b2, x):
        h = np.maximum(0, W1 @ x + b1)
        expected = W2 @ h + b2
        if not np.allclose(yhat, expected, atol=1e-6):
            return f"Kutilgan: {expected}, siz berdingiz: {yhat}"
        return True


class Q4(EqualityCheckProblem):
    """Diqqat (attention) skori: dot product q.k / sqrt(d)."""
    _hints = [
        "Skor = q . k (skalyar ko'paytma).",
        "Masshtablangan: (q @ k) / np.sqrt(len(q)).",
    ]
    _solution = "score = (q @ k) / np.sqrt(len(q))"

    def _do_check(self, score, q, k):
        expected = (q @ k) / np.sqrt(len(q))
        if not np.allclose(score, expected, atol=1e-6):
            return f"Kutilgan: {expected}, siz berdingiz: {score}"
        return True


class Q5(EqualityCheckProblem):
    """Partiya (batch) oldinga yurishi: Z = W X + b."""
    _hints = [
        "X ustunlari namunalar (n x N).",
        "Z = W @ X + b[:, None] — bias ustunlar bo'ylab qo'shiladi.",
    ]
    _solution = "Z = W @ X + b[:, None]"

    def _do_check(self, Z, W, b, X):
        expected = W @ X + b[:, None]
        if not np.allclose(Z, expected, atol=1e-6):
            return f"Kutilgan shakl/qiymat mos kelmadi: {expected}"
        return True


class Q6(EqualityCheckProblem):
    """W ning spektral normasi (eng katta singulyar qiymat)."""
    _hints = [
        "Spektral norma = eng katta singulyar qiymat.",
        "np.linalg.norm(W, 2) yoki np.linalg.svd(W)[1][0].",
    ]
    _solution = "s = np.linalg.norm(W, 2)"

    def _do_check(self, s, W):
        expected = np.linalg.norm(W, 2)
        if not np.allclose(s, expected, atol=1e-5):
            return f"Kutilgan: {expected}, siz berdingiz: {s}"
        return True


# Challenge 1
class C1_Q1(EqualityCheckProblem):
    """Aktivatsiyasiz ikki chiziqli qatlam bitta matritsaga yig'iladi."""
    _hints = [
        "y = W2 (W1 x) = (W2 W1) x.",
        "Yagona ekvivalent matritsa: W = W2 @ W1.",
    ]
    _solution = "W = W2 @ W1   # ikki chiziqli qatlam = bitta matritsa"

    def _do_check(self, W, W1, W2):
        expected = W2 @ W1
        if not np.allclose(W, expected, atol=1e-6):
            return f"Kutilgan: W2 @ W1, siz berdingiz: {W}"
        return True


# Challenge 2
class C2_Q1(ThoughtExperiment):
    """Nima uchun nochiziqlilik MLP ga kuch beradi?"""
    _hints = [
        "Aktivatsiyasiz ketma-ket qatlamlar bitta affin akslantirishga yig'iladi.",
        "ReLU bo'lakli-chiziqli — u kombinatsiyalanganda murakkab egri sirtlar hosil qiladi.",
    ]
    _solution = (
        "Agar sigma bo'lmasa, W2(W1 x) = (W2 W1) x — qancha qatlam bo'lsa ham,\n"
        "natija bitta chiziqli akslantirish bo'lib qoladi va chiziqli bo'lmagan\n"
        "funksiyalarni ifodalay olmaydi. Nochiziqli sigma (masalan ReLU) bu\n"
        "yig'ilishni buzadi: har bir qatlam fazoni egadi/buradi, natijada tarmoq\n"
        "universal approksimator bo'ladi. Chiziqli algebra (W matritsalari) 'qayerga\n"
        "akslantirishni', nochiziqlilik esa 'egishni' beradi."
    )
