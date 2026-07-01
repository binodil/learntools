"""Hints and solutions — Dars 14.4: Backpropagation va Zanjir Qoidasi."""
import numpy as np
from learntools.core import EqualityCheckProblem, ThoughtExperiment


class Q1(EqualityCheckProblem):
    """Chiziqli qatlam Yakobiani: dy/dx = W."""
    _hints = [
        "y = W x + b uchun dy/dx = W.",
        "Shunchaki W ni qaytaring.",
    ]
    _solution = "J = W"

    def _do_check(self, J, W):
        if not np.allclose(J, W, atol=1e-6):
            return f"Kutilgan: W, siz berdingiz: {J}"
        return True


class Q2(EqualityCheckProblem):
    """Aktivatsiya Yakobiani diagonal: diag(sigma'(z)). ReLU."""
    _hints = [
        "ReLU' (z) = 1 agar z>0, aks holda 0.",
        "Yakobian = np.diag((z>0).astype(float)).",
    ]
    _solution = "J = np.diag((z > 0).astype(float))"

    def _do_check(self, J, z):
        expected = np.diag((z > 0).astype(float))
        if not np.allclose(J, expected, atol=1e-6):
            return f"Kutilgan diagonal Yakobian: {expected}, siz berdingiz: {J}"
        return True


class Q3(EqualityCheckProblem):
    """Zanjir qoidasi: dz/dx = Jg @ Jf."""
    _hints = [
        "Kompozitsiya Yakobiani — Yakobianlar ko'paytmasi.",
        "dz/dx = Jg @ Jf (tartibga e'tibor: tashqi @ ichki).",
    ]
    _solution = "J = Jg @ Jf"

    def _do_check(self, J, Jg, Jf):
        expected = Jg @ Jf
        if not np.allclose(J, expected, atol=1e-6):
            return f"Kutilgan: Jg @ Jf, siz berdingiz: {J}"
        return True


class Q4(EqualityCheckProblem):
    """Kvadratik yo'qotish gradienti: dL/dyhat = yhat - y."""
    _hints = [
        "L = 0.5 * ||yhat - y||^2.",
        "dL/dyhat = yhat - y.",
    ]
    _solution = "grad = yhat - y"

    def _do_check(self, grad, yhat, y):
        expected = yhat - y
        if not np.allclose(grad, expected, atol=1e-6):
            return f"Kutilgan: {expected}, siz berdingiz: {grad}"
        return True


class Q5(EqualityCheckProblem):
    """Og'irlik gradienti: dL/dW = outer(dz, x)."""
    _hints = [
        "z = W x bo'lsa, dL/dW = (dL/dz) x^T.",
        "np.outer(dz, x) tashqi ko'paytma matritsasini beradi.",
    ]
    _solution = "dW = np.outer(dz, x)"

    def _do_check(self, dW, dz, x):
        expected = np.outer(dz, x)
        if not np.allclose(dW, expected, atol=1e-6):
            return f"Kutilgan: outer(dz, x), siz berdingiz: {dW}"
        return True


class Q6(EqualityCheckProblem):
    """Sonli gradient (markaziy ayirma) bir komponent uchun."""
    _hints = [
        "df/dtheta ~ (f(theta+eps) - f(theta-eps)) / (2 eps).",
        "f skalyar funksiya, eps kichik (masalan 1e-5).",
    ]
    _solution = "g = (f(theta + eps) - f(theta - eps)) / (2*eps)"

    def _do_check(self, g, f, theta, eps):
        expected = (f(theta + eps) - f(theta - eps)) / (2*eps)
        if not np.allclose(g, expected, atol=1e-5):
            return f"Kutilgan: {expected}, siz berdingiz: {g}"
        return True


# Challenge 1
class C1_Q1(EqualityCheckProblem):
    """2 qatlamli tarmoq uchun dW1 ni orqaga tarqalish bilan toping."""
    _hints = [
        "Oldinga: z1=W1 x, a1=relu(z1), yhat=W2 a1.",
        "Orqaga: dyhat=yhat-y; da1=W2^T dyhat; dz1=da1*relu'(z1); dW1=outer(dz1,x).",
    ]
    _solution = (
        "z1 = W1 @ x; a1 = np.maximum(0, z1); yhat = W2 @ a1\n"
        "dyhat = yhat - y\n"
        "da1 = W2.T @ dyhat\n"
        "dz1 = da1 * (z1 > 0)\n"
        "dW1 = np.outer(dz1, x)"
    )

    def _do_check(self, dW1, W1, W2, x, y):
        z1 = W1 @ x
        a1 = np.maximum(0, z1)
        yhat = W2 @ a1
        dyhat = yhat - y
        da1 = W2.T @ dyhat
        dz1 = da1 * (z1 > 0)
        expected = np.outer(dz1, x)
        if not np.allclose(dW1, expected, atol=1e-6):
            return f"Kutilgan dW1: {expected}, siz berdingiz: {dW1}"
        return True


# Challenge 2
class C2_Q1(ThoughtExperiment):
    """Nima uchun orqaga tarqalish 'orqaga' va Yakobianlar ko'paytmasi?"""
    _hints = [
        "Yo'qotish skalyar — gradient satr vektor (1 x n).",
        "Satr vektorni o'ngdan Yakobianlarga ko'paytirish chapdan-o'ngga arzonroq.",
    ]
    _solution = (
        "Tarmoq — funksiyalar kompozitsiyasi, shuning uchun zanjir qoidasi bo'yicha\n"
        "to'liq Yakobian = J_L J_{L-1} ... J_1 (Yakobianlar ko'paytmasi). Yo'qotish\n"
        "skalyar bo'lgani uchun chiqishdagi gradient satr vektor (1 x d). Uni\n"
        "chapdan boshlab ketma-ket Yakobianlarga ko'paytirsak, har doim vektor x\n"
        "matritsa amalini bajaramiz (arzon), to'liq matritsa x matritsa emas.\n"
        "Shu sababli hisoblash chiqishdan kirishga — 'orqaga' — yuritiladi.\n"
        "Aktivatsiya Yakobiani diagonal bo'lgani uchun, chiziqli qatlam esa W^T\n"
        "bilan ifodalangani uchun bu juda samarali."
    )
