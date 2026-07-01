"""Hints and solutions — Dars 9.1: Kompleks Sonlar."""
import numpy as np
from learntools.linear_algebra_uz.base import UzCheckProblem, ThoughtExperiment


class Q1(UzCheckProblem):
    """z = a + bi kompleks sonning modulini va argumentini toping."""
    _hints = ["abs(z) yoki np.abs(z) — modul. np.angle(z) — argument (radyanda)."]
    _solution = "mod = np.abs(z); arg = np.angle(z)"

    def _do_check(self, mod, arg, z):
        if not np.isclose(mod, np.abs(z), rtol=1e-6):
            return f"Modul: {np.abs(z):.6f}, siz {mod:.6f} berdingiz."
        if not np.isclose(arg, np.angle(z), rtol=1e-6):
            return f"Argument: {np.angle(z):.6f}, siz {arg:.6f} berdingiz."
        return True


class Q2(UzCheckProblem):
    """z1 * z2 va z1 / z2 ni hisoblang."""
    _hints = ["Python kompleks sonlar bilan to'g'ridan-to'g'ri ishlaydi: z1 * z2, z1 / z2."]
    _solution = "prod = z1 * z2; quot = z1 / z2"

    def _do_check(self, prod, quot, z1, z2):
        if not np.isclose(prod, z1 * z2, rtol=1e-8):
            return f"z1*z2 = {z1*z2}, siz {prod} berdingiz."
        if not np.isclose(quot, z1 / z2, rtol=1e-8):
            return f"z1/z2 = {z1/z2}, siz {quot} berdingiz."
        return True


class Q3(UzCheckProblem):
    """Euler formulasi: e^{iθ} = cosθ + i sinθ ni tekshiring."""
    _hints = [
        "np.exp(1j * theta) — kompleks eksponenta.",
        "np.isclose(np.exp(1j*theta), np.cos(theta) + 1j*np.sin(theta))",
    ]
    _solution = "euler = np.exp(1j * theta); expected = np.cos(theta) + 1j * np.sin(theta)"

    def _do_check(self, euler, theta):
        expected = np.cos(theta) + 1j * np.sin(theta)
        if not np.isclose(euler, expected, rtol=1e-8):
            return f"e^(i*theta) = {expected}, siz {euler} berdingiz."
        return True


class Q4(UzCheckProblem):
    """Kompleks konjugat va |z|² = z * z.conj() ni tekshiring."""
    _hints = ["z.conjugate() yoki np.conj(z). |z|² = z * z.conj() = a² + b²."]
    _solution = "z_conj = np.conj(z); mod_sq = z * np.conj(z)"

    def _do_check(self, z_conj, mod_sq, z):
        if not np.isclose(z_conj, np.conj(z), rtol=1e-8):
            return f"Konjugat: {np.conj(z)}, siz {z_conj} berdingiz."
        expected_sq = z * np.conj(z)
        if not np.isclose(mod_sq, expected_sq, rtol=1e-8):
            return f"|z|² = {expected_sq.real:.6f}, siz {mod_sq} berdingiz."
        return True


class Q5(UzCheckProblem):
    """De Moivre teoremasi: (r*e^{iθ})^n = r^n * e^{inθ}."""
    _hints = ["z**n — Python da kompleks son daraja. yoki np.abs(z)**n * np.exp(1j*np.angle(z)*n)."]
    _solution = "zn = z ** n  # yoki: np.abs(z)**n * np.exp(1j * np.angle(z) * n)"

    def _do_check(self, zn, z, n):
        expected = z ** n
        if not np.isclose(zn, expected, rtol=1e-6):
            return f"z^{n} = {expected}, siz {zn} berdingiz."
        return True


class Q6(UzCheckProblem):
    """n-tartibli birlikning ildizlarini toping: z^n = 1."""
    _hints = [
        "z_k = e^{2πik/n}, k = 0,1,...,n-1.",
        "roots = [np.exp(2j*np.pi*k/n) for k in range(n)]",
    ]
    _solution = "roots = np.array([np.exp(2j * np.pi * k / n) for k in range(n)])"

    def _do_check(self, roots, n):
        expected = np.array([np.exp(2j * np.pi * k / n) for k in range(n)])
        if len(roots) != n:
            return f"{n} ta ildiz bo'lishi kerak, siz {len(roots)} ta berdingiz."
        roots_s = sorted(roots, key=lambda z: np.angle(z))
        exp_s = sorted(expected, key=lambda z: np.angle(z))
        if not np.allclose(roots_s, exp_s, atol=1e-8):
            return f"Birlikning {n}-ildizlari: {exp_s}"
        return True


class C1_Q1(UzCheckProblem):
    """DFT matritsasini yarating: F_{jk} = e^{-2πijk/n} / sqrt(n)."""
    _hints = [
        "F[j,k] = np.exp(-2j*np.pi*j*k/n) / np.sqrt(n).",
        "j,k = np.mgrid[0:n, 0:n] bilan vektorizatsiya.",
    ]
    _solution = (
        "j, k = np.mgrid[0:n, 0:n]\n"
        "F = np.exp(-2j * np.pi * j * k / n) / np.sqrt(n)"
    )

    def _do_check(self, F, n):
        j, k = np.mgrid[0:n, 0:n]
        expected = np.exp(-2j * np.pi * j * k / n) / np.sqrt(n)
        if F.shape != (n, n):
            return f"F shakli ({n},{n}) bo'lishi kerak."
        if not np.allclose(F, expected, atol=1e-8):
            return "F[j,k] = exp(-2πijk/n)/sqrt(n) formulasini tekshiring."
        if not np.allclose(F @ F.conj().T, np.eye(n), atol=1e-8):
            return "DFT matritsasi unitariy bo'lishi kerak: F F* = I."
        return True


class C2_Q1(ThoughtExperiment):
    """Kompleks sonlar nima uchun chiziqli algebrada muhim?"""
    _hints = [
        "Haqiqiy matritsaning ham kompleks xususiy qiymatlari bo'lishi mumkin.",
        "Fourier tahlili, to'lqin tenglamalari, kvant mexanikasi.",
    ]
    _solution = (
        "Kompleks sonlar chiziqli algebrada quyidagi holatlarda zarur:\n"
        "1) Haqiqiy matritsaning xususiy qiymatlari kompleks bo'lishi mumkin "
        "(masalan, rotatsiya matritsasi: lambda = e^{±iθ}).\n"
        "2) Fourier tahlili: e^{ikx} — tebranish bazasi.\n"
        "3) Kvant mexanikasi: holat vektorlari kompleks Hilbert fazosida.\n"
        "4) Elektr texnikasi: impedans Z = R + iωL.\n"
        "Fundamental teorema: n-tartibli polinom aynan n ta kompleks ildizga ega — "
        "shuning uchun n×n matritsa n ta xususiy qiymatga ega (C da)."
    )
