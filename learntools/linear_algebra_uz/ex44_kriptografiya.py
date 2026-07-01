"""Hints and solutions — Dars 10.7: Chiziqli Algebra va Kriptografiya."""
import numpy as np
from learntools.linear_algebra_uz.base import UzCheckProblem, ThoughtExperiment


class Q1(UzCheckProblem):
    """Hill shifri: C = A @ P (mod 26)."""
    _hints = [
        "Hill shifri: C = (A @ P) % 26, A — shifr matritsasi (invertibel mod 26).",
    ]
    _solution = "C = (A @ P) % 26"

    def _do_check(self, C, A, P):
        expected = (A @ P) % 26
        if not np.allclose(C, expected):
            return "C = (A @ P) % 26 formulasini tekshiring."
        return True


class Q2(UzCheckProblem):
    """Hill shifrini ochish: P = A⁻¹ @ C (mod 26)."""
    _hints = [
        "Modul bo'yicha teskari: det_inv = pow(int(round(np.linalg.det(A))), -1, 26).",
        "A_inv_mod26 — modulli teskari matritsa.",
    ]
    _solution = (
        "det = int(round(np.linalg.det(A))) % 26\n"
        "det_inv = pow(det, -1, 26)\n"
        "adj = np.round(np.linalg.det(A) * np.linalg.inv(A)).astype(int) % 26\n"
        "A_inv = (det_inv * adj) % 26\n"
        "P = (A_inv @ C) % 26"
    )

    def _do_check(self, P, A, C):
        det = int(round(np.linalg.det(A))) % 26
        det_inv = pow(det, -1, 26)
        adj = np.round(np.linalg.det(A) * np.linalg.inv(A)).astype(int) % 26
        A_inv = (det_inv * adj) % 26
        expected = (A_inv @ C) % 26
        if not np.allclose(P, expected):
            return "A⁻¹ mod 26 bilan P = A⁻¹C mod 26 formulasini tekshiring."
        return True


class Q3(UzCheckProblem):
    """Error-correcting code: parity check matritsasi H @ c = 0 ekanini tekshiring."""
    _hints = [
        "H @ c = 0 (mod 2) bo'lsa — kod so'zi to'g'ri (hech qanday xato yo'q).",
    ]
    _solution = "syndrome = (H @ c) % 2; valid = np.all(syndrome == 0)"

    def _do_check(self, valid, H, c):
        syndrome = (H @ c) % 2
        expected = np.all(syndrome == 0)
        if valid != expected:
            return f"Sindrom: {syndrome}. To'g'ri kod so'zi? {expected}"
        return True


class Q4(UzCheckProblem):
    """Xato pozitsiyasini sindrom bilan toping."""
    _hints = [
        "Sindrom s = H @ r mod 2. s ning ustun raqami xato pozitsiyasini beradi.",
    ]
    _solution = "s = (H @ r) % 2; error_pos = np.where(np.all(H.T == s, axis=1))[0][0]"

    def _do_check(self, error_pos, H, r):
        s = (H @ r) % 2
        cols = H.T
        for i, col in enumerate(cols):
            if np.all(col == s):
                expected = i
                break
        else:
            expected = -1
        if error_pos != expected:
            return f"Xato pozitsiyasi: {expected}, siz {error_pos} berdingiz."
        return True


class Q5(UzCheckProblem):
    """RSA: n = p*q, e*d ≡ 1 (mod φ(n)) ni tekshiring."""
    _hints = [
        "φ(n) = (p-1)*(q-1). e*d mod φ(n) = 1.",
    ]
    _solution = "phi_n = (p-1)*(q-1); valid_rsa = (e * d) % phi_n == 1"

    def _do_check(self, valid_rsa, p, q, e, d):
        phi_n = (p - 1) * (q - 1)
        expected = (e * d) % phi_n == 1
        if valid_rsa != expected:
            return f"φ(n) = {phi_n}, e*d = {e*d}, e*d mod φ(n) = {(e*d)%phi_n}. To'g'ri? {expected}"
        return True


class Q6(UzCheckProblem):
    """AES S-box: GF(2⁸) da affin transformatsiya."""
    _hints = [
        "AES S-box affin transformatsiyasi: b = A*x + c (mod 2, GF(2⁸) da).",
        "Soddalashtirish uchun faqat bit matritsani @ orqali hisoblang.",
    ]
    _solution = "b = (A_bit @ x_bits + c_bits) % 2"

    def _do_check(self, b, A_bit, x_bits, c_bits):
        expected = (A_bit @ x_bits + c_bits) % 2
        if not np.allclose(b, expected):
            return "b = (A @ x + c) % 2 (GF(2) da affin transformatsiya)."
        return True


class C1_Q1(UzCheckProblem):
    """Diffie-Hellman kalitlar almashinuvi: A = g^a mod p."""
    _hints = [
        "A = pow(g, a, p) — Python da modular darajaga ko'tarish.",
        "Umumiy sir: K = pow(B, a, p) = pow(A, b, p) = g^(ab) mod p.",
    ]
    _solution = "A = pow(g, a, p); B = pow(g, b, p); K_a = pow(B, a, p); K_b = pow(A, b, p)"

    def _do_check(self, K_a, K_b, g, a, b, p):
        A = pow(g, a, p)
        B = pow(g, b, p)
        exp_Ka = pow(B, a, p)
        exp_Kb = pow(A, b, p)
        if K_a != exp_Ka:
            return f"K_a = B^a mod p = {exp_Ka}, siz {K_a} berdingiz."
        if K_b != exp_Kb:
            return f"K_b = A^b mod p = {exp_Kb}, siz {K_b} berdingiz."
        if K_a != K_b:
            return f"Umumiy sir mos kelmaydi: K_a={K_a}, K_b={K_b}."
        return True


class C2_Q1(ThoughtExperiment):
    """Chiziqli algebra kriptografiyada qanday ishlatiladi?"""
    _hints = [
        "Hill shifri, error-correcting codes, AES, lattice cryptography.",
    ]
    _solution = (
        "Kriptografiyada chiziqli algebra:\n\n"
        "1) Hill shifri: matrits ko'paytmasi (mod p) — klassik simmetrik shifrlash.\n"
        "2) Error-correcting codes (Hamming, Reed-Solomon):\n"
        "   parity-check matritsa H — xatoni aniqlash va tuzatish.\n"
        "3) AES blok shifri: AddRoundKey, MixColumns — GF(2⁸) da chiziqli operatsiyalar.\n"
        "4) Lattice asosidagi kriptografiya (post-kvant):\n"
        "   LWE (Learning With Errors): Ax + e = b (mod q) — kvant kompyuterga chidamli.\n"
        "5) Elliptik egri chiziqlar: nuqtalar qo'shilishi — chiziqli algebra moduli.\n"
        "Kelajak: kvant kompyuter RSA ni sindiradi → lattice kriptografiya zarurligi ortmoqda."
    )
