"""Hints and solutions — Dars 1.2: Uzunlik va Skalyar Ko'paytma."""
import numpy as np
from learntools.core import EqualityCheckProblem, ThoughtExperiment


class Q1(EqualityCheckProblem):
    """Vektor uzunligini hisoblang."""
    _hints = [
        "||v|| = sqrt(v1^2 + v2^2 + ... + vn^2)",
        "np.linalg.norm(v) yoki np.sqrt(np.dot(v, v)) ishlatishingiz mumkin.",
    ]
    _solution = "norm_v = np.linalg.norm(v)"

    def _do_check(self, result, v):
        expected = np.linalg.norm(v)
        if not np.isclose(result, expected):
            return f"Kutilgan: {expected:.4f}, siz berdingiz: {result}"
        return True


class Q2(EqualityCheckProblem):
    """Skalyar ko'paytma (dot product)."""
    _hints = [
        "u · v = u1*v1 + u2*v2 + ... + un*vn",
        "np.dot(u, v) yoki u @ v ishlatishingiz mumkin.",
    ]
    _solution = "result = np.dot(u, v)  # yoki u @ v"

    def _do_check(self, result, u, v):
        expected = np.dot(u, v)
        if not np.isclose(result, expected):
            return f"Kutilgan: {expected}, siz berdingiz: {result}"
        return True


class Q3(EqualityCheckProblem):
    """Vektorni normalashtiring (unit vector)."""
    _hints = [
        "Unit vektor: û = v / ||v||",
        "Avval normani toping, keyin bo'ling.",
    ]
    _solution = "u_hat = v / np.linalg.norm(v)"

    def _do_check(self, result, v):
        expected = v / np.linalg.norm(v)
        if not np.allclose(result, expected):
            return f"Kutilgan: {expected}, siz berdingiz: {result}"
        if not np.isclose(np.linalg.norm(result), 1.0):
            return "Unit vektorning uzunligi 1 bo'lishi kerak!"
        return True


class Q4(EqualityCheckProblem):
    """Ikki vektor orasidagi burchakni toping."""
    _hints = [
        "cos(θ) = (u · v) / (||u|| * ||v||)",
        "np.arccos() burchakni radianda beradi. Darajaga: np.degrees(...)",
    ]
    _solution = (
        "cos_theta = np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))\n"
        "angle_deg = np.degrees(np.arccos(np.clip(cos_theta, -1, 1)))"
    )

    def _do_check(self, angle_deg, u, v):
        cos_theta = np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))
        expected = np.degrees(np.arccos(np.clip(cos_theta, -1, 1)))
        if not np.isclose(angle_deg, expected, atol=0.01):
            return f"Kutilgan: {expected:.2f}°, siz berdingiz: {angle_deg:.2f}°"
        return True


class Q5(EqualityCheckProblem):
    """Perpendikulyarlikni tekshiring."""
    _hints = [
        "Ikkita vektor perpendikulyar bo'lsa, ularning skalyar ko'paytmasi 0 ga teng.",
        "np.isclose(np.dot(u, v), 0) ishlatishingiz mumkin.",
    ]
    _solution = "is_perp = np.isclose(np.dot(u, v), 0)"

    def _do_check(self, is_perp, u, v):
        expected = np.isclose(np.dot(u, v), 0)
        if is_perp != expected:
            return f"Kutilgan: {expected}, siz berdingiz: {is_perp}"
        return True


class Q6(EqualityCheckProblem):
    """Proyeksiya hisoblang."""
    _hints = [
        "v ni a yo'nalishiga proyeksiyasi: proj = (v · a / ||a||^2) * a",
        "Bu a yo'nalishidagi eng yaqin nuqtani beradi.",
    ]
    _solution = "proj = (np.dot(v, a) / np.dot(a, a)) * a"

    def _do_check(self, proj, v, a):
        expected = (np.dot(v, a) / np.dot(a, a)) * a
        if not np.allclose(proj, expected):
            return f"Kutilgan: {expected}, siz berdingiz: {proj}"
        return True


# Challenge 1
class C1_Q1(EqualityCheckProblem):
    """Cosine o'xshashligi."""
    _hints = [
        "Cosine similarity = (u · v) / (||u|| * ||v||) — bu -1 dan 1 gacha.",
        "1 — bir xil yo'nalish, 0 — perpendikulyar, -1 — qarama-qarshi.",
    ]
    _solution = "sim = np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))"

    def _do_check(self, sim, u, v):
        expected = np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))
        if not np.isclose(sim, expected):
            return f"Kutilgan: {expected:.4f}, siz berdingiz: {sim:.4f}"
        return True


# Challenge 2
class C2_Q1(ThoughtExperiment):
    """Gram-Schmidt: ikki vektordan ortogonal asos yasang."""
    _hints = [
        "Birinchi vektor o'zgarmaydi: u1 = v1.",
        "Ikkinchi: u2 = v2 - proj(v2, u1) — v2 dan u1 yo'nalishidagi qismini olib tashlang.",
        "Keyin normalashtiring: e1 = u1/||u1||, e2 = u2/||u2||.",
    ]
    _solution = (
        "u1 = v1.copy()\n"
        "u2 = v2 - (np.dot(v2, u1) / np.dot(u1, u1)) * u1\n"
        "e1 = u1 / np.linalg.norm(u1)\n"
        "e2 = u2 / np.linalg.norm(u2)\n"
        "# Tekshirish: np.isclose(np.dot(e1, e2), 0) → True"
    )
