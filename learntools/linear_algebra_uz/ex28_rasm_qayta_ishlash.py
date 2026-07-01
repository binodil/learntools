"""Hints and solutions — Dars 7.1: Rasmlarni Chiziqli Algebra bilan Qayta Ishlash."""
import numpy as np
from learntools.linear_algebra_uz.base import UzCheckProblem, ThoughtExperiment


class Q1(UzCheckProblem):
    """SVD ni hisoblang."""
    _hints = [
        "np.linalg.svd(A, full_matrices=False) U, s, Vt qaytaradi.",
        "Singulyar qiymatlar s vektorida kamayib boradi.",
    ]
    _solution = "U, s, Vt = np.linalg.svd(A, full_matrices=False)\nresult = s"

    def _do_check(self, result, A):
        expected = np.linalg.svd(A, compute_uv=False)
        if not np.allclose(np.sort(result)[::-1], expected, atol=1e-6):
            return f"Singulyar qiymatlar mos emas. Kutilgan: {expected}"
        return True


class Q2(UzCheckProblem):
    """Rang-k taxminlash A_k ni quring."""
    _hints = [
        "A_k = U[:,:k] @ diag(s[:k]) @ Vt[:k,:].",
        "Faqat birinchi k ta singulyar komponentni saqlang.",
    ]
    _solution = "A_k = U[:, :k] @ np.diag(s[:k]) @ Vt[:k, :]"

    def _do_check(self, A_k, U, s, Vt, k):
        expected = U[:, :k] @ np.diag(s[:k]) @ Vt[:k, :]
        if not np.allclose(A_k, expected, atol=1e-6):
            return "A_k noto'g'ri. U,s,Vt ni birinchi k bilan kesib oling."
        return True


class Q3(UzCheckProblem):
    """Eckart-Young xatosini hisoblang."""
    _hints = [
        "Frobenius xatosi = sqrt(qolgan sigma kvadratlari yig'indisi).",
        "err = sqrt(sum(s[k:]**2)).",
    ]
    _solution = "err = np.sqrt(np.sum(s[k:]**2))"

    def _do_check(self, err, s, k):
        expected = np.sqrt(np.sum(s[k:]**2))
        if not np.isclose(err, expected, atol=1e-6):
            return f"Kutilgan: {expected}, siz berdingiz: {err}"
        return True


class Q4(UzCheckProblem):
    """Saqlangan energiya nisbati."""
    _hints = [
        "Energiya = sigma kvadratlari.",
        "nisbat = sum(s[:k]**2) / sum(s**2).",
    ]
    _solution = "ratio = np.sum(s[:k]**2) / np.sum(s**2)"

    def _do_check(self, ratio, s, k):
        expected = np.sum(s[:k]**2) / np.sum(s**2)
        if not np.isclose(ratio, expected, atol=1e-6):
            return f"Kutilgan: {expected}, siz berdingiz: {ratio}"
        return True


class Q5(UzCheckProblem):
    """Matritsa rangi = noldan farqli singulyar qiymatlar soni."""
    _hints = [
        "rank = (s > tolerance).sum().",
        "Juda kichik singulyar qiymatlarni nol deb hisoblang.",
    ]
    _solution = "rank = int(np.sum(s > 1e-10))"

    def _do_check(self, rank, A):
        expected = np.linalg.matrix_rank(A)
        if rank != expected:
            return f"Kutilgan rang: {expected}, siz berdingiz: {rank}"
        return True


class Q6(UzCheckProblem):
    """Siqish nisbatini hisoblang (saqlangan sonlar / asl)."""
    _hints = [
        "Rang-k uchun saqlanadi: k*(m + n + 1) son.",
        "Asl matritsa: m*n son.",
    ]
    _solution = "compression = (k*(m + n + 1)) / (m * n)"

    def _do_check(self, compression, m, n, k):
        expected = (k*(m + n + 1)) / (m * n)
        if not np.isclose(compression, expected, atol=1e-6):
            return f"Kutilgan: {expected}, siz berdingiz: {compression}"
        return True


# Challenge 1
class C1_Q1(UzCheckProblem):
    """90% energiya uchun zarur k ni toping."""
    _hints = [
        "Kumulyativ energiya nisbatini hisoblang.",
        "np.cumsum(s**2)/np.sum(s**2) >= 0.9 birinchi indeksini toping (+1).",
    ]
    _solution = (
        "cum = np.cumsum(s**2) / np.sum(s**2)\n"
        "k = int(np.argmax(cum >= 0.9)) + 1"
    )

    def _do_check(self, k, s):
        cum = np.cumsum(s**2) / np.sum(s**2)
        expected = int(np.argmax(cum >= 0.9)) + 1
        if k != expected:
            return f"Kutilgan k: {expected}, siz berdingiz: {k}"
        return True


# Challenge 2
class C2_Q1(ThoughtExperiment):
    """Qaysi rasmlar yaxshi siqiladi?"""
    _hints = [
        "Singulyar qiymatlar tez pasayadigan rasmlar yaxshi siqiladi.",
        "Tekis fon, takrorlanuvchi naqsh -> kam rang. Shovqin -> yuqori rang.",
    ]
    _solution = (
        "Singulyar qiymatlari tez pasayadigan (low-rank) rasmlar yaxshi siqiladi:\n"
        "tekis fon, gradient, takrorlanuvchi naqshlar. Tasodifiy shovqin esa\n"
        "to'liq rangli (sigma'lar sekin pasayadi) -> yomon siqiladi.\n"
        "Eckart-Young: A_k xatosi = sqrt(sum sigma[k:]^2), tez pasayish -> kichik xato."
    )
