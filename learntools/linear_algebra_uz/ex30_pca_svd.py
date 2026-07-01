"""Hints and solutions — Dars 7.3: PCA by the SVD."""
import numpy as np
from learntools.linear_algebra_uz.base import UzCheckProblem, ThoughtExperiment


class Q1(UzCheckProblem):
    """Ma'lumotlarni markazlashtiring (mean-center)."""
    _hints = [
        "Har bir xususiyatning o'rtachasini ayiring: X_centered = X - X.mean(axis=0).",
    ]
    _solution = "X_c = X - X.mean(axis=0)"

    def _do_check(self, X_c, X):
        expected = X - X.mean(axis=0)
        if not np.allclose(X_c, expected, atol=1e-8):
            return "X_c = X - X.mean(axis=0) formulasini tekshiring."
        if not np.allclose(X_c.mean(axis=0), 0, atol=1e-8):
            return f"Markazlashtirilgan ma'lumot o'rtachasi 0 bo'lishi kerak. Got: {X_c.mean(axis=0)}"
        return True


class Q2(UzCheckProblem):
    """SVD orqali asosiy komponentlarni toping."""
    _hints = [
        "U, s, Vt = np.linalg.svd(X_c, full_matrices=False). Asosiy yo'nalishlar — Vt qatorlari.",
    ]
    _solution = "U, s, Vt = np.linalg.svd(X_c, full_matrices=False); components = Vt"

    def _do_check(self, components, X_c):
        _, _, Vt = np.linalg.svd(X_c, full_matrices=False)
        if components.shape != Vt.shape:
            return f"Shaklni tekshiring: kutilgan {Vt.shape}, siz {components.shape} berdingiz."
        # Check that rows span same space (might differ in sign)
        for i in range(min(3, len(Vt))):
            if not (np.allclose(components[i], Vt[i], atol=1e-6) or
                    np.allclose(components[i], -Vt[i], atol=1e-6)):
                return f"{i+1}-komponent noto'g'ri. U, s, Vt = svd(X_c) dan foydalaning."
        return True


class Q3(UzCheckProblem):
    """Explained variance (tushuntirilgan dispersiya) ni hisoblang."""
    _hints = [
        "Dispersiya: s² / sum(s²). U, s, Vt = svd(X_c) dan s — singular qiymatlar.",
    ]
    _solution = "U, s, Vt = np.linalg.svd(X_c, full_matrices=False); var_ratio = s**2 / np.sum(s**2)"

    def _do_check(self, var_ratio, X_c):
        _, s, _ = np.linalg.svd(X_c, full_matrices=False)
        expected = s ** 2 / np.sum(s ** 2)
        if not np.allclose(var_ratio, expected, atol=1e-6):
            return f"var_ratio = s² / sum(s²). Kutilgan: {expected}"
        return True


class Q4(UzCheckProblem):
    """Ma'lumotlarni k-ta asosiy komponentga proyeksiyalang."""
    _hints = [
        "X_k = X_c @ Vt[:k].T — k ta asosiy komponent sifatida.",
    ]
    _solution = "U, s, Vt = np.linalg.svd(X_c, full_matrices=False); X_k = X_c @ Vt[:k].T"

    def _do_check(self, X_k, X_c, k):
        _, _, Vt = np.linalg.svd(X_c, full_matrices=False)
        expected = X_c @ Vt[:k].T
        if X_k.shape != expected.shape:
            return f"Shaklni tekshiring: kutilgan {expected.shape}, siz {X_k.shape} berdingiz."
        # Check reconstruction error is minimal
        X_rec = X_k @ Vt[:k]
        alt_rec = expected @ Vt[:k]
        if not np.allclose(X_rec, alt_rec, atol=1e-5):
            return "Proyeksiya noto'g'ri. X_k = X_c @ Vt[:k].T"
        return True


class Q5(UzCheckProblem):
    """k-komponent bilan qayta tiklash va xatoni hisoblang."""
    _hints = [
        "X_rec = X_k @ Vt[:k]. Xato: ||X_c - X_rec||_F.",
    ]
    _solution = (
        "U, s, Vt = np.linalg.svd(X_c, full_matrices=False)\n"
        "X_k = X_c @ Vt[:k].T\n"
        "X_rec = X_k @ Vt[:k]\n"
        "error = np.linalg.norm(X_c - X_rec, 'fro')"
    )

    def _do_check(self, error, X_c, k):
        _, s, Vt = np.linalg.svd(X_c, full_matrices=False)
        X_k = X_c @ Vt[:k].T
        X_rec = X_k @ Vt[:k]
        expected = np.linalg.norm(X_c - X_rec, 'fro')
        if not np.isclose(error, expected, rtol=1e-4):
            return f"Kutilgan xato: {expected:.6f}, siz {error:.6f} berdingiz."
        return True


class Q6(UzCheckProblem):
    """Nechta komponent kerakligini aniqlang (95% dispersiya)."""
    _hints = [
        "cumsum(var_ratio) >= 0.95 bo'ladigan birinchi k ni toping.",
        "np.searchsorted(np.cumsum(var_ratio), 0.95) + 1",
    ]
    _solution = (
        "_, s, _ = np.linalg.svd(X_c, full_matrices=False)\n"
        "var_ratio = s**2 / np.sum(s**2)\n"
        "k = np.searchsorted(np.cumsum(var_ratio), 0.95) + 1"
    )

    def _do_check(self, k, X_c):
        _, s, _ = np.linalg.svd(X_c, full_matrices=False)
        var_ratio = s ** 2 / np.sum(s ** 2)
        expected = int(np.searchsorted(np.cumsum(var_ratio), 0.95)) + 1
        if k != expected:
            return f"95% dispersiya uchun {expected} komponent kerak, siz {k} berdingiz."
        return True


class C1_Q1(UzCheckProblem):
    """PCA bilan rasm siqishini simulyatsiya qiling."""
    _hints = [
        "20×20 tasodifiy matritsa yarating, SVD dan k=5 komponent bilan siqib, xatoni hisoblang.",
    ]
    _solution = (
        "np.random.seed(42)\n"
        "A = np.random.randn(20, 20)\n"
        "U, s, Vt = np.linalg.svd(A)\n"
        "k = 5\n"
        "A_k = U[:, :k] @ np.diag(s[:k]) @ Vt[:k, :]\n"
        "compression_ratio = k * (20 + 20 + 1) / (20 * 20)"
    )

    def _do_check(self, compression_ratio, k, m, n):
        expected = k * (m + n + 1) / (m * n)
        if not np.isclose(compression_ratio, expected, rtol=0.01):
            return f"Kutilgan nisbat: {expected:.4f}, siz {compression_ratio:.4f} berdingiz."
        return True


class C2_Q1(ThoughtExperiment):
    """PCA nima uchun ma'lumotlar tahlilida keng qo'llaniladi?"""
    _hints = [
        "PCA: o'lchamni kamaytirish, shovqinni yo'qotish, vizualizatsiya.",
        "SVD bilan PCA bir xil natijani beradi — qaysi hollarda farq qiladi?",
    ]
    _solution = (
        "PCA asosiy qo'llanishlari:\n"
        "1) O'lchamni kamaytirish: 1000 xususiyat → 10 komponent (95% axborot saqlanadi).\n"
        "2) Shovqinni yo'qotish: kichik singular qiymatlar shovqinga mos keladi.\n"
        "3) Vizualizatsiya: 2D/3D ga tushirish.\n"
        "4) Ko'p kollinearlikni bartaraf etish (regressiyadan oldin).\n"
        "SVD vs kovariatsiya matritsasi eig: SVD numerik jihatdan barqarorroq, "
        "ayniqsa n >> p hollarda. np.linalg.svd — sanoat standart."
    )
