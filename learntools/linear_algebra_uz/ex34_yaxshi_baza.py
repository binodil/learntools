"""Hints and solutions — Dars 8.3: Yaxshi Bazani Qidirish."""
import numpy as np
from learntools.linear_algebra_uz.base import UzCheckProblem, ThoughtExperiment


class Q1(UzCheckProblem):
    """Ikki matritsa o'xshash (similar) ekanini tekshiring: B = P⁻¹AP."""
    _hints = [
        "O'xshash matritsalar: xususiy qiymatlar va iz (trace) bir xil.",
        "np.allclose(np.sort(np.real(np.linalg.eig(A)[0])), np.sort(np.real(np.linalg.eig(B)[0])))",
    ]
    _solution = (
        "eigs_A = np.sort(np.real(np.linalg.eig(A)[0]))\n"
        "eigs_B = np.sort(np.real(np.linalg.eig(B)[0]))\n"
        "similar = np.allclose(eigs_A, eigs_B) and np.isclose(np.trace(A), np.trace(B))"
    )

    def _do_check(self, similar, A, B):
        eA = np.sort(np.real(np.linalg.eig(A)[0]))
        eB = np.sort(np.real(np.linalg.eig(B)[0]))
        expected = (np.allclose(eA, eB, atol=1e-6) and
                    np.isclose(np.trace(A), np.trace(B), atol=1e-6))
        if similar != expected:
            return f"O'xshash? {expected}. Xususiy qiymatlar: A={eA}, B={eB}"
        return True


class Q2(UzCheckProblem):
    """Jordan normal shakl: blokli diagonal ko'rinishga keltiring."""
    _hints = [
        "Jordan shakl: xususiy qiymatlar bo'yicha bloklarga ajratish.",
        "Agar A diagonallashsa: Jordan shakl = diag(vals).",
        "vals, _ = np.linalg.eig(A); J = np.diag(vals)",
    ]
    _solution = "vals, vecs = np.linalg.eig(A); J = np.diag(vals)  # diagonallashadigan holda"

    def _do_check(self, J, A):
        vals = np.linalg.eig(A)[0]
        if not np.allclose(np.sort(np.real(np.diag(J))), np.sort(np.real(vals)), atol=1e-6):
            return "J ning diagonal elementlari A ning xususiy qiymatlari bo'lishi kerak."
        return True


class Q3(UzCheckProblem):
    """O'xshash matritsaning determinant va izini hisoblang."""
    _hints = [
        "O'xshash matritsalar: det(A) = det(B), tr(A) = tr(B).",
        "det = np.linalg.det(A), tr = np.trace(A)",
    ]
    _solution = "det_val = np.linalg.det(A); tr_val = np.trace(A)"

    def _do_check(self, det_val, tr_val, A):
        exp_det = np.linalg.det(A)
        exp_tr = np.trace(A)
        if not np.isclose(det_val, exp_det, rtol=1e-5):
            return f"det(A) = {exp_det:.6f}, siz {det_val:.6f} berdingiz."
        if not np.isclose(tr_val, exp_tr, rtol=1e-5):
            return f"tr(A) = {exp_tr:.6f}, siz {tr_val:.6f} berdingiz."
        return True


class Q4(UzCheckProblem):
    """Cayley-Hamilton teoremasi: p(A) = 0, bu erda p — xarakteristik polinom."""
    _hints = [
        "2x2 uchun: A² - tr(A)*A + det(A)*I = 0.",
        "n×n uchun: np.linalg.matrix_power va xarakteristik polinom coefficients.",
    ]
    _solution = (
        "# 2x2 holat\n"
        "p_A = np.linalg.matrix_power(A, 2) - np.trace(A)*A + np.linalg.det(A)*np.eye(2)\n"
        "cayley_hamilton = np.allclose(p_A, 0)"
    )

    def _do_check(self, cayley_hamilton, A):
        if A.shape == (2, 2):
            p_A = (np.linalg.matrix_power(A, 2)
                   - np.trace(A) * A
                   + np.linalg.det(A) * np.eye(2))
            expected = np.allclose(p_A, 0, atol=1e-8)
        else:
            coeffs = np.poly(A)
            p_A = sum(c * np.linalg.matrix_power(A, len(coeffs) - 1 - i)
                      for i, c in enumerate(coeffs))
            expected = np.allclose(p_A, 0, atol=1e-8)
        if cayley_hamilton != expected:
            return f"Cayley-Hamilton: p(A) = 0? {expected}. 2x2: A² - tr(A)·A + det(A)·I = 0."
        return True


class Q5(UzCheckProblem):
    """SVD bilan yaxshi baza: A ning singular vektorlari."""
    _hints = [
        "U, s, Vt = np.linalg.svd(A). V ustunlari — kirish uchun yaxshi baza, U — chiqish uchun.",
    ]
    _solution = "U, s, Vt = np.linalg.svd(A, full_matrices=False); V = Vt.T"

    def _do_check(self, V, A):
        _, _, Vt = np.linalg.svd(A, full_matrices=False)
        expected = Vt.T
        if V.shape != expected.shape:
            return f"V shakli {expected.shape} bo'lishi kerak."
        for i in range(V.shape[1]):
            if not (np.allclose(V[:, i], expected[:, i], atol=1e-6) or
                    np.allclose(V[:, i], -expected[:, i], atol=1e-6)):
                return f"{i+1}-ustun noto'g'ri. V = Vt.T (svd natijasi)."
        return True


class Q6(UzCheckProblem):
    """A^k ni tez hisoblash: diagonallashtirish orqali."""
    _hints = [
        "A = PΛP⁻¹ bo'lsa, A^k = PΛ^kP⁻¹.",
        "vals, P = np.linalg.eig(A); A_k = P @ np.diag(vals**k) @ np.linalg.inv(P)",
    ]
    _solution = (
        "vals, P = np.linalg.eig(A)\n"
        "A_k = np.real(P @ np.diag(vals**k) @ np.linalg.inv(P))"
    )

    def _do_check(self, A_k, A, k):
        expected = np.linalg.matrix_power(A, k)
        if not np.allclose(np.real(A_k), expected, atol=1e-5):
            return f"A^{k} = P Λ^{k} P⁻¹ formulasini tekshiring."
        return True


class C1_Q1(UzCheckProblem):
    """Google PageRank: A ning dominant xususiy vektori."""
    _hints = [
        "Power iteration: v ← Av / ||Av|| ni takrorlang (konvergensiya bo'lguncha).",
        "np.linalg.eig(A) dan λ_max ga mos vektorni tanlang.",
    ]
    _solution = (
        "v = np.ones(A.shape[0]) / A.shape[0]\n"
        "for _ in range(100):\n"
        "    v_new = A @ v\n"
        "    v = v_new / np.linalg.norm(v_new)\n"
        "pagerank = v / v.sum()"
    )

    def _do_check(self, pagerank, A):
        if pagerank.shape[0] != A.shape[0]:
            return f"pagerank uzunligi {A.shape[0]} bo'lishi kerak."
        if not np.isclose(pagerank.sum(), 1.0, atol=1e-4):
            return f"pagerank yig'indisi 1 bo'lishi kerak, siz {pagerank.sum():.4f} berdingiz."
        # Check it's approximately an eigenvector
        Av = A @ pagerank
        ratio = Av / pagerank
        if not np.allclose(ratio, ratio[0], rtol=0.01):
            return "pagerank A ning dominant xususiy vektori bo'lishi kerak."
        return True


class C2_Q1(ThoughtExperiment):
    """'Yaxshi baza' nima va uni qanday tanlash kerak?"""
    _hints = [
        "Xususiy vektorlar, singular vektorlar, Fourier bazasi — bular 'yaxshi bazalar'.",
        "Yaxshi baza transformatsiyani diagonal yoki sodda ko'rinishda ifodalaydi.",
    ]
    _solution = (
        "'Yaxshi baza' — muammoning tuzilishiga mos baza:\n\n"
        "1) Xususiy vektor bazasi: A → diagonal Λ. Hisob: A^k, e^A tez.\n"
        "   Foyda: ODE yechish, Markov zanjiri, PageRank.\n\n"
        "2) SVD bazasi (U, V): A → diagonal Σ (rectangular). Hisob: least squares, PCA.\n"
        "   Foyda: rasm siqish, ma'lumot tahlili.\n\n"
        "3) Fourier bazasi: konvolyutsiya → ko'paytma (DFT diagonallashtiradi).\n"
        "   Foyda: signal qayta ishlash, tezkor FFT algoritmi.\n\n"
        "4) Ortonormal baza: numerik barqarorlik. QR, SVD shu uchun afzal.\n\n"
        "Umumiy tamoyil: transformatsiyaning 'tabiiy yo'nalishlari' eng yaxshi baza."
    )
