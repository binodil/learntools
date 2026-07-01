"""Hints and solutions — Dars 12.4: Markov Zanjiri."""
import numpy as np
from learntools.linear_algebra_uz.base import UzCheckProblem, ThoughtExperiment


class Q1(UzCheckProblem):
    """O'tish matritsasini tekshiring: ustunlar yig'indisi = 1, barcha element >= 0."""
    _hints = ["np.allclose(P.sum(axis=0), 1) va np.all(P >= 0)."]
    _solution = "valid = np.allclose(P.sum(axis=0), 1) and np.all(P >= 0)"

    def _do_check(self, valid, P):
        expected = np.allclose(P.sum(axis=0), 1) and np.all(P >= 0)
        if valid != expected:
            return f"Stokastik? {expected}. Ustunlar yig'indisi: {P.sum(axis=0)}"
        return True


class Q2(UzCheckProblem):
    """k-qadam o'tish: P^k."""
    _hints = ["np.linalg.matrix_power(P, k)."]
    _solution = "Pk = np.linalg.matrix_power(P, k)"

    def _do_check(self, Pk, P, k):
        expected = np.linalg.matrix_power(P, k)
        if not np.allclose(Pk, expected, atol=1e-8):
            return "Pk = np.linalg.matrix_power(P, k)."
        return True


class Q3(UzCheckProblem):
    """Barqaror taqsimot: Pπ = π (xususiy vektor lambda=1 uchun)."""
    _hints = [
        "vals, vecs = np.linalg.eig(P). lambda=1 ga mos vektor — barqaror taqsimot.",
        "pi = vecs[:, argmin(|vals-1|)]; pi = |pi|/|pi|.sum()",
    ]
    _solution = (
        "vals, vecs = np.linalg.eig(P)\n"
        "idx = np.argmin(np.abs(vals - 1))\n"
        "pi = np.real(vecs[:, idx])\n"
        "pi = np.abs(pi) / np.abs(pi).sum()"
    )

    def _do_check(self, pi, P):
        if not np.isclose(pi.sum(), 1, rtol=1e-6):
            return f"pi yig'indisi 1 bo'lishi kerak: {pi.sum():.6f}"
        if not np.allclose(P @ pi, pi, atol=1e-6):
            return "Pπ = π bo'lishi kerak (barqarorlik sharti)."
        return True


class Q4(UzCheckProblem):
    """Absorbatsiya ehtimoli: absorbatsiya holatlariga yetish ehtimoli."""
    _hints = [
        "Q — o'tuvchi holatlar o'tish matritsasi. N = (I-Q)^{-1} — fundamental matritsa.",
        "Absorbatsiya ehtimoli: B = N @ R, bu erda R — o'tuvchidan absorbatsiyaga.",
    ]
    _solution = (
        "Q = P[transient_idx][:, transient_idx]\n"
        "R = P[transient_idx][:, absorbing_idx]\n"
        "N = np.linalg.inv(np.eye(len(transient_idx)) - Q)\n"
        "B = N @ R"
    )

    def _do_check(self, B, P, transient_idx, absorbing_idx):
        Q = P[np.ix_(transient_idx, transient_idx)]
        R = P[np.ix_(transient_idx, absorbing_idx)]
        N = np.linalg.inv(np.eye(len(transient_idx)) - Q)
        expected = N @ R
        if not np.allclose(B, expected, atol=1e-6):
            return "B = N @ R, N = (I-Q)^{-1} formulasini tekshiring."
        return True


class Q5(UzCheckProblem):
    """O'rtacha qaytish vaqti: m_i = 1/pi_i."""
    _hints = ["Barqaror taqsimot pi dan: m_i = 1/pi[i]."]
    _solution = (
        "vals, vecs = np.linalg.eig(P)\n"
        "pi = np.abs(vecs[:, np.argmin(np.abs(vals-1))])\n"
        "pi = pi / pi.sum()\n"
        "return_times = 1.0 / pi"
    )

    def _do_check(self, return_times, P):
        vals, vecs = np.linalg.eig(P)
        pi = np.abs(np.real(vecs[:, np.argmin(np.abs(vals - 1))]))
        pi = pi / pi.sum()
        expected = 1.0 / pi
        if not np.allclose(return_times, expected, rtol=1e-4):
            return f"Qaytish vaqtlari = 1/pi_i: {expected}"
        return True


class Q6(UzCheckProblem):
    """MCMC: Metropolis-Hastings bir qadami."""
    _hints = [
        "Taklif: x_new = x + epsilon * randn(). Accept: min(1, p(x_new)/p(x)).",
    ]
    _solution = (
        "x_new = x_current + epsilon * np.random.randn()\n"
        "alpha = min(1, log_prob(x_new) - log_prob(x_current))  # log-ehtimollikda\n"
        "if np.log(np.random.rand()) < alpha:\n"
        "    x_current = x_new"
    )

    def _do_check(self, accepted_ratio, n_total):
        if not (0.1 < accepted_ratio < 0.9):
            return f"Qabul qilish nisbati {accepted_ratio:.2f} — 0.1 va 0.9 orasida bo'lishi kerak."
        return True


class C1_Q1(UzCheckProblem):
    """Google PageRank: telekommunikatsiya matritsasi."""
    _hints = [
        "Damping: P_full = d * P + (1-d)/n * ones_matrix. pi = dominant eig vektor.",
    ]
    _solution = (
        "n = P.shape[0]; d = 0.85\n"
        "P_full = d * P + (1-d)/n * np.ones((n,n))\n"
        "vals, vecs = np.linalg.eig(P_full)\n"
        "pi = np.abs(np.real(vecs[:, np.argmin(np.abs(vals-1))]))\n"
        "pagerank = pi / pi.sum()"
    )

    def _do_check(self, pagerank, P):
        n = P.shape[0]; d = 0.85
        P_full = d * P + (1 - d) / n * np.ones((n, n))
        vals, vecs = np.linalg.eig(P_full)
        pi = np.abs(np.real(vecs[:, np.argmin(np.abs(vals - 1))]))
        expected = pi / pi.sum()
        if not np.isclose(pagerank.sum(), 1, rtol=1e-4):
            return f"PageRank yig'indisi 1 bo'lishi kerak."
        if not np.allclose(np.sort(pagerank), np.sort(expected), rtol=0.01):
            return "Damping = 0.85 bilan PageRank formulasini tekshiring."
        return True


class C2_Q1(ThoughtExperiment):
    """Markov zanjiri va chiziqli algebra: qanday bog'liq?"""
    _hints = [
        "Barqaror taqsimot = lambda=1 xususiy vektor. Konvergentsiya = |lambda_2| < 1.",
    ]
    _solution = (
        "Markov zanjiri va chiziqli algebra:\n\n"
        "1) O'tish matritsasi P — stokastik matritsa (ustunlar/qatorlar yig'indisi = 1).\n"
        "2) Barqaror taqsimot π: Pπ = π → lambda=1 xususiy vektor.\n"
        "3) Konvergentsiya tezligi: |lambda_2| qanchalik kichik → shunchalik tez.\n"
        "4) PageRank: damping factor bilan Markov zanjiri → 4+ mlrd sahifa uchun.\n"
        "5) MCMC: Bayes inference uchun Markov zanjirlari → murakkab taqsimotdan namuna olish.\n"
        "6) Absorb. ehtimoli: (I-Q)^{-1} — matritsa inversiyasi.\n"
        "Asosiy vosita: xususiy qiymatlar va xususiy vektorlar."
    )
