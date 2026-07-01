"""Hints and solutions — Dars 10.5: Furye Qatori."""
import numpy as np
from learntools.linear_algebra_uz.base import UzCheckProblem, ThoughtExperiment


class Q1(UzCheckProblem):
    """Furye koeffitsiyentlarini hisoblang: a_k = (2/T) ∫ f(x) cos(2πkx/T) dx."""
    _hints = [
        "Diskret taqrib: a_k ≈ (2/N) sum(f[n] * cos(2*pi*k*n/N)) for n in range(N).",
    ]
    _solution = (
        "N = len(x)\n"
        "a_k = (2/N) * np.sum(f * np.cos(2*np.pi*k*np.arange(N)/N))"
    )

    def _do_check(self, a_k, f, k):
        N = len(f)
        expected = (2 / N) * np.sum(f * np.cos(2 * np.pi * k * np.arange(N) / N))
        if not np.isclose(a_k, expected, rtol=1e-5):
            return f"a_{k} = {expected:.6f}, siz {a_k:.6f} berdingiz."
        return True


class Q2(UzCheckProblem):
    """Furye sinus koeffitsiyentlarini hisoblang: b_k."""
    _hints = [
        "b_k ≈ (2/N) sum(f[n] * sin(2*pi*k*n/N)) for n in range(N).",
    ]
    _solution = "b_k = (2/N) * np.sum(f * np.sin(2*np.pi*k*np.arange(len(f))/len(f)))"

    def _do_check(self, b_k, f, k):
        N = len(f)
        expected = (2 / N) * np.sum(f * np.sin(2 * np.pi * k * np.arange(N) / N))
        if not np.isclose(b_k, expected, rtol=1e-5):
            return f"b_{k} = {expected:.6f}, siz {b_k:.6f} berdingiz."
        return True


class Q3(UzCheckProblem):
    """Furye qatorini birinchi K a'zo orqali tiklang."""
    _hints = [
        "f_approx = a_0/2 + sum(a_k*cos(2*pi*k*x/T) + b_k*sin(2*pi*k*x/T) for k=1..K).",
    ]
    _solution = (
        "N = len(f)\n"
        "n = np.arange(N)\n"
        "f_approx = np.zeros(N)\n"
        "for k in range(1, K+1):\n"
        "    a_k = (2/N)*np.sum(f*np.cos(2*np.pi*k*n/N))\n"
        "    b_k = (2/N)*np.sum(f*np.sin(2*np.pi*k*n/N))\n"
        "    f_approx += a_k*np.cos(2*np.pi*k*n/N) + b_k*np.sin(2*np.pi*k*n/N)"
    )

    def _do_check(self, f_approx, f, K):
        N = len(f)
        n = np.arange(N)
        expected = np.zeros(N)
        for k in range(1, K + 1):
            a_k = (2 / N) * np.sum(f * np.cos(2 * np.pi * k * n / N))
            b_k = (2 / N) * np.sum(f * np.sin(2 * np.pi * k * n / N))
            expected += a_k * np.cos(2 * np.pi * k * n / N) + b_k * np.sin(2 * np.pi * k * n / N)
        if not np.allclose(f_approx, expected, atol=1e-6):
            return f"K={K} a'zo bilan tiklash xato. Formulani tekshiring."
        return True


class Q4(UzCheckProblem):
    """Ortogonallik: cos va sin funksiyalari ortogonal ekanini tekshiring."""
    _hints = [
        "(1/pi) integral cos(mx)*cos(nx) dx = 0 agar m!=n. Diskret: sum(cos_m * cos_n) / N.",
    ]
    _solution = (
        "N = 1000\n"
        "n = np.linspace(0, 2*np.pi, N, endpoint=False)\n"
        "cos_m = np.cos(m * n)\n"
        "cos_n2 = np.cos(n2 * n)\n"
        "inner = np.sum(cos_m * cos_n2) / N"
    )

    def _do_check(self, inner, m, n2):
        N = 1000
        t = np.linspace(0, 2 * np.pi, N, endpoint=False)
        cos_m = np.cos(m * t)
        cos_n = np.cos(n2 * t)
        expected = np.sum(cos_m * cos_n) / N
        if not np.isclose(inner, expected, atol=1e-4):
            return f"Ichki ko'paytma: {expected:.6f}, siz {inner:.6f} berdingiz."
        return True


class Q5(UzCheckProblem):
    """Gibb hodisasi: kvadrat to'lqin uchun Furye qatorini K=50 bilan hisoblang."""
    _hints = [
        "Kvadrat to'lqin: f(x) = sign(sin(x)). Furye: faqat toq harmonikalar.",
        "b_k = 4/(pi*k) agar k toq, 0 agar k juft.",
    ]
    _solution = (
        "N = 512\n"
        "x = np.linspace(0, 2*np.pi, N, endpoint=False)\n"
        "f_sq = np.sign(np.sin(x))\n"
        "K = 50\n"
        "approx = sum(4/(np.pi*k)*np.sin(k*x) for k in range(1, 2*K, 2))"
    )

    def _do_check(self, approx, K):
        N = 512
        x = np.linspace(0, 2 * np.pi, N, endpoint=False)
        expected = sum(4 / (np.pi * k) * np.sin(k * x) for k in range(1, 2 * K, 2))
        if approx.shape != expected.shape:
            return f"approx uzunligi {N} bo'lishi kerak."
        if not np.allclose(approx, expected, atol=1e-6):
            return "b_k = 4/(pi*k) (toq k), sum for k=1,3,5,...,2K-1."
        return True


class Q6(UzCheckProblem):
    """Furye qatori yordamida tugunlararo interpolyatsiya."""
    _hints = [
        "Furye interpolyatsiya: fft → koeffitsiyentlar → yangi nuqtalarda ifft.",
    ]
    _solution = (
        "X = np.fft.fft(y)\n"
        "# zero-padding uchun yangi N\n"
        "X_pad = np.concatenate([X[:N//2], np.zeros(N_new - N), X[N//2:]])\n"
        "y_interp = np.real(np.fft.ifft(X_pad)) * (N_new / N)"
    )

    def _do_check(self, y_interp, y, N_new):
        N = len(y)
        X = np.fft.fft(y)
        X_pad = np.concatenate([X[:N // 2], np.zeros(N_new - N), X[N // 2:]])
        expected = np.real(np.fft.ifft(X_pad)) * (N_new / N)
        if len(y_interp) != N_new:
            return f"y_interp uzunligi {N_new} bo'lishi kerak, siz {len(y_interp)} berdingiz."
        if not np.allclose(y_interp, expected, atol=1e-6):
            return "Zero-padding va ifft bilan interpolyatsiya formulasini tekshiring."
        return True


class C1_Q1(UzCheckProblem):
    """MP3 audio siqish: quloq eshitmaydigan chastotalarni o'chiring."""
    _hints = [
        "FFT → |X[k]| < threshold bo'lganlarni nolga → IFFT.",
        "Siqish nisbati: (nol bo'lmagan koeffitsiyentlar) / N.",
    ]
    _solution = (
        "X = np.fft.fft(audio)\n"
        "threshold = 0.01 * np.max(np.abs(X))\n"
        "X_compressed = X.copy()\n"
        "X_compressed[np.abs(X) < threshold] = 0\n"
        "ratio = np.sum(X_compressed != 0) / len(X)"
    )

    def _do_check(self, ratio, audio):
        X = np.fft.fft(audio)
        threshold = 0.01 * np.max(np.abs(X))
        X_c = X.copy()
        X_c[np.abs(X) < threshold] = 0
        expected = np.sum(X_c != 0) / len(X)
        if not np.isclose(ratio, expected, rtol=0.01):
            return f"Siqish nisbati: {expected:.4f}, siz {ratio:.4f} berdingiz."
        return True


class C2_Q1(ThoughtExperiment):
    """Chiziqli algebra Furye tahlilini qanday birlashtiradi?"""
    _hints = [
        "Furye bazasi — L² funksiyalar fazosining ortonormal bazasi.",
        "DFT matritsasi unitariy, Parseval — norma saqlanishi.",
    ]
    _solution = (
        "Furye tahlili = chiziqli algebra (sonsiz o'lchamli):\n\n"
        "1) Baza: {1, cos(x), sin(x), cos(2x), sin(2x), ...} — L²[0,2π] da ortonormal.\n"
        "2) Koeffitsiyentlar: a_k = ⟨f, cos(kx)⟩ — ichki ko'paytma (proyeksiya).\n"
        "3) DFT matritsasi F — unitariy: F*ᵀF = I (Parseval = norma saqlanishi).\n"
        "4) FFT = F ning tez hisoblash algoritmi: O(N²) → O(N log N).\n"
        "5) Qo'llanish: audio/video siqish (MP3, JPEG), signal filtri, PDE yechish."
    )
