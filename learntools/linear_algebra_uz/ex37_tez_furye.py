"""Hints and solutions — Dars 9.3: Tez Furye Transformatsiyasi (FFT)."""
import numpy as np
from learntools.linear_algebra_uz.base import UzCheckProblem, ThoughtExperiment


class Q1(UzCheckProblem):
    """DFT ni to'g'ridan-to'g'ri matritsa ko'paytmasi orqali hisoblang."""
    _hints = [
        "DFT[k] = Σ x[n] e^{-2πikn/N}. Matritsa: F[k,n] = e^{-2πikn/N}.",
        "j,k=np.mgrid[0:N,0:N]; F=np.exp(-2j*np.pi*j*k/N); X_dft = F @ x",
    ]
    _solution = (
        "N = len(x)\n"
        "j, k = np.mgrid[0:N, 0:N]\n"
        "F = np.exp(-2j * np.pi * j * k / N)\n"
        "X_dft = F @ x"
    )

    def _do_check(self, X_dft, x):
        expected = np.fft.fft(x)
        if not np.allclose(X_dft, expected, atol=1e-6):
            return "DFT matritsasi F[k,n] = exp(-2πikn/N) dan foydalaning."
        return True


class Q2(UzCheckProblem):
    """np.fft.fft bilan DFT ni hisoblang va tezlikni solishtiring."""
    _hints = ["X = np.fft.fft(x) — FFT yordamida DFT."]
    _solution = "X = np.fft.fft(x)"

    def _do_check(self, X, x):
        expected = np.fft.fft(x)
        if not np.allclose(X, expected, atol=1e-8):
            return "X = np.fft.fft(x) dan foydalaning."
        return True


class Q3(UzCheckProblem):
    """Teskari FFT: x = IFFT(X)."""
    _hints = ["np.fft.ifft(X) — teskari FFT."]
    _solution = "x_rec = np.fft.ifft(X)"

    def _do_check(self, x_rec, x):
        X = np.fft.fft(x)
        expected = np.fft.ifft(X)
        if not np.allclose(x_rec, expected, atol=1e-8):
            return "np.fft.ifft(X) dan foydalaning."
        if not np.allclose(x_rec, x, atol=1e-8):
            return "IFFT(FFT(x)) = x bo'lishi kerak."
        return True


class Q4(UzCheckProblem):
    """Chastota spektrini hisoblang va dominant chastotani toping."""
    _hints = [
        "freqs = np.fft.fftfreq(N, d=1/fs). Dominant: freqs[np.argmax(np.abs(X))].",
    ]
    _solution = (
        "X = np.fft.fft(x)\n"
        "freqs = np.fft.fftfreq(len(x), d=1/fs)\n"
        "dom_freq = freqs[np.argmax(np.abs(X[:len(X)//2]))]"
    )

    def _do_check(self, dom_freq, x, fs):
        X = np.fft.fft(x)
        freqs = np.fft.fftfreq(len(x), d=1 / fs)
        expected = freqs[np.argmax(np.abs(X[:len(X) // 2]))]
        if not np.isclose(dom_freq, expected, rtol=0.01):
            return f"Dominant chastota: {expected:.2f} Hz, siz {dom_freq:.2f} berdingiz."
        return True


class Q5(UzCheckProblem):
    """FFT bilan shovqinli signalni filtrlang (past chastotalar)."""
    _hints = [
        "X = fft(x). Yuqori chastotalarni nolga tenglang (|freq| > cutoff). Keyin ifft.",
    ]
    _solution = (
        "X = np.fft.fft(x)\n"
        "freqs = np.fft.fftfreq(len(x), d=1/fs)\n"
        "X_filtered = X.copy()\n"
        "X_filtered[np.abs(freqs) > cutoff] = 0\n"
        "x_clean = np.real(np.fft.ifft(X_filtered))"
    )

    def _do_check(self, x_clean, x, fs, cutoff):
        X = np.fft.fft(x)
        freqs = np.fft.fftfreq(len(x), d=1 / fs)
        X_f = X.copy()
        X_f[np.abs(freqs) > cutoff] = 0
        expected = np.real(np.fft.ifft(X_f))
        if not np.allclose(x_clean, expected, atol=1e-6):
            return "Yuqori chastotalarni nolga tenglang, keyin np.real(np.fft.ifft(X_filtered))."
        return True


class Q6(UzCheckProblem):
    """Parseval teoremasi: ||x||² = (1/N)||X||²."""
    _hints = [
        "Parseval: sum(|x[n]|²) = (1/N) sum(|X[k]|²).",
        "np.sum(np.abs(x)**2) va np.sum(np.abs(X)**2)/len(x)",
    ]
    _solution = (
        "X = np.fft.fft(x)\n"
        "parseval_ok = np.isclose(np.sum(np.abs(x)**2), np.sum(np.abs(X)**2)/len(x))"
    )

    def _do_check(self, parseval_ok, x):
        X = np.fft.fft(x)
        lhs = np.sum(np.abs(x) ** 2)
        rhs = np.sum(np.abs(X) ** 2) / len(x)
        expected = np.isclose(lhs, rhs, rtol=1e-6)
        if parseval_ok != expected:
            return f"||x||² = {lhs:.4f}, ||X||²/N = {rhs:.4f}. Teng? {expected}"
        return True


class C1_Q1(UzCheckProblem):
    """2D FFT bilan rasm chastota tahlili."""
    _hints = [
        "np.fft.fft2(img) — 2D FFT. np.fft.fftshift — markazlashtirish.",
        "Spektr: np.log(1 + np.abs(np.fft.fftshift(F2))) — vizualizatsiya uchun.",
    ]
    _solution = (
        "F2 = np.fft.fft2(img)\n"
        "F2_shifted = np.fft.fftshift(F2)\n"
        "magnitude = np.abs(F2_shifted)"
    )

    def _do_check(self, magnitude, img):
        F2 = np.fft.fft2(img)
        F2_shifted = np.fft.fftshift(F2)
        expected = np.abs(F2_shifted)
        if not np.allclose(magnitude, expected, atol=1e-6):
            return "magnitude = np.abs(np.fft.fftshift(np.fft.fft2(img))) formulasini tekshiring."
        return True


class C2_Q1(ThoughtExperiment):
    """FFT nima uchun O(N log N) - O(N²) ga nisbatan?"""
    _hints = [
        "Cooley-Tukey algoritmi: N-nuqta DFT = ikki N/2-nuqta DFT.",
        "T(N) = 2T(N/2) + O(N) → T(N) = O(N log N).",
    ]
    _solution = (
        "To'g'ridan-to'g'ri DFT: har bir X[k] uchun N ko'paytma → jami N² amal.\n\n"
        "FFT (Cooley-Tukey, 1965):\n"
        "N-nuqta DFT = toq va juft indekslar uchun ikki N/2-nuqtali DFT + O(N) kombinatsiya.\n"
        "T(N) = 2T(N/2) + N → T(N) = O(N log N).\n\n"
        "Misol: N=10⁶ uchun:\n"
        "  DFT: 10¹² amal ≈ 1000 sekund\n"
        "  FFT: 2·10⁷ amal ≈ 0.02 sekund\n\n"
        "Matrits ko'rinishida: DFT matritsasi F daraxt strukturasiga ega bo'lib,\n"
        "F = (almashtirish) × (blokli diagonal) × ... × (blokli diagonal)\n"
        "— log₂N marta. Bu — zamonaviy raqamli signal qayta ishlashning asosi."
    )
