"""Hints and solutions — Dars 14.2: CNN (Convolutional Neural Networks)."""
import numpy as np
from learntools.core import EqualityCheckProblem, ThoughtExperiment


def _conv_matrix(w, n):
    K = np.zeros((n, n))
    k = len(w) // 2
    for i in range(n):
        for j in range(n):
            d = j - i + k
            if 0 <= d < len(w):
                K[i, j] = w[d]
    return K


class Q1(EqualityCheckProblem):
    """Valid 1D korrelyatsiya: y_i = sum_k w_k x_{i+k}."""
    _hints = [
        "Har bir chiqish: filtrni signal bo'lagiga skalyar ko'paytirish.",
        "y[i] = np.dot(w, x[i:i+len(w)]), i = 0..len(x)-len(w).",
    ]
    _solution = "y = np.array([w @ x[i:i+len(w)] for i in range(len(x)-len(w)+1)])"

    def _do_check(self, y, w, x):
        m = len(x) - len(w) + 1
        expected = np.array([w @ x[i:i+len(w)] for i in range(m)])
        if not np.allclose(y, expected, atol=1e-6):
            return f"Kutilgan: {expected}, siz berdingiz: {y}"
        return True


class Q2(EqualityCheckProblem):
    """O'rtacha (smoothing) filtri natijasi."""
    _hints = [
        "Filtr w = [1/3, 1/3, 1/3].",
        "y[i] = o'rtacha(x[i], x[i+1], x[i+2]).",
    ]
    _solution = "w = np.array([1/3,1/3,1/3]); y = np.convolve(x, w[::-1], 'valid')"

    def _do_check(self, y, x):
        w = np.array([1/3, 1/3, 1/3])
        m = len(x) - 2
        expected = np.array([w @ x[i:i+3] for i in range(m)])
        if not np.allclose(y, expected, atol=1e-6):
            return f"Kutilgan: {expected}, siz berdingiz: {y}"
        return True


class Q3(EqualityCheckProblem):
    """Konvolyutsiyani matritsa ko'paytmasi sifatida: y = K x."""
    _hints = [
        "K — Toeplitz matritsa (diagonallarda filtr koeffitsiyentlari).",
        "Berilgan K uchun shunchaki y = K @ x.",
    ]
    _solution = "y = K @ x"

    def _do_check(self, y, K, x):
        expected = K @ x
        if not np.allclose(y, expected, atol=1e-6):
            return f"Kutilgan: {expected}, siz berdingiz: {y}"
        return True


class Q4(EqualityCheckProblem):
    """Ikkinchi hosila filtri [1,-2,1] ni qo'llang."""
    _hints = [
        "Bu filtr signal egriligini o'lchaydi.",
        "y[i] = x[i] - 2 x[i+1] + x[i+2].",
    ]
    _solution = "w = np.array([1.,-2.,1.]); y = np.array([w@x[i:i+3] for i in range(len(x)-2)])"

    def _do_check(self, y, x):
        w = np.array([1., -2., 1.])
        expected = np.array([w @ x[i:i+3] for i in range(len(x)-2)])
        if not np.allclose(y, expected, atol=1e-6):
            return f"Kutilgan: {expected}, siz berdingiz: {y}"
        return True


class Q5(EqualityCheckProblem):
    """Tsirkulyant matritsaning xususiy qiymatlari = FFT(birinchi ustun)."""
    _hints = [
        "Tsirkulyant matritsa Furye bazisida diagonallashadi.",
        "Xususiy qiymatlar = np.fft.fft(birinchi_ustun).",
    ]
    _solution = "eig = np.fft.fft(c)"

    def _do_check(self, eig, c):
        expected = np.fft.fft(c)
        if not np.allclose(np.sort(eig.real), np.sort(expected.real), atol=1e-5):
            return f"Kutilgan (FFT): {expected}, siz berdingiz: {eig}"
        return True


class Q6(EqualityCheckProblem):
    """CNN qatlamidagi parametrlar soni (weight sharing)."""
    _hints = [
        "Filtr o'lchami k bo'lsa, parametrlar soni = k (signal uzunligidan mustaqil).",
        "To'liq ulangan n x n = n^2 ga taqqoslang.",
    ]
    _solution = "num_params = k   # signal uzunligiga bog'liq emas"

    def _do_check(self, num_params, k):
        if num_params != k:
            return f"Kutilgan: {k} (filtr o'lchami), siz berdingiz: {num_params}"
        return True


# Challenge 1
class C1_Q1(EqualityCheckProblem):
    """1D konvolyutsiyani Toeplitz matritsa qurib hisoblang."""
    _hints = [
        "K[i,j] = w[j-i+k] (chegarada), aks holda 0; k = len(w)//2.",
        "Keyin y = K @ x natijani beradi.",
    ]
    _solution = (
        "def conv_matrix(w, n):\n"
        "    K = np.zeros((n,n)); k = len(w)//2\n"
        "    for i in range(n):\n"
        "        for j in range(n):\n"
        "            d = j-i+k\n"
        "            if 0 <= d < len(w): K[i,j] = w[d]\n"
        "    return K\n"
        "y = conv_matrix(w, len(x)) @ x"
    )

    def _do_check(self, y, w, x):
        expected = _conv_matrix(w, len(x)) @ x
        if not np.allclose(y, expected, atol=1e-6):
            return f"Kutilgan: {expected}, siz berdingiz: {y}"
        return True


# Challenge 2
class C2_Q1(ThoughtExperiment):
    """Nima uchun konvolyutsiya Toeplitz/tsirkulyant matritsa bilan ifodalanadi?"""
    _hints = [
        "Og'irliklarni ulashish (weight sharing) diagonallar bo'ylab takrorlanishni anglatadi.",
        "Davriy chegarada Toeplitz tsirkulyantga aylanadi va Furye bazisida diagonallashadi.",
    ]
    _solution = (
        "Konvolyutsiyada bir xil filtr butun signal bo'ylab 'siljiydi' — demak\n"
        "matritsaning har bir diagonalida bir xil qiymat turadi: bu Toeplitz tuzilma.\n"
        "Bu parametrlarni keskin kamaytiradi (n^2 o'rniga k ta). Davriy chegara\n"
        "shartida matritsa tsirkulyant bo'ladi va K = F^{-1} Lambda F shaklida\n"
        "diagonallashadi (F — DFT matritsasi). Shu sababli konvolyutsiya chastota\n"
        "fazosida oddiy elementma-element ko'paytmaga aylanadi (FFT tezligi)."
    )
