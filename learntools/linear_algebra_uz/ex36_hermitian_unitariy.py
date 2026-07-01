"""Hints and solutions — Dars 9.2: Hermitian va Unitariy Matritsalar."""
import numpy as np
from learntools.linear_algebra_uz.base import UzCheckProblem, ThoughtExperiment


class Q1(UzCheckProblem):
    """A Hermitian ekanini tekshiring: A = A*ᵀ (konjugat transponent)."""
    _hints = ["A.conj().T yoki A.T.conj() — konjugat transponent. np.allclose(A, A.conj().T)."]
    _solution = "is_herm = np.allclose(A, A.conj().T)"

    def _do_check(self, is_herm, A):
        expected = np.allclose(A, A.conj().T)
        if is_herm != expected:
            return f"A Hermitianmi? {expected}. ||A - A*ᵀ|| = {np.linalg.norm(A - A.conj().T):.4f}"
        return True


class Q2(UzCheckProblem):
    """Hermitian matritsaning xususiy qiymatlari haqiqiy ekanini tasdiqlang."""
    _hints = [
        "np.linalg.eigh(A) — Hermitian matritsa uchun haqiqiy xususiy qiymatlar.",
        "np.allclose(np.imag(vals), 0) ni tekshiring.",
    ]
    _solution = "vals, vecs = np.linalg.eigh(A); all_real = np.allclose(np.imag(vals), 0)"

    def _do_check(self, vals, A):
        exp_vals = np.linalg.eigh(A)[0]
        if not np.allclose(np.sort(np.real(vals)), np.sort(exp_vals), atol=1e-6):
            return f"Xususiy qiymatlar: {exp_vals}"
        if not np.allclose(np.imag(vals), 0, atol=1e-10):
            return "Hermitian matritsa xususiy qiymatlari haqiqiy son bo'lishi kerak."
        return True


class Q3(UzCheckProblem):
    """U unitariy ekanini tekshiring: U*ᵀ U = I."""
    _hints = ["U.conj().T @ U = I bo'lsa unitariy. np.allclose(U.conj().T @ U, np.eye(n))."]
    _solution = "is_unitary = np.allclose(U.conj().T @ U, np.eye(U.shape[0]))"

    def _do_check(self, is_unitary, U):
        n = U.shape[0]
        expected = np.allclose(U.conj().T @ U, np.eye(n), atol=1e-8)
        if is_unitary != expected:
            err = np.linalg.norm(U.conj().T @ U - np.eye(n))
            return f"U*ᵀ U = I? {expected}. Xato norma: {err:.4f}"
        return True


class Q4(UzCheckProblem):
    """Unitariy o'zgartirishda vektor normasini saqlang: ||Uv|| = ||v||."""
    _hints = [
        "Unitariy U uchun ||Uv|| = ||v||. np.linalg.norm(U @ v).",
    ]
    _solution = "norm_Uv = np.linalg.norm(U @ v); norm_v = np.linalg.norm(v)"

    def _do_check(self, norm_Uv, norm_v, U, v):
        exp_Uv = np.linalg.norm(U @ v)
        exp_v = np.linalg.norm(v)
        if not np.isclose(norm_Uv, exp_Uv, rtol=1e-8):
            return f"||Uv|| = {exp_Uv:.6f}, siz {norm_Uv:.6f} berdingiz."
        if not np.isclose(norm_v, exp_v, rtol=1e-8):
            return f"||v|| = {exp_v:.6f}, siz {norm_v:.6f} berdingiz."
        if not np.isclose(norm_Uv, norm_v, rtol=1e-6):
            return f"||Uv|| = {norm_Uv:.6f} = ||v|| = {norm_v:.6f} bo'lishi kerak."
        return True


class Q5(UzCheckProblem):
    """Hermitian spektral teorema: A = Q Λ Q*ᵀ."""
    _hints = ["vals, Q = np.linalg.eigh(A). A_rec = Q @ np.diag(vals) @ Q.conj().T"]
    _solution = "vals, Q = np.linalg.eigh(A); A_rec = Q @ np.diag(vals) @ Q.conj().T"

    def _do_check(self, A_rec, A):
        vals, Q = np.linalg.eigh(A)
        expected = Q @ np.diag(vals) @ Q.conj().T
        if not np.allclose(A_rec, expected, atol=1e-8):
            return "A = Q Λ Q*ᵀ formulasini tekshiring."
        if not np.allclose(A_rec, A, atol=1e-8):
            return "Qayta qurilgan matritsa A ga teng bo'lishi kerak."
        return True


class Q6(UzCheckProblem):
    """Pauli matritsalari Hermitian va iz=0 ekanini tekshiring."""
    _hints = [
        "Pauli: σx=[[0,1],[1,0]], σy=[[0,-i],[i,0]], σz=[[1,0],[0,-1]].",
        "np.allclose(P, P.conj().T) va np.trace(P) == 0.",
    ]
    _solution = (
        "sigma_x = np.array([[0,1],[1,0]], dtype=complex)\n"
        "sigma_y = np.array([[0,-1j],[1j,0]])\n"
        "sigma_z = np.array([[1,0],[0,-1]], dtype=complex)\n"
        "all_hermitian = all(np.allclose(P, P.conj().T) for P in [sigma_x, sigma_y, sigma_z])"
    )

    def _do_check(self, all_hermitian, paulis):
        expected = all(np.allclose(P, P.conj().T) for P in paulis)
        if all_hermitian != expected:
            return f"Barcha Pauli matritsalari Hermitianmi? {expected}"
        return True


class C1_Q1(UzCheckProblem):
    """QFT (kvant Fourier transformatsiyasi) matritsasini yarating."""
    _hints = [
        "QFT matritsasi: F[j,k] = ω^(jk) / sqrt(n), ω = e^{2πi/n}.",
        "j, k = np.mgrid[0:n, 0:n]; omega = np.exp(2j*np.pi/n); F = omega**(j*k)/np.sqrt(n)",
    ]
    _solution = (
        "n = 4  # misol\n"
        "j, k = np.mgrid[0:n, 0:n]\n"
        "omega = np.exp(2j * np.pi / n)\n"
        "F_qft = omega ** (j * k) / np.sqrt(n)"
    )

    def _do_check(self, F_qft, n):
        j, k = np.mgrid[0:n, 0:n]
        omega = np.exp(2j * np.pi / n)
        expected = omega ** (j * k) / np.sqrt(n)
        if not np.allclose(F_qft, expected, atol=1e-8):
            return "F[j,k] = ω^(jk)/sqrt(n), ω = e^{2πi/n} formulasini tekshiring."
        if not np.allclose(F_qft @ F_qft.conj().T, np.eye(n), atol=1e-8):
            return "QFT unitariy bo'lishi kerak: F F*ᵀ = I."
        return True


class C2_Q1(ThoughtExperiment):
    """Hermitian va unitariy matritsalar kvant mexanikasida nima uchun muhim?"""
    _hints = [
        "Kuzatuvchi = Hermitian; evolyutsiya = unitariy.",
        "Xususiy qiymatlar = o'lchovning natijalari (haqiqiy sonlar).",
    ]
    _solution = (
        "Kvant mexanikasida:\n"
        "1) Hermitian operatorlar — kuzatuvchilar (energiya, impulse, spin).\n"
        "   Xususiy qiymatlar = o'lchovning mumkin natijalari (haqiqiy son bo'lishi shart).\n"
        "2) Unitariy operatorlar — holat evolyutsiyasi (Shredinger tenglamasi: U = e^{-iHt/h}).\n"
        "   Unitariy = norma va ehtimollik saqlanadi.\n"
        "3) Kvant kompyuter: qubit holati |ψ⟩ ∈ C², logik eshiklar — unitariy matritsalar.\n"
        "Umumiy princip: haqiqiy olamning o'lchovi (Hermitian) va\n"
        "reverstibel evolyutsiya (unitariy) — chiziqli algebrani kvant jismoniyatining tili qiladi."
    )
