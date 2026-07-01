"""Hints and solutions — Dars 2.2: Eliminatsiya G'oyasi."""
import numpy as np
from learntools.core import EqualityCheckProblem, ThoughtExperiment


class Q1(ThoughtExperiment):
    """Eliminatsiyani qo'lda bajaring."""
    _hints = [
        "Birinchi tenglamadan boshlaydigan pivot elementni toping.",
        "Ikkinchi tenglamadan birinchining m21 = a21/a11 marta ko'paytmasini ayringing.",
    ]
    _solution = (
        "Tizim: 2x + 4y = 8  va  x + 3y = 5\n"
        "m = 1/2 (multiplikator)\n"
        "Yangi 2-tenglama: (x+3y) - (1/2)(2x+4y) = 5 - 4\n"
        "→ y = 1, keyin x = 2"
    )


class Q2(EqualityCheckProblem):
    """Gauss eliminatsiyasi — bitta qadam."""
    _hints = [
        "Birinchi satrdan m=A[1,0]/A[0,0] multiplikatorini hisoblang.",
        "2-satrdan birinchi satrning m martaligini ayringing.",
    ]
    _solution = (
        "m = A[1, 0] / A[0, 0]\n"
        "A[1, :] = A[1, :] - m * A[0, :]\n"
        "b[1] = b[1] - m * b[0]"
    )

    def _do_check(self, A_new, b_new, A_orig, b_orig):
        A = A_orig.copy().astype(float)
        b = b_orig.copy().astype(float)
        m = A[1, 0] / A[0, 0]
        A[1, :] -= m * A[0, :]
        b[1] -= m * b[0]
        if not (np.isclose(A_new[1, 0], 0) and np.allclose(A_new[0], A[0])):
            return "Eliminatsiya to'g'ri bajarilmagan. A[1,0] = 0 bo'lishi kerak."
        return True


class Q3(EqualityCheckProblem):
    """To'liq Gauss eliminatsiyasi (yuqori uchburchak ko'rinish)."""
    _hints = [
        "Har bir pivot ustun uchun quyidagi barcha satrlardan o'sha ustun elementini nolga aylantiring.",
        "Ikki marta tsikl: tashqi — pivot uchun, ichki — qatorlar uchun.",
    ]
    _solution = (
        "U = A.copy().astype(float)\n"
        "for col in range(U.shape[1] - 1):\n"
        "    for row in range(col + 1, U.shape[0]):\n"
        "        m = U[row, col] / U[col, col]\n"
        "        U[row, :] -= m * U[col, :]"
    )

    def _do_check(self, U, A):
        n = A.shape[0]
        for i in range(1, n):
            for j in range(i):
                if not np.isclose(U[i, j], 0, atol=1e-9):
                    return f"U[{i},{j}] = {U[i,j]:.4f}, 0 bo'lishi kerak."
        return True


# Challenge 1
class C1_Q1(EqualityCheckProblem):
    """Pivotni toping va nolni aniqlang."""
    _hints = [
        "Pivot — ustundagi noldan farqli birinchi element.",
        "Agar pivot 0 bo'lsa, satrlarni almashtirish kerak.",
    ]
    _solution = (
        "# Pivot = A[i, i]\n"
        "# Agar A[i,i] == 0: satrlarni almashtiramiz (partial pivoting)\n"
        "for i in range(n):\n"
        "    if A[i,i] == 0:\n"
        "        for k in range(i+1, n):\n"
        "            if A[k,i] != 0:\n"
        "                A[[i,k]] = A[[k,i]]\n"
        "                break"
    )

    def _do_check(self, pivots, A):
        expected = [A[i, i] for i in range(min(A.shape))]
        if not np.allclose(pivots, expected):
            return f"Kutilgan pivotlar: {expected}"
        return True


# Challenge 2
class C2_Q1(ThoughtExperiment):
    """Nima uchun eliminatsiya ishlaydi?"""
    _hints = [
        "Satrlarni almashtirishlar yechimni o'zgartirmaydi.",
        "Tenglamaga ko'paytma qo'shish yechimni o'zgartirmaydi.",
    ]
    _solution = (
        "Eliminatsiya ekvivalent transformatsiyalar orqali ishlaydi:\n"
        "1. Satr * skalyar → yechim o'zgarmaydi\n"
        "2. Satr + (boshqa satr * skalyar) → yechim o'zgarmaydi\n"
        "3. Satrlar almashtirish → yechim o'zgarmaydi\n"
        "Maqsad: yuqori uchburchak ko'rinish → orqaga almashtirish (back substitution)"
    )
