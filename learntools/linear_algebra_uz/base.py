"""
UzCheckProblem — compatibility shim for linear_algebra_uz exercises.

The upstream learntools EqualityCheckProblem routes check() through _vars/_expected
and never calls _do_check().  All 60 linear_algebra_uz exercise files define their
checking logic in _do_check(*args), so we override check() here to call it directly
and display rich feedback without requiring the binder.

Usage in exercise files:
    from learntools.linear_algebra_uz.base import UzCheckProblem, ThoughtExperiment

Usage in notebooks (no binder needed):
    import sys, os
    sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), '..', '..', 'learntools')))
    from learntools.linear_algebra_uz import ex1_vektorlar as ex
    q1 = ex.Q1()
    # student writes: answer = ...
    q1.check(answer)          # or q1.check(answer, extra_arg, ...)
    q1.hint()                 # first hint
    q1.hint(2)                # second hint
    q1.solution()
"""

from IPython.display import display

from learntools.core.richtext import Correct, TestFailure, Hint, Solution

__all__ = ['UzCheckProblem', 'ThoughtExperiment']

# ──────────────────────────────────────────────────────────────────────────────

class _DisplayMixin:
    """Shared hint() / solution() display helpers."""

    _hints = []
    _solution = ''

    def hint(self, n=1):
        hints = getattr(self, '_hints', [])
        if not hints:
            display(Hint("Bu savol uchun maslahat mavjud emas."))
            return
        idx = min(n - 1, len(hints) - 1)
        is_last = idx >= len(hints) - 1
        display(Hint(hints[idx], n=n, last=is_last))

    def solution(self):
        sol = str(getattr(self, '_solution', '') or '')
        if not sol:
            display(Solution("Yechim ko'rsatilmagan."))
            return
        # Wrap code-like solutions in a markdown code block
        if '\n' in sol or sol.startswith('def ') or '=' in sol:
            sol = '\n```python\n{}\n```'.format(sol)
        display(Solution(sol))


# ──────────────────────────────────────────────────────────────────────────────

class UzCheckProblem(_DisplayMixin):
    """Base class for auto-checked exercises.

    Subclasses must implement:
        _do_check(self, *args) -> True | "error message"

    The first positional arg is the student's answer; any further args are
    extra values needed to compute the expected result (e.g. the input data).
    """

    def check(self, *args):
        try:
            result = self._do_check(*args)
        except Exception as exc:
            display(TestFailure(
                "Tekshirishda xato yuz berdi: {}: {}".format(type(exc).__name__, exc)
            ))
            return

        if result is True:
            display(Correct(''))
        elif isinstance(result, str) and result:
            display(TestFailure(result))
        else:
            # None or any other truthy value → treat as correct
            display(Correct(''))

    def _do_check(self, *args):
        raise NotImplementedError(
            "Subclass must implement _do_check(*args) -> True | 'error string'"
        )


# ──────────────────────────────────────────────────────────────────────────────

class ThoughtExperiment(_DisplayMixin):
    """A reflective / open-ended problem with no automated checking."""

    def check(self, *args):
        display(Correct(
            "Bu fikrlash masalasi — to'g'ri javob yo'q. "
            "`.solution()` ni chaqirib namunali javobni ko'ring."
        ))
