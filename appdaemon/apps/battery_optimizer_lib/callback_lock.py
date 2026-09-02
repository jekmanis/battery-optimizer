"""
App-wide re-entrant callback lock with a bounded "unlocked" escape hatch.

Purpose
-------
AppDaemon dispatches this app's callbacks from a pool of worker threads (see
CLAUDE.md § Runtime constraints and `docs`/design notes on thread safety).  Every
registered callback already funnels through the orchestrator's `_timed_callback`
decorator, which makes that decorator a complete chokepoint: wrapping its body in
one app-wide re-entrant lock restores the single-threaded semantics the app was
written against, without an async rewrite.

The one thing that must NOT run under that lock is the blocking inverter write
(`growatt_modbus/set_wit_mode`, up to ~15 s).  `unlocked()` is the single,
deliberately narrow escape hatch for that call: it drops the app lock for the
duration of one expression and re-acquires it before anything touches app state
again.

Invariants
----------
1. **Depth bookkeeping lives here and only here.**  The per-thread recursion
   depth is maintained exclusively by `__enter__` / `__exit__` on a
   `threading.local()`.  Never inspect or poke `threading.RLock` internals
   (`_release_save` / `_acquire_restore` / `_count`) — they are private, they
   differ between the C and Python implementations, and they would silently
   drop *all* recursion levels.

2. **Never release more than one recursion level.**  `unlocked()` releases only
   when this thread holds exactly one level (`depth == 1`).  At depth >= 2 an
   outer frame is mid-mutation of shared state (typically a schedule rebuild
   that just deleted future slots); releasing there would expose a torn
   schedule to another worker thread.  In that case the region degrades to
   "run the blocking call with the lock held" — today's behaviour, and correct.

3. **Lock order: app lock -> DirectControl._io_lock -> DirectControl._state_lock.**
   Never acquire the app lock while holding either DirectControl lock.  In
   particular, DirectControl's verification path must report its callback
   duration back to the app *after* both of its own locks are released.

4. `unlocked()` re-acquires in a `finally`, so an exception raised inside the
   body can never leave the app lock dropped; the exception propagates with the
   lock held exactly as it was on entry.

Degrade, don't raise
--------------------
This lock runs inside a home-automation control loop that drives a battery
inverter.  A misuse of `unlocked()` (nested region, or a call made without
holding the lock at all) is a programming error, but raising in production
would abort a callback that may be the one holding the battery in a safe mode.
So the default is to log ONCE at ERROR — repeated logging would itself become
the spam problem — and continue with the lock held, which is always the
conservative choice.  `strict=True` turns the same conditions into
`RuntimeError` and exists for tests only.
"""

from __future__ import annotations

import functools
import threading
from contextlib import contextmanager


class CallbackLock:
    """One app-wide re-entrant lock plus a bounded 'unlocked' escape hatch.

    Args:
        log_func: Optional ``log(msg, level=...)`` callable.  The signature is
            AppDaemon's ``self.log``, so ``CallbackLock(log_func=self.log)``
            works directly.  ``None`` silences the diagnostics.
        strict: When True, misuse of :meth:`unlocked` raises ``RuntimeError``
            instead of logging and degrading.  Tests only — never in production.

    The app creates exactly one instance, so "log once per instance" is "log
    once per process" in practice.
    """

    def __init__(self, log_func=None, strict: bool = False) -> None:
        self._lock = threading.RLock()
        self._local = threading.local()
        self._log_func = log_func
        self._strict = bool(strict)
        # One-shot diagnostics: separate flags so a nested region and a
        # not-held region each get to speak once.
        self._logged_nested = False
        self._logged_not_held = False
        # Guards the two one-shot flags (they are cross-thread, and the whole
        # point of the nested/not-held cases is that the app lock may not be a
        # reliable mutex for them).
        self._diag_lock = threading.Lock()

    # ------------------------------------------------------------------
    # depth bookkeeping (the ONLY place `depth` is written)
    # ------------------------------------------------------------------

    @property
    def depth(self) -> int:
        """Recursion depth held by the *calling* thread (0 if it holds none)."""
        return getattr(self._local, "depth", 0)

    @property
    def locked_by_me(self) -> bool:
        """True if the calling thread currently holds the lock."""
        return self.depth > 0

    def _set_depth(self, value: int) -> None:
        self._local.depth = value

    # ------------------------------------------------------------------
    # context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "CallbackLock":
        self._lock.acquire()
        self._set_depth(self.depth + 1)
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self._set_depth(self.depth - 1)
        self._lock.release()
        return False

    # ------------------------------------------------------------------
    # decorator
    # ------------------------------------------------------------------

    def guard(self, func):
        """Decorate ``func`` so every call runs under this lock.

        The wrapper takes ``(*args, **kwargs)`` verbatim and passes them
        through unchanged, so AppDaemon's positional calling convention
        survives — including the orchestrator's own internal calls such as
        ``execute_scheduled_mode(kwargs, force=True)``.

        This is normally applied to *unbound* methods of another class
        (``lock.guard(App.execute_scheduled_mode)``), which is why the wrapper
        makes no assumption about ``args[0]``.
        """

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with self:
                return func(*args, **kwargs)

        return wrapper

    # ------------------------------------------------------------------
    # escape hatch
    # ------------------------------------------------------------------

    @contextmanager
    def unlocked(self):
        """Temporarily drop the app lock around one blocking call.

        At ``depth == 1`` the lock is released for the duration of the body and
        re-acquired in a ``finally``.  At any other depth the lock is kept (see
        the module docstring, invariant 2): the body still runs, but serialized
        as it is today.
        """
        depth = self.depth

        if depth == 1:
            self._set_depth(0)
            self._lock.release()
            try:
                yield
            finally:
                self._lock.acquire()
                self._set_depth(1)
            return

        if depth == 0:
            self._misuse(
                "CallbackLock.unlocked() called without holding the app lock "
                "(depth=0) - this is a programming error; running the blocking "
                "call unprotected"
            )
        else:
            self._misuse(
                "CallbackLock: nested unlocked region - running the blocking "
                "call under the app lock (depth={})".format(depth)
            )
        yield

    def _misuse(self, message: str) -> None:
        if self._strict:
            raise RuntimeError(message)
        # `depth == 0` and `depth >= 2` are distinct bugs; let each log once.
        nested = "nested" in message
        with self._diag_lock:
            if nested:
                if self._logged_nested:
                    return
                self._logged_nested = True
            else:
                if self._logged_not_held:
                    return
                self._logged_not_held = True
        if self._log_func is not None:
            self._log_func(message, level="ERROR")
