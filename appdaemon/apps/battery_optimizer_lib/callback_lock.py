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

   Depth >= 2 is a DESIGNED path, not a defect.  `full_optimize` /
   `_recalculate_remaining_schedule` / `on_manual_mode_change` all call
   `execute_scheduled_mode` -> `_apply_mode_tracked` from inside their own
   guarded callback, so every scheduled inverter write that follows a rebuild
   arrives here at depth 2.  It is reported once at DEBUG for traceability and
   never raises, not even under `strict=True`.

3. **Lock order: app lock -> DirectControl._io_lock -> DirectControl._state_lock.**
   Never acquire the app lock while holding either DirectControl lock.  In
   particular, DirectControl's verification path must report its callback
   duration back to the app *after* both of its own locks are released.

4. `unlocked()` re-acquires in a `finally`, so an exception raised inside the
   body can never leave the app lock dropped; the exception propagates with the
   lock held exactly as it was on entry.

Two different depths, two different verdicts
--------------------------------------------
`unlocked()` outside `depth == 1` used to be reported as one condition
("misuse"), which conflated a real bug with the app's normal control flow:

* **`depth == 0`** — `unlocked()` was called without holding the app lock at
  all.  That is a programming error (an un-guarded callback, or a call from a
  thread that never entered the lock): the body runs completely unprotected.
  Logged ONCE at ERROR — repeated logging would itself become the spam problem
  — and, under `strict=True`, raised.  Production still degrades rather than
  raising: aborting the callback could be aborting the one holding the battery
  in a safe mode.
* **`depth >= 2`** — the designed nested path (invariant 2).  Keeping the lock
  is the *correct* outcome, so this is not a misuse at all: it is logged ONCE at
  DEBUG for traceability and NEVER raises, `strict=True` included.  A strict
  test double must not explode merely because the app did what it was designed
  to do.
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
        strict: When True, a ``depth == 0`` :meth:`unlocked` call raises
            ``RuntimeError`` instead of logging and degrading.  Tests only —
            never in production.  It deliberately does NOT affect the nested
            (``depth >= 2``) path, which is designed behaviour.

    The app creates exactly one instance, so "log once per instance" is "log
    once per process" in practice.
    """

    def __init__(self, log_func=None, strict: bool = False) -> None:
        self._lock = threading.RLock()
        self._local = threading.local()
        self._log_func = log_func
        self._strict = bool(strict)
        # One-shot diagnostics: separate flags so the (designed) nested region
        # and the (buggy) not-held region each get to speak once.
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

        The two "any other depth" cases are NOT the same thing.  ``depth >= 2``
        is the designed nested path and only gets a one-shot DEBUG line;
        ``depth == 0`` means the body ran with no protection at all and is a
        genuine programming error.
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
            self._not_held()
        else:
            self._nested(depth)
        yield

    def _not_held(self) -> None:
        """`unlocked()` without the lock: a real bug, reported once at ERROR."""
        message = (
            "CallbackLock.unlocked() called without holding the app lock "
            "(depth=0) - this is a programming error; running the blocking "
            "call unprotected"
        )
        if self._strict:
            raise RuntimeError(message)
        with self._diag_lock:
            if self._logged_not_held:
                return
            self._logged_not_held = True
        if self._log_func is not None:
            self._log_func(message, level="ERROR")

    def _nested(self, depth: int) -> None:
        """`unlocked()` inside another guarded frame: designed, reported at DEBUG.

        Never raises — not even under ``strict`` — because keeping the lock IS
        the specified behaviour here (invariant 2).  `full_optimize`,
        `_recalculate_remaining_schedule` and `on_manual_mode_change` all reach
        the inverter write through `execute_scheduled_mode`, so this fires on
        ordinary days.
        """
        with self._diag_lock:
            if self._logged_nested:
                return
            self._logged_nested = True
        if self._log_func is not None:
            self._log_func(
                "CallbackLock: nested unlocked region (depth={}): blocking call "
                "runs under the app lock by design".format(depth),
                level="DEBUG",
            )
