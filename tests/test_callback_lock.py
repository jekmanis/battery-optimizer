"""
Tests for battery_optimizer_lib.callback_lock.CallbackLock.

Every test is designed to finish in well under a second, and every cross-thread
wait carries a timeout so a regression fails the test instead of hanging the
suite.
"""

import threading
import time

import pytest

from battery_optimizer_lib.callback_lock import CallbackLock


JOIN_TIMEOUT = 2.0
WAIT_TIMEOUT = 2.0


class LogRecorder:
    """Stand-in for AppDaemon's ``self.log(msg, level=...)``."""

    def __init__(self):
        self.entries = []
        self._lock = threading.Lock()

    def __call__(self, msg, level="INFO"):
        with self._lock:
            self.entries.append((level, msg))

    def errors(self):
        return self.at_level("ERROR")

    def at_level(self, level):
        return [msg for lvl, msg in self.entries if lvl == level]


def _acquire_from_other_thread(lock, timeout):
    """Try to acquire ``lock``'s underlying RLock on a fresh thread.

    Returns True if the acquire succeeded (and it is released again).
    """
    result = {}

    def worker():
        acquired = lock._lock.acquire(timeout=timeout)
        result["acquired"] = acquired
        if acquired:
            lock._lock.release()

    t = threading.Thread(target=worker)
    t.start()
    t.join(timeout=JOIN_TIMEOUT + timeout)
    assert not t.is_alive(), "helper thread hung"
    return result.get("acquired", False)


# ---------------------------------------------------------------------------
# 1. guard() metadata and calling convention
# ---------------------------------------------------------------------------


def test_guard_preserves_metadata_and_positional_kwargs_call():
    lock = CallbackLock()

    class App:
        def execute_scheduled_mode(self, kwargs=None, force=False):
            """Docstring that must survive."""
            return (self, kwargs, force, lock.depth)

    guarded = lock.guard(App.execute_scheduled_mode)

    assert guarded.__name__ == "execute_scheduled_mode"
    assert guarded.__doc__ == "Docstring that must survive."
    assert guarded.__wrapped__ is App.execute_scheduled_mode

    app = App()
    payload = {"slot": 3}

    # AppDaemon's positional convention: f(kwargs_dict), plus the app's own
    # internal f(kwargs, force=True) call.
    self_obj, got_kwargs, got_force, depth_inside = guarded(app, payload, force=True)
    assert self_obj is app
    assert got_kwargs is payload
    assert got_force is True
    assert depth_inside == 1

    # Purely positional, no extras.
    assert guarded(app, payload)[2] is False
    # Purely keyword.
    assert guarded(app, kwargs=payload, force=True)[1] is payload

    # Lock fully released afterwards.
    assert lock.depth == 0
    assert _acquire_from_other_thread(lock, 0.5) is True


def test_guard_works_on_plain_function():
    lock = CallbackLock()

    @lock.guard
    def cb(a, b=2, *rest, **kw):
        return (a, b, rest, kw, lock.depth)

    assert cb(1, 2, 3, x=4) == (1, 2, (3,), {"x": 4}, 1)
    assert lock.depth == 0


def test_guard_releases_on_exception():
    lock = CallbackLock()

    @lock.guard
    def boom():
        raise ValueError("nope")

    with pytest.raises(ValueError):
        boom()

    assert lock.depth == 0
    assert _acquire_from_other_thread(lock, 0.5) is True


# ---------------------------------------------------------------------------
# 2. Two threads in guarded callbacks are serialized
# ---------------------------------------------------------------------------


def test_two_guarded_callbacks_are_serialized():
    lock = CallbackLock()
    intervals = []
    intervals_lock = threading.Lock()
    barrier = threading.Barrier(2, timeout=WAIT_TIMEOUT)

    @lock.guard
    def callback():
        start = time.monotonic()
        time.sleep(0.03)
        end = time.monotonic()
        with intervals_lock:
            intervals.append((start, end))

    def worker():
        # Sync OUTSIDE the guarded call: the guard serializes, so a barrier
        # inside it would deadlock.
        barrier.wait()
        callback()

    threads = [threading.Thread(target=worker) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=JOIN_TIMEOUT)
        assert not t.is_alive(), "worker thread hung - lock not released?"

    assert len(intervals) == 2
    first, second = sorted(intervals)
    assert first[1] <= second[0], (
        "guarded callbacks overlapped: {} and {}".format(first, second)
    )


# ---------------------------------------------------------------------------
# 3. Re-entrancy
# ---------------------------------------------------------------------------


def test_reentrant_guarded_calls_do_not_deadlock():
    lock = CallbackLock()
    observed = []

    @lock.guard
    def inner():
        observed.append(("inner", lock.depth))
        return "inner-result"

    @lock.guard
    def outer():
        observed.append(("outer", lock.depth))
        return inner()

    assert outer() == "inner-result"
    assert observed == [("outer", 1), ("inner", 2)]
    assert lock.depth == 0
    assert _acquire_from_other_thread(lock, 0.5) is True


def test_manual_nesting_depth_and_locked_by_me():
    lock = CallbackLock()
    assert lock.depth == 0
    assert lock.locked_by_me is False
    with lock:
        assert lock.depth == 1
        assert lock.locked_by_me is True
        with lock:
            assert lock.depth == 2
        assert lock.depth == 1
    assert lock.depth == 0
    assert lock.locked_by_me is False


# ---------------------------------------------------------------------------
# 4. unlocked() at depth 1 really releases
# ---------------------------------------------------------------------------


def test_unlocked_at_depth_one_releases_the_lock():
    recorder = LogRecorder()
    lock = CallbackLock(log_func=recorder)

    a_inside = threading.Event()
    b_acquired = threading.Event()
    b_depth = {}

    def thread_b():
        assert a_inside.wait(timeout=WAIT_TIMEOUT), "A never entered unlocked()"
        with lock:
            b_depth["depth"] = lock.depth
            b_acquired.set()

    t = threading.Thread(target=thread_b)
    t.start()

    observed_release = {}
    with lock:
        assert lock.depth == 1
        with lock.unlocked():
            assert lock.depth == 0, "depth must drop while the lock is released"
            a_inside.set()
            observed_release["b_got_it"] = b_acquired.wait(timeout=WAIT_TIMEOUT)
        assert lock.depth == 1, "depth must be restored after unlocked()"

    t.join(timeout=JOIN_TIMEOUT)
    assert not t.is_alive()

    assert observed_release["b_got_it"] is True, (
        "thread B could not acquire the lock while A was inside unlocked()"
    )
    assert b_depth["depth"] == 1
    assert lock.depth == 0
    assert recorder.errors() == []


# ---------------------------------------------------------------------------
# 5. unlocked() at depth 2 does NOT release
# ---------------------------------------------------------------------------


def test_unlocked_at_depth_two_keeps_the_lock_and_logs_once_at_debug():
    """Depth >= 2 is the DESIGNED path, not a misuse.

    `full_optimize` / `_recalculate_remaining_schedule` /
    `on_manual_mode_change` all reach `_apply_mode_tracked` through
    `execute_scheduled_mode`, so this happens on ordinary days. It must not be
    reported at ERROR alongside a genuine depth-0 bug.
    """
    recorder = LogRecorder()
    lock = CallbackLock(log_func=recorder)

    with lock:
        with lock:
            assert lock.depth == 2
            with lock.unlocked():
                assert lock.depth == 2, "depth must not change in a nested region"
                # Another thread must NOT be able to take the lock.
                assert _acquire_from_other_thread(lock, 0.1) is False
            assert lock.depth == 2

    assert recorder.errors() == []
    debugs = recorder.at_level("DEBUG")
    assert len(debugs) == 1, recorder.entries
    assert "nested unlocked region" in debugs[0]
    assert "depth=2" in debugs[0]
    assert "by design" in debugs[0]
    assert lock.depth == 0


def test_unlocked_at_depth_two_logs_only_once_across_many_regions():
    recorder = LogRecorder()
    lock = CallbackLock(log_func=recorder)

    for _ in range(5):
        with lock:
            with lock:
                with lock.unlocked():
                    pass

    assert len(recorder.at_level("DEBUG")) == 1
    assert recorder.errors() == []


def test_unlocked_at_depth_two_never_raises_even_under_strict():
    """Strict exists to catch bugs; the nested path is not one."""
    lock = CallbackLock(strict=True)
    ran = []
    with lock:
        with lock:
            with lock.unlocked():
                ran.append(lock.depth)
            assert lock.depth == 2
    assert ran == [2], "the body must still run under strict"
    assert lock.depth == 0


def test_unlocked_without_holding_the_lock():
    recorder = LogRecorder()
    lock = CallbackLock(log_func=recorder)

    ran = []
    with lock.unlocked():
        ran.append(lock.depth)

    assert ran == [0]
    errors = recorder.errors()
    assert len(errors) == 1, errors
    assert "depth=0" in errors[0]

    strict = CallbackLock(strict=True)
    with pytest.raises(RuntimeError):
        with strict.unlocked():
            pytest.fail("body must not run under strict")


# ---------------------------------------------------------------------------
# 6. Exception inside the unlocked body
# ---------------------------------------------------------------------------


def test_exception_inside_unlocked_body_reacquires_before_propagating():
    recorder = LogRecorder()
    lock = CallbackLock(log_func=recorder)

    with lock:
        with pytest.raises(ValueError):
            with lock.unlocked():
                assert lock.depth == 0
                raise ValueError("blocking call blew up")

        # Lock is held again, at the right depth ...
        assert lock.depth == 1
        assert _acquire_from_other_thread(lock, 0.1) is False

        # ... and the escape hatch still works afterwards.
        with lock.unlocked():
            assert lock.depth == 0
            assert _acquire_from_other_thread(lock, 0.5) is True
        assert lock.depth == 1

    assert lock.depth == 0
    assert _acquire_from_other_thread(lock, 0.5) is True
    assert recorder.errors() == []


# ---------------------------------------------------------------------------
# 7. depth is per-thread
# ---------------------------------------------------------------------------


def test_depth_is_per_thread():
    lock = CallbackLock()
    other_depth = {}
    holder_ready = threading.Event()
    release_holder = threading.Event()

    def observer():
        assert holder_ready.wait(timeout=WAIT_TIMEOUT)
        # Main thread holds the lock twice; this thread holds none.
        other_depth["depth"] = lock.depth
        other_depth["locked_by_me"] = lock.locked_by_me
        release_holder.set()

    t = threading.Thread(target=observer)
    t.start()

    with lock:
        with lock:
            assert lock.depth == 2
            holder_ready.set()
            assert release_holder.wait(timeout=WAIT_TIMEOUT)

    t.join(timeout=JOIN_TIMEOUT)
    assert not t.is_alive()

    assert other_depth["depth"] == 0
    assert other_depth["locked_by_me"] is False
    assert lock.depth == 0


def test_depth_is_independent_across_sequential_threads():
    lock = CallbackLock()
    depths = []

    def worker():
        depths.append(lock.depth)
        with lock:
            depths.append(lock.depth)
        depths.append(lock.depth)

    for _ in range(2):
        t = threading.Thread(target=worker)
        t.start()
        t.join(timeout=JOIN_TIMEOUT)
        assert not t.is_alive()

    assert depths == [0, 1, 0, 0, 1, 0]


# ---------------------------------------------------------------------------
# 8. log-once behaviour
# ---------------------------------------------------------------------------


def test_nested_unlocked_logs_only_once_across_repeated_calls():
    recorder = LogRecorder()
    lock = CallbackLock(log_func=recorder)

    for _ in range(5):
        with lock:
            with lock:
                with lock.unlocked():
                    pass

    assert len(recorder.at_level("DEBUG")) == 1, recorder.entries
    assert recorder.errors() == []

    # Deeper nesting is the same designed path -> still silent afterwards.
    with lock:
        with lock:
            with lock:
                with lock.unlocked():
                    pass
    assert len(recorder.at_level("DEBUG")) == 1, recorder.entries

    # The depth-0 bug has its own one-shot budget, on its own level.
    with lock.unlocked():
        pass
    with lock.unlocked():
        pass
    errors = recorder.errors()
    assert len(errors) == 1, errors
    assert "depth=0" in errors[0]
    assert len(recorder.at_level("DEBUG")) == 1

    # Fresh instance starts with a fresh budget.
    recorder2 = LogRecorder()
    lock2 = CallbackLock(log_func=recorder2)
    with lock2:
        with lock2:
            with lock2.unlocked():
                pass
    assert len(recorder2.at_level("DEBUG")) == 1


def test_nested_unlocked_logs_once_under_concurrency():
    recorder = LogRecorder()
    lock = CallbackLock(log_func=recorder)
    barrier = threading.Barrier(4, timeout=WAIT_TIMEOUT)

    def worker():
        barrier.wait()
        for _ in range(10):
            with lock:
                with lock:
                    with lock.unlocked():
                        pass

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=JOIN_TIMEOUT)
        assert not t.is_alive()

    assert len(recorder.at_level("DEBUG")) == 1, recorder.entries
    assert recorder.errors() == []


def test_no_log_func_is_safe():
    lock = CallbackLock()
    with lock:
        with lock:
            with lock.unlocked():
                pass
    assert lock.depth == 0
