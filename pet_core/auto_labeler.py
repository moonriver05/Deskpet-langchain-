"""Background scheduler for delayed DeepSeek weak-labeling.

Raw interaction events are cheap to write immediately, but labels are better
after a short delay so feedback events can arrive first. This module keeps the
LLM labeling out of the main chat path.
"""

import threading

from pet_core.config import app_config
from pet_core.learning_labeler import label_pending_events, label_single_event


_lock = threading.Lock()
_batch_timer = None
_event_timers = {}
_running_batch = False
_running_events = set()


def _truthy(value, default=True):
    if value is None or value == "":
        return default
    return str(value).strip().lower() not in ("0", "false", "no", "off")


def _int_config(key, default, min_value=0):
    try:
        value = int(app_config.get(key, default))
    except Exception:
        value = int(default)
    return max(int(min_value), value)


def _observation_window_seconds():
    return _int_config("learning_labeler.observation_window_seconds", 900, min_value=60)


def _feedback_settle_seconds():
    return _int_config("learning_labeler.feedback_settle_seconds", 180, min_value=30)


def _auto_enabled():
    return (
        _truthy(app_config.get("learning_labeler.enabled", "true"), default=True)
        and _truthy(app_config.get("learning_labeler.auto_enabled", "true"), default=True)
    )


def _run_batch(reason):
    global _running_batch
    with _lock:
        if _running_batch:
            return
        _running_batch = True
    try:
        delay = max(
            _int_config("learning_labeler.auto_delay_seconds", 300, min_value=30),
            _observation_window_seconds(),
        )
        limit = _int_config("learning_labeler.max_events_per_run", 20, min_value=1)
        stats = label_pending_events(limit=limit, min_age_seconds=delay)
        print(f"[AutoLabeler] batch reason={reason} stats={stats}")
    except Exception as e:
        print(f"[AutoLabeler] batch failed reason={reason}: {e}")
    finally:
        with _lock:
            _running_batch = False


def schedule_auto_label_batch(reason="new_event", delay_seconds=None):
    """Debounced delayed batch label for pending samples."""
    global _batch_timer
    if not _auto_enabled():
        return False
    delay = (
        int(delay_seconds)
        if delay_seconds is not None
        else max(
            _int_config("learning_labeler.auto_delay_seconds", 300, min_value=30),
            _observation_window_seconds(),
        )
    )
    with _lock:
        if _batch_timer is not None and _batch_timer.is_alive():
            return True
        _batch_timer = threading.Timer(delay, _run_batch, args=(reason,))
        _batch_timer.daemon = True
        _batch_timer.start()
    return True


def _run_single(event_id, reason):
    event_id = str(event_id or "").strip()
    if not event_id:
        return
    with _lock:
        if event_id in _running_events:
            return
        _running_events.add(event_id)
    try:
        stats = label_single_event(event_id, min_age_seconds=_feedback_settle_seconds())
        print(f"[AutoLabeler] relabel event={event_id[:8]} reason={reason} stats={stats}")
    except Exception as e:
        print(f"[AutoLabeler] relabel failed event={event_id[:8]} reason={reason}: {e}")
    finally:
        with _lock:
            _running_events.discard(event_id)
            _event_timers.pop(event_id, None)


def schedule_auto_relabel_event(event_id, reason="feedback", delay_seconds=None):
    """Debounce a single-event relabel after explicit or implicit feedback."""
    event_id = str(event_id or "").strip()
    if not event_id or not _auto_enabled():
        return False
    delay = (
        int(delay_seconds)
        if delay_seconds is not None
        else max(
            _int_config("learning_labeler.feedback_relabel_delay_seconds", 180, min_value=30),
            _feedback_settle_seconds(),
        )
    )
    with _lock:
        old = _event_timers.get(event_id)
        if old is not None and old.is_alive():
            old.cancel()
        timer = threading.Timer(delay, _run_single, args=(event_id, reason))
        timer.daemon = True
        _event_timers[event_id] = timer
        timer.start()
    return True
