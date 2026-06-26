"""Append-only logs for future preference and recommendation training.

The runtime writes raw interaction events first. Feedback is stored as a
separate event stream so training code can join by event_id without mutating
historical data.
"""

import datetime
import json
import os
import threading
import uuid


SCHEMA_VERSION = "preference_recommender.v1"

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LEARNING_DATA_DIR = os.path.join(PROJECT_ROOT, "learning_data")
RAW_INTERACTIONS_PATH = os.path.join(LEARNING_DATA_DIR, "raw_interactions.jsonl")
FEEDBACK_EVENTS_PATH = os.path.join(LEARNING_DATA_DIR, "feedback_events.jsonl")
LABEL_QUEUE_PATH = os.path.join(LEARNING_DATA_DIR, "label_queue.jsonl")
LABEL_RESULTS_PATH = os.path.join(LEARNING_DATA_DIR, "label_results.jsonl")
LABELED_INTERACTIONS_PATH = os.path.join(LEARNING_DATA_DIR, "labeled_interactions.jsonl")
MANUAL_LABELS_PATH = os.path.join(LEARNING_DATA_DIR, "manual_labels.jsonl")

_write_lock = threading.Lock()


def _now():
    return datetime.datetime.now()


def _safe_text(value, limit=1200):
    text = str(value or "").strip()
    if len(text) > limit:
        return text[:limit] + "...[truncated]"
    return text


def preview_text(value, limit=80):
    text = _safe_text(value, limit=max(limit, 8))
    text = " ".join(text.split())
    if len(text) > limit:
        return text[:limit] + "..."
    return text


def _parse_ts(value):
    try:
        if not value:
            return None
        return datetime.datetime.fromisoformat(str(value))
    except Exception:
        return None


def _minutes_ago(ts, now):
    dt = _parse_ts(ts)
    if not dt:
        return None
    return round(max(0, (now - dt).total_seconds()) / 60, 2)


def _infer_turn_source(item):
    source = str((item or {}).get("source") or "").strip().lower()
    if source in {"chat", "proactive", "system"}:
        return source
    if not _safe_text((item or {}).get("user", ""), 20):
        return "proactive"
    return "chat"


def _context_dedupe_key(text):
    return "".join(str(text or "").split())[:80]


def _daypart(hour):
    if 5 <= hour < 11:
        return "morning"
    if 11 <= hour < 14:
        return "noon"
    if 14 <= hour < 18:
        return "afternoon"
    if 18 <= hour < 23:
        return "evening"
    return "night"


def _append_jsonl(path, record):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with _write_lock:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def recent_context_snapshot(turns, now=None, max_turns=8):
    now = now or _now()
    result = []
    seen = set()
    for item in list(turns or [])[-max_turns:]:
        if not isinstance(item, dict):
            continue
        source = _infer_turn_source(item)
        user_text = _safe_text(item.get("user", ""), 300)
        assistant_summary = _safe_text(
            item.get("assistant_summary") or item.get("assistant", ""),
            300,
        )
        key = (source, _context_dedupe_key(user_text or assistant_summary))
        if key in seen:
            continue
        seen.add(key)
        result.append({
            "role_pair": "proactive_message" if source == "proactive" else "user_assistant",
            "source": source,
            "user": user_text,
            "assistant_summary": assistant_summary,
            "timestamp": item.get("timestamp") or item.get("time") or "",
            "minutes_ago": _minutes_ago(item.get("timestamp") or item.get("time"), now),
        })
    return result


def time_features(now, recent_context):
    last_minutes = None
    for item in reversed(recent_context or []):
        if item.get("minutes_ago") is not None:
            last_minutes = item["minutes_ago"]
            break
    return {
        "hour": now.hour,
        "minute": now.minute,
        "weekday": now.weekday(),
        "daypart": _daypart(now.hour),
        "gap_since_last_turn_sec": None if last_minutes is None else int(last_minutes * 60),
    }


def build_interaction_event(
    *,
    user_text,
    assistant_text,
    trigger_type="user_message",
    trigger_source="chat",
    recent_turns=None,
    user_profile_snapshot="",
    system_state_snapshot=None,
    app_state_snapshot=None,
    retrieved_memories=None,
    retrieved_memory_ids=None,
    knowledge_results=None,
    knowledge_tool_info=None,
    search_keywords=None,
    extracted_fact="",
    current_response_card="",
    local_prediction=None,
    emotion_tag="none",
    has_attachment=False,
    attachment_name="",
    models=None,
    rag_params=None,
):
    now = _now()
    recent_context = recent_context_snapshot(recent_turns, now=now)
    event_id = uuid.uuid4().hex
    system_state_snapshot = dict(system_state_snapshot or {})
    if app_state_snapshot:
        system_state_snapshot.setdefault("app_state", {}).update(dict(app_state_snapshot))
    return {
        "schema_version": SCHEMA_VERSION,
        "event_id": event_id,
        "timestamp": now.isoformat(timespec="seconds"),
        "event_type": "proactive_message" if trigger_type.startswith("proactive") else "assistant_reply",
        "trigger": {
            "type": trigger_type,
            "source": trigger_source,
        },
        "user_input": {
            "text": _safe_text(user_text, 3000),
            "has_attachment": bool(has_attachment),
            "attachment_name": os.path.basename(attachment_name or ""),
        },
        "recent_context": recent_context,
        "time_features": time_features(now, recent_context),
        "state_features": {
            "system": system_state_snapshot,
        },
        "user_profile_snapshot": _safe_text(user_profile_snapshot, 5000),
        "retrieval": {
            "retrieved_memory_ids": list(retrieved_memory_ids or []),
            "short_memory_snippets": [_safe_text(x, 500) for x in (retrieved_memories or [])],
            "knowledge_snippets": [_safe_text(x, 500) for x in (knowledge_results or [])],
            "knowledge_tool": dict(knowledge_tool_info or {
                "used": bool(knowledge_results),
                "mode": "legacy",
                "reason": "",
                "query": _safe_text(user_text, 500),
            }),
            "search_keywords": list(search_keywords or []),
            "extracted_fact": _safe_text(extracted_fact, 800),
        },
        "prompt_decision": {
            "current_response_card": _safe_text(current_response_card, 2500),
            "local_prediction": dict(local_prediction or {}),
            "teacher_labels": None,
        },
        "strategy": {
            "selected": None,
            "candidates": [],
            "recommendation_used": None,
            "recommended_action": None,
            "recommendation_candidates": [],
        },
        "assistant_reply": {
            "text": _safe_text(assistant_text, 4000),
            "emotion_tag": emotion_tag or "none",
        },
        "models": dict(models or {}),
        "rag_params": dict(rag_params or {}),
        "feedback": {
            "explicit": None,
            "implicit": {
                "user_replied_after_sec": None,
                "user_continued_task": None,
                "user_rejected": None,
            },
        },
        "label_status": "pending",
    }


def log_interaction_event(event, enqueue_label=True):
    _append_jsonl(RAW_INTERACTIONS_PATH, event)
    if enqueue_label:
        system_state = ((event.get("state_features") or {}).get("system") or {})
        foreground = system_state.get("foreground") or {}
        idle = system_state.get("idle") or {}
        trigger = event.get("trigger") or {}
        user_input = event.get("user_input") or {}
        assistant_reply = event.get("assistant_reply") or {}
        task = {
            "schema_version": SCHEMA_VERSION,
            "event_id": event.get("event_id"),
            "timestamp": event.get("timestamp"),
            "event_type": event.get("event_type"),
            "trigger_type": trigger.get("type", ""),
            "trigger_source": trigger.get("source", ""),
            "status": "pending",
            "task": "teacher_label",
            "user_text_preview": preview_text(user_input.get("text", ""), 120),
            "assistant_preview": preview_text(assistant_reply.get("text", ""), 120),
            "foreground_category": foreground.get("category", "unknown"),
            "idle_bucket": idle.get("seconds_bucket", "unknown"),
            "label_targets": [
                "state",
                "need",
                "feedback_strategy",
                "recommendation_type",
                "proactive_timing",
                "do_nothing_preference",
                "risk",
            ],
        }
        _append_jsonl(LABEL_QUEUE_PATH, task)


def log_feedback_event(
    *,
    event_id="",
    message_id="",
    feedback_value=0,
    feedback_scope="reply_strategy",
    user_text="",
    assistant_text="",
    extra=None,
):
    record = {
        "schema_version": SCHEMA_VERSION,
        "feedback_event_id": uuid.uuid4().hex,
        "timestamp": _now().isoformat(timespec="seconds"),
        "event_id": event_id or "",
        "message_id": message_id or "",
        "feedback_scope": feedback_scope,
        "feedback": int(feedback_value),
        "user_text": _safe_text(user_text, 1200),
        "assistant_text": _safe_text(assistant_text, 1600),
        "extra": dict(extra or {}),
    }
    _append_jsonl(FEEDBACK_EVENTS_PATH, record)
    return record


def log_implicit_state_observation(
    *,
    event_id="",
    scope="implicit_state_observation",
    feedback_value=0,
    user_text="[IMPLICIT_STATE_OBSERVATION]",
    assistant_text="",
    system_state=None,
    delay_seconds=None,
    extra=None,
):
    """Store a delayed local-state observation for future strategy training.

    This is neutral feedback by default. It is not saying the reply was good or
    bad; it gives the labeler and future local models more evidence about what
    happened after the reply.
    """
    payload = dict(extra or {})
    payload.setdefault("observation_kind", scope)
    if delay_seconds is not None:
        try:
            payload["delay_seconds"] = int(delay_seconds)
        except Exception:
            payload["delay_seconds"] = delay_seconds
    if system_state is not None:
        payload["system_state"] = dict(system_state or {})
    return log_feedback_event(
        event_id=event_id,
        feedback_value=feedback_value,
        feedback_scope=scope,
        user_text=user_text,
        assistant_text=assistant_text,
        extra=payload,
    )
