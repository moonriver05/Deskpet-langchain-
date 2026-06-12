"""Pure helpers for proactive companion messages."""

import datetime
import json
import re


PROACTIVE_CONTEXT_FRESH_MINUTES = 90
PROACTIVE_CONTEXT_STALE_MINUTES = 6 * 60


def _extract_json_object(text):
    s = str(text or "").strip()
    try:
        return json.loads(s)
    except Exception:
        pass
    match = re.search(r"\{.*\}", s, flags=re.S)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except Exception:
        return None


def format_proactive_context_for_prompt(turns, now=None, max_turns=8):
    now = now or datetime.datetime.now()
    if not isinstance(turns, list):
        turns = []
    lines = []
    latest_user_age_minutes = None

    for turn in turns[-max_turns:]:
        if not isinstance(turn, dict):
            continue
        ts = turn.get("timestamp")
        try:
            dt = datetime.datetime.fromisoformat(str(ts)) if ts else None
        except Exception:
            dt = None
        if dt:
            age_minutes = max(0, int((now - dt).total_seconds() // 60))
            if age_minutes < 60:
                when = f"{age_minutes}分钟前"
            elif age_minutes < 24 * 60:
                when = f"{age_minutes // 60}小时前"
            else:
                when = f"{age_minutes // (24 * 60)}天前"
        else:
            age_minutes = None
            when = "较早"

        user_text = str(turn.get("user") or "").strip()
        assistant_summary = str(turn.get("assistant_summary") or "").strip()
        if user_text:
            latest_user_age_minutes = age_minutes
            lines.append(f"[{when}] 用户：{user_text[:120]}")
        if assistant_summary:
            lines.append(f"[{when}] 有珠回应摘要：{assistant_summary[:90]}")

    if latest_user_age_minutes is None:
        freshness = "没有可用的近期用户上下文。请只泛泛问候，比如问用户在做什么。"
    elif latest_user_age_minutes <= PROACTIVE_CONTEXT_FRESH_MINUTES:
        freshness = "最近上下文仍有时效性，可以承接用户刚才提到的事情，但不能说你正在看见或监督用户。"
    elif latest_user_age_minutes <= PROACTIVE_CONTEXT_STALE_MINUTES:
        freshness = "上下文已经有些旧，只能轻轻提一句“刚才/之前”，不要当作用户现在还在做同一件事。"
    else:
        freshness = "上下文已经过期，不要继续关心旧任务；请泛泛问一句用户现在在干嘛。"

    return "\n".join(lines) if lines else "无", freshness


def clean_proactive_message(text):
    s = str(text or "").strip()
    obj = _extract_json_object(s)
    if isinstance(obj, dict):
        s = str(obj.get("message") or "").strip()
    s = re.sub(r"\[[a-zA-Z_]+\]\s*$", "", s).strip()
    s = re.sub(r"\s+", " ", s)
    s = s.strip("“”\"'")
    if len(s) > 80:
        s = s[:80].rstrip("，。,. ") + "。"
    return s
