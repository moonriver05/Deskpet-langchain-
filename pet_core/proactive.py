"""Pure helpers for proactive companion messages."""

import datetime
import json
import re


PROACTIVE_CONTEXT_FRESH_MINUTES = 90
PROACTIVE_CONTEXT_STALE_MINUTES = 6 * 60


def _parse_ts(value):
    try:
        if not value:
            return None
        return datetime.datetime.fromisoformat(str(value))
    except Exception:
        return None


def _relative_time(now, ts):
    dt = _parse_ts(ts)
    if not dt:
        return "较早", None
    age_minutes = max(0, int((now - dt).total_seconds() // 60))
    if age_minutes < 60:
        return f"{age_minutes}分钟前", age_minutes
    if age_minutes < 24 * 60:
        return f"{age_minutes // 60}小时前", age_minutes
    return f"{age_minutes // (24 * 60)}天前", age_minutes


def _shorten(text, limit):
    s = re.sub(r"\s+", " ", str(text or "").strip())
    if len(s) <= limit:
        return s
    return s[:limit].rstrip("，。,. ") + "..."


def _dedupe_key(text):
    s = str(text or "").lower()
    s = re.sub(r"!\[.*?\]\(.*?\)", "", s)
    s = re.sub(r"[\s，。！？、,.!?；;：:\"'“”‘’（）()\[\]【】\-—_]+", "", s)
    return s[:80]


def _bigrams(text):
    s = _dedupe_key(text)
    if len(s) < 2:
        return {s} if s else set()
    return {s[i:i + 2] for i in range(len(s) - 1)}


def _is_proactive_turn(turn):
    if not isinstance(turn, dict):
        return False
    source = str(turn.get("source") or "").strip().lower()
    return source == "proactive" or not str(turn.get("user") or "").strip()


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
    user_blocks = []
    proactive_records = []
    seen_user = set()
    seen_assistant = set()
    seen_proactive = set()
    latest_user_age_minutes = None

    for turn in turns[-max_turns:]:
        if not isinstance(turn, dict):
            continue
        when, age_minutes = _relative_time(now, turn.get("timestamp"))

        user_text = str(turn.get("user") or "").strip()
        assistant_text = str(turn.get("assistant") or "").strip()
        assistant_summary = str(turn.get("assistant_summary") or "").strip()

        if _is_proactive_turn(turn):
            if age_minutes is not None and age_minutes > PROACTIVE_CONTEXT_STALE_MINUTES:
                continue
            proactive_text = _shorten(assistant_text or assistant_summary, 80)
            key = _dedupe_key(proactive_text)
            if proactive_text and key and key not in seen_proactive:
                seen_proactive.add(key)
                proactive_records.append(f"[{when}] 已主动说过：{proactive_text}")
            continue

        if user_text:
            latest_user_age_minutes = age_minutes
            user_key = _dedupe_key(user_text)
            if user_key in seen_user:
                continue
            seen_user.add(user_key)
            block = [f"[{when}] 用户：{_shorten(user_text, 120)}"]
            summary = _shorten(assistant_summary or assistant_text, 90)
            summary_key = _dedupe_key(summary)
            if summary and summary_key not in seen_assistant:
                seen_assistant.add(summary_key)
                block.append(f"[{when}] 当时回应要点：{summary}")
            user_blocks.append("\n".join(block))

    if latest_user_age_minutes is None:
        freshness = "没有可用的近期用户上下文。请只泛泛问候，比如问用户在做什么。"
    elif latest_user_age_minutes <= PROACTIVE_CONTEXT_FRESH_MINUTES:
        freshness = "最近上下文仍有时效性，可以承接用户刚才提到的事情，但不能说你正在看见或监督用户。"
    elif latest_user_age_minutes <= PROACTIVE_CONTEXT_STALE_MINUTES:
        freshness = "上下文已经有些旧，只能轻轻提一句“刚才/之前”，不要当作用户现在还在做同一件事。"
    else:
        freshness = "上下文已经过期，不要继续关心旧任务；请泛泛问一句用户现在在干嘛。"

    lines = []
    user_blocks = user_blocks[-4:]
    proactive_records = proactive_records[-4:]
    if user_blocks:
        lines.append("最近用户真实输入（用于判断现在可能在做什么）：")
        lines.extend(user_blocks)
    if proactive_records:
        if lines:
            lines.append("")
        lines.append("近期主动关怀记录（这是避重复清单，不是可模仿文案）：")
        lines.extend(proactive_records)
        lines.append("要求：本次不要复用以上句子的开头、结构和核心问法。")

    return "\n".join(lines) if lines else "无", freshness


def is_repetitive_proactive_message(text, turns, now=None, max_age_minutes=PROACTIVE_CONTEXT_STALE_MINUTES):
    """Return True when a generated proactive message is too close to recent ones."""
    now = now or datetime.datetime.now()
    message_key = _dedupe_key(text)
    if len(message_key) < 6:
        return False
    message_bigrams = _bigrams(message_key)
    if not message_bigrams:
        return False

    for turn in list(turns or []):
        if not _is_proactive_turn(turn):
            continue
        _, age_minutes = _relative_time(now, turn.get("timestamp"))
        if age_minutes is not None and age_minutes > max_age_minutes:
            continue
        old_text = str(turn.get("assistant") or turn.get("assistant_summary") or "").strip()
        old_key = _dedupe_key(old_text)
        if len(old_key) < 6:
            continue
        if message_key == old_key or message_key in old_key or old_key in message_key:
            return True
        old_bigrams = _bigrams(old_key)
        if not old_bigrams:
            continue
        overlap = len(message_bigrams & old_bigrams) / max(1, len(message_bigrams | old_bigrams))
        if overlap >= 0.72:
            return True
    return False


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
