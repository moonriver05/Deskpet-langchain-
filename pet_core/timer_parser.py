"""Parsing helpers for local focus timers and pomodoro requests."""

import re


CHINESE_NUMBER_MAP = {
    "零": 0, "〇": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4,
    "五": 5, "六": 6, "七": 7, "八": 8, "九": 9,
}


def _parse_simple_chinese_number(text):
    s = str(text or "").strip().replace("个", "").replace("鐘", "钟")
    if not s:
        return None
    if s in CHINESE_NUMBER_MAP:
        return float(CHINESE_NUMBER_MAP[s])
    if s == "十":
        return 10.0
    if "百" in s:
        left, _, right = s.partition("百")
        base = CHINESE_NUMBER_MAP.get(left, 1 if not left else None)
        if base is None:
            return None
        tail = _parse_simple_chinese_number(right) if right else 0
        if tail is None:
            return None
        return float(base * 100 + tail)
    if "十" in s:
        left, _, right = s.partition("十")
        tens = CHINESE_NUMBER_MAP.get(left, 1 if not left else None)
        ones = CHINESE_NUMBER_MAP.get(right, 0 if not right else None)
        if tens is None or ones is None:
            return None
        return float(tens * 10 + ones)
    return None


def _parse_timer_amount(raw):
    s = str(raw or "").strip().replace("个", "")
    if not s:
        return None
    if s in ("半", "半个"):
        return 0.5
    if s.endswith("半") and len(s) > 1:
        base = _parse_timer_amount(s[:-1])
        if base is not None:
            return base + 0.5
    try:
        return float(s)
    except ValueError:
        return _parse_simple_chinese_number(s)


def format_focus_duration(seconds):
    seconds = int(max(1, seconds))
    if seconds < 60:
        return f"{seconds}秒"
    minutes = seconds // 60
    rest_seconds = seconds % 60
    if minutes < 60:
        return f"{minutes}分{rest_seconds}秒" if rest_seconds else f"{minutes}分钟"
    hours = minutes // 60
    rest_minutes = minutes % 60
    if rest_minutes:
        return f"{hours}小时{rest_minutes}分钟"
    return f"{hours}小时"


def parse_focus_timer_intent(text):
    s = str(text or "").strip()
    if not s:
        return None
    intent_words = (
        "定时", "计时", "倒计时", "专注", "番茄钟", "计个时", "定个时",
        "定个", "定一个", "设个", "设一个", "帮我定", "帮我设",
        "开个", "开一个", "启动一个", "timer", "focus",
    )
    if not any(w.lower() in s.lower() for w in intent_words):
        return None

    pattern = re.compile(
        r"(?P<num>\d+(?:\.\d+)?|[零〇一二两三四五六七八九十百]+(?:个)?半?|半)"
        r"\s*(?P<unit>小时|钟头|h|H|分钟|分|min|MIN|秒|s|S)"
    )
    match = pattern.search(s)
    if not match:
        default_start_words = ("开", "启动", "开始", "来个", "来一个", "帮我", "给我", "设", "定")
        wants_default_pomodoro = (
            "番茄钟" in s
            or ("专注" in s and any(w in s for w in ("定时", "计时", "倒计时")))
        )
        if wants_default_pomodoro and any(w in s for w in default_start_words):
            return {
                "seconds": 25 * 60,
                "label": "专注",
                "duration_text": format_focus_duration(25 * 60),
            }
        return None

    amount = _parse_timer_amount(match.group("num"))
    if amount is None or amount <= 0:
        return None
    unit = match.group("unit").lower()
    if unit in ("小时", "钟头", "h"):
        seconds = int(amount * 3600)
    elif unit in ("分钟", "分", "min"):
        seconds = int(amount * 60)
    else:
        seconds = int(amount)
    seconds = max(5, min(seconds, 24 * 60 * 60))
    label = "专注" if any(w in s for w in ("专注", "番茄钟", "学习", "复习", "背单词", "做题")) else "定时"
    return {
        "seconds": seconds,
        "label": label,
        "duration_text": format_focus_duration(seconds),
    }

