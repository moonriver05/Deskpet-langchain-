"""Pending skill intents for multi-turn tool confirmation.

This solves the common pattern:

    user: 我明天要复习数据库
    assistant: 要不要帮你记成待办？
    user: 好，帮我记一下

The second user message does not contain the todo content, so a one-turn router
cannot recover it.  We keep a short-lived pending skill call and execute it when
the next user message is a confirmation.
"""

from __future__ import annotations

import datetime
import re
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple


JsonDict = Dict[str, Any]


CONFIRM_PATTERNS = (
    "可以", "行", "好", "嗯", "对", "是", "记吧", "记一下", "帮我记",
    "加到待办", "写进待办", "安排", "开吧", "启动吧", "设吧", "就这样",
)
REJECT_PATTERNS = (
    "不用", "不要", "算了", "别", "先不", "不用了", "不要了", "取消",
)


@dataclass
class PendingIntent:
    skill_name: str
    arguments: JsonDict
    source_user_text: str = ""
    assistant_text: str = ""
    created_at: str = field(default_factory=lambda: datetime.datetime.now().isoformat(timespec="seconds"))
    ttl_seconds: int = 300
    intent_id: str = field(default_factory=lambda: uuid.uuid4().hex)

    def expired(self) -> bool:
        try:
            created = datetime.datetime.fromisoformat(self.created_at)
        except Exception:
            return True
        return (datetime.datetime.now() - created).total_seconds() > self.ttl_seconds

    def to_dict(self) -> JsonDict:
        return {
            "intent_id": self.intent_id,
            "skill_name": self.skill_name,
            "arguments": dict(self.arguments or {}),
            "source_user_text": self.source_user_text,
            "assistant_text": self.assistant_text,
            "created_at": self.created_at,
            "ttl_seconds": self.ttl_seconds,
        }


def _norm(text: str) -> str:
    return re.sub(r"\s+", "", str(text or "").strip().lower())


def is_rejection(text: str) -> bool:
    s = _norm(text)
    return bool(s and any(p in s for p in REJECT_PATTERNS))


def is_confirmation(text: str) -> bool:
    s = _norm(text)
    if not s or is_rejection(s):
        return False
    if len(s) <= 8 and any(s.startswith(p) for p in ("好", "嗯", "行", "可以", "对", "是")):
        return True
    return any(p in s for p in CONFIRM_PATTERNS)


def _looks_like_assistant_asking_todo(reply_text: str) -> bool:
    s = str(reply_text or "")
    return (
        any(w in s for w in ("待办", "清单", "记下", "记成", "提醒"))
        and any(w in s for w in ("要不要", "要不", "可以帮你", "帮你记", "需要的话"))
    )


def _looks_like_assistant_asking_timer(reply_text: str) -> bool:
    s = str(reply_text or "")
    return (
        any(w in s for w in ("计时", "定时", "倒计时", "番茄钟", "专注"))
        and any(w in s for w in ("要不要", "要不", "可以帮你", "帮你开", "需要的话"))
    )


def _todo_category(text: str) -> str:
    s = str(text or "")
    if any(w in s for w in ("学习", "复习", "背", "作业", "考试", "论文", "课程", "单词", "六级")):
        return "study"
    if any(w in s for w in ("买", "洗", "吃", "饭", "药", "收拾", "快递")):
        return "life"
    return "other"


def _clean_todo_text(text: str) -> str:
    s = str(text or "").strip()
    s = re.sub(r"^(我|俺|咱|今天|明天|后天|等会儿|等会|一会儿|一会)\s*", "", s)
    s = re.sub(r"(了|啦|吧|啊|呀|呢)[。.!！?？]*$", "", s)
    s = re.sub(r"\s+", " ", s).strip(" ，,。.!！?？")
    if len(s) > 36:
        s = s[:36].rstrip()
    return s


def build_pending_intent_from_reply(user_text: str, assistant_text: str) -> Optional[PendingIntent]:
    user_text = str(user_text or "").strip()
    assistant_text = str(assistant_text or "").strip()
    if not user_text or not assistant_text:
        return None

    if _looks_like_assistant_asking_todo(assistant_text):
        todo_text = _clean_todo_text(user_text)
        if len(todo_text) >= 3 and not is_confirmation(todo_text):
            return PendingIntent(
                skill_name="todo.add",
                arguments={
                    "text": todo_text,
                    "priority": "medium",
                    "category": _todo_category(todo_text),
                    "due_date": "",
                    "tags": ["对话承接"],
                },
                source_user_text=user_text,
                assistant_text=assistant_text,
            )

    if _looks_like_assistant_asking_timer(assistant_text):
        return PendingIntent(
            skill_name="timer.start",
            arguments={"seconds": 25 * 60, "label": "专注"},
            source_user_text=user_text,
            assistant_text=assistant_text,
        )
    return None


class PendingIntentStore:
    def __init__(self):
        self._intent: Optional[PendingIntent] = None
        self._last_resolution: str = ""

    def set(self, intent: PendingIntent) -> None:
        self._intent = intent
        self._last_resolution = "pending"

    def clear(self, reason: str = "") -> None:
        self._intent = None
        self._last_resolution = reason or "cleared"

    def current(self) -> Optional[PendingIntent]:
        if self._intent and self._intent.expired():
            self.clear("expired")
        return self._intent

    def resolve_user_text(self, user_text: str) -> Tuple[str, Optional[PendingIntent]]:
        intent = self.current()
        if not intent:
            return "none", None
        if is_rejection(user_text):
            self.clear("rejected")
            return "rejected", intent
        if is_confirmation(user_text):
            self.clear("confirmed")
            return "confirmed", intent
        return "pending", None
