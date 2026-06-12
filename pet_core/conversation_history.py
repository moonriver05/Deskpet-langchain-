"""Local recent conversation history used by the prompt builder."""

import datetime
import json
import os
import re
import threading


class ConversationHistory:
    """Keep the latest user/Alice turns on disk for short-term context."""

    DEFAULT_MAX_TURNS = 10

    def __init__(self, file_path=None, max_turns=None, base_dir=None):
        if file_path is None:
            root = base_dir or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            file_path = os.path.join(root, "conversation_history.json")
        self.file_path = file_path
        self.max_turns = int(max_turns) if max_turns else self.DEFAULT_MAX_TURNS
        self._lock = threading.Lock()
        self._turns = self._load()

    def _load(self):
        if not os.path.exists(self.file_path):
            return []
        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, list):
                return []
            normalized = []
            for item in data[-self.max_turns:]:
                if not isinstance(item, dict):
                    continue
                user_text = str(item.get("user", "")).strip()
                assistant_text = str(item.get("assistant", "")).strip()
                timestamp = item.get("timestamp") or item.get("time") or ""
                summary = str(item.get("assistant_summary", "")).strip()
                if not user_text and not assistant_text:
                    continue
                normalized.append({
                    "user": user_text,
                    "assistant": assistant_text,
                    "assistant_summary": summary or self._summarize_assistant_reply(assistant_text),
                    "timestamp": timestamp,
                })
            return normalized
        except Exception as e:
            print(f"[ConversationHistory] 加载历史失败: {e}")
            return []

    @staticmethod
    def _now_iso():
        return datetime.datetime.now().isoformat(timespec="seconds")

    @staticmethod
    def _parse_ts(value):
        try:
            if not value:
                return None
            return datetime.datetime.fromisoformat(str(value))
        except Exception:
            return None

    @staticmethod
    def _relative_time_label(ts):
        dt = ConversationHistory._parse_ts(ts)
        if not dt:
            return "较早"
        seconds = max(0, int((datetime.datetime.now() - dt).total_seconds()))
        if seconds < 60:
            return "刚刚"
        minutes = seconds // 60
        if minutes < 60:
            return f"{minutes}分钟前"
        hours = minutes // 60
        if hours < 24:
            return f"{hours}小时前"
        days = hours // 24
        if days == 1:
            return "昨天"
        return f"{days}天前"

    @staticmethod
    def _summarize_assistant_reply(text):
        s = str(text or "").strip()
        if not s:
            return ""
        if s.startswith("[已回复"):
            return "有珠已经回应过，但旧历史没有保留可用摘要"
        s = re.sub(r"!\[.*?\]\(.*?\)", "", s)
        s = re.sub(r"\[[^\[\]]{0,20}\]$", "", s).strip()
        s = re.sub(r"\s+", " ", s)
        if any(k in s for k in ("休息", "眯", "睡", "困")):
            return "建议用户短暂休息，不要硬撑"
        if any(k in s for k in ("背单词", "复习", "学习", "做题")):
            return "建议用户把学习任务拆小并继续推进"
        if any(k in s for k in ("吃", "胃", "饭", "肠粉")):
            return "围绕饮食给了克制的关心和提醒"
        if any(k in s for k in ("待办", "记下", "提醒")):
            return "说明了待办或提醒相关处理"
        return s[:80]

    def _save_locked(self):
        try:
            with open(self.file_path, "w", encoding="utf-8") as f:
                json.dump(self._turns, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[ConversationHistory] 保存历史失败: {e}")

    def add_turn(self, user_text, assistant_text):
        user_text = str(user_text or "").strip()
        assistant_text = str(assistant_text or "").strip()
        if not user_text and not assistant_text:
            return
        with self._lock:
            self._turns.append({
                "user": user_text,
                "assistant": assistant_text,
                "assistant_summary": self._summarize_assistant_reply(assistant_text),
                "timestamp": self._now_iso(),
            })
            if len(self._turns) > self.max_turns:
                self._turns = self._turns[-self.max_turns:]
            self._save_locked()

    def get_turns(self):
        with self._lock:
            return list(self._turns)

    def format_for_prompt(self):
        """Format history for prompts without copying old assistant wording verbatim."""
        with self._lock:
            turns = list(self._turns)
        if not turns:
            return "无"
        lines = []
        for turn in turns:
            user_text = turn.get("user", "")
            summary = turn.get("assistant_summary") or self._summarize_assistant_reply(
                turn.get("assistant", "")
            )
            when = self._relative_time_label(turn.get("timestamp"))
            if user_text:
                lines.append(f"[{when}] 用户: {user_text}")
            if summary:
                lines.append(f"[{when}] 久远寺有珠回应摘要: {summary}")
        return "\n".join(lines)
