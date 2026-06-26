"""Local recent conversation history used by the prompt builder."""

import datetime
import json
import os
import re
import threading


class ConversationHistory:
    """Keep the latest user/Alice turns on disk for short-term context."""

    DEFAULT_MAX_TURNS = 10
    GENERIC_SUMMARIES = {
        "建议用户短暂休息，不要硬撑",
        "建议用户把学习任务拆小并继续推进",
        "围绕饮食给了克制的关心和提醒",
        "说明了待办或提醒相关处理",
        "有珠已经回应过，但旧历史没有保留可用摘要",
    }

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
                source = self._normalize_source(item.get("source"), user_text)
                summary = str(item.get("assistant_summary", "")).strip()
                if source == "proactive" or summary in self.GENERIC_SUMMARIES:
                    summary = self._summarize_assistant_reply(assistant_text, source=source)
                if not user_text and not assistant_text:
                    continue
                normalized.append({
                    "user": user_text,
                    "assistant": assistant_text,
                    "assistant_summary": summary or self._summarize_assistant_reply(assistant_text, source=source),
                    "timestamp": timestamp,
                    "source": source,
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
    def _age_minutes(ts):
        dt = ConversationHistory._parse_ts(ts)
        if not dt:
            return None
        return max(0, int((datetime.datetime.now() - dt).total_seconds() // 60))

    @staticmethod
    def _normalize_source(source, user_text=""):
        source = str(source or "").strip().lower()
        if source in {"chat", "proactive", "system"}:
            return source
        if not str(user_text or "").strip():
            return "proactive"
        return "chat"

    @staticmethod
    def _is_proactive_turn(turn):
        if not isinstance(turn, dict):
            return False
        source = str(turn.get("source") or "").strip().lower()
        return source == "proactive" or not str(turn.get("user") or "").strip()

    @staticmethod
    def _dedupe_key(text):
        s = str(text or "").lower()
        s = re.sub(r"!\[.*?\]\(.*?\)", "", s)
        s = re.sub(r"[\s，。！？、,.!?；;：:\"'“”‘’（）()\[\]【】\-—_]+", "", s)
        return s[:80]

    @staticmethod
    def _shorten(text, limit):
        s = re.sub(r"\s+", " ", str(text or "").strip())
        if len(s) <= limit:
            return s
        return s[:limit].rstrip("，。,. ") + "..."

    @staticmethod
    def _summarize_assistant_reply(text, source="chat"):
        s = str(text or "").strip()
        if not s:
            return ""
        if s.startswith("[已回复"):
            return "有珠已经回应过，但旧历史没有保留可用摘要"
        s = re.sub(r"!\[.*?\]\(.*?\)", "", s)
        s = re.sub(r"\[[^\[\]]{0,20}\]$", "", s).strip()
        s = re.sub(r"\s+", " ", s)
        limit = 70 if source == "proactive" else 90
        return ConversationHistory._shorten(s, limit)

    def _save_locked(self):
        try:
            with open(self.file_path, "w", encoding="utf-8") as f:
                json.dump(self._turns, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[ConversationHistory] 保存历史失败: {e}")

    def add_turn(self, user_text, assistant_text, source="chat"):
        user_text = str(user_text or "").strip()
        assistant_text = str(assistant_text or "").strip()
        if not user_text and not assistant_text:
            return
        source = self._normalize_source(source, user_text)
        with self._lock:
            self._turns.append({
                "user": user_text,
                "assistant": assistant_text,
                "assistant_summary": self._summarize_assistant_reply(assistant_text, source=source),
                "timestamp": self._now_iso(),
                "source": source,
            })
            if len(self._turns) > self.max_turns:
                self._turns = self._turns[-self.max_turns:]
            self._save_locked()

    def get_turns(self):
        with self._lock:
            return list(self._turns)

    def format_for_prompt(self, max_user_turns=6, max_proactive_turns=2):
        """Format history for prompts without copying old assistant wording verbatim."""
        with self._lock:
            turns = list(self._turns)
        if not turns:
            return "无"

        user_blocks = []
        seen_user_keys = set()
        seen_response_keys = set()
        proactive_records = []
        seen_proactive_keys = set()

        for turn in turns:
            if not isinstance(turn, dict):
                continue
            when = self._relative_time_label(turn.get("timestamp"))
            user_text = self._shorten(turn.get("user", ""), 160)
            assistant_text = self._shorten(turn.get("assistant", ""), 90)
            summary = self._shorten(
                turn.get("assistant_summary") or self._summarize_assistant_reply(
                    turn.get("assistant", ""),
                    source=turn.get("source") or "chat",
                ),
                110,
            )

            if self._is_proactive_turn(turn):
                age_minutes = self._age_minutes(turn.get("timestamp"))
                if age_minutes is not None and age_minutes > 6 * 60:
                    continue
                proactive_text = assistant_text or summary
                key = self._dedupe_key(proactive_text)
                if proactive_text and key and key not in seen_proactive_keys:
                    seen_proactive_keys.add(key)
                    proactive_records.append(
                        f"[{when}] 已主动说过：{proactive_text}"
                    )
                continue

            if not user_text:
                continue
            user_key = self._dedupe_key(user_text)
            if user_key in seen_user_keys:
                continue
            seen_user_keys.add(user_key)
            block = [f"[{when}] 用户：{user_text}"]
            response_key = self._dedupe_key(summary)
            if summary and response_key not in seen_response_keys:
                seen_response_keys.add(response_key)
                block.append(f"[{when}] 有珠回应要点：{summary}")
            user_blocks.append("\n".join(block))

        lines = []
        user_blocks = user_blocks[-max_user_turns:]
        proactive_records = proactive_records[-max_proactive_turns:]
        if user_blocks:
            lines.append("用户近期真实对话：")
            lines.extend(user_blocks)
        if proactive_records:
            if lines:
                lines.append("")
            lines.append("有珠近期主动关怀记录（只用于知道已经说过什么并避免复读，不要模仿这些原句）：")
            lines.extend(proactive_records)
        return "\n".join(lines)
