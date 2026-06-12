"""Todo feature: A2UI tool routing, storage, homework sync, and UI window."""

import datetime
import json
import os
import re
import threading
import time
import uuid

import requests
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from PyQt5.QtCore import QObject, QPoint, QThread, QTimer, Qt, pyqtSignal
from PyQt5.QtGui import QColor, QFont
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFrame,
    QGraphicsDropShadowEffect,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from pet_core.config import app_config
from pet_core.persona import format_capability_registry_for_prompt


def has_explicit_todo_write_intent(text):
    s = str(text or "").strip()
    if not s:
        return False
    explicit_markers = (
        "帮我记", "记一下", "记下", "加到待办", "加入待办", "写进待办",
        "添加待办", "新建待办", "提醒我", "到时候提醒", "设个提醒",
        "安排一下", "列个计划", "列到清单", "放进清单",
    )
    return any(marker in s for marker in explicit_markers)


# ==================== MCP 工具 / A2UI 协议适配层 ====================
# 说明：
#   - 本模块在桌宠进程内提供"MCP 风格"的工具注册表（in-process，非真实的 MCP transport），
#     供大模型在对话时按规则触发"add_todo"工具。
#   - 工具调用的产物用 A2UI v0.9 协议（https://a2ui.org/specification/v0.9-a2ui/）来描述
#     "对待办清单这块 UI surface 的更新"，由前端 TodoWindow 解释执行。
#   - 写入前会做"字面 + 语义"双重去重，避免重复待办。

A2UI_VERSION = "v0.9"
A2UI_CATALOG_ID = "https://yuzu.pet/1.0/todocatalog"
A2UI_SURFACE_TODO = "todo_list"


def get_mcp_tool_definitions():
    """返回 MCP 风格的工具描述列表。会被注入到工具路由 LLM 的 system prompt 里。"""
    return [
        {
            "name": "add_todo",
            "description": (
                "向用户的待办清单写入一条新待办。"
                "仅在用户**明确**表达想要做某事、记下某事、提醒自己、"
                "在某截止时间前完成某事时调用。"
                "不要为聊天闲聊、问候、纯提问、回忆过去的事调用。"
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "待办的核心内容,一句话概括"},
                    "priority": {
                        "type": "string",
                        "enum": ["low", "medium", "high"],
                        "description": "紧急程度,默认 medium",
                    },
                    "category": {
                        "type": "string",
                        "enum": ["work", "study", "life", "homework", "other"],
                        "description": "分类,默认 other",
                    },
                    "due_date": {
                        "type": "string",
                        "description": "截止时间,格式 YYYY-MM-DD 或 YYYY-MM-DD HH:MM,无可留空",
                    },
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "可选标签",
                    },
                },
                "required": ["text"],
            },
        },
        {
            "name": "start_focus_timer",
            "description": (
                "启动本地专注/倒计时定时器。仅当用户明确说出定时、计时、专注、番茄钟等意图，"
                "并且给出具体时长时调用。"
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "seconds": {
                        "type": "integer",
                        "description": "倒计时秒数，范围 5 到 86400",
                    },
                    "label": {
                        "type": "string",
                        "description": "定时器名称，例如 专注、背单词、休息",
                    },
                },
                "required": ["seconds"],
            },
        },
    ]


def _extract_first_json_object(text):
    """容错地从 LLM 输出里提取出第一个完整的 JSON 对象。"""
    if not text:
        return None
    s = text.strip()
    s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s*```\s*$", "", s)
    try:
        return json.loads(s)
    except Exception:
        pass
    depth = 0
    start = -1
    in_str = False
    esc = False
    for i, c in enumerate(s):
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
            continue
        if c == "{":
            if depth == 0:
                start = i
            depth += 1
        elif c == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start >= 0:
                    candidate = s[start:i + 1]
                    try:
                        return json.loads(candidate)
                    except Exception:
                        start = -1
    return None


# ---- 中央待办存储：A2UI 消息 / 工具调用都通过它落盘 + 通知 UI ----
class TodoStore(QObject):
    """A2UI 驱动的中央 Todo 存储。

    数据格式（新）：
        {
          "version": "1.0",
          "items": [
            {"id", "text", "completed", "priority", "category",
             "tags": [...], "endtime", "teacher", "is_homework",
             "created_at", "source"}
          ]
        }
    旧的纯 list 格式会被自动迁移。
    """

    DATA_FILE = "todo_data.json"
    DEDUP_SIM_THRESHOLD = 0.82

    todos_changed = pyqtSignal()
    todo_added = pyqtSignal(dict)

    PRIORITY_ORDER = {"high": 0, "medium": 1, "low": 2}

    def __init__(self):
        super().__init__()
        self.items = []
        # 关键：工具路由线程(后台) 与 主线程都会调用 add/remove/save，
        # 用一把可重入锁保证 self.items 读写、JSON 落盘原子化。
        self._lock = threading.RLock()
        self._load()

    # ----- IO -----
    def _load(self):
        if not os.path.exists(self.DATA_FILE):
            return
        try:
            with open(self.DATA_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"[TodoStore] 加载失败: {e}")
            return
        raw_items = []
        if isinstance(data, list):
            raw_items = data
        elif isinstance(data, dict):
            raw_items = data.get("items", []) or []
        out = []
        for it in raw_items:
            if isinstance(it, str):
                out.append(self._normalize({"text": it}))
            elif isinstance(it, dict):
                out.append(self._normalize(it))
        with self._lock:
            self.items = out

    def save(self):
        try:
            with self._lock:
                payload = {"version": "1.0", "items": list(self.items)}
            with open(self.DATA_FILE, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[TodoStore] 保存失败: {e}")

    def _normalize(self, raw):
        is_hw = bool(raw.get("is_homework", False))
        text = (raw.get("text") or "").strip()
        return {
            "id": raw.get("id") or f"todo_{uuid.uuid4().hex[:10]}",
            "text": text,
            "completed": bool(raw.get("completed", False)),
            "is_homework": is_hw,
            "teacher": raw.get("teacher", "") or "",
            "endtime": raw.get("endtime", "") or raw.get("due_date", "") or "",
            "priority": raw.get("priority") if raw.get("priority") in ("low", "medium", "high") else ("high" if is_hw else "medium"),
            "category": raw.get("category") or ("homework" if is_hw else "other"),
            "tags": list(raw["tags"]) if isinstance(raw.get("tags"), list) else [],
            "created_at": raw.get("created_at") or datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "source": raw.get("source") or ("homework" if is_hw else "user"),
        }

    # ----- 查询 -----
    def all(self):
        with self._lock:
            return list(self.items)

    @staticmethod
    def _norm_text(s):
        s = (s or "").strip().lower()
        return re.sub(r"\s+", "", s)

    def find_similar(self, text):
        """字面 + 模糊去重。返回相似的 item 或 None。"""
        try:
            from difflib import SequenceMatcher
        except Exception:
            SequenceMatcher = None
        target = self._norm_text(text)
        if not target:
            return None
        with self._lock:
            snapshot = list(self.items)
        for it in snapshot:
            existing = self._norm_text(it.get("text"))
            if not existing:
                continue
            if existing == target:
                return it
            # 一方完整包含另一方且长度差不大 → 视为同一条
            if (target in existing or existing in target) and abs(len(existing) - len(target)) <= max(4, int(len(target) * 0.3)):
                return it
            if SequenceMatcher is not None:
                try:
                    ratio = SequenceMatcher(None, target, existing).ratio()
                except Exception:
                    ratio = 0.0
                if ratio >= self.DEDUP_SIM_THRESHOLD:
                    return it
        return None

    # ----- 写入 -----
    def add(self, raw_item, dedup=True):
        """添加一条 todo。返回 (added: bool, item_or_dup: dict)。"""
        item = self._normalize(raw_item)
        if not item["text"]:
            return False, None
        with self._lock:
            if dedup:
                dup = self.find_similar(item["text"])
                if dup is not None:
                    return False, dup
            self.items.append(item)
        self.save()
        self.todo_added.emit(item)
        self.todos_changed.emit()
        return True, item

    def remove(self, todo_id):
        with self._lock:
            n = len(self.items)
            self.items = [it for it in self.items if it.get("id") != todo_id]
            changed = len(self.items) != n
        if changed:
            self.save()
            self.todos_changed.emit()
        return changed

    def set_completed(self, todo_id, completed):
        changed = False
        with self._lock:
            for it in self.items:
                if it.get("id") == todo_id:
                    if bool(it.get("completed")) != bool(completed):
                        it["completed"] = bool(completed)
                        changed = True
                    break
        if changed:
            self.save()
            self.todos_changed.emit()
            return True
        return False

    def clear_completed(self):
        with self._lock:
            before = len(self.items)
            self.items = [it for it in self.items if not it.get("completed")]
            changed = len(self.items) != before
        if changed:
            self.save()
            self.todos_changed.emit()

    # ----- A2UI 消息解释执行 -----
    def apply_a2ui_message(self, msg):
        """解释一条 A2UI v0.9 消息，返回真正落库的 items 列表。

        我们关心的两类 update：
        - updateDataModel: surfaceId=todo_list, path=/items/-, value=dict 或 list[dict]
        - updateComponents: 仅用于显示，不影响数据；忽略。
        其他类型（createSurface/deleteSurface）忽略。
        """
        if not isinstance(msg, dict):
            return []
        added = []
        # 兼容批量包装
        if "messages" in msg and isinstance(msg["messages"], list):
            for m in msg["messages"]:
                added.extend(self.apply_a2ui_message(m))
            return added

        udm = msg.get("updateDataModel")
        if isinstance(udm, dict):
            if udm.get("surfaceId") in (None, A2UI_SURFACE_TODO):
                path = udm.get("path", "/items/-")
                value = udm.get("value")
                values = []
                if isinstance(value, list):
                    values = [v for v in value if isinstance(v, dict)]
                elif isinstance(value, dict):
                    if path in ("/items", "/"):
                        # 整个数据模型被替换的情况：把 items 取出来
                        nested = value.get("items")
                        if isinstance(nested, list):
                            values = [v for v in nested if isinstance(v, dict)]
                        else:
                            values = [value]
                    else:
                        values = [value]
                for v in values:
                    ok, item = self.add(v, dedup=True)
                    if ok and item is not None:
                        added.append(item)
        return added


# 全局单例
_todo_store_instance = None


def todo_store():
    global _todo_store_instance
    if _todo_store_instance is None:
        _todo_store_instance = TodoStore()
    return _todo_store_instance


# ==================== Todo 工具路由线程 ====================
class TodoToolRouterThread(QThread):
    """独立的 LLM 工具路由线程：
    - 输入：用户最近一次输入 + 当前 todo 清单
    - 输出：(added_items, skipped_items)，已经写入 TodoStore
    """

    result_signal = pyqtSignal(list, list)
    error_signal = pyqtSignal(str)

    def __init__(self, user_text, store):
        super().__init__()
        self.user_text = (user_text or "").strip()
        self.store = store

    def run(self):
        try:
            if not self.user_text:
                self.result_signal.emit([], [])
                return
            if not has_explicit_todo_write_intent(self.user_text):
                self.result_signal.emit([], [])
                return

            todo_tools = [
                tool for tool in get_mcp_tool_definitions()
                if tool.get("name") == "add_todo"
            ]
            tools_json = json.dumps(todo_tools, ensure_ascii=False, indent=2)
            existing = [
                {"text": it["text"], "category": it.get("category", "")}
                for it in self.store.items
                if not it.get("completed")
            ][-30:]
            existing_json = json.dumps(existing, ensure_ascii=False, indent=2)
            today_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M (%A)")

            sys_prompt = f"""你是一个 MCP 风格的工具路由器，专门判断"用户的最新输入是否需要调用工具"，并按 A2UI v0.9 协议输出 UI 更新消息。

当前时间: {today_str}

【工具能力边界】:
{format_capability_registry_for_prompt()}

【可用工具(MCP inputSchema 风格)】:
{tools_json}

【当前用户未完成的待办清单(用于去重判断)】:
{existing_json}

【A2UI v0.9 写入约定】:
- catalogId = "{A2UI_CATALOG_ID}"
- surfaceId = "{A2UI_SURFACE_TODO}"
- 每新增一条 todo, 对应一条 updateDataModel 消息, path 固定为 "/items/-", value 为一个 todo 对象。

【触发规则 - 必须严格遵守】:
1. 仅当用户明确表达"帮我记下/加到待办/提醒我/我要在某时间做/安排一下/接下来要做"等执行意图时, 才输出 add_todo 调用。
2. 愿望、祝愿、担心、泛泛目标、闲聊、问候、提问、感叹、回忆已经发生的事 → 不调用任何工具, 返回空列表。
3. 像"希望考试能过/想变好/要加油/但愿顺利"这类没有具体行动的句子不是待办。可以让主回复鼓励用户, 但工具路由必须返回空列表。
4. 用户描述的事项若与【当前未完成的待办清单】中**任意一条语义重复**(同义/包含关系/同一件事的不同说法), 不要再次添加。
5. 一次最多 3 条。每条 text 控制在 30 个字以内, 不要复述用户原话, 提炼出待办主干。
6. 如果用户提到"明天/后天/周一/下午三点"等相对时间, 请基于"当前时间"换算成绝对时间放入 due_date(格式 YYYY-MM-DD 或 YYYY-MM-DD HH:MM); 不确定就留空字符串。

【输出格式 - 必须是纯 JSON, 不要 markdown 围栏, 不要解释】:
没有需要添加的待办时, 严格输出:
{{"tool_calls": [], "a2ui_messages": []}}

需要添加时(示例, 字段必须齐全):
{{
  "tool_calls": [
    {{"name": "add_todo", "arguments": {{"text": "复习数据库 ER 图", "priority": "high", "category": "study", "due_date": "2026-05-15 22:00", "tags": ["期末"]}}}}
  ],
  "a2ui_messages": [
    {{
      "version": "{A2UI_VERSION}",
      "updateDataModel": {{
        "surfaceId": "{A2UI_SURFACE_TODO}",
        "path": "/items/-",
        "value": {{
          "text": "复习数据库 ER 图",
          "priority": "high",
          "category": "study",
          "due_date": "2026-05-15 22:00",
          "tags": ["期末"],
          "source": "agent"
        }}
      }}
    }}
  ]
}}"""

            tool_llm = ChatOpenAI(
                model=app_config.get("ark.model_tool", "") or "doubao-seed-2-0-mini-260428",
                openai_api_key=app_config.get("ark.api_key", "") or "",
                openai_api_base=app_config.get("ark.base_url", "") or "https://ark.cn-beijing.volces.com/api/v3",
                max_tokens=600,
                temperature=0.0,
                model_kwargs={"extra_body": {"thinking": {"type": "disabled"}}},
            )
            sys_message = SystemMessage(content=sys_prompt)
            user_message = HumanMessage(content=f"用户输入: {self.user_text}")

            t0 = time.perf_counter()
            resp = tool_llm.invoke([sys_message, user_message])
            raw = (resp.content or "").strip()
            print(f"[ToolRouter][{time.perf_counter()-t0:.2f}s] raw={raw[:600]}")

            parsed = _extract_first_json_object(raw)
            if not parsed:
                self.result_signal.emit([], [])
                return

            messages = parsed.get("a2ui_messages") or []
            tool_calls = parsed.get("tool_calls") or []

            # 兜底：LLM 只给了 tool_calls 没给 a2ui_messages，本地补
            if not messages and tool_calls:
                for tc in tool_calls:
                    if not isinstance(tc, dict):
                        continue
                    if tc.get("name") != "add_todo":
                        continue
                    args = tc.get("arguments") or {}
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except Exception:
                            args = {}
                    if not isinstance(args, dict):
                        continue
                    messages.append({
                        "version": A2UI_VERSION,
                        "updateDataModel": {
                            "surfaceId": A2UI_SURFACE_TODO,
                            "path": "/items/-",
                            "value": {
                                "text": (args.get("text") or "").strip(),
                                "priority": args.get("priority", "medium"),
                                "category": args.get("category", "other"),
                                "endtime": args.get("due_date", ""),
                                "tags": args.get("tags", []) if isinstance(args.get("tags"), list) else [],
                                "source": "agent",
                            },
                        },
                    })

            added_items = []
            for m in messages:
                added_items.extend(self.store.apply_a2ui_message(m))

            # 计算被跳过(LLM 想加但因去重被拦下)
            skipped_items = []
            wanted_texts = []
            for tc in tool_calls:
                if isinstance(tc, dict) and tc.get("name") == "add_todo":
                    args = tc.get("arguments")
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except Exception:
                            args = {}
                    if isinstance(args, dict):
                        t = (args.get("text") or "").strip()
                        if t:
                            wanted_texts.append(t)
            added_norm = {self.store._norm_text(it["text"]) for it in added_items}
            for t in wanted_texts:
                if self.store._norm_text(t) not in added_norm:
                    dup = self.store.find_similar(t)
                    if dup is not None:
                        skipped_items.append({"text": t, "reason": f"与已有「{dup['text']}」重复"})

            self.result_signal.emit(added_items, skipped_items)
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.error_signal.emit(str(e))


# ==================== To-Do List 窗口 ====================
class HomeworkFetcherThread(QThread):
    finished_signal = pyqtSignal(list)
    error_signal = pyqtSignal(str)

    def run(self):
        try:
            # 优学院的几条 URL + 账号密码全部来自 AppConfig；都没填就直接报错给 UI。
            homework_url = app_config.get("ucollege.homework_url", "") or ""
            clas         = app_config.get("ucollege.course_list_url", "") or ""
            lg_url       = app_config.get("ucollege.login_url", "") or ""
            login_name   = app_config.get("ucollege.login_name", "") or ""
            login_pwd    = app_config.get("ucollege.password", "") or ""
            if not (homework_url and clas and lg_url and login_name and login_pwd):
                self.error_signal.emit("优学院相关字段未填全。请右键桌宠 → ⚙ 设置 → 优学院作业拉取。")
                return

            session = requests.Session()
            lg_data = {
                "loginName": login_name,
                "password": login_pwd,
            }
            login_response = session.post(lg_url, data=lg_data)
            
            if login_response.status_code == 200:
                token = session.cookies.get("token")
                headers = {"Authorization": token}
                
                res = session.get(clas, headers=headers)
                course_list = res.json().get("courseList", [])
                
                all_homeworks = []
                for course in course_list:
                    cid = course["id"]
                    params = {"ocId": cid, "pn": 1, "ps": 10, "Lang": "zh"}
                    howurl = homework_url.format(course_id=cid)
                    homework_response = session.get(howurl, headers=headers, params=params)
                    data = homework_response.json()
                    homework_list = data.get("homeworkList", [])
                    
                    for hw in homework_list:
                        status = hw.get("status")
                        if status in [1, 2]:  # 1: 未开始, 2: 未提交
                            title = hw.get("homeworkTitle", "未命名作业")
                            teacher = hw.get("publisher", "未知")
                            endtime = hw.get("endTime", 0) / 1000
                            if endtime > 0:
                                dt_utc = datetime.datetime.fromtimestamp(endtime, tz=datetime.timezone.utc)
                                dt_beijing = dt_utc.astimezone(datetime.timezone(datetime.timedelta(hours=8)))
                                formatted_time = dt_beijing.strftime("%Y-%m-%d %H:%M:%S")
                            else:
                                formatted_time = "无截止时间"
                            
                            all_homeworks.append({
                                "text": title,
                                "is_homework": True,
                                "teacher": teacher,
                                "endtime": formatted_time,
                                "completed": False,
                                "priority": "high",
                                "category": "homework",
                                "source": "homework",
                            })
                
                self.finished_signal.emit(all_homeworks)
            else:
                self.error_signal.emit("登录失败")
        except Exception as e:
            self.error_signal.emit(str(e))


# ---- 单条 Todo 卡片（新版 UI 的核心可视化组件） ----
class TodoItemCard(QFrame):
    """新 UI 中每一条待办都是一张可缩放的卡片，承载：
    - 复选框（完成态）
    - 优先级竖条（颜色编码）
    - 分类徽标 / 标签 chip
    - 截止时间 / 任课老师 副信息
    - 删除按钮
    点击卡片本身（非按钮区）会切换 completed 状态。
    """

    PRIORITY_COLORS = {
        "high":   "#9FB7CC",
        "medium": "#6F879B",
        "low":    "#46586B",
    }
    CATEGORY_META = {
        "work":     ("工作", "#9FB7CC"),
        "study":    ("学习", "#B7A6D8"),
        "life":     ("生活", "#A8AEB8"),
        "homework": ("作业", "#C6D4E2"),
        "other":    ("其他", "#8F9AAA"),
    }
    SOURCE_BADGE = {
        "agent":    ("AI", "#9FB7CC"),
        "homework": ("同步", "#B7A6D8"),
    }

    completed_changed = pyqtSignal(str, bool)
    delete_requested = pyqtSignal(str)

    def __init__(self, item, parent=None):
        super().__init__(parent)
        self.item = item
        self.setObjectName("todoCard")
        self._build_ui()

    def _build_ui(self):
        item = self.item
        priority = item.get("priority", "medium")
        completed = bool(item.get("completed"))
        category = item.get("category", "other")
        is_hw = bool(item.get("is_homework"))

        border_color = self.PRIORITY_COLORS.get(priority, "#6F879B")
        bg = "#090A0D" if completed else "#050607"

        self.setStyleSheet(f"""
            QFrame#todoCard {{
                background-color: {bg};
                border: 1px solid {border_color};
                border-radius: 8px;
            }}
            QFrame#todoCard:hover {{
                background-color: #0B1018;
                border-color: #9FB7CC;
            }}
        """)

        outer = QHBoxLayout(self)
        outer.setContentsMargins(10, 9, 8, 9)
        outer.setSpacing(9)

        # ----- 复选框 -----
        self.checkbox = QCheckBox()
        self.checkbox.setChecked(completed)
        self.checkbox.setStyleSheet(f"""
            QCheckBox {{ background: transparent; spacing: 0; }}
            QCheckBox::indicator {{
                width: 18px; height: 18px;
                border-radius: 9px;
                border: 1px solid #9FB7CC;
                background: #050607;
            }}
            QCheckBox::indicator:checked {{
                background-color: #34224A;
                border-color: #B7A6D8;
                image: none;
            }}
        """)
        self.checkbox.toggled.connect(self._on_toggled)
        cb_wrap = QVBoxLayout()
        cb_wrap.setContentsMargins(0, 2, 0, 0)
        cb_wrap.addWidget(self.checkbox)
        cb_wrap.addStretch()
        outer.addLayout(cb_wrap)

        # ----- 文字 + 元信息 -----
        text_layout = QVBoxLayout()
        text_layout.setSpacing(3)
        text_layout.setContentsMargins(0, 0, 0, 0)

        # 标题
        text_color = "#6F7783" if completed else "#F1EBDD"
        self.title_label = QLabel(item.get("text", ""))
        self.title_label.setWordWrap(True)
        self.title_label.setStyleSheet(
            f"font-size: 13px; font-weight: 500; color: {text_color}; "
            f"background: transparent; "
            f"{'text-decoration: line-through;' if completed else ''}"
        )
        text_layout.addWidget(self.title_label)

        # 元信息行（分类徽标 / 来源 / 标签）
        chips_row = QHBoxLayout()
        chips_row.setSpacing(4)
        chips_row.setContentsMargins(0, 0, 0, 0)

        cat_label, cat_color = self.CATEGORY_META.get(category, self.CATEGORY_META["other"])
        chips_row.addWidget(self._make_chip(cat_label, cat_color))

        src = item.get("source")
        if src in self.SOURCE_BADGE:
            sb_label, sb_color = self.SOURCE_BADGE[src]
            chips_row.addWidget(self._make_chip(sb_label, sb_color))

        for tag in (item.get("tags") or [])[:3]:
            chips_row.addWidget(self._make_chip(f"#{tag}", "#9BA3AD"))

        chips_row.addStretch()
        if any([category, src in self.SOURCE_BADGE, item.get("tags")]):
            text_layout.addLayout(chips_row)

        # 截止/老师 副信息
        detail_parts = []
        if is_hw and item.get("teacher"):
            detail_parts.append(f"教师 {item['teacher']}")
        if item.get("endtime"):
            detail_parts.append(f"截止 {item['endtime']}")
        if detail_parts:
            details_label = QLabel("  ·  ".join(detail_parts))
            details_label.setWordWrap(True)
            details_label.setStyleSheet(
                "color: #A8AEB8; font-size: 11px; background: transparent;"
            )
            text_layout.addWidget(details_label)

        outer.addLayout(text_layout, 1)

        # ----- 删除按钮 -----
        self.del_btn = QPushButton("×")
        self.del_btn.setFixedSize(26, 26)
        self.del_btn.setCursor(Qt.PointingHandCursor)
        self.del_btn.setStyleSheet("""
            QPushButton {
                background: transparent;
                border: 1px solid transparent;
                border-radius: 13px;
                font-size: 16px;
                color: #6F879B;
            }
            QPushButton:hover {
                background-color: #20162E;
                color: #F1EBDD;
                border-color: #775E90;
            }
        """)
        self.del_btn.clicked.connect(lambda: self.delete_requested.emit(self.item["id"]))
        del_wrap = QVBoxLayout()
        del_wrap.setContentsMargins(0, 0, 0, 0)
        del_wrap.addWidget(self.del_btn)
        del_wrap.addStretch()
        outer.addLayout(del_wrap)

    @staticmethod
    def _make_chip(text, color):
        lbl = QLabel(text)
        lbl.setStyleSheet(
            f"background-color: #0B1018;"
            f"color: {color};"
            f"border: 1px solid {color}55;"
            f"border-radius: 7px;"
            f"padding: 1px 7px;"
            f"font-size: 10px;"
            f"font-weight: 500;"
        )
        return lbl

    def _on_toggled(self, checked):
        self.completed_changed.emit(self.item["id"], bool(checked))

class TodoWindow(QWidget):
    """新版待办清单窗口。
    - 数据层走 TodoStore 单例（A2UI 消息和 LLM 工具调用都能直接写入）。
    - UI 升级：搜索、状态/分类过滤、卡片化每项、优先级颜色、分类徽标、标签 chip。
    - 拖拽：在顶部 header 区域按住才能移动，避免和列表/按钮冲突。
    """

    STATUS_FILTERS = [
        ("all",     "全部"),
        ("pending", "未完成"),
        ("done",    "已完成"),
        ("today",   "今日"),
    ]
    CATEGORY_FILTERS = [
        ("all",      "全部分类"),
        ("homework", "作业"),
        ("study",    "学习"),
        ("work",     "工作"),
        ("life",     "生活"),
        ("other",    "其他"),
    ]

    def __init__(self, pet=None):
        super().__init__()
        self.pet = pet
        self.store = todo_store()
        self.drag_offset = QPoint()
        self._drag_anchor = None
        self.current_status = "all"
        self.current_category = "all"
        self.search_text = ""
        self._connected = False
        self.init_ui()
        self._connect_store()
        self.refresh_list()

    # ----- Store 连接 / 断开 -----
    def _connect_store(self):
        if self._connected:
            return
        self.store.todos_changed.connect(self.refresh_list)
        self.store.todo_added.connect(self._on_todo_added_flash)
        self._connected = True

    def _disconnect_store(self):
        if not self._connected:
            return
        try:
            self.store.todos_changed.disconnect(self.refresh_list)
        except Exception:
            pass
        try:
            self.store.todo_added.disconnect(self._on_todo_added_flash)
        except Exception:
            pass
        self._connected = False

    def closeEvent(self, event):
        self._disconnect_store()
        super().closeEvent(event)

    # ----- UI 构建 -----
    def init_ui(self):
        self.setWindowTitle("有珠的待办清单")
        self.setFixedSize(400, 600)
        self.setWindowFlags(Qt.Window | Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground)

        # 外圆角容器 + 投影
        container = QWidget(self)
        container.setObjectName("rootContainer")
        container.setGeometry(10, 10, 380, 580)
        container.setStyleSheet("""
            QWidget#rootContainer {
                background-color: #050607;
                border: 1px solid #273546;
                border-radius: 12px;
            }
        """)
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(24)
        shadow.setColor(QColor(0, 0, 0, 150))
        shadow.setOffset(0, 8)
        container.setGraphicsEffect(shadow)

        root = QVBoxLayout(container)
        root.setContentsMargins(16, 14, 16, 14)
        root.setSpacing(9)

        # ============== 顶部标题栏 ==============
        self.header = QWidget()
        self.header.setObjectName("titleBar")
        self.header.setFixedHeight(36)
        self.header.setStyleSheet("QWidget#titleBar { background: transparent; }")
        title_layout = QHBoxLayout(self.header)
        title_layout.setContentsMargins(2, 0, 0, 0)
        title_layout.setSpacing(6)

        title = QLabel("我的待办")
        title.setFont(QFont("Microsoft YaHei", 15, QFont.Light))
        title.setStyleSheet("color: #9FB7CC; background: transparent; letter-spacing: 0px;")
        title_layout.addWidget(title)
        title_layout.addStretch()

        self.sync_btn = QPushButton("获取作业")
        self.sync_btn.setFixedSize(82, 28)
        self.sync_btn.setCursor(Qt.PointingHandCursor)
        self.sync_btn.setStyleSheet("""
            QPushButton {
                background-color: #07090D;
                color: #F1EBDD;
                border: 1px solid #46586B;
                border-radius: 7px;
                font-size: 12px;
                font-weight: 500;
            }
            QPushButton:hover {
                background-color: #101724;
                border-color: #9FB7CC;
            }
        """)
        self.sync_btn.clicked.connect(self.sync_homework)
        title_layout.addWidget(self.sync_btn)

        close_btn = QPushButton("关闭")
        close_btn.setFixedSize(48, 28)
        close_btn.setCursor(Qt.PointingHandCursor)
        close_btn.setStyleSheet("""
            QPushButton {
                background-color: #07090D;
                color: #F1EBDD;
                border: 1px solid #46586B;
                border-radius: 7px;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #20162E;
                border-color: #775E90;
            }
        """)
        close_btn.clicked.connect(self.close)
        title_layout.addWidget(close_btn)
        root.addWidget(self.header)

        # ============== 概览卡片（动态统计） ==============
        self.overview_label = QLabel()
        self.overview_label.setWordWrap(True)
        self.overview_label.setStyleSheet("""
            background: transparent;
            color: #C6D4E2;
            padding: 2px 2px;
            font-size: 12px;
            font-weight: 400;
        """)
        root.addWidget(self.overview_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setFixedHeight(5)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                background-color: #10131A;
                border: none;
                border-radius: 2px;
            }
            QProgressBar::chunk {
                border-radius: 2px;
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #34224A, stop:1 #9FB7CC);
            }
        """)
        root.addWidget(self.progress_bar)

        # ============== 输入行 ==============
        input_layout = QHBoxLayout()
        input_layout.setSpacing(6)
        self.input_field = QLineEdit()
        self.input_field.setPlaceholderText("添加新的待办事项")
        self.input_field.setStyleSheet("""
            QLineEdit {
                border: 1px solid #6F879B;
                border-radius: 7px;
                padding: 9px 12px;
                font-size: 13px;
                background-color: #07090D;
                color: #F1EBDD;
                selection-background-color: #34224A;
            }
            QLineEdit:focus { border-color: #9FB7CC; background-color: #090C12; }
        """)
        self.input_field.returnPressed.connect(self.add_todo_from_input)
        input_layout.addWidget(self.input_field, 1)

        self.priority_combo = QComboBox()
        self.priority_combo.addItem("高", "high")
        self.priority_combo.addItem("中", "medium")
        self.priority_combo.addItem("低", "low")
        self.priority_combo.setCurrentIndex(1)
        self.priority_combo.setFixedHeight(38)
        self.priority_combo.setStyleSheet("""
            QComboBox {
                background-color: #07090D;
                color: #F1EBDD;
                border: 1px solid #6F879B;
                border-radius: 7px;
                padding: 0 8px;
                font-size: 12px;
                min-width: 64px;
            }
            QComboBox:focus { border-color: #9FB7CC; }
            QComboBox::drop-down { width: 16px; border: none; }
        """)
        input_layout.addWidget(self.priority_combo)

        add_btn = QPushButton("+")
        add_btn.setFixedSize(38, 38)
        add_btn.setCursor(Qt.PointingHandCursor)
        add_btn.setStyleSheet("""
            QPushButton {
                background-color: #20162E;
                color: #F1EBDD;
                border: 1px solid #775E90;
                border-radius: 7px;
                font-size: 22px;
                font-weight: 400;
            }
            QPushButton:hover { background-color: #34224A; border-color: #B7A6D8; }
        """)
        add_btn.clicked.connect(self.add_todo_from_input)
        input_layout.addWidget(add_btn)
        root.addLayout(input_layout)

        # ============== 搜索 + 分类 ==============
        search_row = QHBoxLayout()
        search_row.setSpacing(6)
        self.search_field = QLineEdit()
        self.search_field.setPlaceholderText("搜索待办 / 标签")
        self.search_field.setStyleSheet("""
            QLineEdit {
                border: 1px solid #6F879B;
                border-radius: 7px;
                padding: 6px 10px;
                font-size: 12px;
                background-color: #07090D;
                color: #F1EBDD;
                selection-background-color: #34224A;
            }
            QLineEdit:focus { border-color: #9FB7CC; background-color: #090C12; }
        """)
        self.search_field.textChanged.connect(self._on_search_changed)
        search_row.addWidget(self.search_field, 1)

        self.category_combo = QComboBox()
        for key, label in self.CATEGORY_FILTERS:
            self.category_combo.addItem(label, key)
        self.category_combo.setStyleSheet("""
            QComboBox {
                background-color: #07090D;
                color: #F1EBDD;
                border: 1px solid #6F879B;
                border-radius: 7px;
                padding: 4px 8px;
                font-size: 12px;
                min-width: 80px;
            }
            QComboBox:focus { border-color: #9FB7CC; }
            QComboBox::drop-down { width: 16px; border: none; }
        """)
        self.category_combo.currentIndexChanged.connect(self._on_category_changed)
        search_row.addWidget(self.category_combo)
        root.addLayout(search_row)

        # ============== 状态过滤 tab pills ==============
        tab_row = QHBoxLayout()
        tab_row.setSpacing(4)
        self._status_btns = {}
        for key, label in self.STATUS_FILTERS:
            btn = QPushButton(label)
            btn.setCursor(Qt.PointingHandCursor)
            btn.setCheckable(True)
            btn.setFixedHeight(26)
            btn.clicked.connect(lambda _checked, k=key: self._set_status(k))
            tab_row.addWidget(btn)
            self._status_btns[key] = btn
        tab_row.addStretch()
        self.clear_done_btn = QPushButton("清理已完成")
        self.clear_done_btn.setCursor(Qt.PointingHandCursor)
        self.clear_done_btn.setFixedHeight(26)
        self.clear_done_btn.setStyleSheet("""
            QPushButton {
                background-color: transparent;
                color: #A8AEB8;
                border: none;
                font-size: 11px;
            }
            QPushButton:hover { color: #F1EBDD; text-decoration: underline; }
        """)
        self.clear_done_btn.clicked.connect(self._on_clear_done)
        tab_row.addWidget(self.clear_done_btn)
        root.addLayout(tab_row)
        self._apply_status_styles()

        # ============== 卡片列表（滚动区） ==============
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFrameShape(QFrame.NoFrame)
        self.scroll_area.setStyleSheet("""
            QScrollArea { background: transparent; border: none; }
            QScrollArea > QWidget > QWidget { background: transparent; }
            QScrollBar:vertical {
                background: transparent; width: 6px; margin: 0;
            }
            QScrollBar::handle:vertical {
                background: #46586B; border-radius: 3px; min-height: 24px;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { background: none; border: none; }
        """)
        self.list_container = QWidget()
        self.list_container.setStyleSheet("background: transparent;")
        self.list_layout = QVBoxLayout(self.list_container)
        self.list_layout.setContentsMargins(2, 2, 6, 2)
        self.list_layout.setSpacing(6)
        self.list_layout.addStretch()
        self.scroll_area.setWidget(self.list_container)
        root.addWidget(self.scroll_area, 1)

        # ============== 底部 stats ==============
        self.stats_label = QLabel("还没有待办事项哦~")
        self.stats_label.setAlignment(Qt.AlignCenter)
        self.stats_label.setStyleSheet("color: #A8AEB8; font-size: 11px; background: transparent;")
        root.addWidget(self.stats_label)

    # ----- 状态过滤按钮样式 -----
    def _apply_status_styles(self):
        for key, btn in self._status_btns.items():
            active = key == self.current_status
            btn.setChecked(active)
            if active:
                btn.setStyleSheet("""
                    QPushButton {
                        background-color: #20162E;
                        color: #F1EBDD;
                        border: 1px solid #775E90;
                        border-radius: 7px;
                        padding: 0 12px;
                        font-size: 11px;
                        font-weight: 500;
                    }
                """)
            else:
                btn.setStyleSheet("""
                    QPushButton {
                        background-color: transparent;
                        color: #A8AEB8;
                        border: 1px solid #46586B;
                        border-radius: 7px;
                        padding: 0 12px;
                        font-size: 11px;
                    }
                    QPushButton:hover {
                        border-color: #9FB7CC;
                        color: #F1EBDD;
                    }
                """)

    def _set_status(self, key):
        self.current_status = key
        self._apply_status_styles()
        self.refresh_list()

    def _on_category_changed(self, _idx):
        self.current_category = self.category_combo.currentData() or "all"
        self.refresh_list()

    def _on_search_changed(self, text):
        self.search_text = (text or "").strip().lower()
        self.refresh_list()

    def _on_clear_done(self):
        self.store.clear_completed()
        if self.pet:
            self.pet.show_bubble("打扫干净啦~", duration=2500)

    # ----- 数据 → UI -----
    def _filtered_items(self):
        items = self.store.all()
        # 优先级排序：完成排后，再按优先级 high→low，最后按创建时间
        items.sort(key=lambda x: (
            bool(x.get("completed")),
            self.store.PRIORITY_ORDER.get(x.get("priority", "medium"), 9),
            x.get("created_at", ""),
        ))
        today = datetime.datetime.now().strftime("%Y-%m-%d")
        result = []
        for it in items:
            if self.current_status == "pending" and it.get("completed"):
                continue
            if self.current_status == "done" and not it.get("completed"):
                continue
            if self.current_status == "today":
                end = it.get("endtime") or ""
                if today not in end:
                    continue
            if self.current_category != "all" and it.get("category") != self.current_category:
                continue
            if self.search_text:
                blob = (it.get("text", "") + " " + " ".join(it.get("tags") or [])).lower()
                if self.search_text not in blob:
                    continue
            result.append(it)
        return result

    def refresh_list(self):
        # 清空旧卡片
        while self.list_layout.count() > 0:
            item = self.list_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.setParent(None)
                w.deleteLater()
        # 重新填充
        filtered = self._filtered_items()
        if not filtered:
            empty = QLabel(self._empty_text())
            empty.setAlignment(Qt.AlignCenter)
            empty.setWordWrap(True)
            empty.setStyleSheet(
                "color: #6F879B; font-size: 12px; padding: 28px 8px; background: transparent;"
            )
            self.list_layout.addWidget(empty)
        else:
            for it in filtered:
                card = TodoItemCard(it)
                card.completed_changed.connect(self._on_card_completed)
                card.delete_requested.connect(self._on_card_delete)
                self.list_layout.addWidget(card)
        self.list_layout.addStretch()
        self._update_stats()

    def _empty_text(self):
        if self.search_text:
            return "没有匹配的待办。"
        if self.current_status == "done":
            return "目前还没有完成的待办。"
        if self.current_status == "today":
            return "今天没有截止的待办。"
        if self.current_status == "pending":
            return "所有待办都完成了。"
        return "还没有待办事项。\n可以直接添加，也可以在聊天里让有珠记录。"

    def _update_stats(self):
        items = self.store.all()
        total = len(items)
        done = sum(1 for it in items if it.get("completed"))
        pending = total - done
        # 概览大卡片
        if total == 0:
            self.overview_label.setText("还没有待办。")
            self.progress_bar.setValue(0)
        else:
            ratio = int((done / total) * 100) if total else 0
            self.overview_label.setText(
                f"共 {total} 条 · 待完成 {pending} · 已完成 {done} · {ratio}%"
            )
            self.progress_bar.setValue(ratio)
        self.stats_label.setText(
            f"当前显示 {len(self._filtered_items())} 条 / 共 {total} 条 / 已完成 {done}"
        )

    # ----- 卡片回调 -----
    def _on_card_completed(self, todo_id, completed):
        self.store.set_completed(todo_id, completed)
        if completed and self.pet:
            self.pet.set_happy()
            self.pet.show_bubble("太棒了！完成一个！🎉", duration=3000)

    def _on_card_delete(self, todo_id):
        self.store.remove(todo_id)

    def _on_todo_added_flash(self, _item):
        # 简单视觉提示：滚动到底部
        QTimer.singleShot(80, lambda: self.scroll_area.verticalScrollBar().setValue(
            self.scroll_area.verticalScrollBar().maximum()
        ))

    # ----- 输入区 -> 添加 -----
    def add_todo_from_input(self):
        text = self.input_field.text().strip()
        if not text:
            return
        priority = self.priority_combo.currentData() or "medium"
        ok, item = self.store.add(
            {"text": text, "priority": priority, "category": "other", "source": "user"},
            dedup=True,
        )
        self.input_field.clear()
        if not ok and item is not None:
            # 命中已有的同义条目
            self.stats_label.setText(f"⚠️ 已有相似待办：「{item['text']}」")
            if self.pet:
                self.pet.show_bubble(f"诶, 已经记过这条啦~", duration=3000)
            return
        if self.pet and ok:
            self.pet.set_happy()
            self.pet.show_bubble("收到！加油完成它！✨", duration=3000)

    # ----- 作业同步 -----
    def sync_homework(self):
        self.sync_btn.setText("获取中...")
        self.sync_btn.setEnabled(False)
        self.fetcher_thread = HomeworkFetcherThread()
        self.fetcher_thread.finished_signal.connect(self.on_homework_fetched)
        self.fetcher_thread.error_signal.connect(self.on_homework_error)
        self.fetcher_thread.start()

    def on_homework_fetched(self, homeworks):
        self.sync_btn.setText("获取作业")
        self.sync_btn.setEnabled(True)
        added = 0
        for hw in homeworks:
            ok, _ = self.store.add(hw, dedup=True)
            if ok:
                added += 1
        if self.pet:
            if added > 0:
                self.pet.set_happy()
                self.pet.show_bubble(f"获取成功！新增了 {added} 个作业任务！📚")
            else:
                self.pet.show_bubble("获取成功！没有新的作业哦~ 🎉")

    def on_homework_error(self, error_msg):
        self.sync_btn.setText("获取作业")
        self.sync_btn.setEnabled(True)
        if self.pet:
            self.pet.show_bubble(f"获取失败：{error_msg} 😢")

    # ----- 拖拽：仅在 header 区域生效，避免和列表冲突 -----
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            if hasattr(self, "header") and self.header.geometry().contains(event.pos() - QPoint(10, 10)):
                self._drag_anchor = event.pos()
                self.drag_offset = event.pos()
            else:
                self._drag_anchor = None

    def mouseMoveEvent(self, event):
        if event.buttons() == Qt.LeftButton and self._drag_anchor is not None:
            self.move(self.mapToGlobal(event.pos() - self.drag_offset))

    def mouseReleaseEvent(self, event):
        self._drag_anchor = None

