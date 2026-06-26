"""Manual labeling window for future preference/recommendation training."""

import copy
import datetime
import json
import os
import uuid

from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from pet_core.learning_logger import (
    FEEDBACK_EVENTS_PATH,
    LABELED_INTERACTIONS_PATH,
    LABEL_RESULTS_PATH,
    MANUAL_LABELS_PATH,
    RAW_INTERACTIONS_PATH,
    SCHEMA_VERSION,
)
from pet_core.learning_labeler import label_pending_events, label_single_event
from pet_core.recommender import record_manual_recommendation_label
from pet_core import window_chrome


def _iter_jsonl(path):
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception as e:
                print(f"[ManualLabel] 跳过损坏 JSONL: {path}:{line_no} {e}")
                continue
            if isinstance(obj, dict):
                yield obj


def _append_jsonl(path, record):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _text(value, limit=None):
    s = str(value or "").strip()
    if limit and len(s) > limit:
        return s[:limit] + "...[truncated]"
    return s


def _preview(value, limit=90):
    s = " ".join(_text(value).split())
    return s[:limit] + ("..." if len(s) > limit else "")


def _now_iso():
    return datetime.datetime.now().isoformat(timespec="seconds")


class DeepSeekLabelThread(QThread):
    finished_signal = pyqtSignal(dict)
    error_signal = pyqtSignal(str)

    def __init__(self, limit=10):
        super().__init__()
        self.limit = max(1, int(limit or 10))

    def run(self):
        try:
            stats = label_pending_events(limit=self.limit, dry_run=False)
            self.finished_signal.emit(dict(stats or {}))
        except Exception as e:
            self.error_signal.emit(str(e))


class DeepSeekSingleLabelThread(QThread):
    finished_signal = pyqtSignal(dict)
    error_signal = pyqtSignal(str)

    def __init__(self, event_id):
        super().__init__()
        self.event_id = str(event_id or "")

    def run(self):
        try:
            stats = label_single_event(self.event_id, force=True)
            self.finished_signal.emit(dict(stats or {}))
        except Exception as e:
            self.error_signal.emit(str(e))


class LearningLabelWindow(QWidget):
    """Browse raw learning samples and append high-priority user labels."""

    EMOTIONS = ["未知", "疲惫", "焦虑", "开心", "平静", "烦躁", "困惑", "身体不适"]
    TASK_STATES = ["未知", "学习中", "拖延中", "休息中", "饮食中", "写代码中", "闲聊"]
    NEEDS = ["未知", "安慰", "督促", "陪伴", "具体建议", "拆任务", "少说话", "查资料", "记录"]
    TONES = ["中性", "冷淡关心", "温柔安慰", "轻微督促", "具体建议", "吐槽"]
    LENGTHS = ["短", "中", "长"]
    ACTIONS = ["只回应", "允许短休", "给下一步", "推荐活动", "调用工具", "保持沉默"]
    REC_INTENTS = ["none", "suggest_action", "tool_action"]
    REC_CATEGORIES = ["none", "rest", "study", "food", "drawing", "music", "timer", "todo", "knowledge", "other"]
    TIMING_QUALITY = ["not_applicable", "good", "too_soon", "too_late", "interruptive", "unknown"]

    def __init__(self, parent=None):
        # Keep this as a normal top-level window: minimizable, coverable, and
        # not owned by the always-on-top desktop pet widget.
        super().__init__(None)
        
        self.setWindowFlags(Qt.Window | Qt.FramelessWindowHint | Qt.WindowMinimizeButtonHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setWindowTitle("训练样本人工标注")
        self.resize(1120, 720)
        window_chrome.setup_resizable_frameless_window(self, minimum_size=(760, 520))
        self.maximize_btn = None
        
        self.container = QFrame(self)
        self.container.setObjectName("MainContainer")
        
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.addWidget(self.container)
        
        self.label_thread = None
        self.single_label_thread = None
        self.events = []
        self.events_by_id = {}
        self.manual_labels = {}
        self.result_labels = {}
        self.feedback_by_event_id = {}
        self.filtered_events = []
        self.current_event = None
        self._dragging = False
        self._drag_offset = None
        self._build_ui()
        self.reload()

    def _build_ui(self):
        self.setStyleSheet("""
            QFrame#MainContainer {
                background: #1A1525;
                border-radius: 20px;
                border: 1px solid #3D2E55;
            }
            QWidget {
                background: transparent;
                color: #EAE5F2;
                font-family: 'Segoe UI', 'Microsoft YaHei', sans-serif;
                font-size: 13px;
            }
            QListWidget, QPlainTextEdit, QLineEdit, QComboBox, QSpinBox {
                background: #231B32;
                color: #EAE5F2;
                border: 1px solid #4E3C6B;
                border-radius: 16px;
                padding: 6px;
                selection-background-color: #4E3C6B;
            }
            QListWidget::item {
                padding: 8px;
                border-bottom: 1px solid #2A203B;
            }
            QListWidget::item:selected {
                background: #3D2E55;
                color: #EAE5F2;
            }
            QLabel#title {
                color: #B886F8;
                font-size: 18px;
                font-weight: 300;
            }
            QLabel#hint {
                color: #B8ADC9;
                font-size: 12px;
            }
            QGroupBox {
                border: 1px solid #3D2E55;
                border-radius: 16px;
                margin-top: 12px;
                padding-top: 12px;
                color: #D1C8E1;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 4px;
                color: #B886F8;
            }
            QPushButton {
                background: #2A203B;
                color: #EAE5F2;
                border: 1px solid #4E3C6B;
                border-radius: 16px;
                padding: 8px 14px;
            }
            QPushButton:hover {
                background: #3D2E55;
                border-color: #B886F8;
            }
            QPushButton#primary {
                background: #4E3C6B;
                border-color: #B886F8;
            }
            QPushButton#primary:hover {
                background: #46305F;
            }
            QCheckBox {
                color: #EAE5F2;
                spacing: 8px;
            }
        """)

        # --- 修改为主容器布局 ---
        root = QVBoxLayout(self.container)
        root.setContentsMargins(15, 15, 15, 15)
        root.setSpacing(10)

        self.header = QFrame()
        self.header.setObjectName("TitleBar")
        self.header.setFixedHeight(32)
        header = QHBoxLayout(self.header)
        header.setContentsMargins(0, 0, 0, 0)
        header.setSpacing(8)
        title = QLabel("训练样本人工标注")
        title.setObjectName("title")
        header.addWidget(title)
        header.addStretch()

        self.stats_label = QLabel("")
        self.stats_label.setObjectName("hint")
        header.addWidget(self.stats_label)

        min_btn = QPushButton("—")
        min_btn.setFixedSize(24, 24)
        min_btn.setToolTip("最小化")
        min_btn.setStyleSheet("""
            QPushButton { background: transparent; color: #B8ADC9; border: none; font-size: 14px; font-weight: bold; }
            QPushButton:hover { color: #EAE5F2; background: #2A203B; }
        """)
        min_btn.clicked.connect(self.showMinimized)
        header.addWidget(min_btn)

        self.maximize_btn = QPushButton("□")
        self.maximize_btn.setFixedSize(24, 24)
        self.maximize_btn.setToolTip("最大化/还原")
        self.maximize_btn.setStyleSheet("""
            QPushButton { background: transparent; color: #B8ADC9; border: none; font-size: 14px; font-weight: bold; }
            QPushButton:hover { color: #EAE5F2; background: #2A203B; }
        """)
        self.maximize_btn.clicked.connect(lambda: window_chrome.toggle_maximize_restore(self))
        header.addWidget(self.maximize_btn)

        close_btn = QPushButton("✕")
        close_btn.setFixedSize(24, 24)
        close_btn.setToolTip("关闭")
        close_btn.setStyleSheet("""
            QPushButton { background: transparent; color: #B886F8; border: none; font-size: 14px; font-weight: bold; }
            QPushButton:hover { color: #EAE5F2; background: #2A203B; }
        """)
        close_btn.clicked.connect(self.close)
        header.addWidget(close_btn)
        root.addWidget(self.header)

        splitter = QSplitter(Qt.Horizontal)
        root.addWidget(splitter, 1)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 10, 0)
        left_layout.setSpacing(8)

        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("搜索用户输入 / 回复 / event_id")
        self.search_input.textChanged.connect(self.refresh_event_list)
        left_layout.addWidget(self.search_input)

        self.only_unlabeled = QCheckBox("只看未人工标注")
        self.only_unlabeled.stateChanged.connect(self.refresh_event_list)
        left_layout.addWidget(self.only_unlabeled)

        self.event_filter_combo = QComboBox()
        self.event_filter_combo.addItem("全部样本", "all")
        self.event_filter_combo.addItem("普通聊天", "assistant_reply")
        self.event_filter_combo.addItem("主动消息", "proactive_sent")
        self.event_filter_combo.addItem("静默样本", "proactive_silence")
        self.event_filter_combo.addItem("有反馈", "has_feedback")
        self.event_filter_combo.currentIndexChanged.connect(self.refresh_event_list)
        left_layout.addWidget(self.event_filter_combo)

        self.event_list = QListWidget()
        self.event_list.currentItemChanged.connect(self._on_event_selected)
        left_layout.addWidget(self.event_list, 1)

        left_buttons = QHBoxLayout()
        refresh_btn = QPushButton("刷新")
        refresh_btn.clicked.connect(self.reload)
        left_buttons.addWidget(refresh_btn)
        clear_filter_btn = QPushButton("清筛选")
        clear_filter_btn.clicked.connect(self._clear_filter)
        left_buttons.addWidget(clear_filter_btn)
        left_layout.addLayout(left_buttons)

        labeler_row = QHBoxLayout()
        labeler_row.setSpacing(6)
        self.ds_limit_spin = QSpinBox()
        self.ds_limit_spin.setRange(1, 100)
        self.ds_limit_spin.setValue(10)
        self.ds_limit_spin.setToolTip("本次最多让 DeepSeek 处理多少条 pending 样本")
        labeler_row.addWidget(QLabel("DS条数"))
        labeler_row.addWidget(self.ds_limit_spin)
        self.ds_label_btn = QPushButton("DeepSeek打标")
        self.ds_label_btn.clicked.connect(self.run_deepseek_labeler)
        labeler_row.addWidget(self.ds_label_btn, 1)
        self.ds_current_btn = QPushButton("重跑当前")
        self.ds_current_btn.clicked.connect(self.run_deepseek_for_current)
        labeler_row.addWidget(self.ds_current_btn)
        left_layout.addLayout(labeler_row)
        splitter.addWidget(left)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(10, 0, 0, 0)
        right_layout.setSpacing(8)

        self.detail_text = QPlainTextEdit()
        self.detail_text.setReadOnly(True)
        self.detail_text.setMinimumHeight(230)
        right_layout.addWidget(self.detail_text, 1)

        form_scroll = QScrollArea()
        form_scroll.setWidgetResizable(True)
        form_scroll.setFrameShape(QFrame.NoFrame)
        form_scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")
        form_holder = QWidget()
        form_layout = QVBoxLayout(form_holder)
        form_layout.setContentsMargins(0, 0, 0, 0)
        form_layout.setSpacing(8)

        form_layout.addWidget(self._build_score_group())
        form_layout.addWidget(self._build_state_group())
        form_layout.addWidget(self._build_recommendation_group())
        form_layout.addWidget(self._build_proactive_group())
        form_layout.addWidget(self._build_risk_group())

        notes_group = QGroupBox("备注")
        notes_layout = QVBoxLayout(notes_group)
        self.notes_input = QPlainTextEdit()
        self.notes_input.setPlaceholderText("你为什么这样标？训练时可以保留为人工解释。")
        self.notes_input.setMaximumHeight(90)
        notes_layout.addWidget(self.notes_input)
        form_layout.addWidget(notes_group)

        form_scroll.setWidget(form_holder)
        right_layout.addWidget(form_scroll, 2)

        footer = QHBoxLayout()
        self.status_label = QLabel("")
        self.status_label.setObjectName("hint")
        footer.addWidget(self.status_label, 1)
        prev_btn = QPushButton("上一条")
        prev_btn.clicked.connect(lambda: self._move_selection(-1))
        footer.addWidget(prev_btn)
        next_btn = QPushButton("下一条")
        next_btn.clicked.connect(lambda: self._move_selection(1))
        footer.addWidget(next_btn)
        save_btn = QPushButton("保存人工标注")
        save_btn.setObjectName("primary")
        save_btn.clicked.connect(self.save_current_label)
        footer.addWidget(save_btn)
        right_layout.addLayout(footer)

        splitter.addWidget(right)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 3)

    def _build_score_group(self):
        box = QGroupBox("人工反馈分数")
        layout = QFormLayout(box)
        self.sample_weight_spin = self._spin(1, 5, 5)
        self.reply_score_spin = self._spin(-2, 2, 0)
        self.recommendation_score_spin = self._spin(-2, 2, 0)
        layout.addRow("人工权重", self.sample_weight_spin)
        layout.addRow("回复策略喜好", self.reply_score_spin)
        layout.addRow("推荐内容喜好", self.recommendation_score_spin)
        hint = QLabel("分数：-2 很不喜欢，-1 不合适，0 中性/无此项，1 可以，2 很喜欢。人工权重越高，未来训练越优先相信你。")
        hint.setObjectName("hint")
        hint.setWordWrap(True)
        layout.addRow("", hint)
        return box

    def _build_state_group(self):
        box = QGroupBox("状态与回复策略")
        grid = QGridLayout(box)
        self.emotion_combo = self._combo(self.EMOTIONS)
        self.task_state_combo = self._combo(self.TASK_STATES)
        self.need_combo = self._combo(self.NEEDS)
        self.tone_combo = self._combo(self.TONES)
        self.length_combo = self._combo(self.LENGTHS)
        self.action_combo = self._combo(self.ACTIONS)
        self.supervision_spin = self._spin(0, 100, 50)
        self.comfort_spin = self._spin(0, 100, 50)
        fields = [
            ("情绪", self.emotion_combo),
            ("任务状态", self.task_state_combo),
            ("主要需求", self.need_combo),
            ("语气", self.tone_combo),
            ("长度", self.length_combo),
            ("动作", self.action_combo),
            ("督促强度%", self.supervision_spin),
            ("安慰强度%", self.comfort_spin),
        ]
        for idx, (label, widget) in enumerate(fields):
            grid.addWidget(QLabel(label), idx // 2, (idx % 2) * 2)
            grid.addWidget(widget, idx // 2, (idx % 2) * 2 + 1)
        return box

    def _build_recommendation_group(self):
        box = QGroupBox("推荐/工具动作")
        layout = QFormLayout(box)
        self.should_recommend_cb = QCheckBox("这轮应该推荐具体行动")
        self.rec_intent_combo = self._combo(self.REC_INTENTS)
        self.rec_category_combo = self._combo(self.REC_CATEGORIES)
        self.rec_action_input = QLineEdit()
        self.rec_action_input.setPlaceholderText("例如：画画十分钟 / 开专注计时 / 先喝热水")
        self.rec_reason_input = QLineEdit()
        self.rec_reason_input.setPlaceholderText("为什么这轮该/不该推荐")
        layout.addRow("", self.should_recommend_cb)
        layout.addRow("推荐意图", self.rec_intent_combo)
        layout.addRow("推荐类别", self.rec_category_combo)
        layout.addRow("候选动作", self.rec_action_input)
        layout.addRow("理由", self.rec_reason_input)
        return box

    def _build_proactive_group(self):
        box = QGroupBox("主动关怀时机")
        layout = QFormLayout(box)
        self.timing_score_spin = self._spin(-2, 2, 0)
        self.timing_quality_combo = self._combo(self.TIMING_QUALITY)
        self.should_silent_cb = QCheckBox("这次更应该保持沉默")
        self.do_nothing_spin = self._spin(0, 100, 50)
        self.proactive_reason_input = QLineEdit()
        self.proactive_reason_input.setPlaceholderText("例如：用户刚回复过，不该继续打扰")
        layout.addRow("时机喜好", self.timing_score_spin)
        layout.addRow("时机质量", self.timing_quality_combo)
        layout.addRow("", self.should_silent_cb)
        layout.addRow("保持沉默倾向%", self.do_nothing_spin)
        layout.addRow("理由", self.proactive_reason_input)
        return box

    def _build_risk_group(self):
        box = QGroupBox("风险/问题")
        layout = QGridLayout(box)
        self.risk_reality_cb = QCheckBox("编造现实能力")
        self.risk_overlong_cb = QCheckBox("太长")
        self.risk_pushy_cb = QCheckBox("太强势/说教")
        self.risk_irrelevant_cb = QCheckBox("记忆/知识无关")
        checks = [
            self.risk_reality_cb,
            self.risk_overlong_cb,
            self.risk_pushy_cb,
            self.risk_irrelevant_cb,
        ]
        for i, cb in enumerate(checks):
            layout.addWidget(cb, i // 2, i % 2)
        return box

    def _spin(self, minimum, maximum, value):
        spin = QSpinBox()
        spin.setRange(minimum, maximum)
        spin.setValue(value)
        spin.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        return spin

    def _combo(self, values):
        combo = QComboBox()
        combo.addItems(values)
        combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        return combo

    def reload(self):
        self.events = list(_iter_jsonl(RAW_INTERACTIONS_PATH) or [])
        self.events_by_id = {str(e.get("event_id")): e for e in self.events if e.get("event_id")}
        self.manual_labels = self._load_latest_manual_labels()
        self.result_labels = self._load_latest_result_labels()
        self.feedback_by_event_id = self._load_feedback_events()
        self.refresh_event_list()

    def run_deepseek_labeler(self):
        if self.label_thread is not None and self.label_thread.isRunning():
            self.status_label.setText("DeepSeek 正在打标，稍等。")
            return
        self.ds_label_btn.setEnabled(False)
        self.status_label.setText("DeepSeek 正在给 pending 样本打草稿标签...")
        self.label_thread = DeepSeekLabelThread(limit=self.ds_limit_spin.value())
        self.label_thread.finished_signal.connect(self._on_deepseek_label_finished)
        self.label_thread.error_signal.connect(self._on_deepseek_label_error)
        self.label_thread.start()

    def _on_deepseek_label_finished(self, stats):
        self.ds_label_btn.setEnabled(True)
        processed = stats.get("processed", 0)
        failed = stats.get("failed", 0)
        skipped = stats.get("skipped", 0)
        message = stats.get("message", "")
        self.status_label.setText(
            f"DeepSeek打标完成：成功 {processed}，失败 {failed}，跳过 {skipped}。{message}"
        )
        self.reload()

    def _on_deepseek_label_error(self, error):
        self.ds_label_btn.setEnabled(True)
        self.status_label.setText(f"DeepSeek打标失败：{error}")
        QMessageBox.warning(self, "DeepSeek打标失败", str(error))

    def run_deepseek_for_current(self):
        if not self.current_event:
            QMessageBox.information(self, "没有样本", "先选择一条训练样本。")
            return
        if self.single_label_thread is not None and self.single_label_thread.isRunning():
            self.status_label.setText("当前样本正在重跑 DeepSeek。")
            return
        event_id = str(self.current_event.get("event_id") or "")
        self.ds_current_btn.setEnabled(False)
        self.status_label.setText(f"DeepSeek 正在重标当前样本：{event_id[:8]}...")
        self.single_label_thread = DeepSeekSingleLabelThread(event_id)
        self.single_label_thread.finished_signal.connect(self._on_current_label_finished)
        self.single_label_thread.error_signal.connect(self._on_current_label_error)
        self.single_label_thread.start()

    def _on_current_label_finished(self, stats):
        self.ds_current_btn.setEnabled(True)
        event_id = stats.get("event_id", "")
        self.status_label.setText(
            f"当前样本重标完成：成功 {stats.get('processed', 0)}，失败 {stats.get('failed', 0)}。{stats.get('message', '')}"
        )
        self.reload()
        if event_id:
            self._select_event_id(event_id)

    def _on_current_label_error(self, error):
        self.ds_current_btn.setEnabled(True)
        self.status_label.setText(f"当前样本重标失败：{error}")
        QMessageBox.warning(self, "当前样本重标失败", str(error))

    def _load_latest_manual_labels(self):
        latest = {}
        for record in _iter_jsonl(MANUAL_LABELS_PATH) or []:
            event_id = record.get("event_id")
            if event_id:
                latest[str(event_id)] = record
        return latest

    def _load_latest_result_labels(self):
        latest = {}
        for record in _iter_jsonl(LABEL_RESULTS_PATH) or []:
            if record.get("status") not in ("labeled", "manual_labeled"):
                continue
            event_id = record.get("event_id")
            if event_id:
                latest[str(event_id)] = record
        return latest

    def _load_feedback_events(self):
        grouped = {}
        for record in _iter_jsonl(FEEDBACK_EVENTS_PATH) or []:
            event_id = record.get("event_id")
            if not event_id:
                continue
            grouped.setdefault(str(event_id), []).append(record)
        return grouped

    def _clear_filter(self):
        self.search_input.clear()
        self.only_unlabeled.setChecked(False)
        self.event_filter_combo.setCurrentIndex(0)

    def refresh_event_list(self):
        query = self.search_input.text().strip().lower()
        only_unlabeled = self.only_unlabeled.isChecked()
        event_filter = self.event_filter_combo.currentData() or "all"
        previous_id = self.current_event.get("event_id") if self.current_event else ""

        filtered = []
        for event in reversed(self.events):
            event_id = str(event.get("event_id") or "")
            if only_unlabeled and event_id in self.manual_labels:
                continue
            if not self._match_event_filter(event, event_filter):
                continue
            if query:
                blob = " ".join([
                    event_id,
                    _text((event.get("user_input") or {}).get("text")),
                    _text((event.get("assistant_reply") or {}).get("text")),
                ]).lower()
                if query not in blob:
                    continue
            filtered.append(event)
        self.filtered_events = filtered

        self.event_list.blockSignals(True)
        self.event_list.clear()
        for event in filtered:
            item = QListWidgetItem(self._event_item_text(event))
            item.setData(Qt.UserRole, event.get("event_id"))
            self.event_list.addItem(item)
        self.event_list.blockSignals(False)

        self.stats_label.setText(
            f"样本 {len(self.events)} 条｜当前 {len(filtered)} 条｜人工标注 {len(self.manual_labels)} 条"
        )

        target_row = 0
        if previous_id:
            for row, event in enumerate(filtered):
                if event.get("event_id") == previous_id:
                    target_row = row
                    break
        if filtered:
            self.event_list.setCurrentRow(target_row)
        else:
            self.current_event = None
            self.detail_text.setPlainText("没有匹配的样本。")

    def _is_silence_sample(self, event):
        trigger = event.get("trigger") or {}
        assistant_text = ((event.get("assistant_reply") or {}).get("text") or "").strip()
        strategy = event.get("strategy") or {}
        return (
            trigger.get("type") == "proactive_timer"
            and (
                trigger.get("source") == "proactive_silence"
                or strategy.get("selected") == "do_nothing"
                or not assistant_text
            )
        )

    def _match_event_filter(self, event, event_filter):
        event_id = str(event.get("event_id") or "")
        if event_filter == "all":
            return True
        if event_filter == "has_feedback":
            return bool(self.feedback_by_event_id.get(event_id))
        if event_filter == "assistant_reply":
            return event.get("event_type") == "assistant_reply"
        if event_filter == "proactive_sent":
            return (
                event.get("event_type") == "proactive_message"
                and not self._is_silence_sample(event)
            )
        if event_filter == "proactive_silence":
            return self._is_silence_sample(event)
        return True

    def _event_item_text(self, event):
        event_id = str(event.get("event_id") or "")
        timestamp = event.get("timestamp") or ""
        trigger = (event.get("trigger") or {}).get("type", "")
        user_text = _preview((event.get("user_input") or {}).get("text"), 70)
        reply = _preview((event.get("assistant_reply") or {}).get("text"), 70)
        if self._is_silence_sample(event):
            trigger = "proactive_silence"
            reply = "[保持沉默样本]"
        elif not reply:
            reply = "[空回复]"
        mark = "已人工" if event_id in self.manual_labels else "待标"
        fb = f" 反馈{len(self.feedback_by_event_id.get(event_id, []))}" if self.feedback_by_event_id.get(event_id) else ""
        return f"[{mark}] {timestamp}  {trigger}{fb}\n用户：{user_text}\n回复：{reply}"

    def _on_event_selected(self, current, _previous):
        if current is None:
            return
        event_id = str(current.data(Qt.UserRole) or "")
        event = self.events_by_id.get(event_id)
        if not event:
            return
        self.current_event = event
        self.detail_text.setPlainText(self._format_event_detail(event))
        self._populate_form(event)

    def _format_event_detail(self, event):
        retrieval = event.get("retrieval") or {}
        trigger = event.get("trigger") or {}
        user_input = event.get("user_input") or {}
        assistant_reply = event.get("assistant_reply") or {}
        state = ((event.get("state_features") or {}).get("system") or {})
        foreground = state.get("foreground") or {}
        idle = state.get("idle") or {}
        event_id = str(event.get("event_id") or "")
        feedback_events = self.feedback_by_event_id.get(event_id, [])
        is_silence = self._is_silence_sample(event)
        assistant_text = _text(assistant_reply.get("text"))
        if is_silence:
            assistant_text = "[静默样本：本轮主动关怀检查后选择不发消息。这个空回复不是坏数据。]"
        elif not assistant_text:
            assistant_text = "[空回复：需要检查是否为异常或主动沉默样本。]"

        parts = [
            f"event_id: {event.get('event_id')}",
            f"timestamp: {event.get('timestamp')}",
            f"event_type: {event.get('event_type')}  trigger: {trigger.get('type')} / {trigger.get('source')}",
            f"sample_type: {'静默样本(do_nothing)' if is_silence else '已发送消息'}",
            "",
            "【用户输入】",
            _text(user_input.get("text")),
            "",
            "【桌宠回复】",
            assistant_text,
            f"emotion_tag: {assistant_reply.get('emotion_tag', '')}",
            "",
            "【时间/本地状态】",
            json.dumps(event.get("time_features") or {}, ensure_ascii=False),
            f"foreground={foreground.get('category', 'unknown')} idle={idle.get('seconds_bucket', 'unknown')}",
            "",
            "【后续反馈 / 隐式反应】",
        ]
        if feedback_events:
            for fb in feedback_events:
                parts.append(
                    "- "
                    f"{fb.get('timestamp', '')} "
                    f"scope={fb.get('feedback_scope', '')} "
                    f"feedback={fb.get('feedback', '')} "
                    f"user={_text(fb.get('user_text'), 160)} "
                    f"extra={json.dumps(fb.get('extra') or {}, ensure_ascii=False)}"
                )
        else:
            parts.append("无。没有后续反馈时，DS 标签只能当低置信度草稿。")
        parts.extend([
            "",
            "【最近上下文】",
        ])
        for item in event.get("recent_context") or []:
            source = str(item.get("source") or "").strip().lower()
            if source == "proactive" or item.get("role_pair") == "proactive_message":
                parts.append(
                    f"- {item.get('minutes_ago')}分钟前 主动关怀:{_text(item.get('assistant_summary'), 180)}"
                )
            else:
                parts.append(
                    f"- {item.get('minutes_ago')}分钟前 用户:{_text(item.get('user'), 160)} / 回应要点:{_text(item.get('assistant_summary'), 160)}"
                )
        parts.extend([
            "",
            "【短期记忆】",
        ])
        for mem in retrieval.get("short_memory_snippets") or []:
            parts.append(f"- {_text(mem, 260)}")
        parts.extend([
            "",
            "【知识库工具】",
            json.dumps(retrieval.get("knowledge_tool") or {}, ensure_ascii=False),
        ])
        for kb in retrieval.get("knowledge_snippets") or []:
            parts.append(f"- {_text(kb, 260)}")
        parts.extend([
            "",
            "【用户画像快照】",
            _text(event.get("user_profile_snapshot"), 900),
        ])

        manual = self.manual_labels.get(event_id)
        result = self.result_labels.get(event_id)
        if manual:
            parts.extend(["", "【已有人工标签】", json.dumps(manual.get("labels") or {}, ensure_ascii=False, indent=2)])
        elif result and result.get("labels"):
            parts.extend(["", "【已有模型标签】", json.dumps(result.get("labels") or {}, ensure_ascii=False, indent=2)])
        return "\n".join(parts)

    def _populate_form(self, event):
        event_id = str(event.get("event_id") or "")
        source = self.manual_labels.get(event_id) or self.result_labels.get(event_id) or {}
        labels = source.get("labels") or (event.get("prompt_decision") or {}).get("teacher_labels") or {}
        scores = source.get("scores") or {}
        is_silence = self._is_silence_sample(event)

        self.sample_weight_spin.setValue(int(scores.get("sample_weight", 5 if source.get("source") == "user_manual" else 3)))
        self.reply_score_spin.setValue(int(scores.get("reply_strategy_score", 0)))
        self.recommendation_score_spin.setValue(int(scores.get("recommendation_score", 0)))
        self.timing_score_spin.setValue(int(scores.get("proactive_timing_score", 0)))

        state = labels.get("state") or {}
        need = labels.get("need") or {}
        strategy = labels.get("feedback_strategy") or {}
        rec = labels.get("recommendation_type") or {}
        timing = labels.get("proactive_timing") or {}
        do_nothing = labels.get("do_nothing_preference") or {}
        risk = labels.get("risk") or {}

        self._set_combo(self.emotion_combo, state.get("emotion") or "未知")
        self._set_combo(self.task_state_combo, state.get("task_state") or "未知")
        self._set_combo(self.need_combo, need.get("primary") or "未知")
        self._set_combo(self.tone_combo, strategy.get("tone") or "中性")
        self._set_combo(self.length_combo, strategy.get("length") or "短")
        self._set_combo(self.action_combo, strategy.get("action") or ("保持沉默" if is_silence else "只回应"))
        self.supervision_spin.setValue(self._percent(strategy.get("supervision_level"), 0 if is_silence else 50))
        self.comfort_spin.setValue(self._percent(strategy.get("comfort_level"), 0 if is_silence else 50))
        self._set_combo(self.rec_intent_combo, strategy.get("recommendation_intent") or "none")
        self.should_recommend_cb.setChecked(bool(rec.get("should_recommend", False)))
        self._set_combo(self.rec_category_combo, rec.get("category") or "none")
        self.rec_action_input.setText(_text(rec.get("candidate_action")))
        self.rec_reason_input.setText(_text(rec.get("reason")))
        self._set_combo(
            self.timing_quality_combo,
            timing.get("timing_quality") or ("unknown" if is_silence else "not_applicable"),
        )
        self.should_silent_cb.setChecked(bool(timing.get("should_have_stayed_silent", is_silence)))
        self.do_nothing_spin.setValue(self._percent(do_nothing.get("score"), 85 if is_silence else 50))
        default_silence_reason = "主动关怀检查选择保持沉默；请结合后续反馈判断这次沉默是否合适。"
        self.proactive_reason_input.setText(
            _text(timing.get("reason") or do_nothing.get("reason") or (default_silence_reason if is_silence else ""))
        )
        self.risk_reality_cb.setChecked(bool(risk.get("fabricated_reality_action", False)))
        self.risk_overlong_cb.setChecked(bool(risk.get("overlong", False)))
        self.risk_pushy_cb.setChecked(bool(risk.get("too_pushy", False)))
        self.risk_irrelevant_cb.setChecked(bool(risk.get("irrelevant_memory_or_knowledge", False)))
        self.notes_input.setPlainText(_text(source.get("notes")))
        self.status_label.setText("已加载人工标签。" if source.get("source") == "user_manual" else "")

    def _percent(self, value, default):
        try:
            number = float(value)
            if 0 <= number <= 1:
                number *= 100
            return int(max(0, min(100, round(number))))
        except Exception:
            return default

    def _set_combo(self, combo, value):
        value = str(value or "")
        idx = combo.findText(value)
        if idx < 0:
            combo.addItem(value)
            idx = combo.findText(value)
        combo.setCurrentIndex(max(0, idx))

    def _collect_labels(self):
        recommendation_intent = self.rec_intent_combo.currentText()
        return {
            "state": {
                "emotion": self.emotion_combo.currentText(),
                "task_state": self.task_state_combo.currentText(),
                "energy": "未知",
                "confidence": 1.0,
            },
            "need": {
                "primary": self.need_combo.currentText(),
                "secondary": [],
                "confidence": 1.0,
            },
            "feedback_strategy": {
                "tone": self.tone_combo.currentText(),
                "length": self.length_combo.currentText(),
                "action": self.action_combo.currentText(),
                "supervision_level": round(self.supervision_spin.value() / 100, 2),
                "comfort_level": round(self.comfort_spin.value() / 100, 2),
                "recommendation_intent": recommendation_intent,
                "confidence": 1.0,
            },
            "recommendation_type": {
                "should_recommend": self.should_recommend_cb.isChecked(),
                "category": self.rec_category_combo.currentText(),
                "candidate_action": self.rec_action_input.text().strip(),
                "reason": self.rec_reason_input.text().strip(),
            },
            "proactive_timing": {
                "is_proactive": ((self.current_event or {}).get("trigger") or {}).get("type") == "proactive_timer",
                "timing_quality": self.timing_quality_combo.currentText(),
                "should_have_stayed_silent": self.should_silent_cb.isChecked(),
                "reason": self.proactive_reason_input.text().strip(),
            },
            "do_nothing_preference": {
                "score": round(self.do_nothing_spin.value() / 100, 2),
                "reason": self.proactive_reason_input.text().strip(),
            },
            "risk": {
                "fabricated_reality_action": self.risk_reality_cb.isChecked(),
                "overlong": self.risk_overlong_cb.isChecked(),
                "too_pushy": self.risk_pushy_cb.isChecked(),
                "irrelevant_memory_or_knowledge": self.risk_irrelevant_cb.isChecked(),
                "notes": [],
            },
        }

    def _collect_scores(self):
        return {
            "sample_weight": self.sample_weight_spin.value(),
            "reply_strategy_score": self.reply_score_spin.value(),
            "recommendation_score": self.recommendation_score_spin.value(),
            "proactive_timing_score": self.timing_score_spin.value(),
        }

    def save_current_label(self):
        if not self.current_event:
            QMessageBox.information(self, "没有样本", "先选择一条训练样本。")
            return

        event = self.current_event
        event_id = str(event.get("event_id") or "")
        labels = self._collect_labels()
        scores = self._collect_scores()
        notes = self.notes_input.toPlainText().strip()
        now = _now_iso()
        record = {
            "schema_version": SCHEMA_VERSION,
            "manual_label_id": uuid.uuid4().hex,
            "event_id": event_id,
            "timestamp": now,
            "source": "user_manual",
            "scores": scores,
            "labels": labels,
            "notes": notes,
            "sample_preview": {
                "event_timestamp": event.get("timestamp"),
                "event_type": event.get("event_type"),
                "trigger": event.get("trigger") or {},
                "user_text": _text((event.get("user_input") or {}).get("text"), 600),
                "assistant_text": _text((event.get("assistant_reply") or {}).get("text"), 800),
            },
        }

        result = {
            "schema_version": SCHEMA_VERSION,
            "event_id": event_id,
            "timestamp": now,
            "status": "manual_labeled",
            "task": "teacher_label",
            "labeler": {
                "source": "user_manual",
                "sample_weight": scores["sample_weight"],
            },
            "scores": scores,
            "labels": labels,
            "notes": notes,
        }

        labeled_event = copy.deepcopy(event)
        labeled_event.setdefault("prompt_decision", {})["teacher_labels"] = labels
        labeled_event["label_status"] = "manual_labeled"
        labeled_event["manual_label_meta"] = {
            "manual_label_id": record["manual_label_id"],
            "timestamp": now,
            "sample_weight": scores["sample_weight"],
            "scores": scores,
        }

        try:
            _append_jsonl(MANUAL_LABELS_PATH, record)
            _append_jsonl(LABEL_RESULTS_PATH, result)
            _append_jsonl(LABELED_INTERACTIONS_PATH, labeled_event)
            record_manual_recommendation_label(event, labels, scores)
        except Exception as e:
            QMessageBox.warning(self, "保存失败", str(e))
            return

        self.manual_labels[event_id] = record
        self.result_labels[event_id] = result
        self.status_label.setText(f"已保存人工标注：{event_id[:8]}")
        self.refresh_event_list()
        self._select_event_id(event_id)

    def _select_event_id(self, event_id):
        for row in range(self.event_list.count()):
            item = self.event_list.item(row)
            if str(item.data(Qt.UserRole) or "") == str(event_id):
                self.event_list.setCurrentRow(row)
                return

    def _move_selection(self, delta):
        if self.event_list.count() == 0:
            return
        row = self.event_list.currentRow()
        if row < 0:
            row = 0
        row = max(0, min(self.event_list.count() - 1, row + delta))
        self.event_list.setCurrentRow(row)

    def mousePressEvent(self, event):
        if window_chrome.begin_window_resize(self, event):
            return
        if window_chrome.begin_title_drag(self, event, getattr(self, "header", None)):
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if window_chrome.continue_window_resize(self, event):
            return
        if window_chrome.continue_title_drag(self, event):
            return
        window_chrome.update_resize_cursor(self, event)
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        window_chrome.end_window_resize(self)
        window_chrome.end_title_drag(self)
        super().mouseReleaseEvent(event)

    def nativeEvent(self, eventType, message):
        handled = window_chrome.native_resize_event(self, eventType, message)
        if handled is not None:
            return handled
        return super().nativeEvent(eventType, message)

    def mouseDoubleClickEvent(self, event):
        if window_chrome.title_double_click_maximize(self, event, getattr(self, "header", None)):
            return
        super().mouseDoubleClickEvent(event)

    def leaveEvent(self, event):
        window_chrome.leave_resize_area(self, event)
        super().leaveEvent(event)

    def resizeEvent(self, event):
        window_chrome.sync_maximize_button(self)
        super().resizeEvent(event)
