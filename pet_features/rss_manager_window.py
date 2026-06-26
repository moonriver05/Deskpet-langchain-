"""RSS source and cache management window."""

import os

import requests

from PyQt5.QtCore import QSize, Qt, QThread, pyqtSignal, QUrl
from PyQt5.QtGui import QDesktopServices, QIcon, QPixmap
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QComboBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from pet_core import window_chrome
from pet_core.config import app_config
from pet_core.rss_content import (
    add_source,
    clear_cached_items,
    delete_source,
    list_cached_items,
    list_sources,
    refresh_source,
    refresh_sources,
    update_source,
)


MAX_VISIBLE_ITEMS = 200
MAX_COVER_DOWNLOADS = 40


def _rsshub_url(path):
    path = str(path or "").strip()
    if path.startswith(("http://", "https://", "rsshub://")):
        return path
    base = str(app_config.get("rss_recommender.base_url", "https://rsshub.rssforever.com") or "").rstrip("/")
    return f"{base}{path}"


SOURCE_PRESETS = [
    {
        "label": "B站综合热门",
        "name": "Bilibili 综合热门",
        "url": "/bilibili/popular/all/noembed",
        "type": "video",
        "platform": "bilibili",
        "tags": "热门, 视频, B站",
    },
    {
        "label": "B站搜索：本地AI",
        "name": "Bilibili 搜索：本地AI",
        "url": "/bilibili/vsearch/%E6%9C%AC%E5%9C%B0AI/pubdate/noembed",
        "type": "video",
        "platform": "bilibili",
        "tags": "本地AI, 搜索, 技术",
    },
    {
        "label": "Pixiv插画日榜",
        "name": "Pixiv 插画日榜",
        "url": "/pixiv/ranking/day",
        "type": "image",
        "platform": "pixiv",
        "tags": "插画, Pixiv, 排行榜, 日榜",
        "max_items": 20,
    },
    {
        "label": "Pixiv插画周榜",
        "name": "Pixiv 插画周榜",
        "url": "/pixiv/ranking/week",
        "type": "image",
        "platform": "pixiv",
        "tags": "插画, Pixiv, 排行榜, 周榜",
        "max_items": 20,
    },
    {
        "label": "Pixiv用户作品模板",
        "name": "Pixiv 用户作品",
        "url": "/pixiv/user/15288095",
        "type": "image",
        "platform": "pixiv",
        "tags": "插画, Pixiv",
    },
    {
        "label": "知乎热榜",
        "name": "知乎热榜",
        "url": "/zhihu/hotlist",
        "type": "article",
        "platform": "zhihu",
        "tags": "知乎, 热榜",
    },
    {
        "label": "GitHub Releases模板",
        "name": "GitHub Releases",
        "url": "https://github.com/huggingface/transformers/releases.atom",
        "type": "repo",
        "platform": "github",
        "tags": "GitHub, releases, AI",
    },
    {
        "label": "arXiv cs.AI",
        "name": "arXiv cs.AI",
        "url": "https://rss.arxiv.org/rss/cs.AI",
        "type": "paper",
        "platform": "arxiv",
        "tags": "AI, 论文, arXiv",
    },
    {
        "label": "HuggingFace Blog",
        "name": "HuggingFace Blog",
        "url": "https://huggingface.co/blog/feed.xml",
        "type": "article",
        "platform": "huggingface",
        "tags": "AI, 模型, HuggingFace",
    },
]


def _text(value, limit=None):
    text = str(value or "").strip()
    if limit and len(text) > limit:
        return text[:limit] + "...[truncated]"
    return text


def _preview(value, limit=80):
    text = " ".join(_text(value).split())
    return text[:limit] + ("..." if len(text) > limit else "")


class RSSRefreshThread(QThread):
    finished_signal = pyqtSignal(dict)
    error_signal = pyqtSignal(str)

    def __init__(self, source_id=None):
        super().__init__()
        self.source_id = str(source_id or "")

    def run(self):
        try:
            if self.source_id:
                result = refresh_source(self.source_id, force=True)
            else:
                result = refresh_sources(force=True, ignore_platform_filter=True)
            self.finished_signal.emit(dict(result or {}))
        except Exception as e:
            self.error_signal.emit(str(e))


class CoverDownloadThread(QThread):
    finished_signal = pyqtSignal(int, object)

    def __init__(self, row, url):
        super().__init__()
        self.row = int(row)
        self.url = str(url or "")

    def run(self):
        content = None
        try:
            resp = requests.get(
                self.url,
                timeout=8,
                headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                    "Referer": "https://www.bilibili.com/",
                },
            )
            resp.raise_for_status()
            content = resp.content
        except Exception:
            content = None
        self.finished_signal.emit(self.row, content)


class RSSManagerWindow(QWidget):
    """Manage user-provided RSS feeds and inspect cached items."""

    def __init__(self, parent=None):
        super().__init__(None)
        self.setWindowTitle("RSS 管理")
        self.setWindowFlags(Qt.Window | Qt.FramelessWindowHint | Qt.WindowMinimizeButtonHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.resize(1080, 700)
        window_chrome.setup_resizable_frameless_window(self, minimum_size=(820, 540))

        self.container = None
        self.title_bar = None
        self.maximize_btn = None
        self.refresh_thread = None
        self.cover_threads = []
        self.sources = []
        self.items = []
        self.current_preset_max_items = None

        self._build_ui()
        self.reload_sources()
        self.reload_items()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)

        self.container = QFrame()
        self.container.setObjectName("RSSContainer")
        container_layout = QVBoxLayout(self.container)
        container_layout.setContentsMargins(14, 12, 14, 14)
        container_layout.setSpacing(10)
        root.addWidget(self.container)

        self.title_bar = QFrame()
        self.title_bar.setObjectName("RSSTitleBar")
        title_layout = QHBoxLayout(self.title_bar)
        title_layout.setContentsMargins(0, 0, 0, 0)
        title = QLabel("RSS 管理")
        title.setObjectName("RSSTitle")
        self.status_label = QLabel("管理 RSS 源、手动抓取、查看标题/简介/原链接")
        self.status_label.setObjectName("RSSStatus")
        title_layout.addWidget(title)
        title_layout.addWidget(self.status_label, 1)
        min_btn = QPushButton("—")
        min_btn.setObjectName("WindowButton")
        min_btn.clicked.connect(self.showMinimized)
        self.maximize_btn = QPushButton("□")
        self.maximize_btn.setObjectName("WindowButton")
        self.maximize_btn.clicked.connect(lambda: window_chrome.toggle_maximize_restore(self))
        close_btn = QPushButton("×")
        close_btn.setObjectName("WindowButton")
        close_btn.clicked.connect(self.close)
        for btn in (min_btn, self.maximize_btn, close_btn):
            btn.setFixedSize(34, 30)
            title_layout.addWidget(btn)
        container_layout.addWidget(self.title_bar)

        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)
        container_layout.addWidget(splitter, 1)

        left = QFrame()
        left.setObjectName("Panel")
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(10, 10, 10, 10)
        left_layout.setSpacing(8)
        splitter.addWidget(left)

        source_title = QLabel("订阅源")
        source_title.setObjectName("SectionTitle")
        left_layout.addWidget(source_title)

        preset_row = QHBoxLayout()
        self.preset_combo = QComboBox()
        self.preset_combo.addItem("选择常用源模板...")
        for preset in SOURCE_PRESETS:
            self.preset_combo.addItem(preset["label"])
        self.apply_preset_btn = QPushButton("套用示例")
        self.apply_preset_btn.clicked.connect(self.apply_selected_preset)
        preset_row.addWidget(self.preset_combo, 1)
        preset_row.addWidget(self.apply_preset_btn)
        left_layout.addLayout(preset_row)

        self.source_list = QListWidget()
        self.source_list.currentItemChanged.connect(self.on_source_selected)
        left_layout.addWidget(self.source_list, 1)

        form = QGridLayout()
        form.setHorizontalSpacing(8)
        form.setVerticalSpacing(8)
        left_layout.addLayout(form)
        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("源名称，例如 泛式动态")
        self.url_input = QLineEdit()
        self.url_input.setPlaceholderText("RSS 链接，例如 https://rsshub.../bilibili/user/dynamic/63231")
        self.platform_combo = QComboBox()
        self.platform_combo.addItems(["auto", "bilibili", "pixiv", "zhihu", "github", "arxiv", "huggingface", "youtube", "rsshub", "custom"])
        self.type_combo = QComboBox()
        self.type_combo.addItems(["feed", "video", "dynamic", "image", "article", "paper", "repo", "audio"])
        self.tags_input = QLineEdit()
        self.tags_input.setPlaceholderText("标签，用逗号分隔")
        self.enabled_check = QCheckBox("启用")
        self.enabled_check.setChecked(True)
        form.addWidget(QLabel("名称"), 0, 0)
        form.addWidget(self.name_input, 0, 1)
        form.addWidget(QLabel("链接"), 1, 0)
        form.addWidget(self.url_input, 1, 1)
        form.addWidget(QLabel("平台"), 2, 0)
        form.addWidget(self.platform_combo, 2, 1)
        form.addWidget(QLabel("类型"), 3, 0)
        form.addWidget(self.type_combo, 3, 1)
        form.addWidget(QLabel("标签"), 4, 0)
        form.addWidget(self.tags_input, 4, 1)
        form.addWidget(self.enabled_check, 5, 1)

        action_row_1 = QHBoxLayout()
        self.add_btn = QPushButton("添加并抓取")
        self.add_btn.clicked.connect(self.add_current_source)
        self.save_btn = QPushButton("保存修改")
        self.save_btn.clicked.connect(self.save_current_source)
        action_row_1.addWidget(self.add_btn)
        action_row_1.addWidget(self.save_btn)
        left_layout.addLayout(action_row_1)

        action_row_2 = QHBoxLayout()
        self.refresh_source_btn = QPushButton("刷新选中")
        self.refresh_source_btn.clicked.connect(self.refresh_selected_source)
        self.refresh_all_btn = QPushButton("刷新全部")
        self.refresh_all_btn.clicked.connect(self.refresh_all_sources)
        self.delete_btn = QPushButton("删除源")
        self.delete_btn.setObjectName("DangerButton")
        self.delete_btn.clicked.connect(self.delete_current_source)
        action_row_2.addWidget(self.refresh_source_btn)
        action_row_2.addWidget(self.refresh_all_btn)
        action_row_2.addWidget(self.delete_btn)
        left_layout.addLayout(action_row_2)

        hint = QLabel("窗口只管理 RSS 文本条目和原链接；不会播放或内嵌展示内容。UP 主投稿/动态若被 B 站风控，需要在设置里填写 Bilibili Cookie。")
        hint.setWordWrap(True)
        hint.setObjectName("Hint")
        left_layout.addWidget(hint)

        right = QFrame()
        right.setObjectName("Panel")
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(10, 10, 10, 10)
        right_layout.setSpacing(8)
        splitter.addWidget(right)

        top_row = QHBoxLayout()
        item_title = QLabel("抓取内容")
        item_title.setObjectName("SectionTitle")
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("搜索标题、简介、来源...")
        self.search_input.returnPressed.connect(self.reload_items)
        self.only_current_check = QCheckBox("只看选中源")
        self.only_current_check.stateChanged.connect(self.reload_items)
        self.search_btn = QPushButton("搜索/刷新")
        self.search_btn.clicked.connect(self.reload_items)
        top_row.addWidget(item_title)
        top_row.addWidget(self.search_input, 1)
        top_row.addWidget(self.only_current_check)
        top_row.addWidget(self.search_btn)
        right_layout.addLayout(top_row)

        self.item_table = QTableWidget(0, 6)
        self.item_table.setHorizontalHeaderLabels(["封面", "时间", "来源", "类型", "标题", "原链接"])
        self.item_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.item_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.item_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.item_table.verticalHeader().setVisible(False)
        self.item_table.setIconSize(QSize(96, 54))
        self.item_table.verticalHeader().setDefaultSectionSize(64)
        self.item_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.item_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.item_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.item_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)
        self.item_table.horizontalHeader().setSectionResizeMode(4, QHeaderView.Stretch)
        self.item_table.horizontalHeader().setSectionResizeMode(5, QHeaderView.ResizeToContents)
        self.item_table.itemSelectionChanged.connect(self.show_selected_item)
        self.item_table.itemDoubleClicked.connect(self.open_selected_item)
        right_layout.addWidget(self.item_table, 2)

        detail_row = QHBoxLayout()
        self.open_btn = QPushButton("打开原链接")
        self.open_btn.clicked.connect(self.open_selected_item)
        self.copy_btn = QPushButton("复制原链接")
        self.copy_btn.clicked.connect(self.copy_selected_url)
        self.clear_items_btn = QPushButton("清空选中源缓存")
        self.clear_items_btn.setObjectName("DangerButton")
        self.clear_items_btn.clicked.connect(self.clear_current_items)
        detail_row.addWidget(self.open_btn)
        detail_row.addWidget(self.copy_btn)
        detail_row.addStretch(1)
        detail_row.addWidget(self.clear_items_btn)
        right_layout.addLayout(detail_row)

        self.detail_text = QPlainTextEdit()
        self.detail_text.setReadOnly(True)
        self.detail_text.setPlaceholderText("选择一条内容后，在这里查看标题、原链接、简介、标签。")
        right_layout.addWidget(self.detail_text, 1)

        splitter.setSizes([360, 720])
        self._apply_style()

    def _apply_style(self):
        self.setStyleSheet("""
            QWidget {
                color: #EAE5F2;
                font-family: "Microsoft YaHei", "Segoe UI";
                font-size: 13px;
            }
            QFrame#RSSContainer {
                background: #050507;
                border: 1px solid #6F86A2;
                border-radius: 18px;
            }
            QFrame#RSSTitleBar {
                background: transparent;
                min-height: 34px;
            }
            QLabel#RSSTitle {
                color: #AFC3D7;
                font-size: 20px;
                font-weight: 500;
            }
            QLabel#RSSStatus, QLabel#Hint {
                color: #8D98A8;
            }
            QLabel#SectionTitle {
                color: #D8D0E6;
                font-size: 15px;
                font-weight: 600;
            }
            QFrame#Panel {
                background: #0A0A0E;
                border: 1px solid #394A63;
                border-radius: 12px;
            }
            QLineEdit, QPlainTextEdit, QComboBox, QListWidget, QTableWidget {
                background: #050507;
                border: 1px solid #536B86;
                border-radius: 8px;
                color: #F0ECF6;
                padding: 7px;
                selection-background-color: #342449;
            }
            QListWidget::item {
                padding: 8px;
                border-bottom: 1px solid #161C25;
            }
            QListWidget::item:selected {
                background: #21172F;
                color: #F4EDF8;
            }
            QHeaderView::section {
                background: #21172F;
                color: #F1EAF7;
                padding: 7px;
                border: none;
                border-right: 1px solid #342449;
            }
            QTableWidget {
                gridline-color: #1B2330;
            }
            QTableWidget::item {
                padding: 5px;
            }
            QTableWidget::item:selected {
                background: #223245;
            }
            QPushButton {
                background: #132235;
                color: #F0ECF6;
                border: 1px solid #6F86A2;
                border-radius: 9px;
                padding: 8px 12px;
            }
            QPushButton:hover {
                background: #1D334D;
                border-color: #93A8BF;
            }
            QPushButton#DangerButton {
                background: #2C1730;
                border-color: #8B5A98;
            }
            QPushButton#WindowButton {
                background: transparent;
                border: none;
                color: #C7B8D8;
                padding: 0;
                border-radius: 6px;
            }
            QPushButton#WindowButton:hover {
                background: #21172F;
            }
            QSplitter::handle {
                background: #121823;
                width: 5px;
            }
            QCheckBox {
                color: #D8D0E6;
                spacing: 6px;
            }
        """)

    def selected_source_id(self):
        item = self.source_list.currentItem()
        return item.data(Qt.UserRole) if item else ""

    def selected_item(self):
        row = self.item_table.currentRow()
        if row < 0 or row >= len(self.items):
            return None
        return self.items[row]

    def reload_sources(self):
        current_id = self.selected_source_id()
        self.sources = list_sources()
        self.source_list.clear()
        restore_row = -1
        for idx, source in enumerate(self.sources):
            enabled = "●" if source.get("enabled", True) else "○"
            label = f"{enabled} {source.get('name') or source.get('id')}\n{_preview(source.get('feed_url'), 64)}"
            item = QListWidgetItem(label)
            item.setData(Qt.UserRole, source.get("id"))
            self.source_list.addItem(item)
            if source.get("id") == current_id:
                restore_row = idx
        if self.source_list.count():
            self.source_list.setCurrentRow(restore_row if restore_row >= 0 else 0)
        self.status_label.setText(f"订阅源 {len(self.sources)} 个")

    def reload_items(self):
        source_id = self.selected_source_id() if self.only_current_check.isChecked() else ""
        query = self.search_input.text().strip()
        self.items = list_cached_items(source_id=source_id, query=query, limit=MAX_VISIBLE_ITEMS)
        self.cover_threads = [t for t in self.cover_threads if t.isRunning()]
        self.item_table.setRowCount(len(self.items))
        for row, item in enumerate(self.items):
            cover_cell = QTableWidgetItem("封面")
            cover_cell.setToolTip(item.get("cover_url") or "")
            self.item_table.setItem(row, 0, cover_cell)
            values = [
                item.get("published_at") or item.get("fetched_at") or "",
                item.get("source_name") or item.get("source_id") or "",
                item.get("source_type") or "",
                item.get("title") or "",
                item.get("url") or "",
            ]
            for col, value in enumerate(values):
                cell = QTableWidgetItem(_text(value, 260))
                cell.setToolTip(_text(value))
                self.item_table.setItem(row, col + 1, cell)
            if row < MAX_COVER_DOWNLOADS and item.get("cover_url"):
                self.start_cover_download(row, item.get("cover_url"))
        self.status_label.setText(f"缓存条目 {len(self.items)} 条")
        if self.items:
            self.item_table.selectRow(0)
        else:
            self.detail_text.clear()

    def on_source_selected(self, current, previous=None):
        source_id = current.data(Qt.UserRole) if current else ""
        source = next((s for s in self.sources if s.get("id") == source_id), None)
        if not source:
            return
        self.current_preset_max_items = None
        self.name_input.setText(source.get("name") or "")
        self.url_input.setText(source.get("feed_url") or "")
        self.tags_input.setText(", ".join(source.get("tags") or []))
        self.enabled_check.setChecked(bool(source.get("enabled", True)))
        self._set_combo(self.platform_combo, source.get("platform") or "custom")
        self._set_combo(self.type_combo, source.get("type") or "feed")
        if self.only_current_check.isChecked():
            self.reload_items()

    def _set_combo(self, combo, value):
        value = str(value or "")
        idx = combo.findText(value)
        if idx < 0:
            combo.addItem(value)
            idx = combo.findText(value)
        combo.setCurrentIndex(max(0, idx))

    def apply_selected_preset(self):
        idx = self.preset_combo.currentIndex() - 1
        if idx < 0 or idx >= len(SOURCE_PRESETS):
            return
        preset = SOURCE_PRESETS[idx]
        self.name_input.setText(preset["name"])
        self.url_input.setText(_rsshub_url(preset["url"]))
        self.tags_input.setText(preset["tags"])
        self.current_preset_max_items = preset.get("max_items")
        self._set_combo(self.platform_combo, preset.get("platform") or "auto")
        self._set_combo(self.type_combo, preset["type"])
        self.enabled_check.setChecked(True)
        self.status_label.setText("已套用常用源模板，可以按需改链接后点“添加并抓取”。")

    def add_current_source(self):
        try:
            max_items = getattr(self, "current_preset_max_items", None)
            if "/pixiv/ranking/" not in self.url_input.text().lower():
                max_items = None
            source = add_source(
                name=self.name_input.text(),
                feed_url=self.url_input.text(),
                source_type=self.type_combo.currentText(),
                platform=self.platform_combo.currentText(),
                tags=self.tags_input.text(),
                enabled=self.enabled_check.isChecked(),
                max_items=max_items,
            )
        except Exception as e:
            QMessageBox.warning(self, "添加失败", str(e))
            return
        self.reload_sources()
        for row in range(self.source_list.count()):
            if self.source_list.item(row).data(Qt.UserRole) == source.get("id"):
                self.source_list.setCurrentRow(row)
                break
        self.start_refresh(source.get("id"))

    def save_current_source(self):
        source_id = self.selected_source_id()
        if not source_id:
            QMessageBox.information(self, "未选择", "先选择一个 RSS 源。")
            return
        try:
            update_source(
                source_id,
                name=self.name_input.text(),
                feed_url=self.url_input.text(),
                type=self.type_combo.currentText(),
                platform=self.platform_combo.currentText(),
                tags=self.tags_input.text(),
                enabled=self.enabled_check.isChecked(),
            )
        except Exception as e:
            QMessageBox.warning(self, "保存失败", str(e))
            return
        self.reload_sources()

    def delete_current_source(self):
        source_id = self.selected_source_id()
        if not source_id:
            return
        reply = QMessageBox.question(
            self,
            "删除 RSS 源",
            "删除这个 RSS 源，并清除它抓到的缓存条目？",
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return
        delete_source(source_id, remove_items=True)
        self.reload_sources()
        self.reload_items()

    def refresh_selected_source(self):
        source_id = self.selected_source_id()
        if not source_id:
            QMessageBox.information(self, "未选择", "先选择一个 RSS 源。")
            return
        self.start_refresh(source_id)

    def refresh_all_sources(self):
        self.start_refresh("")

    def start_refresh(self, source_id):
        if self.refresh_thread and self.refresh_thread.isRunning():
            QMessageBox.information(self, "正在抓取", "上一次 RSS 抓取还没结束。")
            return
        self.status_label.setText("RSS 抓取中...")
        self.refresh_thread = RSSRefreshThread(source_id)
        self.refresh_thread.finished_signal.connect(self.on_refresh_finished)
        self.refresh_thread.error_signal.connect(self.on_refresh_error)
        self.refresh_thread.start()

    def on_refresh_finished(self, result):
        if result.get("reason") == "refresh_already_running":
            self.status_label.setText("已有 RSS 抓取任务在运行，稍后再试。")
            return
        added = result.get("added", 0)
        updated = result.get("updated", 0)
        errors = result.get("errors") or []
        msg = f"抓取完成：新增 {added}，更新 {updated}"
        if errors:
            msg += f"，错误 {len(errors)}"
        self.status_label.setText(msg)
        self.reload_sources()
        self.reload_items()
        if errors:
            self.detail_text.setPlainText("抓取错误：\n" + "\n".join(str(e) for e in errors))

    def on_refresh_error(self, error):
        self.status_label.setText("RSS 抓取失败")
        QMessageBox.warning(self, "RSS 抓取失败", str(error))

    def show_selected_item(self):
        item = self.selected_item()
        if not item:
            self.detail_text.clear()
            return
        lines = [
            f"标题：{item.get('title') or ''}",
            f"原链接：{item.get('url') or ''}",
            f"封面：{item.get('cover_url') or ''}",
            f"来源：{item.get('source_name') or item.get('source_id') or ''}",
            f"平台/类型：{item.get('platform') or ''} / {item.get('source_type') or ''}",
            f"发布时间：{item.get('published_at') or 'unknown'}",
            f"抓取时间：{item.get('fetched_at') or ''}",
            f"标签：{', '.join(item.get('tags') or [])}",
            "",
            "简介：",
            item.get("summary") or "",
        ]
        self.detail_text.setPlainText("\n".join(lines))

    def start_cover_download(self, row, url):
        thread = CoverDownloadThread(row, url)
        thread.finished_signal.connect(self.on_cover_downloaded)
        self.cover_threads.append(thread)
        thread.start()

    def on_cover_downloaded(self, row, content):
        if not content or row < 0 or row >= self.item_table.rowCount():
            return
        pixmap = QPixmap()
        if not pixmap.loadFromData(content):
            return
        cell = self.item_table.item(row, 0)
        if cell is None:
            return
        scaled = pixmap.scaled(96, 54, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        cell.setIcon(QIcon(scaled))
        cell.setText("")

    def open_selected_item(self):
        item = self.selected_item()
        url = (item or {}).get("url") or ""
        if url:
            QDesktopServices.openUrl(QUrl(url))

    def copy_selected_url(self):
        item = self.selected_item()
        url = (item or {}).get("url") or ""
        if not url:
            return
        QApplication.clipboard().setText(url)
        self.status_label.setText("原链接已复制")

    def clear_current_items(self):
        source_id = self.selected_source_id()
        if not source_id:
            return
        reply = QMessageBox.question(
            self,
            "清空缓存",
            "清空选中 RSS 源已经抓到的条目？源本身会保留。",
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return
        clear_cached_items(source_id)
        self.reload_items()

    def mousePressEvent(self, event):
        if window_chrome.begin_window_resize(self, event):
            return
        if window_chrome.begin_title_drag(self, event, self.title_bar):
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

    def mouseDoubleClickEvent(self, event):
        if window_chrome.title_double_click_maximize(self, event, self.title_bar):
            return
        super().mouseDoubleClickEvent(event)

    def leaveEvent(self, event):
        window_chrome.leave_resize_area(self, event)

    def nativeEvent(self, event_type, message):
        result = window_chrome.native_resize_event(self, event_type, message)
        if result:
            return result
        return super().nativeEvent(event_type, message)
