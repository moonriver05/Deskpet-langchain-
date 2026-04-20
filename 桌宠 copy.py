import os
import sys
import json
import requests
import datetime
import random
import base64
import re
import PyQt5
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

# 关键：手动指定 Qt 插件目录
plugin_path = os.path.join(os.path.dirname(PyQt5.__file__), "Qt5", "plugins")
os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = plugin_path

from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QVBoxLayout, QHBoxLayout,
    QPushButton, QLineEdit, QListWidget, QListWidgetItem,
    QCheckBox, QMessageBox, QMenu, QScrollArea, QDesktopWidget,
    QGraphicsDropShadowEffect
)
from PyQt5.QtCore import Qt, QPoint, QTimer, QSize, QThread, pyqtSignal
from PyQt5.QtGui import QPixmap, QPainter, QFont, QColor, QPolygon, QMovie, QIcon


# ==================== 桌宠主体 ====================
class DesktopPet(QWidget):
    def __init__(self):
        super().__init__()
        self.todo_window = None
        self.chat_window = None
        self.offset = QPoint()
        self.current_frame = 0
        self.is_happy = False
        self.happy_timer = 0
        
        # 聊天和提醒相关计时
        self.chat_timer = QTimer()
        self.chat_timer.timeout.connect(self.random_chat)
        self.chat_timer.start(1000 * 60 * 15)  # 15分钟随机闲聊一次
        
        self.sit_timer = QTimer()
        self.sit_timer.timeout.connect(self.remind_stand_up)
        self.sit_timer.start(1000 * 60 * 60)  # 60分钟提醒一次久坐
        
        self.water_timer = QTimer()
        self.water_timer.timeout.connect(self.remind_drink_water)
        self.water_timer.start(1000 * 60 * 45)  # 45分钟提醒一次喝水

        self.init_ui()
        self.init_animation()
        self.check_special_day()

    def init_ui(self):
        self.setWindowFlags(
            Qt.FramelessWindowHint |
            Qt.WindowStaysOnTopHint |
            Qt.Tool
        )
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setFixedSize(180, 200)

        self.pet_label = QLabel(self)
        self.pet_label.setAlignment(Qt.AlignCenter)
        self.pet_label.setGeometry(0, 30, 180, 170)

        # 气泡容器
        self.bubble_container = QWidget(self)
        self.bubble_container.setGeometry(0, 0, 180, 50)
        self.bubble_container.hide()
        
        # 气泡背景和文字
        self.bubble = QLabel(self.bubble_container)
        self.bubble.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.bubble.setWordWrap(True)
        self.bubble.setStyleSheet("""
            QLabel {
                background-color: rgba(255, 255, 230, 230);
                border: 2px solid #FFB347;
                border-radius: 10px;
                padding: 5px 25px 5px 10px;
                font-size: 12px;
                color: #333;
                font-family: "Microsoft YaHei";
            }
        """)
        self.bubble.setGeometry(0, 0, 180, 50)
        
        # 气泡关闭按钮
        self.close_bubble_btn = QPushButton("✕", self.bubble_container)
        self.close_bubble_btn.setGeometry(160, 5, 15, 15)
        self.close_bubble_btn.setStyleSheet("""
            QPushButton {
                background-color: transparent;
                color: #999;
                border: none;
                font-size: 12px;
                font-weight: bold;
            }
            QPushButton:hover {
                color: #FF6B6B;
            }
        """)
        self.close_bubble_btn.clicked.connect(self.hide_bubble)
        
        # 打字机效果相关属性
        self.typewriter_timer = QTimer()
        self.typewriter_timer.timeout.connect(self.type_next_char)
        self.full_text = ""
        self.current_text = ""
        self.char_index = 0
        self.bubble_hide_timer = QTimer()
        self.bubble_hide_timer.timeout.connect(self.hide_bubble)
        self.bubble_hide_timer.setSingleShot(True)

        screen_obj = QApplication.primaryScreen()
        if screen_obj:
            screen = screen_obj.geometry()
            self.move(screen.width() - 250, screen.height() - 300)
        else:
            self.move(100, 100)

    def init_animation(self):
        self.pet_movie = None
        gif_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "有珠.gif")
        if os.path.exists(gif_path):
            self.pet_movie = QMovie(gif_path)
            self.pet_movie.setScaledSize(QSize(180, 170))
            self.pet_label.setMovie(self.pet_movie)
            self.pet_movie.start()
        else:
            self.normal_frames = [
                self.create_pet_pixmap("😺", "ᓚᘏᗢ"),
                self.create_pet_pixmap("😺", "ᓚᘏᗢ ~"),
                self.create_pet_pixmap("😺", "ᓚᘏᗢ  ~"),
                self.create_pet_pixmap("😸", "ᓚᘏᗢ"),
            ]
            self.happy_frames = [
                self.create_pet_pixmap("😻", "♡"),
                self.create_pet_pixmap("🥰", "♡♡"),
                self.create_pet_pixmap("😻", "♡♡♡"),
            ]
            self.pet_label.setPixmap(self.normal_frames[0])

        self.anim_timer = QTimer()
        self.anim_timer.timeout.connect(self.update_animation)
        self.anim_timer.start(500)

    def create_pet_pixmap(self, emoji, decoration=""):
        pixmap = QPixmap(180, 170)
        pixmap.fill(Qt.transparent)

        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)

        # 身体
        painter.setBrush(QColor(255, 200, 100, 200))
        painter.setPen(QColor(200, 150, 50))
        painter.drawEllipse(30, 30, 120, 120)

        # 耳朵
        painter.setBrush(QColor(255, 180, 80, 200))

        left_ear = QPolygon([
            QPoint(45, 45),
            QPoint(30, 5),
            QPoint(70, 35)
        ])
        painter.drawPolygon(left_ear)

        right_ear = QPolygon([
            QPoint(135, 45),
            QPoint(150, 5),
            QPoint(110, 35)
        ])
        painter.drawPolygon(right_ear)

        # 脸
        font = QFont("Segoe UI Emoji", 40)
        painter.setFont(font)
        painter.drawText(55, 115, emoji)

        # 装饰文字
        font2 = QFont("Arial", 14)
        painter.setFont(font2)
        painter.setPen(QColor(255, 100, 100))
        painter.drawText(50, 165, decoration)

        painter.end()
        return pixmap

    def update_animation(self):
        if self.pet_movie is not None:
            if self.is_happy:
                self.pet_movie.setSpeed(150)
                self.happy_timer -= 1
                if self.happy_timer <= 0:
                    self.is_happy = False
            else:
                self.pet_movie.setSpeed(100)
            return

        if self.is_happy:
            frames = self.happy_frames
            self.happy_timer -= 1
            if self.happy_timer <= 0:
                self.is_happy = False
        else:
            frames = self.normal_frames

        self.current_frame = (self.current_frame + 1) % len(frames)
        self.pet_label.setPixmap(frames[self.current_frame])

    def check_special_day(self):
        now = datetime.datetime.now()
        month, day = now.month, now.day
        
        # 定义节日和对应的问候语、皮肤
        # 默认生日设为 5月20日，你可以修改为自己的生日
        self.special_days = {
            (1, 1): {"greeting": "元旦快乐！新的一年也要加油哦！🎉", "skin": "有珠.gif"},
            (2, 14): {"greeting": "情人节快乐！今天也要开心呀~ 💖", "skin": "有珠.gif"},
            (9, 19): {"greeting": "生日快乐！愿你今天是最幸福的人！🎂🎁", "skin": "有珠.gif"}, # 修改为你的生日
            (10, 1): {"greeting": "国庆节快乐！好好休息一下吧！🇨🇳", "skin": "有珠.gif"},
            (12, 25): {"greeting": "圣诞快乐！收到礼物了吗？🎄🎅", "skin": "有珠.gif"}
        }
        
        today = (month, day)
        if today in self.special_days:
            info = self.special_days[today]
            self.show_bubble(info["greeting"], duration=8000)
            self.set_happy()
            
            # 尝试加载节日专属皮肤，如果找不到则回退到默认的 有珠.gif
            skin_name = info.get("skin", "有珠.gif")
            skin_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), skin_name)
            if not os.path.exists(skin_path):
                skin_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "有珠.gif")
                
            if os.path.exists(skin_path) and self.pet_movie:
                self.pet_movie.stop()
                self.pet_movie.setFileName(skin_path)
                self.pet_movie.start()

    def random_chat(self):
        # 只有在没有气泡显示时才闲聊
        if self.bubble_container.isVisible():
            return
            
        now = datetime.datetime.now()
        hour = now.hour
        
        messages = [
            "今天也要元气满满哦！✨",
            "有需要帮忙的随时叫我~ ",
            "发呆中... o(￣▽￣)ｄ",
            "代码写得怎么样啦？💻",
            "偶尔看看窗外，让眼睛休息一下吧~ 🌲",
            "好想吃年糕呀... "
        ]
        
        # 根据时间段增加特定对话
        if 6 <= hour < 9:
            messages.append("早上好呀！新的一天开始了！🌅")
            messages.append("吃早饭了吗？一定要吃早饭哦！🍞")
        elif 11 <= hour <= 13:
            messages.append("到饭点啦，准备吃什么好吃的？🍱")
            messages.append("午休时间，小憩一会吧~ 💤")
        elif 22 <= hour or hour < 2:
            messages.append("夜深了，该准备睡觉啦！不要熬夜哦~ 🌙")
            messages.append("还在肝代码吗？注意身体呀！🦉")
            
        msg = random.choice(messages)
        self.show_bubble(msg, duration=5000)

    def remind_drink_water(self):
        self.set_happy()
        self.show_bubble("叮铃铃~ 喝水时间到！快去喝杯水吧！💧", duration=6000)
        
    def remind_stand_up(self):
        self.set_happy()
        self.show_bubble("坐了很久啦，站起来活动一下筋骨吧！🏃‍♂️", duration=6000)

    def show_bubble(self, text, duration=3000):
        self.typewriter_timer.stop()
        self.bubble_hide_timer.stop()
        
        self.full_text = text
        self.current_text = ""
        self.char_index = 0
        self.bubble.setText("")
        
        # 自适应气泡高度
        self.bubble.setText(text)
        self.bubble.adjustSize()
        height = max(50, self.bubble.height() + 10)
        self.bubble_container.setGeometry(0, 0, 180, height)
        self.bubble.setGeometry(0, 0, 180, height)
        self.bubble.setText("")
        
        self.bubble_container.show()
        
        # 存储气泡停留时间
        self.bubble_duration = duration
        self.typewriter_timer.start(50)  # 每个字 50ms 的速度

    def type_next_char(self):
        if self.char_index < len(self.full_text):
            self.current_text += self.full_text[self.char_index]
            self.bubble.setText(self.current_text)
            self.char_index += 1
        else:
            self.typewriter_timer.stop()
            if self.bubble_duration > 0:
                self.bubble_hide_timer.start(self.bubble_duration)
                
    def hide_bubble(self):
        self.typewriter_timer.stop()
        self.bubble_hide_timer.stop()
        self.bubble_container.hide()

    def set_happy(self):
        self.is_happy = True
        self.happy_timer = 6
        if self.pet_movie is not None:
            self.pet_movie.setSpeed(150)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.offset = event.pos()

    def mouseMoveEvent(self, event):
        if event.buttons() == Qt.LeftButton:
            self.move(self.mapToGlobal(event.pos() - self.offset))

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.open_todo()

    def contextMenuEvent(self, event):
        menu = QMenu(self)
        menu.setStyleSheet("""
            QMenu {
                background-color: #FFF8E7;
                border: 2px solid #FFB347;
                border-radius: 8px;
                padding: 5px;
                font-size: 13px;
            }
            QMenu::item {
                padding: 8px 25px;
                border-radius: 4px;
            }
            QMenu::item:selected {
                background-color: #FFE0B2;
            }
        """)

        todo_action = menu.addAction("📋 打开待办清单")
        chat_action = menu.addAction("💬 聊天")
        pet_action = menu.addAction("🐱 摸摸我")
        menu.addSeparator()
        quit_action = menu.addAction("👋 退出")

        action = menu.exec_(event.globalPos())
        if action == todo_action:
            self.open_todo()
        elif action == chat_action:
            self.open_chat()
        elif action == pet_action:
            self.set_happy()
            self.show_bubble("喵~ 好舒服！(=^･ω･^=)")
        elif action == quit_action:
            reply = QMessageBox.question(
                self, '确认', '真的要离开我吗？🥺',
                QMessageBox.Yes | QMessageBox.No
            )
            if reply == QMessageBox.Yes:
                QApplication.quit()

    def open_chat(self):
        if self.chat_window is None or not self.chat_window.isVisible():
            self.chat_window = ChatWindow(self)
            
            # 让聊天窗口居中显示
            screen = QDesktopWidget().screenGeometry()
            size = self.chat_window.geometry()
            x = (screen.width() - size.width()) // 2
            y = (screen.height() - size.height()) // 2
            self.chat_window.move(x, y)
            
            self.chat_window.show()
            self.show_bubble("来聊天吧！", duration=3000)
        else:
            self.chat_window.activateWindow()

    def open_todo(self):
        if self.todo_window is None or not self.todo_window.isVisible():
            self.todo_window = TodoWindow(self)
            pos = self.mapToGlobal(QPoint(-270, -200))
            self.todo_window.move(pos)
            self.todo_window.show()
            self.show_bubble("要开始干活啦！💪")
        else:
            self.todo_window.activateWindow()


# ==================== 远程图片下载线程 ====================
class ImageDownloader(QThread):
    finished_signal = pyqtSignal(object, object)  # QPixmap, QLabel

    def __init__(self, url, label):
        super().__init__()
        self.url = url
        self.label = label

    def run(self):
        try:
            resp = requests.get(self.url, timeout=15)
            if resp.status_code == 200:
                pixmap = QPixmap()
                pixmap.loadFromData(resp.content)
                self.finished_signal.emit(pixmap, self.label)
            else:
                self.finished_signal.emit(QPixmap(), self.label)
        except Exception:
            self.finished_signal.emit(QPixmap(), self.label)

# ==================== LLM 请求线程 ====================
class LLMFetcherThread(QThread):
    finished_signal = pyqtSignal(str)
    error_signal = pyqtSignal(str)

    def __init__(self, user_text, image_path=None):
        super().__init__()
        self.user_text = user_text
        self.image_path = image_path

    def run(self):
        try:
            api_key = "你的api"
            base_url = "你的模型地址"

            # 使用 langchain_openai 的 ChatOpenAI 接口
            llm = ChatOpenAI(
                model="你的模型",
                openai_api_key=api_key,
                openai_api_base=base_url,
                max_tokens=2048,
                temperature=0.7,
            )

            # 构造符合 langchain 格式的消息
            content = []
            if self.user_text:
                content.append({"type": "text", "text": self.user_text})
            else:
                content.append({"type": "text", "text": "请看这张图片"})
                
            # 支持本地图片上传（转换为 Base64 格式发送给大模型）
            if self.image_path and os.path.exists(self.image_path):
                with open(self.image_path, "rb") as f:
                    base64_data = base64.b64encode(f.read()).decode('utf-8')
                
                # 获取后缀名
                ext = os.path.splitext(self.image_path)[1].lower().replace('.', '')
                if ext == 'jpg':
                    ext = 'jpeg'
                    
                content.insert(0, {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/{ext};base64,{base64_data}"}
                })
            elif "图片" in self.user_text or "看看" in self.user_text:
                # 如果用户说了“图片”相关的词且没有拖拽图片，则自动发送演示图片
                content.insert(0, {
                    "type": "image_url",
                    "image_url": {"url": "img的地址"}
                })

            message = HumanMessage(content=content)
            
            # 调用大模型
            response = llm.invoke([message])
            reply_text = response.content

            # 修复可能存在的转义换行符，将其转换为自然换行
            if isinstance(reply_text, str):
                reply_text = reply_text.replace("\\n", "\n").strip()
                
            self.finished_signal.emit(reply_text)

        except Exception as e:
            self.error_signal.emit(f"API 请求出错: {str(e)}")

# ==================== 聊天窗口 ====================
class ChatWindow(QWidget):
    def __init__(self, pet=None):
        super().__init__()
        self.pet = pet
        self.pending_image_path = None
        self.setAcceptDrops(True)
        self.init_ui()

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            if urls and urls[0].isLocalFile():
                ext = urls[0].toLocalFile().lower()
                if ext.endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp')):
                    event.accept()
                    return
        event.ignore()

    def dropEvent(self, event):
        path = event.mimeData().urls()[0].toLocalFile()
        self.pending_image_path = path
        pixmap = QPixmap(path).scaled(60, 60, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.img_preview_label.setPixmap(pixmap)
        self.img_preview_container.show()

    def init_ui(self):
        self.setWindowTitle("💬 与有珠聊天")
        
        # 隐藏左上角默认的程序图标（使用 1x1 像素的透明图标替代）
        transparent_pixmap = QPixmap(1, 1)
        transparent_pixmap.fill(Qt.transparent)
        self.setWindowIcon(QIcon(transparent_pixmap))
        
        self.resize(400, 600)
        self.setStyleSheet("background-color: #F5F5F5;")
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # 聊天记录区域
        self.scroll_area = QScrollArea(self)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setStyleSheet("border: none; background: #F5F5F5;")
        self.scroll_area.verticalScrollBar().setStyleSheet("""
            QScrollBar:vertical {
                border: none;
                background: transparent;
                width: 8px;
                margin: 0px 0px 0px 0px;
            }
            QScrollBar::handle:vertical {
                background: #CCCCCC;
                min-height: 20px;
                border-radius: 4px;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                border: none;
                background: none;
            }
        """)
        
        self.msg_container = QWidget()
        self.msg_container.setStyleSheet("background: transparent;")
        self.msg_layout = QVBoxLayout(self.msg_container)
        self.msg_layout.setContentsMargins(15, 15, 15, 15)
        self.msg_layout.setSpacing(15)
        self.msg_layout.addStretch()
        
        self.scroll_area.setWidget(self.msg_container)
        layout.addWidget(self.scroll_area)
        
        # 底部输入区域
        input_area = QWidget()
        input_area.setStyleSheet("background-color: #F5F5F5; border-top: 1px solid #E5E5E5;")
        input_area_layout = QVBoxLayout(input_area)
        input_area_layout.setContentsMargins(15, 10, 15, 15)
        input_area_layout.setSpacing(5)
        
        # 图片预览容器
        self.img_preview_container = QWidget()
        self.img_preview_container.hide()
        img_preview_layout = QHBoxLayout(self.img_preview_container)
        img_preview_layout.setContentsMargins(0, 0, 0, 0)
        
        self.img_preview_label = QLabel()
        self.img_preview_label.setFixedSize(60, 60)
        self.img_preview_label.setStyleSheet("background-color: #E0E0E0; border-radius: 4px;")
        self.img_preview_label.setAlignment(Qt.AlignCenter)
        
        self.clear_img_btn = QPushButton("✕")
        self.clear_img_btn.setFixedSize(20, 20)
        self.clear_img_btn.setStyleSheet("""
            QPushButton { background-color: #FF6B6B; color: white; border-radius: 10px; font-weight: bold; }
            QPushButton:hover { background-color: #FF4C4C; }
        """)
        self.clear_img_btn.clicked.connect(self.clear_pending_image)
        
        img_preview_layout.addWidget(self.img_preview_label)
        img_preview_layout.addWidget(self.clear_img_btn, 0, Qt.AlignTop | Qt.AlignLeft)
        img_preview_layout.addStretch()
        
        input_area_layout.addWidget(self.img_preview_container)
        
        # 文本输入和发送按钮
        input_layout = QHBoxLayout()
        input_layout.setContentsMargins(0, 0, 0, 0)
        
        self.input_field = QLineEdit()
        self.input_field.setPlaceholderText("想说点什么... (支持拖拽图片上传)")
        self.input_field.setStyleSheet("""
            QLineEdit {
                background-color: white;
                border: none;
                border-radius: 6px;
                padding: 10px;
                font-size: 14px;
                font-family: "Microsoft YaHei";
            }
        """)
        self.input_field.returnPressed.connect(self.send_message)
        
        self.send_btn = QPushButton("发送")
        self.send_btn.setFixedSize(65, 36)
        self.send_btn.setStyleSheet("""
            QPushButton {
                background-color: #07C160;
                color: white;
                border: none;
                border-radius: 6px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #06AD56;
            }
            QPushButton:pressed {
                background-color: #059A4C;
            }
        """)
        self.send_btn.clicked.connect(self.send_message)
        
        input_layout.addWidget(self.input_field)
        input_layout.addWidget(self.send_btn)
        
        input_area_layout.addLayout(input_layout)
        layout.addWidget(input_area)
        
        # 初始问候语
        QTimer.singleShot(200, lambda: self.add_message("你好呀！找我有什么事吗？喵~ (可以拖拽图片给我哦)", is_user=False))

    def clear_pending_image(self):
        self.pending_image_path = None
        self.img_preview_container.hide()
        self.img_preview_label.clear()

    def send_message(self):
        text = self.input_field.text().strip()
        img_path = self.pending_image_path
        
        if not text and not img_path:
            return
            
        # 添加用户消息
        self.add_message(text, is_user=True, image_path=img_path)
        self.input_field.clear()
        self.clear_pending_image()
        
        # 如果绑定了桌宠，让它做出回应
        if self.pet:
            self.pet.set_happy()
            
        # 禁用发送按钮，防止重复提交
        self.send_btn.setEnabled(False)
        self.send_btn.setText("思考中...")
        
        # 启动 LLM 线程进行异步调用
        self.llm_thread = LLMFetcherThread(text, img_path)
        self.llm_thread.finished_signal.connect(self.on_llm_reply)
        self.llm_thread.error_signal.connect(self.on_llm_error)
        self.llm_thread.start()

    def on_llm_reply(self, reply_text):
        self.send_btn.setEnabled(True)
        self.send_btn.setText("发送")
        self.add_message(reply_text, is_user=False)
        
        if self.pet:
            self.pet.show_bubble("我回复你啦~", duration=3000)

    def on_llm_error(self, error_msg):
        self.send_btn.setEnabled(True)
        self.send_btn.setText("发送")
        self.add_message(f"⚠️ {error_msg}", is_user=False)

    def add_message(self, text, is_user=True, image_path=None):
        msg_widget = QWidget()
        msg_widget.setStyleSheet("background: transparent;")
        h_layout = QHBoxLayout(msg_widget)
        h_layout.setContentsMargins(0, 0, 0, 0)
        
        # 头像
        avatar = QLabel()
        avatar.setFixedSize(36, 36)
        avatar.setAlignment(Qt.AlignCenter)
        
        # 加载本地图片作为头像
        img_name = "用户头像.jpg" if is_user else "有珠.png"
        img_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), img_name)
        
        if os.path.exists(img_path):
            pixmap = QPixmap(img_path)
            # 缩放图片以适应固定大小，保持比例并平滑转换
            pixmap = pixmap.scaled(36, 36, Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation)
            avatar.setPixmap(pixmap)
            avatar.setStyleSheet("""
                border-radius: 6px;
            """)
        else:
            # 如果图片不存在，回退到文字/emoji 占位
            avatar.setText("👤" if is_user else "🐱")
            avatar.setStyleSheet(f"""
                background-color: {'#E2E2E2' if is_user else '#FFF'};
                border-radius: 6px;
                font-size: 20px;
            """)
        
        # 消息内容容器
        bubble_container = QWidget()
        bubble_layout = QVBoxLayout(bubble_container)
        bubble_layout.setContentsMargins(10, 10, 10, 10)
        bubble_layout.setSpacing(5)
        
        # 处理本地发送的图片
        if image_path and os.path.exists(image_path):
            img_label = QLabel()
            pixmap = QPixmap(image_path)
            max_img_w = 200
            if pixmap.width() > max_img_w:
                pixmap = pixmap.scaledToWidth(max_img_w, Qt.SmoothTransformation)
            img_label.setPixmap(pixmap)
            img_label.setStyleSheet("border-radius: 4px;")
            bubble_layout.addWidget(img_label)
            
        # 提取大模型返回的 Markdown 图片链接
        remote_image_urls = []
        if not is_user and text:
            pattern = r"!\[.*?\]\((.*?)\)"
            remote_image_urls = re.findall(pattern, text)
            # 从原文本中移除 markdown 语法
            text = re.sub(pattern, "", text).strip()
        
        # 气泡文字
        if text:
            bubble = QLabel(text)
            bubble.setWordWrap(True)
            bubble.setTextInteractionFlags(Qt.TextSelectableByMouse)
            
            # 限制气泡最大宽度
            max_width = int(self.width() * 0.65)
            bubble.setMaximumWidth(max_width)
            bubble.setStyleSheet("""
                font-size: 14px;
                font-family: "Microsoft YaHei";
                color: #333;
                border: none;
                background: transparent;
            """)
            bubble_layout.addWidget(bubble)
            
        # 异步加载远程图片
        for url in remote_image_urls:
            img_label = QLabel("加载图片中...")
            img_label.setStyleSheet("color: #888; font-style: italic; background: transparent; border: none;")
            bubble_layout.addWidget(img_label)
            
            downloader = ImageDownloader(url, img_label)
            if not hasattr(self, 'downloaders'):
                self.downloaders = []
            self.downloaders.append(downloader)
            
            def on_download_finished(pixmap, label):
                if not pixmap.isNull():
                    max_img_w = 200
                    if pixmap.width() > max_img_w:
                        pixmap = pixmap.scaledToWidth(max_img_w, Qt.SmoothTransformation)
                    label.setPixmap(pixmap)
                    label.setText("")
                else:
                    label.setText("图片加载失败")
                self.scroll_to_bottom()
                
            downloader.finished_signal.connect(on_download_finished)
            downloader.start()
        
        if is_user:
            # 用户：靠右，绿色气泡
            bubble_container.setStyleSheet("""
                background-color: #95EC69;
                border-radius: 8px;
            """)
            h_layout.addStretch()
            h_layout.addWidget(bubble_container)
            h_layout.addWidget(avatar)
        else:
            # 机器人：靠左，白色气泡
            bubble_container.setStyleSheet("""
                background-color: white;
                border-radius: 8px;
            """)
            h_layout.addWidget(avatar)
            h_layout.addWidget(bubble_container)
            h_layout.addStretch()
            
        # 插入到弹簧的前面
        self.msg_layout.insertWidget(self.msg_layout.count() - 1, msg_widget)
        
        # 自动滚动到底部
        QTimer.singleShot(50, self.scroll_to_bottom)

    def scroll_to_bottom(self):
        scrollbar = self.scroll_area.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())


# ==================== To-Do List 窗口 ====================
class HomeworkFetcherThread(QThread):
    finished_signal = pyqtSignal(list)
    error_signal = pyqtSignal(str)
    #这里是登录之后找到你的登录token获取课程表，再根据课程表遍历获取作业。
    def run(self):
        try:
            homework_url = "这是作业地址"
            clas = "课程地址"
            session = requests.Session()
            lg_url = "登陆位置"
            lg_data = {
                "loginName": "登陆账号",
                "password": "登陆密码"
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
                                "completed": False
                            })
                
                self.finished_signal.emit(all_homeworks)
            else:
                self.error_signal.emit("登录失败")
        except Exception as e:
            self.error_signal.emit(str(e))

class TodoWindow(QWidget):
    DATA_FILE = "todo_data.json"

    def __init__(self, pet=None):
        super().__init__()
        self.pet = pet
        self.drag_offset = QPoint()
        self.init_ui()
        self.load_data()

    def init_ui(self):
        self.setWindowTitle("🐱 喵喵待办清单")
        self.setFixedSize(320, 480)
        self.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground)

        container = QWidget(self)
        container.setGeometry(10, 10, 300, 460)
        container.setStyleSheet("""
            QWidget {
                background-color: #FFF8E7;
                border-radius: 20px;
            }
        """)

        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(20)
        shadow.setColor(QColor(0, 0, 0, 60))
        shadow.setOffset(0, 5)
        container.setGraphicsEffect(shadow)

        layout = QVBoxLayout(container)
        layout.setContentsMargins(20, 15, 20, 15)
        layout.setSpacing(10)

        title_layout = QHBoxLayout()
        title_layout.setSpacing(5) # 减小标题栏控件之间的间距
        
        title = QLabel("📋 我的待办")
        # 适当减小标题字号防止拥挤
        title.setFont(QFont("Microsoft YaHei", 14, QFont.Bold))
        title.setStyleSheet("color: #E67E22; background: transparent;")
        title_layout.addWidget(title)
        
        # 添加一个弹簧将按钮推向右侧
        title_layout.addStretch()

        self.sync_btn = QPushButton("🔄 获取作业")
        self.sync_btn.setFixedSize(76, 26) # 稍微减小按钮宽度
        self.sync_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                border-radius: 13px;
                font-size: 12px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        self.sync_btn.clicked.connect(self.sync_homework)
        title_layout.addWidget(self.sync_btn)

        close_btn = QPushButton("✕")
        close_btn.setFixedSize(30, 30)
        close_btn.setStyleSheet("""
            QPushButton {
                background-color: #FF6B6B;
                color: white;
                border: none;
                border-radius: 15px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #EE5A5A;
            }
        """)
        close_btn.clicked.connect(self.close)
        title_layout.addWidget(close_btn)
        layout.addLayout(title_layout)

        input_layout = QHBoxLayout()
        self.input_field = QLineEdit()
        self.input_field.setPlaceholderText("✨ 添加新的待办事项...")
        self.input_field.setStyleSheet("""
            QLineEdit {
                border: 2px solid #FFB347;
                border-radius: 12px;
                padding: 10px 15px;
                font-size: 13px;
                background-color: white;
            }
            QLineEdit:focus {
                border-color: #E67E22;
            }
        """)
        self.input_field.returnPressed.connect(self.add_todo)
        input_layout.addWidget(self.input_field)

        add_btn = QPushButton("+")
        add_btn.setFixedSize(40, 40)
        add_btn.setStyleSheet("""
            QPushButton {
                background-color: #FFB347;
                color: white;
                border: none;
                border-radius: 20px;
                font-size: 22px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #E67E22;
            }
        """)
        add_btn.clicked.connect(self.add_todo)
        input_layout.addWidget(add_btn)
        layout.addLayout(input_layout)

        self.todo_list = QListWidget()
        self.todo_list.setStyleSheet("""
            QListWidget {
                border: none;
                background-color: transparent;
                font-size: 14px;
            }
            QListWidget::item {
                background-color: white;
                border-radius: 10px;
                margin: 3px 0px;
                padding: 5px;
            }
            QListWidget::item:hover {
                background-color: #FFF0D0;
            }
        """)
        layout.addWidget(self.todo_list)

        self.stats_label = QLabel("还没有待办事项哦~")
        self.stats_label.setAlignment(Qt.AlignCenter)
        self.stats_label.setStyleSheet("""
            color: #999;
            font-size: 12px;
            background: transparent;
        """)
        layout.addWidget(self.stats_label)

    def sync_homework(self):
        self.sync_btn.setText("获取中...")
        self.sync_btn.setEnabled(False)
        self.fetcher_thread = HomeworkFetcherThread()
        self.fetcher_thread.finished_signal.connect(self.on_homework_fetched)
        self.fetcher_thread.error_signal.connect(self.on_homework_error)
        self.fetcher_thread.start()

    def on_homework_fetched(self, homeworks):
        self.sync_btn.setText("🔄 获取作业")
        self.sync_btn.setEnabled(True)
        
        # Check existing to avoid duplicates based on title
        existing_titles = set()
        for i in range(self.todo_list.count()):
            item = self.todo_list.item(i)
            widget = self.todo_list.itemWidget(item)
            if widget:
                checkbox = widget.findChild(QCheckBox)
                if checkbox:
                    existing_titles.add(checkbox.text())
        
        added_count = 0
        for hw in homeworks:
            if hw["text"] not in existing_titles:
                self.add_todo_item(hw)
                added_count += 1
                
        self.save_data()
        self.update_stats()
        
        if self.pet:
            if added_count > 0:
                self.pet.set_happy()
                self.pet.show_bubble(f"获取成功！新增了 {added_count} 个作业任务！📚")
            else:
                self.pet.show_bubble("获取成功！没有新的作业哦~ 🎉")

    def on_homework_error(self, error_msg):
        self.sync_btn.setText("🔄 获取作业")
        self.sync_btn.setEnabled(True)
        if self.pet:
            self.pet.show_bubble(f"获取失败：{error_msg} 😢")

    def add_todo(self):
        text = self.input_field.text().strip()
        if not text:
            return

        self.add_todo_item({"text": text, "is_homework": False, "completed": False})
        self.input_field.clear()
        self.save_data()
        self.update_stats()

        if self.pet:
            self.pet.set_happy()
            self.pet.show_bubble("收到！加油完成它！✨")

    def add_todo_item(self, item_data, legacy_completed=False):
        if isinstance(item_data, str):
            text = item_data
            is_homework = False
            completed = legacy_completed
            teacher = ""
            endtime = ""
        else:
            text = item_data.get("text", "")
            is_homework = item_data.get("is_homework", False)
            completed = item_data.get("completed", False)
            teacher = item_data.get("teacher", "")
            endtime = item_data.get("endtime", "")

        item = QListWidgetItem()
        if is_homework:
            item.setSizeHint(QSize(0, 85))
        else:
            item.setSizeHint(QSize(0, 45))
        self.todo_list.addItem(item)

        widget = QWidget()
        widget.setProperty("is_homework", is_homework)
        widget.setProperty("teacher", teacher)
        widget.setProperty("endtime", endtime)
        widget.setStyleSheet("background: transparent;")
        h_layout = QHBoxLayout(widget)
        h_layout.setContentsMargins(10, 5, 5, 5)

        text_layout = QVBoxLayout()
        text_layout.setSpacing(4)
        
        checkbox = QCheckBox(text)
        checkbox.setChecked(completed)
        checkbox.setStyleSheet(f"""
            QCheckBox {{
                font-size: 13px;
                color: {"#999" if completed else "#333"};
                background: transparent;
                spacing: 8px;
            }}
            QCheckBox::indicator {{
                width: 20px;
                height: 20px;
                border-radius: 10px;
                border: 2px solid #FFB347;
            }}
            QCheckBox::indicator:checked {{
                background-color: #4CAF50;
                border-color: #4CAF50;
            }}
        """)

        def on_toggle(checked):
            style_color = "#999" if checked else "#333"
            checkbox.setStyleSheet(f"""
                QCheckBox {{
                    font-size: 13px;
                    color: {style_color};
                    background: transparent;
                    spacing: 8px;
                }}
                QCheckBox::indicator {{
                    width: 20px;
                    height: 20px;
                    border-radius: 10px;
                    border: 2px solid #FFB347;
                }}
                QCheckBox::indicator:checked {{
                    background-color: #4CAF50;
                    border-color: #4CAF50;
                }}
            """)
            self.save_data()
            self.update_stats()
            if checked and self.pet:
                self.pet.set_happy()
                self.pet.show_bubble("太棒了！完成一个！🎉")

        checkbox.toggled.connect(on_toggle)
        text_layout.addWidget(checkbox)
        
        if is_homework:
            details_label = QLabel(f"👨‍🏫 {teacher}\n⏰ {endtime}")
            details_label.setWordWrap(True)
            details_label.setStyleSheet("color: #888; font-size: 11px; margin-left: 28px; line-height: 1.2;")
            text_layout.addWidget(details_label)

        h_layout.addLayout(text_layout)

        del_btn = QPushButton("🗑")
        del_btn.setFixedSize(30, 30)
        del_btn.setStyleSheet("""
            QPushButton {
                background: transparent;
                border: none;
                font-size: 16px;
            }
            QPushButton:hover {
                background-color: #FFE0E0;
                border-radius: 15px;
            }
        """)

        def on_delete():
            row = self.todo_list.row(item)
            self.todo_list.takeItem(row)
            self.save_data()
            self.update_stats()

        del_btn.clicked.connect(on_delete)
        
        btn_layout = QVBoxLayout()
        btn_layout.addWidget(del_btn)
        btn_layout.setAlignment(Qt.AlignTop)
        
        h_layout.addLayout(btn_layout)

        self.todo_list.setItemWidget(item, widget)

    def update_stats(self):
        total = self.todo_list.count()
        if total == 0:
            self.stats_label.setText("还没有待办事项哦~")
            return

        completed = 0
        for i in range(total):
            item = self.todo_list.item(i)
            widget = self.todo_list.itemWidget(item)
            if widget:
                checkbox = widget.findChild(QCheckBox)
                if checkbox and checkbox.isChecked():
                    completed += 1

        self.stats_label.setText(
            f"共 {total} 项 | 已完成 {completed} 项 | 还剩 {total - completed} 项 💪"
        )

    def save_data(self):
        todos = []
        for i in range(self.todo_list.count()):
            item = self.todo_list.item(i)
            widget = self.todo_list.itemWidget(item)
            if widget:
                checkbox = widget.findChild(QCheckBox)
                if checkbox:
                    is_hw = widget.property("is_homework")
                    todos.append({
                        "text": checkbox.text(),
                        "completed": checkbox.isChecked(),
                        "is_homework": is_hw,
                        "teacher": widget.property("teacher") if is_hw else "",
                        "endtime": widget.property("endtime") if is_hw else ""
                    })

        with open(self.DATA_FILE, "w", encoding="utf-8") as f:
            json.dump(todos, f, ensure_ascii=False, indent=2)

    def load_data(self):
        if os.path.exists(self.DATA_FILE):
            try:
                with open(self.DATA_FILE, "r", encoding="utf-8") as f:
                    todos = json.load(f)
                for item in todos:
                    if isinstance(item, dict):
                        if "is_homework" in item:
                            self.add_todo_item(item)
                        else:
                            self.add_todo_item(item["text"], item.get("completed", False))
                    else:
                        pass
                self.update_stats()
            except Exception as e:
                print("加载数据失败：", e)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drag_offset = event.pos()

    def mouseMoveEvent(self, event):
        if event.buttons() == Qt.LeftButton:
            self.move(self.mapToGlobal(event.pos() - self.drag_offset))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)

    pet = DesktopPet()
    pet.show()
    pet.show_bubble("你好呀！双击我打开待办清单~ 🐱")

    sys.exit(app.exec_())
