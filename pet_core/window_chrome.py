"""Reusable helpers for frameless, resizable Qt windows."""

import ctypes
import ctypes.wintypes

from PyQt5.QtCore import QPoint, QRect, Qt
from PyQt5.QtWidgets import (
    QComboBox,
    QLineEdit,
    QListWidget,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QTableWidget,
    QTextEdit,
)


WM_NCHITTEST = 0x0084
HTLEFT = 10
HTRIGHT = 11
HTTOP = 12
HTTOPLEFT = 13
HTTOPRIGHT = 14
HTBOTTOM = 15
HTBOTTOMLEFT = 16
HTBOTTOMRIGHT = 17


def setup_resizable_frameless_window(window, *, minimum_size=None, resize_margin=10):
    window._resize_margin = int(resize_margin)
    window._resizing = False
    window._resize_edges = set()
    window._resize_start_pos = None
    window._resize_start_geometry = None
    window._dragging = getattr(window, "_dragging", False)
    window._drag_offset = getattr(window, "_drag_offset", None)
    if minimum_size:
        window.setMinimumSize(*minimum_size)
    window.setMouseTracking(True)


def sync_maximize_button(window):
    btn = getattr(window, "maximize_btn", None)
    if btn is not None:
        btn.setText("❐" if window.isMaximized() else "□")


def toggle_maximize_restore(window):
    if window.isMaximized():
        window.showNormal()
    else:
        window.showMaximized()
    sync_maximize_button(window)


def _interactive_widget(widget):
    while widget is not None:
        if isinstance(
            widget,
            (
                QPushButton,
                QLineEdit,
                QTextEdit,
                QPlainTextEdit,
                QComboBox,
                QSpinBox,
                QListWidget,
                QTableWidget,
                QScrollArea,
            ),
        ):
            return True
        widget = widget.parentWidget()
    return False


def _resize_reference_rect(window):
    container = getattr(window, "container", None)
    if container is not None:
        return QRect(container.geometry())
    return window.rect()


def resize_edges_at(window, pos):
    if window.isMaximized():
        return set()
    margin = int(getattr(window, "_resize_margin", 10))
    rect = _resize_reference_rect(window)
    edges = set()
    if rect.left() - margin <= pos.x() <= rect.left() + margin:
        edges.add("left")
    if rect.right() - margin <= pos.x() <= rect.right() + margin:
        edges.add("right")
    if rect.top() - margin <= pos.y() <= rect.top() + margin:
        edges.add("top")
    if rect.bottom() - margin <= pos.y() <= rect.bottom() + margin:
        edges.add("bottom")
    return edges


def cursor_for_edges(edges):
    if {"left", "top"} <= edges or {"right", "bottom"} <= edges:
        return Qt.SizeFDiagCursor
    if {"right", "top"} <= edges or {"left", "bottom"} <= edges:
        return Qt.SizeBDiagCursor
    if "left" in edges or "right" in edges:
        return Qt.SizeHorCursor
    if "top" in edges or "bottom" in edges:
        return Qt.SizeVerCursor
    return Qt.ArrowCursor


def begin_window_resize(window, event):
    if event.button() != Qt.LeftButton:
        return False
    edges = resize_edges_at(window, event.pos())
    if not edges:
        return False
    window._resizing = True
    window._resize_edges = edges
    window._resize_start_pos = event.globalPos()
    window._resize_start_geometry = window.geometry()
    event.accept()
    return True


def continue_window_resize(window, event):
    if not getattr(window, "_resizing", False):
        return False
    if event.buttons() != Qt.LeftButton:
        return False

    delta = event.globalPos() - window._resize_start_pos
    geom = QRect(window._resize_start_geometry)
    min_w = max(1, window.minimumWidth())
    min_h = max(1, window.minimumHeight())
    edges = getattr(window, "_resize_edges", set())

    if "left" in edges:
        new_left = min(geom.left() + delta.x(), geom.right() - min_w + 1)
        geom.setLeft(new_left)
    if "right" in edges:
        geom.setRight(max(geom.right() + delta.x(), geom.left() + min_w - 1))
    if "top" in edges:
        new_top = min(geom.top() + delta.y(), geom.bottom() - min_h + 1)
        geom.setTop(new_top)
    if "bottom" in edges:
        geom.setBottom(max(geom.bottom() + delta.y(), geom.top() + min_h - 1))

    window.setGeometry(geom)
    event.accept()
    return True


def end_window_resize(window):
    window._resizing = False
    window._resize_edges = set()
    window._resize_start_pos = None
    window._resize_start_geometry = None


def update_resize_cursor(window, event):
    if getattr(window, "_resizing", False) or getattr(window, "_dragging", False):
        return
    window.setCursor(cursor_for_edges(resize_edges_at(window, event.pos())))


def native_resize_event(window, event_type, message):
    if "windows_generic_msg" not in str(event_type).lower():
        return None
    msg = ctypes.wintypes.MSG.from_address(int(message))
    if msg.message != WM_NCHITTEST:
        return None

    x = ctypes.c_short(msg.lParam & 0xFFFF).value
    y = ctypes.c_short((msg.lParam >> 16) & 0xFFFF).value
    local_pos = window.mapFromGlobal(QPoint(x, y))
    edges = resize_edges_at(window, local_pos)
    if not edges:
        return None
    if {"left", "top"} <= edges:
        return True, HTTOPLEFT
    if {"right", "top"} <= edges:
        return True, HTTOPRIGHT
    if {"left", "bottom"} <= edges:
        return True, HTBOTTOMLEFT
    if {"right", "bottom"} <= edges:
        return True, HTBOTTOMRIGHT
    if "left" in edges:
        return True, HTLEFT
    if "right" in edges:
        return True, HTRIGHT
    if "top" in edges:
        return True, HTTOP
    if "bottom" in edges:
        return True, HTBOTTOM
    return None


def begin_title_drag(window, event, title_widget):
    if title_widget is None or window.isMaximized() or event.button() != Qt.LeftButton:
        return False
    top_left = title_widget.mapTo(window, QPoint(0, 0))
    if not title_widget.rect().translated(top_left).contains(event.pos()):
        return False
    if _interactive_widget(window.childAt(event.pos())):
        return False
    window._dragging = True
    window._drag_offset = event.globalPos() - window.frameGeometry().topLeft()
    event.accept()
    return True


def continue_title_drag(window, event):
    if getattr(window, "_dragging", False) and getattr(window, "_drag_offset", None) is not None:
        if event.buttons() == Qt.LeftButton:
            window.move(event.globalPos() - window._drag_offset)
            event.accept()
            return True
    return False


def end_title_drag(window):
    window._dragging = False
    window._drag_offset = None


def title_double_click_maximize(window, event, title_widget):
    if title_widget is None or event.button() != Qt.LeftButton:
        return False
    top_left = title_widget.mapTo(window, QPoint(0, 0))
    if not title_widget.rect().translated(top_left).contains(event.pos()):
        return False
    if _interactive_widget(window.childAt(event.pos())):
        return False
    toggle_maximize_restore(window)
    event.accept()
    return True


def leave_resize_area(window, event):
    if not getattr(window, "_resizing", False) and not getattr(window, "_dragging", False):
        window.setCursor(Qt.ArrowCursor)
    event.ignore()
