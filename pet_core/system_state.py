"""Lightweight local state sampling for preference/recommendation logs.

This module intentionally keeps the signal low-intrusion:
- foreground process name and coarse category
- idle time bucket
- active app duration bucket

It does not collect window titles, screenshots, keystrokes, file contents, or
browser page contents.
"""

import ctypes
import os
import sys
import threading
import time


_state_lock = threading.Lock()
_last_foreground_key = None
_last_foreground_since = time.time()


_CATEGORY_RULES = {
    "ide": {
        "code.exe", "pycharm64.exe", "pycharm.exe", "idea64.exe", "devenv.exe",
        "cursor.exe", "sublime_text.exe", "notepad++.exe", "webstorm64.exe",
    },
    "terminal": {
        "windowsterminal.exe", "cmd.exe", "powershell.exe", "pwsh.exe",
        "conhost.exe", "wezterm-gui.exe",
    },
    "browser": {
        "chrome.exe", "msedge.exe", "firefox.exe", "brave.exe", "opera.exe",
        "vivaldi.exe", "browser.exe",
    },
    "game": {
        "steam.exe", "steamwebhelper.exe", "genshinimpact.exe", "starrail.exe",
        "zenlesszonezero.exe", "eldenring.exe",
    },
    "drawing": {
        "photoshop.exe", "clipstudiopaint.exe", "sai.exe", "painttool sai.exe",
        "krita.exe", "mspaint.exe", "illustrator.exe",
    },
    "document": {
        "winword.exe", "excel.exe", "powerpnt.exe", "wps.exe", "wpp.exe",
        "et.exe", "notepad.exe", "typora.exe", "obsidian.exe",
    },
    "chat": {
        "wechat.exe", "qq.exe", "telegram.exe", "discord.exe",
    },
    "media": {
        "potplayermini64.exe", "potplayermini.exe", "vlc.exe",
        "cloudmusic.exe", "music.ui.exe", "spotify.exe",
    },
}


def _bucket_seconds(seconds):
    try:
        seconds = max(0, int(seconds))
    except Exception:
        return "unknown"
    if seconds < 30:
        return "0-30s"
    if seconds < 120:
        return "30-120s"
    if seconds < 300:
        return "2-5m"
    if seconds < 900:
        return "5-15m"
    if seconds < 1800:
        return "15-30m"
    if seconds < 3600:
        return "30-60m"
    if seconds < 7200:
        return "1-2h"
    return "2h+"


def _category_for_process(process_name):
    name = os.path.basename(str(process_name or "")).lower()
    if not name:
        return "unknown"
    for category, names in _CATEGORY_RULES.items():
        if name in names:
            return category
    return "other"


def _get_idle_seconds_windows():
    class LASTINPUTINFO(ctypes.Structure):
        _fields_ = [
            ("cbSize", ctypes.c_uint),
            ("dwTime", ctypes.c_uint),
        ]

    info = LASTINPUTINFO()
    info.cbSize = ctypes.sizeof(LASTINPUTINFO)
    if not ctypes.windll.user32.GetLastInputInfo(ctypes.byref(info)):
        return None
    tick = ctypes.windll.kernel32.GetTickCount()
    return max(0, (tick - info.dwTime) / 1000.0)


def _get_foreground_process_windows():
    user32 = ctypes.windll.user32
    kernel32 = ctypes.windll.kernel32
    hwnd = user32.GetForegroundWindow()
    if not hwnd:
        return {"available": False, "process_name": "", "pid": None, "error": "no_foreground_window"}

    pid = ctypes.c_ulong()
    user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
    if not pid.value:
        return {"available": False, "process_name": "", "pid": None, "error": "no_pid"}

    PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
    handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid.value)
    if not handle:
        return {"available": False, "process_name": "", "pid": int(pid.value), "error": "open_process_failed"}

    try:
        size = ctypes.c_ulong(4096)
        buf = ctypes.create_unicode_buffer(size.value)
        ok = kernel32.QueryFullProcessImageNameW(handle, 0, buf, ctypes.byref(size))
        if not ok:
            return {"available": False, "process_name": "", "pid": int(pid.value), "error": "query_name_failed"}
        return {
            "available": True,
            "process_name": os.path.basename(buf.value),
            "pid": int(pid.value),
            "error": "",
        }
    finally:
        kernel32.CloseHandle(handle)


def collect_system_state(extra=None):
    """Return a privacy-preserving snapshot of local runtime state."""
    now = time.time()
    if sys.platform.startswith("win"):
        try:
            fg = _get_foreground_process_windows()
        except Exception as e:
            fg = {"available": False, "process_name": "", "pid": None, "error": str(e)}
        try:
            idle_seconds = _get_idle_seconds_windows()
        except Exception:
            idle_seconds = None
    else:
        fg = {
            "available": False,
            "process_name": "",
            "pid": None,
            "error": f"unsupported_platform:{sys.platform}",
        }
        idle_seconds = None

    process_name = fg.get("process_name", "")
    foreground_key = process_name.lower() or fg.get("error") or "unknown"
    global _last_foreground_key, _last_foreground_since
    with _state_lock:
        if foreground_key != _last_foreground_key:
            _last_foreground_key = foreground_key
            _last_foreground_since = now
        active_seconds = now - _last_foreground_since

    snapshot = {
        "platform": sys.platform,
        "privacy_level": "coarse_process_category",
        "foreground": {
            "available": bool(fg.get("available")),
            "process_name": process_name,
            "category": _category_for_process(process_name),
            "active_duration_bucket": _bucket_seconds(active_seconds),
            "window_title_collected": False,
            "error": fg.get("error", ""),
        },
        "idle": {
            "available": idle_seconds is not None,
            "seconds_bucket": _bucket_seconds(idle_seconds) if idle_seconds is not None else "unknown",
            "is_idle": bool(idle_seconds is not None and idle_seconds >= 300),
        },
    }
    if extra:
        snapshot["app_state"] = dict(extra)
    return snapshot
