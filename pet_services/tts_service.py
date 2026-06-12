"""GPT-SoVITS text-to-speech service.

The UI only needs TTSSynthThread, play_tts_file, cleanup_tts_artifacts,
and configure_tts. HTTP details, cache paths, and playback fallbacks stay here.
"""

import atexit
import os
import re
import shutil
import threading

import requests
from PyQt5.QtCore import QThread, pyqtSignal


_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TTS_GPT_SOVITS_DIR = os.path.join(_BASE_DIR, "voice", "GPT-SoVITS-v2pro-20250604")
TTS_GRADIO_TEMP_DIR = os.path.join(TTS_GPT_SOVITS_DIR, "TEMP", "gradio")
TTS_CACHE_DIR = os.path.join(_BASE_DIR, "tts_cache")

TTS_API_BASE = os.environ.get("PET_TTS_API") or "http://127.0.0.1:9880"
TTS_REF_AUDIO = os.path.join(TTS_GPT_SOVITS_DIR, "output", "slicer_opt", "A40_1_5_0008.mp3")
TTS_REF_TEXT = "あなたが人を批評するのは珍しいわね。そういうダメな人。気にするたちだったの。。"
TTS_REF_LANG = "ja"
TTS_TEXT_LANG = "zh"
TTS_GPT_WEIGHTS = "GPT_weights_v2Pro/有珠语音-e15.ckpt"
TTS_SOVITS_WEIGHTS = "SoVITS_weights_v2Pro/有珠语音_e8_s392.pth"

TTS_REQUEST_TIMEOUT = 150
TTS_MAX_TEXT_LEN = 300

_tts_player_instance = None


def _ensure_cache_dir():
    try:
        os.makedirs(TTS_CACHE_DIR, exist_ok=True)
    except Exception as e:
        print(f"[TTS] 建立 tts_cache 目录失败: {e}")


_ensure_cache_dir()


def configure_tts(config=None, base_dir=None):
    """Apply AppConfig values to the TTS service without recreating UI code."""
    global _BASE_DIR
    global TTS_GPT_SOVITS_DIR, TTS_GRADIO_TEMP_DIR, TTS_CACHE_DIR
    global TTS_API_BASE, TTS_REF_AUDIO, TTS_REF_TEXT, TTS_REF_LANG, TTS_TEXT_LANG
    global TTS_GPT_WEIGHTS, TTS_SOVITS_WEIGHTS

    config = dict(config or {})
    if base_dir:
        _BASE_DIR = base_dir

    TTS_GPT_SOVITS_DIR = os.path.join(_BASE_DIR, "voice", "GPT-SoVITS-v2pro-20250604")
    TTS_GRADIO_TEMP_DIR = os.path.join(TTS_GPT_SOVITS_DIR, "TEMP", "gradio")
    TTS_CACHE_DIR = os.path.join(_BASE_DIR, "tts_cache")
    _ensure_cache_dir()

    TTS_API_BASE = (
        os.environ.get("PET_TTS_API")
        or config.get("api_base")
        or "http://127.0.0.1:9880"
    )
    TTS_REF_AUDIO = config.get("ref_audio") or os.path.join(
        TTS_GPT_SOVITS_DIR,
        "output", "slicer_opt", "A40_1_5_0008.mp3",
    )
    TTS_REF_TEXT = config.get("ref_text") or TTS_REF_TEXT
    TTS_REF_LANG = config.get("ref_lang") or "ja"
    TTS_TEXT_LANG = config.get("text_lang") or "zh"
    TTS_GPT_WEIGHTS = config.get("gpt_weights") or "GPT_weights_v2Pro/有珠语音-e15.ckpt"
    TTS_SOVITS_WEIGHTS = config.get("sovits_weights") or "SoVITS_weights_v2Pro/有珠语音_e8_s392.pth"
    reset_tts_client_state()


def _tts_clean_text(text):
    if not text:
        return ""
    s = re.sub(r"!\[.*?\]\(.*?\)", "", text)
    s = s.strip()
    s = re.sub(r"\[[^\[\]]{0,15}\]$", "", s).strip()
    s = re.sub(r"\([^\(\)]{0,15}\)$", "", s).strip()
    s = re.sub(r"https?://\S+", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s[:TTS_MAX_TEXT_LEN]


def _tts_cache_path_for_id(msg_id):
    return os.path.join(TTS_CACHE_DIR, f"tts_{msg_id}.wav")


class TTSClient:
    """Minimal GPT-SoVITS api_v2.py client; thread-safe around weight switching."""

    def __init__(self):
        self._session = requests.Session()
        self._weights_set = False
        self._lock = threading.Lock()

    def reset(self):
        with self._lock:
            self._weights_set = False

    def _ensure_weights(self):
        with self._lock:
            if self._weights_set:
                return
            try:
                self._session.get(
                    f"{TTS_API_BASE}/set_gpt_weights",
                    params={"weights_path": TTS_GPT_WEIGHTS},
                    timeout=30,
                )
                self._session.get(
                    f"{TTS_API_BASE}/set_sovits_weights",
                    params={"weights_path": TTS_SOVITS_WEIGHTS},
                    timeout=30,
                )
                print(f"[TTS] 已切换到有珠微调权重 ({TTS_GPT_WEIGHTS})")
            except Exception as e:
                print(f"[TTS] 切换权重失败，沿用 api_v2 默认权重: {e}")
            self._weights_set = True

    def synthesize_to_file(self, text, out_path):
        clean = _tts_clean_text(text)
        if not clean:
            return False
        if os.path.exists(out_path) and os.path.getsize(out_path) > 200:
            return True

        self._ensure_weights()
        payload = {
            "text": clean,
            "text_lang": TTS_TEXT_LANG,
            "ref_audio_path": TTS_REF_AUDIO,
            "prompt_text": TTS_REF_TEXT,
            "prompt_lang": TTS_REF_LANG,
            "text_split_method": "cut5",
            "media_type": "wav",
            "streaming_mode": False,
            "batch_size": 1,
            "speed_factor": 1.0,
        }
        try:
            resp = self._session.post(
                f"{TTS_API_BASE}/tts",
                json=payload,
                timeout=TTS_REQUEST_TIMEOUT,
            )
        except Exception as e:
            print(f"[TTS] /tts 请求失败: {e}")
            return False
        if resp.status_code != 200:
            print(f"[TTS] /tts 返回 {resp.status_code}: {resp.text[:200]}")
            return False

        tmp_path = out_path + ".part"
        try:
            with open(tmp_path, "wb") as f:
                f.write(resp.content)
            os.replace(tmp_path, out_path)
            return True
        except Exception as e:
            print(f"[TTS] 写缓存 {out_path} 失败: {e}")
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except OSError:
                pass
            return False


tts_client = TTSClient()


def reset_tts_client_state():
    tts_client.reset()


class TTSSynthThread(QThread):
    """Background synthesis for one assistant message."""

    finished_signal = pyqtSignal(str, str)

    def __init__(self, msg_id, text, parent=None):
        super().__init__(parent)
        self.msg_id = msg_id
        self.text = text
        self.out_path = _tts_cache_path_for_id(msg_id)

    def run(self):
        ok = False
        try:
            ok = tts_client.synthesize_to_file(self.text, self.out_path)
        except Exception as e:
            print(f"[TTS] 合成线程异常: {e}")
        self.finished_signal.emit(self.msg_id, self.out_path if ok else "")


def _purge_dir_contents(dir_path):
    if not dir_path or not os.path.isdir(dir_path):
        return 0
    n = 0
    for name in os.listdir(dir_path):
        p = os.path.join(dir_path, name)
        try:
            if os.path.isdir(p) and not os.path.islink(p):
                shutil.rmtree(p, ignore_errors=True)
            else:
                os.remove(p)
            n += 1
        except Exception:
            pass
    return n


def cleanup_tts_artifacts(purge_local_cache=False):
    try:
        removed = _purge_dir_contents(TTS_GRADIO_TEMP_DIR)
        if removed:
            print(f"[TTS] 已清理 gradio 临时音频 {removed} 项")
    except Exception as e:
        print(f"[TTS] 清理 gradio temp 失败: {e}")
    if purge_local_cache:
        try:
            removed = _purge_dir_contents(TTS_CACHE_DIR)
            if removed:
                print(f"[TTS] 已清理本地 tts_cache {removed} 项")
        except Exception as e:
            print(f"[TTS] 清理 tts_cache 失败: {e}")


atexit.register(lambda: cleanup_tts_artifacts(purge_local_cache=True))


def _get_tts_player():
    global _tts_player_instance
    if _tts_player_instance is not None:
        return _tts_player_instance
    try:
        from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
        from PyQt5.QtCore import QUrl

        player = QMediaPlayer()
        player._make_content = lambda path: QMediaContent(QUrl.fromLocalFile(path))
        _tts_player_instance = player
        return player
    except Exception as e:
        print(f"[TTS] QMediaPlayer 不可用，回退到 winsound: {e}")
        return None


def play_tts_file(path):
    if not path or not os.path.exists(path):
        return False
    player = _get_tts_player()
    if player is not None:
        try:
            player.stop()
            player.setMedia(player._make_content(path))
            player.play()
            return True
        except Exception as e:
            print(f"[TTS] QMediaPlayer 播放失败，尝试 winsound: {e}")
    try:
        import winsound

        winsound.PlaySound(path, winsound.SND_FILENAME | winsound.SND_ASYNC)
        return True
    except Exception as e:
        print(f"[TTS] winsound 播放也失败: {e}")
    return False


def stop_tts_playback():
    player = _tts_player_instance
    if player is not None:
        try:
            player.stop()
        except Exception:
            pass
    try:
        import winsound

        winsound.PlaySound(None, winsound.SND_PURGE)
    except Exception:
        pass
