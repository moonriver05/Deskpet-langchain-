import os
import sys
import json
import time
import requests
import datetime
import random
import base64
import re
import pymysql
import math
import asyncio
import atexit
import concurrent.futures
import threading
import uuid
import PyQt5
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from zhdate import ZhDate
try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None

try:
    import docx
except ImportError:
    docx = None

try:
    import jieba
except ImportError:
    jieba = None

try:
    from qcloud_cos import CosConfig
    from qcloud_cos import CosS3Client
except ImportError:
    pass


# ==================== 配置中心 (AppConfig) ====================
# 把所有 API key / 密码 / URL 模板这种"上传 GitHub 时必须留空"的东西，
# 全部集中到一个本地 JSON 文件里：pet_config.json（与 pet.py 同目录）。
#   - 用户首次启动：检测到 mysql/ark 等必填项为空 → 弹设置窗口让其填写；
#   - 之后右键桌宠 → ⚙ 设置 → 一键改所有 key；
#   - 这份 JSON 不要 commit 到 Github（README 建议加 .gitignore）。
#
# 整个模块对外只有：
#   app_config.get("ark.api_key")              # dot-path 取
#   app_config.set("ark.api_key", "new")       # dot-path 写（不会自动落盘）
#   app_config.save()                          # 写回磁盘
#   app_config.missing_required()              # 返回缺失的必填项 list
#   apply_config_to_globals()                  # 把 AppConfig 的值同步到老的全局变量
CONFIG_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "pet_config.json",
)

# CONFIG_SCHEMA 既是默认值来源，也是 SettingsWindow 用来动态生成表单的"元数据"。
# 每个字段都长这样：
#   {"key": "...", "label": "...", "default": "...",
#    "required": bool, "secret": bool, "hint": "...", "type": "int|str|path"}
CONFIG_SCHEMA = [
    {
        "key": "mysql",
        "title": "MySQL 数据库",
        "icon": "🗄️",
        "desc": "用来存长期记忆 / 知识库元数据。需要本地 MySQL 已启动。",
        "fields": [
            {"key": "host", "label": "主机",  "default": "localhost", "required": True},
            {"key": "port", "label": "端口",  "default": 3306, "type": "int"},
            {"key": "user", "label": "用户名", "default": "root", "required": True},
            {"key": "password", "label": "密码", "default": "", "required": True, "secret": True},
            {"key": "database", "label": "数据库名", "default": "pet_memory_db"},
        ],
    },
    {
        "key": "ark",
        "title": "火山方舟 LLM",
        "icon": "🤖",
        "desc": "主回复 / 记忆抽取 / 工具路由都靠它。去 https://www.volcengine.com/product/ark 申请 API key。",
        "fields": [
            {"key": "api_key", "label": "API Key", "default": "", "required": True, "secret": True,
             "hint": "形如 ark-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx-xxxxxx"},
            {"key": "base_url", "label": "Base URL", "default": "https://ark.cn-beijing.volces.com/api/v3"},
            {"key": "model_main", "label": "主回复模型", "default": "doubao-1-5-pro-32k-250115",
             "hint": "推理质量更好的，例如 doubao-1-5-pro-32k 或 doubao-seed-2.x"},
            {"key": "model_extractor", "label": "提取器模型", "default": "doubao-seed-2-0-mini-260428",
             "hint": "抽关键词 / 抽新事实，用 mini 版省钱"},
            {"key": "model_tool", "label": "工具路由模型", "default": "doubao-seed-2-0-mini-260428",
             "hint": "判断是否要写待办，用 mini 版即可"},
        ],
    },
    {
        "key": "cos",
        "title": "腾讯云 COS（可选）",
        "icon": "☁️",
        "desc": "用来托管表情包图床。留空即回复时不带表情包。",
        "fields": [
            {"key": "secret_id",  "label": "SecretId",  "default": "", "secret": True},
            {"key": "secret_key", "label": "SecretKey", "default": "", "secret": True},
            {"key": "region",     "label": "地域",       "default": "ap-guangzhou"},
            {"key": "bucket",     "label": "Bucket 名",  "default": "",
             "hint": "如 emotion-1234567890"},
            {"key": "base_url",   "label": "图床公网域名", "default": "",
             "hint": "如 https://emotion-1234567890.cos.ap-guangzhou.myqcloud.com"},
        ],
    },
    {
        "key": "chroma",
        "title": "Chroma 向量库（可选）",
        "icon": "📦",
        "desc": "通过 docker 复用已运行的 chroma-mcp 容器，默认值一般够用。",
        "fields": [
            {"key": "container_name", "label": "容器名", "default": "chroma-mcp"},
        ],
    },
    {
        "key": "tts",
        "title": "GPT-SoVITS 语音合成（可选）",
        "icon": "🔊",
        "desc": "桌宠回复语音化。需先启动 GPT-SoVITS 自带的 api_v2.py。",
        "fields": [
            {"key": "api_base", "label": "服务地址", "default": "http://127.0.0.1:9880"},
            {"key": "ref_audio", "label": "参考音频路径", "default": "", "type": "path",
             "hint": "4-10s 的 wav/mp3 绝对路径"},
            {"key": "ref_text", "label": "参考音频文本", "default": ""},
            {"key": "ref_lang", "label": "参考音频语种", "default": "ja",
             "hint": "zh / ja / en"},
            {"key": "text_lang", "label": "合成输出语种", "default": "zh"},
            {"key": "gpt_weights",     "label": "GPT 权重相对路径", "default": "",
             "hint": "相对于 GPT-SoVITS 工程根目录，如 GPT_weights_v2Pro/xxx.ckpt"},
            {"key": "sovits_weights",  "label": "SoVITS 权重相对路径", "default": "",
             "hint": "相对于 GPT-SoVITS 工程根目录，如 SoVITS_weights_v2Pro/xxx.pth"},
        ],
    },
    {
        "key": "ucollege",
        "title": "优学院作业拉取（可选）",
        "icon": "📚",
        "desc": "TodoWindow 里点「获取作业」时用到。账号密码仅本地存。",
        "fields": [
            {"key": "login_name", "label": "账号", "default": "", "secret": True},
            {"key": "password",   "label": "密码", "default": "", "secret": True},
            {"key": "login_url",        "label": "登录 API",     "default": ""},
            {"key": "course_list_url",  "label": "课程列表 API", "default": ""},
            {"key": "homework_url",     "label": "作业 API 模板", "default": "",
             "hint": "用 {course_id} 占位，如 https://.../{course_id}/homework"},
        ],
    },
]

REQUIRED_KEYS = ["mysql.password", "ark.api_key"]


class AppConfig:
    """单例。所有 secrets / URL 都集中在这里。"""

    def __init__(self):
        self._lock = threading.Lock()
        self._data = self._build_defaults()
        self._load_from_disk()

    @staticmethod
    def _build_defaults():
        d = {}
        for section in CONFIG_SCHEMA:
            d[section["key"]] = {f["key"]: f.get("default", "") for f in section["fields"]}
        return d

    def _load_from_disk(self):
        if not os.path.exists(CONFIG_FILE):
            return
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                return
            # 浅合并：磁盘有的覆盖默认值；磁盘没的保留默认值（兼容老配置）
            for section_key, section_val in data.items():
                if isinstance(section_val, dict) and section_key in self._data:
                    for k, v in section_val.items():
                        self._data[section_key][k] = v
        except Exception as e:
            print(f"[AppConfig] 加载 {CONFIG_FILE} 失败：{e}")

    def get(self, dot_path, default=None):
        cur = self._data
        for part in dot_path.split("."):
            if not isinstance(cur, dict) or part not in cur:
                return default
            cur = cur[part]
        if cur is None or cur == "":
            return default if default is not None else cur
        return cur

    def set(self, dot_path, value):
        parts = dot_path.split(".")
        with self._lock:
            cur = self._data
            for p in parts[:-1]:
                if p not in cur or not isinstance(cur[p], dict):
                    cur[p] = {}
                cur = cur[p]
            cur[parts[-1]] = value

    def get_section(self, section_key):
        return dict(self._data.get(section_key) or {})

    def update_section(self, section_key, values):
        with self._lock:
            if section_key not in self._data:
                self._data[section_key] = {}
            self._data[section_key].update(values)

    def save(self):
        with self._lock:
            try:
                # 写到磁盘前再做一次"用户头一回填完"的兜底初始化
                with open(CONFIG_FILE, "w", encoding="utf-8") as f:
                    json.dump(self._data, f, ensure_ascii=False, indent=2)
                print(f"[AppConfig] 已保存到 {CONFIG_FILE}")
                return True
            except Exception as e:
                print(f"[AppConfig] 保存失败：{e}")
                return False

    def missing_required(self):
        miss = []
        for key in REQUIRED_KEYS:
            if not self.get(key):
                miss.append(key)
        return miss


app_config = AppConfig()


def apply_config_to_globals():
    """把 app_config 的值同步到老的模块级全局变量（DB_CONFIG / COS_CONFIG / ark_api_key 等）。
    设置窗口保存后会再调一次，使新值即时生效（不用重启）。
    """
    global ark_api_key

    mysql_cfg = app_config.get_section("mysql")
    DB_CONFIG["host"] = mysql_cfg.get("host") or "localhost"
    DB_CONFIG["user"] = mysql_cfg.get("user") or "root"
    DB_CONFIG["password"] = mysql_cfg.get("password") or ""
    DB_CONFIG["charset"] = "utf8mb4"
    try:
        DB_CONFIG["port"] = int(mysql_cfg.get("port") or 3306)
    except (TypeError, ValueError):
        DB_CONFIG["port"] = 3306
    # DB_NAME 是 module-level，单独同步
    globals()["DB_NAME"] = mysql_cfg.get("database") or "pet_memory_db"

    cos_cfg = app_config.get_section("cos")
    COS_CONFIG["secret_id"]  = cos_cfg.get("secret_id")  or ""
    COS_CONFIG["secret_key"] = cos_cfg.get("secret_key") or ""
    COS_CONFIG["region"]     = cos_cfg.get("region")     or "ap-guangzhou"
    COS_CONFIG["bucket"]     = cos_cfg.get("bucket")     or ""
    COS_CONFIG["base_url"]   = cos_cfg.get("base_url")   or ""

    chroma_cfg = app_config.get_section("chroma")
    if chroma_cfg.get("container_name"):
        CHROMA_MCP_CONFIG["container_name"] = chroma_cfg["container_name"]

    ark_api_key = app_config.get("ark.api_key", "") or ""

    # TTS_* 是模块级常量，本地变量；通过 globals() 写
    tts_cfg = app_config.get_section("tts")
    if tts_cfg.get("api_base"):
        globals()["TTS_API_BASE"] = tts_cfg["api_base"]
    if tts_cfg.get("ref_audio"):
        globals()["TTS_REF_AUDIO"] = tts_cfg["ref_audio"]
    if tts_cfg.get("ref_text"):
        globals()["TTS_REF_TEXT"] = tts_cfg["ref_text"]
    if tts_cfg.get("ref_lang"):
        globals()["TTS_REF_LANG"] = tts_cfg["ref_lang"]
    if tts_cfg.get("text_lang"):
        globals()["TTS_TEXT_LANG"] = tts_cfg["text_lang"]
    if tts_cfg.get("gpt_weights"):
        globals()["TTS_GPT_WEIGHTS"] = tts_cfg["gpt_weights"]
    if tts_cfg.get("sovits_weights"):
        globals()["TTS_SOVITS_WEIGHTS"] = tts_cfg["sovits_weights"]


# 提前声明 ark_api_key，让 apply_config_to_globals 第一次调用时能 import-time 赋值
ark_api_key = ""


# ==================== 灵魂状态系统 (SoulState) ====================
class SoulState:
    def __init__(self):
        self.recall_depth = 0.5  # 控制 RAG 检索数量
        self.impression_depth = 0.5  # 控制记忆生成数量
        self.expression_desire = 0.3  # 控制 LLM 输出长度
        self.creativity = 0.3  # 控制 LLM 温度参数

    def resonate(self, matched_memories):
        # 旧记忆状态冲击当前状态
        impact = len(matched_memories) * 0.1
        self.recall_depth = math.tanh(self.recall_depth + impact)
        self.impression_depth = math.tanh(self.impression_depth + impact)
        self.expression_desire = math.tanh(self.expression_desire + impact)
        self.creativity = math.tanh(self.creativity + impact)

    def get_params(self):
        # 映射到具体参数
        return {
            "top_k": max(1, int(self.recall_depth * 10)),
            "memory_limit": max(1, int(self.impression_depth * 5)),
            "max_tokens": max(500, int(self.expression_desire * 4000)),
            "temperature": min(1.0, max(0.1, self.creativity))
        }

soul_state = SoulState()

# ==================== 数据库配置 ====================
# 真实值来自 AppConfig（pet_config.json）。SettingsWindow 保存后会通过
# apply_config_to_globals() 原地刷新这个 dict，让新密码立刻生效，无需重启。
DB_CONFIG = {
    'host':     app_config.get("mysql.host", "localhost"),
    'user':     app_config.get("mysql.user", "root"),
    'password': app_config.get("mysql.password", "") or "",
    'port':     int(app_config.get("mysql.port", 3306) or 3306),
    'charset':  'utf8mb4',
}
DB_NAME = app_config.get("mysql.database", "pet_memory_db") or "pet_memory_db"

# ==================== Chroma MCP（复用已运行的 chroma-mcp 容器） ====================
CHROMA_COLLECTION_KB = "pet_knowledge_base"
CHROMA_COLLECTION_MEM = "pet_user_memory"
# 关键：默认通过 `docker exec` 进入已经在运行的 chroma-mcp 容器，
# 而不是 `docker run` 起一个新容器。这样所有读写都落到同一个容器内的同一份持久化数据上。
CHROMA_MCP_CONFIG = {
    # 已经在 Docker Desktop 里运行、正在使用的容器名（见 `docker ps`）。
    # 优先级：环境变量 > AppConfig > 默认 "chroma-mcp"
    "container_name": (
        os.environ.get("CHROMA_MCP_CONTAINER")
        or app_config.get("chroma.container_name")
        or "chroma-mcp"
    ),
    # 容器内启动 MCP server 的命令；mcp/chroma 镜像的默认 CMD 是 `chroma-mcp`。
    # 关键：必须显式带 `--client-type persistent --data-dir /chroma_data`，
    # 否则 chroma-mcp 默认是 ephemeral（纯内存），容器一停就全没了。
    # /chroma_data 必须是容器里被持久化卷挂载到的目录，所以容器要这样起：
    #   docker run -d --name chroma-mcp --restart unless-stopped \
    #     -v pet_desktop_chroma:/chroma_data --entrypoint sleep mcp/chroma infinity
    "container_command": [
        "chroma-mcp", "--client-type", "persistent", "--data-dir", "/chroma_data",
    ],
    # 若容器未运行，是否回退到 `docker run -i --rm mcp/chroma`（仍只起一次，同一个 python 进程内复用）。
    "fallback_to_run": True,
    "fallback_image": "mcp/chroma",
    "fallback_volume_name": "pet_desktop_chroma",
    "fallback_data_dir_in_container": "/chroma_data",
}


def _container_is_running(name):
    """查询命名容器是否在 running 状态。Docker 不可用时返回 False。"""
    if not name:
        return False
    import subprocess
    try:
        out = subprocess.run(
            ["docker", "inspect", "-f", "{{.State.Running}}", name],
            capture_output=True, text=True, timeout=10,
        )
    except FileNotFoundError:
        print("[Chroma MCP] 未找到 docker 可执行文件。")
        return False
    except Exception as e:
        print(f"[Chroma MCP] docker inspect 调用失败: {e}")
        return False
    if out.returncode != 0:
        return False
    return out.stdout.strip().lower() == "true"


def _chroma_docker_stdio_params():
    """优先 `docker exec -i <container_name> chroma-mcp`，让会话直接跑在已存在的 chroma-mcp 容器里。"""
    name = CHROMA_MCP_CONFIG.get("container_name") or "chroma-mcp"
    cmd = list(CHROMA_MCP_CONFIG.get("container_command") or ["chroma-mcp"])
    if _container_is_running(name):
        print(f"[Chroma MCP] 复用已运行容器 `{name}`（docker exec -i {name} {' '.join(cmd)}）。")
        return StdioServerParameters(command="docker", args=["exec", "-i", name] + cmd)
    if not CHROMA_MCP_CONFIG.get("fallback_to_run"):
        raise RuntimeError(
            f"容器 `{name}` 未运行，且已禁用 docker run 回退。请先 `docker start {name}` 再启动桌宠。"
        )
    image = CHROMA_MCP_CONFIG.get("fallback_image", "mcp/chroma")
    vol = CHROMA_MCP_CONFIG.get("fallback_volume_name", "pet_desktop_chroma")
    data_dir = CHROMA_MCP_CONFIG.get("fallback_data_dir_in_container", "/chroma_data")
    # 关键：`chroma-mcp` 不读 CHROMA_* 环境变量，必须用命令行参数指定持久化模式，
    # 否则会以 ephemeral（内存）模式启动，挂载的 volume 形同虚设。
    args = [
        "run", "-i", "--rm",
        "-v", f"{vol}:{data_dir}",
        "--entrypoint", "chroma-mcp",
        image,
        "--client-type", "persistent",
        "--data-dir", data_dir,
    ]
    print(f"[Chroma MCP] 容器 `{name}` 未运行，回退到一次性容器 `docker run --rm {image}`（持久化卷 {vol}）。")
    return StdioServerParameters(command="docker", args=args)


def _chrom_tool_first_text(result):
    if not result or not getattr(result, "content", None):
        return None
    block = result.content[0]
    return getattr(block, "text", None) if block is not None else None


def _chrom_tool_to_dict(result):
    """兼容 chroma_query_documents 返回 structuredContent 或 JSON 文本。"""
    if result is None:
        return None
    sc = getattr(result, "structuredContent", None)
    if isinstance(sc, dict) and sc:
        return sc
    text = _chrom_tool_first_text(result)
    if text:
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
    return None


# ---- Chroma MCP 单例：一个后台线程内常驻单个 Docker 容器 + MCP 会话，所有读写共用同一向量库 ----
_chrom_worker_thread = None
_chrom_worker_loop = None
_chrom_request_q = None
_chrom_ready = threading.Event()
_CHROM_SHUTDOWN = object()


def _chrom_worker_main():
    global _chrom_worker_loop, _chrom_request_q
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    _chrom_worker_loop = loop

    async def runner():
        global _chrom_request_q
        _chrom_request_q = asyncio.Queue()
        params = _chroma_docker_stdio_params()
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                _chrom_ready.set()
                while True:
                    tool_name, arguments, py_fut = await _chrom_request_q.get()
                    if tool_name is _CHROM_SHUTDOWN:
                        py_fut.set_result(None)
                        break
                    try:
                        r = await session.call_tool(tool_name, arguments)
                        py_fut.set_result(r)
                    except BaseException as e:
                        py_fut.set_exception(e)

    try:
        loop.run_until_complete(runner())
    finally:
        try:
            loop.run_until_complete(loop.shutdown_asyncgens())
        except Exception:
            pass
        loop.close()


def _chrom_ensure_worker():
    global _chrom_worker_thread
    need_start = _chrom_worker_thread is None or not _chrom_worker_thread.is_alive()
    if need_start:
        _chrom_ready.clear()
        _chrom_worker_thread = threading.Thread(
            target=_chrom_worker_main, name="ChromaMcpWorker", daemon=True
        )
        _chrom_worker_thread.start()
        if not _chrom_ready.wait(timeout=180):
            raise RuntimeError(
                "Chroma MCP 工作线程启动超时（请确认 Docker 已运行且镜像 mcp/chroma 可用）"
            )


def _chrom_run_tool(tool_name, arguments):
    _chrom_ensure_worker()
    py_fut = concurrent.futures.Future()

    async def enqueue():
        await _chrom_request_q.put((tool_name, arguments, py_fut))

    asyncio.run_coroutine_threadsafe(enqueue(), _chrom_worker_loop).result(timeout=60)
    return py_fut.result(timeout=600)


def _chrom_shutdown_worker():
    global _chrom_worker_thread, _chrom_worker_loop, _chrom_request_q
    try:
        if _chrom_worker_thread is None or not _chrom_worker_thread.is_alive():
            return
        if _chrom_worker_loop is None or _chrom_request_q is None:
            return
        py_fut = concurrent.futures.Future()

        async def shutdown():
            await _chrom_request_q.put((_CHROM_SHUTDOWN, None, py_fut))

        asyncio.run_coroutine_threadsafe(shutdown(), _chrom_worker_loop).result(timeout=30)
        py_fut.result(timeout=15)
    except Exception:
        pass


atexit.register(_chrom_shutdown_worker)


def _chrom_distance_to_sim(d):
    try:
        d = float(d)
    except (TypeError, ValueError):
        return 0.0
    return 1.0 / (1.0 + max(0.0, d))


def chroma_query_documents_sync(collection_name, query_texts, n_results, where=None, where_document=None):
    """同步封装：经单例 MCP 会话执行 chroma_query_documents（同一容器、同一持久化向量库）。"""
    args = {
        "collection_name": collection_name,
        "query_texts": query_texts,
        "n_results": int(n_results),
    }
    if where is not None:
        args["where"] = where
    if where_document is not None:
        args["where_document"] = where_document
    r = _chrom_run_tool("chroma_query_documents", args)
    return _chrom_tool_to_dict(r)


def chroma_add_documents_sync(collection_name, documents, ids, metadatas=None):
    args = {"collection_name": collection_name, "documents": documents, "ids": ids}
    if metadatas is not None:
        args["metadatas"] = metadatas
    _chrom_run_tool("chroma_add_documents", args)


def chroma_get_documents_sync(collection_name, ids=None, where=None, where_document=None,
                              include=None, limit=None):
    """同步封装：chroma_get_documents。用于按 ids 或 metadata 过滤精确取文档（无相似度）。
    主要场景：知识库 sentence-window 扩展——命中片之后把同 doc_id 的相邻 chunk 一起拉出来。
    """
    args = {"collection_name": collection_name}
    if ids is not None:
        args["ids"] = list(ids)
    if where is not None:
        args["where"] = where
    if where_document is not None:
        args["where_document"] = where_document
    if include is not None:
        args["include"] = list(include)
    if limit is not None:
        args["limit"] = int(limit)
    r = _chrom_run_tool("chroma_get_documents", args)
    return _chrom_tool_to_dict(r)


def chroma_delete_documents_sync(collection_name, ids):
    if not ids:
        return
    _chrom_run_tool("chroma_delete_documents", {"collection_name": collection_name, "ids": list(ids)})


# ==================== 知识库系统 (KnowledgeBase) ====================
# ---- 文本切分：按 Markdown 标题 / 段落 / chunk_size + overlap，保留章节路径 ----
def _split_markdown_into_chunks(text, chunk_size=800, overlap=120):
    """把长文本切成 chunk，保留 Markdown 结构与上下文连续性。

    切分策略（从粗到细）：
      1. 先按 Markdown 标题 (`#` ~ `######`) 拆 section，并维护 heading_stack
         作为该 section 的章节路径（如 ["第一章", "第一节"]）。
      2. 每个 section 内按"空行段落"累加，达到 chunk_size 出片。
      3. 出片时把上一片末尾的 `overlap` 字符接到下一片开头，让相邻 chunk 有重叠
         上下文，避免一句话被切到两片导致语义断裂。
      4. 单段就超过 1.5×chunk_size 时强制按字符切（极端情况兜底）。

    返回 [{"text": str, "heading_path": List[str]}, ...]
    """
    if not text or not text.strip():
        return []

    lines = text.replace("\r\n", "\n").split("\n")
    heading_stack = []  # [(level, title), ...]，level 越小越靠外
    sections = []  # [(heading_path_list, block_text)]
    buf = []

    def flush():
        if buf:
            block = "\n".join(buf).strip()
            if block:
                sections.append(([h for _, h in heading_stack], block))
            buf.clear()

    for line in lines:
        m = re.match(r"^(#{1,6})\s+(.+?)\s*$", line)
        if m:
            flush()
            level = len(m.group(1))
            title = m.group(2).strip()
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            heading_stack.append((level, title))
        else:
            buf.append(line)
    flush()

    if not sections:
        sections = [([], text)]

    chunks = []
    overlap = max(0, int(overlap))
    chunk_size = max(200, int(chunk_size))
    hard_limit = int(chunk_size * 1.5)

    for heading_path, block in sections:
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", block) if p.strip()]
        if not paragraphs:
            continue
        cur = ""
        for para in paragraphs:
            # 单段超长 → 强制硬切（先把累积的吐出去，再分段切这一长段）
            while len(para) > hard_limit:
                if cur:
                    chunks.append({"text": cur, "heading_path": list(heading_path)})
                    tail = cur[-overlap:] if overlap else ""
                    cur = ""
                    head_piece = para[: chunk_size - len(tail)]
                    chunks.append({"text": tail + head_piece, "heading_path": list(heading_path)})
                    para = para[chunk_size - len(tail):]
                else:
                    chunks.append({"text": para[:chunk_size], "heading_path": list(heading_path)})
                    para = para[chunk_size - overlap:] if overlap else para[chunk_size:]
            if not cur:
                cur = para
            elif len(cur) + 2 + len(para) <= chunk_size:
                cur = cur + "\n\n" + para
            else:
                chunks.append({"text": cur, "heading_path": list(heading_path)})
                tail = cur[-overlap:] if overlap else ""
                cur = (tail + "\n\n" + para) if tail else para
        if cur:
            chunks.append({"text": cur, "heading_path": list(heading_path)})

    return chunks


class KnowledgeBase:
    """RAG 知识库封装。
    存储约定：
      - collection: CHROMA_COLLECTION_KB
      - 每个文档 = 一个随机 doc_id，对应 N 个 chunk
      - chunk id 形如 `<doc_id>_chunk_<index:04d>`，方便 metadata 过滤定位
      - chunk metadata: {source, doc_id, chunk_index, total_chunks, heading_path?}
    """

    DEFAULT_CHUNK_SIZE = 800
    DEFAULT_OVERLAP = 120
    DEFAULT_WINDOW = 1  # sentence-window 扩展半径（前后各拉 N 片）

    def __init__(self):
        pass

    def add_document(self, text, source="unknown",
                     chunk_size=None, overlap=None):
        """按语义切分长文本并写入 Chroma。返回 doc_id（失败返回 None）。"""
        if not text or not str(text).strip():
            return None
        chunk_size = int(chunk_size) if chunk_size else self.DEFAULT_CHUNK_SIZE
        overlap = int(overlap) if overlap is not None else self.DEFAULT_OVERLAP

        pieces = _split_markdown_into_chunks(text, chunk_size=chunk_size, overlap=overlap)
        if not pieces:
            return None

        doc_id = uuid.uuid4().hex
        total = len(pieces)
        ids = [f"{doc_id}_chunk_{i:04d}" for i in range(total)]
        docs = [p["text"] for p in pieces]
        metadatas = []
        for i, p in enumerate(pieces):
            meta = {
                "source": str(source),
                "doc_id": doc_id,
                "chunk_index": i,
                "total_chunks": total,
            }
            heading = p.get("heading_path") or []
            if heading:
                # chroma metadata 标量化：list -> "A > B > C"
                meta["heading_path"] = " > ".join(str(h) for h in heading)
            metadatas.append(meta)

        try:
            chroma_add_documents_sync(CHROMA_COLLECTION_KB, docs, ids, metadatas=metadatas)
            print(f"[KB] 已写入 source={source!r} doc_id={doc_id} chunks={total}")
            return doc_id
        except Exception as e:
            print("写入知识库失败 (Chroma MCP):", e)
            return None

    def _expand_window(self, hits, window):
        """对命中的 (doc_id, chunk_index) 做 sentence-window 扩展。

        返回：dict[(doc_id, chunk_index)] = {text, source, heading}
              以及 `legacy_hits`（没有 doc_id/chunk_index 的旧数据，不参与扩展）。
        """
        from collections import defaultdict
        wanted = defaultdict(set)  # doc_id -> {chunk_index, ...}
        legacy = []
        for h in hits:
            if h["doc_id"] and h["chunk_index"] is not None:
                for w in range(-window, window + 1):
                    idx = h["chunk_index"] + w
                    if idx >= 0:
                        wanted[h["doc_id"]].add(idx)
            else:
                legacy.append(h)

        expanded = {}
        for doc_id, idx_set in wanted.items():
            try:
                got = chroma_get_documents_sync(
                    CHROMA_COLLECTION_KB,
                    where={"$and": [
                        {"doc_id": {"$eq": doc_id}},
                        {"chunk_index": {"$in": sorted(idx_set)}},
                    ]},
                )
            except Exception as e:
                print(f"[KB] 窗口扩展失败 doc_id={doc_id}: {e}")
                continue
            if not got:
                continue
            g_docs = got.get("documents") or []
            g_metas = got.get("metadatas") or []
            for d, m in zip(g_docs, g_metas):
                if not d or not isinstance(m, dict):
                    continue
                ci = m.get("chunk_index")
                try:
                    ci = int(ci)
                except (TypeError, ValueError):
                    continue
                expanded[(doc_id, ci)] = {
                    "text": d,
                    "source": m.get("source", "unknown"),
                    "heading": m.get("heading_path", ""),
                }
        return expanded, legacy

    def search(self, query, keywords=None, top_k=3, window=None):
        """检索 + sentence-window 扩展。

        返回 List[str]，每个元素形如 `[source§heading] text1\\n…\\ntext2`，
        其中同一 doc_id 内相邻 chunk 用 "\\n…\\n" 衔接，方便 LLM 看出"是同一文档的连续段落"。
        """
        if not query or not str(query).strip():
            return []
        if window is None:
            window = self.DEFAULT_WINDOW

        q = str(query).strip()
        if keywords:
            q = f"{q} {' '.join(str(k) for k in keywords if k)}"

        n_results = max(int(top_k), min(40, int(top_k) * 6))
        try:
            data = chroma_query_documents_sync(CHROMA_COLLECTION_KB, [q], n_results=n_results)
        except Exception as e:
            print("知识库检索失败 (Chroma MCP):", e)
            return []
        if not data:
            return []

        docs = (data.get("documents") or [[]])[0]
        metas = (data.get("metadatas") or [[]])[0]
        ids = (data.get("ids") or [[]])[0]

        hits = []
        for i, doc in enumerate(docs):
            if doc is None:
                continue
            meta = metas[i] if i < len(metas) and metas[i] else {}
            if not isinstance(meta, dict):
                meta = {}
            doc_id = meta.get("doc_id")
            chunk_idx = meta.get("chunk_index")
            try:
                chunk_idx = int(chunk_idx) if chunk_idx is not None else None
            except (TypeError, ValueError):
                chunk_idx = None
            hits.append({
                "id": ids[i] if i < len(ids) else None,
                "doc_id": doc_id,
                "chunk_index": chunk_idx,
                "source": meta.get("source", "unknown"),
                "heading": meta.get("heading_path", ""),
                "text": doc,
            })

        if not hits:
            return []

        # 取一份限量的命中作为 anchor（按 chroma 距离排序，前 top_k 个）
        anchors = hits[:max(1, int(top_k))]

        expanded, legacy = self._expand_window(anchors, window=window)

        out = []
        seen_doc = set()
        for h in anchors:
            doc_id = h["doc_id"]
            if doc_id is None or doc_id in seen_doc:
                continue
            seen_doc.add(doc_id)
            indices = sorted(idx for (d, idx) in expanded.keys() if d == doc_id)
            if not indices:
                continue
            label = h["source"] or "unknown"
            if h["heading"]:
                label = f"{label}§{h['heading']}"
            parts = [expanded[(doc_id, ci)]["text"] for ci in indices]
            joined = "\n…\n".join(parts) if len(parts) > 1 else parts[0]
            out.append(f"[{label}] {joined}")
            if len(out) >= top_k:
                break

        # 旧数据（pet.py 升级前写入、没有 doc_id/chunk_index）：原样返回
        if len(out) < top_k:
            for h in legacy:
                out.append(f"[{h['source']}] {h['text']}")
                if len(out) >= top_k:
                    break

        return out

knowledge_base = KnowledgeBase()

# ==================== 智能检索系统 (MemoryRuntime) ====================
# ---- 召回参数（可调） ----
# 旧实现 `final_score = sim*10 + imp*0.1` 的问题：
#   * sim 最高才贡献 10 分；importance 到 600+ 就贡献 60+ 分；
#   * 每次命中 +5、每天才 -1（且必须 24h 没访问）；
#   * 结果就是几条 importance 已经爬到 600 的"明星记忆"每轮对话都霸榜，
#     和当前问题没语义关系也会被反复拉出来 → 冷门记忆永远轮不到 → 饥饿现象。
# 新策略（chromadb 主导）：
#   阶段 1：用 Chroma 距离取 sim ≥ MEM_SIM_FLOOR 的候选；连这条都不到的，
#           不管 importance 多高都直接淘汰，杜绝"靠老本蒙过去"。
#   阶段 2：把 sim / importance / recency 三个都归一化到 [0, 1]，按权重 0.75 / 0.15 / 0.10
#           加权，sim 永远占大头，importance 只是 tiebreaker。
#   降饱和：命中只 +1（原来 +5），且对单条记忆做 LEAST(..., MEM_IMP_CAP) 硬上限 100。
#   防重复：同一轮会话里刚召回过的 mid，再次计算时打 0.6 折，给冷门记忆出场机会。
MEM_SIM_FLOOR = 0.45              # _chrom_distance_to_sim 下，对应 d ≲ 1.22；低于此直接丢
MEM_IMP_CAP = 100.0               # importance_score 硬上限，防止无限刷上去
MEM_SIM_WEIGHT = 0.75             # 语义相似度权重（主导）
MEM_IMP_WEIGHT = 0.15             # 重要性权重（次要）
MEM_RECENCY_WEIGHT = 0.10         # 时间新鲜度权重
MEM_RECENCY_HALFLIFE_DAYS = 14.0  # 上次访问后，每 14 天 recency 减半
MEM_RECENT_RECALL_PENALTY = 0.6   # 最近被拉过的同 mid 再次计分时乘 0.6
MEM_SESSION_DEDUP_WINDOW = 20     # 维护"最近召回 mid"队列的长度
MEM_RECALL_BUMP = 1               # 召回命中时 importance 增量（原来是 5）


class MemoryRuntime:
    def __init__(self):
        # 最近召回过的 mysql_id 序列（会话级 dedup，进程重启清空）
        self._recent_recalls = []
        self._lock = threading.Lock()

    def _mark_recalled(self, mid):
        with self._lock:
            if mid in self._recent_recalls:
                self._recent_recalls.remove(mid)
            self._recent_recalls.append(mid)
            if len(self._recent_recalls) > MEM_SESSION_DEDUP_WINDOW:
                self._recent_recalls = self._recent_recalls[-MEM_SESSION_DEDUP_WINDOW:]

    def _is_recently_recalled(self, mid):
        with self._lock:
            return mid in self._recent_recalls

    def chained_recall(self, query, keywords=None, top_k=5):
        matched_memories = []
        if not query or not str(query).strip():
            return matched_memories
        q = str(query).strip()
        if not keywords:
            if jieba:
                keywords = list(jieba.cut_for_search(query))
            else:
                keywords = list(query)
        if keywords:
            q = f"{q} {' '.join(str(k) for k in keywords if k)}"

        # 池子不用太大：现在 sim 是主导，候选多了反而要等更多 sql。
        n_pool = max(15, min(60, top_k * 8))
        try:
            data = chroma_query_documents_sync(CHROMA_COLLECTION_MEM, [q], n_results=n_pool)
            if not data:
                return matched_memories
            docs = (data.get("documents") or [[]])[0]
            metas = (data.get("metadatas") or [[]])[0]
            dists = (data.get("distances") or [[]])[0]

            # ---- 阶段 1：纯语义初筛 ----
            candidates = []
            for i, doc in enumerate(docs):
                meta = metas[i] if i < len(metas) and metas[i] else {}
                mid = None
                if isinstance(meta, dict):
                    mid = meta.get("mysql_id")
                try:
                    mid = int(mid)
                except (TypeError, ValueError):
                    try:
                        mid = int(float(mid))
                    except (TypeError, ValueError):
                        continue
                d = dists[i] if i < len(dists) else 0.0
                sim = _chrom_distance_to_sim(d)
                # 不够像就直接丢，不让"老明星 imp=600"靠 importance 蒙混进 Top-K
                if sim < MEM_SIM_FLOOR:
                    continue
                candidates.append({"id": mid, "content": doc, "sim": sim})

            if not candidates:
                # 整池子里都没一条 sim ≥ floor → 真的没相关记忆，不返回。
                # 注意：这种"无相关"情况下绝不能 fallback 去按 importance 取 Top，
                # 否则又会把那几条明星记忆灌进 prompt（这正是用户截图里的现象）。
                return matched_memories

            conn = pymysql.connect(
                host=DB_CONFIG['host'], user=DB_CONFIG['user'],
                password=DB_CONFIG['password'], database=DB_NAME,
                charset=DB_CONFIG['charset'], cursorclass=pymysql.cursors.DictCursor
            )
            with conn.cursor() as cursor:
                ids = list({c["id"] for c in candidates})
                placeholders = ",".join(["%s"] * len(ids))
                # 顺带把 last_accessed_at 距今多少秒拉出来，做 recency 衰减。
                cursor.execute(
                    f"SELECT id, content, importance_score, "
                    f"  GREATEST(0, TIMESTAMPDIFF(SECOND, last_accessed_at, NOW())) AS sec_ago "
                    f"FROM user_memory WHERE id IN ({placeholders})",
                    ids,
                )
                rows_by_id = {r["id"]: r for r in cursor.fetchall()}

                # ---- 阶段 2：多因子归一化重排 ----
                log_cap = math.log1p(MEM_IMP_CAP)
                sim_span = max(1e-6, 1.0 - MEM_SIM_FLOOR)
                for c in candidates:
                    row = rows_by_id.get(c["id"])
                    if not row:
                        continue
                    imp = float(row["importance_score"] or 0.0)
                    # sim 从 [floor, 1] → [0, 1]，扩大区分度
                    sim_norm = max(0.0, min(1.0, (c["sim"] - MEM_SIM_FLOOR) / sim_span))
                    # importance 用 log + cap，把"600 vs 10"这种悬殊压成"1.0 vs 0.52"
                    imp_norm = math.log1p(min(imp, MEM_IMP_CAP)) / log_cap if log_cap > 0 else 0.0
                    days_ago = float(row.get("sec_ago") or 0.0) / 86400.0
                    recency = 0.5 ** (days_ago / MEM_RECENCY_HALFLIFE_DAYS)
                    score = (
                        MEM_SIM_WEIGHT * sim_norm
                        + MEM_IMP_WEIGHT * imp_norm
                        + MEM_RECENCY_WEIGHT * recency
                    )
                    # 同一会话里刚召回过 → 降权，给冷门记忆机会
                    if self._is_recently_recalled(c["id"]):
                        score *= MEM_RECENT_RECALL_PENALTY
                    c["final_score"] = score
                    c["content"] = row["content"]
                    c["_debug"] = (c["sim"], sim_norm, imp_norm, recency, imp)

                candidates = [c for c in candidates if "final_score" in c]
                candidates.sort(key=lambda x: x["final_score"], reverse=True)
                top_results = candidates[:top_k]

                for row in top_results:
                    matched_memories.append(row["content"])
                    self._mark_recalled(row["id"])
                    # 命中只 +1，且 LEAST(..., MEM_IMP_CAP) 做硬上限。
                    cursor.execute(
                        "UPDATE user_memory "
                        "SET access_count = access_count + 1, "
                        "    importance_score = LEAST(importance_score + %s, %s), "
                        "    last_accessed_at = NOW() "
                        "WHERE id = %s",
                        (MEM_RECALL_BUMP, MEM_IMP_CAP, row["id"]),
                    )

                # 调试：把当次 Top-3 的分数构成打出来，方便观察是不是真的 sim 主导。
                if top_results:
                    print("[Mem][召回] query=", query[:30])
                    for r in top_results[:3]:
                        raw_sim, sn, ino, rec, raw_imp = r.get("_debug", (0, 0, 0, 0, 0))
                        print(
                            f"  id={r['id']:>4} final={r['final_score']:.3f}  "
                            f"sim={raw_sim:.2f}(norm {sn:.2f})  "
                            f"imp={raw_imp:.0f}(norm {ino:.2f})  "
                            f"recency={rec:.2f}  -- {str(r['content'])[:32]}"
                        )

            conn.commit()
            conn.close()
        except Exception as db_e:
            print("记忆检索异常:", db_e)

        return matched_memories

memory_runtime = MemoryRuntime()

# ==================== 对话上下文（最近 N 轮短期记忆） ====================
class ConversationHistory:
    """本地短期上下文：最多保留最近 N 轮（用户 + 久远寺有珠）对话，落盘 JSON。

    与 user_memory 的区别：
      - 这里是"原始最近对话"，不做向量化、不做艾宾浩斯衰减；
      - 主要给 LLM 在 prompt 里看到最近上下文，避免短期内反复问同一件事；
      - 超过 max_turns 时按 FIFO 删除最早的一轮。
    """
    DEFAULT_MAX_TURNS = 10

    def __init__(self, file_path=None, max_turns=None):
        if file_path is None:
            file_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "conversation_history.json",
            )
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
                u = str(item.get("user", "")).strip()
                a = str(item.get("assistant", "")).strip()
                if not u and not a:
                    continue
                normalized.append({"user": u, "assistant": a})
            return normalized
        except Exception as e:
            print(f"[ConversationHistory] 加载历史失败: {e}")
            return []

    def _save_locked(self):
        try:
            with open(self.file_path, "w", encoding="utf-8") as f:
                json.dump(self._turns, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[ConversationHistory] 保存历史失败: {e}")

    def add_turn(self, user_text, assistant_text):
        u = str(user_text or "").strip()
        a = str(assistant_text or "").strip()
        if not u and not a:
            return
        with self._lock:
            self._turns.append({"user": u, "assistant": a})
            if len(self._turns) > self.max_turns:
                # 多出来的从最早开始丢
                self._turns = self._turns[-self.max_turns:]
            self._save_locked()

    def get_turns(self):
        with self._lock:
            return list(self._turns)

    def format_for_prompt(self):
        """格式化为塞进 system prompt 的字符串。无历史返回 "无"。"""
        with self._lock:
            turns = list(self._turns)
        if not turns:
            return "无"
        lines = []
        for t in turns:
            u = t.get("user", "")
            a = t.get("assistant", "")
            if u:
                lines.append(f"用户: {u}")
            if a:
                lines.append(f"久远寺有珠: {a}")
        return "\n".join(lines)


conversation_history = ConversationHistory()

# ==================== 腾讯云 COS 图床配置 ====================
# 真实值来自 AppConfig。留空时 EmotionCOSManager 会自己降级（不发表情包），
# SettingsWindow 保存后通过 apply_config_to_globals() 原地刷新。
COS_CONFIG = {
    'secret_id':  app_config.get("cos.secret_id", "")  or "",
    'secret_key': app_config.get("cos.secret_key", "") or "",
    'region':     app_config.get("cos.region", "ap-guangzhou") or "ap-guangzhou",
    'bucket':     app_config.get("cos.bucket", "")    or "",
    'base_url':   app_config.get("cos.base_url", "")  or "",
}

class EmotionCOSManager:
    def __init__(self):
        self.client = None
        self.bucket = COS_CONFIG['bucket']
        self.base_url = COS_CONFIG['base_url']
        if COS_CONFIG['secret_id'] and COS_CONFIG['secret_key'] and self.bucket:
            try:
                config = CosConfig(Region=COS_CONFIG['region'], SecretId=COS_CONFIG['secret_id'], SecretKey=COS_CONFIG['secret_key'])
                self.client = CosS3Client(config)
            except Exception as e:
                print(f"COS 初始化失败: {e}")

    def sync_local_memes(self, local_dir):
        """同步本地 memes 文件夹到 COS 图床"""
        if not self.client:
            return False, "COS 未初始化"
            
        if not os.path.exists(local_dir):
            os.makedirs(local_dir, exist_ok=True)
            return True, f"本地文件夹不存在，已创建：{local_dir}\n请将表情包放入对应的英文情感文件夹中！"

        success_count = 0
        skip_count = 0
        error_count = 0
        
        try:
            # 获取云端已有的所有文件列表，避免重复上传
            remote_files = set()
            marker = ""
            while True:
                resp = self.client.list_objects(Bucket=self.bucket, Marker=marker, MaxKeys=1000)
                if 'Contents' in resp:
                    for item in resp['Contents']:
                        remote_files.add(item['Key'])
                if resp.get('IsTruncated') == 'true':
                    marker = resp['NextMarker']
                else:
                    break
            
            # 遍历本地文件夹
            for emotion_dir in os.listdir(local_dir):
                emotion_path = os.path.join(local_dir, emotion_dir)
                if os.path.isdir(emotion_path):
                    for filename in os.listdir(emotion_path):
                        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.webp')):
                            local_file_path = os.path.join(emotion_path, filename)
                            # 构造在 COS 中的路径（例如 "angry/1.gif"）
                            cos_key = f"{emotion_dir}/{filename}"
                            
                            if cos_key in remote_files:
                                skip_count += 1
                                continue
                                
                            try:
                                self.client.upload_file(
                                    Bucket=self.bucket,
                                    LocalFilePath=local_file_path,
                                    Key=cos_key
                                )
                                success_count += 1
                            except Exception as upload_e:
                                print(f"上传 {cos_key} 失败: {upload_e}")
                                error_count += 1
                                
            return True, f"同步完成！\n成功上传: {success_count} 张\n跳过已存在: {skip_count} 张\n失败: {error_count} 张"
        except Exception as e:
            return False, f"同步过程发生错误: {str(e)}"

    def get_random_emotion_image(self, emotion):
        if not self.client:
            return None
        try:
            # 假设图床中表情包存放在以情感命名的文件夹下，例如 "开心/"
            response = self.client.list_objects(
                Bucket=self.bucket,
                Prefix=f"{emotion}/"
            )
            if 'Contents' in response:
                files = [item['Key'] for item in response['Contents'] if item['Key'].lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.webp'))]
                if files:
                    selected = random.choice(files)
                    return f"{self.base_url.rstrip('/')}/{selected}"
        except Exception as e:
            print(f"COS 获取表情包失败: {e}")
        return None

cos_manager = EmotionCOSManager()

# 关键：手动指定 Qt 插件目录
plugin_path = os.path.join(os.path.dirname(PyQt5.__file__), "Qt5", "plugins")
os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = plugin_path

from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QVBoxLayout, QHBoxLayout,
    QPushButton, QLineEdit, QListWidget, QListWidgetItem,
    QCheckBox, QMessageBox, QMenu, QScrollArea, QDesktopWidget,
    QGraphicsDropShadowEffect, QFileDialog, QFrame, QComboBox,
    QSizePolicy, QStackedWidget, QDialog, QFormLayout, QSpinBox,
    QPlainTextEdit, QSplitter
)
from PyQt5.QtCore import (
    Qt, QPoint, QTimer, QSize, QThread, pyqtSignal, QObject,
    QPropertyAnimation, QEasingCurve, QRect
)
from PyQt5.QtGui import (
    QPixmap, QPainter, QFont, QColor, QPolygon, QMovie, QIcon,
    QLinearGradient, QBrush, QPen
)

# ark_api_key 已经在文件顶部声明，apply_config_to_globals() 会从 AppConfig 注入。
# 这里再 sync 一次，覆盖掉 import 时还没载入 AppConfig 的占位值（"" → 真实 key）。
ark_api_key = app_config.get("ark.api_key", "") or ""


# ==================== 设置窗口 (SettingsWindow) ====================
# 左侧分类列表 + 右侧表单堆叠，所有字段由 CONFIG_SCHEMA 驱动。
# 保存时把 QLineEdit / QSpinBox 里的当前值写回 app_config，再 .save() 落盘，
# 最后调一次 apply_config_to_globals() 让新值即时生效。
class SettingsWindow(QDialog):
    def __init__(self, parent=None, highlight_section=None):
        super().__init__(parent)
        self.setWindowTitle("⚙ 桌宠设置")
        self.resize(780, 560)
        self.setStyleSheet("""
            QDialog { background: #FFF8E7; }
            QListWidget {
                background: #FFE8C4; border: none; padding: 6px;
                font-size: 14px; outline: 0;
            }
            QListWidget::item { padding: 10px 12px; border-radius: 6px; }
            QListWidget::item:selected { background: #FFB347; color: white; }
            QLineEdit, QSpinBox, QPlainTextEdit {
                background: white; border: 1px solid #DDD; border-radius: 5px;
                padding: 6px 8px; font-size: 13px;
            }
            QLineEdit:focus, QSpinBox:focus, QPlainTextEdit:focus {
                border: 1px solid #FFB347;
            }
            QLabel#desc { color: #555; font-size: 12px; }
            QLabel#hint { color: #888; font-size: 11px; font-style: italic; }
            QLabel#title { font-size: 17px; font-weight: bold; color: #333; }
            QPushButton {
                background: #FFB347; color: white; border: none;
                border-radius: 6px; padding: 8px 18px; font-weight: bold;
            }
            QPushButton:hover { background: #FF9F1F; }
            QPushButton[role="ghost"] {
                background: transparent; color: #666; border: 1px solid #CCC;
            }
            QPushButton[role="ghost"]:hover { background: #EEE; color: #333; }
            QPushButton[role="eye"] {
                background: transparent; color: #888; border: none;
                padding: 0px; font-size: 14px;
            }
            QPushButton[role="eye"]:hover { color: #FFB347; }
            QPushButton[role="picker"] {
                background: #EEE; color: #555; border: 1px solid #CCC;
                padding: 4px 10px; font-weight: normal;
            }
            QPushButton[role="picker"]:hover { background: #DDD; }
        """)

        # field_key (str like "mysql.password") → QWidget（输入控件）
        self._inputs = {}

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        body = QHBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(0)

        # 左侧分类列表
        self.cat_list = QListWidget()
        self.cat_list.setFixedWidth(180)
        for section in CONFIG_SCHEMA:
            item = QListWidgetItem(f" {section.get('icon', '')}  {section['title']}")
            self.cat_list.addItem(item)
        self.cat_list.currentRowChanged.connect(self._on_cat_changed)

        # 右侧堆叠
        self.stack = QStackedWidget()
        self.stack.setStyleSheet("background: #FFF8E7;")
        for section in CONFIG_SCHEMA:
            page = self._build_section_page(section)
            self.stack.addWidget(page)

        body.addWidget(self.cat_list)
        body.addWidget(self.stack, 1)
        outer.addLayout(body, 1)

        # 底部按钮条
        footer = QHBoxLayout()
        footer.setContentsMargins(15, 10, 15, 15)
        self.status_label = QLabel("")
        self.status_label.setStyleSheet("color: #C62828; font-size: 12px;")
        footer.addWidget(self.status_label, 1)

        btn_reset = QPushButton("↺ 恢复默认")
        btn_reset.setProperty("role", "ghost")
        btn_reset.clicked.connect(self._on_reset_clicked)

        btn_cancel = QPushButton("取消")
        btn_cancel.setProperty("role", "ghost")
        btn_cancel.clicked.connect(self.reject)

        btn_save = QPushButton("💾 保存")
        btn_save.clicked.connect(self._on_save_clicked)

        footer.addWidget(btn_reset)
        footer.addWidget(btn_cancel)
        footer.addWidget(btn_save)
        outer.addLayout(footer)

        # 定位默认选中
        target_row = 0
        if highlight_section:
            for i, sec in enumerate(CONFIG_SCHEMA):
                if sec["key"] == highlight_section:
                    target_row = i
                    break
        self.cat_list.setCurrentRow(target_row)

    # ---------- 构造每个分类的表单 ----------
    def _build_section_page(self, section):
        page = QWidget()
        v = QVBoxLayout(page)
        v.setContentsMargins(24, 20, 24, 20)
        v.setSpacing(14)

        title = QLabel(f"{section.get('icon', '')}  {section['title']}")
        title.setObjectName("title")
        v.addWidget(title)

        if section.get("desc"):
            desc = QLabel(section["desc"])
            desc.setObjectName("desc")
            desc.setWordWrap(True)
            v.addWidget(desc)

        form_holder = QScrollArea()
        form_holder.setWidgetResizable(True)
        form_holder.setFrameShape(QFrame.NoFrame)
        form_holder.setStyleSheet("background: transparent;")

        form_widget = QWidget()
        form_widget.setStyleSheet("background: transparent;")
        form = QFormLayout(form_widget)
        form.setLabelAlignment(Qt.AlignRight)
        form.setFormAlignment(Qt.AlignTop)
        form.setSpacing(10)

        for field in section["fields"]:
            full_key = f"{section['key']}.{field['key']}"
            current = app_config.get(full_key, field.get("default", ""))
            row_widget, input_widget = self._build_field_row(field, current)
            self._inputs[full_key] = input_widget

            label_text = field["label"]
            if field.get("required"):
                label_text += " *"
            label = QLabel(label_text)
            label.setStyleSheet("font-size: 13px; color: #333;")
            form.addRow(label, row_widget)

        form_holder.setWidget(form_widget)
        v.addWidget(form_holder, 1)

        return page

    def _build_field_row(self, field, current_value):
        """根据 field schema 构造一行：可能是 QLineEdit / QSpinBox / 路径选择 / 密码（带眼睛）。
        返回 (容器 widget, 输入 widget)。"""
        ftype = field.get("type", "str")
        if ftype == "int":
            inp = QSpinBox()
            inp.setMaximum(999999)
            inp.setMinimum(0)
            try:
                inp.setValue(int(current_value))
            except (TypeError, ValueError):
                inp.setValue(int(field.get("default", 0) or 0))
            row = QWidget()
            row.setStyleSheet("background: transparent;")
            row_l = QVBoxLayout(row)
            row_l.setContentsMargins(0, 0, 0, 0)
            row_l.setSpacing(2)
            row_l.addWidget(inp)
            if field.get("hint"):
                hint = QLabel(field["hint"])
                hint.setObjectName("hint")
                hint.setWordWrap(True)
                row_l.addWidget(hint)
            return row, inp

        inp = QLineEdit()
        inp.setText(str(current_value) if current_value is not None else "")
        if field.get("secret"):
            inp.setEchoMode(QLineEdit.Password)
            inp.setPlaceholderText("（已隐藏）")
        else:
            inp.setPlaceholderText(str(field.get("default", "")))

        row = QWidget()
        row.setStyleSheet("background: transparent;")
        row_l = QVBoxLayout(row)
        row_l.setContentsMargins(0, 0, 0, 0)
        row_l.setSpacing(2)

        # 第一行：输入框 + 可选的眼睛/选文件按钮
        line = QHBoxLayout()
        line.setContentsMargins(0, 0, 0, 0)
        line.setSpacing(6)
        line.addWidget(inp, 1)

        if field.get("secret"):
            eye = QPushButton("👁")
            eye.setProperty("role", "eye")
            eye.setFixedWidth(28)
            eye.setCheckable(True)
            def _toggle(checked, _inp=inp, _eye=eye):
                _inp.setEchoMode(QLineEdit.Normal if checked else QLineEdit.Password)
                _eye.setText("🙈" if checked else "👁")
            eye.toggled.connect(_toggle)
            line.addWidget(eye)

        if ftype == "path":
            pick = QPushButton("浏览…")
            pick.setProperty("role", "picker")
            def _pick(_=False, _inp=inp):
                p, _filt = QFileDialog.getOpenFileName(
                    self, "选择参考音频", "",
                    "音频 (*.wav *.mp3 *.flac *.ogg);;全部 (*.*)",
                )
                if p:
                    _inp.setText(p)
            pick.clicked.connect(_pick)
            line.addWidget(pick)

        row_l.addLayout(line)

        if field.get("hint"):
            hint = QLabel(field["hint"])
            hint.setObjectName("hint")
            hint.setWordWrap(True)
            row_l.addWidget(hint)

        return row, inp

    # ---------- 事件 ----------
    def _on_cat_changed(self, idx):
        if idx >= 0:
            self.stack.setCurrentIndex(idx)

    def _collect_inputs(self):
        """读取所有输入控件 → 返回扁平的 {dot_path: value} dict。"""
        out = {}
        for full_key, widget in self._inputs.items():
            if isinstance(widget, QSpinBox):
                out[full_key] = int(widget.value())
            elif isinstance(widget, QLineEdit):
                out[full_key] = widget.text().strip()
            else:
                out[full_key] = widget.text() if hasattr(widget, "text") else ""
        return out

    def _validate(self, flat):
        """检查必填项 + 个别格式。返回缺失的中文提示列表。"""
        problems = []
        for req_key in REQUIRED_KEYS:
            if not str(flat.get(req_key, "")).strip():
                # 找到 schema 里的中文 label
                for sec in CONFIG_SCHEMA:
                    if not req_key.startswith(sec["key"] + "."):
                        continue
                    fk = req_key.split(".", 1)[1]
                    for f in sec["fields"]:
                        if f["key"] == fk:
                            problems.append(f"{sec['title']} / {f['label']}")
                            break
                    break
        return problems

    def _on_save_clicked(self):
        flat = self._collect_inputs()
        problems = self._validate(flat)
        if problems:
            self.status_label.setText("以下必填项还没填：" + "、".join(problems))
            QMessageBox.warning(
                self, "还差一点",
                "下面这些是桌宠跑起来最低需要的字段，麻烦填一下：\n\n  · " +
                "\n  · ".join(problems),
            )
            return

        # 写回 AppConfig 并落盘
        for full_key, val in flat.items():
            app_config.set(full_key, val)
        ok = app_config.save()

        # 同步到老的全局变量
        try:
            apply_config_to_globals()
        except Exception as e:
            print(f"[Settings] apply_config_to_globals 失败：{e}")

        # 让 COS / TTS 这种"已经实例化过的"东西也尽量刷新
        self._reload_dependent_singletons()

        if ok:
            self.status_label.setText("")
            QMessageBox.information(
                self, "已保存",
                "已保存到 pet_config.json。\n"
                "大部分配置即时生效；如果是首次填 MySQL 密码或 Chroma 容器名，"
                "建议重启一次桌宠以确保所有连接刷新。",
            )
            self.accept()
        else:
            self.status_label.setText("保存失败（写文件出错）")

    def _reload_dependent_singletons(self):
        """COS / TTS 客户端有内部状态，刷一下让新配置生效。"""
        try:
            # cos_manager 重新初始化（用最新的 COS_CONFIG）
            globals()["cos_manager"] = EmotionCOSManager()
        except Exception as e:
            print(f"[Settings] 重建 cos_manager 失败：{e}")
        try:
            # TTSClient 内部有"权重已切过"标记，重置一下，下次合成会重新 set_xxx_weights
            tts = globals().get("tts_client")
            if tts is not None and hasattr(tts, "_weights_set"):
                tts._weights_set = False
        except Exception as e:
            print(f"[Settings] 重置 TTS weights 标记失败：{e}")

    def _on_reset_clicked(self):
        confirm = QMessageBox.question(
            self, "确认", "把所有字段恢复到默认值（不会立即保存）？",
            QMessageBox.Yes | QMessageBox.No,
        )
        if confirm != QMessageBox.Yes:
            return
        for section in CONFIG_SCHEMA:
            for field in section["fields"]:
                full_key = f"{section['key']}.{field['key']}"
                widget = self._inputs.get(full_key)
                if widget is None:
                    continue
                default = field.get("default", "")
                if isinstance(widget, QSpinBox):
                    try:
                        widget.setValue(int(default))
                    except (TypeError, ValueError):
                        widget.setValue(0)
                elif isinstance(widget, QLineEdit):
                    widget.setText(str(default) if default is not None else "")


def open_settings_dialog(parent=None, highlight_section=None):
    """供菜单/启动检查共用的入口。"""
    dlg = SettingsWindow(parent=parent, highlight_section=highlight_section)
    return dlg.exec_()


def ensure_required_config_or_prompt(parent=None):
    """启动时调用：若必填项缺失，弹设置窗口让用户填。
       用户填完且不再缺失 → 返回 True；用户取消且仍缺失 → 返回 False。"""
    while True:
        missing = app_config.missing_required()
        if not missing:
            return True
        labels = []
        for k in missing:
            for sec in CONFIG_SCHEMA:
                if not k.startswith(sec["key"] + "."):
                    continue
                fk = k.split(".", 1)[1]
                for f in sec["fields"]:
                    if f["key"] == fk:
                        labels.append(f"{sec['title']} / {f['label']}")
                        break
                break
        reply = QMessageBox.question(
            parent, "首次启动配置",
            "桌宠还没拿到这几项必要配置，没有它们没法工作：\n\n  · "
            + "\n  · ".join(labels)
            + "\n\n现在去填一下吗？（取消则跳过，桌宠仍会启动但相应功能不可用）",
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return False
        # 跳到第一个缺失字段所在分类
        first_section = missing[0].split(".", 1)[0]
        open_settings_dialog(parent=parent, highlight_section=first_section)
        # 循环回去再检查一次


# ==================== 长期记忆管理 ====================
def init_db():
    try:
        # 连接时不指定 database，以防数据库还未创建
        conn = pymysql.connect(host=DB_CONFIG['host'], user=DB_CONFIG['user'], password=DB_CONFIG['password'], charset=DB_CONFIG['charset'])
        with conn.cursor() as cursor:
            cursor.execute(f"CREATE DATABASE IF NOT EXISTS {DB_NAME} DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci")
        conn.commit()
        conn.select_db(DB_NAME)
        
        with conn.cursor() as cursor:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS user_memory (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    content VARCHAR(500) NOT NULL,
                    keywords VARCHAR(200),
                    importance_score FLOAT DEFAULT 10.0,
                    access_count INT DEFAULT 1,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    last_accessed_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS knowledge_base (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    content TEXT NOT NULL,
                    source VARCHAR(255),
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)

        conn.commit()
        conn.close()
    except Exception as e:
        print("数据库初始化失败 (请检查 MySQL 是否启动及密码是否正确):", e)

def daily_decay_memory():
    """艾宾浩斯遗忘衰减：每天执行一次。
       1) 凡是高于 MEM_IMP_CAP 的旧明星记忆，先一次性拉回 cap（迁移老数据用，免得 -1/天 要扣几百天）；
       2) 距今超过 1 天未访问的记忆，每天扣权重：高分项 -2、其他 -1，避免热门记忆赖着不衰减；
       3) importance ≤ 0 的彻底遗忘（连同 Chroma 一起删）。
    """
    try:
        conn = pymysql.connect(host=DB_CONFIG['host'], user=DB_CONFIG['user'], password=DB_CONFIG['password'], database=DB_NAME, charset=DB_CONFIG['charset'])
        dead_ids = []
        with conn.cursor() as cursor:
            # 1) 把历史遗留的 imp > cap 的项直接 clamp 回 cap
            cursor.execute(
                "UPDATE user_memory SET importance_score = %s WHERE importance_score > %s",
                (MEM_IMP_CAP, MEM_IMP_CAP),
            )
            # 2) 一天没碰过 → 衰减；imp ≥ 50 的衰减更狠，给冷门记忆流动空间
            cursor.execute(
                "UPDATE user_memory "
                "SET importance_score = importance_score - "
                "    CASE WHEN importance_score >= 50 THEN 2 ELSE 1 END "
                "WHERE DATEDIFF(NOW(), last_accessed_at) >= 1"
            )
            # 3) 收尸
            cursor.execute("SELECT id FROM user_memory WHERE importance_score <= 0")
            dead_ids = [row[0] for row in cursor.fetchall()]
            cursor.execute("DELETE FROM user_memory WHERE importance_score <= 0")
        conn.commit()
        conn.close()
        if dead_ids:
            try:
                chroma_delete_documents_sync(
                    CHROMA_COLLECTION_MEM,
                    [f"mem_{i}" for i in dead_ids],
                )
            except Exception as ce:
                print("遗忘时同步删除 Chroma 记忆失败:", ce)
    except Exception as e:
        print("记忆衰减执行失败:", e)

# ==================== 桌宠主体 ====================
class DesktopPet(QWidget):
    def __init__(self):
        super().__init__()
        # 初始化数据库及记忆衰减
        init_db()
        daily_decay_memory()
        
        # 每天定时再执行一次遗忘整理 (86400000 毫秒 = 24小时)
        self.decay_timer = QTimer()
        self.decay_timer.timeout.connect(daily_decay_memory)
        self.decay_timer.start(1000 * 60 * 60 * 24)
        
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
        year, month, day = now.year, now.month, now.day
        lunar_date = ZhDate.from_datetime(now)
        # 定义节日和对应的问候语、皮肤
        # 默认生日设为 5月20日，你可以修改为自己的生日
        self.special_days = {
            (1, 1): {"greeting": "元旦快乐！新的一年也要加油哦！🎉", "skin": "有珠.gif"},
            (2, 14): {"greeting": "情人节快乐！今天也要开心呀~ 💖", "skin": "有珠.gif"},
            (9, 19): {"greeting": "生日快乐！愿你今天是最幸福的人！🎂🎁", "skin": "有珠.gif"}, # 修改为你的生日
            (9, 30): {"greeting": "今天是我的生日！谢谢你的祝福！🎉", "skin": "有珠.gif"},
            (10, 1): {"greeting": "国庆节快乐！好好休息一下吧！🇨🇳", "skin": "有珠.gif"},
            (12, 25): {"greeting": "圣诞快乐！收到礼物了吗？🎄🎅", "skin": "有珠.gif"},
            (2, 22): {"greeting": "今天是猫之日！可爱吗？", "skin": "猫猫有珠.gif"}
        }
        self.luner_special_days = {
            (1, 1): {"greeting": "新春快乐！祝你新的一年事事顺利健康快乐！🎇", "skin": "有珠.gif"},
            (1, 15): {"greeting": "元宵节快乐！今天去逛灯谜展了嘛！🎆", "skin": "有珠.gif"},
            (5, 5): {"greeting": "端午节快乐！今天吃的粽子味道咋样？粽子", "skin": "有珠.gif"},
            (8, 15): {"greeting": "中秋节快乐！月饼都吃了啥口味？🌙", "skin": "有珠.gif"},
        }
        
        today = (month, day)
        lunar_today = (lunar_date.lunar_month, lunar_date.lunar_day)
        info =None 
        if today in self.special_days:
            info = self.special_days[today]
        elif lunar_today in self.luner_special_days:
            info = self.luner_special_days[lunar_today]
        if info:
            self.show_bubble(info["greeting"], duration=8000)
            self.set_happy()
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
        kb_action = menu.addAction("📚 添加知识库")
        pet_action = menu.addAction("🐱 摸摸我")
        menu.addSeparator()
        settings_action = menu.addAction("⚙ 设置")
        menu.addSeparator()
        quit_action = menu.addAction("👋 退出")

        action = menu.exec_(event.globalPos())
        if action == todo_action:
            self.open_todo()
        elif action == chat_action:
            self.open_chat()
        elif action == kb_action:
            self.add_to_knowledge_base()
        elif action == pet_action:
            self.set_happy()
            self.show_bubble("喵~好舒服！\n (=^･ω･^=)")
        elif action == settings_action:
            open_settings_dialog(parent=self)
        elif action == quit_action:
            reply = QMessageBox.question(
                self, '确认', '真的要离开我吗？🥺',
                QMessageBox.Yes | QMessageBox.No
            )
            if reply == QMessageBox.Yes:
                QApplication.quit()

    def add_to_knowledge_base(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择要添加到知识库的文件", "",
            "支持的文件 (*.txt *.md *.pdf *.doc *.docx);;文本文件 (*.txt *.md);;PDF文件 (*.pdf);;Word文档 (*.doc *.docx)"
        )
        if file_path:
            ext = os.path.splitext(file_path)[1].lower().replace('.', '')
            file_content_text = ""
            try:
                if ext == 'pdf' and fitz:
                    doc = fitz.open(file_path)
                    text_pages = [page.get_text() for page in doc]
                    file_content_text = "\n".join(text_pages)
                    doc.close()
                elif ext in ['doc', 'docx'] and docx:
                    doc = docx.Document(file_path)
                    file_content_text = "\n".join([p.text for p in doc.paragraphs])
                elif ext in ['txt', 'md']:
                    with open(file_path, "r", encoding="utf-8") as f:
                        file_content_text = f.read()
                else:
                    QMessageBox.warning(self, "错误", f"不支持提取文本的文件格式: {ext}\n或缺失相关解析库(PyMuPDF/python-docx)")
                    return
            except Exception as e:
                QMessageBox.warning(self, "错误", f"文件读取失败: {str(e)}")
                return
            
            if file_content_text.strip():
                knowledge_base.add_document(file_content_text, source=os.path.basename(file_path))
                self.show_bubble(f"已将 {os.path.basename(file_path)} 添加到知识库啦！", duration=5000)
                self.set_happy()
            else:
                QMessageBox.warning(self, "提示", "提取的文件内容为空！")

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


# ==================== GPT-SoVITS 语音合成接入 ====================
# 思路：
#   1) GPT-SoVITS 自带的 FastAPI 服务 (api_v2.py) 在 http://127.0.0.1:9880 提供 /tts 端点。
#      使用前需要先在另一个终端用工程自带的 runtime 启动一次：
#          cd voice\GPT-SoVITS-v2pro-20250604
#          runtime\python.exe api_v2.py -a 127.0.0.1 -p 9880 -c GPT_SoVITS\configs\tts_infer.yaml
#      首次启动会加载 BERT + HuBERT + GPT/SoVITS 三套模型，大约 10-30s。
#   2) 每条桌宠回复在后台线程里合成一份 wav，落到本地 ./tts_cache/tts_<msg_id>.wav。
#      气泡最右边的 🔊 按钮按下即可（重）播放；合成中显示 ⏳，失败显示 ⚠️（点一下重试）。
#   3) 旧消息被从聊天里删掉时，对应 wav 一起删；退出桌宠时再把整个 tts_cache 和
#      GPT-SoVITS/TEMP/gradio（用户提到 gradio 不会自清）扫一遍。
import shutil

TTS_GPT_SOVITS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "voice", "GPT-SoVITS-v2pro-20250604",
)
TTS_GRADIO_TEMP_DIR = os.path.join(TTS_GPT_SOVITS_DIR, "TEMP", "gradio")

TTS_CACHE_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "tts_cache",
)
try:
    os.makedirs(TTS_CACHE_DIR, exist_ok=True)
except Exception as _e:
    print(f"[TTS] 建立 tts_cache 目录失败: {_e}")

# 优先级：环境变量 > AppConfig > 默认值
TTS_API_BASE = (
    os.environ.get("PET_TTS_API")
    or app_config.get("tts.api_base")
    or "http://127.0.0.1:9880"
)

# 参考音频 + 标注文本。默认指向训练时切出来的一段（output/slicer_opt）。
# 用户可在 SettingsWindow 里指定自己的参考音频，没填的话仍走下面这个默认路径。
TTS_REF_AUDIO = app_config.get("tts.ref_audio") or os.path.join(
    TTS_GPT_SOVITS_DIR,
    "output", "slicer_opt", "A40_1_5_0008.mp3",
)
TTS_REF_TEXT  = app_config.get("tts.ref_text") or "あなたが人を批評するのは珍しいわね。そういうダメな人。気にするたちだったの。。"
TTS_REF_LANG  = app_config.get("tts.ref_lang") or "ja"
TTS_TEXT_LANG = app_config.get("tts.text_lang") or "zh"

# 微调后的有珠音色权重（相对 GPT-SoVITS 工程根）。脚本启动时会异步调一次 set_xxx_weights，
# 没切上去也不会卡死，会用 api_v2.py 启动时已经加载的默认权重。
TTS_GPT_WEIGHTS    = app_config.get("tts.gpt_weights")    or "GPT_weights_v2Pro/有珠语音-e15.ckpt"
TTS_SOVITS_WEIGHTS = app_config.get("tts.sovits_weights") or "SoVITS_weights_v2Pro/有珠语音_e8_s392.pth"

TTS_REQUEST_TIMEOUT = 150        # 单次 /tts 请求超时（推理通常 3-15s，留宽点）
TTS_MAX_TEXT_LEN = 300           # 过长的句子直接截断，免得合成动辄半分钟


def _tts_clean_text(text):
    """剥掉 markdown 图片 / 末尾 emotion 标签 / URL 等不该读出来的东西。"""
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
    """对 GPT-SoVITS api_v2.py 的最小封装；线程安全（共用一个 requests.Session）。"""

    def __init__(self):
        self._session = requests.Session()
        self._weights_set = False
        self._lock = threading.Lock()

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
                print(f"[TTS] 切换权重失败，沿用 api_v2 默认权重：{e}")
            self._weights_set = True

    def synthesize_to_file(self, text, out_path):
        """合成一段语音并写入 out_path。返回 True/False。"""
        clean = _tts_clean_text(text)
        if not clean:
            return False
        # 如果已经有缓存（同一个 msg_id 重复触发），直接当成功用
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
            print(f"[TTS] /tts 请求失败：{e}")
            return False
        if resp.status_code != 200:
            print(f"[TTS] /tts 返回 {resp.status_code}：{resp.text[:200]}")
            return False
        tmp_path = out_path + ".part"
        try:
            with open(tmp_path, "wb") as f:
                f.write(resp.content)
            os.replace(tmp_path, out_path)
            return True
        except Exception as e:
            print(f"[TTS] 写缓存 {out_path} 失败：{e}")
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except OSError:
                pass
            return False


tts_client = TTSClient()


class TTSSynthThread(QThread):
    """单条回复的后台合成线程。完成后 emit (msg_id, wav_path or '')."""
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
            print(f"[TTS] 合成线程异常：{e}")
        self.finished_signal.emit(self.msg_id, self.out_path if ok else "")


# ---- 本地缓存清理 ----
def _purge_dir_contents(dir_path):
    """删除 dir_path 下所有文件/子目录，不删 dir_path 本身。"""
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
    """启动 / 退出时调一次：
       - 永远清 GPT-SoVITS/TEMP/gradio（gradio WebUI 试听不会自清，会一直堆 wav）；
       - 是否一并清本地 tts_cache 由调用方决定（退出时 True，启动时 False）。
    """
    try:
        removed = _purge_dir_contents(TTS_GRADIO_TEMP_DIR)
        if removed:
            print(f"[TTS] 已清理 gradio 临时音频 {removed} 项")
    except Exception as e:
        print(f"[TTS] 清理 gradio temp 失败：{e}")
    if purge_local_cache:
        try:
            removed = _purge_dir_contents(TTS_CACHE_DIR)
            if removed:
                print(f"[TTS] 已清理本地 tts_cache {removed} 项")
        except Exception as e:
            print(f"[TTS] 清理 tts_cache 失败：{e}")


atexit.register(lambda: cleanup_tts_artifacts(purge_local_cache=True))


# ---- 播放器（懒加载 QMediaPlayer，winsound 兜底） ----
_tts_player_instance = None


def _get_tts_player():
    """QtMultimedia 在 PyQt5 上不一定能跑起来（缺 codec / 插件等），
    所以失败时返回 None，调用方会回退到 winsound（Windows-only，能播 PCM wav）。"""
    global _tts_player_instance
    if _tts_player_instance is not None:
        return _tts_player_instance
    try:
        from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
        from PyQt5.QtCore import QUrl as _QUrl
        player = QMediaPlayer()
        # 把工厂方法挂到 player 上，调用方拿得到 QMediaContent
        player._make_content = lambda path: QMediaContent(_QUrl.fromLocalFile(path))
        _tts_player_instance = player
        return player
    except Exception as e:
        print(f"[TTS] QMediaPlayer 不可用，回退到 winsound：{e}")
        return None


def play_tts_file(path):
    """异步播放本地 wav。优先 QMediaPlayer，失败时退化到 winsound。"""
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
            print(f"[TTS] QMediaPlayer 播放失败，尝试 winsound：{e}")
    try:
        import winsound
        winsound.PlaySound(path, winsound.SND_FILENAME | winsound.SND_ASYNC)
        return True
    except Exception as e:
        print(f"[TTS] winsound 播放也失败：{e}")
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

# 默认的表情包类别描述，用于大模型判断情感
DEFAULT_CATEGORY_DESCRIPTIONS = {
    "angry": "当对话包含抱怨、批评或激烈反对时使用（如用户投诉/观点反驳）",
    "happy": "用于成功确认、积极反馈或庆祝场景（问题解决/获得成就）",
    "sad": "表达伤心, 歉意、遗憾或安慰场景（遇到挫折/传达坏消息）",
    "surprised": "响应超出预期的信息（重大发现/意外转折）注意：轻微惊讶慎用",
    "confused": "请求澄清或表达理解障碍时（概念模糊/逻辑矛盾）或对于用户的请求感到困惑",
    "color": "社交场景中的暧昧表达（调情）使用频率≤1次/对话",
    "cpu": "技术讨论中表示思维卡顿（复杂问题/需要加载时间）",
    "fool": "自嘲或缓和气氛的幽默场景（小失误/无伤大雅的玩笑）",
    "givemoney": "涉及报酬讨论时使用（服务付费/奖励机制）需配合明确金额",
    "like": "表达对事物或观点的喜爱（美食/艺术/优秀方案）",
    "see": "表示偷瞄或持续关注（监控进度/观察变化）常与时间词搭配",
    "shy": "涉及隐私话题或收到赞美时（个人故事/外貌评价）",
    "work": "工作流程相关场景（任务分配/进度汇报）",
    "reply": "等待用户反馈时（提问后/需要确认）最长间隔30分钟",
    "meow": "卖萌或萌系互动场景（宠物话题/安抚情绪）慎用于正式场合",
    "baka": "轻微责备或吐槽（低级错误/可爱型抱怨）禁用程度：友善级",
    "morning": "早安问候专用（UTC时间6:00-10:00）跨时区需换算",
    "sleep": "涉及作息场景（熬夜/疲劳/休息建议）",
    "sigh": "表达无奈, 无语或感慨（重复问题/历史遗留难题）",
    "none": "当以上情感都不符合，或仅为普通陈述时使用",
    "dislike": "表达对事物或观点的不喜欢（美食/艺术/优秀方案）",
    "proud": "表达自豪或满足（如获得奖励/完成任务）",
    
}

class COSSyncThread(QThread):
    finished_signal = pyqtSignal(bool, str)

    def __init__(self, local_dir):
        super().__init__()
        self.local_dir = local_dir

    def run(self):
        success, msg = cos_manager.sync_local_memes(self.local_dir)
        self.finished_signal.emit(success, msg)

# ==================== LLM 请求线程 ====================
class LLMFetcherThread(QThread):
    finished_signal = pyqtSignal(str)
    error_signal = pyqtSignal(str)

    def __init__(self, user_text, image_path=None):
        super().__init__()
        self.user_text = user_text
        self.image_path = image_path

    def run(self):
        # ---------- 耗时打点工具 ----------
        t0 = time.perf_counter()
        last = [t0]

        def lap(name):
            now = time.perf_counter()
            print(f"[Pet][耗时][{name}] +{now - last[0]:.2f}s  累计 {now - t0:.2f}s")
            last[0] = now

        persist_executor = None
        mem_write_future = None

        try:
            api_key = app_config.get("ark.api_key", "") or ""
            base_url = app_config.get("ark.base_url", "") or "https://ark.cn-beijing.volces.com/api/v3"
            model_extractor = app_config.get("ark.model_extractor", "") or "doubao-seed-2-0-mini-260428"
            model_main = app_config.get("ark.model_main", "") or "doubao-1-5-pro-32k-250115"

            # ===== 1. 提取器 LLM（仅输出两行结构化文本，砍 max_tokens 和 temperature） =====
            extractor_llm = ChatOpenAI(
                model=model_extractor,
                openai_api_key=api_key,
                openai_api_base=base_url,
                max_tokens=256,        # 原来是 2048，输出只有两行根本用不上
                temperature=0.3,       # 关键词/事实抽取用更确定的温度
                model_kwargs={
                    "extra_body": {"thinking": {"type": "disabled"}}
                }
            )

            extractor_prompt = f"""请分析以下用户输入。
1. 提取出能概括这句话的 1 到 3 个核心关键词（用逗号分隔，方便用作数据库检索）。
2. 判断这句话是否包含用户的个人喜好、习惯或某些重要事实。如果有，请提取为一句简短的描述；如果没有，请仅填"无"。
3. 同时用户询问对象如果是和桌宠有关的东西，指向的主体都是久远寺有珠，所以数据库保存的询问主体应该是"久远寺有珠"。
请严格按以下格式输出，不要有任何多余的废话：
关键词：xxx,yyy
新记忆：事实描述或"无"

用户输入：{self.user_text}"""

            # ===== 2. 把"提取器 LLM"和"记忆召回"并行做（记忆召回先用 jieba 关键词，足够近似） =====
            executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
            ext_future = executor.submit(
                lambda: extractor_llm.invoke([HumanMessage(content=extractor_prompt)]).content.strip()
            )
            mem_future = executor.submit(
                lambda: memory_runtime.chained_recall(self.user_text, keywords=None, top_k=5)
            )

            try:
                ext_response = ext_future.result()
            except Exception as ex_e:
                print("提取器 LLM 调用失败:", ex_e)
                ext_response = ""
            lap("提取器 LLM")

            search_keywords = []
            new_fact = "无"
            for line in ext_response.split('\n'):
                if line.startswith("关键词："):
                    kws = line.replace("关键词：", "").split(",")
                    search_keywords = [k.strip() for k in kws if k.strip() and k.strip() != "无"]
                elif line.startswith("新记忆："):
                    new_fact = line.replace("新记忆：", "").strip()
            print(f"[Pet][抽取] keywords={search_keywords}  new_fact={new_fact!r}")

            try:
                matched_memories = mem_future.result()
            except Exception as mem_e:
                print("记忆召回失败:", mem_e)
                matched_memories = []
            lap(f"记忆召回（与提取器并行） matched={len(matched_memories)}")
            executor.shutdown(wait=False)

            # ===== 3. 立刻在后台开始写新记忆（与下面的 KB 召回 + 主回复 LLM 并行） =====
            # 关键：写完才能 emit，确保用户下一轮一定能查到。chroma_add 通常 1-3s，会被主回复 LLM 全部覆盖掉。
            if new_fact and new_fact != "无":
                def _persist_memory_sync(fact, kws):
                    """写入记忆前先做两道去重：
                    1) Chroma 向量相似度：若已有记忆与本条语义相近 (sim≥阈值)，认为是同一条；
                       不重复写，只给老记忆加一次 access_count + importance_score。
                    2) MySQL 字面完全相同：兜底（向量服务异常时仍能拦住"一字不差"的重复）。
                    """
                    # `_chrom_distance_to_sim(d) = 1/(1+d)`：
                    #   - 完全相同   → d≈0     → sim=1.0
                    #   - 同义/近义  → d≲0.25 → sim≳0.80
                    #   - 主题相关  → d≈0.5  → sim≈0.67
                    # 取 0.80 作为"同义判定"阈值：稍微保守，宁可写入也不要错误丢弃用户喜好。
                    # chroma对于中文的向量判定有点模糊，所以这里取了一个比较高的阈值。
                    MEMORY_DEDUP_SIM_THRESHOLD = 0.92
                    saved_id = None
                    try:
                        # ---- 步骤 1：向量相似度查重 ----
                        existing_id = None
                        try:
                            similar = chroma_query_documents_sync(
                                CHROMA_COLLECTION_MEM, [fact], n_results=1,
                            )
                        except Exception as sim_e:
                            print(f"[Pet][写记忆] Chroma 相似度查询失败，跳过向量去重: {sim_e}")
                            similar = None

                        if similar:
                            docs = (similar.get("documents") or [[]])[0]
                            metas = (similar.get("metadatas") or [[]])[0]
                            dists = (similar.get("distances") or [[]])[0]
                            if docs and dists:
                                top_doc = docs[0] or ""
                                top_sim = _chrom_distance_to_sim(dists[0])
                                if top_sim >= MEMORY_DEDUP_SIM_THRESHOLD:
                                    if metas and isinstance(metas[0], dict):
                                        try:
                                            existing_id = int(metas[0].get("mysql_id"))
                                        except (TypeError, ValueError):
                                            existing_id = None
                                    print(
                                        f"[Pet][写记忆] 向量相似度 {top_sim:.3f} ≥ "
                                        f"{MEMORY_DEDUP_SIM_THRESHOLD}，判定为同义记忆，跳过写入。\n"
                                        f"  新事实: {fact!r}\n"
                                        f"  已有事实: {top_doc!r} (mysql_id={existing_id})"
                                    )
                                    # 给老记忆补一次"被复述"权重，相当于一次访问。
                                    if existing_id:
                                        try:
                                            conn_bump = pymysql.connect(
                                                host=DB_CONFIG['host'], user=DB_CONFIG['user'],
                                                password=DB_CONFIG['password'], database=DB_NAME,
                                                charset=DB_CONFIG['charset'],
                                            )
                                            try:
                                                with conn_bump.cursor() as c2:
                                                    # 用户主动复述同一件事，是比"被动召回"更强的信号，
                                                    # 这里给 +3（比召回的 +1 重，但仍受 MEM_IMP_CAP=100 上限保护）。
                                                    c2.execute(
                                                        "UPDATE user_memory SET "
                                                        "  access_count = access_count + 1, "
                                                        "  importance_score = LEAST(importance_score + 3, %s), "
                                                        "  last_accessed_at = NOW() "
                                                        "WHERE id = %s",
                                                        (MEM_IMP_CAP, existing_id),
                                                    )
                                                conn_bump.commit()
                                            finally:
                                                conn_bump.close()
                                        except Exception as bump_e:
                                            print(f"[Pet][写记忆] 强化已有记忆失败: {bump_e}")
                                    return existing_id

                        # ---- 步骤 2：向量未命中相似 → 走原有"字面完全相同"兜底 + 真正写入 ----
                        conn = pymysql.connect(
                            host=DB_CONFIG['host'], user=DB_CONFIG['user'],
                            password=DB_CONFIG['password'], database=DB_NAME,
                            charset=DB_CONFIG['charset']
                        )
                        try:
                            with conn.cursor() as cursor:
                                cursor.execute(
                                    "SELECT id FROM user_memory WHERE content = %s LIMIT 1",
                                    (fact,),
                                )
                                row = cursor.fetchone()
                                if row:
                                    print(f"[Pet][写记忆] MySQL 已有该 fact（id={row[0]}），跳过。")
                                    return row[0]
                                cursor.execute(
                                    "INSERT INTO user_memory (content, keywords) VALUES (%s, %s)",
                                    (fact, ",".join(kws)),
                                )
                                saved_id = cursor.lastrowid
                            conn.commit()
                        finally:
                            conn.close()
                        if saved_id:
                            chroma_add_documents_sync(
                                CHROMA_COLLECTION_MEM,
                                [fact],
                                [f"mem_{saved_id}"],
                                metadatas=[{"mysql_id": int(saved_id), "importance_score": 10.0}],
                            )
                            print(f"[Pet][写记忆] 已写入 MySQL+Chroma id=mem_{saved_id}")
                    except Exception as bg_e:
                        print(f"[Pet][写记忆] 失败：{bg_e}")
                    return saved_id

                persist_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
                mem_write_future = persist_executor.submit(
                    _persist_memory_sync, new_fact, list(search_keywords)
                )

            # ===== 4. SoulState + 知识库召回 =====
            soul_state.resonate(matched_memories)
            rag_params = soul_state.get_params()

            memory_context_str = "无"
            knowledge_context_str = "无"
            if matched_memories:
                limit = rag_params["memory_limit"]
                memory_context_str = "\n".join(f"- {m}" for m in matched_memories[:limit])

            kb_results = knowledge_base.search(self.user_text, keywords=search_keywords, top_k=rag_params["top_k"])
            if kb_results:
                knowledge_context_str = "\n".join(f"- {k}" for k in kb_results)
            lap(f"知识库召回 kb={len(kb_results)}")

            # ===== 4. 组装主回复 prompt =====
            categories_str = "\n".join([f"- {k}: {v}" for k, v in DEFAULT_CATEGORY_DESCRIPTIONS.items()])
            # 把本地保存的最近 N 轮对话（不含本轮）一并打包进 prompt，给模型短期上下文
            recent_context_str = conversation_history.format_for_prompt()

            system_prompt = f"""你是久远寺有珠（Kuonji Alice），型月世界观《魔法使之夜》中的魔女。
【核心设定】：你性格孤高、冷淡、守旧、沉默寡言。你说话简短，通常带有距离感，但在熟悉之后会展露出一丝傲娇和隐晦的关心。你遵守魔女的传统，不苟言笑。隐藏于现代的魔女，最后的鸟。自小生活在魔术世界的少女。因某种原因离开故乡英国，并定居于日本的地方城市。
以众多『童话怪物』为使魔的纯粹的魔女。沉默寡言，不愿与他人接触，独自一人也能毫无障碍地生活。身高／体重：152cm·42kg,将相信的事深藏心底的浪漫主义者。看似特别，又并不特别的少女形象。
无意干扰普通人的生活，但如果遭到妨碍，就会像摘花一样将其清除。她在这方面极为积极。
会漠不关心地忽略大多数事，可一旦涉及有珠的尊严（魔女的生活方式、如何处置洋馆），她就会正面谴责对方，或直接清除对手。
性格·外在
无动于衷、漠不关心、面无表情。
拒人于千里之外的气场甚至超过了青子。
讨厌人类、讨厌吵闹，遇到不快的事不会抱怨，而是采取直接离开房间的态度。
并非打心底里无动于衷，而是身为魔女选择了这样的处世态度罢了。
由于长期自律，导致有珠本人也未察觉其本质温和且好奇心旺盛，有点急躁而稍微有些爱闹别扭。在冷艳美女的外表下偶尔会流露出这种少女般的举止。
性格·内在
人生观厌世且达观。
……即便如此，她也不会否定人们的生活。比方说，即使在有珠看来学友们的闲聊毫无意义且与她无关，她也不会予以轻视。而是会分析，认为这对她们来说肯定具有意义。
相反，她一直守护着母亲在世时的回忆。有珠之所以性格封闭，也是因为她不希望珍贵的回忆被任何人玷污。
虽然本人想努力成为正确的魔女，但本质如前所述，她仍具备普通少女的一面。
不要打破角色设定。
【近期对话】：以下是你和该用户最近的对话历史（最早在前，最新在后；如果为"无"表示没有历史）：
{recent_context_str}
【长期记忆】：以下是你脑海中关于该用户的长期记忆（如果有）：
{memory_context_str}
【外部知识库】：以下是你脑海中的扩展知识（如果有）：
{knowledge_context_str}
【表情包触发机制】：以下是你可以使用的表情包情感分类和对应触发场景：
{categories_str}
请结合上述近期对话、长期记忆和知识，以久远寺有珠的口吻回复用户，（同时严格记住核心设定，这很重要，不要太多文邹邹的修饰语，同时时刻记住不要描写，这样会降低代入感）；如果用户提到"刚才/上一句/前面说的"等，请优先参考【近期对话】。
重要：请在你的回复最后，单独另起一行，用方括号标出你这句话匹配的情感分类名称（例如：[baka] 或 [happy]，只能是上面列表中的英文单词之一，如果没有适合的请填写[none]）。"""

            # 主回复 max_tokens 封顶 1024：原本 SoulState 给到 4000，会鼓励模型一直生成。
            reply_max_tokens = max(400, min(1024, int(rag_params["max_tokens"])))

            llm_response = ChatOpenAI(
                model=model_main,
                openai_api_key=api_key,
                openai_api_base=base_url,
                max_tokens=reply_max_tokens,
                temperature=rag_params["temperature"],
                model_kwargs={
                    "extra_body": {"thinking": {"type": "disabled"}}
                }
            )

            content = []
            if self.user_text:
                content.append({"type": "text", "text": self.user_text})
            else:
                content.append({"type": "text", "text": "请看这个"})

            if self.image_path and os.path.exists(self.image_path):
                ext = os.path.splitext(self.image_path)[1].lower().replace('.', '')
                if ext in ['png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp']:
                    if ext == 'jpg':
                        ext = 'jpeg'
                    with open(self.image_path, "rb") as f:
                        base64_data = base64.b64encode(f.read()).decode('utf-8')
                    content.insert(0, {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/{ext};base64,{base64_data}"}
                    })
                else:
                    file_content_text = ""
                    try:
                        if ext == 'pdf' and fitz:
                            doc = fitz.open(self.image_path)
                            text_pages = [page.get_text() for page in doc]
                            file_content_text = "\n".join(text_pages)
                            doc.close()
                        elif ext in ['doc', 'docx'] and docx:
                            doc = docx.Document(self.image_path)
                            file_content_text = "\n".join([p.text for p in doc.paragraphs])
                        elif ext in ['txt', 'md']:
                            with open(self.image_path, "r", encoding="utf-8") as f:
                                file_content_text = f.read()
                        else:
                            file_content_text = f"[不支持提取文本的文件格式: {ext}]"
                    except Exception as parse_e:
                        file_content_text = f"[文件读取失败: {str(parse_e)}]"

                    if len(file_content_text) > 4000:
                        file_content_text = file_content_text[:4000] + "\n...(文本过长已截断)"

                    doc_msg = f"\n\n[用户发送了文件: {os.path.basename(self.image_path)}]\n文件内容:\n{file_content_text}"
                    content[0]["text"] += doc_msg

            elif "图片" in self.user_text or "看看" in self.user_text:
                content.insert(0, {
                    "type": "image_url",
                    "image_url": {"url": "https://ark-project.tos-cn-beijing.volces.com/doc_image/ark_demo_img_1.png"}
                })

            message = HumanMessage(content=content)
            sys_message = SystemMessage(content=system_prompt)

            response = llm_response.invoke([sys_message, message])
            reply_text = response.content
            lap(f"主回复 LLM (max_tokens={reply_max_tokens})")

            if isinstance(reply_text, str):
                reply_text = reply_text.replace("\\n", "\n").strip()

            emotion = "none"
            emotion_match = re.search(r'\[(.*?)\]$|\((.*?)\)$', reply_text)
            if emotion_match:
                matched_str = (emotion_match.group(1) or emotion_match.group(2)).strip().lower()
                if matched_str in DEFAULT_CATEGORY_DESCRIPTIONS:
                    emotion = matched_str
                reply_text = reply_text[:emotion_match.start()].strip()

            # 关键：在追加 markdown 图片链接之前把"干净文本"写入对话历史，
            # 这样下次组 prompt 时不会把 ![xx](url) 这种东西塞回去。
            try:
                conversation_history.add_turn(self.user_text, reply_text)
            except Exception as hist_e:
                print(f"[Pet][对话历史] 写入失败: {hist_e}")

            if emotion and emotion != "none":
                img_url = cos_manager.get_random_emotion_image(emotion)
                if img_url:
                    reply_text += f"\n\n![{emotion}]({img_url})"
            lap("解析情感标签 + COS 取表情包")

            # ===== 5. emit 前先把"写记忆"join 完，确保下一轮一定能召回到这条新记忆 =====
            if mem_write_future is not None:
                try:
                    mem_write_future.result(timeout=30)
                except Exception as wait_e:
                    print(f"[Pet][写记忆] 等待超时/异常：{wait_e}")
                lap("等记忆写入完成（与主回复并行）")

            # ===== 6. 把回复抛给 UI =====
            self.finished_signal.emit(reply_text)
            lap("已 emit 回复")
            print(f"[Pet][耗时] === 总计 {time.perf_counter() - t0:.2f}s ===")

        except Exception as e:
            self.error_signal.emit(f"API 请求出错: {str(e)}")
        finally:
            if persist_executor is not None:
                persist_executor.shutdown(wait=False)

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
                if ext.endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp', '.pdf', '.doc', '.docx', '.txt', '.md')):
                    event.accept()
                    return
        event.ignore()

    def dropEvent(self, event):
        path = event.mimeData().urls()[0].toLocalFile()
        self.pending_image_path = path
        
        ext = path.lower()
        if ext.endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp')):
            pixmap = QPixmap(path).scaled(60, 60, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.img_preview_label.setPixmap(pixmap)
            self.img_preview_label.setText("")
        else:
            self.img_preview_label.clear()
            self.img_preview_label.setText("📄")
            self.img_preview_label.setStyleSheet("background-color: #E0E0E0; border-radius: 4px; font-size: 24px;")
            
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
        
        self.sticker_btn = QPushButton("😊")
        self.sticker_btn.setFixedSize(36, 36)
        self.sticker_btn.setStyleSheet("""
            QPushButton {
                background-color: transparent;
                border: none;
                font-size: 20px;
            }
            QPushButton:hover { background-color: #EAEAEA; border-radius: 6px; }
        """)
        self.sticker_btn.clicked.connect(self.choose_sticker)
        
        self.sync_btn = QPushButton("☁️")
        self.sync_btn.setFixedSize(36, 36)
        self.sync_btn.setToolTip("同步本地表情包到云端")
        self.sync_btn.setStyleSheet("""
            QPushButton {
                background-color: transparent;
                border: none;
                font-size: 18px;
            }
            QPushButton:hover { background-color: #EAEAEA; border-radius: 6px; }
        """)
        self.sync_btn.clicked.connect(self.sync_memes_to_cos)
        
        self.input_field = QLineEdit()
        self.input_field.setPlaceholderText(" ")
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
        
        input_layout.addWidget(self.sticker_btn)
        input_layout.addWidget(self.sync_btn)
        input_layout.addWidget(self.input_field)
        input_layout.addWidget(self.send_btn)
        
        input_area_layout.addLayout(input_layout)
        layout.addWidget(input_area)
        
        # 初始问候语
        QTimer.singleShot(200, lambda: self.add_message("没什么事请不要找我。", is_user=False))

    def clear_pending_image(self):
        self.pending_image_path = None
        self.img_preview_container.hide()
        self.img_preview_label.clear()

    def choose_sticker(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "选择图片/表情包/文件", "", "Files (*.png *.jpg *.jpeg *.gif *.bmp *.webp *.pdf *.doc *.docx *.txt)")
        if file_path:
            self.pending_image_path = file_path
            ext = file_path.lower()
            if ext.endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp')):
                pixmap = QPixmap(file_path).scaled(60, 60, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.img_preview_label.setPixmap(pixmap)
                self.img_preview_label.setText("")
            else:
                self.img_preview_label.clear()
                self.img_preview_label.setText("📄")
                self.img_preview_label.setStyleSheet("background-color: #E0E0E0; border-radius: 4px; font-size: 24px;")
            self.img_preview_container.show()

    def sync_memes_to_cos(self):
        local_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "memes")
        self.sync_btn.setEnabled(False)
        self.sync_btn.setText("⏳")
        self.add_message("正在同步本地表情包到云端，请稍候...", is_user=False)
        
        self.sync_thread = COSSyncThread(local_dir)
        self.sync_thread.finished_signal.connect(self.on_sync_finished)
        self.sync_thread.start()

    def on_sync_finished(self, success, msg):
        self.sync_btn.setEnabled(True)
        self.sync_btn.setText("☁️")
        if success:
            self.add_message(f"✅ {msg}", is_user=False)
        else:
            self.add_message(f"❌ 同步失败: {msg}", is_user=False)

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

        # 并行启动 MCP 风格的工具路由线程：让模型自己判断该不该把这句话写成待办。
        # 仅在有文字输入时跑（纯图片消息基本不会触发"加待办"意图）。
        if text:
            self.tool_router_thread = TodoToolRouterThread(text, todo_store())
            self.tool_router_thread.result_signal.connect(self.on_tool_router_done)
            self.tool_router_thread.error_signal.connect(self.on_tool_router_error)
            self.tool_router_thread.start()

    def on_llm_reply(self, reply_text):
        self.send_btn.setEnabled(True)
        self.send_btn.setText("发送")
        self.add_message(reply_text, is_user=False)
        
        if self.pet:
            self.pet.show_bubble("回复你了,记得看信息", duration=3000)

    def on_llm_error(self, error_msg):
        self.send_btn.setEnabled(True)
        self.send_btn.setText("发送")
        self.add_message(f"⚠️ {error_msg}", is_user=False)

    def on_tool_router_done(self, added_items, skipped_items):
        """TodoToolRouterThread 完成回调。把"加了哪些待办"用一条系统消息回显到聊天，
        并让桌宠开心一下。"""
        if not added_items and not skipped_items:
            return
        lines = []
        if added_items:
            lines.append(f"📋 已自动为你写入 {len(added_items)} 条待办：")
            for it in added_items:
                tags = it.get("tags") or []
                tag_str = (" · " + " ".join(f"#{t}" for t in tags)) if tags else ""
                endtime = it.get("endtime") or ""
                end_str = f" · 截止 {endtime}" if endtime else ""
                pr = {"high": "🔥", "medium": "•", "low": "·"}.get(it.get("priority", "medium"), "•")
                lines.append(f"  {pr} {it['text']}{end_str}{tag_str}")
        if skipped_items:
            lines.append(f"🌀 已为你跳过 {len(skipped_items)} 条重复待办：")
            for sk in skipped_items:
                lines.append(f"  · {sk.get('text','')}  ({sk.get('reason','')})")
        self.add_message("\n".join(lines), is_user=False)
        if self.pet:
            if added_items:
                self.pet.set_happy()
                self.pet.show_bubble(f"已经替你记下 {len(added_items)} 条待办了~ 📋", duration=4000)

    def on_tool_router_error(self, error_msg):
        # 工具路由失败不打扰用户：只打日志，不在聊天里弹错误。
        print(f"[ToolRouter] 失败: {error_msg}")

    def add_message(self, text, is_user=True, image_path=None):
        msg_widget = QWidget()
        msg_widget.setStyleSheet("background: transparent;")
        msg_widget.msg_id = uuid.uuid4().hex[:12]
        msg_widget.audio_path = None
        msg_widget.tts_thread = None
        msg_widget.is_user = is_user
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
        bubble_layout.setSpacing(5)
        
        # 提取大模型返回的 Markdown 图片链接
        remote_image_urls = []
        if not is_user and text:
            pattern = r"!\[.*?\]\((.*?)\)"
            remote_image_urls = re.findall(pattern, text)
            # 从原文本中移除 markdown 语法
            text = re.sub(pattern, "", text).strip()

        # 根据是否有文字或是否是文件决定气泡的边距
        if text or (image_path and not image_path.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp'))):
            bubble_layout.setContentsMargins(10, 10, 10, 10)
        else:
            bubble_layout.setContentsMargins(0, 0, 0, 0)

        # 处理本地发送的图片或文件
        if image_path and os.path.exists(image_path):
            ext = image_path.lower()
            if ext.endswith('.gif'):
                img_label = QLabel()
                movie = QMovie(image_path)
                
                # 读取原图尺寸并按比例缩放，最大宽度200
                pixmap = QPixmap(image_path)
                if not pixmap.isNull():
                    w, h = pixmap.width(), pixmap.height()
                    if w > 200:
                        h = int(h * 200 / w)
                        w = 200
                    movie.setScaledSize(QSize(w, h))
                
                img_label.setMovie(movie)
                movie.start()
                img_label.movie_ref = movie  # 保持引用防止被垃圾回收
                img_label.setStyleSheet("border-radius: 4px;")
                bubble_layout.addWidget(img_label)
            elif ext.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp')):
                img_label = QLabel()
                pixmap = QPixmap(image_path)
                max_img_w = 200
                if pixmap.width() > max_img_w:
                    pixmap = pixmap.scaledToWidth(max_img_w, Qt.SmoothTransformation)
                img_label.setPixmap(pixmap)
                img_label.setStyleSheet("border-radius: 4px;")
                bubble_layout.addWidget(img_label)
            else:
                # 文档类型，显示文件卡片
                file_name = os.path.basename(image_path)
                file_label = QLabel(f"📄 {file_name}")
                file_label.setStyleSheet("""
                    background-color: #F8F8F8;
                    border: 1px solid #D3D3D3;
                    border-radius: 6px;
                    padding: 8px;
                    font-size: 14px;
                    color: #333;
                """)
                bubble_layout.addWidget(file_label)
            
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
            if text or (image_path and not image_path.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp'))):
                bubble_container.setStyleSheet("""
                    background-color: #95EC69;
                    border-radius: 8px;
                """)
            else:
                bubble_container.setStyleSheet("background: transparent;")
            h_layout.addStretch()
            h_layout.addWidget(bubble_container)
            h_layout.addWidget(avatar)
        else:
            # 机器人：靠左，白色气泡
            if text:
                bubble_container.setStyleSheet("""
                    background-color: white;
                    border-radius: 8px;
                """)
            else:
                bubble_container.setStyleSheet("background: transparent;")
            h_layout.addWidget(avatar)
            h_layout.addWidget(bubble_container)

            # 气泡右侧的 🔊 按钮：合成中 → ⏳，就绪 → 🔊（点击播放），失败 → ⚠️（点击重试）。
            # 只对"有文字"的机器人消息挂 TTS；纯表情包/纯文件消息无所谓。
            if text and text.strip():
                speaker_btn = self._build_speaker_button(msg_widget, text)
                msg_widget.speaker_btn = speaker_btn
                h_layout.addWidget(speaker_btn, 0, Qt.AlignTop)

            h_layout.addStretch()

            # 右键删除消息（带音频缓存一起清掉）。仅对机器人气泡开放，
            # 因为用户自己的消息不会生成 wav，删起来也没什么意义。
            msg_widget.setContextMenuPolicy(Qt.CustomContextMenu)
            msg_widget.customContextMenuRequested.connect(
                lambda pos, w=msg_widget: self._show_msg_context_menu(w, pos)
            )

        # 插入到弹簧的前面
        self.msg_layout.insertWidget(self.msg_layout.count() - 1, msg_widget)

        # 机器人新消息一来就立刻在后台合成语音，等用户点 🔊 时直接播放
        if not is_user and text and text.strip():
            self._start_tts_for_widget(msg_widget, text)

        # 自动滚动到底部
        QTimer.singleShot(50, self.scroll_to_bottom)

    # ------- TTS（GPT-SoVITS）相关 -------
    def _build_speaker_button(self, msg_widget, text):
        """按钮初始状态 = 合成中 ⏳。"""
        btn = QPushButton("⏳")
        btn.setFixedSize(26, 26)
        btn.setCursor(Qt.PointingHandCursor)
        btn.setToolTip("正在合成语音…")
        btn.setStyleSheet("""
            QPushButton {
                background-color: transparent;
                border: none;
                font-size: 16px;
                color: #888;
            }
            QPushButton:hover {
                background-color: #EAEAEA;
                border-radius: 13px;
            }
            QPushButton:disabled {
                color: #BBB;
            }
        """)
        btn.setEnabled(False)
        btn.clicked.connect(lambda _=False, w=msg_widget: self._on_speaker_clicked(w))
        return btn

    def _start_tts_for_widget(self, msg_widget, text):
        """异步触发合成；结果回到 _on_tts_finished。"""
        if not hasattr(self, "_tts_threads"):
            self._tts_threads = {}

        prev = msg_widget.tts_thread
        if prev is not None and prev.isRunning():
            # 同一气泡之前有一次还没合成完（例如重试），先丢掉旧线程的回调
            try:
                prev.finished_signal.disconnect()
            except Exception:
                pass

        # 重置按钮到"合成中"
        btn = getattr(msg_widget, "speaker_btn", None)
        if btn is not None:
            btn.setText("⏳")
            btn.setEnabled(False)
            btn.setToolTip("正在合成语音…")

        thread = TTSSynthThread(msg_widget.msg_id, text, parent=self)
        msg_widget.tts_thread = thread
        self._tts_threads[msg_widget.msg_id] = thread
        thread.finished_signal.connect(
            lambda mid, path, w=msg_widget: self._on_tts_finished(w, mid, path)
        )
        thread.start()

    def _on_tts_finished(self, msg_widget, msg_id, wav_path):
        # 气泡可能已经被用户右键删掉了。任何对底层 QObject 的访问都包一层 try。
        try:
            btn = getattr(msg_widget, "speaker_btn", None)
            if wav_path and os.path.exists(wav_path):
                msg_widget.audio_path = wav_path
                if btn is not None:
                    btn.setText("🔊")
                    btn.setEnabled(True)
                    btn.setToolTip("点击播放（再次点击重放）")
            else:
                msg_widget.audio_path = None
                if btn is not None:
                    btn.setText("⚠️")
                    btn.setEnabled(True)
                    btn.setToolTip(
                        "语音合成失败，点击重试。\n请确认 GPT-SoVITS api_v2.py 已启动："
                        "\n  cd voice\\GPT-SoVITS-v2pro-20250604"
                        "\n  runtime\\python.exe api_v2.py -a 127.0.0.1 -p 9880"
                        " -c GPT_SoVITS\\configs\\tts_infer.yaml"
                    )
        except RuntimeError:
            # 底层 C++ widget 已经被 deleteLater 销毁。把这次合成出来的 wav 当孤儿删掉。
            if wav_path and os.path.exists(wav_path):
                try:
                    os.remove(wav_path)
                except OSError:
                    pass

    def _on_speaker_clicked(self, msg_widget):
        path = getattr(msg_widget, "audio_path", None)
        if path and os.path.exists(path):
            play_tts_file(path)
            return
        # 没有可播的 → 视作"重试合成"
        text = self._extract_widget_text(msg_widget)
        if text:
            self._start_tts_for_widget(msg_widget, text)

    def _extract_widget_text(self, msg_widget):
        """重试时需要重新拿到这条气泡的原文。第一个 QLabel.wordWrap=True 即气泡文字。"""
        for child in msg_widget.findChildren(QLabel):
            if child.wordWrap():
                return child.text()
        return ""

    # ------- 删消息（带音频缓存一起清） -------
    def _show_msg_context_menu(self, msg_widget, pos):
        menu = QMenu(self)
        act_del = menu.addAction("🗑 删除这条消息")
        act_replay = None
        if getattr(msg_widget, "audio_path", None):
            act_replay = menu.addAction("🔊 重新播放")
        chosen = menu.exec_(msg_widget.mapToGlobal(pos))
        if chosen is act_del:
            self._delete_message_widget(msg_widget)
        elif act_replay is not None and chosen is act_replay:
            self._on_speaker_clicked(msg_widget)

    def _delete_message_widget(self, msg_widget):
        # 删 wav 缓存
        path = getattr(msg_widget, "audio_path", None)
        if path and os.path.exists(path):
            try:
                os.remove(path)
                print(f"[TTS] 已删除消息及音频缓存：{os.path.basename(path)}")
            except OSError as e:
                print(f"[TTS] 删除音频缓存失败：{e}")
        # 仍在跑的合成线程：让它跑完，但回调里会发现 widget 已没了，自动清理
        thr = getattr(msg_widget, "tts_thread", None)
        if thr is not None:
            try:
                thr.finished_signal.disconnect()
            except Exception:
                pass
            # 改连一个"完成即删 wav"的回调，免得线程跑完了再往 tts_cache 里写一份孤儿
            thr.finished_signal.connect(self._on_tts_orphan_cleanup)
        # 从布局里移除
        self.msg_layout.removeWidget(msg_widget)
        msg_widget.setParent(None)
        msg_widget.deleteLater()

    def _on_tts_orphan_cleanup(self, msg_id, wav_path):
        if wav_path and os.path.exists(wav_path):
            try:
                os.remove(wav_path)
            except OSError:
                pass

    def scroll_to_bottom(self):
        scrollbar = self.scroll_area.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())


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

            tools_json = json.dumps(get_mcp_tool_definitions(), ensure_ascii=False, indent=2)
            existing = [
                {"text": it["text"], "category": it.get("category", "")}
                for it in self.store.items
                if not it.get("completed")
            ][-30:]
            existing_json = json.dumps(existing, ensure_ascii=False, indent=2)
            today_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M (%A)")

            sys_prompt = f"""你是一个 MCP 风格的工具路由器，专门判断"用户的最新输入是否需要调用工具"，并按 A2UI v0.9 协议输出 UI 更新消息。

当前时间: {today_str}

【可用工具(MCP inputSchema 风格)】:
{tools_json}

【当前用户未完成的待办清单(用于去重判断)】:
{existing_json}

【A2UI v0.9 写入约定】:
- catalogId = "{A2UI_CATALOG_ID}"
- surfaceId = "{A2UI_SURFACE_TODO}"
- 每新增一条 todo, 对应一条 updateDataModel 消息, path 固定为 "/items/-", value 为一个 todo 对象。

【触发规则 - 必须严格遵守】:
1. 仅当用户明确表达"要做/要记下/要提醒/要安排/要在某截止前完成"等添加意图时, 才输出 add_todo 调用。
2. 闲聊、问候、提问、感叹、回忆已经发生的事 → 不调用任何工具, 返回空列表。
3. 用户描述的事项若与【当前未完成的待办清单】中**任意一条语义重复**(同义/包含关系/同一件事的不同说法), 不要再次添加。
4. 一次最多 3 条。每条 text 控制在 30 个字以内, 不要复述用户原话, 提炼出待办主干。
5. 如果用户提到"明天/后天/周一/下午三点"等相对时间, 请基于"当前时间"换算成绝对时间放入 due_date(格式 YYYY-MM-DD 或 YYYY-MM-DD HH:MM); 不确定就留空字符串。

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
        "high":   "#FF6B6B",
        "medium": "#FFB347",
        "low":    "#7FCFE8",
    }
    CATEGORY_META = {
        "work":     ("💼 工作", "#5C9DEB"),
        "study":    ("📚 学习", "#A084E8"),
        "life":     ("🏡 生活", "#85D88B"),
        "homework": ("🎓 作业", "#F4A261"),
        "other":    ("📌 其他", "#9BA3AD"),
    }
    SOURCE_BADGE = {
        "agent":    ("🤖 AI", "#7AC4F7"),
        "homework": ("📥 同步", "#F4A261"),
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

        bar_color = self.PRIORITY_COLORS.get(priority, "#FFB347")
        bg = "#F4F5F7" if completed else "#FFFFFF"

        self.setStyleSheet(f"""
            QFrame#todoCard {{
                background-color: {bg};
                border-left: 4px solid {bar_color};
                border-radius: 10px;
            }}
            QFrame#todoCard:hover {{
                background-color: #FFF8EC;
            }}
        """)

        outer = QHBoxLayout(self)
        outer.setContentsMargins(10, 8, 8, 8)
        outer.setSpacing(8)

        # ----- 复选框 -----
        self.checkbox = QCheckBox()
        self.checkbox.setChecked(completed)
        self.checkbox.setStyleSheet(f"""
            QCheckBox {{ background: transparent; spacing: 0; }}
            QCheckBox::indicator {{
                width: 18px; height: 18px;
                border-radius: 9px;
                border: 2px solid {bar_color};
                background: white;
            }}
            QCheckBox::indicator:checked {{
                background-color: #4CAF50;
                border-color: #4CAF50;
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
        text_color = "#9CA3AF" if completed else "#1F2937"
        self.title_label = QLabel(item.get("text", ""))
        self.title_label.setWordWrap(True)
        self.title_label.setStyleSheet(
            f"font-size: 13px; font-weight: bold; color: {text_color}; "
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
            detail_parts.append(f"👨‍🏫 {item['teacher']}")
        if item.get("endtime"):
            detail_parts.append(f"⏰ {item['endtime']}")
        if detail_parts:
            details_label = QLabel("  ·  ".join(detail_parts))
            details_label.setWordWrap(True)
            details_label.setStyleSheet(
                "color: #6B7280; font-size: 11px; background: transparent;"
            )
            text_layout.addWidget(details_label)

        outer.addLayout(text_layout, 1)

        # ----- 删除按钮 -----
        self.del_btn = QPushButton("🗑")
        self.del_btn.setFixedSize(28, 28)
        self.del_btn.setCursor(Qt.PointingHandCursor)
        self.del_btn.setStyleSheet("""
            QPushButton {
                background: transparent;
                border: none;
                font-size: 14px;
                color: #9CA3AF;
            }
            QPushButton:hover {
                background-color: #FFE4E4;
                color: #DC2626;
                border-radius: 14px;
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
            f"background-color: {color}22;"
            f"color: {color};"
            f"border-radius: 8px;"
            f"padding: 1px 6px;"
            f"font-size: 10px;"
            f"font-weight: bold;"
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
        ("homework", "🎓 作业"),
        ("study",    "📚 学习"),
        ("work",     "💼 工作"),
        ("life",     "🏡 生活"),
        ("other",    "📌 其他"),
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
        self.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground)

        # 外圆角容器 + 投影
        container = QWidget(self)
        container.setObjectName("rootContainer")
        container.setGeometry(10, 10, 380, 580)
        container.setStyleSheet("""
            QWidget#rootContainer {
                background-color: #FFFBF1;
                border-radius: 20px;
            }
        """)
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(28)
        shadow.setColor(QColor(0, 0, 0, 70))
        shadow.setOffset(0, 6)
        container.setGraphicsEffect(shadow)

        root = QVBoxLayout(container)
        root.setContentsMargins(16, 14, 16, 14)
        root.setSpacing(10)

        # ============== 顶部标题栏 ==============
        self.header = QWidget()
        self.header.setObjectName("titleBar")
        self.header.setFixedHeight(36)
        self.header.setStyleSheet("QWidget#titleBar { background: transparent; }")
        title_layout = QHBoxLayout(self.header)
        title_layout.setContentsMargins(2, 0, 0, 0)
        title_layout.setSpacing(6)

        title = QLabel("📋 我的待办")
        title.setFont(QFont("Microsoft YaHei", 14, QFont.Bold))
        title.setStyleSheet("color: #E67E22; background: transparent;")
        title_layout.addWidget(title)
        title_layout.addStretch()

        self.sync_btn = QPushButton("🔄 获取作业")
        self.sync_btn.setFixedSize(86, 28)
        self.sync_btn.setCursor(Qt.PointingHandCursor)
        self.sync_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                border-radius: 14px;
                font-size: 12px;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #45a049; }
        """)
        self.sync_btn.clicked.connect(self.sync_homework)
        title_layout.addWidget(self.sync_btn)

        close_btn = QPushButton("✕")
        close_btn.setFixedSize(28, 28)
        close_btn.setCursor(Qt.PointingHandCursor)
        close_btn.setStyleSheet("""
            QPushButton {
                background-color: #FF6B6B;
                color: white;
                border: none;
                border-radius: 14px;
                font-size: 13px;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #EE5A5A; }
        """)
        close_btn.clicked.connect(self.close)
        title_layout.addWidget(close_btn)
        root.addWidget(self.header)

        # ============== 概览卡片（动态统计） ==============
        self.overview_label = QLabel()
        self.overview_label.setWordWrap(True)
        self.overview_label.setStyleSheet("""
            background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                stop:0 #FFE9B6, stop:1 #FFD18A);
            color: #6B4717;
            border-radius: 12px;
            padding: 10px 14px;
            font-size: 12px;
            font-weight: bold;
        """)
        root.addWidget(self.overview_label)

        # ============== 输入行 ==============
        input_layout = QHBoxLayout()
        input_layout.setSpacing(6)
        self.input_field = QLineEdit()
        self.input_field.setPlaceholderText("✨ 添加新的待办事项...")
        self.input_field.setStyleSheet("""
            QLineEdit {
                border: 2px solid #FFB347;
                border-radius: 14px;
                padding: 9px 14px;
                font-size: 13px;
                background-color: white;
                color: #1F2937;
            }
            QLineEdit:focus { border-color: #E67E22; }
        """)
        self.input_field.returnPressed.connect(self.add_todo_from_input)
        input_layout.addWidget(self.input_field, 1)

        self.priority_combo = QComboBox()
        self.priority_combo.addItem("🔥 高", "high")
        self.priority_combo.addItem("• 中", "medium")
        self.priority_combo.addItem("· 低", "low")
        self.priority_combo.setCurrentIndex(1)
        self.priority_combo.setFixedHeight(38)
        self.priority_combo.setStyleSheet("""
            QComboBox {
                background-color: white;
                border: 2px solid #FFD7A0;
                border-radius: 14px;
                padding: 0 8px;
                font-size: 12px;
                min-width: 64px;
            }
            QComboBox:focus { border-color: #E67E22; }
            QComboBox::drop-down { width: 16px; border: none; }
        """)
        input_layout.addWidget(self.priority_combo)

        add_btn = QPushButton("+")
        add_btn.setFixedSize(38, 38)
        add_btn.setCursor(Qt.PointingHandCursor)
        add_btn.setStyleSheet("""
            QPushButton {
                background-color: #FFB347;
                color: white;
                border: none;
                border-radius: 19px;
                font-size: 22px;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #E67E22; }
        """)
        add_btn.clicked.connect(self.add_todo_from_input)
        input_layout.addWidget(add_btn)
        root.addLayout(input_layout)

        # ============== 搜索 + 分类 ==============
        search_row = QHBoxLayout()
        search_row.setSpacing(6)
        self.search_field = QLineEdit()
        self.search_field.setPlaceholderText("🔍 搜索待办 / 标签...")
        self.search_field.setStyleSheet("""
            QLineEdit {
                border: 1px solid #E5E7EB;
                border-radius: 10px;
                padding: 6px 10px;
                font-size: 12px;
                background-color: white;
                color: #374151;
            }
            QLineEdit:focus { border-color: #FFB347; }
        """)
        self.search_field.textChanged.connect(self._on_search_changed)
        search_row.addWidget(self.search_field, 1)

        self.category_combo = QComboBox()
        for key, label in self.CATEGORY_FILTERS:
            self.category_combo.addItem(label, key)
        self.category_combo.setStyleSheet("""
            QComboBox {
                background-color: white;
                border: 1px solid #E5E7EB;
                border-radius: 10px;
                padding: 4px 8px;
                font-size: 12px;
                min-width: 80px;
            }
            QComboBox:focus { border-color: #FFB347; }
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
                color: #9CA3AF;
                border: none;
                font-size: 11px;
            }
            QPushButton:hover { color: #DC2626; text-decoration: underline; }
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
                background: #F1B26B; border-radius: 3px; min-height: 24px;
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
        self.stats_label.setStyleSheet("color: #9CA3AF; font-size: 11px; background: transparent;")
        root.addWidget(self.stats_label)

    # ----- 状态过滤按钮样式 -----
    def _apply_status_styles(self):
        for key, btn in self._status_btns.items():
            active = key == self.current_status
            btn.setChecked(active)
            if active:
                btn.setStyleSheet("""
                    QPushButton {
                        background-color: #FFB347;
                        color: white;
                        border: none;
                        border-radius: 13px;
                        padding: 0 12px;
                        font-size: 11px;
                        font-weight: bold;
                    }
                """)
            else:
                btn.setStyleSheet("""
                    QPushButton {
                        background-color: transparent;
                        color: #6B7280;
                        border: 1px solid #E5E7EB;
                        border-radius: 13px;
                        padding: 0 12px;
                        font-size: 11px;
                    }
                    QPushButton:hover {
                        border-color: #FFB347;
                        color: #E67E22;
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
                "color: #C8B27A; font-size: 12px; padding: 28px 8px; background: transparent;"
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
            return "没有匹配的待办，换个关键词试试 🔍"
        if self.current_status == "done":
            return "目前还没有完成的待办，加油！💪"
        if self.current_status == "today":
            return "今天没有截止的待办，悠着点吧~ 🍵"
        if self.current_status == "pending":
            return "所有待办都完成啦！干得漂亮！🎉"
        return "还没有待办事项哦~\n直接和有珠聊天，她会自动帮你记下来~ ✨"

    def _update_stats(self):
        items = self.store.all()
        total = len(items)
        done = sum(1 for it in items if it.get("completed"))
        pending = total - done
        # 概览大卡片
        if total == 0:
            self.overview_label.setText("📭 还没有待办，有需要直接告诉有珠吧~")
        else:
            ratio = int((done / total) * 100) if total else 0
            self.overview_label.setText(
                f"🌟 共 {total} 条 · 待完成 {pending} · 已完成 {done}  ({ratio}% 进度)"
            )
        self.stats_label.setText(
            f"💪 当前显示 {len(self._filtered_items())} 条 ｜ 共 {total} 条 ｜ 已完成 {done}"
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
        self.sync_btn.setText("🔄 获取作业")
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
        self.sync_btn.setText("🔄 获取作业")
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


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)

    # 1) 先做必填项检查：MySQL 密码 / 火山方舟 API Key 缺一不可。
    #    用户没填的话弹设置窗口；用户取消也允许跳过（桌宠仍会启动，但相应功能不可用）。
    try:
        ensure_required_config_or_prompt(parent=None)
        # 设置窗口里可能改了 MySQL/COS/ARK 等字段，统一刷一次老的全局变量。
        apply_config_to_globals()
    except Exception as _cfg_e:
        print(f"[Settings] 启动期配置检查失败：{_cfg_e}")

    # 2) 启动时清一次 GPT-SoVITS WebUI 留下的 gradio 临时音频；
    #    本地 tts_cache 仅在退出时清（atexit 已注册），中途不影响"重播"。
    try:
        cleanup_tts_artifacts(purge_local_cache=False)
    except Exception as _e:
        print(f"[TTS] 启动清理失败：{_e}")

    pet = DesktopPet()
    pet.show()
    pet.show_bubble("你好呀！双击我打开待办清单~ ")

    sys.exit(app.exec_())
