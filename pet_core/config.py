"""Configuration schema and local JSON-backed settings."""

import json
import os
import threading


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
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_FILE = os.path.join(PROJECT_ROOT, "pet_config.json")

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
        "key": "profile_refiner",
        "title": "用户画像精炼模型（可选）",
        "icon": "🧠",
        "desc": "只在长期记忆新增/手动刷新时调用，用来把原始记忆提炼成可追溯的画像 claim。留空则回退到本地规则聚合。",
        "fields": [
            {"key": "api_key", "label": "API Key", "default": "", "secret": True,
             "hint": "可填 DeepSeek 或其他 OpenAI-compatible 服务的 key"},
            {"key": "base_url", "label": "Base URL", "default": "https://api.deepseek.com"},
            {"key": "model", "label": "模型名", "default": "deepseek-chat",
             "hint": "如果你的账号可用 deepseek-v4，也可以在这里改成对应模型名"},
            {"key": "enabled", "label": "启用", "default": "true"},
        ],
    },
    {
        "key": "memory_reranker",
        "title": "短期记忆重排模型（可选）",
        "icon": "🧭",
        "desc": "用于在 Chroma/中文词面粗召回之后，发一个很小的 DeepSeek 请求重排候选记忆；留空会复用用户画像精炼模型配置。",
        "fields": [
            {"key": "api_key", "label": "API Key", "default": "", "secret": True,
             "hint": "留空则复用用户画像精炼模型的 API Key"},
            {"key": "base_url", "label": "Base URL", "default": "",
             "hint": "留空则复用用户画像精炼模型 Base URL，默认 https://api.deepseek.com"},
            {"key": "model", "label": "模型名", "default": "deepseek-chat",
             "hint": "记忆重排要快，建议使用便宜快速模型；API key 可继续复用画像精炼配置"},
            {"key": "enabled", "label": "启用", "default": "true"},
            {"key": "max_candidates", "label": "候选数量", "default": 12, "type": "int"},
            {"key": "timeout_seconds", "label": "超时秒数", "default": 8, "type": "int"},
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


