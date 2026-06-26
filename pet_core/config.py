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
        "key": "learning_labeler",
        "title": "学习数据打标模型（可选）",
        "icon": "🏷️",
        "desc": "离线给学习样本打弱标签，用于未来训练偏好预测器；留空会复用用户画像精炼模型配置，不会阻塞主聊天。",
        "fields": [
            {"key": "api_key", "label": "API Key", "default": "", "secret": True,
             "hint": "留空则复用用户画像精炼模型的 API Key"},
            {"key": "base_url", "label": "Base URL", "default": "",
             "hint": "留空则复用用户画像精炼模型 Base URL，默认 https://api.deepseek.com"},
            {"key": "model", "label": "模型名", "default": "deepseek-chat",
             "hint": "建议使用 DeepSeek 这类便宜模型；这里只做标签，不生成最终回复"},
            {"key": "enabled", "label": "启用", "default": "true"},
            {"key": "auto_enabled", "label": "后台自动打标", "default": "true",
             "hint": "开启后会延迟处理 pending 样本，人工窗口只用于抽查修正"},
            {"key": "auto_delay_seconds", "label": "自动打标等待秒数", "default": 300, "type": "int",
             "hint": "自动打标调度延迟；实际处理还会受最小观察窗口限制"},
            {"key": "observation_window_seconds", "label": "最小观察窗口秒数", "default": 900, "type": "int",
             "hint": "样本生成后至少等待这段时间，给点赞/踩、主动回复、前台应用变化等隐式反馈进入日志"},
            {"key": "feedback_relabel_delay_seconds", "label": "反馈后重标等待秒数", "default": 180, "type": "int",
             "hint": "收到点赞/踩或隐式反馈后，延迟重标对应样本"},
            {"key": "feedback_settle_seconds", "label": "反馈沉淀秒数", "default": 180, "type": "int",
             "hint": "反馈触发重标前至少等待这段时间，避免只拿到第一条反馈就过早重标"},
            {"key": "timeout_seconds", "label": "超时秒数", "default": 20, "type": "int"},
            {"key": "max_events_per_run", "label": "每次最多处理", "default": 20, "type": "int"},
        ],
    },
    {
        "key": "recommendation_generator",
        "title": "推荐候选生成模型（可选）",
        "icon": "🪄",
        "desc": "当本地策略判断适合推荐但动作池不够贴合时，用便宜模型生成少量具体候选动作，再交给本地推荐器打分和淘汰。",
        "fields": [
            {"key": "api_key", "label": "API Key", "default": "", "secret": True,
             "hint": "留空则复用学习数据打标模型或用户画像精炼模型的 API Key"},
            {"key": "base_url", "label": "Base URL", "default": "",
             "hint": "留空则复用学习数据打标模型 Base URL，默认 https://api.deepseek.com"},
            {"key": "model", "label": "模型名", "default": "deepseek-chat",
             "hint": "建议用 DeepSeek 这类便宜模型；只生成候选动作，不生成最终回复"},
            {"key": "enabled", "label": "启用", "default": "true"},
            {"key": "timeout_seconds", "label": "超时秒数", "default": 8, "type": "int"},
            {"key": "max_candidates", "label": "每次最多候选", "default": 3, "type": "int"},
            {"key": "min_policy_score", "label": "最低策略分", "default": 0.5, "type": "float",
             "hint": "本地策略分低于该值时不调用模型，避免无意义消耗"},
            {"key": "cooldown_minutes", "label": "生成冷却分钟", "default": 30, "type": "int"},
        ],
    },
    {
        "key": "rsshub_local",
        "title": "本地 RSSHub Docker（可选）",
        "icon": "RSS",
        "desc": "在本机 Docker 中运行 RSSHub，桌宠启动时自动确保容器运行。Pixiv/B站等需要登录态的路由请填写对应 Token/Cookie。",
        "fields": [
            {"key": "enabled", "label": "启用本地 RSSHub", "default": "false",
             "hint": "开启后，RSSHub 路由会优先访问本机 http://127.0.0.1:1200。首次启动会自动创建/启动 Docker 容器。"},
            {"key": "auto_start", "label": "启动桌宠时启动容器", "default": "true"},
            {"key": "recreate_on_env_change", "label": "凭证变化时重建容器", "default": "true",
             "hint": "Docker 容器环境变量不能原地修改。开启后，Cookie/Token/额外环境变量变化时会重建配置里指定的 RSSHub 容器。"},
            {"key": "use_as_base_url", "label": "作为 RSSHub 镜像地址", "default": "true",
             "hint": "开启后会把 RSS 推荐里的 RSSHub 镜像地址临时切到本机地址，并把旧公共镜像源切到本地。"},
            {"key": "base_url", "label": "本地 RSSHub 地址", "default": "http://127.0.0.1:1200"},
            {"key": "container_name", "label": "容器名", "default": "rsshub"},
            {"key": "image", "label": "Docker 镜像", "default": "diygod/rsshub"},
            {"key": "host_port", "label": "本地端口", "default": 1200, "type": "int"},
            {"key": "cache_type", "label": "缓存类型", "default": "memory",
             "hint": "默认 memory 足够先跑起来；以后需要 Redis 再扩展。"},
            {"key": "pixiv_refresh_token", "label": "PIXIV_REFRESHTOKEN（可选）", "default": "", "secret": True,
             "hint": "Pixiv 路由通常需要 RSSHub 服务端配置这个 token。不是账号密码。"},
            {"key": "bilibili_cookie", "label": "Bilibili Cookie（可选）", "default": "", "secret": True,
             "hint": "B站 UP 投稿/动态被风控时可填浏览器 Cookie。只保存在本地 pet_config.json。"},
            {"key": "extra_env", "label": "额外 RSSHub 环境变量", "default": "", "type": "text",
             "hint": "一行一个 KEY=VALUE。某些路由需要专用变量时可填这里，例如 UID 专属 Cookie 变量。修改后如容器已存在，需删除旧容器或手动重建才会更新环境变量。"},
        ],
    },
    {
        "key": "rss_recommender",
        "title": "RSS 外部内容推荐",
        "icon": "RSS",
        "desc": "独立于本地行为推荐的外部内容池。先通过 RSSHub 镜像缓存内容，再在用户明确想看内容时推荐。",
        "fields": [
            {"key": "enabled", "label": "启用", "default": "true"},
            {"key": "base_url", "label": "RSSHub 镜像", "default": "https://rsshub.rssforever.com"},
            {"key": "allowed_platforms", "label": "允许平台", "default": "bilibili",
             "hint": "只限制内置默认源；用户在 RSS 管理窗口手动添加的源会进入推荐候选。"},
            {"key": "bilibili_cookie", "label": "Bilibili Cookie（可选）", "default": "", "secret": True,
             "hint": "抓 UP 投稿/动态被 503、412、-799 风控时可填写浏览器里的 B站 Cookie；普通 RSS 不需要。"},
            {"key": "refresh_on_startup", "label": "启动后后台刷新", "default": "true"},
            {"key": "startup_recommend_enabled", "label": "启动后主动推荐", "default": "true",
             "hint": "启动后低频推荐一条外部内容；会遵守每日上限和等待用户回应规则。"},
            {"key": "idle_recommend_enabled", "label": "空闲时主动推荐", "default": "true",
             "hint": "检测到用户空闲且冷却结束时，后台从 RSS 内容池挑一条内容推荐。"},
            {"key": "background_check_minutes", "label": "后台检查间隔分钟", "default": 60, "type": "int"},
            {"key": "active_recommend_daily_limit", "label": "每日主动推荐上限", "default": 2, "type": "int"},
            {"key": "active_recommend_cooldown_hours", "label": "主动推荐冷却小时", "default": 8, "type": "int"},
            {"key": "refresh_interval_minutes", "label": "刷新间隔分钟", "default": 120, "type": "int"},
            {"key": "request_timeout_seconds", "label": "请求超时秒数", "default": 8, "type": "int"},
            {"key": "max_cached_items", "label": "最大缓存条数", "default": 500, "type": "int"},
            {"key": "max_cached_items_per_source", "label": "每个 RSS 源最大缓存条数", "default": 80, "type": "int",
             "hint": "用于定时清理 RSS 条目。Pixiv 榜单等源会继续按源自己的 max_items 收紧。"},
            {"key": "cache_retention_days", "label": "RSS 缓存保留天数", "default": 30, "type": "int",
             "hint": "0 表示不按时间删除，只按总量和每源上限裁剪。"},
            {"key": "cache_cleanup_interval_hours", "label": "RSS 缓存清理间隔小时", "default": 12, "type": "int"},
            {"key": "prompt_candidates", "label": "候选注入条数", "default": 3, "type": "int"},
            {"key": "review_with_llm", "label": "候选交给模型审阅", "default": "true",
             "hint": "开启后会让便宜模型看 Top 候选的标题、简介、链接，选出最贴合的一条并给出理由。"},
            {"key": "reviewer_api_key", "label": "审阅模型 API Key", "default": "", "secret": True,
             "hint": "留空则复用推荐候选生成/学习打标/画像精炼模型配置。"},
            {"key": "reviewer_base_url", "label": "审阅模型 Base URL", "default": "",
             "hint": "留空则复用 DeepSeek 等便宜模型配置。"},
            {"key": "reviewer_model", "label": "审阅模型名", "default": "",
             "hint": "留空则复用 recommendation_generator / learning_labeler / profile_refiner 的模型。"},
            {"key": "reviewer_timeout_seconds", "label": "审阅超时秒数", "default": 10, "type": "int"},
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
            with open(CONFIG_FILE, "r", encoding="utf-8-sig") as f:
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


