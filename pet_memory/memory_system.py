"""Memory, profile, and recall subsystem for the desktop pet app."""

import concurrent.futures
import json
import math
import re
import threading
import time

import pymysql
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from pet_core.config import app_config
from pet_core.conversation_history import ConversationHistory
from pet_services.chroma_mcp import (
    CHROMA_COLLECTION_MEM,
    chrom_distance_to_sim,
    chroma_add_documents_sync,
    chroma_delete_documents_sync,
    chroma_query_documents_sync,
)
from pet_services.knowledge_base import KnowledgeBase

try:
    import jieba
except ImportError:
    jieba = None


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

def configure_memory_database(mysql_cfg=None):
    """Refresh MySQL connection settings used by memory-related functions."""
    global DB_NAME
    if mysql_cfg is None:
        mysql_cfg = app_config.get_section("mysql")
    DB_CONFIG["host"] = mysql_cfg.get("host") or "localhost"
    DB_CONFIG["user"] = mysql_cfg.get("user") or "root"
    DB_CONFIG["password"] = mysql_cfg.get("password") or ""
    DB_CONFIG["charset"] = "utf8mb4"
    try:
        DB_CONFIG["port"] = int(mysql_cfg.get("port") or 3306)
    except (TypeError, ValueError):
        DB_CONFIG["port"] = 3306
    DB_NAME = mysql_cfg.get("database") or "pet_memory_db"
    return DB_CONFIG, DB_NAME


def get_db_name():
    return DB_NAME


# ==================== ????? (KnowledgeBase) ====================
knowledge_base = KnowledgeBase()

# ==================== 智能检索系统 (MemoryRuntime) ====================
# ---- 召回参数（可调） ----
# 旧实现 `final_score = sim*10 + imp*0.1` 的问题：
#   * sim 最高才贡献 10 分；importance 到 600+ 就贡献 60+ 分；
#   * 每次命中 +5、每天才 -1（且必须 24h 没访问）；
#   * 结果就是几条 importance 已经爬到 600 的"明星记忆"每轮对话都霸榜，
#     和当前问题没语义关系也会被反复拉出来 → 冷门记忆永远轮不到 → 饥饿现象。
# 新策略（多路召回 + LLM 小重排）：
#   阶段 1：当前输入 + 最近 2 轮用户上下文一起生成召回 query。
#   阶段 2：Chroma 向量召回和 MySQL 中文词面召回并行进入同一个候选池。
#   阶段 3：用词面承接、向量相似、重要度、时间新鲜度、RRF 排名融合做本地粗排。
#   阶段 4：把前若干候选交给 DeepSeek 小请求重排；DeepSeek 可返回空数组，避免硬塞无关记忆。
#           如果 DeepSeek 未配置、超时或返回异常，则退回本地融合分数。
#   降饱和：命中只 +1（原来 +5），且对单条记忆做 LEAST(..., MEM_IMP_CAP) 硬上限 100。
#   防重复：同一轮会话里刚召回过的 mid，再次计算时打 0.6 折，给冷门记忆出场机会。
MEM_SIM_FLOOR = 0.45              # chrom_distance_to_sim 下，对应 d ≲ 1.22；低于此直接丢
MEM_STRONG_VECTOR_FLOOR = 0.58    # 没有词面命中时，向量至少要达到这个强度才进入候选池
MEM_IMP_CAP = 100.0               # importance_score 硬上限，防止无限刷上去
MEM_TEXT_WEIGHT = 0.38            # 中文词面/上下文承接权重
MEM_SIM_WEIGHT = 0.29             # Chroma 语义相似度权重
MEM_IMP_WEIGHT = 0.15             # 重要性权重（次要）
MEM_RECENCY_WEIGHT = 0.10         # 时间新鲜度权重
MEM_RRF_WEIGHT = 0.08             # BM25-lite/向量排名融合权重
MEM_RECENCY_HALFLIFE_DAYS = 14.0  # 上次访问后，每 14 天 recency 减半
MEM_RECENT_RECALL_PENALTY = 0.6   # 最近被拉过的同 mid 再次计分时乘 0.6
MEM_SESSION_DEDUP_WINDOW = 20     # 维护"最近召回 mid"队列的长度
MEM_RECALL_BUMP = 2               # 召回命中时 importance 增量；只要相关记忆反复出现，就应较快沉淀
MEM_REPEATED_BUMP = 5             # 用户主动复述/强调同一事实，比被动召回更重要
MEM_LEXICAL_POOL_LIMIT = 180      # 中文词面召回最多取多少条 MySQL 候选
MEM_RERANK_DEFAULT_CANDIDATES = 12
MEM_PROMOTE_TO_LONG_TERM_SCORE = 30.0
MEM_MAX_INITIAL_SCORE_RATIO = 0.60
MEM_DEFAULT_INITIAL_SCORE = 10.0
MEM_DURABLE_INITIAL_SCORE = 18.0
MEM_PROFILE_REFRESH_MIN_SECONDS = 60 * 30
USER_PROFILE_CATEGORY_RULES = (
    ("学习习惯", ("作业", "课程", "考试", "复习", "预习", "论文", "死线", "截止", "DDL", "ddl", "拖延", "熬夜", "学习")),
    ("身体毛病", ("胃", "肚子", "头疼", "头痛", "失眠", "睡眠", "生病", "感冒", "发烧", "疼", "痛", "不舒服", "身体")),
    ("饮食偏好", ("吃", "喝", "咖啡", "奶茶", "甜", "辣", "清淡", "外卖", "早餐", "午饭", "晚饭")),
    ("情绪模式", ("焦虑", "压力", "烦", "难过", "开心", "低落", "崩溃", "害怕", "紧张", "emo", "情绪")),
    ("长期兴趣", ("AI", "人工智能", "机器学习", "神经网络", "编程", "Python", "数据库", "桌宠", "游戏", "动画")),
    ("当前目标", ("目标", "计划", "正在", "想做", "要做", "优化", "实现", "项目", "训练", "准备")),
    ("互动偏好", ("回复", "语气", "解释", "详细", "简短", "陪", "建议", "不要", "希望你", "像")),
    ("个人事实", ("生日", "学校", "专业", "名字", "朋友", "家", "住", "课程", "老师")),
    ("稳定偏好", ("喜欢", "讨厌", "偏好", "爱好", "想要", "不喜欢", "希望", "习惯")),
)
USER_PROFILE_KEYS = tuple(label for label, _ in USER_PROFILE_CATEGORY_RULES)
LEGACY_PROFILE_CATEGORY_MAP = {
    "stable_preferences": "稳定偏好",
    "long_term_interests": "长期兴趣",
    "current_goals": "当前目标",
    "interaction_style": "互动偏好",
    "personal_facts": "个人事实",
}
PROFILE_CLAIM_MAX_CHARS = 42
PROFILE_ROLEPLAY_NOISE_WORDS = (
    "角色扮演", "扮演式", "角色扮演式", "互动式倾向", "表达角色", "表现出",
)
PROFILE_CHARACTER_CONTEXT_WORDS = (
    "久远寺有珠", "有珠", "魔法使之夜", "远寺有珠", "kuonji", "alice",
)
PROFILE_CHARACTER_LORE_WORDS = (
    "角色", "生日", "身高", "体重", "魔女", "使魔", "作品", "设定", "型月",
)
PROFILE_USER_SIGNAL_WORDS = (
    "我", "用户", "主人", "喜欢", "讨厌", "希望", "想要", "不想", "习惯", "偏好",
    "经常", "容易", "需要", "记得", "重视",
)


def _profile_bucket_for_memory(text):
    s = str(text or "")
    for bucket, words in USER_PROFILE_CATEGORY_RULES:
        if any(w in s for w in words):
            return bucket
    return "其他长期信息"


def _profile_label(bucket):
    return LEGACY_PROFILE_CATEGORY_MAP.get(bucket, bucket)


def _normalize_profile_category(category, content):
    label = _profile_label(str(category or "").strip())
    if label in USER_PROFILE_KEYS or label == "其他长期信息":
        return label
    return _profile_bucket_for_memory(content)


def _clean_profile_fact(text):
    s = re.sub(r"\s+", " ", str(text or "")).strip()
    s = re.sub(r"^(用户|该用户|主人)[：:，,\s]*", "", s)
    return s[:120]


def _clean_profile_claim_text(text):
    s = _clean_profile_fact(text)
    s = re.sub(r"^(可能|大概|似乎|应该|表现出|表达出|体现出)[，,\s]*", "", s)
    s = s.replace("远寺有珠", "有珠").replace("久远寺有珠", "有珠")
    s = re.sub(r"《?魔法使之夜》?中的?角色", "", s)
    s = re.sub(r"《?魔法使之夜》?角色", "", s)
    s = re.sub(r"《?魔法使之夜》?", "", s)
    s = s.replace("角色扮演式互动倾向", "喜欢有珠式互动")
    s = s.replace("角色扮演互动倾向", "喜欢有珠式互动")
    s = s.replace("互动式倾向", "互动偏好")
    s = re.sub(r"表现出.*?倾向", "", s).strip("，,。 ")
    if "有珠" in s and "喜欢" in s:
        if _has_any(s, ("主动问候", "主动问好", "问候", "问好")):
            return "喜欢有珠主动问候"
        if _has_any(s, ("主动回应", "回应", "互动", "打招呼")):
            return "喜欢有珠主动互动"
        if _has_any(s, ("生日", "纪念日")):
            return "重视有珠相关纪念日"
        return "喜欢有珠相关内容"
    return s[:PROFILE_CLAIM_MAX_CHARS]


def _has_any(text, words):
    return any(w and w.lower() in str(text or "").lower() for w in words)


def _is_character_lore_only(text):
    s = str(text or "")
    if not _has_any(s, PROFILE_CHARACTER_CONTEXT_WORDS):
        return False
    if not _has_any(s, PROFILE_CHARACTER_LORE_WORDS):
        return False
    return not _has_any(s, PROFILE_USER_SIGNAL_WORDS)


def _is_bad_profile_claim(field, claim):
    s = str(claim or "")
    field = str(field or "")
    if not s.strip():
        return True
    if _has_any(s, PROFILE_ROLEPLAY_NOISE_WORDS):
        return True
    if _has_any(s, PROFILE_CHARACTER_CONTEXT_WORDS) and _has_any(s, ("角色扮演", "扮演", "互动式倾向")):
        return True
    if field == "个人事实" and _has_any(s, PROFILE_CHARACTER_CONTEXT_WORDS):
        return True
    if _is_character_lore_only(s):
        return True
    return False


def _normalize_profile_claim(text):
    s = str(text or "").lower()
    s = re.sub(r"^(用户|该用户|主人)[：:，,\s]*", "", s)
    s = re.sub(r"[，。、“”‘’；：:,.!?！？\s]+", "", s)
    return s


def _char_ngrams(text, n=2):
    s = _normalize_profile_claim(text)
    if len(s) <= n:
        return {s} if s else set()
    return {s[i:i + n] for i in range(len(s) - n + 1)}


def _profile_claim_similarity(a, b):
    a_norm = _normalize_profile_claim(a)
    b_norm = _normalize_profile_claim(b)
    if not a_norm or not b_norm:
        return 0.0
    if a_norm == b_norm:
        return 1.0
    if a_norm in b_norm or b_norm in a_norm:
        shorter = min(len(a_norm), len(b_norm))
        longer = max(len(a_norm), len(b_norm))
        return max(0.82, shorter / max(1, longer))
    a_set = _char_ngrams(a_norm)
    b_set = _char_ngrams(b_norm)
    if not a_set or not b_set:
        return 0.0
    return len(a_set & b_set) / len(a_set | b_set)


def _find_similar_profile_claim(cursor, field, claim):
    cursor.execute(
        "SELECT * FROM profile_claim WHERE field_name = %s ORDER BY confidence DESC, updated_at DESC",
        (field,),
    )
    best_row = None
    best_score = 0.0
    for row in cursor.fetchall():
        score = _profile_claim_similarity(row.get("claim"), claim)
        if score > best_score:
            best_score = score
            best_row = row
    if best_row and best_score >= 0.78:
        return best_row
    return None


def _evidence_id_in_claim(row, evidence_id):
    eid = str(evidence_id)
    return eid in {
        part.strip()
        for part in str(row.get("evidence_ids") or "").split(",")
        if part.strip()
    }


def _detach_profile_evidence(cursor, evidence_id, delete_evidence=False):
    """Remove one evidence row's contribution from profile_claim."""
    eid = str(evidence_id)
    cursor.execute(
        "SELECT id, evidence_ids, evidence_count FROM profile_claim "
        "WHERE FIND_IN_SET(%s, evidence_ids)",
        (eid,),
    )
    for claim in cursor.fetchall():
        remaining = [
            x.strip()
            for x in str(claim.get("evidence_ids") or "").split(",")
            if x.strip() and x.strip() != eid
        ]
        if remaining:
            cursor.execute(
                "UPDATE profile_claim SET evidence_ids = %s, "
                "evidence_count = GREATEST(evidence_count - 1, 0), updated_at = NOW() "
                "WHERE id = %s",
                (",".join(remaining), claim.get("id")),
            )
        else:
            cursor.execute("DELETE FROM profile_claim WHERE id = %s", (claim.get("id"),))
    cursor.execute("DELETE FROM profile_revision_log WHERE evidence_id = %s", (eid,))
    if delete_evidence:
        cursor.execute("DELETE FROM profile_evidence WHERE id = %s", (eid,))


def _build_profile_json(buckets):
    profile = {}
    for key in USER_PROFILE_KEYS + ("其他长期信息",):
        seen = set()
        items = []
        for item in buckets.get(key, []):
            fact = _clean_profile_fact(item)
            if not fact or fact in seen:
                continue
            seen.add(fact)
            items.append(fact)
            if len(items) >= 6:
                break
        if items:
            profile[key] = "；".join(items)
    return profile


def _initial_importance_for_memory(text):
    category = _profile_bucket_for_memory(text)
    durable_categories = {"学习习惯", "身体毛病", "饮食偏好", "互动偏好", "个人事实", "稳定偏好"}
    max_initial_score = MEM_PROMOTE_TO_LONG_TERM_SCORE * MEM_MAX_INITIAL_SCORE_RATIO
    if category in durable_categories:
        return min(MEM_DURABLE_INITIAL_SCORE, max_initial_score)
    return min(MEM_DEFAULT_INITIAL_SCORE, max_initial_score)


_RECALL_STOPWORDS = {
    "这个", "那个", "就是", "然后", "但是", "所以", "因为", "如果", "可以", "什么", "一下",
    "感觉", "觉得", "现在", "今天", "昨天", "明天", "用户", "自己", "还是", "已经", "没有",
    "不要", "不是", "有点", "比较", "真的", "的话", "帮我", "你看", "怎么", "为什么",
}


def _normalize_recall_text(text):
    s = str(text or "").lower()
    s = re.sub(r"\s+", "", s)
    s = re.sub(r"[，。、“”‘’；：:,.!?！？（）()【】\[\]{}<>《》\-/\\|~`·…]+", "", s)
    return s


def _is_cjk_char(ch):
    return "\u4e00" <= ch <= "\u9fff"


def _recall_cjk_ngrams(text, n=2):
    s = "".join(ch for ch in _normalize_recall_text(text) if _is_cjk_char(ch))
    if len(s) < n:
        return []
    return [s[i:i + n] for i in range(len(s) - n + 1)]


def _add_recall_term(terms, term, weight):
    term = _normalize_recall_text(term)
    if len(term) < 2 or term in _RECALL_STOPWORDS:
        return
    if len(term) == 2 and all(_is_cjk_char(ch) for ch in term) and term in _RECALL_STOPWORDS:
        return
    terms[term] = max(float(weight), float(terms.get(term, 0.0)))


def _weighted_recall_terms(text, weight=1.0, keywords=None):
    terms = {}
    text = str(text or "")
    for kw in keywords or []:
        _add_recall_term(terms, kw, weight * 1.35)

    if jieba and text.strip():
        try:
            for token in jieba.cut_for_search(text):
                _add_recall_term(terms, token, weight)
        except Exception:
            pass

    for token in re.findall(r"[a-zA-Z0-9_+#.]{2,}", text.lower()):
        _add_recall_term(terms, token, weight)

    # 中文短句里，jieba 对菜名、外号、新词不一定稳定；2/3 gram 能兜住“口水鸡/凉拌菜”这类词面承接。
    for gram in _recall_cjk_ngrams(text, 3):
        _add_recall_term(terms, gram, weight * 0.72)
    for gram in _recall_cjk_ngrams(text, 2):
        _add_recall_term(terms, gram, weight * 0.46)

    if len(terms) > 90:
        ranked = sorted(terms.items(), key=lambda kv: (kv[1], len(kv[0])), reverse=True)
        terms = dict(ranked[:90])
    return terms


def _merge_recall_terms(*term_maps):
    merged = {}
    for term_map in term_maps:
        for term, weight in (term_map or {}).items():
            _add_recall_term(merged, term, weight)
    return merged


def _select_recall_like_terms(weighted_terms, max_terms=18):
    ranked = sorted(
        ((t, w) for t, w in (weighted_terms or {}).items() if len(t) >= 2 and w >= 0.45),
        key=lambda kv: (kv[1], len(kv[0])),
        reverse=True,
    )
    selected = []
    for term, _ in ranked:
        if any(term in old or old in term for old in selected):
            if len(term) <= max((len(old) for old in selected), default=0):
                continue
        selected.append(term)
        if len(selected) >= max_terms:
            break
    return selected


def _recent_user_text_for_memory_recall(max_turns=2, max_age_minutes=45):
    history = globals().get("conversation_history")
    if not history:
        return ""
    try:
        turns = history.get_turns()
    except Exception:
        return ""
    now = datetime.datetime.now()
    parts = []
    for turn in reversed(turns):
        user_text = str(turn.get("user") or "").strip()
        if not user_text:
            continue
        ts = turn.get("timestamp")
        try:
            dt = datetime.datetime.fromisoformat(str(ts)) if ts else None
        except Exception:
            dt = None
        if dt and (now - dt).total_seconds() > max_age_minutes * 60:
            continue
        parts.append(user_text)
        if len(parts) >= max_turns:
            break
    return "\n".join(reversed(parts))


def _memory_lexical_score(weighted_terms, content, keywords=""):
    target_content = _normalize_recall_text(content)
    target_keywords = _normalize_recall_text(keywords)
    if not target_content and not target_keywords:
        return 0.0, []

    score = 0.0
    matched = []
    for term, weight in (weighted_terms or {}).items():
        hit = False
        if term in target_keywords:
            score += weight * (1.25 + min(len(term), 8) * 0.08)
            hit = True
        if term in target_content:
            score += weight * (1.0 + min(len(term), 8) * 0.06)
            hit = True
        if hit and len(matched) < 8:
            matched.append(term)
    return score, matched


def _recall_rrf_score(*ranks, k=60):
    valid = [int(r) for r in ranks if r and int(r) > 0]
    if not valid:
        return 0.0
    raw = sum(1.0 / (k + r) for r in valid)
    best = len(valid) * (1.0 / (k + 1))
    return raw / best if best else 0.0


def _fetch_lexical_memory_rows(cursor, weighted_terms, limit=MEM_LEXICAL_POOL_LIMIT):
    like_terms = _select_recall_like_terms(weighted_terms)
    if not like_terms:
        return []
    conditions = []
    params = []
    for term in like_terms:
        pattern = f"%{term}%"
        conditions.append("(content LIKE %s OR keywords LIKE %s)")
        params.extend([pattern, pattern])
    params.append(int(limit))
    cursor.execute(
        "SELECT id, content, keywords, importance_score, "
        "  GREATEST(0, TIMESTAMPDIFF(SECOND, last_accessed_at, NOW())) AS sec_ago "
        "FROM user_memory WHERE " + " OR ".join(conditions) + " "
        "ORDER BY last_accessed_at DESC, importance_score DESC LIMIT %s",
        params,
    )
    return cursor.fetchall()


def _memory_reranker_config():
    enabled = str(app_config.get("memory_reranker.enabled", "true") or "true").strip().lower()
    if enabled in ("0", "false", "no", "off"):
        return None
    api_key = (
        app_config.get("memory_reranker.api_key", "")
        or app_config.get("profile_refiner.api_key", "")
        or ""
    )
    if not api_key:
        return None
    base_url = (
        app_config.get("memory_reranker.base_url", "")
        or app_config.get("profile_refiner.base_url", "")
        or "https://api.deepseek.com"
    )
    model = (
        app_config.get("memory_reranker.model", "")
        or "deepseek-chat"
    )
    try:
        max_candidates = int(app_config.get("memory_reranker.max_candidates", MEM_RERANK_DEFAULT_CANDIDATES))
    except (TypeError, ValueError):
        max_candidates = MEM_RERANK_DEFAULT_CANDIDATES
    try:
        timeout_seconds = float(app_config.get("memory_reranker.timeout_seconds", 8))
    except (TypeError, ValueError):
        timeout_seconds = 8.0
    return {
        "api_key": api_key,
        "base_url": base_url,
        "model": model,
        "max_candidates": max(4, min(24, max_candidates)),
        "timeout_seconds": max(1.5, min(12.0, timeout_seconds)),
    }


def _call_memory_reranker_llm(query, recent_context, candidates, top_k):
    cfg = _memory_reranker_config()
    if not cfg or not candidates:
        return None
    compact_candidates = []
    for c in candidates[:cfg["max_candidates"]]:
        compact_candidates.append({
            "id": int(c["id"]),
            "memory": str(c.get("content") or "")[:180],
            "local_score": round(float(c.get("final_score") or 0.0), 3),
            "lexical_hits": list(c.get("matched_terms") or [])[:6],
            "vector_sim": round(float(c.get("sim") or 0.0), 3),
        })

    prompt = f"""你是桌宠应用的短期记忆重排器，只判断候选记忆是否应该进入本轮 prompt，不负责生成回复。
请根据【当前用户输入】和【最近用户上下文】，从候选短期记忆中选出最多 {int(top_k)} 条真正有帮助的记忆。

判断优先级：
1. 具体实体、食物、任务、地点、时间、用户刚刚承接的话题优先。
2. 最近上下文可以建立联系，例如“凉拌菜”可以关联上一句“口水鸡/少辣”。
3. 泛泛相似的学习、疲劳、情绪、时间信息不要选，除非它直接解释当前输入。
4. 不要因为 local_score 高就选；它只是粗召回参考。
5. 如果都不相关，keep 返回空数组。

只输出 JSON，不要解释；score 必须是 0 到 1 的数字：
{{"keep":[{{"id":123,"score":0.82,"reason":"不超过18字"}}]}}

【当前用户输入】
{str(query or "")[:500]}

【最近用户上下文】
{str(recent_context or "无")[:500]}

【候选短期记忆 JSON】
{json.dumps(compact_candidates, ensure_ascii=False)}
"""
    try:
        client = OpenAI(
            api_key=cfg["api_key"],
            base_url=cfg["base_url"],
            timeout=cfg["timeout_seconds"],
            max_retries=0,
        )
        response = client.chat.completions.create(
            model=cfg["model"],
            messages=[
                {"role": "system", "content": "你只输出严格 JSON。"},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            max_tokens=500,
        )
        raw = (response.choices[0].message.content or "").strip()
        obj = _extract_json_object(raw)
        if not isinstance(obj, dict):
            return None
        keep = obj.get("keep", [])
        if not isinstance(keep, list):
            return None
        out = []
        for item in keep:
            if not isinstance(item, dict):
                continue
            try:
                mid = int(item.get("id"))
            except (TypeError, ValueError):
                continue
            try:
                score = float(item.get("score", 0.0))
            except (TypeError, ValueError):
                score = 0.0
            out.append({
                "id": mid,
                "score": max(0.0, min(1.0, score)),
                "reason": str(item.get("reason") or "")[:40],
            })
        return out[:int(top_k)]
    except Exception as e:
        print("[Mem][rerank] DeepSeek 重排失败，退回本地融合分数:", e)
        return None


def _queue_chroma_sync(memory_id, operation, content=None, importance_score=0.0, error=None):
    try:
        conn = pymysql.connect(
            host=DB_CONFIG['host'], user=DB_CONFIG['user'],
            password=DB_CONFIG['password'], database=DB_NAME,
            charset=DB_CONFIG['charset']
        )
        try:
            with conn.cursor() as cursor:
                cursor.execute(
                    "INSERT INTO chroma_sync_queue "
                    "(memory_id, operation, content, importance_score, last_error, retry_count, updated_at) "
                    "VALUES (%s, %s, %s, %s, %s, 0, NOW()) "
                    "ON DUPLICATE KEY UPDATE content = VALUES(content), "
                    "importance_score = VALUES(importance_score), last_error = VALUES(last_error), "
                    "retry_count = retry_count + 1, updated_at = NOW()",
                    (int(memory_id), operation, content, float(importance_score or 0.0), str(error or "")[:1000]),
                )
            conn.commit()
        finally:
            conn.close()
    except Exception as e:
        print("[MemorySync] queue Chroma sync failed:", e)


def _clear_chroma_sync_queue(memory_id, operation=None):
    try:
        conn = pymysql.connect(
            host=DB_CONFIG['host'], user=DB_CONFIG['user'],
            password=DB_CONFIG['password'], database=DB_NAME,
            charset=DB_CONFIG['charset']
        )
        try:
            with conn.cursor() as cursor:
                if operation:
                    cursor.execute(
                        "DELETE FROM chroma_sync_queue WHERE memory_id = %s AND operation = %s",
                        (int(memory_id), operation),
                    )
                else:
                    cursor.execute("DELETE FROM chroma_sync_queue WHERE memory_id = %s", (int(memory_id),))
            conn.commit()
        finally:
            conn.close()
    except Exception as e:
        print("[MemorySync] clear Chroma sync queue failed:", e)


def _looks_like_chroma_duplicate_error(err):
    s = str(err or "").lower()
    return any(x in s for x in ("already exists", "duplicate", "duplicated", "existing id"))


def sync_short_memory_to_chroma(memory_id, content, importance_score=0.0, queue_on_fail=True):
    try:
        mid = int(memory_id)
        text = str(content or "").strip()
        if not text:
            return False
        doc_id = f"mem_{mid}"
        metadata = {"mysql_id": mid, "importance_score": float(importance_score or 0.0)}
        try:
            chroma_add_documents_sync(
                CHROMA_COLLECTION_MEM,
                [text],
                [doc_id],
                metadatas=[metadata],
            )
            _clear_chroma_sync_queue(mid, "upsert")
            return True
        except Exception as add_e:
            if not _looks_like_chroma_duplicate_error(add_e):
                if queue_on_fail:
                    _queue_chroma_sync(mid, "upsert", text, importance_score, add_e)
                raise
            try:
                chroma_delete_documents_sync(CHROMA_COLLECTION_MEM, [doc_id])
                chroma_add_documents_sync(
                    CHROMA_COLLECTION_MEM,
                    [text],
                    [doc_id],
                    metadatas=[metadata],
                )
                _clear_chroma_sync_queue(mid, "upsert")
            except Exception as replace_e:
                if queue_on_fail:
                    _queue_chroma_sync(mid, "upsert", text, importance_score, replace_e)
                raise
        return True
    except Exception as e:
        print("[MemorySync] sync short memory to Chroma failed:", e)
        return False


def delete_short_memory_from_chroma(memory_id, queue_on_fail=True):
    try:
        mid = int(memory_id)
        chroma_delete_documents_sync(CHROMA_COLLECTION_MEM, [f"mem_{mid}"])
        _clear_chroma_sync_queue(mid)
        return True
    except Exception as e:
        if queue_on_fail:
            _queue_chroma_sync(memory_id, "delete", None, 0.0, e)
        print("[MemorySync] delete short memory from Chroma failed:", e)
        return False


def flush_chroma_sync_queue(limit=50):
    conn = None
    fixed = 0
    try:
        conn = pymysql.connect(
            host=DB_CONFIG['host'], user=DB_CONFIG['user'],
            password=DB_CONFIG['password'], database=DB_NAME,
            charset=DB_CONFIG['charset'], cursorclass=pymysql.cursors.DictCursor
        )
        with conn.cursor() as cursor:
            cursor.execute(
                "SELECT id, memory_id, operation, content, importance_score FROM chroma_sync_queue "
                "ORDER BY FIELD(operation, 'delete', 'upsert'), updated_at ASC LIMIT %s",
                (int(limit),),
            )
            rows = list(cursor.fetchall())
        for row in rows:
            qid = row.get("id")
            mid = row.get("memory_id")
            operation = row.get("operation")
            ok = False
            if operation == "delete":
                ok = delete_short_memory_from_chroma(mid, queue_on_fail=False)
            else:
                with conn.cursor() as cursor:
                    cursor.execute(
                        "SELECT content, importance_score FROM user_memory WHERE id = %s",
                        (mid,),
                    )
                    mem = cursor.fetchone()
                if mem:
                    ok = sync_short_memory_to_chroma(
                        mid,
                        mem.get("content"),
                        mem.get("importance_score") or row.get("importance_score") or 0.0,
                        queue_on_fail=False,
                    )
                else:
                    ok = delete_short_memory_from_chroma(mid, queue_on_fail=False)
            with conn.cursor() as cursor:
                if ok:
                    cursor.execute("DELETE FROM chroma_sync_queue WHERE id = %s", (qid,))
                    fixed += 1
                else:
                    cursor.execute(
                        "UPDATE chroma_sync_queue SET retry_count = retry_count + 1, "
                        "last_error = %s, updated_at = NOW() WHERE id = %s",
                        ("retry failed", qid),
                    )
            conn.commit()
    except Exception as e:
        print("[MemorySync] flush Chroma sync queue failed:", e)
    finally:
        try:
            if conn:
                conn.close()
        except Exception:
            pass
    return fixed


def schedule_chroma_sync_repair(limit=50):
    threading.Thread(
        target=lambda: flush_chroma_sync_queue(limit=limit),
        name="ChromaSyncRepair",
        daemon=True,
    ).start()


def _profile_refiner_enabled():
    enabled = str(app_config.get("profile_refiner.enabled", "true") or "true").strip().lower()
    return enabled not in ("0", "false", "no", "off") and bool(app_config.get("profile_refiner.api_key", "") or "")


def _clamp_confidence(value):
    try:
        v = float(value)
    except (TypeError, ValueError):
        v = 0.0
    return max(0.0, min(1.0, v))


def _append_evidence_id(existing, evidence_id):
    ids = []
    for part in str(existing or "").split(","):
        part = part.strip()
        if part:
            ids.append(part)
    eid = str(evidence_id)
    if eid not in ids:
        ids.append(eid)
    return ",".join(ids[-30:])


def _extract_json_object(text):
    s = str(text or "").strip()
    try:
        return json.loads(s)
    except Exception:
        pass
    m = re.search(r"\{.*\}", s, re.S)
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except Exception:
        return {}


def _fetch_existing_profile_claims(cursor, limit=40):
    cursor.execute(
        "SELECT id, field_name, claim, confidence, evidence_count "
        "FROM profile_claim ORDER BY confidence DESC, updated_at DESC LIMIT %s",
        (int(limit),),
    )
    return list(cursor.fetchall())


def _call_profile_refiner_llm(evidence_text, category_hint, existing_claims):
    if not _profile_refiner_enabled():
        return {}
    api_key = app_config.get("profile_refiner.api_key", "") or ""
    base_url = app_config.get("profile_refiner.base_url", "") or "https://api.deepseek.com"
    model = app_config.get("profile_refiner.model", "") or "deepseek-chat"
    claims_for_prompt = [
        {
            "id": c.get("id"),
            "field": c.get("field_name"),
            "claim": c.get("claim"),
            "confidence": c.get("confidence"),
        }
        for c in existing_claims[:30]
    ]
    prompt = f"""你是桌宠应用的用户画像精炼器。请根据一条新的长期记忆证据，给出用户画像更新建议。

应用背景：
- 当前桌宠/助手的身份就是“久远寺有珠”。这是真实应用设定，不要把它描述成“角色扮演”。
- 用户提到“有珠、久远寺有珠、魔法使之夜”时，通常是在谈桌宠对象或作品设定；只有明确表达了用户偏好、习惯、目标、身体状态、情绪模式时，才写入画像。
- 作品设定、角色生日、角色资料、助手身份本身，不属于用户画像。

要求：
1. 不要照搬原话，要提炼成稳定、可用于个性化回复的用户画像 claim。
2. 如果证据只是一次性事件、作品设定、角色资料、助手设定或价值不大，operation 用 ignore。
3. 如果支持已有 claim，用 strengthen 或 revise；如果是新特征，用 add；如果和旧 claim 冲突，用 conflict。
4. claim 要短、具体、可执行，最长 22 个汉字；不要写“可能、倾向、表达、表现出、角色扮演”。
5. 不要写“用户喜欢角色扮演/角色扮演式互动”。如果用户说喜欢有珠主动回应，应写“喜欢有珠主动问候”。
6. 只输出 JSON，不要 markdown。

正例：
- 证据“阴雨天容易情绪低落” -> 情绪模式：“阴雨天容易低落”
- 证据“经常死线前赶作业” -> 学习习惯：“常在截止前赶作业”
- 证据“希望有珠主动问好” -> 互动偏好：“喜欢有珠主动问候”

反例：
- “喜欢《魔法使之夜》角色久远寺有珠，表现出角色扮演式互动倾向”是错误画像。
- “记得有珠生日”不是用户个人事实，除非明确说明这会影响用户安排或偏好。

可用字段示例：
学习习惯、身体毛病、饮食偏好、情绪模式、长期兴趣、当前目标、互动偏好、个人事实、稳定偏好、其他长期信息

新证据：
{evidence_text}

本地规则给出的分类提示：
{category_hint or "无"}

已有画像 claim：
{json.dumps(claims_for_prompt, ensure_ascii=False)}

输出格式：
{{
  "updates": [
    {{
      "operation": "add|strengthen|revise|conflict|ignore",
      "field": "学习习惯",
      "target_claim_id": 123,
      "old_claim": "可空",
      "new_claim": "提炼后的画像 claim",
      "confidence_delta": 0.08,
      "reason": "简短原因"
    }}
  ]
}}"""
    try:
        llm = ChatOpenAI(
            model=model,
            openai_api_key=api_key,
            openai_api_base=base_url,
            max_tokens=700,
            temperature=0.1,
        )
        raw = llm.invoke([HumanMessage(content=prompt)]).content.strip()
        return _extract_json_object(raw)
    except Exception as e:
        print("[ProfileRefiner] LLM call failed:", e)
        return {}


def _apply_profile_update(cursor, evidence_id, update, raw_json):
    op = str(update.get("operation") or "ignore").strip().lower()
    if op in ("ignore", "none", "skip"):
        cursor.execute(
            "INSERT INTO profile_revision_log "
            "(evidence_id, operation, field_name, new_claim, confidence_delta, raw_json) "
            "VALUES (%s, %s, %s, %s, %s, %s)",
            (evidence_id, op, update.get("field"), update.get("new_claim"), 0.0, raw_json),
        )
        return

    field = _profile_label(str(update.get("field") or "").strip()) or "其他长期信息"
    if field not in USER_PROFILE_KEYS and field != "其他长期信息":
        field = "其他长期信息"
    new_claim = _clean_profile_claim_text(update.get("new_claim") or update.get("claim") or "")
    if not new_claim or _is_bad_profile_claim(field, new_claim):
        cursor.execute(
            "INSERT INTO profile_revision_log "
            "(evidence_id, operation, field_name, new_claim, confidence_delta, raw_json) "
            "VALUES (%s, 'ignore_bad_claim', %s, %s, 0.0, %s)",
            (evidence_id, field, new_claim, raw_json),
        )
        return
    delta = _clamp_confidence(update.get("confidence_delta", 0.05))
    if op == "conflict":
        delta = min(delta, 0.05)
    target_id = update.get("target_claim_id")
    row = None
    if target_id:
        try:
            cursor.execute("SELECT * FROM profile_claim WHERE id = %s", (int(target_id),))
            row = cursor.fetchone()
        except Exception:
            row = None
    if row is None:
        cursor.execute(
            "SELECT * FROM profile_claim WHERE field_name = %s AND claim = %s LIMIT 1",
            (field, new_claim),
        )
        row = cursor.fetchone()
    if row is None:
        row = _find_similar_profile_claim(cursor, field, new_claim)

    if row:
        old_claim = row.get("claim")
        evidence_ids = _append_evidence_id(row.get("evidence_ids"), evidence_id)
        already_seen = _evidence_id_in_claim(row, evidence_id)
        new_conf = _clamp_confidence(float(row.get("confidence") or 0.0) + delta)
        evidence_increment = 0 if already_seen else 1
        cursor.execute(
            "UPDATE profile_claim SET field_name = %s, claim = %s, confidence = %s, "
            "evidence_count = evidence_count + %s, evidence_ids = %s, updated_at = NOW() "
            "WHERE id = %s",
            (field, new_claim, new_conf, evidence_increment, evidence_ids, row.get("id")),
        )
    else:
        old_claim = update.get("old_claim")
        cursor.execute(
            "INSERT INTO profile_claim "
            "(field_name, claim, confidence, evidence_count, evidence_ids) "
            "VALUES (%s, %s, %s, 1, %s)",
            (field, new_claim, _clamp_confidence(0.55 + delta), str(evidence_id)),
        )

    cursor.execute(
        "INSERT INTO profile_revision_log "
        "(evidence_id, operation, field_name, old_claim, new_claim, confidence_delta, raw_json) "
        "VALUES (%s, %s, %s, %s, %s, %s, %s)",
        (evidence_id, op, field, old_claim, new_claim, delta, raw_json),
    )


def _fallback_profile_claim_from_evidence(cursor, evidence_id, content, category_hint):
    field = category_hint or _profile_bucket_for_memory(content)
    claim = _clean_profile_claim_text(content)
    if not claim or _is_bad_profile_claim(field, claim):
        return
    row = _find_similar_profile_claim(cursor, field, claim)
    if row:
        evidence_ids = _append_evidence_id(row.get("evidence_ids"), evidence_id)
        evidence_increment = 0 if _evidence_id_in_claim(row, evidence_id) else 1
        cursor.execute(
            "UPDATE profile_claim SET confidence = %s, evidence_count = evidence_count + %s, "
            "evidence_ids = %s, updated_at = NOW() WHERE id = %s",
            (
                _clamp_confidence(float(row.get("confidence") or 0.0) + 0.03),
                evidence_increment,
                evidence_ids,
                row.get("id"),
            ),
        )
    else:
        cursor.execute(
            "INSERT INTO profile_claim "
            "(field_name, claim, confidence, evidence_count, evidence_ids) "
            "VALUES (%s, %s, 0.45, 1, %s)",
            (field, claim, str(evidence_id)),
        )
    cursor.execute(
        "INSERT INTO profile_revision_log "
        "(evidence_id, operation, field_name, new_claim, confidence_delta, raw_json) "
        "VALUES (%s, 'fallback_add', %s, %s, 0.0, %s)",
        (evidence_id, field, claim, "{}"),
    )


def _refresh_user_profile_from_claims(cursor, clear_when_empty=False):
    cursor.execute(
        "SELECT id, field_name, claim, confidence, evidence_count "
        "FROM profile_claim WHERE confidence >= 0.35 "
        "ORDER BY field_name, confidence DESC, updated_at DESC"
    )
    rows = list(cursor.fetchall())
    polished_rows = []
    bad_ids = []
    for row in rows:
        field = row.get("field_name")
        original_claim = row.get("claim")
        cleaned_claim = _clean_profile_claim_text(original_claim)
        if _is_bad_profile_claim(field, cleaned_claim):
            bad_ids.append(row.get("id"))
            continue
        if cleaned_claim and cleaned_claim != original_claim:
            cursor.execute(
                "UPDATE profile_claim SET claim = %s, updated_at = NOW() WHERE id = %s",
                (cleaned_claim, row.get("id")),
            )
            row["claim"] = cleaned_claim
        polished_rows.append(row)
    if bad_ids:
        placeholders = ",".join(["%s"] * len(bad_ids))
        cursor.execute(f"DELETE FROM profile_claim WHERE id IN ({placeholders})", bad_ids)
    rows = polished_rows
    if not rows:
        if clear_when_empty:
            cursor.execute("DELETE FROM user_profile WHERE profile_key <> 'profile_summary'")
            cursor.execute(
                "INSERT INTO user_profile "
                "(profile_key, profile_value, confidence, evidence_count, updated_at) "
                "VALUES ('profile_summary', '无', 0.0, 0, NOW()) "
                "ON DUPLICATE KEY UPDATE profile_value = VALUES(profile_value), "
                "confidence = VALUES(confidence), evidence_count = VALUES(evidence_count), updated_at = NOW()"
            )
            return True
        return False
    profile = {}
    evidence_total = 0
    confidence_values = []
    seen_by_field = {}
    for row in rows:
        field = row.get("field_name") or "其他长期信息"
        profile.setdefault(field, [])
        seen_by_field.setdefault(field, set())
        claim_text = row.get("claim")
        claim_key = _normalize_profile_claim(claim_text)
        is_similar_to_existing = any(
            _profile_claim_similarity(existing, claim_text) >= 0.78
            for existing in profile[field]
        )
        if (
            claim_key
            and claim_key not in seen_by_field[field]
            and not is_similar_to_existing
            and len(profile[field]) < 5
        ):
            seen_by_field[field].add(claim_key)
            profile[field].append(claim_text)
        evidence_total += int(row.get("evidence_count") or 0)
        confidence_values.append(float(row.get("confidence") or 0.0))

    profile_obj = {
        field: "；".join(_clean_profile_fact(x) for x in claims if x)
        for field, claims in profile.items()
        if claims
    }
    summary = json.dumps(profile_obj, ensure_ascii=False, indent=2) if profile_obj else "无"
    avg_conf = sum(confidence_values) / len(confidence_values) if confidence_values else 0.0
    cursor.execute(
        "INSERT INTO user_profile "
        "(profile_key, profile_value, confidence, evidence_count, updated_at) "
        "VALUES ('profile_summary', %s, %s, %s, NOW()) "
        "ON DUPLICATE KEY UPDATE profile_value = VALUES(profile_value), "
        "confidence = VALUES(confidence), evidence_count = VALUES(evidence_count), updated_at = NOW()",
        (summary, _clamp_confidence(avg_conf), evidence_total),
    )
    current_fields = list(profile_obj.keys())
    if current_fields:
        placeholders = ",".join(["%s"] * len(current_fields))
        cursor.execute(
            f"DELETE FROM user_profile WHERE profile_key <> 'profile_summary' "
            f"AND profile_key NOT IN ({placeholders})",
            current_fields,
        )
    else:
        cursor.execute("DELETE FROM user_profile WHERE profile_key <> 'profile_summary'")
    for field, claims in profile_obj.items():
        cursor.execute(
            "INSERT INTO user_profile "
            "(profile_key, profile_value, confidence, evidence_count, updated_at) "
            "VALUES (%s, %s, %s, %s, NOW()) "
            "ON DUPLICATE KEY UPDATE profile_value = VALUES(profile_value), "
            "confidence = VALUES(confidence), evidence_count = VALUES(evidence_count), updated_at = NOW()",
            (field, claims, _clamp_confidence(avg_conf), evidence_total),
        )
    return True


def refine_profile_from_evidence(content, source_type="long_term_memory", source_ref=None, category_hint=None):
    if not content:
        return False
    conn = None
    try:
        conn = pymysql.connect(
            host=DB_CONFIG['host'], user=DB_CONFIG['user'],
            password=DB_CONFIG['password'], database=DB_NAME,
            charset=DB_CONFIG['charset'], cursorclass=pymysql.cursors.DictCursor
        )
        with conn.cursor() as cursor:
            cursor.execute(
                "SELECT id FROM profile_evidence WHERE source_type = %s AND source_ref = %s LIMIT 1",
                (source_type, str(source_ref) if source_ref is not None else None),
            )
            existing = cursor.fetchone()
            if existing:
                evidence_id = existing["id"]
                _detach_profile_evidence(cursor, evidence_id)
                cursor.execute(
                    "UPDATE profile_evidence SET content = %s, category_hint = %s, processed_at = NULL "
                    "WHERE id = %s",
                    (content, category_hint, evidence_id),
                )
            else:
                cursor.execute(
                    "INSERT INTO profile_evidence "
                    "(source_type, source_ref, content, category_hint) VALUES (%s, %s, %s, %s)",
                    (source_type, str(source_ref) if source_ref is not None else None, content, category_hint),
                )
                evidence_id = cursor.lastrowid

            if _is_character_lore_only(content):
                cursor.execute(
                    "INSERT INTO profile_revision_log "
                    "(evidence_id, operation, field_name, new_claim, confidence_delta, raw_json) "
                    "VALUES (%s, 'ignore_local_lore', %s, NULL, 0.0, %s)",
                    (evidence_id, category_hint, "{}"),
                )
                cursor.execute("UPDATE profile_evidence SET processed_at = NOW() WHERE id = %s", (evidence_id,))
                _refresh_user_profile_from_claims(cursor, clear_when_empty=True)
                conn.commit()
                return False

            if not _profile_refiner_enabled():
                _refresh_user_profile_from_claims(cursor, clear_when_empty=True)
                conn.commit()
                refresh_user_profile_from_long_term(force=True)
                return False

            existing_claims = _fetch_existing_profile_claims(cursor)
            proposal = _call_profile_refiner_llm(content, category_hint, existing_claims)
            raw_json = json.dumps(proposal, ensure_ascii=False)
            updates = proposal.get("updates") if isinstance(proposal, dict) else None
            if updates:
                for update in updates[:5]:
                    if isinstance(update, dict):
                        _apply_profile_update(cursor, evidence_id, update, raw_json)
            else:
                _fallback_profile_claim_from_evidence(cursor, evidence_id, content, category_hint)
            cursor.execute("UPDATE profile_evidence SET processed_at = NOW() WHERE id = %s", (evidence_id,))
            _refresh_user_profile_from_claims(cursor, clear_when_empty=True)
        conn.commit()
        return True
    except Exception as e:
        print("[ProfileRefiner] evidence refine failed:", e)
        return False
    finally:
        try:
            if conn:
                conn.close()
        except Exception:
            pass


def schedule_profile_refine(content, source_type="long_term_memory", source_ref=None, category_hint=None):
    def _worker():
        refine_profile_from_evidence(
            content,
            source_type=source_type,
            source_ref=source_ref,
            category_hint=category_hint,
        )
    threading.Thread(target=_worker, name="ProfileRefiner", daemon=True).start()


def schedule_refine_unprocessed_long_term_memories(limit=10):
    def _worker():
        conn = None
        try:
            conn = pymysql.connect(
                host=DB_CONFIG['host'], user=DB_CONFIG['user'],
                password=DB_CONFIG['password'], database=DB_NAME,
                charset=DB_CONFIG['charset'], cursorclass=pymysql.cursors.DictCursor
            )
            with conn.cursor() as cursor:
                cursor.execute(
                    "SELECT source_ref FROM profile_evidence "
                    "WHERE source_type = %s AND processed_at IS NOT NULL",
                    ("long_term_memory",),
                )
                processed_refs = {
                    str(r.get("source_ref"))
                    for r in cursor.fetchall()
                    if r.get("source_ref") is not None
                }
                cursor.execute(
                    "SELECT id, content, category FROM long_term_memory "
                    "ORDER BY promoted_at DESC LIMIT %s",
                    (max(int(limit) * 5, int(limit)),),
                )
                rows = []
                for row in cursor.fetchall():
                    if str(row.get("id")) in processed_refs:
                        continue
                    rows.append(row)
                    if len(rows) >= int(limit):
                        break
        except Exception as e:
            print("[ProfileRefiner] load unprocessed long-term memories failed:", e)
            rows = []
        finally:
            try:
                if conn:
                    conn.close()
            except Exception:
                pass
        for row in rows:
            refine_profile_from_evidence(
                row.get("content"),
                source_type="long_term_memory",
                source_ref=row.get("id"),
                category_hint=row.get("category"),
            )
    threading.Thread(target=_worker, name="ProfileRefinerBackfill", daemon=True).start()


def delete_profile_evidence_for_source(source_type, source_ref):
    conn = None
    try:
        conn = pymysql.connect(
            host=DB_CONFIG['host'], user=DB_CONFIG['user'],
            password=DB_CONFIG['password'], database=DB_NAME,
            charset=DB_CONFIG['charset'], cursorclass=pymysql.cursors.DictCursor
        )
        with conn.cursor() as cursor:
            cursor.execute(
                "SELECT id FROM profile_evidence WHERE source_type = %s AND source_ref = %s",
                (source_type, str(source_ref)),
            )
            evidence_ids = [str(r["id"]) for r in cursor.fetchall()]
            for eid in evidence_ids:
                _detach_profile_evidence(cursor, eid, delete_evidence=True)
            _refresh_user_profile_from_claims(cursor, clear_when_empty=True)
        conn.commit()
    except Exception as e:
        print("[ProfileRefiner] delete evidence failed:", e)
    finally:
        try:
            if conn:
                conn.close()
        except Exception:
            pass


def promote_memory_to_long_term(memory_id, reason="score"):
    """Move a high-value working memory into immutable long_term_memory."""
    if not memory_id:
        return False
    try:
        conn = pymysql.connect(
            host=DB_CONFIG['host'], user=DB_CONFIG['user'],
            password=DB_CONFIG['password'], database=DB_NAME,
            charset=DB_CONFIG['charset'], cursorclass=pymysql.cursors.DictCursor
        )
        moved = False
        inserted = False
        long_term_id = None
        promoted_content = None
        promoted_category = None
        with conn.cursor() as cursor:
            cursor.execute(
                "SELECT id, content, keywords, importance_score FROM user_memory WHERE id = %s",
                (memory_id,),
            )
            row = cursor.fetchone()
            if not row:
                return False
            category = _profile_bucket_for_memory(row["content"])
            promoted_content = row["content"]
            promoted_category = category
            cursor.execute(
                "SELECT id FROM long_term_memory "
                "WHERE source_memory_id = %s OR content_hash = SHA2(%s, 256) LIMIT 1",
                (memory_id, row["content"]),
            )
            exists = cursor.fetchone()
            if not exists:
                cursor.execute(
                    "INSERT INTO long_term_memory "
                    "(source_memory_id, content, content_hash, keywords, category, importance_score, promote_reason) "
                    "VALUES (%s, %s, SHA2(%s, 256), %s, %s, %s, %s)",
                    (
                        memory_id, row["content"], row["content"], row.get("keywords"),
                        category, row.get("importance_score") or 0.0, reason,
                    ),
                )
                inserted = True
                long_term_id = cursor.lastrowid
            cursor.execute("DELETE FROM user_memory WHERE id = %s", (memory_id,))
            moved = cursor.rowcount > 0
        conn.commit()
        if moved:
            delete_short_memory_from_chroma(memory_id)
        if inserted:
            schedule_profile_refine(
                promoted_content,
                source_type="long_term_memory",
                source_ref=long_term_id,
                category_hint=promoted_category,
            )
        return moved
    except Exception as e:
        print("[LongTermMemory] promote failed:", e)
        return False
    finally:
        try:
            conn.close()
        except Exception:
            pass


def promote_eligible_memories(limit=50):
    """Promote all short-term memories whose score has crossed the threshold."""
    try:
        conn = pymysql.connect(
            host=DB_CONFIG['host'], user=DB_CONFIG['user'],
            password=DB_CONFIG['password'], database=DB_NAME,
            charset=DB_CONFIG['charset'], cursorclass=pymysql.cursors.DictCursor
        )
        with conn.cursor() as cursor:
            cursor.execute(
                "SELECT id FROM user_memory "
                "WHERE importance_score >= %s "
                "ORDER BY importance_score DESC, last_accessed_at DESC LIMIT %s",
                (MEM_PROMOTE_TO_LONG_TERM_SCORE, int(limit)),
            )
            ids = [r["id"] for r in cursor.fetchall()]
        for mid in ids:
            promote_memory_to_long_term(mid, reason="score_threshold")
    except Exception as e:
        print("[LongTermMemory] batch promote failed:", e)
    finally:
        try:
            conn.close()
        except Exception:
            pass


def refresh_user_profile_from_long_term(force=False):
    """Build a compact user profile from immutable long-term memories.

    Long-term memories are never returned to the prompt verbatim by retrieval;
    only this aggregate profile is injected into the LLM context.
    """
    try:
        conn = pymysql.connect(
            host=DB_CONFIG['host'], user=DB_CONFIG['user'],
            password=DB_CONFIG['password'], database=DB_NAME,
            charset=DB_CONFIG['charset'], cursorclass=pymysql.cursors.DictCursor
        )
        now = time.time()
        with conn.cursor() as cursor:
            if not force:
                cursor.execute(
                    "SELECT UNIX_TIMESTAMP(updated_at) AS ts FROM user_profile "
                    "WHERE profile_key = 'profile_summary' LIMIT 1"
                )
                row = cursor.fetchone()
                if row and row.get("ts") and now - float(row["ts"]) < MEM_PROFILE_REFRESH_MIN_SECONDS:
                    return

            if _refresh_user_profile_from_claims(cursor):
                conn.commit()
                return

            cursor.execute(
                "SELECT content, category, importance_score, promoted_at "
                "FROM long_term_memory ORDER BY importance_score DESC, promoted_at DESC LIMIT 120"
            )
            rows = cursor.fetchall()
            buckets = {k: [] for k in USER_PROFILE_KEYS + ("其他长期信息",)}
            for row in rows:
                content = str(row.get("content") or "").strip()
                if not content:
                    continue
                bucket = _normalize_profile_category(row.get("category"), content)
                if bucket not in buckets:
                    bucket = "其他长期信息"
                if content not in buckets[bucket]:
                    buckets[bucket].append(content)

            profile_obj = _build_profile_json(buckets)
            summary = json.dumps(profile_obj, ensure_ascii=False, indent=2) if profile_obj else "无"

            cursor.execute(
                "INSERT INTO user_profile "
                "(profile_key, profile_value, confidence, evidence_count, updated_at) "
                "VALUES ('profile_summary', %s, %s, %s, NOW()) "
                "ON DUPLICATE KEY UPDATE profile_value = VALUES(profile_value), "
                "confidence = VALUES(confidence), evidence_count = VALUES(evidence_count), "
                "updated_at = NOW()",
                (summary, min(1.0, len(rows) / 40.0), len(rows)),
            )
            for key in USER_PROFILE_KEYS + ("其他长期信息",):
                value = "\n".join(_clean_profile_fact(x) for x in buckets.get(key, [])[:20]) or "无"
                cursor.execute(
                    "INSERT INTO user_profile "
                    "(profile_key, profile_value, confidence, evidence_count, updated_at) "
                    "VALUES (%s, %s, %s, %s, NOW()) "
                    "ON DUPLICATE KEY UPDATE profile_value = VALUES(profile_value), "
                    "confidence = VALUES(confidence), evidence_count = VALUES(evidence_count), "
                    "updated_at = NOW()",
                    (key, value, min(1.0, len(buckets.get(key, [])) / 10.0), len(buckets.get(key, []))),
                )
        conn.commit()
    except Exception as e:
        print("[UserProfile] refresh failed:", e)
    finally:
        try:
            conn.close()
        except Exception:
            pass


def get_user_profile_prompt_context():
    try:
        conn = pymysql.connect(
            host=DB_CONFIG['host'], user=DB_CONFIG['user'],
            password=DB_CONFIG['password'], database=DB_NAME,
            charset=DB_CONFIG['charset'], cursorclass=pymysql.cursors.DictCursor
        )
        with conn.cursor() as cursor:
            cursor.execute(
                "SELECT profile_value, evidence_count, updated_at FROM user_profile "
                "WHERE profile_key = 'profile_summary' LIMIT 1"
            )
            row = cursor.fetchone()
        if row and row.get("profile_value"):
            return (
                f"{row['profile_value']}\n"
                f"画像证据数: {row.get('evidence_count') or 0}; 更新时间: {row.get('updated_at')}"
            )
    except Exception as e:
        print("[UserProfile] read failed:", e)
    finally:
        try:
            conn.close()
        except Exception:
            pass
    return "无"


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
        keywords = [str(k).strip() for k in (keywords or []) if str(k).strip()]

        recent_context = _recent_user_text_for_memory_recall(max_turns=2, max_age_minutes=45)
        current_terms = _weighted_recall_terms(q, weight=1.0, keywords=keywords)
        context_terms = _weighted_recall_terms(recent_context, weight=0.62)
        weighted_terms = _merge_recall_terms(current_terms, context_terms)

        vector_query_parts = [q]
        if keywords:
            vector_query_parts.append(" ".join(keywords))
        if recent_context:
            vector_query_parts.append(recent_context)
        vector_query = "\n".join(part for part in vector_query_parts if part).strip()

        n_pool = max(20, min(80, top_k * 12))
        vector_hits = {}
        try:
            data = chroma_query_documents_sync(CHROMA_COLLECTION_MEM, [vector_query], n_results=n_pool)
            if data:
                docs = (data.get("documents") or [[]])[0]
                metas = (data.get("metadatas") or [[]])[0]
                dists = (data.get("distances") or [[]])[0]
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
                    sim = chrom_distance_to_sim(d)
                    if sim >= MEM_SIM_FLOOR:
                        vector_hits[mid] = {
                            "id": mid,
                            "content": doc,
                            "sim": sim,
                            "vector_rank": i + 1,
                        }
        except Exception as chroma_e:
            print("[Mem][召回] Chroma 候选召回失败，继续使用中文词面召回:", chroma_e)

        try:
            conn = pymysql.connect(
                host=DB_CONFIG['host'], user=DB_CONFIG['user'],
                password=DB_CONFIG['password'], database=DB_NAME,
                charset=DB_CONFIG['charset'], cursorclass=pymysql.cursors.DictCursor
            )
            try:
                with conn.cursor() as cursor:
                    rows_by_id = {}

                    lexical_rows = _fetch_lexical_memory_rows(cursor, weighted_terms)
                    lexical_rank_by_id = {}
                    for rank, row in enumerate(lexical_rows, start=1):
                        try:
                            mid = int(row.get("id"))
                        except (TypeError, ValueError):
                            continue
                        rows_by_id[mid] = row
                        lexical_rank_by_id[mid] = rank

                    vector_ids = [mid for mid in vector_hits.keys() if mid not in rows_by_id]
                    if vector_ids:
                        placeholders = ",".join(["%s"] * len(vector_ids))
                        cursor.execute(
                            f"SELECT id, content, keywords, importance_score, "
                            f"  GREATEST(0, TIMESTAMPDIFF(SECOND, last_accessed_at, NOW())) AS sec_ago "
                            f"FROM user_memory WHERE id IN ({placeholders})",
                            vector_ids,
                        )
                        for row in cursor.fetchall():
                            try:
                                rows_by_id[int(row.get("id"))] = row
                            except (TypeError, ValueError):
                                continue

                    if not rows_by_id:
                        print(f"[Mem][召回] query={query[:30]} candidates=0")
                        return matched_memories

                    candidates = []
                    max_lex_raw = 0.0
                    for mid, row in rows_by_id.items():
                        v = vector_hits.get(mid, {})
                        lex_raw, matched_terms = _memory_lexical_score(
                            weighted_terms,
                            row.get("content", ""),
                            row.get("keywords", ""),
                        )
                        sim = float(v.get("sim") or 0.0)
                        if lex_raw <= 0 and sim < MEM_STRONG_VECTOR_FLOOR:
                            continue
                        max_lex_raw = max(max_lex_raw, lex_raw)
                        candidates.append({
                            "id": mid,
                            "content": row.get("content", ""),
                            "keywords": row.get("keywords", ""),
                            "importance_score": row.get("importance_score", 0.0),
                            "sec_ago": row.get("sec_ago", 0.0),
                            "sim": sim,
                            "vector_rank": v.get("vector_rank"),
                            "lexical_rank": lexical_rank_by_id.get(mid),
                            "lex_raw": lex_raw,
                            "matched_terms": matched_terms,
                        })

                    if not candidates:
                        print(f"[Mem][召回] query={query[:30]} candidates=0 after floor")
                        return matched_memories

                    # ---- 阶段 2：中文词面/向量/重要度/新鲜度/RRF 融合 ----
                    max_lex_raw = max(1.0, max_lex_raw)
                log_cap = math.log1p(MEM_IMP_CAP)
                sim_span = max(1e-6, 1.0 - MEM_SIM_FLOOR)
                for c in candidates:
                    imp = float(c.get("importance_score") or 0.0)
                    lex_norm = max(0.0, min(1.0, float(c.get("lex_raw") or 0.0) / max_lex_raw))
                    sim = float(c.get("sim") or 0.0)
                    sim_norm = max(0.0, min(1.0, (sim - MEM_SIM_FLOOR) / sim_span)) if sim >= MEM_SIM_FLOOR else 0.0
                    # importance 用 log + cap，把"600 vs 10"这种悬殊压成"1.0 vs 0.52"
                    imp_norm = math.log1p(min(imp, MEM_IMP_CAP)) / log_cap if log_cap > 0 else 0.0
                    days_ago = float(c.get("sec_ago") or 0.0) / 86400.0
                    recency = 0.5 ** (days_ago / MEM_RECENCY_HALFLIFE_DAYS)
                    rrf = _recall_rrf_score(c.get("vector_rank"), c.get("lexical_rank"))
                    score = (
                        MEM_TEXT_WEIGHT * lex_norm
                        + MEM_SIM_WEIGHT * sim_norm
                        + MEM_IMP_WEIGHT * imp_norm
                        + MEM_RECENCY_WEIGHT * recency
                        + MEM_RRF_WEIGHT * rrf
                    )
                    # 同一会话里刚召回过 → 降权，给冷门记忆机会
                    if self._is_recently_recalled(c["id"]):
                        score *= MEM_RECENT_RECALL_PENALTY
                    c["final_score"] = score
                    c["_debug"] = (lex_norm, sim, sim_norm, imp_norm, recency, rrf, imp)

                candidates = [c for c in candidates if "final_score" in c]
                candidates.sort(key=lambda x: x["final_score"], reverse=True)

                reranked = _call_memory_reranker_llm(q, recent_context, candidates, top_k)
                if reranked is not None:
                    by_id = {c["id"]: c for c in candidates}
                    top_results = []
                    for item in reranked:
                        row = by_id.get(item["id"])
                        if not row:
                            continue
                        row["rerank_score"] = item.get("score", 0.0)
                        row["rerank_reason"] = item.get("reason", "")
                        top_results.append(row)
                    print(
                        f"[Mem][rerank] DeepSeek keep={len(top_results)}/{min(len(candidates), MEM_RERANK_DEFAULT_CANDIDATES)} "
                        f"ids={[r['id'] for r in top_results]}"
                    )
                else:
                    top_results = candidates[:top_k]
                    print("[Mem][rerank] 未返回有效重排，使用本地融合分数")

                promote_after_commit = []
                with conn.cursor() as update_cursor:
                    for row in top_results:
                        matched_memories.append(row["content"])
                        self._mark_recalled(row["id"])
                        # 命中只 +1，且 LEAST(..., MEM_IMP_CAP) 做硬上限。
                        update_cursor.execute(
                            "UPDATE user_memory "
                            "SET access_count = access_count + 1, "
                            "    importance_score = LEAST(importance_score + %s, %s), "
                            "    last_accessed_at = NOW() "
                            "WHERE id = %s",
                            (MEM_RECALL_BUMP, MEM_IMP_CAP, row["id"]),
                        )
                        raw_imp = float(row.get("importance_score") or 0.0)
                        if min(raw_imp + MEM_RECALL_BUMP, MEM_IMP_CAP) >= MEM_PROMOTE_TO_LONG_TERM_SCORE:
                            promote_after_commit.append(row["id"])

                # 调试：把当次 Top-3 的分数构成打出来，方便观察召回是否由当前话题承接主导。
                if top_results:
                    print("[Mem][召回] query=", query[:30])
                    if recent_context:
                        print("[Mem][召回] recent_context=", recent_context.replace("\n", " / ")[:80])
                    for r in top_results[:3]:
                        lexn, raw_sim, sn, ino, rec, rrf, raw_imp = r.get("_debug", (0, 0, 0, 0, 0, 0, 0))
                        reason = f" rerank={r.get('rerank_score'):.2f}" if "rerank_score" in r else ""
                        print(
                            f"  id={r['id']:>4} final={r['final_score']:.3f}  "
                            f"lex={lexn:.2f} sim={raw_sim:.2f}(norm {sn:.2f})  "
                            f"imp={raw_imp:.0f}(norm {ino:.2f})  "
                            f"recency={rec:.2f} rrf={rrf:.2f}{reason}  -- {str(r['content'])[:32]}"
                        )

                conn.commit()
            finally:
                conn.close()
            for mid in promote_after_commit:
                promote_memory_to_long_term(mid, reason="recall_score")
        except Exception as db_e:
            print("记忆检索异常:", db_e)

        return matched_memories

memory_runtime = MemoryRuntime()

# ==================== ???????? N ?????? ====================
conversation_history = ConversationHistory()


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
                CREATE TABLE IF NOT EXISTS long_term_memory (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    source_memory_id INT NULL,
                    content TEXT NOT NULL,
                    content_hash CHAR(64) NOT NULL,
                    keywords VARCHAR(200),
                    category VARCHAR(64) DEFAULT '其他长期信息',
                    importance_score FLOAT DEFAULT 0.0,
                    promote_reason VARCHAR(64) DEFAULT 'score_threshold',
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    promoted_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE KEY uniq_long_term_hash (content_hash),
                    INDEX idx_long_term_source (source_memory_id),
                    INDEX idx_long_term_category (category),
                    INDEX idx_long_term_promoted (promoted_at)
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS user_profile (
                    profile_key VARCHAR(64) PRIMARY KEY,
                    profile_value TEXT NOT NULL,
                    confidence FLOAT DEFAULT 0.0,
                    evidence_count INT DEFAULT 0,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                        ON UPDATE CURRENT_TIMESTAMP
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS profile_evidence (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    source_type VARCHAR(64) NOT NULL,
                    source_ref VARCHAR(128),
                    content TEXT NOT NULL,
                    category_hint VARCHAR(64),
                    privacy_policy VARCHAR(64) DEFAULT 'profile_only',
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    processed_at DATETIME NULL,
                    INDEX idx_profile_evidence_source (source_type, source_ref),
                    INDEX idx_profile_evidence_processed (processed_at)
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS profile_claim (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    field_name VARCHAR(64) NOT NULL,
                    claim TEXT NOT NULL,
                    confidence FLOAT DEFAULT 0.5,
                    evidence_count INT DEFAULT 0,
                    evidence_ids TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                        ON UPDATE CURRENT_TIMESTAMP,
                    INDEX idx_profile_claim_field (field_name),
                    INDEX idx_profile_claim_confidence (confidence)
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS profile_revision_log (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    evidence_id INT,
                    operation VARCHAR(64),
                    field_name VARCHAR(64),
                    old_claim TEXT,
                    new_claim TEXT,
                    confidence_delta FLOAT DEFAULT 0.0,
                    raw_json TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    INDEX idx_profile_revision_evidence (evidence_id),
                    INDEX idx_profile_revision_field (field_name)
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS chroma_sync_queue (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    memory_id INT NOT NULL,
                    operation VARCHAR(16) NOT NULL,
                    content TEXT,
                    importance_score FLOAT DEFAULT 0.0,
                    retry_count INT DEFAULT 0,
                    last_error TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                        ON UPDATE CURRENT_TIMESTAMP,
                    UNIQUE KEY uniq_chroma_sync (memory_id, operation),
                    INDEX idx_chroma_sync_updated (updated_at)
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
        promote_eligible_memories()
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
                "    CASE WHEN importance_score >= 25 THEN 1 ELSE 0.5 END "
                "WHERE DATEDIFF(NOW(), last_accessed_at) >= 1"
            )
            # 3) 收尸
            cursor.execute("SELECT id FROM user_memory WHERE importance_score <= 0")
            dead_ids = [row[0] for row in cursor.fetchall()]
            cursor.execute("DELETE FROM user_memory WHERE importance_score <= 0")
        conn.commit()
        conn.close()
        if dead_ids:
            for dead_id in dead_ids:
                delete_short_memory_from_chroma(dead_id)
    except Exception as e:
        print("记忆衰减执行失败:", e)
