"""Lightweight dynamic recommender for preference-data bootstrapping.

This is intentionally not a fixed menu of actions. Concrete actions enter the
pool from user manual labels, accepted/rejected recommendations, profile hints,
and later LLM-generated candidates. The local side owns normalization, scoring,
and lifecycle pruning so the action space can converge instead of exploding.
"""

import datetime
import hashlib
import json
import os
import re

from openai import OpenAI

from pet_core.config import app_config
from pet_core.learning_logger import LABELED_INTERACTIONS_PATH, LEARNING_DATA_DIR


ACTION_SPACE_PATH = os.path.join(LEARNING_DATA_DIR, "recommendation_actions.json")
RECOMMENDATION_EVENTS_PATH = os.path.join(LEARNING_DATA_DIR, "recommendation_events.jsonl")

MAX_ACTIONS = 20
NEW_ACTION_PROTECTION_DAYS = 3
DEFAULT_TOP_K = 3
MIN_POLICY_SAMPLES = 2


def _now():
    return datetime.datetime.now()


def _iso(dt=None):
    return (dt or _now()).isoformat(timespec="seconds")


def _parse_ts(value):
    try:
        if not value:
            return None
        return datetime.datetime.fromisoformat(str(value))
    except Exception:
        return None


def _days_since(value, default=999.0):
    dt = _parse_ts(value)
    if not dt:
        return default
    return max(0.0, (_now() - dt).total_seconds() / 86400.0)


def _safe_text(value, limit=400):
    text = str(value or "").strip()
    if len(text) > limit:
        return text[:limit] + "...[truncated]"
    return text


def _truthy(value, default=True):
    if value is None or value == "":
        return default
    return str(value).strip().lower() not in ("0", "false", "no", "off")


def _int_config(key, default, min_value=0, max_value=None):
    try:
        value = int(app_config.get(key, default))
    except Exception:
        value = int(default)
    value = max(int(min_value), value)
    if max_value is not None:
        value = min(int(max_value), value)
    return value


def _float_config(key, default, min_value=0.0, max_value=None):
    try:
        value = float(app_config.get(key, default))
    except Exception:
        value = float(default)
    value = max(float(min_value), value)
    if max_value is not None:
        value = min(float(max_value), value)
    return value


def normalize_action_name(text):
    """Return a compact canonical action name without forcing a fixed taxonomy."""
    s = str(text or "").strip()
    s = re.sub(r"[\s　]+", "", s)
    s = re.sub(r"[。！？!?,，；;：:\"'“”‘’（）()\[\]【】<>《》]", "", s)
    s = s.replace("一下", "").replace("一会儿", "一会").replace("一小会儿", "一小会")
    return s[:24]


def _action_id(name):
    canonical = normalize_action_name(name)
    digest = hashlib.sha1(canonical.encode("utf-8")).hexdigest()[:12]
    return f"act_{digest}"


def _append_jsonl(path, record):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _iter_jsonl(path):
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if isinstance(obj, dict):
                yield obj


def _extract_json_object(text):
    raw = str(text or "").strip()
    if not raw:
        return {}
    try:
        obj = json.loads(raw)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        pass
    start = raw.find("{")
    end = raw.rfind("}")
    if start >= 0 and end > start:
        try:
            obj = json.loads(raw[start:end + 1])
            return obj if isinstance(obj, dict) else {}
        except Exception:
            return {}
    return {}


def _generator_config():
    enabled = _truthy(app_config.get("recommendation_generator.enabled", "true"), default=True)
    api_key = (
        app_config.get("recommendation_generator.api_key", "")
        or app_config.get("learning_labeler.api_key", "")
        or app_config.get("profile_refiner.api_key", "")
        or app_config.get("memory_reranker.api_key", "")
        or ""
    )
    base_url = (
        app_config.get("recommendation_generator.base_url", "")
        or app_config.get("learning_labeler.base_url", "")
        or app_config.get("profile_refiner.base_url", "")
        or "https://api.deepseek.com"
    )
    model = (
        app_config.get("recommendation_generator.model", "")
        or app_config.get("learning_labeler.model", "")
        or app_config.get("profile_refiner.model", "")
        or "deepseek-chat"
    )
    return {
        "enabled": enabled,
        "api_key": api_key,
        "base_url": base_url,
        "model": model,
        "timeout_seconds": _int_config("recommendation_generator.timeout_seconds", 8, min_value=3, max_value=30),
        "max_candidates": _int_config("recommendation_generator.max_candidates", 3, min_value=1, max_value=5),
        "min_policy_score": _float_config("recommendation_generator.min_policy_score", 0.5, min_value=0.0, max_value=1.0),
        "cooldown_minutes": _int_config("recommendation_generator.cooldown_minutes", 30, min_value=1, max_value=1440),
    }


def _valid_user_action(text):
    name = normalize_action_name(text)
    if len(name) < 2 or len(name) > 24:
        return ""
    # 推荐动作必须是用户自己能做的现实动作，不能让桌宠假装现实执行。
    blocked = ("有珠", "使魔", "我帮", "帮你", "替你", "守着", "碰", "递", "泡", "做饭", "监督你")
    if any(word in name for word in blocked):
        return ""
    return name


def _load_state():
    if not os.path.exists(ACTION_SPACE_PATH):
        return {"schema_version": "dynamic_recommender.v1", "actions": {}, "scene_stats": {}}
    try:
        with open(ACTION_SPACE_PATH, "r", encoding="utf-8") as f:
            state = json.load(f)
        if not isinstance(state, dict):
            raise ValueError("state is not dict")
        state.setdefault("schema_version", "dynamic_recommender.v1")
        state.setdefault("actions", {})
        state.setdefault("scene_stats", {})
        return state
    except Exception as e:
        print(f"[Recommender] 动作池读取失败，重建空状态: {e}")
        return {"schema_version": "dynamic_recommender.v1", "actions": {}, "scene_stats": {}}


def _save_state(state):
    os.makedirs(os.path.dirname(ACTION_SPACE_PATH), exist_ok=True)
    tmp = ACTION_SPACE_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)
    os.replace(tmp, ACTION_SPACE_PATH)


def _beta_mean(positive, negative):
    return (float(positive or 0) + 1.0) / (float(positive or 0) + float(negative or 0) + 2.0)


def scene_key_from_event(event, labels=None):
    labels = labels or {}
    state = labels.get("state") or {}
    need = labels.get("need") or {}
    time_features = event.get("time_features") or {}
    trigger = (event.get("trigger") or {}).get("type") or "unknown"
    parts = [
        str(state.get("task_state") or "未知"),
        str(state.get("emotion") or "未知"),
        str(need.get("primary") or "未知"),
        str(time_features.get("daypart") or "unknown"),
        str(trigger),
    ]
    return "|".join(parts)


def _coarse_scene_key(scene_key):
    parts = str(scene_key or "").split("|")
    if len(parts) < 3:
        return str(scene_key or "")
    return "|".join(parts[:3])


def scene_key_from_runtime(user_text="", time_features=None, trigger_type="user_message", strategy_prediction=None):
    """A small, replaceable scene sketch until the future local model exists."""
    pred = strategy_prediction if isinstance(strategy_prediction, dict) else {}
    emotion = str(pred.get("emotion") or "").strip() or "未知"
    task_state = str(pred.get("task_state") or "").strip() or "未知"
    need = str(pred.get("need") or pred.get("intent") or "").strip() or "未知"
    text = str(user_text or "")
    if any(w in text for w in ("累", "困", "疲惫", "没精神")):
        if emotion == "未知":
            emotion = "疲惫"
        if need == "未知":
            need = "安慰"
    if any(w in text for w in ("焦虑", "压力", "慌", "来不及")):
        if emotion == "未知":
            emotion = "焦虑"
        if need == "未知":
            need = "拆任务"
    if any(w in text for w in ("不想", "拖延", "背不动", "学不动")):
        if task_state == "未知":
            task_state = "拖延中"
        if need == "未知":
            need = "督促"
    if any(w in text for w in ("学习", "复习", "作业", "背单词", "考试")):
        task_state = "学习中" if task_state == "未知" else task_state
    if any(w in text for w in ("代码", "bug", "报错", "编程")):
        task_state = "写代码中"
    daypart = (time_features or {}).get("daypart") or _daypart(_now().hour)
    return "|".join([task_state, emotion, need, daypart, trigger_type])


def _manual_score_signal(event, labels):
    meta = event.get("manual_label_meta") or {}
    scores = meta.get("scores") or {}
    try:
        rec_score = int(scores.get("recommendation_score") or 0)
    except Exception:
        rec_score = 0
    try:
        weight = max(1, int(scores.get("sample_weight") or meta.get("sample_weight") or 3))
    except Exception:
        weight = 3
    if rec_score > 0:
        return True, weight
    if rec_score < 0:
        return False, weight

    rec = (labels or {}).get("recommendation_type") or {}
    if bool(rec.get("should_recommend")):
        return True, max(1, weight - 1)
    return False, max(1, weight - 1)


def _teacher_score_signal(event, labels):
    rec = (labels or {}).get("recommendation_type") or {}
    should = bool(rec.get("should_recommend"))
    feedback = ((labels or {}).get("feedback_strategy") or {})
    try:
        confidence = float(feedback.get("confidence") or 0.35)
    except Exception:
        confidence = 0.35
    weight = max(0.25, min(1.0, confidence))
    return should, weight


def _load_recommendation_policy_stats():
    """Build a tiny learned policy from DS/manual labels.

    This is the current local strategy predictor: it learns whether a scene
    tends to accept concrete recommendations. A future MLP/LightGCN can replace
    this function without changing the runtime prompt contract.
    """
    stats = {
        "global": {"positive": 0.0, "negative": 0.0},
        "scene": {},
        "coarse_scene": {},
    }
    for event in _iter_jsonl(LABELED_INTERACTIONS_PATH) or []:
        labels = ((event.get("prompt_decision") or {}).get("teacher_labels") or {})
        if not labels:
            continue
        scene_key = scene_key_from_event(event, labels)
        coarse_key = _coarse_scene_key(scene_key)
        if event.get("label_status") == "manual_labeled" or event.get("manual_label_meta"):
            should, weight = _manual_score_signal(event, labels)
        else:
            should, weight = _teacher_score_signal(event, labels)

        target = "positive" if should else "negative"
        stats["global"][target] += weight
        scene_bucket = stats["scene"].setdefault(scene_key, {"positive": 0.0, "negative": 0.0})
        scene_bucket[target] += weight
        coarse_bucket = stats["coarse_scene"].setdefault(coarse_key, {"positive": 0.0, "negative": 0.0})
        coarse_bucket[target] += weight
    return stats


def _bucket_count(bucket):
    return float(bucket.get("positive") or 0.0) + float(bucket.get("negative") or 0.0)


def _policy_bucket_score(bucket):
    return _beta_mean(bucket.get("positive"), bucket.get("negative"))


def recommendation_policy_score(scene_key):
    stats = _load_recommendation_policy_stats()
    scene_bucket = (stats.get("scene") or {}).get(scene_key) or {}
    coarse_bucket = (stats.get("coarse_scene") or {}).get(_coarse_scene_key(scene_key)) or {}
    global_bucket = stats.get("global") or {}

    scene_count = _bucket_count(scene_bucket)
    coarse_count = _bucket_count(coarse_bucket)
    global_count = _bucket_count(global_bucket)

    if scene_count >= MIN_POLICY_SAMPLES:
        score = 0.70 * _policy_bucket_score(scene_bucket) + 0.30 * _policy_bucket_score(global_bucket)
        source = "scene_label_stats"
        count = scene_count
    elif coarse_count >= MIN_POLICY_SAMPLES:
        score = 0.66 * _policy_bucket_score(coarse_bucket) + 0.34 * _policy_bucket_score(global_bucket)
        source = "coarse_scene_label_stats"
        count = coarse_count
    elif global_count >= MIN_POLICY_SAMPLES:
        score = _policy_bucket_score(global_bucket)
        source = "global_label_stats"
        count = global_count
    else:
        score = 0.42
        source = "cold_start_conservative"
        count = global_count

    return {
        "score": round(max(0.0, min(1.0, score)), 3),
        "source": source,
        "scene_count": round(scene_count, 2),
        "coarse_count": round(coarse_count, 2),
        "global_count": round(global_count, 2),
    }


def _daypart(hour):
    if 5 <= hour < 11:
        return "morning"
    if 11 <= hour < 14:
        return "noon"
    if 14 <= hour < 18:
        return "afternoon"
    if 18 <= hour < 23:
        return "evening"
    return "night"


def _action_template_from_preference(raw):
    """Turn a profile preference phrase into an action candidate.

    This is not the final generator; it is only a cheap bootstrap so an empty
    action pool can start receiving feedback without a fixed global menu.
    """
    phrase = normalize_action_name(raw)
    if not phrase:
        return ""
    if phrase.endswith("歌") or "陈奕迅" in phrase or "音乐" in phrase:
        return f"听一首{phrase.replace('喜欢的歌手是', '')}"
    return f"{phrase}一小会"


def profile_candidate_actions(profile_text, limit=6):
    text = str(profile_text or "")
    candidates = []
    patterns = [
        r"喜欢([^；;，,\n]+)",
        r"习惯([^；;，,\n]+)",
        r"偏好([^；;，,\n]+)",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, text):
            raw = match.group(1).strip()
            raw = re.sub(r"^(的|是|为|：|:)", "", raw)
            if not raw or len(raw) > 18:
                continue
            action = _action_template_from_preference(raw)
            if action and action not in candidates:
                candidates.append(action)
            if len(candidates) >= limit:
                return candidates
    return candidates


class DynamicRecommender:
    def __init__(self, max_actions=MAX_ACTIONS):
        self.max_actions = int(max_actions or MAX_ACTIONS)

    def _scene_action_count(self, state, scene_key):
        active_ids = {
            aid for aid, action in (state.get("actions") or {}).items()
            if action.get("status") == "active"
        }
        scene_stats = (state.get("scene_stats") or {}).get(scene_key, {})
        return sum(1 for aid in scene_stats if aid in active_ids)

    def _candidate_generation_allowed(self, state, scene_key, policy, actions, top_k):
        cfg = _generator_config()
        if not cfg["enabled"] or not cfg["api_key"]:
            return False, "generator_disabled_or_no_key", cfg
        policy_score = float(policy.get("score") or 0.0)
        if policy_score < cfg["min_policy_score"]:
            return False, "policy_score_too_low", cfg
        last = state.get("last_candidate_generation") or {}
        cooldown_days = cfg["cooldown_minutes"] / 1440.0
        if _days_since(last.get("timestamp"), default=999) < cooldown_days:
            return False, "candidate_generation_cooldown", cfg
        scene_action_count = self._scene_action_count(state, scene_key)
        if scene_action_count < max(2, int(top_k or DEFAULT_TOP_K)):
            return True, "scene_action_pool_sparse", cfg
        if len(actions) < min(6, self.max_actions):
            return True, "global_action_pool_sparse", cfg
        return False, "action_pool_enough", cfg

    def _generate_action_candidates(self, *, scene_key, user_text, user_profile, recent_context,
                                    matched_memories, existing_actions, policy, cfg):
        existing_names = [
            str(action.get("name") or "")
            for action in list(existing_actions or [])[: self.max_actions]
            if str(action.get("name") or "").strip()
        ]
        context_preview = _safe_text(str(recent_context or ""), 900)
        memory_preview = _safe_text("；".join(str(x) for x in list(matched_memories or [])[:4]), 700)
        profile_preview = _safe_text(user_profile, 1200)
        prompt = f"""你是桌宠推荐系统的候选动作生成器。你的任务不是回复用户，而是为本地推荐器生成少量“用户自己可以执行”的具体候选动作。

硬性规则：
1. 只生成用户本人能立刻或短时间内执行的现实动作。
2. 不能生成“有珠/使魔替用户做事、守着用户、触碰用户、递东西、泡饮料、做饭、现实监督用户”之类动作。
3. 动作要具体、低压力、短小，通常 4 到 12 个汉字，例如“画画十分钟”“闭眼休息五分钟”“背十个单词”。
4. 优先复用已有动作池里的近义动作；只有确实不同才生成新动作。
5. 不要生成空泛关心、纯聊天、说教、长计划。

当前场景：{scene_key}
本地推荐策略：{json.dumps(policy, ensure_ascii=False)}
用户当前输入：{_safe_text(user_text, 600)}
用户画像摘要：{profile_preview}
近期上下文摘要：{context_preview}
相关短期记忆：{memory_preview}
已有动作池：{json.dumps(existing_names, ensure_ascii=False)}

只输出严格 JSON：
{{
  "actions": [
    {{"name": "候选动作", "reason": "为什么适合当前场景", "reuse_existing": false}}
  ]
}}
"""
        client = OpenAI(
            api_key=cfg["api_key"],
            base_url=cfg["base_url"],
            timeout=cfg["timeout_seconds"],
            max_retries=0,
        )
        response = client.chat.completions.create(
            model=cfg["model"],
            messages=[
                {"role": "system", "content": "你只输出严格 JSON，不要解释。"},
                {"role": "user", "content": prompt},
            ],
            temperature=0.25,
            max_tokens=500,
        )
        raw = (response.choices[0].message.content or "").strip()
        obj = _extract_json_object(raw)
        actions = []
        seen = set()
        for item in list(obj.get("actions") or [])[: cfg["max_candidates"]]:
            if isinstance(item, str):
                name = item
                reason = "llm_candidate"
                reuse_existing = False
            elif isinstance(item, dict):
                name = item.get("name") or item.get("action") or ""
                reason = item.get("reason") or "llm_candidate"
                reuse_existing = bool(item.get("reuse_existing", False))
            else:
                continue
            name = _valid_user_action(name)
            if not name:
                continue
            key = normalize_action_name(name)
            if key in seen:
                continue
            seen.add(key)
            actions.append({
                "name": name,
                "reason": _safe_text(reason, 220),
                "reuse_existing": reuse_existing,
            })
        return actions, _safe_text(raw, 1200)

    def upsert_action(self, action_text, *, source="unknown", evidence="", scene_key=""):
        name = normalize_action_name(action_text)
        if not name:
            return None
        state = _load_state()
        actions = state.setdefault("actions", {})
        aid = _action_id(name)
        now = _iso()
        action = actions.get(aid)
        if not action:
            action = {
                "id": aid,
                "name": name,
                "canonical_name": name,
                "created_at": now,
                "last_seen_at": now,
                "last_recommended_at": "",
                "last_accepted_at": "",
                "last_rejected_at": "",
                "source": source,
                "evidence": _safe_text(evidence, 600),
                "exposures": 0,
                "positive": 0,
                "negative": 0,
                "manual_positive": 0,
                "manual_negative": 0,
                "status": "active",
                "protected": False,
            }
            actions[aid] = action
        else:
            action["last_seen_at"] = now
            if source and action.get("source") != "user_manual":
                action["source"] = source
            if evidence:
                action["evidence"] = _safe_text(evidence, 600)

        if scene_key:
            scene_stats = state.setdefault("scene_stats", {}).setdefault(scene_key, {})
            scene_stats.setdefault(aid, {"exposures": 0, "positive": 0, "negative": 0})
        self._prune_if_needed(state)
        _save_state(state)
        return action

    def record_feedback(self, *, action_text, scene_key, score=0, source="manual", event_id=""):
        action = self.upsert_action(action_text, source=source, evidence=f"event_id={event_id}", scene_key=scene_key)
        if not action:
            return None
        state = _load_state()
        actions = state.setdefault("actions", {})
        action = actions.get(action["id"])
        if not action:
            return None

        score = int(score or 0)
        action["exposures"] = int(action.get("exposures") or 0) + 1
        if score > 0:
            action["positive"] = int(action.get("positive") or 0) + 1
            action["manual_positive"] = int(action.get("manual_positive") or 0) + 1
            action["last_accepted_at"] = _iso()
        elif score < 0:
            action["negative"] = int(action.get("negative") or 0) + 1
            action["manual_negative"] = int(action.get("manual_negative") or 0) + 1
            action["last_rejected_at"] = _iso()

        scene_stats = state.setdefault("scene_stats", {}).setdefault(scene_key, {})
        s = scene_stats.setdefault(action["id"], {"exposures": 0, "positive": 0, "negative": 0})
        s["exposures"] = int(s.get("exposures") or 0) + 1
        if score > 0:
            s["positive"] = int(s.get("positive") or 0) + 1
        elif score < 0:
            s["negative"] = int(s.get("negative") or 0) + 1

        self._prune_if_needed(state)
        _save_state(state)
        _append_jsonl(RECOMMENDATION_EVENTS_PATH, {
            "schema_version": "dynamic_recommender.v1",
            "timestamp": _iso(),
            "event_id": event_id,
            "type": "manual_feedback",
            "scene_key": scene_key,
            "action_id": action["id"],
            "action": action["name"],
            "score": score,
            "source": source,
        })
        return action

    def suggest(self, *, user_text="", user_profile="", recent_context=None, matched_memories=None,
                time_features=None, trigger_type="user_message", top_k=DEFAULT_TOP_K,
                strategy_prediction=None):
        state = _load_state()
        scene_key = scene_key_from_runtime(
            user_text,
            time_features=time_features,
            trigger_type=trigger_type,
            strategy_prediction=strategy_prediction,
        )
        policy = recommendation_policy_score(scene_key)
        policy = self._apply_strategy_prediction_to_policy(policy, strategy_prediction)
        dynamic_candidates = profile_candidate_actions(user_profile, limit=6)
        for candidate in dynamic_candidates:
            self.upsert_action(candidate, source="profile_hint", evidence="profile_candidate", scene_key=scene_key)
        state = _load_state()
        actions = [a for a in (state.get("actions") or {}).values() if a.get("status") == "active"]
        candidate_generation = {
            "attempted": False,
            "reason": "",
            "generated": [],
        }
        allowed, gen_reason, gen_cfg = self._candidate_generation_allowed(
            state,
            scene_key,
            policy,
            actions,
            top_k,
        )
        candidate_generation["reason"] = gen_reason
        if allowed:
            candidate_generation["attempted"] = True
            try:
                generated, raw = self._generate_action_candidates(
                    scene_key=scene_key,
                    user_text=user_text,
                    user_profile=user_profile,
                    recent_context=recent_context,
                    matched_memories=matched_memories,
                    existing_actions=actions,
                    policy=policy,
                    cfg=gen_cfg,
                )
                for item in generated:
                    self.upsert_action(
                        item["name"],
                        source="llm_candidate",
                        evidence=item.get("reason") or "llm_candidate",
                        scene_key=scene_key,
                    )
                state = _load_state()
                state["last_candidate_generation"] = {
                    "timestamp": _iso(),
                    "scene_key": scene_key,
                    "count": len(generated),
                    "reason": gen_reason,
                }
                _save_state(state)
                actions = [a for a in (state.get("actions") or {}).values() if a.get("status") == "active"]
                candidate_generation.update({
                    "generated": generated,
                    "raw": raw,
                })
                _append_jsonl(RECOMMENDATION_EVENTS_PATH, {
                    "schema_version": "dynamic_recommender.v1",
                    "timestamp": _iso(),
                    "type": "candidate_generation",
                    "scene_key": scene_key,
                    "policy": policy,
                    "reason": gen_reason,
                    "generated": generated,
                })
            except Exception as e:
                candidate_generation["error"] = str(e)
                print(f"[Recommender] DeepSeek 候选生成失败，退回动作池: {e}")
        if not actions:
            return {
                "should_recommend": False,
                "scene_key": scene_key,
                "policy_score": policy.get("score", 0),
                "gate_score": policy.get("score", 0),
                "policy": policy,
                "candidate_generation": candidate_generation,
                "reason": "动作池为空，且本轮没有生成可用候选。",
                "candidates": [],
            }

        scored = []
        scene_stats = (state.get("scene_stats") or {}).get(scene_key, {})
        profile_blob = str(user_profile or "")
        recent_blob = " ".join(str(x) for x in (matched_memories or [])) + " " + str(recent_context or "")
        for action in actions:
            aid = action.get("id")
            global_score = _beta_mean(action.get("positive"), action.get("negative"))
            local = scene_stats.get(aid) or {}
            scene_score = _beta_mean(local.get("positive"), local.get("negative"))
            profile_match = 1.0 if action.get("name") and action.get("name")[:4] in profile_blob else 0.0
            context_match = 0.5 if action.get("name") and action.get("name")[:4] in recent_blob else 0.0
            repeat_penalty = 0.0
            if _days_since(action.get("last_recommended_at"), default=999) < 1:
                repeat_penalty = 0.18
            score = (
                0.38 * global_score
                + 0.34 * scene_score
                + 0.16 * profile_match
                + 0.08 * context_match
                + 0.04 * min(1.0, _days_since(action.get("last_seen_at"), default=0) / 14.0)
                - repeat_penalty
            )
            scored.append({
                "action_id": aid,
                "action": action.get("name"),
                "score": round(max(0.0, min(1.0, score)), 3),
                "global_accept_rate": round(global_score, 3),
                "scene_accept_rate": round(scene_score, 3),
                "source": action.get("source", ""),
                "reason": "由动态动作池、场景反馈矩阵和画像匹配综合打分。",
            })

        scored.sort(key=lambda x: x["score"], reverse=True)
        top = scored[:max(1, int(top_k or DEFAULT_TOP_K))]
        best = top[0]
        policy_score = float(policy.get("score") or 0.0)
        decision_score = (0.72 * policy_score) + (0.28 * float(best["score"] or 0.0))
        if _days_since((state.get("last_recommendation") or {}).get("timestamp"), default=999) < 0.08:
            decision_score -= 0.25
        should_recommend = decision_score >= 0.58
        decision = {
            "should_recommend": bool(should_recommend),
            "scene_key": scene_key,
            "policy_score": round(max(0.0, min(1.0, decision_score)), 3),
            "gate_score": round(max(0.0, min(1.0, decision_score)), 3),
            "policy": policy,
            "candidate_generation": candidate_generation,
            "selected_action": best if should_recommend else None,
            "candidates": top,
            "reason": (
                "本地推荐策略分认为当前场景适合给一个具体行动，且动作池中存在较高分候选。"
                if should_recommend else
                "本地推荐策略分不足，优先普通回应或关心。"
            ),
        }
        if should_recommend:
            state["last_recommendation"] = {"timestamp": _iso(), "action_id": best["action_id"], "action": best["action"]}
            action = state["actions"].get(best["action_id"])
            if action:
                action["last_recommended_at"] = _iso()
                action["exposures"] = int(action.get("exposures") or 0) + 1
            _save_state(state)
            _append_jsonl(RECOMMENDATION_EVENTS_PATH, {
                "schema_version": "dynamic_recommender.v1",
                "timestamp": _iso(),
                "type": "suggestion",
                "scene_key": scene_key,
                "decision": decision,
            })
        return decision

    def _apply_strategy_prediction_to_policy(self, policy, strategy_prediction):
        pred = strategy_prediction if isinstance(strategy_prediction, dict) else {}
        if not pred:
            return policy
        policy = dict(policy or {})
        try:
            confidence = float(pred.get("confidence") or 0.0)
        except Exception:
            confidence = 0.0
        confidence = max(0.0, min(1.0, confidence))
        target_conf = pred.get("target_confidence") or {}
        try:
            recommendation_confidence = float(target_conf.get("recommendation_intent", confidence))
        except Exception:
            recommendation_confidence = confidence
        recommendation_confidence = max(0.0, min(1.0, recommendation_confidence))
        intent = str(pred.get("recommendation_intent") or "").strip()
        action = str(pred.get("response_action") or "").strip()
        base_score = float(policy.get("score") or 0.0)
        adjusted = base_score
        adjustment = "none"

        if intent == "tool_action":
            adjusted = max(adjusted, 0.56 + 0.24 * recommendation_confidence)
            adjustment = "strategy_predictor_tool_action"
        elif intent == "suggest_action" or action == "推荐活动" or pred.get("should_recommend"):
            adjusted = max(adjusted, 0.52 + 0.26 * recommendation_confidence)
            adjustment = "strategy_predictor_suggest_action"
        elif intent == "none" and pred.get("source") == "mlp" and confidence >= 0.55:
            adjusted = min(adjusted, max(0.18, base_score * 0.88))
            adjustment = "strategy_predictor_prefers_no_recommendation"

        policy["score"] = round(max(0.0, min(1.0, adjusted)), 3)
        policy["base_score"] = round(base_score, 3)
        policy["strategy_adjustment"] = adjustment
        policy["strategy_prediction"] = {
            "source": pred.get("source"),
            "confidence": confidence,
            "recommendation_confidence": recommendation_confidence,
            "recommendation_intent": intent or "unknown",
            "response_action": action or "unknown",
        }
        if adjustment != "none":
            policy["source"] = f"{policy.get('source') or 'unknown'}+{adjustment}"
        return policy

    def _prune_if_needed(self, state):
        actions = state.setdefault("actions", {})
        active_ids = [aid for aid, a in actions.items() if a.get("status") == "active"]
        while len(active_ids) > self.max_actions:
            weakest = self._find_weakest_action(actions)
            if not weakest:
                break
            actions[weakest]["status"] = "pruned"
            actions[weakest]["pruned_at"] = _iso()
            active_ids = [aid for aid, a in actions.items() if a.get("status") == "active"]

    def _find_weakest_action(self, actions):
        weakest = None
        min_score = float("inf")
        for aid, action in actions.items():
            if action.get("status") != "active" or action.get("protected"):
                continue
            if _days_since(action.get("created_at"), default=999) < NEW_ACTION_PROTECTION_DAYS:
                continue
            success = _beta_mean(action.get("positive"), action.get("negative"))
            stale = _days_since(action.get("last_accepted_at") or action.get("last_seen_at"), default=999)
            survival_score = success / (stale + 1.0)
            if survival_score < min_score:
                min_score = survival_score
                weakest = aid
        if weakest:
            return weakest
        fallback = [
            (aid, action) for aid, action in actions.items()
            if action.get("status") == "active" and not action.get("protected")
        ]
        fallback.sort(key=lambda item: item[1].get("last_seen_at") or item[1].get("created_at") or "")
        return fallback[0][0] if fallback else None


recommendation_runtime = DynamicRecommender()


def record_manual_recommendation_label(event, labels, scores):
    rec = (labels or {}).get("recommendation_type") or {}
    action_text = rec.get("candidate_action") or ""
    if not action_text:
        return None
    scene_key = scene_key_from_event(event or {}, labels or {})
    score = int((scores or {}).get("recommendation_score") or 0)
    return recommendation_runtime.record_feedback(
        action_text=action_text,
        scene_key=scene_key,
        score=score,
        source="user_manual",
        event_id=(event or {}).get("event_id", ""),
    )


def format_recommendation_for_prompt(decision):
    if not decision:
        return "本地推荐器未运行。"
    candidate_generation = decision.get("candidate_generation") or {}
    generated = candidate_generation.get("generated") or []
    generation_note = ""
    if candidate_generation.get("attempted"):
        generation_note = f"；候选生成=已尝试，新增{len(generated)}个"
    elif candidate_generation.get("reason"):
        generation_note = f"；候选生成={candidate_generation.get('reason')}"
    if not decision.get("should_recommend"):
        return (
            f"本地推荐器判断：本轮不主动推荐具体行动。"
            f"策略分={decision.get('policy_score', decision.get('gate_score', 0))}；"
            f"依据={((decision.get('policy') or {}).get('source') or 'unknown')}；"
            f"原因={decision.get('reason', '')}{generation_note}"
        )
    action = decision.get("selected_action") or {}
    return (
        "本地推荐器判断：可以给一个具体但低压的建议。\n"
        f"- 推荐动作：{action.get('action')}\n"
        f"- 动作分数：{action.get('score')}\n"
        f"- 策略分：{decision.get('policy_score')}\n"
        f"- 场景：{decision.get('scene_key')}\n"
        f"- 策略依据：{((decision.get('policy') or {}).get('source') or 'unknown')}\n"
        f"- 候选生成：{generation_note.lstrip('；') or '未触发'}\n"
        f"- 原因：{decision.get('reason')}\n"
        "表达要求：只把它作为可选建议，不要说教，不要假装你能替用户执行。"
    )
