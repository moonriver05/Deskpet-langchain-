"""Local response-strategy predictor for preference bootstrapping.

The model is deliberately small: hashed text/state features + a one-hidden-layer
MLP trained from DeepSeek/manual labels. It predicts how Alice should respond,
not the final wording.
"""

import datetime
import hashlib
import json
import math
import os
import re
import threading

try:
    import numpy as np
except Exception:  # pragma: no cover - runtime fallback for minimal installs
    np = None

from pet_core.config import app_config
from pet_core.learning_logger import LABELED_INTERACTIONS_PATH


MODEL_VERSION = "strategy_predictor_mlp.v1"
DEFAULT_FEATURE_DIM = 384
DEFAULT_HIDDEN_DIM = 32
DEFAULT_EPOCHS = 70
MIN_TRAIN_SAMPLES = 25

TARGETS = {
    "emotion": ("state", "emotion"),
    "task_state": ("state", "task_state"),
    "need": ("need", "primary"),
    "tone": ("feedback_strategy", "tone"),
    "length": ("feedback_strategy", "length"),
    "response_action": ("feedback_strategy", "action"),
    "recommendation_intent": ("feedback_strategy", "recommendation_intent"),
}

SKIP_VALUES = {"", "未知", "unknown", "None", "none/null"}
KEEP_VALUES = {
    "recommendation_intent": {"none", "suggest_action", "tool_action"},
}

_cache_lock = threading.Lock()


def _truthy(value, default=True):
    if value is None or value == "":
        return default
    return str(value).strip().lower() not in {"0", "false", "no", "off"}


def _int_config(key, default, min_value=1, max_value=None):
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


def _safe_text(value, limit=1200):
    text = str(value or "").strip()
    if len(text) > limit:
        return text[:limit] + "...[truncated]"
    return text


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


def _file_signature(path):
    try:
        st = os.stat(path)
        return (st.st_size, int(st.st_mtime))
    except Exception:
        return (0, 0)


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


def _time_features_from_runtime(recent_context=None):
    now = datetime.datetime.now()
    last_minutes = None
    for item in reversed(list(recent_context or [])):
        if isinstance(item, dict) and item.get("minutes_ago") is not None:
            try:
                last_minutes = float(item.get("minutes_ago"))
                break
            except Exception:
                pass
    return {
        "hour": now.hour,
        "minute": now.minute,
        "weekday": now.weekday(),
        "daypart": _daypart(now.hour),
        "gap_since_last_turn_sec": None if last_minutes is None else int(last_minutes * 60),
    }


def _stable_index(token, dim):
    digest = hashlib.blake2b(str(token).encode("utf-8"), digest_size=8).digest()
    raw = int.from_bytes(digest, "little", signed=False)
    return raw % dim, 1.0 if ((raw >> 17) & 1) else -1.0


def _text_tokens(text, prefix="txt"):
    text = str(text or "").lower()
    tokens = []
    for word in re.findall(r"[a-z0-9_+\-.]{2,}", text):
        tokens.append(f"{prefix}:w:{word[:32]}")
    cjk = "".join(re.findall(r"[\u4e00-\u9fff]", text))
    for n in (1, 2, 3):
        if len(cjk) < n:
            continue
        limit = min(len(cjk) - n + 1, 220)
        for i in range(limit):
            tokens.append(f"{prefix}:c{n}:{cjk[i:i+n]}")
    return tokens


def _add_kv(tokens, key, value):
    if value is None or value == "":
        value = "unknown"
    tokens.append(f"{key}={value}")


def _event_feature_tokens(event):
    tokens = []
    trigger = event.get("trigger") or {}
    user_input = event.get("user_input") or {}
    time_features = event.get("time_features") or {}
    system = ((event.get("state_features") or {}).get("system") or {})
    foreground = system.get("foreground") or {}
    idle = system.get("idle") or {}
    retrieval = event.get("retrieval") or {}

    _add_kv(tokens, "trigger_type", trigger.get("type"))
    _add_kv(tokens, "trigger_source", trigger.get("source"))
    _add_kv(tokens, "daypart", time_features.get("daypart"))
    _add_kv(tokens, "weekday", time_features.get("weekday"))
    try:
        hour_bucket = int(time_features.get("hour", 0)) // 3
    except Exception:
        hour_bucket = "unknown"
    _add_kv(tokens, "hour_bucket", hour_bucket)
    _add_kv(tokens, "foreground", foreground.get("category"))
    _add_kv(tokens, "idle_bucket", idle.get("seconds_bucket"))
    _add_kv(tokens, "has_attachment", bool(user_input.get("has_attachment")))

    text_parts = [
        user_input.get("text") or "",
        event.get("user_profile_snapshot") or "",
        " ".join(str(x) for x in (retrieval.get("short_memory_snippets") or [])[:4]),
    ]
    for item in list(event.get("recent_context") or [])[-6:]:
        if not isinstance(item, dict):
            continue
        age = item.get("minutes_ago")
        if age is not None:
            try:
                age = float(age)
                if age > 360:
                    continue
            except Exception:
                pass
        text_parts.append(item.get("user") or "")
        text_parts.append(item.get("assistant_summary") or "")

    for idx, part in enumerate(text_parts):
        tokens.extend(_text_tokens(_safe_text(part, 900), prefix=f"t{idx}"))
    return tokens


def _runtime_event(user_text="", recent_context=None, user_profile="", system_state=None,
                   matched_memories=None, time_features=None, trigger_type="user_message"):
    tf = dict(time_features or _time_features_from_runtime(recent_context))
    system_state = dict(system_state or {})
    return {
        "trigger": {"type": trigger_type, "source": "chat" if trigger_type == "user_message" else "proactive"},
        "user_input": {"text": _safe_text(user_text, 3000), "has_attachment": False},
        "recent_context": list(recent_context or [])[-8:],
        "time_features": tf,
        "state_features": {"system": system_state},
        "user_profile_snapshot": _safe_text(user_profile, 5000),
        "retrieval": {
            "short_memory_snippets": [_safe_text(x, 500) for x in (matched_memories or [])[:4]],
        },
    }


def _vectorize(tokens, dim):
    if np is None:
        return None
    x = np.zeros(dim, dtype=np.float32)
    for token in tokens:
        idx, sign = _stable_index(token, dim)
        x[idx] += sign
    norm = float(np.linalg.norm(x))
    if norm > 1e-6:
        x /= norm
    return x


def _label_value(labels, target):
    section, key = TARGETS[target]
    obj = labels.get(section) or {}
    value = str(obj.get(key) or "").strip()
    if value in KEEP_VALUES.get(target, set()):
        return value
    if value in SKIP_VALUES:
        return ""
    return value


def _sample_weight(event, labels):
    if event.get("label_status") == "manual_labeled" or event.get("manual_label_meta"):
        meta = event.get("manual_label_meta") or {}
        scores = meta.get("scores") or {}
        try:
            return max(1.5, min(5.0, float(scores.get("sample_weight") or meta.get("sample_weight") or 3.0)))
        except Exception:
            return 3.0
    strategy = labels.get("feedback_strategy") or {}
    try:
        confidence = float(strategy.get("confidence") or 0.55)
    except Exception:
        confidence = 0.55
    return max(0.35, min(1.25, confidence))


class _MlpHead:
    def __init__(self, target, labels, feature_dim, hidden_dim, seed):
        self.target = target
        self.labels = list(labels)
        self.label_to_idx = {label: i for i, label in enumerate(self.labels)}
        rng = np.random.default_rng(seed)
        scale1 = math.sqrt(2.0 / max(1, feature_dim + hidden_dim))
        scale2 = math.sqrt(2.0 / max(1, hidden_dim + len(self.labels)))
        self.w1 = rng.normal(0.0, scale1, size=(feature_dim, hidden_dim)).astype(np.float32)
        self.b1 = np.zeros(hidden_dim, dtype=np.float32)
        self.w2 = rng.normal(0.0, scale2, size=(hidden_dim, len(self.labels))).astype(np.float32)
        self.b2 = np.zeros(len(self.labels), dtype=np.float32)

    def train(self, x, y_text, sample_weights, epochs=DEFAULT_EPOCHS, lr=0.035):
        y = np.array([self.label_to_idx[v] for v in y_text], dtype=np.int64)
        n = max(1, len(y))
        class_counts = np.bincount(y, minlength=len(self.labels)).astype(np.float32)
        class_weights = n / np.maximum(1.0, class_counts * len(self.labels))
        weights = np.asarray(sample_weights, dtype=np.float32) * class_weights[y]
        weights = weights / max(1e-6, float(np.mean(weights)))
        onehot = np.zeros((n, len(self.labels)), dtype=np.float32)
        onehot[np.arange(n), y] = 1.0
        l2 = 1e-4

        for epoch in range(int(epochs)):
            h = np.tanh(x @ self.w1 + self.b1)
            logits = h @ self.w2 + self.b2
            logits -= logits.max(axis=1, keepdims=True)
            exp = np.exp(logits)
            probs = exp / np.maximum(1e-8, exp.sum(axis=1, keepdims=True))

            grad_logits = (probs - onehot) * weights[:, None] / n
            grad_w2 = h.T @ grad_logits + l2 * self.w2
            grad_b2 = grad_logits.sum(axis=0)
            grad_h = grad_logits @ self.w2.T
            grad_z1 = grad_h * (1.0 - h * h)
            grad_w1 = x.T @ grad_z1 + l2 * self.w1
            grad_b1 = grad_z1.sum(axis=0)
            step = lr * (0.35 + 0.65 * (1.0 - epoch / max(1, epochs)))
            self.w2 -= step * grad_w2
            self.b2 -= step * grad_b2
            self.w1 -= step * grad_w1
            self.b1 -= step * grad_b1

    def predict(self, x):
        h = np.tanh(x @ self.w1 + self.b1)
        logits = h @ self.w2 + self.b2
        logits -= logits.max()
        exp = np.exp(logits)
        probs = exp / max(1e-8, float(exp.sum()))
        idx = int(np.argmax(probs))
        return self.labels[idx], float(probs[idx])


class StrategyPredictorRuntime:
    def __init__(self):
        self._signature = None
        self._model = None

    def _enabled(self):
        return _truthy(app_config.get("strategy_predictor.enabled", "true"), default=True)

    def _load_samples(self):
        rows = []
        for event in _iter_jsonl(LABELED_INTERACTIONS_PATH) or []:
            labels = ((event.get("prompt_decision") or {}).get("teacher_labels") or {})
            if not labels:
                continue
            rows.append((event, labels, _sample_weight(event, labels)))
        return rows

    def _build_model(self):
        if np is None:
            return {"kind": "disabled", "reason": "numpy_not_available", "sample_count": 0}
        rows = self._load_samples()
        min_samples = _int_config("strategy_predictor.min_samples", MIN_TRAIN_SAMPLES, min_value=5)
        if len(rows) < min_samples:
            return {"kind": "stats", "reason": "not_enough_labeled_samples", "sample_count": len(rows)}

        feature_dim = _int_config("strategy_predictor.feature_dim", DEFAULT_FEATURE_DIM, min_value=128, max_value=2048)
        hidden_dim = _int_config("strategy_predictor.hidden_dim", DEFAULT_HIDDEN_DIM, min_value=8, max_value=128)
        epochs = _int_config("strategy_predictor.epochs", DEFAULT_EPOCHS, min_value=10, max_value=300)
        lr = _float_config("strategy_predictor.learning_rate", 0.035, min_value=0.001, max_value=0.2)

        x_rows = []
        weights = []
        target_values = {target: [] for target in TARGETS}
        target_indices = {target: [] for target in TARGETS}
        numeric_rows = []

        for idx, (event, labels, weight) in enumerate(rows):
            vec = _vectorize(_event_feature_tokens(event), feature_dim)
            if vec is None:
                continue
            x_rows.append(vec)
            weights.append(weight)
            strategy = labels.get("feedback_strategy") or {}
            numeric_rows.append({
                "supervision_level": _to_float(strategy.get("supervision_level")),
                "comfort_level": _to_float(strategy.get("comfort_level")),
            })
            for target in TARGETS:
                value = _label_value(labels, target)
                if value:
                    target_values[target].append(value)
                    target_indices[target].append(idx)

        if not x_rows:
            return {"kind": "stats", "reason": "no_vectorized_samples", "sample_count": len(rows)}

        x_all = np.stack(x_rows).astype(np.float32)
        weight_all = np.asarray(weights, dtype=np.float32)
        heads = {}
        priors = {}
        for target, values in target_values.items():
            counts = {}
            for value in values:
                counts[value] = counts.get(value, 0) + 1
            priors[target] = counts
            if len(counts) < 2 or len(values) < 8:
                continue
            indices = np.asarray(target_indices[target], dtype=np.int64)
            labels = sorted(counts, key=lambda v: (-counts[v], v))
            head = _MlpHead(
                target,
                labels,
                feature_dim,
                hidden_dim,
                seed=int(hashlib.sha1(target.encode("utf-8")).hexdigest()[:8], 16),
            )
            head.train(x_all[indices], values, weight_all[indices], epochs=epochs, lr=lr)
            heads[target] = head

        numeric_defaults = _numeric_defaults(numeric_rows, weight_all)
        return {
            "kind": "mlp" if heads else "stats",
            "version": MODEL_VERSION,
            "feature_dim": feature_dim,
            "hidden_dim": hidden_dim,
            "sample_count": len(rows),
            "heads": heads,
            "priors": priors,
            "numeric_defaults": numeric_defaults,
        }

    def _ensure_model(self):
        signature = (
            _file_signature(LABELED_INTERACTIONS_PATH),
            app_config.get("strategy_predictor.enabled", "true"),
            app_config.get("strategy_predictor.feature_dim", DEFAULT_FEATURE_DIM),
            app_config.get("strategy_predictor.hidden_dim", DEFAULT_HIDDEN_DIM),
            app_config.get("strategy_predictor.epochs", DEFAULT_EPOCHS),
        )
        with _cache_lock:
            if self._model is None or self._signature != signature:
                self._model = self._build_model()
                self._signature = signature
                if self._model.get("kind") == "mlp":
                    print(
                        f"[StrategyPredictor] MLP ready: samples={self._model.get('sample_count')} "
                        f"heads={','.join(sorted((self._model.get('heads') or {}).keys()))}"
                    )
                else:
                    print(f"[StrategyPredictor] fallback: {self._model.get('reason', self._model.get('kind'))}")
            return self._model

    def predict(self, *, user_text="", recent_context=None, user_profile="", system_state=None,
                matched_memories=None, time_features=None, trigger_type="user_message"):
        if not self._enabled():
            return {
                "schema_version": MODEL_VERSION,
                "source": "disabled",
                "strategy": "本地策略预测器未启用；按当前输入克制回应。",
                "confidence": 0.0,
            }
        event = _runtime_event(
            user_text=user_text,
            recent_context=recent_context,
            user_profile=user_profile,
            system_state=system_state,
            matched_memories=matched_memories,
            time_features=time_features,
            trigger_type=trigger_type,
        )
        model = self._ensure_model()
        prediction = self._predict_with_model(model, event)
        prediction["time_features"] = event.get("time_features") or {}
        return prediction

    def _predict_with_model(self, model, event):
        kind = model.get("kind")
        result = {
            "schema_version": MODEL_VERSION,
            "source": kind,
            "sample_count": model.get("sample_count", 0),
            "confidence": 0.0,
        }
        if kind == "mlp" and np is not None:
            x = _vectorize(_event_feature_tokens(event), int(model.get("feature_dim") or DEFAULT_FEATURE_DIM))
            confidences = []
            target_confidences = {}
            heads = model.get("heads") or {}
            for target in TARGETS:
                if target in heads:
                    value, conf = heads[target].predict(x)
                    result[target] = value
                    confidences.append(conf)
                    target_confidences[target] = round(float(conf), 3)
                else:
                    value, conf = _prior_prediction((model.get("priors") or {}).get(target) or {})
                    if value:
                        result[target] = value
                        confidences.append(conf * 0.55)
                        target_confidences[target] = round(float(conf * 0.55), 3)
            result["confidence"] = round(float(sum(confidences) / max(1, len(confidences))), 3)
            result["target_confidence"] = target_confidences
        else:
            result.update(_heuristic_prediction(event))

        numeric = model.get("numeric_defaults") or {}
        result.setdefault("supervision_level", numeric.get("supervision_level", 0.35))
        result.setdefault("comfort_level", numeric.get("comfort_level", 0.45))
        _finalize_prediction(result)
        return result


def _to_float(value):
    try:
        return float(value)
    except Exception:
        return None


def _numeric_defaults(rows, weights):
    defaults = {}
    for key in ("supervision_level", "comfort_level"):
        vals = []
        w = []
        for i, row in enumerate(rows):
            value = row.get(key)
            if value is None:
                continue
            vals.append(value)
            try:
                w.append(float(weights[i]))
            except Exception:
                w.append(1.0)
        if vals:
            defaults[key] = round(float(np.average(np.asarray(vals), weights=np.asarray(w))), 3)
    return defaults


def _prior_prediction(counts):
    if not counts:
        return "", 0.0
    total = sum(counts.values())
    value, count = max(counts.items(), key=lambda kv: (kv[1], kv[0]))
    return value, float(count) / max(1.0, float(total))


def _heuristic_prediction(event):
    text = ((event.get("user_input") or {}).get("text") or "")
    emotion = "平静"
    task_state = "未知"
    need = "陪伴"
    if any(w in text for w in ("累", "困", "疲惫", "没精神")):
        emotion = "疲惫"
        need = "安慰"
    if any(w in text for w in ("焦虑", "压力", "来不及", "慌")):
        emotion = "焦虑"
        need = "拆任务"
    if any(w in text for w in ("不想", "拖延", "背不动", "学不动")):
        task_state = "拖延中"
        need = "督促"
    if any(w in text for w in ("学习", "复习", "作业", "背单词", "考试")):
        task_state = "学习中" if task_state == "未知" else task_state
    return {
        "emotion": emotion,
        "task_state": task_state,
        "need": need,
        "tone": "冷淡关心",
        "length": "短",
        "response_action": "只回应",
        "recommendation_intent": "none",
        "confidence": 0.32,
    }


def _finalize_prediction(result):
    emotion = result.get("emotion") or "未知"
    task_state = result.get("task_state") or "未知"
    need = result.get("need") or "未知"
    tone = result.get("tone") or "冷淡关心"
    length = result.get("length") or "短"
    action = result.get("response_action") or "只回应"
    rec_intent = result.get("recommendation_intent") or "none"
    should_recommend = rec_intent in {"suggest_action", "tool_action"} or action == "推荐活动"
    result["mood"] = f"{task_state} + {emotion}" if task_state != "未知" else emotion
    result["intent"] = need
    result["need"] = need
    result["strategy"] = f"{length}回复；语气={tone}；动作={action}；推荐倾向={rec_intent}"
    result["response_strategy"] = result["strategy"]
    result["should_recommend"] = bool(should_recommend)
    result["avoid"] = ["长篇说教", "假装现实陪伴", "过度热情客服腔"]
    result["confidence"] = round(float(result.get("confidence") or 0.0), 3)


strategy_predictor_runtime = StrategyPredictorRuntime()
