"""Offline DeepSeek teacher-labeler for preference/recommendation samples.

This module consumes learning_data/label_queue.jsonl, joins each task with the
full raw event by event_id, asks a cheap OpenAI-compatible model for weak
labels, then writes append-only labeled records. It deliberately does not run
inside the main chat request path.
"""

import argparse
import copy
import datetime
import json
import os
import time

from openai import OpenAI

from pet_core.config import app_config
from pet_core.learning_logger import (
    FEEDBACK_EVENTS_PATH,
    LABELED_INTERACTIONS_PATH,
    LABEL_QUEUE_PATH,
    LABEL_RESULTS_PATH,
    RAW_INTERACTIONS_PATH,
    SCHEMA_VERSION,
)


def _jsonl_iter(path):
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception as e:
                print(f"[Labeler] 跳过损坏 JSONL: {path}:{line_no} {e}")
                continue
            if isinstance(obj, dict):
                yield obj


def _append_jsonl(path, record):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _safe_text(value, limit=1800):
    text = str(value or "").strip()
    if len(text) > limit:
        return text[:limit] + "...[truncated]"
    return text


def _short_list(values, limit=4, text_limit=420):
    out = []
    for item in list(values or [])[:limit]:
        out.append(_safe_text(item, text_limit))
    return out


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


def _truthy(value, default=True):
    if value is None or value == "":
        return default
    return str(value).strip().lower() not in ("0", "false", "no", "off")


def _labeler_config():
    enabled = _truthy(app_config.get("learning_labeler.enabled", "true"), default=True)
    api_key = (
        app_config.get("learning_labeler.api_key", "")
        or app_config.get("profile_refiner.api_key", "")
        or app_config.get("memory_reranker.api_key", "")
        or ""
    )
    base_url = (
        app_config.get("learning_labeler.base_url", "")
        or app_config.get("profile_refiner.base_url", "")
        or "https://api.deepseek.com"
    )
    model = (
        app_config.get("learning_labeler.model", "")
        or app_config.get("profile_refiner.model", "")
        or "deepseek-chat"
    )
    try:
        timeout_seconds = float(app_config.get("learning_labeler.timeout_seconds", 20))
    except Exception:
        timeout_seconds = 20.0
    try:
        max_events = int(app_config.get("learning_labeler.max_events_per_run", 20))
    except Exception:
        max_events = 20
    try:
        observation_window_seconds = int(app_config.get("learning_labeler.observation_window_seconds", 900))
    except Exception:
        observation_window_seconds = 900
    try:
        feedback_settle_seconds = int(app_config.get("learning_labeler.feedback_settle_seconds", 180))
    except Exception:
        feedback_settle_seconds = 180
    return {
        "enabled": enabled,
        "api_key": api_key,
        "base_url": base_url,
        "model": model,
        "timeout_seconds": max(5.0, timeout_seconds),
        "max_events": max(1, max_events),
        "observation_window_seconds": max(60, observation_window_seconds),
        "feedback_settle_seconds": max(30, feedback_settle_seconds),
    }


def _processed_event_ids():
    done = set()
    for record in _jsonl_iter(LABEL_RESULTS_PATH) or []:
        if record.get("status") in ("labeled", "manual_labeled") and record.get("event_id"):
            done.add(str(record["event_id"]))
    return done


def _raw_event_index():
    events = {}
    for event in _jsonl_iter(RAW_INTERACTIONS_PATH) or []:
        event_id = event.get("event_id")
        if event_id:
            events[str(event_id)] = event
    return events


def _feedback_events_index():
    feedback = {}
    for record in _jsonl_iter(FEEDBACK_EVENTS_PATH) or []:
        event_id = record.get("event_id")
        if not event_id:
            continue
        feedback.setdefault(str(event_id), []).append(record)
    return feedback


def _pending_tasks(processed_ids):
    for task in _jsonl_iter(LABEL_QUEUE_PATH) or []:
        event_id = task.get("event_id")
        if not event_id:
            continue
        if task.get("status") != "pending":
            continue
        if str(event_id) in processed_ids:
            continue
        yield task


def _age_seconds(timestamp):
    try:
        if not timestamp:
            return None
        dt = datetime.datetime.fromisoformat(str(timestamp))
        return max(0.0, (datetime.datetime.now() - dt).total_seconds())
    except Exception:
        return None


def _pending_tasks_with_age(processed_ids, min_age_seconds=None):
    for task in _pending_tasks(processed_ids):
        if min_age_seconds is not None:
            age = _age_seconds(task.get("timestamp"))
            if age is not None and age < float(min_age_seconds):
                continue
        yield task


def _feedback_summary(feedback_events):
    explicit = []
    implicit = []
    positive = 0
    negative = 0
    for item in feedback_events or []:
        val = item.get("feedback")
        try:
            val = int(val)
        except Exception:
            val = 0
        if val > 0:
            positive += 1
        elif val < 0:
            negative += 1
        scope = str(item.get("feedback_scope") or "")
        packed = {
            "timestamp": item.get("timestamp", ""),
            "scope": scope,
            "feedback": val,
            "user_text": _safe_text(item.get("user_text"), 500),
            "assistant_text": _safe_text(item.get("assistant_text"), 500),
            "extra": item.get("extra") or {},
        }
        if scope == "reply_strategy":
            explicit.append(packed)
        else:
            implicit.append(packed)
    return {
        "positive_count": positive,
        "negative_count": negative,
        "explicit": explicit[-5:],
        "implicit": implicit[-8:],
    }


def _label_payload(event, feedback_events=None):
    retrieval = event.get("retrieval") or {}
    feedback_events = list(feedback_events or [])
    trigger = event.get("trigger") or {}
    assistant = event.get("assistant_reply") or {}
    is_silence_sample = (
        trigger.get("type") == "proactive_timer"
        and (
            trigger.get("source") == "proactive_silence"
            or (event.get("strategy") or {}).get("selected") == "do_nothing"
            or not str(assistant.get("text") or "").strip()
        )
    )
    return {
        "schema_version": event.get("schema_version"),
        "event_id": event.get("event_id"),
        "event_type": event.get("event_type"),
        "trigger": trigger,
        "sample_semantics": {
            "is_silence_sample": bool(is_silence_sample),
            "meaning": (
                "主动关怀检查后选择保持沉默，assistant_reply.text 为空不是缺失数据。"
                if is_silence_sample else
                "普通已发送消息样本。"
            ),
        },
        "user_input": event.get("user_input") or {},
        "recent_context": list(event.get("recent_context") or [])[-8:],
        "time_features": event.get("time_features") or {},
        "state_features": event.get("state_features") or {},
        "user_profile_snapshot": _safe_text(event.get("user_profile_snapshot"), 1600),
        "retrieval": {
            "retrieved_memory_ids": retrieval.get("retrieved_memory_ids") or [],
            "short_memory_snippets": _short_list(retrieval.get("short_memory_snippets"), 4),
            "knowledge_tool": retrieval.get("knowledge_tool") or {},
            "knowledge_snippets": _short_list(retrieval.get("knowledge_snippets"), 2),
            "search_keywords": retrieval.get("search_keywords") or [],
            "extracted_fact": _safe_text(retrieval.get("extracted_fact"), 500),
        },
        "prompt_decision": {
            "current_response_card": _safe_text(
                (event.get("prompt_decision") or {}).get("current_response_card"),
                1500,
            ),
            "local_prediction": (event.get("prompt_decision") or {}).get("local_prediction") or {},
        },
        "strategy": event.get("strategy") or {},
        "assistant_reply": event.get("assistant_reply") or {},
        "feedback": event.get("feedback") or {},
        "post_event_feedback": _feedback_summary(feedback_events),
    }


def _build_label_prompt(event, feedback_events=None):
    payload = _label_payload(event, feedback_events=feedback_events)
    return f"""你是桌宠个性化推荐系统的数据标注器。请根据一次交互样本，为未来的本地偏好预测器打弱标签。

目标不是评价文学性，而是抽取：用户状态、用户可能需要什么、本轮回复策略、是否适合推荐、主动关怀时机、是否更该保持沉默、以及风险。

注意：
1. 标签是训练辅助，不是真理；不确定时降低 confidence。
2. 不要把用户画像当成本轮事实，只能当稳定倾向。
3. 如果知识库片段与本轮用户输入无关，risk.irrelevant_memory_or_knowledge=true。
4. 如果回复声称自己能现实触碰、守着、递东西、泡饮料、做饭或监视用户，risk.fabricated_reality_action=true。
5. 推荐是回复策略的一部分；如果本轮只是关心，不要强行标成推荐。
6. 主动关怀样本 trigger.type 为 proactive_timer；普通聊天为 user_message。
7. post_event_feedback 是更强证据：点赞/踩、主动消息后用户是否打开聊天、是否回复，都应影响 timing_quality、do_nothing_preference、recommendation_type 和 confidence。
8. 如果 sample_semantics.is_silence_sample=true，说明系统当时选择“保持沉默”，assistant_reply.text 为空不是坏数据。此时 feedback_strategy.action 应优先标为“保持沉默”，proactive_timing.should_have_stayed_silent 通常为 true，除非后续反馈明确说明沉默不合适。
9. 如果没有后续反馈，不要脑补用户喜好；用当前上下文标注状态，但把 confidence 降低。

只输出严格 JSON，字段必须完整：
{{
  "state": {{
    "emotion": "疲惫/焦虑/开心/平静/烦躁/困惑/身体不适/未知",
    "task_state": "学习中/拖延中/休息中/饮食中/写代码中/闲聊/未知",
    "energy": "低/中/高/未知",
    "confidence": 0.0
  }},
  "need": {{
    "primary": "安慰/督促/陪伴/具体建议/拆任务/少说话/查资料/记录/未知",
    "secondary": [],
    "confidence": 0.0
  }},
  "feedback_strategy": {{
    "tone": "冷淡关心/温柔安慰/轻微督促/具体建议/吐槽/中性",
    "length": "短/中/长",
    "action": "只回应/允许短休/给下一步/推荐活动/调用工具/保持沉默",
    "supervision_level": 0.0,
    "comfort_level": 0.0,
    "recommendation_intent": "none/suggest_action/tool_action",
    "confidence": 0.0
  }},
  "recommendation_type": {{
    "should_recommend": false,
    "category": "none/rest/study/food/drawing/music/timer/todo/knowledge/other",
    "candidate_action": "",
    "reason": ""
  }},
  "proactive_timing": {{
    "is_proactive": false,
    "timing_quality": "not_applicable/good/too_soon/too_late/interruptive/unknown",
    "should_have_stayed_silent": false,
    "reason": ""
  }},
  "do_nothing_preference": {{
    "score": 0.0,
    "reason": ""
  }},
  "risk": {{
    "fabricated_reality_action": false,
    "overlong": false,
    "too_pushy": false,
    "irrelevant_memory_or_knowledge": false,
    "notes": []
  }}
}}

交互样本：
{json.dumps(payload, ensure_ascii=False)}
"""


def _call_labeler(client, cfg, event, feedback_events=None):
    prompt = _build_label_prompt(event, feedback_events=feedback_events)
    response = client.chat.completions.create(
        model=cfg["model"],
        messages=[
            {"role": "system", "content": "你只输出严格 JSON，不要解释。"},
            {"role": "user", "content": prompt},
        ],
        temperature=0.1,
        max_tokens=1200,
    )
    raw = (response.choices[0].message.content or "").strip()
    labels = _extract_json_object(raw)
    if not labels:
        raise ValueError(f"DeepSeek 未返回有效 JSON: {raw[:240]}")
    return labels, raw


def label_pending_events(limit=None, dry_run=False, min_age_seconds=None, respect_observation_window=True):
    cfg = _labeler_config()
    if not cfg["enabled"]:
        return {"processed": 0, "failed": 0, "skipped": 0, "message": "learning_labeler.enabled=false"}
    if not cfg["api_key"]:
        return {"processed": 0, "failed": 0, "skipped": 0, "message": "未配置 DeepSeek/API key"}

    processed_ids = _processed_event_ids()
    raw_events = _raw_event_index()
    feedback_events = _feedback_events_index()
    max_count = int(limit or cfg["max_events"])
    if min_age_seconds is None and respect_observation_window:
        min_age_seconds = cfg["observation_window_seconds"]
    tasks = list(_pending_tasks_with_age(processed_ids, min_age_seconds=min_age_seconds))[:max_count]
    if dry_run:
        return {
            "processed": 0,
            "failed": 0,
            "skipped": 0,
            "pending": len(tasks),
            "message": "dry_run",
            "event_ids": [task.get("event_id") for task in tasks],
        }

    client = OpenAI(
        api_key=cfg["api_key"],
        base_url=cfg["base_url"],
        timeout=cfg["timeout_seconds"],
        max_retries=0,
    )
    processed = 0
    failed = 0
    skipped = 0

    for task in tasks:
        event_id = str(task.get("event_id"))
        event = raw_events.get(event_id)
        if not event:
            skipped += 1
            _append_jsonl(LABEL_RESULTS_PATH, {
                "schema_version": SCHEMA_VERSION,
                "event_id": event_id,
                "timestamp": task.get("timestamp", ""),
                "status": "missing_raw_event",
                "task": "teacher_label",
            })
            continue

        started = time.time()
        try:
            labels, raw = _call_labeler(client, cfg, event, feedback_events=feedback_events.get(event_id, []))
            elapsed = round(time.time() - started, 2)
            result = {
                "schema_version": SCHEMA_VERSION,
                "event_id": event_id,
                "timestamp": event.get("timestamp"),
                "status": "labeled",
                "task": "teacher_label",
                "labeler": {
                    "base_url": cfg["base_url"],
                    "model": cfg["model"],
                    "elapsed_sec": elapsed,
                },
                "labels": labels,
                "raw_labeler_text": _safe_text(raw, 3000),
            }
            labeled_event = copy.deepcopy(event)
            labeled_event.setdefault("prompt_decision", {})["teacher_labels"] = labels
            labeled_event["label_status"] = "labeled"
            labeled_event["teacher_label_meta"] = result["labeler"]
            _append_jsonl(LABEL_RESULTS_PATH, result)
            _append_jsonl(LABELED_INTERACTIONS_PATH, labeled_event)
            processed += 1
            print(f"[Labeler] labeled event={event_id} {elapsed}s")
        except Exception as e:
            failed += 1
            _append_jsonl(LABEL_RESULTS_PATH, {
                "schema_version": SCHEMA_VERSION,
                "event_id": event_id,
                "timestamp": event.get("timestamp", ""),
                "status": "failed",
                "task": "teacher_label",
                "error": str(e),
            })
            print(f"[Labeler] failed event={event_id}: {e}")

    return {
        "processed": processed,
        "failed": failed,
        "skipped": skipped,
        "pending_considered": len(tasks),
        "message": "ok",
    }


def label_single_event(event_id, min_age_seconds=None, force=False):
    event_id = str(event_id or "").strip()
    if not event_id:
        return {"processed": 0, "failed": 1, "message": "event_id 为空"}
    cfg = _labeler_config()
    if not cfg["enabled"]:
        return {"processed": 0, "failed": 0, "message": "learning_labeler.enabled=false"}
    if not cfg["api_key"]:
        return {"processed": 0, "failed": 0, "message": "未配置 DeepSeek/API key"}

    raw_events = _raw_event_index()
    feedback_events = _feedback_events_index()
    event = raw_events.get(event_id)
    if not event:
        return {"processed": 0, "failed": 1, "message": f"找不到 raw event: {event_id}"}
    if not force:
        if min_age_seconds is None:
            min_age_seconds = cfg["feedback_settle_seconds"]
        age = _age_seconds(event.get("timestamp"))
        if age is not None and age < float(min_age_seconds):
            return {
                "processed": 0,
                "failed": 0,
                "skipped": 1,
                "event_id": event_id,
                "message": f"waiting_for_feedback_window: age={round(age, 1)}s < {int(min_age_seconds)}s",
            }

    client = OpenAI(
        api_key=cfg["api_key"],
        base_url=cfg["base_url"],
        timeout=cfg["timeout_seconds"],
        max_retries=0,
    )
    started = time.time()
    try:
        labels, raw = _call_labeler(client, cfg, event, feedback_events=feedback_events.get(event_id, []))
        elapsed = round(time.time() - started, 2)
        result = {
            "schema_version": SCHEMA_VERSION,
            "event_id": event_id,
            "timestamp": event.get("timestamp"),
            "status": "labeled",
            "task": "teacher_label",
            "rerun": True,
            "labeler": {
                "base_url": cfg["base_url"],
                "model": cfg["model"],
                "elapsed_sec": elapsed,
            },
            "labels": labels,
            "raw_labeler_text": _safe_text(raw, 3000),
        }
        labeled_event = copy.deepcopy(event)
        labeled_event.setdefault("prompt_decision", {})["teacher_labels"] = labels
        labeled_event["label_status"] = "labeled"
        labeled_event["teacher_label_meta"] = result["labeler"]
        _append_jsonl(LABEL_RESULTS_PATH, result)
        _append_jsonl(LABELED_INTERACTIONS_PATH, labeled_event)
        return {"processed": 1, "failed": 0, "event_id": event_id, "elapsed_sec": elapsed, "message": "ok"}
    except Exception as e:
        _append_jsonl(LABEL_RESULTS_PATH, {
            "schema_version": SCHEMA_VERSION,
            "event_id": event_id,
            "timestamp": event.get("timestamp", ""),
            "status": "failed",
            "task": "teacher_label",
            "rerun": True,
            "error": str(e),
        })
        return {"processed": 0, "failed": 1, "event_id": event_id, "message": str(e)}


def main():
    parser = argparse.ArgumentParser(description="Label pending learning samples with DeepSeek.")
    parser.add_argument("--limit", type=int, default=None, help="Max events to label this run.")
    parser.add_argument("--dry-run", action="store_true", help="Only show pending ids; do not call the API.")
    parser.add_argument("--force", action="store_true", help="Ignore the observation window for manual maintenance.")
    args = parser.parse_args()
    stats = label_pending_events(
        limit=args.limit,
        dry_run=args.dry_run,
        respect_observation_window=not args.force,
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
