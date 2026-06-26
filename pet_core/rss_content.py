"""RSSHub-backed external content recommendations.

This module is deliberately separate from the local action recommender.  The
action/strategy model may decide *when* an external recommendation is suitable;
this module only decides *which cached RSS item* is worth showing.
"""

import datetime
import hashlib
import html
import json
import os
import re
import threading
import time
import xml.etree.ElementTree as ET
from email.utils import parsedate_to_datetime
from urllib.parse import parse_qs, unquote, urlparse, urlunparse

import requests
from openai import OpenAI

from pet_core.config import app_config
from pet_core.learning_logger import LEARNING_DATA_DIR


RSS_DATA_DIR = os.path.join(LEARNING_DATA_DIR, "rss_content")
RSS_SOURCES_PATH = os.path.join(RSS_DATA_DIR, "rss_sources.json")
RSS_ITEMS_PATH = os.path.join(RSS_DATA_DIR, "rss_items.json")
RSS_EVENTS_PATH = os.path.join(RSS_DATA_DIR, "rss_events.jsonl")

DEFAULT_RSSHUB_BASE = "https://rsshub.rssforever.com"
DEFAULT_SOURCE_ID = "bilibili_fanshi_63231"
DEFAULT_DYNAMIC_SOURCE_ID = "bilibili_fanshi_dynamic_63231"
REMOVED_DEFAULT_SOURCE_IDS = {"rsshub_36kr_newsflashes", "rsshub_sspai_index"}
WBI_MIXIN_KEY_ENC_TAB = [
    46, 47, 18, 2, 53, 8, 23, 32, 15, 50, 10, 31, 58, 3, 45, 35,
    27, 43, 5, 49, 33, 9, 42, 19, 29, 28, 14, 39, 12, 38, 41, 13,
    37, 48, 7, 16, 24, 55, 40, 61, 26, 17, 0, 1, 60, 51, 30, 4,
    22, 25, 54, 21, 56, 59, 6, 63, 57, 62, 11, 36, 20, 34, 44, 52,
]

_lock = threading.Lock()
_network_backoff_until = 0.0
_network_backoff_reason = ""
NETWORK_BACKOFF_SECONDS = 10 * 60
DEFAULT_CACHE_RETENTION_DAYS = 30
DEFAULT_MAX_CACHED_ITEMS_PER_SOURCE = 80
DEFAULT_CACHE_CLEANUP_INTERVAL_HOURS = 12
DEFAULT_EVENT_RETENTION_DAYS = 60
DEFAULT_MAX_EVENT_LINES = 2000


def _now():
    return datetime.datetime.now()


def _iso(dt=None):
    return (dt or _now()).isoformat(timespec="seconds")


def _truthy(value, default=True):
    if value is None or value == "":
        return default
    return str(value).strip().lower() not in {"0", "false", "no", "off"}


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


def _looks_like_dns_error(exc):
    text = repr(exc).lower()
    markers = [
        "nameresolutionerror",
        "getaddrinfo failed",
        "failed to resolve",
        "failed to establish a new connection",
        "temporary failure in name resolution",
        "errno 11001",
        "winerror 10013",
        "network is unreachable",
    ]
    return any(marker in text for marker in markers)


def _looks_like_timeout_error(exc):
    text = repr(exc).lower()
    markers = [
        "read timed out",
        "connect timeout",
        "timeout",
        "timed out",
    ]
    return any(marker in text for marker in markers)


def _is_pixiv_source(source, url=""):
    platform = str((source or {}).get("platform") or "").lower()
    url = str(url or (source or {}).get("feed_url") or "").lower()
    return platform == "pixiv" or "/pixiv/" in url


def _source_request_timeout(source, timeout):
    url = str((source or {}).get("feed_url") or "")
    if _is_pixiv_source(source, url):
        return max(float(timeout or 0), 25.0)
    return timeout


def _set_network_backoff(exc):
    global _network_backoff_until, _network_backoff_reason
    _network_backoff_until = time.time() + NETWORK_BACKOFF_SECONDS
    _network_backoff_reason = _safe_text(exc, 180)


def _network_backoff_active():
    if time.time() >= _network_backoff_until:
        return False, ""
    remain = int(max(1, _network_backoff_until - time.time()))
    return True, f"network_backoff {remain}s: {_network_backoff_reason}"


def _json_from_text(raw):
    text = str(raw or "").strip()
    if not text:
        return {}
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            obj = json.loads(text[start:end + 1])
            return obj if isinstance(obj, dict) else {}
        except Exception:
            return {}
    return {}


def _allowed_platforms():
    raw = str(app_config.get("rss_recommender.allowed_platforms", "bilibili") or "bilibili")
    values = {
        part.strip().lower()
        for part in re.split(r"[,，\s]+", raw)
        if part.strip()
    }
    return values or {"bilibili"}


def _source_allowed(source):
    if isinstance(source, dict) and source.get("user_added"):
        return True
    allowed = _allowed_platforms()
    if "all" in allowed or "*" in allowed:
        return True
    platform = str((source or {}).get("platform") or "").strip().lower()
    return platform in allowed


def _append_jsonl(path, record):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _cleanup_jsonl_events(path, *, retention_days, max_lines):
    if not os.path.exists(path):
        return {"before": 0, "after": 0, "removed": 0}
    cutoff = _now() - datetime.timedelta(days=retention_days) if retention_days > 0 else None
    before = 0
    kept = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                before += 1
                raw = line.rstrip("\n")
                if not raw:
                    continue
                if cutoff:
                    try:
                        obj = json.loads(raw)
                    except Exception:
                        obj = {}
                    ts = _parse_iso(obj.get("timestamp")) if isinstance(obj, dict) else None
                    if ts and ts < cutoff:
                        continue
                kept.append(raw)
    except Exception:
        return {"before": before, "after": before, "removed": 0, "error": "read_failed"}

    if max_lines > 0 and len(kept) > max_lines:
        kept = kept[-max_lines:]

    removed = max(0, before - len(kept))
    if removed:
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            for line in kept:
                f.write(line + "\n")
        os.replace(tmp, path)
    return {"before": before, "after": len(kept), "removed": removed}


def _read_json(path, default):
    if not os.path.exists(path):
        return default
    try:
        with open(path, "r", encoding="utf-8-sig") as f:
            obj = json.load(f)
        return obj if isinstance(obj, type(default)) else default
    except Exception:
        return default


def _write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def _safe_text(value, limit=1000):
    text = str(value or "").strip()
    text = re.sub(r"\s+", " ", text)
    if len(text) > limit:
        return text[:limit] + "...[truncated]"
    return text


def _strip_html(text):
    text = html.unescape(str(text or ""))
    text = re.sub(r"(?is)<script.*?</script>|<style.*?</style>", " ", text)
    text = re.sub(r"(?s)<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _normalize_image_url(url):
    url = str(url or "").strip()
    if not url:
        return ""
    url = html.unescape(url)
    if url.startswith("//"):
        return "https:" + url
    if url.startswith("http://i") and "hdslb.com" in url:
        return "https://" + url[len("http://"):]
    return url


def _extract_first_image_url(html_text):
    text = str(html_text or "")
    patterns = [
        r"""<img[^>]+(?:src|data-src)=["']([^"']+)["']""",
        r"""<source[^>]+src=["']([^"']+)["']""",
        r"""(https?:)?//i\d?\.hdslb\.com/[^\s"'<>]+""",
    ]
    for pattern in patterns:
        m = re.search(pattern, text, flags=re.I)
        if not m:
            continue
        url = m.group(1) if m.groups() else m.group(0)
        return _normalize_image_url(url)
    return ""


def _item_id(link, title):
    seed = (str(link or "") + "\n" + str(title or "")).strip()
    return "rss_" + hashlib.sha1(seed.encode("utf-8")).hexdigest()[:16]


def _source_id(feed_url, name=""):
    seed = (str(feed_url or "") + "\n" + str(name or "")).strip()
    return "rss_source_" + hashlib.sha1(seed.encode("utf-8")).hexdigest()[:12]


def _normalize_feed_url(feed_url):
    feed_url = str(feed_url or "").strip()
    if feed_url.startswith("rsshub://"):
        path = "/" + feed_url[len("rsshub://"):].lstrip("/")
        base_url = str(app_config.get("rss_recommender.base_url", DEFAULT_RSSHUB_BASE) or DEFAULT_RSSHUB_BASE).rstrip("/")
        return base_url + path
    if feed_url.startswith("/"):
        base_url = str(app_config.get("rss_recommender.base_url", DEFAULT_RSSHUB_BASE) or DEFAULT_RSSHUB_BASE).rstrip("/")
        return base_url + feed_url
    try:
        parsed = urlparse(feed_url)
        local_hosts = {"127.0.0.1", "localhost", "::1"}
        local_port = _int_config("rsshub_local.host_port", 1200, min_value=1, max_value=65535)
        if parsed.scheme == "https" and parsed.hostname in local_hosts and (parsed.port or local_port) == local_port:
            return urlunparse(parsed._replace(scheme="http"))
    except Exception:
        pass
    return feed_url


def _guess_platform(feed_url):
    url = str(feed_url or "").lower()
    if "bilibili.com" in url or "/bilibili/" in url:
        return "bilibili"
    if "pixiv.net" in url or "/pixiv/" in url:
        return "pixiv"
    if "zhihu.com" in url or "/zhihu/" in url:
        return "zhihu"
    if "github.com" in url or "/github/" in url:
        return "github"
    if "arxiv.org" in url:
        return "arxiv"
    if "huggingface.co" in url:
        return "huggingface"
    if "youtube.com" in url or "youtu.be" in url:
        return "youtube"
    if "rsshub" in url:
        return "rsshub"
    return "custom"


def _normalize_platform(platform, feed_url):
    guessed = _guess_platform(feed_url)
    platform = str(platform or "").strip().lower()
    if platform in {"", "auto"}:
        return guessed
    if guessed not in {"custom", "rsshub"} and platform != guessed:
        return guessed
    return platform


def _normalize_tags(tags):
    if isinstance(tags, (list, tuple, set)):
        raw = list(tags)
    else:
        raw = re.split(r"[,，\s]+", str(tags or ""))
    result = []
    for tag in raw:
        tag = str(tag or "").strip()
        if tag and tag not in result:
            result.append(tag[:24])
    return result


def _default_source_max_items(feed_url, platform="", source_type=""):
    url = str(feed_url or "").lower()
    platform = str(platform or "").lower()
    source_type = str(source_type or "").lower()
    if platform == "pixiv" or "/pixiv/" in url:
        return 20
    if source_type == "image":
        return 40
    return 80


def _source_max_items(source, default=80):
    try:
        value = int((source or {}).get("max_items") or 0)
    except Exception:
        value = 0
    if value <= 0:
        value = _default_source_max_items(
            (source or {}).get("feed_url"),
            (source or {}).get("platform"),
            (source or {}).get("type"),
        ) or default
    return max(1, min(80, value))


def _parse_datetime(value):
    if not value:
        return ""
    raw = str(value).strip()
    try:
        return parsedate_to_datetime(raw).astimezone().replace(tzinfo=None).isoformat(timespec="seconds")
    except Exception:
        pass
    try:
        raw = raw.replace("Z", "+00:00")
        return datetime.datetime.fromisoformat(raw).astimezone().replace(tzinfo=None).isoformat(timespec="seconds")
    except Exception:
        return ""


def _parse_iso(value):
    try:
        if not value:
            return None
        return datetime.datetime.fromisoformat(str(value))
    except Exception:
        return None


def _age_hours(value):
    dt = _parse_iso(value)
    if not dt:
        return 9999.0
    return max(0.0, (_now() - dt).total_seconds() / 3600.0)


def _item_time_value(item):
    if not isinstance(item, dict):
        return ""
    return item.get("published_at") or item.get("fetched_at") or ""


def _item_datetime(item):
    if not isinstance(item, dict):
        return None
    return _parse_iso(item.get("fetched_at")) or _parse_iso(item.get("published_at"))


def _source_cache_cap(source, default_cap):
    if not isinstance(source, dict):
        return default_cap
    configured = source.get("max_items")
    if configured not in (None, ""):
        try:
            return max(1, min(default_cap, int(configured)))
        except Exception:
            return default_cap
    return min(default_cap, _source_max_items(source, default=default_cap))


def _prune_item_map(raw_items):
    if not isinstance(raw_items, dict):
        return {}, {"removed_invalid": 0, "removed_expired": 0, "removed_source_cap": 0, "removed_total_cap": 0}

    retention_days = _int_config(
        "rss_recommender.cache_retention_days",
        DEFAULT_CACHE_RETENTION_DAYS,
        min_value=0,
        max_value=3650,
    )
    max_items = _int_config("rss_recommender.max_cached_items", 500, min_value=50, max_value=5000)
    per_source_default = _int_config(
        "rss_recommender.max_cached_items_per_source",
        DEFAULT_MAX_CACHED_ITEMS_PER_SOURCE,
        min_value=5,
        max_value=1000,
    )

    source_caps = {}
    try:
        for source in ensure_sources():
            sid = str((source or {}).get("id") or "")
            if sid:
                source_caps[sid] = _source_cache_cap(source, per_source_default)
    except Exception:
        source_caps = {}

    cutoff = _now() - datetime.timedelta(days=retention_days) if retention_days > 0 else None
    stats = {
        "removed_invalid": 0,
        "removed_expired": 0,
        "removed_source_cap": 0,
        "removed_total_cap": 0,
    }

    candidates = []
    for key, item in raw_items.items():
        if not isinstance(item, dict):
            stats["removed_invalid"] += 1
            continue
        dt = _item_datetime(item)
        if cutoff and dt and dt < cutoff:
            stats["removed_expired"] += 1
            continue
        candidates.append((str(key), item))

    grouped = {}
    for key, item in candidates:
        sid = str(item.get("source_id") or "")
        grouped.setdefault(sid, []).append((key, item))

    kept = []
    for sid, pairs in grouped.items():
        cap = source_caps.get(sid, per_source_default)
        pairs.sort(key=lambda kv: _item_time_value(kv[1]), reverse=True)
        kept.extend(pairs[:cap])
        stats["removed_source_cap"] += max(0, len(pairs) - cap)

    kept.sort(key=lambda kv: _item_time_value(kv[1]), reverse=True)
    if len(kept) > max_items:
        stats["removed_total_cap"] = len(kept) - max_items
        kept = kept[:max_items]

    return dict(kept), stats


def _default_sources():
    base_url = str(app_config.get("rss_recommender.base_url", DEFAULT_RSSHUB_BASE) or DEFAULT_RSSHUB_BASE).rstrip("/")
    return [
        {
            "id": DEFAULT_SOURCE_ID,
            "name": "泛式 Bilibili 视频",
            "type": "video",
            "platform": "bilibili",
            "feed_url": f"{base_url}/bilibili/user/video/63231/noembed",
            "homepage": "https://space.bilibili.com/63231",
            "tags": ["杂谈", "动画", "二次元", "游戏", "视频"],
            "priority": 0.82,
            "enabled": True,
        },
        {
            "id": DEFAULT_DYNAMIC_SOURCE_ID,
            "name": "泛式 Bilibili 动态",
            "type": "dynamic",
            "platform": "bilibili",
            "feed_url": f"{base_url}/bilibili/user/dynamic/63231/embed=0&directLink=1&hideGoods=1",
            "homepage": "https://space.bilibili.com/63231/dynamic",
            "tags": ["动态", "杂谈", "动画", "二次元", "游戏", "视频"],
            "priority": 0.9,
            "enabled": True,
        },
    ]


def ensure_sources():
    os.makedirs(RSS_DATA_DIR, exist_ok=True)
    sources = _read_json(RSS_SOURCES_PATH, [])
    if not sources:
        sources = _default_sources()
        _write_json(RSS_SOURCES_PATH, sources)
        return sources

    cleaned_sources = []
    removed = False
    for source in sources:
        if isinstance(source, dict) and str(source.get("id") or "") in REMOVED_DEFAULT_SOURCE_IDS:
            removed = True
            continue
        cleaned_sources.append(source)
    sources = cleaned_sources

    known = {str(s.get("id") or "") for s in sources if isinstance(s, dict)}
    changed = removed
    for source in sources:
        if not isinstance(source, dict):
            continue
        normalized_feed_url = _normalize_feed_url(source.get("feed_url"))
        if source.get("feed_url") != normalized_feed_url:
            source["feed_url"] = normalized_feed_url
            changed = True
        normalized_platform = _normalize_platform(source.get("platform"), source.get("feed_url"))
        if source.get("platform") != normalized_platform:
            source["platform"] = normalized_platform
            changed = True
        if _is_pixiv_source(source):
            try:
                pixiv_max_items = int(source.get("max_items") or 0)
            except Exception:
                pixiv_max_items = 0
            if pixiv_max_items > 20:
                source["max_items"] = 20
                changed = True
    for src in _default_sources():
        if src["id"] not in known:
            sources.append(src)
            changed = True
        else:
            existing = next((s for s in sources if isinstance(s, dict) and str(s.get("id") or "") == src["id"]), None)
            if existing and not existing.get("user_added"):
                for key in ("name", "type", "platform", "feed_url", "homepage", "tags", "priority", "enabled"):
                    if existing.get(key) != src.get(key):
                        existing[key] = src.get(key)
                        changed = True
    if changed:
        _write_json(RSS_SOURCES_PATH, sources)
    return sources


def list_sources():
    return list(ensure_sources())


def save_sources(sources):
    cleaned = []
    seen = set()
    for source in sources or []:
        if not isinstance(source, dict):
            continue
        source_id = str(source.get("id") or "").strip()
        if not source_id or source_id in seen:
            continue
        seen.add(source_id)
        cleaned.append(source)
    _write_json(RSS_SOURCES_PATH, cleaned)
    return cleaned


def add_source(*, name, feed_url, source_type="feed", platform="", tags=None, priority=0.7, enabled=True, max_items=None):
    feed_url = _normalize_feed_url(feed_url)
    if not re.match(r"^https?://", feed_url, re.I):
        raise ValueError("RSS 链接必须以 http://、https://、/bilibili/... 或 rsshub:// 开头")
    name = _safe_text(name or feed_url, 80)
    platform = _safe_text(_normalize_platform(platform, feed_url), 40)
    source_type = _safe_text(source_type or "feed", 32)
    source_id = _source_id(feed_url, name)
    try:
        priority = float(priority)
    except Exception:
        priority = 0.7
    priority = max(0.0, min(1.0, priority))
    try:
        max_items_value = int(max_items) if max_items not in (None, "") else 0
    except Exception:
        max_items_value = 0
    if max_items_value <= 0:
        max_items_value = _default_source_max_items(feed_url, platform, source_type)
    max_items_value = max(1, min(80, int(max_items_value)))

    sources = ensure_sources()
    existing = next((s for s in sources if isinstance(s, dict) and s.get("feed_url") == feed_url), None)
    payload = {
        "id": existing.get("id") if existing else source_id,
        "name": name,
        "type": source_type,
        "platform": platform,
        "feed_url": feed_url,
        "homepage": "",
        "tags": _normalize_tags(tags),
        "priority": priority,
        "max_items": max_items_value,
        "enabled": bool(enabled),
        "user_added": True,
        "created_at": existing.get("created_at") if existing else _iso(),
        "updated_at": _iso(),
    }
    if existing:
        existing.update(payload)
    else:
        sources.append(payload)
    save_sources(sources)
    return payload


def update_source(source_id, **updates):
    source_id = str(source_id or "")
    sources = ensure_sources()
    for source in sources:
        if not isinstance(source, dict) or str(source.get("id") or "") != source_id:
            continue
        allowed = {"name", "type", "platform", "feed_url", "homepage", "tags", "priority", "enabled", "max_items"}
        for key, value in updates.items():
            if key not in allowed:
                continue
            if key == "tags":
                value = _normalize_tags(value)
            elif key == "priority":
                try:
                    value = max(0.0, min(1.0, float(value)))
                except Exception:
                    value = source.get("priority", 0.7)
            elif key == "enabled":
                value = bool(value)
            elif key == "max_items":
                try:
                    value = max(1, min(80, int(value)))
                except Exception:
                    value = _source_max_items(source)
            elif key == "feed_url":
                value = _normalize_feed_url(value)
                if not re.match(r"^https?://", value, re.I):
                    raise ValueError("RSS 链接必须以 http://、https://、/bilibili/... 或 rsshub:// 开头")
            else:
                value = _safe_text(value, 500)
            source[key] = value
        source["platform"] = _normalize_platform(source.get("platform"), source.get("feed_url"))
        if not source.get("max_items"):
            source["max_items"] = _source_max_items(source)
        source["updated_at"] = _iso()
        save_sources(sources)
        return source
    raise KeyError(f"RSS source not found: {source_id}")


def delete_source(source_id, *, remove_items=True):
    source_id = str(source_id or "")
    sources = ensure_sources()
    kept = [s for s in sources if not (isinstance(s, dict) and str(s.get("id") or "") == source_id)]
    if len(kept) == len(sources):
        return False
    save_sources(kept)
    if remove_items:
        state = _load_items()
        items = state.setdefault("items", {})
        if isinstance(items, dict):
            state["items"] = {
                key: item for key, item in items.items()
                if not (isinstance(item, dict) and str(item.get("source_id") or "") == source_id)
            }
            _save_items(state)
    return True


def _children(node, local_name):
    return [child for child in list(node) if child.tag.split("}")[-1] == local_name]


def _child_text(node, local_name):
    for child in _children(node, local_name):
        return "".join(child.itertext()).strip()
    return ""


def _entry_link(entry):
    for child in _children(entry, "link"):
        href = child.attrib.get("href")
        if href:
            return href.strip()
        text = "".join(child.itertext()).strip()
        if text:
            return text
    return _child_text(entry, "link")


def _parse_feed(xml_text, source):
    root = ET.fromstring(xml_text)
    source_id = str(source.get("id") or "")
    source_tags = list(source.get("tags") or [])
    source_priority = float(source.get("priority") or 0.5)
    items = []

    rss_items = root.findall(".//item")
    atom_entries = [node for node in root.iter() if node.tag.split("}")[-1] == "entry"]
    entries = rss_items or atom_entries

    for entry in entries[:_source_max_items(source)]:
        is_atom = entry.tag.split("}")[-1] == "entry"
        title = _safe_text(_child_text(entry, "title"), 220)
        link = _safe_text(_entry_link(entry), 500)
        summary = (
            _child_text(entry, "description")
            or _child_text(entry, "summary")
            or _child_text(entry, "content")
        )
        cover_url = _extract_first_image_url(summary)
        summary = _safe_text(_strip_html(summary), 700)
        published = (
            _child_text(entry, "pubDate")
            or _child_text(entry, "published")
            or _child_text(entry, "updated")
        )
        published_at = _parse_datetime(published)
        if not title and not link:
            continue
        item = {
            "id": _item_id(link, title),
            "source_id": source_id,
            "source_name": source.get("name") or source_id,
            "source_type": source.get("type") or ("article" if not is_atom else "feed"),
            "platform": source.get("platform") or "",
            "title": title,
            "url": link,
            "summary": summary,
            "cover_url": cover_url,
            "published_at": published_at,
            "fetched_at": _iso(),
            "tags": source_tags,
            "source_priority": source_priority,
            "homepage": source.get("homepage") or "",
        }
        items.append(item)
    return items


def _bili_cookie():
    return (
        str(app_config.get("rsshub_local.bilibili_cookie", "") or "").strip()
        or str(app_config.get("rss_recommender.bilibili_cookie", "") or "").strip()
    )


def _bili_headers(referer="https://www.bilibili.com/"):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/126 Safari/537.36",
        "Referer": referer,
        "Origin": "https://www.bilibili.com",
    }
    cookie = _bili_cookie()
    if cookie:
        headers["Cookie"] = cookie
    return headers


def _has_bili_cookie():
    return bool(_bili_cookie())


def _has_pixiv_token():
    return bool(str(app_config.get("rsshub_local.pixiv_refresh_token", "") or "").strip())


def _missing_background_credentials(source):
    url = str((source or {}).get("feed_url") or "").lower()
    platform = str((source or {}).get("platform") or "").lower()
    if (platform == "bilibili" or "/bilibili/" in url) and (
        "/bilibili/user/video/" in url or "/bilibili/user/dynamic/" in url
    ):
        if not _has_bili_cookie():
            return "缺少 Bilibili Cookie，后台跳过 UP 投稿/动态源"
    if platform == "pixiv" or "/pixiv/" in url:
        if not _has_pixiv_token():
            return "缺少 PIXIV_REFRESHTOKEN，后台跳过 Pixiv 源"
    return ""


def _bili_video_url(item):
    bvid = item.get("bvid") or item.get("param")
    aid = item.get("aid") or item.get("id")
    if bvid and str(bvid).startswith("BV"):
        return f"https://www.bilibili.com/video/{bvid}"
    if aid:
        return f"https://www.bilibili.com/video/av{aid}"
    return str(item.get("arcurl") or item.get("url") or "")


def _bili_ts(value):
    try:
        if not value:
            return ""
        return datetime.datetime.fromtimestamp(int(value)).isoformat(timespec="seconds")
    except Exception:
        return ""


def _item_from_bili_video(raw, source, *, summary_extra=""):
    owner = raw.get("owner") or {}
    author = raw.get("author") or raw.get("uname") or owner.get("name") or ""
    title = _safe_text(raw.get("title") or "", 220)
    url = _safe_text(_bili_video_url(raw), 500)
    desc = raw.get("desc") or raw.get("description") or ""
    if summary_extra:
        desc = f"{summary_extra} {desc}".strip()
    cover = _normalize_image_url(raw.get("pic") or raw.get("cover") or "")
    pub_ts = raw.get("pubdate") or raw.get("ctime") or raw.get("created")
    tags = list(source.get("tags") or [])
    if raw.get("tname") and raw.get("tname") not in tags:
        tags.append(raw.get("tname"))
    return {
        "id": _item_id(url, title),
        "source_id": str(source.get("id") or ""),
        "source_name": source.get("name") or source.get("id") or "",
        "source_type": source.get("type") or "video",
        "platform": source.get("platform") or "bilibili",
        "title": title,
        "url": url,
        "summary": _safe_text(_strip_html(desc), 700),
        "cover_url": cover,
        "published_at": _bili_ts(pub_ts),
        "fetched_at": _iso(),
        "tags": tags,
        "source_priority": float(source.get("priority") or 0.5),
        "homepage": source.get("homepage") or "",
        "author": author,
    }


def _bili_ranking_api(rid):
    rid_map = {
        "all": 0,
        "douga": 1005,
        "game": 1008,
        "kichiku": 1007,
        "music": 1003,
        "dance": 1004,
        "cinephile": 1001,
        "ent": 1002,
        "knowledge": 1010,
        "tech": 1012,
        "food": 1020,
        "car": 1013,
        "fashion": 1014,
        "sports": 1018,
        "animal": 1024,
    }
    numeric = str(rid or "all")
    if not numeric.isdigit():
        numeric = str(rid_map.get(numeric, 0))
    return f"https://api.bilibili.com/x/web-interface/ranking/v2?rid={numeric}&type=all&web_location=333.934"


def _fetch_json(url, *, referer="https://www.bilibili.com/", timeout=8):
    resp = requests.get(url, timeout=timeout, headers=_bili_headers(referer))
    resp.raise_for_status()
    data = resp.json()
    if isinstance(data, dict) and data.get("code") not in (0, None):
        raise RuntimeError(f"Bilibili API error {data.get('code')}: {data.get('message')}")
    return data


def _wbi_mixin_key(timeout=8):
    data = _fetch_json("https://api.bilibili.com/x/web-interface/nav", timeout=timeout)
    wbi = ((data or {}).get("data") or {}).get("wbi_img") or {}
    img_key = (wbi.get("img_url") or "").split("/")[-1].split(".")[0]
    sub_key = (wbi.get("sub_url") or "").split("/")[-1].split(".")[0]
    raw = img_key + sub_key
    if len(raw) < 64:
        return ""
    return "".join(raw[i] for i in WBI_MIXIN_KEY_ENC_TAB)[:32]


def _signed_wbi_params(params, timeout=8):
    mixin_key = _wbi_mixin_key(timeout=timeout)
    if not mixin_key:
        raise RuntimeError("无法获取 WBI key，可能需要登录 Cookie")
    signed = {k: str(v) for k, v in dict(params or {}).items() if v is not None}
    signed["wts"] = str(int(time.time()))
    safe = {}
    for key, value in signed.items():
        safe[key] = re.sub(r"[!'()*]", "", str(value))
    query = requests.compat.urlencode(dict(sorted(safe.items())))
    safe["w_rid"] = hashlib.md5((query + mixin_key).encode("utf-8")).hexdigest()
    return safe


def _fetch_bili_user_videos(uid, timeout=8):
    params = {
        "mid": uid,
        "ps": 30,
        "tid": 0,
        "pn": 1,
        "keyword": "",
        "order": "pubdate",
        "platform": "web",
        "web_location": "1550101",
        "order_avoided": "true",
    }
    try:
        signed = _signed_wbi_params(params, timeout=timeout)
        data = _fetch_json(
            "https://api.bilibili.com/x/space/wbi/arc/search?"
            + requests.compat.urlencode(signed),
            referer=f"https://space.bilibili.com/{uid}/video",
            timeout=timeout,
        )
    except Exception:
        data = _fetch_json(
            f"https://api.bilibili.com/x/space/arc/search?mid={uid}&ps=30&pn=1",
            referer=f"https://space.bilibili.com/{uid}/video",
            timeout=timeout,
        )
    return (((data or {}).get("data") or {}).get("list") or {}).get("vlist") or []


def _parse_bilibili_route(feed_url):
    parsed = urlparse(str(feed_url or ""))
    parts = [unquote(p) for p in parsed.path.split("/") if p]
    try:
        idx = parts.index("bilibili")
    except ValueError:
        return None
    route = parts[idx + 1:]
    return route, parse_qs(parsed.query)


def _fetch_bilibili_fallback(source, timeout=8):
    parsed = _parse_bilibili_route(source.get("feed_url") or "")
    if not parsed:
        return None
    route, query = parsed
    if not route:
        return None

    if route[:2] == ["popular", "all"]:
        data = _fetch_json("https://api.bilibili.com/x/web-interface/popular", timeout=timeout)
        rows = (((data or {}).get("data") or {}).get("list") or [])[:40]
        return [_item_from_bili_video(row, source) for row in rows]

    if route[0] == "ranking":
        rid = route[1] if len(route) > 1 and route[1] != "noembed" else "all"
        data = _fetch_json(_bili_ranking_api(rid), timeout=timeout)
        rows = (((data or {}).get("data") or {}).get("list") or [])[:40]
        return [_item_from_bili_video(row, source) for row in rows]

    if route[0] == "weekly":
        series = _fetch_json(
            "https://app.bilibili.com/x/v2/show/popular/selected/series?type=weekly_selected",
            referer="https://www.bilibili.com/h5/weekly-recommend",
            timeout=timeout,
        )
        first = ((series or {}).get("data") or [{}])[0]
        number = first.get("number")
        weekly_name = first.get("name") or ""
        if not number:
            return []
        data = _fetch_json(
            f"https://app.bilibili.com/x/v2/show/popular/selected?type=weekly_selected&number={number}",
            referer=f"https://www.bilibili.com/h5/weekly-recommend?num={number}&navhide=1",
            timeout=timeout,
        )
        rows = (((data or {}).get("data") or {}).get("list") or [])[:40]
        items = []
        for row in rows:
            row = dict(row or {})
            row.setdefault("bvid", row.get("bvid"))
            row.setdefault("aid", row.get("param"))
            row.setdefault("pic", row.get("cover"))
            items.append(_item_from_bili_video(row, source, summary_extra=weekly_name or row.get("rcmd_reason") or ""))
        return items

    if route[0] == "vsearch" and len(route) >= 2:
        keyword = route[1]
        order = route[2] if len(route) >= 3 and route[2] not in {"noembed", ""} else "pubdate"
        tid = route[4] if len(route) >= 5 else (query.get("tid", ["0"])[0])
        api = (
            "https://api.bilibili.com/x/web-interface/search/type"
            f"?search_type=video&keyword={requests.utils.quote(keyword)}&order={order}&tids={tid or 0}"
        )
        data = _fetch_json(api, referer=f"https://search.bilibili.com/all?keyword={requests.utils.quote(keyword)}", timeout=timeout)
        rows = (((data or {}).get("data") or {}).get("result") or [])[:40]
        return [_item_from_bili_video(row, source) for row in rows]

    if len(route) >= 3 and route[0] == "user" and route[1] == "video":
        uid = route[2]
        if not _has_bili_cookie():
            raise RuntimeError("UP 主投稿接口被风控，需要在设置里填写 Bilibili Cookie 后再试")
        try:
            rows = _fetch_bili_user_videos(uid, timeout=timeout)
        except Exception as e:
            if not _has_bili_cookie():
                raise RuntimeError(f"UP 主投稿接口被风控，需要在设置里填写 Bilibili Cookie 后再试: {e}") from e
            raise
        return [_item_from_bili_video(row, source) for row in rows[:40]]

    if len(route) >= 3 and route[0] == "user" and route[1] == "dynamic":
        uid = route[2]
        if not _has_bili_cookie():
            raise RuntimeError("UP 主动态接口被风控，需要在设置里填写 Bilibili Cookie 后再试")
        api = (
            "https://api.bilibili.com/x/polymer/web-dynamic/v1/feed/space"
            f"?host_mid={uid}&platform=web&features=itemOpusStyle,listOnlyfans,opusBigCover,onlyfansVote"
        )
        try:
            data = _fetch_json(api, referer=f"https://space.bilibili.com/{uid}/dynamic", timeout=timeout)
        except Exception as e:
            if not _has_bili_cookie():
                raise RuntimeError(f"UP 主动态接口被风控，需要在设置里填写 Bilibili Cookie 后再试: {e}") from e
            raise
        rows = (((data or {}).get("data") or {}).get("items") or [])[:40]
        items = []
        for row in rows:
            modules = row.get("modules") or {}
            author = modules.get("module_author") or {}
            dynamic = modules.get("module_dynamic") or {}
            major = dynamic.get("major") or {}
            desc = (dynamic.get("desc") or {}).get("text") or ""
            title = desc[:80]
            cover = ""
            url = f"https://t.bilibili.com/{row.get('id_str')}" if row.get("id_str") else ""
            archive = major.get("archive") or {}
            if archive:
                title = archive.get("title") or title
                url = _bili_video_url(archive) or url
                cover = archive.get("cover") or archive.get("pic") or ""
                desc = f"{desc} {archive.get('desc') or ''}".strip()
            opus = major.get("opus") or {}
            if opus:
                title = ((opus.get("summary") or {}).get("text") or title)[:80]
                pics = opus.get("pics") or []
                if pics:
                    cover = pics[0].get("url") or cover
            item = {
                "aid": archive.get("aid"),
                "bvid": archive.get("bvid"),
                "title": title or f"{author.get('name') or uid} 的动态",
                "desc": desc,
                "pic": cover,
                "pubdate": author.get("pub_ts"),
                "owner": {"name": author.get("name") or ""},
            }
            converted = _item_from_bili_video(item, source)
            converted["url"] = url or converted.get("url")
            converted["id"] = _item_id(converted.get("url"), converted.get("title"))
            converted["source_type"] = "dynamic"
            items.append(converted)
        return items

    return None


def _fetch_source_items(source, timeout):
    url = _normalize_feed_url(source.get("feed_url") or "")
    if not url:
        return []
    request_timeout = _source_request_timeout(source, timeout)
    try:
        resp = requests.get(url, timeout=request_timeout, headers={"User-Agent": "YuzuDeskpetRSS/1.0 (+local desktop pet)"})
        resp.raise_for_status()
        return _parse_feed(resp.text, source)
    except Exception as rss_error:
        fallback_items = None
        platform = str(source.get("platform") or "").lower()
        if platform == "bilibili" or "/bilibili/" in url:
            try:
                fallback_items = _fetch_bilibili_fallback(source, timeout=timeout)
            except Exception as fallback_error:
                raise RuntimeError(f"{rss_error}; Bilibili fallback failed: {fallback_error}") from fallback_error
        if fallback_items is not None:
            return fallback_items
        if _is_pixiv_source(source, url):
            if _looks_like_timeout_error(rss_error):
                raise RuntimeError(
                    f"{rss_error}; Pixiv 本地 RSSHub 响应超时。"
                    f"当前已按 Pixiv 源放宽到 {request_timeout:.0f} 秒，"
                    "如果仍失败，多半是 Pixiv/RSSHub 正在慢速响应或网络代理不稳定，稍后再刷新即可。"
                ) from rss_error
            if not _has_pixiv_token():
                raise RuntimeError(
                    f"{rss_error}; Pixiv 路由需要本地 RSSHub 容器配置 PIXIV_REFRESHTOKEN。"
                    "请在设置里填写后重建/重启 RSSHub 容器。"
                ) from rss_error
            raise RuntimeError(
                f"{rss_error}; Pixiv 路由请求失败。若 token 已配置，通常是 Pixiv/RSSHub 临时风控、网络代理或路由参数问题。"
            ) from rss_error
        if platform == "zhihu" or "/zhihu/" in url:
            raise RuntimeError(
                f"{rss_error}; 知乎路由在公共 RSSHub 镜像上经常受反爬影响。"
                "建议换可用 RSSHub 镜像、自建 RSSHub，或添加知乎相关的公开 RSS/Atom 源。"
            ) from rss_error
        raise


def _load_items():
    obj = _read_json(RSS_ITEMS_PATH, {"schema_version": "rss_content.v1", "items": {}})
    obj.setdefault("schema_version", "rss_content.v1")
    obj.setdefault("items", {})
    return obj


def _save_items(state):
    items, stats = _prune_item_map(state.get("items") or {})
    removed = sum(int(v or 0) for v in stats.values())
    state["items"] = items
    state["last_cleanup_checked_at"] = _iso()
    state["last_cleanup_stats"] = stats
    if removed:
        state["last_cleanup_at"] = _iso()
    _write_json(RSS_ITEMS_PATH, state)


def cleanup_cached_items():
    state = _load_items()
    before = len((state.get("items") or {}) if isinstance(state.get("items"), dict) else {})
    _save_items(state)
    after = len((state.get("items") or {}) if isinstance(state.get("items"), dict) else {})
    stats = state.get("last_cleanup_stats") or {}
    event_stats = _cleanup_jsonl_events(
        RSS_EVENTS_PATH,
        retention_days=_int_config(
            "rss_recommender.event_retention_days",
            DEFAULT_EVENT_RETENTION_DAYS,
            min_value=1,
            max_value=3650,
        ),
        max_lines=_int_config(
            "rss_recommender.max_event_lines",
            DEFAULT_MAX_EVENT_LINES,
            min_value=100,
            max_value=100000,
        ),
    )
    return {
        "before": before,
        "after": after,
        "removed": max(0, before - after),
        "stats": stats,
        "event_stats": event_stats,
        "last_cleanup_at": state.get("last_cleanup_at"),
    }


def refresh_sources(force=False, source_ids=None, ignore_platform_filter=False):
    if not _truthy(app_config.get("rss_recommender.enabled", "true"), default=True):
        return {"ok": False, "reason": "disabled", "added": 0, "updated": 0}

    selected_ids = {str(x) for x in (source_ids or []) if str(x)}
    if not selected_ids:
        active, reason = _network_backoff_active()
        if active:
            return {"ok": False, "reason": reason, "added": 0, "updated": 0, "errors": [reason]}

    cooldown = _int_config("rss_recommender.refresh_interval_minutes", 120, min_value=10, max_value=1440)
    state = _load_items()
    last_refresh = _parse_iso(state.get("last_refresh_at"))
    if not force and last_refresh and (_now() - last_refresh).total_seconds() < cooldown * 60:
        return {"ok": True, "reason": "cooldown", "added": 0, "updated": 0}

    timeout = _float_config("rss_recommender.request_timeout_seconds", 8, min_value=2, max_value=30)
    if not selected_ids:
        timeout = min(timeout, 4.0)
    all_sources = ensure_sources()
    sources = [
        s for s in all_sources
        if (
            isinstance(s, dict)
            and s.get("enabled", True)
            and (not selected_ids or str(s.get("id") or "") in selected_ids)
            and (ignore_platform_filter or _source_allowed(s))
        )
    ]
    added = 0
    updated = 0
    errors = []
    if not _lock.acquire(blocking=False):
        return {"ok": True, "reason": "refresh_already_running", "added": 0, "updated": 0, "errors": []}

    try:
        state = _load_items()
        items = state.setdefault("items", {})
        for source in sources:
            url = str(source.get("feed_url") or "").strip()
            if not url:
                continue
            if not selected_ids:
                missing_credentials = _missing_background_credentials(source)
                if missing_credentials:
                    source["last_error"] = missing_credentials
                    continue
            try:
                parsed = _fetch_source_items(source, timeout)
                for item in parsed:
                    old = items.get(item["id"])
                    if old:
                        old.update({k: v for k, v in item.items() if v or k in {"fetched_at"}})
                        updated += 1
                    else:
                        items[item["id"]] = item
                        added += 1
                source["last_ok_at"] = _iso()
                source["last_error"] = ""
            except Exception as e:
                msg = f"{source.get('name') or source.get('id')}: {e}"
                errors.append(msg)
                source["last_error"] = str(e)
                print(f"[RSS] refresh failed: {msg}")
                if _looks_like_dns_error(e):
                    _set_network_backoff(e)
                    if not selected_ids:
                        print("[RSS] 检测到 DNS/网络不可用，暂停后台 RSS 刷新一段时间。")
                        break
        state["last_refresh_at"] = _iso()
        _save_items(state)
        _write_json(RSS_SOURCES_PATH, all_sources)
    finally:
        _lock.release()

    result = {
        "ok": (not errors) or (added + updated > 0),
        "reason": "refreshed",
        "added": added,
        "updated": updated,
        "errors": errors,
    }
    _append_jsonl(RSS_EVENTS_PATH, {"timestamp": _iso(), "type": "refresh", "result": result})
    return result


def refresh_source(source_id, *, force=True):
    return refresh_sources(force=force, source_ids=[source_id], ignore_platform_filter=True)


def refresh_sources_background(force=False):
    thread = threading.Thread(target=lambda: refresh_sources(force=force), daemon=True)
    thread.start()
    return thread


def _tokens(text):
    text = str(text or "").lower()
    tokens = set()
    for word in re.findall(r"[a-z0-9_+\-.]{2,}", text):
        tokens.add(word[:40])
    cjk = "".join(re.findall(r"[\u4e00-\u9fff]", text))
    for n in (2, 3):
        if len(cjk) < n:
            continue
        for i in range(min(len(cjk) - n + 1, 280)):
            tokens.add(cjk[i:i+n])
    return tokens


def _explicit_content_request(text):
    s = str(text or "")
    markers = (
        "推荐视频", "推荐内容", "推荐文章", "推荐播客", "推荐点", "推点",
        "有什么好看", "有什么有意思", "有啥好看", "有啥有意思",
        "想看视频", "看点东西", "杂谈", "RSS", "rss",
    )
    return any(m in s for m in markers)


def _recently_recommended_ids(days=7):
    result = set()
    if not os.path.exists(RSS_EVENTS_PATH):
        return result
    cutoff = _now() - datetime.timedelta(days=days)
    try:
        with open(RSS_EVENTS_PATH, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                if obj.get("type") != "recommendation":
                    continue
                ts = _parse_iso(obj.get("timestamp"))
                if ts and ts < cutoff:
                    continue
                item_id = ((obj.get("item") or {}).get("id") or obj.get("item_id") or "")
                if item_id:
                    result.add(str(item_id))
    except Exception:
        pass
    return result


def list_cached_items(source_id=None, query="", limit=300):
    state = _load_items()
    items = list((state.get("items") or {}).values())
    source_id = str(source_id or "")
    if source_id:
        items = [item for item in items if isinstance(item, dict) and str(item.get("source_id") or "") == source_id]
    if query:
        tokens = _tokens(query)
        if tokens:
            filtered = []
            for item in items:
                text = " ".join([
                    str(item.get("title") or ""),
                    str(item.get("summary") or ""),
                    str(item.get("url") or ""),
                    str(item.get("source_name") or ""),
                    " ".join(item.get("tags") or []),
                ])
                if _tokens(text) & tokens:
                    filtered.append(item)
            items = filtered
    items.sort(key=lambda item: item.get("published_at") or item.get("fetched_at") or "", reverse=True)
    return items[:max(1, int(limit or 300))]


def clear_cached_items(source_id=None):
    state = _load_items()
    items = state.setdefault("items", {})
    if not isinstance(items, dict):
        state["items"] = {}
    elif source_id:
        sid = str(source_id)
        state["items"] = {
            key: item for key, item in items.items()
            if not (isinstance(item, dict) and str(item.get("source_id") or "") == sid)
        }
    else:
        state["items"] = {}
    _save_items(state)
    return True


def _score_item(item, query_tokens, profile_tokens, recent_ids):
    text = " ".join([
        item.get("title") or "",
        item.get("summary") or "",
        " ".join(item.get("tags") or []),
        item.get("source_name") or "",
    ])
    item_tokens = _tokens(text)
    if not item_tokens:
        return 0.0, {}

    query_hits = item_tokens & query_tokens
    profile_hits = item_tokens & profile_tokens
    query_score = len(query_hits) / max(4.0, len(query_tokens) ** 0.5 if query_tokens else 4.0)
    profile_score = len(profile_hits) / max(8.0, len(profile_tokens) ** 0.5 if profile_tokens else 8.0)
    age = _age_hours(item.get("published_at") or item.get("fetched_at"))
    recency = 1.0 / (1.0 + age / 168.0)
    source_priority = max(0.0, min(1.0, float(item.get("source_priority") or 0.5)))
    novelty = 0.55 if item.get("id") in recent_ids else 1.0
    summary_bonus = 0.08 if item.get("summary") else 0.0
    base = 0.42 * query_score + 0.28 * profile_score + 0.18 * recency + 0.12 * source_priority + summary_bonus
    score = max(0.0, min(1.0, base * novelty))
    return score, {
        "query_hits": sorted(query_hits)[:10],
        "profile_hits": sorted(profile_hits)[:10],
        "recency": round(recency, 3),
        "source_priority": round(source_priority, 3),
        "novelty": novelty,
    }


def _reviewer_config():
    enabled = _truthy(app_config.get("rss_recommender.review_with_llm", "true"), default=True)
    api_key = (
        app_config.get("rss_recommender.reviewer_api_key", "")
        or app_config.get("recommendation_generator.api_key", "")
        or app_config.get("learning_labeler.api_key", "")
        or app_config.get("profile_refiner.api_key", "")
        or app_config.get("memory_reranker.api_key", "")
        or ""
    )
    base_url = (
        app_config.get("rss_recommender.reviewer_base_url", "")
        or app_config.get("recommendation_generator.base_url", "")
        or app_config.get("learning_labeler.base_url", "")
        or app_config.get("profile_refiner.base_url", "")
        or "https://api.deepseek.com"
    )
    model = (
        app_config.get("rss_recommender.reviewer_model", "")
        or app_config.get("recommendation_generator.model", "")
        or app_config.get("learning_labeler.model", "")
        or app_config.get("profile_refiner.model", "")
        or "deepseek-chat"
    )
    return {
        "enabled": enabled,
        "api_key": api_key,
        "base_url": base_url,
        "model": model,
        "timeout_seconds": _int_config("rss_recommender.reviewer_timeout_seconds", 10, min_value=3, max_value=40),
    }


def _llm_review_candidates(*, user_text, user_profile, recent_context, candidates):
    cfg = _reviewer_config()
    if not cfg["enabled"] or not cfg["api_key"] or not candidates:
        return None

    candidate_payload = []
    for idx, candidate in enumerate(candidates, 1):
        item = candidate.get("item") or {}
        candidate_payload.append({
            "rank": idx,
            "id": item.get("id"),
            "title": item.get("title"),
            "url": item.get("url"),
            "summary": _safe_text(item.get("summary"), 500),
            "source": item.get("source_name"),
            "type": item.get("source_type"),
            "platform": item.get("platform"),
            "cover_url": item.get("cover_url"),
            "published_at": item.get("published_at"),
            "tags": item.get("tags") or [],
            "local_score": candidate.get("score"),
            "local_match": candidate.get("match") or {},
        })

    recent_text = []
    for turn in list(recent_context or [])[-6:]:
        if not isinstance(turn, dict):
            continue
        user = _safe_text(turn.get("user"), 180)
        assistant = _safe_text(turn.get("assistant_summary"), 180)
        if user or assistant:
            recent_text.append({"user": user, "assistant_summary": assistant, "minutes_ago": turn.get("minutes_ago")})

    prompt = {
        "task": "从 RSS 候选里挑一个最适合现在推荐给用户的 B 站内容；如果都不贴合，返回 selected_id 为空字符串。",
        "rules": [
            "只基于候选的标题、简介、来源、原链接、发布时间和本地匹配线索判断；不要假装完整看过视频或动态。",
            "输出必须是严格 JSON，不要 Markdown。",
            "推荐理由要说明为什么和用户当前状态/画像/请求相关，短而具体。",
            "如果用户只是要链接，仍然保留 reason，但不要编造内容细节。",
        ],
        "user_text": user_text,
        "user_profile": _safe_text(user_profile, 1200),
        "recent_context": recent_text,
        "candidates": candidate_payload,
        "output_schema": {
            "selected_id": "候选 id；不推荐则空字符串",
            "reason": "推荐原因，40-120字",
            "one_line_summary": "根据标题简介写一句内容概述，不能声称看完",
            "confidence": 0.0,
        },
    }

    try:
        client = OpenAI(
            api_key=cfg["api_key"],
            base_url=cfg["base_url"],
            timeout=float(cfg["timeout_seconds"]),
            max_retries=0,
        )
        response = client.chat.completions.create(
            model=cfg["model"],
            messages=[
                {"role": "system", "content": "你是外部内容推荐审阅器，只做候选选择和推荐理由生成。必须输出严格 JSON。"},
                {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
            ],
            temperature=0.2,
            max_tokens=420,
        )
        raw = response.choices[0].message.content or ""
        obj = _json_from_text(raw)
        selected_id = str(obj.get("selected_id") or "").strip()
        if not selected_id:
            return {
                "selected_id": "",
                "reason": _safe_text(obj.get("reason") or "LLM reviewer found no suitable candidate.", 220),
                "one_line_summary": "",
                "confidence": 0.0,
                "model": cfg["model"],
            }
        ids = {
            str((candidate.get("item") or {}).get("id") or "")
            for candidate in candidates
        }
        if selected_id not in ids:
            return None
        try:
            confidence = float(obj.get("confidence") or 0.0)
        except Exception:
            confidence = 0.0
        return {
            "selected_id": selected_id,
            "reason": _safe_text(obj.get("reason"), 260),
            "one_line_summary": _safe_text(obj.get("one_line_summary"), 180),
            "confidence": max(0.0, min(1.0, confidence)),
            "model": cfg["model"],
        }
    except Exception as e:
        print("[RSS] LLM reviewer failed:", e)
        return None


class RSSContentRecommender:
    def __init__(self):
        self._last_background_refresh = 0.0
        self._last_cleanup = 0.0

    def warmup(self):
        ensure_sources()
        self.cleanup_cache(force=False)
        if _truthy(app_config.get("rss_recommender.refresh_on_startup", "true"), default=True):
            now = time.time()
            if now - self._last_background_refresh > 60:
                self._last_background_refresh = now
                refresh_sources_background(force=False)

    def cleanup_cache(self, *, force=False):
        interval_hours = _int_config(
            "rss_recommender.cache_cleanup_interval_hours",
            DEFAULT_CACHE_CLEANUP_INTERVAL_HOURS,
            min_value=1,
            max_value=168,
        )
        now = time.time()
        if not force and now - self._last_cleanup < interval_hours * 3600:
            return {"skipped": True, "reason": "cleanup_cooldown"}
        self._last_cleanup = now
        result = cleanup_cached_items()
        if result.get("removed"):
            print(f"[RSS] cache cleanup removed {result.get('removed')} items: {result.get('stats')}")
        return result

    def suggest(self, *, user_text="", user_profile="", recent_context=None,
                strategy_prediction=None, trigger_type="user_message", allow_refresh=False):
        if not _truthy(app_config.get("rss_recommender.enabled", "true"), default=True):
            return {"should_recommend": False, "reason": "rss_recommender disabled", "item": None}

        explicit = _explicit_content_request(user_text)
        pred = strategy_prediction if isinstance(strategy_prediction, dict) else {}
        intent = str(pred.get("recommendation_intent") or "")
        source = str(pred.get("source") or "")
        auto_allowed = trigger_type in {"startup", "proactive_timer"} and _truthy(
            app_config.get("rss_recommender.startup_recommend_enabled", "false"),
            default=False,
        )

        if not explicit and not auto_allowed:
            return {
                "should_recommend": False,
                "reason": "not an explicit external-content request",
                "item": None,
                "policy": {"explicit": explicit, "strategy_intent": intent, "strategy_source": source},
            }

        if allow_refresh:
            try:
                refresh_sources(force=False)
            except Exception as e:
                print("[RSS] on-demand refresh failed:", e)

        state = _load_items()
        items = list((state.get("items") or {}).values())
        enabled_source_ids = {
            str(source.get("id") or "")
            for source in ensure_sources()
            if isinstance(source, dict) and source.get("enabled", True) and _source_allowed(source)
        }
        if enabled_source_ids:
            items = [item for item in items if str(item.get("source_id") or "") in enabled_source_ids]
        if not items:
            if allow_refresh:
                try:
                    refresh_sources(force=True)
                    state = _load_items()
                    items = list((state.get("items") or {}).values())
                    enabled_source_ids = {
                        str(source.get("id") or "")
                        for source in ensure_sources()
                        if isinstance(source, dict) and source.get("enabled", True) and _source_allowed(source)
                    }
                    if enabled_source_ids:
                        items = [item for item in items if str(item.get("source_id") or "") in enabled_source_ids]
                except Exception as e:
                    return {"should_recommend": False, "reason": f"RSS refresh failed: {e}", "item": None}
            if not items:
                return {"should_recommend": False, "reason": "RSS cache empty", "item": None}

        profile_text = str(user_profile or "")
        if not profile_text:
            profile_text = " ".join(
                str(x.get("assistant_summary") or x.get("user") or "")
                for x in list(recent_context or [])[-6:]
                if isinstance(x, dict)
            )
        query_tokens = _tokens(user_text or "有意思 视频 杂谈 动画 游戏 学习 AI")
        profile_tokens = _tokens(profile_text)
        recent_ids = _recently_recommended_ids()
        scored = []
        for item in items:
            if not item.get("url"):
                continue
            score, details = _score_item(item, query_tokens, profile_tokens, recent_ids)
            if score <= 0:
                continue
            scored.append((score, item, details))
        scored.sort(key=lambda x: x[0], reverse=True)
        top_k = _int_config("rss_recommender.prompt_candidates", 3, min_value=1, max_value=8)
        candidates = []
        for score, item, details in scored[:top_k]:
            candidates.append({
                "score": round(score, 3),
                "item": {
                    "id": item.get("id"),
                    "title": item.get("title"),
                    "url": item.get("url"),
                    "summary": item.get("summary"),
                    "source_name": item.get("source_name"),
                    "source_type": item.get("source_type"),
                    "platform": item.get("platform"),
                    "cover_url": item.get("cover_url"),
                    "published_at": item.get("published_at"),
                    "tags": item.get("tags") or [],
                },
                "match": details,
            })
        if not candidates:
            return {"should_recommend": False, "reason": "no scored RSS candidate", "item": None}

        reviewer = _llm_review_candidates(
            user_text=user_text,
            user_profile=profile_text,
            recent_context=recent_context,
            candidates=candidates,
        )
        if reviewer and not reviewer.get("selected_id"):
            return {
                "should_recommend": False,
                "reason": reviewer.get("reason") or "LLM reviewer rejected all RSS candidates",
                "item": None,
                "candidates": candidates,
                "reviewer": reviewer,
                "policy": {"explicit": explicit, "strategy_intent": intent, "strategy_source": source},
                "kind": "external_content",
            }

        selected = candidates[0]
        if reviewer and reviewer.get("selected_id"):
            selected = next(
                (candidate for candidate in candidates if (candidate.get("item") or {}).get("id") == reviewer["selected_id"]),
                candidates[0],
            )
        decision = {
            "should_recommend": True,
            "reason": "explicit request" if explicit else "low-frequency startup recommendation",
            "item": selected["item"],
            "score": selected["score"],
            "match": selected["match"],
            "candidates": candidates,
            "reviewer": reviewer,
            "policy": {"explicit": explicit, "strategy_intent": intent, "strategy_source": source},
            "kind": "external_content",
        }
        _append_jsonl(RSS_EVENTS_PATH, {
            "timestamp": _iso(),
            "type": "recommendation",
            "trigger_type": trigger_type,
            "item_id": selected["item"].get("id"),
            "item": selected["item"],
            "score": selected["score"],
            "reason": decision["reason"],
            "reviewer": reviewer,
        })
        return decision


def format_external_content_for_prompt(decision):
    if not decision:
        return "外部内容推荐器未运行。"
    if not decision.get("should_recommend"):
        return f"本轮不推荐外部网页/视频内容。原因：{decision.get('reason', '')}"
    item = decision.get("item") or {}
    match = decision.get("match") or {}
    reviewer = decision.get("reviewer") or {}
    reviewer_text = ""
    if reviewer:
        reviewer_text = (
            f"- 审阅模型：{reviewer.get('model') or 'unknown'}\n"
            f"- 审阅摘要：{reviewer.get('one_line_summary') or '无'}\n"
            f"- 审阅推荐理由：{reviewer.get('reason') or '无'}\n"
            f"- 审阅置信度：{reviewer.get('confidence')}\n"
        )
    return (
        "外部内容推荐器判断：本轮可以推荐一个外部内容。注意：这和“用户行为建议”不是同一个东西。\n"
        "候选已经由审阅模型看过标题、简介、动态文本和原链接后筛选。你给用户时只需要：原链接、内容总结、为什么推荐。\n"
        "不要嵌入视频、不要输出 HTML、不要假装已经完整观看视频；只能基于标题、简介、动态文本、来源和匹配理由描述。\n"
        f"- 标题：{item.get('title')}\n"
        f"- 原链接：{item.get('url')}\n"
        f"- 来源：{item.get('source_name')} / {item.get('platform')} / {item.get('source_type')}\n"
        f"- 发布时间：{item.get('published_at') or 'unknown'}\n"
        f"- 封面：{item.get('cover_url') or 'unknown'}\n"
        f"- 简介：{_safe_text(item.get('summary'), 420)}\n"
        f"- 标签：{', '.join(item.get('tags') or [])}\n"
        f"{reviewer_text}"
        f"- 匹配线索：query_hits={match.get('query_hits', [])}; profile_hits={match.get('profile_hits', [])}\n"
        f"- 推荐器分数：{decision.get('score')}\n"
        "如果你认为这个内容和用户本轮需求不贴合，可以简短说明暂时没有很贴的，不要硬推。"
    )


def format_external_content_recommendation_message(decision):
    if not decision or not decision.get("should_recommend"):
        return ""
    item = decision.get("item") or {}
    reviewer = decision.get("reviewer") or {}
    title = _safe_text(item.get("title"), 120)
    url = _safe_text(item.get("url"), 500)
    summary = _safe_text(
        reviewer.get("one_line_summary") or item.get("summary") or "没有足够摘要，只能先看标题和来源。",
        180,
    )
    reason = _safe_text(reviewer.get("reason") or "和你最近的兴趣或当前状态有些相关。", 180)
    if not (title and url):
        return ""
    return (
        f"看到一个也许合你胃口的内容。\n"
        f"{title}\n"
        f"{url}\n"
        f"大概是：{summary}\n"
        f"推荐理由：{reason}"
    )


rss_content_runtime = RSSContentRecommender()
